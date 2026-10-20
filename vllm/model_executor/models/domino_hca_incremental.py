# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Incremental compressor state for Domino HCA draft layers.

The serving-side compressor must consume one token at a time during decode
while remaining numerically equivalent to the training-side one-shot pooling:

    weights = softmax_j(gate_j + B[j])
    pooled  = RMSNorm(sum_j weights_j * kv_j)

A window's softmax normalizer is **per channel**: ``gate`` and ``B`` are both
``[.., head_dim]``.  The running state therefore keeps, per row and per open
window phase, a channel-wise max, denominator and weighted value accumulator.
"""

from __future__ import annotations

import torch


def window_phases(position: int, hop: int, coff: int) -> tuple[tuple[int, int], ...]:
    """Return ``(window, phase)`` pairs containing ``position``.

    ``coff=1`` (vanilla) windows are ``[w*hop, (w+1)*hop)``; ``coff=2``
    (hat) windows are ``[w*hop, w*hop + 2*hop)`` and overlap by 50%.
    """

    first = position // hop
    if coff == 1:
        return ((first, first % 1),)
    pairs = []
    for window in (first, first - 1):
        if window >= 0:
            pairs.append((window, window % 2))
    return tuple(pairs)


def window_anchor(window: int, hop: int, coff: int, win: int) -> int:
    """RoPE anchor used by the serving/training compressor."""
    return window * hop + (win // 2 if coff > 1 else 0)


class HcaIncrementalState:
    """Per-layer running softmax state for the currently open HCA windows.

    One instance is allocated per HCA layer.  Rows are request slots allocated
    by the model; ``window``/``complete`` let the caller detect a reused slot or
    a mid-window restart.
    """

    def __init__(
        self,
        num_rows: int,
        num_phases: int,
        head_dim: int,
        win: int,
        hop: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_rows = num_rows
        self.num_phases = num_phases
        self.head_dim = head_dim
        self.win = win
        self.hop = hop
        self.dtype = dtype
        self.num = torch.zeros(
            num_rows, num_phases, head_dim, device=device, dtype=dtype
        )
        self.max = torch.full(
            (num_rows, num_phases, head_dim),
            float("-inf"),
            device=device,
            dtype=dtype,
        )
        self.den = torch.zeros(
            num_rows, num_phases, head_dim, device=device, dtype=dtype
        )
        self.window = torch.full(
            (num_rows, num_phases), -1, device=device, dtype=torch.int64
        )
        self.complete = torch.zeros(
            (num_rows, num_phases), device=device, dtype=torch.bool
        )

    @torch.no_grad()
    def reset_phase(self, row: int, phase: int) -> None:
        self.num[row, phase].zero_()
        self.max[row, phase].fill_(float("-inf"))
        self.den[row, phase].zero_()
        self.window[row, phase] = -1
        self.complete[row, phase] = False

    @torch.no_grad()
    def reset_row(self, row: int) -> None:
        self.num[row].zero_()
        self.max[row].fill_(float("-inf"))
        self.den[row].zero_()
        self.window[row].fill_(-1)
        self.complete[row].zero_()

    @torch.no_grad()
    def mark_complete(self, row: int, phase: int, window: int) -> None:
        """Mark a pre-initialized open window as having seen its first token."""

        self.window[row, phase] = window
        self.complete[row, phase] = True

    @torch.no_grad()
    def update(
        self,
        row: int,
        phase: int,
        window: int,
        position: int,
        logits: torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor | None:
        """Fold one token into one open window.

        Returns the pooled (pre-norm) entry when ``position`` closes the
        window, else ``None``.  The returned tensor is fp32 ``[head_dim]``.
        """

        stored_window = int(self.window[row, phase].item())
        if stored_window != window:
            self.reset_phase(row, phase)
            # A window is only valid if it is consumed from its first token.
            self.complete[row, phase] = position == window * self.hop
            self.window[row, phase] = window

        logits = logits.to(self.dtype)
        kv = kv.to(self.dtype)
        old_max = self.max[row, phase]
        new_max = torch.maximum(old_max, logits)
        old_scale = torch.exp(old_max - new_max)
        new_scale = torch.exp(logits - new_max)
        self.num[row, phase].mul_(old_scale).add_(kv * new_scale)
        self.den[row, phase].mul_(old_scale).add_(new_scale)
        self.max[row, phase] = new_max

        if position + 1 != window * self.hop + self.win:
            return None

        if not bool(self.complete[row, phase].item()):
            # Mid-window restart: this entry would be incomplete.  Drop it
            # rather than feeding a biased entry to attention.
            self.reset_phase(row, phase)
            return None

        pooled = self.num[row, phase] / self.den[row, phase]
        self.reset_phase(row, phase)
        return pooled


def append_pool_entry(
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    first_index: torch.Tensor,
    stored: torch.Tensor,
    row: int,
    window: int,
    entry_k: torch.Tensor,
    entry_v: torch.Tensor,
) -> None:
    """Append one entry to a row's drop-oldest pool, keeping indices contiguous.

    ``first_index[row] + i`` must equal the compressor window index of pool row
    ``i``; if a gap appears (e.g. a mid-context restart dropped a window), the
    older entries are cleared and the pool restarts at ``window``.
    """

    capacity = pool_k.shape[1]
    begin = int(first_index[row].item())
    count = int(stored[row].item())

    if count == 0:
        begin = window
    elif window != begin + count:
        # Contiguity is the dense read path's index contract.
        begin = window
        count = 0

    if count >= capacity:
        pool_k[row, :-1].copy_(pool_k[row, 1:].clone())
        pool_v[row, :-1].copy_(pool_v[row, 1:].clone())
        begin += 1
        count = capacity - 1

    pool_k[row, count].copy_(entry_k)
    pool_v[row, count].copy_(entry_v)
    first_index[row] = begin
    stored[row] = count + 1


def write_prefill_pool(
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    first_index: torch.Tensor,
    stored: torch.Tensor,
    row: int,
    entries_k: torch.Tensor,
    entries_v: torch.Tensor,
) -> None:
    """Write the newest ``capacity`` one-shot entries into one row's pool."""

    capacity = pool_k.shape[1]
    emitted = entries_k.shape[0]
    begin = max(0, emitted - capacity)
    take = entries_k[begin:emitted]
    n = take.shape[0]
    pool_k[row, :n].copy_(take)
    pool_v[row, :n].copy_(entries_v[begin:emitted])
    first_index[row] = begin
    stored[row] = n
