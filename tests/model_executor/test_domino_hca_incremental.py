# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the incremental Domino HCA compressor state.

These tests intentionally import only the standalone helper module so they do
not need a GPU attention backend.
"""

import torch

from vllm.model_executor.models.domino_hca_incremental import (
    HcaIncrementalState,
    append_pool_entry,
    window_phases,
)


def _one_shot(kv, gate, bias, win, hop, context_len):
    entries = {}
    emitted = (context_len - win) // hop + 1
    for window in range(emitted):
        start = window * hop
        logits = gate[start : start + win] + bias
        weights = torch.softmax(logits.float(), dim=0)
        entries[window] = (weights * kv[start : start + win].float()).sum(0)
    return entries


def _incremental(kv, gate, bias, win, hop, coff):
    state = HcaIncrementalState(
        num_rows=1,
        num_phases=coff,
        head_dim=kv.shape[-1],
        win=win,
        hop=hop,
        device="cpu",
    )
    entries = {}
    for pos in range(kv.shape[0]):
        for window, phase in window_phases(pos, hop, coff):
            offset = pos - window * hop
            pooled = state.update(
                0,
                phase,
                window,
                pos,
                gate[pos] + bias[offset],
                kv[pos],
            )
            if pooled is not None:
                entries[window] = pooled
    return entries


def test_window_phases():
    assert window_phases(0, 8, 1) == ((0, 0),)
    assert window_phases(8, 8, 1) == ((1, 0),)
    # 50%-overlap hat: every token belongs to the current and previous window.
    assert window_phases(8, 8, 2) == ((1, 1), (0, 0))
    assert window_phases(15, 8, 2) == ((1, 1), (0, 0))


def test_incremental_matches_one_shot_vanilla():
    torch.manual_seed(0)
    win = hop = 8
    context_len = 60
    kv = torch.randn(context_len, 16)
    gate = torch.randn(context_len, 16)
    bias = torch.randn(win, 16) * 0.1
    ref = _one_shot(kv, gate, bias, win, hop, context_len)
    got = _incremental(kv, gate, bias, win, hop, coff=1)
    assert set(ref) == set(got)
    for window in ref:
        assert torch.allclose(ref[window], got[window], atol=1e-5)


def test_incremental_matches_one_shot_hat():
    torch.manual_seed(1)
    hop, coff = 8, 2
    win = coff * hop
    context_len = 61
    kv = torch.randn(context_len, 16)
    gate = torch.randn(context_len, 16)
    bias = torch.randn(win, 16) * 0.1
    ref = _one_shot(kv, gate, bias, win, hop, context_len)
    got = _incremental(kv, gate, bias, win, hop, coff)
    assert set(ref) == set(got)
    for window in ref:
        assert torch.allclose(ref[window], got[window], atol=1e-5)


def test_mid_window_restart_drops_incomplete_entry():
    state = HcaIncrementalState(
        num_rows=1,
        num_phases=1,
        head_dim=4,
        win=8,
        hop=8,
        device="cpu",
    )
    kv = torch.randn(4)
    gate = torch.randn(4)
    pooled = state.update(0, 0, 0, 3, gate, kv)  # starts at offset 3, not 0
    assert pooled is None
    for pos in range(4, 8):
        pooled = state.update(0, 0, 0, pos, gate, kv)
    assert pooled is None  # incomplete window is dropped, not emitted
    assert int(state.window[0, 0]) == -1


def test_append_pool_drop_oldest():
    head_dim = 4
    pool_k = torch.zeros(1, 4, head_dim)
    pool_v = torch.zeros(1, 4, head_dim)
    first = torch.zeros(1, dtype=torch.int32)
    stored = torch.zeros(1, dtype=torch.int32)
    for window in range(7):
        entry = torch.full((head_dim,), float(window))
        append_pool_entry(
            pool_k, pool_v, first, stored, 0, window, entry, entry
        )
    assert int(first[0]) == 3
    assert int(stored[0]) == 4
    assert torch.equal(pool_k[0, :, 0], torch.tensor([3.0, 4.0, 5.0, 6.0]))
    assert torch.equal(pool_v[0, :, 0], torch.tensor([3.0, 4.0, 5.0, 6.0]))
