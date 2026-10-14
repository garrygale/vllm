# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``dflash_config.target_tap_norm`` on the served DFlash (fc) draft.

SpecForge normalizes every captured target tap before the fusion when a draft is
trained with ``target_tap_norm``.  The served fusion must slice, normalize and
re-concatenate in the training order, and ``fc`` must see the normalized concat
(it is the quantized activation).  The Laguna drafts in ``laguna_dflash.py``
are the reference implementation.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

NUM_TAPS = 3
TAP_HIDDEN = 8
DRAFT_HIDDEN = 16


def _reference_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """transformers/vLLM RMSNorm math, written out by hand."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


class _FcProbe(torch.nn.Module):
    """Stands in for ReplicatedLinear and records the input it received."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.proj = torch.nn.Linear(input_size, output_size, bias=False)
        self.seen: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen = x
        return self.proj(x)


def _holder(*, tap_norm: bool) -> SimpleNamespace:
    torch.manual_seed(0)
    norms = None
    if tap_norm:
        norms = torch.nn.ModuleList(
            [RMSNorm(TAP_HIDDEN, eps=1e-6) for _ in range(NUM_TAPS)]
        )
        with torch.no_grad():
            for index, norm in enumerate(norms):
                # Distinct gains so a tap-index swap cannot pass.
                norm.weight.fill_(0.5 + index)
    model = SimpleNamespace(
        use_aux_hidden_state=True,
        fc=_FcProbe(NUM_TAPS * TAP_HIDDEN, DRAFT_HIDDEN),
        aux_hidden_norms=norms,
    )
    return SimpleNamespace(model=model)


def _expected_normed_concat(
    holder: SimpleNamespace, hidden_states: torch.Tensor
) -> torch.Tensor:
    taps = hidden_states.view(-1, NUM_TAPS, TAP_HIDDEN)
    normed = torch.stack(
        [
            _reference_rms_norm(taps[:, index, :], norm.weight)
            for index, norm in enumerate(holder.model.aux_hidden_norms)
        ],
        dim=1,
    )
    return normed.reshape(hidden_states.shape[0], -1)


@torch.inference_mode()
def test_target_tap_norm_normalizes_each_tap_before_fc() -> None:
    holder = _holder(tap_norm=True)
    hidden = torch.randn(4, NUM_TAPS * TAP_HIDDEN)
    expected_input = _expected_normed_concat(holder, hidden)
    actual = DFlashQwen3ForCausalLM.combine_hidden_states(holder, hidden)
    # The quantized ``fc`` must see the normalized concat, not the raw one.
    torch.testing.assert_close(
        holder.model.fc.seen, expected_input, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        actual,
        holder.model.fc.proj(expected_input),
        rtol=1e-5,
        atol=1e-6,
    )

    # A single row takes the squeeze path but the same normalization.
    single = hidden[0]
    DFlashQwen3ForCausalLM.combine_hidden_states(holder, single)
    torch.testing.assert_close(
        holder.model.fc.seen,
        _expected_normed_concat(holder, single.unsqueeze(0)),
        rtol=1e-5,
        atol=1e-6,
    )


@torch.inference_mode()
def test_without_the_flag_fc_sees_the_raw_concat() -> None:
    holder = _holder(tap_norm=False)
    hidden = torch.randn(4, NUM_TAPS * TAP_HIDDEN)
    actual = DFlashQwen3ForCausalLM.combine_hidden_states(holder, hidden)
    assert torch.equal(holder.model.fc.seen, hidden)
    torch.testing.assert_close(
        actual, holder.model.fc.proj(hidden), rtol=0, atol=0
    )


@torch.inference_mode()
def test_tap_norm_rejects_a_width_mismatch() -> None:
    holder = _holder(tap_norm=True)
    with pytest.raises(ValueError, match="concatenated aux hidden"):
        DFlashQwen3ForCausalLM.combine_hidden_states(
            holder, torch.randn(4, NUM_TAPS * TAP_HIDDEN + 1)
        )


@torch.inference_mode()
def test_tap_norm_is_skipped_when_aux_hidden_states_are_unused() -> None:
    holder = _holder(tap_norm=True)
    holder.model.use_aux_hidden_state = False
    hidden = torch.randn(4, DRAFT_HIDDEN)
    actual = DFlashQwen3ForCausalLM.combine_hidden_states(holder, hidden)
    assert torch.equal(actual, hidden)
    assert holder.model.fc.seen is None
