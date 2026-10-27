# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Domino draft model unit tests (0.30 port, surviving subjects)."""

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.qwen3_domino import (
    DominoDraftAttention,
    Qwen3DominoForCausalLM,
    _domino_layer_attention,
    domino_has_any_non_causal,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec


def test_domino_draft_attention_uses_full_attention_cache_spec():
    layer = object.__new__(DominoDraftAttention)
    sliding_spec = SlidingWindowSpec(
        block_size=128,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=3072,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=512),
    )

    with patch.object(Attention, "get_kv_cache_spec", return_value=sliding_spec):
        spec = layer.get_kv_cache_spec(vllm_config)

    assert isinstance(spec, FullAttentionSpec)
    assert spec.block_size == 512
    assert spec.sliding_window is None
    assert spec.num_kv_heads == sliding_spec.num_kv_heads
    assert spec.head_size == sliding_spec.head_size


def _layer_config(layer_types=None, sliding_window=None, causal=None):
    cfg = SimpleNamespace(
        num_hidden_layers=3,
        layer_types=layer_types,
        sliding_window=None,
        dflash_config={},
    )
    if sliding_window is not None:
        cfg.dflash_config["sliding_window"] = sliding_window
    if causal is not None:
        cfg.dflash_config["causal"] = causal
    return cfg


def test_domino_layer_attention_resolves_per_layer_windows():
    cfg = _layer_config(
        layer_types=["sliding_attention"] * 3,
        sliding_window=[2040, 1024, 512],
    )
    assert _domino_layer_attention(cfg, 0) == (2040, False)
    assert _domino_layer_attention(cfg, 1) == (1024, False)
    assert _domino_layer_attention(cfg, 2) == (512, False)
    # Domino layers default non-causal; the causal override wins when set.
    cfg_causal = _layer_config(
        layer_types=["sliding_attention"] * 3,
        sliding_window=[2040, 1024, 512],
        causal=True,
    )
    assert _domino_layer_attention(cfg_causal, 0) == (2040, True)
    # Invalid / negative entries resolve to full attention.
    cfg_bad = _layer_config(
        layer_types=["sliding_attention"] * 3,
        sliding_window=[2040, -1, 512],
    )
    assert _domino_layer_attention(cfg_bad, 1) == (None, False)
    # A non-sliding layer_type ignores the window entirely.
    cfg_mixed = _layer_config(
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        sliding_window=[2040, 1024, 512],
    )
    assert _domino_layer_attention(cfg_mixed, 0) == (None, False)
    assert _domino_layer_attention(cfg_mixed, 1) == (1024, False)


def test_domino_has_any_non_causal():
    # Domino defaults to non-causal on every layer.
    assert domino_has_any_non_causal(_layer_config(layer_types=["full_attention"] * 3))
    assert domino_has_any_non_causal(
        _layer_config(layer_types=["sliding_attention"] * 3)
    )
    # All-causal only when the override is set.
    assert not domino_has_any_non_causal(
        _layer_config(layer_types=["sliding_attention"] * 3, causal=True)
    )


def test_combine_hidden_states_flare_orientation():
    """The fused context feature must be [N, D, H] with the draft layer on
    axis 1: fusion weights [D, T] matmul the per-tap features [N, T, H]."""
    num_draft_layers, num_taps, target_hidden = 3, 4, 8
    drafter = SimpleNamespace(
        num_target_features=num_taps,
        target_hidden_size=target_hidden,
        config=SimpleNamespace(num_hidden_layers=num_draft_layers),
        layer_fusion_weights=torch.full(
            (num_draft_layers, num_taps), -10.0
        ),
    )
    # Route draft layer d to tap d; softmax makes a near-one-hot row, so a
    # transposed fusion (weights applied over the draft-layer axis) would mix
    # O(1)-different taps into every row. Compare with tolerance: softmax is
    # never exactly one-hot.
    for d in range(num_draft_layers):
        drafter.layer_fusion_weights[d, d] = 10.0
    head = SimpleNamespace(model=drafter)
    taps = torch.arange(num_taps).float().view(1, 1, num_taps, 1).expand(
        2, 5, num_taps, target_hidden
    ).clone()
    fused = Qwen3DominoForCausalLM.combine_hidden_states(head, taps.reshape(2 * 5, -1))
    assert fused.shape == (2 * 5, num_draft_layers * target_hidden)
    # Draft layer d's feature must be tap d's rows, not any other tap's.
    per_layer = fused.view(2 * 5, num_draft_layers, target_hidden)
    for d in range(num_draft_layers):
        torch.testing.assert_close(
            per_layer[:, d],
            taps[:, :, d, :].reshape(2 * 5, target_hidden),
            atol=1e-2,
            rtol=1e-2,
        )


# ---------------------------------------------------------------------------
# out_proj_after_norm (fold + unfolded fallback)
# ---------------------------------------------------------------------------

import torch.nn.functional as F  # noqa: E402
import pytest  # noqa: E402

from vllm.model_executor.layers.linear import ReplicatedLinear  # noqa: E402
from vllm.model_executor.layers.logits_processor import LogitsProcessor  # noqa: E402
from vllm.model_executor.layers.vocab_parallel_embedding import (  # noqa: E402
    ParallelLMHead,
)
from vllm.model_executor.models.qwen3_domino import (  # noqa: E402
    _fold_output_proj_into_lm_head,
)
from vllm.platforms import current_platform  # noqa: E402

CPU = ("cpu",)


def _tiny_head_and_proj(device):
    H_d, H_t, V = 8, 16, 32
    lm_head = ParallelLMHead(V, H_t, prefix="lm_head")
    output_proj = ReplicatedLinear(
        H_d, H_t, bias=False, return_bias=False, prefix="model.output_proj"
    )
    with torch.no_grad():
        lm_head.weight.normal_()
        output_proj.weight.normal_()
    return lm_head, output_proj, H_d, H_t, V


def _fold_draft(lm_head, output_proj, hidden_proj=None):
    return SimpleNamespace(
        model=SimpleNamespace(
            out_proj_after_norm=True,
            output_proj=output_proj,
            hidden_proj=hidden_proj,
        ),
        lm_head=lm_head,
    )


@torch.inference_mode()
@pytest.mark.parametrize("device", CPU)
def test_out_proj_fold_matches_unfolded_sequence(
    default_vllm_config, dist_init, device
) -> None:
    torch.set_default_device(device)
    torch.manual_seed(0)

    lm_head, output_proj, H_d, H_t, V = _tiny_head_and_proj(device)
    draft = _fold_draft(lm_head, output_proj)
    assert _fold_output_proj_into_lm_head(draft)
    assert draft.model.output_proj is None
    assert draft.lm_head is not lm_head
    # The folded head keeps the original vocab padding; only the input width
    # collapses to draft width.
    assert draft.lm_head.weight.shape[0] == lm_head.weight.shape[0]
    assert draft.lm_head.weight.shape[1] == H_d

    hidden = torch.randn(4, H_d)
    expected = F.linear(F.linear(hidden, output_proj.weight), lm_head.weight)
    torch.testing.assert_close(
        F.linear(hidden, draft.lm_head.weight), expected, atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("device", CPU)
def test_out_proj_fold_skips_unfoldable_head(
    default_vllm_config, dist_init, device
) -> None:
    torch.set_default_device(device)

    lm_head, output_proj, _, _, _ = _tiny_head_and_proj(device)
    lm_head.quant_method = object()  # not a dense unquantized head
    draft = _fold_draft(lm_head, output_proj)
    assert not _fold_output_proj_into_lm_head(draft)
    assert draft.model.output_proj is output_proj
    assert draft.lm_head is lm_head


@pytest.mark.parametrize("device", CPU)
def test_out_proj_fold_keeps_unfolded_head_for_hidden_proj(
    default_vllm_config, dist_init, device
) -> None:
    torch.set_default_device(device)
    torch.manual_seed(0)

    lm_head, output_proj, H_d, H_t, V = _tiny_head_and_proj(device)
    draft = _fold_draft(lm_head, output_proj, hidden_proj=object())
    assert _fold_output_proj_into_lm_head(draft)
    assert draft._unfolded_lm_head is lm_head


@pytest.mark.parametrize("device", CPU)
def test_compute_draft_logits_projects_unfolded_hidden(
    default_vllm_config, dist_init, device
) -> None:
    torch.set_default_device(device)
    torch.manual_seed(0)

    lm_head, output_proj, H_d, H_t, V = _tiny_head_and_proj(device)
    # Duck-typed draft: newer torch forbids assigning submodules on a
    # Module built via object.__new__ without __init__, so bind the real
    # method onto a plain namespace instead.
    draft = SimpleNamespace(
        model=SimpleNamespace(
            out_proj_after_norm=True, output_proj=output_proj
        ),
        lm_head=lm_head,
        logits_processor=LogitsProcessor(V),
    )
    draft.compute_draft_logits = MethodType(
        Qwen3DominoForCausalLM.compute_draft_logits, draft
    )

    hidden = torch.randn(4, H_d)
    expected = F.linear(F.linear(hidden, output_proj.weight), lm_head.weight)
    # LogitsProcessor trims the vocab padding off the head output.
    expected = expected[:, :V]
    torch.testing.assert_close(
        draft.compute_draft_logits(hidden), expected, atol=1e-4, rtol=1e-4
    )

    # Folded path: no output_proj left, the head does the whole job.
    folded = _fold_draft(lm_head, output_proj)
    _fold_output_proj_into_lm_head(folded)
    draft.model.output_proj = None
    draft.lm_head = folded.lm_head
    torch.testing.assert_close(
        draft.compute_draft_logits(hidden), expected, atol=1e-4, rtol=1e-4
    )
