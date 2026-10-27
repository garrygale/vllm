# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Domino draft model unit tests (0.30 port, surviving subjects)."""

from types import SimpleNamespace
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
        layer_fusion_weights=torch.zeros(num_draft_layers, num_taps),
    )
    # Route everything through tap 1 to make the orientation observable.
    drafter.layer_fusion_weights[:, 1] = 10.0
    head = SimpleNamespace(model=drafter)
    taps = torch.arange(num_taps).float().view(1, 1, num_taps, 1).expand(
        2, 5, num_taps, target_hidden
    ).clone()
    fused = Qwen3DominoForCausalLM.combine_hidden_states(head, taps.reshape(2 * 5, -1))
    assert fused.shape == (2 * 5, num_draft_layers * target_hidden)
    # Every draft layer's feature equals tap 1's rows, not tap d's.
    per_layer = fused.view(2 * 5, num_draft_layers, target_hidden)
    for d in range(num_draft_layers):
        assert torch.equal(per_layer[:, d], taps[:, :, 1, :].reshape(2 * 5, target_hidden))
