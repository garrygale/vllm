# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.qwen3_domino import DominoDraftAttention
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
