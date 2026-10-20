# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill+decode parity for the incremental HCA entry pool.

This exercises the model-side state machine without a GPU backend: one HCA
layer, a fake cache/rotary module, and the real compressor.  The reference is
the training-style one-shot pool over the full context.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.model_executor.models import qwen3_domino as mod


class _IdentityRotary:
    def __call__(self, positions, query, key):
        return query, key


def _rms_norm(out, x, weight, eps):
    var = x.float().pow(2).mean(-1, keepdim=True)
    out.copy_((x.float() * torch.rsqrt(var + eps) * weight.float()).to(x.dtype))


def _reference_pool(comp, normed, capacity):
    pooled, emitted = comp.pool(normed)
    begin = max(0, emitted - capacity)
    return pooled[begin:emitted]


def _make_model(spec, hidden=16, head_dim=8, block_size=8, num_blocks=16):
    comp = mod.DominoHCACompressor(hidden, head_dim, spec, prefix="t.hca")
    inner = SimpleNamespace(
        kv_cache=[
            torch.zeros(num_blocks, block_size, 2 * head_dim),
            torch.zeros(num_blocks, block_size, 2 * head_dim),
        ]
    )
    attn = SimpleNamespace(
        hca=spec,
        head_dim=head_dim,
        hca_compressor=comp,
        rotary_emb=_IdentityRotary(),
        attn=inner,
        q_proj=SimpleNamespace(weight=torch.zeros(1)),
    )
    layer = SimpleNamespace(self_attn=attn)
    model = mod.Qwen3DominoModel.__new__(mod.Qwen3DominoModel)
    model.layer_hca_specs = (spec,)
    model._hca_pool_seqlen = 4
    model._num_attn_layers = 1
    model.context_hidden_size = hidden
    model.is_flare = False
    model.layers = [layer]
    model._hidden_norm_weight = torch.ones(hidden)
    model._hidden_norm_eps = 1e-6
    model._build_hca_pools()
    return model, comp, attn


def _slots_for(positions, block_size):
    return positions.to(torch.int64)


def _check(variant, hop):
    torch.manual_seed(0)
    win = hop if variant == "vanilla" else 2 * hop
    spec = {
        "variant": variant,
        "hop": hop,
        "win": win,
        "coff": 1 if variant == "vanilla" else 2,
        "bias_mode": "learnable",
        "max_entries": 6,
        "window": 64,
    }
    model, comp, attn = _make_model(spec)
    block_size = 8
    prefill_len, decode_len = 63, 9
    total = prefill_len + decode_len
    hidden = model.context_hidden_size
    features = torch.randn(total, hidden)
    positions = torch.arange(total)
    slots = _slots_for(positions, block_size)
    block_table = torch.arange(16).view(1, -1)

    with patch.object(mod.ops, "rms_norm", _rms_norm):
        model._pool_hca_entries(
            features[:prefill_len],
            positions[:prefill_len],
            slots[:prefill_len],
            block_table,
        )
        # Compare prefill against one-shot pooling of the same tokens.
        normed = features[:prefill_len] / features[:prefill_len].pow(2).mean(
            -1, keepdim=True
        ).add(1e-6).sqrt()
        expected = _reference_pool(comp, normed, spec["max_entries"])
        row = 0
        stored = int(attn._hca_pool[3][row].item())
        first = int(attn._hca_pool[2][row].item())
        emitted = (prefill_len - win) // hop + 1
        assert stored == expected.shape[0]
        assert first == max(0, emitted - spec["max_entries"])
        assert torch.allclose(
            attn._hca_pool[1][row, :stored], expected.to(expected.dtype), atol=1e-4
        )

        model._pool_hca_entries(
            features[prefill_len:],
            positions[prefill_len:],
            slots[prefill_len:],
            block_table,
        )
        normed_all = (
            features / features.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        )
        expected_all = _reference_pool(comp, normed_all, spec["max_entries"])

    # Compare the newest pool entries against one-shot pooling of the full
    # context.  The pool keeps only the newest max_entries entries.
    stored = int(attn._hca_pool[3][row].item())
    first = int(attn._hca_pool[2][row].item())
    assert stored == expected_all.shape[0]
    assert first == max(0, (total - win) // hop + 1 - spec["max_entries"])
    assert torch.allclose(
        attn._hca_pool[1][row, :stored], expected_all.to(expected_all.dtype), atol=1e-4
    )
    assert int(model._hca_pos_last[row].item()) == total - 1


def test_incremental_pool_prefill_decode_vanilla():
    _check("vanilla", hop=8)


def test_incremental_pool_prefill_decode_hat():
    _check("hat", hop=8)
