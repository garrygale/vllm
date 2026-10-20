# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HCA (heavily_compressed_attention) draft layers: resolve, pool, dense read.

Server-runnable (needs the vLLM stack importable); the pooling and visibility
math mirrors SpecForge's test_dflash_hca.py so train/serve stay pinned to the
same formulas.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.models.qwen3_domino import (
    DominoHCACompressor,
    DominoQwen3Attention,
    _domino_layer_attention,
    _hat_window_kernel,
    resolve_domino_hca,
)

import torch.nn.functional as F


def hca_config(layer_types, sliding_window, hca, **extra):
    values = {
        "hidden_size": 64,
        "num_hidden_layers": len(layer_types),
        "layer_types": layer_types,
        "block_size": 4,
        "dflash_config": {"sliding_window": sliding_window, "hca": hca},
    }
    values["dflash_config"].update(extra)
    return SimpleNamespace(**values)


class TestResolveDominoHca:
    def test_vanilla_defaults(self):
        config = hca_config(
            ["heavily_compressed_attention", "sliding_attention"],
            [64, 64],
            {"variants": "vanilla", "hops": 16},
        )
        specs = resolve_domino_hca(config)
        assert specs is not None
        assert specs[1] is None
        spec = specs[0]
        assert (spec["hop"], spec["win"], spec["coff"]) == (16, 16, 1)
        assert spec["max_entries"] == 2048 - 64 - 4 - 64
        assert spec["window"] == 64

    def test_hat_overlap(self):
        config = hca_config(
            ["heavily_compressed_attention", "sliding_attention"],
            [64, 64],
            {"variants": "hat", "hops": 8, "bias_mode": "frozen"},
        )
        spec = resolve_domino_hca(config)[0]
        assert (spec["hop"], spec["win"], spec["coff"]) == (8, 16, 2)
        assert spec["bias_mode"] == "frozen"

    def test_layer_window_resolves_for_hca(self):
        config = hca_config(
            ["heavily_compressed_attention", "full_attention"],
            [64, -1],
            {"variants": "vanilla", "hops": 16},
        )
        assert _domino_layer_attention(config, 0) == (64, False)
        assert _domino_layer_attention(config, 1) == (None, False)

    def test_off_by_default(self):
        config = hca_config(
            ["sliding_attention", "full_attention"], [64, -1], None
        )
        assert resolve_domino_hca(config) is None

    def test_rejects_compressed_without_hca(self):
        config = hca_config(
            ["heavily_compressed_attention", "full_attention"], [64, -1], None
        )
        with pytest.raises(ValueError, match="requires"):
            resolve_domino_hca(config)

    def test_rejects_hca_without_compressed(self):
        config = hca_config(
            ["sliding_attention", "full_attention"], [64, -1], {"hops": 16}
        )
        with pytest.raises(ValueError, match="no layer_type"):
            resolve_domino_hca(config)

    def test_rejects_causal(self):
        config = hca_config(
            ["heavily_compressed_attention", "full_attention"],
            [64, -1],
            {"variants": "vanilla", "hops": 16},
            causal=True,
        )
        with pytest.raises(ValueError, match="non-causal"):
            resolve_domino_hca(config)

    def test_rejects_sinks(self):
        config = hca_config(
            ["heavily_compressed_attention", "full_attention"],
            [64, -1],
            {"variants": "vanilla", "hops": 16},
            attn_sink=2,
        )
        with pytest.raises(ValueError, match="attn_sink"):
            resolve_domino_hca(config)

    def test_rejects_short_window(self):
        config = hca_config(
            ["heavily_compressed_attention", "full_attention"],
            [16, -1],
            {"variants": "vanilla", "hops": 16},
        )
        with pytest.raises(ValueError, match="seam"):
            resolve_domino_hca(config)

    def test_rejects_unknown_keys(self):
        config = hca_config(
            ["heavily_compressed_attention", "full_attention"],
            [64, -1],
            {"variants": "vanilla", "hops": 16, "stride": 4},
        )
        with pytest.raises(ValueError, match="unknown"):
            resolve_domino_hca(config)


class TestDominoHCACompressor:
    def _compressor(self, variant="vanilla", hop=16):
        spec = {
            "variant": variant,
            "hop": hop,
            "win": hop if variant == "vanilla" else 2 * hop,
            "coff": 1 if variant == "vanilla" else 2,
            "bias_mode": "learnable",
            "max_entries": 100,
            "window": 64,
        }
        return DominoHCACompressor(64, 8, spec, prefix="t.hca_compressor")

    def test_hat_bias_init(self):
        comp = self._compressor("hat", hop=8)
        expected = torch.log(_hat_window_kernel(16) + 1e-8).unsqueeze(-1)
        assert torch.allclose(comp.position_bias, expected, atol=1e-6)
        # Serving-side bias is always frozen; it ships from the checkpoint.
        assert comp.position_bias.requires_grad is False

    def test_pool_matches_loop_reference(self):
        torch.manual_seed(0)
        comp = self._compressor("hat", hop=8)
        feature = torch.randn(48, 64)
        pooled, capacity = comp.pool(feature)
        assert capacity == (48 - 16) // 8 + 1
        kv = comp.kv_proj(feature)[0]
        gate = comp.gate_proj(feature)[0]
        bias = comp.position_bias
        for j in range(capacity):
            logits = gate[j * 8 : j * 8 + 16] + bias
            w = torch.softmax(logits.float(), dim=0).to(kv.dtype)
            ref = comp.norm((kv[j * 8 : j * 8 + 16] * w).sum(0).unsqueeze(0))[0]
            assert torch.allclose(pooled[j], ref, atol=1e-5)

    def test_pool_empty_context(self):
        comp = self._compressor("vanilla", hop=16)
        pooled, capacity = comp.pool(torch.randn(8, 64))
        assert capacity == 0
        assert pooled.shape[0] == 0


def _make_hca_attention(spec):
    with patch.object(DominoQwen3Attention, "__init__", lambda self: None):
        attn = DominoQwen3Attention()
    attn.hca = spec
    attn.num_heads = 4
    attn.num_kv_heads = 2
    attn.head_dim = 8
    attn.scaling = 8**-0.5
    attn._cache_block_size = 4
    attn._hca_pool = None
    return attn


class TestHcaDenseAttention:
    """Pin the [entries || band || draft] gather+mask math against a reference."""

    def _run(self, variant, hop):
        torch.manual_seed(1)
        win = hop if variant == "vanilla" else 2 * hop
        spec = {
            "variant": variant,
            "hop": hop,
            "win": win,
            "coff": 1 if variant == "vanilla" else 2,
            "bias_mode": "learnable",
            "max_entries": 4,
            "window": 16,
        }
        attn = _make_hca_attention(spec)
        W = 16
        # GQA shapes with num_reqs != num_kv_heads: the mask broadcast only
        # survives both singleton axes (regression guard for the R-vs-kv
        # broadcast bug).
        num_reqs, num_query = 3, 3
        block_size = 4
        num_blocks = 64

        first_pos = torch.tensor([40, 12, 9])
        positions = (
            first_pos.view(-1, 1) + torch.arange(num_query).view(1, -1)
        ).reshape(-1)
        q = torch.randn(num_reqs * num_query, attn.num_heads * attn.head_dim)
        k = torch.randn(num_reqs * num_query, attn.num_kv_heads * attn.head_dim)
        v = torch.randn(num_reqs * num_query, attn.num_kv_heads * attn.head_dim)

        # Paged cache holding rope'd context K/V at each position.
        k_cache = torch.randn(num_blocks, block_size, attn.num_kv_heads * attn.head_dim)
        v_cache = torch.randn(num_blocks, block_size, attn.num_kv_heads * attn.head_dim)
        block_table = torch.arange(num_blocks).view(num_reqs, -1).contiguous()
        # Entries: request 0 keeps its newest 4 of 5 emitted; request 1 has 2.
        capacity = 4
        entries_k = torch.randn(num_reqs, capacity, attn.head_dim)
        entries_v = torch.randn(num_reqs, capacity, attn.head_dim)
        pool_first = torch.tensor([1, 0, 0], dtype=torch.int32)
        pool_stored = torch.tensor([4, 2, 3], dtype=torch.int32)
        attn._hca_pool = (entries_k, entries_v, pool_first, pool_stored)
        attn.attn = SimpleNamespace(kv_cache=[k_cache, v_cache])

        fake_cad = SimpleNamespace(
            query_start_loc=torch.arange(num_reqs + 1) * num_query,
            block_table_tensor=block_table,
        )
        from vllm.model_executor.models import qwen3_domino as mod

        with patch.object(mod, "get_forward_context", lambda: SimpleNamespace(attn_metadata=fake_cad)):
            out = attn._hca_dense_attention(positions, q, k, v)

        # Independent dense reference over the same layout.
        def gather_band(r, lo, hi):
            idx = torch.arange(lo, hi)
            blocks = block_table[r][idx // block_size]
            return (
                k_cache[blocks, idx % block_size].view(-1, attn.num_kv_heads, attn.head_dim),
                v_cache[blocks, idx % block_size].view(-1, attn.num_kv_heads, attn.head_dim),
            )

        for r in range(num_reqs):
            q_r = (
                q.view(num_reqs, num_query, attn.num_heads, attn.head_dim)[r]
                .transpose(0, 1)
            )
            k_draft = (
                k.view(num_reqs, num_query, attn.num_kv_heads, attn.head_dim)[r]
            )
            v_draft = (
                v.view(num_reqs, num_query, attn.num_kv_heads, attn.head_dim)[r]
            )
            band_lo = max(0, int(first_pos[r]) - W)
            k_band, v_band = gather_band(r, band_lo, int(first_pos[r]))
            k_all = torch.cat(
                [
                    entries_k[r].unsqueeze(0).expand(attn.num_kv_heads, -1, -1),
                    k_band.transpose(0, 1),
                    k_draft.transpose(0, 1),
                ],
                dim=1,
            )
            v_all = torch.cat(
                [
                    entries_v[r].unsqueeze(0).expand(attn.num_kv_heads, -1, -1),
                    v_band.transpose(0, 1),
                    v_draft.transpose(0, 1),
                ],
                dim=1,
            )
            n_e, n_b = capacity, int(first_pos[r]) - band_lo
            kv_len = n_e + n_b + num_query
            group = attn.num_heads // attn.num_kv_heads
            for qi in range(num_query):
                t = int(first_pos[r]) + qi
                mask = torch.zeros(kv_len)
                mask[n_e + n_b :] = 1
                for b in range(n_b):
                    if band_lo + b >= t + 1 - W:
                        mask[n_e + b] = 1
                for e in range(int(pool_stored[r])):
                    if (int(pool_first[r]) + e) * hop < t + 1 - W:
                        mask[e] = 1
                # Per-query-head reference with the GQA kv mapping.
                ref = torch.empty(attn.num_heads, attn.head_dim)
                for h in range(attn.num_heads):
                    scores_h = (
                        k_all[h // group].float() * q_r[qi][h].float()
                    ).sum(-1) * attn.scaling
                    scores_h = scores_h.masked_fill(mask == 0, float("-inf"))
                    probs_h = torch.softmax(scores_h, dim=-1)
                    ref[h] = (probs_h.unsqueeze(-1) * v_all[h // group].float()).sum(0)
                got = (
                    out.view(num_reqs, num_query, attn.num_heads, attn.head_dim)[r, qi]
                    .float()
                )
                assert torch.allclose(got, ref, atol=1e-3), (r, qi)

    def test_vanilla(self):
        self._run("vanilla", hop=8)

    def test_hat(self):
        self._run("hat", hop=8)
