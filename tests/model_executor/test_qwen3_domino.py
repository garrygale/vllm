# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models import qwen3_domino
from vllm.model_executor.models.qwen3_domino import (
    DominoDraftAttention,
    DominoGroupedConv,
    DominoQwen3Attention,
    Qwen3DominoForCausalLM,
    Qwen3DominoModel,
    RoutedOuterMLP,
    SharedGLUMLP,
    _fold_output_proj_into_lm_head,
    _fold_shared_emb_readout,
    _resolve_vha_load_spec,
    fold_vha_weights,
    resolve_ffn_sharing,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

import torch.nn.functional as F

DEVICE_TYPE = current_platform.device_type
DEVICES = (
    [f"{DEVICE_TYPE}:{i}" for i in range(min(torch.accelerator.device_count(), 2))]
    if not current_platform.is_cpu()
    else ["cpu"]
)


def _draft_config(**overrides) -> SimpleNamespace:
    values = {
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "dflash_config": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_domino_attention(**overrides) -> DominoQwen3Attention:
    kwargs = dict(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
        target_hidden_size=64,
        rope_parameters={"rope_type": "default"},
        max_position=128,
        head_dim=16,
        sliding_window=None,
        causal=False,
        prefix="test.self_attn",
    )
    kwargs.update(overrides)
    return DominoQwen3Attention(**kwargs)


def _make_grouped_conv(taps=2, group_size=8, hidden=32, block=4):
    return DominoGroupedConv(
        hidden_size=hidden,
        block_size=block,
        taps=taps,
        group_size=group_size,
        dtype=torch.float32,
    )


def test_grouped_conv_identity_base_is_a_noop():
    """base_kernel identity + zero dynamic delta -> prepare/finish change nothing."""

    conv = _make_grouped_conv(taps=3)
    with torch.no_grad():
        conv.kernel_projection.weight.zero_()
    x = torch.randn(2, 12, 32)
    prepared, coeff = conv.prepare(x)
    assert torch.equal(prepared, x)
    assert torch.equal(conv.finish(x, coeff), x)


def test_grouped_conv_flattened_matches_batched():
    """Serving feeds [T, H]; training feeds [B, blocks*block, H]."""

    torch.manual_seed(0)
    conv = _make_grouped_conv()
    with torch.no_grad():
        conv.base_kernel.normal_(0, 0.2)
        conv.kernel_projection.weight.normal_(0, 0.2)
    x = torch.randn(2, 12, 32)
    batched, coeff = conv.prepare(x)
    batched = conv.finish(batched, coeff)
    flat_in = x.reshape(24, 32)
    flat, coeff_flat = conv.prepare(flat_in)
    flat = conv.finish(flat, coeff_flat)
    assert torch.allclose(
        batched.reshape(24, 32), flat, atol=1e-5, rtol=1e-5
    )


def test_grouped_conv_causality_within_block():
    """Later taps must not see future positions, or cross block boundaries."""

    conv = _make_grouped_conv(taps=2)
    with torch.no_grad():
        conv.base_kernel.zero_()
        conv.base_kernel[0, 1] = 1.0  # output = previous position only
        conv.kernel_projection.weight.zero_()
    x = torch.randn(1, 8, 32)
    out, _ = conv.prepare(x)
    # 8 tokens is two 4-token blocks; each block starts with no predecessor.
    assert torch.allclose(out[0, 0], torch.zeros_like(out[0, 0]))
    assert torch.allclose(out[0, 1:4], x[0, 0:3])
    assert torch.allclose(out[0, 4], torch.zeros_like(out[0, 4]))
    assert torch.allclose(out[0, 5:8], x[0, 4:7])


def test_grouped_conv_rejects_bad_geometry():
    with pytest.raises(ValueError):
        _make_grouped_conv(taps=0)
    with pytest.raises(ValueError):
        _make_grouped_conv(group_size=7)


def _vha_weight_dict(nh=4, nkv=2, hd=8, hidden=32, seed=0):
    torch.manual_seed(seed)
    physical = nh // nkv
    q = torch.randn(physical * hd, hidden)
    o = torch.randn(hidden, nh * hd)
    premix = torch.randn(nkv, hd, hd)
    u = torch.randn(nh, 2)
    v = torch.randn(nh, 2)
    weights = {
        "model.layers.0.self_attn.q_proj.weight": q,
        "model.layers.0.self_attn.o_proj.weight": o,
        "model.layers.0.self_attn.vha_premix": premix,
        "model.layers.0.self_attn.vha_postmix_u": u,
        "model.layers.0.self_attn.vha_postmix_v": v,
    }
    return weights, q, o, premix, u, v


def test_vha_weights_fold_in_place():
    weights, q, o, premix, u, v = _vha_weight_dict()
    nh, nkv, hd = 4, 2, 8
    physical = nh // nkv

    assert fold_vha_weights(weights, nh, hd)
    assert not any("vha" in name for name in weights)

    folded_q = weights["model.layers.0.self_attn.q_proj.weight"]
    assert folded_q.shape == (nh * hd, q.shape[1])
    ref_q = torch.zeros_like(folded_q)
    for k in range(nkv):
        for i in range(physical):
            ref_q[(k * physical + i) * hd : (k * physical + i + 1) * hd] = (
                premix[k].t() @ q[i * hd : (i + 1) * hd]
            )
    assert torch.allclose(folded_q, ref_q, atol=1e-5)

    folded_o = weights["model.layers.0.self_attn.o_proj.weight"]
    assert folded_o.shape == o.shape
    ref_o = o.clone().view(o.shape[0], nh, hd)
    for w in range(nh):
        for r in range(u.shape[1]):
            c = sum(v[vv, r] * o.view(o.shape[0], nh, hd)[:, vv, :] for vv in range(nh))
            ref_o[:, w, :] += u[w, r] * c
    assert torch.allclose(folded_o, ref_o.reshape(o.shape), atol=1e-4)


def test_fold_vha_weights_rejects_incomplete_params():
    weights, *_ = _vha_weight_dict()
    del weights["model.layers.0.self_attn.vha_postmix_v"]
    with pytest.raises(ValueError, match="incomplete VHA"):
        fold_vha_weights(weights, 4, 8)


def test_fold_vha_weights_rejects_width_mismatch():
    weights, *_ = _vha_weight_dict()
    weights["model.layers.0.self_attn.q_proj.weight"] = torch.randn(64, 32)
    with pytest.raises(ValueError, match="shape mismatch"):
        fold_vha_weights(weights, 4, 8)


def test_fold_vha_weights_passes_through_plain_checkpoint():
    weights = {"model.layers.0.self_attn.q_proj.weight": torch.randn(32, 8)}
    assert not fold_vha_weights(weights, 4, 8)


def test_resolve_vha_load_spec():
    assert _resolve_vha_load_spec(SimpleNamespace(dflash_config={})) is None
    cfg = SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=2,
        dflash_config={"vha": {"physical_q_heads": 2}},
    )
    assert _resolve_vha_load_spec(cfg) == 2
    with pytest.raises(ValueError, match="physical_q_heads"):
        _resolve_vha_load_spec(SimpleNamespace(dflash_config={"vha": {}}))
    bad = SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=2,
        dflash_config={"vha": {"physical_q_heads": 3}},
    )
    with pytest.raises(ValueError, match="num_attention_heads"):
        _resolve_vha_load_spec(bad)


def test_domino_attention_sink_disabled_by_default():
    attn = _make_domino_attention()
    assert attn.num_sinks == 0
    assert not hasattr(attn, "sink_k")


def test_domino_attention_sink_params_and_causal_guard():
    attn = _make_domino_attention(num_sinks=1)
    assert attn.num_sinks == 1
    assert attn.sink_k.shape == (1, 2, 16)
    assert attn.sink_v.shape == (1, 2, 16)
    assert torch.count_nonzero(attn.sink_k) == 0
    assert torch.count_nonzero(attn.sink_v) == 0
    with pytest.raises(ValueError, match="attn_sink"):
        _make_domino_attention(num_sinks=1, causal=True)


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


def test_resolve_ffn_sharing_defaults_match_training():
    # No knob keeps today's Qwen3MLP; each side defaults to no sharing.
    assert resolve_ffn_sharing(_draft_config()) is None
    assert resolve_ffn_sharing(_draft_config(dflash_config={"ffn_sharing": None})) is None
    assert resolve_ffn_sharing(_draft_config(dflash_config={"ffn_sharing": False})) is None
    # Pure gate sharing k=2 on the 35B-A3B draft.
    assert resolve_ffn_sharing(
        _draft_config(dflash_config={"ffn_sharing": {"gate_groups": 4864}})
    ) == ("lattice", "nested", 4864, 9728)
    # The exactly-once outer lattice that keeps 9728 = 76 * 128.
    assert resolve_ffn_sharing(
        _draft_config(
            dflash_config={
                "ffn_sharing": {"pairing": "outer", "gate_groups": 76, "up_groups": 128}
            }
        )
    ) == ("lattice", "outer", 76, 128)


def test_resolve_ffn_sharing_routed_outer():
    assert resolve_ffn_sharing(
        _draft_config(
            intermediate_size=48,
            dflash_config={
                "ffn_sharing": {
                    "mode": "routed_outer",
                    "experts": 4,
                    "gate_groups": 3,
                    "up_groups": 4,
                }
            },
        )
    ) == ("routed_outer", 4, 3, 4, "expert")
    with pytest.raises(ValueError):  # wrong intermediate_size
        resolve_ffn_sharing(
            _draft_config(
                intermediate_size=47,
                dflash_config={
                    "ffn_sharing": {
                        "mode": "routed_outer",
                        "experts": 4,
                        "gate_groups": 3,
                        "up_groups": 4,
                    }
                },
            )
        )
    with pytest.raises(ValueError):  # gate_slot with a single expert
        resolve_ffn_sharing(
            _draft_config(
                intermediate_size=12,
                dflash_config={
                    "ffn_sharing": {
                        "mode": "routed_outer",
                        "experts": 1,
                        "gate_groups": 3,
                        "up_groups": 4,
                        "router": "gate_slot",
                    }
                },
            )
        )


def test_resolve_ffn_sharing_rejects_invalid_knobs():
    with pytest.raises(ValueError):
        resolve_ffn_sharing(_draft_config(dflash_config={"ffn_sharing": "gate"}))
    with pytest.raises(ValueError):
        resolve_ffn_sharing(
            _draft_config(dflash_config={"ffn_sharing": {"pairing_k": "nested"}})
        )
    with pytest.raises(ValueError):
        resolve_ffn_sharing(
            _draft_config(dflash_config={"ffn_sharing": {"pairing": "weave"}})
        )
    with pytest.raises(ValueError):
        resolve_ffn_sharing(_draft_config(dflash_config={"ffn_sharing": {"gate_groups": 5}}))
    with pytest.raises(ValueError):
        resolve_ffn_sharing(_draft_config(dflash_config={"ffn_sharing": {"gate_groups": 9729}}))


def test_resolve_ffn_sharing_outer_error_suggests_factorizations():
    with pytest.raises(ValueError, match="4096") as ctx:
        resolve_ffn_sharing(
            _draft_config(
                dflash_config={
                    "ffn_sharing": {
                        "pairing": "outer",
                        "gate_groups": 64,
                        "up_groups": 64,
                    }
                }
            )
        )
    assert "76x128" in str(ctx.value)


class _MLPHolder(torch.nn.Module):
    """Replicates Qwen3DominoModel's two-line stacked-mapper load path."""

    hf_to_vllm_mapper = Qwen3DominoModel.hf_to_vllm_mapper

    def __init__(self, mlp: torch.nn.Module) -> None:
        super().__init__()
        self.mlp = mlp

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@torch.inference_mode()
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "pairing,gate_groups,up_groups",
    [("nested", 12, 24), ("nested", 12, 8), ("outer", 2, 12), ("outer", 2, 6)],
)
def test_shared_glu_mlp_matches_the_training_formula(
    default_vllm_config, dist_init, device, pairing, gate_groups, up_groups
) -> None:
    """The served channel sharing must be the index math it trained as."""
    if current_platform.is_cuda_alike():
        torch.accelerator.set_device_index(device)
    torch.set_default_device(device)

    mlp = SharedGLUMLP(
        hidden_size=8,
        intermediate_size=24,
        hidden_act="silu",
        pairing=pairing,
        gate_groups=gate_groups,
        up_groups=up_groups,
        prefix="layers.0.mlp",
    )
    parameters = dict(mlp.named_parameters())
    assert parameters["gate_up_proj.weight"].shape == (
        gate_groups + up_groups,
        8,
    )
    assert parameters["down_proj.weight"].shape == (8, 24)
    # The index maps are derived, not checkpointed.
    assert "gate_idx" not in mlp.state_dict()
    assert "up_idx" not in mlp.state_dict()

    hidden_states = torch.randn(4, 8)
    gate = torch.nn.functional.linear(
        hidden_states, mlp.gate_up_proj.weight[:gate_groups]
    )
    up = torch.nn.functional.linear(
        hidden_states, mlp.gate_up_proj.weight[gate_groups:]
    )
    expected_hidden = torch.nn.functional.silu(gate[..., mlp.gate_idx]) * up[
        ..., mlp.up_idx
    ]
    expected = torch.nn.functional.linear(expected_hidden, mlp.down_proj.weight)
    assert torch.allclose(mlp(hidden_states), expected, atol=1e-5)


@torch.inference_mode()
@pytest.mark.parametrize("device", DEVICES)
def test_shared_glu_mlp_loads_a_specforge_export(
    default_vllm_config, dist_init, device
) -> None:
    """Separate unequal gate/up checkpoint keys land in the fused halves."""

    if current_platform.is_cuda_alike():
        torch.accelerator.set_device_index(device)
    torch.set_default_device(device)

    torch.manual_seed(0)
    mlp = SharedGLUMLP(
        hidden_size=8,
        intermediate_size=24,
        hidden_act="silu",
        gate_groups=12,
        up_groups=24,
        prefix="layers.0.mlp",
    )
    holder = _MLPHolder(mlp)
    gate_w = torch.randn(12, 8)
    up_w = torch.randn(24, 8)
    down_w = torch.randn(8, 24)
    holder.load_weights(
        [
            ("mlp.gate_proj.weight", gate_w),
            ("mlp.up_proj.weight", up_w),
            ("mlp.down_proj.weight", down_w),
        ]
    )
    assert torch.allclose(mlp.gate_up_proj.weight, torch.cat([gate_w, up_w]))

    hidden_states = torch.randn(4, 8)
    gate = torch.nn.functional.linear(hidden_states, gate_w)
    up = torch.nn.functional.linear(hidden_states, up_w)
    expected_hidden = torch.nn.functional.silu(gate[..., mlp.gate_idx]) * up[
        ..., mlp.up_idx
    ]
    expected = torch.nn.functional.linear(expected_hidden, down_w)
    assert torch.allclose(mlp(hidden_states), expected, atol=1e-5)


@torch.inference_mode()
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "experts,gate_groups,up_groups,router",
    [(4, 3, 4, "expert"), (2, 3, 4, "gate_slot")],
)
def test_routed_outer_mlp_matches_the_training_formula(
    default_vllm_config, dist_init, device, experts, gate_groups, up_groups, router
) -> None:
    """The served routed blend must be the math it trained as."""
    if current_platform.is_cuda_alike():
        torch.accelerator.set_device_index(device)
    torch.set_default_device(device)

    mlp = RoutedOuterMLP(
        hidden_size=8,
        intermediate_size=experts * gate_groups * up_groups,
        hidden_act="silu",
        experts=experts,
        gate_groups=gate_groups,
        up_groups=up_groups,
        router=router,
        prefix="layers.0.mlp",
    )
    total = experts * (gate_groups + up_groups)
    assert mlp.gate_up_proj.weight.shape == (total + mlp.router_width, 8)
    assert mlp.down_proj.weight.shape == (8, gate_groups * up_groups)
    with torch.no_grad():
        mlp.gate_up_proj.weight[total:].normal_()

    hidden_states = torch.randn(4, 8)
    fused = torch.nn.functional.linear(
        hidden_states, mlp.gate_up_proj.weight
    )
    feats = fused[..., :total].reshape(4, experts, gate_groups + up_groups)
    route = fused[..., total:]
    ref = torch.zeros(4, gate_groups * up_groups)
    if router == "expert":
        alpha = torch.softmax(route, dim=-1)
        for e in range(experts):
            a = torch.nn.functional.silu(feats[:, e, :gate_groups]) * alpha[
                :, e : e + 1
            ]
            u = feats[:, e, gate_groups:]
            ref += (a.unsqueeze(-1) * u.unsqueeze(-2)).reshape(
                4, gate_groups * up_groups
            )
    else:
        delta = torch.softmax(
            route.reshape(4, gate_groups, experts), dim=-1
        )
        for e in range(experts):
            a = torch.nn.functional.silu(feats[:, e, :gate_groups]) * delta[
                :, :, e
            ]
            u = feats[:, e, gate_groups:]
            ref += (a.unsqueeze(-1) * u.unsqueeze(-2)).reshape(
                4, gate_groups * up_groups
            )
    expected = torch.nn.functional.linear(ref, mlp.down_proj.weight)
    assert torch.allclose(mlp(hidden_states), expected, atol=1e-5)


def test_shared_glu_mlp_rejects_tensor_parallel_drafts():
    with patch(
        "vllm.model_executor.models.qwen3_domino."
        "get_tensor_model_parallel_world_size",
        return_value=2,
    ):
        with pytest.raises(NotImplementedError, match="draft_tensor_parallel_size"):
            SharedGLUMLP(
                hidden_size=8,
                intermediate_size=24,
                hidden_act="silu",
                gate_groups=12,
                up_groups=24,
                prefix="layers.0.mlp",
            )


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
@pytest.mark.parametrize("device", DEVICES)
def test_out_proj_fold_matches_unfolded_sequence(
    default_vllm_config, dist_init, device
) -> None:
    if current_platform.is_cuda_alike():
        torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.manual_seed(0)

    lm_head, output_proj, H_d, H_t, V = _tiny_head_and_proj(device)
    draft = _fold_draft(lm_head, output_proj)
    assert _fold_output_proj_into_lm_head(draft)
    assert draft.model.output_proj is None
    assert draft.lm_head is not lm_head
    assert draft.lm_head.weight.shape == (V, H_d)

    hidden = torch.randn(4, H_d)
    expected = F.linear(F.linear(hidden, output_proj.weight), lm_head.weight)
    torch.testing.assert_close(
        F.linear(hidden, draft.lm_head.weight), expected, atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("device", DEVICES)
def test_out_proj_fold_skips_unfoldable_head(
    default_vllm_config, dist_init, device
) -> None:
    if current_platform.is_cuda_alike():
        torch.accelerator.set_device_index(device)
    torch.set_default_device(device)

    lm_head, output_proj, _, _, _ = _tiny_head_and_proj(device)
    lm_head.quant_method = object()  # not a dense unquantized head
    draft = _fold_draft(lm_head, output_proj)
    assert not _fold_output_proj_into_lm_head(draft)
    assert draft.model.output_proj is output_proj
    assert draft.lm_head is lm_head


@pytest.mark.parametrize("device", DEVICES)
def test_out_proj_fold_keeps_unfolded_head_for_hidden_proj(
    default_vllm_config, dist_init, device
) -> None:
    if current_platform.is_cuda_alike():
        torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.manual_seed(0)

    lm_head, output_proj, H_d, H_t, V = _tiny_head_and_proj(device)
    draft = _fold_draft(lm_head, output_proj, hidden_proj=object())
    assert _fold_output_proj_into_lm_head(draft)
    assert draft._unfolded_lm_head is lm_head


@pytest.mark.parametrize("device", DEVICES)
def test_compute_draft_logits_projects_unfolded_hidden(
    default_vllm_config, dist_init, device
) -> None:
    if current_platform.is_cuda_alike():
        torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.manual_seed(0)

    lm_head, output_proj, H_d, H_t, V = _tiny_head_and_proj(device)
    draft = object.__new__(Qwen3DominoForCausalLM)
    draft.model = SimpleNamespace(
        out_proj_after_norm=True, output_proj=output_proj
    )
    draft.lm_head = lm_head
    draft.logits_processor = LogitsProcessor(V)

    hidden = torch.randn(4, H_d)
    expected = F.linear(F.linear(hidden, output_proj.weight), lm_head.weight)
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


def _shared_readout_draft(lm_head, head_down_proj, head_up_proj):
    return SimpleNamespace(
        model=SimpleNamespace(
            shared_emb_readout=True,
            out_proj_after_norm=True,
            output_proj=None,
            head_down_proj=head_down_proj,
            head_up_proj=head_up_proj,
        ),
        lm_head=lm_head,
    )


def _tiny_shared_readout_heads(device):
    H_d, M, H_t, V = 8, 4, 16, 32
    lm_head = ParallelLMHead(V, H_t, prefix="lm_head")
    head_down_proj = ReplicatedLinear(
        H_d, M, bias=False, return_bias=False, prefix="model.head_down_proj"
    )
    head_up_proj = ReplicatedLinear(
        M, H_t, bias=False, return_bias=False, prefix="model.head_up_proj"
    )
    with torch.no_grad():
        lm_head.weight.normal_()
        head_down_proj.weight.normal_()
        head_up_proj.weight.normal_()
    return lm_head, head_down_proj, head_up_proj, H_d, M, H_t, V


@torch.inference_mode()
@pytest.mark.parametrize("device", DEVICES)
def test_shared_readout_fold_matches_two_path_sequence(
    default_vllm_config, dist_init, device
) -> None:
    if current_platform.is_cuda_alike():
        torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.manual_seed(0)

    lm_head, head_down_proj, head_up_proj, H_d, M, H_t, V = (
        _tiny_shared_readout_heads(device)
    )
    draft = _shared_readout_draft(lm_head, head_down_proj, head_up_proj)
    _fold_shared_emb_readout(draft)
    assert draft.model.head_up_proj is None
    assert draft.lm_head is not lm_head
    assert draft.lm_head.weight.shape == (V, M)

    # u carries base + correction in emb space; the folded head must equal
    # lm_head @ W_up applied after the fact.
    z = torch.randn(4, H_d)
    emb_corr = torch.randn(4, M)
    u = F.linear(z, head_down_proj.weight) + emb_corr
    expected = F.linear(F.linear(u, head_up_proj.weight), lm_head.weight)
    torch.testing.assert_close(
        F.linear(u, draft.lm_head.weight), expected, atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize("device", DEVICES)
def test_shared_readout_fold_raises_on_unfoldable_head(
    default_vllm_config, dist_init, device
) -> None:
    if current_platform.is_cuda_alike():
        torch.accelerator.set_device_index(device)
    torch.set_default_device(device)

    lm_head, head_down_proj, head_up_proj, *_ = _tiny_shared_readout_heads(device)
    lm_head.quant_method = object()  # not a dense unquantized head
    draft = _shared_readout_draft(lm_head, head_down_proj, head_up_proj)
    with pytest.raises(RuntimeError, match="not foldable"):
        _fold_shared_emb_readout(draft)
    assert draft.model.head_up_proj is head_up_proj
    assert draft.lm_head is lm_head


@torch.inference_mode()
@pytest.mark.parametrize("device", DEVICES)
def test_compute_draft_logits_routes_shared_readout_through_down_proj(
    default_vllm_config, dist_init, device
) -> None:
    if current_platform.is_cuda_alike():
        torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.manual_seed(0)

    lm_head, head_down_proj, head_up_proj, H_d, M, H_t, V = (
        _tiny_shared_readout_heads(device)
    )
    draft = _shared_readout_draft(lm_head, head_down_proj, head_up_proj)
    _fold_shared_emb_readout(draft)

    holder = object.__new__(Qwen3DominoForCausalLM)
    holder.model = draft.model
    holder.lm_head = draft.lm_head
    holder.logits_processor = LogitsProcessor(V)

    hidden = torch.randn(4, H_d)
    expected = F.linear(
        F.linear(F.linear(hidden, head_down_proj.weight), head_up_proj.weight),
        lm_head.weight,
    )
    torch.testing.assert_close(
        holder.compute_draft_logits(hidden), expected, atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        holder.emb_logits(F.linear(hidden, head_down_proj.weight)),
        expected,
        atol=1e-4,
        rtol=1e-4,
    )


def _reference_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """transformers/vLLM RMSNorm math, written out by hand."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


def _tap_norm_model(
    *,
    tap_norm: bool,
    num_target_features: int = 3,
    target_hidden_size: int = 8,
    num_layers: int = 2,
) -> SimpleNamespace:
    """The subset of Qwen3DominoModel that ``combine_hidden_states`` touches."""
    torch.manual_seed(0)
    norms = None
    if tap_norm:
        norms = torch.nn.ModuleList(
            [
                RMSNorm(target_hidden_size, eps=1e-6)
                for _ in range(num_target_features)
            ]
        )
        with torch.no_grad():
            for index, norm in enumerate(norms):
                # Distinct gains so a tap-index swap cannot pass.
                norm.weight.fill_(0.5 + index)
    model = SimpleNamespace(
        is_flare=True,
        num_target_features=num_target_features,
        target_hidden_size=target_hidden_size,
        aux_hidden_norms=norms,
        layer_fusion_weights=torch.randn(num_layers, num_target_features),
        config=SimpleNamespace(num_hidden_layers=num_layers),
    )
    return SimpleNamespace(model=model)


def _reference_flare_fusion(
    holder: SimpleNamespace, hidden_states: torch.Tensor
) -> torch.Tensor:
    model = holder.model
    taps = hidden_states.view(
        -1, model.num_target_features, model.target_hidden_size
    )
    if model.aux_hidden_norms is not None:
        taps = torch.stack(
            [
                _reference_rms_norm(taps[:, index, :], norm.weight)
                for index, norm in enumerate(model.aux_hidden_norms)
            ],
            dim=1,
        )
    fusion = torch.softmax(model.layer_fusion_weights, dim=1)
    return torch.matmul(fusion, taps).reshape(
        -1, model.config.num_hidden_layers * model.target_hidden_size
    )


@torch.inference_mode()
def test_target_tap_norm_normalizes_each_tap_before_the_fusion() -> None:
    """dflash_config.target_tap_norm must run per tap, before the softmax mix."""
    holder = _tap_norm_model(tap_norm=True)
    hidden = torch.randn(4, 3 * 8)
    expected = _reference_flare_fusion(holder, hidden)
    actual = Qwen3DominoForCausalLM.combine_hidden_states(holder, hidden)
    assert actual.shape == (4, 2 * 8)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    # A single row takes the squeeze path but the same normalization.
    single = hidden[0]
    torch.testing.assert_close(
        Qwen3DominoForCausalLM.combine_hidden_states(holder, single),
        _reference_flare_fusion(holder, single.unsqueeze(0)).squeeze(0),
        rtol=1e-5,
        atol=1e-6,
    )


@torch.inference_mode()
def test_without_the_flag_the_raw_concat_is_fused() -> None:
    """The default path is bit-identical to the unrestricted flare fusion."""
    holder = _tap_norm_model(tap_norm=False)
    hidden = torch.randn(4, 3 * 8)
    taps = hidden.view(4, 3, 8)
    fusion = torch.softmax(holder.model.layer_fusion_weights, dim=1)
    expected = torch.matmul(fusion, taps).reshape(4, 2 * 8)
    actual = Qwen3DominoForCausalLM.combine_hidden_states(holder, hidden)
    assert torch.equal(actual, expected)


class _TargetTapNormKeyHolder(torch.nn.Module):
    """Replicates the served draft key path (params under ``model.``)."""

    def __init__(self, *, tap_norm: bool) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.fc = torch.nn.Linear(24, 8, bias=False)
        self.model.aux_hidden_norms = (
            torch.nn.ModuleList([RMSNorm(8, eps=1e-6) for _ in range(3)])
            if tap_norm
            else None
        )

    def load_weights(self, weights):
        mapped = []
        for name, weight in weights:
            if name.startswith("model."):
                name = name[len("model.") :]
            if "lm_head" not in name:
                name = "model." + name
            mapped.append((name, weight))
        return AutoWeightsLoader(self).load_weights(mapped)


def test_target_tap_norm_checkpoint_keys_load_under_the_model_prefix() -> None:
    holder = _TargetTapNormKeyHolder(tap_norm=True)
    weights = [
        (f"aux_hidden_norms.{index}.weight", torch.full((8,), 1.0 + index))
        for index in range(3)
    ]
    weights.append(("fc.weight", torch.zeros(8, 24)))
    loaded = holder.load_weights(weights)
    assert {f"model.aux_hidden_norms.{index}.weight" for index in range(3)} <= loaded
    for index in range(3):
        assert torch.equal(
            holder.model.aux_hidden_norms[index].weight,
            torch.full((8,), 1.0 + index),
        )


def test_target_tap_norm_keys_fail_loudly_without_the_flag() -> None:
    """A flag-off model must not silently drop trained normalization."""
    holder = _TargetTapNormKeyHolder(tap_norm=False)
    with pytest.raises(ValueError, match="aux_hidden_norms"):
        holder.load_weights(
            [
                ("aux_hidden_norms.0.weight", torch.ones(8)),
                ("fc.weight", torch.zeros(8, 24)),
            ]
        )


def _reference_rms_norm_into(
    out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    """Stand-in for ``ops.rms_norm``: fp32 reduction, weight on last dims."""
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    normed = xf * torch.rsqrt(variance + eps)
    if weight.numel() == x.shape[-1]:
        normed = normed * weight
    else:
        pad = (1,) * (x.dim() - weight.dim())
        normed = normed * weight.reshape(
            *weight.shape[:-1], *pad, weight.shape[-1]
        )
    out.copy_(normed.to(out.dtype))


def _fc_holder(
    *,
    tap_norm: bool,
    num_target_features: int = 3,
    target_hidden_size: int = 8,
    draft_hidden: int = 12,
) -> SimpleNamespace:
    """The subset of Qwen3DominoModel that the fc fusion touches."""
    torch.manual_seed(0)
    norms = None
    if tap_norm:
        norms = torch.nn.ModuleList(
            [
                RMSNorm(target_hidden_size, eps=1e-6)
                for _ in range(num_target_features)
            ]
        )
        with torch.no_grad():
            for index, norm in enumerate(norms):
                norm.weight.fill_(0.5 + index)
    fc = torch.nn.Linear(
        num_target_features * target_hidden_size, draft_hidden
    )
    with torch.no_grad():
        fc.weight.normal_(0, 0.05)
    model = SimpleNamespace(
        is_flare=False,
        num_target_features=num_target_features,
        target_hidden_size=target_hidden_size,
        context_hidden_size=draft_hidden,
        aux_hidden_norms=norms,
        fc=fc,
        hidden_norm=RMSNorm(draft_hidden, eps=1e-6),
        layer_fusion_weights=None,
        config=SimpleNamespace(num_hidden_layers=2, hidden_size=draft_hidden),
    )
    return SimpleNamespace(model=model)


@torch.inference_mode()
def test_fc_fusion_projects_the_normalized_taps_to_draft_width() -> None:
    """fusion_mode='fc' returns one shared draft-width context feature."""
    holder = _fc_holder(tap_norm=True)
    hidden = torch.randn(4, 3 * 8)
    taps = hidden.view(4, 3, 8)
    normed = torch.stack(
        [
            _reference_rms_norm(taps[:, index, :], norm.weight)
            for index, norm in enumerate(holder.model.aux_hidden_norms)
        ],
        dim=1,
    )
    expected = _reference_rms_norm(
        holder.model.fc(normed.reshape(4, -1)), holder.model.hidden_norm.weight
    )
    actual = Qwen3DominoForCausalLM.combine_hidden_states(holder, hidden)
    assert actual.shape == (4, holder.model.context_hidden_size)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    # A single row takes the squeeze path but the same projection.
    torch.testing.assert_close(
        Qwen3DominoForCausalLM.combine_hidden_states(holder, hidden[0]),
        expected[0],
        rtol=1e-5,
        atol=1e-6,
    )


def _fake_projection_model(
    *, is_flare: bool, num_layers: int = 2
) -> SimpleNamespace:
    """The subset of Qwen3DominoModel that ``_project_context_kv`` touches."""
    torch.manual_seed(0)
    draft_hidden, target_hidden = 12, 8
    num_kv_heads, head_dim = 2, 4
    kv_size = num_kv_heads * head_dim
    width = target_hidden if is_flare else draft_hidden
    model = SimpleNamespace(
        is_flare=is_flare,
        context_hidden_size=width,
        _hidden_norm_weight=torch.rand(width) + 0.5,
        _hidden_norm_eps=1e-6,
        _k_norm_weights=torch.rand(num_layers, head_dim) + 0.5,
        _rms_norm_eps=1e-6,
    )
    if is_flare:
        model._fused_kv_weight_T = (
            torch.randn(num_layers, target_hidden, 2 * kv_size) * 0.1
        )
        model._fused_kv_bias = None
    else:
        model._fused_kv_weight_flat = (
            torch.randn(num_layers * 2 * kv_size, draft_hidden) * 0.1
        )
        model._fused_kv_bias_flat = None
    return model


@pytest.mark.parametrize("is_flare", [False, True])
@torch.inference_mode()
def test_context_kv_projection_matches_the_reference(is_flare: bool) -> None:
    """Both fusion layouts must project and split K/V with layers outermost."""
    model = _fake_projection_model(is_flare=is_flare, num_layers=2)
    num_layers, num_ctx, nkv, hd = 2, 5, 2, 4
    width = model.context_hidden_size
    with patch.object(qwen3_domino.ops, "rms_norm", _reference_rms_norm_into):
        if is_flare:
            target = torch.randn(num_ctx, num_layers * width)
            normed = _reference_rms_norm(
                target.reshape(num_ctx * num_layers, width),
                model._hidden_norm_weight,
            ).view(num_ctx, num_layers, width)
            expected = torch.bmm(
                normed.permute(1, 0, 2), model._fused_kv_weight_T
            ).view(num_layers, num_ctx, 2, nkv, hd)
        else:
            target = torch.randn(num_ctx, width)
            normed = _reference_rms_norm(target, model._hidden_norm_weight)
            expected = (
                (normed @ model._fused_kv_weight_flat.t())
                .view(num_ctx, num_layers, 2, nkv, hd)
                .permute(1, 0, 2, 3, 4)
            )
        all_k, all_v = Qwen3DominoModel._project_context_kv(
            model, target, num_ctx, num_layers, nkv, hd
        )
    assert all_k.shape == (num_layers, num_ctx, nkv, hd)
    torch.testing.assert_close(all_k, expected[:, :, 0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(all_v, expected[:, :, 1], rtol=1e-5, atol=1e-6)


@torch.inference_mode()
def test_context_kv_grouped_k_norm_keeps_the_fc_layout() -> None:
    model = _fake_projection_model(is_flare=False, num_layers=2)
    all_k = torch.randn(2, 5, 2, 4)
    with patch.object(qwen3_domino.ops, "rms_norm", _reference_rms_norm_into):
        normed_k = Qwen3DominoModel._normalize_context_k(model, all_k)
    torch.testing.assert_close(
        normed_k,
        _reference_rms_norm(all_k, model._k_norm_weights),
        rtol=1e-5,
        atol=1e-6,
    )


@torch.inference_mode()
def test_precompute_rejects_the_wrong_context_layout() -> None:
    fc_model = _fake_projection_model(is_flare=False)
    fc_model._num_attn_layers = 2
    fc_model._use_fused_context_kv = True
    with pytest.raises(ValueError, match="fc-fused context states"):
        Qwen3DominoModel.precompute_and_store_context_kv(
            fc_model, torch.randn(3, 2 * fc_model.context_hidden_size), None
        )

    flare_model = _fake_projection_model(is_flare=True)
    flare_model._num_attn_layers = 2
    flare_model._use_fused_context_kv = True
    with pytest.raises(ValueError, match="flare-fused context states"):
        Qwen3DominoModel.precompute_and_store_context_kv(
            flare_model, torch.randn(3, flare_model.context_hidden_size), None
        )
