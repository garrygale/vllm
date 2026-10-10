# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_domino import (
    DominoDraftAttention,
    Qwen3DominoForCausalLM,
    Qwen3DominoModel,
    RoutedOuterMLP,
    SharedGLUMLP,
    _fold_output_proj_into_lm_head,
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
