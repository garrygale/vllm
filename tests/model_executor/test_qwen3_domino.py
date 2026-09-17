# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.qwen3_domino import (
    DominoDraftAttention,
    FoldedSoftmaxMLP,
    FoldedSoftmaxReadout,
    Qwen3DominoModel,
    SharedGLUMLP,
    resolve_ffn_sharing,
    resolve_folded_readout,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

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


def test_resolve_folded_readout_defaults_match_training():
    # No knob (or an explicit dense mode) keeps today's Qwen3MLP.
    assert resolve_folded_readout(_draft_config()) is None
    assert (
        resolve_folded_readout(
            _draft_config(dflash_config={"ffn_readout": "dense"})
        )
        is None
    )
    assert (
        resolve_folded_readout(
            _draft_config(dflash_config={"ffn_readout": None})
        )
        is None
    )
    # 35B-A3B draft: 9728 / 2560 = 3.8 -> 4 chunks of 2432, K = 16.
    assert resolve_folded_readout(
        _draft_config(dflash_config={"ffn_readout": "folded_softmax"})
    ) == (4, 16)
    assert resolve_folded_readout(
        _draft_config(
            dflash_config={
                "ffn_readout": {"branches": 8, "granularity": 32}
            }
        )
    ) == (8, 32)
    # A 3N intermediate keeps the 3N -> N -> N shape of the design note.
    assert resolve_folded_readout(
        _draft_config(
            hidden_size=4096,
            intermediate_size=12288,
            dflash_config={"ffn_readout": "folded_softmax"},
        )
    ) == (3, 16)


def test_resolve_folded_readout_rejects_invalid_knobs():
    with pytest.raises(ValueError):
        resolve_folded_readout(
            _draft_config(dflash_config={"ffn_readout": "not_a_mode"})
        )
    with pytest.raises(ValueError):
        resolve_folded_readout(
            _draft_config(dflash_config={"ffn_readout": {"branches": 5}})
        )
    with pytest.raises(ValueError):
        resolve_folded_readout(
            _draft_config(
                dflash_config={
                    "ffn_readout": {"branches": 4, "granularity": 48}
                }
            )
        )
    with pytest.raises(ValueError):
        resolve_folded_readout(
            _draft_config(dflash_config={"ffn_readout": ["folded_softmax"]})
        )


def test_folded_readout_mixture_matches_the_training_formula():
    """The served mixture must be the chunk-wise softmax average it trained as."""

    readout = object.__new__(FoldedSoftmaxReadout)
    torch.nn.Module.__init__(readout)
    readout.hidden_size = 4
    readout.intermediate_size = 12
    readout.branches = 3
    readout.granularity = 2
    readout.folded_size = 4
    readout.repeats = 2
    readout.fold_logits = torch.nn.Parameter(torch.randn(3, 2))
    readout.proj = torch.nn.Linear(4, 4, bias=False)

    hidden_states = torch.randn(5, 12)
    out = readout(hidden_states)

    weights = torch.softmax(readout.fold_logits.float(), dim=0)
    chunks = hidden_states.reshape(5, 3, 2, 2)
    mixed = (chunks * weights.view(3, 1, 2)).sum(dim=1).reshape(5, 4)
    expected = torch.nn.functional.linear(mixed, readout.proj.weight)
    assert torch.allclose(out, expected, atol=1e-6)


@torch.inference_mode()
@pytest.mark.parametrize("device", DEVICES)
def test_folded_softmax_mlp_shapes_and_names(
    default_vllm_config, dist_init, device
) -> None:
    if current_platform.is_cuda_alike():
        torch.accelerator.set_device_index(device)
    torch.set_default_device(device)

    mlp = FoldedSoftmaxMLP(
        hidden_size=8,
        intermediate_size=24,
        hidden_act="silu",
        branches=3,
        granularity=4,
        prefix="layers.0.mlp",
    )
    parameters = dict(mlp.named_parameters())
    assert parameters["gate_up_proj.weight"].shape == (48, 8)
    # folded width = 24 / 3; the projection is an ordinary (quantizable) linear.
    assert parameters["down_proj.proj.weight"].shape == (8, 8)
    assert parameters["down_proj.fold_logits"].shape == (3, 4)
    # Weight names stay aligned with the SpecForge export.
    state_dict = mlp.state_dict()
    assert "down_proj.proj.weight" in state_dict
    assert "down_proj.fold_logits" in state_dict

    hidden_states = torch.randn(4, 8)
    gate_up, _ = mlp.gate_up_proj(hidden_states)
    hidden = mlp.act_fn(gate_up)
    # Uniform logits (the init) are the plain chunk average.
    chunks = hidden.reshape(4, 3, 2, 4)
    mixed = chunks.mean(dim=1).reshape(4, 8)
    expected = torch.nn.functional.linear(
        mixed, mlp.down_proj.proj.weight
    )
    assert torch.allclose(mlp(hidden_states), expected, atol=1e-2)


def test_resolve_ffn_sharing_defaults_match_training():
    # No knob keeps today's Qwen3MLP; each side defaults to no sharing.
    assert resolve_ffn_sharing(_draft_config()) is None
    assert resolve_ffn_sharing(_draft_config(dflash_config={"ffn_sharing": None})) is None
    assert resolve_ffn_sharing(_draft_config(dflash_config={"ffn_sharing": False})) is None
    # Pure gate sharing k=2 on the 35B-A3B draft.
    assert resolve_ffn_sharing(
        _draft_config(dflash_config={"ffn_sharing": {"gate_groups": 4864}})
    ) == ("nested", 4864, 9728)
    # The exactly-once outer lattice that keeps 9728 = 76 * 128.
    assert resolve_ffn_sharing(
        _draft_config(
            dflash_config={
                "ffn_sharing": {"pairing": "outer", "gate_groups": 76, "up_groups": 128}
            }
        )
    ) == ("outer", 76, 128)


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
def test_shared_glu_mlp_composes_with_the_folded_readout(
    default_vllm_config, dist_init, device
) -> None:
    if current_platform.is_cuda_alike():
        torch.accelerator.set_device_index(device)
    torch.set_default_device(device)

    mlp = SharedGLUMLP(
        hidden_size=8,
        intermediate_size=24,
        hidden_act="silu",
        gate_groups=12,
        up_groups=24,
        folded_readout=(3, 4),
        prefix="layers.0.mlp",
    )
    assert isinstance(mlp.down_proj, FoldedSoftmaxReadout)
    assert mlp.down_proj.proj.weight.shape == (8, 8)
    assert mlp.down_proj.fold_logits.shape == (3, 4)
    assert mlp(torch.randn(4, 8)).shape == (4, 8)


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
