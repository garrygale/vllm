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
    resolve_folded_readout,
)
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
    readout.tp_size = 1
    readout.tp_rank = 0
    readout.local_hidden_size = 12
    readout.local_input_size = 4
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


# (hidden, intermediate, branches, granularity, tp) -- the first two divide tp
# into whole chunks, the last two leave a rank with a partial chunk, which
# switches the readout onto its scatter path.
TP_CASES = (
    (8, 32, 4, 8, 2),
    (8, 32, 4, 8, 4),
    (8, 96, 3, 8, 2),
    (8, 64, 4, 8, 8),
)


def _rank_readout(hidden, intermediate, branches, granularity, rank, tp):
    """One rank's readout with the distributed bits filled in by hand."""

    readout = object.__new__(FoldedSoftmaxReadout)
    torch.nn.Module.__init__(readout)
    readout.hidden_size = hidden
    readout.intermediate_size = intermediate
    readout.branches = branches
    readout.granularity = granularity
    readout.folded_size = intermediate // branches
    readout.repeats = readout.folded_size // granularity
    readout.tp_size = tp
    readout.tp_rank = rank
    readout.local_hidden_size = intermediate // tp
    readout.local_input_size = readout.folded_size // tp
    readout.proj = _StubRowParallelLinear(readout.local_input_size, hidden)
    readout.fold_logits = torch.nn.Parameter(
        torch.zeros(branches, granularity)
    )
    readout._init_tp_mixture()
    return readout


def test_sharded_mixture_reconstructs_the_unsharded_formula():
    for hidden, intermediate, branches, granularity, tp in TP_CASES:
        with torch.inference_mode():
            torch.manual_seed(hidden + intermediate + tp)
            folded_size = intermediate // branches
            logits = torch.randn(branches, granularity)
            gated = torch.randn(2, 5, intermediate)

            # Reference: one rank holding the whole gated hidden.
            weights = torch.softmax(logits, dim=0)
            offsets = torch.arange(folded_size) % granularity
            chunks = gated.reshape(2, 5, branches, folded_size)
            expected = (chunks * weights[:, offsets]).sum(dim=2)

            partials = []
            for rank in range(tp):
                readout = _rank_readout(
                    hidden, intermediate, branches, granularity, rank, tp
                )
                readout.fold_logits.copy_(logits)
                width = intermediate // tp
                local = gated[..., rank * width : (rank + 1) * width]
                partials.append(readout._local_mixture(local))

            # Whole-chunk shards use the reshape path, partial shards the
            # scatter path; make sure both are actually covered.
            aligned = (branches % tp == 0) and (
                (intermediate // tp) % folded_size == 0
            )
            assert _rank_readout(
                hidden, intermediate, branches, granularity, 0, tp
            ).aligned_shards == aligned

            # The all-reduce the module performs is just a sum over ranks.
            total = torch.stack(partials).sum(dim=0).reshape(2, 5, folded_size)
            assert torch.allclose(total, expected, atol=1e-5)
