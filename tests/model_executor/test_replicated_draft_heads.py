# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated (disable_tp) draft correction heads.

The domino correction fc2, the folded shared-readout head and the dspark
markov tables are replicated: they are read once per sequential draft
position, and a vocab shard would add a per-step all-reduce/all-gather. These
tests pin the layer semantics (full tables, collective-free lookups) without
a distributed group: the TP helpers are monkeypatched to simulate TP>1, which
disable_tp layers must ignore entirely.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm.model_executor.layers.vocab_parallel_embedding as vpe_module
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead

V, D, R = 64, 8, 4  # V is a multiple of the default vocab padding (64)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def fake_tp():
    """Pretend the global TP group has 2 ranks (this process is rank 0).

    No real group is initialized, so anything that still consults the
    helpers (a sharded layer, a collective) fails loudly.
    """
    patch = pytest.MonkeyPatch()
    patch.setattr(vpe_module, "get_tensor_model_parallel_rank", lambda: 0)
    patch.setattr(vpe_module, "get_tensor_model_parallel_world_size", lambda: 2)
    yield 2
    patch.undo()


@pytest.fixture
def fake_vllm_config():
    # LogitsProcessor only reads model_config.head_dtype; a minimal config
    # avoids VllmConfig()'s platform device inference.
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(SimpleNamespace(model_config=None)):
        yield


def test_disable_tp_layer_ignores_global_tp(fake_tp):
    sharded = VocabParallelEmbedding(V, D, prefix="sharded")
    assert sharded.tp_size == 2
    assert sharded.weight.shape[0] == V // 2

    layer = VocabParallelEmbedding(V, D, prefix="emb", disable_tp=True)
    assert layer.disable_tp
    assert layer.tp_size == 1
    assert layer.weight.shape[0] == V

    with torch.no_grad():
        layer.weight.copy_(torch.arange(V * D, dtype=torch.float32).view(V, D))
    ids = torch.tensor([0, 5, V - 1])
    # tp_size == 1: plain local lookup, no masked fill, no all-reduce.
    torch.testing.assert_close(layer(ids), F.embedding(ids, layer.weight))


def test_disable_tp_head_loads_the_full_table(fake_tp):
    head = ParallelLMHead(V, D, prefix="head", disable_tp=True)
    full = torch.randn(V, D)
    head.weight_loader(head.weight, full)
    torch.testing.assert_close(head.weight.data, full)


def test_logits_gather_skipped_for_replicated_head(
    fake_tp, fake_vllm_config, monkeypatch
):
    lp = LogitsProcessor(V)

    replicated = ParallelLMHead(V, D, prefix="replicated", disable_tp=True)
    sharded = ParallelLMHead(V, D, prefix="sharded")
    assert sharded.tp_size == 2

    gathered = []
    monkeypatch.setattr(
        LogitsProcessor, "_gather_logits", lambda self, logits: gathered.append(1)
    )

    hidden = torch.randn(3, D)
    full = lp(replicated, hidden)
    assert not gathered
    torch.testing.assert_close(full, F.linear(hidden, replicated.weight))

    # A vocab-sharded head still enters the gather path.
    assert lp(sharded, hidden) is None
    assert gathered


def test_dspark_markov_head_is_replicated(fake_tp, fake_vllm_config):
    head = DSparkMarkovHead(V, V, R, prefix="markov_head")
    assert isinstance(head.markov_w1, nn.Embedding)
    assert head.markov_w1.weight.shape == (V, R)
    assert head.markov_w2.disable_tp
    assert head.markov_w2.tp_size == 1
    assert head.markov_w2.weight.shape[0] == V

    ids = torch.tensor([1, V - 1])
    embed = head.embed(ids)
    torch.testing.assert_close(embed, F.embedding(ids, head.markov_w1.weight))
    bias = head.bias(embed, LogitsProcessor(V))
    torch.testing.assert_close(bias, F.linear(embed, head.markov_w2.weight))


def test_shared_readout_fold_assembles_full_table_at_tp2(
    fake_tp, fake_vllm_config, monkeypatch
):
    import vllm.model_executor.models.qwen3_domino as domino_module

    H_t, M = 6, 4
    full = torch.randn(V, H_t)
    head = ParallelLMHead(V, H_t, prefix="lm_head")

    # Simulate the sharded serving state: this rank holds shard 0 and the
    # fold's one-time all-gather returns the other shard around it.
    half0, half1 = full.chunk(2, dim=0)
    head.weight = nn.Parameter(half0.clone(), requires_grad=False)
    head.tp_size = 2
    monkeypatch.setattr(
        domino_module,
        "tensor_model_parallel_all_gather",
        lambda t, dim=0: torch.cat([t, half1.to(t.dtype)], dim=dim),
    )

    up_w = torch.randn(H_t, M)
    draft = SimpleNamespace(
        model=SimpleNamespace(head_up_proj=SimpleNamespace(weight=SimpleNamespace(data=up_w))),
        lm_head=head,
    )

    domino_module._fold_shared_emb_readout(draft)
    folded = draft.lm_head
    assert folded.disable_tp
    assert folded.tp_size == 1
    torch.testing.assert_close(folded.weight.data, full @ up_w)
    assert draft.model.head_up_proj is None
