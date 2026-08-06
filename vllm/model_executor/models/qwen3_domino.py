# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 Domino draft model for speculative decoding.

Domino keeps the DFlash/DSpark block-parallel draft backbone and adds a
lightweight causal correction head.  The correction head consists of:

  * a prefix GRU that summarizes the already-sampled draft prefix, and
  * ``embed_proj``, a low-rank residual logit correction
    ``Linear(hidden + gru) -> emb_dim -> vocab``.

The checkpoint format follows SpecForge's ``DFlashDraftModel`` with
``dflash_config.projector_type == "domino"``.  The dflare variant additionally
uses per-draft-layer softmax fusion of the target hidden states
(``fusion_mode == "flare"``) and heterogeneous K/V projections
(``heterogeneous_kv == true``).
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config

from vllm.config import VllmConfig, get_current_vllm_config, replace
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta

from .qwen2 import Qwen2MLP as Qwen3MLP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
)

logger = init_logger(__name__)


def _dflash_config(config: Qwen3Config) -> dict:
    return getattr(config, "dflash_config", None) or {}


def _is_domino(config: Qwen3Config) -> bool:
    return _dflash_config(config).get("projector_type") == "domino"


class DominoQwen3Attention(nn.Module):
    """Qwen3 attention with separate draft and target K/V projections.

    The draft query block uses ``q_proj``/``k_proj``/``v_proj``.  The target
    context K/V are projected separately with ``k_proj_target``/``v_proj_target``
    and are pre-inserted into the KV cache before the draft forward, following
    the DFlash contract.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        target_hidden_size: int,
        rope_parameters: dict,
        max_position: int,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.hidden_size = hidden_size
        self.target_hidden_size = target_hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
            return_bias=False,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
            return_bias=False,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
            return_bias=False,
        )
        self.k_proj_target = ColumnParallelLinear(
            target_hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj_target",
            return_bias=False,
        )
        self.v_proj_target = ColumnParallelLinear(
            target_hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj_target",
            return_bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            return_bias=False,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        # Domino draft blocks are non-causal: every query position sees the
        # whole context block and the current draft block.
        self.causal = False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output = self.o_proj(attn_output)
        return output


class DominoQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: Qwen3Config,
        layer_idx: int,
        target_hidden_size: int,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)

        self.self_attn = DominoQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            target_hidden_size=target_hidden_size,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3DominoModel(nn.Module):
    """Qwen3 Domino draft model.

    The parallel backbone is a Qwen3 decoder stack whose attention reads its
    context K/V from the cache (pre-populated by the proposer).  The model
    additionally contains the flare fusion weights, input/output projections,
    the prefix GRU, and the low-rank correction head.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.start_layer_id = start_layer_id
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        dflash_config = _dflash_config(self.config)
        if not _is_domino(self.config):
            raise ValueError(
                "Qwen3DominoModel requires dflash_config.projector_type == 'domino'"
            )
        if dflash_config.get("fusion_mode") != "flare":
            raise NotImplementedError(
                "Only fusion_mode='flare' is supported by Qwen3DominoModel"
            )
        if dflash_config.get("use_correction_target_fusion", False):
            raise NotImplementedError(
                "use_correction_target_fusion is not implemented yet"
            )

        self.target_layer_ids = dflash_config.get("target_layer_ids")
        if not self.target_layer_ids:
            raise ValueError("dflash_config.target_layer_ids must be set")
        self.num_target_features = len(self.target_layer_ids)
        self.target_hidden_size = dflash_config.get(
            "target_hidden_size", self.config.hidden_size
        )
        self.pure_draft_prefix_len = dflash_config.get("pure_draft_prefix_len", 0)
        self.mask_token_id = dflash_config.get("mask_token_id")

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.target_hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.input_proj = None
        self.output_proj = None
        if self.target_hidden_size != self.config.hidden_size:
            self.input_proj = ReplicatedLinear(
                self.target_hidden_size,
                self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "input_proj"),
                return_bias=False,
            )
            self.output_proj = ReplicatedLinear(
                self.config.hidden_size,
                self.target_hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "output_proj"),
                return_bias=False,
            )

        current_vllm_config = get_current_vllm_config()
        self.layers = nn.ModuleList(
            [
                DominoQwen3DecoderLayer(
                    current_vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    target_hidden_size=self.target_hidden_size,
                    cache_config=current_vllm_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )

        # Flare fusion: per-draft-layer softmax combination of target features.
        self.layer_fusion_weights = nn.Parameter(
            torch.empty(
                self.config.num_hidden_layers,
                self.num_target_features,
                dtype=vllm_config.model_config.dtype,
            )
        )
        self._init_fusion_weights()
        self.hidden_norm = RMSNorm(
            self.target_hidden_size, eps=self.config.rms_norm_eps
        )
        self.norm = RMSNorm(
            self.target_hidden_size, eps=self.config.rms_norm_eps
        )

        # Domino correction head.
        self.gru_hidden_dim = dflash_config["gru_hidden_dim"]
        self.prefix_gru = nn.GRU(
            input_size=self.target_hidden_size,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )

        use_embed_proj = dflash_config.get("use_embed_proj", True)
        if use_embed_proj:
            self.emb_dim = dflash_config["emb_dim"]
            in_dim = self.target_hidden_size + self.gru_hidden_dim
            self.embed_proj = nn.Sequential(
                ReplicatedLinear(
                    in_dim,
                    self.emb_dim,
                    bias=False,
                    params_dtype=vllm_config.model_config.dtype,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "embed_proj.0"),
                    return_bias=False,
                ),
                nn.SiLU(),
                ParallelLMHead(
                    self.vocab_size,
                    self.emb_dim,
                    params_dtype=vllm_config.model_config.dtype,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "embed_proj.2"),
                ),
            )
        else:
            self.emb_dim = None
            self.embed_proj = None

        if dflash_config.get("use_hidden_proj", False):
            hidden_proj_dim = dflash_config.get(
                "hidden_proj_dim", self.emb_dim
            )
            if hidden_proj_dim is None:
                raise ValueError(
                    "use_hidden_proj=true requires hidden_proj_dim when "
                    "use_embed_proj=false"
                )
            in_dim = self.target_hidden_size + self.gru_hidden_dim
            self.hidden_proj = nn.Sequential(
                ReplicatedLinear(
                    in_dim,
                    hidden_proj_dim,
                    bias=False,
                    params_dtype=vllm_config.model_config.dtype,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "hidden_proj.0"),
                    return_bias=False,
                ),
                nn.SiLU(),
                ReplicatedLinear(
                    hidden_proj_dim,
                    self.target_hidden_size,
                    bias=False,
                    params_dtype=vllm_config.model_config.dtype,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "hidden_proj.2"),
                    return_bias=False,
                ),
            )
        else:
            self.hidden_proj = None

        if self.embed_proj is None and self.hidden_proj is None:
            raise ValueError(
                "Domino requires at least one of embed_proj or hidden_proj"
            )

        # Optional NPU fp16 GRU parameter cache.  Populated by the Ascend
        # wrapper after weight loading.
        self._gru_fp16 = None

    def _init_fusion_weights(self) -> None:
        nn.init.constant_(self.layer_fusion_weights, 0.0)
        D = self.layer_fusion_weights.shape[0]
        T = self.layer_fusion_weights.shape[1]
        for d in range(D):
            t = min(T - 1, int((d / D) * T))
            self.layer_fusion_weights.data[d, t] = 2.0

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        hidden_states = inputs_embeds
        if self.input_proj is not None:
            hidden_states = self.input_proj(hidden_states)

        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)

        if self.output_proj is not None:
            hidden_states = self.output_proj(hidden_states)
        return self.norm(hidden_states)

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3DominoForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)

        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = Qwen3DominoModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.model.target_hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )

        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(token_ids)

    @property
    def pure_draft_prefix_len(self) -> int:
        return self.model.pure_draft_prefix_len

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.compute_draft_logits(hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(
            self.config.draft_vocab_size, device=logits.device
        )
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Flare-fuse raw target aux features into per-draft-layer states.

        Input: ``[N, num_target_features * target_hidden_size]``.
        Output: ``[N, num_draft_layers * target_hidden_size]``.
        """
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)

        expected = (
            self.model.num_target_features * self.model.target_hidden_size
        )
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                "Domino drafter expects "
                f"{self.model.num_target_features}x{self.model.target_hidden_size} "
                f"({expected}) target aux features but received "
                f"{hidden_states.shape[-1]}"
            )

        target = hidden_states.view(
            -1,
            self.model.num_target_features,
            self.model.target_hidden_size,
        )
        fusion_w = torch.softmax(self.model.layer_fusion_weights, dim=1)
        # [D, T] x [N, T, H] -> [D, N, H] without materializing [N, D, T, H].
        fused = torch.einsum("dt,nth->dnh", fusion_w, target)
        fused = fused.permute(1, 0, 2).reshape(
            -1,
            self.model.config.num_hidden_layers * self.model.target_hidden_size,
        )
        if needs_squeeze:
            fused = fused.squeeze(0)
        return fused

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Project flare-fused context states and write them into draft KV caches.

        ``context_states`` is the output of :meth:`combine_hidden_states`,
        shaped ``[T, D * target_hidden_size]``.
        """
        if context_states.dim() != 2:
            raise ValueError(
                "Domino precompute expects 2D flare-fused context states, got "
                f"{context_states.shape}"
            )

        num_ctx = context_states.shape[0]
        D = self.model.config.num_hidden_layers
        H = self.model.target_hidden_size
        fused = context_states.view(num_ctx, D, H)
        per_layer = isinstance(context_slot_mapping, (list, tuple))

        for i, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            ctx = self.model.hidden_norm(fused[:, i, :])
            k = attn.k_proj_target(ctx)
            v = attn.v_proj_target(ctx)
            k_shape = k.shape
            k = attn.k_norm(
                k.view(
                    *k_shape[:-1],
                    k_shape[-1] // attn.head_dim,
                    attn.head_dim,
                )
            ).view(k_shape)
            # Ascend's rotary op requires a real key tensor (it does not accept
            # None like the CUDA/native path). Passing a clone is a no-op for
            # the key side and keeps the NPU path valid.
            k, _ = attn.rotary_emb(context_positions, k, k.clone())

            if context_slot_mapping is None:
                continue
            slot_mapping = (
                context_slot_mapping[i] if per_layer else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            attn.attn.impl.do_kv_cache_update(
                attn.attn,
                k.reshape(num_ctx, attn.num_kv_heads, attn.head_dim),
                v.reshape(num_ctx, attn.num_kv_heads, attn.head_dim),
                attn.attn.kv_cache,
                slot_mapping,
            )

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        return [layer.self_attn.causal for layer in self.model.layers]

    def correction_bias(self, features: torch.Tensor) -> torch.Tensor:
        """Domino residual logit correction for ``[B, hidden + gru]`` features."""
        bias = None
        if self.model.embed_proj is not None:
            x = self.model.embed_proj[0](features)
            x = self.model.embed_proj[1](x)
            bias = self.logits_processor(self.model.embed_proj[2], x)

        if self.model.hidden_proj is not None:
            h = self.model.hidden_proj(features)
            hidden_bias = self.logits_processor(self.lm_head, h)
            bias = hidden_bias if bias is None else bias + hidden_bias

        assert bias is not None
        return bias

    def _gru_weights(self):
        if self.model._gru_fp16 is not None:
            return (
                self.model._gru_fp16["weight_ih_l0"],
                self.model._gru_fp16["weight_hh_l0"],
            )
        return (
            self.model.prefix_gru.weight_ih_l0,
            self.model.prefix_gru.weight_hh_l0,
        )

    def domino_gru_cell(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[None, torch.Tensor]:
        """One GRU step.  ``x`` is ``[B, target_hidden_size]``, ``h`` is
        ``[1, B, gru_hidden_dim]``.  Returns ``(None, h_n)`` in the input dtype.
        """
        orig_dtype = x.dtype
        w_ih, w_hh = self._gru_weights()

        if self.model._gru_fp16 is not None:
            x = x.to(torch.float16)
            h = h.to(torch.float16)

        gi = F.linear(x, w_ih)
        gh = F.linear(h[0], w_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_new = ((1.0 - z) * n + z * h[0]).to(orig_dtype)
        return None, h_new.unsqueeze(0)

    def domino_gru_forward(
        self,
        embeds: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[None, torch.Tensor]:
        """Run the GRU over ``[B, seq, target_hidden_size]`` embeddings."""
        batch = embeds.shape[0]
        if h is None:
            h = embeds.new_zeros(1, batch, self.model.gru_hidden_dim)
        for t in range(embeds.shape[1]):
            _, h = self.domino_gru_cell(embeds[:, t], h)
        return None, h

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_embed_tokens = False
        includes_lm_head = False
        includes_draft_id_mapping = False

        for name, loaded_weight in weights:
            if name.startswith("model."):
                name = name[len("model."):]
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            else:
                if "lm_head" not in name:
                    name = "model." + name

            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name:
                includes_lm_head = True

            model_weights[name] = loaded_weight

        self.has_own_embed_tokens = includes_embed_tokens
        self.has_own_lm_head = includes_lm_head

        skip_substrs = ["fc.", "mask_embedding"]
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")

        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        loader.load_weights(model_weights.items())


def load_domino_model(
    target_model: nn.Module,
    vllm_config: VllmConfig,
) -> nn.Module:
    """Load the Domino draft model and share target embed/lm_head when absent."""
    from vllm.compilation.backends import set_model_tag
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
        _should_share,
        get_target_lm_head,
    )

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    if get_pp_group().world_size != 1:
        raise NotImplementedError(
            "Domino does not support pipeline parallelism."
        )

    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(
                draft_model_config.hf_config
            ),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )

    with set_model_tag("domino_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config,
            model_config=draft_model_config,
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
        target_inner, "embedding", None
    )
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if target_embed is not None and _should_share(
        draft_model, "has_own_embed_tokens", draft_embed, target_embed
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        draft_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
