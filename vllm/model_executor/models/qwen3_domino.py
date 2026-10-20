# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 Domino draft model for speculative decoding.

Domino keeps the DFlash/DSpark block-parallel draft backbone and adds a
lightweight causal correction head.  The correction head consists of:

  * a prefix GRU that summarizes the already-sampled draft prefix, and
  * ``embed_proj``, a low-rank residual logit correction
    ``Linear(hidden + gru) -> emb_dim -> vocab``.

The checkpoint format follows SpecForge's ``DFlashDraftModel`` with
``dflash_config.projector_type == "domino"``.  Both target-fusion modes are
served:

  * ``fusion_mode == "flare"`` (dflare): per-draft-layer softmax fusion of the
    target hidden states, target-width context features, usually with
    heterogeneous K/V projections (``heterogeneous_kv == true``);
  * ``fusion_mode == "fc"`` (plain DFlash, also the SpecForge default when the
    key is absent): the concatenated target taps are projected to draft width
    by one ``fc`` linear and every draft layer reads that shared feature
    through its own K/V projections (shared with the draft stream unless
    ``heterogeneous_kv`` is set).

Both modes apply ``dflash_config.target_tap_norm`` (one RMSNorm per captured
tap) before the fusion when the checkpoint was trained with it.
"""

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config

from vllm import _custom_ops as ops
from vllm.config import VllmConfig, get_current_vllm_config, replace
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.forward_context import get_forward_context
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from .domino_hca_incremental import (
    HcaIncrementalState,
    append_pool_entry,
    window_anchor,
    window_phases,
    write_prefill_pool,
)
from .qwen2 import Qwen2MLP as Qwen3MLP
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
)

logger = init_logger(__name__)


class _DominoWeightsMapper(WeightsMapper):
    """WeightsMapper that leaves HCA compressor parameters alone.

    The generic ``.gate_proj -> .gate_up_proj`` MLP mapping also matches the
    compressor's ``hca_compressor.gate_proj``; that module has no fused
    ``gate_up_proj`` and its checkpoint name already matches the vLLM module.
    """

    def _map_name_with_shard(self, key: str):
        if ".hca_compressor." in key:
            return key, None
        return super()._map_name_with_shard(key)


def _dflash_config(config: Qwen3Config) -> dict:
    return getattr(config, "dflash_config", None) or {}


def _is_domino(config: Qwen3Config) -> bool:
    return _dflash_config(config).get("projector_type") == "domino"


def _domino_layer_attention(
    config: Qwen3Config, layer_idx: int
) -> tuple[int | None, bool]:
    """Resolve (sliding_window, causal) for one Domino draft layer.

    A layer uses sliding-window attention when its ``config.layer_types``
    entry is ``"sliding_attention"``; a ``"heavily_compressed_attention"``
    (HCA) layer resolves its raw-band window the same way.  The window is the
    per-layer entry of ``config.sliding_window`` (a list with one value per
    draft layer) or the scalar value shared by all sliding layers, mirroring
    SpecForge's ``get_layer_sliding_window``: ``-1``/missing/invalid entries
    resolve to ``None`` (full attention).

    Domino draft layers are trained with block-bidirectional (non-causal)
    attention on every layer, so the default is non-causal; a
    ``dflash_config.causal`` override wins when present.
    """
    dflash_config = _dflash_config(config)
    causal = bool(dflash_config.get("causal", False))

    layer_types = getattr(config, "layer_types", None)
    if not layer_types or layer_idx >= len(layer_types):
        return None, causal
    if layer_types[layer_idx] not in ("sliding_attention", "heavily_compressed_attention"):
        return None, causal

    # The config loader relocates a per-layer list from the top-level
    # `sliding_window` into `dflash_config.sliding_window` to satisfy
    # transformers strict field validation; read the relocated value
    # first and fall back to the top-level field.
    sliding_window = dflash_config.get("sliding_window")
    if sliding_window is None:
        sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        return None, causal
    if isinstance(sliding_window, (list, tuple)):
        if layer_idx >= len(sliding_window):
            return None, causal
        value = sliding_window[layer_idx]
    else:
        value = sliding_window
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None, causal
    return value, causal


def domino_has_any_non_causal(config: Qwen3Config) -> bool:
    """Whether any Domino draft layer is non-causal.

    Config mirror of the model's ``get_draft_attn_causal``, usable before the
    model is built.
    """
    return not all(
        _domino_layer_attention(config, layer_idx)[1]
        for layer_idx in range(config.num_hidden_layers)
    )


# ---------------------------------------------------------------------------
# Heavily compressed attention (HCA): compressed history entries beside the
# raw band (``dflash_config.hca``, layer_types "heavily_compressed_attention")
# ---------------------------------------------------------------------------
#
# Serving mirror of SpecForge's ``resolve_hca_config``/``DFlashHCACompressor``.
# A compressed layer attends [entries || raw band || draft block] with the R2
# seam rule: the band keeps the constant-width per-layer window W, entry j is
# visible to query t iff ``j*hop < t+1-W`` (and, past the entry budget, only
# the newest ``max_entries`` entries are kept).  The context-KV precompute is
# incremental (only this round's target tokens), so the compressor keeps a
# per-row running window state: prefill emits every closed window once, and
# decode advances the state one token at a time.

_HCA_COFF = {"vanilla": 1, "hat": 2}
HCA_WINDOW_BUDGET = 2048
_HCA_FIELDS = {"variants", "hops", "bias_mode", "max_entries"}


def _hat_window_kernel(win: int) -> torch.Tensor:
    idx = torch.arange(win, dtype=torch.float32)
    center = (win - 1) / 2
    kernel = torch.clamp_min(1.0 - (idx - center).abs() / (win / 2), min=0.0)
    return kernel / kernel.sum()


def resolve_domino_hca(config: Qwen3Config) -> Optional[tuple]:
    """Resolve dflash_config.hca into per-layer specs; None when disabled."""

    dflash_config = _dflash_config(config)
    layer_types = list(getattr(config, "layer_types", None) or [])
    num_layers = config.num_hidden_layers
    compressed = [
        idx
        for idx in range(num_layers)
        if layer_types[idx] == "heavily_compressed_attention"
    ]
    raw = dflash_config.get("hca")
    if not compressed:
        if raw is not None:
            raise ValueError(
                "dflash_config.hca is set but no layer_type is "
                "'heavily_compressed_attention'; remove the key or mark layers."
            )
        return None
    if raw is None:
        raise ValueError(
            "layer_types 'heavily_compressed_attention' requires "
            "dflash_config.hca with 'variants' and optional 'hops'."
        )
    if not isinstance(raw, dict):
        raise ValueError(
            f"dflash_config.hca must be a dict, got {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - _HCA_FIELDS)
    if unknown:
        raise ValueError(
            f"unknown dflash_config.hca entries {unknown}; expected a subset "
            f"of {sorted(_HCA_FIELDS)}"
        )
    if bool(dflash_config.get("causal", False)):
        raise ValueError(
            "HCA layers require the non-causal domino band (v1): the dense "
            "[entries || band || draft] path assumes block-bidirectional "
            "draft columns."
        )
    if int(dflash_config.get("attn_sink", 0) or 0):
        raise ValueError(
            "HCA layers do not support dflash_config.attn_sink (v1): sinks "
            "ride the paged cache as extra rows, which the dense HCA read "
            "path does not visit."
        )

    block_size = getattr(config, "block_size", None)
    if not isinstance(block_size, int) or isinstance(block_size, bool):
        raise ValueError("dflash_config.hca requires an integer block_size")

    variants = raw.get("variants", "vanilla")
    if isinstance(variants, str):
        variants = [variants] * len(compressed)
    if not isinstance(variants, (list, tuple)) or len(variants) != len(compressed):
        raise ValueError(
            f"dflash_config.hca.variants must be a variant or a list of "
            f"{len(compressed)} (one per compressed layer), got {variants!r}"
        )
    hops = raw.get("hops", 128)
    if isinstance(hops, int) and not isinstance(hops, bool):
        hops = [hops] * len(compressed)
    if not isinstance(hops, (list, tuple)) or len(hops) != len(compressed):
        raise ValueError(
            f"dflash_config.hca.hops must be an int or a list of "
            f"{len(compressed)} positive ints, got {hops!r}"
        )
    bias_mode = raw.get("bias_mode", "learnable")
    if bias_mode not in ("learnable", "frozen"):
        raise ValueError(
            f"dflash_config.hca.bias_mode must be 'learnable' or 'frozen', "
            f"got {bias_mode!r}"
        )
    max_entries = raw.get("max_entries")
    if max_entries is not None and (
        not isinstance(max_entries, int)
        or isinstance(max_entries, bool)
        or max_entries < 1
    ):
        raise ValueError(
            f"dflash_config.hca.max_entries must be a positive int, got "
            f"{max_entries!r}"
        )

    specs: list[Optional[dict]] = [None] * num_layers
    for slot, layer_idx in enumerate(compressed):
        variant = variants[slot]
        if variant not in _HCA_COFF:
            raise ValueError(
                f"dflash_config.hca.variants supports {sorted(_HCA_COFF)} for "
                f"layer {layer_idx}, got {variant!r}"
            )
        hop = hops[slot]
        if not isinstance(hop, int) or isinstance(hop, bool) or hop < 1:
            raise ValueError(
                f"dflash_config.hca.hops[{slot}] must be a positive int for "
                f"layer {layer_idx}, got {hop!r}"
            )
        coff = _HCA_COFF[variant]
        win = coff * hop
        window = _domino_layer_attention(config, layer_idx)[0]
        if window is None:
            raise ValueError(
                f"Layer {layer_idx} is heavily_compressed_attention but no "
                "positive sliding_window is configured for its raw band."
            )
        if window > HCA_WINDOW_BUDGET:
            raise ValueError(
                f"Layer {layer_idx} is heavily_compressed_attention with "
                f"sliding_window={window} > {HCA_WINDOW_BUDGET} (the served "
                "band budget); lower the window, the history goes compressed."
            )
        if window < win + block_size:
            raise ValueError(
                f"Layer {layer_idx} is heavily_compressed_attention with "
                f"window {window} < win({win}) + block_size({block_size}); "
                "the seam between band and entries would drop tokens."
            )
        entries = max_entries
        if entries is None:
            entries = HCA_WINDOW_BUDGET - window - block_size - 64
        if entries < 1:
            raise ValueError(
                f"Layer {layer_idx} has no compressed-entry budget left "
                f"(window {window} + block_size {block_size} consumes the "
                f"{HCA_WINDOW_BUDGET} attention budget)."
            )
        specs[layer_idx] = {
            "variant": variant,
            "hop": hop,
            "win": win,
            "coff": coff,
            "bias_mode": bias_mode,
            "max_entries": entries,
            "window": window,
        }
    return tuple(specs)


def _hca_open_windows(context_len: int, win: int, hop: int) -> range:
    """Windows that are still open after a context of ``context_len`` tokens.

    A window ``w`` covers ``[w*hop, w*hop + win)``.  Closed windows end at or
    before ``context_len`` and are handled by the one-shot prefill pool; the
    returned range is the (at most ``ceil(win/hop)``) tail of overlapping
    windows that still need running state.
    """

    first = max(0, (context_len - win) // hop + 1)
    last = (context_len - 1) // hop
    if context_len <= 0 or first > last:
        return range(0)
    return range(first, last + 1)


def _hca_cache_block_size(kv_caches) -> int:
    """Block size of the paged draft cache, matching the dense read unpacking."""

    if isinstance(kv_caches, torch.Tensor) and kv_caches.dim() == 5:
        return int(kv_caches.shape[2])
    if len(kv_caches) == 1:
        return int(kv_caches[0].shape[2])
    return int(kv_caches[0].shape[1])


def _hca_cache_num_blocks(kv_caches) -> int:
    """Total physical block count of the paged draft cache."""

    if isinstance(kv_caches, torch.Tensor):
        if kv_caches.dim() == 5:
            return int(kv_caches.shape[1])
        return int(kv_caches.shape[0])
    if len(kv_caches) == 1:
        return int(kv_caches[0].shape[0])
    return int(kv_caches[0].shape[0])


class DominoHCACompressor(nn.Module):
    """Per-layer compressed-entry pooler over the fused context feature.

    Same recipe as SpecForge's ``DFlashHCACompressor``: every closed ``win``
    window emits ``RMSNorm(sum_j softmax(gate_j + B)_j * kv_j)`` with one
    shared head. ``position_bias`` mirrors the training parameter verbatim
    (zero init = vanilla, log tent init = hat). K/V are stored unrotated; the
    owning attention rotates the K view at the anchor positions.
    """

    def __init__(
        self,
        context_hidden_size: int,
        head_dim: int,
        spec: dict,
        rms_norm_eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hop = spec["hop"]
        self.win = spec["win"]
        self.anchor_offset = 0 if spec["coff"] == 1 else spec["win"] // 2
        self.kv_proj = ReplicatedLinear(
            context_hidden_size,
            head_dim,
            bias=False,
            params_dtype=dtype,
            prefix=f"{prefix}.kv_proj",
            return_bias=False,
        )
        self.gate_proj = ReplicatedLinear(
            context_hidden_size,
            head_dim,
            bias=False,
            params_dtype=dtype,
            prefix=f"{prefix}.gate_proj",
            return_bias=False,
        )
        bias = torch.zeros(self.win, head_dim, dtype=torch.float32)
        if spec["variant"] == "hat":
            bias = bias + torch.log(_hat_window_kernel(self.win) + 1e-8).unsqueeze(
                -1
            )
        self.position_bias = nn.Parameter(bias, requires_grad=False)
        self.norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def pool(self, feature: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pool one request's context feature [S, H] into unrotated entries."""

        capacity = (feature.shape[0] - self.win) // self.hop + 1
        if capacity <= 0:
            return feature.new_zeros(0, self.kv_proj.output_size), 0
        kv = self.kv_proj(feature)[0]
        gate = self.gate_proj(feature)[0]
        # Index-gathered hop-strided windows, [capacity, win, hd]: same values
        # as tensor unfold, kept in the serving path so both sides share one
        # formulation (unfold backward has no NPU kernel).
        window_idx = (
            torch.arange(capacity, device=feature.device) * self.hop
        ).unsqueeze(1) + torch.arange(self.win, device=feature.device)
        kv_windows = kv[window_idx]
        gate_windows = gate[window_idx]
        weights = torch.softmax(
            gate_windows + self.position_bias,
            dim=1,
            dtype=torch.float32,
        ).to(kv_windows.dtype)
        pooled = self.norm((kv_windows * weights).sum(dim=1))
        return pooled, capacity

    def anchor_positions(self, num_entries: int, device) -> torch.Tensor:
        return (
            torch.arange(num_entries, device=device) * self.hop + self.anchor_offset
        )


class DominoDraftAttention(Attention):
    """Draft attention with a full (non-evicted) KV cache.

    Sliding-window draft layers keep the full context in the KV cache: the
    DFlash contract incrementally appends each round's new context K/V through
    position-aligned block tables, and window-block eviction would break that
    slot mapping.  The cache allocation and manager must therefore see a
    single, ordinary full-attention layout for every draft layer.  The
    per-layer window is still applied at compute time through
    ``per_layer_sliding_window`` on the Attention implementation.
    """

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if isinstance(spec, SlidingWindowSpec):
            return FullAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=spec.num_kv_heads,
                head_size=spec.head_size,
                head_size_v=spec.head_size_v,
                dtype=spec.dtype,
                kv_quant_mode=spec.kv_quant_mode,
                sliding_window=None,
            )
        return spec


class DominoQwen3Attention(nn.Module):
    """Qwen3 attention with optional dedicated target K/V projections.

    The draft query block uses ``q_proj``/``k_proj``/``v_proj``.  With
    heterogeneous K/V (``target_hidden_size`` given) the target context K/V are
    projected separately with ``k_proj_target``/``v_proj_target`` and are
    pre-inserted into the KV cache before the draft forward, following the
    DFlash contract.  With ``target_hidden_size=None`` the context is projected
    by the same shared ``k_proj``/``v_proj`` as the draft stream (fc fusion).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        target_hidden_size: int | None,
        rope_parameters: dict,
        max_position: int,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        sliding_window: int | None = None,
        causal: bool = False,
        num_sinks: int = 0,
        hca: dict | None = None,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.sliding_window = sliding_window
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
        self.k_proj_target = None
        self.v_proj_target = None
        if target_hidden_size is not None:
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

        self.attn = DominoDraftAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        # Learned attention sinks (dflash_config.attn_sink). Full-size over
        # all KV heads; the TP shard is sliced at write time. Raw constants:
        # no k_norm, no RoPE — written straight into the draft cache as one
        # extra row one position past the draft block, which only block-
        # bidirectional (non-causal) layers can see.
        self.num_sinks = num_sinks
        if self.num_sinks < 0:
            raise ValueError(
                f"dflash_config.attn_sink must be >= 0, got {self.num_sinks}"
            )
        if self.num_sinks:
            if causal:
                raise ValueError(
                    "dflash_config.attn_sink requires non-causal draft layers "
                    "in v1: the sink row sits one position past the draft "
                    "block and is only reachable through block-bidirectional "
                    "attention"
                )
            self.sink_k = nn.Parameter(
                torch.zeros(
                    self.num_sinks,
                    self.total_num_kv_heads,
                    self.head_dim,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            self.sink_v = nn.Parameter(
                torch.zeros(
                    self.num_sinks,
                    self.total_num_kv_heads,
                    self.head_dim,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
        # Domino draft blocks are non-causal: every query position sees the
        # whole context block and the current draft block.  Sliding-window
        # layers additionally restrict the context part of the mask to the
        # last ``sliding_window`` KV positions.
        self.causal = causal
        # HCA (heavily_compressed_attention): compressed history entries read
        # through a dense [entries || band || draft block] attention instead of
        # the paged band call. The band call below still runs (it owns the
        # query K/V cache write and the runner's accounting); its output is
        # replaced by the dense path whenever the entry pool is populated.
        self.hca = hca
        self.hca_dense_fallback = False
        # Fixed read width for the entry branch.  ``None`` means the full
        # ``max_entries`` capacity; a concrete bucket is graph-safe but must be
        # large enough for the deployment's maximum stored entry count.
        self.hca_entry_bucket: int | None = None
        self.hca_compressor: DominoHCACompressor | None = None
        if self.hca is not None:
            # The pooler reads the fused context feature, so its input width
            # is the target width the context K/V projections see.
            compressor_input = (
                target_hidden_size
                if target_hidden_size is not None
                else hidden_size
            )
            self.hca_compressor = DominoHCACompressor(
                compressor_input,
                self.head_dim,
                self.hca,
                rms_norm_eps=rms_norm_eps,
                prefix=f"{prefix}.hca_compressor",
            )
        # Persistent entry pool [max_num_seqs, max_entries, hd]; allocated by
        # the model and written in place every round so CUDA graphs baked into
        # the dense read path keep valid pointers.
        self._hca_pool: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

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
        if self.hca is not None:
            # The band call may expose its softmax LSE from the backend; clear
            # any previous layer's value before the call so a stale LSE can
            # never be merged.
            impl = getattr(self.attn, "impl", None)
            if impl is not None:
                impl._hca_last_lse = None
        attn_output = self.attn(q, k, v)
        if self.hca is not None:
            if self._hca_pool is None:
                raise RuntimeError(
                    "HCA layer is configured but its entry pool was not "
                    "allocated; refusing to silently serve band-only attention"
                )
            attn_output = self._hca_attention(
                positions, q, k, v, attn_output
            )
        output = self.o_proj(attn_output)
        return output

    def _hca_metadata(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(query_start_loc, block_table) for the current draft forward.

        CUDA's forward context carries the CommonAttentionMetadata object;
        vLLM-Ascend carries a per-layer dict (list of dicts under multi-step
        drafting) whose entries expose the same segmentation with Ascend's
        ``block_tables`` spelling. Every layer of one draft forward shares the
        segmentation, so any entry serves the dense path.
        """
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, (list, tuple)):
            metadata = metadata[0]
        if isinstance(metadata, dict):
            for key in (self.layer_name, f"{self.layer_name}.attn"):
                if key in metadata:
                    metadata = metadata[key]
                    break
            else:
                metadata = next(iter(metadata.values()))
        block_table = getattr(metadata, "block_table_tensor", None)
        if block_table is None:
            block_table = metadata.block_tables
        return metadata.query_start_loc, block_table

    def _hca_pool_indices(
        self, block_table: torch.Tensor, num_reqs: int
    ) -> torch.Tensor | None:
        """Map current batch rows to persistent pool rows by first block."""

        block2row = getattr(self, "_hca_block2row", None)
        if block2row is None:
            return None
        first_blocks = block_table[:num_reqs, 0].to(torch.long)
        rows = block2row[first_blocks]
        # An unassigned row here would be a precompute/ordering bug; clamping
        # keeps the graph shape static and lets the (masked) pool read expose
        # the stale row rather than silently indexing out of range.
        return rows.clamp(min=0)

    def _hca_band_lse(
        self,
        lse: torch.Tensor | None,
        num_tokens: int,
        num_heads: int,
    ) -> torch.Tensor | None:
        """Normalize the backend's band-call LSE to ``[tokens, heads]``.

        FIA/FA implementations differ in whether LSE is transposed and whether
        a leading batch axis is present.  Any layout we do not recognize
        returns ``None``, which sends the layer to the exact dense fallback.
        """

        if lse is None or not isinstance(lse, torch.Tensor) or lse.numel() == 0:
            return None
        if lse.dim() == 3:
            if lse.shape[0] == 1:
                lse = lse[0]
            elif lse.shape[-1] == 1:
                lse = lse.squeeze(-1)
        if lse.dim() != 2:
            return None
        if lse.shape == (num_tokens, num_heads):
            return lse.float()
        if lse.shape == (num_heads, num_tokens):
            return lse.float().transpose(0, 1)
        return None

    def _hca_attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        band_output: torch.Tensor,
    ) -> torch.Tensor:
        """Read HCA as band output merged with an entry-only attention call.

        Falls back to the exact dense ``[entries || band || draft]`` path when
        the backend does not expose a band LSE (e.g. CUDA FlashAttention
        without the dedicated varlen call) or when the layout is unrecognized.
        """

        # Dummy/profile drafts run without attention metadata (the speculator
        # passes attn_metadata=None); leave the band output in place instead of
        # dereferencing a missing block table.
        if getattr(get_forward_context(), "attn_metadata", None) is None:
            return band_output
        if self.hca_dense_fallback:
            return self._hca_dense_attention(positions, q, k, v)
        impl = getattr(self.attn, "impl", None)
        lse1 = self._hca_band_lse(
            getattr(impl, "_hca_last_lse", None) if impl is not None else None,
            q.shape[0],
            self.num_heads,
        )
        if lse1 is None:
            logger.info_once(
                "HCA: band LSE unavailable; using the dense fallback read"
            )
            return self._hca_dense_attention(positions, q, k, v)

        entry = self._hca_entry_attention(positions, q)
        if entry is None:
            logger.info_once(
                "HCA: entry attention metadata mismatch; using the dense fallback"
            )
            return self._hca_dense_attention(positions, q, k, v)
        logger.info_once(
            "HCA: merging band and entry attention via softmax LSE"
        )
        o2, lse2 = entry
        o1 = band_output.float().reshape(o2.shape)
        lse = torch.logaddexp(lse1, lse2)
        finite = torch.isfinite(lse)
        safe = torch.where(finite, lse, torch.zeros_like(lse))
        w1 = torch.where(finite, torch.exp(lse1 - safe), torch.zeros_like(lse))
        w2 = torch.where(finite, torch.exp(lse2 - safe), torch.zeros_like(lse))
        out = o1 * w1.unsqueeze(-1) + o2 * w2.unsqueeze(-1)
        return out.to(band_output.dtype).reshape(band_output.shape)

    def _hca_entry_attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Attention over the compressed entries only.

        Returns ``(output [tokens, heads, hd], lse [tokens, heads])``, or
        ``None`` when the batch metadata does not line up.
        """

        query_start_loc, block_table = self._hca_metadata()
        num_reqs = query_start_loc.numel() - 1
        num_query = q.shape[0] // num_reqs
        if num_reqs <= 0 or num_reqs * num_query != q.shape[0]:
            return None
        entries_k, entries_v, pool_first, pool_stored = self._hca_pool
        pool_rows = self._hca_pool_indices(block_table, num_reqs)
        if pool_rows is not None:
            entries_k = entries_k[pool_rows]
            entries_v = entries_v[pool_rows]
            pool_first = pool_first[pool_rows]
            pool_stored = pool_stored[pool_rows]

        spec = self.hca
        hop = spec["hop"]
        W = spec["window"]
        capacity = entries_k.shape[1]
        bucket = getattr(self, "hca_entry_bucket", None)
        if bucket is not None and bucket < capacity:
            capacity = bucket
            entries_k = entries_k[:, :capacity]
            entries_v = entries_v[:, :capacity]
            pool_stored = pool_stored.clamp(max=capacity)
        kv_heads = self.num_kv_heads
        head_dim = self.head_dim
        group = self.num_heads // kv_heads
        device = positions.device

        q4 = q.view(
            num_reqs, num_query, self.num_heads, head_dim
        ).transpose(1, 2).to(entries_k.dtype)
        q_grouped = q4.view(
            num_reqs, kv_heads, group, num_query, head_dim
        )
        k_entries = entries_k.view(num_reqs, 1, 1, capacity, head_dim)
        v_entries = entries_v.view(num_reqs, 1, 1, capacity, head_dim)
        scores = (
            torch.matmul(q_grouped, k_entries.transpose(-1, -2))
            * self.scaling
        )

        t = positions.view(num_reqs, num_query).unsqueeze(-1)
        row_ids = torch.arange(capacity, device=device).view(1, 1, -1)
        first = pool_first.view(-1, 1, 1)
        stored = pool_stored.view(-1, 1, 1)
        entry_pos = (first + row_ids) * hop
        visible = (row_ids < stored) & (entry_pos < t + 1 - W)
        scores = scores.masked_fill(
            ~visible.unsqueeze(1).unsqueeze(1), float("-inf")
        )
        any_visible = visible.any(dim=-1)
        lse = torch.logsumexp(scores.float(), dim=-1)
        lse = torch.where(
            any_visible.unsqueeze(1).unsqueeze(1),
            lse,
            torch.full_like(lse, float("-inf")),
        )
        probs = torch.softmax(scores.float(), dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0).to(entries_k.dtype)
        out = torch.matmul(probs, v_entries)
        out = out.permute(0, 3, 1, 2, 4).reshape(
            num_reqs * num_query, self.num_heads, head_dim
        )
        lse = lse.permute(0, 3, 1, 2).reshape(
            num_reqs * num_query, self.num_heads
        )
        return out, lse

    def _hca_dense_attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Exact [entries || raw band || draft block] attention for one round.

        All shapes are static per graph batch: the entry pool is padded to
        ``max_entries``, the band gather to exactly ``W`` rows (clamped and
        over-masked when the context is shorter), and the draft block carries
        ``n_q`` columns, so the path is CUDA-graph capturable. Per-query
        visibility is the R2 rule shared with training: band = [t+1-W, t]
        plus the block-bidirectional draft columns; entry j visible iff
        ``j*hop < t+1-W`` and ``j >= nE - max_entries``.
        """
        spec = self.hca
        W = spec["window"]
        hop = spec["hop"]

        query_start_loc, block_table = self._hca_metadata()
        num_reqs = query_start_loc.numel() - 1
        num_query = q.shape[0] // num_reqs
        assert num_reqs * num_query == q.shape[0], (
            "HCA dense attention requires the uniform per-request query count "
            "of parallel drafting"
        )
        # Cache layout differs per platform: CUDA hands [k, v] with shape
        # [num_blocks, block, nkv*hd]; vLLM-Ascend hands one
        # (2, num_blocks, block, nkv, hd) tensor whose [0]/[1] are K/V. The
        # gather block size comes from the tensor (Ascend pads blocks to 128).
        kv_caches = self.attn.kv_cache
        if isinstance(kv_caches, torch.Tensor) and kv_caches.dim() == 5:
            k_cache, v_cache = kv_caches[0], kv_caches[1]
            block_size = kv_caches.shape[2]
        elif len(kv_caches) == 1:
            k_cache, v_cache = kv_caches[0][0], kv_caches[0][1]
            block_size = kv_caches[0].shape[2]
        else:
            k_cache, v_cache = kv_caches
            block_size = k_cache.shape[1]
        entries_k, entries_v, pool_first, pool_stored = self._hca_pool
        # Pool rows are keyed by the request's first physical block so a
        # request keeps its state across batch reordering.  Gather the current
        # batch's rows before the dense read.
        pool_rows = self._hca_pool_indices(block_table, num_reqs)
        if pool_rows is not None:
            entries_k = entries_k[pool_rows]
            entries_v = entries_v[pool_rows]
            pool_first = pool_first[pool_rows]
            pool_stored = pool_stored[pool_rows]
        capacity = entries_k.shape[1]
        bucket = getattr(self, "hca_entry_bucket", None)
        if bucket is not None and bucket < capacity:
            capacity = bucket
            entries_k = entries_k[:, :capacity]
            entries_v = entries_v[:, :capacity]
            pool_stored = pool_stored.clamp(max=capacity)
        kv_heads = self.num_kv_heads
        head_dim = self.head_dim
        device = positions.device

        q4 = q.view(num_reqs, num_query, self.num_heads, head_dim).transpose(1, 2)
        k_draft = (
            k.view(num_reqs, num_query, kv_heads, head_dim)
            .transpose(1, 2)
            .to(entries_k.dtype)
        )
        v_draft = (
            v.view(num_reqs, num_query, kv_heads, head_dim)
            .transpose(1, 2)
            .to(entries_k.dtype)
        )

        # Band gather: the W rows ending just below the first query position
        # (the anchor); padded requests produce clamped, over-masked rows.
        first_pos = positions.view(num_reqs, num_query)[:, 0]
        band_pos = (
            first_pos.view(-1, 1) - W + torch.arange(W, device=device).view(1, -1)
        )
        gather_idx = band_pos.clamp(min=0)
        block_ids = torch.div(gather_idx, block_size, rounding_mode="floor").clamp(
            max=block_table.shape[1] - 1
        )
        band_slots = gather_idx % block_size
        band_blocks = torch.gather(block_table, 1, block_ids)
        k_band = (
            k_cache[band_blocks, band_slots]
            .reshape(num_reqs, W, kv_heads, head_dim)
            .transpose(1, 2)
        )
        v_band = (
            v_cache[band_blocks, band_slots]
            .reshape(num_reqs, W, kv_heads, head_dim)
            .transpose(1, 2)
        )

        # [entries || band || draft]: entries broadcast over the KV heads.
        # ACLNN's cat rejects non-contiguous/zero-stride inputs and requires a
        # single dtype, so normalize all three views before concatenating.
        k_band = k_band.to(entries_k.dtype).contiguous()
        v_band = v_band.to(entries_v.dtype).contiguous()
        k_all = torch.cat(
            [
                entries_k.unsqueeze(1)
                .expand(-1, kv_heads, -1, -1)
                .contiguous(),
                k_band,
                k_draft.contiguous(),
            ],
            dim=2,
        )
        v_all = torch.cat(
            [
                entries_v.unsqueeze(1)
                .expand(-1, kv_heads, -1, -1)
                .contiguous(),
                v_band,
                v_draft.contiguous(),
            ],
            dim=2,
        )
        q4 = q4.to(k_all.dtype)

        t = positions.view(num_reqs, num_query, 1).to(torch.float32)
        neg_inf = float("-inf")
        row_ids = torch.arange(capacity, device=device).view(1, 1, -1).to(
            torch.float32
        )
        # The pool holds at most the newest max_entries entries; entry index
        # = first_index + row. Visible iff stored, and its window starts
        # below the query's band lower bound.
        first_idx = pool_first.view(-1, 1, 1).to(torch.float32)
        stored = pool_stored.view(-1, 1, 1).to(torch.float32)
        entry_pos = (first_idx + row_ids) * hop
        visible_entries = (row_ids < stored) & (entry_pos < t + 1 - W)
        visible_band = (band_pos.unsqueeze(1) >= t + 1 - W) & (
            band_pos.unsqueeze(1) >= 0
        )
        # ACLNN's bool cat is also fragile; materialize float masks and use a
        # float cat with contiguous inputs, then cast once.
        entry_mask = torch.where(
            visible_entries, 0.0, neg_inf
        ).contiguous()
        band_mask = torch.where(visible_band, 0.0, neg_inf).contiguous()
        mask = torch.cat([entry_mask, band_mask], dim=-1).to(q4.dtype)
        # Draft block columns are block-bidirectional (non-causal): pad the
        # trailing num_query always-visible columns.
        mask = F.pad(mask, (0, num_query))

        # Grouped matmul over GQA (no sdpa enable_gqa: torch_npu support is
        # unverified); scores/mask are small at draft sizes.
        group = self.num_heads // kv_heads
        q_grouped = q4.reshape(num_reqs, kv_heads, group, num_query, head_dim)
        scores = torch.matmul(
            q_grouped, k_all.transpose(-1, -2).unsqueeze(2)
        )
        # [R, nq, L] -> [R, 1, 1, nq, L]: without both singleton axes the
        # broadcast against [R, kv, group, nq, L] only works when
        # num_reqs == 1 or num_reqs == num_kv_heads.
        scores = scores * self.scaling + mask.reshape(
            num_reqs, 1, 1, num_query, -1
        )
        probs = torch.softmax(scores.float(), dim=-1).to(k_all.dtype)
        out = torch.matmul(probs, v_all.unsqueeze(2))
        return out.permute(0, 3, 1, 2, 4).reshape(
            num_reqs * num_query, self.num_heads * head_dim
        ).to(q.dtype)


# ---------------------------------------------------------------------------
# Shared gate/up projections (``dflash_config.ffn_sharing``)
# ---------------------------------------------------------------------------
#
# Serving counterpart of the training knob in
# ``specforge/modeling/draft/dflash_kernels.py``.  ``gate_proj``/``up_proj``
# shrink to ``gate_groups``/``up_groups`` unique channels fused into one
# ``MergedColumnParallelLinear`` (the stacked weight loader places the two
# halves by their own sizes, so unequal shapes and SpecForge exports with
# separate ``gate_proj``/``up_proj`` weights load as-is), and intermediate
# channel ``j`` pairs them through precomputed index maps:
#
#     h_j = silu( gate[gate_idx[j]] ) * up[up_idx[j]]
#
#     nested:  gate_idx[j] = j // (M / G_g),   up_idx[j] = j // (M / G_u)
#     outer:   gate_idx[j] = j % G_g,           up_idx[j] = (j // G_g) % G_u
#
# ``down_proj`` is untouched (dense, or the folded readout above when
# ``ffn_readout`` is also enabled).  TP > 1 is rejected: after column-parallel
# sharding the channel gather reads gate/up channels owned by other ranks,
# matching the folded readout's restriction.

_SHARING_PAIRINGS = frozenset({"nested", "outer"})


def _divisors(value: int) -> list[int]:
    return [d for d in range(2, min(value, 1024) + 1) if value % d == 0]


_FOLDED_READOUT_KEY = "ffn_readout"  # retired; kept only for the guard below
_SHARING_MODES = frozenset({"lattice", "routed_outer"})
_ROUTER_GRANULARITIES = frozenset({"expert", "gate_slot"})


def _reject_retired_fold(config) -> None:
    """Refuse retired ``ffn_readout`` entries with an actionable message."""

    raw = (getattr(config, "dflash_config", None) or {}).get(_FOLDED_READOUT_KEY)
    if raw is None or raw is False:
        return
    if isinstance(raw, str) and raw in {"dense", "off", "none"}:
        return
    if isinstance(raw, dict) and raw.get("mode") in {"dense", "off", "none"}:
        return
    raise ValueError(
        "dflash_config.ffn_readout is retired: the folded readout measured "
        "equivalent to shrinking the gate lattice (the static mixture is "
        "inert). Drop the key, or serve old checkpoints from the "
        "outer_ffn_fold tag."
    )


def _resolve_vha_load_spec(config) -> Optional[int]:
    """``physical_q_heads`` from ``dflash_config.vha``; None for plain GQA.

    Serving counterpart of SpecForge's training-side resolver: a VHA
    checkpoint keeps its training parameterization (physical-width q_proj
    plus ``vha_premix``/``vha_postmix_*``) and the loader folds it back
    into standard GQA at weight-load time.  QAT-exported int-packed
    checkpoints cannot be folded here and must go through
    ``scripts/fold_vha.py`` before quantization instead.
    """

    raw = (getattr(config, "dflash_config", None) or {}).get("vha")
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict) or "physical_q_heads" not in raw:
        raise ValueError(
            "dflash_config.vha must be a dict with physical_q_heads, got "
            f"{raw!r}"
        )
    physical = raw["physical_q_heads"]
    num_heads = int(getattr(config, "num_attention_heads", 0))
    num_kv = int(getattr(config, "num_key_value_heads", 0))
    if (
        not isinstance(physical, int)
        or isinstance(physical, bool)
        or num_heads != physical * num_kv
    ):
        raise ValueError(
            "dflash_config.vha needs num_attention_heads == physical_q_heads "
            f"* num_key_value_heads, got {num_heads} vs {physical} * {num_kv}"
        )
    return physical


_VHA_PARAMS = ("vha_premix", "vha_postmix_u", "vha_postmix_v")


def fold_vha_weights(
    model_weights: dict, num_heads: int, head_dim: int
) -> bool:
    """Fold VHA premix/postmix into q_proj/o_proj entries, in place.

    Serving mirror of SpecForge ``specforge/modeling/draft/vha.py`` (keep
    the einsum-free formulations in sync).  Returns whether any VHA
    parameters were found; raises on incomplete or shape-mismatched sets.
    """

    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name in list(model_weights):
        for param in _VHA_PARAMS:
            if name.endswith("." + param):
                pending.setdefault(name[: -(len(param) + 1)], {})[
                    param
                ] = model_weights.pop(name)
    if not pending:
        return False

    for prefix, params in pending.items():
        missing = set(_VHA_PARAMS) - set(params)
        if missing:
            raise ValueError(
                f"incomplete VHA parameters under {prefix!r}: missing {sorted(missing)}"
            )
        premix, u, v = (
            params["vha_premix"],
            params["vha_postmix_u"],
            params["vha_postmix_v"],
        )

        q_key = f"{prefix}.q_proj.weight"
        q_weight = model_weights[q_key]
        if not q_weight.is_floating_point():
            raise ValueError(
                f"VHA parameters at {prefix!r} arrive with non-float q_proj "
                "weights: int-packed QAT exports cannot be folded at load. "
                "Run SpecForge scripts/fold_vha.py before the QAT stage."
            )
        num_kv = premix.shape[0]
        physical = q_weight.shape[0] // head_dim
        if (
            q_weight.shape[0] != physical * head_dim
            or num_heads != physical * num_kv
        ):
            raise ValueError(
                f"VHA shape mismatch at {prefix!r}: q_proj {tuple(q_weight.shape)} "
                f"vs premix {tuple(premix.shape)} under {num_heads} heads"
            )
        w = q_weight.float().view(physical, head_dim, -1)
        folded = torch.stack(
            [torch.matmul(pm.t(), w) for pm in premix.float().unbind(0)]
        )
        model_weights[q_key] = folded.reshape(
            num_kv * physical * head_dim, w.shape[-1]
        ).to(q_weight.dtype)
        b_key = f"{prefix}.q_proj.bias"
        if b_key in model_weights:
            b = model_weights[b_key].float().view(physical, head_dim)
            folded_b = torch.stack(
                [torch.matmul(pm.t(), b) for pm in premix.float().unbind(0)]
            )
            model_weights[b_key] = folded_b.reshape(-1).to(model_weights[b_key].dtype)

        o_key = f"{prefix}.o_proj.weight"
        o_weight = model_weights[o_key]
        if u.shape[0] != num_heads or v.shape != u.shape:
            raise ValueError(
                f"VHA postmix shape mismatch at {prefix!r}: u {tuple(u.shape)}, "
                f"v {tuple(v.shape)} under {num_heads} heads"
            )
        m = u.float() @ v.float().t()
        mix = torch.eye(num_heads, dtype=m.dtype) + m
        w3 = o_weight.float().view(-1, num_heads, head_dim)
        model_weights[o_key] = torch.matmul(
            w3.permute(2, 0, 1), mix.t()
        ).permute(1, 2, 0).reshape(o_weight.shape).to(o_weight.dtype)
    return True


def resolve_ffn_sharing(config) -> tuple | None:
    """Resolve ``dflash_config.ffn_sharing``; ``None`` keeps the dense SwiGLU.

    Must mirror the training-side resolver in SpecForge.  Returns
    ``("lattice", pairing, gate_groups, up_groups)`` for the single shared
    lattice or ``("routed_outer", experts, gate_groups, up_groups, router)``
    for the routed outer experts.
    """

    dflash_config = getattr(config, "dflash_config", None) or {}
    raw = dflash_config.get("ffn_sharing")
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            "dflash_config.ffn_sharing must be a dict, got "
            f"{type(raw).__name__}"
        )
    spec = dict(raw)
    mode = spec.pop("mode", "lattice")
    if mode not in _SHARING_MODES:
        raise ValueError(
            f"unknown dflash_config.ffn_sharing mode {mode!r}; expected one "
            f"of {sorted(_SHARING_MODES)}"
        )

    if mode == "routed_outer":
        unknown = sorted(set(spec) - {"experts", "gate_groups", "up_groups", "router"})
        if unknown:
            raise ValueError(
                f"unknown dflash_config.ffn_sharing entries {unknown}; "
                "expected 'experts', 'gate_groups', 'up_groups' and 'router'"
            )
        router = spec.get("router", "expert")
        if router not in _ROUTER_GRANULARITIES:
            raise ValueError(
                f"dflash_config.ffn_sharing.router={router!r} must be one of "
                f"{sorted(_ROUTER_GRANULARITIES)}"
            )
        experts = int(spec.get("experts", 1))
        gate_groups = int(spec.get("gate_groups", 0) or 0)
        up_groups = int(spec.get("up_groups", 0) or 0)
        if min(experts, gate_groups, up_groups) < 1:
            raise ValueError(
                "routed_outer needs experts/gate_groups/up_groups >= 1, got "
                f"{experts}/{gate_groups}/{up_groups}"
            )
        if experts == 1 and router == "gate_slot":
            raise ValueError(
                "router='gate_slot' needs experts >= 2 (with one expert the "
                "blend is identically 1)"
            )
        intermediate_size = int(getattr(config, "intermediate_size", 0) or 0)
        if intermediate_size != experts * gate_groups * up_groups:
            raise ValueError(
                "routed_outer needs intermediate_size == experts * "
                f"gate_groups * up_groups, got {intermediate_size} vs "
                f"{experts} * {gate_groups} * {up_groups}"
            )
        return ("routed_outer", experts, gate_groups, up_groups, router)

    unknown = sorted(set(spec) - {"pairing", "gate_groups", "up_groups"})
    if unknown:
        raise ValueError(
            f"unknown dflash_config.ffn_sharing entries {unknown}; expected "
            "'mode', 'pairing', 'gate_groups' and 'up_groups'"
        )
    pairing = spec.get("pairing", "nested")
    if pairing not in _SHARING_PAIRINGS:
        raise ValueError(
            f"unsupported dflash_config.ffn_sharing pairing {pairing!r}; "
            f"expected one of {sorted(_SHARING_PAIRINGS)}"
        )
    intermediate_size = int(getattr(config, "intermediate_size", 0) or 0)
    if intermediate_size <= 0:
        raise ValueError("ffn sharing needs a positive intermediate_size")
    gate_groups = int(spec.get("gate_groups", intermediate_size))
    up_groups = int(spec.get("up_groups", intermediate_size))
    for name, groups in (("gate_groups", gate_groups), ("up_groups", up_groups)):
        if groups < 1 or groups > intermediate_size:
            raise ValueError(
                f"dflash_config.ffn_sharing.{name}={groups} must be between 1 "
                f"and intermediate_size={intermediate_size}"
            )
    if pairing == "nested":
        if intermediate_size % gate_groups or intermediate_size % up_groups:
            raise ValueError(
                f"nested sharing needs gate_groups={gate_groups} and "
                f"up_groups={up_groups} to divide intermediate_size="
                f"{intermediate_size}"
            )
    elif intermediate_size % (gate_groups * up_groups):
        raise ValueError(
            f"outer sharing needs gate_groups * up_groups to divide "
            f"intermediate_size, got {gate_groups} * {up_groups} = "
            f"{gate_groups * up_groups} with intermediate_size="
            f"{intermediate_size}"
        )
    return ("lattice", pairing, gate_groups, up_groups)


def _sharing_indices(
    intermediate_size: int, gate_groups: int, up_groups: int, pairing: str
) -> tuple[torch.Tensor, torch.Tensor]:
    channel = torch.arange(intermediate_size)
    if pairing == "nested":
        gate_idx = channel // (intermediate_size // gate_groups)
        up_idx = channel // (intermediate_size // up_groups)
    else:  # outer
        gate_idx = channel % gate_groups
        up_idx = (channel // gate_groups) % up_groups
    return gate_idx, up_idx


class SharedGLUMLP(nn.Module):
    """SwiGLU MLP whose fused gate/up halves serve multiple channels."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        pairing: str = "nested",
        gate_groups: int,
        up_groups: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            raise NotImplementedError(
                "ffn sharing requires "
                f"draft_tensor_parallel_size=1 (got {tp_size}); use the dense "
                "gate/up projections or shard the gathered channels first"
            )
        self.intermediate_size = intermediate_size
        self.gate_groups = gate_groups
        self.up_groups = up_groups
        self.pairing = pairing
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [gate_groups, up_groups],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        gate_idx, up_idx = _sharing_indices(
            intermediate_size, gate_groups, up_groups, pairing
        )
        self.register_buffer("gate_idx", gate_idx, persistent=False)
        self.register_buffer("up_idx", up_idx, persistent=False)

    def shared_act(self, gate_up: torch.Tensor) -> torch.Tensor:
        """SwiGLU over the fused ``[gate | up]`` halves with channel sharing.

        Split out so the Ascend fused norm+quant path can reuse the exact
        post-GEMM computation instead of the plain ``npu_swiglu`` halves.
        """

        gate = gate_up[..., : self.gate_groups]
        up = gate_up[..., self.gate_groups :]
        return F.silu(gate[..., self.gate_idx]) * up[..., self.up_idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out = self.down_proj(self.shared_act(gate_up))
        return out[0] if isinstance(out, tuple) else out


class RoutedOuterMLP(nn.Module):
    """``experts`` independent outer-product FFNs blended by a router.

    One ``MergedColumnParallelLinear`` carries every expert's gate and up
    halves expert-major plus the trailing ``router_width`` routing rows
    (zero-initialized: serving matches the trained uniform-blend start), and
    ``combine`` reproduces the training-side post-GEMM computation exactly:

        H(x) = sum_e w_e(x) * silu(W_g^e x) (W_u^e x)^T
        y    = down_proj(vec(H))

    The checkpoint's fused ``gate_up_proj.weight`` loads through the stacked
    shard path unchanged; ``experts=1`` is the plain outer lattice.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        experts: int,
        gate_groups: int,
        up_groups: int,
        router: str = "expert",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            raise NotImplementedError(
                "routed outer sharing requires "
                f"draft_tensor_parallel_size=1 (got {tp_size})"
            )
        self.intermediate_size = intermediate_size
        self.experts = experts
        self.gate_groups = gate_groups
        self.up_groups = up_groups
        self.router = router
        self.router_width = experts if router == "expert" else experts * gate_groups
        output_sizes = [gate_groups, up_groups] * experts + [self.router_width]
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            gate_groups * up_groups,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def combine(self, fused: torch.Tensor) -> torch.Tensor:
        lead = fused.shape[:-1]
        E, G, U = self.experts, self.gate_groups, self.up_groups
        feats = fused[..., : E * (G + U)].reshape(*lead, E, G + U)
        gate = F.silu(feats[..., :G])
        up = feats[..., G:]
        route = fused[..., E * (G + U):]
        if self.router == "expert":
            gate = gate * torch.softmax(route, dim=-1).unsqueeze(-1)
        else:
            delta = torch.softmax(route.reshape(*lead, G, E), dim=-1)
            gate = gate * delta.transpose(-1, -2)
        hidden = (gate.unsqueeze(-1) * up.unsqueeze(-2)).sum(dim=-3)
        out, _ = self.down_proj(hidden.reshape(*lead, G * U))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused, _ = self.gate_up_proj(x)
        return self.combine(fused)


class DominoGroupedConv(nn.Module):
    """Serving port of SpecForge's DFlashGroupedConv.

    Parameter names (``base_kernel``, ``kernel_projection``) match the
    training export and are always supplied by the checkpoint.
    ``_convolve`` accepts any leading batch dims: training feeds
    [B, blocks*block, H], serving feeds the flattened [num_reqs*block, H]
    token stream.
    """

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        taps: int,
        group_size: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if taps < 1 or taps > block_size:
            raise ValueError(
                f"conv_kernel_size must be in [1, block_size={block_size}], "
                f"got {taps}"
            )
        if group_size < 1 or hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide "
                f"hidden_size={hidden_size}"
            )
        self.block_size = int(block_size)
        self.taps = int(taps)
        self.num_groups = int(hidden_size) // int(group_size)

        base_kernel = torch.zeros(
            2, self.taps, int(hidden_size), dtype=dtype
        )
        base_kernel[:, 0] = 1.0
        self.base_kernel = nn.Parameter(base_kernel)
        self.kernel_projection = ReplicatedLinear(
            int(hidden_size),
            2 * self.taps * self.num_groups,
            bias=False,
            params_dtype=dtype,
            quant_config=None,
            prefix="conv.kernel_projection",
        )

    def _convolve(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        *,
        side: int,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[-2]
        hidden_size = hidden_states.shape[-1]
        if seq_len % self.block_size:
            raise ValueError(
                f"conv sequence length {seq_len} not divisible by "
                f"block_size={self.block_size}"
            )
        shape = hidden_states.shape
        blocks = hidden_states.reshape(
            -1,
            seq_len // self.block_size,
            self.block_size,
            self.num_groups,
            hidden_size // self.num_groups,
        )
        dynamic = delta.reshape(
            -1,
            seq_len // self.block_size,
            self.block_size,
            self.taps,
            self.num_groups,
        )
        base = self.base_kernel[side].reshape(
            1, 1, 1, self.taps, self.num_groups, -1
        )
        coefficients = base + dynamic.unsqueeze(-1)

        output = coefficients[:, :, :, 0] * blocks
        for tap in range(1, self.taps):
            shifted = F.pad(
                blocks[:, :, : self.block_size - tap],
                (0, 0, 0, 0, tap, 0),
            )
            output = output + coefficients[:, :, :, tap] * shifted
        return output.reshape(shape)

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.kernel_projection(hidden_states)[0]
        delta = delta.reshape(
            *hidden_states.shape[:-1], 2, self.taps, self.num_groups
        )
        return (
            self._convolve(hidden_states, delta[..., 0, :, :], side=0),
            delta[..., 1, :, :],
        )

    def finish(
        self,
        hidden_states: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)


class DominoQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: Qwen3Config,
        layer_idx: int,
        target_hidden_size: int | None,
        hca: dict | None = None,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)

        sliding_window, causal = _domino_layer_attention(config, layer_idx)
        self.self_attn = DominoQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            target_hidden_size=target_hidden_size,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            sliding_window=sliding_window,
            causal=causal,
            num_sinks=int(_dflash_config(config).get("attn_sink", 0) or 0),
            hca=hca,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
        )
        _reject_retired_fold(config)
        conv_cfg = _dflash_config(config)
        conv_taps = conv_cfg.get("conv_kernel_size")
        conv_group = conv_cfg.get("conv_group_size")
        if conv_taps is None and conv_group is None:
            self.attention_conv: DominoGroupedConv | None = None
            self.mlp_conv: DominoGroupedConv | None = None
        else:
            if (
                not isinstance(conv_taps, int)
                or isinstance(conv_taps, bool)
                or not isinstance(conv_group, int)
                or isinstance(conv_group, bool)
            ):
                raise ValueError(
                    "domino conv requires integer dflash_config."
                    "conv_kernel_size and conv_group_size"
                )
            # Canonical entry: top-level config.block_size (what every draft
            # config carries and SpecForge reads first); dflash_config.
            # block_size is a legacy fallback kept for old exports only.
            block_size = int(
                getattr(config, "block_size", None)
                or conv_cfg.get("block_size", 0)
                or 0
            )
            if block_size <= 0:
                raise ValueError(
                    "domino conv requires dflash_config.block_size in the "
                    "draft config"
                )
            dtype = vllm_config.model_config.dtype

            def grouped_conv() -> DominoGroupedConv:
                return DominoGroupedConv(
                    hidden_size=int(config.hidden_size),
                    block_size=block_size,
                    taps=conv_taps,
                    group_size=conv_group,
                    dtype=dtype,
                )

            self.attention_conv = grouped_conv()
            self.mlp_conv = grouped_conv()
        sharing = resolve_ffn_sharing(config)
        if sharing is None:
            self.mlp = Qwen3MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        elif sharing[0] == "routed_outer":
            _, experts, gate_groups, up_groups, router = sharing
            self.mlp = RoutedOuterMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                experts=experts,
                gate_groups=gate_groups,
                up_groups=up_groups,
                router=router,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            _, pairing, gate_groups, up_groups = sharing
            self.mlp = SharedGLUMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                pairing=pairing,
                gate_groups=gate_groups,
                up_groups=up_groups,
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
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Residual-stream pattern (mirrors Qwen3DecoderLayer): passing the
        # residual into each RMSNorm lets the platform fuse add+norm into
        # one kernel (CUDA fused_add_rmsnorm / NPU npu_add_rms_norm).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )
        if self.attention_conv is not None:
            hidden_states, attention_kernel = self.attention_conv.prepare(
                hidden_states
            )
        hidden_states = self.self_attn(positions, hidden_states)
        if self.attention_conv is not None:
            hidden_states = self.attention_conv.finish(
                hidden_states, attention_kernel
            )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        if self.mlp_conv is not None:
            hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.mlp_conv is not None:
            hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        return hidden_states, residual


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
        # SpecForge's DFlashDraftModel defaults to the fc (concat -> Linear)
        # fusion when the key is absent; flare is the routed softmax variant.
        fusion_mode = dflash_config.get("fusion_mode", "fc")
        if fusion_mode not in ("fc", "flare"):
            raise NotImplementedError(
                "dflash_config.fusion_mode must be 'fc' or 'flare', got "
                f"{fusion_mode!r}"
            )
        self.is_flare = fusion_mode == "flare"
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
        # Dedicated target K/V projections.  Flare checkpoints predate the key
        # and always shipped them, so absent means dedicated for flare; fc
        # defaults to the shared k_proj/v_proj that SpecForge trains with
        # heterogeneous_kv=false.
        self.heterogeneous_kv = bool(
            dflash_config.get("heterogeneous_kv", self.is_flare)
        )
        # The fused context feature: flare keeps target width (one routed
        # feature per draft layer), fc projects the concat to draft width and
        # every layer reads the same feature.
        self.context_hidden_size = (
            self.target_hidden_size if self.is_flare else self.config.hidden_size
        )
        proj_input = (
            self.target_hidden_size
            if self.heterogeneous_kv
            else self.config.hidden_size
        )
        if proj_input != self.context_hidden_size:
            raise ValueError(
                "dflash_config.heterogeneous_kv="
                f"{self.heterogeneous_kv} needs the fused context feature "
                f"({self.context_hidden_size}) to match the target K/V "
                f"projection input ({proj_input}): fc needs "
                "heterogeneous_kv=false unless target_hidden_size == "
                "hidden_size, flare needs true unless they are equal"
            )
        # Mirrors SpecForge's dflash_config.out_proj_after_norm: the final
        # norm runs at draft width and output_proj sits between it and the
        # lm_head (folded away at load time when the head allows it).
        self.out_proj_after_norm = bool(
            dflash_config.get("out_proj_after_norm", False)
        )
        # Mirrors SpecForge's dflash_config.shared_emb_readout: base and
        # correction read out through one emb-width bottleneck (W_down/W_up)
        # plus the frozen lm_head. Load folds lm_head @ W_up into a single
        # [V, emb] head and the correction loop adds in emb space.
        self.shared_emb_readout = bool(
            dflash_config.get("shared_emb_readout", False)
        )
        if self.shared_emb_readout and not self.out_proj_after_norm:
            raise ValueError(
                "shared_emb_readout requires out_proj_after_norm=true"
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
            if not self.shared_emb_readout:
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
        self.layer_hca_specs = resolve_domino_hca(self.config)
        # Max concurrent requests: sizes the persistent HCA entry pools.
        self._hca_pool_seqlen: int | None = (
            int(current_vllm_config.scheduler_config.max_num_seqs)
            if self.layer_hca_specs is not None
            else None
        )
        self.layers = nn.ModuleList(
            [
                DominoQwen3DecoderLayer(
                    current_vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    # None: no dedicated target projections -- the fused context
                    # feature goes through the shared k_proj/v_proj.
                    target_hidden_size=(
                        self.target_hidden_size
                        if self.heterogeneous_kv
                        else None
                    ),
                    hca=(
                        self.layer_hca_specs[layer_idx]
                        if self.layer_hca_specs is not None
                        else None
                    ),
                    cache_config=current_vllm_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(
                        prefix, f"layers.{layer_idx + start_layer_id}"
                    ),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.num_sinks = int(_dflash_config(self.config).get("attn_sink", 0) or 0)
        if self.num_sinks:
            layer_sinks = {layer.self_attn.num_sinks for layer in self.layers}
            if layer_sinks != {self.num_sinks}:
                raise ValueError(
                    "all draft layers must share dflash_config.attn_sink; "
                    f"model={self.num_sinks}, layers={sorted(layer_sinks)}"
                )
        # Serving-only escape hatch: keep the exact dense [entries||band||draft]
        # read path if the platform's band LSE is unavailable or suspect.
        self.hca_dense_fallback = bool(
            _dflash_config(self.config).get("hca_dense_fallback", False)
        )
        if self.layer_hca_specs is not None:
            for layer in self.layers:
                layer.self_attn.hca_dense_fallback = self.hca_dense_fallback
            bucket = _dflash_config(self.config).get("hca_entry_bucket")
            if bucket is not None:
                if (
                    isinstance(bucket, bool)
                    or not isinstance(bucket, int)
                    or bucket < 1
                ):
                    raise ValueError(
                        "dflash_config.hca_entry_bucket must be a positive int"
                    )
                for layer, spec in zip(self.layers, self.layer_hca_specs):
                    if spec is not None:
                        layer.self_attn.hca_entry_bucket = min(
                            bucket, spec["max_entries"]
                        )

        if self.is_flare:
            # Per-draft-layer softmax combination of the target features.
            self.layer_fusion_weights = nn.Parameter(
                torch.empty(
                    self.config.num_hidden_layers,
                    self.num_target_features,
                    dtype=vllm_config.model_config.dtype,
                )
            )
            self._init_fusion_weights()
            self.fc = None
        else:
            # fc fusion: one shared projection of the concatenated target taps.
            self.layer_fusion_weights = None
            self.fc = ReplicatedLinear(
                self.num_target_features * self.target_hidden_size,
                self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.hidden_norm = RMSNorm(
            self.context_hidden_size, eps=self.config.rms_norm_eps
        )
        self.norm = RMSNorm(
            self.config.hidden_size
            if self.out_proj_after_norm
            else self.target_hidden_size,
            eps=self.config.rms_norm_eps,
        )

        # Per-tap RMSNorm on the captured target features before the fusion
        # (dflash_config.target_tap_norm).  The taps of the concatenated feature
        # differ by up to ~50x in RMS on a real target, so one shared per-token
        # activation scale starves the quiet taps -- the ones the trained fusion
        # reads most.  The parameter layout matches the Laguna drafts served by
        # ``laguna_dflash.py``, so a checkpoint trained with the same flag loads
        # unchanged and vice versa.
        self.target_tap_norm = bool(dflash_config.get("target_tap_norm", False))
        if self.target_tap_norm:
            self.aux_hidden_norms = nn.ModuleList(
                [
                    RMSNorm(
                        self.target_hidden_size, eps=self.config.rms_norm_eps
                    )
                    for _ in range(self.num_target_features)
                ]
            )
        else:
            self.aux_hidden_norms = None

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
            if self.shared_emb_readout:
                # Pure-linear GRU readout into the shared emb bottleneck; the
                # prefix projection of z lives in head_down_proj and the
                # lm_head @ head_up_proj product is folded at load.
                self.embed_proj = ReplicatedLinear(
                    self.gru_hidden_dim,
                    self.emb_dim,
                    bias=False,
                    params_dtype=vllm_config.model_config.dtype,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "embed_proj"),
                    return_bias=False,
                )
            else:
                z_dim = (
                    self.config.hidden_size
                    if self.out_proj_after_norm
                    else self.target_hidden_size
                )
                in_dim = z_dim + self.gru_hidden_dim
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

        self.head_down_proj = None
        self.head_up_proj = None
        if self.shared_emb_readout:
            if self.embed_proj is None:
                raise ValueError(
                    "shared_emb_readout requires use_embed_proj=true"
                )
            if dflash_config.get("use_hidden_proj", False):
                raise ValueError(
                    "shared_emb_readout is not compatible with use_hidden_proj"
                )
            self.head_down_proj = ReplicatedLinear(
                self.config.hidden_size,
                self.emb_dim,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "head_down_proj"),
                return_bias=False,
            )
            self.head_up_proj = ReplicatedLinear(
                self.emb_dim,
                self.target_hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "head_up_proj"),
                return_bias=False,
            )

        if dflash_config.get("use_hidden_proj", False):
            hidden_proj_dim = dflash_config.get(
                "hidden_proj_dim", self.emb_dim
            )
            if hidden_proj_dim is None:
                raise ValueError(
                    "use_hidden_proj=true requires hidden_proj_dim when "
                    "use_embed_proj=false"
                )
            z_dim = (
                self.config.hidden_size
                if self.out_proj_after_norm
                else self.target_hidden_size
            )
            in_dim = z_dim + self.gru_hidden_dim
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
        self._use_fused_context_kv = False

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

        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, residual
            )

        if self.out_proj_after_norm:
            # output_proj moves to compute_draft_logits (or is folded into
            # the lm_head at load); the correction head consumes this
            # draft-width hidden directly, and the residual-arg norm fuses
            # the final add into the last RMSNorm.
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        # output_proj sits between the residual add and the final norm, so
        # the add cannot fuse into the norm here.
        hidden_states = hidden_states + residual
        if self.output_proj is not None:
            hidden_states = self.output_proj(hidden_states)
        return self.norm(hidden_states)

    def _build_fused_kv_buffers(self) -> None:
        """Build fused buffers for :meth:`precompute_and_store_context_kv`.

        Stacks the target K/V projection weights, the k-norm weights, and the
        RoPE parameters from every draft layer so the context states can be
        projected with one batched GEMM.  All layers share the same
        pre-projection ``hidden_norm``; flattening ``[T, D, H]`` to
        ``[T*D, H]`` lets a single grouped RMSNorm replace the per-layer loop.

        Only used when the projections are unquantized; with a quantization
        config the per-layer loop remains active.
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        # Dedicated target projections when heterogeneous_kv, otherwise the
        # context is projected by the same k_proj/v_proj as the draft stream.
        k_proj = (
            attn0.k_proj_target
            if attn0.k_proj_target is not None
            else attn0.k_proj
        )

        if (
            self.quant_config is not None
            or not k_proj.weight.data.is_floating_point()
        ):
            logger.warning_once(
                "Domino fused context-KV precompute requires unquantized "
                "target K/V projections; using the per-layer path."
            )
            self._use_fused_context_kv = False
            # Drop fused buffers built from the unquantized weights so a later
            # rebuild (e.g. after on-the-fly quantization) frees the memory.
            for attr in (
                "_fused_kv_weight_T",
                "_fused_kv_weight_flat",
                "_fused_kv_bias",
                "_fused_kv_bias_flat",
                "_k_norm_weights",
                "_attn_layers",
                "_rope_head_size",
                "_rope_cos_sin_cache",
                "_rope_is_neox",
                "_kv_size",
                "_head_dim",
                "_num_kv_heads",
                "_rms_norm_eps",
                "_hidden_norm_weight",
                "_hidden_norm_eps",
            ):
                if hasattr(self, attr):
                    delattr(self, attr)
            # Mark the fused decision as made so the precompute does not
            # re-attempt the build on every step. HCA pooling is checked at
            # precompute time instead: vLLM-Ascend's wrapper can still enable
            # its own quantized fused context-KV path after this build.
            self._num_attn_layers = len(layers_attn)
            self._build_hca_pools()
            return

        # KV projection weights, transposed for bmm:
        # [num_layers, context_width, 2 * kv_size]
        fused_kv_weights_T: list[torch.Tensor] = []
        fused_kv_weights: list[torch.Tensor] = []
        fused_kv_biases: list[torch.Tensor] = []
        has_bias = k_proj.bias is not None
        for attn in layers_attn:
            k_proj_i = (
                attn.k_proj_target
                if attn.k_proj_target is not None
                else attn.k_proj
            )
            v_proj_i = (
                attn.v_proj_target
                if attn.v_proj_target is not None
                else attn.v_proj
            )
            kv_weight = torch.cat(
                [k_proj_i.weight.data, v_proj_i.weight.data],
                dim=0,
            )
            fused_kv_weights_T.append(kv_weight.t().contiguous())
            fused_kv_weights.append(kv_weight)
            if has_bias:
                fused_kv_biases.append(
                    torch.cat(
                        [k_proj_i.bias.data, v_proj_i.bias.data],
                        dim=0,
                    )
                )
        self._fused_kv_weight_T = torch.stack(fused_kv_weights_T, dim=0)
        self._fused_kv_bias = (
            torch.stack(fused_kv_biases, dim=0) if has_bias else None
        )
        # fc reads one shared context feature for every layer, so the stacked
        # per-layer projections collapse to a single [D * 2kv, H] GEMM.
        if self.is_flare:
            self._fused_kv_weight_flat = None
            self._fused_kv_bias_flat = None
        else:
            self._fused_kv_weight_flat = torch.cat(fused_kv_weights, dim=0)
            self._fused_kv_bias_flat = (
                torch.cat(fused_kv_biases, dim=0) if has_bias else None
            )

        self._hidden_norm_weight = self.hidden_norm.weight.data
        self._hidden_norm_eps = self.hidden_norm.variance_epsilon
        self._k_norm_weights = torch.stack(
            [attn.k_norm.weight.data for attn in layers_attn], dim=0
        ).contiguous()

        # RoPE parameters
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        for attn in layers_attn[1:]:
            assert (
                attn.rotary_emb.head_size == self._rope_head_size
                and attn.rotary_emb.is_neox_style == self._rope_is_neox
            ), (
                "All layers must have the same RoPE parameters for Domino "
                "precomputation"
            )

        # Layer metadata
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.k_norm.variance_epsilon
        for attn in layers_attn[1:]:
            assert (
                attn.kv_size == self._kv_size
                and attn.head_dim == self._head_dim
                and attn.num_kv_heads == self._num_kv_heads
                and attn.k_norm.variance_epsilon == self._rms_norm_eps
            ), (
                "All layers must have the same attn config for Domino "
                "precomputation"
            )

        # References to inner Attention layers for direct cache writes
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]
        self._use_fused_context_kv = True
        self._build_hca_pools()

    def _build_hca_pools(self) -> None:
        """Allocate the persistent per-layer HCA pools and compressor states.

        The dense HCA read path may be CUDA-graph captured, so every pool is a
        fixed ``[max_num_seqs, max_entries, head_dim]`` buffer allocated once
        and rewritten in place.  The incremental state is precompute-only
        (outside the graph); it keeps the currently open compressor windows per
        request row so decode does not need the full context feature.
        """

        if self.layer_hca_specs is None or self._hca_pool_seqlen is None:
            return
        if getattr(self, "_hca_pools_built", False):
            return
        device = self.layers[0].self_attn.q_proj.weight.device
        # Never inherit a quantized (integer/packed) storage dtype from q_proj:
        # the entry pool must stay in the floating activation dtype it is
        # concatenated with in the dense read.
        q_norm = getattr(self.layers[0].self_attn, "q_norm", None)
        norm_weight = getattr(q_norm, "weight", None)
        if norm_weight is None or not norm_weight.is_floating_point():
            norm_weight = getattr(getattr(self, "norm", None), "weight", None)
        dtype = (
            norm_weight.dtype
            if norm_weight is not None and norm_weight.is_floating_point()
            else self.layers[0].self_attn.q_proj.weight.dtype
        )
        if not dtype.is_floating_point():
            dtype = torch.bfloat16
        max_seqs = self._hca_pool_seqlen
        self._hca_pools_built = True
        self._hca_pos_last = torch.full(
            (max_seqs,), -1, device=device, dtype=torch.int64
        )
        self._hca_row2block = torch.full(
            (max_seqs,), -1, device=device, dtype=torch.int64
        )
        self._hca_block2row: torch.Tensor | None = None
        self._hca_row_cursor = 0

        for layer in self.layers:
            attn = layer.self_attn
            if attn.hca is None:
                continue
            spec = attn.hca
            capacity = spec["max_entries"]
            attn._hca_pool = (
                torch.zeros(
                    max_seqs, capacity, attn.head_dim, device=device, dtype=dtype
                ),
                torch.zeros(
                    max_seqs, capacity, attn.head_dim, device=device, dtype=dtype
                ),
                torch.zeros(max_seqs, device=device, dtype=torch.int32),
                torch.zeros(max_seqs, device=device, dtype=torch.int32),
            )
            attn._hca_state = HcaIncrementalState(
                max_seqs,
                int(spec["coff"]),
                attn.head_dim,
                int(spec["win"]),
                int(spec["hop"]),
                device=device,
            )
            attn._hca_block2row = None

    def _sync_hca_block_map(self) -> None:
        """Share the block→row map with the dense read path."""

        for layer in self.layers:
            attn = layer.self_attn
            if attn.hca is not None:
                attn._hca_block2row = self._hca_block2row

    def _hca_reset_row(self, row: int) -> None:
        """Drop every HCA layer's pool/state for one row."""

        self._hca_pos_last[row] = -1
        for layer in self.layers:
            attn = layer.self_attn
            if attn.hca is None:
                continue
            attn._hca_state.reset_row(row)
            pool_k, pool_v, first_index, stored = attn._hca_pool
            first_index[row] = 0
            stored[row] = 0
            pool_k[row].zero_()
            pool_v[row].zero_()

    def _hca_row_for_block(self, block_id: int, num_blocks: int) -> int:
        """Map a request's first physical block to a persistent pool row."""

        if self._hca_block2row is None or self._hca_block2row.numel() < num_blocks:
            old = self._hca_block2row
            new = torch.full(
                (num_blocks,),
                -1,
                device=self.layers[0].self_attn.q_proj.weight.device,
                dtype=torch.int64,
            )
            if old is not None:
                new[: old.numel()].copy_(old)
            self._hca_block2row = new
            self._sync_hca_block_map()
        block2row = self._hca_block2row
        row = int(block2row[block_id].item())
        if row >= 0 and int(self._hca_row2block[row].item()) == block_id:
            return row

        max_seqs = self._hca_pool_seqlen
        row = self._hca_row_cursor % max_seqs
        self._hca_row_cursor = (self._hca_row_cursor + 1) % max_seqs
        old_block = int(self._hca_row2block[row].item())
        if old_block >= 0 and old_block < block2row.numel():
            block2row[old_block] = -1
        self._hca_reset_row(row)
        self._hca_row2block[row] = block_id
        block2row[block_id] = row
        return row

    def _hca_group_tokens(
        self,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
        context_block_table: torch.Tensor | None,
    ) -> list[dict] | None:
        """Group new context tokens by request using the paged slot mapping.

        Returns ``None`` when the caller did not supply enough information
        (dummy runs / legacy callers).  Production spec-decode passes both the
        slot mapping and the block table.
        """

        if context_slot_mapping is None or context_block_table is None:
            return None
        if not isinstance(context_block_table, torch.Tensor):
            return None
        slots = context_slot_mapping
        if isinstance(slots, (list, tuple)):
            slots = next((item for item in slots if item is not None), None)
            if slots is None:
                return None
        if not isinstance(slots, torch.Tensor):
            return None
        if context_block_table.dim() != 2 or slots.numel() < context_positions.numel():
            return None

        hca_layer = next(
            layer.self_attn for layer in self.layers if layer.self_attn.hca is not None
        )
        block_size = _hca_cache_block_size(hca_layer.attn.kv_cache)
        positions = context_positions.detach().to("cpu").tolist()
        slot_list = slots[: len(positions)].detach().to("cpu").tolist()
        table = context_block_table.detach().to("cpu").tolist()

        block_to_req: dict[int, int] = {}
        for req, blocks in enumerate(table):
            for block in blocks:
                if block is not None and int(block) >= 0:
                    block_to_req.setdefault(int(block), req)

        per_req: list[list[int]] = [[] for _ in table]
        for token, (slot, _) in enumerate(zip(slot_list, positions)):
            if slot is None or slot < 0:
                continue
            req = block_to_req.get(int(slot) // block_size)
            if req is None:
                continue
            per_req[req].append(token)

        groups = []
        for req, indices in enumerate(per_req):
            if not indices:
                continue
            pos = [positions[token] for token in indices]
            start, end = pos[0], pos[-1] + 1
            if pos != list(range(start, end)):
                # Rejected/padded tokens must not leave a hole in the running
                # window; skip this request this round and keep its old state.
                logger.warning_once(
                    "HCA precompute: non-contiguous context tokens for request "
                    "%d; leaving its entry pool unchanged",
                    req,
                )
                continue
            first_block = next(
                (int(block) for block in table[req] if int(block) >= 0), -1
            )
            if first_block < 0:
                continue
            groups.append(
                {
                    "req": req,
                    "block_id": first_block,
                    "indices": torch.tensor(
                        indices,
                        device=context_positions.device,
                        dtype=torch.long,
                    ),
                    "start": int(start),
                    "end": int(end),
                }
            )
        return groups

    def _hca_layer_normed(
        self,
        context_states: torch.Tensor,
        indices: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Normalized context feature for one HCA layer and one request."""

        num_ctx = context_states.shape[0]
        if self.is_flare:
            feature = context_states.view(
                num_ctx, self._num_attn_layers, self.context_hidden_size
            )[indices, layer_idx]
        else:
            feature = context_states[indices]
        feature = feature.contiguous()
        normed = torch.empty_like(feature)
        ops.rms_norm(
            normed,
            feature,
            self._hidden_norm_weight,
            self._hidden_norm_eps,
        )
        return normed

    def _hca_entry_kv(
        self,
        attn: nn.Module,
        compressor: DominoHCACompressor,
        pooled: torch.Tensor,
        window: int,
        *,
        hop: int,
        coff: int,
        win: int,
        pool_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm + anchor RoPE for one pooled compressor entry."""

        entry = pooled.to(pool_dtype)
        entry = compressor.norm(entry.unsqueeze(0))[0]
        anchor = window_anchor(window, hop, coff, win)
        anchor_t = torch.tensor(
            [anchor], device=entry.device, dtype=torch.long
        )
        k_query = entry.view(1, 1, -1)
        k_entry = attn.rotary_emb(anchor_t, k_query, k_query.clone())[0]
        return k_entry[0], entry

    def _hca_prefill_group(
        self,
        context_len: int,
        normed: torch.Tensor,
        attn: nn.Module,
        row: int,
    ) -> None:
        """One-shot pool the closed windows, then seed the open tail state."""

        compressor = attn.hca_compressor
        spec = attn.hca
        pool = attn._hca_pool
        pool_dtype = pool[0].dtype
        hop, coff, win = spec["hop"], spec["coff"], spec["win"]

        pooled, emitted = compressor.pool(normed)
        if emitted:
            pooled = pooled.to(pool_dtype)
            anchors = torch.arange(
                emitted, device=pooled.device
            ) * hop + compressor.anchor_offset
            k_query = pooled.unsqueeze(1)
            k_entry = attn.rotary_emb(
                anchors, k_query, k_query.clone()
            )[0].squeeze(1)
            write_prefill_pool(
                pool[0],
                pool[1],
                pool[2],
                pool[3],
                row,
                k_entry.to(pool_dtype),
                pooled,
            )
        else:
            pool[2][row] = 0
            pool[3][row] = 0

        state = attn._hca_state
        state.reset_row(row)
        kv = compressor.kv_proj(normed)[0]
        gate = compressor.gate_proj(normed)[0]
        for window in _hca_open_windows(context_len, win, hop):
            start = window * hop
            if start < 0 or start >= context_len:
                continue
            phase = window % coff
            for pos in range(start, context_len):
                pooled_step = state.update(
                    row,
                    phase,
                    window,
                    pos,
                    gate[pos] + compressor.position_bias[pos - start],
                    kv[pos],
                )
                if pooled_step is not None:
                    # Defensive: an "open" window should not close here.
                    k_entry, v_entry = self._hca_entry_kv(
                        attn,
                        compressor,
                        pooled_step,
                        window,
                        hop=hop,
                        coff=coff,
                        win=win,
                        pool_dtype=pool_dtype,
                    )
                    append_pool_entry(
                        pool[0], pool[1], pool[2], pool[3], row, window,
                        k_entry, v_entry,
                    )

    def _hca_decode_group(
        self,
        positions: torch.Tensor,
        normed: torch.Tensor,
        attn: nn.Module,
        row: int,
    ) -> None:
        """Consume one request's new decode tokens token-by-token."""

        compressor = attn.hca_compressor
        spec = attn.hca
        pool = attn._hca_pool
        pool_dtype = pool[0].dtype
        hop, coff, win = spec["hop"], spec["coff"], spec["win"]
        state = attn._hca_state
        kv = compressor.kv_proj(normed)[0]
        gate = compressor.gate_proj(normed)[0]
        bias = compressor.position_bias

        pos_list = positions.detach().to("cpu").tolist()
        for local, pos in enumerate(pos_list):
            for window, phase in window_phases(pos, hop, coff):
                offset = pos - window * hop
                if offset < 0 or offset >= win:
                    continue
                pooled = state.update(
                    row,
                    phase,
                    window,
                    pos,
                    gate[local] + bias[offset],
                    kv[local],
                )
                if pooled is None:
                    continue
                k_entry, v_entry = self._hca_entry_kv(
                    attn,
                    compressor,
                    pooled,
                    window,
                    hop=hop,
                    coff=coff,
                    win=win,
                    pool_dtype=pool_dtype,
                )
                append_pool_entry(
                    pool[0], pool[1], pool[2], pool[3], row, window,
                    k_entry, v_entry,
                )

    @torch.inference_mode()
    def _pool_hca_entries(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
        context_block_table: torch.Tensor | None = None,
    ) -> None:
        """Advance every HCA layer's compressor state by this round's tokens.

        The context states contain only the tokens scheduled for this round;
        the target model keeps the older context in its KV cache.  Closed
        windows are pooled once at prefill and then appended incrementally
        during decode, so the entry pool is never rebuilt from a partial
        context feature.
        """

        self._build_hca_pools()
        if not getattr(self, "_hca_pools_built", False):
            return
        groups = self._hca_group_tokens(
            context_positions, context_slot_mapping, context_block_table
        )
        if groups is None:
            if context_slot_mapping is None:
                return
            logger.warning_once(
                "HCA precompute requires context_slot_mapping and "
                "context_block_table for incremental compressor state; "
                "leaving the entry pool unchanged"
            )
            return

        assert context_block_table is not None
        first_hca = next(
            layer.self_attn for layer in self.layers if layer.self_attn.hca is not None
        )
        num_blocks = _hca_cache_num_blocks(first_hca.attn.kv_cache)
        if num_blocks <= 0:
            num_blocks = int(context_block_table.max().item()) + 1
        for group in groups:
            block_id = group["block_id"]
            row = self._hca_row_for_block(block_id, max(num_blocks, 1))
            start, end = group["start"], group["end"]
            indices = group["indices"]

            expected = start - 1
            if int(self._hca_pos_last[row].item()) != expected:
                self._hca_reset_row(row)
                self._hca_pos_last[row] = expected

            for layer_idx, spec in enumerate(self.layer_hca_specs):
                if spec is None:
                    continue
                attn = self.layers[layer_idx].self_attn
                normed = self._hca_layer_normed(
                    context_states, indices, layer_idx
                )
                if start == 0:
                    self._hca_prefill_group(end, normed, attn, row)
                else:
                    self._hca_decode_group(
                        context_positions[indices], normed, attn, row
                    )
                bucket = getattr(attn, "hca_entry_bucket", None)
                if bucket is not None:
                    stored = int(attn._hca_pool[3][row].item())
                    if stored > bucket:
                        raise ValueError(
                            "HCA entry pool has "
                            f"{stored} entries but hca_entry_bucket={bucket}; "
                            "raise dflash_config.hca_entry_bucket or set it to "
                            "null"
                        )
            self._hca_pos_last[row] = end - 1


    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # --- Fused KV projection (one batched GEMM for all layers) ---
        if self.is_flare:
            # All layers share the same hidden_norm, so the [T, D, H] context is
            # flattened to [T*D, H], normalized with a single kernel, then
            # projected through the per-layer K/V weights with one bmm.
            H = self.context_hidden_size
            normed_context_states = torch.empty_like(context_states)
            ops.rms_norm(
                normed_context_states.view(num_ctx * num_layers, H),
                context_states.reshape(num_ctx * num_layers, H),
                self._hidden_norm_weight,
                self._hidden_norm_eps,
            )
            fused = normed_context_states.view(num_ctx, num_layers, H)
            all_kv_flat = torch.bmm(
                fused.permute(1, 0, 2).contiguous(), self._fused_kv_weight_T
            )
            if self._fused_kv_bias is not None:
                all_kv_flat = all_kv_flat + self._fused_kv_bias.unsqueeze(1)
            # [D, T, 2, nkv, hd]; dim-2 slices are contiguous K and V.
            all_kv = all_kv_flat.view(
                num_layers, num_ctx, 2, num_kv_heads, head_dim
            )
        else:
            # fc: one shared context feature of draft width, every layer
            # projects it with its own (shared with the draft stream) K/V
            # weights -- one plain GEMM against the stacked [D * 2kv, H] weight.
            normed_context_states = torch.empty_like(context_states)
            ops.rms_norm(
                normed_context_states,
                context_states,
                self._hidden_norm_weight,
                self._hidden_norm_eps,
            )
            all_kv_flat = F.linear(
                normed_context_states,
                self._fused_kv_weight_flat,
                self._fused_kv_bias_flat,
            )
            all_kv = all_kv_flat.view(
                num_ctx, num_layers, 2, num_kv_heads, head_dim
            ).permute(1, 0, 2, 3, 4)
        all_k = all_kv[:, :, 0]  # [D, T, nkv, hd]
        all_v = all_kv[:, :, 1]
        return all_k, all_v

    def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
        # --- Grouped RMSNorm K across all layers ([D, T, nkv, hd]) ---
        # The weight is selected per layer by the outermost (layer) index.
        all_k_normed = torch.empty_like(all_k)
        ops.rms_norm(
            all_k_normed,
            all_k,
            self._k_norm_weights,
            self._rms_norm_eps,
        )
        return all_k_normed

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
        context_block_table: torch.Tensor | None = None,
    ) -> None:
        """Project fused context states and write them into draft KV caches.

        ``context_states`` is the output of
        :meth:`Qwen3DominoForCausalLM.combine_hidden_states`: ``[T, D * H_t]``
        for flare (one routed feature per draft layer) or ``[T, H_d]`` for fc
        (one shared feature read by every layer).

        ``context_block_table`` is the paged block table for the draft cache
        group.  HCA needs it (together with ``context_slot_mapping``) to map
        this round's tokens back to a request row and advance the incremental
        compressor state; the dense draft read path does not use it.
        """
        expected_width = (
            self.config.num_hidden_layers * self.context_hidden_size
            if self.is_flare
            else self.context_hidden_size
        )
        if context_states.dim() != 2 or context_states.shape[-1] != expected_width:
            raise ValueError(
                "Domino precompute expects 2D "
                f"{'flare-fused' if self.is_flare else 'fc-fused'} context "
                f"states of width {expected_width}, got "
                f"{tuple(context_states.shape)}"
            )

        if not hasattr(self, "_num_attn_layers"):
            logger.warning_once(
                "Domino fused buffer initialization was skipped. If dummy "
                "weights are not in use, this may indicate an error in weight "
                "loading."
            )
            self._build_fused_kv_buffers()

        if not self._use_fused_context_kv:
            if self.layer_hca_specs is not None:
                raise NotImplementedError(
                    "HCA (heavily_compressed_attention) drafts require a "
                    "fused context-KV precompute (bf16, or vLLM-Ascend's "
                    "quantized w4a8/w8a8 fused path); the per-layer fallback "
                    "carries no pooling buffers (v1)."
                )
            self._precompute_and_store_context_kv_per_layer(
                context_states, context_positions, context_slot_mapping
            )
            return

        num_ctx = context_states.shape[0]
        D = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads

        all_k, all_v = self._project_context_kv(
            context_states, num_ctx, D, nkv, hd
        )
        all_k_normed = self._normalize_context_k(all_k)

        # --- Fused RoPE across all layers ---
        # View as [D * num_ctx, kv] so RoPE sees one big batch (no copy).
        # In-place RoPE: pass K as the "query" arg with key=None.
        all_k_flat = all_k_normed.view(D * num_ctx, kv)
        positions_repeated = context_positions.repeat(D)
        cos_sin_cache = self._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(
            positions_repeated,
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )

        if context_slot_mapping is None:
            return

        # --- Per-layer cache insert ---
        all_k_final = all_k_flat.view(D, num_ctx, nkv, hd)
        per_layer = isinstance(context_slot_mapping, (list, tuple))
        for i in range(D):
            slot_mapping = (
                context_slot_mapping[i] if per_layer else context_slot_mapping
            )
            if slot_mapping is None:
                continue  # dummy run: skip cache ops
            attn = self._attn_layers[i]
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                slot_mapping,
            )

        if self.layer_hca_specs is not None:
            self._pool_hca_entries(
                context_states,
                context_positions,
                context_slot_mapping,
                context_block_table,
            )

    def _precompute_and_store_context_kv_per_layer(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Per-layer fallback of :meth:`precompute_and_store_context_kv`."""
        num_ctx = context_states.shape[0]
        D = self.config.num_hidden_layers
        per_layer = isinstance(context_slot_mapping, (list, tuple))
        if self.is_flare:
            H = self.context_hidden_size
            fused = context_states.view(num_ctx, D, H)
            shared_ctx = None
        else:
            fused = None
            # fc: normalize the one shared feature once, then project it with
            # every layer's own K/V weights.
            shared_ctx = self.hidden_norm(context_states)

        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            ctx = (
                self.hidden_norm(fused[:, i, :])
                if self.is_flare
                else shared_ctx
            )
            k_proj = (
                attn.k_proj_target
                if attn.k_proj_target is not None
                else attn.k_proj
            )
            v_proj = (
                attn.v_proj_target
                if attn.v_proj_target is not None
                else attn.v_proj
            )
            k = k_proj(ctx)
            v = v_proj(ctx)
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

    @torch.inference_mode()
    def write_attention_sinks(
        self,
        sink_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        """Write the static per-layer sink K/V rows into the draft caches.

        ``sink_slot_mapping`` follows the context pass convention: one tensor
        of slot ids per request (one sink row), either shared by all layers or
        a per-layer list. ``None`` (or per-layer ``None`` entries) skips the
        write, mirroring the dummy-run behavior of the context precompute.
        Raw rows: no k_norm, no RoPE — the parameters are the cache content.
        """
        if not self.num_sinks or sink_slot_mapping is None:
            return
        per_layer = isinstance(sink_slot_mapping, (list, tuple))
        tp_rank = get_tensor_model_parallel_rank()
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            slot_mapping = (
                sink_slot_mapping[i] if per_layer else sink_slot_mapping
            )
            if slot_mapping is None:
                continue
            num_reqs = slot_mapping.shape[0]
            if attn.total_num_kv_heads > attn.num_kv_heads:
                head0 = tp_rank * attn.num_kv_heads
                heads = slice(head0, head0 + attn.num_kv_heads)
            else:
                heads = slice(None)
            # kv_cache may be a plain tensor or a runner-owned tuple; take
            # the first tensor's dtype so the sink rows match the cache.
            kv_cache = attn.attn.kv_cache
            cache_dtype = (
                kv_cache.dtype
                if torch.is_tensor(kv_cache)
                else next(t.dtype for t in kv_cache if torch.is_tensor(t))
            )
            k = (
                attn.sink_k[0:1, heads]
                .to(dtype=cache_dtype)
                .expand(num_reqs, -1, -1)
                .contiguous()
            )
            v = (
                attn.sink_v[0:1, heads]
                .to(dtype=cache_dtype)
                .expand(num_reqs, -1, -1)
                .contiguous()
            )
            attn.attn.impl.do_kv_cache_update(
                attn.attn,
                k,
                v,
                kv_cache,
                slot_mapping,
            )

    hf_to_vllm_mapper = _DominoWeightsMapper(
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
        # None for plain-GQA checkpoints; otherwise the loader folds the
        # VHA training parameterization back into q_proj/o_proj at load.
        self._vha_physical = _resolve_vha_load_spec(self.config)

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
            if self.model.shared_emb_readout:
                raise ValueError(
                    "shared_emb_readout requires draft_vocab_size == "
                    "vocab_size (the folded head serves the full target "
                    "vocab)"
                )
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

    @property
    def num_sinks(self) -> int:
        return self.model.num_sinks

    def write_attention_sinks(self, sink_slot_mapping) -> None:
        self.model.write_attention_sinks(sink_slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.model.shared_emb_readout:
            # The lm_head is the folded [V, emb] head; the hidden must pass
            # through head_down_proj first.
            hidden_states = self.model.head_down_proj(hidden_states)
        elif self.model.out_proj_after_norm and self.model.output_proj is not None:
            # Unfolded out_proj_after_norm fallback: output_proj survived load
            # because the lm_head was not foldable.
            hidden_states = self.model.output_proj(hidden_states)
        return self.logits_processor(self.lm_head, hidden_states)

    def head_emb(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[N, hidden] -> [N, emb] base part of the shared readout."""
        return self.model.head_down_proj(hidden_states)

    def correction_emb(self, gru_hidden: torch.Tensor) -> torch.Tensor:
        """[B, gru] -> [B, emb] correction part of the shared readout."""
        return self.model.embed_proj(gru_hidden)

    def emb_logits(self, u: torch.Tensor) -> torch.Tensor:
        """[B, emb] -> [B, V] through the folded shared head."""
        return self.logits_processor(self.lm_head, u)

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
        """Fuse raw target aux features into the draft context feature(s).

        Input: ``[N, num_target_features * target_hidden_size]``.

        Output: ``[N, num_draft_layers * target_hidden_size]`` for flare (one
        routed feature per draft layer) or ``[N, hidden_size]`` for fc (one
        shared feature projected from the concatenated taps).
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
        # Mirror the trained per-tap RMSNorm before the fusion: the served
        # Laguna drafts slice, normalize and re-concatenate in the same order.
        if self.model.aux_hidden_norms is not None:
            normed = torch.empty_like(target)
            for index, norm in enumerate(self.model.aux_hidden_norms):
                normed[:, index, :] = norm(target[:, index, :])
            target = normed
        if not self.model.is_flare:
            # fc fusion: project the normalized concat to draft width once and
            # normalize; every draft layer reads this same feature.
            fused = self.model.hidden_norm(
                self.model.fc(target.reshape(target.shape[0], -1))
            )
            if needs_squeeze:
                fused = fused.squeeze(0)
            return fused
        fusion_w = torch.softmax(self.model.layer_fusion_weights, dim=1)
        # [D, T] @ [N, T, H] -> [N, D, H] (broadcast matmul, no
        # materialization of [N, D, T, H]).  Plain matmul instead of einsum:
        # the generic aclnnEinsum path is much slower on NPU (specforge
        # observed the same speedup by avoiding einsum in the flare fusion).
        fused = torch.matmul(fusion_w, target).reshape(
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
        context_block_table: torch.Tensor | None = None,
    ) -> None:
        """Project the fused context states and write them into draft KV caches."""
        self.model.precompute_and_store_context_kv(
            context_states,
            context_positions,
            context_slot_mapping,
            context_block_table,
        )

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        return [layer.self_attn.causal for layer in self.model.layers]

    @property
    def shared_emb_readout(self) -> bool:
        return self.model.shared_emb_readout

    def correction_bias(self, features: torch.Tensor) -> torch.Tensor:
        """Domino residual logit correction for ``[B, hidden + gru]`` features."""
        if self.model.shared_emb_readout:
            raise RuntimeError(
                "correction_bias is the legacy two-path head; the "
                "shared_emb_readout loop composes in emb space via "
                "head_emb/correction_emb/emb_logits"
            )
        bias = None
        if self.model.embed_proj is not None:
            x = self.model.embed_proj[0](features)
            x = self.model.embed_proj[1](x)
            bias = self.logits_processor(self.model.embed_proj[2], x)

        if self.model.hidden_proj is not None:
            h = self.model.hidden_proj(features)
            # hidden_proj outputs target-width states; under a folded lm_head
            # the unfolded head is kept around for exactly this path.
            head = getattr(self, "_unfolded_lm_head", None) or self.lm_head
            hidden_bias = self.logits_processor(head, h)
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

        # shared_emb_readout is a different parameterization, not a
        # re-parameterization of the legacy two-path head: the draft-owned
        # vocab matrix (embed_proj.2) has no counterpart in the shared head.
        # Catch the mismatch here instead of letting AutoWeightsLoader emit
        # its generic "no module named embed_proj.0" error.
        has_legacy_head = any(
            "embed_proj.0." in name or "embed_proj.2." in name
            for name in model_weights
        )
        has_shared_head = any(
            name == "model.head_down_proj.weight" for name in model_weights
        )
        if self.model.shared_emb_readout and (has_legacy_head or not has_shared_head):
            raise ValueError(
                "shared_emb_readout=true but the draft checkpoint carries "
                "the legacy two-path correction head (embed_proj.0/.2). "
                "The shared head must be trained from scratch with the "
                "flag; serve legacy checkpoints with the flag off."
            )
        if not self.model.shared_emb_readout and has_shared_head:
            raise ValueError(
                "the draft checkpoint carries shared_emb_readout "
                "parameters (head_down_proj) but "
                "dflash_config.shared_emb_readout is off; enable the flag "
                "in the draft config."
            )

        found_vha = fold_vha_weights(
            model_weights,
            num_heads=int(self.config.num_attention_heads),
            head_dim=self.model.layers[0].self_attn.head_dim,
        )
        if self._vha_physical is not None and not found_vha:
            raise ValueError(
                "dflash_config.vha is set but the checkpoint carries no "
                "vha_* parameters; either point at the VHA training "
                "checkpoint or drop the key (already-folded exports)"
            )
        if self._vha_physical is None and found_vha:
            raise ValueError(
                "checkpoint carries vha_* parameters but dflash_config.vha "
                "is unset; add the key (physical_q_heads) to the draft "
                "config so the loader folds them"
            )

        skip_substrs = ["mask_embedding"]
        if self.model.is_flare:
            # flare checkpoints ship layer_fusion_weights instead of fc.
            skip_substrs.append("fc.")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")

        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()


def _fold_output_proj_into_lm_head(draft_model: nn.Module) -> bool:
    """Fold ``output_proj`` into the installed lm_head: ``W_lm @ W_out``.

    The lm_head is frozen, so per TP rank
    ``weight[V_shard, H_target] @ output_proj[H_target, H_draft]`` is an exact
    replacement for the two-GEMM sequence. On success the draft runs one
    draft-width head GEMM per step and ``output_proj`` is dropped; the
    unfolded head is kept for the hidden_proj correction path. Heads that are
    not dense unquantized weights (e.g. a packed quantized target head) stay
    unfolded: ``output_proj`` then runs inside ``compute_draft_logits``.
    """
    model = draft_model.model
    if not model.out_proj_after_norm or model.output_proj is None:
        return False
    head = draft_model.lm_head
    weight = getattr(head, "weight", None)
    if (
        head is None
        or not isinstance(
            getattr(head, "quant_method", None), UnquantizedEmbeddingMethod
        )
        or weight is None
        or not weight.is_floating_point()
    ):
        logger.warning(
            "out_proj_after_norm=true but the draft lm_head is not foldable "
            "(missing or non-dense head); keeping the separate output_proj "
            "GEMM in compute_draft_logits"
        )
        return False

    out_w = model.output_proj.weight.data
    fused = torch.matmul(weight.data, out_w).to(weight.dtype)
    new_head = ParallelLMHead(
        head.num_embeddings,
        out_w.shape[1],
        params_dtype=weight.dtype,
        padding_size=head.padding_size,
        prefix="lm_head",
    )
    assert new_head.weight.shape[0] == fused.shape[0], (
        f"folded lm_head shard {tuple(fused.shape)} does not match a fresh "
        f"head shard {tuple(new_head.weight.shape)}"
    )
    new_head.weight = nn.Parameter(fused, requires_grad=False)
    if model.hidden_proj is not None:
        draft_model._unfolded_lm_head = head
    draft_model.lm_head = new_head
    model.output_proj = None
    logger.info(
        "out_proj_after_norm: folded output_proj into lm_head "
        f"{tuple(weight.shape)} @ {tuple(out_w.shape)} -> {tuple(fused.shape)}"
    )
    return True


def _fold_shared_emb_readout(draft_model: nn.Module) -> None:
    """Fold the shared emb readout: ``W_lm @ W_up`` into a ``[V, emb]`` head.

    Every correction-step projection (base + correction) then runs as one
    emb-width GEMM on ``head_down_proj(z) + embed_proj(gru_state)``. There is
    deliberately no unfolded fallback: without the fold each step would run
    three vocab-width GEMMs (two of them full target-width), strictly worse
    than the flag-off paths, so an unfoldable head raises instead.
    """
    model = draft_model.model
    head = draft_model.lm_head
    weight = getattr(head, "weight", None)
    if (
        head is None
        or not isinstance(
            getattr(head, "quant_method", None), UnquantizedEmbeddingMethod
        )
        or weight is None
        or not weight.is_floating_point()
    ):
        raise RuntimeError(
            "shared_emb_readout=true but the draft lm_head is not foldable "
            "(missing, quantized or non-dense head); keep the lm_head out of "
            "quantization or turn the flag off"
        )

    up_w = model.head_up_proj.weight.data
    fused = torch.matmul(weight.data, up_w).to(weight.dtype)
    new_head = ParallelLMHead(
        head.num_embeddings,
        up_w.shape[1],
        params_dtype=weight.dtype,
        padding_size=head.padding_size,
        prefix="lm_head",
    )
    assert new_head.weight.shape[0] == fused.shape[0], (
        f"folded shared-readout head shard {tuple(fused.shape)} does not "
        f"match a fresh head shard {tuple(new_head.weight.shape)}"
    )
    new_head.weight = nn.Parameter(fused, requires_grad=False)
    draft_model.lm_head = new_head
    model.head_up_proj = None
    logger.info(
        "shared_emb_readout: folded lm_head @ head_up_proj into a "
        f"{tuple(fused.shape)} emb-width head"
    )


def load_domino_model(
    target_model: nn.Module,
    vllm_config: VllmConfig,
) -> nn.Module:
    """Load the Domino draft model and share target embed/lm_head when absent."""
    from vllm.compilation.backends import set_model_tag
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.model_executor.model_loader import get_model
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
            use_non_causal=domino_has_any_non_causal(
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

    if getattr(draft_model.model, "shared_emb_readout", False):
        _fold_shared_emb_readout(draft_model)
    elif getattr(draft_model.model, "out_proj_after_norm", False):
        _fold_output_proj_into_lm_head(draft_model)

    return draft_model
