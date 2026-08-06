# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Domino speculator: block-parallel draft backbone + GRU-corrected sampling."""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


class DominoSpeculator(DSparkSpeculator):
    """Domino is DSpark-shaped, but replaces the Markov head with a GRU
    correction loop.

    Execution shape:
      * ``num_query_per_req == num_speculative_tokens`` (anchor-as-first),
      * one parallel draft forward,
      * left-to-right sampling where each position adds a GRU-conditioned
        residual logit correction.
    """

    _speculator_name = "Domino"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.sample_from_anchor = True
        self.num_query_per_req = self.num_speculative_steps

        hf_config = self.draft_model_config.hf_config
        dflash_config = getattr(hf_config, "dflash_config", None) or {}
        self.target_hidden_size = dflash_config.get(
            "target_hidden_size", hf_config.hidden_size
        )
        self.num_draft_layers = hf_config.num_hidden_layers

        # Domino uses the flare-fused context representation in the proposer
        # buffer: [num_tokens, num_draft_layers * target_hidden_size].
        self.hidden_size = self.num_draft_layers * self.target_hidden_size
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Let DFlash's graph manager decide: FULL / FULL_DECODE_ONLY is
        # captured as an ACL graph on Ascend, while PIECEWISE / NONE stays
        # eager. The fixed-length GRU correction loop is unrolled during
        # capture, so it is part of the same graph.
        super().init_cudagraph_manager(cudagraph_mode)

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        from vllm.model_executor.models.qwen3_domino import load_domino_model

        model = load_domino_model(target_model, self.vllm_config)
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=self.device,
            )
        return model

    def _sample_step(
        self,
        logits_i: torch.Tensor,
        idx_map_i: torch.Tensor,
        sample_pos_i: torch.Tensor,
        col: int,
    ) -> torch.Tensor:
        if self.draft_logits is not None:
            if self._d2t_scatter_index is not None:
                assert self._draft_scatter_buf is not None
                buf = self._draft_scatter_buf[: logits_i.shape[0]]
                buf.index_copy_(1, self._d2t_scatter_index, logits_i.to(buf.dtype))
                logits_i = buf
            return gumbel_sample(
                logits_i,
                idx_map_i,
                self.temperature,
                self.seeds,
                sample_pos_i - 1,
                apply_temperature=True,
                output_processed_logits=self.draft_logits,
                output_processed_logits_col=self._step_cols[col],
                use_fp64=self.use_fp64_gumbel,
            )
        return self.model.map_draft_to_target(logits_i.argmax(dim=-1))

    def _sample_sequential(
        self,
        num_reqs: int,
        head_hidden: torch.Tensor,
    ) -> None:
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]

        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)

        prefix_len = int(getattr(self.model, "pure_draft_prefix_len", 0))
        if prefix_len > n_spec:
            raise ValueError(
                f"Domino pure_draft_prefix_len ({prefix_len}) cannot exceed "
                f"num_speculative_tokens ({n_spec})"
            )

        anchor = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        prefix_ids = torch.empty(
            num_reqs,
            1 + prefix_len,
            dtype=torch.int64,
            device=self.device,
        )
        prefix_ids[:, 0] = anchor

        for i in range(prefix_len):
            draft_i = self._sample_step(
                base_logits[:, i],
                idx_map[:, i],
                sample_pos[:, i],
                i,
            )
            self.draft_tokens[:num_reqs, i] = draft_i
            prefix_ids[:, 1 + i] = draft_i

        prefix_embeds = self.model.embed_tokens(prefix_ids)
        _, gru_hidden = self.model.domino_gru_forward(prefix_embeds)

        sample_hidden_3d = sample_hidden.view(num_reqs, n_spec, -1)
        for i in range(prefix_len, n_spec):
            z_i = sample_hidden_3d[:, i]
            s_i = gru_hidden[0]
            features = torch.cat([z_i, s_i], dim=-1)
            correction = self.model.correction_bias(features)
            logits_i = base_logits[:, i] + correction

            draft_i = self._sample_step(
                logits_i,
                idx_map[:, i],
                sample_pos[:, i],
                i,
            )
            self.draft_tokens[:num_reqs, i] = draft_i

            if i + 1 < n_spec:
                next_embeds = self.model.embed_tokens(draft_i)
                _, gru_hidden = self.model.domino_gru_cell(
                    next_embeds,
                    gru_hidden,
                )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._sample_sequential(num_reqs, head_hidden)
