from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import torch
from freetoken.engine.graph import project_lm_head_all_positions

from freetoken.speculative.dflash.config import DFlashConfig
from freetoken.speculative.dflash.model import DFlashDraftModel
from freetoken.speculative.dflash.weight import iter_dflash_weights

if TYPE_CHECKING:
    from freetoken.engine.engine import Engine


def _dflash_worker_timing_now(device: torch.device) -> float | None:
    if os.environ.get("FREETOKEN_DEBUG_DFLASH_TIMING", "0") != "1":
        return None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _dflash_worker_elapsed_ms(start: float | None, end: float | None) -> float:
    if start is None or end is None:
        return 0.0
    return (end - start) * 1000.0


class DFlashWorker:
    """Orchestrates DFlash draft generation and target-head projection."""

    def __init__(
        self,
        draft_model_path: str,
        target_model,
        engine: Engine,
        device: torch.device,
        block_size: int | None = None,
    ):
        from freetoken.utils import cached_load_hf_config

        # Load draft config
        hf_config = cached_load_hf_config(draft_model_path)
        self.config = DFlashConfig.from_hf_config(hf_config)
        if block_size is not None:
            self.config.block_size = block_size

        self.block_size = self.config.block_size
        self.mask_token_id = self.config.mask_token_id
        self.target_layer_ids = set(self.config.target_layer_ids)
        self.device = device
        self.engine = engine

        # Create and load draft model
        self.draft_model = DFlashDraftModel(self.config)
        state_dict = {}
        for name, t in iter_dflash_weights(draft_model_path, device):
            state_dict[name] = t
        self.draft_model.load_state_dict(state_dict)
        self.draft_model.to(device)

        # Borrow target model's embedding and LM head
        self.target_embed = target_model.model.embed_tokens
        self.target_lm_head = target_model.lm_head

        # Hidden context storage grows lazily and get_context() returns a view into it.
        # Cross-attention needs multiple context positions to differentiate draft tokens.
        self._context_buffer: list[torch.Tensor] = []
        self._context_len = 0
        self._context_storage: torch.Tensor | None = None
        self._context_kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None
        self._context_kv_len = 0
        self.last_context_kv_append_ms = 0.0
        self.last_draft_model_forward_ms = 0.0
        self._mask_embeds: torch.Tensor | None = None
        self._draft_input_storage: torch.Tensor | None = None
        self.last_draft_probs: torch.Tensor | None = None
        self._position_offsets = torch.arange(self.block_size, dtype=torch.int32, device=device)

    def store_hidden_states(self, hidden_states: list[torch.Tensor]) -> None:
        """Store hidden states from a forward pass into the context buffer.

        Args:
            hidden_states: list of [num_tokens, hidden] tensors from target layers.
                           For decode: num_tokens=1. For prefill: num_tokens=prompt_len.
        """
        rows = hidden_states[0].shape[0]
        context_dim = sum(hidden.shape[1] for hidden in hidden_states)
        self._ensure_context_capacity(rows, context_dim, hidden_states[0].dtype, hidden_states[0].device)
        assert self._context_storage is not None
        start = self._context_len
        end = start + rows
        offset = 0
        for hidden in hidden_states:
            width = hidden.shape[1]
            self._context_storage[start:end, offset : offset + width].copy_(hidden)
            offset += width
        context = self._context_storage[start:end]
        self._context_len = end
        append_start = _dflash_worker_timing_now(context.device)
        self._append_context_kv_cache(context, start)
        append_end = _dflash_worker_timing_now(context.device)
        self.last_context_kv_append_ms = _dflash_worker_elapsed_ms(append_start, append_end)

    def _append_context_kv_cache(self, context: torch.Tensor, start_position: int) -> None:
        draft_model = getattr(self, "draft_model", None)
        layers = getattr(draft_model, "layers", None)
        if not layers:
            return
        layer_list = getattr(layers, "op_list", layers)

        project_context_features = getattr(draft_model, "project_context_features", None)
        context_for_attention = project_context_features(context) if project_context_features is not None else context
        positions = torch.arange(
            start_position,
            start_position + context.shape[0],
            dtype=torch.int32,
            device=context_for_attention.device,
        )
        cache = getattr(self, "_context_kv_cache", None)
        if cache is None or len(cache) != len(layer_list):
            cache = [None] * len(layer_list)

        appended = False
        for i, layer in enumerate(layer_list):
            attention = getattr(layer, "self_attn", layer)
            project_context_kv = getattr(attention, "project_context_kv", None)
            if project_context_kv is None:
                continue
            new_k, new_v = project_context_kv(context_for_attention, positions)
            end = start_position + context.shape[0]
            entry = cache[i]
            if entry is None:
                cache[i] = self._alloc_context_kv_entry(new_k, new_v, end)
            else:
                pool_k, pool_v, cap = entry
                if end > cap:
                    cap = max(end, 2 * cap)
                    pool_k = self._grow_context_kv(pool_k, cap)
                    pool_v = self._grow_context_kv(pool_v, cap)
                pool_k[start_position:end].copy_(new_k)
                pool_v[start_position:end].copy_(new_v)
                cache[i] = (pool_k, pool_v, cap)
            appended = True

        if appended:
            self._context_kv_cache = cache
            self._context_kv_len = start_position + context.shape[0]

    @staticmethod
    def _alloc_context_kv_entry(new_k: torch.Tensor, new_v: torch.Tensor, length: int):
        cap = max(length, 2 * new_k.shape[0], 16)
        pool_k = torch.empty((cap, *new_k.shape[1:]), dtype=new_k.dtype, device=new_k.device)
        pool_v = torch.empty((cap, *new_v.shape[1:]), dtype=new_v.dtype, device=new_v.device)
        pool_k[:length].copy_(new_k)
        pool_v[:length].copy_(new_v)
        return (pool_k, pool_v, cap)

    @staticmethod
    def _grow_context_kv(pool: torch.Tensor, cap: int) -> torch.Tensor:
        grown = torch.empty((cap, *pool.shape[1:]), dtype=pool.dtype, device=pool.device)
        grown[: pool.shape[0]].copy_(pool)
        return grown

    def _ensure_context_capacity(
        self,
        add_tokens: int,
        context_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        required = self._context_len + add_tokens
        storage = getattr(self, "_context_storage", None)
        if storage is not None and storage.shape[0] >= required:
            return
        new_capacity = max(required, 2 * (storage.shape[0] if storage is not None else 0), 16)
        new_storage = torch.empty((new_capacity, context_dim), dtype=dtype, device=device)
        if storage is not None and self._context_len > 0:
            new_storage[:self._context_len].copy_(storage[:self._context_len])
        self._context_storage = new_storage

    def target_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project target hidden states without LMHead's prefill last-token slicing."""
        return project_lm_head_all_positions(self.target_lm_head, hidden_states)

    def draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.target_logits(hidden_states)
        config = getattr(self, "config", None)
        logits = logits * float(getattr(config, "output_multiplier", 1.0))
        softcap = getattr(config, "final_logit_softcapping", None)
        if softcap is not None and float(softcap) > 0:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        return logits

    def reset_context(self) -> None:
        """Clear the context buffer (called when a request finishes)."""
        self._context_buffer.clear()
        self._context_len = 0
        self._context_storage = None
        self._context_kv_cache = None
        self._context_kv_len = 0
        self.last_context_kv_append_ms = 0.0
        self.last_draft_model_forward_ms = 0.0
        self._draft_input_storage = None
        self.last_draft_probs = None

    @property
    def context_length(self) -> int:
        return self._context_len

    def get_context(self) -> torch.Tensor:
        """Get the full context features buffer: [seq_len, context_dim]."""
        if self._context_len == 0:
            raise RuntimeError("DFlash context buffer is empty")
        assert self._context_storage is not None
        return self._context_storage[:self._context_len]

    def get_context_kv_cache(self) -> list[tuple[torch.Tensor, torch.Tensor] | None] | None:
        cache = getattr(self, "_context_kv_cache", None)
        if cache is None or getattr(self, "_context_kv_len", 0) != self._context_len:
            return None
        out = []
        for kv in cache:
            if kv is None:
                return None
            if len(kv) == 3:
                pool_k, pool_v, _ = kv
                out.append((pool_k[: self._context_kv_len], pool_v[: self._context_kv_len]))
            else:  # legacy 2-tuple (tests / external callers)
                out.append(kv)
        return out

    def _draft_input_embeds(self, base_token_id: torch.Tensor) -> torch.Tensor:
        if self.block_size == 1:
            return self.target_embed.forward(base_token_id[:1])

        config = getattr(self, "config", None)
        scale = float(getattr(config, "input_embedding_scale", 1.0))
        base_embed = self.target_embed.forward(base_token_id[:1]) * scale
        mask_embeds = getattr(self, "_mask_embeds", None)
        if mask_embeds is None:
            mask_ids = torch.full(
                (self.block_size - 1,),
                self.mask_token_id,
                dtype=torch.int32,
                device=self.device,
            )
            mask_embeds = (self.target_embed.forward(mask_ids) * scale).detach()
            self._mask_embeds = mask_embeds
        storage = getattr(self, "_draft_input_storage", None)
        if (
            storage is None
            or storage.shape != (self.block_size, base_embed.shape[1])
            or storage.dtype != base_embed.dtype
            or storage.device != base_embed.device
        ):
            storage = torch.empty(
                (self.block_size, base_embed.shape[1]),
                dtype=base_embed.dtype,
                device=base_embed.device,
            )
            storage[1:].copy_(mask_embeds)
            self._draft_input_storage = storage
        storage[:1].copy_(base_embed)
        return storage

    def _draft_positions(self, position: int) -> torch.Tensor:
        offsets = getattr(self, "_position_offsets", None)
        if offsets is None or offsets.numel() != self.block_size or offsets.device != self.device:
            offsets = torch.arange(self.block_size, dtype=torch.int32, device=self.device)
            self._position_offsets = offsets
        return offsets + position

    def draft(
        self,
        hidden_states: list[torch.Tensor],  # hidden states from target layers (current pos)
        base_token_id: torch.Tensor,       # [1] — last verified token
        position: int,                      # position of base token
        sampling_args=None,                 # BatchSamplingArgs; None / greedy -> argmax drafts
    ) -> torch.Tensor:
        """Generate block_size draft tokens in parallel.

        Returns: draft_tokens [block_size]
        """
        bs = self.block_size

        # 1. Store current hidden states and get full context
        self.store_hidden_states(hidden_states)
        context_features = self.get_context()  # [seq_len, context_dim]

        mask_embeds = self._draft_input_embeds(base_token_id)
        positions = self._draft_positions(position)

        # 4. Draft model forward
        context_kv_cache = self.get_context_kv_cache()
        forward_start = _dflash_worker_timing_now(self.device)
        if context_kv_cache is None:
            draft_hidden = self.draft_model.forward(mask_embeds, context_features, positions)
        else:
            draft_hidden = self.draft_model.forward(
                mask_embeds,
                context_features,
                positions,
                context_kv_cache=context_kv_cache,
            )
        forward_end = _dflash_worker_timing_now(self.device)
        self.last_draft_model_forward_ms = _dflash_worker_elapsed_ms(forward_start, forward_end)

        # Only positions 1..bs-1 are candidate draft tokens. Position 0 is the
        # sampled target base token and is not consumed by the engine.
        if bs == 1:
            return base_token_id[:1]
        logits = self.draft_logits(draft_hidden[1:])  # [bs - 1, vocab]
        if sampling_args is not None and sampling_args.temperatures is not None:
            # Non-greedy: sample candidates (upstream DFlash samples drafts too) and
            # stash the filtered draft distribution for rejection-sampling verify.
            from freetoken.speculative.utils import sampling_probs

            draft_probs = sampling_probs(logits, sampling_args)
            candidate_tokens = torch.multinomial(draft_probs, 1)[:, 0]
            self.last_draft_probs = draft_probs
        else:
            candidate_tokens = torch.argmax(logits, dim=-1)
            self.last_draft_probs = None
        return torch.cat([base_token_id[:1].to(candidate_tokens.dtype), candidate_tokens])


__all__ = ["DFlashWorker"]
