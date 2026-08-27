from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Tuple

import torch

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend, BaseAttnMetadata
    from freetoken.attention.linear import FLAMetadata
    from freetoken.kvcache import BaseCacheHandle, BaseKVCachePool
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.moe import BaseMoeBackend
    from freetoken.moe.offload_cache import OffloadMoeCache


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024
    # Stop strings (OpenAI `stop` / Anthropic `stop_sequences`). Generation finishes when one
    # appears in the decoded output; the matched substring (and anything after) is trimmed.
    stop_strs: list[str] = field(default_factory=list)

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle
    # Optional precomputed multimodal soft-token embeddings (GPU, [num_image_tokens,
    # hidden]) scattered at image-token positions during this request's prefill.
    mm_embeds: torch.Tensor | None = None
    # Optional Qwen multimodal RoPE coordinates for the original prompt,
    # shaped [3, prompt_tokens] on CPU. Generated tokens use
    # ``logical_position + mrope_position_delta`` on all three axes.
    mrope_position_ids: torch.Tensor | None = None
    mrope_position_delta: int = 0

    # --- hybrid-radix (GDN linear-state) per-request slots; None for non-hybrid models or
    # until allocated from LinearStatePool. Set by the scheduler (P2). ---
    linear_slot_idx: int | None = None              # live GDN state slot (sglang mamba_pool_idx)
    mamba_ping_pong: tuple[int, int] | None = None  # 2 donatable track slots under overlap
    mamba_next_track_idx: int = 0                   # which ping-pong slot is the next snapshot dst (0/1)
    mamba_last_track_seqlen: int | None = None      # chunk-aligned committed len of the last snapshot
    mamba_restore_src: int | None = None            # on a prefix hit: tree snapshot slot to COW into the live slot (first chunk only)
    swa_evicted_seqlen: int = 0                      # SWA radix: positions < this had their swa KV freed (slid out of window) during decode
    decode_batch_idx: int = 0                        # SWA radix: # of decode forwards done; the proactive free_swa skips the first (overlap guard)
    # Set once, at the first sampled tool-call opener token (scheduler detection): the state
    # length just after that token (its index + 1). A client-side rewrite of the echoed tool
    # call diverges strictly after this point, so it is the deepest reuse boundary that
    # survives such a rewrite. GDN: the state is frozen into a ping-pong slot when cached_len
    # reaches it (snapshot_toolcall_anchor) and donated at finish. SWA: caps the proactive
    # out-of-window eviction so the window ending here stays resumable.
    toolcall_anchor_len: int | None = None
    # Abort arrived while this request's forward was in flight (overlap scheduling). The abort
    # handler must not free resources under an in-flight forward; it sets this flag and
    # _process_last_data frees the request when the batch drains (after copy_done.synchronize).
    aborted: bool = False

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
        self._alloc_ids_buf()

    def _alloc_ids_buf(self) -> None:
        self._ids_buf = torch.empty(self.max_device_len, dtype=self.input_ids.dtype)
        self._ids_buf[: self.device_len] = self.input_ids
        self.input_ids = self._ids_buf[: self.device_len]

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        n = self.input_ids.numel()
        m = n + next_token.numel()
        assert m <= self.max_device_len
        self._ids_buf[n:m] = next_token
        self.input_ids = self._ids_buf[:m]

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )



@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    # Model-space rotary coordinates. None means ordinary 1-D ``positions``.
    # Qwen VL uses [3, tokens] temporal/height/width coordinates.
    rope_positions: torch.Tensor | None = field(default=None, init=False)
    out_loc: torch.Tensor | None = field(init=False)
    # Per-(padded-)request table_idx as a GPU int64 tensor, used by GatedDeltaNet
    # decode to gather/scatter recurrent+conv state without host-side loops (so the
    # decode step is CUDA-graph capturable). Set by the scheduler / graph buffer.
    linear_table_idx: torch.Tensor | None = field(default=None, init=False)
    # Per-forward GatedDeltaNet metadata (cu_seqlens / cache_indices / continuation
    # flags), built once and shared by all GDN layers. Lazily built by the GDN op if
    # the scheduler/graph didn't set it.
    fla_metadata: "FLAMetadata | None" = field(default=None, init=False)
    padded_reqs: List[Req] = field(init=False)
    # DSV4 paged-KV out-locations for this batch (None for non-DSV4 models). Set by the scheduler.
    # This decode batch's padded per-row page-table rows. Attention backends that must read
    # positions anywhere in a request's history snapshot those rows before a captured replay
    # (DSV4), since the next batch's allocate_paged mutates the live table.
    active_table_idx: "torch.Tensor | None" = None
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)
    # concatenated multimodal soft-token embeddings for a prefill batch (or None)
    mm_embeds: torch.Tensor | None = field(default=None, init=False)
    # Prefill log stats snapshotted at schedule time (before forward's complete_one()
    # advances cached_len), so the prefill log reports the tokens actually forwarded and
    # the prefix-cache hit -- matching SGLang's #new-token / #cached-token. Set by the
    # PrefillManager; 0 on decode batches.
    log_new_tokens: int = field(default=0, init=False)
    log_cached_tokens: int = field(default=0, init=False)
    # (uid, complete prompt length, prefix-cache hit) for requests entering their first
    # prepared prefill batch. The scheduler turns these into PromptAdmittedMsg only AFTER
    # _prepare_batch succeeds. Continuation chunks leave this empty, so accounting is
    # exactly-once.
    prompt_admissions: List[Tuple[int, int, int]] = field(default_factory=list, init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    moe_offload_cache: OffloadMoeCache | None = None
    kv_cache: BaseKVCachePool = field(init=False)
    # Per-request recurrent state for GatedDeltaNet layers; set by the engine for
    # hybrid linear-attention models, otherwise None.
    linear_state_pool: LinearStatePool | None = None
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
