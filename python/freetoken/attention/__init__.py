from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from freetoken.utils import Registry, init_logger

from .base import AttentionSpec, AttnType, BaseAttnBackend, BaseAttnMetadata, HybridBackend

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

logger = init_logger(__name__)


class BackendCreator(Protocol):
    def __call__(self, config: ModelConfig) -> BaseAttnBackend: ...


@dataclass(frozen=True)
class BackendInfo:
    """Declarative capability matrix entry for one backend. The engine's config-time
    validation interprets the requirement flags through its own (monkeypatch-able)
    availability probes; this stays pure data so registration never imports kernels."""

    supported_types: frozenset[AttnType]
    requires_flashinfer: bool = False
    requires_sgl_kernel: bool = False
    requires_sm100: bool = False
    # Allowed page sizes (None -> any). Config-time resolution coerces to the last
    # entry when the resolved page_size is not in the list.
    page_sizes: tuple[int, ...] | None = None
    # Whether forward() honors a per-call AttentionSpec (window/sm_scale/sinks).
    # Non-consumers raise on a non-None spec instead of silently dropping it.
    consumes_attn_spec: bool = False


SUPPORTED_ATTENTION_BACKENDS = Registry[BackendCreator]("Attention Backend")


@SUPPORTED_ATTENTION_BACKENDS.register(
    "trtllm",
    BackendInfo(
        supported_types=frozenset({AttnType.FULL}),
        requires_flashinfer=True,
        requires_sm100=True,
        page_sizes=(16, 32, 64),
    ),
)
def create_trtllm_backend(config: ModelConfig):
    from .trtllm import TensorRTLLMBackend

    return TensorRTLLMBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register(
    "fi",
    BackendInfo(
        supported_types=frozenset({AttnType.FULL}),
        requires_flashinfer=True,
    ),
)
def create_fi_backend(config: ModelConfig):
    from .fi import FlashInferBackend

    return FlashInferBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register(
    "fa",
    BackendInfo(
        supported_types=frozenset({AttnType.FULL}),
        requires_sgl_kernel=True,
    ),
)
def create_fa_backend(config: ModelConfig):
    from .fa import FlashAttentionBackend

    return FlashAttentionBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register(
    "triton",
    BackendInfo(
        supported_types=frozenset({AttnType.FULL, AttnType.SWA}),
        consumes_attn_spec=True,
    ),
)
def create_triton_backend(config: ModelConfig):
    from .triton import TritonAttentionBackend

    return TritonAttentionBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register(
    "dsv4_sparse",
    BackendInfo(supported_types=frozenset({AttnType.DSV4})),
)
def create_dsv4_sparse_backend(config: ModelConfig):
    from .dsv4_sparse import DSV4SparseAttnBackend

    return DSV4SparseAttnBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register(
    "dsa",
    BackendInfo(supported_types=frozenset({AttnType.MLA, AttnType.DSA})),
)
def create_dsa_backend(config: ModelConfig):
    # MLA with a grouped index (index_ratio > 1) is the kpool indexer layout.
    if any(s.mla and s.index_ratio > 1 for s in config.kv_cache_group_specs()):
        from .dsa_indexer_kpool import Glm5NextDSABackend

        return Glm5NextDSABackend(config)
    from .dsa import DSAAttnBackend

    return DSAAttnBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register(
    "m3_sparse",
    BackendInfo(
        supported_types=frozenset({AttnType.BSA}),
        # One KV page == one 128-token sparse block: the top-k block ids ARE page
        # indices and the block-base-row addressing needs page-aligned 128-row runs.
        # Config-time resolution coerces any other page size here.
        page_sizes=(128,),
    ),
)
def create_m3_sparse_backend(config: ModelConfig):
    from .m3_sparse import M3SparseAttnBackend

    return M3SparseAttnBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register(
    "qsa_sparse",
    BackendInfo(
        supported_types=frozenset({AttnType.QSA}),
        # 64-token pages: a 4-token compress group never straddles a page, so the
        # compressed row of a group is page_base // 4 + block-in-page.
        page_sizes=(64,),
    ),
)
def create_qsa_sparse_backend(config: ModelConfig):
    from .qsa_sparse import QSASparseAttnBackend

    return QSASparseAttnBackend(config)


def attention_backend_info(name: str) -> BackendInfo:
    return SUPPORTED_ATTENTION_BACKENDS.info(name)


def validate_attn_backend(backend: str, allow_auto: bool = True):
    if backend != "auto":
        parts = backend.split(",")
        if len(parts) > 2:
            from argparse import ArgumentTypeError

            raise ArgumentTypeError(
                f"At most two comma-separated attention backends are allowed "
                f"(prefill,decode), got {backend!r}"
            )
        SUPPORTED_ATTENTION_BACKENDS.assert_supported(parts)
    else:
        assert allow_auto, "auto is not allowed here"
    return backend


def create_attention_backend(
    backend: str,
    config: ModelConfig,
) -> BaseAttnBackend:
    validate_attn_backend(backend, allow_auto=False)
    if "," in backend:
        p_backend, d_backend = backend.split(",", 1)
        if p_backend != d_backend:
            logger.info(f"Using hybrid attention backend: prefill={p_backend}, decode={d_backend}")
            p_backend = create_attention_backend(p_backend, config)
            d_backend = create_attention_backend(d_backend, config)
            return HybridBackend(p_backend, d_backend)
        backend = p_backend  # both are the same, fall through to single backend
        logger.warning(f"P/D attention backends are the same: {backend}, using single backend.")

    return SUPPORTED_ATTENTION_BACKENDS[backend](config)


__all__ = [
    "AttnType",
    "BackendInfo",
    "BaseAttnMetadata",
    "BaseAttnBackend",
    "AttentionSpec",
    "attention_backend_info",
    "create_attention_backend",
    "SUPPORTED_ATTENTION_BACKENDS",
    "validate_attn_backend",
]
