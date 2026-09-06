"""Vendored flash-linear-attention (fla) GatedDeltaNet triton kernels.

Borrowed from sglang's fla fork (`sglang/srt/layers/attention/fla/`), which itself
adapts https://github.com/fla-org/flash-linear-attention. We vendor sglang's fork — not
upstream fla — because the fork carries features FreeToken needs that upstream lacks:
the indexed state pool (`initial_state_indices` threaded into the chunk recurrence) and
the fused sigmoid-gating decode kernel, plus an H100-safe single-config autotune that
won't corrupt the in-place state pool.

Kernel code is intentionally "dirty" (pure triton, sglang lineage) and lives here under
``kernel/fla/`` rather than under ``models/`` so the model code stays clean.

Public entry points:
- ``chunk_gated_delta_rule`` — chunked prefill; reads/writes the recurrent state pool
  in place by ``initial_state_indices``, with optional in-kernel q/k l2norm.
- ``fused_sigmoid_gating_delta_rule_update`` — single-token decode; gating + in-kernel
  l2norm + delta-rule update + per-slot state read/write, all in one kernel.

Provenance: https://github.com/sgl-project/sglang, ``python/sglang/srt/layers/attention/fla/``
(NVIDIA path only; the ``is_intel`` XPU branch and the ``torch_release`` sglang import were
stripped/inlined on vendoring). Keep ``chunk_delta_h.py``'s single fixed ``triton.Config`` — restoring
upstream's multi-config autotune corrupts the in-place state pool. Tune via the env knobs
``SGLANG_GDN_CHUNK_H_BV`` / ``SGLANG_GDN_CHUNK_H_NUM_WARPS`` / ``SGLANG_GDN_CHUNK_H_NUM_STAGES``.

KDA (GLM-5.3-Flash Kimi Delta Attention) kernels are vendored separately from vLLM's
``third_party/flash_linear_attention`` (same fla lineage): ``kda.py`` (chunked prefill +
fused gate cumsum + recurrent decode wrapper), ``kda_chunk_delta_h.py`` (exp2-gate chunk
recurrence), ``fused_recurrent.py`` (pool-indexed decode kernel with in-kernel KDA gate),
``solve_tril.py``. Public entry points:
- ``chunk_kda_with_fused_gate`` -- chunked prefill from raw gate logits; gathered initial
  state in, final state out (scatter back to the pool is the caller's job).
  WARNING: clobbers the ``v`` argument (the output is written into that buffer to save
  an allocation, as in vLLM where v is an ephemeral projection). Never pass a tensor
  that is read again afterwards.
- ``fused_recurrent_kda`` -- decode; per-slot state read/write via ``ssm_state_indices``,
  gate + beta-sigmoid + q/k l2norm computed in-kernel. The per-token state store reads
  ``ssm_state_indices`` as a CONTIGUOUS [N, T] block; materialize, never ``expand()``.
"""
from freetoken.kernel.fla.chunk import chunk_gated_delta_rule
from freetoken.kernel.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from freetoken.kernel.fla.kda import (
    chunk_kda_with_fused_gate,
    fused_kda_gate,
    fused_recurrent_kda,
)
from freetoken.kernel.fla.layernorm_gated import rms_norm_gated

__all__ = [
    "chunk_gated_delta_rule",
    "fused_sigmoid_gating_delta_rule_update",
    "chunk_kda_with_fused_gate",
    "fused_kda_gate",
    "fused_recurrent_kda",
    "rms_norm_gated",
]
