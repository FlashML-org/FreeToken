"""Forward-probe state for the Qwen3 CPU debug build.

Activated only by FREETOKEN_PROBE_LAYERS=1; inert in normal serving (no
prints, no files written unless the env flag is set).

Why a custom probe instead of ad-hoc prints:
The engine runs chunked prefill -- a long prompt is split into several
forward passes, each covering a contiguous slice of absolute positions.
HF transformers processes the whole prompt in ONE pass. A naive "dump x
after the layer" only sees the LAST forward call's slice, so comparing it
against HF's full sequence is misaligned (the earlier 7-vs-15 token mess).

This module accumulates activations by ABSOLUTE position across every
forward call of a request, so chunked prefill is reconstructed into one
contiguous, correctly-positioned sequence. Both engine and HF then agree
on "position p", and the comparator can align them 1:1.

It also records, per forward call, the RoPE positions actually handed to
layer 0 (the prime suspect for the L00 divergence seen vs HF) and the
lm_head logits at the final prefill position.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROBE_LAYERS = bool(os.environ.get("FREETOKEN_PROBE_LAYERS"))

# accum[layer_id][abs_pos] -> np.ndarray[hidden]
ACCUM: Dict[int, Dict[int, np.ndarray]] = {}
# embedding[abs_pos] -> np.ndarray[hidden]  (raw, pre-norm, after embed_tokens)
EMBED: Dict[int, np.ndarray] = {}
# per-forward-call metadata: list of {"phase": str, "positions": [int,...]}
CALLS: List[Dict[str, Any]] = []
# lm_head logits at the highest prefill position seen
LMHEAD: Optional[np.ndarray] = None
LMHEAD_POS: Optional[int] = None
# raw input token ids for this request (debug cross-check vs HF tokenizer)
INPUT_IDS: Optional[List[int]] = None
# post-RoPE q/k per absolute position (layer 0 only; isolates the RoPE kernel)
ROPE_Q: Dict[int, np.ndarray] = {}
ROPE_K: Dict[int, np.ndarray] = {}
# pre-RoPE q/k per absolute position (layer 0 only; for self-consistency)
PRE_ROPE_Q: Dict[int, np.ndarray] = {}
PRE_ROPE_K: Dict[int, np.ndarray] = {}
# the ACTUAL positions tensor passed to rotary.forward at layer 0 (the prime suspect)
ATTN_POSITIONS: Optional[np.ndarray] = None
ATTN_CALL_PHASE: Optional[str] = None


def record_attn_positions(positions_tensor: torch.Tensor, phase: str) -> None:
    """Capture the EXACT positions tensor handed to rotary.forward on layer 0,
    prefill only. If this is not [0,1,2,3,4], that's the RoPE bug root cause."""
    global ATTN_POSITIONS, ATTN_CALL_PHASE
    if phase != "prefill":
        return
    ATTN_POSITIONS = positions_tensor.detach().to(torch.int64).cpu().numpy().reshape(-1)
    ATTN_CALL_PHASE = phase


def record_rope(positions_tensor: torch.Tensor, q: np.ndarray, k: np.ndarray) -> None:
    if CURRENT_PHASE != "prefill":
        return
    pos = positions_tensor.detach().to(torch.int64).cpu().numpy().tolist()
    # q/k: [num_heads, head_size]; flatten per-position for comparison
    for i, p in enumerate(pos):
        ROPE_Q[p] = q[i].reshape(-1)
        ROPE_K[p] = k[i].reshape(-1)


def record_pre_rope(positions_tensor: torch.Tensor, q: np.ndarray, k: np.ndarray) -> None:
    if CURRENT_PHASE != "prefill":
        return
    pos = positions_tensor.detach().to(torch.int64).cpu().numpy().tolist()
    for i, p in enumerate(pos):
        PRE_ROPE_Q[p] = q[i].reshape(-1)
        PRE_ROPE_K[p] = k[i].reshape(-1)


def record_input_ids(ids: torch.Tensor) -> None:
    global INPUT_IDS
    try:
        INPUT_IDS = ids.detach().to(torch.int64).cpu().numpy().tolist()
    except Exception:
        INPUT_IDS = None

# Set by Qwen3Model.forward each call so layer.forward can align its rows.
CURRENT_POSITIONS: Optional[torch.Tensor] = None
CURRENT_PHASE: str = "unknown"


def reset_if_new_request(positions_tensor: torch.Tensor) -> None:
    """Clear stale accumulators when a fresh prefill starts (positions begin at 0)."""
    try:
        mn = float(positions_tensor.min().item())
    except Exception:
        mn = -1.0
    if mn == 0.0:
        ACCUM.clear()
        EMBED.clear()
        CALLS.clear()
        global LMHEAD, LMHEAD_POS
        LMHEAD = None
        LMHEAD_POS = None
        global INPUT_IDS
        INPUT_IDS = None
        global ROPE_Q, ROPE_K
        ROPE_Q = {}
        ROPE_K = {}
        global PRE_ROPE_Q, PRE_ROPE_K
        PRE_ROPE_Q = {}
        PRE_ROPE_K = {}
        global ATTN_POSITIONS, ATTN_CALL_PHASE
        ATTN_POSITIONS = None
        ATTN_CALL_PHASE = None


def record_embedding(positions_tensor: torch.Tensor, x: torch.Tensor) -> None:
    if CURRENT_PHASE != "prefill":
        return  # decode batch is padded with position-0 garbage; skip it
    pos = positions_tensor.detach().to(torch.int64).cpu().numpy().tolist()
    xs = x.detach().float().cpu().numpy()
    for i, p in enumerate(pos):
        EMBED[p] = xs[i]


def record_layer(layer_id: int, positions_tensor: torch.Tensor, x_after_inln: torch.Tensor) -> None:
    if CURRENT_PHASE != "prefill":
        return  # decode batch is padded with position-0 garbage; skip it
    pos = positions_tensor.detach().to(torch.int64).cpu().numpy().tolist()
    xs = x_after_inln.detach().float().cpu().numpy()
    d = ACCUM.setdefault(layer_id, {})
    for i, p in enumerate(pos):
        d[p] = xs[i]


def record_forward_meta(phase: str, positions_tensor: torch.Tensor) -> None:
    pos = positions_tensor.detach().to(torch.int64).cpu().numpy().tolist()
    CALLS.append({"phase": phase, "positions": pos})


def record_lmhead(positions_tensor: torch.Tensor, logits: torch.Tensor) -> None:
    global LMHEAD, LMHEAD_POS
    pos = positions_tensor.detach().to(torch.int64).cpu().numpy()
    k = int(pos.max().item())
    row = int(np.argmax((pos == k).astype(np.int32)))
    LMHEAD = logits.detach().float().cpu().numpy()[row].copy()
    LMHEAD_POS = k


def finalize(path: str = "/tmp/ft_probe.npz") -> None:
    out: Dict[str, Any] = {}
    for L in sorted(ACCUM.keys()):
        d = ACCUM[L]
        ps = sorted(d.keys())
        out[f"layer{L}_inln"] = np.stack([d[p] for p in ps], axis=0)
        out[f"layer{L}_pos"] = np.array(ps, dtype=np.int64)
    if EMBED:
        ps = sorted(EMBED.keys())
        out["embed"] = np.stack([EMBED[p] for p in ps], axis=0)
        out["embed_pos"] = np.array(ps, dtype=np.int64)
    # calls metadata as numeric arrays (avoid object-dtype pickle issues)
    out["_calls_phase"] = np.array(
        [0 if c["phase"] == "prefill" else 1 if c["phase"] == "decode" else 2 for c in CALLS],
        dtype=np.int8,
    )
    maxlen = max((len(c["positions"]) for c in CALLS), default=0)
    calls_pos = np.zeros((len(CALLS), maxlen), dtype=np.int64)
    for i, c in enumerate(CALLS):
        plist = c["positions"]
        calls_pos[i, : len(plist)] = plist
    out["_calls_pos"] = calls_pos
    if LMHEAD is not None:
        out["lmhead"] = LMHEAD
        out["lmhead_pos"] = np.array([LMHEAD_POS], dtype=np.int64)
    if INPUT_IDS is not None:
        out["input_ids"] = np.array(INPUT_IDS, dtype=np.int64)
    if ROPE_Q:
        ps = sorted(ROPE_Q.keys())
        out["rope_q"] = np.stack([ROPE_Q[p] for p in ps], axis=0)
        out["rope_k"] = np.stack([ROPE_K[p] for p in ps], axis=0)
        out["rope_pos"] = np.array(ps, dtype=np.int64)
    if PRE_ROPE_Q:
        ps = sorted(PRE_ROPE_Q.keys())
        out["pre_rope_q"] = np.stack([PRE_ROPE_Q[p] for p in ps], axis=0)
        out["pre_rope_k"] = np.stack([PRE_ROPE_K[p] for p in ps], axis=0)
    if ATTN_POSITIONS is not None:
        out["attn_positions"] = ATTN_POSITIONS.astype(np.int64)
    np.savez(path, **out)


# Convenience for the comparator: return a dict of per-layer stacked arrays.
def load(path: str = "/tmp/ft_probe.npz") -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}

