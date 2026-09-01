"""Algorithm-agnostic helpers shared by all speculative decoding workers."""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from freetoken.core import Batch, Req
    from freetoken.engine.sample import BatchSamplingArgs


# ---------------------------------------------------------------------------
# Output selection (greedy exact-match verify)
# ---------------------------------------------------------------------------

def select_output_tokens(
    base_token: torch.Tensor,
    draft_candidates: torch.Tensor,
    verify_tokens: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Greedy verify: accept while draft == target argmax, bonus = first mismatch."""
    if verify_tokens.numel() < draft_candidates.numel() + 1:
        raise ValueError("verify_tokens must contain one prediction per draft plus one bonus")
    accepted = contiguous_accept_len(draft_candidates, verify_tokens)
    bonus = verify_tokens[accepted : accepted + 1]
    return torch.cat([
        base_token[:1],
        draft_candidates[:accepted].to(base_token.dtype),
        bonus.to(base_token.dtype),
    ]), accepted


def contiguous_accept_len(
    draft_candidates: torch.Tensor,
    verify_tokens: torch.Tensor,
) -> int:
    matches = draft_candidates == verify_tokens[: draft_candidates.numel()].to(draft_candidates.dtype)
    if matches.numel() == 0:
        return 0
    mismatches = torch.nonzero(~matches, as_tuple=False)
    if mismatches.numel() == 0:
        return matches.numel()
    return int(mismatches[0, 0].item())


def select_streaming_output_tokens(
    base_token: torch.Tensor,
    draft_candidates: torch.Tensor,
    verify_tokens: torch.Tensor,
) -> tuple[torch.Tensor, int, bool]:
    """Streaming greedy verify with early-stop. Returns (output, accepted, done)."""
    accepted = 0
    checked = min(draft_candidates.numel(), verify_tokens.numel())
    for i in range(checked):
        if int(draft_candidates[i].item()) != int(verify_tokens[i].item()):
            bonus = verify_tokens[i : i + 1]
            return torch.cat([
                base_token[:1],
                draft_candidates[:accepted].to(base_token.dtype),
                bonus.to(base_token.dtype),
            ]), accepted, True
        accepted += 1
    if verify_tokens.numel() > draft_candidates.numel():
        bonus = verify_tokens[accepted : accepted + 1]
        return torch.cat([
            base_token[:1],
            draft_candidates[:accepted].to(base_token.dtype),
            bonus.to(base_token.dtype),
        ]), accepted, True
    return torch.empty((0,), dtype=base_token.dtype, device=base_token.device), accepted, False


# ---------------------------------------------------------------------------
# Speculative (rejection) sampling for non-greedy decoding
# ---------------------------------------------------------------------------

def sampling_probs(logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
    """Temperature + top_k/top_p filtered probability distribution."""
    temperature = float(args.temperatures[0].item())
    top_k = int(args.top_k[0].item()) if args.top_k is not None else 0
    top_p = float(args.top_p[0].item()) if args.top_p is not None else 1.0

    scores = logits.float() / max(temperature, 1e-6)
    vocab_size = scores.shape[-1]
    indices = None
    if 0 < top_k < vocab_size:
        scores, indices = torch.topk(scores, top_k, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    if indices is not None:
        probs = torch.zeros_like(logits, dtype=probs.dtype).scatter(-1, indices, probs)
    return probs


def rejection_residual_sample(p_row: torch.Tensor, q_row: torch.Tensor) -> torch.Tensor:
    residual = (p_row - q_row).clamp_min(0)
    total = residual.sum()
    residual = torch.where(
        total > 0,
        residual / total.clamp_min(torch.finfo(residual.dtype).tiny),
        p_row,
    )
    return torch.multinomial(residual[None], 1)[0]


def rejection_step(
    p_row: torch.Tensor,
    q_row: torch.Tensor,
    draft_token: int,
    uniform: float,
) -> tuple[bool, torch.Tensor | None]:
    if (uniform * q_row[draft_token] < p_row[draft_token]).item():
        return True, None
    return False, rejection_residual_sample(p_row, q_row)


def rejection_sample_chain(
    base_token: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    uniform: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    gamma = draft_tokens.numel()
    tokens64 = draft_tokens.to(torch.int64)
    p = target_probs[:gamma].gather(-1, tokens64[:, None])[:, 0]
    q = draft_probs.gather(-1, tokens64[:, None])[:, 0]
    u = (
        torch.rand(gamma, dtype=torch.float32, device=p.device)
        if uniform is None
        else uniform.to(device=p.device, dtype=torch.float32)
    )
    accepted = int((u * q < p).to(torch.int32).cumprod(0).sum().item())
    if accepted == gamma:
        bonus = torch.multinomial(target_probs[gamma : gamma + 1], 1)[0]
    else:
        bonus = rejection_residual_sample(target_probs[accepted], draft_probs[accepted])
    return torch.cat([
        base_token[:1],
        draft_tokens[:accepted].to(base_token.dtype),
        bonus.to(base_token.dtype),
    ]), accepted


def use_sampling_verify(args: BatchSamplingArgs, worker) -> bool:
    return args.temperatures is not None and getattr(worker, "last_draft_probs", None) is not None


def repeat_sampling_args(args: BatchSamplingArgs, repeat: int) -> BatchSamplingArgs:
    if args.temperatures is None:
        return args
    from freetoken.engine.sample import BatchSamplingArgs
    return BatchSamplingArgs(
        temperatures=args.temperatures.repeat(repeat),
        top_k=args.top_k.repeat(repeat) if isinstance(args.top_k, torch.Tensor) else args.top_k,
        top_p=args.top_p.repeat(repeat) if isinstance(args.top_p, torch.Tensor) else args.top_p,
    )


# ---------------------------------------------------------------------------
# Linear (GDN) state snapshot / restore
# ---------------------------------------------------------------------------

def snapshot_linear_state_slot(pool: Any, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
    return pool.conv_states[:, slot].clone(), pool.recurrent_states[:, slot].clone()


def restore_linear_state_slot(
    pool: Any, slot: int, snapshot: tuple[torch.Tensor, torch.Tensor]
) -> None:
    conv_state, recurrent_state = snapshot
    pool.conv_states[:, slot].copy_(conv_state)
    pool.recurrent_states[:, slot].copy_(recurrent_state)


def restore_linear_state_for_commit(
    pool: Any,
    slot: int,
    pre_verify_snapshot: tuple[torch.Tensor, torch.Tensor],
    verify_snapshots: list[tuple[torch.Tensor, torch.Tensor]],
    commit_len: int,
) -> None:
    if commit_len <= 0:
        snapshot = pre_verify_snapshot
    elif commit_len <= len(verify_snapshots):
        snapshot = verify_snapshots[commit_len - 1]
    else:
        raise ValueError(f"commit_len={commit_len} exceeds verify snapshots={len(verify_snapshots)}")
    restore_linear_state_slot(pool, slot, snapshot)


def restore_graph_linear_state_for_commit(
    pool: Any,
    slot: int,
    pre_verify_snapshot: tuple[torch.Tensor, torch.Tensor],
    graph_snapshots: tuple[torch.Tensor, torch.Tensor],
    commit_len: int,
) -> None:
    if commit_len <= 0:
        snapshot = pre_verify_snapshot
    elif commit_len <= graph_snapshots[0].shape[0]:
        snapshot = (graph_snapshots[0][commit_len - 1], graph_snapshots[1][commit_len - 1])
    else:
        raise ValueError(f"commit_len={commit_len} exceeds graph snapshots={graph_snapshots[0].shape[0]}")
    restore_linear_state_slot(pool, slot, snapshot)


# ---------------------------------------------------------------------------
# Verify state snapshot / restore (batch + req fields)
# ---------------------------------------------------------------------------

class VerifyState(NamedTuple):
    batch_phase: str
    batch_input_ids: torch.Tensor
    batch_positions: torch.Tensor
    batch_out_loc: torch.Tensor | None
    batch_padded_reqs: list
    batch_fla_metadata: Any
    req_input_len: int
    req_cached_len: int
    req_device_len: int


def snapshot_verify_state(batch: Batch, req: Req) -> VerifyState:
    return VerifyState(
        batch_phase=batch.phase,
        batch_input_ids=batch.input_ids,
        batch_positions=batch.positions,
        batch_out_loc=batch.out_loc,
        batch_padded_reqs=batch.padded_reqs,
        batch_fla_metadata=batch.fla_metadata,
        req_input_len=req.input_ids.numel(),
        req_cached_len=req.cached_len,
        req_device_len=req.device_len,
    )


def restore_verify_state(batch: Batch, req: Req, state: VerifyState) -> None:
    req.input_ids = req._ids_buf[: state.req_input_len]
    req.cached_len = state.req_cached_len
    req.device_len = state.req_device_len
    batch.phase = state.batch_phase
    batch.input_ids = state.batch_input_ids
    batch.positions = state.batch_positions
    batch.out_loc = state.batch_out_loc
    batch.padded_reqs = state.batch_padded_reqs
    batch.fla_metadata = state.batch_fla_metadata


def clone_hidden_outputs(hidden_states: list[torch.Tensor]) -> list[torch.Tensor]:
    return [hidden.clone() for hidden in hidden_states]


# ---------------------------------------------------------------------------
# Adaptive gate (shared by all spec algorithms)
# ---------------------------------------------------------------------------

class AdaptiveGate:
    """Per-request measured fallback: disables spec decode when it measures
    slower than the plain target-forward baseline proxy by more than ``margin``."""

    def __init__(
        self,
        *,
        min_cycles: int = 12,
        eval_interval: int = 8,
        margin: float = 1.15,
        warmup_cycles: int = 4,
        window: int = 32,
        auto_disable_after: int = 3,
        reprobe_every: int = 8,
    ):
        self.min_cycles = min_cycles
        self.eval_interval = eval_interval
        self.margin = margin
        self.warmup_cycles = warmup_cycles
        self.auto_disable_after = auto_disable_after
        self.reprobe_every = reprobe_every
        self._window: deque[tuple[float, float, int]] = deque(maxlen=window)
        self._pending_events: list = []
        self._uid: int | None = None
        self.enabled = True
        self._records = 0
        self._evaluated = False
        self._consecutive_disables = 0
        self._off_requests = 0
        self._req_cycle_ms = 0.0
        self._req_tokens = 0
        self._req_target_ms: list[float] = []

    def should_run(self, uid: int) -> bool:
        if self._uid is None:
            self._uid = uid
        elif uid != self._uid:
            self._finish_request()
            self.reset(uid)
        return self.enabled

    def _finish_request(self) -> None:
        self._drain_pending()
        if len(self._req_target_ms) < self.min_cycles:
            return
        overall = self._req_cycle_ms / max(self._req_tokens, 1)
        baseline = statistics.median_low(sorted(self._req_target_ms))
        if overall > baseline * self.margin:
            self._consecutive_disables += 1
        else:
            self._consecutive_disables = 0

    def reset(self, uid: int | None = None) -> None:
        self._uid = uid
        self._window.clear()
        self._pending_events.clear()
        self._records = 0
        self._evaluated = False
        self._req_cycle_ms = 0.0
        self._req_tokens = 0
        self._req_target_ms = []
        self.enabled = True
        if self._consecutive_disables >= self.auto_disable_after:
            self._off_requests += 1
            if self._off_requests % self.reprobe_every != 0:
                self.enabled = False

    def _record_ms(self, cycle_ms: float, target_ms: float, out_tokens: int) -> None:
        self._records += 1
        if self._records <= self.warmup_cycles:
            return
        self._req_cycle_ms += cycle_ms
        self._req_tokens += out_tokens
        self._req_target_ms.append(target_ms)
        self._window.append((cycle_ms, target_ms, out_tokens))

    def record(self, *, cycle_ms: float, target_ms: float, out_tokens: int) -> None:
        if not self.enabled or out_tokens <= 0:
            return
        self._record_ms(cycle_ms, target_ms, out_tokens)
        n = self._records - self.warmup_cycles
        if n >= self.min_cycles and n % self.eval_interval == 0:
            self._evaluate()

    def record_events(self, *, cycle, target, out_tokens: int) -> None:
        if not self.enabled or out_tokens <= 0:
            return
        self._pending_events.append((cycle, target, out_tokens))
        n = self._records + len(self._pending_events) - self.warmup_cycles
        if n < self.min_cycles or n % self.eval_interval:
            return
        self._drain_pending()
        self._evaluate()

    def _drain_pending(self) -> None:
        if not self._pending_events:
            return
        self._pending_events[-1][0][1].synchronize()
        pending, self._pending_events = self._pending_events, []
        for (cycle_start, cycle_end), (target_start, target_end), tokens in pending:
            self._record_ms(
                cycle_start.elapsed_time(cycle_end),
                target_start.elapsed_time(target_end),
                tokens,
            )

    def _evaluate(self) -> None:
        if not self._window:
            return
        self._evaluated = True
        per_token = sorted(c / t for c, _, t in self._window if t > 0)
        baseline = sorted(t for _, t, _ in self._window)
        if not per_token or not baseline:
            return
        cycle_ms_per_token = statistics.median_low(per_token)
        baseline_ms = statistics.median_low(baseline)
        if cycle_ms_per_token > baseline_ms * self.margin:
            self.enabled = False
            from freetoken.utils import init_logger
            logger = init_logger(__name__)
            logger.warning_rank0(
                f"[SPEC_ADAPTIVE] disabling spec decode for this request: "
                f"cycle {cycle_ms_per_token:.3f} ms/token > "
                f"baseline proxy {baseline_ms * self.margin:.3f} ms/token "
                f"(target forward {baseline_ms:.3f} ms, margin {self.margin})"
            )

    @classmethod
    def from_env(cls) -> AdaptiveGate:
        return cls(
            margin=float(os.environ.get("FREETOKEN_SPEC_ADAPTIVE_MARGIN", "1.15")),
            min_cycles=int(os.environ.get("FREETOKEN_SPEC_ADAPTIVE_MIN_CYCLES", "12")),
            eval_interval=int(os.environ.get("FREETOKEN_SPEC_ADAPTIVE_EVAL_INTERVAL", "8")),
            reprobe_every=int(os.environ.get("FREETOKEN_SPEC_ADAPTIVE_REPROBE_EVERY", "8")),
        )
