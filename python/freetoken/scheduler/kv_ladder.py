"""Pure policy for growing KV at request boundaries by trading MoE cache slots.

The scheduler owns the idle/drain mechanics; this module only decides whether a request
reaches the current rung and, if so, computes a budget-respecting next geometry. Keeping the
arithmetic CUDA-free makes the policy cheap to test and aligned with the engine's preflight.
"""

from __future__ import annotations

from dataclasses import dataclass

from freetoken.utils import div_ceil


DEFAULT_KV_LADDER_STEP_TOKENS = 32_768


class KVLadderCapacityError(ValueError):
    """The requested rung cannot fit even after shrinking MoE to its safe floor."""


@dataclass(frozen=True)
class KVLadderPlan:
    required_tokens: int
    current_tokens: int
    target_tokens: int
    target_pages: int
    current_moe_slots: int
    target_moe_slots: int


@dataclass(frozen=True)
class KVLadderPolicy:
    step_tokens: int
    max_context_tokens: int
    page_size: int
    pool_budget_bytes: int
    kv_bytes_per_page: int
    moe_bytes_per_slot: int
    min_moe_slots: int

    def __post_init__(self) -> None:
        for name in (
            "step_tokens", "max_context_tokens", "page_size", "pool_budget_bytes",
            "kv_bytes_per_page", "moe_bytes_per_slot", "min_moe_slots",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def initial_tokens(self) -> int:
        """The intended first rung, capped by the checkpoint's real context window."""
        return min(self.max_context_tokens, 2 * self.step_tokens)

    def plan(
        self,
        *,
        current_pages: int,
        current_moe_slots: int,
        input_tokens: int,
        max_output_tokens: int,
    ) -> KVLadderPlan | None:
        """Return a growth plan when ``prompt + possible output`` reaches the current rung.

        A long request may skip several rungs in one rebuild; the target remains aligned to the
        same ``step_tokens`` ladder. At the model ceiling the ordinary admission path clips an
        over-large output budget, exactly as it does without the ladder.
        """
        current_tokens = current_pages * self.page_size
        required_tokens = input_tokens + max_output_tokens
        if input_tokens >= self.max_context_tokens:
            return None
        if required_tokens < current_tokens or current_tokens >= self.max_context_tokens:
            return None

        # Always move at least one step. The +1 preserves one token of breathing room when the
        # request's theoretical maximum lands exactly on a rung ("could hit the bound").
        required_rung = div_ceil(required_tokens + 1, self.step_tokens) * self.step_tokens
        target_tokens = min(
            self.max_context_tokens,
            max(current_tokens + self.step_tokens, required_rung),
        )
        target_pages = div_ceil(target_tokens, self.page_size)
        if target_pages <= current_pages:
            return None

        bytes_after_kv = self.pool_budget_bytes - target_pages * self.kv_bytes_per_page
        affordable_moe = bytes_after_kv // self.moe_bytes_per_slot
        target_moe_slots = min(current_moe_slots, affordable_moe)
        if target_moe_slots < self.min_moe_slots:
            max_kv_pages = (
                self.pool_budget_bytes - self.min_moe_slots * self.moe_bytes_per_slot
            ) // self.kv_bytes_per_page
            max_kv_tokens = max(0, max_kv_pages * self.page_size)
            raise KVLadderCapacityError(
                f"KV ladder rung {target_tokens} tokens cannot fit while retaining the "
                f"minimum {self.min_moe_slots} MoE slots (budget permits at most "
                f"{max_kv_tokens} KV tokens)"
            )

        return KVLadderPlan(
            required_tokens=required_tokens,
            current_tokens=current_tokens,
            target_tokens=min(target_pages * self.page_size, self.max_context_tokens),
            target_pages=target_pages,
            current_moe_slots=current_moe_slots,
            target_moe_slots=target_moe_slots,
        )
