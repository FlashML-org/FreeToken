from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    silu_and_mul,
)
from freetoken.utils import init_logger
from freetoken.utils.misc import opt_in_env

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

logger = init_logger(__name__)
_FOLD_SAID = False  # the router fold's one line a process
# The router fold runs only under FREETOKEN_ROUTER_FOLD=1 (or true, yes, on). Unset, the router's linear and its top-k
# stay two launches on every step, and with them the shared expert's own kernels, since the ninth route's gate rides the
# fold's launch: every step then routes as a prefill does and no line is logged. Any other value keeps them too and the
# first decode step says so. Read once, at import; a block decides at its first decode step.
ROUTER_FOLD_ENV = "FREETOKEN_ROUTER_FOLD"
_ROUTER_FOLD_SET, _ROUTER_FOLD = opt_in_env(os.environ.get(ROUTER_FOLD_ENV))

# A decode step of up to this many rows (one row, or a verify of up to five drafts) on resident NVFP4 experts runs the
# shared expert as one more route of the routed experts' kernels and its gate as one more row of the router's launch:
# the router fold's own cap (moe_router._FOLD_MAX_ROWS), since the gate rides the fold's launch.
_SHARED_ROUTE_MAX_ROWS = 6
# It runs only under FREETOKEN_SHARED_EXPERT_ROUTE=1 (or true, yes, on), and only with the router fold on. Unset, those
# steps keep the shared expert's own kernels, no route copy is made and no line is logged; any other value keeps them
# too and the load says so once. Each decode step reads _SHARED_ROUTE, so a captured graph runs the path that was set
# when it was captured.
SHARED_ROUTE_ENV = "FREETOKEN_SHARED_EXPERT_ROUTE"
_SHARED_ROUTE_SET, _SHARED_ROUTE = opt_in_env(os.environ.get(SHARED_ROUTE_ENV))
_SHARED_ROUTE_ROLES = ("gate_up", "gate_up_scale", "gate_up_global", "down", "down_scale", "down_global")
# the row counts a step of the served model took the ninth route at: one line each a process
_SHARED_ROUTE_ROWS_SAID: set[int] = set()


def shared_expert_bank_slot(gate_up, down) -> dict[str, torch.Tensor]:
    """The two NVFP4 projections of a shared expert on the triton linear kernel's K-major resident layout
    (``nvfp4_transpose_resident``: [K // 8, N] int32 words, [K // 16, N] block scales, [N] row globals) as one
    expert of the Triton MoE banks: [N, K // 2] uint8, [N, K // 16] e4m3 and [N] fp32 per projection, by the
    layout's roles. The transpose back is an exact permutation; the row globals are the projections' own."""
    route = {}
    for name, proj in (("gate_up", gate_up), ("down", down)):
        route[name] = proj.weight.t().contiguous().view(torch.uint8)
        route[f"{name}_scale"] = proj.weight_scale.t().contiguous()
        route[f"{name}_global"] = proj.weight_global.to(torch.float32).contiguous()
    return route


class _SharedExpert(BaseOP):
    """Always-present shared SwiGLU expert of width ``shared_expert_intermediate_size``."""

    def __init__(
        self, config: ModelConfig, hidden_size: int, intermediate_size: int, *, prefix: str = ""
    ):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = LinearRowParallel(
            intermediate_size, hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen3_5DenseMLP(_SharedExpert):
    """Dense (non-MoE) SwiGLU MLP for dense Qwen3.x checkpoints (e.g. 27B): ``gate_up_proj``
    (fused gate|up) + ``down_proj`` at full ``intermediate_size``. Same structure as the shared
    expert, so it reuses ``_SharedExpert`` directly and keeps the state-dict keys flat
    (``...layers.N.mlp.{gate_up_proj,down_proj}``)."""

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        super().__init__(config, config.hidden_size, config.intermediate_size, prefix=prefix)


class Qwen3_5MoE(BaseOP):
    """Routed MoE (256 experts, top-8) plus a gated shared expert:

        out = routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)

    Router softmaxes over all experts, takes top-k, and renormalizes (HF semantics).
    """

    # True once the engine's load has called prepare_decode_weights: a block of the served model, not of a draft
    _decode_prepared = False

    def __init__(self, config: ModelConfig, layer_id: int | None = None, *, prefix: str = ""):
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        # routers stay bf16 whatever the checkpoint quantizes
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size,
            prefix=f"{prefix}.shared_expert",
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)
        self._router_weight: torch.Tensor | None | bool = False  # False until the first decode step decides
        self._router_off: str | None = None
        # the shared expert in the experts' bank layout; False until the engine's load (prepare_decode_weights) or the
        # first decode step outside a capture decides
        self._shared_route: dict[str, torch.Tensor] | None | bool = False

    def _shared_projections_resident(self) -> bool:
        """Whether both of the shared expert's projections are resident under the triton NVFP4 linear kernel, on one
        rank: the K-major layout ``shared_expert_bank_slot`` reads."""
        from freetoken.layers.quantization.linear.nvfp4 import TritonNvfp4LinearKernel

        gate_up, down = self.shared_expert.gate_up_proj, self.shared_expert.down_proj
        return down._tp_size == 1 and all(isinstance(proj.quant_method.kernel, TritonNvfp4LinearKernel) for proj in (gate_up, down))

    def _router_weight_off(self) -> str | None:
        """Why the router's weight cannot go into its top-k kernel, or None when it can: a bf16 [E, H]
        torch linear without a bias on the GPU, and the fold asked for."""
        from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel

        if not _ROUTER_FOLD:
            return f"{ROUTER_FOLD_ENV}={_ROUTER_FOLD_SET}" if _ROUTER_FOLD_SET else f"{ROUTER_FOLD_ENV} unset"
        layer = self.gate
        w = layer.weight
        if layer.bias is not None:
            return "the router has a bias"
        if not isinstance(layer.quant_method.kernel, TorchLinearKernel):
            return f"the router runs {type(layer.quant_method.kernel).__name__}, not the torch linear"
        if not w.is_cuda:
            return f"the router's weight is on {w.device}"
        if w.dtype != torch.bfloat16:
            return f"the router's weight is {w.dtype}"
        if w.dim() != 2 or w.stride(1) != 1:
            return f"the router's weight is not a row-major matrix (shape {tuple(w.shape)}, strides {w.stride()})"
        return None

    def _router_weight_in_kernel(self) -> torch.Tensor | None:
        """The router's weight when its top-k kernel can take the logits too. Decided once, at the
        first decode step, when the weights are final."""
        if self._router_weight is False:
            self._router_off = self._router_weight_off()
            self._router_weight = self.gate.weight if self._router_off is None else None
        return self._router_weight

    def _fold_off(self, hidden_states: torch.Tensor, w: torch.Tensor | None) -> str | None:
        """Why no decode step folds the router, or None when decode-sized steps do; the step's own row
        count is left out, as CUDA-graph capture also runs steps past the fold's cap."""
        if w is None:
            return self._router_off
        if hidden_states.dtype != torch.bfloat16:
            return f"the hidden states are {hidden_states.dtype}"
        if hidden_states.stride(1) != 1:
            return "the hidden states' rows are not contiguous"
        from freetoken.kernel.triton.moe_router import router_gemv_topk_folds

        if not router_gemv_topk_folds(1, w.shape[0], w.shape[1], self.experts.top_k):
            return f"{w.shape[0]} experts of hidden size {w.shape[1]} and top-{self.experts.top_k} are not the fold's shapes"
        return None

    def _routes_in_kernel(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(topk weights, topk ids) of a decode step from one launch that also computes the router's
        logits, or None where the linear and the router stay two launches. The logits are summed in
        that kernel's order, not cuBLAS's, so a near-tie may route to the other expert."""
        global _FOLD_SAID
        if get_global_ctx().batch.is_prefill:
            return None
        w = self._router_weight_in_kernel()
        # silent when nobody asked for the fold
        if not _FOLD_SAID and (_ROUTER_FOLD or _ROUTER_FOLD_SET):
            _FOLD_SAID = True
            off = self._fold_off(hidden_states, w)
            if off is None:
                logger.info_rank0("router fold on decode: on (the router's linear and top-k in one launch for decode-sized steps)")
            else:
                logger.info_rank0(f"router fold on decode: off ({off})")
        if w is None or hidden_states.dtype != torch.bfloat16 or hidden_states.stride(1) != 1:
            return None
        from freetoken.kernel.triton.moe_router import router_gemv_topk_folds, router_gemv_topk_softmax

        experts = self.experts
        if not router_gemv_topk_folds(hidden_states.shape[0], w.shape[0], w.shape[1], experts.top_k):
            return None
        return router_gemv_topk_softmax(hidden_states, w, experts.top_k, experts.renormalize)

    def prepare_decode_weights(self) -> None:
        """The engine's call once the experts are loaded: the shared expert's route copy is made here, so the free
        memory the KV pool is sized from counts it. A block built without the engine makes it at its first decode.
        With the router fold off no step can take the route, so no copy is made and none of the KV pool goes to it;
        with the route's switch unset no copy is made either."""
        self._decode_prepared = True
        if not _ROUTER_FOLD:
            self._shared_route = None
            if _SHARED_ROUTE:
                fold = f"{ROUTER_FOLD_ENV}={_ROUTER_FOLD_SET}" if _ROUTER_FOLD_SET else f"{ROUTER_FOLD_ENV} unset"
                logger.info_once(f"shared expert as a ninth route on decode: off ({fold})")
        elif _SHARED_ROUTE:
            self._shared_route_banks()
        elif _SHARED_ROUTE_SET and self._shared_route is False:
            logger.info_once(f"shared expert as a ninth route on decode: off ({SHARED_ROUTE_ENV}={_SHARED_ROUTE_SET})")

    def _shared_route_banks(self) -> dict[str, torch.Tensor] | None:
        """The shared expert as one expert of the routed experts' banks: each projection in the bank's role, shape,
        dtype and strides of one slot, or None where it cannot be a route. Built once, from the K-major resident
        weights by the exact permutation back (prefill and wider steps keep reading those)."""
        if self._shared_route is False:
            self._shared_route = self._build_shared_route()
            if self._shared_route is not None:
                logger.info_once("shared expert as a ninth route on decode: on")
        return self._shared_route

    def _build_shared_route(self) -> dict[str, torch.Tensor] | None:
        from freetoken.layers.moe import OffloadMoELayer
        from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel

        experts, gate = self.experts, self.shared_expert_gate
        kernel = getattr(experts.quant_method, "kernel", None)
        if not self._shared_projections_resident() or isinstance(experts, OffloadMoELayer) or not getattr(kernel, "takes_shared_route", False):
            return None
        if experts.tp_size > 1 or experts.activation != "silu" or experts.apply_router_weight_on_input:
            return None
        g = gate.weight
        if gate.bias is not None or not isinstance(gate.quant_method.kernel, TorchLinearKernel):
            return None
        if not (g.dtype == torch.bfloat16 and g.dim() == 2 and g.shape[0] == 1 and g.stride(1) == 1):
            return None
        banks = getattr(experts, "_expert_banks", None)
        if banks is None or any(role not in banks for role in _SHARED_ROUTE_ROLES):
            return None
        route = shared_expert_bank_slot(self.shared_expert.gate_up_proj, self.shared_expert.down_proj)
        for role, t in route.items():
            bank = banks[role]
            if t.shape != bank.shape[1:] or t.dtype != bank.dtype or t.stride() != bank.stride()[1:] or t.device != bank.device:
                logger.info_once(f"shared expert as a ninth route on decode: off (its {role} is not one slot of the experts' bank)")
                return None
        return route

    def _shared_route_for(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor] | None:
        """The shared expert as the last route of this step, or None for today's kernels: a decode step of 1 to
        _SHARED_ROUTE_MAX_ROWS rows whose router logits the top-k launch computes, on resident NVFP4 experts under
        a kernel that takes a route outside the bank."""
        if not _SHARED_ROUTE or self._shared_route is None:
            return None
        rows = hidden_states.shape[0]
        if not 1 <= rows <= _SHARED_ROUTE_MAX_ROWS or hidden_states.dtype != torch.bfloat16 or hidden_states.stride(1) != 1:
            return None
        if get_global_ctx().batch.is_prefill:
            return None
        w = self._router_weight_in_kernel()
        if w is None:
            return None
        from freetoken.kernel.triton.moe_router import router_gemv_topk_folds

        if not router_gemv_topk_folds(rows, w.shape[0], w.shape[1], self.experts.top_k):
            return None
        # built from the loaded weights outside a capture; the capture runs each forward eagerly first
        if self._shared_route is False and hidden_states.is_cuda and torch.cuda.is_current_stream_capturing():
            return None
        return self._shared_route_banks()

    def _forward_shared_route(self, hidden_states: torch.Tensor, shared: dict[str, torch.Tensor]) -> torch.Tensor:
        """A decode step with the shared expert as route top_k + 1 of the routed experts' kernels: weight
        sigmoid(gate), the gate's logit one more row of the router's launch, and the nine routes summed once."""
        from freetoken.kernel.triton.moe_router import router_gemv_topk_softmax

        rows = hidden_states.shape[0]
        if self._decode_prepared and rows not in _SHARED_ROUTE_ROWS_SAID:
            # a capture runs this too, so the line also shows the path a captured step replays
            _SHARED_ROUTE_ROWS_SAID.add(rows)
            logger.info_rank0(f"shared expert as a ninth route on decode: taken at {rows} row{'' if rows == 1 else 's'}")
        experts = self.experts
        weights, ids = router_gemv_topk_softmax(
            hidden_states, self._router_weight_in_kernel(), experts.top_k, experts.renormalize,
            shared_gate=self.shared_expert_gate.weight,
        )
        return experts.routed_forward(hidden_states, weights, ids, shared=shared)

    def _routed(self, hidden_states, router_logits, routes) -> torch.Tensor:
        if routes is None:
            return self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return self.experts.routed_forward(hidden_states, *routes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_route = self._shared_route_for(hidden_states)
        if shared_route is not None:
            return self._forward_shared_route(hidden_states, shared_route).view(num_tokens, hidden_dim)
        # Compute the router + shared expert BEFORE the routed experts: the fused MoE
        # kernel may write into ``hidden_states`` in place, which would corrupt the
        # shared expert's input (HF also evaluates the shared expert first).
        routes = self._routes_in_kernel(hidden_states)
        router_logits = self.gate.forward(hidden_states) if routes is None else None
        shared = self.shared_expert.forward(hidden_states)
        shared = shared * torch.sigmoid(self.shared_expert_gate.forward(hidden_states))
        routed = self._routed(hidden_states, router_logits, routes)
        return (routed + shared).view(num_tokens, hidden_dim)


__all__ = ["Qwen3_5MoE", "Qwen3_5DenseMLP", "shared_expert_bank_slot"]
