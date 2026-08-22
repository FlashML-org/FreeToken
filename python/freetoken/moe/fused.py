import functools
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from freetoken.moe import BaseMoeBackend
from freetoken.utils import div_ceil, init_logger

logger = init_logger(__name__)

_warned_torch_topk = False


def _torch_fused_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch softmax router matching triton_kernels.topk (Windows fallback).

    Softmax over all experts, select the top-k, and (when ``renormalize``) rescale the
    selected weights to sum to 1 -- the standard fused-MoE routing convention.
    """
    probs = torch.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = topk_ids.to(torch.int32)
    if num_token_non_padded is not None:
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
        topk_ids[indices >= num_token_non_padded, :] = -1
    return topk_weights.contiguous(), topk_ids.contiguous()


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    from freetoken.kernel.backend import is_triton_kernels_installed

    # triton_kernels ships no Windows wheel, and unlike flashinfer/sgl_kernel it is not one
    # of the six ops the in-repo triton kernels cover -- so this router needs its own fallback.
    if not is_triton_kernels_installed():
        global _warned_torch_topk
        if not _warned_torch_topk:
            _warned_torch_topk = True
            # Once, not per call: this runs every MoE forward. On Linux a missing
            # triton_kernels used to fail fast with ImportError; keep the misconfiguration
            # visible without giving up the fallback that Windows needs.
            logger.warning_rank0(
                "fused_topk: triton_kernels is not installed -> pure-torch router fallback "
                "(numerically equivalent, slower). Expected on Windows (no wheel); on Linux "
                "install triton_kernels to restore the fused router."
            )
        return _torch_fused_topk(gating_output, topk, renormalize, num_token_non_padded)

    from triton_kernels.topk import topk as triton_kernels_topk

    logits = gating_output.float()
    softmax_first = not renormalize
    if softmax_first:
        logits = torch.softmax(logits, dim=-1)
    sparse_topk = triton_kernels_topk(
        logits,
        topk,
        apply_softmax=not softmax_first,
    )
    if hasattr(sparse_topk, "vals"):
        topk_weights = sparse_topk.vals
        topk_ids = sparse_topk.indx
    else:
        topk_weights, topk_ids = sparse_topk[:2]
    topk_ids = topk_ids.to(torch.int32)
    if num_token_non_padded is not None:
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
        topk_ids[indices >= num_token_non_padded, :] = -1
    return topk_weights, topk_ids


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    from freetoken.kernel.backend import is_sgl_kernel_installed

    if not is_sgl_kernel_installed():
        from freetoken.utils.arch import is_sm70_supported

        if not is_sm70_supported():
            # The fused triton path ranks the scatter with tl.atomic_add, and triton
            # lowers every atomic to a scoped+ordered PTX encoding that only sm_70+
            # can assemble -- there is no sem=/scope= that avoids it. The staged
            # implementation computes the same buffers with a counts/cumsum chain and
            # no atomics, so pre-Volta routes there instead. Slower (5 launches vs 1),
            # which is the right trade against not running at all.
            from freetoken.kernel import moe_align_block_size_triton

            return moe_align_block_size_triton(topk_ids, block_size, num_experts)

        from freetoken.kernel.triton.moe_align import (
            moe_align_block_size as triton_moe_align_block_size,
        )

        return triton_moe_align_block_size(topk_ids, block_size, num_experts)

    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = div_ceil(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)
    sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
) -> Dict[str, int]:

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    if M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


@functools.lru_cache(maxsize=32)
def _get_tuned_moe_configs(
    num_experts: int,
    config_n: int,
    device_name: str,
    triton_version: str,
) -> Dict[int, Dict[str, int]] | None:
    file_name = f"E={num_experts},N={config_n},device_name={device_name}.json"
    version_dir = f"triton_{triton_version.replace('.', '_')}"
    config_roots = []
    if env_dir := os.environ.get("FREETOKEN_MOE_CONFIG_DIR"):
        config_roots.append(Path(env_dir))
    config_roots.append(Path(__file__).with_name("configs"))

    for root in config_roots:
        path = root / version_dir / file_name
        if not path.exists():
            continue
        with path.open() as f:
            configs = json.load(f)
        return {int(batch_size): config for batch_size, config in configs.items()}
    return None


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    M: int,
) -> Dict[str, int]:
    E, N, config_n = w2_shape
    config = None
    if torch.cuda.is_available():
        import triton

        configs = _get_tuned_moe_configs(
            E,
            config_n,
            torch.cuda.get_device_name().replace(" ", "_"),
            triton.__version__,
        )
        if configs:
            config = dict(configs[min(configs.keys(), key=lambda batch: abs(batch - M))])
    if config is None:
        config = get_default_config(M, E, N, w1_shape[2], top_k)
    return config


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Returns ``hidden_states`` itself, overwritten with the routed output. A caller that
    still needs the input afterwards (a shared expert, a residual) must read it BEFORE this
    call or pass a copy. ``fused_experts_decode_impl`` allocates instead, so the contract is
    not shared; the resident bf16 path routes decode through here too."""
    from freetoken.kernel import fused_moe_kernel_triton, moe_sum_reduce_triton
    from freetoken.layers import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

    padded_size = 0
    assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    M = num_tokens
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
    )
    config = get_config_func(M)

    cache = torch.empty(
        M * topk_ids.shape[1] * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(
        (M, topk_ids.shape[1], N),
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2.shape[1]].view(
        (M, topk_ids.shape[1], w2.shape[1]),
    )
    compute_type = hidden_states.dtype

    out_hidden_states = hidden_states
    curr_hidden_states = hidden_states
    tokens_num, _ = curr_hidden_states.shape
    begin_token_idx, end_token_idx = 0, num_tokens

    intermediate_cache1 = intermediate_cache1[:tokens_num]
    intermediate_cache2 = intermediate_cache2[: tokens_num * topk_ids.shape[1]]
    intermediate_cache3 = intermediate_cache3[:tokens_num]
    config = get_config_func(tokens_num)

    curr_topk_ids = topk_ids[begin_token_idx:end_token_idx]
    curr_topk_weights = topk_weights[begin_token_idx:end_token_idx]

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        curr_topk_ids, config["BLOCK_SIZE_M"], E
    )

    fused_moe_kernel_triton(
        curr_hidden_states,
        w1,
        intermediate_cache1,
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk_ids.shape[1],
        config,
        compute_type=compute_type,
    )
    FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}
    FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
    fused_moe_kernel_triton(
        intermediate_cache2,
        w2,
        (intermediate_cache3),
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        config,
        compute_type=compute_type,
    )

    moe_sum_reduce_triton(
        intermediate_cache3,
        out_hidden_states[begin_token_idx:end_token_idx],
    )
    return out_hidden_states


def fused_experts_decode_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    from freetoken.kernel import fused_moe_decode_kernel_triton, moe_sum_reduce_triton
    from freetoken.layers import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert w1.shape[0] == w2.shape[0], "Expert cache size mismatch"
    assert w1.shape[1] == 2 * w2.shape[2], "Intermediate size mismatch"
    assert w2.shape[1] == hidden_states.shape[1], "Output hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert topk_weights.is_contiguous(), "topk_weights must be contiguous"
    assert topk_ids.is_contiguous(), "topk_ids must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert w1.dtype == hidden_states.dtype and w2.dtype == hidden_states.dtype
    assert topk_weights.dtype == torch.float32
    assert topk_ids.dtype == torch.int32
    if activation not in {"silu", "gelu", "gelu_tanh"}:
        raise ValueError(f"Unsupported activation: {activation}")

    M, _ = hidden_states.shape
    top_k = topk_ids.shape[1]
    gate_up_dim = w1.shape[1]
    intermediate_size = gate_up_dim // 2
    config = {
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "num_warps": 8,
    }

    intermediate_cache1 = torch.empty(
        (M, top_k, gate_up_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    fused_moe_decode_kernel_triton(
        hidden_states,
        w1,
        intermediate_cache1,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        False,
        config,
        compute_type=hidden_states.dtype,
    )

    intermediate_cache2 = torch.empty(
        (M * top_k, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}
    FN_MAP[activation](intermediate_cache1.view(-1, gate_up_dim), intermediate_cache2)

    intermediate_cache3 = torch.empty(
        (M, top_k, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    fused_moe_decode_kernel_triton(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        topk_weights,
        topk_ids,
        not apply_router_weight_on_input,
        True,
        config,
        compute_type=hidden_states.dtype,
    )

    out_hidden_states = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(intermediate_cache3, out_hidden_states)
    return out_hidden_states


class FusedMoe(BaseMoeBackend):
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
        )
        return fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
