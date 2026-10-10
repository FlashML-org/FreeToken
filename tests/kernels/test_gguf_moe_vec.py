import numpy as np
import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

# maxGridDim{Y,Z} is 65535 on every CUDA compute capability; grid.x reaches 2**31-1.
MAX_GRID_YZ = 65535


def _q4_0_weights(num_experts: int, nrows: int, ncols: int, seed: int = 0) -> torch.Tensor:
    """Structurally valid Q4_0 banks: each 32-value block is {half d; uint8 qs[16]}.

    The scale has to be a real finite half -- random bytes decode to inf/nan and make any
    comparison vacuous.
    """
    rng = np.random.default_rng(seed)
    nblocks = ncols // 32
    blocks = np.zeros((num_experts, nrows, nblocks, 18), dtype=np.uint8)
    scale = np.array([0.01], dtype=np.float16).view(np.uint8)
    blocks[..., 0], blocks[..., 1] = scale[0], scale[1]
    blocks[..., 2:] = rng.integers(0, 256, blocks[..., 2:].shape, dtype=np.uint8)
    return torch.from_numpy(blocks.reshape(num_experts, nrows, nblocks * 18)).cuda()


def test_moe_vec_above_grid_z_limit_matches_split_batches():
    """A batch whose ``tokens * top_k`` exceeds 65535 must still compute the right thing.

    ``moe_vec_*_q8_1_cuda`` used to launch that product as grid.z, which CUDA rejects with
    ``cudaErrorInvalidValue`` above 65535. With top_k=8 that made any prefill batch of 8192
    tokens fail -- and ``--max-extend-tokens`` defaults to exactly 8192, so a single long
    prompt, or a few concurrent ones packed into one batch, hit it. The launch return code
    was unchecked, so the failure surfaced at whatever unrelated CUDA call ran next.

    Splitting the same batch in half keeps each launch under the old limit, so the halves
    are a reference the pre-fix kernel could actually produce.
    """
    from freetoken.kernel.gguf import ggml_moe_a8_vec
    from freetoken.models.gguf.dequant import GGML_Q4_0

    num_experts, hidden, nrows, top_k = 4, 256, 64, 8
    tokens = (MAX_GRID_YZ // top_k) + 1  # 8192 -> 65536 > 65535, one over the old cap
    assert tokens * top_k > MAX_GRID_YZ

    torch.manual_seed(0)
    weight = _q4_0_weights(num_experts, nrows, hidden)
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, num_experts, (tokens, top_k), dtype=torch.int32, device="cuda")

    out = ggml_moe_a8_vec(x, weight, topk_ids, top_k, int(GGML_Q4_0), nrows, tokens)
    torch.cuda.synchronize()

    half = tokens // 2
    assert half * top_k <= MAX_GRID_YZ
    ref = torch.cat([
        ggml_moe_a8_vec(
            x[s:e].contiguous(), weight, topk_ids[s:e].contiguous(),
            top_k, int(GGML_Q4_0), nrows, e - s,
        )
        for s, e in ((0, half), (half, tokens))
    ])
    torch.cuda.synchronize()

    assert out.shape == (tokens * top_k, nrows)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
