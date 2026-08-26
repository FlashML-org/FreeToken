from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fast_index_copy_multi_strided_copies_only_payload_prefixes() -> None:
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_strided_jit

    device = torch.device("cuda")
    src0 = torch.arange(4 * 32, dtype=torch.uint8, device=device).view(4, 32)
    src1 = (
        (torch.arange(4 * 64, dtype=torch.int32, device=device) % 251)
        .to(torch.uint8)
        .view(4, 64)
    )
    dst0 = torch.full((5, 64), 0xEE, dtype=torch.uint8, device=device)
    dst1 = torch.full((5, 80), 0xEE, dtype=torch.uint8, device=device)

    dst_ptrs = torch.tensor(
        [dst0.data_ptr(), dst1.data_ptr()], dtype=torch.int64, device=device
    )
    src_ptrs = torch.tensor(
        [src0.data_ptr(), src1.data_ptr()], dtype=torch.int64, device=device
    )
    copy_bytes = torch.tensor([32, 64], dtype=torch.int64, device=device)
    dst_row_strides = torch.tensor([64, 80], dtype=torch.int64, device=device)
    src_row_strides = torch.tensor([32, 64], dtype=torch.int64, device=device)
    dst_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    src_indices = torch.tensor([1, 3], dtype=torch.int32, device=device)
    num_indices = torch.tensor([2], dtype=torch.int64, device=device)

    fast_index_copy_multi_strided_jit(
        dst_ptrs,
        src_ptrs,
        copy_bytes,
        dst_row_strides,
        src_row_strides,
        dst_indices,
        src_indices,
        num_indices,
    )
    torch.cuda.synchronize()

    assert torch.equal(dst0[2, :32], src0[1])
    assert torch.equal(dst0[0, :32], src0[3])
    assert torch.all(dst0[[0, 2], 32:] == 0xEE)
    assert torch.all(dst0[[1, 3, 4]] == 0xEE)
    assert torch.equal(dst1[2, :64], src1[1])
    assert torch.equal(dst1[0, :64], src1[3])
    assert torch.all(dst1[[0, 2], 64:] == 0xEE)
    assert torch.all(dst1[[1, 3, 4]] == 0xEE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fast_index_copy_rows_strided_avoids_payload_sized_cuda_temporary() -> None:
    from freetoken.kernel.fast_index_copy import fast_index_copy_rows_strided_jit

    rows = 16
    payload = 1 << 20
    destination_stride = payload * 2
    source = torch.arange(rows * payload, dtype=torch.int64).view(torch.uint8)
    source = source[: rows * payload].reshape(rows, payload).pin_memory()
    destination = torch.full(
        (rows, destination_stride), 0xEE, dtype=torch.uint8, device="cuda"
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()

    fast_index_copy_rows_strided_jit(destination, source)
    torch.cuda.synchronize()

    temporary_peak = torch.cuda.max_memory_allocated() - allocated_before
    assert temporary_peak < 1 << 20
    assert torch.equal(destination[:, :payload].cpu(), source)
    assert torch.equal(
        destination[:, payload:].cpu(),
        torch.full((rows, payload), 0xEE, dtype=torch.uint8),
    )
