from __future__ import annotations

import pytest
import torch

from freetoken.kernel import fast_index_copy as fast_copy


def test_rocm_fallback_copies_only_requested_rows() -> None:
    src = torch.arange(20, dtype=torch.float32).reshape(5, 4).to(torch.bfloat16)
    dst = torch.full((4, 4), -1, dtype=torch.bfloat16)
    src_indices = torch.tensor([4, 1, 3], dtype=torch.int32)
    dst_indices = torch.tensor([2, 0, 3], dtype=torch.int32)
    num_indices = torch.tensor([2], dtype=torch.int64)

    fast_copy._rocm_index_copy_fallback(
        dst,
        dst_indices,
        src,
        src_indices,
        num_indices,
    )

    torch.testing.assert_close(dst[2], src[4], rtol=0, atol=0)
    torch.testing.assert_close(dst[0], src[1], rtol=0, atol=0)
    torch.testing.assert_close(dst[1], torch.full((4,), -1, dtype=torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(dst[3], torch.full((4,), -1, dtype=torch.bfloat16), rtol=0, atol=0)


def test_rocm_dispatch_does_not_build_cuda_jit(monkeypatch: pytest.MonkeyPatch) -> None:
    src = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    dst = torch.zeros((3, 4), dtype=torch.float32)
    src_indices = torch.tensor([2, 0], dtype=torch.int32)
    dst_indices = torch.tensor([1, 2], dtype=torch.int32)

    monkeypatch.setattr(fast_copy, "_is_rocm", lambda: True)

    def fail_jit(**_kwargs):
        raise AssertionError("ROCm dispatch must not compile the CUDA fast-index-copy JIT")

    monkeypatch.setattr(fast_copy, "_jit_fast_index_copy_module", fail_jit)

    fast_copy.fast_index_copy_jit(dst, dst_indices, src, src_indices)

    torch.testing.assert_close(dst[1], src[2], rtol=0, atol=0)
    torch.testing.assert_close(dst[2], src[0], rtol=0, atol=0)


def test_rocm_priority_mode_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    src = torch.zeros((2, 4), dtype=torch.float32)
    dst = torch.zeros((2, 4), dtype=torch.float32)
    indices = torch.tensor([0], dtype=torch.int32)

    monkeypatch.setattr(fast_copy, "_is_rocm", lambda: True)

    with pytest.raises(NotImplementedError, match="priority scheduling"):
        fast_copy.fast_index_copy_jit(
            dst,
            indices,
            src,
            indices,
            priority="high",
            sync_flag=torch.zeros((1,), dtype=torch.int32),
        )
