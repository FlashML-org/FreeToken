"""The fused multi-bank ``copy_missing`` path must move exactly the same bytes as the
legacy per-bank ``fast_index_copy_jit`` loop, for every miss count (including the
zero-copy case), across banks of differing per-row sizes.
"""

from __future__ import annotations

from pathlib import Path
import re
from uuid import uuid4

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

CUDA = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="needs an available CUDA or HIP runtime",
)
HIP = pytest.mark.skipif(
    torch is None or not (torch.cuda.is_available() and torch.version.hip),
    reason="needs an available HIP runtime",
)

# mxfp4_triton 6-bank schema with mixed 16B-aligned per-row sizes (bytes), >=256 so the
# legacy per-bank kernel's vectorized template is valid. Sizes not covered by a model in
# kernel/aot_models.py must be listed in its TEST_FEATURE_SIZES so the per-bank kernels
# stay prebuilt under FREETOKEN_DISABLE_JIT=1.
FEATS = [8192, 512, 256, 4096, 512, 256]


def _build_cache(num_layers, num_experts, cache_size):
    dev = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=num_layers, num_experts=num_experts, cache_size=cache_size,
        device=dev, cache_policy="lru", prefill_overlap=False, quant_format="mxfp4_triton",
    )
    schema = _BANK_SCHEMAS["mxfp4_triton"]
    # Views into one flat tensor: only per-layer addressing matters here, not
    # independent allocations.
    sources = {
        name: list(torch.randint(0, 256, (num_layers * num_experts, feat), dtype=torch.uint8, device=dev)
                   .split(num_experts))
        for name, feat in zip(schema, FEATS)
    }
    cache.set_bank_sources(sources)  # also builds the fused-copy descriptor
    return cache


def _assert_fused_copy_matches_per_bank(num_indices):
    num_layers, num_experts, cache_size = 8, 8, 32
    layer_id = 3  # exercise a non-zero per-layer source selection, not just layer 0
    cache = _build_cache(num_layers, num_experts, cache_size)
    assert cache._copy_fused_ok, "fused copy should activate for 16B-aligned banks"
    # copy_missing resolves the per-layer source through this (normally set by
    # ensure_experts/materialize_layer); poked directly here since this test drives
    # evict_slots/src_indices/num_indices by hand.
    cache._pending_src_layer = layer_id

    cache.num_indices.fill_(num_indices)
    if num_indices:
        dev = torch.device("cuda")
        dst_rows = torch.tensor([19, 3, 24, 7, 14, 29, 0, 11], dtype=torch.int32, device=dev)
        src_rows = torch.tensor([6, 2, 7, 1, 5, 0, 4, 3], dtype=torch.int32, device=dev)
        cache.evict_slots[:num_indices] = dst_rows[:num_indices]
        # src_indices are layer-local expert rows (0..num_experts) under the new contract.
        cache.src_indices[:num_indices] = src_rows[:num_indices]

    # reference: legacy per-bank loop
    for _, c in cache.banks:
        c.zero_()
    cache._copy_fused_ok = False
    cache.copy_missing()
    torch.cuda.synchronize()
    ref = [c.clone() for _, c in cache.banks]

    # fused multi-bank launch
    for _, c in cache.banks:
        c.zero_()
    cache._copy_fused_ok = True
    cache.copy_missing()
    torch.cuda.synchronize()

    for b, (r, (_, c)) in enumerate(zip(ref, cache.banks)):
        assert torch.equal(r, c), f"bank {b} (feat={FEATS[b]}) fused != per-bank at num_indices={num_indices}"


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("num_indices", [0, 1, 4, 8])
def test_fused_copy_matches_per_bank(num_indices):
    _assert_fused_copy_matches_per_bank(num_indices)


def test_multi_index_copy_matchers_admit_only_cuda_and_rocm_devices():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "freetoken"
        / "kernel"
        / "csrc"
        / "jit"
        / "fast_index_copy.cuh"
    )
    source = source_path.read_text(encoding="utf-8")
    start = source.index("struct MultiIndexCopyKernel")
    matcher_block = source[start:source.index("\n};", start)]
    matcher_pattern = re.compile(
        r"TensorMatcher\(\{(?P<size>B|L|1)\}\)\s*"
        r"\.with_dtype<[^;]+?>\([^)]*\)\s*"
        r"\.with_device\(device\)",
        re.DOTALL,
    )
    matches = list(matcher_pattern.finditer(matcher_block))

    assert matcher_block.count("SymbolicDevice{}") == 1
    assert "device.set_options<kDLCUDA, kDLROCM>();" in matcher_block
    assert [match["size"] for match in matches] == ["B", "L", "1"]
    for host_or_cpu_device in ("kDLCUDAHost", "kDLROCMHost", "kDLCPU"):
        assert host_or_cpu_device not in matcher_block


def _load_fresh_hip_multi_index_copy_module(build_directory):
    from freetoken.kernel.utils import load_jit, make_cpp_args

    args = make_cpp_args(256, 2)
    return load_jit(
        f"fast_index_copy_multi_hip_regression_{uuid4().hex}",
        *args,
        cuda_files=["fast_index_copy.cuh"],
        cuda_wrappers=[("launch", f"&MultiIndexCopyKernel<{args}>::run")],
        build_directory=str(build_directory),
    )


@HIP
@pytest.mark.slow
def test_hip_fused_copy_jit_executes_multi_bank_device_descriptors(tmp_path):
    device = torch.device("cuda")
    src_banks = [
        torch.arange(8 * 16, dtype=torch.uint8, device=device).reshape(8, 16),
        torch.arange(8 * 48, dtype=torch.uint8, device=device).reshape(8, 48),
    ]
    dst_banks = [torch.full_like(source, 0xA5) for source in src_banks]
    dst_rows = torch.tensor([6, 1, 4], dtype=torch.int64, device=device)
    src_rows = torch.tensor([2, 7, 0], dtype=torch.int64, device=device)
    dst_indices = dst_rows.to(dtype=torch.int32)
    src_indices = src_rows.to(dtype=torch.int32)
    num_indices = torch.tensor([len(dst_rows)], dtype=torch.int64, device=device)
    dst_ptrs = torch.tensor([bank.data_ptr() for bank in dst_banks], dtype=torch.int64, device=device)
    src_ptrs = torch.tensor([bank.data_ptr() for bank in src_banks], dtype=torch.int64, device=device)
    feat_bytes = torch.tensor([bank.size(1) for bank in src_banks], dtype=torch.int64, device=device)

    module = _load_fresh_hip_multi_index_copy_module(tmp_path)
    module.launch(dst_ptrs, src_ptrs, feat_bytes, dst_indices, src_indices, num_indices)
    torch.cuda.synchronize()

    for source, destination in zip(src_banks, dst_banks):
        torch.testing.assert_close(destination[dst_rows], source[src_rows])
        untouched = torch.ones(destination.size(0), dtype=torch.bool, device=device)
        untouched[dst_rows] = False
        assert torch.all(destination[untouched] == 0xA5)

    # The native matcher must reject a non-GPU descriptor before any raw pointer
    # is consumed, and must preserve the shared-device contract across descriptors.
    with pytest.raises(Exception, match="not in the allowed options"):
        module.launch(dst_ptrs.cpu(), src_ptrs, feat_bytes, dst_indices, src_indices, num_indices)
    with pytest.raises(Exception, match="Device mismatch"):
        module.launch(dst_ptrs, src_ptrs.cpu(), feat_bytes, dst_indices, src_indices, num_indices)

    # This also exercises the real fused OffloadMoeCache.copy_missing caller with
    # multiple banks and a nontrivial source-to-destination row permutation.
    _assert_fused_copy_matches_per_bank(4)
