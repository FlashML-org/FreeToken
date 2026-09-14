"""V4.1's native expert-copy rows must have matching prebuilt-cache spec names."""

import importlib

import pytest
import torch

from freetoken.kernel.aot_models import (
    SUPPORTED_MODELS, expert_bank_row_bytes, index_variants, store_element_sizes,
)


ROWS = {
    "gate_up_packed": 11_796_480,
    "gate_up_scale": 1_474_560,
    "gate_up_global": 9_216,
    "down_packed": 5_898_240,
    "down_scale": 737_280,
    "down_global": 10_240,
}


def test_v41_aot_entry_matches_native_expert_layout():
    entry = next(m for m in SUPPORTED_MODELS if m.architecture == "DeepseekV41ForCausalLM")
    assert (entry.hidden_size, entry.moe_intermediate_size, entry.top_k) == (5120, 2304, 6)
    assert entry.expert_formats == ("nvfp4",)
    assert not store_element_sizes(entry) and not index_variants(entry)
    assert expert_bank_row_bytes("nvfp4", entry.hidden_size, entry.moe_intermediate_size) == ROWS


def test_v41_runtime_copy_names_are_in_default_aot_specs(monkeypatch):
    from freetoken.kernel.aot import default_kernel_specs
    from freetoken.kernel.utils import _make_name

    copy = importlib.import_module("freetoken.kernel.fast_index_copy")
    monkeypatch.setattr(copy, "load_jit", lambda name, *args, **kwargs: _make_name(name, *args))
    claimed = {spec.name for spec in default_kernel_specs()}
    for feature_size in ROWS.values():
        threads, worker_size, blocks = copy.default_worker_args(feature_size)
        runtime_name = copy._jit_fast_index_copy_module.__wrapped__(
            feature_size=feature_size, worker_threads=threads,
            worker_feature_size=worker_size, num_block=blocks,
        )
        assert runtime_name in claimed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("feature_size", ROWS.values(), ids=ROWS.keys())
def test_v41_native_bank_row_copy_matches_host_bytes(feature_size):
    from freetoken.kernel.fast_index_copy import fast_index_copy_jit

    src = torch.randint(0, 256, (3, feature_size), dtype=torch.uint8, pin_memory=True)
    dst = torch.zeros((4, feature_size), dtype=torch.uint8, device="cuda")
    src_indices = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
    dst_indices = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    fast_index_copy_jit(dst, dst_indices, src, src_indices)
    expected = torch.zeros_like(dst, device="cpu")
    expected[[1, 3]] = src[[2, 0]]
    assert torch.equal(dst.cpu(), expected)
