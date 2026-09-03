"""Decode copy-path bench: ensure_experts + copy_missing across model workloads.

Measures the full per-layer decode movement cost (LRU slot remap + per-bank
fast_index_copy_jit gather) for each supported model's bank layout, sweeping batch
size and miss rate. tok_ms extrapolates one decode token's copy cost across all MoE
layers, assuming every layer sees the same miss profile.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_offload_cache_copy.py
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch

from freetoken.gpu_select import assign_gpu, bind_assigned_gpu, single_gpu_arg
from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache


LAYER_ID = 0


@dataclass(frozen=True)
class ModelProfile:
    layers: int
    experts: int
    topk: int
    quant_format: str
    hidden: int
    inter: int


MODELS = {
    "qwen3.5-35B": ModelProfile(40, 256, 8, "bf16", 2048, 512),
    "qwen3-30B": ModelProfile(48, 128, 8, "bf16", 2048, 768),
    "gemma4-26B": ModelProfile(30, 128, 8, "bf16", 2816, 704),
    "minimax-m2.5-marlin": ModelProfile(62, 256, 8, "nvfp4_marlin", 3072, 1536),
    "minimax-m2.5-triton": ModelProfile(62, 256, 8, "nvfp4", 3072, 1536),
    # GLM-4.7-NVFP4: 92 layers, first_k_dense_replace=3 -> 89 MoE layers;
    # 160 experts, top-8, 1 shared; H=5120, moe_inter=1536; triton 6-bank NVFP4 on sm_90.
    # Bank sizes match the nsys fast_index_copy kernels: gate_up_packed=2*1536*2560=7.86M,
    # down_packed=5120*768=3.93M, gate_up_scale=0.98M, down_scale=0.49M.
    "glm4.7-nvfp4": ModelProfile(89, 160, 8, "nvfp4", 5120, 1536),
    # gpt-oss MXFP4 (block-32 e2m1 codes + e8m0 scales), H == I == 2880, top-4 routing
    "gpt-oss-20b": ModelProfile(24, 32, 4, "mxfp4_triton", 2880, 2880),
    "gpt-oss-120b": ModelProfile(36, 128, 4, "mxfp4_triton", 2880, 2880),
}


def bank_specs(profile: ModelProfile) -> dict[str, tuple[int, torch.dtype]]:
    """Per-bank (elems_per_expert, dtype) in schema order, derived from H and I."""
    h, i = profile.hidden, profile.inter
    per_name = {
        "gate_up": (2 * i * h, torch.bfloat16),
        "down": (h * i, torch.bfloat16),
        # packed e2m1 codes (2 weights/byte) + fp8-e4m3 per-16 block scales
        "gate_up_packed": (2 * i * (h // 2), torch.uint8),
        "gate_up_scale": (2 * i * (h // 16), torch.uint8),
        "gate_up_global": (2 * i, torch.float16),
        "down_packed": (h * (i // 2), torch.uint8),
        "down_scale": (h * (i // 16), torch.uint8),
        "down_global": (h, torch.float16),
        # gpt-oss mxfp4 transposed (_t) banks: e2m1 codes (2 weights/byte) + e8m0
        # per-32 block scales (uint8), plus a per-output bias in compute dtype
        "gate_up_blocks": (2 * i * (h // 2), torch.uint8),
        "gate_up_scales": (2 * i * (h // 32), torch.uint8),
        "gate_up_bias": (2 * i, torch.bfloat16),
        "down_blocks": (h * (i // 2), torch.uint8),
        "down_scales": (h * (i // 32), torch.uint8),
        "down_bias": (h, torch.bfloat16),
    }
    return {name: per_name[name] for name in _BANK_SCHEMAS[profile.quant_format]}


def expert_bytes(profile: ModelProfile) -> int:
    return sum(elems * dtype.itemsize for elems, dtype in bank_specs(profile).values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=single_gpu_arg, default=None,
                        help="GPU UUID or nvidia-smi index (default: the first visible GPU)")
    parser.add_argument("--repeat", type=int, default=25)
    parser.add_argument("--models", type=str, nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument(
        "--cache-slots",
        type=int,
        nargs="+",
        default=None,
        help="absolute slot counts; default per model: [E, 0.4 * L * E] (target scenario)",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--miss-rates", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    return parser.parse_args()


def make_cache(profile: ModelProfile, cache_slots: int, device: torch.device) -> OffloadMoeCache:
    cache = OffloadMoeCache(
        num_layers=profile.layers,
        num_experts=profile.experts,
        cache_size=cache_slots,
        device=device,
        quant_format=profile.quant_format,
    )
    # Only LAYER_ID=0 is ever exercised, so a single layer's worth of host rows is
    # enough; wire the banks registry directly instead of set_bank_sources (which
    # would demand -- and allocate against -- the full per-layer banks). Per the
    # per-layer host bank contract, bank_sources/banks hold a list of per-layer
    # sources; only index 0 (LAYER_ID) is ever populated/read here.
    for name, (elems, dtype) in bank_specs(profile).items():
        source = torch.empty(profile.experts, elems, dtype=dtype, device="cpu", pin_memory=True)
        slot_cache = torch.empty(cache_slots, elems, dtype=dtype, device=device)
        cache.bank_sources[name] = [source]
        cache.bank_caches[name] = slot_cache
        cache.banks.append(([source], slot_cache))
    return cache


def prepare_state(
    cache: OffloadMoeCache,
    active_unique: int,
    miss_count: int,
    device: torch.device,
) -> None:
    # Every slot starts owned by some OTHER layer, so it is occupied (eviction has to do
    # real work) but never a hit for LAYER_ID. Slot ownership lives in flashlib's flat id
    # space: id == layer_id * num_experts + expert.
    cache.slot_for_id.fill_(-1)
    other_layer_experts = (
        torch.arange(cache.cache_size, dtype=torch.int32, device=device) % cache.num_experts
    )
    cache.id_of_slot.copy_(cache.num_experts + other_layer_experts)  # layer 1
    cache.usage.copy_(torch.arange(cache.cache_size, dtype=torch.int64, device=device) + 1)
    cache.step.zero_()
    cache.num_indices.zero_()

    cached_count = active_unique - miss_count
    if cached_count:
        experts = torch.arange(cached_count, dtype=torch.int32, device=device)
        cache.slot_for_id[LAYER_ID, :cached_count] = experts
        cache.id_of_slot[:cached_count] = LAYER_ID * cache.num_experts + experts


def make_topk_ids(
    batch_size: int, topk: int, active_unique: int, device: torch.device
) -> torch.Tensor:
    topk_ids = torch.arange(batch_size * topk, dtype=torch.int32, device=device) % active_unique
    return topk_ids.view(batch_size, topk).contiguous()


def time_case(
    cache: OffloadMoeCache,
    profile: ModelProfile,
    batch_size: int,
    miss_rate: float,
    repeat: int,
    device: torch.device,
) -> tuple[int, int, float]:
    active_unique = min(profile.experts, batch_size * profile.topk)
    miss_count = int(round(active_unique * miss_rate))
    base_topk_ids = make_topk_ids(batch_size, profile.topk, active_unique, device)
    topk_ids = torch.empty_like(base_topk_ids)

    prepare_state(cache, active_unique, miss_count, device)
    topk_ids.copy_(base_topk_ids)
    torch.cuda.synchronize(device)
    cache.ensure_experts(LAYER_ID, topk_ids)
    cache.copy_missing()
    torch.cuda.synchronize(device)

    samples = []
    for _ in range(repeat):
        prepare_state(cache, active_unique, miss_count, device)
        topk_ids.copy_(base_topk_ids)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(10**7)
        start.record()
        cache.ensure_experts(LAYER_ID, topk_ids)
        cache.copy_missing()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))

    return active_unique, miss_count, statistics.median(samples)


def print_table(
    name: str,
    profile: ModelProfile,
    cache_slots: int,
    batch_sizes: list[int],
    miss_rates: list[float],
    repeat: int,
    device: torch.device,
) -> None:
    per_expert = expert_bytes(profile)
    cache_gib = cache_slots * per_expert / 2**30
    print(
        f"\n{name} ({profile.quant_format}, {len(bank_specs(profile))} banks, "
        f"L={profile.layers} E={profile.experts} k={profile.topk}) "
        f"ensure_experts + copy_missing | cache_slots={cache_slots} ({cache_gib:.1f} GiB) | "
        f"expert_bytes={per_expert / 2**20:.2f} MiB"
    )
    print("bs active miss_rate misses time_ms copy_MiB bw_GBps tok_ms")
    print("-- ------ --------- ------ ------- -------- ------- ------")

    cache = make_cache(profile, cache_slots, device)
    for batch_size in batch_sizes:
        for miss_rate in miss_rates:
            active, misses, time_ms = time_case(
                cache, profile, batch_size, miss_rate, repeat, device
            )
            copy_mib = misses * per_expert / 2**20
            bandwidth_gbps = (misses * per_expert / (time_ms * 1e6)) if misses else 0.0
            tok_ms = time_ms * profile.layers
            print(
                f"{batch_size:2d} {active:6d} {miss_rate:9.2f} {misses:6d} "
                f"{time_ms:7.3f} {copy_mib:8.1f} {bandwidth_gbps:7.1f} {tok_ms:6.2f}",
                flush=True,
            )
    del cache
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required"
    try:
        assign_gpu(args.gpu)
        device = bind_assigned_gpu()
    except (ValueError, RuntimeError) as e:
        raise SystemExit(f"error: {e}") from e

    print("gpu", torch.cuda.get_device_name(device), flush=True)
    for name in args.models:
        profile = MODELS[name]
        slot_counts = args.cache_slots or [
            profile.experts,
            int(0.4 * profile.layers * profile.experts),
        ]
        for cache_slots in slot_counts:
            print_table(
                name, profile, cache_slots, args.batch_sizes, args.miss_rates, args.repeat, device
            )


if __name__ == "__main__":
    main()
