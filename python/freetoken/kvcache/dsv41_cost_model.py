"""Byte accounting for V4.1's shared compressed KV and per-layer windows."""

from __future__ import annotations

import math

from .dsv4_cost_model import DSV4PoolSizes, _dsv4_window_floor_pages
from .dsv41_layout import dsv41_row_bytes


def source_layers(args) -> tuple[tuple[int | None, ...], tuple[int | None, ...]]:
    ratios = tuple(args.compress_ratios)[:args.n_layers]
    if len(ratios) != args.n_layers or any(r not in (0, 1, 2) for r in ratios):
        raise ValueError("DeepSeek-V4.1 requires one compression ratio (0, 1, or 2) per layer")
    kv_ids, index_ids = set(args.kv_source_layers), set(args.index_source_layers)
    if not kv_ids <= index_ids or any(i < 0 or i >= args.n_layers for i in index_ids):
        raise ValueError("DeepSeek-V4.1 KV sources must be valid index source layers")
    kv_map, index_map = [], []
    kv = index = None
    for layer, ratio in enumerate(ratios):
        if layer in kv_ids:
            kv = layer
        if layer in index_ids:
            index = layer
        if ratio:
            if kv is None or index is None or ratios[kv] != ratio or ratios[index] != ratio:
                raise ValueError(f"DeepSeek-V4.1 layer {layer} has no matching KV/index source")
            kv_map.append(kv)
            index_map.append(index)
        else:
            if layer in kv_ids or layer in index_ids:
                raise ValueError("Sliding-window-only layers cannot publish compressed KV/indexes")
            kv_map.append(None)
            index_map.append(None)
    return tuple(kv_map), tuple(index_map)


def dsv41_pool_sizes(num_pages, args, swa_ratio, P=128, n_win_pages=None):
    source_layers(args)
    if num_pages < 1 or P <= 0 or P % 2:
        raise ValueError("DeepSeek-V4.1 requires positive pages with an even window size")
    if n_win_pages is None:
        n_win_pages = math.ceil(swa_ratio * num_pages)
    n_win_pages = min(num_pages, max(1, n_win_pages))
    sizes = DSV4PoolSizes(P, swa_ratio, num_pages * P, n_win_pages * P, n_win_pages)
    owners = set(args.kv_source_layers)
    for layer, ratio in enumerate(args.compress_ratios[:args.n_layers]):
        owns = layer in owners
        sizes.cmp_blocks.append(sizes.full_token // ratio if owns else None)
        sizes.idx_blocks.append(sizes.full_token // ratio if owns else None)
        sizes.state_slots.append(n_win_pages * 2 if owns and ratio == 2 else None)
        sizes.ring_sizes.append(2 if owns and ratio == 2 else None)
        sizes.idx_state_slots.append(None)
    return sizes


def dsv41_pool_bytes(sizes, args, n_scratch=1, kv_quant="none"):
    window_bytes, cmp_bytes, idx_bytes = dsv41_row_bytes(args, kv_quant)
    total = sizes.n_win_slots * args.n_layers * window_bytes
    total += (sizes.full_token + 1) * 8
    for layer in args.kv_source_layers:
        total += (sizes.cmp_blocks[layer] + n_scratch) * cmp_bytes
        total += (sizes.idx_blocks[layer] + n_scratch) * idx_bytes
        if sizes.state_slots[layer] is not None:
            total += (sizes.state_slots[layer] + 1) * args.head_dim * 8
    return int(total)


def dsv41_unit_bytes(args, P=128, kv_quant="none"):
    window_bytes, cmp_bytes, idx_bytes = dsv41_row_bytes(args, kv_quant)
    full = 8.0
    window = args.n_layers * window_bytes
    for layer in args.kv_source_layers:
        ratio = args.compress_ratios[layer]
        full += (cmp_bytes + idx_bytes) / ratio
        if ratio == 2:
            window += 2 * args.head_dim * 8 / P
    return int(math.ceil(full)), int(math.ceil(window))


def _dsv41_pool_sizes(config, num_pages, num_swa_pages=None):
    args = config.model_config.dsv41_args
    P = args.window_size
    floor = _dsv4_window_floor_pages(config, P)
    target = num_swa_pages if num_swa_pages is not None else config.swa_num_pages_override
    win = max(floor, int(target) + 1 if target is not None else math.ceil(config.swa_full_tokens_ratio * num_pages))
    return dsv41_pool_sizes(num_pages, args, config.swa_full_tokens_ratio, P, win)


def dsv41_auto_cost_model(config):
    args = config.model_config.dsv41_args
    window_bytes, cmp_bytes, idx_bytes = dsv41_row_bytes(args, getattr(config, "kv_quant", "none"))
    P = args.window_size
    floor = _dsv4_window_floor_pages(config, P)
    full = P * 8
    window = P * args.n_layers * window_bytes
    fixed = 8
    for layer in args.kv_source_layers:
        ratio = args.compress_ratios[layer]
        full += P // ratio * (cmp_bytes + idx_bytes)
        fixed += (config.max_running_req + 1) * (cmp_bytes + idx_bytes)
        if ratio == 2:
            window += 2 * args.head_dim * 8
            fixed += args.head_dim * 8
    fixed += full  # The planner counts usable pages; the pool also owns one dummy page.
    if config.swa_num_pages_override is not None:
        fixed += max(floor, config.swa_num_pages_override + 1) * window
        per_page = full
    else:
        ratio = config.swa_full_tokens_ratio
        per_page = math.ceil(full + ratio * window)
        # Bound both the working-set floor and ceil(ratio * physical_pages) rounding.
        fixed += math.ceil(max(floor * (1 - ratio), ratio + 1) * window)
    return per_page, fixed, P, floor * P


def dsv41_solve_num_pages(config, available_bytes):
    args = config.model_config.dsv41_args
    kv_quant = getattr(config, "kv_quant", "none")
    floor = _dsv4_window_floor_pages(config, args.window_size) + 1
    def cost(pages):
        return dsv41_pool_bytes(_dsv41_pool_sizes(config, pages), args, config.max_running_req + 1, kv_quant)
    if cost(floor) > available_bytes:
        raise ValueError("KV budget cannot fit the DeepSeek-V4.1 window working set")
    full_bytes, _ = dsv41_unit_bytes(args, args.window_size, kv_quant)
    lo, hi = floor, max(floor + 1, available_bytes // max(1, full_bytes * args.window_size) + 1)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if cost(mid) <= available_bytes:
            lo = mid
        else:
            hi = mid
    return lo - 1
