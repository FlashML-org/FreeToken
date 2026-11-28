"""Intermediate-dim expert sharding (``layers/quantization/moe/shard.py``).

The kernel's ``piece_axes`` / ``bank_axes`` are the one declaration that loader slicing,
narrow-from-wide bank fetches and shard alignment derive from, so these check them against
what the kernels actually do: slicing pieces then packing must equal packing a wider shard
then taking its byte runs, and the partial outputs of a shard cover must sum to the full expert.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from freetoken.layers.quantization.moe import ExpertView, Interval, MoEConfig
from freetoken.layers.quantization.moe.fp8_block import Fp8BlockMoEMethod
from freetoken.layers.quantization.moe.gguf import GgmlMoEKernel
from freetoken.layers.quantization.moe.mxfp4 import Mxfp4MoEMethod
from freetoken.layers.quantization.moe.nvfp4 import Nvfp4MoEMethod
from freetoken.layers.quantization.moe.shard import ExpertSlicing, standard_piece_axes
from freetoken.layers.quantization.moe.unquantized import UnquantizedMoEMethod

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, H, TOPK = 4, 256, 2


def _u8(g, *shape, lo=0, hi=256, device="cpu"):
    return torch.randint(lo, hi, shape, dtype=torch.uint8, generator=g).to(device)


def _ds_mxfp4(g, i, device="cpu"):
    """DeepSeek-V4 ``ds_fp4`` pieces: separate gate / up, e8m0 scale bytes near 2^0 (pack views them as e8m0)."""
    p = {}
    for proj in ("gate", "up"):
        p[proj] = _u8(g, E, i, H // 2, device=device)
        p[f"{proj}_scale"] = _u8(g, E, i, H // 32, lo=122, hi=128, device=device)
    p["down"] = _u8(g, E, H, i // 2, device=device)
    p["down_scale"] = _u8(g, E, H, i // 32, lo=122, hi=128, device=device)
    return p


def _nvfp4(g, i, device="cpu"):
    """ModelOpt NVFP4 pieces: separate gate / up, fp8 block scales, one global per expert."""
    def scale(*shape):
        return (torch.rand(shape, generator=g) + 0.25).to(torch.float8_e4m3fn).to(device)

    def glob():
        return (torch.rand(E, generator=g) * 0.02 + 0.01).to(device)

    p = {}
    for proj in ("gate", "up"):
        p[proj], p[f"{proj}_scale"], p[f"{proj}_global"] = _u8(g, E, i, H // 2, device=device), scale(E, i, H // 16), glob()
    p["down"], p["down_scale"], p["down_global"] = _u8(g, E, H, i // 2, device=device), scale(E, H, i // 16), glob()
    return p


def _bf16(g, i, device="cpu"):
    return {"gate_up": torch.randn(E, 2 * i, H, generator=g).bfloat16().to(device),
            "down": torch.randn(E, H, i, generator=g).bfloat16().to(device)}


def _gptoss(g, i, device="cpu"):
    """gpt-oss pieces as HF ships them: interleaved gate|up rows, biases."""
    return {
        "gate_up": _u8(g, E, 2 * i, H // 32, 16), "gate_up_scale": _u8(g, E, 2 * i, H // 32, lo=122, hi=128),
        "gate_up_bias": torch.randn(E, 2 * i, generator=g).bfloat16(),
        "down": _u8(g, E, H, i // 32, 16), "down_scale": _u8(g, E, H, i // 32, lo=122, hi=128),
        "down_bias": torch.randn(E, H, generator=g).bfloat16(),
    }


def _cfg(i, **kw):
    return MoEConfig(num_experts=E, hidden=H, intermediate=i, top_k=TOPK, **kw)


FORMATS = {
    "ds_mxfp4": (lambda i: Mxfp4MoEMethod(_cfg(i, act_block=128), "triton"), _ds_mxfp4),
    "nvfp4": (lambda i: Nvfp4MoEMethod(_cfg(i), "triton"), _nvfp4),
    "bf16": (lambda i: UnquantizedMoEMethod(_cfg(i, strategy="offload")), _bf16),
    "gptoss": (lambda i: Mxfp4MoEMethod(_cfg(i, has_bias=True, interleaved=True, alpha=1.702, limit=7.0), "triton_gptoss"), _gptoss),
}


def _pack(slicing: ExpertSlicing, pieces, width, device="cpu"):
    layout = slicing.layout_at(width)
    banks = {r: torch.zeros((E, *s.shape), dtype=s.dtype, device=device) for r, s in layout.items() if not s.resident}
    resident = slicing.kernel.pack(pieces, slicing.config_at(width), banks)
    return banks, resident


@pytest.mark.parametrize("fmt", sorted(FORMATS))
@pytest.mark.parametrize("stored, view", [((0, 1024), (0, 512)), ((0, 1024), (512, 1024)), ((512, 1024), (768, 1024)), ((0, 1024), (0, 1024))])
def test_narrow_bank_is_byte_runs_of_wider_bank(fmt, stored, view):
    """A kernel's bank_axes must match its pack: fetching a view out of a wider stored shard gives
    the bytes of packing the view's own pieces (incl. the bias the shard at 0 alone keeps)."""
    make_method, make_pieces = FORMATS[fmt]
    slicing = make_method(1024).expert_slicing()
    pieces = make_pieces(torch.Generator().manual_seed(0), 1024)
    stored, view = Interval(*stored), Interval(*view)

    wide, _ = _pack(slicing, slicing.slice_pieces(pieces, stored), stored.width)
    narrow, _ = _pack(slicing, slicing.slice_pieces(pieces, view), view.width)
    for role, runs in slicing.bank_runs(stored, view).items():
        if role not in narrow:
            continue  # GPU-resident per-expert values
        rows = wide[role].reshape(E, -1).view(torch.uint8)
        fetched = torch.stack([runs.gather(rows[e]) for e in range(E)])
        assert torch.equal(fetched, narrow[role].reshape(E, -1).view(torch.uint8)), role


@pytest.mark.parametrize("interleaved", [False, True])
def test_piece_shards_sum_to_the_full_expert(interleaved):
    """Each standard piece rule, on a float reference MLP: the shards' partial outputs of a cover
    (two-level, like a CPU pool over GPU shards) add up to the unsharded expert, bias once."""
    g = torch.Generator().manual_seed(1)
    i = 64
    gate, up = torch.randn(E, i, H, generator=g), torch.randn(E, i, H, generator=g)
    fused = torch.stack([gate, up], 2).flatten(1, 2) if interleaved else torch.cat([gate, up], 1)
    pieces = {"gate_up": fused, "gate_up_bias": torch.randn(E, 2 * i, generator=g),
              "down": torch.randn(E, H, i, generator=g), "down_bias": torch.randn(E, H, generator=g),
              "down_global": torch.rand(E, generator=g), "gate_up_global": torch.rand(E, 1, generator=g)}
    x = torch.randn(H, generator=g)
    axes = standard_piece_axes(interleaved)

    def mlp(p, width):
        h = (p["gate_up"] @ x + p["gate_up_bias"]) * p["gate_up_global"]
        gh, uh = (h[:, 0::2], h[:, 1::2]) if interleaved else (h[:, :width], h[:, width:])
        return torch.einsum("ehi,ei->eh", p["down"], F.silu(gh) * uh) * p["down_global"][:, None] + p["down_bias"]

    def shard(p, view):
        from freetoken.layers.quantization.moe.shard import slice_tensor

        return {r: slice_tensor(t, axes[r], view, i) for r, t in p.items()}

    cover = [Interval(0, 16), Interval(16, 32), Interval(32, 64)]
    total = sum(mlp(shard(pieces, v), v.width) for v in cover)
    torch.testing.assert_close(total, mlp(pieces, i), rtol=1e-4, atol=1e-3)


def test_gptoss_piece_rules_agree_with_its_tp_reader():
    """gpt-oss's own TP reader (``_source_slices``) and the generic interleaved piece rules pick the same bytes."""
    from freetoken.distributed import DistributedInfo
    from freetoken.models.gpt_oss.weight import _source_slices

    i, tp = 512, 2
    pieces = _gptoss(torch.Generator().manual_seed(2), i)
    slicing = FORMATS["gptoss"][0](i).expert_slicing()
    sources = {"gate_up": "gate_up_proj_blocks", "gate_up_scale": "gate_up_proj_scales", "gate_up_bias": "gate_up_proj_bias",
               "down": "down_proj_blocks", "down_scale": "down_proj_scales", "down_bias": "down_proj_bias"}
    for rank, view in enumerate(Interval.split(i, tp, align=slicing.alignment)):
        slices = _source_slices(SimpleNamespace(moe_intermediate_size=i), DistributedInfo(rank=rank, size=tp))
        ours = slicing.slice_pieces(pieces, view)
        for role, source in sources.items():
            theirs = pieces[role][slices[source][1]]
            if role == "down_bias" and rank:
                theirs = torch.zeros_like(theirs)  # what its piece reader substitutes off rank 0
            assert torch.equal(ours[role], theirs), (rank, role)


@pytest.mark.parametrize("make_method, alignment", [
    (FORMATS["ds_mxfp4"][0], 256),  # the decode kernel's K tile (and the 128 W4A8 activation block)
    (FORMATS["nvfp4"][0], 16),
    (FORMATS["bf16"][0], 1),
    (FORMATS["gptoss"][0], 32),
    (lambda i: Fp8BlockMoEMethod(_cfg(i)), 128),
])
def test_alignment_rejects_shards_that_split_a_group(make_method, alignment):
    slicing = make_method(512).expert_slicing()
    assert slicing.alignment == alignment
    slicing.check(Interval(alignment, 2 * alignment))
    if alignment > 1:
        with pytest.raises(ValueError, match="aligned"):
            slicing.check(Interval(0, alignment // 2))


def test_unsliceable_formats_say_so():
    ggml = ExpertSlicing(GgmlMoEKernel(), _cfg(512))
    with pytest.raises(ValueError, match="cannot be sliced"):
        ggml.check(Interval(0, 256))
    fp8 = Fp8BlockMoEMethod(_cfg(512)).expert_slicing()
    with pytest.raises(ValueError, match="repack from pieces"):
        fp8.bank_runs(Interval(0, 512), Interval(0, 256))


def _fp8(g, i, device="cpu"):
    """Block-FP8 pieces: e4m3 weights, one fp32 scale per 128x128 block."""
    def weight(*shape):
        return (torch.randn(shape, generator=g) * 0.5).to(torch.float8_e4m3fn)

    def scale(*shape):
        return torch.rand(shape, generator=g) * 0.02 + 0.01

    p = {}
    for proj in ("gate", "up"):
        p[proj], p[f"{proj}_scale"] = weight(E, i, H), scale(E, i // 128, H // 128)
    p["down"], p["down_scale"] = weight(E, H, i), scale(E, H // 128, i // 128)
    return p


@cuda
@pytest.mark.parametrize("fmt", ["nvfp4", "ds_mxfp4", "fp8"])
@pytest.mark.parametrize("is_prefill", [False, True])
def test_kernel_shard_outputs_sum_to_full(fmt, is_prefill):
    """The GPU kernels accept shard-width banks, and a TP-style cover reproduces the unsharded output.

    FP8's padded scale banks cannot be fetched out of a wider bank, so it serves expert TP from banks of
    exactly the GPU's slice (``--moe-host-pool rank``); this is that path."""
    make_method, make_pieces = {**FORMATS, "fp8": (lambda i: Fp8BlockMoEMethod(_cfg(i)), _fp8)}[fmt]
    device = torch.device("cuda")
    i, tokens = 1024, 3
    slicing = make_method(i).expert_slicing()
    g = torch.Generator().manual_seed(3)
    pieces = make_pieces(g, i)  # packed on the host, as the loader does
    x = torch.randn(tokens, H, generator=g).bfloat16().to(device)
    ids = torch.randint(0, E, (tokens, TOPK), generator=g, dtype=torch.int32).to(device)
    weights = torch.rand(tokens, TOPK, generator=g).to(device)

    def run(view):
        banks, resident = _pack(slicing, slicing.slice_pieces(pieces, view), view.width)
        banks = {role: t.to(device) for role, t in banks.items()}
        layer = SimpleNamespace(activation="silu", alpha=1.0, limit=None, apply_router_weight_on_input=False,
                                top_k=TOPK, quant_method=SimpleNamespace(cfg=slicing.config_at(view.width)))
        # a prefill kernel may overwrite its input with the output, and every shard reads the same x
        return slicing.kernel.apply(layer, x.clone(), weights, ids, ExpertView(banks, n=E), is_prefill=is_prefill).float()

    full = run(Interval(0, i))
    total = sum(run(v) for v in Interval.split(i, 4, align=slicing.alignment))
    torch.testing.assert_close(total, full, rtol=2e-2, atol=2e-2 * float(full.abs().max()))


# --- placement: GPU slices, host banks and CPU pools across ranks (moe/placement.py) ---

NODE_CORES = {0: list(range(0, 24, 2)), 1: list(range(1, 24, 2))}


@pytest.mark.parametrize("rank_nodes, mode, banks", [
    ([0, 0, 1, 1], "numa", [((0, 1024), (0, 1)), ((1024, 2048), (2, 3))]),
    ([0, 1, 0, 1], "numa", [((0, 1024), (0, 2)), ((1024, 2048), (1, 3))]),  # ranks interleave over the nodes
    ([0, 0, 0, 1], "numa", [((0, 1536), (0, 1, 2)), ((1536, 2048), (3,))]),
    ([0, 0, 1, 1], "rank", [((0, 512), (0,)), ((512, 1024), (1,)), ((1024, 1536), (2,)), ((1536, 2048), (3,))]),
    ([None, None], "numa", [((0, 2048), (0, 1))]),
])
def test_placement_sums_every_route_once_and_fetches_locally(rank_nodes, mode, banks):
    from freetoken.moe.placement import plan_expert_placement

    cores = NODE_CORES if None not in rank_nodes else {None: list(range(12))}
    p = plan_expert_placement(2048, rank_nodes, mode=mode, alignment=256, node_cores=cores)
    assert [(b.interval.start, b.interval.stop) for b in p.banks] == [iv for iv, _ in banks]
    assert [b.ranks for b in p.banks] == [ranks for _, ranks in banks]
    for rank, node in enumerate(rank_nodes):
        bank = p.bank_of(rank)
        assert bank.node == node and bank.interval.contains(p.gpu[rank])
    # one pool per bank, in its owner's process, on that bank's node only
    assert [(pool.owner, pool.interval) for pool in p.pools] == [(b.owner, b.interval) for b in p.banks]
    for pool in p.pools:
        assert set(pool.cores) <= set(cores[pool.node])
    used = [c for pool in p.pools for c in pool.cores]
    assert len(used) == len(set(used))
    # every unit of I is computed by exactly one GPU slice and exactly one pool
    for slices in (p.gpu, [pool.interval for pool in p.pools]):
        assert sum(iv.width for iv in slices) == 2048 and len({u for iv in slices for u in range(iv.start, iv.stop)}) == 2048


@pytest.mark.parametrize("rank_nodes", [[None, None], [0, None]])
@pytest.mark.parametrize("mode", ["numa", "rank"])
def test_placement_pools_the_whole_host_when_a_gpu_locality_is_unknown(rank_nodes, mode):
    from freetoken.moe.placement import plan_expert_placement

    p = plan_expert_placement(2048, rank_nodes, mode=mode, alignment=256, node_cores=NODE_CORES)
    assert all(bank.node is None for bank in p.banks)  # nothing to bind to
    used = [c for pool in p.pools for c in pool.cores]
    assert used and len(used) == len(set(used)) and set(used) <= set(range(24))


def test_placement_rejects_what_cannot_be_served():
    import dataclasses

    from freetoken.moe.placement import plan_expert_placement

    with pytest.raises(ValueError, match="cannot split"):
        plan_expert_placement(2048, [0, 0, 1], mode="numa", alignment=256)
    with pytest.raises(ValueError, match="too few"):
        plan_expert_placement(2048, [0, 0], mode="rank", node_cores={0: [0, 1, 2]})
    p = plan_expert_placement(2048, [0, 1], mode="numa", alignment=256, node_cores=NODE_CORES)
    swapped = dataclasses.replace(p, gpu=(p.gpu[1], p.gpu[0]))
    with pytest.raises(ValueError, match="outside its bank"):
        swapped.validate(256)
    doubled = dataclasses.replace(p, pools=(p.pools[0], dataclasses.replace(p.pools[1], interval=p.pools[0].interval)))
    with pytest.raises(ValueError, match="exactly once"):
        doubled.validate(256)
    # banks are bound to their GPUs' node: two GPUs of one node put every expert there, in either pool
    from freetoken.moe.shared_banks import check_node_memory

    experts, node_ram = 137 << 30, lambda node: 125 << 30
    for mode in ("rank", "numa"):
        one_node = plan_expert_placement(2048, [0, 0], mode=mode)
        with pytest.raises(ValueError, match="137.0 GiB of expert banks to NUMA node 0"):
            check_node_memory(one_node, [experts // len(one_node.banks)] * len(one_node.banks), node_ram)
        check_node_memory(plan_expert_placement(2048, [0, 1], mode=mode), [experts // 2] * 2, node_ram)


# --- placed host banks: a shard packed into private or NUMA-shared banks (moe/shared_banks.py) ---


def _nvfp4_banks(stored, allocate):
    from freetoken.moe.expert_banks import build_expert_banks

    method = FORMATS["nvfp4"][0](1024)
    pieces = _nvfp4(torch.Generator().manual_seed(5), 1024)
    return build_expert_banks(method, 2, ((l, 0, E, pieces) for l in range(2)), device=torch.device("cpu"),
                              shard=stored, allocate=allocate)


def test_shared_arena_banks_hold_the_shard_and_map_across_processes():
    from freetoken.moe.shared_banks import SharedBankArena, arena_bytes, attach_arena

    method = FORMATS["nvfp4"][0](1024)
    shard = Interval(512, 1024)
    layout = method.expert_slicing().layout_at(shard.width)
    arena = SharedBankArena(arena_bytes(layout, E, 2), node=None)
    banks = _nvfp4_banks(shard, arena.allocate)
    assert banks.stored == shard and banks.layout == layout
    narrow, _ = _pack(method.expert_slicing(), method.expert_slicing().slice_pieces(_nvfp4(torch.Generator().manual_seed(5), 1024), shard), 512)
    mapping, _ = attach_arena(arena.handle())  # what a peer rank maps: the same pages, not a copy
    for role, per_layer in banks.sources.items():
        for t in per_layer:
            assert torch.equal(t.view(torch.uint8), narrow[role].view(torch.uint8)), role
            peer = torch.frombuffer(mapping, dtype=torch.uint8, count=t.nbytes, offset=t.data_ptr() - arena.addr)
            assert torch.equal(peer, t.reshape(-1).view(torch.uint8)), role
    flat = banks.sources["down"][0].view(torch.uint8).reshape(-1)
    flat[0] ^= 0xFF
    assert torch.frombuffer(mapping, dtype=torch.uint8, count=1, offset=flat.data_ptr() - arena.addr)[0] == flat[0]


def _placed_worker(rank, world, store_path, mode, out):
    import torch.distributed as dist

    from freetoken.moe.placement import plan_expert_placement
    from freetoken.moe.shared_banks import load_placed_expert_banks

    torch.cuda.set_device(0)
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=world)
    method = FORMATS["nvfp4"][0](1024)
    placement = plan_expert_placement(1024, [0] * world, mode=mode, alignment=16)
    read = []

    def read_pieces(layers):
        read.extend(layers)
        return ((l, 0, E, _nvfp4(torch.Generator().manual_seed(20 + l), 1024)) for l in layers)

    banks = load_placed_expert_banks(placement, rank, method, read_pieces, num_experts=E, num_layers=4, device=torch.device("cuda"))
    out[rank] = (banks.stored, {r: [t.clone() for t in ts] for r, ts in banks.sources.items()}, read)
    dist.barrier()
    dist.destroy_process_group()


@cuda
@pytest.mark.parametrize("mode", ["numa", "rank"])
def test_placed_banks_read_each_layer_once_into_every_bank(mode, tmp_path):
    import torch.multiprocessing as mp

    with mp.get_context("spawn").Manager() as manager:
        out = manager.dict()
        mp.spawn(_placed_worker, args=(2, str(tmp_path / "store"), mode, out), nprocs=2)
        results = [out[0], out[1]]
    assert sorted(l for _, _, read in results for l in read) == [0, 1, 2, 3]  # the checkpoint is read once in total
    slicing = FORMATS["nvfp4"][0](1024).expert_slicing()
    for rank, (stored, banks, _) in enumerate(results):
        want = Interval(0, 1024) if mode == "numa" else Interval(rank * 512, rank * 512 + 512)  # the node's bank, or the rank's
        assert stored == (None if want.width == 1024 else want)
        for layer in range(4):
            ref, _ = _pack(slicing, slicing.slice_pieces(_nvfp4(torch.Generator().manual_seed(20 + layer), 1024), want), want.width)
            for role, per_layer in banks.items():
                assert torch.equal(per_layer[layer].view(torch.uint8), ref[role].view(torch.uint8)), (rank, layer, role)


# --- the offload cache fetching a GPU shard out of wider host banks (kernel/triton/shard_copy.py) ---


@cuda
@pytest.mark.parametrize("fmt", ["nvfp4", "ds_mxfp4"])
@pytest.mark.parametrize("stored, view", [((0, 1024), (512, 1024)), ((0, 1024), (0, 256)), ((0, 512), (0, 512))])
def test_offload_cache_fetches_the_gpu_shard_out_of_wide_banks(fmt, stored, view):
    from freetoken.moe.expert_banks import build_expert_banks
    from freetoken.moe.offload_cache import OffloadMoeCache

    make_method, make_pieces = FORMATS[fmt]
    device = torch.device("cuda")
    method = make_method(1024)
    slicing = method.expert_slicing()
    stored, view = Interval(*stored), Interval(*view)
    if any(e % slicing.alignment for e in (view.start, view.stop) if e not in (0, 1024)):
        pytest.skip("shard edge inside a quantization group of this format")
    pieces = make_pieces(torch.Generator().manual_seed(6), 1024)
    layers = 2
    wide = build_expert_banks(method, layers, ((l, 0, E, pieces) for l in range(layers)), device=device, shard=stored)
    narrow, _ = _pack(slicing, slicing.slice_pieces(pieces, view), view.width)
    cache = OffloadMoeCache(num_layers=layers, num_experts=E, cache_size=3 * E, device=device, prefill_overlap=True,
                            layout=slicing.layout_at(view.width))
    cache.set_bank_sources(wide.sources, source_runs=slicing.bank_runs(stored, view))
    cache.reset()

    def rows(views, slots, experts):
        for (role, ref), bank in zip(narrow.items(), views):
            got = bank[slots.long()].cpu().view(torch.uint8)
            assert torch.equal(got, ref[experts.long()].view(torch.uint8)), role

    ids = torch.tensor([[3, 1]], dtype=torch.int32, device=device)  # decode misses into the slot pool
    cache.ensure_experts(1, slots := ids.clone())
    cache.copy_missing()
    rows(cache.pool_bank_views(1), slots.flatten(), ids.flatten().cpu())
    cache.materialize_layer(0)  # whole-layer prefill, position == expert id
    cache.copy_missing()
    every = torch.arange(E, dtype=torch.int32)
    rows(cache.stage_views(0, 0), every, every)
    cache.begin_prefill()  # overlapped prefill on the copy stream
    rows(cache.wait_prefill_layer(1), every, every)
    torch.cuda.synchronize()


# --- CPU pools: one pool over a node's wide bank equals the per-rank pools over its slices ---


@cuda
@pytest.mark.parametrize("fmt", ["nvfp4", "ds_mxfp4"])
def test_one_cpu_pool_over_the_wide_bank_sums_the_rank_pools(fmt):
    from freetoken.moe.cpu_executor import CpuMoeExecutor, physical_core_cpus
    from freetoken.moe.expert_banks import build_expert_banks

    make_method, make_pieces = FORMATS[fmt]
    device = torch.device("cuda")
    method = make_method(1024)
    pieces = make_pieces(torch.Generator().manual_seed(7), 1024)
    cores = physical_core_cpus()
    if len(cores) < 12:
        pytest.skip("needs 12 physical cores for three disjoint pools")
    activation = "swiglu_clamp" if fmt == "ds_mxfp4" else "silu"

    def pool(stored, my_cores):
        banks = build_expert_banks(method, 1, iter([(0, 0, E, pieces)]), device=device, shard=stored)
        cache = SimpleNamespace(num_layers=1, num_experts=E, quant_format=banks.quant_format, bank_sources=banks.sources)
        return CpuMoeExecutor(cache, top_k=TOPK, activation=activation, apply_router_weight_on_input=False, num_threads=0,
                              max_tokens=8, device=device, swiglu_limit=10.0 if fmt == "ds_mxfp4" else None,
                              layer_formats=[method.cpu_spec()], core_ids=my_cores)

    g = torch.Generator().manual_seed(8)
    x = torch.randn(3, H, generator=g).bfloat16().to(device)
    ids = torch.randint(0, E, (3, TOPK), generator=g, dtype=torch.int32).to(device)
    w = torch.rand(3, TOPK, generator=g).to(device)
    # the pools must outlive their queued CPU work, as the engine's does
    pools = [pool(Interval(0, 1024), cores[:4]), pool(Interval(0, 512), cores[4:8]), pool(Interval(512, 1024), cores[8:12])]
    wide, *halves = (p.decode(0, x, w, ids.clone()).float() for p in pools)
    torch.cuda.synchronize()
    torch.testing.assert_close(sum(halves), wide, rtol=2e-2, atol=2e-2 * float(wide.abs().max()))


# --- numa hybrid: the rank that owns a node's CPU pool and the rank that shares it split the routes alike ---


@cuda
def test_numa_hybrid_pool_owner_and_sharer_sum_to_the_whole_layer(monkeypatch):
    """Under --moe-host-pool numa hybrid the owner's pool computes the CPU share of the routes across the node's width
    and each rank computes its own GPU share, so the two ranks' outputs sum to the whole layer only while both split
    the routes alike; an extra cache hit on one rank double-counts or drops an expert, which split_check sees."""
    import freetoken.core
    from freetoken.core import Context
    from freetoken.distributed.plan import ExpertGroup
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig
    from freetoken.moe import split_check
    from freetoken.moe.cpu_executor import CpuMoeExecutor, physical_core_cpus
    from freetoken.moe.expert_banks import build_expert_banks
    from freetoken.moe.offload_cache import OffloadMoeCache

    pytest.importorskip("freetoken.kernel._cpu_moe")
    cores = physical_core_cpus()
    if len(cores) < 8:
        pytest.skip("needs 8 physical cores for two disjoint pools")
    i, device = 1024, torch.device("cuda")
    pieces = _bf16(torch.Generator().manual_seed(11), i)
    method = FORMATS["bf16"][0](i)
    wide = build_expert_banks(method, 1, iter([(0, 0, E, pieces)]), device=device, shard=Interval(0, i))

    def pool(my_cores):
        cache = SimpleNamespace(num_layers=1, num_experts=E, quant_format=wide.quant_format, bank_sources=wide.sources)
        return CpuMoeExecutor(cache, top_k=TOPK, activation="silu", apply_router_weight_on_input=False, num_threads=0,
                              max_tokens=8, device=device, layer_formats=[method.cpu_spec()], core_ids=my_cores)

    def rank(shard, executor):
        layer = OffloadMoELayer(0, E, TOPK, H, i, expert_group=ExpertGroup(2, i, shard=shard), quant_config=NoQuantConfig(),
                                strategy="hybrid", decode_target="hybrid", prefix="experts")
        slicing = layer.quant_method.expert_slicing()
        cache = OffloadMoeCache(num_layers=1, num_experts=E, cache_size=E, device=device, decode_target="hybrid",
                                hybrid_max_fetch=1, layout=layer.quant_method.layout())
        cache.set_bank_sources(wide.sources, source_runs=slicing.bank_runs(Interval(0, i), Interval(*shard)))
        cache.reset()
        if executor is None:
            cache.shares_cpu_pool = True
        else:
            cache.set_cpu_executor(executor)
        layer.offload_cache = cache
        return layer

    # the pools must outlive their queued CPU work, as the engine's does
    owner_pool, whole_pool = pool(cores[:4]), pool(cores[4:8])
    owner, sharer = rank((0, i // 2), owner_pool), rank((i // 2, i), None)
    g = torch.Generator().manual_seed(12)
    x = torch.randn(2, H, generator=g).bfloat16().to(device)
    w = (torch.rand(2, TOPK, generator=g) + 0.5).to(device)
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)  # four misses: one fetched, three on the CPU

    seen = []
    monkeypatch.setattr(split_check, "_group", object())
    monkeypatch.setattr(split_check, "_split", True)
    monkeypatch.setattr(split_check, "_agree", lambda value: seen.append(value) or True)
    ctx = Context(page_size=1)
    monkeypatch.setattr(freetoken.core, "_GLOBAL_CTX", ctx)

    def step(layer):
        seen.clear()
        with ctx.forward_batch(SimpleNamespace(is_prefill=False)):
            out = layer.routed_forward(x.clone(), w.clone(), ids.clone(), reduce=False).float()
        return out, list(seen)

    def tol(ref):
        return dict(rtol=2e-2, atol=2e-2 * float(ref.abs().max()))

    whole = whole_pool.decode(0, x, w, ids.clone()).float()
    (a, checked_a), (b, checked_b) = step(owner), step(sharer)
    torch.cuda.synchronize()
    assert checked_a == checked_b  # same routes, same GPU/CPU split
    torch.testing.assert_close(a + b, whole, **tol(whole))

    # the sharer alone fetches expert 3, so next step it has a hit the owner lacks and their splits part
    sharer._decode_routed(x[:1].clone(), w[:1].clone(), torch.tensor([[3, 2]], dtype=torch.int32, device=device))
    (a, checked_a), (b, checked_b) = step(owner), step(sharer)
    torch.cuda.synchronize()
    assert checked_a[0] == checked_b[0] and checked_a[1] != checked_b[1]  # same routes, different split
    assert not torch.allclose(a + b, whole, **tol(whole))


# --- expert tensor parallelism end to end: an offload MoE layer split over two GPU processes ---


def _expert_tp_worker(rank, store_path, is_prefill, out):
    import torch.distributed as dist

    from freetoken.core import Context, set_global_ctx
    from freetoken.distributed import set_tp_info
    from freetoken.distributed.plan import ExpertGroup
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig
    from freetoken.moe.expert_banks import build_expert_banks
    from freetoken.moe.offload_cache import OffloadMoeCache

    i = 512
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{store_path}", rank=rank, world_size=2)
    # the dense layers split too, and the experts must still see their slice as the whole local width
    set_tp_info(rank, 2)
    group = ExpertGroup(2, i, shard=(rank * i // 2, (rank + 1) * i // 2))
    ctx = Context(page_size=1)
    set_global_ctx(ctx)
    layer = OffloadMoELayer(0, E, TOPK, H, i, expert_group=group, quant_config=NoQuantConfig(), prefix="experts")
    method = layer.quant_method
    slicing = method.expert_slicing()
    pieces = _bf16(torch.Generator().manual_seed(9), i)
    banks = build_expert_banks(method, 1, iter([(0, 0, E, pieces)]), device=torch.device("cuda"), shard=Interval(0, i))
    cache = OffloadMoeCache(num_layers=1, num_experts=E, cache_size=2 * E, device=torch.device("cuda"), layout=method.layout())
    cache.set_bank_sources(banks.sources, source_runs=slicing.bank_runs(Interval(0, i), Interval(*layer.expert_shard)))
    cache.reset()
    layer.offload_cache = cache
    g = torch.Generator().manual_seed(10)
    x = torch.randn(3, H, generator=g).bfloat16().cuda()
    ids = torch.randint(0, E, (3, TOPK), generator=g, dtype=torch.int32).cuda()
    w = torch.rand(3, TOPK, generator=g).cuda()
    with ctx.forward_batch(SimpleNamespace(is_prefill=is_prefill)):
        y = layer.routed_forward(x.clone(), w.clone(), ids.clone())
    torch.cuda.synchronize()
    out[rank] = (y.float().cpu(), x.cpu(), ids.cpu(), w.cpu())
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("is_prefill", [False, True])
def test_expert_tp_layer_over_two_gpus_matches_one_gpu(is_prefill, tmp_path):
    import torch.multiprocessing as mp

    with mp.get_context("spawn").Manager() as manager:
        out = manager.dict()
        mp.spawn(_expert_tp_worker, args=(str(tmp_path / "store"), is_prefill, out), nprocs=2)
        (y0, x, ids, w), (y1, *_) = out[0], out[1]
    assert torch.equal(y0, y1)  # the all-reduce leaves every rank with the same output
    pieces = _bf16(torch.Generator().manual_seed(9), 512)
    gate_up, down = pieces["gate_up"].float(), pieces["down"].float()
    ref = torch.zeros(x.shape[0], H)
    for t in range(x.shape[0]):
        for k in range(TOPK):
            e = int(ids[t, k])
            h = gate_up[e] @ x[t].float()
            ref[t] += w[t, k] * (down[e] @ (F.silu(h[:512]) * h[512:]))
    torch.testing.assert_close(y0, ref, rtol=2e-2, atol=2e-2 * float(ref.abs().max()))
