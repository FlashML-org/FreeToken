"""Routed experts split across this host's GPUs: ``--tp N`` with an expert group in the model's parallel plan.

Model-independent. Given the plan's expert group and the expert kernel the MoE layers bind, ``SplitExperts.place``
places each rank's slice of the intermediate dim, the host banks and the CPU pools (``moe/placement.py``); the object
then answers what the engine needs from a split-expert rank: the banks it loads and pins, where its GPU slice lies in
them, whether its node's ranks share a CPU pool, and whether this rank runs that pool.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from freetoken.layers.quantization import QuantKind
from freetoken.moe import is_offload_moe_strategy, split_check
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.distributed.plan import ExpertGroup
    from freetoken.engine.config import EngineConfig
    from freetoken.layers.quantization.moe.shard import Interval
    from freetoken.moe.placement import ExpertPlacement

logger = init_logger(__name__)


class SplitExperts:
    def __init__(self, placement: ExpertPlacement, rank: int):
        self.placement = placement
        self.rank = rank

    @classmethod
    def place(cls, config: EngineConfig, group: ExpertGroup, method, layer_methods, *, device, cpu_group) -> SplitExperts:
        """Place ``group`` over the TP ranks (a collective over ``cpu_group``). ``method`` is the offload expert method the
        layers bind at a provisional whole-width shard, ``layer_methods`` the per-layer ones when they differ."""
        from freetoken.moe.expert_banks import is_mixed
        from freetoken.moe.placement import gpu_numa_node, node_physical_cores, plan_expert_placement

        model_config = config.model_config
        if group.size != config.tp_info.size:
            raise ValueError(f"an expert group of {group.size} ranks in a world of {config.tp_info.size} is not served")
        if not is_offload_moe_strategy(config.moe_strategy):
            raise ValueError("--tp splits the routed experts across GPUs only when they are offloaded; use --moe-strategy offload, hybrid or cpu")
        if config.moe_prefill_hit_d2d:
            raise ValueError("routed experts split across GPUs do not combine with --moe-prefill-hit-d2d")
        if method is None or is_mixed(layer_methods) or method.kind is QuantKind.GGUF:
            raise ValueError("routed experts split across GPUs must all be in one non-GGUF format")
        slicing = method.expert_slicing()
        records: list = [None] * group.size
        gpu = (torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device))
        torch.distributed.all_gather_object(
            records, (os.uname().nodename, gpu_numa_node(device.index), node_physical_cores(), method.kernel.name, gpu),
            group=cpu_group,
        )
        if len({host for host, *_ in records}) != 1:
            raise ValueError("GPUs that split the routed experts share host expert banks, so they must all be on one host")
        # every rank routes the tokens, and with numa pools splits them between GPU and CPU, for itself: the ranks agree
        # only while their GPUs compute alike (FREETOKEN_CHECK_SPLIT_EXPERTS=1 checks it)
        gpus = [g for *_, g in records]
        if len(set(gpus)) != 1:
            raise ValueError(f"GPUs that split the routed experts must be one model, but they are {[name for name, _ in gpus]} (by rank)")
        # each GPU picks its expert kernel for itself (W4A4 NVFP4 only where the GPU runs it), but the shared banks, the
        # placement's alignment and the bank loader's collectives follow one kernel
        kernels = [kernel for _, _, _, kernel, _ in records]
        if len(set(kernels)) != 1:
            raise ValueError(
                f"GPUs that split the routed experts share host banks packed for one {method.kind} expert kernel, but they picked "
                f"{kernels} (by rank); pass --quant-backend moe.{method.kind.value}=<name> with a kernel every GPU runs"
            )
        cpu = model_config.decode_target in ("cpu", "hybrid")
        nodes, node_cores = [node for _, node, *_ in records], records[0][2]
        mode = config.moe_host_pool
        if mode == "numa" and slicing.bank_axes is None:
            logger.warning_rank0(
                f"--moe-host-pool numa: the {method.kernel.name} {method.kind} expert kernel's banks are repacked or padded, so a GPU "
                "cannot fetch its slice out of a node's shared bank; each GPU keeps its own bank and CPU pool (--moe-host-pool rank)"
            )
            mode = "rank"
        if cpu and None in nodes and set(node_cores) != {None}:
            logger.warning_rank0(
                f"splitting the routed experts: the NUMA node of GPU(s) {[r for r, n in enumerate(nodes) if n is None]} is unknown; the expert "
                "banks stay unbound and the CPU pools share all of the host's cores"
            )
        placement = plan_expert_placement(
            group.intermediate, nodes, mode=mode, alignment=slicing.alignment,
            node_cores=node_cores if cpu else None, threads=config.moe_cpu_threads,
        )
        logger.info_rank0(placement.describe())
        if split_check.ENABLED:
            split_check.enable(cpu_group, hybrid_split=cpu and mode == "numa")
            object.__setattr__(config, "cuda_graph_bs", [])  # the check syncs the host at every MoE layer
            logger.info_rank0("FREETOKEN_CHECK_SPLIT_EXPERTS=1: comparing every MoE layer's routes across ranks, without CUDA graphs")
        return cls(placement, config.tp_info.rank)

    @property
    def shard(self) -> Interval:
        """This rank's slice of the intermediate dim."""
        return self.placement.gpu[self.rank]

    @property
    def shares_cpu_pools(self) -> bool:
        """Some rank computes its CPU share of the routes in another rank's pool (--moe-host-pool numa), so the ranks on a
        node must split the routes between GPU and CPU alike."""
        return len(self.placement.pools) < self.placement.world_size

    def load_banks(self, config: EngineConfig, method, device):
        """This rank's host banks: its own slice, or its NUMA node's shared one.

        Every rank reads its share of the MoE layers once; either reader reads just those layers' bytes. --expert-load
        picks the reader as it does on one GPU, ``auto`` falling back to serial when free RAM cannot also hold one
        whole-shard buffer of the parallel reader per rank."""
        from freetoken.moe.expert_banks import _PARALLEL_READER_SUPPORTED, _host_ram_fits_parallel
        from freetoken.moe.expert_pieces import iter_expert_pieces
        from freetoken.moe.shared_banks import load_placed_expert_banks

        parallel = {"serial": False, "parallel": True}.get(config.expert_load)
        if parallel is None:
            parallel = _PARALLEL_READER_SUPPORTED
            if parallel and not config.use_dummy_weight and not _host_ram_fits_parallel(config.model_path, readers=config.tp_info.size):
                logger.warning_rank0("expert banks: low free RAM -> serial read (avoids parallel-reader OOM; override with --expert-load parallel)")
                parallel = False
        elif parallel and not _PARALLEL_READER_SUPPORTED:
            raise ValueError("--expert-load parallel: this platform has no O_DIRECT/preadv; use --expert-load serial")
        logger.info_rank0(f"expert banks: {'parallel' if parallel else 'serial'} read, each rank its share of the MoE layers")

        def read(layers):
            return iter_expert_pieces(config.model_path, config.model_config, method.kind, layers=layers, parallel=parallel)

        return load_placed_expert_banks(
            self.placement, self.rank, method, None if config.use_dummy_weight else read,
            num_experts=config.model_config.num_experts, num_layers=config.model_config.num_moe_layers, device=device,
        )

    def pinned_bank_bytes(self, config: EngineConfig, method) -> int:
        """Host bytes of the bank this rank pins: the whole bank it fetches from, its NUMA node's under --moe-host-pool
        numa, which is wider than the GPU slice the layers' layout describes."""
        from freetoken.moe.shared_banks import arena_bytes

        bank = self.placement.bank_of(self.rank)
        layout = method.expert_slicing().layout_at(bank.interval.width)
        return arena_bytes(layout, config.model_config.num_experts, config.model_config.num_moe_layers)

    def pin_hint(self) -> str:
        """What frees pin budget (split experts take no OS-locked layers)."""
        placement = self.placement
        if any(bank.interval.width > min(placement.gpu[r].width for r in bank.ranks) for bank in placement.banks):
            return "pass --moe-host-pool rank, so each GPU pins only its own slice of the experts, or split them over more GPUs"
        return "split the routed experts over more GPUs, each pinning a narrower slice"

    def slot_rows(self, config: EngineConfig, method, banks):
        """``(source_runs, slot_sources)``: where this rank's GPU slice lies in rows of wider host banks (None when the
        banks hold exactly it), and per-layer tensors shaped like the slot rows for the cache budget."""
        from freetoken.layers.quantization.moe.shard import Interval

        slicing = method.expert_slicing()
        stored = banks.stored or Interval(0, slicing.intermediate)
        if stored == self.shard:
            return None, banks.sources
        L, E = config.model_config.num_moe_layers, config.model_config.num_experts
        rows = {
            role: [torch.empty((E, *spec.shape), dtype=spec.dtype, device="meta")] * L
            for role, spec in method.layout().items() if not spec.resident
        }
        return slicing.bank_runs(stored, self.shard), rows

    def attach_cpu_pool(self, cache) -> tuple[int, ...] | None:
        """The cores of the CPU expert pool this rank runs, or None when its node's pool runs in another rank: then
        ``cache`` leaves the CPU share of the routes to that pool, and this process leaves the pool's cores to it as the
        owner does (its torch threads)."""
        from freetoken.moe.cpu_executor import leave_cores_to_pool

        if not self.placement.pools:
            raise ValueError("routed experts split across GPUs get CPU expert pools for --moe-strategy hybrid or cpu only; drop --moe-cpu-layers")
        pool = self.placement.pool_owned_by(self.rank)
        if pool is not None:
            return pool.cores
        cache.shares_cpu_pool = True
        leave_cores_to_pool(c for p in self.placement.pools for c in p.cores)
        return None


__all__ = ["SplitExperts"]
