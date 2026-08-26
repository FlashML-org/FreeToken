"""Checkpoint-loading entry points, routed per model architecture.

Each loader resolves the model spec from the checkpoint config and dispatches to the
model module (``models/<name>/weight.py``). Passing ``dummy=True`` replaces the
checkpoint read with randomly filled tensors that keep the loader's exact output
contract (shapes, dtypes, pinning), so everything downstream — repack, offload cache,
kernels — runs unchanged without weights on disk. A model whose banks differ from the
default layout opts out by defining the same-named ``dummy_*`` hook in its weight
module; otherwise the defaults below (built purely from the parsed config) apply.
"""

from __future__ import annotations

import glob
import json
import mmap
import os
import queue
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterator, Tuple

import torch
from freetoken.distributed import get_tp_info
from freetoken.kernel.pinned import alloc_pinned_tensor, copy_to_pinned_tensor
from freetoken.models.loader import stream_moe_expert_sources
from freetoken.utils import cached_load_hf_config, div_even

from .register import _load_attr, get_model_spec

# safetensors header dtype strings -> torch dtypes (for the parallel reader below)
_ST_DTYPE = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2, "F8_E8M0": torch.float8_e8m0fnu,
}
_ODIRECT_BLK = 4096


def _read_shard_odirect_parallel(path: str, workers: int, chunk: int) -> mmap.mmap:
    """Read a whole shard into a page-aligned mmap via CHUNKED multi-threaded O_DIRECT.
    Multi-threading one fd scales even for single-shard checkpoints (measured ~7x at 8
    threads): the kernel issues the parallel preads at high queue depth. DMA bypasses the
    page cache, so there's nothing to drop afterwards."""
    size = os.path.getsize(path)
    asize = ((size + _ODIRECT_BLK - 1) // _ODIRECT_BLK) * _ODIRECT_BLK
    buf = mmap.mmap(-1, asize)
    mv = memoryview(buf)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    offs = list(range(0, size, chunk))

    def rd(o):
        want = min(chunk, asize - o)
        want = min(want, ((size - o + _ODIRECT_BLK - 1) // _ODIRECT_BLK) * _ODIRECT_BLK)
        os.preadv(fd, [mv[o:o + want]], o)

    try:
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        os.close(fd)
    return buf


def iter_expert_tensors_parallel(
    model_path: str,
    is_expert: Callable[[str], bool],
    *,
    workers: int = 8,
    chunk: int = 8 << 20,
    drop_cache: bool = True,
    prefetch: int = 2,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Parallel O_DIRECT analog of a model's serial expert ``iter_weights``.

    Common scaffolding for the "parallel" load path: reads every shard that holds >=1 expert
    tensor (``is_expert(name)`` is the model's per-model predicate) with chunked
    multi-threaded O_DIRECT, parses the safetensors header, and yields ``(name, tensor)``
    in checkpoint dtype/shape for the expert tensors.

    A background reader PREFETCHES the next ``prefetch`` shards (each chunked O_DIRECT)
    while the consumer places the current one, so the disk stays busy during placement
    instead of idling between shards (the gap that made the naive sequential version slow).
    Peak host memory is ~(prefetch+1) shards + the banks the caller fills. Order is
    shard-then-header order (NOT global), so the consumer must place by ``name``.
    """
    from freetoken.utils.hf import download_hf_weight

    model_path = download_hf_weight(model_path)  # resolve hub id -> local (parity w/ serial)
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
    else:  # single-file / no-index checkpoint: map name -> shard from each shard's header
        weight_map = {}
        for shard in sorted(os.path.basename(p) for p in glob.glob(os.path.join(model_path, "*.safetensors"))):
            with open(os.path.join(model_path, shard), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for nm in hdr:
                if nm != "__metadata__":
                    weight_map[nm] = shard
    shards: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if is_expert(name):
            shards.setdefault(shard, []).append(name)
    shard_list = sorted(shards)

    q: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
    _DONE = object()
    err: list[BaseException] = []

    def _reader():
        try:
            for shard in shard_list:
                path = os.path.join(model_path, shard)
                if drop_cache:
                    try:
                        fd = os.open(path, os.O_RDONLY)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        os.close(fd)
                    except OSError:
                        pass
                buf = _read_shard_odirect_parallel(path, workers, chunk)  # overlaps placement
                n = struct.unpack("<Q", bytes(buf[:8]))[0]
                hdr = json.loads(bytes(buf[8:8 + n]))
                q.put((buf, hdr, 8 + n, shards[shard], os.path.getsize(path)))
        except BaseException as e:  # surface reader errors to the consumer
            err.append(e)
        finally:
            q.put(_DONE)

    from freetoken.utils.progress import byte_bar

    th = threading.Thread(target=_reader, name="expert-prefetch", daemon=True)
    th.start()
    bar = byte_bar(sum(os.path.getsize(os.path.join(model_path, s)) for s in shard_list),
                   "Loading experts (parallel)")
    try:
        while True:
            item = q.get()
            if item is _DONE:
                break
            buf, hdr, base, names, shard_sz = item
            mv = memoryview(buf)
            for name in names:
                meta = hdr[name]
                b, e = meta["data_offsets"]
                t = torch.frombuffer(mv[base + b: base + e], dtype=_ST_DTYPE[meta["dtype"]])
                yield name, (t.view(*meta["shape"]) if meta["shape"] else t)
            bar.update(shard_sz)
            del mv, buf  # freed once the consumer drops the last yielded tensor of this shard
    finally:
        bar.close()
        th.join()
    if err:
        raise err[0]


_SCATTERED_AVG_BYTES = 16 << 20  # avg expert tensor below this -> "scattered" -> prefer parallel


def experts_scattered(model_path: str) -> bool:
    """Slow-path strategy signal: are the experts stored as many SMALL tensors?

    If yes (per-expert / quantized layouts -> avg expert tensor a few MiB), the serial
    baseline pays per-tensor overhead on thousands of tiny reads and is slow, so the parallel
    parallel whole-shard O_DIRECT reader wins. If experts are pre-packed into a few large
    tensors, the serial read already saturates the disk and parallel only adds read amplification.
    Measured cheaply from the safetensors headers (no tensor data read). This is a best-
    effort heuristic: ANY failure (unresolvable path, no safetensors, GGUF, unreadable
    header) -> False (serial), so the real loader still runs and reports real errors."""
    try:
        from freetoken.utils.hf import download_hf_weight

        model_path = download_hf_weight(model_path)  # resolve hub ids -> local (parity w/ serial)
        index = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                shards = sorted(set(json.load(f)["weight_map"].values()))
        else:
            shards = sorted(os.path.basename(p) for p in glob.glob(os.path.join(model_path, "*.safetensors")))
        sizes: list[int] = []
        for shard in shards:
            try:
                with open(os.path.join(model_path, shard), "rb") as fh:
                    n = struct.unpack("<Q", fh.read(8))[0]
                    hdr = json.loads(fh.read(n))
            except (OSError, ValueError, struct.error):  # unreadable/partial shard -> skip
                continue
            for name, meta in hdr.items():
                if name != "__metadata__" and ".experts." in name:
                    b, e = meta["data_offsets"]
                    sizes.append(e - b)
        if not sizes:
            return False
        return (sum(sizes) / len(sizes)) < _SCATTERED_AVG_BYTES
    except Exception:  # heuristic only -> default to serial; the real loader reports errors
        return False


def _spec_for_model_path(model_path: str):
    hf_config = cached_load_hf_config(model_path)
    spec = get_model_spec(hf_config.architectures[0])
    parse_config = _load_attr(spec.module, spec.parse_config)
    return parse_config(hf_config), spec


def _model_override(spec, name: str):
    """The model module's same-named hook, if it defines one."""
    try:
        return _load_attr(spec.module, name)
    except AttributeError:
        return None


def load_weight(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool = True,
) -> Iterator[Tuple[str, torch.Tensor]]:
    # FTW checkpoint: dense weights are stored post-iter_weights, so we replay them
    # model-agnostically instead of re-running the per-model reader. Which tensors exist is
    # decided at conversion (offload -> experts live in banks, not here); a backend mismatch
    # fails loudly in load_state_dict (strict missing/unexpected expert keys), so the reader
    # just yields the stored weight tensors regardless of the include_moe_experts flag.
    from freetoken.checkpoint.ftw import is_ftw_checkpoint, iter_ftw_weights
    from freetoken.models.config import VISION_KEY_PREFIXES, vision_load_enabled

    if is_ftw_checkpoint(model_path):
        # The FTW dense shard stores whatever existed at conversion, including the vision
        # stack. Vision is opt-in (default OFF, see vision_load_enabled): when it is off the
        # model never builds the tower, so replaying those tensors would trip load_state_dict's
        # strict unexpected-key check. Skip them here to match the model the engine built.
        skip_vision = not vision_load_enabled()
        for name, tensor in iter_ftw_weights(model_path):
            if skip_vision and name.startswith(VISION_KEY_PREFIXES):
                continue
            yield name, tensor
        return

    _config, spec = _spec_for_model_path(model_path)
    iter_weights = _load_attr(spec.module, spec.iter_weights)
    yield from iter_weights(
        model_path,
        device,
        include_moe_experts=include_moe_experts,
        include_non_moe=True,
    )


def load_moe_expert_sources(
    model_path: str,
    *,
    dtype: torch.dtype,
    dummy: bool = False,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    config, spec = _spec_for_model_path(model_path)
    if not config.is_moe:
        raise ValueError(
            f"{config.architectures[0]} does not provide MoE expert source loading"
        )
    if dummy:
        builder = _model_override(spec, "dummy_moe_expert_sources") or dummy_moe_expert_sources
        return builder(config, dtype=dtype)
    if parallel:  # parallel: experts read via the common chunked multi-threaded O_DIRECT reader
        iter_weights = _model_override(spec, "iter_weights_parallel")
        if iter_weights is None:  # model has no parallel reader -> let the caller fall back to serial
            raise NotImplementedError(
                f"{spec.module} provides no iter_weights_parallel")
        src = iter_weights(model_path, torch.device("cpu"), include_moe_experts=True,
                           include_non_moe=False, workers=workers, chunk=chunk)
    else:
        iter_weights = _load_attr(spec.module, spec.iter_weights)
        src = iter_weights(model_path, torch.device("cpu"), include_moe_experts=True,
                           include_non_moe=False)
    return stream_moe_expert_sources(
        src,
        config,
        dtype=dtype,
        layer_sink=layer_sink,
    )


def load_nvfp4_moe_expert_sources(
    model_path: str,
    model_config,
    *,
    dummy: bool = False,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    layer_sink=None,
) -> dict:
    """Load (or fabricate, with ``dummy=True``) packed NVFP4 expert source banks.
    ``parallel=True`` uses the model's ``load_nvfp4_expert_sources_parallel`` (common
    chunked multi-threaded O_DIRECT reader). ``layer_sink``: see
    ``models.nvfp4_banks.load_nvfp4_expert_source_banks``; forwarded to the per-model
    loader, which forwards it on."""
    _config, spec = _spec_for_model_path(model_path)
    if dummy:
        builder = (
            _model_override(spec, "dummy_nvfp4_expert_sources") or dummy_nvfp4_expert_sources
        )
        return builder(model_config)
    if parallel:
        loader = _model_override(spec, "load_nvfp4_expert_sources_parallel")
        if loader is None:  # no parallel reader -> let the caller fall back to serial
            raise NotImplementedError(
                f"{spec.module} provides no load_nvfp4_expert_sources_parallel")
        return loader(model_path, model_config, workers=workers, chunk=chunk, layer_sink=layer_sink)
    loader = _load_attr(spec.module, "load_nvfp4_expert_sources")
    return loader(model_path, model_config, layer_sink=layer_sink)


def load_q4_0_moe_expert_sources(
    model_path: str,
    model_config,
    *,
    dummy: bool = False,
    layer_sink=None,
) -> dict:
    """Load (or fabricate, with ``dummy=True``) packed GGUF Q4_0 expert source banks.
    ``layer_sink`` (converter) streams each completed layer's banks; ignored for dummy."""
    _config, spec = _spec_for_model_path(model_path)
    if dummy:
        builder = _model_override(spec, "dummy_q4_0_expert_sources")
        assert builder is not None, "model defines no dummy_q4_0_expert_sources"
        return builder(model_config)
    loader = _load_attr(spec.module, "load_q4_0_expert_sources")
    return loader(model_path, model_config, layer_sink=layer_sink)


def load_gguf_moe_expert_sources(
    model_path: str,
    model_config,
    *,
    dummy: bool = False,
    layer_sink=None,
) -> dict:
    """Load (or fabricate) mixed-type GGUF expert banks (flat padded uint8 slots).

    Per-model: dispatches to the model module's ``load_gguf_expert_sources`` /
    ``dummy_gguf_expert_sources`` (laguna is the first user)."""
    _config, spec = _spec_for_model_path(model_path)
    if dummy:
        builder = _model_override(spec, "dummy_gguf_expert_sources")
        assert builder is not None, "model defines no dummy_gguf_expert_sources"
        return builder(model_config)
    loader = _load_attr(spec.module, "load_gguf_expert_sources")
    return loader(model_path, model_config, layer_sink=layer_sink)


def _num_moe_layers(config) -> int:
    value = getattr(config, "num_moe_layers", None)
    if value is not None:
        return int(value)
    return int(config.num_layers) - int(getattr(config, "first_k_dense_replace", 0))


def dummy_moe_expert_sources(
    config, *, dtype: torch.dtype
) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Random BF16 expert banks shaped like ``stream_moe_expert_sources`` output:
    one independently allocated ``[num_experts, ...]`` tensor per layer."""
    num_layers = _num_moe_layers(config)
    intermediate_size = div_even(config.moe_intermediate_size, get_tp_info().size)
    gate_up = [
        torch.randn(config.num_experts, 2 * intermediate_size, config.hidden_size, dtype=dtype)
        for _ in range(num_layers)
    ]
    down = [
        torch.randn(config.num_experts, config.hidden_size, intermediate_size, dtype=dtype)
        for _ in range(num_layers)
    ]
    if torch.cuda.is_available():
        gate_up = [copy_to_pinned_tensor(t) for t in gate_up]
        down = [copy_to_pinned_tensor(t) for t in down]
    return gate_up, down


def dummy_nvfp4_expert_sources(config) -> dict[str, list[torch.Tensor]]:
    """Random NVFP4 (ModelOpt-layout) expert banks shaped like the real loader's.

    Same pinned allocation and per-layer bank shapes as ``load_nvfp4_expert_sources``,
    so the repack/offload path downstream is exercised unchanged. Packed codes are
    random nibbles; block scales are 1.0 and globals small because random e4m3 bytes
    reach 448 (and include NaN encodings), which would blow up the dummy activations.
    """
    num_layers = _num_moe_layers(config)
    E = config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    fp8 = torch.float8_e4m3fn

    def bank(*shape: int, dtype: torch.dtype) -> list[torch.Tensor]:
        return [alloc_pinned_tensor(*shape, dtype=dtype) for _ in range(num_layers)]

    sources = {
        "gate_up_packed": bank(E, 2 * I, H // 2, dtype=torch.uint8),
        "gate_up_scale": bank(E, 2 * I, H // 16, dtype=fp8),
        "gate_up_global": bank(E, 2 * I, dtype=torch.float16),
        "down_packed": bank(E, H, I // 2, dtype=torch.uint8),
        "down_scale": bank(E, H, I // 16, dtype=fp8),
        "down_global": bank(E, H, dtype=torch.float16),
    }
    for t in sources["gate_up_packed"] + sources["down_packed"]:
        t.random_(0, 256)
    for t in sources["gate_up_scale"] + sources["down_scale"]:
        t.fill_(1.0)
    for t in sources["gate_up_global"] + sources["down_global"]:
        t.fill_(0.01)
    return sources


__all__ = [
    "load_weight",
    "load_moe_expert_sources",
    "load_gguf_moe_expert_sources",
    "load_nvfp4_moe_expert_sources",
    "dummy_moe_expert_sources",
    "dummy_nvfp4_expert_sources",
    "iter_expert_tensors_parallel",
]
