"""ft info: forecast a model's GPU memory under a serve configuration without loading its weights."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from freetoken.engine.forecast import CATEGORY_LABELS

GiB = 1 << 30

USAGE = """usage: {prog} <model> [ft serve flags] [--json] [--gpu-memory-gib N | --gpu-free-gib N] [--verbose]

Forecast what `ft serve <model> [flags]` puts on the GPU -- weights by category, KV cache,
fixed pools, workspaces -- and whether it fits, without reading any weight. Accepts every
`ft serve` flag, so "what if" combinations can be tried before a long load.

Options of ft info itself:
  --json               print the forecast as JSON
  --gpu-memory-gib N   plan for an empty card with N GiB of memory instead of the local GPU
  --gpu-free-gib N     plan for N GiB free (as nvidia-smi reports it) instead of the local GPU
  --verbose            keep the engine's INFO logs

The `ft serve` flags follow:"""


def _info_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpu-memory-gib", type=float, default=None)
    group.add_argument("--gpu-free-gib", type=float, default=None)
    return parser


def split_argv(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    """(ft info's own options, the argv for the serve parser with the model as --model)."""
    opts, rest = _info_parser().parse_known_args(list(argv))
    if rest and not rest[0].startswith("-"):
        rest = ["--model", rest[0], *rest[1:]]
    return opts, rest


def checkpoint_tensors(model_path: str) -> tuple[dict[str, tuple[str, int]], str] | None:
    """Tensor name -> (dtype, bytes) from the safetensors headers alone, and where they came from:
    a local directory or the HF cache, else the Hub through HTTP range requests (headers only)."""
    from freetoken.models.loader import read_safetensors_header

    folder = model_path if os.path.isdir(model_path) else None
    if folder is None:
        try:
            from huggingface_hub import snapshot_download

            folder = snapshot_download(model_path, local_files_only=True)
        except Exception:  # noqa: BLE001 -- not cached: fall back to the Hub below
            folder = None
    if folder is not None:
        index = os.path.join(folder, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index, encoding="utf-8") as f:
                shards = [os.path.join(folder, s) for s in sorted(set(json.load(f)["weight_map"].values()))]
        else:
            shards = [p for p in sorted(glob.glob(os.path.join(folder, "*.safetensors")))
                      if not p.endswith("consolidated.safetensors")]
        if shards and all(os.path.isfile(s) for s in shards):
            out: dict[str, tuple[str, int]] = {}
            for shard in shards:
                for name, meta in read_safetensors_header(shard).items():
                    if name != "__metadata__":
                        begin, end = meta["data_offsets"]
                        out[name] = (meta["dtype"], end - begin)
            return out, "local safetensors headers"
    if os.path.isdir(model_path):
        return None
    try:
        from huggingface_hub import get_safetensors_metadata

        meta = get_safetensors_metadata(model_path)
    except Exception:  # noqa: BLE001 -- offline or not a safetensors repo: no checkpoint view
        return None
    out = {}
    for file_meta in meta.files_metadata.values():
        for name, info in file_meta.tensors.items():
            begin, end = info.data_offsets
            out[name] = (info.dtype, end - begin)
    return out, "Hub safetensors headers (HTTP range requests)"


@dataclass
class GpuInfo:
    free_before: int | None
    source: str
    name: str | None = None
    total: int | None = None
    free_now: int | None = None


def gpu_info(opts: argparse.Namespace, config) -> GpuInfo:
    """Free memory the engine would measure before the load: the card's free memory minus the
    CUDA context the engine process creates first. NVML only -- no CUDA context here."""
    from freetoken.engine.forecast import CUDA_CONTEXT_BYTES
    from freetoken.gpu_select import nvml_memory_info

    if opts.gpu_memory_gib is not None:
        total = int(opts.gpu_memory_gib * GiB)
        return GpuInfo(total - CUDA_CONTEXT_BYTES, f"--gpu-memory-gib {opts.gpu_memory_gib:g} (empty card)",
                       total=total, free_now=total)
    if opts.gpu_free_gib is not None:
        free = int(opts.gpu_free_gib * GiB)
        return GpuInfo(free - CUDA_CONTEXT_BYTES, f"--gpu-free-gib {opts.gpu_free_gib:g}", free_now=free)
    spec = config.gpu[0] if config.gpu else None
    if spec is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            entries = [e.strip() for e in visible.split(",") if e.strip()]
            if not entries:
                return GpuInfo(None, "no GPU visible (CUDA_VISIBLE_DEVICES is empty)")
            spec = entries[0]
    info = nvml_memory_info(spec or "0")
    if info is None:
        return GpuInfo(None, "no GPU found (NVML unavailable)")
    name, total, free = info
    return GpuInfo(free - CUDA_CONTEXT_BYTES, "NVML", name=name, total=total, free_now=free)


@dataclass
class InfoReport:
    config: Any
    gpu: GpuInfo
    inputs: Any
    forecast: Any
    tips: list
    combo: Any
    checkpoint: dict[str, int] | None = None
    checkpoint_dtypes: dict[str, int] | None = None
    checkpoint_source: str | None = None
    quant: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _quant_summary(report: "InfoReport") -> dict[str, str]:
    return {CATEGORY_LABELS.get(cat, cat): ", ".join(kinds) for cat, kinds in report.quant.items()}


def _layout_summary(mc) -> str:
    kinds = sorted({spec.attn_type.value for spec in mc.kv_cache_group_specs() if spec.num_layers > 0})
    if mc.linear_attention_group() is not None:
        kinds.append(f"linear ({mc.linear_attention_group().variant}, {len(mc.linear_attention_group().layer_ids)} layers)")
    text = f"{mc.num_layers} layers, hidden {mc.hidden_size}, attention: {', '.join(kinds)}"
    if getattr(mc, "is_moe", False):
        text += f", MoE {mc.num_experts} experts top-{mc.num_experts_per_tok} on {mc.num_moe_layers} layers"
    return text


def analyze(config, opts: argparse.Namespace) -> InfoReport:
    """Resolve ``config`` as Engine.__init__ does, build the model on the meta device and
    forecast; reads only config files and safetensors headers."""
    import torch

    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.engine.engine import _adjust_config, _adjust_ftw_quant_backend
    from freetoken.engine.forecast import forecast_inputs, quant_by_category, suggest_tips, tensor_category
    from freetoken.layers import set_rope_device
    from freetoken.layers.quantization import QuantBackend, set_quant_backend
    from freetoken.models import create_model
    from freetoken.utils import torch_dtype

    tp = try_get_tp_info()
    if tp is None:
        set_tp_info(rank=0, size=config.tp_info.size)
    elif tp.size != config.tp_info.size:
        raise RuntimeError(f"TP size {tp.size} is already set in this process, not {config.tp_info.size}")
    set_quant_backend(_adjust_ftw_quant_backend(config.model_path, QuantBackend.parse(config.quant_backend)))
    _adjust_config(config)
    set_rope_device(torch.device("cpu"))
    gpu = gpu_info(opts, config)
    found = checkpoint_tensors(config.model_path)
    try:
        with torch.device("meta"), torch_dtype(config.dtype):
            model = create_model(config.model_config)
    except Exception as exc:  # noqa: BLE001 -- fall back to the checkpoint headers, flagged as such
        if found is None:
            raise
        model, meta_error = None, exc
    if model is not None:
        inputs = forecast_inputs(config, model, gpu.free_before)
        quant = quant_by_category(model)
    else:
        inputs = header_inputs(config, found[0], gpu.free_before)
        inputs.weights.notes.append(
            f"The model could not be built on the meta device ({type(meta_error).__name__}: {meta_error}); "
            "GPU weights are the checkpoint's stored bytes, so re-quantized or upcast tensors are not priced."
        )
        quant = {}
    fc = inputs.forecast()
    tips, combo = suggest_tips(inputs, fc)
    report = InfoReport(config, gpu, inputs, fc, tips, combo, quant=quant)
    if found is not None:
        tensors, report.checkpoint_source = found
        by_cat: Counter[str] = Counter()
        by_dtype: Counter[str] = Counter()
        for name, (dtype, nbytes) in tensors.items():
            by_cat[tensor_category(name)] += nbytes
            by_dtype[dtype] += nbytes
        report.checkpoint, report.checkpoint_dtypes = dict(by_cat), dict(by_dtype)
    report.notes = inputs.weights.notes + [
        "Kernels that repack weights in finalize_quant are assumed to keep the byte count.",
        "Load overhead, prefill activations and CUDA graph memory are estimates; the verdict only "
        "refuses when the engine's own sizing fails without them.",
    ]
    if gpu.free_before is not None and gpu.source == "NVML":
        report.notes.append("Free memory is read now; other processes on the GPU change it.")
    return report


def header_inputs(config, tensors: dict[str, tuple[str, int]], free_before: int | None):
    """Forecast inputs from the safetensors headers alone, for a family the meta build cannot run:
    the tensors the engine would not put on the GPU (an inactive vision tower, MTP heads,
    offloaded experts) are dropped by category."""
    from freetoken.engine.forecast import ForecastInputs, WeightReport, _is_offload, tensor_category

    mc = config.model_config
    gpu: Counter[str] = Counter()
    host: Counter[str] = Counter()
    offload = _is_offload(config)
    for name, (_, nbytes) in tensors.items():
        cat = tensor_category(name)
        if cat == "mtp" or (cat == "vision" and not config.active_encoders):
            continue
        (host if cat == "experts" and offload else gpu)[cat] += nbytes
    total_experts = getattr(mc, "num_moe_layers", 0) * getattr(mc, "num_experts", 0)
    per_expert = host["experts"] // total_experts if offload and total_experts else 0
    item = config.dtype.itemsize
    return ForecastInputs(
        config=config, weights=WeightReport(gpu=dict(gpu), host=dict(host)), free_before=free_before,
        per_expert_bytes=per_expert,
        activation_per_token=item * (4 * mc.hidden_size + 3 * mc.intermediate_size),
    )


def report_dict(r: InfoReport) -> dict:
    config, mc = r.config, r.config.model_config
    fc = r.forecast
    return {
        "model": config.model_path,
        "architecture": config.hf_config.architectures[0],
        "model_type": mc.model_type,
        "layout": _layout_summary(mc),
        "quantization": _quant_summary(r),
        "resolved": {
            "attention_backend": config.attention_backend,
            "cache_type": config.cache_type,
            "page_size": config.page_size,
            "moe_strategy": config.moe_strategy if getattr(mc, "is_moe", False) else None,
            "max_running_req": config.max_running_req,
            "memory_ratio": config.memory_ratio,
            "max_prefill_length": getattr(config, "max_extend_tokens", None),
            "max_seq_len": config.max_seq_len,
            "dtype": str(config.dtype).removeprefix("torch."),
        },
        "gpu": vars(r.gpu),
        "checkpoint": None if r.checkpoint is None else {
            "source": r.checkpoint_source, "bytes_by_category": r.checkpoint, "bytes_by_dtype": r.checkpoint_dtypes,
        },
        "weights": {"gpu": r.inputs.weights.gpu, "host": r.inputs.weights.host, "load_overhead": fc.load_overhead},
        "forecast": fc.to_dict(),
        "tips": [t.to_dict() for t in r.tips],
        "suggested": r.combo.to_dict() if r.combo else None,
        "notes": r.notes,
    }


def format_report(r: InfoReport) -> str:
    from freetoken.engine.forecast import _gib, format_forecast

    config, mc = r.config, r.config.model_config
    quant = "; ".join(f"{k}: {v}" for k, v in _quant_summary(r).items())
    lines = [
        f"Model         {config.model_path}",
        f"Architecture  {config.hf_config.architectures[0]} ({mc.model_type}): {_layout_summary(mc)}",
        f"Quantization  {quant}",
    ]
    if r.checkpoint_dtypes:
        dtypes = ", ".join(f"{k} {_gib(v)}" for k, v in sorted(r.checkpoint_dtypes.items(), key=lambda kv: -kv[1]))
        lines.append(f"Checkpoint    {dtypes} ({r.checkpoint_source})")
    resolved = (f"attention {config.attention_backend}, cache {config.cache_type}, page size {config.page_size}, "
                f"max_running_req {config.max_running_req}, memory_ratio {config.memory_ratio:g}, "
                f"max prefill {getattr(config, 'max_extend_tokens', '-')}, max_seq_len {config.max_seq_len}")
    if getattr(mc, "is_moe", False):
        resolved = f"moe {config.moe_strategy}, " + resolved
    lines.append(f"Resolved      {resolved}")
    g = r.gpu
    if g.free_before is None:
        lines.append(f"GPU           {g.source}")
    else:
        card = f"{g.name}: " if g.name else ""
        total = f"{_gib(g.total)} total, " if g.total else ""
        lines.append(f"GPU           {card}{total}{_gib(g.free_now)} free ({g.source}) -> "
                     f"{_gib(g.free_before)} free before loading (after the CUDA context)")
    lines.append("")
    lines.append(format_forecast(r.inputs, r.forecast, r.tips, r.combo, r.checkpoint))
    lines.append("")
    lines.extend(f"Note: {n}" for n in r.notes)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, prog: str = "ft info") -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE.format(prog=prog), file=sys.stdout if argv else sys.stderr)
        if not argv:
            return 2
        from freetoken.server.args import parse_args

        try:
            parse_args(["--help"], prog=prog)
        except SystemExit:
            pass
        return 0
    opts, serve_argv = split_argv(argv)
    if not opts.verbose:
        os.environ.setdefault("LOG_LEVEL", "WARNING")
    from freetoken.server.args import parse_args

    config, _ = parse_args(serve_argv, prog=prog)
    try:
        report = analyze(config, opts)
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"{prog}: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if opts.json:
        print(json.dumps(report_dict(report), indent=2, default=str))
    else:
        print(format_report(report))
    return 1 if report.forecast.verdict == "does not fit" else 0
