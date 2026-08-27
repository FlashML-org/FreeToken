"""Convert an HF safetensors checkpoint into a FreeToken Weight (FTW) checkpoint.

Model-agnostic: it drives the *existing* per-model loaders once and stores their output, so
no per-model conversion code is needed.

* dense weights = exactly what ``load_weight(include_moe_experts=...)`` yields (post
  fusion/TP-shard) -> ``kind="weight"``; at load they feed ``model.load_state_dict``.
* offload experts = exactly what ``load_expert_banks(parallel=True)`` produces (post
  backend-repack pinned banks + alpha scale vectors) -> ``kind="experts_bank"`` (alphas are
  told apart at load by their reserved names, so they need no separate kind).

The output directory is a self-contained checkpoint (config + tokenizer copied), so you can
point ``--model`` straight at it; the load path auto-detects the FTW and reads it (FTW).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import threading

import torch

from .ftw import DEFAULT_SHARD_LIMIT, FTWWriter, layer_bank_entry_name

# Machine-readable convert progress for a supervising process (e.g. a GUI frontend parses
# these `FTCONVERT <phase> <done> <total>` stdout lines to drive its convert bar). Gated by
# FREETOKEN_CONVERT_PROGRESS=1 so plain CLI use isn't spammed; the human tqdm bars stay on
# stderr. Phases: `dense` (indeterminate, done=total=0), `experts` (byte totals), `finalize`.
_EMIT_PROGRESS = os.environ.get("FREETOKEN_CONVERT_PROGRESS") == "1"


def _progress(phase: str, done: int = 0, total: int = 0) -> None:
    if _EMIT_PROGRESS:
        print(f"FTCONVERT {phase} {done} {total}", flush=True)


def _source_fingerprint(model_path: str, model_config, *, device) -> str:
    """Identity of (checkpoint + quant + GPU capability), stored in the FTW index so
    it's clear what an FTW was built from. Cheap (stat only)."""
    h = hashlib.sha256()
    h.update(f"quant={getattr(model_config, 'expert_quant', None)}|".encode())
    h.update(f"arch={getattr(model_config, 'architectures', None)}|".encode())
    try:  # nvfp4 marlin/b12x layout depends on compute capability
        h.update(f"cc={torch.cuda.get_device_capability(device)}|".encode())
    except Exception:
        pass
    files = sorted(
        glob.glob(os.path.join(model_path, "*.safetensors")) + glob.glob(os.path.join(model_path, "*.gguf"))
    )
    for f in files:
        st = os.stat(f)
        h.update(f"{os.path.basename(f)}:{st.st_size}:{int(st.st_mtime)}|".encode())
    return h.hexdigest()[:16]

# Checkpoint metadata to carry over so the FTW dir is a usable checkpoint on its own.
# (Weight shards + the safetensors index are intentionally NOT copied.)
# Everything that is NOT a weight shard is metadata we carry over verbatim. A whitelist
# misses model-specific layouts (e.g. DSV4's inference/config.json + encoding/ live in
# subdirs), so we copy every non-weight file preserving its relative path instead.
_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".ftw")  # .ftw: a nested FTW, not source
_SKIP_NAMES = ("model.safetensors.index.json",)  # indexes shards the FTW replaces
# .freetoken_expert_cache: the legacy per-bank cache (can be tens of GB of stale .bin)
_SKIP_DIRS = (".git", ".cache", ".freetoken_expert_cache")
_QWEN4_PLE_FRAGMENT = ".ple.ple_embedding.ngram_embedding."


def _copy_host_mapped_weights(model_path: str, out_dir: str) -> list[str]:
    """Preserve large CPU-mapped weights that are intentionally absent from FTW.

    Qwen4-Exp PLE tables are random-access host embeddings.  They must remain
    memory-mapped safetensors instead of being copied into resident RAM or GPU
    weights.  Copy only the shards named by those keys and write a small index
    that contains only the preserved host weights.
    """
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return []
    with open(index_path, encoding="utf-8") as index_file:
        index = json.load(index_file)
    source_map = index.get("weight_map", {})
    weight_map = {
        name: filename
        for name, filename in source_map.items()
        if _QWEN4_PLE_FRAGMENT in name
    }
    if not weight_map:
        return []

    copied: list[str] = []
    total_size = 0
    for rel in sorted(set(weight_map.values())):
        src = os.path.join(model_path, rel)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"host-mapped checkpoint shard is missing: {src}")
        dst = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
        total_size += os.path.getsize(src)

    slim_index = {
        "metadata": {"total_size": total_size, "freetoken_host_mapped_only": True},
        "weight_map": weight_map,
    }
    dst_index = os.path.join(out_dir, "model.safetensors.index.json")
    with open(dst_index, "w", encoding="utf-8") as index_file:
        json.dump(slim_index, index_file, indent=2, sort_keys=True)
        index_file.write("\n")
    copied.append("model.safetensors.index.json")
    return copied


def _copy_metadata(model_path: str, out_dir: str) -> list[str]:
    """Copy all non-weight files (config, tokenizer, remote-code, nested model configs)
    preserving directory structure, so the FTW dir is a self-contained checkpoint."""
    if os.path.isfile(model_path):
        # A single-file source has no sibling metadata to walk: a .gguf carries its config
        # AND tokenizer in its own KV section. Emit a metadata-only copy (header + KV, no
        # weight data) the FTW dir resolves those from. We deliberately do NOT sweep the
        # file's parent dir -- an HF gguf snapshot dir can hold unrelated blobs.
        from freetoken.models.gguf.reader import (
            FTW_METADATA_GGUF,
            is_gguf_path,
            write_metadata_gguf,
        )

        if is_gguf_path(model_path):
            os.makedirs(out_dir, exist_ok=True)
            write_metadata_gguf(model_path, os.path.join(out_dir, FTW_METADATA_GGUF))
            return [FTW_METADATA_GGUF]
        return []

    out_abs = os.path.abspath(out_dir)
    copied = []
    for root, dirs, files in os.walk(model_path):
        dirs[:] = [d for d in dirs
                   if d not in _SKIP_DIRS and os.path.abspath(os.path.join(root, d)) != out_abs]
        for name in files:
            if name.endswith(_WEIGHT_SUFFIXES) or name in _SKIP_NAMES:
                continue
            src = os.path.join(root, name)
            rel = os.path.relpath(src, model_path)
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    copied.extend(_copy_host_mapped_weights(model_path, out_dir))
    return copied


class _ConvertSink:
    """Layer-completion sink for ``load_expert_banks(layer_sink=...)``: writes each
    completed layer's banks as their own FTW entries immediately (name
    ``f"{bank_name}#L{layer_id:05d}"``, kind ``"experts_bank"``) and releases them, so
    conversion RAM peaks at ~in-flight layers instead of the whole bank set.

    Only engaged for streamable formats -- ``ExpertBanks.streamed`` reports whether this
    actually fired; if not, the caller falls back to the materialize-and-write path
    instead. The progress bar is created lazily on the first call, so a format that never
    streams never shows one.

    ``FTWWriter`` buffers file/shard state and is not thread-safe; completion callbacks
    can fire from the loader's own reader threads, so the write+release is serialized
    under one lock (disk-bound anyway).
    """

    def __init__(self, writer: FTWWriter, desc: str = "Converting expert banks") -> None:
        self._writer = writer
        self._desc = desc
        self._bar = None
        self._lock = threading.Lock()
        self._seen: set[int] = set()
        self.n_written = 0
        self.n_bytes = 0

    def __call__(self, layer_id: int, banks: dict) -> None:
        with self._lock:
            assert layer_id not in self._seen, f"layer {layer_id} streamed to the sink twice"
            self._seen.add(layer_id)
            if self._bar is None:
                from freetoken.utils.progress import byte_bar

                self._bar = byte_bar(0, self._desc)  # total unknown up front (streamed)
            nbytes = 0
            for bank_name, bank in banks.items():
                self._writer.add_tensor(
                    layer_bank_entry_name(bank_name, layer_id), bank.tensor, kind="experts_bank"
                )
                nbytes += bank.nbytes
                bank.release()
                self.n_written += 1
            self.n_bytes += nbytes
            self._bar.update(nbytes)
            # Cumulative BYTES (not the bank count): the supervisor maps this against the
            # known expert-pool size for a smooth phase-budgeted bar. Total stays 0 (unknown
            # up front while streaming); the materialize path below emits a real total.
            _progress("experts", self.n_bytes, 0)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    @property
    def num_layers(self) -> int:
        return len(self._seen)


def convert_checkpoint(
    model_path: str,
    out_dir: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    moe_backend: str = "offload",
    shard_limit: int = DEFAULT_SHARD_LIMIT,
    device: str | None = None,
) -> dict:
    """Write ``model_path`` as an FTW checkpoint at ``out_dir``. Returns the index dict.

    The FTW format is TP-agnostic and conversion runs single-process, so the resulting
    checkpoint records no TP layout and loads independently of the runtime TP setting."""
    from freetoken.distributed import DistributedInfo, set_tp_info, try_get_tp_info
    from freetoken.engine.config import EngineConfig
    from freetoken.models.weight import load_weight
    from freetoken.moe.expert_banks import load_expert_banks
    from .ftw import is_ftw_checkpoint

    if is_ftw_checkpoint(model_path):
        raise SystemExit(f"{model_path} is already an FTW checkpoint")
    tp = try_get_tp_info()
    if tp is None:
        set_tp_info(rank=0, size=1)
        tp = try_get_tp_info()
    elif tp.size != 1:
        raise SystemExit(
            f"FTW conversion runs single-process and the format records no TP layout, "
            f"but TP is already set to size={tp.size}"
        )
    dev = torch.device(device or "cuda:0")
    torch.cuda.set_device(dev)
    torch.zeros(1, device=dev)  # init CUDA context (needed by nvfp4 backend pick / pinning)

    cfg = EngineConfig(model_path=model_path, tp_info=DistributedInfo(tp.rank, tp.size),
                       dtype=dtype, moe_backend=moe_backend)
    mc = cfg.model_config
    offload = moe_backend == "offload" and getattr(mc, "is_moe", False)
    include_moe_experts = not offload

    from freetoken.utils.progress import byte_bar, count_bar

    writer = FTWWriter(out_dir, shard_limit=shard_limit)
    n_weight = n_bank = n_alpha = 0

    # 1) dense weights (host tensors; load straight to CPU to avoid GPU pressure)
    _progress("dense", 0, 0)  # phase start; per-tensor cumulative bytes follow (total unknown)
    dense_bytes = 0
    for name, tensor in count_bar(load_weight(model_path, torch.device("cpu"),
                                              include_moe_experts=include_moe_experts),
                                  "Converting dense weights"):
        writer.add_tensor(name, tensor, kind="weight")
        n_weight += 1
        dense_bytes += tensor.numel() * tensor.element_size()
        _progress("dense", dense_bytes, 0)

    # 2) offload expert banks (post-repack) + alpha scales (slow path auto-picks parallel/serial)
    quant_format = None
    num_layers = None
    if offload:
        # Streamable formats (bf16, ds_fp4, nvfp4 on the triton backend, gpt-oss mxfp4, q4_0,
        # qwen3_5 fp8/bf16-dequant) write each layer to its own FTW entry as it completes (via
        # the sink) instead of materializing the whole bank set first; the non-streamable ones
        # (nvfp4 marlin/b12x -- repack mutates the whole bank set in place after load) ignore
        # the sink. Which happened is per-provider (e.g. nvfp4's backend pick), so it's read
        # back from ExpertBanks.streamed, not guessed here.
        sink = _ConvertSink(writer)
        banks = load_expert_banks(model_path, mc, device=dev, dtype=dtype, layer_sink=sink)
        quant_format = banks.quant_format
        if banks.streamed:
            sink.close()
            num_layers = sink.num_layers  # however many distinct layers the sink actually saw
            n_bank = sink.n_written
            assert num_layers > 0, (
                "provider reported streamed=True but the sink never fired -- the FTW "
                "would silently have no expert banks"
            )
            # Formats that fold their global scales (nvfp4 marlin/b12x) stream the weight
            # banks per layer but keep the alphas as flat [L*E] GPU vectors; write those as
            # flat reserved-name entries (same kind + names the materialize branch uses, so
            # the reader's reserved-name path reconstructs them identically).
            for an in ("gate_up_alpha", "down_alpha"):
                alpha = getattr(banks, an, None)
                if alpha is not None:
                    writer.add_tensor(an, alpha, kind="experts_bank")
                    n_alpha += 1
        else:
            # The on-disk format keeps one contiguous region per bank and the writer only
            # has whole-tensor add_tensor, so the per-layer sources reassemble into one
            # flat tensor (a per-bank host RAM spike during conversion).
            items = []
            for name, per_layer in banks.sources.items():
                if num_layers is None:
                    num_layers = len(per_layer)
                else:
                    assert len(per_layer) == num_layers, (name, len(per_layer), num_layers)
                items.append((name, torch.cat(per_layer, dim=0) if len(per_layer) > 1 else per_layer[0]))
            for an in ("gate_up_alpha", "down_alpha"):
                if getattr(banks, an, None) is not None:
                    items.append((an, getattr(banks, an)))
            total_bytes = sum(t.numel() * t.element_size() for _, t in items)
            bar = byte_bar(total_bytes, "Converting expert banks")
            done_bytes = 0
            _progress("experts", 0, total_bytes)
            for name, tensor in items:
                writer.add_tensor(name, tensor, kind="experts_bank")
                nbytes = tensor.numel() * tensor.element_size()
                bar.update(nbytes)
                done_bytes += nbytes
                _progress("experts", done_bytes, total_bytes)
                n_bank += name not in ("gate_up_alpha", "down_alpha")
                n_alpha += name in ("gate_up_alpha", "down_alpha")
            bar.close()

    _progress("finalize")  # writing shard index + copying config/tokenizer
    copied = _copy_metadata(model_path, out_dir)

    try:
        fingerprint = _source_fingerprint(model_path, mc, device=dev)
    except Exception:
        fingerprint = None

    index = writer.finalize({
        "source_model_path": os.path.abspath(model_path),
        "fingerprint": fingerprint,
        # quant_format records the actual on-disk bank layout (e.g. nvfp4_marlin vs
        # nvfp4_b12x): the suffix is a runtime backend pick (GPU capability / env), NOT in
        # config, and the stored bytes are physically repacked into it -- so it's kept and
        # read back at load (ftw.load_ftw_banks). dtype/moe_backend were dropped: each
        # tensor already carries its own dtype, and nothing reads a model-level backend.
        "quant_format": quant_format,
        # The reader takes num_layers from the model config (copied into this
        # checkpoint); recording it here too gives load_ftw_banks a cross-check that
        # the banks match the config they ship with. None for non-offload checkpoints.
        "expert_bank_num_layers": num_layers,
        "counts": {"weight": n_weight, "experts_bank": n_bank + n_alpha},
        "copied_metadata": copied,
    })
    return index


__all__ = ["convert_checkpoint"]
