"""Run one repeatable Qwen3.8-Flash-Next text or image smoke test.

This harness is intentionally offline.  It loads the source checkout while
allowing the installed FreeToken wheel to supply unchanged native extensions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def _nvidia_memory() -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free_mib, total_mib = (int(value.strip()) for value in output.splitlines()[0].split(","))
    return free_mib * 2**20, total_mib * 2**20


def _bootstrap_native_extensions() -> None:
    import freetoken.kernel

    override = os.getenv("FREETOKEN_INSTALLED_KERNEL_DIR")
    candidate = (
        Path(override)
        if override
        else Path(sys.executable).resolve().parent.parent
        / "Lib"
        / "site-packages"
        / "freetoken"
        / "kernel"
    )
    if candidate.is_dir() and str(candidate) not in freetoken.kernel.__path__:
        freetoken.kernel.__path__.append(str(candidate))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Reply with exactly: FREETOKEN_QWEN4_OK")
    parser.add_argument("--synthetic-context-tokens", type=int)
    parser.add_argument("--needle", default="FREETOKEN_CONTEXT_48291")
    parser.add_argument("--image")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--max-prefill", type=int, default=2048)
    parser.add_argument("--moe-cache-size", type=int, default=1024)
    parser.add_argument("--moe-collect-stats", action="store_true")
    parser.add_argument("--moe-backend", choices=("offload", "hybrid", "cpu"), default="offload")
    parser.add_argument("--moe-cpu-layers")
    parser.add_argument("--expert-load", choices=("auto", "serial", "parallel"), default="serial")
    return parser.parse_args()


def _prepare_prompt(
    model_path: str,
    prompt: str,
    image_path: str | None,
    disable_thinking: bool = False,
):
    from transformers import AutoProcessor, AutoTokenizer

    if image_path is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=not disable_thinking,
        )
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        elif ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(token_id) for token_id in ids]
        return ids, None

    from PIL import Image

    processor = AutoProcessor.from_pretrained(model_path)
    assistant_prefix = "<|im_start|>assistant\n"
    if disable_thinking:
        assistant_prefix += "<think>\n\n</think>\n\n"
    text = (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{prompt}<|im_end|>\n{assistant_prefix}"
    )
    with Image.open(image_path) as image:
        encoded = processor(text=[text], images=[image.convert("RGB")], return_tensors="pt")
    ids = encoded["input_ids"][0].tolist()
    mm = {
        "pixel_values": encoded["pixel_values"],
        "image_grid_thw": encoded["image_grid_thw"],
        "mm_token_type_ids": encoded["mm_token_type_ids"][0],
    }
    return ids, mm


def _prepare_synthetic_context(
    model_path: str,
    target_tokens: int,
    needle: str,
    disable_thinking: bool,
) -> tuple[list[int], None]:
    """Build a repeatable needle-recall prompt close to ``target_tokens`` long."""
    from transformers import AutoTokenizer

    if target_tokens < 256:
        raise ValueError("--synthetic-context-tokens must be at least 256")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    filler = "This sentence is filler and does not contain the verification code. "
    prefix = "Read the full document and remember the verification code.\n\n"
    needle_text = f"The verification code is {needle}.\n\n"
    suffix = "\nWhat is the verification code? Reply with only the code."

    def encode(repetitions: int) -> list[int]:
        before = repetitions // 2
        prompt = prefix + filler * before + needle_text + filler * (repetitions - before) + suffix
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=not disable_thinking,
        )
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        elif ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(token_id) for token_id in ids]

    low, high = 0, target_tokens
    best = encode(0)
    while low <= high:
        mid = (low + high) // 2
        candidate = encode(mid)
        if len(candidate) <= target_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best, None


def main() -> None:
    args = _parse_args()
    _bootstrap_native_extensions()

    from freetoken.core import SamplingParams
    from freetoken.llm import LLM

    if args.synthetic_context_tokens is not None:
        if args.image is not None:
            raise ValueError("--synthetic-context-tokens cannot be combined with --image")
        prompt_ids, mm = _prepare_synthetic_context(
            args.model,
            args.synthetic_context_tokens,
            args.needle,
            disable_thinking=args.disable_thinking,
        )
    else:
        prompt_ids, mm = _prepare_prompt(
            args.model,
            args.prompt,
            args.image,
            disable_thinking=args.disable_thinking,
        )
    # Engine requires CUDA to be uninitialized when it selects the process GPU.
    # nvidia-smi gives us the baseline without creating a CUDA context.
    free_before, total = _nvidia_memory()
    load_start = time.perf_counter()
    llm = LLM(
        args.model,
        dtype=torch.bfloat16,
        max_running_req=1,
        attention_backend="auto",
        moe_backend=args.moe_backend,
        nvfp4_backend="triton",
        expert_load=args.expert_load,
        moe_cache_size=args.moe_cache_size,
        moe_collect_stats=args.moe_collect_stats,
        moe_cpu_layers=args.moe_cpu_layers,
        cache_type="naive",
        max_seq_len_override=args.max_seq_len,
        num_token_override=args.num_tokens,
        max_extend_tokens=args.max_prefill,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    free_loaded, _ = torch.cuda.mem_get_info()

    from freetoken.message import DetokenizeMsg

    token_times: list[float] = []
    original_send_result = llm.send_result

    def timed_send_result(reply):
        now = time.perf_counter()
        for msg in reply:
            if isinstance(msg, DetokenizeMsg) and not (
                msg.finished and msg.next_token in llm.eos_token_ids
            ):
                token_times.append(now)
        original_send_result(reply)

    llm.send_result = timed_send_result
    generation_start = time.perf_counter()
    result = llm.generate(
        [prompt_ids],
        SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
        mm_inputs=[mm] if mm is not None else None,
    )[0]
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_start
    output_tokens = len(result["token_ids"])
    cache_stats = None
    if args.moe_collect_stats and llm.engine.moe_offload_cache is not None:
        cache = llm.engine.moe_offload_cache
        cache_stats = cache.decode_miss_stats()
        rates = [
            layer["miss_rate"]
            for layer in cache.decode_miss_stats_per_layer()["per_layer"]
            if layer["steps"]
        ]
        if rates:
            cache_stats["layer_miss_rate_min"] = min(rates)
            cache_stats["layer_miss_rate_max"] = max(rates)
    ttft_seconds = token_times[0] - generation_start if token_times else None
    decode_seconds = token_times[-1] - token_times[0] if len(token_times) > 1 else None
    steady_decode_tps = (
        (len(token_times) - 1) / decode_seconds
        if decode_seconds is not None and decode_seconds > 0
        else None
    )
    print(
        "QWEN4_SMOKE_RESULT "
        + json.dumps(
            {
                "prompt_tokens": len(prompt_ids),
                "output_tokens": output_tokens,
                "load_seconds": round(load_seconds, 3),
                "generation_seconds": round(generation_seconds, 3),
                "overall_output_tps": round(output_tokens / generation_seconds, 3),
                "ttft_seconds": round(ttft_seconds, 3) if ttft_seconds is not None else None,
                "decode_seconds": round(decode_seconds, 3) if decode_seconds is not None else None,
                "steady_decode_tps": (
                    round(steady_decode_tps, 3) if steady_decode_tps is not None else None
                ),
                "gpu_total_gib": round(total / 2**30, 3),
                "gpu_used_by_load_gib": round((free_before - free_loaded) / 2**30, 3),
                "moe_backend": args.moe_backend,
                "moe_cpu_layers": args.moe_cpu_layers,
                "moe_cache_size": args.moe_cache_size,
                "moe_cache_stats": cache_stats,
                "text": result["text"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
