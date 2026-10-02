#!/usr/bin/env python3
"""Run one deterministic OpenAI image control against an isolated Gemma4 server.

The generated solid-red PNG removes network and copyrighted-image variables. It
still exercises every production-relevant multimodal boundary: OpenAI content
parts, data-URL decoding, Gemma4 resize/patching, shaped ZMQ tensors, the ROCm
vision tower, projector, image-slot scatter, and response formatting.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image


def _png_data_url(image: Image.Image) -> str:
    """Encode one generated RGB fixture as an OpenAI-compatible data URL."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _post_json(url: str, payload: dict) -> dict:
    """Send one bounded JSON request and decode the server's JSON response."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:  # nosec B310: caller controls local base URL
        return json.loads(response.read().decode("utf-8"))


def _post_json_stream(url: str, payload: dict) -> tuple[dict, dict]:
    """Stream one OpenAI response and retain timing plus the final message.

    The returned metrics deliberately distinguish the server-reported completion
    token count from the number of network chunks. A chunk is not necessarily a
    token, so TPS is calculated only when final OpenAI usage is available.
    """
    payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first_chunk: float | None = None
    last_chunk: float | None = None
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(request, timeout=300) as response:  # nosec B310: caller controls local base URL
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                piece = delta.get("content") or ""
                thought = delta.get("reasoning_content") or ""
                if piece or thought:
                    now = time.perf_counter()
                    first_chunk = first_chunk if first_chunk is not None else now
                    last_chunk = now
                    content.append(piece)
                    reasoning.append(thought)
    elapsed = time.perf_counter() - started
    completion = usage.get("completion_tokens")
    generated_window = (last_chunk - first_chunk) if first_chunk is not None and last_chunk is not None else 0.0
    metrics = {
        "wall_s": elapsed,
        "ttft_ms": (first_chunk - started) * 1000 if first_chunk is not None else None,
        "stream_window_s": generated_window,
        "completion_tokens": completion,
        "completion_tok_s": completion / generated_window if completion and generated_window else None,
    }
    return {
        "choices": [{"message": {"role": "assistant", "content": "".join(content), "reasoning_content": "".join(reasoning)}}],
        "usage": usage,
    }, metrics


def main() -> int:
    """Run color and spatial fixtures, validate exact answers, and save evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--max-tokens", type=int, default=16,
        help="per-case generation cap; llama.cpp needs a larger cap when it emits thought first",
    )
    parser.add_argument("--stream", action="store_true", help="capture stream timing and final usage")
    args = parser.parse_args()

    split = Image.new("RGB", (96, 48), (0, 0, 255))
    for x in range(48):
        for y in range(48):
            split.putpixel((x, y), (255, 0, 0))
    cases = [
        ("solid_red", Image.new("RGB", (16, 16), (255, 0, 0)),
         "What is the dominant color in the image? Reply with one lowercase word.", "red"),
        ("solid_green", Image.new("RGB", (16, 16), (0, 255, 0)),
         "What is the dominant color in the image? Reply with one lowercase word.", "green"),
        ("red_left_blue_right", split,
         "What color is the left half of the image? Reply with one lowercase word.", "red"),
    ]
    records = []
    for name, image, prompt, expected in cases:
        request = {
            "model": args.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _png_data_url(image)}},
            ]}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        started = time.perf_counter()
        metrics = None
        if args.stream:
            response, metrics = _post_json_stream(args.base_url.rstrip("/") + "/v1/chat/completions", request)
        else:
            response = _post_json(args.base_url.rstrip("/") + "/v1/chat/completions", request)
        text = response["choices"][0]["message"]["content"].strip().lower()
        records.append({
            "control": name,
            "prompt": prompt,
            "expected": expected,
            "actual": text,
            "elapsed_s": time.perf_counter() - started,
            "stream_metrics": metrics,
            "usage": response.get("usage"),
            "response": response,
        })
    record = {"schema_version": 2, "passed": all(item["actual"] == item["expected"] for item in records), "cases": records}
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if not record["passed"]:
        failures = [f"{item['control']}: expected {item['expected']!r}, got {item['actual']!r}" for item in records if item["actual"] != item["expected"]]
        raise SystemExit("Gemma4 image control failed: " + "; ".join(failures))
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
