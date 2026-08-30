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


def _red_png_data_url() -> str:
    """Return a small, valid RGB PNG encoded as an OpenAI-compatible data URL."""
    image = Image.new("RGB", (16, 16), (255, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _post_json(url: str, payload: dict) -> dict:
    """Send one bounded JSON request and decode the server's JSON response."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:  # nosec B310: caller controls local base URL
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    """Submit the red-image question, validate the exact short answer, and save evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()

    prompt = "What is the dominant color in the image? Reply with one lowercase word."
    request = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _red_png_data_url()}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 16,
    }
    started = time.perf_counter()
    response = _post_json(args.base_url.rstrip("/") + "/v1/chat/completions", request)
    elapsed_s = time.perf_counter() - started
    text = response["choices"][0]["message"]["content"].strip().lower()
    record = {
        "control": "solid_red_png_data_url",
        "prompt": prompt,
        "expected": "red",
        "actual": text,
        "elapsed_s": elapsed_s,
        "usage": response.get("usage"),
        "response": response,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if text != "red":
        raise SystemExit(f"Gemma4 image control failed: expected 'red', got {text!r}")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
