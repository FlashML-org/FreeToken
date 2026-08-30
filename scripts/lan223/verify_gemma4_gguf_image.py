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


def main() -> int:
    """Run color and spatial fixtures, validate exact answers, and save evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
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
            "max_tokens": 16,
        }
        started = time.perf_counter()
        response = _post_json(args.base_url.rstrip("/") + "/v1/chat/completions", request)
        text = response["choices"][0]["message"]["content"].strip().lower()
        records.append({
            "control": name,
            "prompt": prompt,
            "expected": expected,
            "actual": text,
            "elapsed_s": time.perf_counter() - started,
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
