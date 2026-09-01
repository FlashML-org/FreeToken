#!/usr/bin/env python3
"""Measure a quality-gated long visible Gemma4 image response over OpenAI SSE."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image


def data_url(image: Image.Image) -> str:
    """Encode the deterministic fixture without network access."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> int:
    """Request a constrained visual description and preserve quality and TPS evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    image = Image.new("RGB", (96, 48), (0, 0, 255))
    for x in range(48):
        for y in range(48):
            image.putpixel((x, y), (255, 0, 0))
    prompt = (
        "Describe this image in 45 to 65 words. State the colors, their left-to-right "
        "arrangement, and the image shape. Do not use headings, bullet points, or reasoning."
    )
    payload = {"model": args.model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url(image)}},
    ]}], "temperature": 0, "max_tokens": args.max_tokens, "stream": True,
        "stream_options": {"include_usage": True}}
    request = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter(); stamps: list[float] = []; content: list[str] = []; reasoning: list[str] = []; usage: dict = {}
    with urllib.request.urlopen(request, timeout=300) as response:  # nosec B310: local caller URL
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "): continue
            event_data = line[6:]
            if event_data == "[DONE]": break
            event = json.loads(event_data); usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {}); text = delta.get("content") or ""; thought = delta.get("reasoning_content") or ""
                if text or thought: stamps.append(time.perf_counter()); content.append(text); reasoning.append(thought)
    visible = "".join(content).strip(); normalized = visible.lower(); words = visible.split()
    duration = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0
    completion = usage.get("completion_tokens")
    record = {"schema_version": 1, "prompt": prompt, "visible": visible, "reasoning": "".join(reasoning), "usage": usage,
        "quality": {"word_count": len(words), "has_red": "red" in normalized, "has_blue": "blue" in normalized,
                    "has_left": "left" in normalized, "has_right": "right" in normalized,
                    "visible_words_45_to_65": 45 <= len(words) <= 65},
        "metrics": {"events": len(stamps), "ttft_ms": (stamps[0]-started)*1000 if stamps else None,
                    "stream_window_s": duration, "completion_tokens": completion,
                    "completion_tok_s": completion/duration if completion and duration else None,
                    "wall_s": time.perf_counter()-started}}
    record["passed"] = all(record["quality"].values())
    args.artifact.parent.mkdir(parents=True, exist_ok=True); args.artifact.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))
    if not record["passed"]: raise SystemExit("visual TPS quality gate failed")
    return 0


if __name__ == "__main__": raise SystemExit(main())
