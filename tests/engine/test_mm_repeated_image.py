"""A request that carries the same image twice, chunked between the two occurrences (a real checkpoint)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

CHECKPOINT = os.environ.get("FREETOKEN_QWEN3VL_MODEL", "")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.path.exists(os.path.join(CHECKPOINT, "config.json")), reason="FREETOKEN_QWEN3VL_MODEL not set"),
]

# the engine wants a fresh CUDA context, so each run gets its own process
_SCRIPT = """
import io, sys, torch
from PIL import Image
from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.mm.config import MultimodalConfig

checkpoint, storage = sys.argv[1], sys.argv[2]
llm = LLM(model_path=checkpoint, dtype=torch.bfloat16, attention_backend="auto", max_running_req=1, cuda_graph_max_bs=1,
          max_extend_tokens=128, max_seq_len_override=4096, mm=MultimodalConfig(embed_cache_device=storage))
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                         {"type": "text", "text": "What color are these two images? Answer in one word."}]}]
prompt = llm.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
ids = llm.tokenizer.encode(prompt, add_special_tokens=False)
buf = io.BytesIO(); Image.new("RGB", (256, 256), (255, 0, 0)).save(buf, format="PNG"); red = buf.getvalue()
text = llm.generate([ids], SamplingParams(max_tokens=8, temperature=0.0), images=[[red, red]])[0]["text"]
print("RESULT", repr(text), llm.engine.encoder_cache.stats())
"""


@pytest.mark.parametrize("storage", ["cpu", "cuda"])
def test_repeated_image_survives_the_chunk_boundary(storage):
    run = subprocess.run(
        [sys.executable, "-c", _SCRIPT, CHECKPOINT, storage],
        capture_output=True, text=True, timeout=900, env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    assert run.returncode == 0, run.stderr[-2000:]
    result = [line for line in run.stdout.splitlines() if line.startswith("RESULT")][-1]
    assert "red" in result.lower(), result
    assert result.endswith("(0, 0)"), result  # every claim gathered, nothing left behind
