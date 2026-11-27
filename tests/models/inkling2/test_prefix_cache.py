"""Prefix-cache continuation against a cold prefill on the real checkpoint (``FREETOKEN_INKLING2_MODEL``).

A hit reads the convolution inputs of the tokens before it and the window KV from the tree, then
prefills the rest. When the cached prefix was itself prefilled, the continuation must be bitwise
equal to a cold prefill chunked at the same point (the same forwards, nothing restored): a row kept
for the wrong token, a row freed too early or a window freed too early all break that. A previous
turn's output was decoded, and no prefill reproduces decoded KV bit for bit, so a hit there is held to
the decode-vs-prefill noise of its first logits. A greedy continuation is not compared there, as any
near tie may flip it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

CHECKPOINT = os.environ.get("FREETOKEN_INKLING2_MODEL", "")

pytestmark = [
    pytest.mark.needs_weights,
    pytest.mark.slow,
    pytest.mark.skipif(not os.path.exists(os.path.join(CHECKPOINT, "config.json")), reason="FREETOKEN_INKLING2_MODEL not set"),
]

# the engine wants a fresh CUDA context: every scenario runs in one child process
_SCRIPT = r'''
import io, json, sys
import torch
from PIL import Image, ImageDraw
from freetoken.core import SamplingParams
from freetoken.llm import LLM

llm = LLM(sys.argv[1], moe_strategy="fused", max_running_req=4, memory_ratio=0.85)
assert llm.config.cache_type == "swa_radix", llm.config.cache_type
logits, admissions = [], []
forward = llm.engine.model.forward


def capture():
    out = forward()
    if llm.engine.ctx.batch.is_prefill:
        logits.append(out[-1].float().cpu())
    return out


llm.engine.model.forward = capture
report = llm._report_prompt_admissions


def admitted(batch):
    admissions.extend((prompt, cached) for _, prompt, cached in batch.prompt_admissions or [])
    return report(batch)


llm._report_prompt_admissions = admitted
budget = llm.prefill_budget
tok = llm.tokenizer
eos = tok.convert_tokens_to_ids("<|content_model_end_sampling|>")
ids = lambda text: tok.encode(text, add_special_tokens=False)
user = lambda text: ids("<|message_user|><|content_text|>" + text + "<|end_message|>")


def run(prompt, n=24, image=None, chunk=None):
    """(greedy tokens, first-step logits, prefix hit, prompt tokens after image expansion)"""
    logits.clear()
    admissions.clear()
    llm.prefill_budget = chunk or budget
    try:
        out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=n), images=[[image]] if image else None)
    finally:
        llm.prefill_budget = budget
    return out[0]["token_ids"], logits[-1], admissions[0][1], admissions[0][0]


def turn(prompt, n=48, image=None):
    out = run(prompt, n, image)[0]
    return prompt + out + ([eos] if len(out) < n else [])  # the end-sampling token closes a re-rendered reply


def cold(prompt, chunk=None, image=None):
    llm.cache_manager.flush()
    return run(prompt, image=image, chunk=chunk)


def report_case(name, warm, ref, **extra):
    print("RESULT " + json.dumps({
        "name": name, "hit": warm[2], "tokens_equal": warm[0] == ref[0], "bitwise": bool(torch.equal(warm[1], ref[1])),
        "max_diff": float((warm[1] - ref[1]).abs().max()), **extra}), flush=True)


def price_list(question):
    # longer than the 1024-token window, so a hit needs the window KV the tree kept
    items = " ".join(f"Item {i} costs {i * 7 % 101} coins." for i in range(260))
    return user("You are a careful shopkeeper. Here is the price list: " + items + " " + question)


llm.cache_manager.flush()
t1 = price_list("What does item 5 cost? Answer briefly.")
reply = turn(t1)[len(t1):]
t2 = t1 + reply[:10] + ids(" rewritten history") + user("And item 9?")
warm = run(t2)
report_case("rewritten", warm, cold(t2, chunk=warm[2]), expect=len(t1))

llm.cache_manager.flush()
t1 = price_list("What does item 7 cost?")
run(t1)
warm = run(t1)
report_case("resent", warm, cold(t1, chunk=warm[2]), expect=len(t1) - 1)

llm.cache_manager.flush()
a = price_list("What does item 3 cost?")
run(a)
b = a + user("Also, which item is the cheapest?")  # another request that extends the first one's prompt
warm = run(b)
report_case("cross_request", warm, cold(b, chunk=warm[2]), expect=len(a))

llm.cache_manager.flush()
t1 = turn(price_list("What does item 5 cost? Answer briefly."))
t2 = t1 + user("And item 9?")
warm = run(t2)
report_case("next_turn", warm, cold(t2), expect=len(t1) - 1)

img = Image.new("RGB", (200, 120), "white")
draw = ImageDraw.Draw(img)
draw.rectangle((10, 10, 90, 70), fill="red")
draw.ellipse((110, 30, 190, 110), fill="blue")
buf = io.BytesIO()
img.save(buf, format="PNG")
png = buf.getvalue()
llm.cache_manager.flush()
i1 = ids("<|message_user|><|content_image|><|unused_200054|><|end_message|>") + user("Describe the image.")
expanded = run(i1, image=png)[3]
i2 = i1 + user("How many shapes are there?")
warm = run(i2, image=png)
report_case("image", warm, cold(i2, chunk=warm[2], image=png), expect=expanded)

llm.cache_manager.flush()
llm.cache_manager.check_integrity()
print("RESULT " + json.dumps({"name": "integrity", "ok": True}), flush=True)
'''


@pytest.fixture(scope="module")
def results() -> dict:
    run = subprocess.run(
        [sys.executable, "-c", _SCRIPT, CHECKPOINT],
        capture_output=True, text=True, timeout=1800, env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    assert run.returncode == 0, run.stderr[-3000:]
    rows = [json.loads(line[len("RESULT "):]) for line in run.stdout.splitlines() if line.startswith("RESULT ")]
    return {row["name"]: row for row in rows}


def _bitwise_hit(row: dict) -> None:
    assert row["hit"] == row["expect"], row
    assert row["tokens_equal"] and row["bitwise"], row


def test_a_rewritten_reply_resumes_at_the_end_of_its_prompt(results):
    row = results["rewritten"]
    if row["hit"] == row["expect"]:
        return _bitwise_hit(row)
    # with overlap the prompt is committed after the first decode step was launched, so the
    # resume point holds that decoded token too
    assert row["hit"] == row["expect"] + 1, row
    _close_hit(row)


def test_a_resent_prompt_prefills_only_its_last_token(results):
    _bitwise_hit(results["resent"])  # the retain gap before the prompt's end keeps the point one token back


def test_a_request_that_extends_another_requests_prompt_reuses_it(results):
    _bitwise_hit(results["cross_request"])


def _close_hit(row: dict) -> None:
    # decoded vs prefilled KV moves the first logits by 0.8-1.9 here; a lost state restore by 5-9
    assert row["max_diff"] < 3.0, row


def test_the_next_turn_resumes_at_the_end_of_the_previous_one(results):
    row = results["next_turn"]
    assert row["hit"] >= row["expect"], row  # everything the previous turn fed through the model
    _close_hit(row)


def test_an_image_prefix_is_reused(results):
    _bitwise_hit(results["image"])


def test_the_cache_returns_every_slot(results):
    assert results["integrity"]["ok"]
