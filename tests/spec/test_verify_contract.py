"""A k+1-row verify step commits exactly what k+1 decode steps would: with drafts taken from a plain greedy run and wrong at a different place each step (so every accept count occurs), speculative decoding must reproduce that run token for token.

A tiny hybrid (GatedDeltaNet + attention) model with dummy weights, served end to end through the scheduler; the drafts come from a stub proposer, so this pins the target side alone.
A verify row and a decode step run different GEMM and attention shapes, so their logits can differ by an ulp: the runs may part at a near tie, and the test checks they part only there. fp16 keeps such ties rare (bf16 has 3 fewer mantissa bits).
"""

import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

TOKENIZER_SOURCE = os.environ.get("FREETOKEN_QWEN36_MODEL", "")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not os.path.exists(os.path.join(TOKENIZER_SOURCE, "tokenizer.json")), reason="FREETOKEN_QWEN36_MODEL not set"),
]

K = 3

_SCRIPT = """
import json, os, sys, torch
from freetoken.core import SamplingParams
from freetoken.engine import engine as E, sample as S
from freetoken.engine.spec_protocol import SpecProposal
from freetoken.llm import LLM
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.spec import PROPOSERS
from freetoken.spec import runner as R
from freetoken.spec.proposer import Proposer

model, out, reference, kwargs = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])

# (uid, output index) -> top-2 token ids and logits of the row that picked it
top2 = {}
current = {}


def watch(cls, name):
    original = getattr(cls, name)

    def wrapped(self, batch, *args):
        current["batch"] = batch
        return original(self, batch, *args)

    setattr(cls, name, wrapped)


watch(E.Engine, "forward_batch")
watch(R.SpecRunner, "verify_step")


def record(rows, logits):
    top = logits.float().topk(2, dim=-1)
    for key, ids, vals in zip(rows, top.indices.tolist(), top.values.tolist()):
        if key is not None:
            top2[key] = ids + vals


def decode_rows(batch):
    # sampling runs after complete_one: the token lands at device_len - 1
    return [None if isinstance(r, ChunkedReq) else (r.uid, r.device_len - 1 - r.prompt_len) for r in batch.reqs]


def verify_rows(batch, width):
    # row j of a request predicts position cached_len + j + 1
    return [(r.uid, r.cached_len + j + 1 - r.prompt_len) for r in batch.reqs for j in range(width)]


sample, sample_verify = S.Sampler.sample, S.Sampler.sample_verify
S.Sampler.sample = lambda self, logits, args: (record(decode_rows(current["batch"]), logits), sample(self, logits, args))[1]
S.Sampler.sample_verify = lambda self, logits, args, drafts: (
    record(verify_rows(current["batch"], drafts.shape[1] + 1), logits), sample_verify(self, logits, args, drafts))[1]


class OracleProposer(Proposer):
    name = "oracle"
    default_k = 3
    max_k = 8

    def __init__(self, config, target_model):
        super().__init__(config, target_model)
        self.reference = json.load(open(reference))
        self.steps = 0

    def propose(self, reqs, cs, rows, token_pool):
        drafts = []
        for req, c in zip(reqs, cs):
            done = c - req.prompt_len  # outputs committed before the pending token at c
            want = self.reference["outputs"][req.uid][done + 1 : done + 1 + self.k]
            want = want + [0] * (self.k - len(want))
            wrong = self.steps % (self.k + 1)  # k: every draft right
            if wrong < self.k:
                want[wrong] = (want[wrong] + 1) % 1000
            drafts.append(want)
        self.steps += 1
        return SpecProposal(torch.tensor(drafts, dtype=torch.int64, device=rows.device))


if reference != "-":
    PROPOSERS.register("oracle")("__main__:OracleProposer")
llm = LLM(model_path=model, dtype=torch.float16, use_dummy_weight=True, max_running_req=4, **kwargs)
prompts = [llm.tokenizer.encode(p) for p in ("The capital of France is", "1, 2, 3, 4,", "def fibonacci(n):")]
params = [SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, presence_penalty=p, frequency_penalty=p)
          for n, p in ((48, 0.0), (37, 0.6), (41, 0.0))]
res = llm.generate(prompts, params)
json.dump({"outputs": [r["token_ids"] for r in res], "top2": [[u, i, *t] for (u, i), t in top2.items()]}, open(out, "w"))
"""


def _tiny_model(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if os.path.exists(os.path.join(TOKENIZER_SOURCE, name)):
            shutil.copy(os.path.join(TOKENIZER_SOURCE, name), path)
    config = {
        "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text", "vocab_size": 248320,
        "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "full_attention_interval": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 32, "linear_num_key_heads": 2,
        "linear_num_value_heads": 4, "linear_value_head_dim": 32, "attn_output_gate": True, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "max_position_embeddings": 4096, "tie_word_embeddings": False, "eos_token_id": 248044,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                            "mrope_section": [3, 3, 2], "mrope_interleaved": True},
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    return path


def _run(model: str, out: str, reference: str, **kwargs) -> dict:
    run = subprocess.run([sys.executable, "-c", _SCRIPT, model, out, reference, json.dumps(kwargs)],
                         capture_output=True, text=True, timeout=900, env=dict(os.environ))
    assert run.returncode == 0, run.stderr[-3000:]
    with open(out) as f:
        return json.load(f)


@pytest.mark.parametrize("graphs", [True, False])
@pytest.mark.parametrize("backend", ["fa", "triton"])
def test_verify_steps_reproduce_greedy_decode(tmp_path, backend, graphs):
    if backend == "fa":
        pytest.importorskip("sgl_kernel")
    model = _tiny_model(str(tmp_path / "model"))
    kwargs = {"attention_backend": backend, "cuda_graph_max_bs": 4 if graphs else 0}
    reference = _run(model, str(tmp_path / "decode.json"), "-", **kwargs)
    spec = _run(model, str(tmp_path / "spec.json"), str(tmp_path / "decode.json"),
                speculative_algorithm="oracle", speculative_num_steps=K, **kwargs)
    agreed = 0
    for uid, (want, got) in enumerate(zip(reference["outputs"], spec["outputs"])):
        assert len(got) == len(want)
        split = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), len(want))
        agreed += split
        if split < len(want):
            # same prefix, so both runs must rank the two tokens top-2 and within two fp16 ulps of logits in [64, 128)
            pair = {want[split], got[split]}
            for run in (reference, spec):
                a, b, va, vb = next(t[2:] for t in run["top2"] if t[:2] == [uid, split])
                assert {a, b} == pair and va - vb <= 0.125, f"request {uid} parts at {split}: top-2 {(a, b)} gap {va - vb}"
    # most of the run must still be compared, or a tie early in every request would hide a bug
    assert agreed >= 0.5 * sum(map(len, reference["outputs"]))
