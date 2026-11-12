"""A k+1-row verify step commits exactly what k+1 decode steps would: with drafts taken from a plain greedy run and wrong at a different place each step (so every accept count occurs), speculative decoding must reproduce that run token for token.

Tiny hybrid models with dummy weights, served end to end through the scheduler: qwen3_5 (GatedDeltaNet + attention) and qwen4_exp (GatedDeltaNet + QSA, PLE, hyper-connections). The drafts come from a stub proposer, so this pins the target side alone; a second run drafts with a DFlash 2 draft of random weights, whose wrong drafts must leave the target's output alone through its ring, window and graphs; a third drafts with the target's own MTP layer of random weights, which needs the tiny target materialized as a real checkpoint since MTP rejects dummy weights.
A gemma4 target runs the same contract on a windowed KV pool: its oracle case asserts the out-of-window frees and the verify rollback fired, and its mtp case drafts from a tiny gemma4-assistant GGUF, which keeps no KV and reads the target's through the read-only view.
A verify row and a decode step run different GEMM and attention shapes, so their logits can differ by an ulp: the runs may part at a near tie, and the test checks they part only there. fp16 keeps such ties rare (bf16 has 3 fewer mantissa bits), and fan-in scaled dummy weights keep the activations O(1), so ulp noise does not grow through the layers.
"""

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from freetoken.models.gemma4.mtp import DRAFT_LAYERS

# tests/ is not a package, so a plain `pytest` run cannot import a sibling test by name
_spec = importlib.util.spec_from_file_location(
    "_qwen_gguf_writer", os.path.join(os.path.dirname(__file__), "..", "models", "test_qwen3_5_moe_weight.py"))
_writer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_writer)
BF16, F32 = _writer.BF16, _writer.F32
_gguf_bytes, _write_gguf = _writer._gguf_bytes, _writer._write_gguf

TOKENIZER_SOURCE = os.environ.get("FREETOKEN_QWEN36_MODEL", "")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not os.path.exists(os.path.join(TOKENIZER_SOURCE, "tokenizer.json")), reason="FREETOKEN_QWEN36_MODEL not set"),
]

K = 3
# longer than the tiny gemma4 target's 16-position window, so its prefill and verify rows both slide past it
WINDOWED_PROMPTS = [
    "The capital of France is Paris, and the capital of Germany is Berlin. The capital of Italy is",
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
]

_SCRIPT = """
import json, os, sys, torch
from freetoken.core import SamplingParams
from freetoken.engine import engine as E, sample as S
from freetoken.llm import LLM
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.spec import PROPOSERS
from freetoken.spec import runner as R
from freetoken.spec.mtp.shared_proposer import SharedKVMTPProposer
from freetoken.spec.proposer import Proposer, SpecProposal

model, out, reference, kwargs = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
dump_weights, use_dummy_weight = kwargs.pop("dump_weights", None), kwargs.pop("use_dummy_weight", True)
dump_prefix, norm_ones = kwargs.pop("dump_prefix", ""), kwargs.pop("norm_ones", False)
family = kwargs.pop("family", "")
prompt_texts = kwargs.pop("prompts", None) or ["The capital of France is", "1, 2, 3, 4,", "def fibonacci(n):"]
TAPS, H = (1, 2), 256

# (uid, output index) -> top-3 token ids and logits of the row that picked it
top3 = {}
current = {}
# the tapped layers' outputs as the sink got them, and as the next layer read them
taps_got, taps_ref = {}, {}
taps = {"prefill": 0, "verify": 0, "mismatch": 0, "width": None}
steps = {"verify": 0}


def check_taps(batch):
    # gemma4's decoder hands the sink nothing, so there is no tap reference to compare against
    if family == "gemma4" or not isinstance(batch.features, RecordingSink):
        return
    n, width = batch.input_ids.shape[0], taps["width"]
    taps["prefill" if batch.is_prefill else "verify"] += 1
    taps["mismatch"] += sum(not torch.equal(taps_got[i][:n, :width], taps_ref[i][:n, :width]) for i in TAPS)


def watch(cls, name):
    original = getattr(cls, name)

    def wrapped(self, batch, *args):
        current["batch"] = batch
        result = original(self, batch, *args)
        check_taps(batch)
        steps["verify"] += batch.phase == "verify"
        return result

    setattr(cls, name, wrapped)


watch(E.Engine, "forward_batch")
watch(R.SpecRunner, "verify_step")

# SWA window machinery: _free_swa is the only place slots come back, so attribute each call to the
# entry point that made it. A run that never reaches the decode or rollback paths leaves them at 0.
from freetoken.scheduler.cache import CacheManager

evict = {"decode": 0, "extend": 0, "rollback": 0, "other": 0, "slots": 0}
bucket = ["other"]
free_swa = CacheManager._free_swa


def counting_free_swa(self, indices):
    if len(indices):
        evict[bucket[0]] += 1
        evict["slots"] += len(indices)
    return free_swa(self, indices)


CacheManager._free_swa = counting_free_swa
for _name, _key in (("maybe_free_swa_out_of_window", "decode"), ("free_swa_out_of_window_extend", "extend"),
                    ("rollback_tail", "rollback")):
    _original = getattr(CacheManager, _name)

    def _wrapped(self, *args, _original=_original, _key=_key):
        previous, bucket[0] = bucket[0], _key
        try:
            return _original(self, *args)
        finally:
            bucket[0] = previous

    setattr(CacheManager, _name, _wrapped)

# the draft side: whether the drafts came from the draft's own head reading the target's slabs
from freetoken.models.gemma4.mtp import Gemma4MTPDraft
from freetoken.spec.mtp.shared_kv import ReadOnlyTargetKVView

draft = {"head_calls": 0, "reads": {}, "top3": []}
draft_logits = Gemma4MTPDraft.draft_logits


def counting_draft_logits(self, hidden):
    draft["head_calls"] += 1
    logits = draft_logits(self, hidden)
    top = logits.float().topk(3, dim=-1)
    draft["top3"].append((top.indices.tolist(), top.values.tolist()))
    return logits


Gemma4MTPDraft.draft_logits = counting_draft_logits
k_cache = ReadOnlyTargetKVView.k_cache


def counting_k_cache(self, index):
    layer = self._route[index]
    draft["reads"][layer] = draft["reads"].get(layer, 0) + 1
    return k_cache(self, index)


ReadOnlyTargetKVView.k_cache = counting_k_cache

# the identity sidecar's drafts are known exactly, so record each proposal to pin the seed row
propose_orig = SharedKVMTPProposer.propose
proposals = []


def recording_propose(self, reqs, rows, pending):
    proposal = propose_orig(self, reqs, rows, pending)
    # this call's k head calls: per request row, one [ids, vals] top-3 per step
    tops = draft["top3"][len(draft["top3"]) - self.k:]
    proposals.append([
        (r.uid, p, [[ids[i], vals[i]] for ids, vals in tops])
        for r, p, i in zip(reqs, pending.tolist(), range(len(reqs)))
    ])
    return proposal


SharedKVMTPProposer.propose = recording_propose


class RecordingSink:
    # copies the tapped layers' outputs into persistent buffers, so a captured verify graph writes them too
    wanted_layers = frozenset(TAPS)

    def final_hidden(self, x):
        pass

    def layer_output(self, layer_id, x, residual=None, num_streams=1):
        if residual is not None:
            x = x + residual
        taps["width"] = x.shape[1]
        taps_got[layer_id][: x.shape[0], : x.shape[1]].copy_(x)


def tap_next_layer_inputs(model):
    # the reference for layer i's output is what layer i+1 reads: the residual its fused add forms (qwen3_5), or its input streams (qwen4_exp)
    layers = model.model.layers.op_list
    for i in TAPS:
        taps_got[i] = torch.zeros(4096, 4 * H, dtype=torch.float16, device="cuda")
        taps_ref[i] = torch.zeros(4096, 4 * H, dtype=torch.float16, device="cuda")
    if hasattr(layers[0], "input_layernorm"):
        cls, by_id = type(layers[0].input_layernorm), {id(layers[i + 1].input_layernorm): i for i in TAPS}
        original = cls.forward

        def wrapped(self, x, residual=None):
            x, residual = original(self, x, residual)
            if (i := by_id.get(id(self))) is not None:
                taps_ref[i][: x.shape[0], : x.shape[1]].copy_(residual)
            return x, residual
    else:
        cls, by_id = type(layers[0]), {id(layers[i + 1]): i for i in TAPS}
        original = cls.forward

        def wrapped(self, hidden, batch):
            if (i := by_id.get(id(self))) is not None:
                taps_ref[i][: hidden.shape[0], : hidden.shape[1]].copy_(hidden)
            return original(self, hidden, batch)

    cls.forward = wrapped


def record(rows, logits):
    top = logits.float().topk(3, dim=-1)
    for key, ids, vals in zip(rows, top.indices.tolist(), top.values.tolist()):
        if key is not None:
            top3[key] = ids + vals


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
    @classmethod
    def num_steps(cls, config):
        return 3, 8

    @classmethod
    def target_may_be_windowed(cls, config):
        # the oracle drafts from a reference but reads nothing: answer as SharedKVMTPProposer does, or
        # the engine refuses a hybrid SWA target before the case exercises anything
        return SharedKVMTPProposer.target_may_be_windowed(config)

    def __init__(self, config, target_model):
        super().__init__(config, target_model)
        self.reference = json.load(open(reference))
        self.steps = 0

    def feature_sink(self, batch):
        return RecordingSink()

    def propose(self, reqs, rows, pending):
        drafts = []
        for req in reqs:
            done = req.cached_len - req.prompt_len  # outputs committed before the pending token at cached_len
            want = self.reference["outputs"][req.uid][done + 1 : done + 1 + self.k]
            want = want + [0] * (self.k - len(want))
            wrong = self.steps % (self.k + 1)  # k: every draft right
            if wrong < self.k:
                want[wrong] = (want[wrong] + 1) % 1000
            drafts.append(want)
        self.steps += 1
        return SpecProposal(torch.tensor(drafts, dtype=torch.int64, device=rows.device))


def fan_in_scaled(tensor):
    # unit-variance dummy matrices grow the activations by sqrt(fan_in) per layer, until fp16 noise decides the argmax
    if tensor.is_floating_point() and tensor.dim() >= 2:
        tensor.mul_(tensor.shape[-1] ** -0.5)
    return tensor


dummy_state = E._make_dummy_weight_state_dict


def dummy_scaled(*args, **kw):
    state = {name: fan_in_scaled(tensor) for name, tensor in dummy_state(*args, **kw).items()}
    if norm_ones:
        # the gemma4 draft reproduces the target's head over its post-norm hidden, so that hidden has to
        # stay normed after a second norm; an all-ones final scale makes that second norm the identity
        state["model.norm.weight"] = torch.ones_like(state["model.norm.weight"])
    if dump_weights:
        # the seeded dummy tensors are the target's weights; the MTP run loads them as a real checkpoint.
        # gemma4's loader only renames keys under model.language_model., so those need that prefix.
        from safetensors.torch import save_file

        def key(name):
            return f"{dump_prefix}{name.removeprefix('model.')}" if dump_prefix else name

        save_file({key(k): v.detach().cpu().contiguous() for k, v in state.items()}, dump_weights)
    return state


E._make_dummy_weight_state_dict = dummy_scaled
create_model = E.create_model


def create_model_with_ple_rows(*args, **kwargs):
    # imported here, once CUDA is up
    from freetoken.models.qwen4_exp import ple
    from freetoken.moe import expert_banks

    dummy_fill = expert_banks._dummy_fill

    def scaled_fill(role, tensor):
        dummy_fill(role, tensor)
        if not role.endswith(("_scale", "_global")):
            fan_in_scaled(tensor)

    expert_banks._dummy_fill = scaled_fill

    # seeded PLE rows in place of the dummy zeros, so the PLE path carries signal
    class RandomTable(ple.GpuResidentTable):
        def __init__(self, num_rows, head_dim, dtype=torch.bfloat16):
            generator = torch.Generator(device="cuda").manual_seed(0)
            super().__init__(torch.randn(num_rows, head_dim, generator=generator, device="cuda", dtype=dtype))

    ple.ZeroTable = RandomTable
    model = create_model(*args, **kwargs)
    # gemma4's decoder layers hand the sink nothing, and their norms take no residual to tap
    if reference != "-" and family != "gemma4":
        tap_next_layer_inputs(model)
    return model


E.create_model = create_model_with_ple_rows
if reference != "-":
    PROPOSERS.register("oracle")("__main__:OracleProposer")
llm = LLM(model_path=model, dtype=torch.float16, use_dummy_weight=use_dummy_weight, max_running_req=4, **kwargs)
prompts = [llm.tokenizer.encode(p) for p in prompt_texts]
penalties = (0.0, 0.6, 0.0)
params = [SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, presence_penalty=p, frequency_penalty=p)
          for n, p in zip((48, 37, 41), penalties)]
res = llm.generate(prompts, params)
pool = llm.engine.linear_state_pool
window = list(pool.slot_state("dflash_window").shape) if pool is not None and pool.has_slot_state("dflash_window") else None
cache = llm.cache_manager
json.dump({"outputs": [r["token_ids"] for r in res], "top3": [[u, i, *t] for (u, i), t in top3.items()], "verify_steps": steps["verify"],
           "taps": taps, "num_slots": None if pool is None else pool.num_slots, "window": window,
           "swa_window": cache.sliding_window_size, "swa": cache.swa_paged, "rows": llm.config.decode_rows_per_req,
           "prompt_lens": [len(p) for p in prompts], "penalties": list(penalties),
           "evict": evict, "proposals": proposals,
           "draft": {"head_calls": draft["head_calls"], "reads": sorted(draft["reads"].items())}}, open(out, "w"))
"""


def _tiny_model(path: str, family: str) -> str:
    """A tiny target of one family; gemma4 is the dense hybrid sliding-window/full release shape with a 16-position window."""
    os.makedirs(path, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if os.path.exists(os.path.join(TOKENIZER_SOURCE, name)):
            shutil.copy(os.path.join(TOKENIZER_SOURCE, name), path)
    text = {
        "vocab_size": 248320, "hidden_size": 256, "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "full_attention_interval": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 32, "linear_num_key_heads": 2,
        "linear_num_value_heads": 4, "linear_value_head_dim": 32, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "max_position_embeddings": 4096, "tie_word_embeddings": False, "eos_token_id": 248044,
    }
    if family == "qwen3_5":
        config = {
            "architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text", **text,
            "intermediate_size": 512, "head_dim": 64, "attn_output_gate": True,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                                "mrope_section": [3, 3, 2], "mrope_interleaved": True},
        }
    elif family == "gemma4":
        # rope head sizes are 64 / 128 / 256 / 512; the window is shorter than the windowed prompts below
        config = {
            "architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4_text",
            "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 4,
            "num_attention_heads": 4, "head_dim": 64, "num_key_value_heads": 2,
            "global_head_dim": 128, "num_global_key_value_heads": 2,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "sliding_window": 16, "vocab_size": 248320, "rms_norm_eps": 1e-6, "max_position_embeddings": 4096,
            "tie_word_embeddings": True, "eos_token_id": 248044, "final_logit_softcapping": 30.0,
            "hidden_size_per_layer_input": 0, "vocab_size_per_layer_input": 0,
            "num_kv_shared_layers": 0, "use_double_wide_mlp": False,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 10000.0},
            },
        }
    else:
        # the shipping QSA geometry (head_dim 256, index head_dim 128, ratio 4), scaled-down widths
        text.update({
            "model_type": "qwen4_exp_text", "head_dim": 256,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                                "mrope_section": [11, 11, 10], "mrope_interleaved": True},
            "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64,
            "output_gate_type": "sigmoid", "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128,
            "indexer_budget": 2048, "indexer_compress_ratio": 4, "hc_count": 4, "hc_lowrank": 16,
            "ple_layer_ids": [2], "ple_embed_dim": 64, "ple_conv_kernel_size": 4, "ngram_size": 3, "heads_per_ngram": 2,
            "ngram_vocab_size_base": 1000, "make_ngram_vocab_size_divisible_by": 8, "split_ngram_parts": 4,
        })
        config = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
                  "tie_word_embeddings": False, "text_config": text}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    return path


def _tiny_draft(path: str) -> str:
    """A DFlash 2 draft of the tiny targets: two layers over target layers 1 and 2, a 16-position window, random fan-in scaled weights."""
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    hidden, layers, heads, kv_heads, head_dim, inter, group = 256, 2, 4, 2, 64, 512, 16
    config = {
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "is_causal": False, "hidden_size": hidden,
        "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv_heads, "head_dim": head_dim,
        "intermediate_size": inter, "hidden_act": "silu", "rms_norm_eps": 1e-6, "vocab_size": 248320, "num_target_layers": 4,
        "layer_types": ["sliding_attention"] * layers, "use_sliding_window": True, "sliding_window": 16, "max_position_embeddings": 4096,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "dflash_config": {"block_size": 8, "conv_kernel_size": 2, "conv_group_size": group, "mask_token_id": 248300,
                          "selector_rank": 8, "selector_top_k": 4, "target_layer_ids": [1, 2]},
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    shapes = {"fc.weight": (hidden, 2 * hidden), "hidden_norm.weight": (hidden,), "norm.weight": (hidden,),
              "candidate_selector.hidden_projection.weight": (8, hidden),
              "candidate_selector.predecessor_codebook": (248320, 8), "candidate_selector.successor_codebook": (248320, 8)}
    for i in range(layers):
        p = f"layers.{i}"
        shapes.update({
            f"{p}.self_attn.q_proj.weight": (heads * head_dim, hidden), f"{p}.self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
            f"{p}.self_attn.v_proj.weight": (kv_heads * head_dim, hidden), f"{p}.self_attn.o_proj.weight": (hidden, heads * head_dim),
            f"{p}.self_attn.q_norm.weight": (head_dim,), f"{p}.self_attn.k_norm.weight": (head_dim,),
            f"{p}.mlp.gate_proj.weight": (inter, hidden), f"{p}.mlp.up_proj.weight": (inter, hidden), f"{p}.mlp.down_proj.weight": (hidden, inter),
            f"{p}.input_layernorm.weight": (hidden,), f"{p}.post_attention_layernorm.weight": (hidden,),
        })
        for conv in ("attention_conv", "mlp_conv"):
            shapes.update({f"{p}.{conv}.base_kernel": (2, 2, hidden), f"{p}.{conv}.kernel_projection.weight": (4 * hidden // group, hidden)})
    g = torch.Generator().manual_seed(0)
    weights = {}
    for name, shape in shapes.items():
        t = torch.randn(*shape, generator=g)
        weights[name] = (t * shape[-1] ** -0.5 if len(shape) > 1 else 1 + 0.1 * t).to(torch.bfloat16)
    save_file(weights, os.path.join(path, "model.safetensors"))
    return path


def _tiny_mtp_draft(path: str, target: str, family: str = "qwen3_5") -> str:
    """The tiny target's own MTP draft as a checkpoint.

    qwen3_5: one draft attention layer over random fan-in scaled weights, written in the pre-fusion HF names
    iter_mtp_weights consumes (q_proj carries the output gate). gemma4: the assistant sidecar GGUF, whose four
    KV-shared layers are the identity and whose head is twice the target's tied table, so the draft still
    reads the target's slabs and its own head predicts the target's greedy token.
    """
    from safetensors.torch import load_file, save_file

    os.makedirs(path, exist_ok=True)
    if family == "gemma4":
        cfg = json.load(open(os.path.join(target, "config.json")))
        hidden, heads, kv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        swa_hd, full_hd, window = cfg["head_dim"], cfg["global_head_dim"], cfg["sliding_window"]
        ffn = 2 * hidden
        embed = load_file(os.path.join(target, "model.safetensors"))["model.language_model.embed_tokens.weight"]
        tensors = {}

        def put(name, tensor, ggml_type=BF16):
            tensor = tensor.to(torch.bfloat16) if ggml_type == BF16 else tensor.float()
            tensors[name] = (ggml_type, tuple(reversed(tensor.shape)), _gguf_bytes(tensor, ggml_type))

        def zeros(*shape):
            return torch.zeros(*shape)

        # twice the target's head: the same argmax, and a value only the draft's own table can produce
        put("token_embd.weight", 2 * embed)
        freqs = torch.full((full_hd // 2,), 1e30)
        freqs[: full_hd // 8] = 1.0  # partial rotary 0.25, as the full layer's rope_freqs
        put("rope_freqs.weight", freqs, F32)
        pre = zeros(hidden, 2 * hidden)
        pre[:, hidden:] = torch.eye(hidden)  # keep the target hidden, drop the token embedding
        put("nextn.pre_projection.weight", pre)
        put("nextn.post_projection.weight", torch.eye(hidden))
        put("output_norm.weight", torch.ones(hidden), F32)
        for layer in range(DRAFT_LAYERS):
            hd = full_hd if layer == 3 else swa_hd
            blk = f"blk.{layer}"
            put(f"{blk}.attn_q.weight", zeros(heads * hd, hidden))
            put(f"{blk}.attn_output.weight", zeros(hidden, heads * hd))
            put(f"{blk}.ffn_gate.weight", zeros(ffn, hidden))
            put(f"{blk}.ffn_up.weight", zeros(ffn, hidden))
            put(f"{blk}.ffn_down.weight", zeros(hidden, ffn))
            for name, width in (("attn_norm", hidden), ("post_attention_norm", hidden), ("ffn_norm", hidden),
                                ("post_ffw_norm", hidden), ("attn_q_norm", hd)):
                put(f"{blk}.{name}.weight", torch.ones(width), F32)
            put(f"{blk}.layer_output_scale.weight", torch.ones(1), F32)
        metadata = {
            "block_count": DRAFT_LAYERS, "context_length": cfg["max_position_embeddings"], "embedding_length": hidden,
            "feed_forward_length": ffn, "attention.head_count": heads, "attention.head_count_kv": [kv] * DRAFT_LAYERS,
            "rope.freq_base": 10000.0, "rope.freq_base_swa": 10000.0,
            "attention.layer_norm_rms_epsilon": cfg["rms_norm_eps"],
            "attention.key_length": full_hd, "attention.value_length": full_hd,
            "attention.key_length_swa": swa_hd, "attention.value_length_swa": swa_hd,
            "attention.sliding_window": window,
            "embedding_length_per_layer_input": 0, "attention.sliding_window_pattern": [1] * (DRAFT_LAYERS - 1) + [0],
            "rope.dimension_count_swa": swa_hd,
            "embedding_length_out": hidden, "nextn_predict_layers": DRAFT_LAYERS,
        }
        draft = os.path.join(path, "assistant.gguf")
        _write_gguf(Path(draft), "gemma4-assistant", metadata, tensors)
        return draft

    shutil.copy(os.path.join(target, "config.json"), os.path.join(path, "config.json"))
    hidden, heads, kv_heads, head_dim, inter = 256, 4, 2, 64, 512
    shapes = {
        "mtp.fc.weight": (hidden, 2 * hidden), "mtp.pre_fc_norm_embedding.weight": (hidden,),
        "mtp.pre_fc_norm_hidden.weight": (hidden,), "mtp.norm.weight": (hidden,),
        "mtp.layers.0.self_attn.q_proj.weight": (2 * heads * head_dim, hidden),
        "mtp.layers.0.self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
        "mtp.layers.0.self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
        "mtp.layers.0.self_attn.o_proj.weight": (hidden, heads * head_dim),
        "mtp.layers.0.self_attn.q_norm.weight": (head_dim,), "mtp.layers.0.self_attn.k_norm.weight": (head_dim,),
        "mtp.layers.0.mlp.gate_proj.weight": (inter, hidden), "mtp.layers.0.mlp.up_proj.weight": (inter, hidden),
        "mtp.layers.0.mlp.down_proj.weight": (hidden, inter),
        "mtp.layers.0.input_layernorm.weight": (hidden,), "mtp.layers.0.post_attention_layernorm.weight": (hidden,),
    }
    g = torch.Generator().manual_seed(0)
    weights = {}
    for name, shape in shapes.items():
        t = torch.randn(*shape, generator=g)
        weights[name] = (t * shape[-1] ** -0.5 if len(shape) > 1 else 1 + 0.1 * t).to(torch.bfloat16)
    save_file(weights, os.path.join(path, "model.safetensors"))
    return path


def _run(model: str, out: str, reference: str, **kwargs) -> dict:
    run = subprocess.run([sys.executable, "-c", _SCRIPT, model, out, reference, json.dumps(kwargs)],
                         capture_output=True, text=True, timeout=900, env=dict(os.environ))
    assert run.returncode == 0, run.stderr[-3000:]
    with open(out) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def plain_runs(tmp_path_factory):
    # the plain greedy run of each (family, backend, graphs, prompt/norm variant), shared by every algorithm: model dir, output path, result
    runs = {}

    def get(family, backend, graphs, kwargs):
        # the prompts, the dummy-weight knobs and the eviction interval change the reference
        variant = json.dumps({k: kwargs.get(k) for k in ("prompts", "norm_ones")}, sort_keys=True)
        variant += os.environ.get("FREETOKEN_SWA_EVICTION_INTERVAL", "")
        key = (family, backend, graphs, variant)
        if key not in runs:
            root = tmp_path_factory.mktemp(f"plain-{family}-{backend}-{graphs}")
            model = _tiny_model(str(root / "model"), family)
            runs[key] = model, str(root / "decode.json"), _run(model, str(root / "decode.json"), "-", **kwargs)
        return runs[key]

    return get


@pytest.mark.parametrize("family, backend, algorithm, graphs", [
    *[(f, b, "oracle", g) for f, b in [("qwen3_5", "fa"), ("qwen3_5", "triton"), ("qwen4_exp", "qsa_sparse")] for g in (True, False)],
    # the draft side once per family, with graphs
    ("qwen3_5", "triton", "dflash", True), ("qwen4_exp", "qsa_sparse", "dflash", True),
    ("qwen3_5", "triton", "mtp", True),
    # a windowed target: the oracle case frees out-of-window slots on every verify step, the mtp case
    # drafts with the read-only gemma4-assistant sidecar
    ("gemma4", "triton", "oracle", True), ("gemma4", "triton", "mtp", True),
])
def test_verify_steps_reproduce_greedy_decode(tmp_path, plain_runs, monkeypatch, family, backend, algorithm, graphs):
    if backend == "fa":
        pytest.importorskip("sgl_kernel")
    # resident experts take the model dtype
    kwargs = {"attention_backend": backend, "cuda_graph_max_bs": 4 if graphs else 0, "moe_strategy": "fused", "family": family}
    if family == "gemma4":
        kwargs["norm_ones"] = True
    if family == "gemma4" and algorithm == "oracle":
        # a window shorter than each prompt, evicted every two tokens: the verify rows, the rollback and the frees all run
        kwargs["prompts"] = WINDOWED_PROMPTS
        monkeypatch.setenv("FREETOKEN_SWA_EVICTION_INTERVAL", "2")
    model, decode, reference = plain_runs(family, backend, graphs, kwargs)
    draft = {"speculative_draft_model_path": _tiny_draft(str(tmp_path / "draft"))} if algorithm == "dflash" else {}
    extra = {}
    if algorithm == "mtp":
        # MTP rejects dummy weights: write the same seeded dummy tensors a plain run would generate into a
        # per-test copy of the target, never the fixture's shared model dir
        target = str(tmp_path / "target")
        shutil.copytree(model, target)
        _run(model, str(tmp_path / "materialize.json"), "-", dump_weights=os.path.join(target, "model.safetensors"),
             dump_prefix="model.language_model." if family == "gemma4" else "", **kwargs)
        draft = {"speculative_draft_model_path": _tiny_mtp_draft(str(tmp_path / "draft"), target, family)}
        extra = {"use_dummy_weight": False}
        model = target
    spec = _run(model, str(tmp_path / "spec.json"), decode, speculative_algorithm=algorithm, speculative_num_steps=K,
                **draft, **extra, **kwargs)
    if algorithm == "mtp" and family == "gemma4":
        # the draft must earn its keep: its own head produced every draft (k per proposal), and each
        # forward read the target's slabs through the view (3 SWA layers -> 2, the full layer -> 3)
        assert spec["proposals"] and spec["draft"]["head_calls"] == len(spec["proposals"]) * K, spec["draft"]
        # the identity sidecar proposes the pending token itself on any machine, so the proposals pin
        # the seed row: a hidden stashed from another verify row proposes that row's token instead.
        # the run's own dumped penalties say which requests are penalized; their pending token need
        # not be that row's argmax, and a row whose candidates sit within two fp16 ulps may still
        # flip under the head's own tiling
        for call in spec["proposals"]:
            for uid, pending, steps in call:
                if spec["penalties"][uid]:
                    continue
                for ids, vals in steps:
                    by_id = dict(zip(ids, vals))
                    ulp = 2.0 ** (math.floor(math.log2(abs(vals[0]) or 1.0)) - 10)
                    assert ids[0] == pending or (pending in by_id and by_id[ids[0]] - by_id[pending] <= 2 * ulp), (
                        uid, pending, ids, vals)
        cfg = json.load(open(os.path.join(model, "config.json")))
        swa = [i for i, t in enumerate(cfg["layer_types"]) if t == "sliding_attention"]
        last_swa, last_full = max(swa), cfg["num_hidden_layers"] - 1
        reads = dict(spec["draft"]["reads"])
        assert set(reads) == {last_swa, last_full} and reads[last_swa] == (DRAFT_LAYERS - 1) * reads[last_full], reads
    agreed = 0
    for uid, (want, got) in enumerate(zip(reference["outputs"], spec["outputs"])):
        assert len(got) == len(want)
        split = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), len(want))
        agreed += split
        if split < len(want):
            # same prefix, so both runs must hold the two tokens inside one top-3 tie within two fp16
            # ulps: verify's differently-shaped GEMMs can reorder a third candidate an ulp behind
            pair = {want[split], got[split]}
            for run in (reference, spec):
                a, b, c, va, vb, vc = next(t[2:] for t in run["top3"] if t[:2] == [uid, split])
                ulp = 2.0 ** (math.floor(math.log2(abs(va) or 1.0)) - 10)
                by_id = {a: va, b: vb, c: vc}
                assert pair <= set(by_id) and max(by_id[t] for t in pair) - min(by_id[t] for t in pair) <= 2 * ulp, (
                    f"request {uid} parts at {split}: top-3 {(a, va), (b, vb), (c, vc)}")
    # most of the run must still be compared, or a tie early in every request would hide a bug
    assert agreed >= 0.5 * sum(map(len, reference["outputs"]))
    assert spec["verify_steps"] > 0
    if algorithm == "dflash":
        # the draft's context window rides the target's GDN slots: per slot, rows of the draft width covering its 16-position window and a block
        window = spec["window"]
        assert window is not None and window[0] == spec["num_slots"] and window[1] >= 16 + K + 1 and window[2] == 256
    elif algorithm == "mtp":
        # MTPProposer installs its own prefill and verify sinks, so the recording sink must have seen no target forward
        assert spec["taps"]["prefill"] == 0 and spec["taps"]["verify"] == 0
    elif family == "gemma4":
        # the window machinery ran: prompt longer than the window, slots freed out of window on verify steps
        # and the rejected verify rows' slots returned
        assert spec["swa"] and spec["rows"] == K + 1
        assert min(spec["prompt_lens"]) > spec["swa_window"]
        assert spec["evict"]["decode"] > 0 and spec["evict"]["rollback"] > 0 and spec["evict"]["slots"] > 0, spec["evict"]
    else:
        # every prefill and verify forward handed the sink each tapped layer's output, as the next layer read it
        taps = spec["taps"]
        assert taps["prefill"] > 0 and taps["verify"] > 0 and taps["mismatch"] == 0
