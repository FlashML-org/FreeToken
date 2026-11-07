"""Ripple-65B against a transcription of the checkpoint's modeling_ricursive.py and a ModelOpt checkpoint layout."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
import torch.nn.functional as F

from freetoken.layers.quantization import set_quant_config
from freetoken.models.register import checkpoint_quant_config, get_model_spec
from freetoken.models.ricursive.config import parse_config
from freetoken.models.ricursive.model import RicursiveForCausalLM
from freetoken.models.ricursive.weight import iter_weights
from freetoken.utils import cached_load_hf_config

H, INTER, VOCAB, HEADS, KV_HEADS, HEAD_DIM, WINDOW, CLAMP, EPS, SCALE = 128, 256, 64, 4, 2, 64, 8, 0.75, 1e-6, 11.25
LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention"]


def _hf_dict(**overrides) -> dict:
    return dict(
        architectures=["RicursiveForCausalLM"], model_type="ricursive", num_hidden_layers=len(LAYER_TYPES),
        hidden_size=H, intermediate_size=INTER, vocab_size=VOCAB, num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM, layer_types=LAYER_TYPES, hidden_act="silu",
        sliding_window=WINDOW, max_position_embeddings=2097152, rope_theta=10000.0, rms_norm_eps=EPS,
        embed_scale=SCALE, ffn_clamp_threshold=CLAMP, use_gated_attention=True, gate_full_precision=True,
        use_valres=True, tie_word_embeddings=False,
    ) | overrides


def _rms(x, weight):
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def _attend(q, k, v, window):
    """[T, heads, D] q and [T, kv_heads, D] k, v through the reference's SDPA and sliding-window mask."""
    T = q.shape[0]
    pos = torch.arange(T, device=q.device)
    mask = pos[None] <= pos[:, None]
    if window is not None:
        mask &= pos[None] >= pos[:, None] - (window - 1)
    group = q.shape[1] // k.shape[1]
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.repeat_interleave(group, 1).transpose(0, 1), v.repeat_interleave(group, 1).transpose(0, 1),
        attn_mask=mask,
    )
    return out.transpose(0, 1)


def _reference_logits(sd: dict, lambdas: list[torch.Tensor], ids: torch.Tensor, clamp: float) -> torch.Tensor:
    """RicursiveForCausalLM.forward with q, k, v, g and gate, up as the checkpoint stores them."""
    x = sd["model.embed_tokens.weight"][ids] * SCALE
    value_residual = x
    T = ids.shape[0]
    half = HEAD_DIM // 2
    inv_freq = 1.0 / 10000.0 ** (torch.arange(0, HEAD_DIM, 2, device=x.device).float() / HEAD_DIM)
    freqs = torch.arange(T, device=x.device).float()[:, None] * inv_freq[None]
    emb = torch.cat((freqs, freqs), dim=-1)[:, None]
    assert emb.shape[-1] == 2 * half
    for i, kind in enumerate(LAYER_TYPES):
        p, a = f"model.layers.{i}", f"model.layers.{i}.self_attn"
        normed = _rms(x, sd[f"{p}.input_layernorm.weight"])
        q_all, k_all = (normed @ sd[f"{a}.qk_proj.weight"].T).split((HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM), dim=-1)
        v = torch.lerp(value_residual, normed, torch.sigmoid(lambdas[i])) @ sd[f"{a}.v_proj.weight"].T
        q = _rms(q_all.view(T, HEADS, HEAD_DIM), sd[f"{a}.q_norm.weight"])
        k = _rms(k_all.view(T, KV_HEADS, HEAD_DIM), sd[f"{a}.k_norm.weight"])
        if kind == "sliding_attention":
            q = q * emb.cos() + _rotate_half(q) * emb.sin()
            k = k * emb.cos() + _rotate_half(k) * emb.sin()
        out = _attend(q, k, v.view(T, KV_HEADS, HEAD_DIM), WINDOW if kind == "sliding_attention" else None)
        gated = out.reshape(T, -1) * torch.sigmoid(normed @ sd[f"{a}.g_proj.weight"].T)
        x = x + _rms(gated @ sd[f"{a}.o_proj.weight"].T, sd[f"{p}.post_attention_layernorm.weight"])
        normed = _rms(x, sd[f"{p}.pre_feedforward_layernorm.weight"])
        gate, up = (normed @ sd[f"{p}.mlp.gate_up_proj.weight"].T).chunk(2, dim=-1)
        if clamp:
            gate, up = gate.clamp(max=clamp), up.clamp(-clamp, clamp)
        ff = F.silu(gate) * up
        x = x + _rms(ff @ sd[f"{p}.mlp.down_proj.weight"].T, sd[f"{p}.post_feedforward_layernorm.weight"])
    return _rms(x, sd["model.norm.weight"]) @ sd["lm_head.weight"].T


class _SDPABackend:
    def forward(self, q, k, v, layer_id, batch, attn_spec=None):
        T = q.shape[0]
        return _attend(q, k.view(T, KV_HEADS, HEAD_DIM), v.view(T, KV_HEADS, HEAD_DIM), attn_spec.sliding_window)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused CUDA norm and activation kernels")
@pytest.mark.parametrize("clamp", [CLAMP, 0])  # the reference skips a zero clamp
def test_model_matches_the_reference_on_every_row(monkeypatch, clamp):
    from freetoken.distributed import info

    monkeypatch.setattr(info, "_TP_INFO", info.DistributedInfo(rank=0, size=1))
    prev_dtype, prev_device = torch.get_default_dtype(), torch.get_default_device()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    try:
        model = RicursiveForCausalLM(parse_config(SimpleNamespace(**_hf_dict(ffn_clamp_threshold=clamp))))
        torch.manual_seed(0)
        sd = {k: (torch.randn(v.shape, device="cuda") * 0.1).to(v.dtype) for k, v in model.state_dict().items()}
        lambdas = [torch.randn(1, device="cuda").to(torch.bfloat16) for _ in LAYER_TYPES]
        for name in sd:
            if name.endswith("norm.weight"):
                sd[name] = sd[name].abs() + 0.5
        for i, lam in enumerate(lambdas):
            sd[f"model.layers.{i}.self_attn.value_res_blend"] = torch.sigmoid(lam)
        model.load_state_dict(dict(sd))

        ids = torch.randint(0, VOCAB, (3 * WINDOW,), device="cuda")
        batch = SimpleNamespace(
            input_ids=ids, size=1, is_prefill=False, features=None, get_attn_positions=lambda: torch.arange(ids.numel(), device="cuda"),
        )
        ctx = SimpleNamespace(batch=batch, attn_backend=_SDPABackend())
        for mod in ("attention", "model"):
            monkeypatch.setattr(f"freetoken.models.ricursive.{mod}.get_global_ctx", lambda: ctx)
        monkeypatch.setattr("freetoken.layers.embedding.get_global_ctx", lambda: ctx)
        got = model.forward().float()
        torch.set_default_dtype(torch.float32)
        want = _reference_logits({k: v.float() for k, v in sd.items()}, [lam.float() for lam in lambdas], ids, clamp)
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.set_default_device(prev_device)
    # What is left is the model's bf16 rounding against the fp32 reference, row by row so one bad position shows.
    assert (got - want).abs().mean(-1).max() < 0.02 * want.std(), ((got - want).abs().mean(-1).max(), want.std())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the attention gate runs on the GPU")
def test_bf16_sigmoid_is_the_fp32_sigmoid_the_reference_asks_for():
    x = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.uint16).view(torch.bfloat16)
    assert torch.equal(torch.sigmoid(x).nan_to_num(), torch.sigmoid(x.float()).to(torch.bfloat16).nan_to_num())


def _modelopt_checkpoint(path, attention_quantized: bool) -> tuple[dict, dict]:
    ignore = ["lm_head", "model.embed_tokens"]
    if not attention_quantized:
        ignore += [f"model.layers.{i}.self_attn*" for i in range(len(LAYER_TYPES))]
    quant = {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ignore,
             "config_groups": {"group_0": {"input_activations": {"num_bits": 4}, "weights": {"num_bits": 4}, "targets": ["Linear"]}}}
    (path / "config.json").write_text(json.dumps(_hf_dict(quantization_config=quant)))
    torch.manual_seed(0)
    bf16 = lambda *shape: torch.randn(*shape).to(torch.bfloat16)
    def nvfp4(prefix, rows, cols, scale):
        return {
            f"{prefix}.weight": torch.randint(0, 255, (rows, cols // 2), dtype=torch.uint8),
            f"{prefix}.weight_scale": torch.ones(rows, cols // 16).to(torch.float8_e4m3fn),
            f"{prefix}.weight_scale_2": torch.tensor(scale),
            f"{prefix}.input_scale": torch.tensor(scale * 2),
        }
    sd = {"model.embed_tokens.weight": bf16(VOCAB, H), "lm_head.weight": bf16(VOCAB, H), "model.norm.weight": bf16(H)}
    for i in range(len(LAYER_TYPES)):
        p, a = f"model.layers.{i}", f"model.layers.{i}.self_attn"
        for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            sd[f"{p}.{norm}.weight"] = bf16(H)
        sd |= {f"{a}.q_norm.weight": bf16(HEAD_DIM), f"{a}.k_norm.weight": bf16(HEAD_DIM), f"{a}.value_res_lambda": bf16(1)}
        for leaf, rows, cols in (("q", HEADS * HEAD_DIM, H), ("k", KV_HEADS * HEAD_DIM, H), ("v", KV_HEADS * HEAD_DIM, H), ("g", HEADS * HEAD_DIM, H), ("o", H, HEADS * HEAD_DIM)):
            sd |= nvfp4(f"{a}.{leaf}_proj", rows, cols, 0.25) if attention_quantized else {f"{a}.{leaf}_proj.weight": bf16(rows, cols)}
        sd |= nvfp4(f"{p}.mlp.gate_proj", INTER, H, 2.0 ** -i) | nvfp4(f"{p}.mlp.up_proj", INTER, H, 2.0 ** -i)
        sd |= nvfp4(f"{p}.mlp.down_proj", H, INTER, 0.5)
    safetensors.torch.save_file(sd, path / "model.safetensors")
    return sd, quant


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the NVFP4 linear picks its kernel from the device")
@pytest.mark.parametrize("attention_quantized", [False, True])
def test_modelopt_checkpoint_fills_the_model_the_quant_config_builds(tmp_path, monkeypatch, attention_quantized):
    from freetoken.distributed import info

    monkeypatch.setattr(info, "_TP_INFO", info.DistributedInfo(rank=0, size=1))
    stored, _ = _modelopt_checkpoint(tmp_path, attention_quantized)
    hf = cached_load_hf_config(str(tmp_path))
    quant = checkpoint_quant_config(str(tmp_path), hf, get_model_spec("RicursiveForCausalLM"))
    set_quant_config(quant)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = RicursiveForCausalLM(replace(parse_config(hf), quant=quant))
        got = dict(iter_weights(str(tmp_path), torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    finally:
        torch.set_default_dtype(prev_dtype)
    want = model.state_dict()
    assert set(got) == set(want)
    assert all(got[k].shape == want[k].shape and got[k].dtype == want[k].dtype for k in want)
    a = "model.layers.1.self_attn"
    assert torch.equal(got[f"{a}.qk_proj.weight"], torch.cat([stored[f"{a}.q_proj.weight"], stored[f"{a}.k_proj.weight"]]))
    if attention_quantized:
        assert torch.equal(got[f"{a}.g_proj.weight_global"], torch.full((HEADS * HEAD_DIM,), 0.25))
    assert got[f"{a}.value_res_blend"].item() == torch.sigmoid(stored[f"{a}.value_res_lambda"].float()).to(torch.bfloat16).item()
    mlp = "model.layers.1.mlp.gate_up_proj"
    assert torch.equal(got[f"{mlp}.weight_global"], torch.full((2 * INTER,), 0.5))
    assert got[f"{mlp}.input_scale"].item() == 1.0


def _gguf_shim(monkeypatch, clamp, layer_types=LAYER_TYPES):
    import freetoken.models.ricursive.gguf as ricursive_gguf
    from freetoken.models.gguf.config import GgufConfigShim

    monkeypatch.setattr(ricursive_gguf, "check_scales", lambda path: None)
    md = {f"ricursive.{k}": v for k, v in {
        "block_count": len(layer_types), "embedding_length": H, "feed_forward_length": INTER,
        "attention.head_count": HEADS, "attention.head_count_kv": KV_HEADS, "attention.key_length": HEAD_DIM,
        "context_length": 2097152, "attention.layer_norm_rms_epsilon": EPS, "rope.freq_base": 10000.0,
        "attention.sliding_window_pattern": [kind == "sliding_attention" for kind in layer_types],
        "attention.sliding_window": WINDOW, "embedding_scale": SCALE, "swiglu_clamp": clamp,
    }.items()}
    return GgufConfigShim(architectures=["RicursiveGGUFForCausalLM"], model_path="", model_type="ricursive",
                          metadata=md, vocab_size=VOCAB, tie_word_embeddings=False)


# the first community GGUFs store one clamp, llama.cpp's converter one per layer
@pytest.mark.parametrize("clamp, hf_clamp", [(CLAMP, CLAMP), ([CLAMP] * len(LAYER_TYPES), CLAMP), ([0.0] * len(LAYER_TYPES), 0)])
def test_gguf_config_matches_the_checkpoint_config(monkeypatch, clamp, hf_clamp):
    from freetoken.models.ricursive.gguf import parse_gguf_config

    want = replace(parse_config(SimpleNamespace(**_hf_dict(ffn_clamp_threshold=hf_clamp))), architectures=["RicursiveGGUFForCausalLM"])
    assert parse_gguf_config(_gguf_shim(monkeypatch, clamp)) == want


def test_gguf_reads_the_keys_llama_cpp_treats_as_optional(monkeypatch):
    """Absent: no clamp, no embedding scale, and every fifth layer global, as llama.cpp loads such a file."""
    from freetoken.models.ricursive.gguf import parse_gguf_config

    types = ["sliding_attention"] * 4 + ["full_attention"]
    shim = _gguf_shim(monkeypatch, CLAMP, types)
    for key in ("swiglu_clamp", "embedding_scale", "attention.sliding_window_pattern"):
        del shim.metadata[f"ricursive.{key}"]
    hf = _hf_dict(num_hidden_layers=len(types), layer_types=types, ffn_clamp_threshold=None, embed_scale=None)
    assert parse_gguf_config(shim) == replace(parse_config(SimpleNamespace(**hf)), architectures=["RicursiveGGUFForCausalLM"])


def test_gguf_refuses_per_layer_clamps_that_differ(monkeypatch):
    from freetoken.models.ricursive.gguf import parse_gguf_config

    with pytest.raises(NotImplementedError, match="swiglu_clamp"):
        parse_gguf_config(_gguf_shim(monkeypatch, [CLAMP, CLAMP, 2 * CLAMP]))


@pytest.mark.needs_weights
def test_gguf_matches_the_checkpoint_tokenizer_and_config():
    from transformers import AutoTokenizer

    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.models.gguf.tokenizer import gguf_eos_token_ids, load_gguf_tokenizer
    from freetoken.models.ricursive.gguf import parse_gguf_config

    gguf, model = os.environ.get("FREETOKEN_RIPPLE_GGUF"), os.environ.get("FREETOKEN_RIPPLE_MODEL")
    if not gguf or not model:
        pytest.skip("set FREETOKEN_RIPPLE_GGUF and FREETOKEN_RIPPLE_MODEL")
    hf, gg = AutoTokenizer.from_pretrained(model), load_gguf_tokenizer(gguf)
    for text in ("x = 1234567890 + 3.14159; I'M here, you'RE", "a  b\n\n\n  d\t\te \r\n f", "caf\u00e9 \u65e5\u672c\u8a9e \U0001f642", "<|im_start|>user\nHi<|im_end|>\n<think>\nok\n</think>"):
        assert gg.encode(text, add_special_tokens=False) == hf.encode(text, add_special_tokens=False), text
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    messages = [
        {"role": "user", "content": "Read it."},
        {"role": "assistant", "content": "", "reasoning_content": "Look first.",
         "tool_calls": [{"type": "function", "function": {"name": "read", "arguments": {"path": "a.py"}}}]},
        {"role": "tool", "content": "print(42)"},
    ]
    assert gg.apply_chat_template(messages, tools=tools, add_generation_prompt=True) == hf.apply_chat_template(messages, tools=tools, add_generation_prompt=True)
    assert 100265 in gguf_eos_token_ids(gguf)
    want, got = parse_config(cached_load_hf_config(model)), parse_gguf_config(build_gguf_shim(gguf))
    for name in ("num_layers", "num_qo_heads", "num_kv_heads", "head_dim", "hidden_size", "intermediate_size", "vocab_size", "swiglu_limit", "attention_groups"):
        assert getattr(got, name) == getattr(want, name), name
    assert got.rms_norm_eps == pytest.approx(want.rms_norm_eps)
    assert got.embedding_scale == pytest.approx(want.embedding_scale)


_CALL = '<tool_call>\n{"name": "read", "arguments": {"path": "a.py"}}\n</tool_call>'
# Ripple's output in the layout chat_template.jinja renders an assistant turn back to
_TURNS = (
    "Wrap it in a <think> tag.",
    "<think>\nOne line.\n</think>\nUse <think> here.",
    "<think>\nline one\n\nline two\n</think>\nDone.\n\nBye.",
    "<think>\nRead it first.\n</think>\n\n" + _CALL,
    "<think>\nCheck.\n</think>\nLet me check.\n" + _CALL,
    "<think>\nRead both.\n</think>\n\n" + _CALL + "\n" + _CALL.replace("a.py", "b.py"),
)


def _serve(text: str, tools: list, chunk: int | None):
    """(reasoning, content, calls) the server returns for ``text``, streamed in ``chunk``-char deltas or not."""
    import asyncio

    from freetoken.message.frontend import UserReply
    from freetoken.server.generation import (
        ContentDelta, GenSpec, ReasoningDelta, ToolCallsDelta, generate_events, generate_full,
    )

    pieces = [text] if chunk is None else [text[i:i + chunk] for i in range(0, len(text), chunk)]

    class State:
        config = SimpleNamespace(tool_call_parser="ricursive", reasoning_parser="ricursive")

        async def wait_for_ack(self, uid):
            for i, piece in enumerate(pieces):
                yield UserReply(uid=uid, incremental_output=piece, finished=i == len(pieces) - 1, finish_reason=None,
                                prompt_tokens_delta=0, completion_tokens_delta=1)

    spec = GenSpec(messages=[], sampling_params=None, parser_tools=tools, template_tools=tools)
    if chunk is None:
        r = asyncio.run(generate_full(0, spec, State()))
        return r.reasoning, r.content, [(c.name, c.parameters) for c in r.tool_calls]

    async def events():
        return [ev async for ev in generate_events(0, spec, State())]

    evs = asyncio.run(events())
    return (
        "".join(ev.text for ev in evs if isinstance(ev, ReasoningDelta)),
        "".join(ev.text for ev in evs if isinstance(ev, ContentDelta)),
        [(c.name, c.parameters) for ev in evs if isinstance(ev, ToolCallsDelta) for c in ev.calls],
    )


@pytest.mark.needs_weights
@pytest.mark.parametrize("chunk", [None, 1, 3])
def test_a_parsed_turn_sent_back_renders_to_the_generated_text(chunk):
    """Each extra separator newline the parsers keep comes back once per turn and accumulates in a long conversation."""
    from transformers import AutoTokenizer

    from freetoken.server.generation import render_messages

    model = os.environ.get("FREETOKEN_RIPPLE_MODEL")
    if not model:
        pytest.skip("set FREETOKEN_RIPPLE_MODEL")
    tokenizer = AutoTokenizer.from_pretrained(model)
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    for text in _TURNS:
        reasoning, content, calls = _serve(text, tools, chunk)
        turn = {"role": "assistant", "content": content, "reasoning_content": reasoning or None,
                "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls] or None}
        rendered = tokenizer.apply_chat_template(render_messages([{"role": "user", "content": "Hi."}, turn]), tools=tools, tokenize=False)
        assert rendered.endswith(f"<|im_start|>assistant\n{text}<|im_end|>\n"), (text, reasoning, content)
