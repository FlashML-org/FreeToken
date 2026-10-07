"""Logits parity against the vendored reference implementation on the tiny synthetic checkpoint.

The reference runs bf16 or FP8/FP4 weights with its own torch quantizers, sparse attention, engram
hash and single-pass mHC; FreeToken runs the same checkpoint through its kernels and packed pools.
Compared: exact-mode prefill, greedy decode steps, a two-request batch, and the bounded-replay path
against the reference's ``forward_bounded`` oracle.
"""

from __future__ import annotations

from dataclasses import asdict, fields

import pytest
import torch

from .common import VOCAB, requires_cuda, tiny_text_config, write_tiny_checkpoint

pytestmark = requires_cuda

# The sparse-attention implementations accumulate in different orders.
ATOL = RTOL = 1e-2
# The fp4 experts also round the routed sum to bf16 before the shared-expert add and weight each
# expert's down output; the reference keeps that sum fp32 and weights the intermediate before its fp8
# round-trip (10 seeds x 16 checks: max |err| 0.019, against 0.007 in the reference's order).
QUANT_TOL = 2e-2


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dsv41-parity")
    tensors = write_tiny_checkpoint(str(folder), seed=11)
    return str(folder), tensors


def _tokens(n: int, seed: int) -> torch.Tensor:
    return torch.randint(3, VOCAB, (n,), generator=torch.Generator().manual_seed(seed))


class Reference:
    """The vendored ``Transformer`` on CUDA, loaded from the synthetic checkpoint (identity token map)."""

    def __init__(self, tensors: dict, text: dict, *, max_seq_len: int, max_batch_size: int, quantized: bool = False, vision_config: dict | None = None):
        from freetoken.models.deepseek_v41.args import DeepseekV41Args
        from types import SimpleNamespace

        from . import reference
        from .reference import engram as ref_engram
        from .reference import model as ref_model

        args = DeepseekV41Args.from_hf(SimpleNamespace(text_config=text))
        names = {f.name for f in fields(ref_model.ModelArgs)}
        kwargs = {k: v for k, v in asdict(args).items() if k in names}
        kwargs.update(dtype="fp8" if quantized else "bf16", expert_dtype="fp4" if quantized else None,
                      max_seq_len=max_seq_len, max_batch_size=max_batch_size, vision_n_layers=0, temperature=0.0)
        if vision_config:
            kwargs.update(vision_n_layers=vision_config["num_hidden_layers"], vision_dim=vision_config["hidden_size"],
                          vision_n_heads=vision_config["num_attention_heads"], vision_inter_dim=vision_config["intermediate_size"],
                          vision_patch_size=vision_config["patch_size"], vision_downsample_ratio=vision_config["downsample_ratio"])
        self.margs = ref_model.ModelArgs(**kwargs)
        # the synthetic checkpoint's tokenizer is the identity over VOCAB ids
        ref_engram.build_compressed_token_map = lambda tokenizer: (list(range(VOCAB)), VOCAB)
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                self.model = ref_model.Transformer(self.margs, tokenizer=object())
        finally:
            torch.set_default_dtype(torch.float32)
        state = {k: v for k, v in tensors.items() if not k.startswith("mtp.")
                 and (vision_config or (not k.startswith("vision.") and not k.endswith("bias_vl")))}
        if quantized:
            state = {k: v.view(torch.float4_e2m1fn_x2) if ".ffn.experts." in k and k.endswith(".weight") else v
                     for k, v in state.items()}
            # The reference's grouped output projection holds dequantized bf16 weights.
            for name in list(state):
                if name.endswith(".wo_a.scale"):
                    scale = state.pop(name).float().repeat_interleave(32, 0).repeat_interleave(32, 1)
                    weight_name = name.removesuffix(".scale") + ".weight"
                    state[weight_name] = (state[weight_name].float() * scale).bfloat16()
        missing, unexpected = self.model.load_state_dict({k: v.cuda() for k, v in state.items()}, strict=False)
        assert not unexpected, unexpected
        assert not [m for m in missing if "freqs_cis" not in m], missing
        self.reference = reference

    # the reference builds its index tensors on the default device (generate.py sets it to cuda)
    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.device("cuda"):
            return self.model(ids.view(1, -1).cuda(), 0)[1].float()

    def prefill_batch(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.device("cuda"):
            return self.model(ids.cuda(), 0)[1].float()

    def prefill_bounded(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.device("cuda"):
            return self.model.forward_bounded(ids.reshape(-1, ids.shape[-1]).cuda(), self.margs.window_size).float()

    def decode(self, tokens: torch.Tensor, pos: int) -> torch.Tensor:
        with torch.device("cuda"):
            return self.model(tokens.view(-1, 1).cuda(), pos)[1].float()


def _engram_table_for(engine, tensors, text):
    """Our engine reads the Engram rows straight from the synthetic shard (the disk table), hashed
    with the same identity token map the reference oracle uses."""
    from freetoken.models.deepseek_v41.engram import EngramHash
    from freetoken.models.deepseek_v41.engram_table import EngramDiskTable, EngramHost, engram_row_source

    hash = EngramHash(engine.args, list(range(VOCAB)), VOCAB)
    tables = [
        EngramDiskTable(engram_row_source(engine.checkpoint, layer.layer_id), hash.n_cols, engine.device, max_graph_rows=8, max_extend_tokens=64)
        for layer in engine.model.engram_layers()
    ]
    for layer, table in zip(engine.model.engram_layers(), tables):
        layer.attach_table(table)
    host = EngramHost(hash, tables, engine.device)
    engine.model.forward_host_ctx = host.forward_host_ctx
    engine.engram_host = host


def _compare(name, got, want, *, tol=ATOL):
    err = (got - want).abs().max().item()
    assert torch.equal(got.argmax(-1), want.argmax(-1)), f"{name}: argmax differs (max abs err {err:.4f})"
    torch.testing.assert_close(got, want, atol=tol, rtol=tol, msg=lambda m: f"{name}: {m}")


@pytest.mark.parametrize("mode", ["exact", "bounded"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_quantized_offload_prefill_and_decode_match_reference(tmp_path, mode, batch_size):
    from .harness import TinyEngine

    tensors = write_tiny_checkpoint(str(tmp_path), seed=11, quantized=True)
    text = tiny_text_config(moe_intermediate_size=256)
    ref = Reference(tensors, text, max_seq_len=1024, max_batch_size=3, quantized=True)
    eng = TinyEngine(str(tmp_path), swa_decoder_replay=mode, quantized=True)
    _engram_table_for(eng, tensors, text)
    ids = torch.stack([_tokens(300, 1 + i) for i in range(batch_size)])
    want = ref.prefill_batch(ids) if mode == "exact" else ref.prefill_bounded(ids)
    reqs = [eng.new_request(i, row.tolist()) for i, row in enumerate(ids)]
    got = eng.prefill(reqs)
    _compare(f"quantized {mode} prefill", got, want, tol=QUANT_TOL)
    eng.finish_prefill(reqs)
    for step in range(3):
        token = want.argmax(-1)
        want = ref.decode(token, ids.shape[-1] + step)
        got = eng.decode(reqs, token.tolist())
        _compare(f"quantized {mode} decode {step}", got, want, tol=QUANT_TOL)


def test_exact_prefill_and_greedy_decode_match_the_reference(checkpoint):
    from .harness import TinyEngine

    folder, tensors = checkpoint
    text = tiny_text_config()
    ref = Reference(tensors, text, max_seq_len=1024, max_batch_size=3)
    eng = TinyEngine(folder, max_seq_len=1024, max_running_req=2, swa_decoder_replay="exact")
    _engram_table_for(eng, tensors, text)

    ids = _tokens(300, 1)
    want = ref.prefill(ids)
    req = eng.new_request(0, ids.tolist())
    got = eng.prefill([req])
    _compare("prefill", got, want)
    eng.finish_prefill([req])
    # greedy decode from the reference's own picks keeps both sides on one trajectory
    pos = 300
    for step in range(12):
        nxt = int(want.argmax(-1).item())
        want = ref.decode(torch.tensor([nxt]), pos)
        got = eng.decode([req], [nxt])
        _compare(f"decode step {step}", got, want)
        pos += 1


def test_batched_prefill_and_decode_match_the_reference(checkpoint):
    from .harness import TinyEngine

    folder, tensors = checkpoint
    text = tiny_text_config()
    ref = Reference(tensors, text, max_seq_len=1024, max_batch_size=3)
    eng = TinyEngine(folder, max_seq_len=1024, max_running_req=2, swa_decoder_replay="exact")
    _engram_table_for(eng, tensors, text)

    ids = torch.stack([_tokens(200, 2), _tokens(200, 3)])
    want = ref.prefill_batch(ids)
    reqs = [eng.new_request(i, ids[i].tolist()) for i in range(2)]
    got = eng.prefill(reqs)
    _compare("batched prefill", got, want)
    eng.finish_prefill(reqs)
    pos = 200
    for step in range(6):
        nxt = want.argmax(-1).cpu()
        want = ref.decode(nxt, pos)
        got = eng.decode(reqs, nxt.tolist())
        _compare(f"batched decode step {step}", got, want)
        pos += 1


def test_bounded_replay_matches_the_reference_oracle(checkpoint):
    from .harness import TinyEngine

    folder, tensors = checkpoint
    text = tiny_text_config()
    ref = Reference(tensors, text, max_seq_len=1024, max_batch_size=3)
    eng = TinyEngine(folder, max_seq_len=1024, max_running_req=2, swa_decoder_replay="bounded")
    _engram_table_for(eng, tensors, text)

    ids = _tokens(300, 4)
    want = ref.prefill_bounded(ids)
    req = eng.new_request(0, ids.tolist())
    got = eng.prefill([req])
    _compare("bounded prefill", got, want)
    eng.finish_prefill([req])
    pos = 300
    for step in range(6):
        nxt = int(want.argmax(-1).item())
        want = ref.decode(torch.tensor([nxt]), pos)
        got = eng.decode([req], [nxt])
        _compare(f"decode after bounded prefill, step {step}", got, want)
        pos += 1
