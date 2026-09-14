"""One QSA layer against the HF reference math at L = 3000.

The fp32 reference here is transcribed from ``modeling_qwen4_exp.py``
(``Qwen4ExpTextQSAIndexer``:611, ``Qwen4ExpTextAttention``:757): pool a group of raw index
keys in fp32, ``(1 + w)`` rmsnorm it, rope it at the group's FIRST position, score
``sum_h relu(<q_h, k_bar_b>) / sqrt(index_head_dim)`` over complete blocks, keep the top
``budget // ratio``, expand, then attend to that set only.

Two claims: the selected sets agree (ties near the top-k boundary may differ, so the bar is a
Jaccard floor) and, GIVEN the backend's own selection, the attention output matches. Set
``FREETOKEN_QWEN4_HF_PYTHON`` to an interpreter whose transformers ships ``qwen4_exp`` to run
the same comparison against the real HF module in a subprocess.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from .common import Fixture, requires_cuda, parsed_config, selection_spy

QSA_LAYER = 3
LENGTH = 3000


def _plus_one_rmsnorm(x, weight, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())


def _hf_rope(x, positions, rotary_dim, base):
    """HF apply_rotary_pos_emb on [T, H, D], rotating only the first rotary_dim dims."""
    inv = 1.0 / (
        base
        ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim)
    )
    freqs = positions.float().unsqueeze(-1) * inv
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(1)
    rotated = x[..., :rotary_dim].float()
    half = rotary_dim // 2
    swapped = torch.cat([-rotated[..., half:], rotated[..., :half]], dim=-1)
    return torch.cat([rotated * cos + swapped * sin, x[..., rotary_dim:].float()], dim=-1)


def _hf_block_scores(x, indexer, config, positions):
    """[T, blocks] indexer scores; the pooled block keys do not depend on the query row."""
    args = config.qwen4_args
    rotary = config.rotary_config
    heads, dim, ratio = args.index_n_heads, args.index_head_dim, args.index_ratio
    qk = F.linear(x.float(), indexer.index_qk_proj.weight.float())
    q = _plus_one_rmsnorm(
        qk[:, : heads * dim].view(-1, heads, dim), indexer.q_layernorm.weight, config.rms_norm_eps
    )
    q = _hf_rope(q, positions, rotary.rotary_dim, rotary.base)
    raw = qk[:, heads * dim :]
    blocks = raw.shape[0] // ratio
    pooled = raw[: blocks * ratio].view(blocks, ratio, dim).mean(1)
    pooled = _plus_one_rmsnorm(pooled, indexer.k_layernorm.weight, config.rms_norm_eps)
    kbar = _hf_rope(
        pooled.unsqueeze(1), positions[: blocks * ratio : ratio], rotary.rotary_dim, rotary.base
    ).squeeze(1)
    return torch.relu(torch.einsum("thd,bd->tbh", q, kbar)).sum(-1) / math.sqrt(dim)


def _hf_selection(scores, positions, ratio, budget):
    """Per-query token ids: the top-(budget // ratio) complete blocks plus the open tail."""
    offsets = torch.arange(ratio, device=scores.device)
    selected = []
    for row, position in enumerate(positions.tolist()):
        visible = (position + 1) // ratio
        chosen = torch.empty(0, dtype=torch.int64, device=scores.device)
        if visible:
            top = scores[row, :visible].topk(min(budget // ratio, visible)).indices
            chosen = (top.unsqueeze(-1) * ratio + offsets).flatten()
        tail = torch.arange(visible * ratio, position + 1, device=scores.device)
        selected.append(torch.cat([chosen, tail]).sort().values)
    return selected


def _hf_layer_output(x, attn, config, positions, selection):
    """The HF gated-GQA layer restricted to a given per-query token selection."""
    rotary = config.rotary_config
    length, dim = x.shape[0], attn.head_dim
    qkv = F.linear(x.float(), attn.qkv_proj.weight.float())
    qg, k, v = qkv.split(attn._qkv_split, dim=-1)
    qg = qg.view(length, attn.num_q, dim * 2)
    q = _plus_one_rmsnorm(qg[..., :dim], attn.q_norm.weight, config.rms_norm_eps)
    q = _hf_rope(q, positions, rotary.rotary_dim, rotary.base)
    k = _plus_one_rmsnorm(k.view(length, attn.num_kv, dim), attn.k_norm.weight, config.rms_norm_eps)
    k = _hf_rope(k, positions, rotary.rotary_dim, rotary.base)
    repeat = attn.num_q // attn.num_kv
    k = k.repeat_interleave(repeat, dim=1)
    v = v.view(length, attn.num_kv, dim).repeat_interleave(repeat, dim=1).float()
    out = torch.zeros(length, attn.num_q, dim, device=x.device, dtype=torch.float32)
    for row, tokens in enumerate(selection):
        scores = torch.einsum("hd,khd->hk", q[row], k[tokens]) * dim**-0.5
        out[row] = torch.einsum("hk,khd->hd", scores.softmax(-1), v[tokens])
    gate = torch.sigmoid(qg[..., dim:].reshape(length, -1).float())
    return F.linear(out.reshape(length, -1) * gate, attn.o_proj.weight.float())


def _jaccard(indices, selection):
    scores = []
    for row, tokens in enumerate(selection):
        mine = set(indices[row][indices[row] >= 0].tolist())
        theirs = set(tokens.tolist())
        scores.append(len(mine & theirs) / max(len(mine | theirs), 1))
    return torch.tensor(scores)


@requires_cuda
def test_single_layer_matches_hf_reference(monkeypatch):
    config = parsed_config()
    fixture = Fixture(config, num_pages=128, max_running_req=4)
    attn = fixture.layer(QSA_LAYER)
    generator = torch.Generator(device=fixture.device).manual_seed(13)
    x = (
        torch.randn(
            LENGTH, config.hidden_size, device=fixture.device, dtype=fixture.dtype,
            generator=generator,
        )
        * 0.5
    )
    seen = selection_spy(monkeypatch, fixture.backend)
    batch = fixture.batch([fixture.req(0, 0, LENGTH)], "prefill")
    got = attn.forward(x, batch)
    indices = seen["indices"]

    args = config.qwen4_args
    scores = _hf_block_scores(x, attn.indexer, config, batch.positions)
    reference_selection = _hf_selection(
        scores, batch.positions, args.index_ratio, args.index_budget
    )
    jaccard = _jaccard(indices, reference_selection)
    assert jaccard.min() >= 0.97, f"worst-row Jaccard {jaccard.min():.4f}"

    own_selection = [row[row >= 0].long().sort().values for row in indices]
    reference = _hf_layer_output(x, attn, config, batch.positions, own_selection)
    torch.testing.assert_close(got.float(), reference, rtol=2e-2, atol=2e-2)


_HF_DRIVER = '''
import sys, torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention

payload = torch.load(sys.argv[1], map_location="cuda", weights_only=False)
meta = payload["meta"]
config = Qwen4ExpTextConfig(
    hidden_size=meta["hidden_size"], num_attention_heads=meta["num_q"],
    num_key_value_heads=meta["num_kv"], head_dim=meta["head_dim"], rms_norm_eps=meta["eps"],
    max_position_embeddings=meta["max_position"],
    rope_parameters={"rope_type": "default", "rope_theta": meta["base"],
                     "partial_rotary_factor": meta["rotary_dim"] / meta["head_dim"],
                     "mrope_section": [11, 11, 10]},
    indexer_n_heads=meta["index_heads"], indexer_kv_heads=1, indexer_head_dim=meta["index_dim"],
    indexer_budget=meta["budget"], indexer_compress_ratio=meta["ratio"],
)
config._attn_implementation = "eager"
torch.set_grad_enabled(False)
attn = Qwen4ExpTextAttention(config, layer_idx=0).to("cuda", torch.float32)
attn.load_state_dict({k: v.to("cuda", torch.float32) for k, v in payload["weights"].items()})

x = payload["x"].to(torch.float32).unsqueeze(0)
positions = payload["positions"].to("cuda").to(torch.long)
rotary_dim = meta["rotary_dim"]
pairs = torch.arange(0, rotary_dim, 2, device="cuda", dtype=torch.float32)
inv = 1.0 / (meta["base"] ** (pairs / rotary_dim))
freqs = positions.float().unsqueeze(-1) * inv
cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(0)
sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(0)
length = x.shape[1]
causal = torch.arange(length, device="cuda")
mask = torch.zeros(1, 1, length, length, device="cuda", dtype=torch.float32)
mask.masked_fill_(causal[None, :] > causal[:, None], torch.finfo(torch.float32).min)

out, _ = attn(x, (cos, sin), mask)
selected = attn.indexer(x, (cos, sin), mask, None)[0, 0] == 0
torch.save({"out": out[0].cpu(), "selected": selected.cpu()}, sys.argv[2])
'''


@requires_cuda
@pytest.mark.skipif(
    not os.environ.get("FREETOKEN_QWEN4_HF_PYTHON"),
    reason="set FREETOKEN_QWEN4_HF_PYTHON to a transformers build that ships qwen4_exp",
)
def test_single_layer_matches_upstream_hf(tmp_path, monkeypatch):
    config = parsed_config()
    fixture = Fixture(config, num_pages=128, max_running_req=4)
    attn = fixture.layer(QSA_LAYER)
    generator = torch.Generator(device=fixture.device).manual_seed(13)
    x = (
        torch.randn(
            LENGTH, config.hidden_size, device=fixture.device, dtype=fixture.dtype,
            generator=generator,
        )
        * 0.5
    )
    seen = selection_spy(monkeypatch, fixture.backend)
    batch = fixture.batch([fixture.req(0, 0, LENGTH)], "prefill")
    got = attn.forward(x, batch)

    args = config.qwen4_args
    rotary = config.rotary_config
    q_rows, kv_rows = attn.qo_attn_dim * 2, attn.kv_attn_dim
    fused = attn.qkv_proj.weight
    payload = tmp_path / "payload.pt"
    result = tmp_path / "hf.pt"
    driver = tmp_path / "driver.py"
    driver.write_text(_HF_DRIVER)
    torch.save(
        {
            "weights": {
                "q_proj.weight": fused[:q_rows].cpu(),
                "k_proj.weight": fused[q_rows : q_rows + kv_rows].cpu(),
                "v_proj.weight": fused[q_rows + kv_rows :].cpu(),
                "o_proj.weight": attn.o_proj.weight.cpu(),
                "q_norm.weight": attn.q_norm.weight.cpu(),
                "k_norm.weight": attn.k_norm.weight.cpu(),
                "indexer.index_qk_proj.weight": attn.indexer.index_qk_proj.weight.cpu(),
                "indexer.q_layernorm.weight": attn.indexer.q_layernorm.weight.cpu(),
                "indexer.k_layernorm.weight": attn.indexer.k_layernorm.weight.cpu(),
            },
            "x": x.cpu(),
            "positions": batch.positions.cpu(),
            "meta": {
                "hidden_size": config.hidden_size, "num_q": attn.num_q, "num_kv": attn.num_kv,
                "head_dim": attn.head_dim, "eps": config.rms_norm_eps,
                "max_position": rotary.max_position, "base": rotary.base,
                "rotary_dim": rotary.rotary_dim, "index_heads": args.index_n_heads,
                "index_dim": args.index_head_dim, "budget": args.index_budget,
                "ratio": args.index_ratio,
            },
        },
        payload,
    )
    subprocess.run(
        [os.environ["FREETOKEN_QWEN4_HF_PYTHON"], str(driver), str(payload), str(result)],
        check=True, stdout=sys.stderr, timeout=1800,
    )
    upstream = torch.load(result, map_location=fixture.device, weights_only=False)
    selection = [row.nonzero().flatten() for row in upstream["selected"].to(fixture.device)]
    jaccard = _jaccard(seen["indices"], selection)
    assert jaccard.min() >= 0.97, f"worst-row Jaccard {jaccard.min():.4f}"
    torch.testing.assert_close(got.float(), upstream["out"].float(), rtol=2e-2, atol=2e-2)
