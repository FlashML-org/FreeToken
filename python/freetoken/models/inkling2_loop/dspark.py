"""Inkling 2 DSpark draft (the checkpoint's ``mtp.safetensors``), after vLLM ``models/inkling2/nvidia/dspark.py`` and sglang ``models/inkling2_mtp.py``.

One forward drafts the block ``[anchor, mask x (block-1)]`` at positions ``p .. p+block-1``, ``p`` the anchor's: three dense layers whose
attention is bidirectional over the block and reads context K/V below ``p`` (local layers only the ``window`` positions before ``p``).
Context K/V of position ``q`` come from the target's features of the token at ``q``: the inputs of the tapped layer visits and the
final-norm output, mixed per layer by ``softmax(fuse_weight)``, normalized and projected through the layer's ``k/v_proj_ctx``.
A recurrent Markov head turns each mask row into a token in order: its rank-``r`` state reads the previous token and biases the
target head's logits, which are softcapped like the target's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

# the draft's own tensors; ``mtp.safetensors`` holds exactly these (``*`` = layer index)
_LAYER_TENSORS = (
    "dflare_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "self_attn.k_proj_ctx.weight", "self_attn.v_proj_ctx.weight", "self_attn.k_norm_ctx.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)
_HEAD_TENSORS = (
    "fuse_weight", "mask_embed", "norm.weight",
    "markov_head.dflare_norm.weight", "markov_head.markov_w1.weight",
    "markov_head.rnn_proj.weight", "markov_head.rnn_proj.bias",
    "markov_head.hidden_out.weight", "markov_head.hidden_out.bias",
)
MTP_FILE = "mtp.safetensors"


def _section(obj: Any) -> dict:
    """A config section (dict, PretrainedConfig or RawConfigShim) as a plain dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    return obj.to_dict()


@dataclass(frozen=True)
class DSparkConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    local_layer_ids: tuple[int, ...]
    window: int
    block_size: int
    markov_rank: int
    # target layer visits (loop passes counted) whose INPUT feeds tap i; the last, num_layer_visits, is the final-norm output
    target_layer_ids: tuple[int, ...]
    num_layer_visits: int
    # visits that restart the loop core from the prelude's state: their input is the prelude's output, not the visit before
    pass_start_visits: tuple[int, ...]
    loop_core_start: int
    vocab_size: int  # unpadded: logits past it are never drafted
    target_rms_norm_eps: float
    softcap: float
    max_position: int

    @property
    def num_taps(self) -> int:
        return len(self.target_layer_ids)

    def output_taps(self) -> dict[int, list[int]]:
        """Target visit -> the feature columns its OUTPUT fills; the final-norm sentinel is not in it (``final_hidden`` fills the last column)."""
        taps: dict[int, list[int]] = {}
        for col, visit in enumerate(self.target_layer_ids[:-1]):
            # the input of a pass-restart visit is the prelude's output; any other visit reads the one before it
            src = self.loop_core_start - 1 if visit in self.pass_start_visits else visit - 1
            taps.setdefault(src, []).append(col)
        return taps


def parse_dspark_config(hf_config: Any) -> DSparkConfig:
    """The draft config from the target checkpoint's ``mtp_config``; only the shape the checkpoint ships (rnn_hidden head, BF16, no attention bias) is served."""
    draft = dict(_section(getattr(hf_config, "mtp_config", None)))
    if not draft:
        raise ValueError("the checkpoint has no mtp_config, so it ships no DSpark draft")
    text = _section(getattr(hf_config, "text_config", None)) or _section(hf_config)
    required = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size",
                "rms_norm_eps", "rope_theta", "local_layer_ids", "sliding_window_size", "block_size", "markov_rank",
                "markov_head_type", "target_layer_ids")
    if missing := [key for key in required if key not in draft]:
        raise ValueError(f"mtp_config lacks {missing}")
    if draft["markov_head_type"] != "rnn_hidden":
        raise ValueError(f"only the rnn_hidden DSpark Markov head is supported, got {draft['markov_head_type']!r}")
    if draft.get("q_bias") or draft.get("o_bias"):
        raise ValueError("DSpark drafts with attention projection biases are not supported")
    if any(draft.get(key, "bfloat16") != "bfloat16" for key in ("dtype", "torch_dtype")):
        raise ValueError("the DSpark draft must be bfloat16")
    hidden = int(draft["hidden_size"])
    if hidden != int(text["hidden_size"]):
        raise ValueError(f"the DSpark draft's hidden size {hidden} is not the target's {text['hidden_size']}")
    num_layers, heads, kv_heads = int(draft["num_hidden_layers"]), int(draft["num_attention_heads"]), int(draft["num_key_value_heads"])
    local = tuple(int(i) for i in draft["local_layer_ids"])
    if len(set(local)) != len(local) or any(not 0 <= i < num_layers for i in local):
        raise ValueError(f"mtp_config local_layer_ids {list(local)} must be distinct layers below {num_layers}")
    if heads % kv_heads:
        raise ValueError(f"{heads} query heads do not group over {kv_heads} KV heads")
    block, rank = int(draft["block_size"]), int(draft["markov_rank"])
    if block < 2 or rank <= 0:
        raise ValueError(f"DSpark needs block_size >= 2 and markov_rank > 0, got {block} and {rank}")

    layers = int(text["num_hidden_layers"])
    loop = _section(text.get("loop_config"))
    if loop:
        start, end, passes = int(loop["core_start"]), int(loop["core_end"]), int(loop["num_passes"])
        if loop.get("is_varloop"):
            raise ValueError("DSpark taps fixed layer visits, so variable-loop checkpoints are not supported")
        visits = layers + (passes - 1) * (end - start)
        restarts = tuple(start + p * (end - start) for p in range(1, passes))
    else:
        start, visits, restarts = 0, layers, ()
    taps = tuple(int(i) for i in draft["target_layer_ids"])
    # the input of visit 0 is the embedding, which no layer outputs
    if len(taps) < 2 or list(taps) != sorted(set(taps)) or taps[0] < 1 or taps[-1] != visits:
        raise ValueError(f"mtp_config target_layer_ids {list(taps)} must be sorted distinct layer visits in [1, {visits}) followed by {visits}, the final-norm output")
    softcap = text.get("final_logit_softcapping")
    vocab = int(text.get("unpadded_vocab_size") or text["vocab_size"])
    return DSparkConfig(
        hidden_size=hidden, num_layers=num_layers, num_heads=heads, num_kv_heads=kv_heads, head_dim=int(draft["head_dim"]),
        intermediate_size=int(draft["intermediate_size"]), rms_norm_eps=float(draft["rms_norm_eps"]),
        rope_theta=float(draft["rope_theta"]), local_layer_ids=local, window=int(draft["sliding_window_size"]),
        block_size=block, markov_rank=rank, target_layer_ids=taps, num_layer_visits=visits, pass_start_visits=restarts,
        loop_core_start=start, vocab_size=vocab, target_rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)), softcap=float(softcap) if softcap else 0.0,
        max_position=int(draft.get("max_position_embeddings") or text.get("model_max_length") or 1 << 20),
    )


def mtp_file(path: str) -> str:
    """Local path of the draft's safetensors for a checkpoint dir, a Hub repo id or the file itself."""
    if path.endswith(".safetensors"):
        if not os.path.isfile(path):
            raise ValueError(f"{path} does not exist")
        return path
    from freetoken.utils.hf import optional_hf_file

    found = optional_hf_file(path, MTP_FILE)
    if found is None:
        raise ValueError(f"{path} has no {MTP_FILE}, the DSpark draft")
    return found


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32 RMSNorm rounded once to ``x``'s dtype."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual + x`` (rounded, the new residual) and the RMSNorm of the unrounded sum."""
    s = residual.float() + x.float()
    return (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype), s.to(residual.dtype)


def rope_inv_freq(theta: float, head_dim: int, device) -> torch.Tensor:
    """fp32 NeoX inverse frequencies over the whole head, as the reference's rope cache builds them."""
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))


def apply_rope(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """NeoX rotary of ``x [n, heads, d]`` at ``positions [n]`` in fp32, rounded to ``x``'s dtype."""
    half = x.shape[-1] // 2
    freqs = positions.float()[:, None] * inv_freq[None, :]
    cos, sin = freqs.cos()[:, None], freqs.sin()[:, None]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)


def mixture_weights(fuse_weight: torch.Tensor) -> torch.Tensor:
    return torch.softmax(fuse_weight.float(), dim=-1).to(fuse_weight.dtype)


class Inkling2DSparkDraft:
    """The draft's weights and math; the proposer owns the context K/V, the per-request state and the verify."""

    def __init__(self, config: DSparkConfig, weights: dict[str, torch.Tensor], embed: Callable[[torch.Tensor], torch.Tensor],
                 lm_head_weight: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> None:
        c = self.config = config
        self.device, self.dtype = lm_head_weight.device, dtype
        if lm_head_weight.dtype != torch.bfloat16:
            raise ValueError(f"the DSpark draft reads the target's LM head as stored, in bf16; got {lm_head_weight.dtype}")
        if lm_head_weight.shape[0] < c.vocab_size or lm_head_weight.shape[1] != c.hidden_size:
            raise ValueError(f"target LM head {tuple(lm_head_weight.shape)} does not cover vocab {c.vocab_size} x hidden {c.hidden_size}")
        # serving shares the target's bf16 head; an fp32 draft (tests pin the math with one) copies it
        self.embed, self.lm_head_weight = embed, lm_head_weight.to(dtype)
        weights = {name: t.to(dtype) for name, t in weights.items()}
        q, kv = c.num_heads * c.head_dim, c.num_kv_heads * c.head_dim
        self.q_size, self.kv_size = q, kv
        self.layers = []
        for i in range(c.num_layers):
            p = f"layers.{i}."
            w = lambda name: weights[p + name]  # noqa: E731
            self.layers.append({
                "qkv": torch.cat([w("self_attn.q_proj.weight"), w("self_attn.k_proj.weight"), w("self_attn.v_proj.weight")]),
                "kv_ctx": torch.cat([w("self_attn.k_proj_ctx.weight"), w("self_attn.v_proj_ctx.weight")]),
                "o": w("self_attn.o_proj.weight"),
                "gate_up": torch.cat([w("mlp.gate_proj.weight"), w("mlp.up_proj.weight")]),
                "down": w("mlp.down_proj.weight"),
                "q_norm": w("self_attn.q_norm.weight"), "k_norm": w("self_attn.k_norm.weight"),
                "k_norm_ctx": w("self_attn.k_norm_ctx.weight"),
                "input_ln": w("input_layernorm.weight"), "post_ln": w("post_attention_layernorm.weight"),
                "dflare_norm": w("dflare_norm.weight"),
            })
        self.norm, self.mask_embed = weights["norm.weight"], weights["mask_embed"]
        # rounded to the draft dtype like the reference's mixture buffer; row i mixes the taps for layer i, the last row for the Markov head
        self.fuse_alpha = mixture_weights(weights["fuse_weight"])
        r = c.markov_rank
        self.markov_norm = weights["markov_head.dflare_norm.weight"]
        self.markov_w1 = weights["markov_head.markov_w1.weight"]
        rnn_w = weights["markov_head.rnn_proj.weight"]
        # rnn_proj reads [state, emb(prev), hidden, f_rnn]: the first two change every step, the last two once per block
        self.rnn_w_step, self.rnn_w_ctx = rnn_w[:, : 2 * r].contiguous(), rnn_w[:, 2 * r :].contiguous()
        self.rnn_bias = weights["markov_head.rnn_proj.bias"].float()
        self.fold_weight, self.fold_bias = self._fold(weights["markov_head.hidden_out.weight"], weights["markov_head.hidden_out.bias"])
        self.inv_freq = rope_inv_freq(c.rope_theta, c.head_dim, self.device)

    @torch.no_grad()
    def _fold(self, out_w: torch.Tensor, out_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``hidden_out`` folded into the LM head rows in fp64: the Markov bias over the vocab is ``latent @ W^T + b``."""
        head = self.lm_head_weight[: self.config.vocab_size]
        weight = torch.empty(head.shape[0], self.config.markov_rank, dtype=self.dtype, device=self.device)
        bias = torch.empty(head.shape[0], dtype=self.dtype, device=self.device)
        proj, b = out_w.double(), out_b.double()
        for start in range(0, head.shape[0], 8192):
            rows = head[start : start + 8192].double()
            weight[start : start + 8192] = rows @ proj
            bias[start : start + 8192] = rows @ b
        return weight, bias

    @classmethod
    def load(cls, config: DSparkConfig, target_model, model_path: str, device: torch.device, dtype: torch.dtype) -> Inkling2DSparkDraft:
        """Read ``mtp.safetensors`` next to the target's weights; the anchor embedding and the LM head are the target's."""
        import safetensors

        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"the DSpark draft runs in bf16 (or fp32 to check its math), not {dtype}")
        path = mtp_file(model_path)
        weights = {}
        expected = set(_HEAD_TENSORS) | {f"layers.{i}.{name}" for i in range(config.num_layers) for name in _LAYER_TENSORS}
        with safetensors.safe_open(path, framework="pt", device=str(device)) as f:
            for name in f.keys():
                if name not in expected:
                    raise ValueError(f"{path}: unexpected DSpark tensor {name!r}")
                tensor = f.get_tensor(name)
                if tensor.dtype != torch.bfloat16:
                    raise ValueError(f"{path}: {name} is {tensor.dtype}, the DSpark draft is bf16")
                weights[name] = tensor
        if missing := sorted(expected - set(weights)):
            raise ValueError(f"{path}: DSpark tensors missing: {missing}")
        cls._check_shapes(config, weights, path)
        embed, head = target_parts(target_model, config.target_rms_norm_eps)
        return cls(config, weights, embed, head, dtype)

    @staticmethod
    def _check_shapes(c: DSparkConfig, w: dict[str, torch.Tensor], path: str) -> None:
        H, q, kv, r = c.hidden_size, c.num_heads * c.head_dim, c.num_kv_heads * c.head_dim, c.markov_rank
        shapes = {"fuse_weight": (c.num_layers + 1, c.num_taps), "mask_embed": (H,), "norm.weight": (H,),
                  "markov_head.dflare_norm.weight": (H,), "markov_head.rnn_proj.weight": (3 * r, 2 * r + 2 * H),
                  "markov_head.rnn_proj.bias": (3 * r,), "markov_head.hidden_out.weight": (H, r), "markov_head.hidden_out.bias": (H,)}
        for i in range(c.num_layers):
            p = f"layers.{i}."
            shapes.update({p + "self_attn.q_proj.weight": (q, H), p + "self_attn.k_proj.weight": (kv, H), p + "self_attn.v_proj.weight": (kv, H),
                           p + "self_attn.o_proj.weight": (H, q), p + "self_attn.k_proj_ctx.weight": (kv, H), p + "self_attn.v_proj_ctx.weight": (kv, H),
                           p + "mlp.gate_proj.weight": (c.intermediate_size, H), p + "mlp.up_proj.weight": (c.intermediate_size, H),
                           p + "mlp.down_proj.weight": (H, c.intermediate_size)})
        for name, shape in shapes.items():
            if tuple(w[name].shape) != shape:
                raise ValueError(f"{path}: {name} is {tuple(w[name].shape)}, expected {shape}")
        if w["markov_head.markov_w1.weight"].shape[1] != r or w["markov_head.markov_w1.weight"].shape[0] < c.vocab_size:
            raise ValueError(f"{path}: markov_w1 {tuple(w['markov_head.markov_w1.weight'].shape)} does not cover vocab {c.vocab_size} x rank {r}")

    # ---------------------------------------------------------------- target features

    def mix(self, features: torch.Tensor, row: int, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm of the ``fuse_alpha[row]``-weighted sum of the taps in ``features [n, taps*H]`` (fp32 math)."""
        c = self.config
        taps = features.view(features.shape[0], c.num_taps, c.hidden_size).float()
        mixed = (taps * self.fuse_alpha[row].float()[None, :, None]).sum(1)
        return (mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + c.rms_norm_eps) * weight.float()).to(self.dtype)

    def context_kv(self, features: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Every layer's context K/V ``[layers, n, kv_heads, d]`` of the target features at ``positions``."""
        c, n = self.config, features.shape[0]
        ks, vs = [], []
        for i, layer in enumerate(self.layers):
            kv = F.linear(self.mix(features, i, layer["dflare_norm"]), layer["kv_ctx"])
            k = rms_norm(kv[:, : self.kv_size].reshape(n, c.num_kv_heads, c.head_dim), layer["k_norm_ctx"], c.rms_norm_eps)
            ks.append(apply_rope(k, positions, self.inv_freq))
            vs.append(kv[:, self.kv_size :].reshape(n, c.num_kv_heads, c.head_dim))
        return torch.stack(ks), torch.stack(vs)

    def f_rnn(self, features: torch.Tensor) -> torch.Tensor:
        """The Markov head's context from the features of the last committed token, ``[n, H]``."""
        return self.mix(features, -1, self.markov_norm)

    # ---------------------------------------------------------------- block forward

    def forward_block(self, anchor: torch.Tensor, starts: torch.Tensor, context: Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Final hidden ``[bs, block, H]`` of the blocks ``[anchor, mask x (block-1)]`` at positions ``starts + j``.

        ``context(layer)`` returns that layer's context ``k, v [bs, C, kv_heads, d]`` and ``valid [bs, C]``: the columns a query of the block may read.
        """
        c = self.config
        bs, block = anchor.shape[0], c.block_size
        x = self.mask_embed.expand(bs, block, -1).clone()
        x[:, 0] = self.embed(anchor).to(x.dtype)
        x = x.view(bs * block, -1)
        positions = (starts.view(-1, 1) + torch.arange(block, device=anchor.device)).view(-1)
        group = c.num_heads // c.num_kv_heads
        residual = None
        for i, layer in enumerate(self.layers):
            if residual is None:
                residual, h = x, rms_norm(x, layer["input_ln"], c.rms_norm_eps)
            else:
                h, residual = add_rms_norm(x, residual, layer["input_ln"], c.rms_norm_eps)
            qkv = F.linear(h, layer["qkv"])
            q = rms_norm(qkv[:, : self.q_size].reshape(-1, c.num_heads, c.head_dim), layer["q_norm"], c.rms_norm_eps)
            k = rms_norm(qkv[:, self.q_size : self.q_size + self.kv_size].reshape(-1, c.num_kv_heads, c.head_dim), layer["k_norm"], c.rms_norm_eps)
            q, k = apply_rope(q, positions, self.inv_freq), apply_rope(k, positions, self.inv_freq)
            v = qkv[:, self.q_size + self.kv_size :].reshape(-1, c.num_kv_heads, c.head_dim)
            ctx_k, ctx_v, valid = context(i)
            # bidirectional over the block, which every query reads in full after the context
            keys = torch.cat([ctx_k, k.view(bs, block, c.num_kv_heads, c.head_dim)], dim=1).transpose(1, 2)
            values = torch.cat([ctx_v, v.view(bs, block, c.num_kv_heads, c.head_dim)], dim=1).transpose(1, 2)
            mask = torch.cat([valid, valid.new_ones(bs, block)], dim=1)[:, None, None, :]
            queries = q.view(bs, block, c.num_heads, c.head_dim).transpose(1, 2)
            keys, values = keys.repeat_interleave(group, dim=1), values.repeat_interleave(group, dim=1)
            o = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask, scale=c.head_dim**-0.5)
            o = F.linear(o.transpose(1, 2).reshape(bs * block, self.q_size), layer["o"])
            h, residual = add_rms_norm(o, residual, layer["post_ln"], c.rms_norm_eps)
            gate, up = F.linear(h, layer["gate_up"]).float().chunk(2, dim=-1)
            x = F.linear((F.silu(gate) * up).to(h.dtype), layer["down"])
        out, _ = add_rms_norm(x, residual, self.norm, c.rms_norm_eps)
        return out.view(bs, block, -1)

    # ---------------------------------------------------------------- Markov head

    def markov_context(self, hidden: torch.Tensor, f_rnn: torch.Tensor) -> torch.Tensor:
        """fp32 ``rnn_proj`` bias plus its ``[hidden, f_rnn]`` term, ``[bs, steps, 3r]`` for ``hidden [bs, steps, H]``."""
        inputs = torch.cat([hidden, f_rnn[:, None].expand_as(hidden)], dim=-1)
        return (inputs.float() @ self.rnn_w_ctx.float().T) + self.rnn_bias

    def markov_step(self, state: torch.Tensor, prev: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrence step from ``state [bs, r]`` and the previous token ``prev [bs]``: the new state and the bf16 vocab bias."""
        r = self.config.markov_rank
        inputs = torch.cat([state, self.markov_w1[prev.long()]], dim=-1)
        raw = (context + inputs.float() @ self.rnn_w_step.float().T).to(self.dtype).float()
        # each op rounds to bf16 like the reference's bf16 torch ops
        rnd = lambda t: t.to(self.dtype).float()  # noqa: E731
        gate, cand, out = raw[:, :r], raw[:, r : 2 * r], raw[:, 2 * r :]
        gate, cand = rnd(torch.sigmoid(gate)), rnd(torch.tanh(cand))
        new_state = (rnd(gate * state.float()) + rnd(rnd(1.0 - gate) * cand)).to(self.dtype)
        latent = torch.tanh(out).to(self.dtype)
        bias = torch.addmm(self.fold_bias, latent, self.fold_weight.T)
        return new_state, bias

    def step_logits(self, base: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """bf16 ``softcap * tanh(bf16(base + bias) / softcap)`` over the unpadded vocab."""
        x = (base.float() + bias.float()).to(self.dtype).float()
        if self.config.softcap:
            x = torch.tanh(x / self.config.softcap) * self.config.softcap
        return x.to(self.dtype)

    def base_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """bf16 GEMM of the draft's hidden with the target head's rows of the unpadded vocab."""
        return F.linear(hidden, self.lm_head_weight[: self.config.vocab_size])

    def walk(self, hidden: torch.Tensor, anchor: torch.Tensor, f_rnn: torch.Tensor, pick: Callable[[int, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Tokens ``[bs, steps]`` from the mask rows' ``hidden [bs, steps, H]``: step j's logits read the token picked at j-1 (the anchor at 0); ``pick(j, logits)`` chooses."""
        bs, steps, _ = hidden.shape
        base = self.base_logits(hidden)
        context = self.markov_context(hidden, f_rnn)
        state = hidden.new_zeros(bs, self.config.markov_rank)
        prev, tokens = anchor, []
        for j in range(steps):
            state, bias = self.markov_step(state, prev, context[:, j])
            prev = pick(j, self.step_logits(base[:, j], bias))
            tokens.append(prev)
        return torch.stack(tokens, dim=1)


def target_parts(target_model, eps: float) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]:
    """What the draft shares with the target: its first layer's input of a token (``embed_norm(embed(ids))``) and its bf16 LM head weight ``[padded vocab, H]``, as stored."""
    model = getattr(target_model, "model", None)
    embed_tokens, embed_norm = getattr(model, "embed_tokens", None), getattr(model, "embed_norm", None)
    head = getattr(getattr(target_model, "lm_head", None), "weight", None)
    if embed_tokens is None or embed_norm is None or head is None:
        raise ValueError(f"{type(target_model).__name__} lacks model.embed_tokens, model.embed_norm or lm_head.weight, which the DSpark draft shares")

    def embed(ids: torch.Tensor) -> torch.Tensor:
        # the target's own order: fp32 weight times the normalized row, rounded once
        x = embed_tokens.forward(ids)
        xf = x.float()
        return (embed_norm.float() * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps))).to(x.dtype)

    return embed, head


__all__ = ["DSparkConfig", "Inkling2DSparkDraft", "MTP_FILE", "mtp_file", "parse_dspark_config", "target_parts"]
