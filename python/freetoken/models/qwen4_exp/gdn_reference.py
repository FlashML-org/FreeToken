"""Pure-torch Gated DeltaNet reference (text-only, no cache).

Correctness oracle for the kernel-backed GDN op (``gdn.Qwen4ExpGatedDeltaNet``).
The two delta rules and the forward are transcribed from
``transformers.models.qwen4_exp.modeling_qwen4_exp`` (``torch_chunk_gated_delta_rule``,
``torch_recurrent_gated_delta_rule`` and ``Qwen4ExpTextGatedDeltaNet.forward`` no-cache path).
Qwen3.8-Flash-Next gates the output norm with ``config.output_gate_type`` (sigmoid) where
Qwen3.5 hardcodes silu; the conv keeps ``config.hidden_act`` (silu).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_GATE_ACTS = {"silu": F.silu, "swish": F.silu, "sigmoid": torch.sigmoid}


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + eps)


def recurrent_gated_delta_rule(
    query: torch.Tensor,  # [B, T, Hv, Dk]
    key: torch.Tensor,    # [B, T, Hv, Dk]
    value: torch.Tensor,  # [B, T, Hv, Dv]
    g: torch.Tensor,      # [B, T, Hv]   (log-decay; per-step decay = exp(g))
    beta: torch.Tensor,   # [B, T, Hv]
    *,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verbatim port of HF ``torch_recurrent_gated_delta_rule`` (output_final_state=True)."""
    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = _l2norm(query, eps=1e-6)
        key = _l2norm(key, eps=1e-6)
    query, key, value, beta, g = [
        t.transpose(1, 2).contiguous().to(torch.float32)
        for t in (query, key, value, beta, g)
    ]
    b, h, t_len, dk = key.shape
    dv = value.shape[-1]
    scale = 1.0 / (dk ** 0.5)
    query = query * scale

    out = torch.zeros(b, h, t_len, dv, dtype=value.dtype, device=value.device)
    state = (
        torch.zeros(b, h, dk, dv, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    for i in range(t_len):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    out = out.transpose(1, 2).contiguous().to(initial_dtype)  # [B, T, Hv, Dv]
    return out, state


def chunk_gated_delta_rule(
    query: torch.Tensor,  # [B, T, Hv, Dk]
    key: torch.Tensor,    # [B, T, Hv, Dk]
    value: torch.Tensor,  # [B, T, Hv, Dv]
    g: torch.Tensor,      # [B, T, Hv]
    beta: torch.Tensor,   # [B, T, Hv]
    *,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verbatim port of HF ``torch_chunk_gated_delta_rule`` (output_final_state=True).

    Same recurrence as ``recurrent_gated_delta_rule`` in exact arithmetic; it is the form the
    fla chunk kernel implements, so it is the closer oracle for the prefill path."""
    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = _l2norm(query, eps=1e-6)
        key = _l2norm(key, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim,
            dtype=value.dtype, device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    # for each chunk; decay_mask is already lower-triangular so no extra causal mask is needed
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2)
            @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class _RMSNormGated(nn.Module):
    """RMSNorm of x followed by an ``activation(z)`` gate (norm_before_gate=True).

    Mirrors ``Qwen4ExpTextRMSNormGated`` over head_v_dim groups.
    """

    def __init__(self, dim: int, eps: float, activation: str = "sigmoid"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.act = _GATE_ACTS[activation]

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * self.weight.float()
        x = x * self.act(z.float())
        return x.to(in_dtype)


class Qwen4ExpGatedDeltaNetReference(nn.Module):
    """Pure-torch Gated DeltaNet (text-only, no cache)."""

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        rms_norm_eps: float,
        hidden_act: str = "silu",
        output_gate: str = "sigmoid",
    ):
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"GDN reference only supports silu conv activation, got {hidden_act!r}")
        if output_gate not in _GATE_ACTS:
            raise ValueError(f"unsupported GDN output gate {output_gate!r}")
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = conv_kernel_size

        self.in_proj_qkv = nn.Linear(hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, kernel_size=conv_kernel_size,
            groups=self.conv_dim, padding=conv_kernel_size - 1, bias=False,
        )
        self.dt_bias = nn.Parameter(torch.zeros(num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(num_v_heads))
        self.norm = _RMSNormGated(head_v_dim, eps=rms_norm_eps, activation=output_gate)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    @torch.no_grad()
    def load_from_hf(self, hf_gdn) -> None:
        """Copy weights from a transformers ``Qwen4ExpTextGatedDeltaNet``."""
        self.in_proj_qkv.weight.copy_(hf_gdn.in_proj_qkv.weight)
        self.in_proj_z.weight.copy_(hf_gdn.in_proj_z.weight)
        self.in_proj_b.weight.copy_(hf_gdn.in_proj_b.weight)
        self.in_proj_a.weight.copy_(hf_gdn.in_proj_a.weight)
        # HF stores conv1d weight as [conv_dim, 1, K]; our depthwise Conv1d matches.
        self.conv1d.weight.copy_(hf_gdn.conv1d.weight.view_as(self.conv1d.weight))
        if hf_gdn.conv1d.bias is not None and self.conv1d.bias is not None:
            self.conv1d.bias.copy_(hf_gdn.conv1d.bias)
        self.dt_bias.copy_(hf_gdn.dt_bias)
        self.A_log.copy_(hf_gdn.A_log)
        self.norm.weight.copy_(hf_gdn.norm.weight)
        self.out_proj.weight.copy_(hf_gdn.out_proj.weight)

    def forward(self, hidden_states: torch.Tensor, *, use_chunk_rule: bool = False) -> torch.Tensor:
        b, t_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [B, conv_dim, T]
        z = self.in_proj_z(hidden_states).reshape(b, t_len, -1, self.head_v_dim)
        a = self.in_proj_a(hidden_states)
        bb = self.in_proj_b(hidden_states)

        # causal depthwise conv + silu (drop the right padding back to T)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[..., :t_len]).transpose(1, 2)  # [B, T, conv_dim]
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(b, t_len, -1, self.head_k_dim)
        key = key.reshape(b, t_len, -1, self.head_k_dim)
        value = value.reshape(b, t_len, -1, self.head_v_dim)

        beta = bb.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # GQA expand: replicate q/k heads up to num_v_heads
        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        rule = chunk_gated_delta_rule if use_chunk_rule else recurrent_gated_delta_rule
        core, _ = rule(query, key, value, g, beta, use_qk_l2norm=True)

        core = core.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core = self.norm(core, z).reshape(b, t_len, -1)
        return self.out_proj(core)


__all__ = [
    "Qwen4ExpGatedDeltaNetReference",
    "chunk_gated_delta_rule",
    "recurrent_gated_delta_rule",
]
