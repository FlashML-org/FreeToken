"""Pure-torch Gated DeltaNet reference (text-only, no cache).

Correctness oracle for the kernel-backed GDN op (``freetoken.layers.linear_attn.GatedDeltaNet``).
The delta rule and the forward are transcribed from
``transformers.models.qwen4_exp.modeling_qwen4_exp`` (``torch_recurrent_gated_delta_rule``
and ``Qwen4ExpTextGatedDeltaNet.forward`` no-cache path).
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


class _RMSNormGated(nn.Module):
    """RMSNorm of x followed by an ``activation(z)`` gate (norm_before_gate=True).

    Mirrors ``Qwen4ExpTextRMSNormGated`` over head_v_dim groups.
    """

    def __init__(self, dim: int, eps: float, activation: str = "silu"):
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


class GatedDeltaNetReference(nn.Module):
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
        output_gate: str = "silu",
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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

        core, _ = recurrent_gated_delta_rule(query, key, value, g, beta, use_qk_l2norm=True)

        core = core.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core = self.norm(core, z).reshape(b, t_len, -1)
        return self.out_proj(core)


__all__ = [
    "GatedDeltaNetReference",
    "recurrent_gated_delta_rule",
]
