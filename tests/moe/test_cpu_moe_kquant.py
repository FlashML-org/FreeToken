"""CPU MoE executor -- native GGUF K-quant experts (Q4_K, Q6_K).

Companion to test_cpu_moe_q4_0.py. Same contract: the CPU GEMV reads the *same* packed
banks the GPU offload path streams and dequantizes a block inside the K-loop, so it is
checked against the canonical CUDA dequant + the production bf16 GPU decode on
byte-identical banks. Both sides are W4A16, so the only spread is weight bf16-rounding and
reduction order.

Blocks are synthesized rather than carved out of a real checkpoint: every byte pattern is
a legal Q4_K/Q6_K block once the fp16 scale fields hold finite values, and a self-contained
fixture keeps this runnable without a multi-GB download.

Three bugs these tests exist to catch, all of which produce a model that loads, runs at
full speed and emits fluent nonsense:

* Q4_K quants are UNSIGNED 0..15 offset by a per-sub-block min. Reusing Q4_0's ``q - 8``
  bias reads as a plausible kernel and destroys the output (cosine ~0.008).
* Within a Q4_K 64-element group, a byte's high nibble is element l+32, not l+1, and it
  carries a different scale/min pair.
* The Q6_K ``qh`` shift is per-group only. Adding a term in the lane index corrupts
  exactly the upper half of every group, which still correlates ~0.17 and so can be
  mistaken for a tolerance problem.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

QK_K = 256
_Q4_K_BYTES, _Q6_K_BYTES = 144, 210
GGML_Q4_0, GGML_Q4_K, GGML_Q6_K = 2, 12, 14


def _fp16_bytes(vals: torch.Tensor) -> torch.Tensor:
    """[...] float -> [..., 2] uint8 little-endian fp16."""
    return vals.to(torch.float16).view(torch.uint8).reshape(*vals.shape, 2)


def _make_q4_k_rows(S: int, OUT: int, K: int, gen) -> torch.Tensor:
    """Random valid Q4_K rows: [S, OUT, K//256*144].

    Layout per block (ggml-common.h): half2 dm | uint8 scales[12] | uint8 qs[128]. The
    scale and quant bytes are unconstrained, so only dm has to be finite and sanely scaled.
    """
    nb = K // QK_K
    d = 0.02 + 0.03 * torch.rand(S, OUT, nb, generator=gen)
    dmin = 0.01 + 0.02 * torch.rand(S, OUT, nb, generator=gen)
    dm = torch.cat([_fp16_bytes(d), _fp16_bytes(dmin)], dim=-1)          # [S, OUT, nb, 4]
    rest = torch.randint(0, 256, (S, OUT, nb, 140), dtype=torch.uint8, generator=gen)
    return torch.cat([dm, rest], dim=-1).reshape(S, OUT, nb * _Q4_K_BYTES).contiguous()


def _make_q6_k_rows(S: int, OUT: int, K: int, gen) -> torch.Tensor:
    """Random valid Q6_K rows: [S, OUT, K//256*210].

    Layout per block: uint8 ql[128] | uint8 qh[64] | int8 scales[16] | half d.
    """
    nb = K // QK_K
    body = torch.randint(0, 256, (S, OUT, nb, 208), dtype=torch.uint8, generator=gen)
    d = 0.01 + 0.02 * torch.rand(S, OUT, nb, generator=gen)
    return torch.cat([body, _fp16_bytes(d)], dim=-1).reshape(
        S, OUT, nb * _Q6_K_BYTES).contiguous()


_MAKERS = {"q4_k": (_make_q4_k_rows, GGML_Q4_K), "q6_k": (_make_q6_k_rows, GGML_Q6_K)}


def _make_cache(fmt: str, L: int, E: int, H: int, I: int, seed: int = 0,
                *, as_gguf: bool = False):
    """Pinned host banks in the native K-quant schema, as the offload path builds them."""
    from freetoken.kernel.pinned import alloc_pinned_tensor

    make, ggml_type = _MAKERS[fmt]
    gen = torch.Generator().manual_seed(seed)
    S = L * E

    def rows(OUT, K):
        packed = make(S, OUT, K, gen)
        pinned = alloc_pinned_tensor(*packed.shape, dtype=torch.uint8)
        pinned.copy_(packed)
        return pinned

    return SimpleNamespace(
        # A real GGUF checkpoint tags the cache "gguf" and carries the ggml types
        # separately; the executor resolves that to the concrete format.
        quant_format="gguf" if as_gguf else fmt,
        gguf_expert_types=(ggml_type, ggml_type) if as_gguf else None,
        bank_sources={"gate_up": list(rows(2 * I, H).split(E)),
                      "down": list(rows(H, I).split(E))},
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


def _dequant_bank(packed: torch.Tensor, ggml_type: int, K: int, dev) -> torch.Tensor:
    """[S, OUT, row_bytes] packed -> [S, OUT, K] bf16 via the vendored CUDA dequant."""
    from freetoken.kernel.gguf import ggml_dequantize

    S, OUT, row_bytes = packed.shape
    flat = ggml_dequantize(
        packed.reshape(-1, row_bytes).to(dev).contiguous(), ggml_type, S * OUT, K,
        torch.bfloat16,
    )
    return flat.reshape(S, OUT, K)


@pytest.mark.parametrize("fmt", ["q4_k", "q6_k"])
@pytest.mark.parametrize("bs", [1, 3, 8])
def test_cpu_decode_kquant_matches_dequant_then_gpu(fmt, bs):
    """CPU inline-dequant K-quant GEMV vs. the CUDA dequant + bf16 GPU decode."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    L, E, H, I, top_k, layer = 2, 8, 512, 256, 4, 1
    dev = torch.device("cuda")
    cache = _make_cache(fmt, L, E, H, I, seed=100 + bs)
    ex = CpuMoeExecutor(cache, top_k=top_k, activation="silu",
                        apply_router_weight_on_input=False, num_threads=0,
                        max_tokens=bs, device=dev)

    torch.manual_seed(400 + bs)
    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k]
                       for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    ggml_type = _MAKERS[fmt][1]
    gu = _dequant_bank(cache.bank_sources["gate_up"][layer], ggml_type, H, dev)
    dn = _dequant_bank(cache.bank_sources["down"][layer], ggml_type, I, dev)
    gpu_out = fused_experts_decode_impl(hidden, gu, dn, w, ids.clone(), "silu", False).float()

    cos = torch.nn.functional.cosine_similarity(
        cpu_out.flatten(), gpu_out.flatten(), dim=0).item()
    rel = ((cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)).item()
    assert cos > 0.999, f"{fmt} bs={bs}: cosine {cos} (rel {rel})"
    assert rel < 5e-2, f"{fmt} bs={bs}: rel {rel} (cosine {cos})"


@pytest.mark.parametrize("fmt", ["q4_k", "q6_k"])
def test_gguf_cache_resolves_to_the_cpu_kernel(fmt):
    """A checkpoint tagged quant_format 'gguf' must reach the same kernel.

    The bank types live on the cache rather than in the format tag, so without the bridge
    the executor refuses every real GGUF checkpoint while accepting the literal 'q4_k'
    string that nothing actually produces.
    """
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    L, E, H, I = 2, 8, 512, 256
    cache = _make_cache(fmt, L, E, H, I, seed=7, as_gguf=True)
    ex = CpuMoeExecutor(cache, top_k=4, activation="silu",
                        apply_router_weight_on_input=False, num_threads=0,
                        max_tokens=2, device=torch.device("cuda"))
    assert ex.quant_format == fmt


class TestGgufFormatResolution:
    """The bridge's refusals. Each must name what is wrong and what to use instead."""

    def test_mixed_banks_refused(self):
        """Q4_K_M stores gate_up Q4_K and down Q6_K; one weight_format cannot serve both."""
        from freetoken.moe.cpu_executor import _resolve_gguf_format

        c = SimpleNamespace(quant_format="gguf",
                            gguf_expert_types=(GGML_Q4_K, GGML_Q6_K))
        with pytest.raises(NotImplementedError, match="(?i)mixed-type"):
            _resolve_gguf_format(c)

    def test_uniform_but_unsupported_type_refused(self):
        """IQ3_S banks are uniform but have no CPU dot kernel; offload must be named."""
        from freetoken.moe.cpu_executor import _resolve_gguf_format

        c = SimpleNamespace(quant_format="gguf", gguf_expert_types=(21, 21))
        with pytest.raises(NotImplementedError, match="(?i)no cpu kernel"):
            _resolve_gguf_format(c)

    @pytest.mark.parametrize("t,want", [(GGML_Q4_0, "q4_0"), (GGML_Q4_K, "q4_k"),
                                        (GGML_Q6_K, "q6_k")])
    def test_uniform_supported_types_resolve(self, t, want):
        from freetoken.moe.cpu_executor import _resolve_gguf_format

        c = SimpleNamespace(quant_format="gguf", gguf_expert_types=(t, t))
        assert _resolve_gguf_format(c) == want

    def test_missing_types_refused(self):
        from freetoken.moe.cpu_executor import _resolve_gguf_format

        c = SimpleNamespace(quant_format="gguf", gguf_expert_types=None)
        with pytest.raises(NotImplementedError):
            _resolve_gguf_format(c)
