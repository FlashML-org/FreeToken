"""Paged relative-bias attention kernel against the dense reference (``reference.attention``, checked against
transformers #51050's ``eager_attention_forward`` + ``Inkling2LoopRelativePositionBias``).

Covers a cached prefix plus a chunk (keys read back from scattered pool slots), distances past the 1024-token bias
extent, the sliding-window edge, decode and multi-row verify batches, and the sink logit."""

from __future__ import annotations

import pytest
import torch

from .reference import attention as ref_attention

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")

H, KVH, D, R, E = 24, 8, 128, 16, 1024


def _case(lens, seed):
    """Per request ``(prefix, rows)``: a pool with every key in a shuffled slot, the metadata the backend builds, and inputs."""
    g = torch.Generator().manual_seed(seed)
    total_kv = sum(p + r for p, r in lens)
    slots = torch.randperm(total_kv + 7, generator=g)[:total_kv] + 1  # slot 0 stays a sentinel
    k_cache = torch.randn(total_kv + 8, KVH, D, generator=g) * 0.4
    v_cache = torch.randn(total_kv + 8, KVH, D, generator=g)
    T = sum(r for _, r in lens)
    q = torch.randn(T, H, D, generator=g) * 0.4
    rel = torch.randn(T, H, R, generator=g)
    table = torch.randn(R, E, generator=g) * 0.5
    sinks = torch.randn(T, H, generator=g) * 2
    qo, kv, pos = [0], [0], []
    for p, r in lens:
        qo.append(qo[-1] + r)
        kv.append(kv[-1] + p + r)
        pos.extend(range(p, p + r))
    return dict(
        k_cache=k_cache, v_cache=v_cache, q=q, rel=rel, table=table, sinks=sinks, slots=slots, lens=lens,
        qo=torch.tensor(qo, dtype=torch.int32), kv=torch.tensor(kv, dtype=torch.int32), pos=torch.tensor(pos, dtype=torch.int32),
    )


def _run(case, window, dtype=torch.bfloat16):
    from freetoken.kernel.triton.inkling2_attention import inkling2_paged_attention

    dev = "cuda"
    c = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in case.items()}
    out = inkling2_paged_attention(
        c["q"].to(dtype), c["k_cache"].to(dtype), c["v_cache"].to(dtype),
        qo_indptr=c["qo"], kv_indptr=c["kv"], kv_indices=c["slots"].to(torch.int32), q_positions=c["pos"],
        max_q_len=max(r for _, r in case["lens"]), rel=c["rel"].to(dtype), table=c["table"].to(dtype),
        sinks=c["sinks"].to(dtype), sm_scale=D**-0.5, sliding_window=window,
    )
    errs = []
    for b, (p, r) in enumerate(case["lens"]):
        q0, k0 = int(case["qo"][b]), int(case["kv"][b])
        keys = case["slots"][k0 : k0 + p + r]
        cast = lambda t: t.to(dtype).float()  # the kernel sees bf16 inputs
        ref = ref_attention(
            cast(case["q"][q0 : q0 + r]), cast(case["k_cache"][keys]), cast(case["v_cache"][keys]),
            cast(case["rel"][q0 : q0 + r]), cast(case["table"]), cast(case["sinks"][q0 : q0 + r]),
            case["pos"][q0 : q0 + r].long(), torch.arange(p + r), D**-0.5, window,
        )
        got = out[q0 : q0 + r].float().cpu()
        errs.append(((got - ref).abs().max() / ref.abs().max()).item())
    return max(errs)


@requires_cuda
@pytest.mark.parametrize("window", [None, 1024])
@pytest.mark.parametrize(
    "lens",
    [
        [(0, 300)],  # fresh prefill
        [(1100, 77), (0, 5), (3000, 130)],  # prefix + chunk past the extent and the window
        [(0, 1), (1500, 1), (1023, 1), (1024, 1)],  # decode at the window edge
        [(2047, 4), (17, 4)],  # verify rows
    ],
)
def test_matches_reference(lens, window):
    assert _run(_case(lens, seed=len(lens)), window) < 2e-2


@requires_cuda
def test_sink_absorbs_mass():
    # a large sink takes most of the mass: the kernel must treat it as a zero-value key, not a skipped one
    case = _case([(40, 9)], seed=3)
    case["sinks"] = torch.full_like(case["sinks"], 6.0)
    assert _run(case, None) < 2e-2
