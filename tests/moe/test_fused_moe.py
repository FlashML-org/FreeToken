import pytest
import torch


def _activation_and_mul(gate_up: torch.Tensor, activation: str, alpha: float = 1.0, limit: float = float("inf")) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    if activation == "silu":
        return torch.nn.functional.silu(gate) * up
    if activation == "gelu":
        return torch.nn.functional.gelu(gate) * up
    if activation == "swigluoai":
        gate = gate.clamp(max=limit)
        return gate / (1.0 + torch.exp(-alpha * gate)) * (up.clamp(-limit, limit) + 1.0)
    raise AssertionError(f"unsupported test activation {activation}")


def _reference_fused_experts_decode(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.0,
    act_limit: float = float("inf"),
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    hidden_fp32 = hidden_states.float()
    w1_fp32 = w1.float()
    w2_fp32 = w2.float()
    for token_idx in range(hidden_states.shape[0]):
        for route_idx in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_idx, route_idx])
            route_weight = topk_weights[token_idx, route_idx].float()
            gate_up = torch.matmul(w1_fp32[expert_id], hidden_fp32[token_idx])
            if w1_bias is not None:
                gate_up = gate_up + w1_bias[expert_id].float()
            if apply_router_weight_on_input:
                gate_up = gate_up * route_weight
            activated = _activation_and_mul(gate_up, activation, alpha=act_alpha, limit=act_limit)
            contribution = torch.matmul(w2_fp32[expert_id], activated)
            if w2_bias is not None:
                contribution = contribution + w2_bias[expert_id].float()
            if not apply_router_weight_on_input:
                contribution = contribution * route_weight
            output[token_idx] += contribution
    return output.to(hidden_states.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_topk_accepts_triton_kernel_tuple_output():
    from freetoken.moe.fused import fused_topk

    logits = torch.tensor(
        [[4.0, 1.0, -1.0, 2.0], [0.5, 3.0, 2.0, -2.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden_states = torch.zeros((2, 8), device="cuda", dtype=torch.bfloat16)

    weights, ids = fused_topk(hidden_states, logits, topk=2, renormalize=True)

    ref_logits, ref_ids = torch.topk(logits.float(), 2, dim=-1)
    ref_weights = torch.softmax(ref_logits, dim=-1)
    torch.testing.assert_close(ids, ref_ids.to(torch.int32))
    torch.testing.assert_close(weights, ref_weights, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 24])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_experts_decode_matches_reference_for_non_contiguous_slots(batch_size, dtype):
    from freetoken.moe.fused import fused_experts_decode_impl

    device = torch.device("cuda")
    cache_size = 37
    hidden_size = 32
    intermediate_size = 24
    top_k = 8
    torch.manual_seed(42 + batch_size)

    hidden_states = 0.5 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        cache_size,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    down = torch.randn(
        cache_size,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    weights = torch.rand(batch_size, top_k, device=device, dtype=torch.float32)
    topk_weights = weights / weights.sum(dim=-1, keepdim=True)
    slot_ids = torch.tensor([31, 4, 18, 0, 29, 7, 35, 12], device=device, dtype=torch.int32)
    topk_ids = slot_ids.repeat(batch_size, 1).contiguous()

    output = fused_experts_decode_impl(hidden_states, gate_up, down, topk_weights, topk_ids)
    expected = _reference_fused_experts_decode(hidden_states, gate_up, down, topk_weights, topk_ids)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_fused_experts_decode_matches_grouped_impl(dtype, apply_router_weight_on_input):
    from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl

    device = torch.device("cuda")
    batch_size = 4
    num_experts = 37
    hidden_size = 32
    intermediate_size = 24
    top_k = 4
    torch.manual_seed(91)

    hidden_states = 0.25 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    down = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    weights = torch.rand(batch_size, top_k, device=device, dtype=torch.float32)
    topk_weights = weights / weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.tensor(
        [[31, 4, 18, 0], [29, 7, 35, 12], [6, 21, 1, 33], [16, 3, 28, 9]],
        device=device,
        dtype=torch.int32,
    )

    output = fused_experts_decode_impl(
        hidden_states.clone(),
        gate_up,
        down,
        topk_weights,
        topk_ids.clone(),
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    expected = fused_experts_impl(
        hidden_states.clone(),
        gate_up,
        down,
        topk_weights,
        topk_ids.clone(),
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("impl_name", ["decode", "grouped"])
def test_fused_experts_with_per_expert_biases_match_reference(impl_name):
    """gpt-oss's expert banks carry per-expert biases: w1 before the gated activation, w2 before the routing weight."""
    from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl

    device = torch.device("cuda")
    batch_size = 4
    num_experts = 37
    hidden_size = 32
    intermediate_size = 24
    top_k = 4
    torch.manual_seed(104)

    hidden_states = 0.25 * torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
    gate_up = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    ) * 0.25
    down = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=torch.bfloat16,
    ) * 0.25
    w1_bias = 0.5 * torch.randn(num_experts, intermediate_size * 2, device=device, dtype=torch.bfloat16)
    w2_bias = 0.5 * torch.randn(num_experts, hidden_size, device=device, dtype=torch.bfloat16)
    weights = torch.rand(batch_size, top_k, device=device, dtype=torch.float32)
    topk_weights = weights / weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.tensor(
        [[31, 4, 18, 0], [29, 7, 35, 12], [6, 21, 1, 33], [16, 3, 28, 9]],
        device=device,
        dtype=torch.int32,
    )

    impl = fused_experts_decode_impl if impl_name == "decode" else fused_experts_impl
    output = impl(
        hidden_states.clone(),
        gate_up,
        down,
        topk_weights,
        topk_ids,
        activation="swigluoai",
        act_alpha=1.702,
        act_limit=7.0,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )
    expected = _reference_fused_experts_decode(
        hidden_states,
        gate_up,
        down,
        topk_weights,
        topk_ids,
        activation="swigluoai",
        act_alpha=1.702,
        act_limit=7.0,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_experts_grouped_impl_is_cuda_graph_capturable():
    from freetoken.moe.fused import fused_experts_impl

    device = torch.device("cuda")
    dtype = torch.float16
    batch_size = 1
    num_experts = 8
    hidden_size = 32
    intermediate_size = 24
    torch.manual_seed(17)

    hidden_states = 0.25 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    down = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.25
    topk_weights = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device=device, dtype=torch.float32)
    topk_ids = torch.tensor([[6, 1, 4, 7]], device=device, dtype=torch.int32)
    output = torch.empty_like(hidden_states)

    def run():
        output.copy_(
            fused_experts_impl(
                hidden_states,
                gate_up,
                down,
                topk_weights,
                topk_ids,
            )
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("activation", ["silu", "gelu"])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_fused_experts_decode_activation_and_router_weight_modes(
    activation,
    apply_router_weight_on_input,
):
    from freetoken.moe.fused import fused_experts_decode_impl

    device = torch.device("cuda")
    batch_size = 3
    cache_size = 19
    hidden_size = 32
    intermediate_size = 24
    top_k = 4
    dtype = torch.float16
    torch.manual_seed(123)

    hidden_states = 0.5 * torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    gate_up = torch.randn(
        cache_size,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    down = torch.randn(
        cache_size,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * 0.5
    topk_weights = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1], [0.15, 0.35, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]],
        device=device,
        dtype=torch.float32,
    )
    topk_ids = torch.tensor(
        [[17, 2, 11, 5], [3, 17, 0, 14], [8, 6, 12, 1]],
        device=device,
        dtype=torch.int32,
    )

    output = fused_experts_decode_impl(
        hidden_states,
        gate_up,
        down,
        topk_weights,
        topk_ids,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    expected = _reference_fused_experts_decode(
        hidden_states,
        gate_up,
        down,
        topk_weights,
        topk_ids,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_topk_handles_non_power_of_2_k():
    """A top-10 router (qwen4_exp) must route like any other k."""
    from freetoken.moe.fused import _torch_fused_topk, fused_topk

    gating = torch.randn(5, 64, device="cuda")
    hidden = torch.randn(5, 8, device="cuda")
    weights, ids = fused_topk(hidden, gating, 10, renormalize=True)
    ref_w, ref_i = _torch_fused_topk(gating, 10, True, None)
    assert torch.equal(ids, ref_i)
    torch.testing.assert_close(weights, ref_w, rtol=1e-5, atol=1e-6)


# The in-repo triton router behind fused_topk; fp32 logits keep the reference top-k tie-free.
# Ties get their own case below, because torch.topk does not break them by expert id.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [(1, 512, 10), (7, 512, 10), (129, 512, 10), (33, 512, 6), (4, 64, 3)],
)
def test_fused_topk_softmax_matches_torch_reference(num_tokens, num_experts, topk, renormalize):
    from freetoken.kernel.triton.moe_router import fused_topk_softmax
    from freetoken.moe.fused import _torch_fused_topk

    gen = torch.Generator(device="cuda").manual_seed(num_tokens * 31 + topk)
    gating = torch.randn(num_tokens, num_experts, generator=gen, device="cuda")

    weights, ids = fused_topk_softmax(gating, topk, renormalize)
    ref_w, ref_i = _torch_fused_topk(gating, topk, renormalize, None)

    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert torch.equal(ids, ref_i)
    torch.testing.assert_close(weights, ref_w, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_topk_softmax_ties_pick_the_lowest_expert_id():
    from freetoken.kernel.triton.moe_router import fused_topk_softmax

    gating = torch.full((2, 8), -10.0, device="cuda")
    gating[0, [6, 2, 5]] = 1.0  # three-way tie for two slots
    gating[1] = 0.0  # whole row tied

    weights, ids = fused_topk_softmax(gating, 3, renormalize=True)

    assert ids[0].tolist() == [2, 5, 6]
    assert ids[1].tolist() == [0, 1, 2]
    torch.testing.assert_close(weights, torch.full((2, 3), 1.0 / 3.0, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("limit_dtype", [torch.int32, torch.int64])
def test_fused_topk_softmax_masks_padded_rows(limit_dtype):
    from freetoken.kernel.triton.moe_router import fused_topk_softmax
    from freetoken.moe.fused import _torch_fused_topk

    gen = torch.Generator(device="cuda").manual_seed(5)
    gating = torch.randn(16, 512, generator=gen, device="cuda")
    limit = torch.tensor(5, dtype=limit_dtype, device="cuda")

    weights, ids = fused_topk_softmax(gating, 10, True, limit)
    ref_w, ref_i = _torch_fused_topk(gating, 10, True, limit)

    assert (ids[5:] == -1).all()
    assert torch.equal(ids, ref_i)
    torch.testing.assert_close(weights, ref_w, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_topk_softmax_is_cuda_graph_capturable():
    """The padded-row limit must come off the device tensor, not a host read baked into the graph."""
    from freetoken.kernel.triton.moe_router import fused_topk_softmax

    gen = torch.Generator(device="cuda").manual_seed(11)
    gating = torch.randn(8, 512, generator=gen, device="cuda")
    limit = torch.tensor(8, dtype=torch.int32, device="cuda")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused_topk_softmax(gating, 10, True, limit)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, ids = fused_topk_softmax(gating, 10, True, limit)

    limit.fill_(3)
    graph.replay()
    torch.cuda.synchronize()
    assert (ids[:3] != -1).all()
    assert (ids[3:] == -1).all()


@pytest.mark.parametrize(
    "rows,num_experts,hidden,topk,folds",
    [
        (1, 256, 2048, 8, True),  # Qwen3.5/3.6 MoE, one row
        (6, 256, 2048, 8, True),  # MTP-5 verify
        (7, 256, 2048, 8, False),  # past the decode-sized batches
        (16, 256, 2048, 8, False),
        (0, 256, 2048, 8, False),
        (1, 512, 1024, 10, True),
        (1, 128, 2048, 8, False),  # the router packs several rows a program here
        (1, 1024, 2048, 8, False),  # and runs four warps here
        (1, 256, 2000, 8, False),  # no whole GEMV slices
    ],
)
def test_router_gemv_topk_folds_only_where_the_router_runs_one_row_a_warp(rows, num_experts, hidden, topk, folds):
    """The fold routes its logits with the softmax router's own program; that holds only for its one-row, one-warp tile."""
    from freetoken.kernel.triton.moe_router import router_gemv_topk_folds

    assert router_gemv_topk_folds(rows, num_experts, hidden, topk) is folds


def test_qwen3_5_moe_says_once_why_its_router_does_not_fold(monkeypatch):
    """One line a process, at the first decode step, naming the predicate that keeps the router's linear apart."""
    from types import SimpleNamespace

    from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel
    from freetoken.models.qwen3_5_moe import moe as qmoe

    said = []
    monkeypatch.setattr(qmoe, "_FOLD_SAID", False)
    monkeypatch.setattr(qmoe, "_ROUTER_FOLD", True)
    monkeypatch.setattr(qmoe.logger, "info_rank0", lambda msg, *a, **k: said.append(msg), raising=False)
    monkeypatch.setattr(qmoe, "get_global_ctx", lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=False)))

    def block(weight, bias=None):
        m = object.__new__(qmoe.Qwen3_5MoE)
        m.gate = SimpleNamespace(weight=weight, bias=bias, quant_method=SimpleNamespace(kernel=TorchLinearKernel()))
        m.experts = SimpleNamespace(top_k=8, renormalize=True)
        m._router_weight, m._router_off = False, None
        return m

    x = torch.zeros(1, 2048, dtype=torch.bfloat16)
    first = block(torch.zeros(256, 2048, dtype=torch.bfloat16))
    assert first._routes_in_kernel(x) is None
    assert block(torch.zeros(256, 2048, dtype=torch.bfloat16), bias=torch.zeros(256))._routes_in_kernel(x) is None
    assert said == ["router fold on decode: off (the router's weight is on cpu)"]
    assert block(torch.zeros(256, 2048), bias=torch.zeros(256))._router_weight_off() == "the router has a bias"
    assert block(torch.zeros(256, 2048))._router_weight_off() == "the router's weight is on cpu"


def test_qwen3_5_moe_router_fold_off_keeps_the_ninth_route_off_too(monkeypatch):
    """FREETOKEN_ROUTER_FOLD=0: no decode step folds the router, so none takes the ninth route whose gate rides the
    fold's launch; the one line names the switch, checked before any property of the weight."""
    from types import SimpleNamespace

    from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel
    from freetoken.models.qwen3_5_moe import moe as qmoe

    said = []
    monkeypatch.setattr(qmoe, "_FOLD_SAID", False)
    monkeypatch.setattr(qmoe, "_ROUTER_FOLD", False)
    monkeypatch.setattr(qmoe, "_ROUTER_FOLD_SET", "0")
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE", True)
    monkeypatch.setattr(qmoe.logger, "info_rank0", lambda msg, *a, **k: said.append(msg), raising=False)
    monkeypatch.setattr(qmoe, "get_global_ctx", lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=False)))

    m = object.__new__(qmoe.Qwen3_5MoE)
    m.gate = SimpleNamespace(weight=torch.zeros(256, 2048, dtype=torch.bfloat16), bias=torch.zeros(256),
                             quant_method=SimpleNamespace(kernel=TorchLinearKernel()))
    m.experts = SimpleNamespace(top_k=8, renormalize=True)
    m._router_weight, m._router_off = False, None
    m._shared_route = {"gate_up": torch.empty(0)}   # built: only the fold's decision keeps it off
    x = torch.zeros(6, 2048, dtype=torch.bfloat16)
    assert m._shared_route_for(x) is None
    assert m._routes_in_kernel(x) is None
    assert m._router_weight is None and m._router_off == "FREETOKEN_ROUTER_FOLD=0"
    assert said == ["router fold on decode: off (FREETOKEN_ROUTER_FOLD=0)"]


def _router_gemv_inputs(rows, num_experts, hidden, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, hidden, generator=gen, device="cuda").to(torch.bfloat16)
    w = (torch.randn(num_experts, hidden, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    return x, w


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize(
    "rows,num_experts,hidden,topk",
    [
        (1, 256, 2048, 8),
        (6, 256, 2048, 8),
        (3, 512, 1024, 10),
        # expert counts whose row stride is not a multiple of 16: the kernels' loads vectorize differently
        (2, 200, 512, 8),
        (6, 200, 512, 8),
        (5, 136, 768, 6),
    ],
)
def test_router_gemv_topk_routes_its_logits_as_the_router_does(rows, num_experts, hidden, topk, renormalize):
    from freetoken.kernel.triton.moe_router import fused_topk_softmax, router_gemv_topk_softmax

    x, w = _router_gemv_inputs(rows, num_experts, hidden, rows * 7 + num_experts)
    logits = torch.empty(rows, num_experts, dtype=torch.bfloat16, device="cuda")
    weights, ids = router_gemv_topk_softmax(x, w, topk, renormalize, logits_out=logits)
    ref_w, ref_i = fused_topk_softmax(logits, topk, renormalize)

    assert torch.equal(ids, ref_i)
    if not renormalize:
        assert torch.equal(weights.view(torch.int32), ref_w.view(torch.int32))
        return
    # The same picks, renormalized by their sum in selection order (the router sums its slots in its own
    # layout's order, which the prefill keeps): the picks summed one by one in fp32, then Triton's division.
    picks, _ = fused_topk_softmax(logits, topk, False)
    total = torch.zeros(rows, dtype=torch.float32, device="cuda")
    for k in range(topk):
        total = total + picks[:, k]
    want = picks / total[:, None]
    ulp = torch.ldexp(torch.ones_like(want), torch.frexp(want).exponent - 24)
    assert ((weights - want).abs() <= 3 * ulp).all()
    torch.testing.assert_close(weights, ref_w, rtol=4e-6, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows,num_experts,hidden", [(1, 256, 2048), (6, 256, 2048), (2, 200, 512), (6, 200, 512)])
def test_router_gemv_topk_logits_round_the_exact_dot(rows, num_experts, hidden):
    """Against the fp64 dot, not F.linear, whose split-K partials cuBLAS may round: one bf16 step from the final
    rounding plus fp32 summation error; a slice read stale or twice is off by far more."""
    from freetoken.kernel.triton.moe_router import router_gemv_topk_softmax

    x, w = _router_gemv_inputs(rows, num_experts, hidden, 100 + rows)
    logits = torch.empty(rows, num_experts, dtype=torch.bfloat16, device="cuda")
    router_gemv_topk_softmax(x, w, 8, True, logits_out=logits)
    exact = x.double() @ w.double().t()

    big = torch.maximum(logits.double().abs(), exact.abs())
    step = torch.ldexp(torch.ones_like(big), torch.frexp(big).exponent - 8)
    summed = x.double().abs() @ w.double().abs().t()
    assert ((logits.double() - exact).abs() <= step + 1e-5 * summed).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_router_gemv_topk_rows_do_not_depend_on_the_batch():
    from freetoken.kernel.triton.moe_router import router_gemv_topk_softmax

    x, w = _router_gemv_inputs(6, 256, 2048, 3)
    logits = torch.empty(6, 256, dtype=torch.bfloat16, device="cuda")
    weights, ids = router_gemv_topk_softmax(x, w, 8, True, logits_out=logits)
    for r in range(6):
        one = torch.empty(1, 256, dtype=torch.bfloat16, device="cuda")
        w1, i1 = router_gemv_topk_softmax(x[r : r + 1], w, 8, True, logits_out=one)
        assert torch.equal(one.view(torch.int16), logits[r : r + 1].view(torch.int16))
        assert torch.equal(i1, ids[r : r + 1])
        assert torch.equal(w1.view(torch.int32), weights[r : r + 1].view(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", [1, 6])
def test_router_gemv_topk_replays_in_a_cuda_graph(rows):
    """The arrival counters reset themselves, so every replay routes its own input."""
    from freetoken.kernel.triton.moe_router import router_gemv_topk_softmax

    x, w = _router_gemv_inputs(rows, 256, 2048, 21)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        router_gemv_topk_softmax(x, w, 8, True)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = router_gemv_topk_softmax(x, w, 8, True)

    gen = torch.Generator(device="cuda").manual_seed(22)
    for _ in range(3):
        x.copy_(torch.randn(x.shape, generator=gen, device="cuda").to(torch.bfloat16))
        graph.replay()
        ref_w, ref_i = router_gemv_topk_softmax(x, w, 8, True)
        torch.cuda.synchronize()
        assert torch.equal(ids, ref_i)
        assert torch.equal(weights.view(torch.int32), ref_w.view(torch.int32))


def _shared_gate(hidden, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(1, hidden, generator=gen, device="cuda") * 0.05).to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize(
    "rows,num_experts,hidden",
    [
        (1, 256, 2048),
        (2, 256, 2048),
        (6, 256, 2048),
        # a row stride Triton cannot prove a multiple of 16: the gate's partials must not move the tail's layout
        (2, 200, 512),
        (6, 200, 512),
    ],
)
def test_router_gemv_topk_shared_gate_adds_the_shared_experts_route(rows, num_experts, hidden, renormalize):
    """The shared expert gate as one more row of the fold: its top-k columns are the fold's without it, bit for bit,
    and the last column is id num_experts with torch's bf16 sigmoid of the gate's own bf16 logit, which keeps the
    fp64 bound."""
    from freetoken.kernel.triton.moe_router import router_gemv_topk_softmax

    x, w = _router_gemv_inputs(rows, num_experts, hidden, 40 + rows + num_experts)
    gate = _shared_gate(hidden, 50 + rows)
    logits = torch.empty(rows, num_experts, dtype=torch.bfloat16, device="cuda")
    ref_w, ref_i = router_gemv_topk_softmax(x, w, 8, renormalize, logits_out=logits)
    logits2 = torch.empty_like(logits)
    glog = torch.empty(rows, dtype=torch.bfloat16, device="cuda")
    weights, ids = router_gemv_topk_softmax(x, w, 8, renormalize, logits_out=logits2, shared_gate=gate, gate_logit_out=glog)

    assert weights.shape == (rows, 9) and ids.shape == (rows, 9)
    assert torch.equal(logits2.view(torch.int16), logits.view(torch.int16))
    assert torch.equal(ids[:, :8], ref_i)
    assert torch.equal(weights[:, :8].contiguous().view(torch.int32), ref_w.view(torch.int32))
    assert (ids[:, 8] == num_experts).all()
    assert torch.equal(weights[:, 8].contiguous().view(torch.int32), torch.sigmoid(glog).float().view(torch.int32))

    exact = (x.double() @ gate.double().t()).view(-1)
    big = torch.maximum(glog.double().abs(), exact.abs())
    step = torch.ldexp(torch.ones_like(big), torch.frexp(big).exponent - 8)
    summed = (x.double().abs() @ gate.double().abs().t()).view(-1)
    assert ((glog.double() - exact).abs() <= step + 1e-5 * summed).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_router_gemv_topk_shared_gate_weight_is_torch_sigmoid_of_the_bf16_logit():
    """Logits set exactly (one nonzero product a row) over a spread of bf16 values, the saturated tails and the
    infinities included: the gate's route weight is torch.sigmoid of the bf16 logit, bit for bit."""
    from freetoken.kernel.triton.moe_router import router_gemv_topk_softmax

    values = torch.arange(0, 1 << 16, 3, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    values = values[~values.isnan()]
    values = torch.cat([values, torch.tensor([float("inf"), float("-inf"), 0.0, -0.0])]).cuda()
    w = (torch.randn(256, 2048, device="cuda") * 0.02).to(torch.bfloat16)
    gate = torch.zeros(1, 2048, dtype=torch.bfloat16, device="cuda")
    gate[0, 0] = 1.0
    for chunk in values.split(6):  # the fold takes up to six rows a call
        x = torch.zeros(chunk.numel(), 2048, dtype=torch.bfloat16, device="cuda")
        x[:, 0] = chunk.to(torch.bfloat16)
        glog = torch.empty(chunk.numel(), dtype=torch.bfloat16, device="cuda")
        weights, _ = router_gemv_topk_softmax(x, w, 8, True, shared_gate=gate, gate_logit_out=glog)
        # values, not bits: the dot's sum starts from +0, so a -0 logit comes back +0
        assert torch.equal(glog.float(), x[:, 0].float())
        assert torch.equal(weights[:, 8].contiguous().view(torch.int32), torch.sigmoid(glog).float().view(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", [1, 6])
def test_router_gemv_topk_shared_gate_replays_in_a_cuda_graph(rows):
    """The extra tile counts among the arrivals, so the counters still reset themselves at every replay."""
    from freetoken.kernel.triton.moe_router import router_gemv_topk_softmax

    x, w = _router_gemv_inputs(rows, 256, 2048, 23)
    gate = _shared_gate(2048, 24)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        router_gemv_topk_softmax(x, w, 8, True, shared_gate=gate)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = router_gemv_topk_softmax(x, w, 8, True, shared_gate=gate)

    gen = torch.Generator(device="cuda").manual_seed(25)
    for _ in range(3):
        x.copy_(torch.randn(x.shape, generator=gen, device="cuda").to(torch.bfloat16))
        graph.replay()
        ref_w, ref_i = router_gemv_topk_softmax(x, w, 8, True, shared_gate=gate)
        # and the fold without the gate after it: the counters were left at zero
        plain_w, plain_i = router_gemv_topk_softmax(x, w, 8, True)
        torch.cuda.synchronize()
        assert torch.equal(ids, ref_i) and torch.equal(weights.view(torch.int32), ref_w.view(torch.int32))
        assert torch.equal(plain_i, ref_i[:, :8])


def _reference_sigmoid_topk(logits, bias, topk, renormalize, scale, n_group, topk_group, num_token_non_padded):
    """The torch routing of the GLM / MiniMax MoE blocks: sigmoid, bias for selection only, optional group-limited top-k."""
    scores = logits.float().sigmoid()
    choice = scores + bias.float()
    if n_group > 1:
        m, e = choice.shape
        group_scores = choice.view(m, n_group, e // n_group).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0).bool()
        choice = choice.masked_fill(~group_mask.repeat_interleave(e // n_group, dim=1), float("-inf"))
    _, ids = torch.topk(choice, topk, dim=-1)
    weights = scores.gather(-1, ids)
    if renormalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    ids = ids.to(torch.int32)
    ids[torch.arange(ids.shape[0], device=ids.device) >= num_token_non_padded] = -1
    return weights * scale, ids


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_tokens", [1, 7, 2048])
@pytest.mark.parametrize(
    "num_experts,topk,n_group,topk_group,renormalize,scale",
    [
        (128, 8, 1, 1, True, 1.0),  # GLM-4.5-Air
        (160, 8, 1, 1, True, 2.5),  # GLM-4.7
        (288, 8, 1, 1, True, 2.5),  # GLM-5.3-Flash
        (128, 4, 1, 1, True, 2.0),  # MiniMax-M3
        (256, 8, 8, 4, True, 2.5),  # DeepSeek-V3 group-limited top-k
        (96, 6, 3, 2, False, 1.0),  # odd group count, no renormalization
    ],
)
def test_fused_topk_sigmoid_matches_torch_reference(num_tokens, num_experts, topk, n_group, topk_group, renormalize, scale):
    from freetoken.kernel.triton.moe_router import fused_topk_sigmoid

    gen = torch.Generator(device="cuda").manual_seed(num_tokens * 31 + num_experts * 7 + topk)
    logits = torch.randn(num_tokens, num_experts, generator=gen, device="cuda")
    bias = torch.randn(num_experts, generator=gen, device="cuda") * 0.1
    limit = torch.tensor(max(1, num_tokens - 3), dtype=torch.int32, device="cuda")

    weights, ids = fused_topk_sigmoid(logits, bias, topk, renormalize, scale, n_group, topk_group, limit)
    ref_w, ref_i = _reference_sigmoid_topk(logits, bias, topk, renormalize, scale, n_group, topk_group, limit)

    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert torch.equal(ids, ref_i)
    torch.testing.assert_close(weights, ref_w, rtol=1e-5, atol=1e-6)


def test_qwen3_5_moe_router_fold_is_off_and_silent_with_its_switch_unset(monkeypatch):
    """FREETOKEN_ROUTER_FOLD unset: the router's linear and its top-k stay two launches and no line is logged."""
    from types import SimpleNamespace

    from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel
    from freetoken.models.qwen3_5_moe import moe as qmoe
    from freetoken.utils.misc import opt_in_env

    assert opt_in_env(None) == ("", False)
    said = []
    monkeypatch.setattr(qmoe, "_FOLD_SAID", False)
    monkeypatch.setattr(qmoe, "_ROUTER_FOLD", False)
    monkeypatch.setattr(qmoe, "_ROUTER_FOLD_SET", "")
    monkeypatch.setattr(qmoe.logger, "info_rank0", lambda msg, *a, **k: said.append(msg), raising=False)
    monkeypatch.setattr(qmoe, "get_global_ctx", lambda: SimpleNamespace(batch=SimpleNamespace(is_prefill=False)))

    m = object.__new__(qmoe.Qwen3_5MoE)
    m.gate = SimpleNamespace(weight=torch.zeros(256, 2048, dtype=torch.bfloat16), bias=None,
                             quant_method=SimpleNamespace(kernel=TorchLinearKernel()))
    m.experts = SimpleNamespace(top_k=8, renormalize=True)
    m._router_weight, m._router_off = False, None
    assert m._routes_in_kernel(torch.zeros(1, 2048, dtype=torch.bfloat16)) is None
    assert m._router_weight is None and m._router_off == "FREETOKEN_ROUTER_FOLD unset"
    assert said == []
