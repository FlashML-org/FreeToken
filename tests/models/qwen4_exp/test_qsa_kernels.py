"""The modified and original QSA Triton kernels against pure-torch references.

Only kernels FreeToken changed or wrote get unit tests: the compression kernel (re-addressed
pending ring, its own torch check) and the block top-k (original radix select, checked against
torch.topk and, through the expansion chain, against the vLLM reference semantics). score.py
and attend.py are vendored from vLLM and are covered by the backend and e2e tests.
``_qsa_mqa_paged_reference`` / ``_qsa_relative_topk_reference`` / ``_expand_qsa_indices_reference``
are transcribed from ``vllm/tests/test_qsa_reference.py`` (Apache-2.0).
"""

from __future__ import annotations

import math

import pytest
import torch

from .common import Fixture, requires_cuda, parsed_config

PAGE_SIZE = 64
RATIO = 4
BUDGET = 2048
INDEX_DIM = 128
CMP_PAGE = PAGE_SIZE // RATIO


# --------------------------------------------------------------------------------------
# vLLM pure-torch references (tests/test_qsa_reference.py:87-227)
# --------------------------------------------------------------------------------------


def _qsa_mqa_paged_reference(q, k_cache, page_table, token_to_req, visible_lengths):
    pages = page_table.index_select(0, token_to_req.long()).long()
    keys = k_cache[pages, :, 0, :].flatten(1, 2)
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys.float())
    logits = torch.relu(scores).sum(dim=-1) / math.sqrt(q.shape[-1])
    positions = torch.arange(keys.shape[1], device=q.device).unsqueeze(0)
    return logits.masked_fill(positions >= visible_lengths.unsqueeze(1), -torch.inf)


def _qsa_relative_topk_reference(logits, row_starts, row_ends, topk):
    output = torch.full((logits.shape[0], topk), -1, dtype=torch.int32, device=logits.device)
    for row in range(logits.shape[0]):
        start = int(row_starts[row].item())
        length = int((row_ends[row] - row_starts[row]).item())
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(
                logits[row, start : start + length], width
            ).indices.to(torch.int32)
    return output


def _expand_qsa_indices_reference(
    block_indices, query_positions, sequence_lengths, compress_ratio, token_topk
):
    rows = block_indices.shape[0]
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    offsets = torch.arange(compress_ratio, device=block_indices.device)
    blocks = block_indices.long()
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=block_indices.device)
    visible_tokens = query_positions + 1
    tail_start = visible_tokens // compress_ratio * compress_ratio
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_count = (visible_tokens - tail_start).unsqueeze(1)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat((expanded, tail), dim=1)
    order = torch.arange(output_width, device=result.device).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + output_width)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


class _Case:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def _paged_case(length: int, bs: int, rows_per_req: int, seed: int, index_heads: int = 4):
    """Synthetic paged geometry: shuffled pages, the last ``rows_per_req`` queries per request."""
    device = torch.device("cuda")
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    pages_per_req = -(-length // PAGE_SIZE)
    total_pages = bs * pages_per_req
    block_table = (
        torch.randperm(total_pages, device=device).reshape(bs, pages_per_req).to(torch.int32)
    )
    q = torch.randn(
        bs * rows_per_req, index_heads, INDEX_DIM, device=device, dtype=torch.bfloat16,
        generator=generator,
    )
    token_to_req = torch.repeat_interleave(
        torch.arange(bs, device=device, dtype=torch.int32), rows_per_req
    )
    query_positions = torch.cat(
        [
            torch.arange(length - rows_per_req, length, device=device, dtype=torch.int32)
            for _ in range(bs)
        ]
    )
    seq_lens = torch.full((bs,), length, device=device, dtype=torch.int32)
    return _Case(
        device=device,
        generator=generator,
        length=length,
        bs=bs,
        pages_per_req=pages_per_req,
        total_pages=total_pages,
        block_table=block_table,
        q=q,
        token_to_req=token_to_req,
        query_positions=query_positions,
        seq_lens=seq_lens,
    )


@requires_cuda
@pytest.mark.parametrize("length", [20000])
@pytest.mark.parametrize("bs", [1])
@pytest.mark.parametrize("torch_topk", [False, True])
def test_top_blocks_and_expansion_match_vllm_reference(length: int, bs: int, torch_topk: bool):
    from freetoken.kernel.triton.qsa import expand_qsa_block_indices, qsa_mqa_paged

    config = parsed_config()
    fixture = Fixture(config, num_pages=4, max_running_req=2)
    case = _paged_case(length, bs, rows_per_req=4, seed=7 * length + bs)
    cache = torch.randn(
        case.total_pages, CMP_PAGE, 1, INDEX_DIM, device=case.device,
        dtype=torch.bfloat16, generator=case.generator,
    )
    rows, columns = case.q.shape[0], case.pages_per_req * CMP_PAGE
    logits = torch.empty(rows, columns, dtype=torch.float32, device=case.device)
    visible = torch.empty(rows, dtype=torch.int32, device=case.device)
    qsa_mqa_paged(
        case.q, cache, case.block_table, case.token_to_req, case.query_positions,
        case.seq_lens, RATIO, logits, visible,
    )
    blocks = torch.empty(rows, BUDGET // RATIO, dtype=torch.int32, device=case.device)
    reference_logits = _qsa_mqa_paged_reference(
        case.q, cache, case.block_table, case.token_to_req, visible
    )
    if torch_topk:
        fixture.backend._block_topk_kernel = None
    fixture.backend._top_blocks(logits, visible, blocks)

    expected_blocks = _qsa_relative_topk_reference(
        reference_logits, torch.zeros_like(visible), visible, BUDGET // RATIO
    )
    # Ties between equal scores may land on either index; the SET is what selection means.
    torch.testing.assert_close(blocks.sort(-1).values, expected_blocks.sort(-1).values)

    row_seq_lens = case.seq_lens.index_select(0, case.token_to_req.long())
    indices = torch.empty(rows, BUDGET + RATIO - 1, dtype=torch.int32, device=case.device)
    expand_qsa_block_indices(
        expected_blocks, case.query_positions, case.seq_lens, case.token_to_req,
        RATIO, BUDGET, indices,
    )
    expected = _expand_qsa_indices_reference(
        expected_blocks, case.query_positions, row_seq_lens, RATIO, BUDGET
    )
    torch.testing.assert_close(indices, expected)


@requires_cuda
@pytest.mark.parametrize("ring_capacity", [4, 8])
def test_compression_reads_both_sources(ring_capacity: int):
    """Members already consumed come from the ring, the rest from this forward's raw rows."""
    from freetoken.kernel.triton.qsa import qsa_compress_groups, qsa_store_rows

    device = torch.device("cuda")
    dim, slots = 8, 3
    pairs = [(0, p) for p in range(2, 9)] + [(1, p) for p in range(5, 11)]

    def key(request: int, position: int) -> torch.Tensor:
        return (torch.arange(dim, dtype=torch.float32) + request * 1000 + position * 10).to(
            torch.bfloat16
        )

    raw = torch.stack([key(*pair) for pair in pairs]).to(device)
    token_to_req = torch.tensor([r for r, _ in pairs], dtype=torch.int32, device=device)
    positions = torch.tensor([p for _, p in pairs], dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor([0, 7, 13], dtype=torch.int32, device=device)
    ring_slots = torch.tensor([2, 0], dtype=torch.int32, device=device)
    ring = torch.zeros(slots, ring_capacity, dim, device=device, dtype=torch.bfloat16)
    for request, position, slot in ((0, 0, 2), (0, 1, 2), (1, 4, 0)):
        ring[slot, position % ring_capacity] = key(request, position).to(device)

    pooled = torch.empty(len(pairs), dim, device=device, dtype=torch.bfloat16)
    first = torch.empty(len(pairs), dtype=torch.int32, device=device)
    qsa_compress_groups(
        raw, ring, ring_slots, token_to_req, cu_seqlens, positions, RATIO, pooled, first
    )

    for row, (request, position) in enumerate(pairs):
        if (position + 1) % RATIO:
            continue
        group = torch.stack([key(request, position - RATIO + 1 + k).float() for k in range(RATIO)])
        expected = group.mean(0).to(torch.bfloat16).to(device)
        assert torch.equal(pooled[row], expected), (request, position)
        assert int(first[row]) == position - RATIO + 1

    # The ring keeps only each request's last ring_capacity rows.
    rows = torch.arange(len(pairs), device=device)
    ends = cu_seqlens.long().index_select(0, token_to_req.long() + 1)
    slot = torch.where(
        rows >= ends - ring_capacity,
        ring_slots.long().index_select(0, token_to_req.long()) * ring_capacity
        + positions.long() % ring_capacity,
        torch.full_like(rows, -1),
    )
    qsa_store_rows(ring, slot.to(torch.int32), raw)
    for request, ring_slot in ((0, 2), (1, 0)):
        for position in [p for r, p in pairs if r == request][-ring_capacity:]:
            assert torch.equal(
                ring[ring_slot, position % ring_capacity], key(request, position).to(device)
            )


# --------------------------------------------------------------------------------------
# Block top-k (kernel/triton/qsa/topk.py)
# --------------------------------------------------------------------------------------

TOPK_SHAPES = [(512, 512), (4096, 512)]


def _torch_topk_blocks(logits, visible, width):
    """The torch.topk fallback of ``qsa_sparse._top_blocks``, kept here as the reference."""
    columns = logits.shape[1]
    out = torch.full((logits.shape[0], width), -1, dtype=torch.int32, device=logits.device)
    column = torch.arange(columns, device=logits.device)
    masked = logits.masked_fill(column.unsqueeze(0) >= visible.unsqueeze(1), -float("inf"))
    take = min(width, columns)
    values, chosen = torch.topk(masked, take, dim=-1)
    out[:, :take] = torch.where(values > -float("inf"), chosen.to(torch.int32), -1)
    return out


def _topk_case(n_blocks: int, bs: int, mode: str, seed: int):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(bs, n_blocks, device=device, generator=generator)
    visible = torch.full((bs,), n_blocks, dtype=torch.int32, device=device)
    if mode == "ties":
        # Three distinct scores over every column: almost every selection sits on a tie.
        logits = torch.randint(0, 3, (bs, n_blocks), device=device, generator=generator).float()
    if mode == "ragged":
        visible = torch.randint(
            0, n_blocks + 1, (bs,), dtype=torch.int32, device=device, generator=generator
        )
    if mode == "dead":
        logits[:, ::5] = -float("inf")
    return logits, visible


@requires_cuda
@pytest.mark.parametrize("n_blocks,width", TOPK_SHAPES)
@pytest.mark.parametrize("bs", [4])
@pytest.mark.parametrize("mode", ["random", "ties", "ragged", "dead"])
def test_block_topk_matches_torch_topk(n_blocks: int, width: int, bs: int, mode: str):
    from freetoken.kernel.triton.qsa import qsa_block_topk

    logits, visible = _topk_case(n_blocks, bs, mode, seed=31 * n_blocks + 7 * width + bs)
    blocks = torch.empty(bs, width, dtype=torch.int32, device=logits.device)
    qsa_block_topk(logits, visible, blocks)
    expected = _torch_topk_blocks(logits, visible, width)

    # Selection is a set: torch.topk orders by descending score, the kernel by column id.
    torch.testing.assert_close(blocks.sort(-1).values, expected.sort(-1).values)
    live = (blocks >= 0).sum(-1)
    torch.testing.assert_close(live, (expected >= 0).sum(-1))
    # expand.py reads ranks [0, complete_blocks), so a -1 may only sit in the tail.
    ranks = torch.arange(width, device=blocks.device)
    assert torch.equal(blocks >= 0, ranks.unsqueeze(0) < live.unsqueeze(1))
    assert bool(((blocks[:, 1:] > blocks[:, :-1]) | (blocks[:, 1:] < 0)).all())


@requires_cuda
def test_block_topk_replays_in_a_cuda_graph():
    """Fixed grid, no host read: one capture serves every later sequence length."""
    from freetoken.kernel.triton.qsa import qsa_block_topk

    rows, columns, width = 4, 4096, 512
    device = torch.device("cuda")
    logits = torch.randn(rows, columns, device=device)
    visible = torch.full((rows,), columns, dtype=torch.int32, device=device)
    blocks = torch.empty(rows, width, dtype=torch.int32, device=device)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        qsa_block_topk(logits, visible, blocks)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qsa_block_topk(logits, visible, blocks)

    for lengths in ([columns] * rows, [columns // 2, 900, 37, 0], [4095, 512, 511, 4096]):
        logits.normal_()
        visible.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        blocks.fill_(0)
        graph.replay()
        expected = _torch_topk_blocks(logits, visible, width)
        torch.testing.assert_close(blocks.sort(-1).values, expected.sort(-1).values)


def test_torch_topk_env_picks_the_fallback(monkeypatch):
    from freetoken.attention.qsa_sparse import TORCH_TOPK_ENV, _resolve_block_topk

    assert _resolve_block_topk() is not None
    monkeypatch.setenv(TORCH_TOPK_ENV, "1")
    assert _resolve_block_topk() is None


# --------------------------------------------------------------------------------------
# Block top-k, split + merge path (wide buffers)
# --------------------------------------------------------------------------------------

SPLIT_CHUNK = 4096  # _split_plan's chunk for every buffer these tests use


def _policy_topk_blocks(logits, visible, width):
    """The kernel's documented order: highest score wins, lowest column breaks a tie."""
    columns = logits.shape[1]
    column = torch.arange(columns, device=logits.device)
    masked = logits.masked_fill(column.unsqueeze(0) >= visible.unsqueeze(1), -float("inf"))
    take = min(width, columns)
    order = masked.argsort(dim=-1, descending=True, stable=True)[:, :take]
    out = torch.full((logits.shape[0], width), -1, dtype=torch.int32, device=logits.device)
    out[:, :take] = torch.where(
        masked.gather(1, order) > -float("inf"), order.to(torch.int32), -1
    )
    return out


def _split_topk_case(n_blocks: int, width: int, bs: int, mode: str, seed: int):
    if mode != "boundary":
        return _topk_case(n_blocks, bs, mode, seed)
    # width - 212 columns beat the tie, so the 212 remaining winners start 100 columns below
    # a chunk boundary and run past it into a chunk that is all tie.
    logits = torch.zeros(bs, n_blocks, device="cuda")
    for row in range(bs):
        cut = SPLIT_CHUNK * (1 + row % (n_blocks // SPLIT_CHUNK - 1))
        logits[row, : width - 212] = 2.0
        logits[row, cut - 100 :] = 1.0
    return logits, torch.full((bs,), n_blocks, dtype=torch.int32, device="cuda")


@requires_cuda
@pytest.mark.parametrize("n_blocks", [65536])
@pytest.mark.parametrize("bs", [4])
@pytest.mark.parametrize("mode", ["random", "boundary", "ragged", "dead"])
def test_block_topk_split_path_matches_torch_topk(n_blocks: int, bs: int, mode: str):
    from freetoken.kernel.triton.qsa import qsa_block_topk, qsa_block_topk_scratch_width

    width = 512
    assert qsa_block_topk_scratch_width(n_blocks, width) > 0, "case must take the split path"
    logits, visible = _split_topk_case(n_blocks, width, bs, mode, seed=n_blocks + bs + len(mode))
    blocks = torch.empty(bs, width, dtype=torch.int32, device=logits.device)
    qsa_block_topk(logits, visible, blocks)

    torch.testing.assert_close(
        blocks.sort(-1).values, _torch_topk_blocks(logits, visible, width).sort(-1).values
    )
    # Tie determinism: the winners are the exact set the lowest-column-first policy names,
    # including the ties that straddle a chunk boundary.
    torch.testing.assert_close(
        blocks.sort(-1).values, _policy_topk_blocks(logits, visible, width).sort(-1).values
    )
    live = (blocks >= 0).sum(-1)
    ranks = torch.arange(width, device=blocks.device)
    assert torch.equal(blocks >= 0, ranks.unsqueeze(0) < live.unsqueeze(1))
    assert bool(((blocks[:, 1:] > blocks[:, :-1]) | (blocks[:, 1:] < 0)).all())


@requires_cuda
@pytest.mark.parametrize("preallocated", [False, True])
def test_block_topk_split_path_replays_in_a_cuda_graph(preallocated: bool):
    """The split geometry comes from the buffer width, so one capture serves every length."""
    from freetoken.kernel.triton.qsa import qsa_block_topk, qsa_block_topk_scratch_width

    rows, columns, width = 4, 65536, 512
    device = torch.device("cuda")
    scratch_width = qsa_block_topk_scratch_width(columns, width)
    assert scratch_width > 0
    logits = torch.randn(rows, columns, device=device)
    visible = torch.full((rows,), columns, dtype=torch.int32, device=device)
    blocks = torch.empty(rows, width, dtype=torch.int32, device=device)
    # The scratch never needs clearing: every split rewrites its own slots on every replay.
    scratch = (
        torch.empty(rows, scratch_width, dtype=torch.int32, device=device)
        if preallocated
        else None
    )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        qsa_block_topk(logits, visible, blocks, scratch)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qsa_block_topk(logits, visible, blocks, scratch)

    for lengths in ([columns] * rows, [4096, 40000, 0, 65535], [1, 4097, 8192, columns]):
        logits.normal_()
        visible.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        blocks.fill_(0)
        graph.replay()
        expected = _torch_topk_blocks(logits, visible, width)
        torch.testing.assert_close(blocks.sort(-1).values, expected.sort(-1).values)


@requires_cuda
def test_block_topk_split_path_cost_tracks_live_blocks():
    """A wide buffer with a short row must not pay for the splits past its visible tail."""
    from freetoken.kernel.triton.qsa import qsa_block_topk, qsa_block_topk_scratch_width

    # wide enough that the split work dwarfs the 20-launch floor even at boosted clocks
    rows, columns, width = 1, 262144, 512
    device = torch.device("cuda")
    logits = torch.randn(rows, columns, device=device)
    visible = torch.full((rows,), columns, dtype=torch.int32, device=device)
    blocks = torch.empty(rows, width, dtype=torch.int32, device=device)
    scratch = torch.empty(
        rows, qsa_block_topk_scratch_width(columns, width), dtype=torch.int32, device=device
    )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        qsa_block_topk(logits, visible, blocks, scratch)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            qsa_block_topk(logits, visible, blocks, scratch)

    def replay_us(live: int) -> float:
        visible.fill_(live)
        best = float("inf")
        for _ in range(5):
            start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            graph.replay()
            stop.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(stop) * 1000.0 / 20)
        return best

    full, short = replay_us(columns), replay_us(SPLIT_CHUNK)
    assert full > 2.0 * short, f"{columns} live {full:.1f}us vs {SPLIT_CHUNK} live {short:.1f}us"


@requires_cuda
def test_capture_graph_provisions_the_block_topk_scratch():
    from freetoken.kernel.triton.qsa import qsa_block_topk_scratch_width

    fixture = Fixture(parsed_config(), num_pages=320, max_running_req=2)
    backend = fixture.backend
    table_width = fixture.page_table.shape[1]
    columns = table_width // PAGE_SIZE * CMP_PAGE
    width = qsa_block_topk_scratch_width(columns, backend.block_topk)
    assert width > 0, "the fixture must be wide enough to reach the split path"

    backend.init_capture_graph(table_width, [2])
    static = backend._graph["topk_scratch"]
    assert static.shape[1] == width
    assert backend._scratch("topk_scratch", 2, width, dtype=torch.int32).data_ptr() == (
        static.data_ptr()
    )
