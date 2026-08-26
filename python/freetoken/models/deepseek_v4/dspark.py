"""dSpark: DeepSeek-V4-Flash's semi-autoregressive draft model.

DSV4-Flash ships a drafter under ``mtp.{0..n_mtp_layers-1}`` in the checkpoint. It is
NOT a one-token-per-step MTP head: it proposes a whole BLOCK of ``dspark_block_size``
tokens per draft pass, scores them with a low-rank Markov transition head, and gates
acceptance with a confidence head. That block structure is what makes it worth having
on an offload MoE -- one pass over the experts yields five candidate tokens instead of
one, and expert traffic is what bounds decode here.

Shape of a draft pass:

1. The target hands over its hidden state at ``dspark_target_layer_ids`` (40, 41, 42),
   concatenated. ``main_norm(main_proj(...))`` projects that back to one hidden width.
2. Each draft layer derives its CONTEXT KV from that same projected tensor, through its
   own ``wkv``/``kv_norm``, and writes it at the context slots -- the drafter never runs
   the prompt itself.
3. The block's tokens run through ``n_mtp_layers`` full DSV4 blocks (MLA attention,
   routed FP4 experts, hyper-connections), then ``hc_head`` and ``norm``.
4. ``markov_head`` adds a transition bias keyed on the previously sampled token, and
   ``confidence_head`` scores how likely each position is to survive verification.

Draft layers carry no compressor and no Lightning Indexer -- the checkpoint has no such
weights for them -- so they attend over the sliding window only. Their layer ids
continue the target's (``n_layers + k``), which is what lets the expert banks, the GPU
slot cache and the KV pools address draft and target layers with one index space.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.hc import hc_pre_combine

from .args import DeepseekV4Args
from .layers import Linear, RMSNorm
from .parallel import div_tp
from .rollback import needs_rollback


class MarkovHead(nn.Module):
    """Low-rank transition bias over the vocabulary (V x r, then r x V).

    ``markov_w1[token]`` embeds the previously sampled token; ``markov_w2`` turns that
    into a per-vocabulary bias added to the draft logits. Both stay REPLICATED under
    tensor parallelism: the head runs once per draft position, so sharding it would buy
    a little memory and cost an all-reduce plus a full-vocabulary gather every position.
    """

    def __init__(self, args: DeepseekV4Args):
        super().__init__()
        self.rank = args.dspark_markov_rank
        self.markov_w1 = nn.Embedding(args.vocab_size, self.rank)
        self.markov_w1.weight.requires_grad_(False)
        self.markov_w2 = nn.Parameter(
            torch.empty(args.vocab_size, self.rank, dtype=torch.bfloat16),
            requires_grad=False,
        )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """``[B]`` token ids -> ``[B, r]`` Markov embedding."""
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        """``[B, r]`` -> ``[B, vocab]`` transition bias."""
        return F.linear(markov_embed.to(self.markov_w2.dtype), self.markov_w2)


class ConfidenceHead(nn.Module):
    """Per-position acceptance confidence, from the head hidden plus the Markov embedding.

    Its output is what makes verification ADAPTIVE: a block whose tail positions score
    low can be verified short instead of paying for tokens that will be rejected.
    Runs in fp32 -- it is one row of arithmetic per position, and the threshold it feeds
    decides how many tokens the verify step covers.
    """

    def __init__(self, args: DeepseekV4Args):
        super().__init__()
        self.proj = nn.Parameter(
            torch.empty(1, args.dim + args.dspark_markov_rank, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hidden, markov_embed], dim=-1).float()
        return torch.sigmoid(F.linear(x, self.proj).squeeze(-1))


class DSparkDrafter(nn.Module):
    """The ``mtp.*`` stack: block proposal for DeepSeek-V4-Flash.

    Built only when dSpark is both shipped by the checkpoint and enabled at runtime;
    see ``DeepseekV4Args.n_draft_layers``. Holds no KV of its own -- its layers read and
    write the same paged pools as the target, at layer ids ``n_layers + k``.
    """

    def __init__(self, args: DeepseekV4Args):
        super().__init__()
        # Imported here: model.py imports this module, so a top-level import would cycle.
        from .model import Block

        self.args = args
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.hc_eps = args.hc_eps
        self.norm_eps = args.norm_eps
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        self.target_layer_ids = tuple(args.dspark_target_layer_ids)
        self.n_layers = n_draft = args.n_draft_layers

        # The target's hidden state at three layers, concatenated, projected back to one
        # hidden width. Replicated: it is the drafter's single input, and sharding it
        # would need a collective before the first draft layer.
        self.main_proj = Linear(self.dim * len(self.target_layer_ids), self.dim, kind="fp8")
        self.main_norm = RMSNorm(self.dim, self.norm_eps)

        # Full DSV4 blocks, continuing the target's layer ids. compress_ratio=0: the
        # checkpoint ships no compressor or indexer for these layers.
        self.layers = nn.ModuleList(
            [Block(args.n_layers + k, args, compress_ratio=0) for k in range(n_draft)]
        )
        # The block is proposed as a unit, so a drafted query may see the whole block
        # rather than only what precedes it. Without this the five positions would be
        # five serial steps wearing a block's clothing.
        for layer in self.layers:
            layer.attn.non_causal = True

        self.norm = RMSNorm(self.dim, self.norm_eps)
        hc_dim = self.hc_mult * self.dim
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

        self.markov_head = MarkovHead(args)
        self.confidence_head = ConfidenceHead(args)
        # Bound by the Transformer: the drafter shares the target's embedding table and
        # output head, both vocabulary-parallel under TP, so it must reach them through
        # the target's methods rather than holding tensors of its own.
        self._embed_tokens = None

        # Sanity: the draft layers shard exactly like the target's, so any TP split that
        # the target accepts must work here too.
        div_tp(args.moe_inter_dim, "moe_inter_dim", multiple_of=128)

    def bind(self, pool, device: torch.device) -> None:
        for layer in self.layers:
            layer.attn.bind(pool, device)

    def combine_target_hidden(self, aux_hidden: torch.Tensor) -> torch.Tensor:
        """``[T, dim * len(target_layer_ids)]`` -> ``[T, dim]``.

        The drafter's whole view of the context arrives through this projection, which
        is why the target must tap exactly ``dspark_target_layer_ids`` and in that order.
        """
        expect = self.dim * len(self.target_layer_ids)
        if aux_hidden.shape[-1] != expect:
            raise ValueError(
                f"dSpark expects the target hidden at layers {self.target_layer_ids} "
                f"concatenated ({expect} wide), got {aux_hidden.shape[-1]}"
            )
        return self.main_norm(self.main_proj(aux_hidden))

    def store_context_kv(
        self, main_x: torch.Tensor, positions: torch.Tensor, window_slots: torch.Tensor
    ) -> None:
        """Give every draft layer the context's KV, without running the context.

        The drafter never sees the prompt. Each draft layer instead derives its own KV
        from ``main_x`` -- the projected target hidden -- through that layer's ``wkv``,
        ``kv_norm``, RoPE and FP8 quant, and writes it at the same window slots the
        target used. So a draft block attends over the real context while costing one
        projection per layer instead of a full prefill.

        ``main_x`` is ``[T, dim]``, ``positions`` the tokens' ABSOLUTE positions ``[T]``,
        and ``window_slots`` their global window slots ``[T]``.

        RoPE keys on the absolute position, so the frequencies must be gathered by
        ``positions`` rather than sliced from the front. A context that does not start
        at zero -- which is every context after the first block -- would otherwise be
        rotated as though it did, and the drafter would attend against phases the target
        never used. That degrades acceptance without ever raising.
        """
        from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace

        # apply_rotary_emb_decode, not apply_rotary_emb: the frequencies here are
        # PER ROW (gathered by absolute position), which is the decode variant's
        # contract. apply_rotary_emb broadcasts ONE row of frequencies across the
        # sequence and reshapes freqs_cis to [1, T, 1, D] -- with T rows of frequencies
        # it fails on the view rather than rotating anything wrongly.
        from .ops import apply_rotary_emb_decode

        if positions.shape[0] != main_x.shape[0]:
            raise ValueError(
                f"positions ({positions.shape[0]}) must cover every context token "
                f"({main_x.shape[0]})"
            )
        rd = self.args.rope_head_dim
        for layer in self.layers:
            attn = layer.attn
            freqs = attn.freqs_cis.index_select(0, positions)
            kv = attn.kv_norm(attn.wkv(main_x))
            # [T, rd] -> [T, 1, rd]: the decode rotary wants a head dim to broadcast
            # each row's frequencies over. The view aliases kv, so the in-place write
            # lands in the tensor that gets stored.
            rope_view = kv[..., -rd:].unsqueeze(1)
            apply_rotary_emb_decode(rope_view, freqs)
            act_quant_fp8_inplace(kv[..., :-rd], 64)
            attn.attn.store_window(kv, attn.layer_id, window_slots)

    def catch_up_context(
        self, aux_hidden: torch.Tensor, positions: torch.Tensor, window_slots: torch.Tensor
    ) -> None:
        """Give the draft layers KV for the positions the target just committed.

        The drafter never runs the prompt. Instead each draft layer derives KV for
        already-committed positions from the target's projected hidden, and writes it at
        those positions' window slots -- so the draft layers' sliding window fills in
        behind the target, one step at a time, for the price of one projection per layer.

        Skipped silently when the tap is unavailable, because a drafter attending to a
        stale window is merely a bad drafter, not a broken one.
        """
        if aux_hidden is None or positions.numel() == 0:
            return
        main_x = self.combine_target_hidden(aux_hidden.view(-1, aux_hidden.shape[-1]))
        self.store_context_kv(main_x, positions, window_slots)

    def propose(
        self, aux_hidden: torch.Tensor, input_ids: torch.Tensor, segments,
        positions: torch.Tensor, target_logits_fn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the draft stack over one block and propose its tokens.

        ``aux_hidden`` is the target's tap at ``dspark_target_layer_ids`` for these
        positions, ``[1, T, dim * len(target_layer_ids)]``. ``input_ids`` is what the
        block is fed -- the last committed token then noise -- and ``segments`` /
        ``positions`` are the batch's, so the draft blocks address the same paged slots
        the caller allocated.

        Returns ``(draft_logits [T, vocab], confidence [T])`` -- the LOGITS, not a token
        choice. Speculative sampling needs the draft's whole distribution q, not just
        its argmax: a sampled request accepts a proposal with probability
        min(1, p(x)/q(x)) and, on rejection, resamples from the residual p - q. Reducing
        to argmax here would throw q away and force the caller into greedy-only
        acceptance.
        """
        # The block runs on token embeddings ALONE. The projected target hidden
        # (main_x) is not added here -- it is what the draft layers derive their CONTEXT
        # KV from, over the positions the target has already committed. Adding it to the
        # block would also mismatch: aux covers the previous forward's positions, which
        # is the prompt on the first step and one token thereafter, never the block.
        embeds = self.embed_block(input_ids)
        head_hidden = self.head_hidden(embeds, input_ids, segments, positions)
        return self.logits(head_hidden[0], target_logits_fn, input_ids.view(-1))

    def embed_block(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings for the block, ``[1, T, dim]``.

        The drafter has no embedding table of its own -- it shares the target's, which is
        vocabulary-parallel under TP, so the lookup has to go through the target's method
        rather than a bare index.
        """
        return self._embed_tokens(input_ids)

    def block_input_ids(self, last_token: int, device: torch.device) -> torch.Tensor:
        """The token ids a draft block is fed: the real last token, then noise.

        dSpark is a BLOCK predictor, not an autoregressive one. It is handed the last
        committed token followed by ``block_size - 1`` copies of the checkpoint's noise
        token, and fills every position in a single pass -- which is the whole reason
        one pass yields several candidates. Feeding it anything else at those positions
        (zeros, the last token repeated) puts it off its training distribution and the
        proposals stop being accepted.
        """
        if self.noise_token_id < 0:
            raise ValueError(
                "this checkpoint declares no dspark_noise_token_id, so a draft block "
                "has nothing to place at its unknown positions"
            )
        ids = torch.full(
            (self.block_size,), self.noise_token_id, dtype=torch.long, device=device
        )
        ids[0] = last_token
        return ids

    def hc_head(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce the hyper-connection copies to one hidden state (mirrors Transformer)."""
        shape, dtype = x.size(), x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(xf, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        M = shape[0] * shape[1]
        return hc_pre_combine(
            xf.view(M, self.hc_mult, self.dim), pre.view(M, self.hc_mult), dtype
        ).view(*shape[:2], self.dim)

    def head_hidden(
        self, embeds: torch.Tensor, input_ids: torch.Tensor, segments, positions: torch.Tensor
    ) -> torch.Tensor:
        """Run the draft stack over one block and return the pre-logits hidden.

        ``embeds`` is ``[1, T, dim]`` -- the block's token embeddings, already summed with
        the projected target hidden by the caller. Returns ``[1, T, dim]``.
        """
        h = embeds.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer.prefill_batched(h, input_ids, segments, positions)
        return self.norm(self.hc_head(h))

    def logits(self, head_hidden: torch.Tensor, target_logits_fn,
               prev_token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft logits and per-position acceptance confidence.

        ``target_logits_fn`` must be the TARGET's own logits method, not its head
        weight. Under tensor parallelism that head is vocabulary-parallel: a bare
        ``F.linear`` against it returns this rank's vocabulary SLICE, while the Markov
        bias is full-vocabulary because the head is replicated. Adding the two would
        line a [B, vocab/tp] tensor up against a [B, vocab] one -- broadcasting either
        into nonsense or a shape error, depending on the TP size. Going through the
        target's method keeps the all-gather in one place.
        """
        markov = self.markov_head.embed(prev_token_ids)
        logits = target_logits_fn(head_hidden) + self.markov_head.bias(markov)
        return logits, self.confidence_head(head_hidden, markov)


def accepted_prefix(
    proposed: torch.Tensor, target_argmax: torch.Tensor
) -> tuple[int, int]:
    """How much of a proposed block survives verification, and what follows it.

    Speculative decoding is only correct if it emits exactly what the target would
    have emitted. So acceptance is a PREFIX: scan the block in order and stop at the
    first position where the target disagrees. Everything before it is what the target
    would have produced anyway; everything after it was conditioned on a token the
    target rejected, and is worthless -- even if it happens to match.

    Returns ``(n_accepted, bonus_token)``. The bonus is the target's own token at the
    first rejected position, which verification already computed: a block always
    advances by at least one token, so a fully-rejected block is not a wasted step.
    """
    n = int(proposed.numel())
    for i in range(n):
        if int(proposed[i]) != int(target_argmax[i]):
            return i, int(target_argmax[i])
    # Whole block accepted; the target's extra position supplies the next token.
    return n, int(target_argmax[n]) if target_argmax.numel() > n else -1


def sampling_probs(
    logits: torch.Tensor, temperature: float, top_p: float, top_k: int = -1
) -> torch.Tensor:
    """Turn logits into the distribution the sampler would actually draw from.

    Speculative sampling's guarantee is that the emitted token matches what the TARGET
    would have produced -- which means the target's distribution AFTER temperature and
    top-p, not the raw softmax. Both the draft q and the target p have to be shaped the
    same way, or the ratio test compares two different things and the guarantee is void.
    """
    if temperature <= 0:
        # Greedy: a point mass on the argmax. The ratio test then degenerates to an
        # equality check, which is exactly the greedy acceptance rule.
        out = torch.zeros_like(logits)
        out.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return out
    scaled = logits.float() / temperature
    if top_k and top_k > 0:
        kth = scaled.topk(min(top_k, scaled.shape[-1]), dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    if 0 < top_p < 1:
        ordered, idx = probs.sort(dim=-1, descending=True)
        cum = ordered.cumsum(dim=-1)
        # Keep the smallest prefix whose mass reaches top_p; the shift keeps the first
        # token even when it alone already exceeds the threshold.
        drop = cum - ordered > top_p
        ordered = ordered.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, idx, ordered)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def rejection_accept(
    proposed: torch.Tensor, q: torch.Tensor, p: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Speculative sampling: accept a prefix, then resample from the residual.

    This is what lets a SAMPLED request speculate without changing its distribution.
    Comparing against the target's argmax (what greedy acceptance does) would bias the
    output towards the mode -- faster, and not what the caller asked for.

    For each drafted position the proposal x is accepted with probability
    ``min(1, p(x) / q(x))``. On rejection the replacement is drawn from the normalized
    residual ``max(0, p - q)``, and the block stops there. Together those two steps make
    the emitted token exactly p-distributed -- the standard result: speculation becomes
    invisible in the output, not merely close.

    ``q`` and ``p`` are the draft and target probabilities, ``[k, vocab]`` and
    ``[k+1, vocab]``. Returns ``(n_accepted, next_token)``.
    """
    n = int(proposed.numel())
    for i in range(n):
        x = int(proposed[i])
        px, qx = float(p[i, x]), float(q[i, x])
        u = float(torch.rand((), generator=generator))
        # The test is u < p(x)/q(x), but evaluated as p(x) > u * q(x) so nothing is
        # divided by a probability. q(x) is denormal for a token the drafter would
        # essentially never pick, and the ratio there is meaningless while the product
        # is exact. (vLLM's kernel does the same thing in log space.)
        if px > u * qx:
            continue
        residual = torch.clamp(p[i] - q[i], min=0.0)
        total = float(residual.sum())
        # A degenerate residual (p entirely inside q) means there is nothing left to
        # prefer; fall back to p itself rather than dividing by zero.
        dist = residual / total if total > 0 else p[i]
        return i, int(torch.multinomial(dist, 1, generator=generator))
    # Whole block accepted: the verify's extra position supplies the next token, drawn
    # from the target's own distribution so the bonus is p-distributed too.
    return n, int(torch.multinomial(p[n], 1, generator=generator))


def draft_width(confidence: torch.Tensor, threshold: float, block_size: int) -> int:
    """How many of the block's positions are worth verifying.

    The confidence head scores each drafted position. Verification costs a full target
    pass over whatever width it covers, so carrying tail positions the drafter itself
    doubts is paid-for work that will be thrown away. Cut the block at the first
    position that scores below ``threshold``, keeping at least one -- that is the
    "adaptive" in adaptive verification.
    """
    below = (confidence < threshold).nonzero()
    if below.numel() == 0:
        return min(block_size, int(confidence.numel()))
    return max(1, int(below[0]))


class StepResult(NamedTuple):
    """What one speculative step produced."""

    tokens: list[int]      # what the request actually emits, in order
    drafted: int           # positions the drafter proposed
    verified: int          # positions verification covered (<= drafted)
    accepted: int          # positions the target agreed with
    rolled_back: bool      # whether the compressor carry had to be restored


class SpeculativeLoop:
    """One dSpark step: draft a block, verify it, keep the prefix the target agrees with.

    This class exists for the ORDER, which is where speculative decoding goes wrong in
    ways that do not crash. Three orderings matter and all three are enforced here:

    1. Snapshot the compressor carry BEFORE drafting. The draft advances it in place;
       once that has happened the state a rejection must return to no longer exists.
    2. Choose the verify width from the drafter's confidence BEFORE verifying, not
       after. Verification costs a full target pass over whatever width it covers, so
       deciding afterwards spends exactly what the confidence head exists to save.
    3. Accept a PREFIX, and take the target's own token at the first disagreement. A
       block therefore always advances by at least one token and can never stall.

    The model-facing operations are injected, so the loop's control flow is testable
    without a GPU -- and so the same logic serves both the eager and captured paths.
    """

    def __init__(self, block_size: int, confidence_threshold: float = 0.5,
                 page_size: int = 128):
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.block_size = block_size
        self.confidence_threshold = confidence_threshold
        self.page_size = page_size

    def step(
        self,
        *,
        draft: "Callable[[], tuple[torch.Tensor, torch.Tensor]]",
        verify: "Callable[[torch.Tensor], torch.Tensor]",
        positions: torch.Tensor,
        snapshot=None,
    ) -> StepResult:
        """Run one block.

        ``draft()`` returns ``(proposed_ids [k], confidence [k])``.
        ``verify(tokens [w])`` returns the target's argmax at each of the ``w`` drafted
        positions plus one more -- the token that follows a fully accepted block.
        ``positions`` are the block's absolute positions, used only to decide whether a
        rejection can strand the carry.
        ``snapshot`` is called BEFORE drafting; its result is restored on rejection.
        """
        saved = snapshot() if snapshot is not None else None

        proposed, confidence = draft()
        width = draft_width(confidence, self.confidence_threshold, self.block_size)
        proposed = proposed[:width]

        target = verify(proposed)
        n_accepted, bonus = accepted_prefix(proposed, target)

        tokens = [int(t) for t in proposed[:n_accepted]]
        if bonus >= 0:
            tokens.append(bonus)

        rolled_back = False
        if n_accepted < width and saved is not None:
            if needs_rollback(n_accepted, positions[:width], self.page_size):
                saved.restore()
                rolled_back = True
        return StepResult(tokens, self.block_size, width, n_accepted, rolled_back)


def window_cols_for_block(
    start_pos: int, n: int, window: int, non_causal: bool,
    device: torch.device | None = None,
) -> torch.Tensor:
    """The window candidate columns a segment's queries may read, as ``[n, window]``.

    Extracted from ``Attention._prefill_segment`` so the masking rule -- the one thing
    that decides whether a block is genuinely semi-autoregressive -- can be checked
    without a GPU or a KV pool. ``-1`` marks a column no query may read.

    Causal: row i sees ``[pos_i - window + 1, pos_i]``.
    Non-causal: EVERY row sees ``[last - window + 1, last]``, where ``last`` is the
    block's final position, so each drafted query sees the whole block.
    """
    end = start_pos + n
    w_lo = max(0, start_pos - window + 1)
    ar = torch.arange(window, device=device)
    if non_causal:
        last = end - 1
        lo = max(w_lo, last - window + 1)
        cand = lo + ar
        return torch.where(cand > last, -1, cand - w_lo).unsqueeze(0).expand(n, window)
    abs_p = start_pos + torch.arange(n, device=device).unsqueeze(1)
    cand = (abs_p - window + 1).clamp(min=w_lo) + ar
    return torch.where(cand > abs_p, -1, cand - w_lo)


__all__ = [
    "ConfidenceHead",
    "DSparkDrafter",
    "MarkovHead",
    "SpeculativeLoop",
    "StepResult",
    "accepted_prefix",
    "draft_width",
    "window_cols_for_block",
]
