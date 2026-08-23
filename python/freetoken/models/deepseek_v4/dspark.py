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

import math
import statistics
from collections import deque
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.kernel.triton.dsv4.hc import hc_pre_combine
from freetoken.utils import init_logger

from .args import DeepseekV4Args
from .layers import Linear, RMSNorm
from .parallel import div_tp

logger = init_logger(__name__)


def choose_adaptive_draft_width(
    stale_confidence: Sequence[float] | torch.Tensor,
    draft_cost_ms: float,
    verify_cost_ms: Sequence[float],
) -> int:
    """Choose the DSpark prefix that maximizes expected tokens per millisecond.

    This is Algorithm 1 / Section 5.2 specialized to FreeToken's present DSV4
    serving shape of one active request.  ``verify_cost_ms[k]`` is the measured
    target cost for anchor + ``k`` draft rows.  The drafter has already produced
    its whole block, so its measured cost is common to every candidate width.

    For one request, globally sorting cumulative survival probabilities is exactly
    the same as considering its prefixes in order: the survival product cannot
    increase as the prefix grows.  Ties retain the smaller prefix, matching
    ``argmax`` in the vLLM implementation.
    """
    confidence = [float(c) for c in stale_confidence]
    if len(verify_cost_ms) != len(confidence) + 1:
        raise ValueError(
            "adaptive DSpark needs one verify cost for every width 0..gamma; "
            f"got {len(verify_cost_ms)} costs for gamma={len(confidence)}"
        )
    if not math.isfinite(draft_cost_ms) or draft_cost_ms < 0:
        raise ValueError(f"invalid DSpark draft cost {draft_cost_ms}")

    survival = 1.0
    expected = 1.0  # target bonus token; a verify always advances at least once
    best_width = 0
    best_throughput = expected / max(
        draft_cost_ms + float(verify_cost_ms[0]), 1e-6
    )
    for width, token_confidence in enumerate(confidence, start=1):
        survival *= token_confidence
        expected += survival
        throughput = expected / max(
            draft_cost_ms + float(verify_cost_ms[width]), 1e-6
        )
        if throughput > best_throughput:
            best_width = width
            best_throughput = throughput
    return best_width


class DSparkAdaptiveVerification:
    """Two-buffer stale-confidence scheduler from DSpark Section 5.2.

    Current confidences are copied asynchronously to pinned host memory.  Capacity
    selection reads the other buffer, creating the paper's causal barrier across
    jagged CUDA-graph buckets.  The target curve is profiled at graph capture; the
    fixed-cost draft pass is priced from five real executions, like vLLM's five
    startup profiling replays.
    """

    _DRAFT_PROFILE_SAMPLES = 5

    def __init__(
        self,
        block_size: int,
        verify_curve: Sequence[tuple[int, float]],
        device: torch.device,
    ) -> None:
        expected_spans = list(range(1, block_size + 2))
        curve = sorted((int(span), float(cost)) for span, cost in verify_curve)
        if [span for span, _ in curve] != expected_spans:
            raise ValueError(
                "adaptive DSpark needs profiled graphs for every anchor+prefix span "
                f"{expected_spans}, got {[span for span, _ in curve]}"
            )
        high = 0.0
        self.verify_cost_ms: list[float] = []
        for _span, cost in curve:
            high = max(high, cost, 1e-6)
            self.verify_cost_ms.append(high)

        self.block_size = block_size
        self.device = device
        self._copy_stream = torch.cuda.Stream(device=device)
        self._stale = [
            torch.ones(block_size, dtype=torch.float32, pin_memory=True)
            for _ in range(2)
        ]
        self._events: list[torch.cuda.Event | None] = [None, None]
        self._stale_idx = 0
        self._active_uid: int | None = None
        self._draft_samples: deque[float] = deque(
            maxlen=self._DRAFT_PROFILE_SAMPLES
        )
        self._debug_left = 12

    @property
    def needs_draft_profile(self) -> bool:
        return len(self._draft_samples) < self._DRAFT_PROFILE_SAMPLES

    @property
    def draft_cost_ms(self) -> float | None:
        if self.needs_draft_profile:
            return None
        return float(statistics.median(self._draft_samples))

    def record_draft_cost(self, cost_ms: float) -> None:
        if self.needs_draft_profile and math.isfinite(cost_ms) and cost_ms >= 0:
            self._draft_samples.append(float(cost_ms))
            if not self.needs_draft_profile:
                logger.info_rank0(
                    "DSpark profiled draft cost: %.2fms (median of %d real steps)",
                    self.draft_cost_ms,
                    self._DRAFT_PROFILE_SAMPLES,
                )

    def _reset_request(self, uid: int) -> None:
        for event in self._events:
            if event is not None:
                event.synchronize()
        for slot in self._stale:
            slot.fill_(1.0)
        self._events = [None, None]
        self._stale_idx = 0
        self._active_uid = uid

    def record_and_choose(self, confidence: torch.Tensor, uid: int) -> int:
        """Publish this block's confidence and choose from the stale buffer."""
        flat = confidence.detach().float().reshape(-1)
        if flat.numel() != self.block_size:
            raise RuntimeError(
                f"DSpark confidence has {flat.numel()} rows, expected {self.block_size}"
            )
        if self._active_uid != uid:
            self._reset_request(uid)

        ready_idx = self._stale_idx ^ 1
        ready = self._events[ready_idx]
        if ready is not None:
            ready.synchronize()
        self._stale_idx, write_idx = ready_idx, self._stale_idx
        stale = self._stale[self._stale_idx]

        current_stream = torch.cuda.current_stream(self.device)
        self._copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(self._copy_stream):
            self._stale[write_idx].copy_(flat, non_blocking=True)
            event = torch.cuda.Event(blocking=True)
            event.record(self._copy_stream)
            self._events[write_idx] = event

        draft_cost = self.draft_cost_ms
        # Collect five real drafter samples before changing shape.  This is a
        # measured warmup, not a guessed initial cost.
        if draft_cost is None:
            return self.block_size
        width = choose_adaptive_draft_width(
            stale, draft_cost, self.verify_cost_ms
        )
        if self._debug_left > 0:
            self._debug_left -= 1
            logger.info_rank0(
                "DSpark adaptive verify: stale=%s draft=%.2fms width=%d/%d",
                [round(float(c), 3) for c in stale],
                draft_cost,
                width,
                self.block_size,
            )
        return width


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

    Its output feeds DSpark's hardware-aware prefix scheduler: cumulative survival
    probabilities are ranked against profiled draft/verify costs. Runs in fp32 because
    the checkpoint defines this as a calibrated scalar probability per position.
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
        self,
        input_ids: torch.Tensor,
        segments,
        positions: torch.Tensor,
        target_logits_fn,
        sampling_params: Sequence,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the paper's parallel backbone and sequential Markov sampler.

        The prepared target VERIFY span has ``1 + gamma`` rows per request: the anchor
        followed by ``gamma`` proposal slots.  The DSpark backbone takes only ``gamma``
        inputs -- anchor plus ``gamma - 1`` masks -- and produces ``gamma`` base-logit
        rows.  Equation 4 then samples those rows left to right, conditioning row ``k``
        on the token sampled at ``k - 1`` (the anchor for the first row).

        Returns the proposed tokens, the exact shaped distribution each proposal was
        drawn from, and the confidence from equation 7, all flattened request-major.
        """
        gamma = self.block_size
        draft_segments = []
        rows = []
        for i, (off, n, ti, start) in enumerate(segments):
            if n != gamma + 1:
                raise ValueError(
                    f"a DSpark verify span has {n} rows, expected gamma + 1 = "
                    f"{gamma + 1}"
                )
            rows.append(torch.arange(off, off + gamma, device=input_ids.device))
            draft_segments.append((i * gamma, gamma, ti, start))
        row_idx = torch.cat(rows)
        # The scheduler extends the CPU Req with noise placeholders, but the model's
        # flat input is gathered from the GPU token pool; those newly allocated pool
        # positions have never been initialized.  Construct the reference layout here
        # from device-resident data: one live anchor followed by gamma-1 checkpoint
        # noise tokens per request (vLLM's sample_from_anchor path).
        if self.noise_token_id < 0:
            raise ValueError("DSpark requires the checkpoint's noise token id")
        draft_ids = torch.full(
            (len(segments), gamma),
            self.noise_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        draft_ids[:, 0] = input_ids[0, row_idx.view(len(segments), gamma)[:, 0]]
        draft_ids = draft_ids.view(1, -1)
        draft_pos = positions[row_idx]
        head_hidden = self.head_hidden(
            self.embed_block(draft_ids), draft_ids, draft_segments, draft_pos
        )[0]
        # The checkpoint's two heads consume different forms of this tensor.  vLLM's
        # DSV4 DSpark model returns the PRE-norm hc_head output to the speculator:
        # base logits use norm(head_hidden), while equation 7's confidence projection
        # consumes head_hidden itself.  Normalizing before returning silently changes
        # every confidence score even though the proposal logits still look plausible.
        base_logits = target_logits_fn(self.norm(head_hidden)).view(
            len(segments), gamma, -1
        )
        return self.sample_block(
            base_logits,
            head_hidden.view(len(segments), gamma, -1),
            draft_ids.view(len(segments), gamma)[:, 0],
            sampling_params,
        )

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
        """Run the draft stack and return the pre-norm ``hc_head`` hidden.

        ``embeds`` is ``[1, T, dim]`` -- the block's token embeddings, already summed with
        the projected target hidden by the caller. Returns ``[1, T, dim]``.
        """
        h = embeds.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer.prefill_batched(h, input_ids, segments, positions)
        return self.hc_head(h)

    def sample_block(
        self,
        base_logits: torch.Tensor,
        head_hidden: torch.Tensor,
        anchor: torch.Tensor,
        sampling_params: Sequence,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Equation 4/5/7 over ``[request, gamma, ...]`` backbone outputs."""
        requests, gamma, vocab = base_logits.shape
        if len(sampling_params) != requests:
            raise ValueError(
                f"got {len(sampling_params)} sampling parameter sets for {requests} requests"
            )
        prev = anchor.long()
        proposed = torch.empty(
            (requests, gamma), dtype=torch.long, device=base_logits.device
        )
        q = torch.empty(
            (requests, gamma, vocab), dtype=torch.float32, device=base_logits.device
        )
        confidence = torch.empty(
            (requests, gamma), dtype=torch.float32, device=base_logits.device
        )
        for k in range(gamma):
            markov = self.markov_head.embed(prev)
            logits_k = base_logits[:, k].float() + self.markov_head.bias(markov).float()
            step_tokens = []
            for r, params in enumerate(sampling_params):
                q_r = sampling_probs(
                    logits_k[r : r + 1],
                    params.temperature,
                    params.top_p,
                    params.top_k,
                )
                q[r, k].copy_(q_r[0])
                step_tokens.append(
                    q_r.argmax(dim=-1)
                    if params.is_greedy
                    else torch.multinomial(q_r, 1).squeeze(-1)
                )
            next_token = torch.cat(step_tokens)
            proposed[:, k].copy_(next_token)
            confidence[:, k].copy_(
                self.confidence_head(head_hidden[:, k], markov)
            )
            prev = next_token
        return proposed.flatten(), q.flatten(0, 1), confidence.flatten()


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
    # DSpark evaluates standard speculative sampling at temperature 1.0. Dividing by
    # exactly one is numerically redundant, but it still launches an elementwise kernel
    # for every draft and target row in every cycle.
    scaled = logits.float() if temperature == 1.0 else logits.float() / temperature
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


def rejection_accept_device(
    proposed: torch.Tensor,
    q: torch.Tensor,
    p: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[int, int]:
    """Device-resident form of vLLM's standard rejection sampler.

    The acceptance ratios, prefix break, residual distribution, and recovered-token
    draw stay on the logits device.  Only the accepted length and recovered token
    cross to the host, which the scheduler needs to resize the request.  This has the
    same p/q rule as :func:`rejection_accept`; it removes the full-vocabulary D2H copy,
    not any part of the distribution-correct algorithm.
    """
    n = int(proposed.numel())
    if q.shape[0] != n or p.shape[0] < n + 1:
        raise ValueError(
            f"rejection sampler shape mismatch: proposed={n}, q={q.shape}, p={p.shape}"
        )
    if n:
        token_ids = proposed.long()
        rows = torch.arange(n, device=proposed.device)
        p_token = p[:n][rows, token_ids]
        q_token = q[:n][rows, token_ids]
        uniforms = torch.rand(
            n, device=p.device, dtype=torch.float32, generator=generator
        )
        # Keep the prefix reduction on device. Converting the first rejection to a
        # Python int here would synchronize once for the length and again for the
        # recovered token below. vLLM publishes both results together as well.
        positions = torch.arange(n, device=p.device, dtype=torch.int64)
        n_acc_device = torch.where(
            p_token <= uniforms * q_token,
            positions,
            torch.full_like(positions, n),
        ).amin()
    else:
        n_acc_device = torch.zeros((), dtype=torch.int64, device=p.device)

    if n:
        rejected_row = n_acc_device.clamp_max(n - 1)
        residual = torch.clamp(p[rejected_row] - q[rejected_row], min=0.0)
        total = residual.sum()
        # clamp_min protects the unselected division branch from producing NaNs.
        residual_dist = torch.where(
            total > 0,
            residual / total.clamp_min(torch.finfo(residual.dtype).tiny),
            p[rejected_row],
        )
        dist = torch.where(n_acc_device < n, residual_dist, p[n])
    else:
        dist = p[n]
    bonus_device = torch.multinomial(dist, 1, generator=generator).squeeze(0)
    result = torch.stack((n_acc_device, bonus_device.to(torch.int64))).to(
        "cpu", non_blocking=False
    )
    return int(result[0]), int(result[1])


def window_cols_for_block(
    start_pos: int, n: int, window: int, non_causal: bool,
    device: torch.device | None = None,
) -> torch.Tensor:
    """The window candidate columns a segment's queries may read.

    Extracted from ``Attention._prefill_segment`` so the masking rule -- the one thing
    that decides whether a block is genuinely semi-autoregressive -- can be checked
    without a GPU or a KV pool. ``-1`` marks a column no query may read.

    Causal: row i sees ``[pos_i - window + 1, pos_i]``.
    Non-causal DSpark: every row sees the trailing ``window`` CONTEXT tokens plus all
    ``n`` block tokens, matching the paper and vLLM's sparse-index construction.
    """
    end = start_pos + n
    if non_causal:
        w_lo = max(0, start_pos - window)
        width = end - w_lo
        return torch.arange(width, device=device).unsqueeze(0).expand(n, width)
    w_lo = max(0, start_pos - window + 1)
    ar = torch.arange(window, device=device)
    abs_p = start_pos + torch.arange(n, device=device).unsqueeze(1)
    cand = (abs_p - window + 1).clamp(min=w_lo) + ar
    return torch.where(cand > abs_p, -1, cand - w_lo)


__all__ = [
    "ConfidenceHead",
    "DSparkAdaptiveVerification",
    "DSparkDrafter",
    "MarkovHead",
    "accepted_prefix",
    "choose_adaptive_draft_width",
    "rejection_accept_device",
    "window_cols_for_block",
]
