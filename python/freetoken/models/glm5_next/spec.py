"""GLM-5.3-Flash MTP self-speculative decoding (engine-side state machine).

One request, greedy, bs==1. Round layout (steady state, all inside one scheduler
"decode step" at position p = device_len-1, input token x_p):

  carry-in : d1 = draft of x_{p+1} (from last round's refresh), h_mtp = MTP hidden
             at position p-1, main-stack KV/state committed through position p-1.
  chain    : drafts d2..dk -- MTP-only forwards at positions p..p+k-2, each feeding
             the previous MTP hidden (DeepSeek-style depth-1 chaining, approximate).
  verify   : ONE main-model forward of k+1 SINGLE-TOKEN fake reqs at positions
             p..p+k with inputs [x_p, d1..dk]. The prefill/extend path gives exact
             per-position causal attention; lm_head's per-request last-index gather
             returns all k+1 logits rows. KDA layers run a dedicated one-sequence
             branch that archives inputs + pre-states (batch.spec_stash).
  accept   : m = longest prefix with verify-argmax == draft; emit m+1 tokens
             (d1..dm + bonus). The step returns the first; the rest go to a token
             cache the next m scheduler steps pop without any forward.
  commit   : KDA state re-run over the accepted m+1 tokens from the archived
             pre-state (kernels only expose the final state). Everything else is
             append-only: rejected positions are plain overwrites next round.
  refresh  : MTP-only span forward over positions p..p+m with EXACT inputs
             (verify hiddens + emitted tokens) -- overwrites the chain's approximate
             layer-45 KV and its last logits row IS the next round's d1.

Fallbacks are free: any gate failure (bs>1, sampling, horizon missing) runs the
normal path; stale spec state is dropped by the (uid, device_len) key; junk KV
beyond the accepted point is overwritten before it can ever be attended (stores
happen before attention in mla_forward, one position at a time).

Known v1 gap: MTP layer-45 KV at PROMPT positions is never written (the prefill
runs only the main stack), so drafts attend junk there -- the measured 0.78
acceptance already includes this handicap. A prompt-time MTP pass is a later
quality upgrade, not a correctness issue (verify guarantees exactness).
"""

from __future__ import annotations

import os
import time
from collections import Counter
from typing import List

import torch

from freetoken.core import Batch


class _FakeReq:
    """Just enough of Req for prepare_metadata / build_fla_metadata / MoE prefill."""

    __slots__ = ("table_idx", "cached_len", "device_len", "linear_slot_idx",
                 "mamba_ping_pong", "mm_embeds")

    def __init__(self, table_idx: int, cached_len: int, device_len: int):
        self.table_idx = table_idx
        self.cached_len = cached_len
        self.device_len = device_len
        self.linear_slot_idx = None
        self.mamba_ping_pong = None
        self.mm_embeds = None

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len


class Glm5MtpSpec:
    def __init__(self, engine, k: int):
        self.eng = engine
        self.k = k
        self.log = os.environ.get("FREETOKEN_GLM5_SPEC_LOG", "0") == "1"
        # KL measurement mode: after each verify, roll KDA state back and REPLAY the
        # accepted prefix through the real single-token decode path, reporting
        # KL(sequential || verify) per position + argmax agreement. The replay's final
        # state then IS the committed state (spec_commit is skipped). ~3x slower;
        # measurement only. Note the replay attends verify-written KV (1-ulp GEMM
        # noise on stores), which is exactly the operative continuation question.
        self.kl_mode = os.environ.get("FREETOKEN_GLM5_SPEC_KL", "0") == "1"
        self.kl_sum = 0.0
        self.kl_max = 0.0
        self.kl_n = 0
        self.kl_disagree = 0
        # single-slot spec state, keyed by (uid, device_len)
        self.uid: int | None = None
        self.expect_dl = -1
        self.d1: torch.Tensor | None = None      # [1] draft of the next token
        self.h_mtp: torch.Tensor | None = None   # [1, D] MTP hidden at the draft position
        self.cache_gpu: torch.Tensor | None = None  # [n] int32 pending tokens
        self.cache_pos = 0
        self.cache_len = 0
        self._pending = False
        self.stats = Counter()
        self.m_hist = Counter()
        self._round_t = 0.0
        self._ph = [0.0, 0.0, 0.0, 0.0]  # chain / verify / commit / refresh (cuda-synced)
        self._kda_layers = [
            layer.self_attn
            for layer in engine.model.model.layers.op_list
            if type(layer.self_attn).__name__ == "KdaAttention"
        ]
        # ---- captured verify graph (Phase B). Lazily captured at the first eligible
        # round when the engine serves with CUDA graphs (their attn staging buffers and
        # memory pool are reused). Eager verify remains the fallback (KL mode, graphs
        # off, capture failure).
        self._vg = None           # torch.cuda.CUDAGraph once captured
        self._vg_state = None     # dict of static buffers/batch/stash
        self._vg_failed = self.kl_mode  # KL replay compares against EAGER verify

    # ------------------------------------------------------------------ fake batches
    def _fake_batch(self, table_idx: int, start: int, toks: torch.Tensor,
                    span: bool = False) -> Batch:
        """Fabricated batch at positions [start, start+n).

        ``span`` (MTP chain/refresh): ONE request of n tokens on the prefill path --
        exact causal extend semantics, and only the last logits row is gathered.
        Verify (span=False): n single-token fake reqs on the DECODE path -- per-row
        staggered kvlen gives exact causal attention, lm_head returns every row, and
        crucially MoE takes the decode fetch (union of activated experts through the
        LRU cache). The prefill MoE path STREAMS WHOLE LAYERS from host (~130GB for
        the 39 offload layers) and costs seconds per verify."""
        eng = self.eng
        n = toks.numel()
        if span:
            reqs: List[_FakeReq] = [_FakeReq(table_idx, start, start + n)]
            b = Batch(reqs=reqs, phase="prefill")
        else:
            reqs = [_FakeReq(table_idx, start + i, start + i + 1) for i in range(n)]
            b = Batch(reqs=reqs, phase="decode")
        b.padded_reqs = b.reqs
        b.input_ids = toks
        b.positions = torch.arange(start, start + n, dtype=torch.int32, device=eng.device)
        b.out_loc = eng.page_table[table_idx, start:start + n].clone()
        if not span:
            b.active_table_idx = torch.full(
                (n,), table_idx, dtype=torch.int64, device=eng.device)
            b.linear_table_idx = torch.full(
                (n,), table_idx, dtype=torch.int32, device=eng.device)
        eng.attn_backend.prepare_metadata(b)
        if not span:
            # decode _update_pools: rows share ONE table row -- make same-pool writers
            # produce identical bytes (see DSAAttnBackend._update_pools).
            b.attn_metadata.spec_shared_row = True
        return b

    def _fake_decode_batch(self, table_idx: int, pos: int, tok: torch.Tensor) -> Batch:
        """Fabricated single-token DECODE batch at ``pos`` (KL replay only)."""
        eng = self.eng
        b = Batch(reqs=[_FakeReq(table_idx, pos, pos + 1)], phase="decode")
        b.padded_reqs = b.reqs
        b.input_ids = tok
        b.positions = torch.tensor([pos], dtype=torch.int32, device=eng.device)
        b.out_loc = eng.page_table[table_idx, pos:pos + 1].clone()
        b.active_table_idx = torch.tensor([table_idx], dtype=torch.int64, device=eng.device)
        b.linear_table_idx = torch.tensor([table_idx], dtype=torch.int32, device=eng.device)
        eng.attn_backend.prepare_metadata(b)
        return b

    # ------------------------------------------------------------------ entry points
    def try_step(self, batch: Batch):
        if not batch.is_decode or batch.size != 1:
            return None
        req = batch.reqs[0]
        # Effective greediness: temperature<=0 samples argmax REGARDLESS of top_p/top_k
        # (Sampler: greedy path, and even the 1e-6-clamped path degenerates to argmax).
        # SamplingParams.is_greedy is stricter (requires top_p==1.0) and the server's
        # --sampling-defaults model merges top_p=0.95 into temp=0 requests -- that gate
        # would disable spec for every normal client call.
        sp = req.sampling_params
        if req.aborted or not (sp.temperature <= 0.0 or sp.top_k == 1) or not req.can_decode:
            return None
        keyed = req.uid == self.uid and req.device_len == self.expect_dl
        if keyed and self.cache_pos < self.cache_len:
            return self._pop(req)
        if (
            keyed
            and self.d1 is not None
            and self.cache_pos >= self.cache_len
            and req.spec_alloc_len >= req.device_len + self.k
            and req.device_len + self.k <= self.eng.max_seq_len
        ):
            return self._round(batch, req)
        self._pending = True  # bootstrap on the normal forward that follows
        return None

    def wants_eager(self, batch: Batch) -> bool:
        """Bootstrap steps must run the model EAGERLY: model.last_pre_norm under a
        graph replay points at whichever graph was captured LAST (the verify graph,
        once it exists), not this batch's buffer. One eager step per request."""
        return self._pending

    def after_main(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None:
        """Bootstrap: after a normal decode forward, draft d1 under the live batch
        ctx (probe pattern: same positions/out_loc -> layer-45 KV at position p)."""
        if not self._pending:
            return
        self._pending = False
        if not batch.is_decode:
            return
        eng, model = self.eng, self.eng.model
        req = batch.reqs[0]
        h = model.model.last_pre_norm[:1]
        t1 = next_tokens_gpu[:1].long()
        with eng.ctx.forward_batch(batch):
            logits, x = model.mtp.draft_step(
                h, model.model.embed_tokens.weight[t1], model.lm_head
            )
        self.d1 = logits.argmax(dim=-1)
        self.h_mtp = x[-1:]
        self.uid = req.uid
        self.expect_dl = req.device_len  # complete_one already advanced it
        self.cache_pos = self.cache_len = 0
        self.stats["bootstrap"] += 1

    # ------------------------------------------------------------------ internals
    def _pop(self, req):
        from freetoken.engine.engine import ForwardOutput

        tok = self.cache_gpu[self.cache_pos:self.cache_pos + 1]
        self.cache_pos += 1
        self.expect_dl += 1
        req.complete_one()
        cpu = tok.to("cpu", non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(self.eng.stream)
        self.stats["pop"] += 1
        return ForwardOutput(tok, cpu, ev)

    def _round(self, batch: Batch, req):
        from freetoken.engine.engine import ForwardOutput

        eng, model, k = self.eng, self.eng.model, self.k
        t0 = time.monotonic()
        p = req.device_len - 1
        x_p = batch.input_ids[:1].to(torch.int32)
        emb_w = model.model.embed_tokens.weight

        def _tick(i0):
            torch.cuda.synchronize()
            t = time.monotonic()
            if i0 >= 0:
                self._ph[i0] += t - _tick.last
            _tick.last = t
        _tick(-1)

        # -- draft chain: d1 carried; d2..dk via MTP at positions p..p+k-2
        drafts = [self.d1]
        h = self.h_mtp
        for j in range(2, k + 1):
            prev = drafts[-1]
            fb = self._fake_batch(req.table_idx, p + j - 2, prev.to(torch.int32), span=True)
            with eng.ctx.forward_batch(fb):
                lg, h = model.mtp.draft_step(h, emb_w[prev], model.lm_head)
            drafts.append(lg.argmax(dim=-1))

        _tick(0)
        # -- verify: one main-model forward, k+1 single-token fake reqs
        d_t = torch.cat(drafts)  # [k] int64
        toks = torch.cat([x_p, d_t.to(torch.int32)])
        if self._vg is None and not self._vg_failed:
            self._try_capture_verify(req, p, toks)
        if self._vg is not None:
            vb, vlogits, h_v = self._replay_verify(req, p, toks)
        else:
            vb = self._fake_batch(req.table_idx, p, toks)
            vb.spec_stash = {}
            with eng.ctx.forward_batch(vb):
                vlogits = model.forward()  # [k+1, V]
            h_v = model.model.last_pre_norm  # [k+1, D]
        greedy = vlogits.argmax(dim=-1)  # [k+1]

        _tick(1)
        # -- accept: longest draft prefix + bonus (one host sync for m)
        matched = torch.cumprod((greedy[:k] == d_t).int(), 0)
        m = int(matched.sum().item())
        # never emit past the output budget (scheduler finishes the req at remain 0)
        m = min(m, req.remain_len - 1)
        emitted = torch.cat([d_t[:m], greedy[m:m + 1]])  # [m+1]

        # -- KDA state commit at the accepted prefix
        if self.kl_mode:
            self._kl_replay(req, vb, vlogits, toks, emitted, p, m)
        elif m < k:
            for attn in self._kda_layers:
                attn.spec_commit(vb.spec_stash[attn.layer_id], m + 1, req.table_idx)

        _tick(2)
        # -- MTP refresh over accepted positions p..p+m: exact layer-45 KV + next d1
        rb = self._fake_batch(req.table_idx, p, emitted.to(torch.int32), span=True)
        with eng.ctx.forward_batch(rb):
            rlg, x = model.mtp.draft_step(h_v[:m + 1], emb_w[emitted], model.lm_head)
        self.d1 = rlg.argmax(dim=-1)
        self.h_mtp = x[-1:]

        _tick(3)
        # -- bookkeeping: emit first token now, cache the rest
        req.complete_one()
        self.cache_gpu = emitted[1:].to(torch.int32)
        self.cache_pos, self.cache_len = 0, m
        self.expect_dl = req.device_len

        self.stats["round"] += 1
        self.stats["accepted"] += m
        self.m_hist[m] += 1
        self._round_t += time.monotonic() - t0
        if self.log and (self.stats["round"] % 25 == 0 or self.stats["round"] == 1):
            r = self.stats["round"]
            print(
                f"[mtp-spec] rounds={r} tok/round={(self.stats['accepted'] + r) / r:.2f}"
                f" m_hist={dict(sorted(self.m_hist.items()))}"
                f" ms/round={1000 * self._round_t / r:.1f}"
                f" phases(chain/verify/commit/refresh)ms="
                f"{'/'.join(f'{1000 * x / r:.0f}' for x in self._ph)}"
                f" pops={self.stats['pop']} boots={self.stats['bootstrap']}",
                flush=True,
            )

        if self.kl_mode and self.log and self.stats["round"] % 25 == 0 and self.kl_n:
            print(
                f"[mtp-spec-kl] n={self.kl_n} kl_mean={self.kl_sum / self.kl_n:.3e}"
                f" kl_max={self.kl_max:.3e} argmax_disagree={self.kl_disagree}",
                flush=True,
            )
        out = emitted[:1].to(torch.int32)
        cpu = out.to("cpu", non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(eng.stream)
        return ForwardOutput(out, cpu, ev)


    # ------------------------------------------------------------------ verify graph
    def _try_capture_verify(self, req, p: int, toks: torch.Tensor) -> None:
        """Capture the verify forward as a standalone CUDA graph (once per boot).
        Reuses the engine graphs' attention staging buffers (init_capture_graph) and
        memory pool. On any precondition failure, marks eager-forever."""
        eng = self.eng
        runner = eng.graph_runner
        n = self.k + 1
        if not runner.graph_map or getattr(eng.attn_backend, "_rows_buf", None) is None                 or runner.max_graph_bs < n:
            self._vg_failed = True
            return
        try:
            dev = eng.device
            st: dict = {}
            st["input_ids"] = torch.zeros(n, dtype=torch.int32, device=dev)
            st["positions"] = torch.zeros(n, dtype=torch.int32, device=dev)
            st["out_loc"] = torch.zeros(n, dtype=torch.int32, device=dev)
            st["active"] = torch.zeros(n, dtype=torch.int64, device=dev)
            st["linear"] = torch.zeros(n, dtype=torch.int32, device=dev)
            b = Batch(reqs=[_FakeReq(req.table_idx, p + i, p + i + 1) for i in range(n)],
                      phase="decode")
            b.padded_reqs = b.reqs
            b.input_ids = st["input_ids"]
            b.positions = st["positions"]
            b.out_loc = st["out_loc"]
            b.active_table_idx = st["active"]
            b.linear_table_idx = st["linear"]
            eng.attn_backend.prepare_metadata(b)   # persistent md (pinned kv_len_cpu)
            b.attn_metadata.spec_shared_row = True
            b.spec_stash = {}
            st["batch"] = b
            # Capture ON THE DUMMY SLOT (GraphRunner pattern): its page-table row points
            # at the sink page and its state slot is scratch, so the warm forward and
            # the capture run have ZERO side effects on the live request. Replays then
            # stage the real request's addressing into the same buffers.
            dummy = eng.dummy_req
            self._stage_verify(st, dummy, 0, torch.zeros_like(toks))
            model = eng.model
            g = torch.cuda.CUDAGraph()
            pool = next(iter(runner.graph_map.values())).pool()
            with eng.ctx.forward_batch(b):
                warm = model.forward()             # warmup + triton compile (uncaptured)
                st["logits"] = torch.empty_like(warm)
                b.spec_stash.clear()
                with torch.cuda.graph(g, pool=pool, stream=eng.stream):
                    st["logits"].copy_(model.forward())
            st["h"] = model.model.last_pre_norm    # captured buffer, refreshed per replay
            self._vg = g
            self._vg_state = st
            print(f"[mtp-spec] verify graph captured (bs={n})", flush=True)
        except Exception as e:  # noqa: BLE001
            self._vg_failed = True
            print(f"[mtp-spec] verify graph capture FAILED ({e!r}); staying eager", flush=True)

    def _stage_verify(self, st: dict, req, p: int, toks: torch.Tensor) -> None:
        n = self.k + 1
        eng = self.eng
        st["input_ids"].copy_(toks)
        st["positions"].copy_(torch.arange(p, p + n, dtype=torch.int32, device=eng.device))
        st["out_loc"].copy_(eng.page_table[req.table_idx, p:p + n])
        st["active"].fill_(req.table_idx)
        st["linear"].fill_(req.table_idx)
        b = st["batch"]
        md = b.attn_metadata
        for i, r in enumerate(b.reqs):
            r.table_idx = req.table_idx
            r.cached_len = p + i
            r.device_len = p + i + 1
        for i in range(n):
            md.kv_len_cpu[i] = p + i + 1
        # decode staging: rows/kvlen/pool_rows into the shared static buffers the
        # captured kernels read (same call the engine's own replays use).
        eng.attn_backend.prepare_for_replay(b)

    def _replay_verify(self, req, p: int, toks: torch.Tensor):
        st = self._vg_state
        self._stage_verify(st, req, p, toks)
        self._vg.replay()
        b = st["batch"]
        return b, st["logits"], st["h"]

    def _kl_replay(self, req, vb, vlogits, toks, emitted, p, m):
        """Roll KDA back to pre-verify and replay the accepted prefix through the
        REAL decode path; measure KL(sequential || verify) and argmax agreement.
        Leaves KDA state exactly as sequential execution would (replaces commit)."""
        eng, model = self.eng, self.eng.model
        for attn in self._kda_layers:
            attn.spec_restore(vb.spec_stash[attn.layer_id], req.table_idx)
        for i in range(m + 1):
            db = self._fake_decode_batch(req.table_idx, p + i, toks[i:i + 1])
            with eng.ctx.forward_batch(db):
                slog = model.forward()[0]
            ps = torch.log_softmax(slog.float(), -1)
            pv = torch.log_softmax(vlogits[i].float(), -1)
            kl = float(torch.sum(ps.exp() * (ps - pv)).item())
            self.kl_sum += kl
            self.kl_max = max(self.kl_max, kl)
            self.kl_n += 1
            if int(ps.argmax().item()) != int(emitted[i].item()):
                self.kl_disagree += 1


__all__ = ["Glm5MtpSpec"]
