# `ft daemon` — a simple daemon mode for FreeToken

A small, durable, **torch-free** control plane that owns an `ft serve` child's lifecycle and
exposes control / logs / metrics over HTTP. The engine becomes a persistent service; anything that
speaks HTTP is a thin client. This file is the design reference.

```
client (ft ctl / curl / any HTTP client)             chat traffic → serve DIRECTLY
        │ HTTP control plane (loopback :1900)                     │
        ▼                                                         ▼
   ft daemon  ──spawn / signal / tail──▶  ft serve  (model · inference · MAY crash)
   (no torch)                              └─ /health /v1/stats  (per-serve control API)
        ▲
   systemd  Restart=always · RestartSec=1 · KillMode=process
```

## Why torch-free (the one non-negotiable)

The daemon imports **only** stdlib + `fastapi` + `uvicorn` (+ tiny `/proc` reads + optional
`pynvml`). It never imports `torch` / CUDA / `flashinfer` / `sgl_kernel`, nor anything under
`freetoken.server.*` (its `__init__` pulls torch) or `freetoken.utils.*` (its `__init__` pulls
transformers). This is what makes it un-crashable: a CUDA fault or native-extension segfault kills
the process that loaded it — and the daemon loads none of them. All risky work lives in the
isolated `ft serve` child. `tests/daemon/test_daemon_import_safety.py` enforces this with an import
sentinel.

## Run the server

```bash
ft daemon --host 127.0.0.1 --port 1900         # bare/flags = run the daemon server
# or as a service (survives logout, auto-restarts): see ft-daemon.service
```

State (single-instance lock, serve pidfile for re-adoption, per-serve logs) lives under
`--state-dir` (default `~/.freetoken/daemon`, override with `$FREETOKEN_DAEMON_DIR`).

## Control it (`ft daemon <verb>` — its own entry, distinct from `ft ctl`)

`ft daemon` with a **verb** is the client (controls a running daemon over HTTP); bare `ft daemon`
runs the server. `ft ctl` is left untouched — it targets a running *serve*, not the daemon.

```bash
ft daemon self                                 # daemon self-health
ft daemon start MODEL --port 1919 -- --moe-cache-auto   # args after -- go to ft serve
ft daemon status
ft daemon logs                                 # stream engine logs (SSE)
ft daemon health                               # proxied serve /health (camelCased)
ft daemon metrics                              # engine-only RAM(PSS)+VRAM footprint
ft daemon switch OTHER_MODEL                    # stop old + start new
ft daemon stop
# Recovery only: permit a degraded receipt if the failed engine cannot seal final totals.
ft daemon stop --force
```

Target a non-default daemon with `--url http://host:1900` (or `$FREETOKEN_DAEMON_URL`) and
`--token`/`$FREETOKEN_DAEMON_TOKEN`.

## HTTP API (camelCase JSON, loopback by default)

| Method / path | Notes |
| --- | --- |
| `GET /health` | Daemon self-health; always answers, never gated by `--token`. |
| `POST /engine/start` `{model,port,args[],apiKey?}` | Idempotent on the full `(model,port,args,apiKey)`; a differing config on the same port → `409`. |
| `POST /engine/stop` `{force?:false}` | Close admission, drain/abort, durably enqueue the final-accounting receipt, then `SIGTERM`→grace→`SIGKILL`. A prepare/outbox failure preserves the engine. |
| `POST /engine/switch` `{model,port,args[],apiKey?,force?:false}` | One serialized stop-accounting-start transaction. |
| `GET /engine/status` | `{running,pid,model,port,uptimeS,lastExitCode,…}`; outlives any single serve. |
| `GET /engine/logs?since=` | SSE, ANSI-stripped, tqdm-`\r` collapsed, ring replay, `id:<seq>`, `Last-Event-ID` resume. |
| `GET /engine/metrics` | `{ramBytes,vramBytes}` — the serve tree's own footprint only. |
| `GET /engine/health` | Proxied serve `/health` + daemon reachability. |
| `GET /engine/stats` | Proxied serve `/v1/stats`. |
| `GET /accounting/pending` | Unacknowledged durable final-accounting receipts, replayable after a Desktop/client crash. |
| `POST /accounting/ack` `{receiptId}` | Idempotently removes a receipt only after the client has durably applied it. |
| `POST /checkpoint/start\|cancel` | Supervised `ft checkpoint` (GPU-exclusive: stops the serve first). |

Set `--token` (or `$FREETOKEN_DAEMON_TOKEN`) to require an `X-FT-Token` header on everything
except `/health`.

`apiKey` is the serve's own `--api-key`. The daemon hands it to the serve as `FREETOKEN_API_KEY` in
its environment (never on its command line), sends it on its own calls to the serve, and keeps it in
the owner-only state file so a re-adopted serve stays reachable. The daemon's own
`FREETOKEN_API_KEY` is never passed down: a serve requires a key only when its start request carries
one, and an `--api-key` inside `args` is rejected with a 400 because the daemon would not know it.
`ft daemon start|switch --api-key KEY` (default `$FREETOKEN_API_KEY`) sets it.

The daemon reaches the serve's destructive `POST /v1/admin/prepare-stop` endpoint only over
loopback; the serve rejects non-loopback callers even when its inference API is bound to
`0.0.0.0`. `force` is never implicit: it is an explicit recovery choice that may record null
totals for an unobservable tail when a broken engine cannot be drained or queried.

## Self-preservation (the point of the thing)

- **Single owner:** flock pidfile → at most one daemon; on boot it **re-adopts** a still-running
  serve recorded in the pidfile (PID-reuse-safe via start-time + argv identity), so a daemon blip
  never orphans a live engine.
- **Engine outlives the daemon:** on `SIGTERM` the daemon *detaches* (leaves the serve running)
  by default; only `POST /engine/stop` kills the engine. Pair with `KillMode=process` in systemd.
- **OOM policy:** the daemon periodically raises the serve tree's `oom_score_adj`, making the
  22 GB serve — not the tiny daemon — the kernel's preferred victim.
- **Never blocks / never throws out:** spawn/kill/`/proc`/NVML/proxy calls all run off the event
  loop; handler errors become 5xx; degraded start boots even with no serve / stale pidfile.
- **Crash policy:** a serve crash is recorded (`lastExitCode`) and reported, not blindly
  restarted (blind restart loops on an OOM-from-too-big-a-model). `--auto-restart` opts in.
