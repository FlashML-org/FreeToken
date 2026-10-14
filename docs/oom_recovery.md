# CUDA OOM recovery

FreeToken normally leaves backend-worker recovery to the outer daemon. This remains the default.
Set `--oom-recovery` to opt into backend-process-group replacement up to the configured limit after a **post-ready** scheduler
`torch.cuda.OutOfMemoryError`:

```bash
ft serve --model /path/to/model --oom-recovery
```

The replacement limit, readiness wait, and bounded admission queue are configurable with
`--oom-recovery-max-restarts` (default `1`), `--oom-recovery-ready-timeout` (default `900`
seconds), and `--oom-recovery-wait-queue` (default `128`). Requests already dispatched when the
OOM occurs receive one terminal retryable error and are never replayed. New requests can wait for
the fresh readiness handshake; queue overflow, timeout, or replacement failure returns 503.

Only a typed worker receipt for `torch.cuda.OutOfMemoryError` starts this path. Other worker
failures keep the existing fail-closed process-exit behavior. Before allocation starts, every old
scheduler/tokenizer process is terminated and reaped. Each replacement has a new backend epoch;
workers drop buffered IPC work from retired epochs, including requests, aborts, and cache rebuild
traffic.

`/health` and `/v1/stats` expose `maintenance`, `backend_epoch`, `restart_attempts`, and
`restart_waiters` under their recovery information.
