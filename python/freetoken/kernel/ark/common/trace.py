"""Optional Python dispatch census; Graph captures are distinct from replays."""

import atexit
from collections import Counter
import json
import os
from pathlib import Path
import torch

_PATH = os.environ.get("FREETOKEN_ARK_TRACE")
_COUNTS = Counter()
_TOTAL = 0


def record(op, shape, backend):
    global _TOTAL
    if _PATH:
        mode = "capture" if torch.cuda.is_current_stream_capturing() else "eager"
        item = (op, shape, backend, mode)
        first = item not in _COUNTS
        _COUNTS[item] += 1
        _TOTAL += 1
        if first or _TOTAL % 1000 == 0:
            flush()


def flush():
    if _PATH:
        path = Path(_PATH.replace("{pid}", str(os.getpid())))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "counts": [dict(operator=o, shape=s, backend=b, phase=m, calls=n) for (o, s, b, m), n in sorted(_COUNTS.items())],
                    "note": "Python eager calls and Graph captures only; replay GPU execution needs profiler evidence.",
                },
                indent=2,
            )
        )


atexit.register(flush)
