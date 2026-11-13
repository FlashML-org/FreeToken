"""Default-off overlap of the decode miss gather with the shared expert.

``QWEN36_DECODE_SHARED_OVERLAP=1`` forks ``copy_missing`` onto a side stream
after ``ensure_experts`` and runs the shared expert on the compute stream.
The compute stream waits for the gather before the routed GEMM. Unset or
``0`` leaves the layer serial. ``QWEN36_DECODE_OVERLAP_PRIORITY=1`` gives the
side stream the highest priority; that was slower on the measured host, so
it stays off.
"""
import os

ENV = "QWEN36_DECODE_SHARED_OVERLAP"
PRIORITY_ENV = "QWEN36_DECODE_OVERLAP_PRIORITY"


def _flag(name):
    raw = os.environ.get(name, "0")
    if raw not in ("", "0", "1"):
        raise ValueError(f"{name}={raw!r}: expected unset, 0 or 1")
    return raw == "1"


def enabled():
    return _flag(ENV)


class DecodeSharedOverlap:
    def __init__(self, cuda, device):
        self.cuda = cuda
        self.device = device
        if _flag(PRIORITY_ENV):
            self.stream = cuda.Stream(device=device, priority=-1)
        else:
            self.stream = cuda.Stream(device=device)
        self.ready = cuda.Event()
        self.done = cuda.Event()
        # Create the CUDA handles before any graph capture.
        producer = cuda.current_stream(device)
        self.ready.record(producer)
        self.stream.wait_event(self.ready)
        self.done.record(self.stream)
        producer.wait_event(self.done)

    def run(self, copy_missing, shared):
        producer = self.cuda.current_stream(self.device)
        # ensure_experts has already written the slot map on this stream.
        self.ready.record(producer)
        try:
            with self.cuda.stream(self.stream):
                self.stream.wait_event(self.ready)
                try:
                    copy_missing()
                finally:
                    self.done.record(self.stream)
            return shared()
        finally:
            # Join before the routed GEMM or the next layer reuses the slots.
            producer.wait_event(self.done)
