# Lazy: ``launch`` (and the CUDA scheduler it wires) imports torch, which is not
# installed on macOS/Metal builds. ``ft serve-metal`` imports
# ``freetoken.server.metal_main`` via this package; importing ``launch`` eagerly
# would break that path on machines without torch.
from typing import TYPE_CHECKING

if TYPE_CHECKING:

    from .launch import launch_server

__all__ = ["launch_server"]


def __getattr__(name: str):
    if name == "launch_server":
        from .launch import launch_server

        return launch_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
