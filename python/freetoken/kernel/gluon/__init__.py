"""Gluon versions of Triton kernels for specific GPUs, written against the Gluon API of Triton :data:`TRITON_VERSION`.

Callers route to a Gluon kernel only when :func:`available` is true and the kernel supports the device and inputs; otherwise the Triton kernel runs.
"""

from __future__ import annotations

import functools

from freetoken.kernel.backend import is_rocm
from freetoken.kernel.utils import _env_enabled
from freetoken.utils import init_logger

logger = init_logger(__name__)

DISABLE_GLUON_ENV = "FREETOKEN_DISABLE_GLUON"
# Gluon is experimental and its API changes between Triton releases
TRITON_VERSION = (3, 8)


@functools.cache
def available() -> bool:
    """True on CUDA with Triton :data:`TRITON_VERSION`, unless FREETOKEN_DISABLE_GLUON is set; call it only for a GPU that has a Gluon kernel, as another Triton version logs a warning."""
    if _env_enabled(DISABLE_GLUON_ENV) or is_rocm():
        return False
    import triton
    from packaging.version import Version

    if Version(triton.__version__).release[:2] == TRITON_VERSION:
        return True
    logger.warning(
        "Gluon kernels need Triton %s, found %s; this GPU runs the slower Triton kernels instead (%s=1 silences this)",
        ".".join(map(str, TRITON_VERSION)), triton.__version__, DISABLE_GLUON_ENV,
    )
    return False
