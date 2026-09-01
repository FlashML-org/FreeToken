"""GLM-5.3-Flash (glm5_next): hybrid KDA+MLA/DSA attention, mHC hyper-connections,
NVFP4 experts with a VRAM/host residency split. Text-only serving.

Milestone build: config parse is live; the model / weight-loader modules land next, so
`model`-level names are imported lazily to keep `parse_config` usable on its own.
"""

from .args import Glm5NextArgs, KdaArgs, load_args
from .config import parse_config
from .model import Glm5NextForCausalLM
from .weight import (
    iter_weights,
    load_nvfp4_expert_sources,
    load_nvfp4_expert_sources_parallel,
)

__all__ = [
    "Glm5NextArgs", "KdaArgs", "load_args", "parse_config", "Glm5NextForCausalLM",
    "iter_weights", "load_nvfp4_expert_sources", "load_nvfp4_expert_sources_parallel",
]
