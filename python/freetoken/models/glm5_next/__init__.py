"""Experimental GLM-5.3-Flash configuration support.

The model is intentionally not registered until its KDA and mHC forward paths are
implemented. This package establishes the engine/checkpoint contract first.
"""

from .config import Glm5NextArgs, parse_config

__all__ = ["Glm5NextArgs", "parse_config"]
