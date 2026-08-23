from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from gguf.quants import IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL


@lru_cache(maxsize=1)
def _cpu_iq_tables() -> dict[str, np.ndarray]:
    IQ1_S.init_grid()
    IQ2_XXS.init_grid()
    IQ2_S.init_grid()
    IQ2_XS.init_grid()
    IQ3_S.init_grid()
    IQ3_XXS.init_grid()

    # IQ1 kernels expect the packed-table indices 0/1/2, not the mapped values -1/0/1.
    iq1s_grid = (IQ1_S.grid[0, 0].astype(np.int8) + 1).copy()
    iq2xxs_grid = IQ2_XXS.grid[0, 0].astype(np.uint8).copy()
    iq2xs_grid = IQ2_XS.grid[0, 0].astype(np.uint8).copy()
    iq2s_grid = IQ2_S.grid[0, 0].astype(np.uint8).copy()
    iq3xxs_grid = IQ3_XXS.grid[0, 0].astype(np.uint8).copy()
    iq3s_grid = IQ3_S.grid[0, 0].astype(np.uint8).copy()

    return {
        "iq2xs_grid": iq2xs_grid,
        "iq2xxs_grid": iq2xxs_grid,
        "iq2s_grid": iq2s_grid,
        "iq3xxs_grid": iq3xxs_grid,
        "iq3s_grid": iq3s_grid,
        "iq1s_grid": iq1s_grid,
        "ksigns_iq2xs": np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).copy(),
        "kvalues_iq4nl": np.array(IQ4_NL.kvalues, dtype=np.int8),
    }


_DEVICE_TABLES: dict[tuple[str, int | None], dict[str, torch.Tensor]] = {}


def get_iq_table_tensors(device: torch.device) -> dict[str, torch.Tensor]:
    key = (device.type, device.index)
    if key in _DEVICE_TABLES:
        return _DEVICE_TABLES[key]

    cpu_tables = _cpu_iq_tables()
    device_tables = {
        name: torch.tensor(values, device=device) for name, values in cpu_tables.items()
    }
    _DEVICE_TABLES[key] = device_tables
    return device_tables
