"""Controllable RNG for evolutionary operators (distinct from LLM GGUF seed)."""

from __future__ import annotations

import random
from types import ModuleType
from typing import Union

import numpy as np

_RUN_RNG: random.Random | None = None


def init_run_rng(seed: int | None) -> random.Random | None:
    """Seed Python and NumPy RNGs for this process; return the ``Random`` instance."""
    global _RUN_RNG
    if seed is None:
        _RUN_RNG = None
        return None
    _RUN_RNG = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)
    return _RUN_RNG


def get_run_rng() -> Union[random.Random, ModuleType]:
    """Return process run RNG, or the module-level ``random`` wrapper."""
    return _RUN_RNG if _RUN_RNG is not None else random
