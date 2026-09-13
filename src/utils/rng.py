"""Controllable RNG for evolutionary operators (distinct from LLM GGUF seed)."""

from __future__ import annotations

import random
from types import ModuleType
from typing import Union

import numpy as np

_RUN_RNG: random.Random | None = None
_NP_GENERATOR: np.random.Generator | None = None


def init_run_rng(seed: int | None) -> random.Random | None:
    """Seed Python and NumPy RNGs for this process; return the ``Random`` instance."""
    global _RUN_RNG, _NP_GENERATOR
    if seed is None:
        _RUN_RNG = None
        _NP_GENERATOR = None
        return None
    _RUN_RNG = random.Random(seed)
    random.seed(seed)
    _NP_GENERATOR = np.random.default_rng(seed)
    np.random.seed(seed)
    return _RUN_RNG


def get_run_rng() -> Union[random.Random, ModuleType]:
    """Return process run RNG, or the module-level ``random`` wrapper."""
    return _RUN_RNG if _RUN_RNG is not None else random


def get_np_generator() -> np.random.Generator:
    """Return seeded NumPy generator or a fresh default."""
    if _NP_GENERATOR is not None:
        return _NP_GENERATOR
    return np.random.default_rng()


def choice(seq):
    return get_run_rng().choice(seq)


def sample(population, k):
    return get_run_rng().sample(population, k)


def choices(population, k=1):
    return get_run_rng().choices(population, k=k)
