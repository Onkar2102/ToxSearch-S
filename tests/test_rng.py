"""Tests for run RNG seeding."""

from __future__ import annotations

from utils import rng as run_rng


def test_init_run_rng_deterministic() -> None:
    run_rng.init_run_rng(12345)
    a = [run_rng.get_run_rng().random() for _ in range(5)]
    run_rng.init_run_rng(12345)
    b = [run_rng.get_run_rng().random() for _ in range(5)]
    assert a == b


def test_init_run_rng_none_does_not_crash() -> None:
    run_rng.init_run_rng(None)
    _ = run_rng.get_run_rng().random()
