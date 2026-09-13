"""Tests for SpeciationConfig validation and serialization."""

from __future__ import annotations

import pytest

from speciation.config import DEFAULT_CONFIG, SpeciationConfig


def test_default_config_valid() -> None:
    cfg = SpeciationConfig()
    assert cfg.theta_sim == DEFAULT_CONFIG.theta_sim


def test_theta_merge_must_be_lte_theta_sim() -> None:
    with pytest.raises(AssertionError):
        SpeciationConfig(theta_sim=0.2, theta_merge=0.25)


def test_to_dict_from_dict_roundtrip() -> None:
    cfg = SpeciationConfig(theta_sim=0.3, species_capacity=50)
    restored = SpeciationConfig.from_dict(cfg.to_dict())
    assert restored.theta_sim == 0.3
    assert restored.species_capacity == 50
