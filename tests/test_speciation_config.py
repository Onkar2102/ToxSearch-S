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


def test_clustering_and_distance_fields_roundtrip() -> None:
    cfg = SpeciationConfig(
        clustering_method="dbscan",
        distance_method="nli+embedding",
        distance_alpha=0.55,
        dbscan_eps=0.2,
        dbscan_min_samples=3,
        theta_sim=0.25,
        theta_merge=0.2,
    )
    restored = SpeciationConfig.from_dict(cfg.to_dict())
    assert restored.clustering_method == "dbscan"
    assert restored.distance_method == "nli+embedding"
    assert restored.distance_alpha == pytest.approx(0.55)
    assert restored.dbscan_eps == pytest.approx(0.2)
    assert restored.dbscan_min_samples == 3
