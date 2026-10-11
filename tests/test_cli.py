"""CLI parser tests."""

from __future__ import annotations

import pytest

from cli import build_parser
from speciation.distance import DISTANCE_METHODS


def test_max_total_genomes_parsed() -> None:
    parser = build_parser()
    args = parser.parse_args(["--max-total-genomes", "10"])
    assert args.max_total_genomes == 10


def test_parser_has_required_flags() -> None:
    parser = build_parser()
    dests = {a.dest for a in parser._actions}
    assert "max_total_genomes" in dests
    assert "theta_sim" in dests
    assert "evaluator" in dests
    assert "clustering_method" in dests
    assert "distance_method" in dests
    assert "config" not in dests


def test_theta_sim_default_matches_speciation_config() -> None:
    from speciation.config import SpeciationConfig

    parser = build_parser()
    args = parser.parse_args(["--max-total-genomes", "10"])
    assert args.theta_sim == SpeciationConfig().theta_sim


@pytest.mark.parametrize("method", ["leader_follower", "dbscan"])
def test_clustering_method_choices(method: str) -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--max-total-genomes", "10", "--clustering-method", method]
    )
    assert args.clustering_method == method


@pytest.mark.parametrize("method", list(DISTANCE_METHODS))
def test_distance_method_choices(method: str) -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--max-total-genomes", "10", "--distance-method", method]
    )
    assert args.distance_method == method


def test_invalid_distance_method_rejected() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-total-genomes", "10", "--distance-method", "cosine"])
