"""CLI parser tests."""

from __future__ import annotations

from cli import build_parser


def test_max_total_genomes_parsed() -> None:
    parser = build_parser()
    args = parser.parse_args(["--max-total-genomes", "10"])
    assert args.max_total_genomes == 10


def test_parser_help_includes_config_flag() -> None:
    parser = build_parser()
    assert any(a.dest == "config" for a in parser._actions)


def test_theta_sim_default_matches_speciation_config() -> None:
    from speciation.config import SpeciationConfig

    parser = build_parser()
    args = parser.parse_args(["--max-total-genomes", "10"])
    assert args.theta_sim == SpeciationConfig().theta_sim
