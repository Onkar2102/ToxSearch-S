"""Command-line entry for ToxSearch-S evolution runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from speciation.config import SpeciationConfig
from utils import get_custom_logging, get_system_utils
from utils.model_config_patch import update_model_configs as patch_model_configs
from utils.rng import init_run_rng
from utils.run_config import build_run_config, write_run_config

get_project_root, get_config_path, get_data_path, get_outputs_path, _, initialize_system, set_outputs_path = (
    get_system_utils()
)


def _load_experiment_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a YAML mapping: {path}")
    return data


def _apply_config_defaults(parser: argparse.ArgumentParser, config: dict[str, Any]) -> None:
    """Map YAML keys (snake_case) onto argparse dest names (hyphens → underscores)."""
    dest_map = {a.dest: a for a in parser._actions if a.dest != "help"}
    for key, value in config.items():
        dest = key.replace("-", "_")
        if dest in dest_map and value is not None:
            dest_map[dest].default = value


def build_parser() -> argparse.ArgumentParser:
    defaults = SpeciationConfig()
    parser = argparse.ArgumentParser(description="Evolutionary Text Generation and Safety Analysis Framework")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML experiment config (CLI flags override file values)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Not used for termination. Kept for compatibility; termination is only by --max-total-genomes.",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        choices=["google", "openai"],
        default=None,
        help="Moderation backend for this run: google (Perspective) or openai (omni-moderation)",
    )
    parser.add_argument(
        "--north-star-metric",
        type=str,
        default=None,
        help="North-star score key for fitness (valid choices depend on --evaluator)",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="omni-moderation-latest",
        help="OpenAI moderation model when --evaluator openai (default: omni-moderation-latest)",
    )
    parser.add_argument(
        "--moderation-methods",
        nargs="+",
        choices=["google", "perspective", "all", "openai", "omni"],
        default=["google"],
        help="Deprecated alias for --evaluator; --evaluator wins when both are set",
    )
    parser.add_argument(
        "--stagnation-limit",
        type=int,
        default=5,
        help="Number of generations without improvement before switching to explore mode (default: 5)",
    )
    parser.add_argument(
        "--theta-sim",
        type=float,
        default=defaults.theta_sim,
        help=f"Species similarity threshold (ensemble distance, default: {defaults.theta_sim})",
    )
    parser.add_argument(
        "--theta-merge",
        type=float,
        default=defaults.theta_merge,
        help=f"Species merge threshold (default: {defaults.theta_merge})",
    )
    parser.add_argument(
        "--min-stability-gens",
        type=int,
        default=defaults.min_stability_gens,
        help=f"Minimum species age before merge (default: {defaults.min_stability_gens})",
    )
    parser.add_argument(
        "--species-capacity",
        type=int,
        default=defaults.species_capacity,
        help=f"Maximum individuals per species (default: {defaults.species_capacity})",
    )
    parser.add_argument(
        "--cluster0-max-capacity",
        type=int,
        default=defaults.cluster0_max_capacity,
        help=f"Maximum individuals in cluster 0/reserves (default: {defaults.cluster0_max_capacity})",
    )
    parser.add_argument(
        "--cluster0-min-cluster-size",
        type=int,
        default=defaults.cluster0_min_cluster_size,
        help=f"Minimum cluster size for cluster 0 speciation (default: {defaults.cluster0_min_cluster_size})",
    )
    parser.add_argument(
        "--min-island-size",
        type=int,
        default=defaults.min_island_size,
        help=f"Minimum island size before extinction (default: {defaults.min_island_size})",
    )
    parser.add_argument(
        "--species-stagnation",
        type=int,
        default=defaults.species_stagnation,
        help=f"Generations before species extinction (default: {defaults.species_stagnation})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=defaults.embedding_model,
        help=f"Sentence-transformer model (default: {defaults.embedding_model})",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=defaults.embedding_dim,
        help=f"Embedding dimensionality (default: {defaults.embedding_dim})",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=defaults.embedding_batch_size,
        help=f"Embedding batch size (default: {defaults.embedding_batch_size})",
    )
    parser.add_argument(
        "--rg",
        type=str,
        default="models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf",
        help="Response generator model path or models/ alias",
    )
    parser.add_argument(
        "--pg",
        type=str,
        default="models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf",
        help="Prompt generator model path or models/ alias",
    )
    parser.add_argument(
        "--operators",
        type=str,
        choices=["ie", "cm", "all"],
        default="all",
        help="Operator mode: ie, cm, or all",
    )
    parser.add_argument("--max-variants", type=int, default=1, help="Variants per evolution cycle")
    parser.add_argument(
        "--seed-file",
        type=str,
        default="data/prompt.csv",
        help="CSV seed prompts (questions column)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed seed for LLM generation and Python EA randomness",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Parallel merge batch threshold (K)")
    parser.add_argument(
        "--max-total-genomes",
        type=int,
        default=None,
        help="Required termination cap (elites + reserves + archive)",
    )
    parser.add_argument("--parallel", action="store_true", help="MPI master-worker mode")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Run output directory (default timestamp under data/outputs/)",
    )
    parser.add_argument(
        "--profile",
        nargs="?",
        const="profile_main.prof",
        default=None,
        metavar="OUTPUT.prof",
        help="Enable cProfile; writes profile_main.prof under output dir",
    )
    return parser


def _update_model_configs(rg: str, pg: str, logger) -> None:
    patch_model_configs(rg, pg, logger, get_project_root=get_project_root, get_config_path=get_config_path)


def _persist_run_config(
    args: argparse.Namespace,
    *,
    evaluator_name: str,
    north_star_metric: str,
    speciation_config: SpeciationConfig,
    parallel: bool,
) -> None:
    cfg = build_run_config(
        evaluator=evaluator_name,
        north_star_metric=north_star_metric,
        max_total_genomes=args.max_total_genomes,
        seed_file=args.seed_file,
        seed=args.seed,
        operators=args.operators,
        max_variants=args.max_variants,
        stagnation_limit=args.stagnation_limit,
        rg_model=args.rg,
        pg_model=args.pg,
        openai_model=args.openai_model,
        parallel=parallel,
        output_dir=args.output_dir,
        speciation_config=speciation_config,
    )
    write_run_config(get_outputs_path(), cfg)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    pre_args, _ = parser.parse_known_args(argv)
    if pre_args.config:
        cfg_path = Path(pre_args.config)
        if not cfg_path.is_absolute():
            cfg_path = get_project_root() / cfg_path
        _apply_config_defaults(parser, _load_experiment_config(cfg_path))
    args = parser.parse_args(argv)

    if args.output_dir:
        set_outputs_path(args.output_dir)

    if args.max_total_genomes is None:
        parser.error(
            "--max-total-genomes is required (sequential and parallel): primary termination is total genomes."
        )

    init_run_rng(args.seed)

    from utils.evaluator_profiles import (
        moderation_methods_to_evaluator,
        resolve_evaluator,
        set_active_evaluator,
        set_active_north_star,
        validate_north_star,
    )

    if args.evaluator is not None and args.moderation_methods != ["google"]:
        print("Warning: --evaluator takes precedence over --moderation-methods", file=sys.stderr)
    evaluator_name = args.evaluator or moderation_methods_to_evaluator(args.moderation_methods)
    profile = resolve_evaluator(evaluator_name)
    set_active_evaluator(profile.name)
    north_star_metric = args.north_star_metric or profile.default_north_star
    try:
        north_star_metric = validate_north_star(profile, north_star_metric)
    except ValueError as exc:
        parser.error(str(exc))
    set_active_north_star(north_star_metric)

    speciation_config = SpeciationConfig(
        theta_sim=args.theta_sim,
        theta_merge=args.theta_merge,
        min_stability_gens=args.min_stability_gens,
        species_capacity=args.species_capacity,
        cluster0_max_capacity=args.cluster0_max_capacity,
        cluster0_min_cluster_size=args.cluster0_min_cluster_size,
        min_island_size=args.min_island_size,
        species_stagnation=args.species_stagnation,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        embedding_batch_size=args.embedding_batch_size,
    )

    _persist_run_config(
        args,
        evaluator_name=evaluator_name,
        north_star_metric=north_star_metric,
        speciation_config=speciation_config,
        parallel=bool(args.parallel),
    )

    prof = None
    if args.profile is not None:
        import cProfile

        prof = cProfile.Profile()

    def _dump_profile() -> None:
        if prof is not None:
            prof.disable()
            profile_path = str(get_outputs_path() / "profile_main.prof")
            prof.dump_stats(profile_path)
            print(f"Profile saved to {profile_path}")
            print(f"  Inspect with: python -m pstats {profile_path}")

    get_logger, get_log_filename, _, PerformanceLogger = get_custom_logging()

    if args.parallel:
        from mpi4py import MPI

        from parallel.master_worker import _rank_log_file
        from parallel.master_worker import run as run_parallel
        from speciation.run_speciation import run_speciation
        from utils import custom_logging as _parallel_log_state

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if args.output_dir is None:
            _parallel_outputs = None
            if rank == 0:
                _parallel_outputs = str(get_outputs_path())
            _parallel_outputs = comm.bcast(_parallel_outputs, root=0)
            set_outputs_path(_parallel_outputs)

        log_file = get_log_filename() if rank == 0 else None
        log_file = comm.bcast(log_file, root=0)
        if log_file is not None:
            _parallel_log_state._CURRENT_LOG_FILE = log_file
        per_rank_log = _rank_log_file(log_file, rank)
        logger = get_logger("master_worker", per_rank_log)
        logger.info("Starting in parallel (MPI) mode.")

        if rank == 0:
            try:
                with PerformanceLogger(logger, "Update model configs (parallel)"):
                    _update_model_configs(args.rg, args.pg, logger)
            except Exception as e:
                logger.error("Config update failed: %s", e, exc_info=True)
                return 1
        comm.Barrier()

        try:
            if prof is not None:
                prof.enable()
            run_parallel(
                logger,
                K=args.batch_size,
                seed_file=args.seed_file,
                seed=args.seed,
                operators_mode=args.operators,
                moderation_methods=args.moderation_methods,
                max_generations=args.generations,
                max_total_genomes=args.max_total_genomes,
                north_star_metric=north_star_metric,
                evaluator_backend=evaluator_name,
                openai_model=args.openai_model,
                speciation_config=speciation_config,
                log_file=log_file,
                run_speciation_fn=run_speciation,
                stagnation_limit=args.stagnation_limit,
                max_variants=args.max_variants,
            )
        finally:
            _dump_profile()
        return 0

    import main as main_module

    try:
        if prof is not None:
            prof.enable()
        main_module.main(
            max_generations=args.generations,
            moderation_methods=args.moderation_methods,
            rg_model=args.rg,
            pg_model=args.pg,
            operators=args.operators,
            max_variants=args.max_variants,
            stagnation_limit=args.stagnation_limit,
            seed_file=args.seed_file,
            max_total_genomes=args.max_total_genomes,
            seed=args.seed,
            theta_sim=args.theta_sim,
            theta_merge=args.theta_merge,
            min_stability_gens=args.min_stability_gens,
            species_capacity=args.species_capacity,
            cluster0_max_capacity=args.cluster0_max_capacity,
            cluster0_min_cluster_size=args.cluster0_min_cluster_size,
            min_island_size=args.min_island_size,
            species_stagnation=args.species_stagnation,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            embedding_batch_size=args.embedding_batch_size,
            evaluator=evaluator_name,
            north_star_metric=north_star_metric,
            openai_model=args.openai_model,
        )
        return 0
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        return 1
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    finally:
        _dump_profile()


if __name__ == "__main__":
    sys.exit(main())
