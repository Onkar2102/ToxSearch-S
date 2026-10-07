

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
from utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts

configure_matplotlib_embedded_fonts()
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import get_custom_logging, get_system_utils

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()


def _generations_chronological(tracker: Dict[str, Any]) -> List[Dict[str, Any]]:
    
    gens = tracker.get("generations") or []
    return sorted(gens, key=lambda g: int(g.get("generation_number", 0) or 0))


def _is_inc_dbscan_tracker(tracker: Dict[str, Any], outputs_path: Optional[str] = None) -> bool:
    summary = tracker.get("speciation_summary") or {}
    if summary.get("population_mode") == "inc_dbscan":
        return True
    gens = tracker.get("generations") or []
    for g in reversed(gens):
        sp = g.get("speciation") or {}
        if sp.get("population_mode") == "inc_dbscan" or sp.get("density_size"):
            return True
        if g.get("population_mode") == "inc_dbscan":
            return True
    if outputs_path:
        try:
            from speciation.clustering_mode import is_inc_dbscan_mode
            return is_inc_dbscan_mode(outputs_path=outputs_path)
        except Exception:
            pass
    return False


def _cumulative_genome_counts(generations: List[Dict[str, Any]]) -> List[int]:
    """X-axis values: cumulative genomes (total_population), not generation index."""
    xs: List[int] = []
    for i, g in enumerate(generations):
        total = g.get("total_population")
        if total is None:
            dens = g.get("density_size")
            if dens is not None:
                total = dens
            else:
                elites = int(g.get("elites_count", 0) or 0)
                archived = int(g.get("archived_count", 0) or 0)
                total = elites + archived
        xs.append(int(total or 0))
    return xs


def _max_total_genomes_xlim(tracker: Dict[str, Any], xs: List[int]) -> float:
    meta = tracker.get("run_metadata") or {}
    cap = meta.get("max_total_genomes")
    if cap is not None:
        try:
            return max(float(cap), float(max(xs) if xs else 0))
        except (TypeError, ValueError):
            pass
    return float(max(xs) if xs else 0)


def load_evolution_tracker(outputs_path: Optional[str] = None) -> Dict[str, Any]:
    
    if outputs_path is None:
        outputs_path = str(get_outputs_path())
    
    tracker_path = Path(outputs_path) / "EvolutionTracker.json"
    if not tracker_path.exists():
        return {}
    
    with open(tracker_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_fitness_evolution_plot(outputs_path: Optional[str] = None, logger=None) -> Optional[str]:
    
    _logger = logger or get_logger("LiveAnalysis")
    
    try:
        tracker = load_evolution_tracker(outputs_path)
        if not tracker or "generations" not in tracker:
            _logger.warning("No generation data found for fitness plot")
            return None
        
        generations = _generations_chronological(tracker)
        if not generations:
            return None
        
        x_genomes = _cumulative_genome_counts(generations)

        max_scores = [float(g.get("max_score_variants", g.get("best_fitness", 0.0)) or 0.0) for g in generations]
        min_scores = [float(g.get("min_score_variants", 0.0) or 0.0) for g in generations]
        avg_scores = [float(g.get("avg_fitness_generation", g.get("avg_fitness", 0.0)) or 0.0) for g in generations]
        
        cumulative_best = []
        current_max = 0.0
        for score in max_scores:
            current_max = max(current_max, float(score))
            cumulative_best.append(current_max)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_genomes, max_scores, 'o-', label='Max Fitness', linewidth=2, markersize=6, color='#1f77b4')
        plt.plot(x_genomes, min_scores, '^-', label='Min Fitness', linewidth=2, markersize=6, color='#2ca02c')
        plt.plot(x_genomes, avg_scores, 's-', label='Avg Fitness', linewidth=2, markersize=6, color='#ff7f0e')
        plt.plot(x_genomes, cumulative_best, '--', label='Cumulative Max Score', linewidth=2, color='red', alpha=0.75)
        
        plt.xlabel('Cumulative genomes', fontsize=12)
        plt.ylabel('Fitness Score', fontsize=12)
        plt.title('Fitness Evolution vs Cumulative Genomes', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(left=0, right=_max_total_genomes_xlim(tracker, x_genomes))
        plt.ylim(bottom=0)
        plt.tight_layout()
        
        if outputs_path is None:
            outputs_path = str(get_outputs_path())
        
        figures_dir = Path(outputs_path) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = figures_dir / "fitness_evolution.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        _logger.info("Generated fitness evolution plot: %s", plot_path)
        return str(plot_path)
        
    except Exception as e:
        _logger.error("Failed to generate fitness evolution plot: %s", e, exc_info=True)
        return None


def generate_speciation_plot(outputs_path: Optional[str] = None, logger=None) -> Optional[str]:
    
    _logger = logger or get_logger("LiveAnalysis")
    
    try:
        tracker = load_evolution_tracker(outputs_path)
        if not tracker or "generations" not in tracker:
            _logger.warning("No generation data found for speciation plot")
            return None
        
        generations = _generations_chronological(tracker)
        if not generations:
            return None
        
        x_genomes = _cumulative_genome_counts(generations)
        species_counts = []
        secondary_panel = []
        use_inc = _is_inc_dbscan_tracker(tracker, outputs_path)
        
        for g in generations:
            speciation = g.get("speciation") or {}
            species_counts.append(speciation.get("species_count", 0))
            if use_inc:
                secondary_panel.append(
                    int(
                        speciation.get("noise_count", g.get("noise_count", g.get("archived_count", 0)))
                        or 0
                    )
                )
            else:
                # Cumulative dead pool: archive_size (not per-gen archived_count).
                secondary_panel.append(
                    int(
                        speciation.get(
                            "archive_size",
                            g.get("archived_count", 0),
                        )
                        or 0
                    )
                )
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        xlim_right = _max_total_genomes_xlim(tracker, x_genomes)
        
        ax1.plot(x_genomes, species_counts, 'o-', color='#377eb8', linewidth=2, markersize=6)
        ax1.set_ylabel('Species Count', fontsize=12)
        ax1.set_title('Species Count vs Cumulative Genomes', fontsize=12, fontweight='bold')
        ax1.set_xlim(left=0, right=xlim_right)
        ax1.set_ylim(bottom=0)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(x_genomes, secondary_panel, 's-', color='#984ea3', linewidth=2, markersize=6)
        ax2.set_xlabel('Cumulative genomes', fontsize=12)
        if use_inc:
            ax2.set_ylabel('Noise count (in D_t)', fontsize=12)
            ax2.set_title('DBSCAN Noise vs Cumulative Genomes', fontsize=12, fontweight='bold')
        else:
            ax2.set_ylabel('Archive size (cumulative)', fontsize=12)
            ax2.set_title('Non-elites (archive) vs Cumulative Genomes', fontsize=12, fontweight='bold')
        ax2.set_xlim(left=0, right=xlim_right)
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if outputs_path is None:
            outputs_path = str(get_outputs_path())
        
        figures_dir = Path(outputs_path) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = figures_dir / "speciation_evolution.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        _logger.info("Generated speciation evolution plot: %s", plot_path)
        return str(plot_path)
        
    except Exception as e:
        _logger.error("Failed to generate speciation plot: %s", e, exc_info=True)
        return None


def generate_operator_statistics_plot(outputs_path: Optional[str] = None, logger=None) -> Optional[str]:
    
    _logger = logger or get_logger("LiveAnalysis")
    
    try:
        tracker = load_evolution_tracker(outputs_path)
        if not tracker or "generations" not in tracker:
            _logger.warning("No generation data found for operator statistics plot")
            return None
        
        generations = _generations_chronological(tracker)
        if not generations:
            return None
        
        operator_counts = defaultdict(int)
        
        for g in generations:
            op_stats = g.get("operator_statistics") or {}
            if isinstance(op_stats, dict):
                for op_name, stats in op_stats.items():
                    if isinstance(stats, dict):
                        operator_counts[op_name] += stats.get("count", 0)
                    else:
                        operator_counts[op_name] += int(stats) if isinstance(stats, (int, float)) else 0
        
        if not operator_counts:
            _logger.warning("No operator statistics found")
            return None
        
        operators = sorted(operator_counts.keys(), key=lambda o: operator_counts[o], reverse=True)
        counts = [operator_counts[op] for op in operators]
        
        x = np.arange(len(operators))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x, counts, width=0.6, color='#377eb8', label='Count')
        
        ax.set_xlabel('Operator', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Operator Usage (Cumulative)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(operators, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if outputs_path is None:
            outputs_path = str(get_outputs_path())
        
        figures_dir = Path(outputs_path) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = figures_dir / "operator_statistics.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        _logger.info("Generated operator statistics plot: %s", plot_path)
        return str(plot_path)
        
    except Exception as e:
        _logger.error("Failed to generate operator statistics plot: %s", e, exc_info=True)
        return None


def generate_population_composition_plot(outputs_path: Optional[str] = None, logger=None) -> Optional[str]:
    
    _logger = logger or get_logger("LiveAnalysis")
    
    try:
        tracker = load_evolution_tracker(outputs_path)
        if not tracker or "generations" not in tracker:
            _logger.warning("No generation data found for population composition plot")
            return None
        
        generations = _generations_chronological(tracker)
        if not generations:
            return None
        
        x_genomes = _cumulative_genome_counts(generations)
        use_inc = _is_inc_dbscan_tracker(tracker, outputs_path)
        if use_inc:
            top_counts = [
                int(g.get("clustered_count", g.get("elites_count", 0)) or 0) for g in generations
            ]
            bottom_counts = [
                int(g.get("noise_count", g.get("archived_count", 0)) or 0) for g in generations
            ]
            labels = ["Clustered (density)", "Noise (density)"]
            title = "Population composition (IncDBSCAN: clustered + noise in D_t)"
        else:
            top_counts = [int(g.get("elites_count", 0) or 0) for g in generations]
            bottom_counts = [int(g.get("archived_count", 0) or 0) for g in generations]
            labels = ["Elites (cumulative)", "Archive (cumulative)"]
            title = "Population composition (cumulative: elites + archive)"
        
        plt.figure(figsize=(10, 6))
        plt.stackplot(
            x_genomes,
            top_counts,
            bottom_counts,
            labels=labels,
            colors=["#377eb8", "#984ea3"],
            alpha=0.88,
        )
        
        plt.xlabel("Cumulative genomes", fontsize=12)
        plt.ylabel("Cumulative genome count (per pool)", fontsize=12)
        plt.title(
            title,
            fontsize=14,
            fontweight="bold",
        )
        plt.xlim(left=0, right=_max_total_genomes_xlim(tracker, x_genomes))
        plt.ylim(bottom=0)
        plt.legend(loc="upper left", fontsize=10)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        
        if outputs_path is None:
            outputs_path = str(get_outputs_path())
        
        figures_dir = Path(outputs_path) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = figures_dir / "population_composition.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        _logger.info("Generated population composition plot: %s", plot_path)
        return str(plot_path)
        
    except Exception as e:
        _logger.error("Failed to generate population composition plot: %s", e, exc_info=True)
        return None


def generate_gdp_projection_plot(outputs_path: Optional[str] = None, logger=None) -> Optional[str]:
    
    _logger = logger or get_logger("LiveAnalysis")
    if outputs_path is None:
        outputs_path = str(get_outputs_path())
    base = Path(outputs_path)
    elites_path = base / "elites.json"
    archive_path = base / "archive.json"
    temp_path = base / "temp.json"
    figures_dir = base / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        from utils.gdp_projection import (
            run_gdp_projection,
            generate_gdp_3d_toxicity_figure,
            generate_gdp_3d_generation_axis_toxicity_color,
            DEFAULT_VIEW_ANGLES,
            is_gdp_available,
            get_gdp_import_error,
        )
        if not is_gdp_available():
            err = get_gdp_import_error()
            _logger.warning(
                "GDP package not available; skipping GDP figures. Import error: %s",
                err or "unknown",
            )
            return None
        use_inc = _is_inc_dbscan_tracker(load_evolution_tracker(outputs_path), outputs_path)
        if use_inc and temp_path.exists():
            _, reduced = run_gdp_projection(
                elites_path=temp_path,
                output_dir=base,
                archive_path=None,
                reduced_size=2,
                save_json=True,
            )
        else:
            _, reduced = run_gdp_projection(
                elites_path=elites_path,
                output_dir=base,
                archive_path=archive_path if archive_path.exists() else None,
                reduced_size=2,
                save_json=True,
            )
        if reduced is None:
            _logger.debug("No genomes with embeddings for GDP projection; skipping plot")
            return None
        result_path = None
        plot_3d_gen_path = figures_dir / "genetic_distance_projection_3d_toxicity_by_generation.png"
        if generate_gdp_3d_toxicity_figure(
            reduced,
            str(plot_3d_gen_path),
            color_by="species_archive",
            publication_style=True,
            view_angles=DEFAULT_VIEW_ANGLES,
        ):
            _logger.info("Generated GDP 3D (species + archive): %s", plot_3d_gen_path)
            result_path = str(plot_3d_gen_path)
        plot_3d_gen_axis_path = figures_dir / "genetic_distance_projection_3d_generation_axis_toxicity_color.png"
        if generate_gdp_3d_generation_axis_toxicity_color(
            reduced,
            str(plot_3d_gen_axis_path),
            view_angles=DEFAULT_VIEW_ANGLES,
        ):
            _logger.info("Generated GDP 3D (Z=generation, color=toxicity): %s", plot_3d_gen_axis_path)
        return result_path
    except Exception as e:
        _logger.warning("Failed to generate GDP projection plot (non-fatal): %s", e)
    return None


def run_live_analysis(outputs_path: Optional[str] = None, logger=None) -> Dict[str, Optional[str]]:
    
    _logger = logger or get_logger("LiveAnalysis")
    
    _logger.info("Running live analysis and generating visualizations...")
    
    results = {
        "fitness_evolution": generate_fitness_evolution_plot(outputs_path, _logger),
        "speciation_evolution": generate_speciation_plot(outputs_path, _logger),
        "operator_statistics": generate_operator_statistics_plot(outputs_path, _logger),
        "population_composition": generate_population_composition_plot(outputs_path, _logger),
        "gdp_projection": generate_gdp_projection_plot(outputs_path, _logger),
    }
    
    successful = sum(1 for v in results.values() if v is not None)
    _logger.info("Live analysis complete: %d/%d visualizations generated", successful, len(results))
    
    return results


__all__ = [
    "run_live_analysis",
    "generate_fitness_evolution_plot",
    "generate_speciation_plot",
    "generate_operator_statistics_plot",
    "generate_population_composition_plot",
    "generate_gdp_projection_plot",
    "load_evolution_tracker",
]
