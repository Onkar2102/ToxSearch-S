"""
live_analysis.py

Live analysis and visualization generation after each generation.
Since we're not keeping historic data, we calculate and visualize metrics live.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import get_custom_logging, get_system_utils
from utils.population_io import _extract_north_star_score

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()


def load_evolution_tracker(outputs_path: Optional[str] = None) -> Dict[str, Any]:
    """Load EvolutionTracker.json."""
    if outputs_path is None:
        outputs_path = str(get_outputs_path())
    
    tracker_path = Path(outputs_path) / "EvolutionTracker.json"
    if not tracker_path.exists():
        return {}
    
    with open(tracker_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_fitness_evolution_plot(outputs_path: Optional[str] = None, logger=None) -> Optional[str]:
    """Generate fitness evolution plot showing max, avg, min scores over generations."""
    _logger = logger or get_logger("LiveAnalysis")
    
    try:
        tracker = load_evolution_tracker(outputs_path)
        if not tracker or "generations" not in tracker:
            _logger.warning("No generation data found for fitness plot")
            return None
        
        generations = tracker["generations"]
        if not generations:
            return None
        
        gen_nums = [g.get("generation_number", 0) for g in generations]
        # best_fitness = per-gen max over elites+reserves; fall back to max_score_variants
        best_fitness_scores = [
            g.get("best_fitness", g.get("max_score_variants", 0.0)) for g in generations
        ]
        avg_scores = [g.get("avg_fitness_generation", 0.0) for g in generations]
        avg_elites = [g.get("avg_fitness_elites", 0.0) for g in generations]
        
        # Calculate cumulative best fitness (running maximum across all generations)
        cumulative_best = []
        current_max = 0.0
        for score in best_fitness_scores:
            current_max = max(current_max, score)
            cumulative_best.append(current_max)
        
        plt.figure(figsize=(10, 6))
        plt.plot(gen_nums, best_fitness_scores, 'o-', label='Best Fitness (elites+reserves)', linewidth=2, markersize=6)
        plt.plot(gen_nums, avg_scores, 's-', label='Avg Fitness (population)', linewidth=2, markersize=6)
        plt.plot(gen_nums, avg_elites, '^-', label='Avg Fitness (elites)', linewidth=2, markersize=6)
        plt.plot(gen_nums, cumulative_best, '--', label='Cumulative Best', linewidth=2, color='red', alpha=0.7)
        
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness Score', fontsize=12)
        plt.title('Fitness Evolution Over Generations', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if outputs_path is None:
            outputs_path = str(get_outputs_path())
        
        # Create figures directory
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
    """Generate speciation plot showing species count and reserves size over generations."""
    _logger = logger or get_logger("LiveAnalysis")
    
    try:
        tracker = load_evolution_tracker(outputs_path)
        if not tracker or "generations" not in tracker:
            _logger.warning("No generation data found for speciation plot")
            return None
        
        generations = tracker["generations"]
        if not generations:
            return None
        
        gen_nums = []
        species_counts = []
        reserves_counts = []
        
        for g in generations:
            gen_nums.append(g.get("generation_number", 0))
            speciation = g.get("speciation") or {}
            species_counts.append(speciation.get("species_count", 0))
            reserves_counts.append(speciation.get("reserves_size", 0))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        ax1.plot(gen_nums, species_counts, 'o-', color='#377eb8', linewidth=2, markersize=6)
        ax1.set_ylabel('Species Count', fontsize=12)
        ax1.set_title('Species Count Over Generations', fontsize=12, fontweight='bold')
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(gen_nums, reserves_counts, 's-', color='#4daf4a', linewidth=2, markersize=6)
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Reserves Size', fontsize=12)
        ax2.set_title('Reserves Size Over Generations', fontsize=12, fontweight='bold')
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if outputs_path is None:
            outputs_path = str(get_outputs_path())
        
        # Create figures directory
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
    """Generate operator statistics plot showing operator usage and rejections."""
    _logger = logger or get_logger("LiveAnalysis")
    
    try:
        tracker = load_evolution_tracker(outputs_path)
        if not tracker or "generations" not in tracker:
            _logger.warning("No generation data found for operator statistics plot")
            return None
        
        generations = tracker["generations"]
        if not generations:
            return None
        
        # Aggregate operator statistics across all generations.
        # Supports both count-only format {op: count} and legacy {op: {count, mutation, crossover}}.
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
        
        # Sort by count descending; single bar per operator
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
        
        # Create figures directory
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
    """Generate population composition plot showing elites vs reserves over generations."""
    _logger = logger or get_logger("LiveAnalysis")
    
    try:
        tracker = load_evolution_tracker(outputs_path)
        if not tracker or "generations" not in tracker:
            _logger.warning("No generation data found for population composition plot")
            return None
        
        generations = tracker["generations"]
        if not generations:
            return None
        
        gen_nums = [g.get("generation_number", 0) for g in generations]
        elites_counts = [g.get("elites_count", 0) for g in generations]
        reserves_counts = [g.get("reserves_count", 0) for g in generations]
        
        plt.figure(figsize=(10, 6))
        plt.plot(gen_nums, elites_counts, 'o-', label='Elites', linewidth=2, markersize=6, color='#377eb8')
        plt.plot(gen_nums, reserves_counts, 's-', label='Reserves', linewidth=2, markersize=6, color='#4daf4a')
        
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Population Count', fontsize=12)
        plt.title('Population Composition Over Generations', fontsize=14, fontweight='bold')
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if outputs_path is None:
            outputs_path = str(get_outputs_path())
        
        # Create figures directory
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
    """
    Generate Genetic Distance Projection (GDP) figures from generation 0 through final.
    Loads elites.json, reserves.json, and archive.json (full run). Runs MDS-GDP (cosine),
    saves gdp_projection.json, and generates only:
    - genetic_distance_projection_3d_toxicity_by_generation.png (X=MDS1, Y=MDS2, Z=toxicity; color=species/archive)
    - genetic_distance_projection_3d_generation_axis_toxicity_color.png (X=MDS1, Y=MDS2, Z=generation; color=toxicity)
    Returns path to the 3D toxicity-by-generation figure or None if GDP unavailable.
    """
    _logger = logger or get_logger("LiveAnalysis")
    if outputs_path is None:
        outputs_path = str(get_outputs_path())
    base = Path(outputs_path)
    elites_path = base / "elites.json"
    reserves_path = base / "reserves.json"
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
        _, reduced = run_gdp_projection(
            elites_path=elites_path,
            reserves_path=reserves_path,
            output_dir=base,
            archive_path=base / "archive.json",
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
    """
    Run live analysis and generate all visualizations.
    
    Args:
        outputs_path: Path to outputs directory (defaults to get_outputs_path())
        logger: Optional logger instance
        
    Returns:
        Dict with paths to generated plots (None if generation failed)
    """
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
