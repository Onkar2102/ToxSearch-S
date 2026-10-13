

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
    meta = tracker.get("run_metadata") or {}
    if meta.get("clustering_method") == "dbscan":
        return True
    if meta.get("population_mode") == "inc_dbscan":
        return True
    gens = tracker.get("generations") or []
    for g in reversed(gens):
        sp = g.get("speciation") or {}
        if sp.get("population_mode") == "inc_dbscan":
            return True
        if g.get("population_mode") == "inc_dbscan":
            return True
    if outputs_path:
        try:
            run_cfg = Path(outputs_path) / "run_config.json"
            if run_cfg.is_file():
                with open(run_cfg, encoding="utf-8") as f:
                    rc = json.load(f)
                clustering = (rc.get("speciation") or {}).get("clustering_method")
                if clustering == "dbscan":
                    return True
        except Exception:
            pass
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

        # Population max/avg over durable store; this-gen min from variant cohort.
        max_scores = [
            float(
                g.get("population_max_fitness")
                or g.get("best_fitness")
                or g.get("max_score_variants")
                or 0.0
            )
            for g in generations
        ]
        min_scores = [float(g.get("min_score_variants", 0.0) or 0.0) for g in generations]
        avg_scores = [float(g.get("avg_fitness_generation", g.get("avg_fitness", 0.0)) or 0.0) for g in generations]
        
        cumulative_best = []
        current_max = 0.0
        for score in max_scores:
            current_max = max(current_max, float(score))
            cumulative_best.append(current_max)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_genomes, max_scores, 'o-', label='Max Fitness (population)', linewidth=2, markersize=6, color='#1f77b4')
        plt.plot(x_genomes, min_scores, '^-', label='Min Fitness (this gen)', linewidth=2, markersize=6, color='#2ca02c')
        plt.plot(x_genomes, avg_scores, 's-', label='Avg Fitness (population)', linewidth=2, markersize=6, color='#ff7f0e')
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
                        speciation.get(
                            "singleton_species_count",
                            g.get("singleton_species_count", 0),
                        )
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
            ax2.set_ylabel('Singleton species count', fontsize=12)
            ax2.set_title(
                'Singleton Species (ex-noise) vs Cumulative Genomes',
                fontsize=12,
                fontweight='bold',
            )
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


def _load_population_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _operator_usage_from_population(outputs_path: str, tracker: Dict[str, Any]) -> Dict[str, int]:
    """Count accepted variants by ``genome['operator']`` (ground-truth usage)."""
    base = Path(outputs_path)
    counts: Dict[str, int] = defaultdict(int)
    use_inc = _is_inc_dbscan_tracker(tracker, outputs_path)
    if use_inc:
        paths = [base / "temp.json"]
    else:
        # L–F: elites + archive hold the durable population; temp is transient.
        paths = [base / "elites.json", base / "archive.json", base / "temp.json"]
    seen_ids = set()
    for path in paths:
        for genome in _load_population_list(path):
            if not genome:
                continue
            gid = genome.get("id")
            if gid is not None:
                if gid in seen_ids:
                    continue
                seen_ids.add(gid)
            op = genome.get("operator")
            if not op or op in ("initial", "seed", "None"):
                continue
            counts[str(op)] += 1
    return dict(counts)


def _operator_usage_from_tracker(generations: List[Dict[str, Any]]) -> Dict[str, int]:
    """Fallback: sum per-gen operator_statistics usage-like fields."""
    counts: Dict[str, int] = defaultdict(int)
    usage_keys = (
        "count",
        "variants_created",
        "total_variants",
        "accepted",
        "variants_integrated",
    )
    for g in generations:
        op_stats = g.get("operator_statistics") or {}
        if not isinstance(op_stats, dict):
            continue
        for op_name, stats in op_stats.items():
            if isinstance(stats, dict):
                n = 0
                for key in usage_keys:
                    if stats.get(key) is not None:
                        try:
                            n = int(stats.get(key) or 0)
                        except (TypeError, ValueError):
                            n = 0
                        break
                counts[str(op_name)] += max(0, n)
            elif isinstance(stats, (int, float)):
                counts[str(op_name)] += int(stats)
    return {k: v for k, v in counts.items() if v > 0}


def generate_operator_statistics_plot(outputs_path: Optional[str] = None, logger=None) -> Optional[str]:
    
    _logger = logger or get_logger("LiveAnalysis")
    
    try:
        if outputs_path is None:
            outputs_path = str(get_outputs_path())

        tracker = load_evolution_tracker(outputs_path)
        if not tracker or "generations" not in tracker:
            _logger.warning("No generation data found for operator statistics plot")
            return None
        
        generations = _generations_chronological(tracker)
        if not generations:
            return None

        # Prefer genome.operator counts: EvolutionTracker.operator_statistics only
        # records duplicates/rejections (no "count"), which produced empty bars.
        operator_counts = _operator_usage_from_population(outputs_path, tracker)
        if not operator_counts:
            operator_counts = _operator_usage_from_tracker(generations)
        
        if not operator_counts:
            _logger.warning("No operator usage found in population or tracker")
            return None
        
        operators = sorted(operator_counts.keys(), key=lambda o: operator_counts[o], reverse=True)
        counts = [operator_counts[op] for op in operators]
        
        x = np.arange(len(operators))
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(x, counts, width=0.6, color='#377eb8', label='Accepted variants')
        for rect, val in zip(bars, counts):
            ax.annotate(
                str(val),
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        
        ax.set_xlabel('Operator', fontsize=12)
        ax.set_ylabel('Accepted variants', fontsize=12)
        ax.set_title('Operator Usage (Cumulative)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(operators, rotation=45, ha='right')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        figures_dir = Path(outputs_path) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = figures_dir / "operator_statistics.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        _logger.info(
            "Generated operator statistics plot: %s (%d operators, %d accepted)",
            plot_path,
            len(operators),
            sum(counts),
        )
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
            # Prefer full density memory size; clustered_count is mature-only.
            top_counts = [
                int(
                    g.get("density_size")
                    or g.get("elites_count")
                    or g.get("total_population")
                    or 0
                )
                for g in generations
            ]
            bottom_counts = [0 for _ in generations]
            labels = ["Density memory D_t", ""]
            title = "Population composition (IncDBSCAN: |D_t|)"
        else:
            top_counts = [int(g.get("elites_count", 0) or 0) for g in generations]
            bottom_counts = [int(g.get("archived_count", 0) or 0) for g in generations]
            labels = ["Elites (cumulative)", "Archive (cumulative)"]
            title = "Population composition (cumulative: elites + archive)"
        
        plt.figure(figsize=(10, 6))
        if use_inc:
            plt.plot(x_genomes, top_counts, color="#377eb8", linewidth=2.0, label=labels[0])
            plt.fill_between(x_genomes, top_counts, color="#377eb8", alpha=0.25)
        else:
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
            generate_gdp_3d_generation_axis_fitness_color,
            generate_gdp_3d_plotly_generation_axis_fitness_color,
            GENERATION_AXIS_VIEW_ANGLES,
            is_gdp_available,
            get_gdp_import_error,
            metric_display_label,
            metric_file_tag,
            resolve_north_star_metric,
        )
        if not is_gdp_available():
            err = get_gdp_import_error()
            _logger.warning(
                "GDP package not available; skipping GDP figures. Import error: %s",
                err or "unknown",
            )
            return None
        tracker_for_name = load_evolution_tracker(outputs_path) or {}
        metric = resolve_north_star_metric(
            (tracker_for_name.get("run_metadata") or {}).get("north_star_metric")
            or tracker_for_name.get("north_star_metric")
        )
        metric_tag = metric_file_tag(metric)
        metric_label = metric_display_label(metric)
        use_inc = _is_inc_dbscan_tracker(tracker_for_name, outputs_path)
        if use_inc and temp_path.exists():
            payload, reduced = run_gdp_projection(
                elites_path=temp_path,
                output_dir=base,
                archive_path=None,
                reduced_size=2,
                save_json=True,
                incremental=True,
                north_star_metric=metric,
            )
        else:
            payload, reduced = run_gdp_projection(
                elites_path=elites_path,
                output_dir=base,
                archive_path=archive_path if archive_path.exists() else None,
                reduced_size=2,
                save_json=True,
                incremental=True,
                north_star_metric=metric,
            )
        if reduced is None:
            _logger.debug("No genomes with embeddings for GDP projection; skipping plot")
            return None
        if payload and payload.get("incremental"):
            _logger.info(
                "GDP incremental update: reused=%s new=%s metric=%s",
                payload.get("reused_positions"),
                payload.get("new_positions"),
                metric,
            )
        plot_3d_gen_axis_path = (
            figures_dir / f"genetic_distance_projection_3d_generation_axis_{metric_tag}_color.png"
        )
        # Remove legacy Z=fitness / species-archive GDP figure if present.
        legacy = figures_dir / f"genetic_distance_projection_3d_{metric_tag}_by_generation.png"
        if legacy.exists():
            try:
                legacy.unlink()
            except OSError:
                pass
        png_ok = generate_gdp_3d_generation_axis_fitness_color(
            reduced,
            str(plot_3d_gen_axis_path),
            view_angles=GENERATION_AXIS_VIEW_ANGLES,
            metric_label=metric_label,
            north_star_metric=metric,
        )
        if png_ok:
            _logger.info(
                "Generated GDP 3D (Z=generation, color=%s): %s",
                metric_label,
                plot_3d_gen_axis_path,
            )
        # Interactive HTML (drag to rotate) alongside the static PNG.
        plot_html_path = (
            figures_dir / f"genetic_distance_projection_3d_generation_axis_{metric_tag}_color.html"
        )
        html_ok = generate_gdp_3d_plotly_generation_axis_fitness_color(
            reduced,
            save_fpath=str(plot_html_path),
            metric_label=metric_label,
            north_star_metric=metric,
        )
        if html_ok:
            _logger.info("Generated rotatable GDP 3D (HTML): %s", plot_html_path)
        if png_ok:
            return str(plot_3d_gen_axis_path)
        if html_ok:
            return str(plot_html_path)
        return None
    except Exception as e:
        _logger.warning("Failed to generate GDP projection plot (non-fatal): %s", e)
    return None


def run_live_analysis(outputs_path: Optional[str] = None, logger=None) -> Dict[str, Optional[str]]:
    """Regenerate all per-generation figures (fitness, speciation, operators, composition, GDP)."""
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
