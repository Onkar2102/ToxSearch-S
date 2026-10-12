"""
gdp_projection.py

Genetic Distance Projection (GDP) integration for ToxSearch-S.
Maps elites.json + archive.json (non-elites) to GDP's GenomeData format.
Uses cosine-MDS reduction in embedding space.
Labels and coloring use the active north-star metric (any evaluator/metric).
Per-generation updates freeze prior genome coordinates and place new genomes out-of-sample.
Does not modify core evolution or speciation.
"""

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add genetic-distance-projection-main to path so we can import gdp (without modifying core project)
_project_root = Path(__file__).resolve().parents[2]
_gdp_root = _project_root / "genetic-distance-projection-main"
if _gdp_root.exists() and str(_gdp_root) not in sys.path:
    sys.path.insert(0, str(_gdp_root))

_GDP_AVAILABLE = False
_GDP_IMPORT_ERROR: Optional[str] = None
try:
    from gdp import GenomeData, ReducedGenomeData
    _GDP_AVAILABLE = True
except ImportError as e:
    GenomeData = None
    ReducedGenomeData = None
    _GDP_IMPORT_ERROR = str(e)

# For cosine-MDS we use sklearn (already used in experiments)
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances

from .population_io import _extract_north_star_score

def _metric_display_label(metric: Optional[str]) -> str:
    """Human-readable north-star label for axes/legends (evaluator-agnostic)."""
    name = (metric or "fitness").strip()
    if not name:
        name = "fitness"
    return name.replace("_", " ").replace("/", " / ").title()


def _metric_file_tag(metric: Optional[str]) -> str:
    """Filesystem-safe token for figure filenames."""
    name = (metric or "fitness").strip() or "fitness"
    return re.sub(r"[^\w.\-]+", "_", name)


def _resolve_north_star_metric(metric: Optional[str] = None) -> str:
    if metric:
        return str(metric)
    try:
        from utils.evaluator_profiles import get_active_north_star
        return str(get_active_north_star(fallback="fitness") or "fitness")
    except Exception:
        return "fitness"


def _load_previous_gdp_positions(output_dir: Path) -> Dict[int, np.ndarray]:
    """Load prior 2D coords from gdp_projection.json for incremental placement."""
    path = Path(output_dir) / "gdp_projection.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        ids = payload.get("genome_ids") or []
        pos = payload.get("positions_2d") or []
        out: Dict[int, np.ndarray] = {}
        for gid, xy in zip(ids, pos):
            if xy is None or len(xy) < 2:
                continue
            out[int(gid)] = np.asarray(xy[:2], dtype=np.float32)
        return out
    except Exception:
        return {}


def _reduce_using_cosine_mds(
    genes_matrix: np.ndarray,
    reduced_size: int = 2,
    random_state: int = 42,
    init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reduce embedding matrix to 2D/3D using MDS with precomputed cosine distances.
    Signature matches GDP's dim_reduction_function: (genes_matrix) -> (n, reduced_size).
    """
    if genes_matrix.shape[0] < 2:
        # MDS needs at least 2 points; return zeros
        return np.zeros((genes_matrix.shape[0], reduced_size), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        distances = cosine_distances(genes_matrix)
        kwargs: Dict[str, Any] = dict(
            n_components=reduced_size,
            dissimilarity="precomputed",
            random_state=random_state,
            normalized_stress="auto",
        )
        if init is not None and init.shape == (genes_matrix.shape[0], reduced_size):
            kwargs["init"] = init
            kwargs["n_init"] = 1
        mds = MDS(**kwargs)
        return mds.fit_transform(distances).astype(np.float32)


def _reduce_using_cosine_mds_incremental(
    genes_matrix: np.ndarray,
    genome_ids: List[int],
    prev_positions: Dict[int, np.ndarray],
    reduced_size: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Incremental GDP: keep prior coords for known genome IDs; place only new IDs
    out-of-sample against fixed landmarks (classical MDS is not natively incremental).
    Falls back to full MDS when fewer than 2 prior points overlap the current set.
    """
    n = int(genes_matrix.shape[0])
    if n < 2:
        return np.zeros((n, reduced_size), dtype=np.float32)

    ids = [int(g) for g in genome_ids]
    if len(ids) != n:
        return _reduce_using_cosine_mds(genes_matrix, reduced_size=reduced_size, random_state=random_state)

    fixed: Dict[int, np.ndarray] = {}
    for i, gid in enumerate(ids):
        prev = prev_positions.get(gid)
        if prev is not None and len(prev) >= reduced_size:
            fixed[i] = np.asarray(prev[:reduced_size], dtype=np.float32)

    if len(fixed) < 2:
        init = None
        if fixed:
            init = np.zeros((n, reduced_size), dtype=np.float32)
            rng = np.random.RandomState(random_state)
            init += rng.normal(scale=0.01, size=init.shape).astype(np.float32)
            for i, pos in fixed.items():
                init[i] = pos
        return _reduce_using_cosine_mds(
            genes_matrix, reduced_size=reduced_size, random_state=random_state, init=init
        )

    if len(fixed) == n:
        out = np.zeros((n, reduced_size), dtype=np.float32)
        for i, pos in fixed.items():
            out[i] = pos
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        distances = cosine_distances(genes_matrix)

    out = np.zeros((n, reduced_size), dtype=np.float32)
    for i, pos in fixed.items():
        out[i] = pos

    fixed_indices = list(fixed.keys())
    new_indices = [i for i in range(n) if i not in fixed]

    try:
        from scipy.optimize import least_squares
    except ImportError:
        # Warm-start full MDS if scipy unavailable
        init = out.copy()
        rng = np.random.RandomState(random_state)
        for i in new_indices:
            nearest = fixed_indices[int(np.argmin(distances[i, fixed_indices]))]
            init[i] = out[nearest] + rng.normal(scale=0.01, size=reduced_size).astype(np.float32)
        return _reduce_using_cosine_mds(
            genes_matrix, reduced_size=reduced_size, random_state=random_state, init=init
        )

    for i in new_indices:
        nearest = fixed_indices[int(np.argmin(distances[i, fixed_indices]))]
        y0 = out[nearest].astype(np.float64).copy()

        def _residuals(y, i=i):
            res = []
            for j in fixed_indices:
                dij = float(distances[i, j])
                dist = float(np.linalg.norm(y - out[j])) + 1e-12
                res.append(dist - dij)
            return np.asarray(res, dtype=np.float64)

        try:
            result = least_squares(_residuals, y0, method="lm", max_nfev=200)
            out[i] = result.x.astype(np.float32)
        except Exception:
            out[i] = y0.astype(np.float32)
    return out


def _load_genomes_from_json(path: Path) -> List[Dict[str, Any]]:
    """Load a list of genome dicts from a population JSON file."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _genomes_to_genome_data(
    genomes: List[Dict[str, Any]],
    alive_ids: Optional[set] = None,
    north_star_metric: Optional[str] = None,
) -> Tuple[Optional[Any], List[int]]:
    """
    Build GDP GenomeData from a list of genome dicts (each with id, prompt_embedding, etc.).
    If alive_ids is provided, adds "alive": 1 for ids in alive_ids (elites), 0 for archive.
    Returns (GenomeData or None if GDP unavailable, list of genome ids in order).
    """
    if not genomes:
        return None, []
    if not _GDP_AVAILABLE:
        return None, [g["id"] for g in genomes]

    metric = _resolve_north_star_metric(north_star_metric)
    population = {}
    population_info = {}
    for g in genomes:
        gid = int(g["id"])
        emb = g["prompt_embedding"]
        if isinstance(emb, list):
            emb = np.asarray(emb, dtype=np.float32)
        population[gid] = {f"emb_{i}": float(v) for i, v in enumerate(emb)}
        parents_raw = g.get("parents") or []
        parent_ids = []
        for p in parents_raw:
            if isinstance(p, dict) and "id" in p:
                parent_ids.append(int(p["id"]))
            elif isinstance(p, (int, float)):
                parent_ids.append(int(p))
        # North-star fitness (0–1) for the active/requested metric (any evaluator)
        try:
            fitness = float(_extract_north_star_score(g, metric))
        except ValueError:
            fitness = float(g["fitness"]) if "fitness" in g else 0.0001
        info: Dict[str, Any] = {
            "fitness": fitness,
            "parents": parent_ids,
            "species_id": g.get("species_id") if g.get("species_id") is not None else 0,
            "generation": g.get("generation", 0),
        }
        if alive_ids is not None:
            info["alive"] = "alive" if gid in alive_ids else "archived"
        population_info[gid] = info
    genome_data = GenomeData()
    genome_data._population = population
    genome_data._population_info = population_info
    genome_data._minimizing_fitness = False
    return genome_data, [g["id"] for g in genomes]


def build_genome_data_from_elites_archive(
    elites_path: Path,
    archive_path: Optional[Path] = None,
    north_star_metric: Optional[str] = None,
) -> Tuple[Optional[Any], List[Dict], List[int]]:
    """
    Build GDP GenomeData from elites.json and optionally archive.json (non-elites).

    For Incremental DBSCAN, pass ``temp.json`` as ``elites_path`` (density memory D_t)
    and omit ``archive_path``. Mature clusters (species_id/density_label > 0) and
    the singleton pool (0) are treated as "alive".
    """
    elites = _load_genomes_from_json(Path(elites_path))
    archive = _load_genomes_from_json(Path(archive_path)) if archive_path else []
    # Prefer elites over archive on id collision
    all_raw = list(elites) + list(archive)
    seen: Dict[int, Dict[str, Any]] = {}
    for g in all_raw:
        gid = g.get("id")
        if gid is None or g.get("prompt_embedding") is None:
            continue
        gid = int(gid)
        if gid not in seen:
            seen[gid] = g
    genomes = list(seen.values())
    if not genomes:
        return None, [], []

    # All elites.json rows are alive (L–F elites and IncDBSCAN D_t including singleton species).
    alive_ids = {g.get("id") for g in elites if g.get("id") is not None}
    genome_data, genome_ids = _genomes_to_genome_data(
        genomes, alive_ids=alive_ids, north_star_metric=north_star_metric
    )
    return genome_data, genomes, genome_ids


# Backward-compatible alias

def run_gdp_projection(
    elites_path: Path,
    output_dir: Path,
    archive_path: Optional[Path] = None,
    reduced_size: int = 2,
    save_json: bool = True,
    random_state: int = 42,
    reserves_path: Optional[Path] = None,  # legacy alias for archive_path
    incremental: bool = True,
    north_star_metric: Optional[str] = None,
) -> Tuple[Optional[Dict], Optional[Any]]:
    """Run GDP projection over elites.json + archive.json (non-elites).

    When ``incremental=True`` (default), reuse coordinates from an existing
    ``gdp_projection.json`` for known genome IDs and place only new genomes
    out-of-sample so the plot stays stable across generations.
    """
    if archive_path is None:
        archive_path = reserves_path
    output_dir = Path(output_dir)
    metric = _resolve_north_star_metric(north_star_metric)
    genome_data, genomes, genome_ids = build_genome_data_from_elites_archive(
        elites_path, archive_path=archive_path, north_star_metric=metric
    )
    if not genomes:
        return None, None
    if not _GDP_AVAILABLE or genome_data is None:
        return None, None
    prev_positions = _load_previous_gdp_positions(output_dir) if incremental else {}
    ids_int = [int(g) for g in genome_ids]

    def _cosine_mds(genes_matrix: np.ndarray) -> np.ndarray:
        if incremental and prev_positions:
            return _reduce_using_cosine_mds_incremental(
                genes_matrix,
                genome_ids=ids_int,
                prev_positions=prev_positions,
                reduced_size=reduced_size,
                random_state=random_state,
            )
        return _reduce_using_cosine_mds(
            genes_matrix, reduced_size=reduced_size, random_state=random_state
        )

    reduced = ReducedGenomeData.perform_reduction(source=genome_data, dim_reduction_function=_cosine_mds)
    positions_2d = [reduced.reduced_positions[gid].tolist() for gid in genome_ids]

    n_reused = sum(1 for gid in ids_int if gid in prev_positions)
    payload = {
        "genome_ids": genome_ids,
        "positions_2d": positions_2d,
        "method": "cosine_mds_incremental" if (incremental and n_reused >= 2) else "cosine_mds",
        "north_star_metric": metric,
        "incremental": bool(incremental and n_reused >= 2),
        "reused_positions": n_reused,
        "new_positions": max(0, len(ids_int) - n_reused),
    }
    if reduced_size == 3:
        payload["positions_3d"] = positions_2d

    if save_json:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "gdp_projection.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return payload, reduced


ELEV_DEFAULT = 22.0
# Three orbital views ~120° apart; avoid exact 270° (reads as a flat side-on squash).
DEFAULT_VIEW_ANGLES: List[Tuple[float, float]] = [
    (ELEV_DEFAULT, 35.0),
    (ELEV_DEFAULT, 155.0),
    (ELEV_DEFAULT, 255.0),
]
GENERATION_AXIS_VIEW_ANGLES: List[Tuple[float, float]] = list(DEFAULT_VIEW_ANGLES)


def generate_gdp_3d_generation_axis_fitness_color(
    reduced_genome_data: Any,
    save_fpath: str,
    view_angles: Optional[List[Tuple[float, float]]] = None,
    metric_label: Optional[str] = None,
    north_star_metric: Optional[str] = None,
) -> bool:
    """
    Publication-style 3D GDP: X/Y = MDS, Z = generation; color = north-star fitness.

    Three orbital viewpoints with a compact layout and an external colorbar
    immediately to the right of panel (c).
    """
    if reduced_genome_data is None:
        return False
    color_label = metric_label or _metric_display_label(_resolve_north_star_metric(north_star_metric))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.ticker import FormatStrFormatter, MaxNLocator

        positions = reduced_genome_data.reduced_positions
        population_info = reduced_genome_data._population_info
        if not positions or not population_info:
            return False

        ids = list(positions.keys())
        xy = np.array([positions[i] for i in ids], dtype=float)
        x = xy[:, 0]
        y = xy[:, 1]
        z_gen = np.array([int(population_info.get(i, {}).get("generation", 0)) for i in ids], dtype=float)
        fitness = np.array([float(population_info.get(i, {}).get("fitness", 0.0)) for i in ids], dtype=float)

        # Draw low-fitness first so high-fitness points remain visible when overlapping.
        order = np.argsort(fitness)
        x, y, z_gen, fitness = x[order], y[order], z_gen[order], fitness[order]

        angles = list(view_angles) if view_angles is not None else list(GENERATION_AXIS_VIEW_ANGLES)
        n_views = max(1, len(angles))
        n_pts = len(x)

        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.9,
        })

        fig_w = 3.55 * n_views + 1.0
        fig_h = 4.0
        fig, axes_arr = plt.subplots(
            1,
            n_views,
            figsize=(fig_w, fig_h),
            subplot_kw={"projection": "3d"},
            facecolor="white",
        )
        axes = list(axes_arr) if n_views > 1 else [axes_arr]
        # Leave headroom for title and right margin for colorbar; avoid bbox_inches=tight
        # which often collapses 3D label pads into overlapping text.
        fig.subplots_adjust(left=0.03, right=0.86, top=0.84, bottom=0.06, wspace=0.12)

        pad = 0.10

        def _lims(arr: np.ndarray) -> Tuple[float, float]:
            lo, hi = float(np.min(arr)), float(np.max(arr))
            if abs(hi - lo) < 1e-9:
                return lo - 0.5, hi + 0.5
            span = hi - lo
            return lo - pad * span, hi + pad * span

        xlim, ylim, zlim = _lims(x), _lims(y), _lims(z_gen)
        z_floor = zlim[0]
        vmin = float(np.min(fitness))
        vmax = float(max(np.max(fitness), vmin + 1e-6))
        if vmax > vmin:
            vmax = vmax + 0.05 * (vmax - vmin)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        panel_letters = "abcdefghijklmnopqrstuvwxyz"
        marker_size = 28 if n_pts <= 80 else (18 if n_pts <= 250 else 12)
        draw_stems = n_pts <= 100

        for i, ((elev, azim), ax) in enumerate(zip(angles, axes)):
            ax.set_facecolor("white")

            if draw_stems:
                for xi, yi, zi in zip(x, y, z_gen):
                    ax.plot(
                        [xi, xi], [yi, yi], [z_floor, zi],
                        color="0.75",
                        linewidth=0.4,
                        alpha=0.25,
                        zorder=1,
                    )

            ax.scatter(
                x, y, z_gen,
                c=fitness,
                cmap=cmap,
                norm=norm,
                s=marker_size,
                alpha=0.96,
                depthshade=False,
                edgecolors="0.15",
                linewidths=0.3,
                zorder=3,
            )

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(z_floor, zlim[1])
            try:
                ax.set_box_aspect((1.0, 1.0, 0.88))
            except Exception:
                pass

            ax.set_xlabel("MDS 1", labelpad=12, fontsize=10)
            ax.set_ylabel("MDS 2", labelpad=12, fontsize=10)
            ax.set_zlabel("")  # ticks show generation; omit axis title for cleaner panels
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.zaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = True
            ax.zaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.75))
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_edgecolor("0.55")
            ax.grid(True, linestyle=":", linewidth=0.35, alpha=0.35, color="0.40")
            ax.view_init(elev=elev, azim=azim)
            ax.tick_params(axis="both", which="major", pad=4, labelsize=8)

            if n_views > 1:
                ax.text2D(
                    0.02, 0.98,
                    f"({panel_letters[i]})",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=12,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
                )

        cb = fig.colorbar(
            sm,
            ax=axes,
            fraction=0.03,
            pad=0.04,
            shrink=0.82,
            aspect=24,
        )
        cb.set_label(color_label, rotation=270, labelpad=16, fontsize=11)
        cb.ax.tick_params(labelsize=9, length=3, width=0.6)
        cb.outline.set_linewidth(0.7)

        fig.suptitle(
            "Genetic distance projection (MDS)",
            fontsize=13,
            fontweight="bold",
            y=0.95,
        )
        fig.savefig(save_fpath, dpi=300, facecolor="white")
        plt.close(fig)
        return True
    except Exception:
        return False




# ============================================================================
# PLOTLY INTERACTIVE 3D VISUALIZATION (NEW)
# ============================================================================

def _gdp_reduced_population_info(reduced_genome_data: Any) -> Any:
    """GDP ReducedGenomeData stores metadata on `_population_info` (see matplotlib GDP paths)."""
    info = getattr(reduced_genome_data, "_population_info", None)
    if info is not None:
        return info
    return getattr(reduced_genome_data, "population_info", None)


def generate_gdp_3d_plotly_fitness_figure(
    reduced_genome_data: Any,
    genomes: List[Dict[str, Any]],
    save_fpath: str,
    color_by: str = "species_id",
    use_pub_style: bool = False,
    metric_label: Optional[str] = None,
    north_star_metric: Optional[str] = None,
) -> bool:
    """
    Generate an interactive 3D Plotly figure: X=MDS1, Y=MDS2, Z=north-star fitness.
    
    Args:
        reduced_genome_data: ReducedGenomeData object with reduced_positions and population_info
        genomes: List of genome dicts (with 'id', fitness scores, 'species_id', 'generation', etc.)
        save_fpath: Path to save the interactive HTML file
        color_by: "species_id", "alive", "generation", or "fitness"
        use_pub_style: If True, use publication styling
        metric_label / north_star_metric: axis/hover labels for the active fitness metric
    
    Returns:
        True on success, False otherwise
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    
    try:
        if reduced_genome_data is None:
            return False
        z_label = metric_label or _metric_display_label(_resolve_north_star_metric(north_star_metric))
        
        positions_2d = reduced_genome_data.reduced_positions
        population_info = _gdp_reduced_population_info(reduced_genome_data)
        if not positions_2d or population_info is None:
            return False

        ids = list(positions_2d.keys())
        if not ids:
            return False

        x = np.array([positions_2d[i][0] for i in ids])
        y = np.array([positions_2d[i][1] for i in ids])

        z = np.array([population_info.get(i, {}).get("fitness", 0.5) for i in ids])

        # Determine colors based on color_by
        if color_by == "alive":
            source = [population_info.get(i, {}).get("alive", "archived") for i in ids]
            color_map = {"alive": "green", "archived": "red"}
            colors = [color_map.get(s, "gray") for s in source]
            color_label = "Alive vs Archived"
        elif color_by == "species_id":
            species_ids = [population_info.get(i, {}).get("species_id", 0) for i in ids]
            colors = species_ids
            color_label = "Species ID"
        elif color_by == "generation":
            gens = [int(population_info.get(i, {}).get("generation", 0)) for i in ids]
            colors = gens
            color_label = "Generation"
        else:  # fitness / north-star score
            colors = z
            color_label = z_label

        # alive: discrete HTML color strings — omit colorscale/colorbar (Plotly rejects colorscale=None)
        marker_kw: Dict[str, Any] = dict(size=4, color=colors, opacity=0.8)
        if color_by != "alive":
            marker_kw["colorscale"] = "Viridis"
            marker_kw["colorbar"] = dict(title=color_label)
            marker_kw["line"] = dict(width=0)

        # Create interactive scatter plot
        fig = go.Figure(data=[
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=marker_kw,
                text=[f"ID: {i}<br>{z_label}: {population_info.get(i, {}).get('fitness', 0):.3f}<br>Gen: {population_info.get(i, {}).get('generation', 0)}<br>Species: {population_info.get(i, {}).get('species_id', 0)}" for i in ids],
                hoverinfo='text',
            )
        ])
        
        # Update layout with better styling
        fig.update_layout(
            title=f"Genetic Distance Projection (MDS) + {z_label} [color = {color_by}]",
            scene=dict(
                xaxis=dict(title='MDS 1'),
                yaxis=dict(title='MDS 2'),
                zaxis=dict(title=z_label),
                bgcolor='rgba(240, 240, 240, 0.5)' if use_pub_style else 'white',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            hovermode='closest',
            height=800 if use_pub_style else 700,
            width=1000 if use_pub_style else 900,
            template='plotly' if use_pub_style else 'plotly_white',
            font=dict(family="Arial, sans-serif", size=11),
        )
        
        fig.write_html(save_fpath)
        return True
    except Exception:
        return False


generate_gdp_3d_plotly_toxicity_figure = generate_gdp_3d_plotly_fitness_figure


def generate_gdp_3d_plotly_generation_axis_fitness_color(
    reduced_genome_data: Any,
    genomes: Optional[List[Dict[str, Any]]] = None,
    save_fpath: str = "",
    use_pub_style: bool = False,
    metric_label: Optional[str] = None,
    north_star_metric: Optional[str] = None,
) -> bool:
    """
    Interactive rotatable 3D Plotly figure: X/Y = MDS, Z = generation; color = fitness.

    Drag to rotate; scroll to zoom. Matches the static PNG encoding (no Z-axis title).
    ``genomes`` is unused (kept for call-site compatibility).
    """
    del genomes  # metadata comes from reduced_genome_data
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        return False

    try:
        if reduced_genome_data is None or not save_fpath:
            return False
        fit_label = metric_label or _metric_display_label(_resolve_north_star_metric(north_star_metric))

        positions_2d = reduced_genome_data.reduced_positions
        population_info = _gdp_reduced_population_info(reduced_genome_data)
        if not positions_2d or population_info is None:
            return False

        ids = list(positions_2d.keys())
        if not ids:
            return False

        x = np.array([positions_2d[i][0] for i in ids], dtype=float)
        y = np.array([positions_2d[i][1] for i in ids], dtype=float)
        z_gen = np.array([int(population_info.get(i, {}).get("generation", 0)) for i in ids], dtype=float)
        fit = np.array([float(population_info.get(i, {}).get("fitness", 0.5)) for i in ids], dtype=float)
        species_ids = [int(population_info.get(i, {}).get("species_id", 0) or 0) for i in ids]

        palette = list(px.colors.qualitative.Plotly) + list(px.colors.qualitative.Dark24)
        uniq_spec = sorted(set(species_ids))
        sid_to_hex = {sid: palette[j % len(palette)] for j, sid in enumerate(uniq_spec)}
        species_hex = [sid_to_hex[sid] for sid in species_ids]

        alive_hex = []
        for i in ids:
            a = population_info.get(i, {}).get("alive", "archived")
            alive_hex.append("#2ca02c" if str(a).lower() == "alive" else "#d62728")

        hover = [
            f"ID: {i}<br>{fit_label}: {population_info.get(i, {}).get('fitness', 0):.3f}<br>"
            f"Gen: {population_info.get(i, {}).get('generation', 0)}<br>"
            f"Species: {population_info.get(i, {}).get('species_id', 0)}<br>"
            f"Alive: {population_info.get(i, {}).get('alive', 'archived')}"
            for i in ids
        ]

        n = len(ids)
        marker_size = 5 if n <= 80 else (4 if n <= 250 else 3)

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z_gen,
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        color=fit,
                        colorscale="Viridis",
                        colorbar=dict(title=dict(text=fit_label)),
                        opacity=0.9,
                        showscale=True,
                        line=dict(width=0.4, color="rgba(30,30,30,0.55)"),
                    ),
                    text=hover,
                    hoverinfo="text",
                )
            ]
        )

        def _btn(label: str, m: Dict[str, Any]) -> Dict[str, Any]:
            return dict(label=label, method="restyle", args=[m])

        buttons = [
            _btn(
                f"Color: {fit_label}",
                {
                    "marker.color": [fit],
                    "marker.colorscale": "Viridis",
                    "marker.autocolorscale": True,
                    "marker.showscale": True,
                    "marker.colorbar": [{"title": {"text": fit_label}}],
                },
            ),
            _btn(
                "Color: species",
                {
                    "marker.color": [species_hex],
                    "marker.colorscale": [None],
                    "marker.autocolorscale": False,
                    "marker.showscale": False,
                    "marker.colorbar": [None],
                },
            ),
            _btn(
                "Color: generation",
                {
                    "marker.color": [z_gen],
                    "marker.colorscale": "Plasma",
                    "marker.autocolorscale": True,
                    "marker.showscale": True,
                    "marker.colorbar": [{"title": {"text": "Generation"}}],
                },
            ),
            _btn(
                "Color: alive vs archived",
                {
                    "marker.color": [alive_hex],
                    "marker.colorscale": [None],
                    "marker.autocolorscale": False,
                    "marker.showscale": False,
                    "marker.colorbar": [None],
                },
            ),
        ]

        fig.update_layout(
            title=dict(
                text="Genetic distance projection (MDS)<br>"
                     "<span style='font-size:12px;font-weight:normal;color:#555'>"
                     "Drag to rotate · scroll to zoom</span>",
                x=0.5,
                xanchor="center",
            ),
            scene=dict(
                xaxis=dict(title="MDS 1", backgroundcolor="white", gridcolor="rgba(0,0,0,0.15)"),
                yaxis=dict(title="MDS 2", backgroundcolor="white", gridcolor="rgba(0,0,0,0.15)"),
                # Match static PNG: generation ticks only, no axis title.
                zaxis=dict(title="", backgroundcolor="rgb(245,245,245)", gridcolor="rgba(0,0,0,0.15)"),
                bgcolor="white",
                camera=dict(eye=dict(x=1.45, y=1.45, z=1.15)),
                aspectmode="cube",
                dragmode="orbit",
            ),
            hovermode="closest",
            height=780 if use_pub_style else 720,
            width=980 if use_pub_style else 920,
            template="plotly_white",
            font=dict(family="Times New Roman, Times, serif", size=12),
            margin=dict(l=10, r=10, t=70, b=10),
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=0.99,
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#888",
                    borderwidth=1,
                )
            ],
        )

        fig.write_html(
            save_fpath,
            include_plotlyjs="cdn",
            full_html=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True,
                "modeBarButtonsToAdd": ["orbitRotation", "resetCameraDefault3d"],
            },
        )
        return True
    except Exception:
        return False


def is_gdp_available() -> bool:
    """Return True if the GDP package can be imported."""
    return _GDP_AVAILABLE


def get_gdp_import_error() -> Optional[str]:
    """Return the ImportError message if GDP failed to load, else None."""
    return _GDP_IMPORT_ERROR if not _GDP_AVAILABLE else None


# Public aliases for metric helpers used by live_analysis and callers.
metric_display_label = _metric_display_label
metric_file_tag = _metric_file_tag
resolve_north_star_metric = _resolve_north_star_metric
