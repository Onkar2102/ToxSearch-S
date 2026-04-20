

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
_project_root = Path(__file__).resolve().parents[2]
_gdp_root = _project_root / "genetic-distance-projection-main"
if _gdp_root.exists() and str(_gdp_root) not in sys.path:
    sys.path.insert(0, str(_gdp_root))

_GDP_AVAILABLE = False
_GDP_IMPORT_ERROR: Optional[str] = None
try:
    from gdp import GenomeData, ReducedGenomeData, GenomeVisualizer
    _GDP_AVAILABLE = True
except ImportError as e:
    GenomeData = None
    ReducedGenomeData = None
    GenomeVisualizer = None
    _GDP_IMPORT_ERROR = str(e)
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances

from .population_io import _extract_north_star_score

_UMAP_AVAILABLE = False
try:
    import umap
    _UMAP_AVAILABLE = True
except ImportError:
    umap = None


def _reduce_using_cosine_mds(genes_matrix: np.ndarray, reduced_size: int = 2, random_state: int = 42) -> np.ndarray:
    
    if genes_matrix.shape[0] < 2:
        return np.zeros((genes_matrix.shape[0], reduced_size), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        distances = cosine_distances(genes_matrix)
        mds = MDS(
            n_components=reduced_size,
            dissimilarity="precomputed",
            random_state=random_state,
            normalized_stress="auto",
        )
        return mds.fit_transform(distances).astype(np.float32)


def _reduce_using_umap(
    genes_matrix: np.ndarray,
    reduced_size: int = 2,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    
    n = genes_matrix.shape[0]
    if n < 2 or not _UMAP_AVAILABLE:
        return np.zeros((n, reduced_size), dtype=np.float32)
    k = min(n_neighbors, max(2, n - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        reducer = umap.UMAP(
            n_components=reduced_size,
            metric="cosine",
            random_state=random_state,
            n_neighbors=k,
            min_dist=min_dist,
        )
        out = reducer.fit_transform(genes_matrix)
    return out.astype(np.float32)


def _load_genomes_from_json(path: Path) -> List[Dict[str, Any]]:
    
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _genomes_to_genome_data(
    genomes: List[Dict[str, Any]],
    alive_ids: Optional[set] = None,
) -> Tuple[Optional[Any], List[int]]:
    
    if not genomes:
        return None, []
    if not _GDP_AVAILABLE:
        return None, [g["id"] for g in genomes]

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
        toxicity = _extract_north_star_score(g, "toxicity")
        if toxicity <= 0.0001 and "fitness" in g:
            toxicity = float(g["fitness"])
        info: Dict[str, Any] = {
            "fitness": float(toxicity),
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


def build_genome_data_from_elites_reserves(
    elites_path: Path,
    reserves_path: Path,
    archive_path: Optional[Path] = None,
) -> Tuple[Optional[Any], List[Dict], List[int]]:
    
    elites = _load_genomes_from_json(Path(elites_path))
    reserves = _load_genomes_from_json(Path(reserves_path))
    all_raw = list(elites) + list(reserves)
    if archive_path and Path(archive_path).exists():
        all_raw = all_raw + _load_genomes_from_json(Path(archive_path))
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

    alive_ids = {g.get("id") for g in elites if g.get("id") is not None}
    alive_ids |= {g.get("id") for g in reserves if g.get("id") is not None}
    genome_data, genome_ids = _genomes_to_genome_data(genomes, alive_ids=alive_ids)
    return genome_data, genomes, genome_ids


def run_gdp_projection(
    elites_path: Path,
    reserves_path: Path,
    output_dir: Path,
    archive_path: Optional[Path] = None,
    reduced_size: int = 2,
    save_json: bool = True,
    random_state: int = 42,
) -> Tuple[Optional[Dict], Optional[Any]]:
    
    genome_data, genomes, genome_ids = build_genome_data_from_elites_reserves(
        elites_path, reserves_path, archive_path=archive_path
    )
    if not genomes:
        return None, None
    if not _GDP_AVAILABLE or genome_data is None:
        return None, None

    def _cosine_mds(genes_matrix: np.ndarray) -> np.ndarray:
        return _reduce_using_cosine_mds(genes_matrix, reduced_size=reduced_size, random_state=random_state)

    reduced = ReducedGenomeData.perform_reduction(source=genome_data, dim_reduction_function=_cosine_mds)
    positions_2d = [reduced.reduced_positions[gid].tolist() for gid in genome_ids]

    payload = {
        "genome_ids": genome_ids,
        "positions_2d": positions_2d,
        "method": "cosine_mds",
    }
    if reduced_size == 3:
        payload["positions_3d"] = positions_2d

    if save_json:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "gdp_projection.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return payload, reduced


def run_gdp_projection_nn(
    elites_path: Path,
    reserves_path: Path,
    output_dir: Path,
    archive_path: Optional[Path] = None,
    save_json: bool = True,
    model_save_fname: str = "gdp_nn_model.pt",
) -> Tuple[Optional[Dict], Optional[Any]]:
    
    genome_data, genomes, genome_ids = build_genome_data_from_elites_reserves(
        elites_path, reserves_path, archive_path=archive_path
    )
    if not genomes:
        return None, None
    if not _GDP_AVAILABLE or genome_data is None:
        return None, None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_models = output_dir / "saved_models"
    saved_models.mkdir(parents=True, exist_ok=True)
    old_cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        reduced = ReducedGenomeData.perform_reduction_nn(
            source=genome_data,
            model_save_fname=model_save_fname,
        )
    finally:
        os.chdir(old_cwd)

    positions_2d = [reduced.reduced_positions[gid].tolist() for gid in genome_ids]
    payload = {
        "genome_ids": genome_ids,
        "positions_2d": positions_2d,
        "method": "nn",
    }

    if save_json:
        out_path = output_dir / "gdp_projection_nn.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return payload, reduced


def run_gdp_projection_umap(
    elites_path: Path,
    reserves_path: Path,
    output_dir: Path,
    archive_path: Optional[Path] = None,
    reduced_size: int = 2,
    save_json: bool = True,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> Tuple[Optional[Dict], Optional[Any]]:
    
    genome_data, genomes, genome_ids = build_genome_data_from_elites_reserves(
        elites_path, reserves_path, archive_path=archive_path
    )
    if not genomes:
        return None, None
    if not _GDP_AVAILABLE or genome_data is None or not _UMAP_AVAILABLE:
        return None, None

    def _umap_fn(genes_matrix: np.ndarray) -> np.ndarray:
        return _reduce_using_umap(
            genes_matrix,
            reduced_size=reduced_size,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )

    reduced = ReducedGenomeData.perform_reduction(source=genome_data, dim_reduction_function=_umap_fn)
    positions_2d = [reduced.reduced_positions[gid].tolist() for gid in genome_ids]
    payload = {
        "genome_ids": genome_ids,
        "positions_2d": positions_2d,
        "method": "umap",
    }

    if save_json:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "gdp_projection_umap.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return payload, reduced


def load_gdp_projection(projection_path: Path) -> Optional[Dict[str, Any]]:
    
    path = Path(projection_path)
    if path.is_dir():
        path = path / "gdp_projection.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_gdp_projection_nn(projection_path: Path) -> Optional[Dict[str, Any]]:
    
    path = Path(projection_path)
    if path.is_dir():
        path = path / "gdp_projection_nn.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_gdp_projection_umap(projection_path: Path) -> Optional[Dict[str, Any]]:
    
    path = Path(projection_path)
    if path.is_dir():
        path = path / "gdp_projection_umap.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def generate_gdp_figure(
    reduced_genome_data: Any,
    save_fpath: str,
    color_by: str = "alive",
    **kwargs: Any,
) -> bool:
    
    if not _GDP_AVAILABLE or reduced_genome_data is None:
        return False
    try:
        viz = GenomeVisualizer(source=reduced_genome_data)
        if color_by == "alive":
            viz.set_genome_colors_by_group(group_key="alive")
        elif color_by == "species_id":
            viz.set_genome_colors_by_group(group_key="species_id")
        else:
            viz.set_colors_by_fitness((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
        viz.visualize_genomes2D(
            save_fpath=save_fpath,
            vis_image_type="png",
            node_size=kwargs.get("node_size", 10),
            trace_best=kwargs.get("trace_best", False),
            trace_gene_origins=kwargs.get("trace_gene_origins", False),
            transform_to_01=kwargs.get("transform_to_01", False),
        )
        return True
    except Exception:
        return False

ELEV_DEFAULT = 15.0
DEFAULT_VIEW_ANGLES: List[Tuple[float, float]] = [
    (ELEV_DEFAULT, 30.0),
    (ELEV_DEFAULT, 150.0),
    (ELEV_DEFAULT, 270.0),
]


def generate_gdp_3d_toxicity_figure(
    reduced_genome_data: Any,
    save_fpath: str,
    color_by: str = "alive",
    publication_style: bool = False,
    view_angles: Optional[List[Tuple[float, float]]] = None,
) -> bool:
    
    if reduced_genome_data is None:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        from utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts

        configure_matplotlib_embedded_fonts()
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        positions = reduced_genome_data.reduced_positions
        population_info = reduced_genome_data._population_info
        if not positions or not population_info:
            return False

        ids = list(positions.keys())
        xy = np.array([positions[i] for i in ids])
        x = xy[:, 0]
        y = xy[:, 1]
        z = np.array([float(population_info.get(i, {}).get("fitness", 0.0)) for i in ids])

        use_pub_style = publication_style or (color_by == "species_archive")
        if use_pub_style:
            plt.rcParams.update({
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
                "font.size": 11,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
            })

        if color_by == "species_archive":
            alive = np.array([population_info.get(i, {}).get("alive", "archived") == "alive" for i in ids])
            species_ids = np.array([population_info.get(i, {}).get("species_id", 0) for i in ids])
            dark_blue = (0.05, 0.05, 0.35)
            uniq_species = sorted(set(species_ids[alive]))
            n_species = len(uniq_species) or 1
            species_to_idx = {s: i for i, s in enumerate(uniq_species)}
            cmap = plt.cm.tab20
            denom = max(len(uniq_species) - 1, 1)
            alive_colors = [cmap(species_to_idx.get(s, 0) / denom) for s in species_ids[alive]] if np.any(alive) else []
            mask_arch = ~alive

            angles = view_angles if view_angles else [(ELEV_DEFAULT, 0.0)]
            n_views = len(angles)
            fig, axes = plt.subplots(
                1, n_views,
                subplot_kw={"projection": "3d"},
                figsize=(7 * n_views, 6),
                facecolor="white",
            )
            if n_views == 1:
                axes = [axes]
            for ax, (elev, azim) in zip(axes, angles):
                ax.set_facecolor("white")
                if np.any(mask_arch):
                    ax.scatter(
                        x[mask_arch], y[mask_arch], z[mask_arch],
                        c=dark_blue, s=2, alpha=0.7, depthshade=True
                    )
                if np.any(alive):
                    ax.scatter(
                        x[alive], y[alive], z[alive],
                        c=alive_colors, s=28, alpha=0.85, depthshade=True
                    )
                ax.set_xlabel("MDS 1")
                ax.set_ylabel("MDS 2")
                ax.set_zlabel("Toxicity (fitness)")
                ax.view_init(elev=elev, azim=azim)
            plt.tight_layout()
            plt.savefig(save_fpath, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()
            return True

        if color_by == "alive":
            source = [population_info.get(i, {}).get("alive", "archived") for i in ids]
            col_map = {"alive": (0.2, 0.6, 0.2), "archived": (0.7, 0.3, 0.3)}
            colors = [col_map.get(s, (0.5, 0.5, 0.5)) for s in source]
        elif color_by == "species_id":
            species_ids = [population_info.get(i, {}).get("species_id", 0) for i in ids]
            uniq = sorted(set(species_ids))
            cmap = plt.cm.tab20
            colors = [cmap((uniq.index(s) % 20) / 19.0) for s in species_ids]
        elif color_by == "generation":
            gens = np.array([int(population_info.get(i, {}).get("generation", 0)) for i in ids])
            norm = plt.Normalize(vmin=gens.min(), vmax=max(gens.max(), 1))
            sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma)
            colors = sm.to_rgba(gens)
        else:
            norm = plt.Normalize(vmin=z.min(), vmax=z.max())
            cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
            colors = cmap.to_rgba(z)

        fig = plt.figure(figsize=(9, 7) if use_pub_style else (8, 6), facecolor="white" if use_pub_style else None)
        ax = fig.add_subplot(111, projection="3d", facecolor="white" if use_pub_style else None)
        ax.scatter(x, y, z, c=colors, s=20, alpha=0.8)
        ax.set_xlabel("MDS 1")
        ax.set_ylabel("MDS 2")
        ax.set_zlabel("Toxicity (fitness)")
        title = "Genetic distance (gen 0 → final) + Toxicity (Z)"
        if color_by == "alive":
            title += " [color = alive vs archived]"
        elif color_by == "generation":
            title += " [color = generation: purple = early, yellow = late]"
            fig.colorbar(sm, ax=ax, shrink=0.6, label="Generation")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(
            save_fpath,
            dpi=300 if use_pub_style else 150,
            bbox_inches="tight",
            facecolor="white" if use_pub_style else None,
        )
        plt.close()
        return True
    except Exception:
        return False


def generate_gdp_3d_generation_axis_toxicity_color(
    reduced_genome_data: Any,
    save_fpath: str,
    view_angles: Optional[List[Tuple[float, float]]] = None,) -> bool:
    
    if reduced_genome_data is None:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        from utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts

        configure_matplotlib_embedded_fonts()
        import matplotlib.pyplot as plt

        positions = reduced_genome_data.reduced_positions
        population_info = reduced_genome_data._population_info
        if not positions or not population_info:
            return False

        ids = list(positions.keys())
        xy = np.array([positions[i] for i in ids])
        x = xy[:, 0]
        y = xy[:, 1]
        z_gen = np.array([int(population_info.get(i, {}).get("generation", 0)) for i in ids])
        toxicity = np.array([float(population_info.get(i, {}).get("fitness", 0.0)) for i in ids])

        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })

        angles = view_angles if view_angles else [(ELEV_DEFAULT, 0.0)]
        n_views = len(angles)
        fig, axes = plt.subplots(
            1, n_views,
            subplot_kw={"projection": "3d"},
            figsize=(7 * n_views, 6),
            facecolor="white",
        )
        if n_views == 1:
            axes = [axes]

        norm = plt.Normalize(vmin=toxicity.min(), vmax=max(toxicity.max(), 0.001))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
        colors = sm.to_rgba(toxicity)

        for ax, (elev, azim) in zip(axes, angles):
            ax.set_facecolor("white")
            ax.scatter(x, y, z_gen, c=colors, s=20, alpha=0.85, depthshade=True)
            ax.set_xlabel("MDS 1")
            ax.set_ylabel("MDS 2")
            ax.set_zlabel("Generation (time)")
            ax.view_init(elev=elev, azim=azim)

        fig.colorbar(sm, ax=axes, shrink=0.5, aspect=25, label="Toxicity (fitness)")
        plt.suptitle("Genetic distance (MDS) + Generation (Z), color = Toxicity", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_fpath, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        return True
    except Exception:
        return False


def _gdp_reduced_population_info(reduced_genome_data: Any) -> Any:
    
    info = getattr(reduced_genome_data, "_population_info", None)
    if info is not None:
        return info
    return getattr(reduced_genome_data, "population_info", None)


def generate_gdp_3d_plotly_toxicity_figure(
    reduced_genome_data: Any,
    genomes: List[Dict[str, Any]],
    save_fpath: str,
    color_by: str = "species_id",
    use_pub_style: bool = False,
) -> bool:
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False
    
    try:
        if reduced_genome_data is None:
            return False
        
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
        else:
            colors = z
            color_label = "Toxicity (Z-axis)"

        marker_kw: Dict[str, Any] = dict(size=4, color=colors, opacity=0.8)
        if color_by != "alive":
            marker_kw["colorscale"] = "Viridis"
            marker_kw["colorbar"] = dict(title=color_label)
            marker_kw["line"] = dict(width=0)

        fig = go.Figure(data=[
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=marker_kw,
                text=[f"ID: {i}<br>Toxicity: {population_info.get(i, {}).get('fitness', 0):.3f}<br>Gen: {population_info.get(i, {}).get('generation', 0)}<br>Species: {population_info.get(i, {}).get('species_id', 0)}" for i in ids],
                hoverinfo='text',
            )
        ])
        
        fig.update_layout(
            title=f"Genetic Distance Projection (MDS) + Toxicity [color = {color_by}]",
            scene=dict(
                xaxis=dict(title='MDS 1'),
                yaxis=dict(title='MDS 2'),
                zaxis=dict(title='Toxicity (fitness)'),
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


def generate_gdp_3d_plotly_generation_axis_toxicity_color(
    reduced_genome_data: Any,
    genomes: List[Dict[str, Any]],
    save_fpath: str,
    use_pub_style: bool = False,
) -> bool:
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        return False
    
    try:
        if reduced_genome_data is None:
            return False
        
        positions_2d = reduced_genome_data.reduced_positions
        population_info = _gdp_reduced_population_info(reduced_genome_data)
        if not positions_2d or population_info is None:
            return False

        ids = list(positions_2d.keys())
        if not ids:
            return False

        x = np.array([positions_2d[i][0] for i in ids])
        y = np.array([positions_2d[i][1] for i in ids])

        z_gen = np.array([int(population_info.get(i, {}).get("generation", 0)) for i in ids])
        tox = np.array([float(population_info.get(i, {}).get("fitness", 0.5)) for i in ids], dtype=float)
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
            f"ID: {i}<br>Toxicity: {population_info.get(i, {}).get('fitness', 0):.3f}<br>"
            f"Gen: {population_info.get(i, {}).get('generation', 0)}<br>"
            f"Species: {population_info.get(i, {}).get('species_id', 0)}<br>"
            f"Alive: {population_info.get(i, {}).get('alive', 'archived')}"
            for i in ids
        ]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z_gen,
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=tox,
                        colorscale="Viridis",
                        colorbar=dict(title=dict(text="Toxicity")),
                        opacity=0.8,
                        showscale=True,
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
                "Color: toxicity (Viridis)",
                {
                    "marker.color": [tox],
                    "marker.colorscale": "Viridis",
                    "marker.autocolorscale": True,
                    "marker.showscale": True,
                    "marker.colorbar": [{"title": {"text": "Toxicity"}}],
                },
            ),
            _btn(
                "Color: species (discrete)",
                {
                    "marker.color": [species_hex],
                    "marker.colorscale": [None],
                    "marker.autocolorscale": False,
                    "marker.showscale": False,
                    "marker.colorbar": [None],
                },
            ),
            _btn(
                "Color: generation (Plasma)",
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
                text="Genetic Distance Projection (MDS) + Generation (Z-axis) — use dropdown to change coloring",
                x=0.5,
                xanchor="center",
            ),
            scene=dict(
                xaxis=dict(title="MDS 1"),
                yaxis=dict(title="MDS 2"),
                zaxis=dict(title="Generation"),
                bgcolor="rgba(240, 240, 240, 0.5)" if use_pub_style else "white",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            ),
            hovermode="closest",
            height=800 if use_pub_style else 720,
            width=1000 if use_pub_style else 900,
            template="plotly" if use_pub_style else "plotly_white",
            font=dict(family="Arial, sans-serif", size=11),
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.02,
                    xanchor="left",
                    y=1.18,
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#888",
                    borderwidth=1,
                    pad=dict(r=6, t=6, b=6, l=6),
                )
            ],
            annotations=[
                dict(
                    text="Point color mode:",
                    x=0.02,
                    y=1.22,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(size=12),
                )
            ],
        )

        fig.write_html(
            save_fpath,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True,
            },
        )
        return True
    except Exception:
        return False


def generate_gdp_3d_plotly_unified(
    reduced_genome_data: Any,
    genomes: List[Dict[str, Any]],
    save_fpath: str,
    use_pub_style: bool = False,
) -> bool:
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        return False

    try:
        if reduced_genome_data is None:
            return False

        positions_2d = reduced_genome_data.reduced_positions
        population_info = _gdp_reduced_population_info(reduced_genome_data)
        if not positions_2d or population_info is None:
            return False

        ids = list(positions_2d.keys())
        if not ids:
            return False

        x = np.array([positions_2d[i][0] for i in ids])
        y = np.array([positions_2d[i][1] for i in ids])
        z_gen = np.array([int(population_info.get(i, {}).get("generation", 0)) for i in ids])
        tox = np.array([float(population_info.get(i, {}).get("fitness", 0.5)) for i in ids], dtype=float)
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
            f"ID: {i}<br>Toxicity: {population_info.get(i, {}).get('fitness', 0):.3f}<br>"
            f"Gen: {population_info.get(i, {}).get('generation', 0)}<br>"
            f"Species: {population_info.get(i, {}).get('species_id', 0)}<br>"
            f"Alive: {population_info.get(i, {}).get('alive', 'archived')}"
            for i in ids
        ]

        def _marker_restyle(color_mode: str) -> Dict[str, Any]:
            if color_mode == "species":
                return {
                    "marker.color": [species_hex],
                    "marker.colorscale": [None],
                    "marker.autocolorscale": False,
                    "marker.showscale": False,
                    "marker.colorbar": [None],
                }
            if color_mode == "alive":
                return {
                    "marker.color": [alive_hex],
                    "marker.colorscale": [None],
                    "marker.autocolorscale": False,
                    "marker.showscale": False,
                    "marker.colorbar": [None],
                }
            if color_mode == "toxicity":
                return {
                    "marker.color": [tox],
                    "marker.colorscale": "Viridis",
                    "marker.autocolorscale": True,
                    "marker.showscale": True,
                    "marker.colorbar": [{"title": {"text": "Toxicity"}}],
                }
            if color_mode == "generation":
                return {
                    "marker.color": [z_gen],
                    "marker.colorscale": "Plasma",
                    "marker.autocolorscale": True,
                    "marker.showscale": True,
                    "marker.colorbar": [{"title": {"text": "Generation"}}],
                }
            raise ValueError(f"unknown color_mode: {color_mode}")

        default_marker_size = 4
        default_marker_opacity = 0.8

        def _trace_layout(z_is_toxicity: bool, color_mode: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            z_arr = tox if z_is_toxicity else z_gen.astype(float)
            z_title = "Toxicity (fitness)" if z_is_toxicity else "Generation"
            trace_patch = {"z": [z_arr]}
            trace_patch.update(_marker_restyle(color_mode))
            trace_patch["marker.size"] = default_marker_size
            trace_patch["marker.opacity"] = default_marker_opacity
            layout_patch = {
                "scene": {
                    "zaxis": {"title": {"text": z_title}},
                },
            }
            return trace_patch, layout_patch

        modes: List[Tuple[str, bool, str]] = [
            ("Toxicity (Z) · species", True, "species"),
            ("Toxicity (Z) · alive vs archived", True, "alive"),
            ("Toxicity (Z) · generation (color)", True, "generation"),
            ("Toxicity (Z) · toxicity (color)", True, "toxicity"),
            ("Generation (Z) · toxicity (color)", False, "toxicity"),
            ("Generation (Z) · species", False, "species"),
            ("Generation (Z) · generation (color)", False, "generation"),
            ("Generation (Z) · alive vs archived", False, "alive"),
        ]

        buttons = []
        for label, z_is_toxicity, cm in modes:
            tp, lp = _trace_layout(z_is_toxicity, cm)
            buttons.append(
                dict(
                    label=label,
                    method="update",
                    args=[tp, lp],
                )
            )

        marker_sizes = [2, 4, 6, 8, 12]
        size_buttons = [
            dict(label=f"{s}px", method="restyle", args=[{"marker.size": s}, [0]]) for s in marker_sizes
        ]
        size_active = marker_sizes.index(default_marker_size)

        opacities = [0.25, 0.5, 0.75, 0.8, 1.0]
        opacity_buttons = []
        for o in opacities:
            if o == 0.8:
                lbl = "80% (default)"
            elif o == 1.0:
                lbl = "100%"
            else:
                lbl = f"{int(round(o * 100))}%"
            opacity_buttons.append(dict(label=lbl, method="restyle", args=[{"marker.opacity": o}, [0]]))
        opacity_active = opacities.index(default_marker_opacity)

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z_gen.astype(float),
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=tox,
                        colorscale="Viridis",
                        colorbar=dict(title=dict(text="Toxicity")),
                        opacity=0.8,
                        showscale=True,
                    ),
                    text=hover,
                    hoverinfo="text",
                )
            ]
        )

        _sidebar_x = 0.02
        _menu_pad = dict(r=4, t=4, b=4, l=4)
        _scene_x0 = 0.26
        _control_rows = (
            (0.82, 0.765),
            (0.54, 0.485),
            (0.26, 0.205),
        )

        fig.update_layout(
            title=dict(
                text="GDP 3D",
                x=0.5,
                xanchor="center",
                y=0.98,
                yref="paper",
                yanchor="top",
                font=dict(size=20, family="Arial, sans-serif"),
            ),
            margin=dict(t=36, l=8, r=96, b=48),
            scene=dict(
                xaxis=dict(title="MDS 1"),
                yaxis=dict(title="MDS 2"),
                zaxis=dict(title="Generation"),
                bgcolor="rgba(240, 240, 240, 0.5)" if use_pub_style else "white",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
                domain=dict(x=[_scene_x0, 0.99], y=[0.02, 0.90]),
            ),
            hovermode="closest",
            height=800 if use_pub_style else 720,
            width=1120 if use_pub_style else 1020,
            template="plotly" if use_pub_style else "plotly_white",
            font=dict(family="Arial, sans-serif", size=11),
            uirevision="gdp3d_unified",
            updatemenus=[
                dict(
                    active=4,
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=_sidebar_x,
                    xanchor="left",
                    y=_control_rows[0][1],
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.96)",
                    bordercolor="#ccc",
                    borderwidth=1,
                    pad=_menu_pad,
                ),
                dict(
                    active=size_active,
                    buttons=size_buttons,
                    direction="down",
                    showactive=True,
                    x=_sidebar_x,
                    xanchor="left",
                    y=_control_rows[1][1],
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.96)",
                    bordercolor="#ccc",
                    borderwidth=1,
                    pad=_menu_pad,
                ),
                dict(
                    active=opacity_active,
                    buttons=opacity_buttons,
                    direction="down",
                    showactive=True,
                    x=_sidebar_x,
                    xanchor="left",
                    y=_control_rows[2][1],
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.96)",
                    bordercolor="#ccc",
                    borderwidth=1,
                    pad=_menu_pad,
                ),
            ],
            annotations=[
                dict(
                    text="<b>View</b>",
                    x=_sidebar_x,
                    y=_control_rows[0][0],
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(size=11, color="#333"),
                ),
                dict(
                    text="<b>Size</b>",
                    x=_sidebar_x,
                    y=_control_rows[1][0],
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(size=11, color="#333"),
                ),
                dict(
                    text="<b>Opacity</b>",
                    x=_sidebar_x,
                    y=_control_rows[2][0],
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(size=11, color="#333"),
                ),
            ],
        )

        fig.write_html(
            save_fpath,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True,
            },
        )
        return True
    except Exception:
        return False


def is_gdp_available() -> bool:
    
    return _GDP_AVAILABLE


def get_gdp_import_error() -> Optional[str]:
    
    return _GDP_IMPORT_ERROR if not _GDP_AVAILABLE else None
