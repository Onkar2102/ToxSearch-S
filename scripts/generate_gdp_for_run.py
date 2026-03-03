#!/usr/bin/env python3
"""
Generate GDP (Genetic Distance Projection) diagram for a completed run.

Usage:
  From project root with PYTHONPATH=src:
    python scripts/generate_gdp_for_run.py <output_dir>
    python scripts/generate_gdp_for_run.py --output-dir data/outputs/20260302_2028

  Or run after execution: the pipeline also runs GDP at the end of a run.
"""
import argparse
import sys
from pathlib import Path

# Allow importing from src
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Generate GDP diagram (genetic_distance_projection.png) for a run output directory."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Path to run output directory (e.g. data/outputs/20260302_2028)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir_flag",
        default=None,
        help="Path to run output directory (alternative to positional)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir_flag or args.output_dir
    if not output_dir:
        parser.error("Provide output_dir as positional argument or --output-dir")
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        print(f"Error: not a directory: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # Run from project root so utils resolve
    import os
    os.chdir(_project_root)
    sys.path.insert(0, str(_project_root / "src"))

    from utils.gdp_projection import (
        run_gdp_projection,
        run_gdp_projection_nn,
        generate_gdp_figure,
        generate_gdp_3d_toxicity_figure,
        DEFAULT_VIEW_ANGLES,
        is_gdp_available,
        get_gdp_import_error,
    )
    if not is_gdp_available():
        err = get_gdp_import_error()
        print(f"Error: GDP package not available. {err}", file=sys.stderr)
        sys.exit(1)

    elites_path = output_dir / "elites.json"
    reserves_path = output_dir / "reserves.json"
    archive_path = output_dir / "archive.json"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_path = figures_dir / "genetic_distance_projection.png"
    plot_3d_gen_path = figures_dir / "genetic_distance_projection_3d_toxicity_by_generation.png"

    # MDS-GDP (cosine distance)
    payload, reduced = run_gdp_projection(
        elites_path=elites_path,
        reserves_path=reserves_path,
        output_dir=output_dir,
        archive_path=archive_path,
        reduced_size=2,
        save_json=True,
    )
    if reduced is None:
        print("No genomes with embeddings found; nothing to plot.", file=sys.stderr)
        sys.exit(0)
    ok = False
    if generate_gdp_figure(reduced, str(plot_path), color_by="alive"):
        print(f"GDP 2D (MDS) saved: {plot_path}")
        ok = True
    if generate_gdp_3d_toxicity_figure(
        reduced,
        str(plot_3d_gen_path),
        color_by="species_archive",
        publication_style=True,
        view_angles=DEFAULT_VIEW_ANGLES,
    ):
        print(f"GDP 3D (MDS, species + archive, publication style) saved: {plot_3d_gen_path}")
        ok = True
    # Same 3D-by-generation but elites+reserves only (no archive)
    _, reduced_no_arch = run_gdp_projection(
        elites_path=elites_path,
        reserves_path=reserves_path,
        output_dir=output_dir,
        archive_path=None,
        reduced_size=2,
        save_json=False,
    )
    if reduced_no_arch is not None:
        plot_3d_no_arch_path = figures_dir / "genetic_distance_projection_3d_toxicity_by_generation_no_archive.png"
        if generate_gdp_3d_toxicity_figure(reduced_no_arch, str(plot_3d_no_arch_path), color_by="generation"):
            print(f"GDP 3D (MDS, elites+reserves only, color=generation) saved: {plot_3d_no_arch_path}")
            ok = True
    print(f"MDS projection data: {output_dir / 'gdp_projection.json'}")

    # NN-GDP (Euclidean; requires torch)
    try:
        _, reduced_nn = run_gdp_projection_nn(
            elites_path=elites_path,
            reserves_path=reserves_path,
            output_dir=output_dir,
            archive_path=archive_path,
            save_json=True,
        )
        if reduced_nn is not None:
            plot_nn_path = figures_dir / "genetic_distance_projection_nn.png"
            if generate_gdp_figure(reduced_nn, str(plot_nn_path), color_by="alive"):
                print(f"GDP 2D (NN) saved: {plot_nn_path}")
                ok = True
            plot_3d_nn_path = figures_dir / "genetic_distance_projection_3d_toxicity_by_generation_nn.png"
            if generate_gdp_3d_toxicity_figure(reduced_nn, str(plot_3d_nn_path), color_by="generation"):
                print(f"GDP 3D (NN, color=generation) saved: {plot_3d_nn_path}")
                ok = True
            print(f"NN projection data: {output_dir / 'gdp_projection_nn.json'}")
    except Exception as e:
        print(f"NN-GDP skipped (e.g. torch not installed): {e}", file=sys.stderr)

    if not ok:
        print("Failed to generate figures.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
