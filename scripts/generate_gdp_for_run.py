#!/usr/bin/env python3
"""
Generate GDP (Genetic Distance Projection) figures for a completed run.

Generates only the two 3D diagrams:
  - genetic_distance_projection_3d_toxicity_by_generation.png (X=MDS1, Y=MDS2, Z=toxicity; color=species/archive)
  - genetic_distance_projection_3d_generation_axis_toxicity_color.png (X=MDS1, Y=MDS2, Z=generation; color=toxicity)

Usage:
  From project root:
    PYTHONPATH=src python scripts/generate_gdp_for_run.py <output_dir>
    PYTHONPATH=src python scripts/generate_gdp_for_run.py --output-dir data/outputs/20260211_2122

  Or run after execution: the pipeline also runs GDP at the end of a run (run_live_analysis).
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
        description="Generate GDP 3D figures for a run output directory (MDS only)."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Path to run output directory (e.g. data/outputs/20260211_2122)",
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

    import os
    os.chdir(_project_root)
    sys.path.insert(0, str(_project_root / "src"))

    from utils.live_analysis import generate_gdp_projection_plot

    result = generate_gdp_projection_plot(outputs_path=str(output_dir), logger=None)
    if result:
        print(f"GDP figures saved under {output_dir / 'figures'}")
        print(f"  - genetic_distance_projection_3d_toxicity_by_generation.png")
        print(f"  - genetic_distance_projection_3d_generation_axis_toxicity_color.png")
        print(f"Data: {output_dir / 'gdp_projection.json'}")
    else:
        print("No GDP figures generated (check elites.json/reserves.json and genetic-distance-projection-main).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
