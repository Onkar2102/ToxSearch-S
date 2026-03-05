#!/usr/bin/env python3
"""
Generate interactive 3D Plotly visualizations for ToxSearch-S GDP analysis.

Creates standalone HTML files with full interactive controls:
- Rotate, zoom, pan with mouse
- Hover for genome details (ID, toxicity, generation, species)
- Legend toggle for data series

Usage:
    python scripts/generate_interactive_3d_plots.py <outputs_dir> [--output-dir <dir>]

Example:
    python scripts/generate_interactive_3d_plots.py data/outputs/20260227_0029 --output-dir data/outputs/20260227_0029/figures
"""

import argparse
import sys
from pathlib import Path

# Add src directory to Python path so we can import utils
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from utils.gdp_projection import (
    run_gdp_projection,
    generate_gdp_3d_plotly_generation_axis_toxicity_color,
)
from utils import get_custom_logging

get_logger, setup_file_logging, close_all_logs, _ = get_custom_logging()
logger = get_logger("Generate3DPlots")


def generate_interactive_plots(outputs_path: Path, output_dir: Path = None) -> bool:
    """
    Generate interactive 3D Plotly plots from GDP projection.
    
    Args:
        outputs_path: Path to run outputs directory (contains elites.json, reserves.json, archive.json)
        output_dir: Where to save the plots (defaults to outputs_path/figures)
    
    Returns:
        True if successful, False otherwise
    """
    outputs_path = Path(outputs_path)
    if not outputs_path.exists():
        logger.error(f"Outputs directory not found: {outputs_path}")
        return False
    
    if output_dir is None:
        output_dir = outputs_path / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for required files
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    archive_path = outputs_path / "archive.json"
    
    if not elites_path.exists() or not reserves_path.exists():
        logger.error(f"Missing elites.json or reserves.json in {outputs_path}")
        return False
    
    logger.info(f"Generating interactive 3D plots from {outputs_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Run MDS-GDP projection
    logger.info("Step 1: Computing MDS-GDP projection...")
    payload, reduced_data = run_gdp_projection(
        elites_path=elites_path,
        reserves_path=reserves_path,
        output_dir=outputs_path,
        archive_path=archive_path,
        reduced_size=2,
        save_json=True,
    )
    
    if reduced_data is None:
        logger.error("Failed to generate GDP projection. GDP package may not be available.")
        logger.info("Install with: pip install genetic-distance-projection")
        return False
    
    logger.info(f"GDP projection complete: {len(payload['genome_ids'])} genomes projected")
    
    # Step 2: Load genomes from population files
    logger.info("Step 2: Loading genome data...")
    import json
    
    genomes = []
    for fpath in [elites_path, reserves_path]:
        if fpath.exists():
            with open(fpath, 'r', encoding='utf-8') as f:
                genomes.extend(json.load(f) or [])
    
    if archive_path.exists():
        with open(archive_path, 'r', encoding='utf-8') as f:
            genomes.extend(json.load(f) or [])
    
    logger.info(f"Loaded {len(genomes)} total genomes")
    
    # Step 3: Generate single interactive 3D plot (Z = Generation or Toxicity, Color = Species/Toxicity/Generation, Shape = All or Alive/Archived)
    logger.info("Step 3: Generating interactive Plotly visualization...")
    plot_path = output_dir / "genetic_distance_projection_3d_generation_axis_interactive.html"
    success = generate_gdp_3d_plotly_generation_axis_toxicity_color(
        reduced_data,
        genomes,
        str(plot_path),
        use_pub_style=False,
    )
    if success:
        logger.info(f"✓ Generated: {plot_path}")
    else:
        logger.warning(f"✗ Failed to generate: {plot_path}")
    logger.info("="*60)
    logger.info(f"Interactive 3D plot generation complete: {'1/1' if success else '0/1'} successful")
    logger.info("="*60)
    logger.info("\nGenerated file:")
    logger.info(f"  {plot_path.name}")
    logger.info("  Options: Z-axis (Generation | Toxicity), Color (Species | Toxicity | Generation), Shape (All | Alive/Archived)")
    logger.info("\nInteractive: rotate (drag), zoom (scroll), pan (right-drag), hover for details, legend toggle")
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive 3D Plotly visualizations for GDP analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots for a specific run
  python scripts/generate_interactive_3d_plots.py data/outputs/20260227_0029

  # Generate plots and save to custom directory
  python scripts/generate_interactive_3d_plots.py data/outputs/20260227_0029 \\
      --output-dir data/outputs/20260227_0029/interactive_plots
        """,
    )
    parser.add_argument(
        "outputs_path",
        type=str,
        help="Path to run outputs directory containing elites.json, reserves.json, archive.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save interactive plots (default: <outputs_path>/figures)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (optional)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log_file:
        setup_file_logging(args.log_file, logger)
    
    # Handle relative paths - resolve from project root if not absolute
    outputs_path = Path(args.outputs_path)
    if not outputs_path.is_absolute():
        outputs_path = project_root / outputs_path
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir and not output_dir.is_absolute():
        output_dir = project_root / output_dir
    
    success = generate_interactive_plots(outputs_path, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
