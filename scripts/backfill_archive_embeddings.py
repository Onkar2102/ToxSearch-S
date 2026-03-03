#!/usr/bin/env python3
"""
Backfill prompt_embedding for all genomes in archive.json that have a prompt but no embedding.

Use this for runs where archive was written before we kept embeddings in archive, so that
GDP (and any other code) can use the full archive from generation 0 to final.

Usage (from project root):
  PYTHONPATH=src python scripts/backfill_archive_embeddings.py <output_dir>
  PYTHONPATH=src python scripts/backfill_archive_embeddings.py --output-dir data/outputs/20260211_2122

  Optional:
  --model-name all-MiniLM-L6-v2   (must match the model used for the run)
  --batch-size 64
"""
import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Add prompt_embedding to genomes in archive.json that are missing it."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Run output directory (e.g. data/outputs/20260211_2122)",
    )
    parser.add_argument("--output-dir", dest="output_dir_flag", default=None, help="Same as positional output_dir")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="Embedding model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding")
    args = parser.parse_args()
    output_dir = args.output_dir_flag or args.output_dir
    if not output_dir:
        parser.error("Provide output_dir as positional argument or --output-dir")
    output_dir = Path(output_dir).resolve()
    archive_path = output_dir / "archive.json"
    if not archive_path.exists():
        print(f"Error: {archive_path} not found", file=sys.stderr)
        sys.exit(1)

    from speciation.embeddings import backfill_embeddings_for_genomes
    from utils import get_custom_logging

    get_logger, _, _, _ = get_custom_logging()
    logger = get_logger("BackfillArchive")

    with open(archive_path, "r", encoding="utf-8") as f:
        genomes = json.load(f)
    if not isinstance(genomes, list):
        print("Error: archive.json is not a list of genomes", file=sys.stderr)
        sys.exit(1)
    if not genomes:
        print("Archive is empty, nothing to backfill.")
        return
    need = sum(1 for g in genomes if isinstance(g, dict) and g.get("prompt") and not g.get("prompt_embedding"))
    if need == 0:
        print("All archive genomes already have prompt_embedding.")
        return
    print(f"Backfilling embeddings for {need} genomes in {archive_path} ...")
    n = backfill_embeddings_for_genomes(
        genomes,
        model_name=args.model_name,
        batch_size=args.batch_size,
        show_progress=True,
        logger=logger,
    )
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(genomes, f, indent=2, ensure_ascii=False)
    print(f"Done. Backfilled {n} genomes and saved {archive_path}")


if __name__ == "__main__":
    main()
