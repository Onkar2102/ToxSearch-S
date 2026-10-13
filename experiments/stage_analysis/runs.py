"""Index run directories into factorial cells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

DISTANCE_LABEL = {
    "embedding": "d_E",
    "nli": "d_N",
    "embedding+objective": "d_EP",
    "nli+embedding": "d_H",
    "objective": "d_P",
}
CLUSTERING_LABEL = {"leader_follower": "LF", "dbscan": "IncDBSCAN", "none": "control"}
ORACLE_LABEL = {"google": "M0", "openai": "M1", "laya": "M2"}

PRIMARY_DISTANCES = ("d_E", "d_N", "d_EP")
PRIMARY_CLUSTERINGS = ("LF", "IncDBSCAN")


def _load(path: Path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def find_run_dirs(roots: Iterable[Path]) -> List[Path]:
    out = []
    for root in roots:
        root = Path(root)
        if (root / "run_config.json").is_file():
            out.append(root)
            continue
        out.extend(sorted(p.parent for p in root.rglob("run_config.json")))
    return out


def index_runs(roots: Iterable[Path]) -> pd.DataFrame:
    """One row per run with factor levels, seed, thresholds, and completion status."""
    rows = []
    for run_dir in find_run_dirs(roots):
        cfg = _load(run_dir / "run_config.json") or {}
        sp = cfg.get("speciation") or {}
        tracker = _load(run_dir / "EvolutionTracker.json") or {}
        meta = tracker.get("run_metadata") or {}
        clustering = sp.get("clustering_method") or "none"
        distance = sp.get("distance_method")
        rows.append({
            "run_dir": str(run_dir),
            "evaluator": cfg.get("evaluator"),
            "oracle": ORACLE_LABEL.get(cfg.get("evaluator"), cfg.get("evaluator")),
            "north_star_metric": cfg.get("north_star_metric"),
            "clustering_method": clustering,
            "clustering": CLUSTERING_LABEL.get(clustering, clustering),
            "distance_method": distance,
            "distance": DISTANCE_LABEL.get(distance, distance) if clustering != "none" else "control",
            "seed": cfg.get("seed"),
            "max_total_genomes": cfg.get("max_total_genomes"),
            "theta_sim": sp.get("theta_sim"),
            "theta_merge": sp.get("theta_merge"),
            "dbscan_eps": sp.get("dbscan_eps") if sp.get("dbscan_eps") is not None else sp.get("theta_sim"),
            "dbscan_min_samples": sp.get("dbscan_min_samples"),
            "rg_model": cfg.get("rg_model"),
            "git_commit": cfg.get("git_commit"),
            "status": tracker.get("status"),
            "n_generations": len(tracker.get("generations") or []),
            "run_duration_seconds": meta.get("run_duration_seconds"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["cell"] = df["distance"].astype(str) + "|" + df["clustering"].astype(str)
    return df


def check_design(df: pd.DataFrame, min_seeds: int = 5) -> pd.DataFrame:
    """Seeds per (oracle, cell); flags cells below ``min_seeds`` or with mixed settings."""
    if df.empty:
        return df
    g = df.groupby(["oracle", "cell"])
    out = g.agg(
        n_runs=("run_dir", "size"),
        n_complete=("status", lambda s: int((s == "complete").sum())),
        n_seeds=("seed", "nunique"),
        theta_sim=("theta_sim", lambda s: sorted(set(s.dropna()))),
        max_total_genomes=("max_total_genomes", lambda s: sorted(set(s.dropna()))),
        rg_model=("rg_model", "nunique"),
    ).reset_index()
    out["ok"] = (out["n_complete"] >= min_seeds) & (out["rg_model"] == 1) & (out["max_total_genomes"].map(len) == 1)
    return out
