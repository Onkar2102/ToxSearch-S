#!/usr/bin/env python3
"""Recompute RQ2 endpoint inter / intra species diversity under pure MiniLM d_E.

Inter: mean pairwise d_E = (1 - cos) / 2 between species leaders
(``speciation_state.json`` leader embeddings). Intra: mean over species of the
mean pairwise d_E among endpoint members (LF ``elites.json``, IncDBSCAN
``temp.json``) carrying that ``species_id``; species with < 2 members are skipped.

Overwrites ``inter_diversity`` / ``intra_diversity`` in ``rq2_run_level.csv``
(a ``.bak`` copy of the previous file is written first).

Example::

    PYTHONPATH=src:. python experiments/evostar_2027/rq2/scripts/rq2_de_diversity.py
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CSV = ROOT / "experiments/evostar_2027/rq2/reports/rq2_run_level.csv"


def _load(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _unit(v: Any) -> Optional[np.ndarray]:
    if not isinstance(v, list) or len(v) < 8:
        return None
    x = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(x)
    return x / n if n > 0 else None


def _mean_pairwise_de(x: np.ndarray) -> float:
    sims = np.clip(x @ x.T, -1.0, 1.0)
    iu = np.triu_indices(len(x), k=1)
    return float(np.mean((1.0 - sims[iu]) / 2.0))


def run_diversity(run_dir: Path, clustering: str) -> Tuple[Optional[float], Optional[float]]:
    state = _load(run_dir / "speciation_state.json") or {}
    species = state.get("species") or {}
    leaders = [_unit(sp.get("leader_embedding")) for sp in species.values()]
    leaders = [e for e in leaders if e is not None]
    inter = _mean_pairwise_de(np.stack(leaders)) if len(leaders) >= 2 else None

    names = ("temp.json",) if clustering == "dbscan" else ("elites.json",)
    by_sid: Dict[int, List[np.ndarray]] = {}
    seen = set()
    for name in names:
        for rec in _load(run_dir / name) or []:
            rid = rec.get("id")
            if rid in seen:
                continue
            seen.add(rid)
            try:
                sid = int(rec.get("species_id"))
            except (TypeError, ValueError):
                continue
            if sid <= 0 or str(sid) not in species:
                continue
            e = _unit(rec.get("prompt_embedding"))
            if e is not None:
                by_sid.setdefault(sid, []).append(e)
    intras = [_mean_pairwise_de(np.stack(v)) for v in by_sid.values() if len(v) >= 2]
    intra = float(np.mean(intras)) if intras else None
    return inter, intra


def main() -> int:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    fields = list(rows[0].keys())
    shutil.copyfile(CSV, CSV.with_suffix(".csv.bak"))
    for r in rows:
        inter, intra = run_diversity(ROOT / r["run_dir"], r["clustering"])
        r["inter_diversity"] = "" if inter is None else round(inter, 4)
        r["intra_diversity"] = "" if intra is None else round(intra, 4)
        print(f"{r['cell_id']:40s} inter={r['inter_diversity']} intra={r['intra_diversity']}")
    with CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
