#!/usr/bin/env python3
"""One-command RQ1 Part-1 (+ optional Part-2) orchestrator.

Stages (comma-separated via --stages / env STAGES)::

    pool → materialize → nli → materialize → p1d → p1e → pilots

Defaults match the current campaign::

    - artifacts under experiments/evostar_2027/data/
    - provisional theta_sim = eps = 0.1
    - NLI gated by --nli-max-n (smoke=50, full=2000, large=3139 + --allow-large-n)
    - with --run-p1d: full LF+IncDBSCAN grids for each batch in --batches
      (hparam + threshold grid + coherence + LaTeX tables)

Examples::

    # Local smoke (cheap + NLI@50, no 1d, no EA)
    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/run_rq1_pipeline.py --smoke

    # Cluster campaign: NLI@2000, attach, full grids for B500/B1000/B2000
    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/run_rq1_pipeline.py \\
      --stages nli,materialize,p1d --nli-max-n 2000 --run-p1d \\
      --batches 500,1000,2000 --t-min 0.05 --t-max 0.5 --t-step 0.05

    # After NLI@2000, full-pool NLI
    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/run_rq1_pipeline.py \\
      --stages nli,materialize --nli-max-n 3139 --allow-large-n
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[4]  # ToxSearch-S
SCRIPTS = Path(__file__).resolve().parent
DATA = Path("experiments/evostar_2027/data")
RQ1 = Path("experiments/evostar_2027/rq1")
NOTES = Path("notes/LaTeX2e+Proceedings+Templates+download")
PROVISIONAL_THETA = 0.1
PROVISIONAL_EPS = 0.1
DEFAULT_BATCHES = (500, 1000, 2000)

ALL_STAGES = (
    "pool",
    "materialize",
    "nli",
    "p1d",
    "p1e",
    "finalize",
    "pilots",
)


def _run(cmd: Sequence[str], *, dry_run: bool = False) -> None:
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.check_call(list(cmd), cwd=str(ROOT))


def _py() -> str:
    return sys.executable


def _script(name: str) -> str:
    return str(SCRIPTS / name)


def stage_pool(args: argparse.Namespace) -> None:
    out = DATA / "pool"
    marker = out / "p0_full.pkl"
    if marker.is_file() and not args.force:
        print(f"[SKIP] pool: {marker} exists")
        return
    cmd = [
        _py(),
        _script("p1a_build_p0_pool.py"),
        "--out-dir",
        str(out),
    ]
    if args.skip_encode:
        cmd.append("--skip-encode")
    _run(cmd, dry_run=args.dry_run)


def stage_materialize(args: argparse.Namespace) -> None:
    cmd = [
        _py(),
        _script("p1_materialize_data.py"),
        "--data-dir",
        str(DATA),
        "--src-p0",
        str(DATA / "pool"),
        "--src-mats",
        str(RQ1 / "mats"),
    ]
    if args.rebuild_cheap or args.force:
        cmd.append("--force-rewrite")
    else:
        cmd.append("--skip-existing")
    if args.rebuild_cheap:
        cmd.append("--rebuild-cheap")
    _run(cmd, dry_run=args.dry_run)


def stage_nli(args: argparse.Namespace) -> None:
    flat = DATA / "mats" / "_flat"
    flat.mkdir(parents=True, exist_ok=True)
    n = int(args.nli_max_n)
    cmd = [
        _py(),
        _script("p1c_d_negE_and_d_N_mats.py"),
        "--p0-dir",
        str(DATA / "pool"),
        "--mats-dir",
        str(flat),
        "--max-n",
        str(n),
        "--which",
        args.nli_which,
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--batch-size",
        str(args.batch_size),
    ]
    if n > 2000:
        cmd.append("--allow-large-n")
    if args.force:
        cmd.append("--force")
    if args.nli_model:
        cmd.extend(["--nli-model", args.nli_model])
    _run(cmd, dry_run=args.dry_run)


def _parse_batches(raw: str) -> List[int]:
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip().upper().lstrip("B")
        if not part:
            continue
        out.append(int(part))
    return out or list(DEFAULT_BATCHES)


def _batch_p1d_done(sweeps: Path, *, expect_nli: bool) -> bool:
    """True when grid+coherence look complete (resume-safe skip)."""
    import csv

    grid = sweeps / "threshold_grid.csv"
    coh = sweeps / "coherence.csv"
    if not grid.is_file() or not coh.is_file():
        return False
    try:
        with grid.open(newline="", encoding="utf-8") as f:
            g_rows = list(csv.DictReader(f))
        with coh.open(newline="", encoding="utf-8") as f:
            c_rows = list(csv.DictReader(f))
    except Exception:
        return False
    if not g_rows or not c_rows:
        return False
    dists = {str(r.get("distance", "")) for r in g_rows}
    need = {"d_E", "d_EP"}
    if expect_nli:
        need |= {"d_N", "d_negE"}
    if not need.issubset(dists):
        return False
    # Require classic + cross-canvas columns (forces rescore after schema bump).
    coh_cols = set(c_rows[0].keys())
    for col in (
        "silhouette_d_E",
        "davies_bouldin",
        "calinski_harabasz",
        "separation_on_d_E",
        "separation_on_d_EP",
        "separation_on_forming",
    ):
        if col not in coh_cols:
            return False
    # Coherence should cover the same method×distance×t cells (approx).
    return len(c_rows) >= max(1, len(g_rows) // 2)


def _run_batch_p1d(args: argparse.Namespace, n: int) -> None:
    """Hparam + threshold grid + coherence + LaTeX tables for one batch."""
    bid = f"B{n}"
    mats = DATA / "mats" / bid
    p0 = DATA / "batches" / bid / "p0.pkl"
    sweeps = RQ1 / "sweeps" / bid
    sweeps.mkdir(parents=True, exist_ok=True)

    if not mats.is_dir():
        raise SystemExit(f"Missing mats dir {mats}")
    if not (mats / "d_E.npy").is_file():
        raise SystemExit(f"Missing {mats / 'd_E.npy'}")
    if not p0.is_file():
        raise SystemExit(f"Missing {p0}")

    missing_nli = [d for d in ("d_N", "d_negE") if not (mats / f"{d}.npy").is_file()]
    if missing_nli:
        print(
            f"[WARN] {bid}: missing NLI mats {missing_nli} — "
            "threshold grid will skip those distances. "
            "Re-run materialize after NLI@2000 completes.",
            flush=True,
        )

    if not args.force and _batch_p1d_done(sweeps, expect_nli=not missing_nli):
        print(f"[SKIP] {bid}: threshold_grid+coherence already present (use --force to redo)")
        # Still refresh LaTeX tables from existing CSVs
        notes = NOTES
        notes.mkdir(parents=True, exist_ok=True)
        _run(
            [
                _py(),
                _script("p1d_generate_rq1_tables.py"),
                "--batch",
                bid,
                "--sweeps-dir",
                str(sweeps),
                "--out",
                str(notes / f"rq1_{bid.lower()}_tables.tex"),
                "--n",
                str(n),
            ],
            dry_run=args.dry_run,
        )
        return

    print(f"\n----- {bid}: hparam sweep -----", flush=True)
    _run(
        [
            _py(),
            _script("p1d_hparam_sweep.py"),
            "--mats-dir",
            str(mats),
            "--sweeps-dir",
            str(sweeps),
            "--prefer-n",
            str(n),
            "--p0",
            str(p0),
            "--reference-radius",
            str(args.reference_radius),
        ],
        dry_run=args.dry_run,
    )

    print(f"\n----- {bid}: threshold grid -----", flush=True)
    _run(
        [
            _py(),
            _script("p1d_threshold_grid.py"),
            "--mats-dir",
            str(mats),
            "--p0",
            str(p0),
            "--sweeps-dir",
            str(sweeps),
            "--t-min",
            str(args.t_min),
            "--t-max",
            str(args.t_max),
            "--t-step",
            str(args.t_step),
        ],
        dry_run=args.dry_run,
    )

    print(f"\n----- {bid}: coherence -----", flush=True)
    _run(
        [
            _py(),
            _script("p1d_cluster_coherence.py"),
            "--species-dir",
            str(sweeps / "species"),
            "--mats-dir",
            str(mats),
            "--p0",
            str(p0),
            "--out",
            str(sweeps / "coherence.csv"),
        ],
        dry_run=args.dry_run,
    )

    notes = NOTES
    notes.mkdir(parents=True, exist_ok=True)
    print(f"\n----- {bid}: LaTeX tables -----", flush=True)
    _run(
        [
            _py(),
            _script("p1d_generate_rq1_tables.py"),
            "--batch",
            bid,
            "--sweeps-dir",
            str(sweeps),
            "--out",
            str(notes / f"rq1_{bid.lower()}_tables.tex"),
            "--n",
            str(n),
        ],
        dry_run=args.dry_run,
    )


def stage_p1d(args: argparse.Namespace) -> None:
    if not args.run_p1d:
        print("[HOLD] p1d — pass --run-p1d (or RUN_P1D=1) to enable full batch grids")
        return

    # Backfill NLI wall timing when progress exists but meta is incomplete.
    flat = DATA / "mats" / "_flat"
    if (flat / f"d_N_{args.nli_max_n}.progress.json").is_file():
        print("\n----- backfill NLI timing -----", flush=True)
        _run(
            [
                _py(),
                _script("backfill_nli_timing.py"),
                "--mats-dir",
                str(flat),
                "--max-n",
                str(int(args.nli_max_n)),
            ],
            dry_run=args.dry_run,
        )

    batches = _parse_batches(args.batches)
    print(f"p1d batches={batches}  t=[{args.t_min}, {args.t_max}] step={args.t_step}", flush=True)
    for n in batches:
        _run_batch_p1d(args, n)

    marker = RQ1 / "sweeps" / "CROSS_BATCH_READY.json"
    if not args.dry_run:
        marker.write_text(
            json.dumps(
                {
                    "batches": [f"B{n}" for n in batches],
                    "ready": True,
                    "t_min": args.t_min,
                    "t_max": args.t_max,
                    "t_step": args.t_step,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {marker}", flush=True)


def stage_finalize(args: argparse.Namespace) -> None:
    """Cross-batch notebook + paper prose + PDF (best-effort on cluster)."""
    cmd = [_py(), _script("finalize_cross_batch_rq1.py")]
    try:
        _run(cmd, dry_run=args.dry_run)
    except subprocess.CalledProcessError as exc:
        print(
            f"[WARN] finalize failed (exit {exc.returncode}); "
            "grids/tables are still usable — sync and finalize locally if needed.",
            flush=True,
        )


def _write_provisional_frozen(args: argparse.Namespace) -> None:
    frozen = {
        "schema_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source": "provisional (1d held)",
        "seed_file": "data/prompt.csv",
        "max_total_genomes": int(args.pilot_max_genomes),
        "shortlist_cli_methods": [
            "embedding",
            "embedding+objective",
            "nli",
            "nli_entail",
        ],
        "clustering_methods": ["leader_follower", "dbscan"],
        "temporal_knobs": "code_defaults_untouched",
        "distances": {
            m: {
                "plan_name": pn,
                "theta_sim": PROVISIONAL_THETA,
                "eps": PROVISIONAL_EPS,
                "dbscan_eps": PROVISIONAL_EPS,
                "min_samples": 2,
                "dbscan_min_samples": 2,
                "provisional": True,
            }
            for m, pn in (
                ("embedding", "d_E"),
                ("embedding+objective", "d_EP"),
                ("nli", "d_N"),
                ("nli_entail", "d_negE"),
            )
        },
        "part1d": "held",
        "provisional_radii": {"theta_sim": PROVISIONAL_THETA, "eps": PROVISIONAL_EPS},
    }
    out = RQ1 / "frozen_config.json"
    if args.dry_run:
        print(f"[DRY] would write provisional {out}")
        return
    out.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"Wrote provisional {out}")


def stage_p1e(args: argparse.Namespace) -> None:
    # Prefer B500 shortlist for pilots (campaign baseline); fall back to larger batches.
    candidates = []
    freeze_n = getattr(args, "freeze_batch", None)
    if freeze_n:
        candidates.append(int(freeze_n))
    candidates.extend([500, 1000, 2000, int(args.nli_max_n)])
    prop = None
    for n in candidates:
        hit = RQ1 / "sweeps" / f"B{n}" / "shortlist_proposal.json"
        if hit.is_file():
            prop = hit
            break
    if prop is None:
        prop = RQ1 / "sweeps" / "shortlist_proposal.json"
    if args.run_p1d and prop.is_file():
        print(f"p1e freeze from {prop}", flush=True)
        cmd = [
            _py(),
            _script("p1e_freeze_config.py"),
            "--sweeps-dir",
            str(prop.parent),
            "--force",
            "--max-total-genomes",
            str(int(args.pilot_max_genomes)),
            "--pilot-seeds",
            str(args.pilot_seeds),
        ]
        # RQ2-style shortlist override (env / CLI); empty ⇒ keep proposal shortlist
        cli_m = str(getattr(args, "freeze_cli_methods", "") or "").strip()
        if cli_m:
            cmd.extend(["--cli-methods", cli_m])
        clus_m = str(getattr(args, "freeze_clustering_methods", "") or "").strip()
        if clus_m:
            cmd.extend(["--clustering-methods", clus_m])
        _run(cmd, dry_run=args.dry_run)
        return
    print("[INFO] p1e: 1d held — writing provisional frozen_config @ theta/eps=0.1")
    _write_provisional_frozen(args)


def _pilot_cell_done(pilots_dir: Path, distance: str, clustering: str, seed: int) -> bool:
    d = distance.replace("+", "_").replace("/", "_")
    c = clustering.replace("-", "_")
    cell_dir = pilots_dir / f"{d}__{c}__seed{seed}"
    meta_path = cell_dir / "cell.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if meta.get("mode") == "run" and meta.get("exit_code") == 0:
        return True
    # Heuristic: EA wrote run artifacts
    run_dir = cell_dir / "run"
    if run_dir.is_dir() and any(run_dir.iterdir()):
        return True
    return False


def stage_pilots(args: argparse.Namespace) -> None:
    if not args.run_pilots:
        print("[SKIP] pilots — pass --run-pilots / RUN_PILOTS=1 to launch Part 2 cells")
        return
    frozen_path = RQ1 / "frozen_config.json"
    if not frozen_path.is_file():
        stage_p1e(args)
    if not frozen_path.is_file():
        raise SystemExit("pilots: missing frozen_config.json after p1e")

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if args.pilot_distances.strip().lower() in ("", "auto", "shortlist"):
        distances = list(frozen.get("shortlist_cli_methods") or [])
    else:
        distances = [d.strip() for d in args.pilot_distances.split(",") if d.strip()]
    if args.pilot_clusterings.strip().lower() in ("", "auto", "shortlist"):
        clusterings = list(frozen.get("clustering_methods") or ["leader_follower", "dbscan"])
    else:
        clusterings = [c.strip() for c in args.pilot_clusterings.split(",") if c.strip()]
    seeds = [int(x) for x in str(args.pilot_seeds).split(",") if x.strip() != ""]
    if not seeds:
        seeds = [int(args.pilot_seed)]

    pilots_dir = RQ1 / "pilots"
    pilots_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Part 2 pilots: distances={distances} clusterings={clusterings} "
        f"seeds={seeds} run={bool(args.pilot_run)}",
        flush=True,
    )
    failures = 0
    for dist in distances:
        for clus in clusterings:
            for seed in seeds:
                if not args.force and _pilot_cell_done(pilots_dir, dist, clus, seed):
                    print(f"[SKIP] pilot {dist} × {clus} × seed{seed} already done")
                    continue
                cmd = [
                    _py(),
                    _script("p2a_launch_pilot_cell.py"),
                    "--frozen-config",
                    str(frozen_path),
                    "--pilots-dir",
                    str(pilots_dir),
                    "--distance",
                    dist,
                    "--clustering",
                    clus,
                    "--seed",
                    str(seed),
                ]
                if args.pilot_run:
                    cmd.append("--run")
                else:
                    cmd.append("--dry-run")
                try:
                    _run(cmd, dry_run=args.dry_run)
                except subprocess.CalledProcessError as exc:
                    failures += 1
                    print(
                        f"[WARN] pilot failed {dist}×{clus}×seed{seed} "
                        f"(exit {exc.returncode}); continuing",
                        flush=True,
                    )
    if failures:
        print(f"[WARN] {failures} pilot cell(s) failed", flush=True)


def _parse_stages(raw: str, smoke: bool) -> List[str]:
    if raw.strip().lower() in ("all", ""):
        stages = ["pool", "materialize", "nli", "materialize", "p1d", "p1e", "finalize"]
        if smoke:
            # smoke still lists p1d (held unless RUN_P1D) + provisional p1e
            pass
        return stages
    out: List[str] = []
    for s in raw.split(","):
        s = s.strip().lower()
        if not s:
            continue
        if s not in ALL_STAGES and s != "materialize":
            raise SystemExit(f"Unknown stage {s!r}; choose from {ALL_STAGES}")
        out.append(s)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1 one-command pipeline")
    p.add_argument(
        "--stages",
        type=str,
        default=os.environ.get("STAGES", "all"),
        help="Comma list or 'all' (default). Env: STAGES",
    )
    p.add_argument("--smoke", action="store_true", help="NLI max-n=50; skip heavy defaults")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    p.add_argument("--force", action="store_true", help="Rebuild even if artifacts exist")
    p.add_argument("--skip-encode", action="store_true")
    p.add_argument("--rebuild-cheap", action="store_true")
    p.add_argument("--nli-max-n", type=int, default=None)
    p.add_argument("--allow-large-n", action="store_true")
    p.add_argument("--nli-which", choices=["both", "d_N", "d_negE"], default="both")
    p.add_argument("--nli-model", type=str, default=os.environ.get("NLI_MODEL"))
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--run-p1d",
        dest="run_p1d",
        action="store_true",
        default=None,
        help="Enable full batch grids (hparam+threshold+coherence+tables)",
    )
    p.add_argument(
        "--no-p1d",
        dest="run_p1d",
        action="store_false",
        help="Hold p1d even if the stage is listed",
    )
    p.add_argument(
        "--batches",
        type=str,
        default=os.environ.get("RQ1_BATCHES", "500,1000,2000"),
        help="Comma batch sizes for p1d grids (default 500,1000,2000). Env: RQ1_BATCHES",
    )
    p.add_argument("--reference-radius", type=float, default=float(os.environ.get("REF_RADIUS", "0.1")))
    p.add_argument("--t-min", type=float, default=float(os.environ.get("T_MIN", "0.05")))
    p.add_argument("--t-max", type=float, default=float(os.environ.get("T_MAX", "0.5")))
    p.add_argument("--t-step", type=float, default=float(os.environ.get("T_STEP", "0.05")))
    p.add_argument(
        "--run-pilots",
        dest="run_pilots",
        action="store_true",
        default=None,
        help="Enable Part 2 pilot stage",
    )
    p.add_argument(
        "--no-pilots",
        dest="run_pilots",
        action="store_false",
        help="Disable Part 2 pilots",
    )
    p.add_argument(
        "--pilot-run",
        dest="pilot_run",
        action="store_true",
        default=None,
        help="Actually run EA (else dry-run stubs). Env: PILOT_RUN=1",
    )
    p.add_argument(
        "--pilot-dry-run",
        dest="pilot_run",
        action="store_false",
        help="Write pilot cell stubs only (no EA)",
    )
    p.add_argument(
        "--pilot-distances",
        type=str,
        default=os.environ.get("PILOT_DISTANCES", "auto"),
        help="Comma CLI distances, or 'auto' = frozen shortlist",
    )
    p.add_argument(
        "--pilot-clusterings",
        type=str,
        default=os.environ.get("PILOT_CLUSTERINGS", "auto"),
        help="Comma clusterings, or 'auto' = frozen clustering_methods",
    )
    p.add_argument("--pilot-seed", type=int, default=int(os.environ.get("PILOT_SEED", "0")))
    p.add_argument(
        "--pilot-seeds",
        type=str,
        default=os.environ.get("PILOT_SEEDS", "0"),
        help="Comma seeds for Part 2 grid (default: 0). Env: PILOT_SEEDS",
    )
    p.add_argument(
        "--pilot-max-genomes",
        type=int,
        default=int(os.environ.get("PILOT_MAX_GENOMES", "150")),
        help="Part 2 EA --max-total-genomes (default 150). Env: PILOT_MAX_GENOMES",
    )
    p.add_argument(
        "--freeze-batch",
        type=int,
        default=int(os.environ.get("FREEZE_BATCH", "500")),
        help="Prefer shortlist from this batch for p1e freeze (default 500)",
    )
    p.add_argument(
        "--freeze-cli-methods",
        type=str,
        default=os.environ.get("FREEZE_CLI_METHODS", ""),
        help="Comma CLI distances for p1e freeze (RQ2: embedding,nli_entail). Env: FREEZE_CLI_METHODS",
    )
    p.add_argument(
        "--freeze-clustering-methods",
        type=str,
        default=os.environ.get("FREEZE_CLUSTERING_METHODS", ""),
        help="Comma clusterings for p1e freeze (RQ2: leader_follower,dbscan). Env: FREEZE_CLUSTERING_METHODS",
    )
    args = p.parse_args()

    if args.run_p1d is None:
        # Default: enable grids unless SMOKE or RUN_P1D=0
        if args.smoke and os.environ.get("RUN_P1D") is None:
            args.run_p1d = False
        else:
            args.run_p1d = os.environ.get("RUN_P1D", "1") != "0"
    if args.run_pilots is None:
        if args.smoke and os.environ.get("RUN_PILOTS") is None:
            args.run_pilots = False
        else:
            args.run_pilots = os.environ.get("RUN_PILOTS", "0") == "1"
    if args.pilot_run is None:
        args.pilot_run = os.environ.get("PILOT_RUN", "0") == "1"

    if args.smoke:
        if args.nli_max_n is None:
            args.nli_max_n = 50
        print("[SMOKE] nli_max_n=", args.nli_max_n)
    elif args.nli_max_n is None:
        args.nli_max_n = int(os.environ.get("NLI_MAX_N", "2000"))

    if args.nli_max_n > 2000 and not args.allow_large_n and os.environ.get("ALLOW_LARGE_N") != "1":
        # allow via flag or env
        if "--allow-large-n" not in sys.argv:
            args.allow_large_n = os.environ.get("ALLOW_LARGE_N") == "1"
        if not args.allow_large_n:
            raise SystemExit("nli_max_n>2000 requires --allow-large-n (or ALLOW_LARGE_N=1)")

    os.chdir(ROOT)
    stages = _parse_stages(args.stages, smoke=args.smoke)
    if args.run_pilots and "pilots" not in stages:
        stages = list(stages) + ["pilots"]

    status = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": stages,
        "smoke": bool(args.smoke),
        "nli_max_n": args.nli_max_n,
        "run_p1d": bool(args.run_p1d),
        "batches": _parse_batches(args.batches) if args.run_p1d else [],
        "threshold_grid": {"t_min": args.t_min, "t_max": args.t_max, "t_step": args.t_step},
        "provisional_radii": {"theta_sim": PROVISIONAL_THETA, "eps": PROVISIONAL_EPS},
        "part1d": "enabled" if args.run_p1d else "held",
    }
    print(json.dumps(status, indent=2), flush=True)

    dispatch = {
        "pool": stage_pool,
        "materialize": stage_materialize,
        "nli": stage_nli,
        "p1d": stage_p1d,
        "p1e": stage_p1e,
        "finalize": stage_finalize,
        "pilots": stage_pilots,
    }
    for s in stages:
        print(f"\n===== STAGE: {s} =====", flush=True)
        dispatch[s](args)

    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "PIPELINE_LAST_RUN.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("\n[OK] RQ1 pipeline finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
