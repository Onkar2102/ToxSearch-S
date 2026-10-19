#!/usr/bin/env python3
"""One-command RQ1 Part-1 (+ optional Part-2) orchestrator.

Stages (comma-separated via --stages / env STAGES)::

    pool → materialize → nli → materialize → p1d → p1e → pilots

Defaults match the current campaign::

    - artifacts under experiments/evostar_2027/data/
    - provisional theta_sim = eps = 0.1
    - p1d held unless --run-p1d
    - NLI gated by --nli-max-n (smoke=50, full=2000, large=3139 + --allow-large-n)

Examples::

    # Local smoke (cheap + NLI@50, no 1d, no EA)
    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/run_rq1_pipeline.py --smoke

    # Full Part 1 through NLI@2000 (1d still held)
    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/run_rq1_pipeline.py \\
      --stages pool,materialize,nli,materialize --nli-max-n 2000

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
PROVISIONAL_THETA = 0.1
PROVISIONAL_EPS = 0.1

ALL_STAGES = (
    "pool",
    "materialize",
    "nli",
    "p1d",
    "p1e",
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


def stage_p1d(args: argparse.Namespace) -> None:
    if not args.run_p1d:
        print("[HOLD] p1d — pass --run-p1d to enable (campaign default: held)")
        return
    prefer_n = int(args.nli_max_n)
    # Prefer per-batch folder (B500/…) when present; else flat + rq1/mats
    batch_mats = DATA / "mats" / f"B{prefer_n}"
    flat = DATA / "mats" / "_flat"
    mats = batch_mats if batch_mats.is_dir() else (RQ1 / "mats")
    if flat.is_dir() and mats == RQ1 / "mats":
        mats.mkdir(parents=True, exist_ok=True)
        for p in flat.glob("d_*.npy"):
            dest = mats / p.name
            if not dest.exists():
                try:
                    dest.symlink_to(p.resolve())
                except OSError:
                    import shutil

                    shutil.copy2(p, dest)
    sweeps = RQ1 / "sweeps" / f"B{prefer_n}"
    sweeps.mkdir(parents=True, exist_ok=True)
    cmd = [
        _py(),
        _script("p1d_hparam_sweep.py"),
        "--mats-dir",
        str(mats),
        "--sweeps-dir",
        str(sweeps),
        "--prefer-n",
        str(prefer_n),
        "--reference-radius",
        str(PROVISIONAL_THETA),
    ]
    _run(cmd, dry_run=args.dry_run)


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
    prefer_n = int(args.nli_max_n)
    prop = RQ1 / "sweeps" / f"B{prefer_n}" / "shortlist_proposal.json"
    if not prop.is_file():
        prop = RQ1 / "sweeps" / "shortlist_proposal.json"
    if args.run_p1d and prop.is_file():
        cmd = [
            _py(),
            _script("p1e_freeze_config.py"),
            "--sweeps-dir",
            str(prop.parent),
            "--force",
        ]
        _run(cmd, dry_run=args.dry_run)
        return
    print("[INFO] p1e: 1d held — writing provisional frozen_config @ theta/eps=0.1")
    _write_provisional_frozen(args)


def stage_pilots(args: argparse.Namespace) -> None:
    if not args.run_pilots:
        print("[SKIP] pilots — pass --run-pilots to launch Part 2 cells")
        return
    frozen_path = RQ1 / "frozen_config.json"
    if not frozen_path.is_file():
        stage_p1e(args)
    distances = [d.strip() for d in args.pilot_distances.split(",") if d.strip()]
    clusterings = [c.strip() for c in args.pilot_clusterings.split(",") if c.strip()]
    for dist in distances:
        for clus in clusterings:
            cmd = [
                _py(),
                _script("p2a_launch_pilot_cell.py"),
                "--distance",
                dist,
                "--clustering",
                clus,
                "--seed",
                str(args.pilot_seed),
            ]
            if args.pilot_run:
                cmd.append("--run")
            else:
                cmd.append("--dry-run")
            _run(cmd, dry_run=args.dry_run)


def _parse_stages(raw: str, smoke: bool) -> List[str]:
    if raw.strip().lower() in ("all", ""):
        stages = ["pool", "materialize", "nli", "materialize", "p1d", "p1e"]
        if smoke:
            # smoke still lists p1d (held) + provisional p1e; no pilots unless asked
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
    p.add_argument("--run-p1d", action="store_true", default=os.environ.get("RUN_P1D") == "1")
    p.add_argument("--run-pilots", action="store_true", default=os.environ.get("RUN_PILOTS") == "1")
    p.add_argument("--pilot-run", action="store_true", help="Actually run EA (else p2a dry-run)")
    p.add_argument("--pilot-distances", type=str, default="embedding")
    p.add_argument("--pilot-clusterings", type=str, default="leader_follower")
    p.add_argument("--pilot-seed", type=int, default=0)
    p.add_argument("--pilot-max-genomes", type=int, default=250)
    args = p.parse_args()

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
