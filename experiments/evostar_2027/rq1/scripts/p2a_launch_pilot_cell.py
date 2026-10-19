#!/usr/bin/env python3
"""Part 2a: launch ONE pilot cell from frozen_config + data/prompt.csv.

One invoke = one (distance_method × clustering_method × seed) cell.
Does not run a full grid.

Dry-run (default; verifies wiring without a full EA)::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p2a_launch_pilot_cell.py \\
      --distance embedding --clustering leader_follower --dry-run

Real launch (N_max=250)::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p2a_launch_pilot_cell.py \\
      --distance embedding --clustering leader_follower --run
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_FROZEN = Path("experiments/evostar_2027/rq1/frozen_config.json")
DEFAULT_PILOTS = Path("experiments/evostar_2027/rq1/pilots")
DEFAULT_SEED = Path("data/prompt.csv")


def _cell_id(distance: str, clustering: str, seed: int) -> str:
    d = distance.replace("+", "_").replace("/", "_")
    c = clustering.replace("-", "_")
    return f"{d}__{c}__seed{seed}"


def build_cli_argv(
    *,
    frozen: Dict[str, Any],
    distance: str,
    clustering: str,
    seed: int,
    out_dir: Path,
) -> List[str]:
    dist_cfg = frozen["distances"][distance]
    theta = float(dist_cfg["theta_sim"])
    eps = float(dist_cfg.get("dbscan_eps") or dist_cfg["eps"])
    min_samples = int(dist_cfg.get("dbscan_min_samples") or dist_cfg["min_samples"])
    max_genomes = int(frozen.get("max_total_genomes", 250))
    seed_file = str(frozen.get("seed_file", "data/prompt.csv"))

    argv = [
        sys.executable,
        "-m",
        "cli",
        "--seed-file",
        seed_file,
        "--seed",
        str(seed),
        "--max-total-genomes",
        str(max_genomes),
        "--distance",
        distance,
        "--clustering-method",
        clustering,
        "--theta-sim",
        str(theta),
        "--output-dir",
        str(out_dir),
    ]
    if clustering == "dbscan":
        argv.extend(["--dbscan-eps", str(eps), "--dbscan-min-samples", str(min_samples)])
    return argv


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1 p2a: launch one pilot cell")
    p.add_argument("--frozen-config", type=Path, default=DEFAULT_FROZEN)
    p.add_argument("--pilots-dir", type=Path, default=DEFAULT_PILOTS)
    p.add_argument("--distance", type=str, required=True, help="CLI distance_method from shortlist")
    p.add_argument(
        "--clustering",
        type=str,
        required=True,
        choices=["leader_follower", "dbscan"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-file", type=Path, default=None, help="Override (default: frozen / prompt.csv)")
    p.add_argument(
        "--run",
        action="store_true",
        help="Actually invoke the EA (default is dry-run: write cell stub + command only)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit dry-run (default when --run is omitted)",
    )
    args = p.parse_args()
    do_run = bool(args.run) and not bool(args.dry_run)

    if not args.frozen_config.is_file():
        raise SystemExit(f"Missing {args.frozen_config}; run p1e first")
    frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    if args.distance not in frozen.get("shortlist_cli_methods", []):
        raise SystemExit(
            f"distance {args.distance!r} not in shortlist "
            f"{frozen.get('shortlist_cli_methods')}"
        )
    if args.distance not in frozen.get("distances", {}):
        raise SystemExit(f"No radii for {args.distance!r} in frozen_config")

    seed_file = args.seed_file or Path(frozen.get("seed_file", DEFAULT_SEED))
    if not seed_file.is_file():
        raise SystemExit(f"Seed file missing: {seed_file}")
    # Ensure frozen seed is prompt.csv only for this campaign
    frozen["seed_file"] = str(seed_file)

    cell = _cell_id(args.distance, args.clustering, args.seed)
    cell_dir = args.pilots_dir / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    run_out = cell_dir / "run"
    argv = build_cli_argv(
        frozen=frozen,
        distance=args.distance,
        clustering=args.clustering,
        seed=args.seed,
        out_dir=run_out,
    )
    cmd = " ".join(shlex.quote(a) for a in argv)

    meta = {
        "cell_id": cell,
        "distance_method": args.distance,
        "clustering_method": args.clustering,
        "seed": args.seed,
        "seed_file": str(seed_file),
        "max_total_genomes": frozen.get("max_total_genomes", 250),
        "frozen_config": str(args.frozen_config),
        "distance_hparams": frozen["distances"][args.distance],
        "command": cmd,
        "argv": argv,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "run" if do_run else "dry-run",
    }
    (cell_dir / "cell.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (cell_dir / "launch.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(Path.cwd()))}\n"
        f"export PYTHONPATH=src:.\n"
        f"{cmd}\n",
        encoding="utf-8",
    )
    print(f"Cell {cell}")
    print(f"  wrote {cell_dir / 'cell.json'}")
    print(f"  command: {cmd}")

    if do_run:
        print("Launching EA...")
        run_out.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        root = str(Path.cwd())
        env["PYTHONPATH"] = f"src:.{(':' + env['PYTHONPATH']) if env.get('PYTHONPATH') else ''}"
        proc = subprocess.run(argv, cwd=root, env=env)
        meta["exit_code"] = proc.returncode
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        (cell_dir / "cell.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return int(proc.returncode)

    print("Dry-run only (pass --run to execute). Wiring verified against frozen_config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
