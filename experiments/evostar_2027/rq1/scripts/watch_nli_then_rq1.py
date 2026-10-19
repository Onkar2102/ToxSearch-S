#!/usr/bin/env python3
"""Daemon: wait for N=2000 NLI complete, then materialize + B1000/B2000 grids + finalize.

Launch detached::

    ./venv/bin/python -c "import subprocess,sys; subprocess.Popen(
        [sys.executable, 'experiments/evostar_2027/rq1/scripts/watch_nli_then_rq1.py'],
        start_new_session=True,
        cwd='.',
    )"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FLAT = ROOT / "experiments/evostar_2027/data/mats/_flat"
DATA = ROOT / "experiments/evostar_2027/data"
RQ1 = ROOT / "experiments/evostar_2027/rq1"
NOTES = ROOT / "notes/LaTeX2e+Proceedings+Templates+download"
LOG = FLAT / "watch_nli_then_rq1.log"
PIDFILE = FLAT / "watch_nli_then_rq1.pid"
PY = ROOT / "venv/bin/python"
ENV = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}", "PYTHONUNBUFFERED": "1"}


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def nli_done(label: str) -> bool:
    prog = FLAT / f"{label}_2000.progress.json"
    if not prog.is_file():
        return False
    meta = json.loads(prog.read_text(encoding="utf-8"))
    return bool(meta.get("complete"))


def next_i(label: str) -> str:
    prog = FLAT / f"{label}_2000.progress.json"
    if not prog.is_file():
        return "?"
    return str(json.loads(prog.read_text(encoding="utf-8")).get("next_i", "?"))


def run(cmd: list[str]) -> None:
    log("+ " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT), env=ENV)


def run_batch(n: int) -> None:
    bid = f"B{n}"
    mats = DATA / "mats" / bid
    sw = RQ1 / "sweeps" / bid
    p0 = DATA / "batches" / bid / "p0.pkl"
    sw.mkdir(parents=True, exist_ok=True)
    if not (mats / "d_N.npy").is_file() or not (mats / "d_negE.npy").is_file():
        raise SystemExit(f"missing NLI under {mats}")
    run(
        [
            str(PY),
            "experiments/evostar_2027/rq1/scripts/p1d_hparam_sweep.py",
            "--mats-dir",
            str(mats),
            "--sweeps-dir",
            str(sw),
            "--prefer-n",
            str(n),
            "--p0",
            str(p0),
            "--reference-radius",
            "0.1",
        ]
    )
    run(
        [
            str(PY),
            "experiments/evostar_2027/rq1/scripts/p1d_threshold_grid.py",
            "--mats-dir",
            str(mats),
            "--p0",
            str(p0),
            "--sweeps-dir",
            str(sw),
            "--t-min",
            "0.1",
            "--t-max",
            "0.5",
            "--t-step",
            "0.05",
        ]
    )
    run(
        [
            str(PY),
            "experiments/evostar_2027/rq1/scripts/p1d_cluster_coherence.py",
            "--species-dir",
            str(sw / "species"),
            "--d-e",
            str(mats / "d_E.npy"),
            "--out",
            str(sw / "coherence.csv"),
        ]
    )
    run(
        [
            str(PY),
            "experiments/evostar_2027/rq1/scripts/p1d_generate_rq1_tables.py",
            "--batch",
            bid,
            "--sweeps-dir",
            str(sw),
            "--out",
            str(NOTES / f"rq1_{bid.lower()}_tables.tex"),
            "--n",
            str(n),
        ]
    )


def main() -> int:
    FLAT.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    log(f"=== watch_nli_then_rq1 start pid={os.getpid()} ===")
    log(f"cwd={ROOT}")

    while not (nli_done("d_N") and nli_done("d_negE")):
        log(f"waiting NLI… d_N next_i={next_i('d_N')}  d_negE next_i={next_i('d_negE')}")
        time.sleep(120)

    log("NLI @2000 complete — ensure timing meta")
    run(
        [
            str(PY),
            "experiments/evostar_2027/rq1/scripts/backfill_nli_timing.py",
            "--max-n",
            "2000",
        ]
    )

    log("NLI @2000 complete — materialize")
    run([str(PY), "experiments/evostar_2027/rq1/scripts/p1_materialize_data.py", "--data-dir", str(DATA)])
    run_batch(1000)
    run_batch(2000)
    (RQ1 / "sweeps" / "CROSS_BATCH_READY.json").write_text(
        json.dumps({"batches": ["B500", "B1000", "B2000"], "ready": True}, indent=2),
        encoding="utf-8",
    )
    log("finalize notebook + paper")
    run([str(PY), "experiments/evostar_2027/rq1/scripts/finalize_cross_batch_rq1.py"])
    log("=== ALL DONE ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log(f"FATAL: {e!r}")
        raise
