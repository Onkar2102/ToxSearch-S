#!/usr/bin/env python3
"""
Sample GPU utilization and memory for a given duration and write gpu_metrics.json
into the specified output directory. Does not modify any application code; run as
a separate process (e.g. in another terminal or alongside the experiment).

Usage (from project root):

  python scripts/sample_gpu_metrics.py --output-dir data/outputs/20260311_1742 [--duration 60] [--interval 5]

  --output-dir: Required. Run output directory where gpu_metrics.json will be written.
  --duration:   Seconds to sample (default 60). Use 0 to sample once and exit.
  --interval:   Seconds between samples (default 5).

Output: <output_dir>/gpu_metrics.json with per-GPU avg/peak utilization and memory.
Requires nvidia-smi on PATH. If nvidia-smi is not available, exits with a message.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def run_nvidia_smi_query() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return ""
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def parse_nvidia_smi_line(line: str) -> dict:
    # index, name, utilization.gpu %, memory.used MiB, memory.total MiB
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        return {}
    gpu_id = parts[0]
    name = parts[1].strip()
    util = 0
    try:
        util = int(re.sub(r"[^0-9]", "", parts[2]) or "0")
    except ValueError:
        pass
    mem_used = 0
    try:
        mem_used = int(float(re.sub(r"[^0-9.]", "", parts[3]) or "0"))
    except ValueError:
        pass
    mem_total = 0
    try:
        mem_total = int(float(re.sub(r"[^0-9.]", "", parts[4]) or "0"))
    except ValueError:
        pass
    return {
        "gpu_id": int(gpu_id) if gpu_id.isdigit() else gpu_id,
        "device_name": name,
        "utilization_percent": min(100, max(0, util)),
        "memory_used_mb": mem_used,
        "memory_total_mb": mem_total,
    }


def sample_once() -> list:
    raw = run_nvidia_smi_query()
    if not raw:
        return []
    result = []
    for line in raw.split("\n"):
        if not line.strip():
            continue
        row = parse_nvidia_smi_line(line)
        if row:
            result.append(row)
    return result


def aggregate_samples(samples: list) -> list:
    """samples: list of list of per-GPU dicts (one list per sample)."""
    if not samples:
        return []
    n_gpus = len(samples[0])
    aggregated = []
    for gpu_idx in range(n_gpus):
        utils = []
        mem_used = []
        name = None
        mem_total = 0
        gpu_id = None
        for sample in samples:
            if gpu_idx >= len(sample):
                continue
            g = sample[gpu_idx]
            utils.append(g.get("utilization_percent", 0))
            mem_used.append(g.get("memory_used_mb", 0))
            if name is None:
                name = g.get("device_name", "")
            if mem_total == 0:
                mem_total = g.get("memory_total_mb", 0)
            if gpu_id is None:
                gpu_id = g.get("gpu_id", gpu_idx)
        aggregated.append({
            "gpu_id": gpu_id,
            "device_name": name or f"GPU{gpu_idx}",
            "avg_utilization_percent": round(sum(utils) / len(utils), 1) if utils else None,
            "peak_utilization_percent": max(utils) if utils else None,
            "avg_memory_used_mb": round(sum(mem_used) / len(mem_used), 1) if mem_used else None,
            "peak_memory_used_mb": max(mem_used) if mem_used else None,
            "memory_total_mb": mem_total,
            "sample_count": len(samples),
        })
    return aggregated


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sample GPU metrics and write gpu_metrics.json to run output directory.",
    )
    ap.add_argument("--output-dir", type=Path, required=True, help="Run output directory (e.g. data/outputs/20260311_1742)")
    ap.add_argument("--duration", type=float, default=60.0, help="Seconds to sample (default 60; use 0 for single sample)")
    ap.add_argument("--interval", type=float, default=5.0, help="Seconds between samples (default 5)")
    args = ap.parse_args()

    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        print(f"Output directory does not exist: {output_dir}", file=sys.stderr)
        return 1

    first = sample_once()
    if not first:
        print("nvidia-smi did not return GPU data; ensure NVIDIA drivers and nvidia-smi are available.", file=sys.stderr)
        return 1

    samples = [first]
    duration = max(0.0, args.duration)
    interval = max(0.5, args.interval)
    end_time = time.time() + duration
    while duration > 0 and time.time() < end_time:
        time.sleep(interval)
        s = sample_once()
        if s and len(s) == len(first):
            samples.append(s)
        if time.time() >= end_time:
            break

    gpu_metrics = aggregate_samples(samples)
    out_path = output_dir / "gpu_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"gpu_metrics": gpu_metrics}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path} ({len(samples)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
