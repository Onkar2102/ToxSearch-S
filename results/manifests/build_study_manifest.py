#!/usr/bin/env python3
"""Build SHA256 manifests for tracked study outputs under results/."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFESTS_DIR = Path(__file__).resolve().parent

STUDY_ROOTS = (
    REPO_ROOT / "results" / "emnlp2026",
    REPO_ROOT / "results" / "comparison",
    REPO_ROOT / "results" / "cluster_analysis",
)


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(study_name: str, root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    if root.is_dir():
        for fp in sorted(root.rglob("*")):
            if not fp.is_file():
                continue
            rel = fp.relative_to(REPO_ROOT).as_posix()
            files.append(
                {
                    "path": rel,
                    "bytes": fp.stat().st_size,
                    "sha256": _sha256(fp),
                }
            )
    return {
        "study": study_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "root": root.relative_to(REPO_ROOT).as_posix() if root.exists() else str(root),
        "dependency_files": ["requirements.txt", "requirements-analysis.txt"],
        "regenerate_notes": "See experiments/ drivers and results/manifests/README.md",
        "files": files,
    }


def main() -> int:
    for root in STUDY_ROOTS:
        if not root.exists():
            continue
        name = root.relative_to(REPO_ROOT / "results").as_posix().replace("/", "_")
        manifest = build_manifest(name, root)
        out = MANIFESTS_DIR / f"{name}.json"
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {out} ({len(manifest['files'])} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
