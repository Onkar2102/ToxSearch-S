#!/usr/bin/env python3
"""Cross-platform timeout wrapper: _run_with_timeout.py SECONDS -- cmd..."""
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        print("usage: _run_with_timeout.py SECONDS -- cmd [args...]", file=sys.stderr)
        return 2
    seconds = float(sys.argv[1])
    cmd = sys.argv[3:]
    try:
        return subprocess.call(cmd, timeout=seconds)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {seconds}s: {' '.join(cmd)}", file=sys.stderr)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
