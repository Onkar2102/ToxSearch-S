#!/usr/bin/env python3
"""Quick check: can the GDP package be imported? Run from project root with PYTHONPATH=src."""
import sys
from pathlib import Path

# Same path setup as gdp_projection
_project_root = Path(__file__).resolve().parents[1]
_gdp_root = _project_root / "genetic-distance-projection-main"
if _gdp_root.exists() and str(_gdp_root) not in sys.path:
    sys.path.insert(0, str(_gdp_root))

print("Project root:", _project_root)
print("GDP root:", _gdp_root, "exists:", _gdp_root.exists())
print("Python:", sys.executable)
print()

try:
    from gdp import GenomeData, ReducedGenomeData, GenomeVisualizer
    print("OK: GDP imported successfully.")
except ImportError as e:
    print("FAILED:", e)
    sys.exit(1)
