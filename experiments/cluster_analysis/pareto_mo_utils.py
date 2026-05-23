"""Deprecated MO helper module.

The canonical implementations now live in
[`src/utils/objectives.py`](../../src/utils/objectives.py). This module
re-exports the same names so existing experiment scripts keep importing
``pareto_mo_utils.fast_non_dominated_sort`` / ``crowding_distance`` /
``dominates`` without modification.

New code should import directly from ``utils.objectives``::

    from utils.objectives import (
        PERSPECTIVE_AXIS_ORDER,
        dominates,
        fast_non_dominated_sort,
        crowding_distance,
        normalize_per_front,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
import warnings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.objectives import (  # noqa: E402
    PERSPECTIVE_AXIS_ORDER,
    crowding_distance,
    dominates as _dominates_max,
    fast_non_dominated_sort,
    normalize_per_front,
)

warnings.warn(
    "pareto_mo_utils is deprecated; use utils.objectives directly.",
    DeprecationWarning,
    stacklevel=2,
)


def dominates(a, b, *, maximize: bool = True) -> bool:  # noqa: D401
    """Backwards-compatible wrapper. The canonical implementation maximizes; flip
    arguments for minimization.
    """
    if maximize:
        return _dominates_max(a, b)
    return _dominates_max(b, a)


__all__ = [
    "PERSPECTIVE_AXIS_ORDER",
    "crowding_distance",
    "dominates",
    "fast_non_dominated_sort",
    "normalize_per_front",
]
