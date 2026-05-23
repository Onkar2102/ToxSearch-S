"""pymoo PCP figures for Phase 2 (ported from AMBER ``utils.post_hoc.pymoo_viz``)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis_utils import PERSPECTIVE_AXIS_ORDER, fast_non_dominated_sort, get_axis_order

_logger = logging.getLogger("emnlp.pymoo_viz")

COLOR_COHORT_PF = "#1a1a1a"
COLOR_GLOBAL_PF = "#C9A227"
COLOR_PF_OVERLAY = "#7B61FF"  # per-cohort PCP only (not combined)
LIGHTEN_PASTEL = 0.62
N_SPECIES_HUES = 9  # hue = index / 9 (cluster_analysis table)

# Fixed species index → pastel (peach, yellow-lime, … pink); see cluster_analysis pymoo_pcp_pareto
SPECIES_HUE_INDEX: Dict[str, int] = {
    "species_2415": 0,
    "species_2421": 1,
    "species_2422": 2,
    "species_2423": 3,
    "species_2424": 4,
    "species_2425": 5,
    "species_2426": 6,
    "species_2427": 7,
    "species_2428": 8,
}

# Non-species cohorts: same pastel recipe, distinct hues (no grey/black)
EXTRA_COHORT_HUE: Dict[str, float] = {
    "reserves": 0.50,  # soft aqua
    "archive": 0.72,   # soft lavender
    "other": 0.38,     # soft mint
}

_PASTEL_CACHE: Optional[Dict[str, Tuple[float, float, float, float]]] = None


def _pastel_from_hue(hue: float) -> Tuple[float, float, float, float]:
    """HSV (hue, 0.55, 0.95) blended 62% toward white — cluster_analysis rule."""
    import matplotlib as mpl

    hsv = (float(hue) % 1.0, 0.55, 0.95)
    r, g, b = mpl.colors.hsv_to_rgb(hsv)
    lighten = LIGHTEN_PASTEL
    r = r + (1.0 - r) * lighten
    g = g + (1.0 - g) * lighten
    b = b + (1.0 - b) * lighten
    return (float(r), float(g), float(b), 1.0)


def _pastel_from_hue_index(i: int) -> Tuple[float, float, float, float]:
    return _pastel_from_hue(float(i) / float(N_SPECIES_HUES))


def _cohort_pastel_table() -> Dict[str, Tuple[float, float, float, float]]:
    global _PASTEL_CACHE
    if _PASTEL_CACHE is None:
        _PASTEL_CACHE = {
            cohort: _pastel_from_hue_index(i) for cohort, i in SPECIES_HUE_INDEX.items()
        }
        for cohort, hue in EXTRA_COHORT_HUE.items():
            _PASTEL_CACHE[cohort] = _pastel_from_hue(hue)
    return _PASTEL_CACHE


def cohort_pastel_color(cohort: str) -> Tuple[float, float, float, float]:
    """Color for a viz cohort label (all cohorts use light pastels)."""
    table = _cohort_pastel_table()
    if cohort in table:
        return table[cohort]
    if cohort.startswith("species_"):
        try:
            sid = int(cohort.split("_", 1)[1])
            key = f"species_{sid}"
            if key in table:
                return table[key]
        except (IndexError, ValueError):
            pass
    return table["other"]


def _safe_stem(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return s.strip("_") or "cohort"


def _pcp_figsize(evaluator: str) -> Tuple[float, float]:
    """Wider canvas when many axes (OpenAI 13-D) so slanted labels fit."""
    if evaluator == "openai":
        return (22.0, 7.5)
    return (14.0, 7.0)


def _slant_pcp_axis_labels(ax: Any, *, rotation: float = 42.0) -> None:
    """Rotate parallel-axis names so long moderation labels do not overlap."""
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    # pymoo uses pad=25 when bounds are shown; keep room below slanted text
    ax.tick_params(axis="x", which="major", pad=32)
    ax.figure.subplots_adjust(bottom=0.22)


# Default PCP x-axis labels: "abbrev" (readable short) or "full" (API attribute names).
PCP_LABEL_STYLE: str = "abbrev"

_GOOGLE_ABBREV: Dict[str, str] = {
    "toxicity": "Toxicity",
    "severe_toxicity": "Sev. toxicity",
    "identity_attack": "Identity attack",
    "insult": "Insult",
    "profanity": "Profanity",
    "threat": "Threat",
    "sexually_explicit": "Sex. explicit",
    "flirtation": "Flirtation",
}

_OPENAI_ABBREV: Dict[str, str] = {
    "harassment": "Harassment",
    "harassment/threatening": "Har. threatening",
    "hate": "Hate",
    "hate/threatening": "Hate threatening",
    "violence": "Violence",
    "violence/graphic": "Viol. graphic",
    "sexual": "Sexual",
    "sexual/minors": "Sexual minors",
    "self-harm": "Self-harm",
    "self-harm/intent": "Self-harm intent",
    "self-harm/instructions": "Self-harm instruct.",
    "illicit": "Illicit",
    "illicit/violent": "Illicit violent",
}


def _full_axis_labels(axes: List[str]) -> List[str]:
    return [a.replace("_", " ").replace("/", " / ") for a in axes]


def _abbrev_axis_labels(axes: List[str], evaluator: str) -> List[str]:
    if evaluator == "google":
        table = _GOOGLE_ABBREV
    elif evaluator == "openai":
        table = _OPENAI_ABBREV
    else:
        table = {}
    out: List[str] = []
    for a in axes:
        out.append(table.get(a, a.replace("_", " ").replace("/", " / ")[:14]))
    return out


def _pcp_axis_labels(
    axes: List[str],
    evaluator: str,
    label_style: str,
) -> List[str]:
    style = (label_style or PCP_LABEL_STYLE).lower()
    if style == "full":
        return _full_axis_labels(axes)
    if style in ("abbrev", "short"):
        return _abbrev_axis_labels(axes, evaluator)
    raise ValueError(f"label_style must be 'abbrev' or 'full', got {label_style!r}")


def _write_label_legend(
    out_dir: Path,
    evaluator: str,
    axes: List[str],
    label_style: str,
) -> None:
    """Map PCP tick text -> full API attribute (for captions / appendix)."""
    import json

    if label_style == "full":
        mapping = { _full_axis_labels(axes)[i]: axes[i] for i in range(len(axes)) }
    else:
        ticks = _abbrev_axis_labels(axes, evaluator)
        full = _full_axis_labels(axes)
        mapping = {ticks[i]: full[i] for i in range(len(axes))}
    path = out_dir / f"pcp_axis_labels_{evaluator}.json"
    path.write_text(
        json.dumps({"label_style": label_style, "tick_to_full": mapping}, indent=2),
        encoding="utf-8",
    )


def _short_attr_labels(axes: List[str]) -> List[str]:
    """Radviz spoke labels (compact)."""
    if tuple(axes) == tuple(PERSPECTIVE_AXIS_ORDER):
        return ["Tox", "ST", "IA", "Ins", "Prof", "Thr", "SE", "Flirt"]
    if all(a in _OPENAI_ABBREV for a in axes):
        return [_OPENAI_ABBREV[a].split()[0][:6] for a in axes]
    return [a.replace("_", " ")[:8] for a in axes]


def _label_encode(cohort_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    classes = sorted(set(cohort_names))
    idx_by_class = {c: i for i, c in enumerate(classes)}
    codes = np.asarray([idx_by_class[c] for c in cohort_names], dtype=np.int64)
    return codes, classes


def global_pareto_mask(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=bool)
    fronts = fast_non_dominated_sort(X)
    mask = np.zeros(n, dtype=bool)
    if fronts:
        mask[fronts[0]] = True
    return mask


def _cohort_front_mask(rows: List[Dict[str, Any]]) -> np.ndarray:
    """First non-dominated front within each ``cohort`` group."""
    n = len(rows)
    mask = np.zeros(n, dtype=bool)
    by_cohort: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        by_cohort.setdefault(str(r["cohort"]), []).append(i)
    for idxs in by_cohort.values():
        if not idxs:
            continue
        F = np.vstack([rows[i]["objectives"] for i in idxs])
        fronts = fast_non_dominated_sort(F)
        if fronts:
            for local in fronts[0]:
                mask[idxs[local]] = True
    return mask


def generate_pymoo_viz(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    *,
    evaluator: str = "google",
    label_style: str = PCP_LABEL_STYLE,
    title_suffix: str = "",
) -> Dict[str, Optional[str]]:
    """Render pymoo PCP (+ Radviz) for one evaluator's objective space.

    Global and cohort F₀ masks are computed on the passed ``objectives`` rows
    (8-D Google or 13-D OpenAI), not mixed across evaluators.

    Hero output: a single combined PCP (``pymoo_pcp_pareto_<evaluator>.png``)
    in ``out_dir``. Per-cohort PCPs and Radviz are demoted to ``out_dir/appendix/``
    to honour the EMNLP plan's <=3 hero-figure budget per phase.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir = out_dir / "appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Optional[str]] = {}
    if not rows:
        return results

    try:
        import matplotlib as mpl
        import matplotlib.colors  # noqa: F401
        import matplotlib.lines  # noqa: F401
        from pymoo.visualization.pcp import PCP
        from pymoo.visualization.radviz import Radviz
    except Exception as e:
        _logger.warning("pymoo not available (%s); skipping pymoo MO figures.", e)
        return results

    X = np.vstack([r["objectives"] for r in rows])
    if X.size == 0:
        return results

    cohort_names = [str(r["cohort"]) for r in rows]
    codes, classes = _label_encode(cohort_names)
    n_cls = len(classes)
    cohort_pf = _cohort_front_mask(rows)
    global_pf = global_pareto_mask(X)

    axis_order = list(get_axis_order(evaluator))
    n_dim = X.shape[1]
    if n_dim != len(axis_order):
        axis_order = axis_order[:n_dim]
    labels_pcp = _pcp_axis_labels(axis_order, evaluator, label_style)
    _write_label_legend(out_dir, evaluator, axis_order, label_style)
    axis_short = _short_attr_labels(axis_order)
    eval_tag = evaluator.replace("/", "_")
    figsize = _pcp_figsize(evaluator)
    slant_rotation = 50.0 if label_style == "full" and evaluator == "openai" else 42.0

    for c in range(n_cls):
        cohort = classes[c]
        m = codes == c
        if not np.any(m):
            continue
        Xc = X[m]
        pf_m = cohort_pf[m]
        pcp = PCP(
            bounds=(0.0, 1.0),
            show_bounds=True,
            n_ticks=6,
            normalize_each_axis=False,
            labels=labels_pcp,
            figsize=figsize,
            title="",
            tight_layout=False,
        )
        col = cohort_pastel_color(cohort)
        pcp.add(Xc, color=col, alpha=0.22, linewidth=0.6)
        if np.any(pf_m):
            pcp.add(Xc[pf_m], color=COLOR_PF_OVERLAY, alpha=0.7, linewidth=1.6)
        pcp.do()
        _slant_pcp_axis_labels(pcp.ax, rotation=slant_rotation)
        try:
            if np.any(pf_m):
                pcp.ax.legend(
                    handles=[
                        mpl.lines.Line2D(
                            [0], [0], color=COLOR_PF_OVERLAY, lw=2.4, label="Cohort 1st front"
                        )
                    ],
                    loc="upper right",
                    frameon=True,
                    fontsize=9,
                )
        except Exception:
            pass
        path = appendix_dir / f"pymoo_pcp_{_safe_stem(cohort)}.png"
        pcp.fig.savefig(str(path), dpi=160, bbox_inches="tight", pad_inches=0.35)
        results[f"appendix_pcp_{cohort}"] = str(path)

    combined = PCP(
        bounds=(0.0, 1.0),
        show_bounds=True,
        n_ticks=6,
        normalize_each_axis=False,
        labels=labels_pcp,
        figsize=figsize,
        title="",
        tight_layout=False,
    )
    cohort_to_counts: Dict[str, Any] = {}
    for c in range(n_cls):
        cohort = classes[c]
        m = codes == c
        if not np.any(m):
            continue
        cohort_to_counts[cohort] = (int(np.sum(cohort_pf[m])), int(np.sum(global_pf[m])))
    for c in range(n_cls):
        cohort = classes[c]
        m = codes == c
        if not np.any(m):
            continue
        pf_m = cohort_pf[m]
        if not np.any(pf_m):
            continue
        col = cohort_pastel_color(cohort)
        combined.add(X[m][pf_m], color=col, alpha=0.9, linewidth=1.4, linestyle=":")
    for c in range(n_cls):
        cohort = classes[c]
        m = codes == c
        if not np.any(m):
            continue
        g_m = global_pf[m]
        if not np.any(g_m):
            continue
        col = cohort_pastel_color(cohort)
        combined.add(X[m][g_m], color=col, alpha=0.85, linewidth=2.2)
    combined_path = out_dir / f"pymoo_pcp_pareto_{eval_tag}.png"
    combined.do()
    _slant_pcp_axis_labels(combined.ax, rotation=slant_rotation)
    style_handles = [
        mpl.lines.Line2D([0], [0], color="0.35", lw=2.0, linestyle=":", label="Cohort F0"),
        mpl.lines.Line2D([0], [0], color="0.35", lw=2.4, linestyle="-", label="Global F0"),
    ]
    cohort_handles: List[Any] = []
    for c in range(n_cls):
        cohort = classes[c]
        if cohort not in cohort_to_counts:
            continue
        nf, ng = cohort_to_counts[cohort]
        col = cohort_pastel_color(cohort)
        cohort_handles.append(
            mpl.lines.Line2D(
                [0],
                [0],
                color=col,
                lw=3.0,
                label=f"{cohort}  (species F0={nf}, global F0={ng})",
            )
        )
    try:
        combined.ax.legend(
            handles=style_handles + cohort_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="0.7",
            fontsize=7.6,
            ncol=1,
        )
    except Exception:
        pass
    combined.fig.savefig(str(combined_path), dpi=160, bbox_inches="tight", pad_inches=0.35)
    results["pcp_combined"] = str(combined_path)

    bulk = ~(cohort_pf | global_pf)
    c_only = cohort_pf & ~global_pf
    g_only = global_pf & ~cohort_pf
    both = cohort_pf & global_pf
    n_global = int(global_pf.sum())
    rad = Radviz(
        labels=axis_short,
        figsize=(9, 8),
        title=(f"pymoo Radviz - global F0={n_global}", {"pad": 12}),
        endpoint_style={"color": "0.25", "s": 55, "alpha": 0.35, "edgecolors": "0.4"},
        tight_layout=True,
    )
    if np.any(bulk):
        rad.add(X[bulk], s=14, alpha=0.28, color="0.45", edgecolors="none")
    if np.any(c_only):
        rad.add(
            X[c_only],
            s=38,
            alpha=0.9,
            facecolors="none",
            edgecolors=COLOR_COHORT_PF,
            linewidths=1.1,
        )
    if np.any(g_only):
        rad.add(
            X[g_only],
            s=52,
            alpha=0.95,
            facecolors="none",
            edgecolors=COLOR_GLOBAL_PF,
            linewidths=1.7,
        )
    if np.any(both):
        rad.add(
            X[both],
            s=58,
            alpha=1.0,
            facecolors="none",
            edgecolors=COLOR_GLOBAL_PF,
            linewidths=2.0,
        )
        rad.add(
            X[both],
            s=40,
            alpha=0.6,
            facecolors="none",
            edgecolors=COLOR_COHORT_PF,
            linewidths=0.9,
        )
    rad_path = appendix_dir / f"pymoo_radviz_pareto_{eval_tag}.png"
    rad.save(str(rad_path))
    results["appendix_radviz"] = str(rad_path)

    return results


__all__ = [
    "PCP_LABEL_STYLE",
    "SPECIES_HUE_INDEX",
    "cohort_pastel_color",
    "generate_pymoo_viz",
    "global_pareto_mask",
]
