"""pymoo PCP/Radviz for phase-2 Pareto fronts (from cluster_analysis)."""

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
COLOR_PF_OVERLAY = "#7B61FF"  # appendix cohort PCPs only

# cohort colours (shared Google/OpenAI)
COHORT_LIGHT_HEX: Dict[str, str] = {
    "species_2415": "#F5A0A0",
    "species_2421": "#F5C070",
    "species_2422": "#EDE060",
    "species_2423": "#90D890",
    "species_2424": "#70D0C8",
    "species_2425": "#70A8E8",
    "species_2426": "#9090E8",
    "species_2427": "#B888E8",
    "species_2428": "#E888B8",
    "reserves": "#C8B080",
    "archive": "#88C898",
}

# lighten toward white for PCP lines
COHORT_COLOR_LIGHTEN = 0.18

_PASTEL_CACHE: Optional[Dict[str, Tuple[float, float, float, float]]] = None


def _rgba_from_hex(
    hex_color: str,
    *,
    lighten: float = COHORT_COLOR_LIGHTEN,
) -> Tuple[float, float, float, float]:
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(hex_color)
    if lighten > 0:
        r = r + (1.0 - r) * lighten
        g = g + (1.0 - g) * lighten
        b = b + (1.0 - b) * lighten
    return (float(r), float(g), float(b), 1.0)


def _cohort_pastel_table() -> Dict[str, Tuple[float, float, float, float]]:
    global _PASTEL_CACHE
    if _PASTEL_CACHE is None:
        _PASTEL_CACHE = {
            cohort: _rgba_from_hex(hex_color)
            for cohort, hex_color in COHORT_LIGHT_HEX.items()
        }
    return _PASTEL_CACHE


def _cohort_legend_label(cohort: str) -> str:
    if cohort.startswith("species_"):
        return cohort.split("_", 1)[1]
    return cohort


def _cohort_sort_key(cohort: str) -> Tuple[int, Any]:
    if cohort.startswith("species_"):
        try:
            return (0, int(cohort.split("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    fallback = {"archive": (1, 0), "reserves": (2, 0), "other": (3, 0)}
    return fallback.get(cohort, (4, cohort))


def cohort_pastel_color(cohort: str) -> Tuple[float, float, float, float]:
    """RGBA for cohort label."""
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
    return _rgba_from_hex("#C0C0C0")


def _safe_stem(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return s.strip("_") or "cohort"


def _pcp_figsize(evaluator: str) -> Tuple[float, float]:
    """Figure size; wider canvas for OpenAI axis labels."""
    if evaluator == "openai":
        return (26.0, 9.0)
    return (16.0, 8.5)


# combined PCP export (dpi + fonts)
PCP_SAVE_DPI = 300
PCP_AXIS_LABEL_SIZE = 12.0
PCP_AXIS_LABEL_SIZE_OPENAI = 12.0
PCP_YTICK_LABEL_SIZE = 10.5
PCP_LEGEND_FONT_SIZE = 10.5
PCP_LEGEND_TITLE_SIZE = 11.5


def _apply_publication_rcparams() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": PCP_YTICK_LABEL_SIZE,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _style_pcp_axes(ax: Any, *, evaluator: str, label_style: str) -> None:
    """PCP axis label rotation/size."""
    axis_fs = (
        PCP_AXIS_LABEL_SIZE_OPENAI
        if evaluator == "openai"
        else PCP_AXIS_LABEL_SIZE
    )
    rotation = 48.0 if evaluator == "openai" and label_style == "full" else 40.0
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
        label.set_fontsize(axis_fs)
        label.set_fontweight("medium")
    for label in ax.get_yticklabels():
        label.set_fontsize(PCP_YTICK_LABEL_SIZE)
    ax.tick_params(axis="x", which="major", pad=38, length=3, width=0.7)
    ax.tick_params(axis="y", which="major", length=3, width=0.7, labelsize=PCP_YTICK_LABEL_SIZE)
    bottom = 0.26 if evaluator == "openai" else 0.24
    ax.figure.subplots_adjust(bottom=bottom, left=0.06, right=0.82 if evaluator == "google" else 0.78)


# PCP_LABEL_STYLE: abbrev | full
PCP_LABEL_STYLE: str = "abbrev"

_GOOGLE_ABBREV: Dict[str, str] = {
    "toxicity": "Toxicity",
    "severe_toxicity": "Severe\ntoxicity",
    "identity_attack": "Identity\nattack",
    "insult": "Insult",
    "profanity": "Profanity",
    "threat": "Threat",
    "sexually_explicit": "Sexual\nexplicit",
    "flirtation": "Flirtation",
}

_OPENAI_ABBREV: Dict[str, str] = {
    "harassment": "Harassment",
    "harassment/threatening": "Harass.\nthreatening",
    "hate": "Hate",
    "hate/threatening": "Hate\nthreatening",
    "violence": "Violence",
    "violence/graphic": "Violence\ngraphic",
    "sexual": "Sexual",
    "sexual/minors": "Sexual\nminors",
    "self-harm": "Self-harm",
    "self-harm/intent": "Self-harm\nintent",
    "self-harm/instructions": "Self-harm\ninstruct.",
    "illicit": "Illicit",
    "illicit/violent": "Illicit\nviolent",
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
    """Combined PCP in out_dir; per-cohort copies under appendix/."""
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

    _apply_publication_rcparams()

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
        _style_pcp_axes(pcp.ax, evaluator=evaluator, label_style=label_style)
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
        pcp.fig.savefig(str(path), dpi=PCP_SAVE_DPI, bbox_inches="tight", pad_inches=0.35)
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
    cohorts_in_plot: List[str] = []
    for c in range(n_cls):
        cohort = classes[c]
        m = codes == c
        if not np.any(m):
            continue
        col = cohort_pastel_color(cohort)
        combined.add(X[m], color=col, alpha=0.14, linewidth=0.35)
        g_m = global_pf[m]
        if np.any(g_m):
            combined.add(X[m][g_m], color=col, alpha=0.72, linewidth=0.95)
        cohorts_in_plot.append(cohort)
    combined_path = out_dir / f"pymoo_pcp_pareto_{eval_tag}.pdf"
    combined.do()
    _style_pcp_axes(combined.ax, evaluator=evaluator, label_style=label_style)
    cohort_handles: List[Any] = []
    for cohort in sorted(cohorts_in_plot, key=_cohort_sort_key):
        col = cohort_pastel_color(cohort)
        cohort_handles.append(
            mpl.lines.Line2D(
                [0],
                [0],
                color=col,
                lw=2.4,
                label=_cohort_legend_label(cohort),
            )
        )
    try:
        combined.ax.legend(
            handles=cohort_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.65",
            fontsize=PCP_LEGEND_FONT_SIZE,
            title_fontsize=PCP_LEGEND_TITLE_SIZE,
            ncol=1,
            title="Species / cohort",
        )
    except Exception:
        pass
    combined.fig.savefig(
        str(combined_path),
        dpi=PCP_SAVE_DPI,
        bbox_inches="tight",
        pad_inches=0.4,
        facecolor="white",
    )
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
    "COHORT_LIGHT_HEX",
    "cohort_pastel_color",
    "generate_pymoo_viz",
    "global_pareto_mask",
]
