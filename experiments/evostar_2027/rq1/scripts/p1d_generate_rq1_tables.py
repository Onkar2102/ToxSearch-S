#!/usr/bin/env python3
"""Generate LaTeX RQ1 tables from a batch sweeps directory.

Matches the B500 paper style (multi-row \\thead headers, full forms, dagger marks).

Example::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1d_generate_rq1_tables.py \\
      --batch B1000 \\
      --sweeps-dir experiments/evostar_2027/rq1/sweeps/B1000 \\
      --out notes/LaTeX2e+Proceedings+Templates+download/rq1_b1000_tables.tex
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DIST_TEX = {
    "d_E": r"$d_E$",
    "d_P": r"$d_P$",
    "d_EP": r"$d_{EP}$",
    "d_N": r"$d_N$",
    "d_negE": r"$d_{\neg E}$",
}
DIST_ORDER = ["d_E", "d_EP", "d_N", "d_negE", "d_P"]


def _fmt_f(x: Any, nd: int = 3) -> str:
    return f"{float(x):.{nd}f}"


def _fmt_i(x: Any) -> str:
    return str(int(float(x)))


def _coh_key(
    coh: List[Dict[str, str]],
    method: str,
    dist: str,
    t: float,
    ms: Optional[Any] = None,
) -> Optional[Dict[str, str]]:
    for r in coh:
        if r["method"] != method or r["distance"] != dist:
            continue
        if abs(float(r["threshold"]) - float(t)) > 1e-9:
            continue
        if method == "incdbscan" and ms is not None and r.get("min_samples") not in (None, ""):
            try:
                if abs(float(r["min_samples"]) - float(ms)) > 1e-9:
                    continue
            except Exception:
                continue
        return r
    return None


def _dist_sort_key(d: str) -> int:
    try:
        return DIST_ORDER.index(d)
    except ValueError:
        return 99


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt_wall(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    s = float(seconds)
    if s >= 3600:
        return f"{s / 3600.0:.2f} h ({s:.0f} s)"
    if s >= 60:
        return f"{s / 60.0:.1f} min ({s:.0f} s)"
    return f"{s:.1f} s"


def _collect_timing(sweeps_dir: Path, n: int) -> List[Tuple[str, str]]:
    """Return (item, setting) rows for computation-time lines in the setup table."""
    rows: List[Tuple[str, str]] = []
    flat = Path("experiments/evostar_2027/data/mats/_flat")
    # Prefer exact-N meta; else N=2000 source (B1000 is a slice).
    meta_candidates = [
        flat / f"nli_mats_meta_{n}.json",
        flat / "nli_mats_meta_2000.json",
        Path("experiments/evostar_2027/rq1/mats") / f"nli_mats_meta_{n}.json",
    ]
    nli_meta: Dict[str, Any] = {}
    for c in meta_candidates:
        nli_meta = _load_json(c)
        if nli_meta:
            break
    timing = nli_meta.get("timing") or {}
    for label, tex in (("d_N", r"$d_N$"), ("d_negE", r"$d_{\neg E}$")):
        # Prefer progress.json at the source N used for this batch
        src_n = int(nli_meta.get("N", n)) if nli_meta else n
        prog = _load_json(flat / f"{label}_{src_n}.progress.json")
        wall = prog.get("wall_seconds")
        rate = prog.get("pairs_per_sec")
        if wall is None and label in timing:
            wall = timing[label].get("wall_seconds")
            rate = timing[label].get("pairs_per_sec")
        if wall is None:
            continue
        rate_s = f"; {float(rate):.0f} pairs/s" if rate is not None else ""
        note = f" (from N={src_n} flat mats)" if src_n != n else ""
        rows.append(
            (
                f"Computation time (NLI {tex})",
                f"{_fmt_wall(float(wall))}{rate_s}{note}",
            )
        )
    if nli_meta.get("timing_total_wall_seconds") is not None:
        rows.append(
            (
                "Computation time (NLI total)",
                _fmt_wall(float(nli_meta["timing_total_wall_seconds"])),
            )
        )

    grid_meta = _load_json(sweeps_dir / "threshold_grid_meta.json")
    gwall = (grid_meta.get("timing") or {}).get("wall_seconds")
    if gwall is not None:
        rows.append(("Computation time (threshold grid)", _fmt_wall(float(gwall))))

    prop = _load_json(sweeps_dir / "shortlist_proposal.json")
    hwall = (prop.get("timing") or {}).get("wall_seconds")
    if hwall is not None:
        rows.append(("Computation time (calibration / structure)", _fmt_wall(float(hwall))))
    return rows


def generate(batch: str, sweeps_dir: Path, out: Path, n: int) -> None:
    calib = list(csv.DictReader(open(sweeps_dir / "calibration.csv")))
    struct = [
        r for r in csv.DictReader(open(sweeps_dir / "structure.csv")) if r["min_samples"] == "2"
    ]
    grid = list(csv.DictReader(open(sweeps_dir / "threshold_grid.csv")))
    coh = list(csv.DictReader(open(sweeps_dir / "coherence.csv")))
    target = float(calib[0]["target_fraction"]) if calib else 0.0
    batch_lower = batch.lower()
    timing_rows = _collect_timing(sweeps_dir, n)
    t_vals = sorted({round(float(r["threshold"]), 2) for r in grid})
    n_t = len(t_vals)
    n_dist = len({r["distance"] for r in grid if r["distance"] in DIST_TEX and r["distance"] != "d_P"})
    n_methods = len({r["method"] for r in grid})
    n_runs = len([r for r in grid if r["distance"] in ("d_E", "d_EP", "d_N", "d_negE")])
    if n_t >= 2:
        t_grid_tex = (
            f"$t\\in\\{{{t_vals[0]:.2f},{t_vals[1]:.2f},\\ldots,{t_vals[-1]:.2f}\\}}$ "
            f"({n_t} values $\\times$ {n_dist} distances $\\times$ {n_methods} methods "
            f"$=$ {n_runs} runs)"
        )
    elif n_t == 1:
        t_grid_tex = f"$t={t_vals[0]:.2f}$ ({n_runs} runs)"
    else:
        t_grid_tex = "threshold grid (empty)"

    lines: List[str] = [
        f"% Auto-generated from {sweeps_dir} — do not hand-edit.",
        "% Regenerated by p1d_generate_rq1_tables.py; included from 04-experimentation.tex",
        "",
    ]

    # Setup
    setup_body = [
        (
            f"Held-out batch & Fitness-top $N{{=}}{n}$ prompts "
            r"(B500 $\subset$ B1000 $\subset$ B2000 $\subset$ B3139) \\"
        ),
        r"Distances & $d_E$, $d_{EP}$, $d_N$, $d_{\neg E}$ (plus $d_P$ for calibration only) \\",
        (
            r"Speciation methods & Leader--Follower (offline) and Incremental DBSCAN "
            r"(\texttt{seed\_from\_precomputed\_matrix}, fitness-descending insert order) \\"
        ),
        f"Threshold grid & {t_grid_tex} \\\\",
        (
            r"Incremental DBSCAN minimum samples & $\textit{min\_samples}=2$ "
            r"(structure also probed at $m\in\{2,3,5\}$) \\"
        ),
        (
            f"Calibration & Match neighbor fraction of $d_E$ at radius $0.1$ "
            f"($\\approx{_fmt_f(target, 3)}$) \\\\"
        ),
        (
            r"Topic-clustering proxy & Within-/between-cluster Separation and silhouette "
            r"scored under \emph{all} canvases $d_E$, $d_{EP}$, $d_N$, $d_{\neg E}$ "
            r"(primary ranking on held-out $d_E$); Davies--Bouldin / Calinski--Harabasz on "
            r"MiniLM vectors; prompt samples from the largest niches \\"
        ),
    ]
    for item, setting in timing_rows:
        # Escape underscores in plain settings for LaTeX safety
        setting_tex = setting.replace("_", r"\_")
        setup_body.append(f"{item} & {setting_tex} \\\\")
    setup_body.append(
        r"Artifacts & \texttt{threshold\_grid.csv}, \texttt{species/}, "
        r"\texttt{coherence.csv}, \texttt{calibration.csv}, \texttt{structure.csv} \\"
    )

    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Batch~{batch} research-question~1 offline execution setup.}}",
        f"\\label{{tab:rq1-{batch_lower}-setup}}",
        r"\scriptsize",
        r"\begin{tabular}{@{}lp{0.62\textwidth}@{}}",
        r"\toprule",
        r"Item & Setting \\",
        r"\midrule",
        *setup_body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    # Calibration
    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            f"\\caption{{Batch~{batch} neighbor-fraction calibration "
            f"(reference: $d_E$ at radius $0.1$; target neighbor fraction "
            f"$\\approx{_fmt_f(target, 3)}$). "
            r"$\theta_{\mathrm{sim}}$ is the Leader--Follower join threshold (strict $<$); "
            r"$\varepsilon$ is the Incremental DBSCAN neighborhood radius (non-strict $\le$).}"
        ),
        f"\\label{{tab:rq1-{batch_lower}-calib}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        (
            r"Distance & \thead{Command-line\\method} & $N$ & "
            r"\thead{Target neighbor\\fraction} & \thead{Reference\\radius} & "
            r"$\theta_{\mathrm{sim}}$ & $\varepsilon$ \\"
        ),
        r"\midrule",
    ]
    for r in calib:
        # Escape underscores outside the f-string (Python 3.11 forbids \\ in {...}).
        cli_tex = str(r["cli_method"]).replace("_", r"\_")
        lines.append(
            f"{DIST_TEX[r['distance']]} & {cli_tex} & {r['N']} & "
            f"{_fmt_f(r['target_fraction'], 3)} & {_fmt_f(r['reference_radius'], 2)} & "
            f"{_fmt_f(r['theta_sim'], 4)} & {_fmt_f(r['eps'], 4)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Structure
    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            f"\\caption{{Batch~{batch} Incremental DBSCAN structure probes at calibrated "
            r"$\varepsilon$ ($\textit{min\_samples}=2$; fitness-descending seed on "
            r"precomputed distance matrices).}"
        ),
        f"\\label{{tab:rq1-{batch_lower}-structure}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        (
            r"Distance & $\varepsilon$ & \thead{Number of\\clusters} & "
            r"\thead{Noise\\fraction} & \thead{Maximum\\cluster size} & "
            r"\thead{Singleton\\fraction} \\"
        ),
        r"\midrule",
    ]
    for r in struct:
        if r["distance"] not in DIST_TEX:
            continue
        lines.append(
            f"{DIST_TEX[r['distance']]} & {_fmt_f(r['eps'], 4)} & {r['n_clusters']} & "
            f"{_fmt_f(r['noise_frac'], 3)} & {_fmt_i(r['max_cluster'])} & "
            f"{_fmt_f(r['singleton_frac'], 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    def _coh_f(c: Optional[Dict[str, str]], key: str, nd: int) -> str:
        if not c or c.get(key) in (None, ""):
            return "---"
        try:
            return _fmt_f(c[key], nd)
        except Exception:
            return "---"

    def _dagger_mark(method: str, r: Dict[str, str]) -> str:
        mark = ""
        if (
            method in ("leader_follower", "incdbscan")
            and r["distance"] in ("d_E", "d_EP")
            and float(r["threshold"]) <= 0.15 + 1e-9
            and float(r["threshold"]) >= 0.05 - 1e-9
        ):
            mark = r"$^\dagger$"
        if (
            method == "incdbscan"
            and r["distance"] in ("d_N", "d_negE")
            and float(r["max_cluster"]) >= 0.5 * n
        ):
            mark = r"$^\ddagger$"
        return mark

    def grid_table(method: str, label: str, caption: str) -> List[str]:
        out = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{1.8pt}",
            r"\begin{tabular}{@{}lrrrrrrrrrl@{}}",
            r"\toprule",
        ]
        if method == "leader_follower":
            out.append(
                r"Distance & \thead{Threshold\\$t$} & \thead{Mature\\niches} & Singletons & "
                r"\thead{Maximum\\niche size} & \thead{Fraction in\\mature niches} & "
                r"\thead{Neighbor\\fraction} & \thead{Within\\(intra $d_E$)} & "
                r"\thead{Between\\(inter $d_E$)} & Separation & \thead{Structurally\\healthy} \\"
            )
        else:
            out.append(
                r"Distance & \thead{Threshold\\$t$} & \thead{Number of\\clusters} & "
                r"\thead{Noise\\fraction} & \thead{Maximum\\cluster size} & "
                r"\thead{Fraction in\\mature niches} & \thead{Neighbor\\fraction} & "
                r"\thead{Within\\(intra $d_E$)} & \thead{Between\\(inter $d_E$)} & "
                r"Separation & \thead{Structurally\\healthy} \\"
            )
        out.append(r"\midrule")
        rows = [r for r in grid if r["method"] == method]
        rows.sort(key=lambda r: (_dist_sort_key(r["distance"]), float(r["threshold"])))
        prev = None
        for r in rows:
            if r["distance"] not in DIST_TEX:
                continue
            c = _coh_key(coh, method, r["distance"], r["threshold"], r.get("min_samples"))
            intra = _coh_f(c, "intra_d_E", 3)
            inter = _coh_f(c, "inter_d_E", 3)
            sep = _coh_f(c, "separation", 2)
            ok = "Yes" if c and str(c.get("structurally_healthy")) == "True" else "No"
            mark = _dagger_mark(method, r)
            dist = DIST_TEX[r["distance"]] + mark
            if prev == r["distance"]:
                dist = ""
            prev = r["distance"]
            if method == "leader_follower":
                out.append(
                    f"{dist} & {_fmt_f(r['threshold'], 2)} & {r['n_mature']} & "
                    f"{r['n_singletons']} & {_fmt_i(r['max_cluster'])} & "
                    f"{_fmt_f(r['frac_in_mature'], 3)} & {_fmt_f(r['neighbor_frac_lf'], 3)} & "
                    f"{intra} & {inter} & {sep} & {ok} \\\\"
                )
            else:
                out.append(
                    f"{dist} & {_fmt_f(r['threshold'], 2)} & {r['n_clusters']} & "
                    f"{_fmt_f(r['noise_frac'], 3)} & {_fmt_i(r['max_cluster'])} & "
                    f"{_fmt_f(r['frac_in_mature'], 3)} & {_fmt_f(r['neighbor_frac_lf'], 3)} & "
                    f"{intra} & {inter} & {sep} & {ok} \\\\"
                )
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        return out

    def quality_table(method: str, label: str, caption: str) -> List[str]:
        """Silhouette / Davies--Bouldin / Calinski--Harabasz for one speciation method."""
        out = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.2pt}",
            r"\begin{tabular}{@{}lrrrrrl@{}}",
            r"\toprule",
            (
                r"Distance & \thead{Threshold\\$t$} & Silhouette & "
                r"\thead{Davies--\\Bouldin} & \thead{Calinski--\\Harabasz} & "
                r"Separation & \thead{Structurally\\healthy} \\"
            ),
            r"\midrule",
        ]
        rows = [r for r in grid if r["method"] == method and r["distance"] in DIST_TEX]
        rows.sort(key=lambda r: (_dist_sort_key(r["distance"]), float(r["threshold"])))
        prev_dist = None
        for r in rows:
            c = _coh_key(coh, method, r["distance"], r["threshold"], r.get("min_samples"))
            mark = _dagger_mark(method, r)
            dist_tex = DIST_TEX[r["distance"]] + mark
            dist_show = "" if prev_dist == r["distance"] else dist_tex
            prev_dist = r["distance"]
            ok = "Yes" if c and str(c.get("structurally_healthy")) == "True" else "No"
            out.append(
                f"{dist_show} & {_fmt_f(r['threshold'], 2)} & "
                f"{_coh_f(c, 'silhouette_d_E', 3)} & {_coh_f(c, 'davies_bouldin', 3)} & "
                f"{_coh_f(c, 'calinski_harabasz', 1)} & {_coh_f(c, 'separation', 2)} & "
                f"{ok} \\\\"
            )
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        return out

    sep_caption_tail = (
        r"Within / Between are mean intra- / inter-cluster $d_E$ on the embedding canvas; "
        r"Separation $=$ Between/Within. Structurally healthy marks cells that pass collapse "
        r"and fragmentation filters. Recommended working-band cells are marked~$\dagger$."
    )
    quality_caption_tail = (
        r"Silhouette and Separation here use the held-out embedding canvas $d_E$ "
        r"(full cross-canvas Within / Between / Separation / silhouette appear in the "
        r"cross-canvas tables). Davies--Bouldin (lower better) and Calinski--Harabasz "
        r"(higher better) require feature vectors and are computed on MiniLM embeddings "
        r"only (not distance-canvas transferable). Structurally healthy matches the "
        r"corresponding threshold-grid table. Recommended working-band cells are marked~$\dagger$."
    )
    lines += grid_table(
        "leader_follower",
        f"tab:rq1-{batch_lower}-grid-lf",
        (
            f"Batch~{batch} full Leader--Follower threshold grid ($N{{=}}{n}$; join if $d<t$; "
            f"fitness-descending order). {sep_caption_tail}"
        ),
    )
    lines += grid_table(
        "incdbscan",
        f"tab:rq1-{batch_lower}-grid-inc",
        (
            f"Batch~{batch} full Incremental DBSCAN threshold grid ($N{{=}}{n}$; true incremental "
            r"seed on precomputed distance matrices; $\textit{min\_samples}=2$). "
            + sep_caption_tail
            + r" Oversized natural-language-inference components are marked~$\ddagger$."
        ),
    )
    lines += quality_table(
        "leader_follower",
        f"tab:rq1-{batch_lower}-quality-lf",
        (
            f"Batch~{batch} classic cluster-quality indices for Leader--Follower threshold-grid "
            f"cells ($N{{=}}{n}$). {quality_caption_tail}"
        ),
    )
    lines += quality_table(
        "incdbscan",
        f"tab:rq1-{batch_lower}-quality-inc",
        (
            f"Batch~{batch} classic cluster-quality indices for Incremental DBSCAN threshold-grid "
            f"cells ($N{{=}}{n}$; $\\textit{{min\\_samples}}=2$). {quality_caption_tail} "
            r"Oversized natural-language-inference components are marked~$\ddagger$."
        ),
    )

    def cross_canvas_metric_table(
        method: str,
        label: str,
        caption: str,
        *,
        metric_prefix: str,
        short_name: str,
        nd: int,
        higher_better: bool,
    ) -> List[str]:
        """One clustering metric under every scoring canvas for one speciation method."""
        col_heads = (
            rf"\thead{{{short_name}@$d_E$}} & \thead{{{short_name}@$d_{{EP}}$}} & "
            rf"\thead{{{short_name}@$d_N$}} & \thead{{{short_name}@$d_{{\neg E}}$}}"
        )
        out = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.0pt}",
            r"\begin{tabular}{@{}lrrrrrl@{}}",
            r"\toprule",
            (
                r"Forming & \thead{Threshold\\$t$} & "
                + col_heads
                + r" & \thead{Structurally\\healthy} \\"
            ),
            r"\midrule",
        ]
        rows = [r for r in grid if r["method"] == method and r["distance"] in DIST_TEX]
        rows.sort(key=lambda r: (_dist_sort_key(r["distance"]), float(r["threshold"])))
        prev_dist = None
        for r in rows:
            c = _coh_key(coh, method, r["distance"], r["threshold"], r.get("min_samples"))
            mark = _dagger_mark(method, r)
            dist_tex = DIST_TEX[r["distance"]] + mark
            dist_show = "" if prev_dist == r["distance"] else dist_tex
            prev_dist = r["distance"]
            ok = "Yes" if c and str(c.get("structurally_healthy")) == "True" else "No"
            out.append(
                f"{dist_show} & {_fmt_f(r['threshold'], 2)} & "
                f"{_coh_f(c, f'{metric_prefix}_on_d_E', nd)} & "
                f"{_coh_f(c, f'{metric_prefix}_on_d_EP', nd)} & "
                f"{_coh_f(c, f'{metric_prefix}_on_d_N', nd)} & "
                f"{_coh_f(c, f'{metric_prefix}_on_d_negE', nd)} & "
                f"{ok} \\\\"
            )
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        return out

    def _emit_cross_family(method: str, method_name: str, label_suffix: str, extra: str = "") -> None:
        if method == "incdbscan":
            head_scope = (
                f"Batch~{batch} cross-canvas {{what}} for {method_name} "
                f"($N{{=}}{n}$; $\\textit{{min\\_samples}}=2$). "
            )
        else:
            head_scope = f"Batch~{batch} cross-canvas {{what}} for {method_name} ($N{{=}}{n}$). "
        shared = (
            r"Each niche partition (forming distance) is rescored under all four canvases "
            r"$d_E$, $d_{EP}$, $d_N$, $d_{\neg E}$. "
            r"Diagonal entries (score under the forming canvas) can look strong by construction; "
            r"off-diagonal entries test transfer. Recommended working-band cells are marked~$\dagger$."
            + extra
        )
        specs = [
            ("within", "intra", "Within", 3, "within-cluster (intra) distances",
             "mean intra-cluster pairwise distance (lower tighter)"),
            ("between", "inter", "Between", 3, "between-cluster (inter) distances",
             "mean inter-cluster pairwise distance (higher more separated)"),
            ("sep", "separation", "Sep", 2, "Separation (inter/intra)",
             r"Separation $=$ Between/Within (higher better)"),
            ("sil", "silhouette", "Sil", 3, "silhouette",
             "silhouette on the precomputed canvas (higher better; negative implies mismatched assignment)"),
        ]
        for short, prefix, head, nd, what, gloss in specs:
            caption = (
                head_scope.replace("{what}", what)
                + shared
                + f" {head}@$d$: {gloss}."
            )
            label = f"tab:rq1-{batch_lower}-cross-{short}-{label_suffix}"
            lines.extend(
                cross_canvas_metric_table(
                    method,
                    label,
                    caption,
                    metric_prefix=prefix,
                    short_name=head,
                    nd=nd,
                    higher_better=True,
                )
            )

    _emit_cross_family("leader_follower", "Leader--Follower", "lf")
    _emit_cross_family(
        "incdbscan",
        "Incremental DBSCAN",
        "inc",
        extra=r" Oversized natural-language-inference components are marked~$\ddagger$.",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(lines)} lines)")


def main() -> int:
    p = argparse.ArgumentParser(description="Generate RQ1 LaTeX tables from sweeps")
    p.add_argument("--batch", required=True, help="e.g. B500 / B1000 / B2000")
    p.add_argument("--sweeps-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=None, help="Batch size (default: parse from --batch)")
    args = p.parse_args()
    batch = args.batch if args.batch.startswith("B") else f"B{args.batch}"
    n = args.n if args.n is not None else int(batch[1:])
    generate(batch, args.sweeps_dir, args.out, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
