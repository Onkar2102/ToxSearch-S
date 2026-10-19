#!/usr/bin/env python3
"""After B1000/B2000 sweeps exist: update RQ1 notebook + paper prose, rebuild PDF.

Idempotent: re-running replaces the cross-batch notebook section and regenerates
cross-batch / B1000 / B2000 table includes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
EVO = ROOT / "experiments" / "evostar_2027"
RQ1 = EVO / "rq1"
NOTES = ROOT / "notes" / "LaTeX2e+Proceedings+Templates+download"
NB_PATH = RQ1 / "notebooks" / "B500_rq1_summary.ipynb"
EXP_TEX = NOTES / "04-experimentation.tex"
CROSS_TEX = NOTES / "rq1_cross_batch_tables.tex"
MARKER = "<!-- CROSS_BATCH_SECTION -->"
END_MARKER = "<!-- /CROSS_BATCH_SECTION -->"


def _load_sw(batch: str) -> Dict[str, Any]:
    sw = RQ1 / "sweeps" / batch
    out: Dict[str, Any] = {"batch": batch, "sw": sw}
    out["calib"] = pd.read_csv(sw / "calibration.csv")
    out["struct"] = pd.read_csv(sw / "structure.csv")
    out["grid"] = pd.read_csv(sw / "threshold_grid.csv")
    out["coh"] = pd.read_csv(sw / "coherence.csv")
    return out


def _cell_metrics(coh: pd.DataFrame, method: str, dist: str, t: float) -> Optional[pd.Series]:
    m = coh[(coh["method"] == method) & (coh["distance"] == dist)]
    m = m[abs(m["threshold"].astype(float) - t) < 1e-9]
    if method == "incdbscan" and "min_samples" in m.columns:
        m = m[m["min_samples"].astype(float) == 2]
    if m.empty:
        return None
    return m.iloc[0]


def _working_band_summary(data: Dict[str, Any]) -> List[str]:
    """Bullet lines for working-band cells at t=0.10 and 0.15 on d_E/d_EP."""
    lines = []
    coh = data["coh"]
    grid = data["grid"]
    n = int(data["calib"]["N"].iloc[0])
    for method, mlabel in [("leader_follower", "LF"), ("incdbscan", "IncDBSCAN")]:
        for dist in ("d_E", "d_EP"):
            for t in (0.10, 0.15):
                g = grid[
                    (grid["method"] == method)
                    & (grid["distance"] == dist)
                    & (abs(grid["threshold"].astype(float) - t) < 1e-9)
                ]
                c = _cell_metrics(coh, method, dist, t)
                if g.empty:
                    continue
                r = g.iloc[0]
                max_c = int(float(r["max_cluster"]))
                if method == "leader_follower":
                    niches = int(float(r["n_mature"]))
                else:
                    niches = int(float(r["n_clusters"]))
                sep = float(c["separation"]) if c is not None and pd.notna(c.get("separation")) else float("nan")
                healthy = bool(c["structurally_healthy"]) if c is not None else False
                lines.append(
                    f"- **{data['batch']}** {mlabel}×`{dist}` @ t={t:.2f}: "
                    f"niches/clusters={niches}, max={max_c}/{n}, "
                    f"sep={sep:.2f}, healthy={healthy}"
                )
    return lines


def _nli_collapse_note(data: Dict[str, Any]) -> str:
    grid = data["grid"]
    n = int(data["calib"]["N"].iloc[0])
    rows = grid[
        (grid["method"] == "incdbscan")
        & (grid["distance"].isin(["d_N", "d_negE"]))
        & (grid["threshold"].astype(float) <= 0.15 + 1e-9)
    ]
    if rows.empty:
        return f"- **{data['batch']}**: no NLI+IncDBSCAN rows at t≤0.15"
    max_frac = float(rows["max_cluster"].astype(float).max()) / n
    return (
        f"- **{data['batch']}**: NLI+IncDBSCAN at t≤0.15 max component "
        f"up to {max_frac:.0%} of N (collapse if ≥50%)"
    )


def _eps_row(data: Dict[str, Any], dist: str) -> Tuple[float, float, int]:
    calib = data["calib"]
    struct = data["struct"]
    c = calib[calib["distance"] == dist].iloc[0]
    s = struct[(struct["distance"] == dist) & (struct["min_samples"] == 2)].iloc[0]
    return float(c["eps"]), float(s["max_cluster"]), int(s["n_clusters"])


def build_cross_batch_md(batches: List[Dict[str, Any]]) -> str:
    wb = []
    nli = []
    calib_lines = []
    for d in batches:
        wb.extend(_working_band_summary(d))
        nli.append(_nli_collapse_note(d))
        for dist in ("d_E", "d_EP", "d_N", "d_negE"):
            if dist not in set(d["calib"]["distance"]):
                continue
            eps, mx, nc = _eps_row(d, dist)
            calib_lines.append(
                f"| {d['batch']} | `{dist}` | {eps:.4f} | {nc} | {int(mx)} |"
            )

    parts = [
        MARKER,
        "# Cross-batch (B500 / B1000 / B2000)",
        "",
        "Same offline protocol as B500: neighbor-fraction calibration vs $d_E$ at "
        "radius **0.1**, LF + true IncDBSCAN `seed_from_precomputed_matrix`, "
        "threshold grid $t\\in\\{0.10,\\ldots,0.50\\}$, canvas Separation on $d_E$.",
        "",
        "## Loaders",
        "",
    ]
    return "\n".join(parts)


def build_cross_batch_code() -> str:
    return '''from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BATCHES = ["B500", "B1000", "B2000"]
SWEEPS = {b: RQ1 / "sweeps" / b for b in BATCHES}
for b, p in SWEEPS.items():
    need = ["calibration.csv", "structure.csv", "threshold_grid.csv", "coherence.csv"]
    missing = [f for f in need if not (p / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{b}: missing {missing} under {p}")

calib = pd.concat(
    [pd.read_csv(SWEEPS[b] / "calibration.csv").assign(batch=b) for b in BATCHES],
    ignore_index=True,
)
struct = pd.concat(
    [pd.read_csv(SWEEPS[b] / "structure.csv").assign(batch=b) for b in BATCHES],
    ignore_index=True,
)
grid = pd.concat(
    [pd.read_csv(SWEEPS[b] / "threshold_grid.csv").assign(batch=b) for b in BATCHES],
    ignore_index=True,
)
coh = pd.concat(
    [pd.read_csv(SWEEPS[b] / "coherence.csv").assign(batch=b) for b in BATCHES],
    ignore_index=True,
)

print("Loaded cross-batch sweeps:")
for b in BATCHES:
    n = int(calib.loc[calib["batch"] == b, "N"].iloc[0])
    print(f"  {b}: N={n}  grid_rows={len(grid[grid.batch==b])}  coh_rows={len(coh[coh.batch==b])}")

# Calibration eps comparison
fig, ax = plt.subplots(figsize=(9, 4))
dists = ["d_E", "d_EP", "d_N", "d_negE"]
x = np.arange(len(dists))
w = 0.25
for i, b in enumerate(BATCHES):
    ys = []
    for d in dists:
        row = calib[(calib.batch == b) & (calib.distance == d)]
        ys.append(float(row["eps"].iloc[0]) if len(row) else np.nan)
    ax.bar(x + (i - 1) * w, ys, w, label=b)
ax.set_xticks(x)
ax.set_xticklabels([LABEL[d] for d in dists])
ax.set_ylabel(r"calibrated $\\varepsilon$")
ax.set_title("Neighbor-fraction calibration vs N")
ax.legend()
plt.tight_layout()
plt.show()

# Structure at calibrated eps (min_samples=2)
ms2 = struct[struct["min_samples"] == 2]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, col, title in zip(
    axes,
    ["n_clusters", "noise_frac", "max_cluster"],
    ["#clusters", "noise frac", "max cluster"],
):
    for b in BATCHES:
        sub = ms2[ms2.batch == b].set_index("distance").reindex(dists)
        ax.plot(range(len(dists)), sub[col], marker="o", label=b)
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels([LABEL[d] for d in dists], rotation=20)
    ax.set_title(title)
axes[0].legend(fontsize=8)
plt.suptitle("IncDBSCAN structure at calibrated ε (min_samples=2)", y=1.02)
plt.tight_layout()
plt.show()

# LF vs IncDBSCAN curves at shared t (mature/clusters + max size)
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for ax, dist in zip(axes.ravel(), dists):
    for b, ls in zip(BATCHES, ["-", "--", ":"]):
        for method, color in [("leader_follower", "#1f77b4"), ("incdbscan", "#ff7f0e")]:
            sub = grid[(grid.batch == b) & (grid.distance == dist) & (grid.method == method)].sort_values("threshold")
            y = sub["n_mature"] if method == "leader_follower" else sub["n_clusters"]
            ax.plot(sub["threshold"], y, ls=ls, color=color, alpha=0.85,
                    label=f"{b} {'LF' if method=='leader_follower' else 'Inc'}")
    ax.set_title(LABEL[dist])
    ax.set_ylabel("#mature / #clusters")
axes[0, 0].legend(fontsize=7, ncol=2)
axes[1, 0].set_xlabel("t")
axes[1, 1].set_xlabel("t")
plt.suptitle("Structure vs threshold across N", y=1.01)
plt.tight_layout()
plt.show()

# Coherence: top healthy separation by batch
healthy = coh[coh["structurally_healthy"] == True].copy()
fig, ax = plt.subplots(figsize=(9, 4.2))
for b, marker in zip(BATCHES, ["o", "s", "^"]):
    sub = healthy[healthy.batch == b]
    for method, c in [("leader_follower", "#1f77b4"), ("incdbscan", "#ff7f0e")]:
        s2 = sub[sub.method == method]
        ax.scatter(s2["threshold"], s2["separation"], c=c, marker=marker, alpha=0.7,
                   label=f"{b}/{method[:2]}")
ax.set_xlabel("t")
ax.set_ylabel("separation (inter/intra $d_E$)")
ax.set_title("Healthy cells: separation vs t (all batches)")
ax.legend(fontsize=7, ncol=3)
plt.tight_layout()
plt.show()

display(calib.pivot_table(index="distance", columns="batch", values="eps"))
display(
    ms2.pivot_table(index="distance", columns="batch", values=["n_clusters", "max_cluster"])
)
'''


def build_conclusion_md(batches: List[Dict[str, Any]]) -> str:
    wb_lines = []
    nli_lines = []
    for d in batches:
        wb_lines.extend(_working_band_summary(d))
        nli_lines.append(_nli_collapse_note(d))

    # Stability: does t≈0.10–0.15 on d_E/d_EP stay healthy?
    stable = True
    details = []
    for d in batches:
        for method in ("leader_follower", "incdbscan"):
            for dist in ("d_E", "d_EP"):
                for t in (0.10, 0.15):
                    c = _cell_metrics(d["coh"], method, dist, t)
                    ok = bool(c["structurally_healthy"]) if c is not None else False
                    if not ok:
                        stable = False
                    details.append(f"{d['batch']} {method} {dist} t={t:.2f} healthy={ok}")

    verdict = (
        f"Working band **$t\\approx0.10$–$0.15$ on $d_E$/$d_{{EP}}$** "
        + ("**holds** across B500/B1000/B2000." if stable else "is **mixed** across N — see per-cell flags.")
    )
    return "\n".join(
        [
            "## Cross-batch conclusions",
            "",
            verdict,
            "",
            "### Working-band cells (t=0.10 / 0.15)",
            "",
            *wb_lines,
            "",
            "### NLI + IncDBSCAN collapse check",
            "",
            *nli_lines,
            "",
            "### Stability checklist",
            "",
            *[f"- `{x}`" for x in details],
            "",
            END_MARKER,
        ]
    )


def update_notebook(batches: List[Dict[str, Any]]) -> None:
    # Notebook lives under notebooks/ (often gitignored) — skip on cluster if absent.
    if not NB_PATH.is_file():
        print(f"[SKIP] notebook missing (not fatal): {NB_PATH}")
        return
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    # Drop prior cross-batch section if present
    new_cells = []
    skipping = False
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if MARKER in src:
            skipping = True
            continue
        if skipping:
            if END_MARKER in src:
                skipping = False
            continue
        new_cells.append(c)
    nb["cells"] = new_cells

    md1 = build_cross_batch_md(batches)
    code = build_cross_batch_code()
    md2 = build_conclusion_md(batches)

    def md_cell(text: str) -> Dict[str, Any]:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")],
        }

    def code_cell(text: str) -> Dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.split("\n")],
        }

    # Insert before Part 2 / Summary if present; else append
    insert_at = len(nb["cells"])
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if src.startswith("# Part 2") or src.startswith("# Summary"):
            insert_at = i
            break

    block = [md_cell(md1), code_cell(code), md_cell(md2)]
    nb["cells"] = nb["cells"][:insert_at] + block + nb["cells"][insert_at:]
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated notebook {NB_PATH} (+{len(block)} cells at index {insert_at})")


def write_cross_batch_tex(batches: List[Dict[str, Any]]) -> None:
    lines = [
        "% Auto-generated cross-batch RQ1 summary — do not hand-edit.",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Cross-batch RQ1 calibration and calibrated Incremental DBSCAN structure "
        r"($d_E$ reference radius $0.1$; $\textit{min\_samples}=2$).}",
        r"\label{tab:rq1-cross-calib-structure}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Batch & Distance & $\varepsilon$ & \thead{Number of\\clusters} & "
        r"\thead{Maximum\\cluster size} & \thead{Noise\\fraction} \\",
        r"\midrule",
    ]
    dist_tex = {
        "d_E": r"$d_E$",
        "d_EP": r"$d_{EP}$",
        "d_N": r"$d_N$",
        "d_negE": r"$d_{\neg E}$",
    }
    for d in batches:
        struct = d["struct"]
        calib = d["calib"]
        for dist in ("d_E", "d_EP", "d_N", "d_negE"):
            c = calib[calib["distance"] == dist]
            s = struct[(struct["distance"] == dist) & (struct["min_samples"] == 2)]
            if c.empty or s.empty:
                continue
            lines.append(
                f"{d['batch']} & {dist_tex[dist]} & {float(c.iloc[0]['eps']):.4f} & "
                f"{int(s.iloc[0]['n_clusters'])} & {int(float(s.iloc[0]['max_cluster']))} & "
                f"{float(s.iloc[0]['noise_frac']):.3f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Working-band snapshot at t=0.10
    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Cross-batch working-band snapshot at $t{=}0.10$ on $d_E$/$d_{EP}$ "
        r"(Leader--Follower mature niches; Incremental DBSCAN clusters; Separation on embedding canvas).}",
        r"\label{tab:rq1-cross-working-band}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\begin{tabular}{@{}lllrrrr@{}}",
        r"\toprule",
        r"Batch & Method & Distance & Niches/clusters & Max size & Separation & Healthy \\",
        r"\midrule",
    ]
    for d in batches:
        n = int(d["calib"]["N"].iloc[0])
        for method, mlabel in [("leader_follower", "LF"), ("incdbscan", "IncDBSCAN")]:
            for dist in ("d_E", "d_EP"):
                g = d["grid"][
                    (d["grid"]["method"] == method)
                    & (d["grid"]["distance"] == dist)
                    & (abs(d["grid"]["threshold"].astype(float) - 0.10) < 1e-9)
                ]
                c = _cell_metrics(d["coh"], method, dist, 0.10)
                if g.empty:
                    continue
                r = g.iloc[0]
                niches = int(float(r["n_mature" if method == "leader_follower" else "n_clusters"]))
                mx = int(float(r["max_cluster"]))
                sep = float(c["separation"]) if c is not None and pd.notna(c.get("separation")) else float("nan")
                healthy = "Yes" if c is not None and bool(c["structurally_healthy"]) else "No"
                sep_s = f"{sep:.2f}" if sep == sep else "---"
                lines.append(
                    f"{d['batch']} & {mlabel} & {dist_tex[dist]} & {niches} & "
                    f"{mx}/{n} & {sep_s} & {healthy} \\\\"
                )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    CROSS_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {CROSS_TEX}")


def update_experimentation_tex(batches: List[Dict[str, Any]]) -> None:
    text = EXP_TEX.read_text(encoding="utf-8")
    # Ensure inputs for B1000/B2000 + cross tables exist after B500 input
    for name in ("rq1_cross_batch_tables", "rq1_b1000_tables", "rq1_b2000_tables"):
        needle = f"\\input{{{name}}}"
        if needle not in text:
            text = text.replace(
                "\\input{rq1_b500_tables}",
                "\\input{rq1_b500_tables}\n" + needle,
            )

    # Build / replace cross-N paragraph
    # Stability verdict
    stable = True
    for d in batches:
        for method in ("leader_follower", "incdbscan"):
            for dist in ("d_E", "d_EP"):
                for t in (0.10, 0.15):
                    c = _cell_metrics(d["coh"], method, dist, t)
                    if c is None or not bool(c["structurally_healthy"]):
                        stable = False

    nli_bits = []
    for d in batches:
        n = int(d["calib"]["N"].iloc[0])
        rows = d["grid"][
            (d["grid"]["method"] == "incdbscan")
            & (d["grid"]["distance"].isin(["d_N", "d_negE"]))
            & (d["grid"]["threshold"].astype(float) <= 0.15 + 1e-9)
        ]
        if not rows.empty:
            frac = float(rows["max_cluster"].astype(float).max()) / n
            nli_bits.append(f"{d['batch']} up to {frac:.0%}")

    hold = (
        "still holds at $N{=}1000$ and $N{=}2000$"
        if stable
        else "is only partially stable at larger $N$ (see per-batch healthy flags)"
    )
    para = (
        r"\paragraph{Cross-batch scale-up (B500 / B1000 / B2000).}" + "\n"
        "We repeated the same offline protocol on the nested fitness-top batches "
        r"B1000 and B2000 (NLI matrices taken from the completed $N{=}2000$ flat mats, "
        "with B1000 as the leading $1000\\times1000$ slice). "
        "Table~\\ref{tab:rq1-cross-calib-structure} compares calibrated radii and "
        "Incremental DBSCAN structure across $N$; "
        "Table~\\ref{tab:rq1-cross-working-band} snapshots the $t{=}0.10$ working band. "
        "Full Leader--Follower and Incremental DBSCAN grids for B1000 and B2000 appear in "
        "Tables~\\ref{tab:rq1-b1000-grid-lf}--\\ref{tab:rq1-b2000-grid-inc}. "
        f"The B500 finding that $t\\approx0.10$--$0.15$ on $d_E$/$d_{{EP}}$ forms a usable "
        f"topic-niche band {hold}. "
        "NLI+IncDBSCAN continues to collapse into oversized components at these radii "
        f"({'; '.join(nli_bits)}). "
        "As with B500, these offline geometry results do not finalize the RQ1 pilot shortlist."
    )

    start = r"\paragraph{Cross-batch scale-up"
    if start in text:
        # replace existing paragraph through next \paragraph or \subsection or \input
        i = text.index(start)
        rest = text[i:]
        # end at next structural command after first newline block
        end_candidates = []
        for tok in (r"\paragraph{", r"\subsection{", r"\input{", r"\emph{Caveat:}"):
            j = rest.find(tok, 1)
            if j != -1:
                end_candidates.append(j)
        j = min(end_candidates) if end_candidates else len(rest)
        # if we hit Caveat that belongs to B500 para, don't eat it when replacing cross para only
        text = text[:i] + para + "\n\n" + rest[j:]
    else:
        # insert after B500 caveat / before RQ2
        anchor = r"\subsection{RQ2:"
        if anchor not in text:
            raise SystemExit("Cannot find RQ2 anchor in 04-experimentation.tex")
        # place before RQ2, after inputs
        text = text.replace(anchor, para + "\n\n" + anchor)

    EXP_TEX.write_text(text, encoding="utf-8")
    print(f"Updated {EXP_TEX}")


def rebuild_pdf() -> None:
    tex = "main.tex"
    # Prefer paper.tex if that is the built artifact
    if (NOTES / "paper.tex").is_file():
        # check which was used previously
        pass
    main = NOTES / "main.tex"
    cmd = ["pdflatex", "-interaction=nonstopmode", "-jobname=paper", "main.tex"]
    for _ in range(2):
        r = subprocess.run(cmd, cwd=str(NOTES), capture_output=True, text=True)
        print(f"pdflatex exit={r.returncode}")
    pdf = NOTES / "paper.pdf"
    if pdf.is_file():
        print(f"Rebuilt {pdf} ({pdf.stat().st_size} bytes)")
    else:
        # fallback: main.pdf
        alt = NOTES / "main.pdf"
        print(f"paper.pdf missing; main.pdf exists={alt.is_file()}")
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-1000:])


def main() -> int:
    needed = ["B500", "B1000", "B2000"]
    for b in needed:
        sw = RQ1 / "sweeps" / b
        for f in ("calibration.csv", "structure.csv", "threshold_grid.csv", "coherence.csv"):
            if not (sw / f).is_file():
                print(f"ERROR: missing {sw / f}", file=sys.stderr)
                return 1

    # Ensure per-batch LaTeX tables exist
    gen = ROOT / "experiments/evostar_2027/rq1/scripts/p1d_generate_rq1_tables.py"
    for b, n in [("B1000", 1000), ("B2000", 2000)]:
        out = NOTES / f"rq1_{b.lower()}_tables.tex"
        subprocess.check_call(
            [
                sys.executable,
                str(gen),
                "--batch",
                b,
                "--sweeps-dir",
                str(RQ1 / "sweeps" / b),
                "--out",
                str(out),
                "--n",
                str(n),
            ]
        )

    batches = [_load_sw(b) for b in needed]
    write_cross_batch_tex(batches)
    update_notebook(batches)
    update_experimentation_tex(batches)
    rebuild_pdf()
    (RQ1 / "sweeps" / "CROSS_BATCH_FINALIZED.json").write_text(
        json.dumps({"ok": True, "batches": needed}, indent=2),
        encoding="utf-8",
    )
    print("FINALIZED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
