#!/usr/bin/env python3
"""Triangle-inequality checks on the RQ1 archive (default: N=3139 only).

RQ1: B500 / B1000 / B2000 / B3139 matrices for d_E, d_negE, d_EP.
Optional --include-executions: also audit all 50 RQ2 runs for d_E / d_EP
(full pairwise d_negE is not stored online).

Writes notes/rough_draft/app_triangle.tex
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MATS = ROOT / "experiments/evostar_2027/data/mats"
RUNS = ROOT / "data/outputs/evostar2027"
OUT_TEX = ROOT / "notes/rough_draft/app_triangle.tex"

# Prefer underscore OpenAI category keys when both slash and underscore forms exist.
SCORE_KEYS = [
    "harassment",
    "harassment_threatening",
    "hate",
    "hate_threatening",
    "illicit",
    "illicit_violent",
    "self_harm",
    "self_harm_instructions",
    "self_harm_intent",
    "sexual",
    "sexual_minors",
    "violence",
    "violence_graphic",
]


def violation_stats(D: np.ndarray, n_triples: int, seed: int = 0) -> dict:
    n = D.shape[0]
    assert D.shape == (n, n)
    rng = np.random.default_rng(seed)
    n_triples = min(n_triples, comb(n, 3)) if n >= 3 else 0
    if n_triples == 0:
        return {
            "n": n,
            "n_triples": 0,
            "n_checks": 0,
            "n_viol": 0,
            "viol_rate": float("nan"),
            "mean_slack": 0.0,
            "max_slack": 0.0,
        }

    triples = set()
    while len(triples) < n_triples:
        a, b, c = rng.choice(n, size=3, replace=False)
        triples.add(tuple(sorted((int(a), int(b), int(c)))))

    checks = 0
    viol = 0
    slacks: list[float] = []
    for i, j, k in triples:
        for u, v, w in (
            (i, j, k),
            (i, k, j),
            (j, i, k),
            (j, k, i),
            (k, i, j),
            (k, j, i),
        ):
            lhs = float(D[u, w])
            rhs = float(D[u, v] + D[v, w])
            checks += 1
            slack = lhs - rhs
            if slack > 1e-12:
                viol += 1
                slacks.append(slack)

    return {
        "n": n,
        "n_triples": len(triples),
        "n_checks": checks,
        "n_viol": viol,
        "viol_rate": viol / checks if checks else float("nan"),
        "mean_slack": float(np.mean(slacks)) if slacks else 0.0,
        "max_slack": float(np.max(slacks)) if slacks else 0.0,
    }


def d_E_from_embeddings(emb: np.ndarray) -> np.ndarray:
    X = np.asarray(emb, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(nrm, 1e-12)
    sims = np.clip(X @ X.T, -1.0, 1.0)
    D = (1.0 - sims) / 2.0
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


def d_P_from_scores(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    sq = np.sum(x * x, axis=1, keepdims=True)
    d = np.sqrt(np.maximum(sq + sq.T - 2.0 * (x @ x.T), 0.0)) / np.sqrt(x.shape[1])
    np.fill_diagonal(d, 0.0)
    return d


def _scores_from_genome(g: dict) -> np.ndarray | None:
    mr = g.get("moderation_result") or {}
    openai = mr.get("openai") if isinstance(mr, dict) else None
    sc = openai.get("scores") if isinstance(openai, dict) else None
    if not isinstance(sc, dict):
        return None
    vals = []
    for k in SCORE_KEYS:
        if k in sc:
            vals.append(float(sc[k]))
        else:
            alt = k.replace("_", "/")
            if alt in sc:
                vals.append(float(sc[alt]))
            else:
                vals.append(0.0)
    return np.asarray(vals, dtype=np.float64)


_EMBEDDER = None


def _encode_prompts(prompts: list[str]) -> np.ndarray:
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return np.asarray(_EMBEDDER.encode(prompts, normalize_embeddings=False), dtype=np.float64)


def load_run_genomes(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Unique prompts from elites/archive/temp → embedding and score matrices."""
    by_prompt: dict[str, tuple[np.ndarray | None, np.ndarray]] = {}
    for fname in ("elites.json", "archive.json", "temp.json"):
        path = run_dir / fname
        if not path.is_file() or path.stat().st_size < 10:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for g in data:
            prompt = g.get("prompt")
            if not prompt:
                continue
            scores = _scores_from_genome(g)
            if scores is None:
                continue
            emb = g.get("prompt_embedding")
            emb_arr = np.asarray(emb, dtype=np.float64) if emb is not None else None
            by_prompt[prompt] = (emb_arr, scores)
    if not by_prompt:
        state_path = run_dir / "inc_dbscan_state.json"
        if state_path.is_file():
            st = json.loads(state_path.read_text(encoding="utf-8"))
            texts = st.get("texts") or []
            embs = st.get("embeddings") or []
            objs = st.get("objectives") or []
            for prompt, emb, obj in zip(texts, embs, objs):
                by_prompt[prompt] = (
                    np.asarray(emb, dtype=np.float64),
                    np.asarray(obj, dtype=np.float64),
                )
    if not by_prompt:
        raise RuntimeError(f"no genomes with scores in {run_dir}")

    missing = [p for p, (e, _) in by_prompt.items() if e is None]
    if missing:
        encoded = _encode_prompts(missing)
        for prompt, row in zip(missing, encoded):
            _, scores = by_prompt[prompt]
            by_prompt[prompt] = (row, scores)

    emb = np.vstack([v[0] for v in by_prompt.values()])
    scores = np.vstack([v[1] for v in by_prompt.values()])
    return emb, scores


def summarize_rates(rates: list[float], ns: list[int]) -> dict:
    r = np.asarray(rates, dtype=float)
    n = np.asarray(ns, dtype=float)
    return {
        "n_runs": len(rates),
        "n_med": float(np.median(n)),
        "n_min": float(np.min(n)),
        "n_max": float(np.max(n)),
        "rate_med": float(np.median(r)),
        "rate_min": float(np.min(r)),
        "rate_max": float(np.max(r)),
        "slack_note": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--triples", type=int, default=20000)
    ap.add_argument(
        "--include-executions",
        action="store_true",
        help="Also audit all 50 RQ2 runs for d_E / d_EP (slow; d_negE omitted).",
    )
    args = ap.parse_args()

    archive_rows = []
    # Report the full RQ1 archive in the table; smaller nested prefixes are checked
    # only for logging (same matrices, leading principal submatrices).
    for N, seed_off in ((500, 0), (1000, 1), (2000, 2), (3139, 3)):
        batch = MATS / f"B{N}"
        for j, (tex, fname) in enumerate(
            (
                (r"$d_E$", "d_E.npy"),
                (r"$d_{\neg E}$", "d_negE.npy"),
                (r"$d_{EP}$", "d_EP.npy"),
            )
        ):
            D = np.load(batch / fname)
            if D.shape[0] != N:
                raise SystemExit(f"{batch/fname}: expected N={N}, got {D.shape}")
            st = violation_stats(D, args.triples, seed=args.seed + 10 * seed_off + j)
            print(
                f"archive N={N:4d} {tex:12s} viol={st['viol_rate']:.4%} "
                f"checks={st['n_checks']}"
            )
            if N == 3139:
                archive_rows.append((tex, st))

    exec_summary = None
    if args.include_executions:
        configs = [
            "lf_embedding",
            "inc_embedding",
            "lf_nli",
            "inc_nli",
            "no_speciation",
        ]
        run_stats = {r"$d_E$": [], r"$d_{EP}$": []}
        run_ns = {r"$d_E$": [], r"$d_{EP}$": []}
        run_i = 0
        for cfg in configs:
            for run_dir in sorted((RUNS / cfg).glob("rep*_s*")):
                emb, scores = load_run_genomes(run_dir)
                mats = {
                    r"$d_E$": d_E_from_embeddings(emb),
                    r"$d_{EP}$": 0.7 * d_E_from_embeddings(emb)
                    + 0.3 * d_P_from_scores(scores),
                }
                for name, D in mats.items():
                    st = violation_stats(D, args.triples, seed=args.seed + 100 + run_i)
                    run_stats[name].append(st["viol_rate"])
                    run_ns[name].append(st["n"])
                    run_i += 1
                print(f"run {cfg}/{run_dir.name}: N={emb.shape[0]}")
        exec_summary = {
            k: summarize_rates(run_stats[k], run_ns[k]) for k in run_stats
        }

    if exec_summary is None:
        lines = [
            r"\section{Triangle-Inequality Checks}",
            r"\label{app:triangle}",
            r"",
            r"A distance is a metric only if it satisfies the triangle inequality: for any "
            r"three prompts $u$, $v$, and $w$,",
            r"\[",
            r"d(u,w)\le d(u,v)+d(v,w).",
            r"\]",
            r"In words, the direct dissimilarity between two prompts should never exceed the "
            r"sum of the dissimilarities along a path through a third prompt. Speciation in "
            r"this paper never uses that property. Leader--follower and IncDBSCAN only compare "
            r"a single pair against a threshold, so the operators remain valid even when the "
            r"inequality fails.",
            r"",
            r"Table~\ref{tab:app-triangle} reports checks on the RQ1 calibration archive "
            r"($N{=}3139$) with precomputed $d_E$, $d_{\neg E}$, and $d_{EP}$.",
            r"For every matrix we draw up to $20{,}000$ random triples and test all six "
            r"oriented inequalities $d(u,w)\le d(u,v)+d(v,w)$ (failure if the left-hand side "
            r"exceeds the right-hand side by more than $10^{-12}$).",
            r"",
            r"The columns are:",
            r"\begin{itemize}",
            r"\item \textbf{Checks:} oriented inequalities tested.",
            r"\item \textbf{Failed checks:} how many inequalities were false.",
            r"\item \textbf{Failure rate:} fraction of failed checks.",
            r"\item \textbf{Mean / largest excess:} among failures, the average / maximum of "
            r"$d(u,w)-d(u,v)-d(v,w)$.",
            r"\end{itemize}",
            r"Failures remain rare. This supports calling the functions dissimilarities "
            r"rather than metrics and does not change the speciation operators.",
            r"",
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Triangle-inequality checks on the RQ1 archive ($N{=}3139$). "
            r"Each matrix uses up to $20{,}000$ random triples / $120{,}000$ oriented checks.}",
            r"\label{tab:app-triangle}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrr@{}}",
            r"\toprule",
            r"Dissim. & $N$ & Checks & Failed & Failure rate & "
            r"Mean / max excess \\",
            r"\midrule",
        ]
        for tex, st in archive_rows:
            lines.append(
                f"{tex} & {st['n']} & {st['n_checks']:,} & "
                f"{st['n_viol']:,} & {100.0 * st['viol_rate']:.2f}\\% & "
                f"{st['mean_slack']:.4f} / {st['max_slack']:.4f} \\\\"
            )
    else:
        lines = [
            r"\section{Triangle-Inequality Checks}",
            r"\label{app:triangle}",
            r"",
            r"A distance is a metric only if it satisfies the triangle inequality: for any "
            r"three prompts $u$, $v$, and $w$,",
            r"\[",
            r"d(u,w)\le d(u,v)+d(v,w).",
            r"\]",
            r"In words, the direct dissimilarity between two prompts should never exceed the "
            r"sum of the dissimilarities along a path through a third prompt. Speciation in "
            r"this paper never uses that property. Leader--follower and IncDBSCAN only compare "
            r"a single pair against a threshold, so the operators remain valid even when the "
            r"inequality fails.",
            r"",
            r"Table~\ref{tab:app-triangle} reports checks on two sources of prompts.",
            r"\textbf{RQ1 archive.} The $N{=}3139$ calibration archive with precomputed "
            r"$d_E$, $d_{\neg E}$, and $d_{EP}$.",
            r"\textbf{Evolutionary executions.} All $50$ RQ2 runs (five configurations "
            r"$\times$ ten seeds). For each run we rebuild $d_E$ and $d_{EP}$ from the stored "
            r"prompt embeddings and moderation score vectors of that run's unique evaluated "
            r"prompts. Full pairwise $d_{\neg E}$ matrices were not retained online, so "
            r"NLI non-entailment is reported only for the RQ1 archive.",
            r"",
            r"For every matrix we draw up to $20{,}000$ random triples and test all six "
            r"oriented inequalities $d(u,w)\le d(u,v)+d(v,w)$ (failure if the left-hand side "
            r"exceeds the right-hand side by more than $10^{-12}$).",
            r"Archive rows report the failure rate on that matrix.",
            r"Execution rows report the median failure rate across the $50$ runs "
            r"(range in brackets).",
            r"",
            r"The columns are:",
            r"\begin{itemize}",
            r"\item \textbf{Checks:} oriented inequalities tested.",
            r"\item \textbf{Failed checks:} how many inequalities were false (RQ1 rows).",
            r"\item \textbf{Failure rate:} fraction of failed checks "
            r"(median [min, max] over runs for executions).",
            r"\item \textbf{Mean / largest excess:} among failures, the average / maximum of "
            r"$d(u,w)-d(u,v)-d(v,w)$ (RQ1 rows).",
            r"\end{itemize}",
            r"Failures remain rare in both the calibration archive and the evolutionary "
            r"executions. This supports calling the functions dissimilarities rather than "
            r"metrics and does not change the speciation operators.",
            r"",
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Triangle-inequality checks on the RQ1 archive ($N{=}3139$) and all "
            r"$50$ RQ2 evolutionary runs. "
            r"Execution rows give the median failure rate over runs (range in brackets). "
            r"Full pairwise $d_{\neg E}$ is available only for the RQ1 archive. "
            r"Each matrix uses up to $20{,}000$ random triples / $120{,}000$ oriented checks.}",
            r"\label{tab:app-triangle}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{@{}llrrrrr@{}}",
            r"\toprule",
            r"Source & Dissim. & $N$ & Checks & Failed & Failure rate & "
            r"Mean / max excess \\",
            r"\midrule",
        ]
        for tex, st in archive_rows:
            lines.append(
                f"RQ1 archive & {tex} & {st['n']} & {st['n_checks']:,} & "
                f"{st['n_viol']:,} & {100.0 * st['viol_rate']:.2f}\\% & "
                f"{st['mean_slack']:.4f} / {st['max_slack']:.4f} \\\\"
            )
        lines.append(r"\midrule")
        for tex in (r"$d_E$", r"$d_{EP}$"):
            s = exec_summary[tex]
            n_med = int(round(s["n_med"]))
            checks_typ = 6 * min(args.triples, comb(n_med, 3))
            lines.append(
                f"RQ2 executions & {tex} & "
                f"{n_med} & "
                f"{checks_typ:,} & -- & "
                f"{100.0 * s['rate_med']:.2f}\\% "
                f"[{100.0 * s['rate_min']:.2f},\\,{100.0 * s['rate_max']:.2f}] & "
                f"-- \\\\"
            )
        lines.append(
            r"RQ2 executions & $d_{\neg E}$ & -- & -- & -- & "
            r"not retained online & -- \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        r"",
    ]
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_TEX}")
    if exec_summary is not None:
        for tex, s in exec_summary.items():
            print(
                f"exec {tex}: median rate={100*s['rate_med']:.3f}% "
                f"range=[{100*s['rate_min']:.3f},{100*s['rate_max']:.3f}] "
                f"N_med={s['n_med']:.0f}"
            )


if __name__ == "__main__":
    main()
