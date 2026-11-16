#!/usr/bin/env python3
"""Prompt perplexity for RQ2 genomes with refusal-penalized fitness > 0.6.

The generation file Meta-Llama-3.1-8B-Instruct.Q8_0.gguf is not on disk.
Scores use the same model's Q5_K_M GGUF, the finest quantization present.
No chat template is applied: perplexity is of the raw prompt text.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq2_metrics import illicit_of, load_json  # noqa: E402

LIVE = ROOT / "data" / "outputs" / "evostar2027"
REPORTS = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports"
MODEL = ROOT / "models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf"
HIGH_CSV = REPORTS / "rq2_response_duration_gt06.csv"
CHECKPOINT = REPORTS / "rq2_prompt_perplexity_checkpoint.csv"

N_CTX = 4096
N_BATCH = 512
STRIDE = 2048

CELLS = [
    ("lf_embedding", "leader_follower", "MiniLM LF"),
    ("inc_embedding", "dbscan", "MiniLM Inc"),
    ("lf_nli", "leader_follower", "NLI LF"),
    ("inc_nli", "dbscan", "NLI Inc"),
    ("no_speciation", "none", "Global EA"),
]
ORDER = [label for _, _, label in CELLS]


def pool_names(clustering: str) -> Tuple[str, ...]:
    if clustering == "dbscan":
        return ("temp.json",)
    return ("elites.json", "reserves.json", "archive.json")


def load_wanted() -> Dict[Tuple[str, int, str], dict]:
    wanted = {}
    with HIGH_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wanted[(row["cell"], int(row["seed"]), row["genome_id"])] = row
    return wanted


def collect_prompts(wanted: Dict[Tuple[str, int, str], dict]) -> List[dict]:
    found: Dict[Tuple[str, int, str], dict] = {}
    for cell, clustering, label in CELLS:
        for seed in range(10):
            run_dir = LIVE / cell / f"rep{seed}_s{seed}"
            seen = set()
            for name in pool_names(clustering):
                data = load_json(run_dir / name)
                if not isinstance(data, list):
                    continue
                for rec in data:
                    if not isinstance(rec, dict):
                        continue
                    gid = str(rec.get("id"))
                    if gid in seen:
                        continue
                    seen.add(gid)
                    key = (cell, seed, gid)
                    if key not in wanted:
                        continue
                    fit = illicit_of(rec)
                    found[key] = {
                        "cell": cell,
                        "label": label,
                        "seed": seed,
                        "genome_id": gid,
                        "fitness": fit if fit is not None else float(wanted[key]["fitness"]),
                        "prompt_chars": len(rec.get("prompt") or ""),
                        "prompt": rec.get("prompt") or "",
                    }
            print(f"[load] {label} seed {seed}", flush=True)
    missing = [k for k in wanted if k not in found]
    if missing:
        raise SystemExit(f"{len(missing)} fitness>0.6 genomes were not found in the population files")
    rows = list(found.values())
    rows.sort(key=lambda r: (ORDER.index(r["label"]), r["seed"], int(r["genome_id"]) if str(r["genome_id"]).isdigit() else 0))
    return rows


def token_nll(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Negative log-likelihood of ``targets`` under each row of ``logits``."""
    m = logits.max(axis=1)
    shifted = logits - m[:, None]
    logsumexp = m + np.log(np.exp(shifted).sum(axis=1))
    chosen = logits[np.arange(targets.shape[0]), targets]
    return logsumexp - chosen


def window_nll(llm, tokens: Sequence[int], score_from: int) -> List[float]:
    """Teacher-forced NLL of tokens[score_from:] given the preceding tokens in this window."""
    llm._ctx.kv_cache_seq_rm(-1, 0, -1)
    n_past = 0
    nlls: List[float] = []
    n_vocab = llm._n_vocab
    seq = list(tokens)
    while n_past < len(seq):
        batch = seq[n_past : n_past + N_BATCH]
        llm._batch.set_batch(batch, n_past, logits_all=True)
        llm._ctx.decode(llm._batch)
        rows = len(batch)
        logits = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(rows, n_vocab))
        # Row j predicts the token at absolute index n_past + j + 1.
        pred_lo = n_past + 1
        pred_hi = n_past + rows + 1
        take_lo = max(score_from, pred_lo)
        take_hi = min(len(seq), pred_hi)
        if take_lo < take_hi:
            local = np.arange(take_lo, take_hi) - n_past - 1
            targets = np.asarray(seq[take_lo:take_hi], dtype=np.int64)
            nlls.extend(token_nll(np.asarray(logits[local], dtype=np.float64), targets).tolist())
        n_past += rows
    return nlls


def prompt_perplexity(llm, text: str) -> Tuple[float, int, int]:
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=True, special=False)
    n_tokens = len(tokens)
    if n_tokens < 2:
        return float("nan"), n_tokens, 0
    nlls: List[float] = []
    windows = 0
    if n_tokens <= N_CTX:
        nlls.extend(window_nll(llm, tokens, score_from=1))
        windows = 1
    else:
        start = 0
        scored_through = 0
        while scored_through < n_tokens - 1:
            end = min(start + N_CTX, n_tokens)
            local_from = max(scored_through + 1, start + (0 if start == 0 else 1)) - start
            if start == 0:
                local_from = 1
            else:
                local_from = (scored_through + 1) - start
            nlls.extend(window_nll(llm, tokens[start:end], score_from=local_from))
            windows += 1
            scored_through = end - 1
            if end == n_tokens:
                break
            start += STRIDE
            if start >= n_tokens - 1:
                break
    if not nlls:
        return float("nan"), n_tokens, windows
    return float(np.exp(np.mean(nlls))), n_tokens, windows


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    a, b = x[mask], y[mask]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    if method == "spearman":
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def summarize(rows: Sequence[dict]) -> dict:
    ppl = np.asarray([r["perplexity"] for r in rows], dtype=float)
    ppl = ppl[np.isfinite(ppl)]
    if ppl.size == 0:
        return {}
    q25, med, q75 = np.percentile(ppl, [25, 50, 75])
    fit = np.asarray([r["fitness"] for r in rows], dtype=float)
    chars = np.asarray([r["prompt_chars"] for r in rows], dtype=float)
    toks = np.asarray([r["n_tokens"] for r in rows], dtype=float)
    ppl_all = np.asarray([r["perplexity"] for r in rows], dtype=float)
    return {
        "n": int(ppl.size),
        "mean": float(ppl.mean()),
        "median": float(med),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(ppl.min()),
        "max": float(ppl.max()),
        "pearson_fitness": corr(ppl_all, fit, "pearson"),
        "spearman_fitness": corr(ppl_all, fit, "spearman"),
        "pearson_chars": corr(ppl_all, chars, "pearson"),
        "spearman_chars": corr(ppl_all, chars, "spearman"),
        "pearson_tokens": corr(ppl_all, toks, "pearson"),
        "spearman_tokens": corr(ppl_all, toks, "spearman"),
        "n_multi_window": sum(1 for r in rows if r["n_windows"] > 1),
    }


def write_outputs(rows: List[dict]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    per_path = REPORTS / "rq2_prompt_perplexity_gt06.csv"
    fields = ["cell", "label", "seed", "genome_id", "fitness", "prompt_chars", "n_tokens", "n_windows", "perplexity"]
    with per_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    groups = [("Overall", rows)] + [(label, [r for r in rows if r["label"] == label]) for label in ORDER]
    groups = [(name, subset) for name, subset in groups if subset]
    summaries = []
    for name, subset in groups:
        stats = summarize(subset)
        stats["configuration"] = name
        summaries.append(stats)
    sum_fields = [
        "configuration", "n", "mean", "median", "q25", "q75", "iqr", "min", "max",
        "pearson_fitness", "spearman_fitness", "pearson_chars", "spearman_chars",
        "pearson_tokens", "spearman_tokens", "n_multi_window",
    ]
    with (REPORTS / "rq2_prompt_perplexity_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        for stats in summaries:
            w.writerow({k: stats.get(k, "") for k in sum_fields})

    finite = np.asarray([r["perplexity"] for r in rows if np.isfinite(r["perplexity"])], dtype=float)
    lo, hi = np.percentile(finite, [1, 99])
    edges = np.linspace(lo, hi, 16)
    hist_rows = []
    with (REPORTS / "rq2_prompt_perplexity_hist.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["configuration", "bin_lo", "bin_hi", "count"])
        w.writeheader()
        for name, subset in groups:
            vals = [r["perplexity"] for r in subset if np.isfinite(r["perplexity"])]
            counts, _ = np.histogram(vals, bins=edges)
            for a, b, c in zip(edges[:-1], edges[1:], counts):
                rec = {"configuration": name, "bin_lo": float(a), "bin_hi": float(b), "count": int(c)}
                w.writerow(rec)
                hist_rows.append(rec)

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), sharex=True)
    palette = {
        "Overall": "#333333",
        "MiniLM LF": "#0F4D92",
        "MiniLM Inc": "#2A9D8F",
        "NLI LF": "#B64342",
        "NLI Inc": "#E9C46A",
        "Global EA": "#4D4D4D",
    }
    for ax, (name, subset) in zip(axes.ravel(), groups):
        vals = [r["perplexity"] for r in subset if np.isfinite(r["perplexity"])]
        ax.hist(vals, bins=edges, color=palette[name], edgecolor="white", linewidth=0.4)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Prompt perplexity")
        ax.set_ylabel("Prompts")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Refusal-penalized fitness > 0.6", fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORTS / "rq2_prompt_perplexity_hist.png", dpi=160)
    fig.savefig(REPORTS / "rq2_prompt_perplexity_hist.pdf")
    plt.close(fig)

    def fmt(x: float, nd: int = 2) -> str:
        if x != x:
            return "—"
        return f"{x:.{nd}f}"

    lines = [
        "# Prompt perplexity for fitness greater than 0.6",
        "",
        "Perplexity of the raw prompt text under Meta-Llama-3.1-8B-Instruct.",
        "The runs were generated with `Meta-Llama-3.1-8B-Instruct.Q8_0.gguf`.",
        "That file is not on disk, so these scores use `Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf`, the finest copy of the same model that is present.",
        "No other model family was substituted, and no chat template was applied.",
        "",
        "Each prompt is tokenized with a beginning-of-sequence token.",
        "That token is context only.",
        "Perplexity is exp(mean negative log-likelihood) of the remaining tokens.",
        f"Prompts longer than {N_CTX} tokens are scored in windows of {N_CTX} with stride {STRIDE}, each token once.",
        "The set is the 2,318 genomes with refusal-penalized fitness greater than 0.6 from the 50 RQ2 runs.",
        "",
        "Per-prompt scores: `rq2_prompt_perplexity_gt06.csv`.",
        "Histogram: `rq2_prompt_perplexity_hist.png`.",
        "",
        "## Overall",
        "",
    ]
    overall = summaries[0]
    lines += [
        f"{overall['n']} prompts. Mean {fmt(overall['mean'])}, median {fmt(overall['median'])}, IQR {fmt(overall['iqr'])} (Q1 {fmt(overall['q25'])}, Q3 {fmt(overall['q75'])}), range {fmt(overall['min'])}–{fmt(overall['max'])}.",
        f"{overall['n_multi_window']} prompts needed more than one context window.",
        f"Spearman correlation with fitness {fmt(overall['spearman_fitness'], 3)}, with character length {fmt(overall['spearman_chars'], 3)}, with token length {fmt(overall['spearman_tokens'], 3)}.",
        "",
        "| Configuration | n | Mean | Median | Q1 | Q3 | IQR | Min | Max | Windows > 1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stats in summaries:
        lines.append(
            f"| {stats['configuration']} | {stats['n']} | {fmt(stats['mean'])} | {fmt(stats['median'])} | "
            f"{fmt(stats['q25'])} | {fmt(stats['q75'])} | {fmt(stats['iqr'])} | {fmt(stats['min'])} | "
            f"{fmt(stats['max'])} | {stats['n_multi_window']} |"
        )
    lines += [
        "",
        "| Configuration | Spearman fitness | Pearson fitness | Spearman characters | Spearman tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stats in summaries:
        lines.append(
            f"| {stats['configuration']} | {fmt(stats['spearman_fitness'], 3)} | {fmt(stats['pearson_fitness'], 3)} | "
            f"{fmt(stats['spearman_chars'], 3)} | {fmt(stats['spearman_tokens'], 3)} |"
        )
    lines.append("")
    for stats in summaries[1:]:
        name = stats["configuration"]
        lines += [
            f"## {name}",
            "",
            f"{stats['n']} prompts with fitness greater than 0.6.",
            f"Mean perplexity {fmt(stats['mean'])}, median {fmt(stats['median'])}, IQR {fmt(stats['iqr'])} "
            f"(Q1 {fmt(stats['q25'])}, Q3 {fmt(stats['q75'])}), range {fmt(stats['min'])}–{fmt(stats['max'])}.",
            f"{stats['n_multi_window']} prompts were longer than {N_CTX} tokens and used a second window.",
            f"Correlation with fitness: Spearman {fmt(stats['spearman_fitness'], 3)}, Pearson {fmt(stats['pearson_fitness'], 3)}.",
            f"Correlation with prompt length: Spearman {fmt(stats['spearman_chars'], 3)} on characters, {fmt(stats['spearman_tokens'], 3)} on tokens.",
            "",
        ]
    (REPORTS / "rq2_prompt_perplexity.md").write_text("\n".join(lines), encoding="utf-8")
    print("[wrote]", REPORTS / "rq2_prompt_perplexity.md", flush=True)


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if not MODEL.is_file():
        raise SystemExit(f"model missing: {MODEL}")
    if not HIGH_CSV.is_file():
        raise SystemExit(f"missing {HIGH_CSV}; run rq2_text_diagnostics.py first")
    rows = collect_prompts(load_wanted())
    done: Dict[Tuple[str, int, str], dict] = {}
    if limit == 0 and CHECKPOINT.is_file():
        with CHECKPOINT.open(encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                rec["seed"] = int(rec["seed"])
                rec["fitness"] = float(rec["fitness"])
                rec["prompt_chars"] = int(rec["prompt_chars"])
                rec["n_tokens"] = int(rec["n_tokens"])
                rec["n_windows"] = int(rec["n_windows"])
                rec["perplexity"] = float(rec["perplexity"])
                done[(rec["cell"], rec["seed"], rec["genome_id"])] = rec
        print(f"[resume] {len(done)} already scored", flush=True)
    if limit:
        rows = rows[:limit]
    pending = [r for r in rows if (r["cell"], r["seed"], r["genome_id"]) not in done]
    print(f"[score] {len(pending)} prompts to score ({len(rows)} in this run)", flush=True)
    from llama_cpp import Llama

    llm = Llama(
        model_path=str(MODEL),
        n_ctx=N_CTX,
        n_batch=N_BATCH,
        n_gpu_layers=-1,
        logits_all=False,
        verbose=False,
        use_mmap=True,
    )
    t0 = time.time()
    fields = ["cell", "label", "seed", "genome_id", "fitness", "prompt_chars", "n_tokens", "n_windows", "perplexity"]
    checkpoint_new = limit == 0
    if checkpoint_new:
        write_header = not CHECKPOINT.is_file()
        ckpt = CHECKPOINT.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(ckpt, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
    else:
        ckpt = None
        writer = None
    for i, row in enumerate(pending, start=1):
        ppl, n_tokens, n_windows = prompt_perplexity(llm, row["prompt"])
        row["perplexity"] = ppl
        row["n_tokens"] = n_tokens
        row["n_windows"] = n_windows
        row.pop("prompt", None)
        if writer is not None:
            writer.writerow(row)
            if i % 20 == 0:
                ckpt.flush()
        if i == 1 or i % 25 == 0 or i == len(pending):
            elapsed = time.time() - t0
            rate = elapsed / i
            left = rate * (len(pending) - i)
            print(
                f"[score] {i}/{len(pending)} ppl={ppl:.2f} tokens={n_tokens} windows={n_windows} "
                f"elapsed={elapsed/60:.1f}m eta={left/60:.1f}m",
                flush=True,
            )
    if ckpt is not None:
        ckpt.close()
    if limit:
        print("[limit] smoke test only; reports not written", flush=True)
        return
    by_key = dict(done)
    for row in rows:
        key = (row["cell"], row["seed"], row["genome_id"])
        if "perplexity" in row:
            by_key[key] = row
    write_outputs([by_key[k] for k in ((r["cell"], r["seed"], r["genome_id"]) for r in rows)])


if __name__ == "__main__":
    main()
