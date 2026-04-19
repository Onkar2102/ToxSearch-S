"""
ensemble_distance_analysis.py

Threshold-calibration diagnostic and formal-bound sanity checks for the
ensemble compatibility distance used by ToxSearch-S. Operates exclusively
on saved Gen-0 artifacts (elites.json + reserves.json with generation==0)
from the ppsn2026/toxsearch_s runs; no new multi-generation execution is
required.

This script is the analytical basis for Appendix A.2 of the paper. It
produces:

  experiments/outputs/ensemble_distance_diagnostic/summary.json
  docs/figures/ensemble_distance_diagnostic_hist.pdf
  docs/figures/ensemble_distance_diagnostic_scatter.pdf
  docs/figures/ensemble_distance_diagnostic_slack.pdf

Run:
  python experiments/ensemble_distance_analysis.py

Options:
  --runs-root PATH          primary directory of run folders (default: ppsn2026)
  --extra-runs-root PATH    repeatable; merge Gen-0 from additional roots
  --max-exhaustive-triples-n  when unique prompt count exceeds this, triple
                            slack ECDFs use a fixed random sample instead of
                            all \\binom{n}{3} triples (default: 130)
  --random-triple-samples   number of random triples when sampling (default:
                            400000)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from speciation.gen0_clustering import Gen0Clustering  # noqa: E402
from speciation.reserves import CLUSTER_0_ID  # noqa: E402
from speciation.species import Individual  # noqa: E402


PHENOTYPE_SCORE_ORDER = [
    "toxicity",
    "threat",
    "profanity",
    "sexually_explicit",
    "identity_attack",
    "flirtation",
    "insult",
    "severe_toxicity",
]

W_GENOTYPE = 0.7
W_PHENOTYPE = 0.3
THETA_SIM = 0.25
THETA_MERGE = 0.25


def extract_phenotype(genome: dict) -> np.ndarray | None:
    mr = genome.get("moderation_result") or {}
    google = mr.get("google") if isinstance(mr, dict) else None
    if not isinstance(google, dict):
        return None
    scores = google.get("scores")
    if not isinstance(scores, dict):
        return None
    v = np.array(
        [float(scores.get(k, 0.0)) for k in PHENOTYPE_SCORE_ORDER],
        dtype=np.float64,
    )
    return np.clip(v, 0.0, 1.0)


def extract_embedding(genome: dict) -> np.ndarray | None:
    emb = genome.get("prompt_embedding")
    if not emb or not isinstance(emb, list):
        return None
    e = np.asarray(emb, dtype=np.float64)
    if e.ndim != 1 or e.size == 0:
        return None
    n = np.linalg.norm(e)
    if not np.isfinite(n) or n == 0.0:
        return None
    return e / n


def load_gen0_from_run(run_dir: Path) -> list[dict]:
    collected: list[dict] = []
    for fname in ("elites.json", "reserves.json"):
        fp = run_dir / fname
        if not fp.exists():
            continue
        with fp.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            continue
        for g in payload:
            if g.get("generation") != 0:
                continue
            e = extract_embedding(g)
            p = extract_phenotype(g)
            if e is None or p is None:
                continue
            prompt = (g.get("prompt") or "").strip()
            if not prompt:
                continue
            collected.append(
                {
                    "prompt": prompt,
                    "embedding": e,
                    "phenotype": p,
                    "run": run_dir.name,
                    "id": g.get("id"),
                }
            )
    return collected


def build_levels(records: list[dict]) -> tuple[dict, dict]:
    """Aggregate prompt-level (Level A) and keep evaluation-level (Level B)."""
    by_prompt: dict[str, list[dict]] = {}
    for r in records:
        by_prompt.setdefault(r["prompt"], []).append(r)

    level_a_prompts: list[str] = []
    level_a_embeddings: list[np.ndarray] = []
    level_a_phenotypes: list[np.ndarray] = []
    for prompt, rs in by_prompt.items():
        es = np.stack([r["embedding"] for r in rs])
        ps = np.stack([r["phenotype"] for r in rs])
        e_bar = es.mean(axis=0)
        e_bar = e_bar / np.linalg.norm(e_bar)
        p_bar = ps.mean(axis=0)
        level_a_prompts.append(prompt)
        level_a_embeddings.append(e_bar)
        level_a_phenotypes.append(p_bar)

    level_a = {
        "prompts": level_a_prompts,
        "E": np.stack(level_a_embeddings),
        "P": np.stack(level_a_phenotypes),
        "n_prompts": len(level_a_prompts),
        "n_evaluations_per_prompt": {
            prompt: len(rs) for prompt, rs in by_prompt.items()
        },
    }

    level_b = {
        "E": np.stack([r["embedding"] for r in records]),
        "P": np.stack([r["phenotype"] for r in records]),
        "runs": [r["run"] for r in records],
        "prompts": [r["prompt"] for r in records],
        "n_instances": len(records),
    }
    return level_a, level_b


def pairwise_unweighted_matrices(E: np.ndarray, P: np.ndarray) -> dict:
    """Genotype- and phenotype-only pairwise matrices (full $n\\times n$)."""
    n = E.shape[0]
    cos_sim = np.clip(E @ E.T, -1.0, 1.0)
    d_g_full = (1.0 - cos_sim) / 2.0
    np.fill_diagonal(d_g_full, 0.0)

    diff = P[:, None, :] - P[None, :, :]
    d_p_full = np.linalg.norm(diff, axis=-1) / np.sqrt(P.shape[1])
    np.fill_diagonal(d_p_full, 0.0)

    iu = np.triu_indices(n, k=1)
    return {
        "n": n,
        "d_g_full": d_g_full,
        "d_p_full": d_p_full,
        "d_g": d_g_full[iu],
        "d_p": d_p_full[iu],
        "iu": iu,
    }


def pairwise_components(E: np.ndarray, P: np.ndarray) -> dict:
    """Same as :func:`pairwise_unweighted_matrices` plus default-weight ensemble."""
    base = pairwise_unweighted_matrices(E, P)
    d_g_full = base["d_g_full"]
    d_p_full = base["d_p_full"]
    d_e_full = W_GENOTYPE * d_g_full + W_PHENOTYPE * d_p_full
    iu = base["iu"]
    return {
        **base,
        "d_e_full": d_e_full,
        "d_e": d_e_full[iu],
    }


def leader_follower_gen0_weight_sweep(
    E: np.ndarray,
    P: np.ndarray,
    prompts: list[str],
    *,
    d_g_full: np.ndarray,
    d_p_full: np.ndarray,
    theta: float,
    min_island_size: int,
    weight_pairs: list[tuple[float, float]],
) -> dict:
    """Same Gen-0 two-phase leader--follower as production (`Gen0Clustering.run`).

    For each $(w_g,w_p)$, builds `Individual` objects and runs
    :class:`Gen0Clustering` with that pair's ensemble distance. Also reports
    pairwise summaries (median $d$, fraction $<\\theta$) on the weighted matrix
    for comparison with the threshold-calibration block.
    """
    n = E.shape[0]
    iu = np.triu_indices(n, k=1)
    logger = logging.getLogger("ensemble_weight_sweep")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.WARNING)

    toxicity_idx = PHENOTYPE_SCORE_ORDER.index("toxicity")
    fitnesses = P[:, toxicity_idx].astype(np.float64)

    rows: list[dict] = []
    for wg, wp in weight_pairs:
        if abs(wg + wp - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1, got {wg=}, {wp=}")
        d_e_full = wg * d_g_full + wp * d_p_full
        d_upper = d_e_full[iu]
        rho_ge, _ = spearmanr(d_upper, d_g_full[iu])
        rho_pe, _ = spearmanr(d_upper, d_p_full[iu])

        individuals: list[Individual] = []
        for i in range(n):
            individuals.append(
                Individual(
                    id=i + 1,
                    prompt=prompts[i][:200],
                    fitness=float(fitnesses[i]),
                    embedding=np.asarray(E[i], dtype=np.float64),
                    phenotype=np.asarray(P[i], dtype=np.float64),
                    species_id=None,
                    generation=0,
                )
            )
        species, _ = Gen0Clustering.run(
            individuals,
            theta_sim=theta,
            min_island_size=min_island_size,
            w_genotype=wg,
            w_phenotype=wp,
            current_generation=0,
            logger=logger,
        )
        n_species = len(species)
        n_reserves = sum(
            1 for ind in individuals if ind.species_id == CLUSTER_0_ID
        )
        rows.append(
            {
                "w_genotype": wg,
                "w_phenotype": wp,
                "median_pairwise_d": float(np.median(d_upper)),
                "mean_pairwise_d": float(d_upper.mean()),
                "frac_pairs_below_theta": float((d_upper < theta).mean()),
                "n_species_formed": int(n_species),
                "n_individuals_in_reserves": int(n_reserves),
                "spearman_d_vs_d_genotype": float(rho_ge),
                "spearman_d_vs_d_phenotype": float(rho_pe),
            }
        )
    return {
        "method": "Gen0Clustering.run (production leader-follower, two-phase Gen-0)",
        "theta_sim": theta,
        "min_island_size": min_island_size,
        "n_points": n,
        "rows": rows,
    }


def five_number(x: np.ndarray) -> dict:
    x = np.asarray(x)
    q = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": float(q[0]),
        "q1": float(q[1]),
        "median": float(q[2]),
        "q3": float(q[3]),
        "max": float(q[4]),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
    }


def triple_slacks_exhaustive(d_e_full: np.ndarray) -> dict:
    """Exhaustive triangle- and 2-inframetric-slack over all triples.

    For every ordered directional claim d(u,w) <= d(u,v)+d(v,w), we iterate
    over all 3 roles within each unordered triple and take the tightest
    violation/slack, which corresponds to the standard formulation
    s = min_{v in {a,b,c}} [ d(a,c-but-not-v) + d(b,c-but-not-v) - d(a,b) ].

    Returns statistics on:
      s1 := min over the 3 roles of [ d(x,v)+d(v,y) - d(x,y) ]  (the tightest
             triangle-slack; negative iff TI fails for this triple)
      s2 := min over the 3 roles of [ 2*(d(x,v)+d(v,y)) - d(x,y) ] (tightest
             2-inframetric slack; must be >=0 by the A.2.2 lemma)

    Implementation: iterate over all unordered triples (n choose 3) with
    numpy lookups. For n<=150 this is well below 1M iterations and runs
    in a few seconds.
    """
    n = d_e_full.shape[0]
    idx = np.array(list(combinations(range(n), 3)), dtype=np.int64)
    a, b, c = idx[:, 0], idx[:, 1], idx[:, 2]
    d_ab = d_e_full[a, b]
    d_ac = d_e_full[a, c]
    d_bc = d_e_full[b, c]

    s_via_a = (d_ab + d_ac) - d_bc
    s_via_b = (d_ab + d_bc) - d_ac
    s_via_c = (d_ac + d_bc) - d_ab
    s1 = np.minimum(np.minimum(s_via_a, s_via_b), s_via_c)

    s2_via_a = 2.0 * (d_ab + d_ac) - d_bc
    s2_via_b = 2.0 * (d_ab + d_bc) - d_ac
    s2_via_c = 2.0 * (d_ac + d_bc) - d_ab
    s2 = np.minimum(np.minimum(s2_via_a, s2_via_b), s2_via_c)

    ti_violations_sided = np.concatenate(
        [
            d_bc - (d_ab + d_ac),
            d_ac - (d_ab + d_bc),
            d_ab - (d_ac + d_bc),
        ]
    )

    return {
        "n_triples": int(idx.shape[0]),
        "s1_tight_summary": five_number(s1),
        "s2_tight_summary": five_number(s2),
        "n_triples_with_any_TI_violation": int((ti_violations_sided > 0).sum()),
        "max_TI_violation_magnitude": float(
            max(0.0, ti_violations_sided.max())
        ),
        "min_s2_over_all_sides": float(
            min(s2_via_a.min(), s2_via_b.min(), s2_via_c.min())
        ),
        "fraction_triples_s1_negative": float((s1 < 0.0).mean()),
        "mode": "exhaustive",
    }


def triple_slacks_random_sample(
    d_e_full: np.ndarray, rng: np.random.Generator, n_samples: int
) -> dict:
    """Monte Carlo estimate of slack distributions when n is large."""
    n = d_e_full.shape[0]
    if n < 3:
        raise ValueError("need at least 3 points")
    idx = rng.choice(n, size=(n_samples, 3), replace=True)
    # reject samples with duplicate indices
    mask = (idx[:, 0] != idx[:, 1]) & (idx[:, 0] != idx[:, 2]) & (idx[:, 1] != idx[:, 2])
    idx = idx[mask]
    if len(idx) < n_samples // 2:
        # rare for large n; pad with fresh draws
        extra = n_samples - len(idx)
        add = rng.choice(n, size=(extra * 2, 3), replace=True)
        m2 = (add[:, 0] != add[:, 1]) & (add[:, 0] != add[:, 2]) & (add[:, 1] != add[:, 2])
        idx = np.vstack([idx, add[m2]])[:n_samples]

    a, b, c = idx[:, 0], idx[:, 1], idx[:, 2]
    d_ab = d_e_full[a, b]
    d_ac = d_e_full[a, c]
    d_bc = d_e_full[b, c]

    s_via_a = (d_ab + d_ac) - d_bc
    s_via_b = (d_ab + d_bc) - d_ac
    s_via_c = (d_ac + d_bc) - d_ab
    s1 = np.minimum(np.minimum(s_via_a, s_via_b), s_via_c)

    s2_via_a = 2.0 * (d_ab + d_ac) - d_bc
    s2_via_b = 2.0 * (d_ab + d_bc) - d_ac
    s2_via_c = 2.0 * (d_ac + d_bc) - d_ab
    s2 = np.minimum(np.minimum(s2_via_a, s2_via_b), s2_via_c)

    ti_violations_sided = np.concatenate(
        [
            d_bc - (d_ab + d_ac),
            d_ac - (d_ab + d_bc),
            d_ab - (d_ac + d_bc),
        ]
    )

    return {
        "n_triples": int(len(s1)),
        "s1_tight_summary": five_number(s1),
        "s2_tight_summary": five_number(s2),
        "n_triples_with_any_TI_violation": int((ti_violations_sided > 0).sum()),
        "max_TI_violation_magnitude": float(
            max(0.0, float(ti_violations_sided.max()))
        ),
        "min_s2_over_all_sides": float(
            min(float(s2_via_a.min()), float(s2_via_b.min()), float(s2_via_c.min()))
        ),
        "fraction_triples_s1_negative": float((s1 < 0.0).mean()),
        "mode": "random_sample",
    }


def cosine_vs_angular_agreement(E: np.ndarray) -> dict:
    cos_sim = np.clip(E @ E.T, -1.0, 1.0)
    iu = np.triu_indices(E.shape[0], k=1)
    d_cos = (1.0 - cos_sim[iu]) / 2.0
    d_ang = np.arccos(cos_sim[iu]) / np.pi
    rho, _ = spearmanr(d_cos, d_ang)
    return {
        "spearman_rho_cosine_vs_angular": float(rho),
        "max_abs_gap": float(np.max(np.abs(d_cos - d_ang))),
    }


def calibration_fractions(d_e: np.ndarray) -> dict:
    return {
        "frac_pairs_below_theta_sim": float((d_e < THETA_SIM).mean()),
        "frac_pairs_below_theta_merge": float((d_e < THETA_MERGE).mean()),
        "frac_pairs_above_theta_sim": float((d_e >= THETA_SIM).mean()),
    }


def plot_hist(d_e: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.hist(d_e, bins=50, edgecolor="black", alpha=0.75)
    ax.axvline(
        THETA_SIM,
        color="red",
        linestyle="--",
        linewidth=1.4,
        label=r"$\theta_{\mathrm{sim}}=\theta_{\mathrm{merge}}=0.25$",
    )
    ax.set_xlabel(r"Pairwise $d_{\mathrm{ensemble}}$ on Gen-0 seeds (Level A)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of pairwise ensemble distance on Gen-0 seeds")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(d_g: np.ndarray, d_p: np.ndarray, d_e: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    sc = ax.scatter(d_g, d_p, c=d_e, s=8, alpha=0.5, cmap="viridis")
    ax.set_xlabel(r"$d_{\mathrm{gen\text{-}norm}}(u,v)$")
    ax.set_ylabel(r"$d_{\mathrm{phenotype}}(u,v)$")
    ax.set_title(
        r"Component decomposition of $d_{\mathrm{ensemble}}$ on Gen-0 pairs"
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$d_{\mathrm{ensemble}}(u,v)$")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_slack_ecdf(slacks: dict, out_path: Path) -> None:
    """Two-panel ECDF of tight triangle slack s1 and 2-inframetric slack s2.

    s1 being >=0 is an empirical observation (not guaranteed).
    s2 being >=0 is guaranteed by the A.2.2 lemma; we plot it as a
    sanity/implementation check.
    """
    n_triples = slacks["n_triples"]
    mode = slacks.get("mode", "exhaustive")
    mode_note = "" if mode == "exhaustive" else f" ({mode}, {n_triples:,} draws)"
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    for ax, key, title, vline_label in (
        (axes[0], "s1_samples", r"Triangle slack $s_1$ (tight, all triples)", r"$s_1=0$"),
        (
            axes[1],
            "s2_samples",
            r"2-inframetric slack $s_2$ (tight, all triples)",
            r"$s_2=0$ (guaranteed)",
        ),
    ):
        x = np.sort(slacks[key])
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y, linewidth=1.5)
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2, label=vline_label)
        ax.set_xlabel("slack")
        ax.set_ylabel("ECDF")
        ax.set_title(title)
        ax.legend(loc="lower right", frameon=False)

    fig.suptitle(
        f"Slack ECDFs over triples (n_triples={n_triples:,}){mode_note}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def compute_slack_samples_for_plot(
    d_e_full: np.ndarray,
    rng: np.random.Generator,
    exhaustive_limit: int,
    random_samples: int,
) -> dict:
    n = d_e_full.shape[0]
    if n <= exhaustive_limit:
        idx = np.array(list(combinations(range(n), 3)), dtype=np.int64)
    else:
        idx = rng.choice(n, size=(random_samples, 3), replace=True)
        m = (idx[:, 0] != idx[:, 1]) & (idx[:, 0] != idx[:, 2]) & (idx[:, 1] != idx[:, 2])
        idx = idx[m][:random_samples]
    a, b, c = idx[:, 0], idx[:, 1], idx[:, 2]
    d_ab = d_e_full[a, b]
    d_ac = d_e_full[a, c]
    d_bc = d_e_full[b, c]

    s_via_a = (d_ab + d_ac) - d_bc
    s_via_b = (d_ab + d_bc) - d_ac
    s_via_c = (d_ac + d_bc) - d_ab
    s1 = np.minimum(np.minimum(s_via_a, s_via_b), s_via_c)

    s2_via_a = 2.0 * (d_ab + d_ac) - d_bc
    s2_via_b = 2.0 * (d_ab + d_bc) - d_ac
    s2_via_c = 2.0 * (d_ac + d_bc) - d_ab
    s2 = np.minimum(np.minimum(s2_via_a, s2_via_b), s2_via_c)

    return {
        "n_triples": int(idx.shape[0]),
        "s1_samples": s1,
        "s2_samples": s2,
        "mode": "exhaustive" if n <= exhaustive_limit else "random_sample",
    }


def discover_run_dirs(repo_root: Path, rel_roots: list[str]) -> list[Path]:
    """Collect immediate child directories under each rel_root that contain elites.json."""
    seen: set[str] = set()
    out: list[Path] = []
    for rel in rel_roots:
        root = repo_root / rel
        if not root.is_dir():
            raise FileNotFoundError(f"runs root not found: {root}")
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            key = str(d.resolve())
            if key in seen:
                continue
            if not (d / "elites.json").exists():
                continue
            seen.add(key)
            out.append(d)
    return out


def run(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    rel_roots = [args.runs_root] + list(args.extra_runs_root or [])
    run_dirs = discover_run_dirs(repo_root, rel_roots)
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories with elites.json under: {rel_roots}"
        )

    print(f"[load] scanning {len(run_dirs)} run directories from {rel_roots}")
    records: list[dict] = []
    per_run_counts: dict[str, int] = {}
    for d in run_dirs:
        rs = load_gen0_from_run(d)
        label = f"{d.parent.name}/{d.name}"
        per_run_counts[label] = len(rs)
        records.extend(rs)
        print(f"  {label}: {len(rs)} Gen-0 genomes with (embedding, phenotype)")

    if not records:
        raise RuntimeError("No Gen-0 records with both embedding and phenotype were found")

    level_a, level_b = build_levels(records)
    print(
        f"[levels] Level A (prompt-level): {level_a['n_prompts']} unique prompts"
    )
    print(
        f"[levels] Level B (evaluation-level): {level_b['n_instances']} total instances"
    )

    print("[compute] pairwise components on Level A ...")
    pa = pairwise_components(level_a["E"], level_a["P"])
    base_m = pairwise_unweighted_matrices(level_a["E"], level_a["P"])
    weight_grid = [
        (1.0, 0.0),
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
    ]
    print("[compute] Gen-0 weight sweep (production Gen0Clustering / leader-follower) ...")
    weight_sweep = leader_follower_gen0_weight_sweep(
        level_a["E"],
        level_a["P"],
        level_a["prompts"],
        d_g_full=base_m["d_g_full"],
        d_p_full=base_m["d_p_full"],
        theta=THETA_SIM,
        min_island_size=args.min_island_size,
        weight_pairs=weight_grid,
    )
    rng = np.random.default_rng(42)

    print("[compute] pairwise components on Level B (sampled) ...")
    nB = level_b["n_instances"]
    if nB <= 120:
        pb_full = pairwise_components(level_b["E"], level_b["P"])
        d_eB = pb_full["d_e"]
        d_gB = pb_full["d_g"]
        d_pB = pb_full["d_p"]
    else:
        target = min(5000, nB * (nB - 1) // 2)
        sampled_pairs = set()
        d_eB_list: list[float] = []
        d_gB_list: list[float] = []
        d_pB_list: list[float] = []
        E = level_b["E"]
        P = level_b["P"]
        while len(d_eB_list) < target:
            i, j = rng.integers(0, nB, size=2)
            if i == j:
                continue
            key = (int(min(i, j)), int(max(i, j)))
            if key in sampled_pairs:
                continue
            sampled_pairs.add(key)
            e_i = E[i]; e_j = E[j]
            p_i = P[i]; p_j = P[j]
            cos_ij = float(np.clip(e_i @ e_j, -1.0, 1.0))
            d_g_norm = (1.0 - cos_ij) / 2.0
            d_p_ij = float(np.linalg.norm(p_i - p_j) / np.sqrt(P.shape[1]))
            d_eB_list.append(W_GENOTYPE * d_g_norm + W_PHENOTYPE * d_p_ij)
            d_gB_list.append(d_g_norm)
            d_pB_list.append(d_p_ij)
        d_eB = np.asarray(d_eB_list)
        d_gB = np.asarray(d_gB_list)
        d_pB = np.asarray(d_pB_list)

    n_a = pa["n"]
    print(
        "[compute] triangle- and 2-inframetric slack on Level A "
        f"(n={n_a}, exhaustive limit={args.max_exhaustive_triples_n}) ..."
    )
    if n_a <= args.max_exhaustive_triples_n:
        slacks_stats = triple_slacks_exhaustive(pa["d_e_full"])
    else:
        slacks_stats = triple_slacks_random_sample(
            pa["d_e_full"], rng, args.random_triple_samples
        )
    slacks_samples = compute_slack_samples_for_plot(
        pa["d_e_full"],
        rng,
        exhaustive_limit=args.max_exhaustive_triples_n,
        random_samples=min(args.random_triple_samples, 200_000),
    )

    print("[compute] cosine vs angular ordering agreement ...")
    agreement = cosine_vs_angular_agreement(level_a["E"])

    summary = {
        "inputs": {
            "runs_roots": rel_roots,
            "per_run_gen0_counts": per_run_counts,
            "max_exhaustive_triples_n": args.max_exhaustive_triples_n,
            "random_triple_samples": args.random_triple_samples,
            "n_records": len(records),
            "n_unique_prompts_level_a": level_a["n_prompts"],
            "n_instances_level_b": level_b["n_instances"],
            "phenotype_score_order": PHENOTYPE_SCORE_ORDER,
            "weights": {"w_genotype": W_GENOTYPE, "w_phenotype": W_PHENOTYPE},
            "thresholds": {"theta_sim": THETA_SIM, "theta_merge": THETA_MERGE},
        },
        "level_a": {
            "n_pairs": int(pa["d_e"].size),
            "d_genotype_norm": five_number(pa["d_g"]),
            "d_phenotype": five_number(pa["d_p"]),
            "d_ensemble": five_number(pa["d_e"]),
            "calibration": calibration_fractions(pa["d_e"]),
            "triple_slack": slacks_stats,
            "triple_slack_plot": {
                "n_triples": slacks_samples["n_triples"],
                "mode": slacks_samples["mode"],
            },
            "cosine_vs_angular": agreement,
        },
        "level_b": {
            "n_pairs_reported": int(d_eB.size),
            "d_genotype_norm": five_number(d_gB),
            "d_phenotype": five_number(d_pB),
            "d_ensemble": five_number(d_eB),
            "calibration": calibration_fractions(d_eB),
        },
        "weight_sweep_gen0": weight_sweep,
    }

    out_dir = repo_root / "experiments" / "outputs" / "ensemble_distance_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    print(f"[write] {out_dir / 'summary.json'}")

    fig_dir = repo_root / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_hist(pa["d_e"], fig_dir / "ensemble_distance_diagnostic_hist.pdf")
    plot_scatter(pa["d_g"], pa["d_p"], pa["d_e"], fig_dir / "ensemble_distance_diagnostic_scatter.pdf")
    plot_payload = {
        "n_triples": slacks_samples["n_triples"],
        "s1_samples": slacks_samples["s1_samples"],
        "s2_samples": slacks_samples["s2_samples"],
        "mode": slacks_samples["mode"],
    }
    plot_slack_ecdf(plot_payload, fig_dir / "ensemble_distance_diagnostic_slack.pdf")
    print(f"[write] figures -> {fig_dir}")

    print("\n== Level A summary (unique Gen-0 seeds) ==")
    print(f"  n_pairs = {pa['d_e'].size:,}")
    print(f"  d_ensemble mean = {pa['d_e'].mean():.4f}, median = {np.median(pa['d_e']):.4f}")
    print(
        f"  TI violations = {slacks_stats['n_triples_with_any_TI_violation']:,} / "
        f"{slacks_stats['n_triples']:,} sided checks "
        f"(max magnitude = {slacks_stats['max_TI_violation_magnitude']:.6f})"
    )
    print(
        f"  2-inframetric slack (tight) min = "
        f"{slacks_stats['s2_tight_summary']['min']:.6f} (must be >= 0)"
    )
    print(
        f"  cosine/angular Spearman rho = "
        f"{agreement['spearman_rho_cosine_vs_angular']:.6f}"
    )
    print("\n== Gen-0 weight sweep (leader-follower, theta=theta_sim, C_min) ==")
    for r in weight_sweep["rows"]:
        print(
            f"  w_g={r['w_genotype']:.1f} w_p={r['w_phenotype']:.1f}  "
            f"median_d={r['median_pairwise_d']:.3f}  "
            f"frac<theta={r['frac_pairs_below_theta']:.3f}  "
            f"species={r['n_species_formed']}  reserves={r['n_individuals_in_reserves']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default="data/outputs/ppsn2026/toxsearch_s",
        help="Path (relative to repo root) to a directory of run folders (each with elites.json).",
    )
    parser.add_argument(
        "--extra-runs-root",
        action="append",
        default=[],
        help="Additional runs root (repeatable). All immediate subdirs with elites.json are merged.",
    )
    parser.add_argument(
        "--max-exhaustive-triples-n",
        type=int,
        default=130,
        help="If unique prompt count exceeds this, use random triple sampling for slack stats.",
    )
    parser.add_argument(
        "--random-triple-samples",
        type=int,
        default=400_000,
        help="Number of random triples when sampling slack statistics.",
    )
    parser.add_argument(
        "--min-island-size",
        type=int,
        default=5,
        help="Minimum island size $C_{\\min}$ for Gen0Clustering (same as reported experiments).",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
