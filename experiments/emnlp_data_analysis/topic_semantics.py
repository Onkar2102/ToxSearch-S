"""Phase 12: TF-IDF tokens, exemplars, optional gpt-4o-mini labels (cached)."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_logger = logging.getLogger("emnlp.topic_semantics")

# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

# Specific harmful-content nouns/verbs that get masked in the exemplar markdown.
# Topic-typical content is preserved structurally but specifics are removed.
_REDACT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\b(weapon|weapons|firearm|firearms|gun|guns|rifle|pistol|knife|knives|"
     r"explosive|explosives|bomb|bombs|grenade|ammunition|ammo)\b",
     "[REDACTED-WEAPON]"),
    (r"\b(drug|drugs|cocaine|heroin|fentanyl|meth(amphetamine)?|opioid|opioids|"
     r"narcotic|narcotics|cannabis|marijuana)\b",
     "[REDACTED-DRUG]"),
    (r"\b(suicide|self-harm|self harm|kill myself|end my life|"
     r"ending my life|end my own life)\b",
     "[REDACTED-SELFHARM]"),
    (r"\b(murder|kill|assassinate|behead|execute|stab|shoot|shooting|"
     r"shootings|massacre|slaughter)\b",
     "[REDACTED-VIOLENCE]"),
    (r"\b(torture|abuse|rape|molest|assault|attack|attacking)\b",
     "[REDACTED-HARM]"),
    (r"\b(hack|hacking|exploit|malware|ransomware|phishing|ddos|"
     r"keylogger|trojan|virus)\b",
     "[REDACTED-CYBER]"),
    (r"\b(nuclear|radiological|biological agent|chemical weapon|"
     r"anthrax|sarin|ricin|nerve agent)\b",
     "[REDACTED-WMD]"),
    (r"\b(child(ren)?|minor|kid|kids|underage)\b",
     "[REDACTED-MINOR]"),
    (r"\b(racist|racial slur|slur|ethnic slur|homophobic)\b",
     "[REDACTED-SLUR]"),
    (r"\b(launder(ing)?|fraud|forge(ry)?|counterfeit|embezzle|tax evasion|"
     r"insider trading|ponzi)\b",
     "[REDACTED-FINCRIME]"),
    (r"\b(stalk(ing)?|harass(ing|ment)?|dox+(ing|ed)?|swat(ting)?)\b",
     "[REDACTED-STALK]"),
)


def redact_text(text: str) -> str:
    """Apply deterministic regex-based redaction to a single prompt."""
    out = text
    for pat, repl in _REDACT_PATTERNS:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Phase12Config:
    """Knobs for the phase 12 build."""

    min_topic_size: int = 5
    top_k_tokens: int = 8
    top_k_exemplars: int = 3
    max_exemplar_chars: int = 220
    tfidf_ngram_max: int = 2
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95
    enable_llm: bool = True
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_seed: int = 42
    llm_max_tokens: int = 24
    llm_request_delay_sec: float = 0.5
    extra_stopwords: Tuple[str, ...] = field(
        default_factory=lambda: (
            "how", "what", "why", "way", "ways", "best",
            "good", "make", "get", "use", "using", "without",
            "would", "could", "should", "tell", "give", "list",
            "explain", "step", "steps", "method", "methods",
            "describe", "say", "asked", "ask", "tips", "tip",
        )
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _coerce_species_id(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def collect_topic_documents(
    rows: Sequence[Dict[str, Any]],
    *,
    min_topic_size: int,
) -> Tuple[Dict[int, List[int]], List[str]]:
    """Return ``({sid -> indices}, prompts_list)`` keeping topics with size >= min_topic_size."""
    prompts: List[str] = []
    by_topic: Dict[int, List[int]] = {}
    for r in rows:
        sid = _coerce_species_id(r.get("species_id"))
        prompt = str(r.get("prompt") or "").strip()
        idx = len(prompts)
        prompts.append(prompt)
        if sid > 0 and prompt:
            by_topic.setdefault(sid, []).append(idx)
    by_topic = {sid: idxs for sid, idxs in by_topic.items() if len(idxs) >= min_topic_size}
    return by_topic, prompts


# ---------------------------------------------------------------------------
# TF-IDF over topics
# ---------------------------------------------------------------------------


def topic_top_tokens(
    prompts: Sequence[str],
    topic_indices: Dict[int, List[int]],
    *,
    top_k_tokens: int,
    ngram_max: int,
    min_df: int,
    max_df: float,
    extra_stopwords: Sequence[str],
) -> Dict[int, List[Tuple[str, float]]]:
    """Discriminative TF-IDF tokens per topic relative to the full corpus."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:  # pragma: no cover - sklearn is in requirements
        raise RuntimeError("scikit-learn is required for TF-IDF") from exc

    redacted_corpus = [redact_text(p) for p in prompts]
    stopwords = "english"  # sklearn baseline
    vec = TfidfVectorizer(
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df,
        stop_words=stopwords,
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
    )
    X = vec.fit_transform(redacted_corpus)
    vocab = np.array(vec.get_feature_names_out())
    extra = {w.lower() for w in extra_stopwords}
    out: Dict[int, List[Tuple[str, float]]] = {}
    for sid, idxs in topic_indices.items():
        if not idxs:
            out[sid] = []
            continue
        # Mean TF-IDF within the topic minus mean TF-IDF outside (discriminative).
        in_mask = np.zeros(X.shape[0], dtype=bool)
        in_mask[idxs] = True
        in_mean = np.asarray(X[in_mask].mean(axis=0)).ravel()
        if in_mask.sum() < X.shape[0]:
            out_mean = np.asarray(X[~in_mask].mean(axis=0)).ravel()
        else:
            out_mean = np.zeros_like(in_mean)
        score = in_mean - out_mean
        order = np.argsort(score)[::-1]
        picks: List[Tuple[str, float]] = []
        seen: set = set()
        for j in order:
            tok = str(vocab[j])
            if tok in extra:
                continue
            if any(tok.startswith(p) and p in seen for p in seen):
                continue
            if score[j] <= 0 and len(picks) >= 1:
                break
            picks.append((tok, float(score[j])))
            seen.add(tok)
            if len(picks) >= top_k_tokens:
                break
        out[sid] = picks
    return out


# ---------------------------------------------------------------------------
# Nearest-centroid exemplars
# ---------------------------------------------------------------------------


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < 1e-12:
        return x
    return x / n


def topic_exemplars(
    rows: Sequence[Dict[str, Any]],
    prompts: Sequence[str],
    topic_indices: Dict[int, List[int]],
    *,
    top_k_exemplars: int,
    max_chars: int,
) -> Dict[int, List[Dict[str, Any]]]:
    """Return ``{sid -> [exemplar dicts]}`` ordered by similarity to topic centroid."""
    out: Dict[int, List[Dict[str, Any]]] = {}
    for sid, idxs in topic_indices.items():
        embs: List[Tuple[int, np.ndarray]] = []
        for i in idxs:
            emb = rows[i].get("_embedding")
            if emb is None:
                continue
            embs.append((i, np.asarray(emb, dtype=np.float64).ravel()))
        if not embs:
            out[sid] = []
            continue
        E = np.vstack([_l2_normalize(e) for _, e in embs])
        centroid = _l2_normalize(E.mean(axis=0))
        sims = E @ centroid
        order = np.argsort(sims)[::-1]
        picks: List[Dict[str, Any]] = []
        for k in order[: top_k_exemplars]:
            i_orig = embs[int(k)][0]
            row = rows[i_orig]
            text = prompts[i_orig].strip()
            redacted = redact_text(text)
            if len(redacted) > max_chars:
                redacted = redacted[: max_chars - 1].rsplit(" ", 1)[0] + "…"
            picks.append({
                "genome_id": row.get("genome_id"),
                "generation": row.get("generation"),
                "source_file": row.get("source_file"),
                "similarity_to_centroid": round(float(sims[int(k)]), 4),
                "redacted_prompt": redacted,
            })
        out[sid] = picks
    return out


# ---------------------------------------------------------------------------
# LLM label call (cached, deterministic, optional)
# ---------------------------------------------------------------------------


def _label_cache_path(out_dir: Path) -> Path:
    return Path(out_dir) / "label_cache.json"


def _load_label_cache(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    p = _label_cache_path(out_dir)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_label_cache(out_dir: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    p = _label_cache_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _label_cache_key(
    species_id: int,
    top_tokens: Sequence[str],
    exemplar_texts: Sequence[str],
    *,
    model: str,
    seed: int,
) -> str:
    payload = {
        "species_id": int(species_id),
        "tokens": list(top_tokens),
        "exemplars": [t.strip() for t in exemplar_texts],
        "model": model,
        "seed": int(seed),
        "version": "phase12.v1",
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_LABEL_PROMPT_TEMPLATE = (
    "You are categorizing harmful-prompt clusters from a red-teaming dataset for "
    "a research paper. Given the discriminative tokens and three redacted "
    "representative prompts below, output a 2-4 word academic label naming the "
    "harm category in clinical, neutral language (e.g. \"weapons of mass "
    "destruction\", \"online financial fraud\", \"self-harm narrative\"). "
    "Output ONLY the label, nothing else, and do not include the word 'topic'.\n\n"
    "Discriminative tokens: {tokens}\n\n"
    "Representative prompts (redacted):\n{exemplars}\n\n"
    "Label:"
)


def _format_label_prompt(
    top_tokens: Sequence[str],
    exemplar_texts: Sequence[str],
) -> str:
    tokens_str = ", ".join(top_tokens) or "(none)"
    bullets = "\n".join(f"- {t.strip()}" for t in exemplar_texts) or "- (no exemplars)"
    return _LABEL_PROMPT_TEMPLATE.format(tokens=tokens_str, exemplars=bullets)


def _sanitize_label(raw: str) -> str:
    s = raw.strip().strip('"').strip("'")
    s = re.sub(r"^(label\s*[:\-])\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".")
    if not s:
        return ""
    words = s.split()
    if len(words) > 6:
        words = words[:6]
    return " ".join(words)


def call_llm_label(
    species_id: int,
    top_tokens: Sequence[str],
    exemplar_texts: Sequence[str],
    *,
    cache: Dict[str, Dict[str, Any]],
    model: str,
    seed: int,
    temperature: float,
    max_tokens: int,
    request_delay_sec: float,
) -> Tuple[Optional[str], str, str]:
    """Return ``(label, source, cache_key)`` where source ∈ {cache, api, error, missing-key}."""
    key = _label_cache_key(
        species_id, top_tokens, exemplar_texts, model=model, seed=seed
    )
    cached = cache.get(key)
    if cached and cached.get("label"):
        return _sanitize_label(str(cached["label"])), "cache", key

    if not os.environ.get("OPENAI_API_KEY"):
        return None, "missing-key", key

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        _logger.warning("openai client not installed; skipping LLM label")
        return None, "error", key

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write concise, clinical labels for harmful-content "
                        "clusters in academic NLP red-teaming research."
                    ),
                },
                {
                    "role": "user",
                    "content": _format_label_prompt(top_tokens, exemplar_texts),
                },
            ],
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content if resp.choices else ""
        label = _sanitize_label(raw or "")
        cache[key] = {
            "species_id": int(species_id),
            "label": label,
            "raw": raw,
            "model": model,
            "seed": int(seed),
        }
        if request_delay_sec > 0:
            time.sleep(request_delay_sec)
        return label, "api", key
    except Exception as exc:  # network / quota
        _logger.warning("LLM label call failed for sid=%s: %s", species_id, exc)
        return None, "error", key


def fallback_label_from_tokens(
    top_tokens: Sequence[Tuple[str, float]],
    *,
    n: int = 3,
) -> str:
    """Build a deterministic readable label from top TF-IDF tokens."""
    if not top_tokens:
        return ""
    picks: List[str] = []
    for tok, _ in top_tokens[: max(n * 2, n)]:
        if any(tok in p or p in tok for p in picks):
            continue
        picks.append(tok)
        if len(picks) >= n:
            break
    return " / ".join(picks)


# ---------------------------------------------------------------------------
# Build / IO
# ---------------------------------------------------------------------------


def build_phase12_artifacts(
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    *,
    config: Optional[Phase12Config] = None,
) -> Dict[str, Any]:
    """Compute Phase 12 artifacts and write them under ``out_dir``."""
    cfg = config or Phase12Config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    topic_indices, prompts = collect_topic_documents(rows, min_topic_size=cfg.min_topic_size)
    n_topics = len(topic_indices)
    _logger.info("Phase 12: building labels for %d topics", n_topics)

    tokens_by_topic = topic_top_tokens(
        prompts,
        topic_indices,
        top_k_tokens=cfg.top_k_tokens,
        ngram_max=cfg.tfidf_ngram_max,
        min_df=cfg.tfidf_min_df,
        max_df=cfg.tfidf_max_df,
        extra_stopwords=cfg.extra_stopwords,
    )
    exemplars_by_topic = topic_exemplars(
        rows,
        prompts,
        topic_indices,
        top_k_exemplars=cfg.top_k_exemplars,
        max_chars=cfg.max_exemplar_chars,
    )

    cache = _load_label_cache(out_dir)
    label_rows: List[Dict[str, Any]] = []
    n_cache_hits = 0
    n_api_calls = 0
    n_missing_key = 0
    n_label_errors = 0

    for sid in sorted(topic_indices):
        toks = tokens_by_topic.get(sid, [])
        token_strs = [t for t, _ in toks]
        ex_records = exemplars_by_topic.get(sid, [])
        ex_texts = [e["redacted_prompt"] for e in ex_records]
        fb_label = fallback_label_from_tokens(toks)
        llm_label: Optional[str] = None
        source = "fallback"
        if cfg.enable_llm and token_strs:
            llm_label, source, _ = call_llm_label(
                sid,
                token_strs,
                ex_texts,
                cache=cache,
                model=cfg.llm_model,
                seed=cfg.llm_seed,
                temperature=cfg.llm_temperature,
                max_tokens=cfg.llm_max_tokens,
                request_delay_sec=cfg.llm_request_delay_sec,
            )
            if source == "cache":
                n_cache_hits += 1
            elif source == "api":
                n_api_calls += 1
            elif source == "missing-key":
                n_missing_key += 1
            elif source == "error":
                n_label_errors += 1

        chosen = llm_label.strip() if llm_label else fb_label
        label_rows.append({
            "species_id": int(sid),
            "n_members": len(topic_indices[sid]),
            "label": chosen,
            "label_source": source if llm_label else "fallback",
            "fallback_label": fb_label,
            "llm_label": llm_label or "",
            "top_tokens": "|".join(token_strs),
            "top_token_scores": "|".join(f"{s:.4f}" for _, s in toks),
            "exemplar_genome_ids": "|".join(str(e["genome_id"]) for e in ex_records),
            "exemplar_similarities": "|".join(
                f"{e['similarity_to_centroid']:.4f}" for e in ex_records
            ),
        })

    # Always persist the label cache so subsequent runs are free.
    _save_label_cache(out_dir, cache)

    csv_path = out_dir / "topic_semantic_labels.csv"
    if label_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(label_rows[0].keys()))
            w.writeheader()
            w.writerows(label_rows)

    md_path = out_dir / "topic_exemplars.md"
    md_path.write_text(
        _render_exemplars_markdown(label_rows, exemplars_by_topic, tokens_by_topic),
        encoding="utf-8",
    )

    summary = {
        "n_topics": n_topics,
        "n_label_cache_hits": n_cache_hits,
        "n_label_api_calls": n_api_calls,
        "n_label_missing_key": n_missing_key,
        "n_label_errors": n_label_errors,
        "llm_enabled": bool(cfg.enable_llm),
        "llm_model": cfg.llm_model,
        "llm_seed": cfg.llm_seed,
        "min_topic_size": cfg.min_topic_size,
        "top_k_tokens": cfg.top_k_tokens,
        "top_k_exemplars": cfg.top_k_exemplars,
    }
    summary_path = out_dir / "phase12_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    artifacts = {
        "topic_semantic_labels_csv": str(csv_path),
        "topic_exemplars_md": str(md_path),
        "phase12_summary_json": str(summary_path),
        "label_cache_json": str(_label_cache_path(out_dir)),
    }
    manifest_path = out_dir / "phase12_manifest.json"
    manifest_path.write_text(
        json.dumps({"artifacts": artifacts, "meta": summary}, indent=2),
        encoding="utf-8",
    )
    artifacts["phase12_manifest"] = str(manifest_path)

    return {
        "artifacts": artifacts,
        "meta": summary,
        "label_rows": label_rows,
        "topic_indices": topic_indices,
        "tokens_by_topic": tokens_by_topic,
        "exemplars_by_topic": exemplars_by_topic,
    }


def _render_exemplars_markdown(
    label_rows: Sequence[Dict[str, Any]],
    exemplars_by_topic: Dict[int, List[Dict[str, Any]]],
    tokens_by_topic: Dict[int, List[Tuple[str, float]]],
) -> str:
    lines: List[str] = [
        "# Phase 12 — Topic semantic labels and redacted exemplars",
        "",
        "Each topic shows: short label, top discriminative tokens, and "
        "nearest-centroid representative prompts. All prompts are passed through "
        "a deterministic redactor that masks specifics of harmful nouns/verbs "
        "(weapons, drugs, self-harm, violence, cyber, WMD, minors, slurs, "
        "financial crime, stalking) with `[REDACTED-…]` markers; the linguistic "
        "scaffolding is preserved.",
        "",
    ]
    for r in label_rows:
        sid = int(r["species_id"])
        lines.append(f"## Topic species_id={sid} — {r['label']}")
        lines.append("")
        lines.append(f"- Members: {r['n_members']}")
        toks = ", ".join(t for t, _ in tokens_by_topic.get(sid, [])) or "(none)"
        lines.append(f"- Top discriminative tokens: {toks}")
        lines.append(f"- Label source: {r['label_source']}")
        lines.append("")
        lines.append("Representative prompts (redacted):")
        lines.append("")
        for j, e in enumerate(exemplars_by_topic.get(sid, []), 1):
            lines.append(
                f"{j}. `gid={e['genome_id']} gen={e['generation']} "
                f"src={e['source_file']} sim={e['similarity_to_centroid']:.3f}` "
                f"— {e['redacted_prompt']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI helpers (load unified inputs from disk)
# ---------------------------------------------------------------------------


def _load_unified_rows_from_csv(
    csv_path: Path,
    embeddings_path: Path,
    genome_ids_path: Path,
) -> List[Dict[str, Any]]:
    """Reconstruct minimal row dicts (with ``_embedding``) from saved unified artifacts."""
    rows: List[Dict[str, Any]] = []
    embeddings: Optional[np.ndarray] = None
    if embeddings_path.is_file():
        embeddings = np.load(embeddings_path, allow_pickle=False)
    id_index: Dict[str, int] = {}
    if genome_ids_path.is_file():
        try:
            ids = json.loads(genome_ids_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            ids = []
        for k, gid in enumerate(ids):
            id_index[str(gid)] = k

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row: Dict[str, Any] = dict(r)
            try:
                row["species_id"] = int(r.get("species_id") or 0)
            except (TypeError, ValueError):
                row["species_id"] = 0
            try:
                row["generation"] = int(r.get("generation") or 0)
            except (TypeError, ValueError):
                row["generation"] = 0
            row["prompt"] = str(r.get("prompt") or "")
            gid = str(r.get("genome_id") or "")
            if embeddings is not None and gid in id_index:
                k = id_index[gid]
                if 0 <= k < embeddings.shape[0]:
                    row["_embedding"] = np.asarray(embeddings[k], dtype=np.float64)
            rows.append(row)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 12 semantic topic labels")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Where to read unified inputs from and write phase12 outputs.",
    )
    parser.add_argument(
        "--unified-dir",
        type=Path,
        default=None,
        help="Override unified artifacts directory (defaults to <results-dir>/unified).",
    )
    parser.add_argument("--run-id", default="20260211_2122")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip the LLM label call; use TF-IDF fallback only.")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-seed", type=int, default=42)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--top-k-tokens", type=int, default=8)
    parser.add_argument("--top-k-exemplars", type=int, default=3)
    parser.add_argument("--min-topic-size", type=int, default=5)
    args = parser.parse_args(argv)

    unified_dir = Path(args.unified_dir or args.results_dir / "unified")
    csv_path = unified_dir / f"{args.run_id}_genomes.csv"
    emb_path = unified_dir / f"{args.run_id}_embeddings.npy"
    ids_path = unified_dir / f"{args.run_id}_genome_ids.json"
    if not csv_path.is_file():
        print(f"Missing unified CSV: {csv_path}", file=sys.stderr)
        return 2

    rows = _load_unified_rows_from_csv(csv_path, emb_path, ids_path)
    out_dir = Path(args.results_dir) / "phase12"
    cfg = Phase12Config(
        min_topic_size=args.min_topic_size,
        top_k_tokens=args.top_k_tokens,
        top_k_exemplars=args.top_k_exemplars,
        enable_llm=not args.no_llm,
        llm_model=args.llm_model,
        llm_seed=args.llm_seed,
        llm_temperature=args.llm_temperature,
    )
    bundle = build_phase12_artifacts(rows, out_dir, config=cfg)
    print(json.dumps(bundle["meta"], indent=2))
    return 0


__all__ = [
    "Phase12Config",
    "build_phase12_artifacts",
    "call_llm_label",
    "collect_topic_documents",
    "fallback_label_from_tokens",
    "redact_text",
    "topic_exemplars",
    "topic_top_tokens",
]


if __name__ == "__main__":
    raise SystemExit(main())
