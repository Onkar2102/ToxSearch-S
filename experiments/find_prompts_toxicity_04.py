#!/usr/bin/env python3
"""Find prompts in speciated executions with toxicity around 0.25, prompt length < 20 words, and is_refusal == 1."""

import json
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
TOXICITY_LOW, TOXICITY_HIGH = 0.20, 0.30  # "around 0.25"
MAX_WORDS = 20
REQUIRE_REFUSAL = True  # only include genomes where is_refusal == 1


def get_toxicity(genome):
    if not genome or not isinstance(genome, dict):
        return None
    mr = genome.get("moderation_result")
    if not mr:
        return genome.get("toxicity") or (genome.get("scores") or {}).get("toxicity")
    if isinstance(mr, dict) and "google" in mr and "scores" in mr["google"]:
        return mr["google"]["scores"].get("toxicity")
    if isinstance(mr, dict) and "scores" in mr:
        return mr["scores"].get("toxicity")
    return None


def main():
    speciated_dirs = sorted(
        d for d in OUTPUTS_DIR.iterdir()
        if d.is_dir() and d.name.endswith("_speciated")
    )
    if not speciated_dirs:
        print(f"No run*_speciated dirs in {OUTPUTS_DIR}")
        return

    results = []
    for run_dir in speciated_dirs:
        run_id = run_dir.name
        for fname in ["elites.json", "reserves.json", "archive.json"]:
            p = run_dir / fname
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Skip {run_id}/{fname}: {e}")
                continue
            if not isinstance(data, list):
                continue
            for g in data:
                prompt = (g or {}).get("prompt") or ""
                tox = get_toxicity(g)
                if tox is None:
                    continue
                if REQUIRE_REFUSAL and (g or {}).get("is_refusal") != 1:
                    continue
                words = len(prompt.split())
                if TOXICITY_LOW <= tox <= TOXICITY_HIGH and words < MAX_WORDS:
                    results.append({
                        "run": run_id,
                        "source": fname.replace(".json", ""),
                        "toxicity": round(tox, 4),
                        "words": words,
                        "prompt": prompt,
                        "is_refusal": (g or {}).get("is_refusal", None),
                    })

    print(f"Speciated runs: {[d.name for d in speciated_dirs]}")
    print(f"Criteria: toxicity in [{TOXICITY_LOW}, {TOXICITY_HIGH}], prompt length < {MAX_WORDS} words, is_refusal == 1")
    print(f"Found {len(results)} prompt(s)\n")
    for i, r in enumerate(results, 1):
        print(f"--- {i} ---")
        print(f"  Run: {r['run']} ({r['source']})")
        print(f"  Toxicity: {r['toxicity']}  |  Words: {r['words']}  |  is_refusal: {r.get('is_refusal')}")
        print(f"  Prompt: {r['prompt']!r}")
        print()


if __name__ == "__main__":
    main()
