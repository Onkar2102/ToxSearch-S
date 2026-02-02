#!/usr/bin/env python3
"""Analyze top toxicity species and why multiple species have high scores."""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE = Path("data/outputs")
runs = [BASE / f"run0{i}_speciated" for i in range(1, 6)]

print("=" * 80)
print("ANALYZING TOP TOXICITY SPECIES - WHY MULTIPLE HIGH SCORES?")
print("=" * 80)

# Find top species and analyze them
top_species_data = []

for run_dir in sorted(runs):
    if not run_dir.exists():
        continue
    
    run_id = run_dir.name
    elites = json.load(open(run_dir / "elites.json")) if (run_dir / "elites.json").exists() else []
    reserves = json.load(open(run_dir / "reserves.json")) if (run_dir / "reserves.json").exists() else []
    spec_state = json.load(open(run_dir / "speciation_state.json")) if (run_dir / "speciation_state.json").exists() else {}
    
    # Collect toxicity by species
    species_tox = defaultdict(list)
    
    for genome in elites + reserves:
        sid = genome.get("species_id")
        if sid is not None and sid != 0 and sid != -1:
            tox = (genome.get("toxicity") or 
                   genome.get("north_star_score") or 
                   genome.get("fitness") or
                   genome.get("moderation_result", {}).get("google", {}).get("scores", {}).get("toxicity"))
            if tox is not None and isinstance(tox, (int, float)) and not np.isnan(tox):
                species_tox[str(sid)].append(float(tox))
    
    # Get labels and metadata
    species_labels = {}
    species_metadata = {}
    if spec_state and "species" in spec_state:
        for sid, sdata in spec_state["species"].items():
            labels = sdata.get("labels", [])
            valid_labels = [l for l in labels if l and l.strip()]
            if valid_labels:
                species_labels[sid] = valid_labels
                species_metadata[sid] = {
                    "state": sdata.get("species_state", ""),
                    "created_at": sdata.get("created_at", 0),
                    "cluster_origin": sdata.get("cluster_origin", "unknown")
                }
    
    # Store top species
    for sid, toxicities in species_tox.items():
        if toxicities:
            max_tox = max(toxicities)
            if max_tox > 0.4:  # Focus on high-toxicity species
                top_species_data.append({
                    "run_id": run_id,
                    "species_id": sid,
                    "max_toxicity": max_tox,
                    "median_toxicity": np.median(toxicities),
                    "mean_toxicity": np.mean(toxicities),
                    "count": len(toxicities),
                    "labels": species_labels.get(sid, []),
                    "state": species_metadata.get(sid, {}).get("state", ""),
                    "created_at": species_metadata.get(sid, {}).get("created_at", 0),
                    "origin": species_metadata.get(sid, {}).get("cluster_origin", "unknown"),
                    "all_toxicities": sorted(toxicities, reverse=True)[:5]  # Top 5 values
                })

# Sort by max toxicity
top_species_data.sort(key=lambda x: x["max_toxicity"], reverse=True)

print(f"\nTop 15 Species by Max Toxicity (>0.4):")
print("-" * 100)
print(f"{'Run':<15} {'Species':<10} {'Max':<8} {'Median':<8} {'Count':<6} {'Origin':<10} {'Top 3 Labels'}")
print("-" * 100)

for sp in top_species_data[:15]:
    labels_str = ", ".join(sp["labels"][:3]) if sp["labels"] else "No labels"
    print(f"{sp['run_id']:<15} {sp['species_id']:<10} {sp['max_toxicity']:<8.4f} {sp['median_toxicity']:<8.4f} {sp['count']:<6} {sp['origin']:<10} {labels_str}")

# Analyze why multiple species have high scores
print("\n" + "=" * 80)
print("WHY MULTIPLE SPECIES HAVE HIGH TOXICITY?")
print("=" * 80)

# Group by run
runs_with_high_tox = defaultdict(list)
for sp in top_species_data[:10]:
    runs_with_high_tox[sp["run_id"]].append(sp)

print("\nHigh-toxicity species distribution by run:")
for run_id, species_list in sorted(runs_with_high_tox.items()):
    print(f"\n  {run_id}: {len(species_list)} high-toxicity species")
    for sp in sorted(species_list, key=lambda x: x["max_toxicity"], reverse=True):
        print(f"    Species {sp['species_id']}: max={sp['max_toxicity']:.4f}, count={sp['count']}, origin={sp['origin']}")

# Check if they're from same or different semantic clusters
print("\n" + "=" * 80)
print("SEMANTIC CLUSTER ANALYSIS")
print("=" * 80)

# Check label similarity between top species
top_3 = top_species_data[:3]
print(f"\nTop 3 species label comparison:")
for i, sp1 in enumerate(top_3):
    print(f"\n  {i+1}. {sp1['run_id']} Species {sp1['species_id']} (max={sp1['max_toxicity']:.4f}):")
    print(f"     Labels: {', '.join(sp1['labels'][:10]) if sp1['labels'] else 'No labels'}")
    print(f"     Origin: {sp1['origin']}, Created at gen: {sp1['created_at']}")
    print(f"     Toxicity values: {sp1['all_toxicities']}")

# Compare labels
if len(top_3) >= 2:
    print(f"\n  Label overlap between top species:")
    for i, sp1 in enumerate(top_3):
        for j, sp2 in enumerate(top_3[i+1:], i+1):
            labels1 = set(l.lower().strip() for l in sp1["labels"] if l)
            labels2 = set(l.lower().strip() for l in sp2["labels"] if l)
            if labels1 and labels2:
                overlap = labels1.intersection(labels2)
                similarity = len(overlap) / max(len(labels1), len(labels2)) if max(len(labels1), len(labels2)) > 0 else 0
                print(f"    Species {sp1['species_id']} vs {sp2['species_id']}: {similarity:.1%} similarity ({len(overlap)} common labels)")

# Check if they're from same run
print(f"\n  Are top species from same run?")
runs_of_top3 = [sp["run_id"] for sp in top_3]
if len(set(runs_of_top3)) == 1:
    print(f"    Yes - all from {runs_of_top3[0]}")
else:
    print(f"    No - from different runs: {set(runs_of_top3)}")

# Check generation when created
print(f"\n  Generation when created:")
for sp in top_3:
    print(f"    Species {sp['species_id']}: gen {sp['created_at']}")

# Hypothesis: Why multiple high-toxicity species?
print("\n" + "=" * 80)
print("HYPOTHESIS: WHY MULTIPLE SPECIES WITH HIGH TOXICITY?")
print("=" * 80)
print("""
Possible reasons:
1. Different semantic clusters discovered similar high-toxicity strategies
2. Species formed at different times, each finding high-toxicity prompts independently
3. Merging didn't occur (species remained separate despite similar toxicity)
4. Semantic diversity: different prompts achieve similar toxicity through different approaches
5. No capacity enforcement: species could grow large, increasing chance of finding high-toxicity prompts
""")
