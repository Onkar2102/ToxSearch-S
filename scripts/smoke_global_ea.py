#!/usr/bin/env python3
"""Multi-generation smoke for clustering_method=none (Global-EA).

Mocks RG/PG/moderation/operators so the sequential main loop can run without GPUs.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _mock_generate_single_variant(parents, pg, **kwargs):
    p0 = parents[0] if parents else {}
    return [
        {
            "prompt": f"evolved_{str(p0.get('prompt', 'x'))[:20]}?",
            "operator": "MockOp",
            "variant_type": "mutation",
            "parents": [{"id": p.get("id"), "score": 0.5} for p in parents],
            "status": "pending_generation",
            "creation_info": {"type": "mutation", "operator": "MockOp"},
        }
    ]


def _fake_process_population(pop_path: str) -> None:
    path = Path(pop_path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    for i, g in enumerate(data):
        if not g:
            continue
        g["generated_output"] = f"resp_{g.get('id', i)}"
        g["model_name"] = "mock-rg"
        g["status"] = "pending_evaluation"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _fake_moderation(pop_path: str, **kwargs) -> None:
    metric = kwargs.get("north_star_metric") or "toxicity"
    path = Path(pop_path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    for i, g in enumerate(data):
        if not g:
            continue
        score = 0.3 + 0.05 * ((int(g.get("id") or i) % 10))
        g["moderation_result"] = {"google": {"scores": {metric: round(score, 4)}}}
        g["status"] = "complete"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class _MockRG:
    model_name = "mock-rg"
    model_cfg = {"name": "mock-rg"}

    def generate_response(self, prompt, **kwargs):
        return (f"resp_{str(prompt)[:12]}", 0.01)

    def process_population(self, pop_path="data/outputs/temp.json"):
        _fake_process_population(pop_path)


class _MockPG:
    model_cfg = {"name": "mock-pg"}


def _assert_run(outputs: Path, min_gens: int = 2) -> None:
    elites = json.loads((outputs / "elites.json").read_text(encoding="utf-8"))
    archive = json.loads((outputs / "archive.json").read_text(encoding="utf-8"))
    temp = json.loads((outputs / "temp.json").read_text(encoding="utf-8"))
    state = json.loads((outputs / "speciation_state.json").read_text(encoding="utf-8"))
    tracker = json.loads((outputs / "EvolutionTracker.json").read_text(encoding="utf-8"))
    parents = json.loads((outputs / "parents.json").read_text(encoding="utf-8"))

    assert temp == [], f"temp should be empty after integrate, got {len(temp)}"
    assert archive == [], f"archive should stay empty in Global-EA, got {len(archive)}"
    assert len(elites) >= 3, f"expected growing elites, got {len(elites)}"
    assert all(int(g.get("species_id", -1)) == 1 for g in elites), "all elites need species_id=1"
    assert state.get("clustering_method") == "none"
    assert state.get("population_mode") == "global"
    gens = tracker.get("generations") or []
    assert len(gens) >= min_gens + 1, f"expected >= {min_gens + 1} tracker gens, got {len(gens)}"
    # post-gen0 entries should report global mode when present
    for g in gens[1:]:
        mode = g.get("population_mode")
        if mode is not None:
            assert mode == "global", f"tracker population_mode={mode!r}"
    assert parents, "parents.json should retain last selected parents"
    ids = [g["id"] for g in elites]
    assert len(ids) == len(set(ids)), f"duplicate elite ids: {ids}"
    print(
        f"SMOKE OK elites={len(elites)} gens={len(gens)} "
        f"selection_mode={tracker.get('selection_mode')} "
        f"parents={len(parents)}"
    )


def main() -> int:
    import ea.evolution_engine as ee_module
    from utils.population_io import set_outputs_path

    tmp = Path(tempfile.mkdtemp(prefix="global_ea_smoke_"))
    outputs = tmp / "outputs"
    seed_csv = tmp / "seed.csv"
    with open(seed_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["questions"])
        for q in ["prompt_a?", "prompt_b?", "prompt_c?"]:
            w.writerow([q])

    set_outputs_path(outputs)
    original_gsv = ee_module.generate_single_variant
    ee_module.generate_single_variant = _mock_generate_single_variant

    # Plain object — MagicMock auto-attrs break _create_child_genome via
    # hasattr(op, "_last_operation_time") + round(MagicMock, 4).
    class _MockOp:
        name = "MockOp"
        operator_type = "mutation"

        def apply(self, inp):
            parent = inp.get("parent_data") or {}
            if isinstance(parent, list):
                parent = parent[0] if parent else {}
            return [f"child_{parent.get('id', 0)}_g?"]

    mock_op = _MockOp()

    from utils import population_io as pio

    def _init_system(logger, log_file, seed_file="data/prompt.csv", seed=None, clustering_method=None):
        from ea.evolution_engine import set_global_generators

        set_global_generators(_MockRG(), _MockPG())
        pio.load_and_initialize_population(
            input_path=str(seed_csv),
            output_path=str(outputs),
            log_file=log_file,
            clustering_method=clustering_method,
        )
        return _MockRG(), _MockPG()

    try:
        with patch("main.initialize_system", side_effect=_init_system), \
             patch("main.update_model_configs", return_value=None), \
             patch("gne.get_run_moderation_on_population", return_value=_fake_moderation), \
             patch("utils.live_analysis.run_live_analysis", return_value={}), \
             patch(
                 "utils.operator_effectiveness.generate_operator_effectiveness_visualizations",
                 return_value={},
             ):
            with patch.object(
                ee_module.EvolutionEngine,
                "_get_single_parent_operators",
                return_value=[mock_op],
            ), patch.object(
                ee_module.EvolutionEngine,
                "_get_multi_parent_operators",
                return_value=[],
            ):
                from main import main as tox_main

                rc = tox_main(
                    max_total_genomes=8,
                    seed_file=str(seed_csv),
                    seed=0,
                    evaluator="google",
                    north_star_metric="toxicity",
                    clustering_method="none",
                    operators="all",
                    max_variants=1,
                    theta_sim=0.1,
                    theta_merge=0.1,
                    distance_method="embedding",
                )
                if rc not in (0, None):
                    print(f"main returned {rc}", file=sys.stderr)
                    return 1
        _assert_run(outputs, min_gens=2)
        print(f"outputs={outputs}")
        return 0
    finally:
        ee_module.generate_single_variant = original_gsv


if __name__ == "__main__":
    raise SystemExit(main())
