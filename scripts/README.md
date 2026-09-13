# Scripts

Helper utilities that are **not** part of the core evolution loop.

| Script | Purpose |
|--------|---------|
| `validate_run_outputs.py` | Lightweight checks on a finished `data/outputs/<run>/` directory |
| `emnlp_4pager_analysis.py` | Legacy EMNLP 4-pager analysis entry (prefer `experiments/emnlp_data_analysis/` for current pipelines) |

Primary research drivers live under `experiments/`. Generated artifacts go under `results/` (see [`../results/manifests/README.md`](../results/manifests/README.md)).

Shell runners remain at the repo root (`run_experiments_local.sh`, `run_experiments_parallel_2w.sh`) and are the supported way to launch full evolution experiments.
