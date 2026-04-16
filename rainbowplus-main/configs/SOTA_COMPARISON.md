# SOTA comparison with ToxSearch-S

**`configs/base.yml`** (the default `--config_file` when run from `rainbowplus-main/`) is aligned with ToxSearch-S decoding hyperparameters. **ToxSearch-S YAMLs** live at the **parent repo root**: `../config/RGConfig.yaml` and `../config/PGConfig.yaml` relative to `rainbowplus-main/`.

| Role | ToxSearch-S source | `sampling_params` in `base.yml` |
|------|-------------------|----------------------------------|
| Target (response) | `config/RGConfig.yaml` (repo root) → `response_generator.generation_args` | `temperature: 0.8`, `top_p: 0.9`, `top_k: 40`, `max_tokens: 2048`, `repetition_penalty: 1.0` |
| Mutator (prompt variants) | `config/PGConfig.yaml` (repo root) → `prompt_generator.generation_args` | `temperature: 0.8`, `top_p: 0.9`, `top_k: 40`, `max_tokens: 2048`, `repetition_penalty: 1.1` |

- vLLM uses `max_tokens`; ToxSearch-S llama.cpp uses `max_new_tokens` — same numeric cap (2048).

**CLI defaults** (`python -m rainbowplus.rainbowplus`, no flags) — current fork uses a lighter budget; scale up to mirror `src/main.py` if needed:

| ToxSearch-S (`main.py`) | RainbowPlus (defaults) |
|-------------------------|------------------------|
| **`--batch-size`**: merge **K** in **`--parallel` MPI mode only** (sequential runs ignore it; omit for auto K). Legacy auto default **100** when operators are `cm`/`ie`. | `--num_mutations` default **3** (raise to mirror a chosen merge batch, e.g. **100**) |
| **`--max-total-genomes`** (required seq + parallel) | `--max_genomes` default **1000** |
| Seed file | `--num_samples` default **100** |
| (runs until genome budget) | `--max_iters` default **10000** (ceiling) |
| `--fitness_threshold` N/A | `--fitness_threshold` default **0.3** |

Example to align with **50 × 100** evaluations: `--num_mutations 100 --max_genomes 5000` (and raise `--max_iters` if needed).

**Model paths:** set `model:` (and matching `tokenizer:`) in `base.yml` to your local GGUF + HF tokenizer id, or pass `--target_llm /path/to/model.gguf` to override the **target** weights path only.

**Fitness:** the fork uses `PerspectiveScorer` (Google Perspective API); `fitness_llm` in YAML remains for reference or alternate runs.

**Other configs:** `base-opensource.yml`, `base-openai.yml`, and `eval.yml` are unchanged for upstream-style experiments.
