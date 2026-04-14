# SOTA comparison with ToxSearch-S

**`configs/base.yml`** (the default `--config_file`) is aligned with ToxSearch-S decoding hyperparameters:

| Role | ToxSearch-S source | `sampling_params` in `base.yml` |
|------|-------------------|----------------------------------|
| Target (response) | `config/RGConfig.yaml` → `response_generator.generation_args` | `temperature: 0.8`, `top_p: 0.9`, `top_k: 40`, `max_tokens: 2048`, `repetition_penalty: 1.0` |
| Mutator (prompt variants) | `config/PGConfig.yaml` → `prompt_generator.generation_args` (PG uses 0.9) | `temperature: 0.8`, `top_p: 0.9`, `top_k: 40`, `max_tokens: 2048`, `repetition_penalty: 1.1` |

- vLLM uses `max_tokens`; ToxSearch-S llama.cpp uses `max_new_tokens` — same numeric cap (2048).

**CLI defaults** (`python -m rainbowplus.rainbowplus`, no flags) — current fork uses a lighter budget; scale up to mirror `src/main.py` if needed:

| ToxSearch-S (`main.py`) | RainbowPlus (defaults) |
|-------------------------|------------------------|
| `--batch-size` default **100** | `--num_mutations` default **3** (use **100** to match batch size) |
| `--max-total-genomes` | `--max_genomes` default **1000** |
| Seed file | `--num_samples` default **100** |
| (runs until genome budget) | `--max_iters` default **10000** (ceiling) |
| `--fitness_threshold` N/A | `--fitness_threshold` default **0.3** |

Example to align with **50 × 100** evaluations: `--num_mutations 100 --max_genomes 5000` (and raise `--max_iters` if needed).

**Model paths:** set `model:` (and matching `tokenizer:`) in `base.yml` to your local GGUF + HF tokenizer id, or pass `--target_llm /path/to/model.gguf` to override the **target** weights path only.

**Fitness:** the fork uses `PerspectiveScorer` (Google Perspective API); `fitness_llm` in YAML remains for reference or alternate runs.

**Other configs:** `base-opensource.yml`, `base-openai.yml`, and `eval.yml` are unchanged for upstream-style experiments.
