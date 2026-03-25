# SOTA comparison with ToxSearch-S

**`configs/base.yml`** (the default `--config_file`) is aligned with ToxSearch-S decoding hyperparameters:

| Role | ToxSearch-S source | `sampling_params` in `base.yml` |
|------|-------------------|----------------------------------|
| Target (response) | `config/RGConfig.yaml` → `response_generator.generation_args` | `temperature: 0.8`, `top_p: 0.9`, `top_k: 40`, `max_tokens: 2048`, `repetition_penalty: 1.0` |
| Mutator (prompt variants) | `config/PGConfig.yaml` → `prompt_generator.generation_args` | `temperature: 0.9`, `top_p: 0.9`, `top_k: 40`, `max_tokens: 2048`, `repetition_penalty: 1.1` |

- vLLM uses `max_tokens`; ToxSearch-S llama.cpp uses `max_new_tokens` — same numeric cap (2048).

**CLI defaults** (`python -m rainbowplus.rainbowplus`, no flags) mirror `src/main.py` comparison knobs where there is a direct mapping:

| ToxSearch-S (`main.py`) | RainbowPlus |
|-------------------------|-------------|
| `--batch-size` default **100** | `--num_mutations` default **100** (children scored per iteration) |
| `--max-total-genomes` (e.g. **5000** = 50×100) | `--max_genomes` default **5000** |
| Seed count from `--seed-file` (you use ~**100** genomes at gen 0) | `--num_samples` default **100** |
| (no hard iter cap; runs until genome budget) | `--max_iters` default **20000** (safety ceiling) |
| `--fitness_threshold` N/A (different archive mechanics) | `--fitness_threshold` default **0.3** (archive + `above_threshold` in jsonl) |

With `--max_genomes 5000` and `--num_mutations 100`, the run stops after ~50 mutation-heavy iterations if dedup is low (same order as **50 generations × 100** evaluations). To score a full **100×100** seed wave before the cap, use e.g. `--max_genomes 10000`.

**Model paths:** set `model:` (and matching `tokenizer:`) in `base.yml` to your local GGUF + HF tokenizer id, or pass `--target_llm /path/to/model.gguf` to override the **target** weights path only.

**Fitness:** the fork uses `PerspectiveScorer` (Google Perspective API); `fitness_llm` in YAML remains for reference or alternate runs.

**Other configs:** `base-opensource.yml`, `base-openai.yml`, and `eval.yml` are unchanged for upstream-style experiments.
