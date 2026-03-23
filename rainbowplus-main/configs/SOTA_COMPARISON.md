# SOTA comparison with ToxSearch-S

**`configs/base.yml`** (the default `--config_file`) is aligned with ToxSearch-S decoding hyperparameters:

| Role | ToxSearch-S source | `sampling_params` in `base.yml` |
|------|-------------------|----------------------------------|
| Target (response) | `config/RGConfig.yaml` → `response_generator.generation_args` | `temperature: 0.8`, `top_p: 0.9`, `top_k: 40`, `max_tokens: 2048` |
| Mutator (prompt variants) | `config/PGConfig.yaml` → `prompt_generator.generation_args` | `temperature: 0.9`, `top_p: 0.9`, `top_k: 40`, `max_tokens: 2048` |

- vLLM uses `max_tokens`; ToxSearch-S llama.cpp uses `max_new_tokens` — same numeric cap (2048).
- `repetition_penalty` is not forwarded to vLLM (llama.cpp-only in ToxSearch-S).

**Model paths:** set `model:` (and matching `tokenizer:`) in `base.yml` to your local GGUF + HF tokenizer id, or pass `--target_llm /path/to/model.gguf` to override the **target** weights path only.

**Fitness:** the fork uses `PerspectiveScorer` (Google Perspective API); `fitness_llm` in YAML remains for reference or alternate runs.

**Other configs:** `base-opensource.yml`, `base-openai.yml`, and `eval.yml` are unchanged for upstream-style experiments.
