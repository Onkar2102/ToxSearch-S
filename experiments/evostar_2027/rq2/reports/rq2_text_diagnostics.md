# RQ2 text diagnostics

Diagnostics from the 50 completed RQ2 runs (`data/outputs/evostar2027`, five configurations, seeds 0–9). Nothing in `notes/rough_draft` was changed.

Fitness is the stored illicit score. That score already includes the refusal penalty when a refusal was applied. Pools match the RQ2 species analysis: Leader–Follower and Global EA use `elites.json`, `reserves.json` when present, and `archive.json`; IncDBSCAN uses live `temp.json`.

## 1. Response time for fitness greater than 0.6

`response_duration` is the wall-clock time of one Llama response, in seconds, rounded to four decimals when the genome was written. Run-level `total_response_time` is only the sum, so the distribution below is rebuilt from the population files.

Of 2,318 genomes with fitness greater than 0.6, every genome has a `response_duration`. None were missing, and none were treated as zero.

| Configuration | n | Mean (s) | Median (s) | Q1 (s) | Q3 (s) | IQR (s) | Min (s) | Max (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 2318 | 17.3 | 24.3 | 3.7 | 25.2 | 21.5 | 0.15 | 28.3 |
| MiniLM LF | 478 | 17.0 | 24.7 | 3.1 | 25.3 | 22.2 | 0.15 | 27.2 |
| MiniLM Inc | 488 | 18.5 | 24.3 | 6.0 | 25.3 | 19.4 | 0.19 | 27.8 |
| NLI LF | 406 | 15.4 | 24.1 | 1.7 | 24.8 | 23.2 | 0.22 | 26.2 |
| NLI Inc | 569 | 17.1 | 24.0 | 3.1 | 24.8 | 21.8 | 0.20 | 27.1 |
| Global EA | 377 | 18.5 | 25.1 | 5.0 | 26.8 | 21.8 | 0.32 | 28.3 |

The distribution is two modes, not a single pile around the mean. In 5-second bins, 678 prompts (29%) finish in 0–5 s and 1,508 (65%) finish in 20–30 s. The 10–20 s range holds 18 prompts. The median sits in the slow mode. The mean is lower because of the fast mode. The same split appears in every configuration. The histogram is `rq2_response_duration_hist.png` (also `.pdf` and `.csv`). Per-prompt rows are `rq2_response_duration_gt06.csv`.

## 2. Perplexity

Perplexity is not stored on genomes or in the run logs. It was computed afterwards on this same fitness-greater-than-0.6 set. The generation file `Meta-Llama-3.1-8B-Instruct.Q8_0.gguf` is not on disk, so the scores use `Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf`. Per configuration: median perplexity is 10.4 (MiniLM LF), 10.8 (MiniLM Inc), 18.4 (NLI LF), 13.8 (NLI Inc), and 20.9 (Global EA). The full distributions and the correlations with fitness and length are in `rq2_prompt_perplexity.md`.

## 3. Same seed, prompt length, and species

Seed labels are the ancestry topics already stored in `topic_vocabulary_by_execution.xlsx` (generation-0 seed reached through parent links). A prompt is kept when that ancestry is exactly one seed and the stored source is ancestry, ancestry agreeing with nearest-seed text, or an archive exact match. Conflicts and multi-seed ancestries are left out. Global EA is left out because it does not form species.

Within each run and single seed, length is character length of the stored prompt. Membership uses the run’s species sizes: a mature species has size at least 2; noise and size-1 species are unassigned. The largest species is the mature species that holds the most members of that seed in that run. Topics with fewer than 8 such prompts, or with no mature species, are dropped. That leaves 71, 97, 64, and 133 topics for MiniLM LF, MiniLM Inc, NLI LF, and NLI Inc.

The table is the median, across topics, of each topic’s median length.

| Configuration | Topics | Largest species | Other mature | Unassigned | Topics with another mature species | Median share in the largest species | Median long-quartile excess | Median Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM LF | 71 | 124 | 142 | 117 | 52 | 0.25 | +0.03 | +0.07 |
| MiniLM Inc | 97 | 103 | 125 | 122 | 50 | 0.64 | −0.06 | −0.09 |
| NLI LF | 64 | 121 | 148 | 118 | 40 | 0.10 | −0.07 | −0.08 |
| NLI Inc | 133 | 116 | 116 | 119 | 4 | 0.63 | −0.06 | −0.07 |

Long-quartile excess is the share of the topic’s longest quartile that sits in the largest species, minus the share of the whole topic that sits there. A positive excess would mean long prompts pile into that species beyond the seed’s baseline. The median excess is about zero in every configuration, and the rank correlation of length with membership in the largest species is weak.

NLI Inc is the configuration where one species already absorbs a seed: only 4 of 133 topics place any member in a second mature species, and the median share in the largest species is 0.63. Length does not add to that. The three group medians are 116, 116, and 119 characters. The single-species pattern is the seed’s species assignment. It is not a length effect. The other three configurations show the same thing, with other mature species slightly longer than the largest species where a second species exists.

Per-topic rows are `rq2_length_species_topics.csv`. The summary is `rq2_length_species_summary.csv`.

## 4. What can wait for an appendix

The manuscript’s claims are niche geometry (RQ1) and, for RQ2, search under a matched evaluation budget, diversity of the top-50 prompts, and high-fitness occupancy. Response time, perplexity, length within a fixed seed, and worked examples of variation operators do not carry those claims. They are diagnostics and illustrations. They can be an appendix later. This pass does not add that appendix.

The text-variation operators named for that appendix, from the LLM-using set in `calculate_budget_metrics` (`src/utils/population_io.py`), are:

- `LLMBasedParaphrasing` / `LLMBasedParaphrasingOperator`
- `StylisticMutator`
- `LLM_POSAwareSynonymReplacement`
- `POSAwareAntonymReplacement`
- `MLM` / `MLMOperator`

RQ2 launches with the full operator set (`OPERATORS=all`). The same budget function also counts back-translation, negation, typographical errors, concept addition, semantic fusion, and informed evolution when those operators write a genome. An operator example for a later appendix is the paraphrase template in `src/ea/paraphrasing.py`: the model must return exactly one `<paraphrase>...</paraphrase>` question. The template is not copied here, and it is not added to the paper.
