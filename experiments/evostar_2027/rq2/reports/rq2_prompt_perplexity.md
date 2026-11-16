# Prompt perplexity for fitness greater than 0.6

Perplexity of the raw prompt text under Meta-Llama-3.1-8B-Instruct.
The runs were generated with `Meta-Llama-3.1-8B-Instruct.Q8_0.gguf`.
That file is not on disk, so these scores use `Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf`, the finest copy of the same model that is present.
No other model family was substituted, and no chat template was applied.

Each prompt is tokenized with a beginning-of-sequence token.
That token is context only.
Perplexity is exp(mean negative log-likelihood) of the remaining tokens.
Prompts longer than 4096 tokens are scored in windows of 4096 with stride 2048, each token once.
The set is the 2,318 genomes with refusal-penalized fitness greater than 0.6 from the 50 RQ2 runs.

Per-prompt scores: `rq2_prompt_perplexity_gt06.csv`.
Histogram: `rq2_prompt_perplexity_hist.png`.

## Overall

2318 prompts. Median 14.53, IQR 35.70 (Q1 5.13, Q3 40.83). The mean is 44.76 because a few prompts sit far above the rest (maximum 5171). Compare configurations by the median.
7 prompts needed more than one context window.
Spearman correlation with fitness is -0.266. Spearman correlation with character length is -0.898, and with token length -0.903. Longer prompts are the predictable ones. Fitness moves perplexity only weakly, and in the same direction: higher fitness, lower perplexity.

| Configuration | n | Mean | Median | Q1 | Q3 | IQR | Min | Max | Windows > 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 2318 | 44.76 | 14.53 | 5.13 | 40.83 | 35.70 | 1.16 | 5171.39 | 7 |
| MiniLM LF | 478 | 31.93 | 10.40 | 4.99 | 33.47 | 28.47 | 1.16 | 569.70 | 5 |
| MiniLM Inc | 488 | 58.75 | 10.77 | 3.29 | 39.90 | 36.61 | 1.17 | 1239.13 | 2 |
| NLI LF | 406 | 55.56 | 18.41 | 2.81 | 41.16 | 38.35 | 1.24 | 5171.39 | 0 |
| NLI Inc | 569 | 28.29 | 13.76 | 6.73 | 34.23 | 27.50 | 1.40 | 380.02 | 0 |
| Global EA | 377 | 56.16 | 20.87 | 6.47 | 62.26 | 55.79 | 1.98 | 526.07 | 0 |

| Configuration | Spearman fitness | Pearson fitness | Spearman characters | Spearman tokens |
| --- | ---: | ---: | ---: | ---: |
| Overall | -0.266 | -0.100 | -0.898 | -0.903 |
| MiniLM LF | -0.422 | -0.248 | -0.927 | -0.938 |
| MiniLM Inc | -0.179 | -0.051 | -0.944 | -0.951 |
| NLI LF | -0.234 | -0.066 | -0.830 | -0.856 |
| NLI Inc | -0.250 | -0.172 | -0.816 | -0.827 |
| Global EA | -0.259 | -0.194 | -0.910 | -0.862 |

## MiniLM LF

478 prompts with fitness greater than 0.6.
Mean perplexity 31.93, median 10.40, IQR 28.47 (Q1 4.99, Q3 33.47), range 1.16–569.70.
5 prompts were longer than 4096 tokens and were scored across a second window.
Correlation with fitness: Spearman -0.422, Pearson -0.248.
Correlation with prompt length: Spearman -0.927 on characters, -0.938 on tokens.

## MiniLM Inc

488 prompts with fitness greater than 0.6.
Mean perplexity 58.75, median 10.77, IQR 36.61 (Q1 3.29, Q3 39.90), range 1.17–1239.13.
2 prompts were longer than 4096 tokens and were scored across a second window.
Correlation with fitness: Spearman -0.179, Pearson -0.051.
Correlation with prompt length: Spearman -0.944 on characters, -0.951 on tokens.

## NLI LF

406 prompts with fitness greater than 0.6.
Mean perplexity 55.56, median 18.41, IQR 38.35 (Q1 2.81, Q3 41.16), range 1.24–5171.39.
Every prompt fit in one context window.
Correlation with fitness: Spearman -0.234, Pearson -0.066.
Correlation with prompt length: Spearman -0.830 on characters, -0.856 on tokens.

## NLI Inc

569 prompts with fitness greater than 0.6.
Mean perplexity 28.29, median 13.76, IQR 27.50 (Q1 6.73, Q3 34.23), range 1.40–380.02.
Every prompt fit in one context window.
Correlation with fitness: Spearman -0.250, Pearson -0.172.
Correlation with prompt length: Spearman -0.816 on characters, -0.827 on tokens.

## Global EA

377 prompts with fitness greater than 0.6.
Mean perplexity 56.16, median 20.87, IQR 55.79 (Q1 6.47, Q3 62.26), range 1.98–526.07.
Every prompt fit in one context window.
Correlation with fitness: Spearman -0.259, Pearson -0.194.
Correlation with prompt length: Spearman -0.910 on characters, -0.862 on tokens.
