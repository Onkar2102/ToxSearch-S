# RQ1 per-source archive robustness

Threshold $t=0.1$, batch DBSCAN MinPts=2, identical subsets per source run.

- NLI giant CC > embedding on every source archive: **True**
- NLI $D_{assigned}$ > embedding on every source archive: **True**

| run | n | dE dens | dE giant | dE D | NLI dens | NLI giant | NLI D |
|---|---:|---:|---:|---:|---:|---:|---:|
| run01_speciated | 691 | 0.0398 | 0.31 | 0.42 | 0.0086 | 0.41 | 0.70 |
| run02_speciated | 573 | 0.0126 | 0.12 | 0.18 | 0.0034 | 0.27 | 0.53 |
| run03_speciated | 642 | 0.0134 | 0.17 | 0.25 | 0.0075 | 0.45 | 0.79 |
| run04_speciated | 639 | 0.0217 | 0.35 | 0.48 | 0.0065 | 0.51 | 0.80 |
| run05_speciated | 594 | 0.0177 | 0.16 | 0.24 | 0.0086 | 0.54 | 0.81 |
| B3139_pooled | 3139 | 0.0049 | 0.11 | 0.14 | 0.0026 | 0.61 | 0.88 |

Disclosure: subset sizes differ by source run; this is not a large population of
independent datasets and does not claim universal generalization.
