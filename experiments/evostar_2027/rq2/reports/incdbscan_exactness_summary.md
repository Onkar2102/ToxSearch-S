# IncDBSCAN exactness audit

Compares grow-only IncDBSCAN partitions against classical batch DBSCAN.
At MinPts=2 the methods must match (connected-component niching).
At MinPts≥3 the live insert path can under-count mature clusters relative to batch.

## Archive (calibration matrices)

| distance | t | MinPts | inc n_mat | batch n_mat | ARI | exact |
|---|---:|---:|---:|---:|---:|---:|
| d_E | 0.05 | 2 | 237 | 237 | 1.0000 | 0 |
| d_E | 0.05 | 3 | 164 | 177 | 0.9652 | 0 |
| d_E | 0.05 | 5 | 74 | 94 | 0.9073 | 0 |
| d_E | 0.1 | 2 | 167 | 167 | 1.0000 | 0 |
| d_E | 0.1 | 3 | 128 | 139 | 0.9669 | 0 |
| d_E | 0.1 | 5 | 88 | 98 | 0.9467 | 0 |
| d_E | 0.15 | 2 | 83 | 83 | 1.0000 | 0 |
| d_E | 0.15 | 3 | 64 | 68 | 0.9972 | 0 |
| d_E | 0.15 | 5 | 49 | 52 | 0.9784 | 0 |
| d_negE | 0.05 | 2 | 107 | 107 | 1.0000 | 0 |
| d_negE | 0.05 | 3 | 38 | 46 | 0.9857 | 0 |
| d_negE | 0.05 | 5 | 13 | 19 | 0.9814 | 0 |
| d_negE | 0.1 | 2 | 86 | 86 | 1.0000 | 0 |
| d_negE | 0.1 | 3 | 32 | 38 | 0.9910 | 0 |
| d_negE | 0.1 | 5 | 10 | 13 | 0.9891 | 0 |
| d_negE | 0.15 | 2 | 66 | 66 | 1.0000 | 0 |
| d_negE | 0.15 | 3 | 23 | 28 | 0.9944 | 0 |
| d_negE | 0.15 | 5 | 9 | 11 | 0.9878 | 0 |

## Online (final live graphs)

| cell | seed | MinPts | live n_mat | batch n_mat | ARI | exact |
|---|---:|---:|---:|---:|---:|---:|
| inc_embedding | 0 | 2 | 69 | 69 | 1.0000 | 1 |
| inc_embedding | 1 | 2 | 49 | 49 | 1.0000 | 1 |
| inc_embedding | 2 | 2 | 51 | 51 | 1.0000 | 1 |
| inc_embedding | 3 | 2 | 57 | 57 | 1.0000 | 1 |
| inc_embedding | 4 | 2 | 53 | 53 | 1.0000 | 1 |
| inc_embedding | 5 | 2 | 40 | 40 | 1.0000 | 1 |
| inc_embedding | 6 | 2 | 50 | 50 | 1.0000 | 1 |
| inc_embedding | 7 | 2 | 61 | 61 | 1.0000 | 1 |
| inc_embedding | 8 | 2 | 91 | 91 | 1.0000 | 1 |
| inc_embedding | 9 | 2 | 63 | 63 | 1.0000 | 1 |
| inc_nli | 0 | 5 | 2 | 3 | 0.9961 | 0 |
| inc_nli | 1 | 5 | 1 | 6 | 0.9777 | 0 |
| inc_nli | 2 | 5 | 3 | 4 | 0.9952 | 0 |
| inc_nli | 3 | 5 | 1 | 3 | 0.9951 | 0 |
| inc_nli | 4 | 5 | 1 | 5 | 0.9794 | 0 |
| inc_nli | 5 | 5 | 1 | 4 | 0.9920 | 0 |
| inc_nli | 6 | 5 | 2 | 3 | 0.9966 | 0 |
| inc_nli | 7 | 5 | 3 | 5 | 0.9970 | 0 |
| inc_nli | 8 | 5 | 3 | 10 | 0.9742 | 0 |
| inc_nli | 9 | 5 | 8 | 11 | 0.9888 | 0 |

## Online ARI summary (median [min, max])

- `inc_embedding`: 1.0000 [1.0000, 1.0000] (n=10)
- `inc_nli`: 0.9935 [0.9742, 0.9970] (n=10)

## Caveat

The online comparison applies batch DBSCAN to the *final* ε-graph and
therefore bounds final-partition disagreement. It does not replay the
evolutionary trajectory under exact IncrementalDBSCAN updates.

Regenerated from CSV by `validation/mandatory_checks.py`.
