# Study output manifests

Generated CSVs, figures, and JSON under `results/` are **gitignored** except this folder. Drivers stay under `experiments/`; bulky outputs do not mix with source.

## Layout

| Path | Role |
|------|------|
| `results/emnlp2026/` | Earlier EMNLP analysis artifacts |
| `results/comparison/` | Earlier comparison-report outputs |
| `results/cluster_analysis/` | Earlier Pareto figure exports |
| `results/manifests/` | Tracked SHA256 manifests + this README |

The current study drivers are under `experiments/evostar_2027/`. The three manifests above record earlier result trees.

## Restore vs regenerate

- **Restore locally:** copy a paper backup into the matching `results/…` tree, then rebuild manifests to verify checksums.
- **Regenerate:** run the drivers under `experiments/evostar_2027/`.

## Build manifests

From the repository root:

```bash
python results/manifests/build_study_manifest.py
```

Writes (or refreshes):

- `results/manifests/emnlp2026.json`
- `results/manifests/comparison.json`
- `results/manifests/cluster_analysis.json`

Each file lists relative paths, byte sizes, SHA256 digests, optional git commit, and dependency file names.
