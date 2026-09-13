# Study output manifests

Generated CSVs, figures, and JSON under `results/` are **gitignored** except this folder. Drivers stay under `experiments/`; bulky outputs do not mix with source.

## Layout

| Path | Role |
|------|------|
| `results/emnlp2026/` | EMNLP / multi-phase analysis artifacts |
| `results/comparison/` | C1 / C2 / C3 comparison report outputs |
| `results/cluster_analysis/` | Cluster / Pareto figure exports |
| `results/manifests/` | Tracked SHA256 manifests + this README |

Default analysis `RESULTS_DIR` points at `results/emnlp2026` (see `experiments/emnlp_data_analysis/analysis_utils.py`). Comparison report scripts write under `results/comparison/<study>/`.

## Restore vs regenerate

- **Restore locally:** copy a paper backup into the matching `results/…` tree, then rebuild manifests to verify checksums.
- **Regenerate:** run drivers in `experiments/` (EMNLP phase scripts, `*_report.py` under `comparison_results/`, cluster analysis tools).

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
