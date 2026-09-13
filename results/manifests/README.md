# Study output manifests

Generated CSVs, figures, and JSON under `results/` are **gitignored** except this folder.

## Restore vs regenerate

- **Restore locally:** copy artifacts from paper backup or run `build_study_manifest.py` after a local `results/` tree exists to verify checksums.
- **Regenerate:** run the driver scripts in `experiments/` (comparison `*_report.py`, EMNLP analysis phases). Paths default to `results/emnlp2026`, `results/comparison/*`, and `results/cluster_analysis/`.

## Build manifests

From repository root:

```bash
python results/manifests/build_study_manifest.py
```

Writes `results/manifests/<study>.json` with SHA256 hashes and file sizes.
