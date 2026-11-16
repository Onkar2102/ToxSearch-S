# Study output manifests

Generated files under `results/` are gitignored except this folder. The current study reports live under `experiments/evostar_2027/`.

`build_study_manifest.py` writes a SHA256 manifest for each directory listed in `STUDY_ROOTS`. That list is empty until a new result tree should be tracked.
