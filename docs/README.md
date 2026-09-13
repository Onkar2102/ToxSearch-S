# Documentation index

| Document | Audience | Contents |
|----------|----------|----------|
| [../README.md](../README.md) | Everyone | Install, run, layout, reproducibility |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Developers / researchers | Packages, evolution loop, distances, configs, outputs |
| [../tests/README.md](../tests/README.md) | Contributors | Pytest layout, markers, CI notes |
| [../results/manifests/README.md](../results/manifests/README.md) | Paper / analysis | SHA256 manifests for generated artifacts |
| [../verfier/README.md](../verfier/README.md) | Formal methods | Lean 4 audit of distance and speciation claims |
| [../data/dataset.md](../data/dataset.md) | Experiments | Shared dataset location |

## Generated Doxygen trees

`docs/html/` and `docs/latex/` are **build products** (ignored by git). Source of truth for API-style docs is the code plus this folder’s Markdown. Rebuild from the repo root with:

```bash
doxygen Doxyfile
```

(requires Doxygen; LaTeX backend may be configured in `Doxyfile`).
