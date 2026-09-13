# ToxSearch-S Lean verifier (`verfier/`)

Lean 4 + Mathlib formalization of the mathematical core of ToxSearch-S
(ensemble distance, config well-formedness, NSGA-II dominance, speciation
invariants, refusal penalty).

The directory name `verfier/` is historical; Python docs and imports still
point here. For the runtime architecture, see [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

See:

- [AUDIT.md](AUDIT.md) — claim-by-claim correct / incorrect / empirical-only verdicts
- [CORRECTIONS.md](CORRECTIONS.md) — **how to correct** each incorrect claim mathematically
  (replacement statements + Lean theorem names)

## Requirements

- [elan](https://github.com/leanprover/elan) (Lean version manager)
- Network access the first time (downloads Lean `v4.22.0` and Mathlib)

```bash
# If needed:
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --no-modify-path
export PATH="$HOME/.elan/bin:$PATH"
```

## Build

From this directory:

```bash
lake update
lake exe cache get   # optional but much faster
lake build
```

Pinned toolchain: `leanprover/lean4:v4.22.0` (see `lean-toolchain`).
Mathlib tag: `v4.22.0`.

## Layout

| Path | Role |
|------|------|
| `ToxSearch/Distance.lean` | Distance definitions |
| `ToxSearch/DistanceProperties.lean` | Range / symmetry / zero-characterization |
| `ToxSearch/Inframetric.lean` | Cosine not a metric; 2-relaxed triangle |
| `ToxSearch/Config.lean` | `SpeciationConfig` well-formedness |
| `ToxSearch/NSGA2.lean` | Pareto dominance + capacity selection |
| `ToxSearch/SpeciationSpec.lean` | Abstract species / population invariants |
| `ToxSearch/RefusalPenalty.lean` | `0.85` fitness multiplier |
| `AUDIT.md` | Human-readable validation report |
| `CORRECTIONS.md` | Mathematical fixes for incorrect / overstated claims |
