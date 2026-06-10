# SSRA — Scale-Shared Recursive Attention

Research code and design documentation. **Private until publication.**

A causal language-modeling attention block in which a single shared
(attention + pooling) rule generates the entire scale hierarchy over the
sequence — from tokens to the tree root, including the token-level read-out.

## Status
- Phase: M0 design closed (D1–D6, Q1–Q5). No training has been run yet.
- Single source of truth for project state: `docs/00-stav-a-triaz.md`
- Implementation spec (once it exists): `docs/spec.md`
- Development history: `docs/handover/`

## Layout
```
docs/         design docs (00-03), spec.md, handover series
src/ssra/     implementation (level-wise batched core, pool operators, read-out)
baselines/    flat Transformer, Log-Linear GatedDeltaNet, MEGABYTE-style
tests/        shift test, completion test, gradient-flow checks
experiments/  one YAML config per run (committed BEFORE launch)
results/      runs.md ledger, curves, tables
scripts/      reproduction and benchmarking scripts
logs/         versioned run/sanity log artifacts
```

## Reproduction
Will be a single command (`scripts/repro.sh`) by publication time.

## License
Apache-2.0 (see `LICENSE`).
