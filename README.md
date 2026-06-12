# SSRA — Scale-Shared Recursive Attention

Research code and design documentation. **Private until publication.**

A causal language-modeling attention block in which a single shared
(attention + pooling) rule generates the entire scale hierarchy over the
sequence — from tokens to the tree root, including the token-level read-out.

## Status
- Phase: M1 implementation & verification (spec v1.0, G0 passed; see `results/M1-report.md`).
- Single source of truth for project state: `docs/00-stav-a-triaz.md`
- Implementation spec: `docs/spec.md`
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

## How to run tests and benchmarks
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -e .

# M1 verification suite (judged on CPU fp32; spec §14.1-.3, .7)
.venv/bin/python -m pytest tests/

# informative MPS re-run of the same checks (AP-2)
.venv/bin/python scripts/run_mps_informative.py

# test 4: throughput/VRAM curves SSRA vs flat (G1a), MPS
.venv/bin/python scripts/bench_throughput.py

# smoke run (one YAML in experiments/ = one run, committed before launch)
.venv/bin/python scripts/train.py experiments/M1-smoke-p1.yaml
```

## Reproduction
Will be a single command (`scripts/repro.sh`) by publication time.

## License
Apache-2.0 (see `LICENSE`).
