# SSRA — Scale-Shared Recursive Attention

Research record of a pre-registered proof-of-concept: a causal language-modeling
attention block in which a single weight-shared (attention + pooling) rule per
layer generates the entire scale hierarchy over the sequence — from tokens to
the tree root, including the token-level read-out.

**The empirical outcome is negative on every pre-registered axis.** Either
outcome was declared publishable in advance; this repository is the public,
append-only methodology record backing both publications.

## Publications

1. **Technical note** (stage 1 — design, causality proof, derived complexity,
   falsification plan, fixed before any training run):
   DOI [10.5281/zenodo.20647034](https://doi.org/10.5281/zenodo.20647034)
2. **Results paper** (stage 2 — pre-registered empirical evaluation, negative
   results): DOI [10.5281/zenodo.21439493](https://doi.org/10.5281/zenodo.21439493)
   — PDF: `paper/ssra-results-paper-v1.0.pdf`, source: `paper/results-paper.md`

Headline results (stated plainly): +10.22 % validation-perplexity gap against a
flat Transformer at the training context length 1,024 (outside the
pre-registered ±5 % parity band); an empirically narrower stable learning-rate
range at this scale; the length-extrapolation prior violated from N = 2,048
with no crossover at any measured length; 0 % needle-lite retrieval in every
cell, subject to a pre-registered caveat. See the paper for the full protocol
and diagnostics.

## Methodology artifacts

- `docs/spec.md` — frozen implementation specification (v1.2), normative for all code
- `docs/00-stav-a-triaz.md` — append-only decision log (single source of truth)
- `docs/cc/` — pre-registered assignments committed before each run
- `experiments/` — one YAML config per run, committed before launch
- `logs/` — raw JSONL training logs incl. per-step diagnostics
- `tests/` — verification suite (shift, completion, frozen-reference A/B, gradient-flow)
- `results/runs.md` — append-only run ledger incl. costs

## Layout

```
docs/         design docs (00-03), spec.md, handover series, CC assignments
paper/        results paper (source + published PDF + export source)
src/ssra/     implementation (level-wise batched core, pool operators, read-out)
baselines/    flat Transformer (the §4 comparison); loglinear/megabyte modules
              not exercised in the reported runs
tests/        shift test, completion test, gradient-flow checks
experiments/  one YAML config per run (committed BEFORE launch)
results/      runs.md ledger, curves, tables, figures
scripts/      training, evaluation, benchmarking, plotting
logs/         versioned run/sanity log artifacts
```

## How to run tests and benchmarks

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -e .

# verification suite (CPU fp32; spec §14)
.venv/bin/python -m pytest tests/

# informative MPS re-run of the same checks
.venv/bin/python scripts/run_mps_informative.py

# throughput curves SSRA vs flat
.venv/bin/python scripts/bench_throughput.py
```

## Reproduction

Each run is fully specified by its committed YAML: `scripts/train.py
experiments/<run>.yaml`. Length-extrapolation and needle-lite evaluations:
`scripts/g2lite_eval.py` (see the paper, §3.6 and §4.7–4.8). Checkpoints and
raw evaluation outputs are mirrored in object storage and are not part of this
repository.

## License

Code: Apache-2.0 (see `LICENSE`). Paper text and documentation: CC BY 4.0.
