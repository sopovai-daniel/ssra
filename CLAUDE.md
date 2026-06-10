# CLAUDE.md — instructions for Claude Code in the ssra repo

## What this repo is
SSRA (Scale-Shared Recursive Attention) research project: a causal LM attention
block where one shared (attention + pooling) rule generates the whole scale
hierarchy, including the token-level read-out. Goal: PoC with matched-compute
baselines and an open publication (positive OR negative result).

## Sources of truth (in this order)
1. `docs/spec.md` — implementation spec. If it exists, code follows it exactly.
2. `docs/00-stav-a-triaz.md` — project state and decision log (D-log).
3. `docs/01-mechanizmus.md` — design rationale, complexity derivations (§7).
Until spec.md exists, implement nothing beyond scaffolding and tests.

## Hard rules
- **No silent design changes.** If the spec is ambiguous or seems wrong, STOP and
  write the question + proposed D-log entry into the task report. Never "improve"
  the architecture on your own. Design decisions live in the Claude.ai project,
  not here.
- **No fabrication.** No invented APIs, prices, citations, benchmark numbers.
  Any performance claim must point to a committed log or script output.
- **Causality is test-first.** `tests/test_shift.py` (changing token t must not
  change logits at positions < t) and `tests/test_completion.py` (incremental
  decoding logits == full-forward logits) must pass before any training code is
  trusted. Treat a red causality test as a release blocker.
- **Level-wise batching only (D6).** No Python recursion over tree branches.
  Nodes of one level are processed as one batched tensor.
- **Run discipline.** One run = one YAML in `experiments/` committed BEFORE
  launch + a row appended to `results/runs.md`. A run without a committed config
  does not exist.
- **History will become public.** No secrets, no client references, no personal
  data — anywhere, including commit messages. Commit messages in English,
  imperative mood. Commits should be signed (developer's key).
- **Language:** code, comments, commit messages, identifiers — English.
  Documents in `docs/` — Slovak with diacritics. Do not translate docs.

## Architecture quick reference (details in docs/01 §5, §7)
- Binary tree (k=2), pass-through schedule s_l = min(2^l, m), m = 16.
- Node block: pre-norm residual, bidirectional attention INSIDE the node
  (legal — structural gating guarantees causality), slot-RoPE positions 1..2m,
  additive level embedding e_l.
- Pool_phi operators behind one interface: P1 latent-query (default),
  P2 strided merge (control), P3 top-(m-1) selection + context residual
  (challenger; STE/Gumbel in training only).
- Read-out (variant A): token attends to exact window w=64 plus summaries of
  the <= log2(t) completed Fenwick nodes covering the prefix; read-out shares
  theta with node blocks.
- Complexity targets (must hold empirically): training Theta(N*(w + m*log N)*d)
  per layer; decode state O(m*d*log N) per layer; throughput curve must grow
  sub-quadratically (Gate G1a).

## Workflow with the Claude.ai project
Tasks arrive as "Zadanie pre CC" blocks: goal, spec references, acceptance
criteria, anti-goals. After finishing: commit, then write a short run/task
report into `results/` (what was done, test status, open questions). The
Claude.ai side reads reports via Filesystem MCP and answers design questions
through D-log updates.
