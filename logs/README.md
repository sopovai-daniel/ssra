# Logs
Versioned sanity/run log artifacts. First expected entry: T0 — output log of
`scripts/v1_legacy.py` (training loop on a dummy corpus; expect 5 epochs of
decreasing loss). NOTE: the loss decrease is a target-leakage artifact (no
causal mask), not a learning signal — see docs/00, [OVERENÉ] section.
