"""Fenwick (binary indexed tree) prefix decomposition (spec §8) and the
gather-index table for the batched read-out."""

from __future__ import annotations

import torch


def fenwick_blocks(p: int) -> list[tuple[int, int]]:
    """Standard BIT decomposition of prefix [1, p] into aligned blocks.

    Returns [(level, j)] with node (level, j) spanning
    [(j-1)*2^level + 1, j*2^level]; empty if p <= 0. Emitted right-to-left.
    """
    blocks = []
    while p > 0:
        b = p & (-p)
        level = b.bit_length() - 1
        blocks.append((level, p >> level))
        p -= b
    return blocks


def level_row_layout(n: int, s_l) -> tuple[list[int], int]:
    """Row offsets of each level inside the flattened summary table.

    Level l contributes floor(n / 2^l) nodes x s_l(l) rows, levels 0..floor(log2 n).
    Returns (offsets per level, total rows).
    """
    offsets, off = [], 0
    level, nodes = 0, n
    while nodes >= 1:
        offsets.append(off)
        off += nodes * s_l(level)
        level += 1
        nodes = n >> level
    return offsets, off


def build_readout_index(n: int, w: int, s_l, device=None):
    """Per-position gather indices into the flattened summary table.

    For 1-indexed position t the Fenwick key set is the rows of all nodes in
    fenwick_blocks(t - w - 1). Returns (idx [n, k_max] long, mask [n, k_max]
    bool valid, vpos [n] long = max(t-w-1, 0) for summary_pos=virtual).
    """
    offsets, _ = level_row_layout(n, s_l)
    rows_per_t: list[list[int]] = []
    for t in range(1, n + 1):
        rows: list[int] = []
        for level, j in fenwick_blocks(t - w - 1):
            s = s_l(level)
            base = offsets[level] + (j - 1) * s
            rows.extend(range(base, base + s))
        rows_per_t.append(rows)
    k_max = max(1, max(len(r) for r in rows_per_t))
    idx = torch.zeros(n, k_max, dtype=torch.long, device=device)
    mask = torch.zeros(n, k_max, dtype=torch.bool, device=device)
    for i, rows in enumerate(rows_per_t):
        if rows:
            idx[i, : len(rows)] = torch.tensor(rows, dtype=torch.long, device=device)
            mask[i, : len(rows)] = True
    vpos = torch.clamp(torch.arange(1, n + 1, device=device) - w - 1, min=0)
    return idx, mask, vpos
