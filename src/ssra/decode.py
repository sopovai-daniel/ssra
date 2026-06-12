"""Autoregressive decoding per spec §9: window ring + node store with the
retention rule.

Retention as implemented: level-0 states live in the window ring (last w+2
positions, covering both window keys [t-w, t] and the level-0 Fenwick block
at t-w-1). Internal nodes (l >= 1, odd j -- even-j nodes are consumed by
their parent within the same step and are never Fenwick blocks) are retained
for t in [end(u), end(u) + 2^l + w], the closure of Frontier membership
[end, end + 2^l - 1], parent formation at end + 2^l, and read-out use
[end + w + 1, end + 2^l + w]. NOTE: the pointwise set Frontier(t) u
Fenwick(t-w-1) of spec §9 has a gap for nodes with 2^l <= w (a node leaves
Frontier at parent formation but is still consumed by the read-out until
end(parent) + w); the interval above is the minimal correct closure, same
complexity class. Flagged in results/M1-report.md as a proposed D-log entry.

Python-level loops here are legal: D6 prohibits recursion in the *training*
path only."""

from __future__ import annotations

import math

import torch

from .fenwick import fenwick_blocks
from .model import SSRALM, SSRALayer
from .rope import rope_rotate


class _LayerState:
    def __init__(self):
        self.ring: dict[int, torch.Tensor] = {}   # pos -> z [B, d]
        self.store: dict[tuple[int, int], torch.Tensor] = {}  # (l, j) -> [B, s, d]


class IncrementalDecoder:
    """Stateful per-token decoder. step(tokens) must reproduce the full
    batched forward logits at every position (spec §14.2)."""

    def __init__(self, model: SSRALM):
        self.model = model
        self.cfg = model.cfg
        self.reset()

    def reset(self):
        self.t = 0
        self.states = [_LayerState() for _ in self.model.layers]

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B] int64 (position t = current step) -> logits [B, vocab]."""
        self.t += 1
        t = self.t
        if t > self.cfg.n_max:
            raise ValueError(f"t={t} exceeds n_max={self.cfg.n_max}")
        x = self.model.emb(tokens)  # [B, d]
        for layer, state in zip(self.model.layers, self.states):
            z = layer.ln1(x)
            x = x + self._readout_step(layer, state, z, t)
            x = x + layer.ffn(layer.ln2(x))
            self._tree_update(layer, state, z, t)
        return self.model.lm_head(self.model.ln_f(x))

    # ---- read-out over W(t) and Fenwick(t-w-1) -------------------------------

    def _readout_step(self, layer: SSRALayer, state: _LayerState,
                      z: torch.Tensor, t: int) -> torch.Tensor:
        cfg, attn = self.cfg, layer.readout_attn
        w, base = cfg.w, cfg.rope_base
        dev = z.device

        q = attn.split_heads(attn.w_q(z.unsqueeze(1)))  # [B, h, 1, dh]
        q = rope_rotate(q, torch.tensor([t], device=dev), base)

        win_pos = list(range(max(1, t - w), t))  # current z appended below
        win = [state.ring[p] for p in win_pos] + [z]
        k_in = torch.stack(win, dim=1)  # [B, n_win, d]
        pos = torch.tensor(win_pos + [t], device=dev)
        k = rope_rotate(attn.split_heads(attn.w_k(k_in)), pos, base)
        v = attn.split_heads(attn.w_v(k_in))

        blocks = fenwick_blocks(t - w - 1)
        if blocks:
            key_rows, val_rows = [], []
            for level, j in blocks:
                s_u = (state.ring[j].unsqueeze(1) if level == 0
                       else state.store[(level, j)])
                tagged = (s_u + layer.level_emb[level]
                          if cfg.level_emb == "on" else s_u)
                key_rows.append(tagged)
                val_rows.append(s_u)
            k_sum = attn.split_heads(attn.w_k(torch.cat(key_rows, 1)))
            if cfg.summary_pos == "virtual":
                k_sum = rope_rotate(
                    k_sum, torch.tensor([t - w - 1], device=dev), base)
            k = torch.cat([k, k_sum], dim=-2)
            v = torch.cat([v, attn.split_heads(attn.w_v(torch.cat(val_rows, 1)))],
                          dim=-2)

        probs = (q @ k.transpose(-1, -2) / math.sqrt(attn.d_h)).softmax(-1)
        out = attn.merge_heads(probs @ v)  # [B, 1, d]
        return attn.w_o(out).squeeze(1)

    # ---- tree maintenance ----------------------------------------------------

    def _tree_update(self, layer: SSRALayer, state: _LayerState,
                     z: torch.Tensor, t: int) -> None:
        cfg = self.cfg
        state.ring[t] = z
        state.ring.pop(t - cfg.w - 2, None)  # keep positions >= t-w-1

        # form parents while the rightmost node's sibling is complete (§9)
        aux: dict = {}
        level, j = 0, t
        while j % 2 == 0:
            level, j = level + 1, j // 2
            c1 = self._child(state, level - 1, 2 * j - 1)
            c2 = self._child(state, level - 1, 2 * j)
            x = torch.cat([c1, c2], dim=1)
            state.store[(level, j)] = layer.node_step(x, level, aux)

        # evict: retain (l, j odd) while t <= end + 2^l + w; even-j nodes are
        # consumed within their formation step
        for (lvl, jj) in list(state.store):
            end = jj * (1 << lvl)
            if jj % 2 == 0 or end + (1 << lvl) + cfg.w < t:
                del state.store[(lvl, jj)]

    def _child(self, state: _LayerState, level: int, j: int) -> torch.Tensor:
        if level == 0:
            return state.ring[j].unsqueeze(1)
        return state.store[(level, j)]


@torch.no_grad()
def decode_logits(model: SSRALM, tokens: torch.Tensor) -> torch.Tensor:
    """Run the incremental decoder over a full sequence [B, N]; returns
    logits [B, N, vocab] for direct comparison with model(tokens)."""
    dec = IncrementalDecoder(model)
    return torch.stack([dec.step(tokens[:, i]) for i in range(tokens.shape[1])],
                       dim=1)
