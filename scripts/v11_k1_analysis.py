"""V11 K1-analysis: T-A...T-E + C-T1 verdict from the committed NPZ extracts.

Governing: docs/cc/V11-data-exploitation.md v1 SS5 (pre-registered metrics and
criteria, unchanged). Execution supplement: docs/cc/V11-k1-analysis.md v1
(operationalization rules [OP], fixed before any result was computed).
Pure tensor statistics over committed inputs - no forward passes, no
gradients, no GPU, no GCS (SS6 anti-goals).

Pinned inputs:
  results/v11/v11-k1-extract-ssra.npz   (52 steps x 393 named params + full
                                         phi/e_l tensors, S_min reference)
  results/v11/v11-k1-extract-flat.npz   (52 steps x 183 named params)
  logs/m2-core-ssra-s2-850m-lr6e4.log   (per-step `lr` field, T-D overlay)
  logs/m2-core-flat-s2-850m-lr6e4.log

Binding rules (supplement SS2): dedup by meta_json.alias_groups (canonical =
first member of each sorted group), filter to meta_json.trainable, reference
exclusively S_min = step 1000 (init DROPPED - pre-registered branch fired);
assert |population| = 273 (ssra) / 183 (flat) before computing anything.

Outputs (AP-21, no overwrites):
  results/figures/v11/v11-k1-ta-ssra.png      T-A  ssra class trajectories
  results/figures/v11/v11-k1-ta-flat.png      T-E  flat class trajectories
  results/figures/v11/v11-k1-tb-phi-ssra.png  T-B  phi series vs step
  results/figures/v11/v11-k1-rho-ssra.png     C-T1 rho strip plot
  results/figures/v11/v11-k1-tc-levelemb-ssra.png  T-C rows 0-10 norms
  results/figures/v11/v11-k1-td-ssra.png      T-D  rel update rate + lr
  results/figures/v11/v11-k1-td-flat.png      T-E  flat T-D + lr
  results/v11/v11-k1-rho.csv                  full population, both arms
  results/v11/v11-k1-analysis-summary.json    every reported number

Usage: .venv/bin/python scripts/v11_k1_analysis.py
Any assert failure = STOP and report (supplement SS5 gate 1).
Observations only; no architecture conclusions (spec SS16).
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures" / "v11"
OUT_DIR = ROOT / "results" / "v11"

EXPECTED_STEPS = list(range(1000, 52000, 1000)) + [51880]
EXPECTED = {  # supplement SS1/SS5 gate 1 (verified 2026-07-20; assert, not re-derive)
    "ssra": {"P": 393, "alias_groups": 60, "population": 273},
    "flat": {"P": 183, "alias_groups": 0, "population": 183},
}
LOGS = {  # committed run JSONL logs of the Phase 3b core pair (T-D lr source)
    "ssra": ROOT / "logs" / "m2-core-ssra-s2-850m-lr6e4.log",
    "flat": ROOT / "logs" / "m2-core-flat-s2-850m-lr6e4.log",
}
CROSS_CHECK_RTOL = 1e-9  # T-B gate (supplement SS3)

# Module-class mapping (T-A): ordered, first match wins; asserted to be an
# exact partition of the population. Printed verbatim in the report.
CLASS_TABLE = [
    ("emb", r"^emb\.weight$"),  # tied embeddings = output head (config)
    ("pool.latent_q", r"^layers\.\d+\.pool\.latent_q$"),
    ("pool.ln", r"^layers\.\d+\.pool\.ln_pool\.(weight|bias)$"),
    ("level_emb", r"^layers\.\d+\.level_emb$"),
    ("attn", r"^layers\.\d+\.attn\."),
    ("ffn", r"^layers\.\d+\.ffn\."),
    ("ln", r"^layers\.\d+\.(ln1|ln2|ln_node)\."),
    ("ln_f", r"^ln_f\.(weight|bias)$"),
]
# Okabe-Ito (colorblind-safe), fixed per class across ALL figures/arms.
CLASS_COLOR = {
    "emb": "#0072B2", "attn": "#E69F00", "ffn": "#009E73", "ln": "#56B4E9",
    "ln_f": "#CC79A7", "pool.latent_q": "#D55E00", "pool.ln": "#F0E442",
    "level_emb": "#000000",
}


def class_of(name: str) -> str:
    for cls, pat in CLASS_TABLE:
        if re.search(pat, name):
            return cls
    raise AssertionError(f"population key matches no module class: {name}")


def load_arm(arm: str) -> dict:
    z = np.load(OUT_DIR / f"v11-k1-extract-{arm}.npz")
    meta = json.loads(str(z["meta_json"]))
    names = [str(n) for n in z["param_names"]]
    exp = EXPECTED[arm]

    # --- supplement SS5 gate 1: input asserts (STOP on failure) ---
    assert list(z["steps"]) == EXPECTED_STEPS, f"{arm}: step set mismatch"
    assert len(names) == exp["P"], f"{arm}: P {len(names)} != {exp['P']}"
    assert meta["arm"] == arm and meta["s_min_step"] == 1000
    assert np.all(z["delta_ref_l2"][0] == 0.0), \
        f"{arm}: delta_ref_l2[0] not exactly 0.0 (S_min reference row)"
    assert meta["init_validated"] is False, \
        f"{arm}: init_validated {meta['init_validated']!r} != False " \
        f"(SS2 rule 3 presumes the recorded drop verdict)"
    groups = meta["alias_groups"] or []
    assert len(groups) == exp["alias_groups"], \
        f"{arm}: {len(groups)} alias groups != {exp['alias_groups']}"
    assert all(g == sorted(g) and len(g) > 1 for g in groups)

    # --- SS2 rules 1+2: dedup by alias_groups, filter to trainable ---
    canonical = set(names)
    alias_members = {}  # canonical -> full group
    for g in groups:
        alias_members[g[0]] = g
        canonical.difference_update(g[1:])
    trainable = meta["trainable"]
    assert len(trainable) == exp["population"], \
        f"{arm}: trainable list {len(trainable)} != {exp['population']}"
    population = [n for n in trainable if n in canonical]
    assert len(population) == exp["population"], \
        f"{arm}: |population| {len(population)} != {exp['population']} - STOP"

    col = {n: i for i, n in enumerate(names)}
    classes = {n: class_of(n) for n in population}
    counts = {c: sum(1 for v in classes.values() if v == c)
              for c, _ in CLASS_TABLE}
    assert sum(counts.values()) == len(population)
    return {"arm": arm, "z": z, "meta": meta, "names": names, "col": col,
            "population": population, "alias_members": alias_members,
            "classes": classes, "class_counts": counts,
            "steps": np.array(EXPECTED_STEPS)}


def class_agg(d: dict, arr2d: np.ndarray, cls: str) -> np.ndarray:
    """[OP] class aggregate = sqrt(sum of squared per-tensor values), i.e.
    the norm of the concatenation, over POPULATION tensors only (alias
    duplicates never double-counted)."""
    idx = [d["col"][n] for n in d["population"] if d["classes"][n] == cls]
    return np.sqrt(np.sum(arr2d[:, idx].astype(np.float64) ** 2, axis=1))


def arm_classes(d: dict) -> list[str]:
    return [c for c, _ in CLASS_TABLE if d["class_counts"][c] > 0]


def load_lr(arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-step lr from the committed run JSONL log; field name `lr` in the
    train records (verified; logged every 25 steps)."""
    steps, lrs = [], []
    with open(LOGS[arm], errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and "lr" in rec and "step" in rec:
                steps.append(rec["step"])
                lrs.append(rec["lr"])
    assert steps, f"{arm}: no lr records in {LOGS[arm]}"
    return np.array(steps), np.array(lrs)


def style_ax(ax):
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


# ---- T-A / T-E: class norm + delta trajectories --------------------------

def fig_ta(d: dict):
    arm, steps = d["arm"], d["steps"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for cls in arm_classes(d):
        c = CLASS_COLOR[cls]
        n = d["class_counts"][cls]
        lab = f"{cls} ({n})"
        ax1.plot(steps, class_agg(d, d["z"]["l2"], cls), color=c, lw=2,
                 label=lab)
        # delta_ref row 0 is exactly 0 (S_min reference) - skipped on log y
        ax2.plot(steps[1:], class_agg(d, d["z"]["delta_ref_l2"], cls)[1:],
                 color=c, lw=2, label=lab)
    for ax, ttl in ((ax1, "L2 norm"), (ax2, "L2 delta to S_min (step 1000)")):
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_title(ttl, fontsize=11)
        style_ax(ax)
    ax1.set_ylabel("class L2 (norm of concatenation)")
    ax1.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"V11 K1 T-{'A' if arm == 'ssra' else 'E'}: {arm} arm — "
                 f"module-class trajectories ({len(d['population'])} tensors)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"v11-k1-ta-{arm}.png", dpi=150)
    plt.close(fig)


# ---- T-B: phi series from full tensors + cross-check gate ----------------

def tb_cross_check(d: dict) -> dict:
    """Recompute Frobenius norm / delta-to-S_min / cos-to-S_min from the
    full/* tensors and compare to the l2 / delta_ref_l2 / cos_ref columns
    (rel tol 1e-9); mismatch = STOP (supplement SS3)."""
    z, col = d["z"], d["col"]
    phi_keys = sorted(k[len("full/"):] for k in z.files
                      if k.startswith("full/"))
    assert len(phi_keys) == 60, f"expected 60 full/ keys, got {len(phi_keys)}"
    worst = 0.0
    series = {}
    for k in phi_keys:
        full = z[f"full/{k}"].astype(np.float64)
        flat = full.reshape(52, -1)
        frob = np.linalg.norm(flat, axis=1)
        delta = np.linalg.norm(flat - flat[0], axis=1)
        cosv = (flat @ flat[0]) / (frob * frob[0])
        j = col[k]
        for a, b in ((frob, z["l2"][:, j]), (delta, z["delta_ref_l2"][:, j]),
                     (cosv, z["cos_ref"][:, j])):
            rel = np.abs(a - b) / np.maximum(np.maximum(np.abs(a),
                                                        np.abs(b)), 1e-300)
            rel[(a == 0.0) & (b == 0.0)] = 0.0
            worst = max(worst, float(rel.max()))
        series[k] = {"frob": frob, "delta": delta, "cos": cosv}
    assert worst < CROSS_CHECK_RTOL, \
        f"T-B cross-check FAILED: max rel diff {worst:.3e} >= 1e-9 - STOP"
    return {"series": series, "max_rel_diff": worst, "n_keys": len(phi_keys)}


def fig_tb(d: dict, tb: dict):
    steps = d["steps"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    lq = [f"layers.{i}.pool.latent_q" for i in range(15)]
    layer_col = [cm.viridis(i / 14) for i in range(15)]
    ln_keys = [k for k in tb["series"] if "ln_pool" in k]
    for ax, what, ttl in ((axes[0], "frob", "Frobenius norm"),
                          (axes[1], "delta", "delta to S_min"),
                          (axes[2], "cos", "cosine to S_min")):
        for i, k in enumerate(lq):
            y = tb["series"][k][what]
            ax.plot(steps[1:] if what == "delta" else steps,
                    y[1:] if what == "delta" else y,
                    color=layer_col[i], lw=1.2)
        # phi LayerNorm minor series: class aggregate (norm of concatenation)
        if what != "cos":
            agg = np.sqrt(sum(tb["series"][k][what] ** 2 for k in ln_keys))
            ax.plot(steps[1:] if what == "delta" else steps,
                    agg[1:] if what == "delta" else agg,
                    color="0.4", lw=2, ls="--", label="ln_pool (30, agg)")
            ax.legend(fontsize=8)
        ax.set_xlabel("step")
        ax.set_title(ttl, fontsize=11)
        style_ax(ax)
    axes[1].set_yscale("log")
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=0, vmax=14))
    fig.colorbar(sm, ax=axes[2], label="layer i (latent_q)", fraction=0.05)
    fig.suptitle("V11 K1 T-B: phi latent queries (15 layers) + phi LayerNorm "
                 "(minor series), from full/* tensors — cross-check vs "
                 "scalar columns PASS (rel tol 1e-9)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v11-k1-tb-phi-ssra.png", dpi=150)
    plt.close(fig)


# ---- C-T1: rho over the population, mechanical verdict -------------------

def ct1(d: dict) -> dict:
    z, col = d["z"], d["col"]
    rows, zero_ref = [], []
    for n in d["population"]:
        j = col[n]
        l2_ref = float(z["l2"][0, j])
        row = {"name": n, "cls": d["classes"][n],
               "numel": int(z["numel"][j]), "l2_ref": l2_ref,
               "l2_final": float(z["l2"][-1, j]),
               "delta_ref_final": float(z["delta_ref_l2"][-1, j]),
               "is_latent_q": bool(re.search(
                   r"^layers\.\d+\.pool\.latent_q$", n))}
        if l2_ref == 0.0:  # [OP] guard: excluded from median, reported
            row["rho"] = float("nan")
            zero_ref.append(n)
        else:
            row["rho"] = row["delta_ref_final"] / l2_ref
        rows.append(row)
    med = float(np.median([r["rho"] for r in rows
                           if not math.isnan(r["rho"])]))
    lq = sorted((r for r in rows if r["is_latent_q"]),
                key=lambda r: int(r["name"].split(".")[1]))
    verdict = "inconclusive"
    if lq:
        if max(r["rho"] for r in lq) < 0.1 * med:
            verdict = "supported"
        elif min(r["rho"] for r in lq) >= med:
            verdict = "refuted"
    return {"rows": rows, "median": med, "zero_ref": zero_ref,
            "latent_q": lq, "verdict": verdict}


def fig_rho(d: dict, ct: dict):
    classes = arm_classes(d)
    fig, ax = plt.subplots(figsize=(9, 5))
    rng_pos = {c: i for i, c in enumerate(classes)}
    for r in ct["rows"]:
        if math.isnan(r["rho"]):
            continue
        y = rng_pos[r["cls"]]
        ax.plot(r["rho"], y, "o", ms=5 if r["is_latent_q"] else 3.5,
                color=CLASS_COLOR[r["cls"]], alpha=0.75,
                mec="black" if r["is_latent_q"] else "none", mew=0.6)
    ax.axvline(ct["median"], color="0.2", lw=1.2, ls="--")
    ax.axvline(0.1 * ct["median"], color="0.2", lw=1.2, ls=":")
    ymax = len(classes) - 0.5
    ax.text(ct["median"] * 1.05, ymax, "median", fontsize=8, va="top")
    ax.text(0.1 * ct["median"] * 1.05, ymax, "0.1 × median", fontsize=8,
            va="top")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"{c} ({d['class_counts'][c]})" for c in classes],
                       fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho(m)$ = delta_ref_l2[-1, m] / l2[0, m]")
    style_ax(ax)
    ax.set_title(f"V11 K1 C-T1: per-tensor rho over the ssra population "
                 f"(273 tensors) — verdict: {ct['verdict']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v11-k1-rho-ssra.png", dpi=150)
    plt.close(fig)


def write_csv(arms: dict, cts: dict):
    """Raw appendix (oversight recount input): full population, both arms."""
    out = OUT_DIR / "v11-k1-rho.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "canonical_name", "alias_group_members", "numel",
                    "l2_ref", "l2_final", "delta_ref_final", "rho",
                    "is_latent_q"])
        for arm in ("ssra", "flat"):
            d = arms[arm]
            for r in cts[arm]["rows"]:
                members = d["alias_members"].get(r["name"], [])
                w.writerow([arm, r["name"], ";".join(members), r["numel"],
                            repr(r["l2_ref"]), repr(r["l2_final"]),
                            repr(r["delta_ref_final"]), repr(r["rho"]),
                            r["is_latent_q"]])
    return out


# ---- T-C: e_l table exact-zero check + rows 0-10 figure ------------------

def tc(d: dict) -> dict:
    z = d["z"]
    flags = []
    norms = {}  # (layer) -> (52, 16) per-row norms
    zero_rows = None  # observation: rows exactly 0.0 at ALL steps + init
    for i in range(15):
        k = f"layers.{i}.level_emb"
        arr = z[f"full/{k}"]          # (52, 16, 640) float32
        init = z[f"full_init/{k}"]    # (16, 640)
        assert arr.shape[1] == 16
        if not np.all(arr[:, 11:, :] == 0.0):
            bad = np.argwhere(arr[:, 11:, :] != 0.0)
            flags.append({"key": k, "where": "checkpoints",
                          "first": bad[0].tolist(), "count": int(len(bad))})
        if not np.all(init[11:, :] == 0.0):
            bad = np.argwhere(init[11:, :] != 0.0)
            flags.append({"key": k, "where": "init",
                          "first": bad[0].tolist(), "count": int(len(bad))})
        zr = [r for r in range(16)
              if np.all(arr[:, r, :] == 0.0) and np.all(init[r] == 0.0)]
        assert zero_rows is None or zr == zero_rows, (k, zr, zero_rows)
        zero_rows = zr
        norms[i] = np.linalg.norm(arr.astype(np.float64), axis=2)
    return {"flags": flags, "norms": norms, "zero_rows": zero_rows}


def fig_tc(d: dict, t: dict):
    steps = d["steps"]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True)
    layer_col = [cm.viridis(i / 14) for i in range(15)]
    for row in range(11):
        ax = axes.flat[row]
        for i in range(15):
            ax.plot(steps, t["norms"][i][:, row], color=layer_col[i], lw=1)
        ax.set_title(f"row {row}", fontsize=9)
        if row in t["zero_rows"]:
            ax.text(0.5, 0.5, "exactly 0.0\n(all steps + init)", fontsize=9,
                    ha="center", va="center", transform=ax.transAxes,
                    color="0.35")
        style_ax(ax)
    axes.flat[11].axis("off")
    axes.flat[11].text(
        0.05, 0.45, "rows 11–15: exactly 0.0\nat all 52 steps + init,\n"
        "all 15 layers (PASS)\n\nobservation: row 10 is\nALSO exactly 0.0 "
        "at all\nsteps + init, all layers", fontsize=10)
    for ax in axes[2]:
        ax.set_xlabel("step")
    for r in range(3):
        axes[r][0].set_ylabel("row L2 norm")
    fig.suptitle("V11 K1 T-C: e_l table per-row L2 norms, rows 0–10 "
                 "(15 layers, viridis by layer)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v11-k1-tc-levelemb-ssra.png", dpi=150)
    plt.close(fig)


# ---- T-D / T-E: relative update rate + lr overlay ------------------------

def fig_td(d: dict):
    arm, steps, z = d["arm"], d["steps"], d["z"]
    lr_steps, lrs = load_lr(arm)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for cls in arm_classes(d):
        num = class_agg(d, z["upd_l2"], cls)      # (51,) interval k -> k+1
        den = class_agg(d, z["l2"], cls)[:-1]     # [OP] start-of-interval
        ax.plot(steps[1:], num / den, color=CLASS_COLOR[cls], lw=1.8,
                label=f"{cls} ({d['class_counts'][cls]})")
    ax.set_yscale("log")
    ax.set_xlabel("step (end of interval)")
    ax.set_ylabel(r"class $\|\Delta\theta\|_2 / \|\theta\|_2$ per interval")
    style_ax(ax)
    ax.annotate("final interval 51000→51880\n(880 steps < 1000 stride)",
                xy=(51880, ax.get_ylim()[0] * 1.5), fontsize=8,
                ha="right", color="0.3")
    # lr overlay mandated by the pre-registered T-D spec; secondary axis is
    # recessive (gray, no grid) and reports lr only - not a second data axis
    ax2 = ax.twinx()
    ax2.plot(lr_steps, lrs, color="0.55", lw=1, ls="-", alpha=0.8)
    ax2.set_ylabel("lr (JSONL `lr` field)", color="0.45")
    ax2.tick_params(axis="y", colors="0.45")
    ax2.spines[["top"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_title(f"V11 K1 T-{'D' if arm == 'ssra' else 'E'}: {arm} arm — "
                 f"relative update rate between consecutive checkpoints, "
                 f"lr schedule overlaid (gray)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"v11-k1-td-{arm}.png", dpi=150)
    plt.close(fig)
    return {"lr_records": len(lr_steps), "lr_field": "lr",
            "log": str(LOGS[arm].relative_to(ROOT))}


# ---- main ----------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    arms = {arm: load_arm(arm) for arm in ("ssra", "flat")}
    for d in arms.values():
        print(f"[k1-analysis] {d['arm']}: P={len(d['names'])} "
              f"population={len(d['population'])} classes={d['class_counts']}")

    summary = {"inputs": {a: f"results/v11/v11-k1-extract-{a}.npz"
                          for a in arms},
               "class_table": [[c, p] for c, p in CLASS_TABLE],
               "class_counts": {a: arms[a]["class_counts"] for a in arms}}

    for d in arms.values():
        fig_ta(d)

    tb = tb_cross_check(arms["ssra"])
    fig_tb(arms["ssra"], tb)
    print(f"[k1-analysis] T-B cross-check PASS: max rel diff "
          f"{tb['max_rel_diff']:.3e} < 1e-9 over {tb['n_keys']} keys")
    summary["tb_cross_check"] = {"max_rel_diff": tb["max_rel_diff"],
                                 "n_keys": tb["n_keys"], "pass": True}

    cts = {arm: ct1(arms[arm]) for arm in ("ssra", "flat")}
    ct = cts["ssra"]
    csv_path = write_csv(arms, cts)
    fig_rho(arms["ssra"], ct)
    print(f"[k1-analysis] C-T1: median rho = {ct['median']:.6f}, "
          f"latent_q rho in [{min(r['rho'] for r in ct['latent_q']):.6f}, "
          f"{max(r['rho'] for r in ct['latent_q']):.6f}] "
          f"-> verdict: {ct['verdict']} (zero-ref excluded: "
          f"{ct['zero_ref'] or 'none'})")
    summary["ct1"] = {
        "median_rho": ct["median"],
        "zero_ref_excluded": ct["zero_ref"],
        "latent_q_per_layer": {r["name"]: r["rho"] for r in ct["latent_q"]},
        "latent_q_min": min(r["rho"] for r in ct["latent_q"]),
        "latent_q_max": max(r["rho"] for r in ct["latent_q"]),
        "threshold_supported": 0.1 * ct["median"],
        "threshold_refuted": ct["median"],
        "verdict": ct["verdict"],
        "flat_median_rho": cts["flat"]["median"]}

    t = tc(arms["ssra"])
    fig_tc(arms["ssra"], t)
    assert not t["flags"], f"T-C exact-zero FLAG: {t['flags']}"
    print(f"[k1-analysis] T-C rows 11-15 exactly 0.0: PASS "
          f"(52 steps + init, 15 layers); rows exactly 0.0 at all steps + "
          f"init: {t['zero_rows']}")
    summary["tc"] = {"rows_11_15_exact_zero": True, "flags": [],
                     "rows_exact_zero_all_steps_and_init": t["zero_rows"]}

    summary["td"] = {arm: fig_td(arms[arm]) for arm in ("ssra", "flat")}

    out = OUT_DIR / "v11-k1-analysis-summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"[k1-analysis] done -> {csv_path.name}, {out.name}, "
          f"7 figures in results/figures/v11/")


if __name__ == "__main__":
    main()
