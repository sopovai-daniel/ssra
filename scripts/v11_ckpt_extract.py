"""V11 K1 checkpoint-trajectory extraction (docs/cc/V11-data-exploitation.md §5).

Pure tensor statistics over the Phase 3b step-tagged checkpoints — no
gradients, no forward passes, no fine-tuning (assignment §5 anti-goals).
Subcommands:

  manifest         build the explicit per-arm step list (NO prefix globbing)
                   from the committed S0-A `gsutil ls -l` listings; asserts
                   the exact 52-step set and constant per-arm object size.
  smoke-fixture    build tiny dummy checkpoints LOCALLY through the repo's
                   own save_checkpoint code path (production blob structure)
                   so `extract` can be smoke-tested end-to-end incl. the
                   pairwise T-D path. No gradients: weights are perturbed
                   under no_grad; optimizer state is populated synthetically
                   with production-shaped tensors.
  reconstruct-init best-effort init reconstruction (assignment §5 delta
                   reference, secondary): replays train.py's construction
                   order — torch.manual_seed(training.seed) then
                   BUILDERS[arch](cfg) on CPU. Used ONLY if validated
                   against S_min inside `extract` (no silent switching).
  extract          stream a manifest one checkpoint at a time (download →
                   load → metrics → delete local copy), keep the previous
                   state dict in RAM only for consecutive update norms, and
                   write one compressed NPZ per arm.

Per named parameter and checkpoint: L2 norm, L2 delta and cosine to the
S_min reference (first manifest row), and the consecutive update norm
||theta_k - theta_{k-1}||_2. Full tensors are saved ONLY for the phi-side
pooling parameters (MD-3: latent queries + phi LayerNorm) and the e_l table:

  layers.{i}.pool.latent_q       [m_max, d]   phi latent queries
  layers.{i}.pool.ln_pool.weight [d]          phi LayerNorm scale
  layers.{i}.pool.ln_pool.bias   [d]          phi LayerNorm shift
  layers.{i}.level_emb           [l_max+1, d] e_l table

`torch.load(map_location="cpu", weights_only=True)` is REQUIRED; any load
failure is a hard stop (report to Daniel) — the flag is never relaxed.

Init validation (pre-registered here, before any production data is read):
over every parameter tensor with nonzero init norm, rel(theta) =
||theta_Smin - theta_init||_F / ||theta_init||_F. Validated iff
max rel < 0.5 (a correct same-RNG reconstruction drifts far less by step
1000; an independent random draw sits near sqrt(2)). If any tensor fails,
init-reference columns are dropped and only the verdict + per-tensor rel
table is recorded — S_min-reference results stand alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

STEP_RE = re.compile(r"step-(\d+)\.pt$")
PHI_KEY_RE = re.compile(
    r"^layers\.\d+\.(pool\.latent_q|pool\.ln_pool\.(weight|bias)|level_emb)$")
EXPECTED_STEPS = list(range(1000, 52000, 1000)) + [51880]  # 52 per arm (S0-A)
ARM_BYTES = {"ssra": 1016124393, "flat": 1011848651}  # constant per arm (S0-A)
REL_DRIFT_MAX = 0.5  # init-validation threshold (module docstring)
FLUSH_EVERY = 10  # partial NPZ rewrite cadence (abort-rule partial scp)


def _repo_imports():
    """Repo imports only where needed (manifest/extract stay torch+numpy-only
    so the VM needs no package installs beyond the bootstrap)."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    from baselines.flat import FlatLM  # noqa: E402
    from ssra import SSRALM, config_from_dict  # noqa: E402
    from ssra.checkpoint import save_checkpoint  # noqa: E402
    return {"ssra": SSRALM, "flat": FlatLM}, config_from_dict, save_checkpoint


# ---- manifest ---------------------------------------------------------------

def cmd_manifest(args):
    """Explicit step list from the committed S0-A listing (no globbing)."""
    rows = []
    for line in Path(args.ls_file).read_text().splitlines():
        parts = line.split()
        if len(parts) != 3 or not STEP_RE.search(parts[2]):
            continue  # skips latest.pt, the stray nested object, TOTAL line
        rows.append((int(STEP_RE.search(parts[2]).group(1)),
                     int(parts[0]), parts[2]))
    rows.sort()
    steps = [r[0] for r in rows]
    assert steps == EXPECTED_STEPS, \
        f"step set mismatch vs S0-A: got {len(steps)} steps {steps[:3]}..."
    sizes = {r[1] for r in rows}
    assert sizes == {ARM_BYTES[args.arm]}, \
        f"object sizes {sizes} != committed constant {ARM_BYTES[args.arm]}"
    out = Path(args.out)
    out.write_text("".join(f"{s}\t{b}\t{u}\n" for s, b, u in rows))
    print(f"[manifest] {args.arm}: {len(rows)} objects -> {out}")


# ---- smoke fixture ----------------------------------------------------------

def _tiny_raw(arch: str) -> dict:
    return {"model": {"d": 64, "h": 4, "n_layers": 2, "vocab": 256,
                      "n_max": 1024, "m": 16, "w": 64, "pool": "p1",
                      "tied_embeddings": True}}


def _fake_adamw_state(model, opt):
    """Production-shaped optimizer state without running any step/gradient."""
    for p in model.parameters():
        opt.state[p] = {"step": torch.tensor(1000.0),
                        "exp_avg": torch.zeros_like(p),
                        "exp_avg_sq": torch.zeros_like(p)}


def cmd_smoke_fixture(args):
    builders, config_from_dict, save_checkpoint = _repo_imports()
    out = Path(args.out_dir)
    for arch in ("ssra", "flat"):
        raw = _tiny_raw(arch)
        cfg = config_from_dict(json.loads(json.dumps(raw)))
        torch.manual_seed(1337)
        model = builders[arch](cfg)
        arm_dir = out / arch
        arm_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "seed": 1337, "arch": arch,
                    "trainable": [n for n, _ in model.named_parameters()],
                    "torch": str(torch.__version__)}, arm_dir / "init.pt")
        torch.manual_seed(4242)
        wrong = builders[arch](cfg)
        torch.save({"model": wrong.state_dict(), "seed": 4242, "arch": arch,
                    "trainable": [n for n, _ in wrong.named_parameters()],
                    "torch": str(torch.__version__)}, arm_dir / "init-wrong.pt")
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4,
                                weight_decay=0.01, betas=(0.9, 0.95))
        _fake_adamw_state(model, opt)
        gen = torch.Generator().manual_seed(1337)
        rows = []
        for i, (step, scale) in enumerate([(1000, 1e-3), (2000, 1e-2),
                                           (3000, 1e-2)]):
            with torch.no_grad():  # NOT a training step: pure perturbation
                for p in model.parameters():
                    p.add_(scale * torch.randn_like(p))
            path = save_checkpoint(
                arm_dir / f"step-{step}.pt", step=step, model=model,
                optimizer=opt, data_gen=gen, config_raw=raw,
                run_name=f"v11-smoke-{arch}",
                extra={"arch": arch, "vocab": 256})
            rows.append((step, path.stat().st_size, str(path)))
        (arm_dir / "manifest.tsv").write_text(
            "".join(f"{s}\t{b}\t{u}\n" for s, b, u in rows))
        print(f"[smoke-fixture] {arch}: 3 ckpts + init/init-wrong "
              f"-> {arm_dir}")


# ---- init reconstruction ----------------------------------------------------

def cmd_reconstruct_init(args):
    import yaml
    builders, config_from_dict, _ = _repo_imports()
    raw = yaml.safe_load(Path(args.config).read_text())
    arch = raw.pop("arch", "ssra")
    raw.pop("run_name", None)
    data_cfg = raw.pop("data")
    t = raw.pop("training")
    # exact train.py replay: vocab injected, then seed, then construction on
    # CPU; nothing else touches the global torch RNG in between (train.py
    # main(): load_data uses no torch RNG; data_gen is a separate Generator)
    raw.setdefault("model", {})["vocab"] = int(data_cfg["vocab"])
    cfg = config_from_dict(raw)
    torch.manual_seed(t["seed"])
    model = builders[arch](cfg)
    torch.save({"model": model.state_dict(), "seed": t["seed"], "arch": arch,
                "trainable": [n for n, _ in model.named_parameters()],
                "config": str(args.config), "torch": str(torch.__version__)},
               Path(args.out))
    n = sum(p.numel() for p in model.parameters())
    print(f"[reconstruct-init] {arch} seed={t['seed']} params={n} "
          f"torch={torch.__version__} -> {args.out}")


# ---- extract ----------------------------------------------------------------

def _norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.double()))


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(torch.dot(a.double().flatten(), b.double().flatten())
                 / (na * nb))


def _load_blob(path: Path) -> dict:
    """weights_only=True is REQUIRED (assignment §5). Never relaxed."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise SystemExit(
            f"[extract] STOP: torch.load(weights_only=True) failed on "
            f"{path.name}: {type(e).__name__}: {e}\n"
            f"Production blob format incompatible with weights_only=True — "
            f"report to Daniel; do NOT relax the flag.") from e


def _validate_init(init_sd: dict, ref_sd: dict) -> tuple[bool, dict]:
    """Pre-registered check (module docstring): max rel drift at S_min over
    tensors with nonzero init norm must be < REL_DRIFT_MAX."""
    rels = {}
    for k, v in init_sd.items():
        n0 = _norm(v)
        if n0 == 0.0:
            continue  # zero-init tensors (e_l rows) carry no RNG signal
        rels[k] = _norm(ref_sd[k].double() - v.double()) / n0
    ok = bool(rels) and max(rels.values()) < REL_DRIFT_MAX
    return ok, rels


def _save_npz(out: Path, arrays: dict):
    tmp = out.with_suffix(out.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)  # atomic: a partial scp never sees a torn file


def cmd_extract(args):
    manifest = [line.split("\t") for line in
                Path(args.manifest).read_text().splitlines()]
    manifest = [(int(s), int(b), u) for s, b, u in manifest]
    assert manifest == sorted(manifest), "manifest must be step-ascending"
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    init_sd, init_meta, init_trainable = None, None, None
    if args.init:
        blob = torch.load(Path(args.init), map_location="cpu",
                          weights_only=True)
        init_sd = blob["model"]
        init_meta = {k: blob.get(k) for k in ("seed", "arch", "torch")}
        # C-T1 is over trainable parameter tensors: carry the explicit
        # named_parameters() list (alias-deduped) regardless of the
        # validation verdict — validation gates only delta/cos columns
        init_trainable = blob.get("trainable")

    names, meta, alias_groups = None, None, None
    l2, dref, cref, dinit, cinit, upd, timings = [], [], [], [], [], [], []
    full, full_init = {}, {}
    init_l2_row = None
    steps_done = []
    ref_sd = prev_sd = None
    init_ok, init_rels = False, {}
    t_start = time.time()

    def flush():
        arrays = {
            "steps": np.array(steps_done, dtype=np.int64),
            "param_names": np.array(names),
            "numel": np.array([meta[k] for k in names], dtype=np.int64),
            "l2": np.array(l2), "delta_ref_l2": np.array(dref),
            "cos_ref": np.array(cref),
            "upd_l2": np.array(upd) if upd else np.zeros((0, len(names))),
            "timings_s": np.array(timings),
            "meta_json": np.array(json.dumps({
                "arm": args.arm, "manifest": str(args.manifest),
                "s_min_step": manifest[0][0],
                "torch": str(torch.__version__), "numpy": np.__version__,
                "weights_only": True,
                "alias_groups": alias_groups,
                "trainable": init_trainable,
                "init": init_meta,
                "init_validated": init_ok if args.init else None,
                "init_rel_drift_at_smin": init_rels,
                "rel_drift_max": REL_DRIFT_MAX,
                "wall_s": round(time.time() - t_start, 1)})),
        }
        if args.init and init_ok:
            arrays["delta_init_l2"] = np.array(dinit)
            arrays["cos_init"] = np.array(cinit)
        if init_l2_row is not None:
            # init reference survives VM deletion: norms for every key +
            # full phi/e_l init tensors (rho denominators, phi-vs-init)
            arrays["init_l2"] = np.array(init_l2_row)
            for k, v in full_init.items():
                arrays[f"full_init/{k}"] = v
        for k, v in full.items():
            arrays[f"full/{k}"] = np.stack(v)
        _save_npz(out, arrays)

    for i, (step, nbytes, uri) in enumerate(manifest):
        t0 = time.time()
        if uri.startswith("gs://"):
            local = workdir / Path(uri).name
            r = subprocess.run(["gcloud", "storage", "cp", uri, str(local)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"[extract] download failed {uri}: "
                                 f"{r.stderr.strip()}")
            downloaded = True
        else:
            local, downloaded = Path(uri), False
        got = local.stat().st_size
        assert got == nbytes, f"{local.name}: {got} B != manifest {nbytes} B"
        t1 = time.time()
        blob = _load_blob(local)
        assert int(blob["step"]) == step, \
            f"{local.name}: blob step {blob['step']} != manifest {step}"
        if args.expect_run_name:
            assert blob["run_name"] == args.expect_run_name, \
                f"run_name {blob['run_name']!r} != {args.expect_run_name!r}"
        n_layers = int(blob["config_raw"]["model"]["n_layers"])
        sd = blob["model"]
        del blob
        t2 = time.time()

        if names is None:
            names = sorted(sd.keys())
            meta = {k: sd[k].numel() for k in names}
            # F1: alias groups by tensor identity (shared attention appears
            # under layers.i.attn.*, layers.i.pool.attn.*,
            # layers.i.readout_attn.*). Per-key series stay as-is (raw
            # fidelity); this map drives downstream dedup — the C-T1
            # median is over tensors, not state-dict keys.
            ident: dict = {}
            for k in names:
                ident.setdefault(
                    (sd[k].data_ptr(), tuple(sd[k].shape)), []).append(k)
            alias_groups = sorted(g for g in ident.values() if len(g) > 1)
            print(f"[extract] alias groups (>1 key per tensor): "
                  f"{len(alias_groups)}; canonical = first sorted member")
            phi_keys = sorted(k for k in names if PHI_KEY_RE.match(k))
            if args.arm == "ssra":
                by = {"latent_q": 0, "ln_pool": 0, "level_emb": 0}
                for k in phi_keys:
                    for tag in by:
                        by[tag] += tag in k
                assert by == {"latent_q": n_layers, "ln_pool": 2 * n_layers,
                              "level_emb": n_layers}, \
                    f"phi/e_l key census mismatch: {by} (n_layers={n_layers})"
                for k in phi_keys:
                    assert sd[k].dtype == torch.float32, (k, sd[k].dtype)
                full = {k: [] for k in phi_keys}
                print(f"[extract] phi/e_l full-tensor keys ({len(phi_keys)}): "
                      + ", ".join(phi_keys))
            else:
                assert not phi_keys, f"flat arm has phi keys?! {phi_keys}"
        else:
            assert sorted(sd.keys()) == names, "state-dict key set changed"

        if ref_sd is None:  # first row = S_min reference
            ref_sd = {k: v.clone() for k, v in sd.items()}
            if init_sd is not None:
                assert sorted(init_sd.keys()) == names, \
                    "init state-dict key set != checkpoint key set"
                init_ok, init_rels = _validate_init(init_sd, ref_sd)
                worst = max(init_rels.values())
                print(f"[extract] init validation at S_min={step}: "
                      f"{'VALIDATED' if init_ok else 'DROPPED'} "
                      f"(max rel drift {worst:.4f}, threshold "
                      f"{REL_DRIFT_MAX}); no silent switching — "
                      f"recorded in NPZ meta")
                # F1b: persist the init reference itself either way (norms
                # for all keys, full tensors for phi/e_l) — the verdict
                # gates only the delta_init/cos_init columns
                init_l2_row = [_norm(init_sd[k]) for k in names]
                full_init = {k: init_sd[k].numpy().copy() for k in full}

        l2.append([_norm(sd[k]) for k in names])
        dref.append([_norm(sd[k].double() - ref_sd[k].double())
                     for k in names])
        cref.append([_cos(sd[k], ref_sd[k]) for k in names])
        if init_sd is not None and init_ok:
            dinit.append([_norm(sd[k].double() - init_sd[k].double())
                          for k in names])
            cinit.append([_cos(sd[k], init_sd[k]) for k in names])
        if prev_sd is not None:  # T-D pairwise, streamed: prev kept in RAM
            upd.append([_norm(sd[k].double() - prev_sd[k].double())
                        for k in names])
        for k in full:
            full[k].append(sd[k].numpy().copy())
        prev_sd = sd
        steps_done.append(step)
        if downloaded:
            local.unlink()  # stream: never hold two downloads on disk
        t3 = time.time()
        timings.append([t1 - t0, t2 - t1, t3 - t2])
        print(f"[extract] {args.arm} {i + 1}/{len(manifest)} step={step} "
              f"dl={t1 - t0:.1f}s load={t2 - t1:.1f}s "
              f"compute={t3 - t2:.1f}s elapsed={t3 - t_start:.0f}s",
              flush=True)
        if (i + 1) % FLUSH_EVERY == 0 or i + 1 == len(manifest):
            flush()

    print(f"[extract] done: {len(steps_done)}/{len(manifest)} ckpts, "
          f"{len(names)} named params -> {out} "
          f"({out.stat().st_size / 2**20:.1f} MiB, "
          f"wall {time.time() - t_start:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest")
    m.add_argument("--arm", choices=("ssra", "flat"), required=True)
    m.add_argument("--ls-file", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_manifest)

    s = sub.add_parser("smoke-fixture")
    s.add_argument("--out-dir", required=True)
    s.set_defaults(fn=cmd_smoke_fixture)

    r = sub.add_parser("reconstruct-init")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_reconstruct_init)

    e = sub.add_parser("extract")
    e.add_argument("--arm", choices=("ssra", "flat"), required=True)
    e.add_argument("--manifest", required=True)
    e.add_argument("--workdir", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--init", default=None)
    e.add_argument("--expect-run-name", default=None)
    e.set_defaults(fn=cmd_extract)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
