# HO-28 — V11: S0 + K3 uzavreté (oversight PASS ×2), K2 descoped, K1 pending

Dátum: 2026-07-20 (session 19.–20. 7. v noci). Nadväzuje: HO-27. Detailný záznam: D-log 2026-07-20.

## 1. Stav po tomto sedení

- **Commity:** `5e16d90` (S0 batch: report §S0 + 4 evidencie + `scripts/v11_price_check.py`) a `db9a181` (K3 batch: report + 3 skripty + 5 výsledkových súborov + 4 figúry). V čase písania **len lokálne** — push je súčasť záverečného batchu tohto sedenia. Po pushi vykonať G-V11-4 re-check z GitHub file listov oboch commitov (0 súborov pod `src/ssra/` a `baselines/`).
- **S0:** inventár presne per očakávanie (106 top-level objektov / 100,10 GiB); K1 set = **explicitný 104-zoznam** (dedup: `latest.pt` md5-identické so `step-51880.pt` v oboch ramenách); **stray** vnorený objekt `…ssra…/latest.pt/latest.pt` (md5 = flat final) — nemazať teraz; **pre-delete listing post-K1 očakáva 107 objektov**. K2 DESCOPED (G2-lite artefakty = len per-cell agregáty; re-run forwardov neautorizovaný). Ceny: Catalog API v EUR, SKU + tiery v `results/v11/v11-s0c-prices.json`; **G-V11-2 PASS — worst-case 0,727 ≤ 3,00 EUR** (e2-standard-4 €0,151139/h; realisticky ≈ 0,47 EUR).
- **K3:** hotové; oversight nezávisle prepočítal z raw vstupov (needle 720/720 riadková zhoda; entropia core runov priamo z logov; bf16 kvantá 8/8 binád pure-python round-tripom). Kľúčové čísla: 15/17 sérií ≤ 0,00024 nat od ln 32; lr6e4 min 3,4287 / final 3,4348; lr1e3 min 2,9396 / final 3,2006; needle flat 36/4/320, SSRA 0/0/360. Sémantika kód-overená: entropia nad 32 kľúčmi (publikovaná P-C interpretácia drží), participácia nad 16 latentnými queries (`p1_participation_min/max`, `src/ssra/pool.py::_p1_diagnostics`).
- Spend 0 EUR; kumulatív **72,37 EUR ≈ 24,1 %** (ledger append 2026-07-20 v `results/runs.md`).

## 2. K1 session — plán (jediná platená časť V11)

1. **CC lokálne (0 EUR):** extrakčný skript (`scripts/v11_` prefix) + smoke test na lokálnom dummy checkpointe; presné `gcloud` príkazy na create/delete VM (e2-standard-4, europe-west3, attached default compute SA so scope `devstorage.read_only` cez metadata server — žiadne user-managed kľúče, R7 nedotknuté; nič sa do bucketu nezapisuje).
2. **Daniel:** VM create (konzola/terminál, príkazy od CC). Štartuje 4,0 h wall cap (G-V11-3).
3. **CC cez `gcloud compute ssh`:** streaming EXPLICITNÉHO 104-zoznamu zo S0-A po jednom (download → extract → delete lokálnej kópie); metriky T-A…T-E vrátane povinného T-D; delta referencia S_min = `step-1000`; rekonštrukcia initu len ak validovaná (inak len S_min-reference, bez tichého prepínania); výstup 1 komprimovaný archív/rameno → `gcloud compute scp` na Mac (~21 MiB, bound 1 GiB).
4. **Daniel: VM delete IHNEĎ po scp** (idle lekcie ×3 z M2). Fakturácia per pravidlo ≥ 2 h; ledger riadok.
5. CC dopíše §K1 + §Deviations + §Ledger → **oversight review (Claude)** → po PASS: **Daniel ckpt delete** (pre-delete listing, očakávanie **107 objektov**, AP-25 vzor: listing + kontrola úplnosti extraktov pred príkazom) → D-log + ledger.

Vyžaduje Danielovu prítomnosť na create/delete; plánovať ~2 h súvislé okno.

## 3. Standing

- Nota v1.1: GO, **bez termínu** (checklist HO-27 §2) — nezávislá malá session kedykoľvek.
- #2 arXiv endorsement open; teardown (SA delete + bucket IAM cleanup + GCS zvyšky) ≤ 2026-08-31.

## 4. Otvárací prompt pre nový chat (paste-ready)

```
K1 exekucia (V11). Cez Filesystem MCP precitaj docs/handover/HO-28-2026-07-20-v11-s0-k3-uzavrete-k1-pending.md a docs/00-stav-a-triaz.md (tail 12), over git log -1 (<HASH>) a zhodu remote HEAD == local HEAD. Kontext: S0+K3 uzavrete, oversight PASS (D-log 2026-07-20), K2 DESCOPED, G-V11-2 PASS (0,727 <= 3,00 EUR), K1 = cesta (ii) e2-standard-4 @ europe-west3, wall cap 4 h, explicitny 104-zoznam bez prefix globbingu, T-D povinne, stray objekt -> pre-delete listing 107. Kumulativ 72,37 EUR ~ 24,1 %. Dnes: CC priprava + lokalny smoke extrakcneho skriptu (0 EUR) -> VM lifecycle (Daniel konzola) -> beh -> scp extraktov -> VM delete -> report §K1 -> oversight.
```

(`<HASH>` doplň po commite + pushi tohto batchu.)
