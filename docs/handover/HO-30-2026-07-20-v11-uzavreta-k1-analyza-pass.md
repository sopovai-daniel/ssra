# HO-30 — V11 UZAVRETÁ (K1-analýza oversight PASS, C-T1 INCONCLUSIVE); ďalej nota v1.1

Dátum: 2026-07-20 (session 3, popoludní). Nadväzuje: HO-29. Detailný záznam: D-log 2026-07-20 (štvrtý riadok).

## 1. Stav po tomto sedení

- **V11 KOMPLETNÁ:** S0 ✔, K3 ✔, K2 descoped, K1 extrakcia + ckpt delete ✔, **K1-analýza ✔** (CC commit `1566a08`; close batch tohto sedenia = D-log + tento HO). Spend celej vlny K1-analýzy 0 EUR; kumulatív [ODHAD] 72,43 EUR ≈ 24,1 %.
- **C-T1 verdikt: INCONCLUSIVE** (mechanicky, pre-registrované prahy): medián ρ nad 273-tenzorovou populáciou **2,763482**; ρ(latent_q) per-layer 1,533105 (L13) – 7,836802 (L9); supported vyžadovalo max < 0,276348, refuted min ≥ 2,763482 — ani jedno nesplnené.
- **Oversight PASS (0 vecných korekcií), nezávisle:** flat rameno plný prepočet z NPZ (183 riadkov × 4 hodnoty + numel vs CSV: 0 nezhôd, worst rel diff 0,00e+00; medián 2,864481); ssra cez snippet exekuovaný Danielom — populačný hash `d150251b…` = hash z CSV (set-identita), medián + všetkých 15 ρ na 6 dp, T-B endpointy, T-C (riadok 9 max 1,89531; riadky 10–15 presne 0,0), T-D okno 26k–36k — 100 % zhoda. G-V11-4 z GitHub file listu (11 súborov, 0 pod `src/ssra/`, `baselines/`); G-V11-5 splnené.
- **Observácie zaznamenané (bez architektonických záverov, spec §16):** (a) entropia ≈ ln(32) NIE je sprevádzaná nehybnosťou latent queries (všetkých 15 ρ v pásme 0,55–2,84× mediánu); (b) **trénované riadky e_ℓ sú 0–9, nie 0–10** (riadok 10 presne 0,0 všade vrátane initu); (c) T-D: pool.latent_q jediná trieda s ne-monotónnym poklesom (bump 0,0538 @ 28k → 0,0667 @ 33k, bez flat náprotivku).

## 2. Pending

- **Billing korekcia K1 okna** (~0,06 EUR [ODHAD]): append-only riadok do `results/runs.md` + §Ledger po objavení v GCP konzole (report typicky T+1; nie časovo kritické, materialita zanedbateľná).
- **Nota v1.1** (GO, bez termínu; checklist HO-27 §2 + AP-25 blok): kandidátsky obsah += V11 výsledky — C-T1 inconclusive + observácie (a)–(c), **row-10 spresnenie** (skontrolovať formulácie paper/G2-lite, či niekde netvrdia riadok 10 ako trénovaný), V2b kvantová figúra, needle kategorizácia, P-C entropia overlay.
- **Row-10 follow-up (voliteľné, 0 EUR, pred alebo v rámci v1.1):** overiť z read-out plánovača (kód, žiadne forwardy), či sa level-10 (root) súhrn @ N=1024 vôbec konzumuje — očakávanie: nie (root sa stáva konzumovateľným až za koncom sekvencie), čo by riadok 10 = 0 plne vysvetlilo.
- Teardown ≤ 2026-08-31 (checklist per HO-29 §3: SA final delete, bucket legacy IAM, GCS zvyšky g2lite + data, lokálny gcloud ssh keypair + pubkey v projekt metadátach).
- #2 arXiv endorsement open.

## 3. Otvárací prompt pre nový chat (paste-ready)

```
Nota v1.1 (0 EUR, lokalne). Cez Filesystem MCP precitaj docs/handover/HO-30-2026-07-20-v11-uzavreta-k1-analyza-pass.md a docs/00-stav-a-triaz.md (tail 2), over git log -1 (<HASH>) a zhodu remote HEAD == local HEAD. Kontext: V11 kompletna (C-T1 INCONCLUSIVE, oversight PASS — D-log 2026-07-20 stvrty riadok). Dnes: checklist HO-27 §2 — obsah v1.1 (AI disclosure vo file, spec §9 erratum, pointer na stage-2 DOI 10.5281/zenodo.21439493, V11 kandidati vratane row-10 spresnenia), MD zdroj -> PDF export, AP-25 brana (identita artefaktu pred Publish), Zenodo New version noty 20647034.
```

(`<HASH>` doplň po commite + pushi close batchu tohto sedenia.)
