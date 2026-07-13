# HO-12 — M2 re-kalibrácia UZAVRETÁ: verdikt SPLNENÉ, strop 30 EUR pre S2 run; ďalej = Phase 2 zadanie

**Dátum:** 2026-07-13 · **Autor:** Claude (Fable 5) + Daniel (verdikty) · **Predchádzajúci:** HO-11
**Vstupný bod nového chatu:** tento dokument + `docs/00-stav-a-triaz.md` (D-log 2026-07-13, dva riadky) + `results/M2-recalibration.md` + `results/M2-readout-optimization.md` (referencie)

---

## 1. Stav jednou vetou

Kroky 2 a 3 rozhodnutia (a) z D-logu 2026-07-12 sú hotové a **merateľný cieľ je SPLNENÝ (verdikt Daniel 2026-07-13)**: read-out optimalizácia prešla oversight review 6/6, re-kalibrácia na A100 SXM (~1.14 EUR) namerala SSRA S2 b16 = 12 334.7 tok/s / 41.21 GiB (Phase 1: OOM), projekcia 850M = 23.3–25.1 EUR podľa karty; Daniel predschválil **strop 30 EUR pre S2 850M SSRA run** (scoped výnimka AP-12 single-run brány); **chat, ktorý číta tento dokument, píše Phase 2 zadanie (symetrický lr/dropout sweep per M2-assignment v1.1 §4) a rieši otvorené položky §4.**

## 2. Čo sa v tomto sedení stalo

1. **Oversight review CC reportu read-out optimalizácie:** gate check 6/6 PASS; nezávislý prepočet D5 (k_max=95 vlastnou enumeráciou, 6/6 totálov na cent, margin backout 1.1075, senzitivita: brána 60 GiB padne až pri margine 1.88); bonusová korobácia — 3 kalibračné OOM last-alloc hodnoty (4.45/3.71/1.27 GiB) = presné closed-form členy; commit hygiena overená cez GitHub API (file list `9417399`). Nity bez akcie v D-logu.
2. **GO → zadanie `docs/cc/M2-recalibration.md` v1** + D-log zápis (jeden commit, Daniel).
3. **Launch:** A100 PCIe nedostupná → **SXM $1.49/hr (rebrík krok 2)**; pod `ssra-m2-recal` (`1u7wmoy6l71ull`), template overrides z minula (image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 50 GB disk, env var `GCP_SA_KEY_B64`, start command default). Community cena not capturable — zaznamenané, žiadny backfill.
4. **CC exekúcia:** všetkých 6 prób, report `results/M2-recalibration.md`, commit `819fcb2` (signed); pod terminovaný po signále, ~52 min.
5. **Verdikt Daniel: SPLNENÉ + strop 30 EUR** → D-log zápis aplikovaný (2. riadok 2026-07-13).

## 3. Kľúčové čísla [záväzné, z reportu — nere-merať]

| run | tok/s | peak GiB | D5 proj. | chyba modelu |
|---|---|---|---|---|
| flat S1 b16 (kotva) | 319 945 | 6.345 | — | — (Phase 1: 300 978 ⇒ box +6.3 %) |
| SSRA S1 b16 | 27 079 | 18.557 | 15.40 | +20.5 % |
| **SSRA S2 b16 (BRÁNA)** | **12 334.7** | **41.207** | 36.59 | +12.6 % |
| SSRA S1 b32 | 31 759 | 36.584 | 30.44 | +20.2 % |
| SSRA S1 b64 | 33 357 | 72.624 | 60.51 | +20.0 % |
| SSRA S2 b32 | OOM (1. backward) | >79 | 71.92 | konzist. (≈86) |

- **D5 error bar = ×1.20** (systematický podstrel) — používať pri Phase 2+ pamäťovom plánovaní.
- Projekcia S2 @ 850M: 23.28 EUR @ $1.39 báza / 24.95 GPU-only / 25.07 s diskom @ SXM $1.497. **Strop runu: 30 EUR (D-log).** Plný 1.7B budget NIE je v hre — planning na 850M.
- Gap SSRA vs flat: 32× → 11.8× @ S1 b16 (konštanta sa v paperi reportuje vždy).
- Pytest na boxe 64/1 (známy fla×transformers Phase-4 fail); A/B testy zelené aj na CUDA.

## 4. Otvorené položky

| # | položka | kedy |
|---|---|---|
| 1 | ~~Doplniť konzolovú fakturovanú sumu~~ **✔ hotové:** $0.9700 ≈ 0.85 EUR zapísané v ledgeri, D-logu aj reporte (≈ 25 % pod CC odhadom, vzorec ako Phase 1; región US-MD-1); kumulatívne 3.89 EUR ≈ 1.3 % | hotové 2026-07-13 |
| 2 | **Phase 2 zadanie pre CC** (symetrický S1 lr/dropout sweep per M2-assignment §4/AP-14; do zadania zapracovať: run_name pravidlo pre re-runy, D5 ×1.20 error bar, strop 30 EUR pre S2 run vo Phase 3 kontexte) | nový chat, prvá úloha |
| 3 | **AP-19 Community cena — 3. prenos:** zachytiť **pred** deployom ako krok 0 pre-flightu (nie po; post-deploy nebola dohľadateľná) | ďalší launch |
| 4 | transformers pin kompatibilný s fla 0.5.0 | pred Phase 4 |
| 5 | ⚠ Revert záväzok: zmazať SA kľúč + vrátiť org-policy enforcement | po M2/M3 |
| 6 | Zenodo erratum/v1.1 (retenčné pravidlo + read-out konštanta) | po zozbieraní M2 nálezov |
| 7 | `p1_attn_entropy` monitoring štandardne v Phase 2 logoch | Phase 2 |
| 8 | Project files vymeniť: `00`, `results/M2-recalibration.md`, `results/M2-readout-optimization.md`, `docs/cc/M2-recalibration.md`, tento HO | po commite |

## 5. Čoho sa nedotýkať

Uzavreté rozhodnutia (D1–D6, Q1–Q5, MD-1…MD-13, AP-1…AP-20) sa neotvárajú. Spec v1.2 normatívna, existujúce testy sa nemodifikujú (AP-20 platí pre každý ďalší perf rework). Žiadne kvalitatívne závery z lossov (spec §16) — kalibračné/re-kalibračné loss hodnoty sa NIKDY necitujú ako evidencia kvality. Flat baseline sa nedropuje. G1 kritérium (±5 % val ppl @ ctx 1024) sa nemení bez D-logu. Strop 30 EUR platí len pre S2 850M SSRA run — všetko ostatné pod AP-12 (25 EUR/run, prahy 50/80 % z 300 EUR). Virtual fallback (`_readout_gathered`) a `build_readout_index` ostávajú — decoder a contingency ich potrebujú. R5 = prvý revert kandidát, ak sa v Phase 2 objaví up-pass anomália.
