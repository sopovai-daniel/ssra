# HO-14 — M2 Phase 2 UZAVRETÁ: sweep 8/8, selekcie (1e-3, 0.0) pre oba modely; ďalej = Phase 3 zadanie (S2 850M)

**Dátum:** 2026-07-14 · **Autor:** Claude (Fable 5) + Daniel (rozhodnutia) · **Predchádzajúci:** HO-13
**Vstupný bod nového chatu:** tento dokument + `docs/00-stav-a-triaz.md` (D-log 2026-07-14, štyri riadky) + `results/M2-sweep.md` (§B.1–B.9)

---

## 1. Stav jednou vetou

M2 Phase 2 je celá uzavretá — Task A (913.6M tokenov v GCS) aj Task B (symetrický S1 lr/dropout sweep, 8/8 runov bez divergencie, oversight review VERIFIED) — s mechanickými selekciami **flat (lr 1e-3, dropout 0.0) = 4.28121** a **SSRA (lr 1e-3, dropout 0.0) = 4.23127** (final_eval_loss na `val-eval-2M`); **chat, ktorý číta tento dokument, autoruje zadanie Phase 3: S2 850M core pair s vybranými hyperparametrami pod predschváleným stropom 30 EUR.**

## 2. Čo sa v tomto sedení stalo

1. **Task B lokálna príprava** (commit `482bdb5`): 6 stage-1 configov (AP-21 mená), harness sha256 hard gates (train/val/val-eval-2M/tokenizer pred každým tréningovým krokom), `eval_bin` ako samostatný eval set + deterministický full-coverage `final_eval` (selekčná metrika, fp32 akumulácia), `--dry-run` režim; lokálny smoke + negatívny gate test; „READY FOR POD" signál.
2. **Task B sweep na pode `ssra-m2-sweep`** (`bq0ky2rcudcsf4`, A100 SXM 80 GB Secure, US-MD-1, $1.50/hr vrátane disku, repo cez git bundle — GitHub HTTPS clone bez auth zlyhá): flat stage 1 → SSRA stage 1 → stage-2 configy commitnuté pred behom (`21a4d9d`) → stage 2 → uploady → signál. Výsledky commit `d752762`.
3. **Selekcie (mechanické):** stage-1 víťaz lr 1e-3 u oboch modelov; dropout 0.1 zhoršil oba (flat 4.36339, SSRA 4.35232) ⇒ finálne (1e-3, 0.0) pre oba. Žiadne kvalitatívne závery zo strát (spec §16).
4. **Oversight review (Claude): §B.1–B.9 VERIFIED** — selekcie, ledger aritmetika, tok/s konzistencia nezávisle prepočítané. Jediný materiálny nález = idle položka v ledgeri (bod 5).
5. **Ledger FINAL:** $7.2679 ≈ 6.36 EUR (rozhodnutie Daniela; odpočet ≈ 2.5 h po terminácii, RunPod ~1 h billing delay). Dekompozícia [ODHAD]: ≈ $5.25 práca + ≈ $2.03 post-signal idle (signál ≈ 12:10 UTC → terminácia 13:31 UTC; **3. výskyt idle vzoru v M2**). **Kumulatív 11.75 EUR ≈ 3.9 % z 300.**
6. **Nové pravidlá:** AP-19 úprava prijatá (krok 0 ostáva; Secure-vs-Community porovnanie nie je deliverable, kým sa Community tier neobjaví v deploy flow — 4× not capturable). Pravidlo odpočtu: konzolová hodnota finálna ≥ 2 h po terminácii; neskoršia korekcia = append (nahrádza plošný T+1 re-check).

## 3. Kľúčové fakty [záväzné — nere-merať, nere-hashovať]

| artefakt | hodnota |
|---|---|
| selekcia flat | **(lr 1e-3, dropout 0.0)**, final_eval_loss 4.28121 (`val-eval-2M`) |
| selekcia SSRA | **(lr 1e-3, dropout 0.0)**, final_eval_loss 4.23127 (`val-eval-2M`) |
| výkonové kotvy S1 b16 | SSRA 27 062 tok/s / 18.557 GiB (−0.06 % vs recal) · flat ≈ 312k tok/s / 6.345 GiB |
| S2 kotvy (recal, pre Phase 3 plán) | SSRA S2 b16 = 12 335 tok/s / 41.21 GiB; projekcia 850M ≈ 25 EUR @ SXM |
| dáta | `gs://ssra-poc-ew3/m2/data/m2-data-900m/` (train 913 605 620 / val 48 050 671 / val-eval-2M 2 000 000 tok; sha v HO-13 tabuľke, nezmenené) |
| sweep artefakty | logy + `latest.pt` v `gs://ssra-poc-ew3/m2/sweep/{run_name}/`, ploty v `.../m2/sweep/plots/`; repo commity `482bdb5`/`21a4d9d`/`d752762` |
| P-C | `p1_attn_entropy` ≈ ln(32) trvá aj pri 60M tok (informatívne; participácia bez kolapsu) |
| kumulatívny spend | **11.75 EUR ≈ 3.9 %** z 300 EUR (4.05 + 0.85 + 0.49 + 6.36; všetko FINAL) |

## 4. Otvorené položky

| # | položka | kedy |
|---|---|---|
| 1 | **Phase 3 zadanie autorovať** (S2 850M core pair, vybrané hyperparametre lr 1e-3 / dropout 0.0, strop 30 EUR scoped na tento run, AP-12 brána ≤ 25 EUR/run vnútri; `val-eval-2M` ako G1 eval set potvrdiť v zadaní) | nový chat, prvá úloha |
| 2 | **Idle cost vzor (3. výskyt):** pri dlhých behoch plánovať terminate okno — signál CC vs. Danielova dostupnosť v konzole (S2 850M run ≈ 19 h ⇒ idle hodina je drahšia než pri sweepe) | do Phase 3 zadania |
| 3 | Zenodo erratum/v1.1 | odložené, po zozbieraní M2/M3 nálezov |
| 4 | fla/transformers/torchvision pin task (známy fail `test_loglinear_integration`; na sweep pode `operator torchvision::nms does not exist`, report §B.2) | pred Phase 4 |
| 5 | ⚠ Revert záväzok: zmazať SA kľúč + vrátiť org-policy enforcement | po M2/M3 |
| 6 | Project files vymeniť: `00`, `results/M2-sweep.md`, tento HO | po push |

## 5. Čoho sa nedotýkať

Uzavreté rozhodnutia (D1–D6, Q1–Q5, MD-1…MD-13, AP-1…AP-22 + úprava AP-19 a pravidla odpočtu z D-logu 2026-07-14) sa neotvárajú. Spec v1.2 normatívna; existujúce testy sa nemodifikujú. Selekcie (1e-3, 0.0) sú vstup Phase 3 — už sa nere-behujú ani „nevylepšujú". Dáta a tokenizer finálne (`data_scale.py` sa nespúšťa). Žiadne kvalitatívne závery zo sweep lossov; SSRA-vs-flat loss porovnania nie sú evidencia ničoho (spec §16). Strop 30 EUR = výhradne Phase 3 S2 850M run. Committed logy a GCS objekty sa neprepisujú (AP-21).

## 6. Otvárací prompt pre Phase 3 chat (copy-paste)

```
Vstup: docs/handover/HO-14-2026-07-14-m2-phase2-closed.md + docs/00
(D-log 2026-07-14, štyri riadky) + results/M2-sweep.md (§B.4 selekcie).
Úloha: autorovať zadanie Phase 3 (docs/cc/) — S2 850M core pair SSRA vs
flat, hyperparametre lr 1e-3 / dropout 0.0 (selekcie zo sweepu), seed 1337,
strop 30 EUR scoped na S2 SSRA run, AP-12 vnútri, val-eval-2M potvrdiť ako
G1 eval set, terminate okno plánovať vopred (idle vzor 3×). Žiadny beh bez
committed YAML; AP-21 mená.
```
