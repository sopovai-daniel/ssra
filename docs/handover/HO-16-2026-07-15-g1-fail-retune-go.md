# HO-16 — Oversight + diagnostika UZAVRETÉ: G1 = FAIL, AP-24 prijaté, RETUNE GO (Phase 3b @ lr 6e-4); ďalej = supervízia CC exekúcie Phase 3b

**Dátum:** 2026-07-15 · **Autor:** Claude (Fable 5) + Daniel (rozhodnutia) · **Predchádzajúci:** HO-15 (+ addendum A1)
**Vstupný bod nového chatu:** tento dokument + `docs/00-stav-a-triaz.md` (D-log 2026-07-15, druhý zápis) + `docs/cc/M2-phase3b-retune.md` (celé)

---

## 1. Stav jednou vetou

Oversight review Phase 3 reportu (VERIFIED; vecná korekcia = druhá epizóda nestability, append-only §xi) aj spike diagnostika (per-step checkpointy neexistujú ⇒ evidencia log-only; T5 dátové okno korpusovo typické) sú uzavreté; Daniel vyniesol **G1 = FAIL**, prijal **AP-24** (auto-STOP tvar) a schválil **jedinú povolenú retune iteráciu** — symetrický pár @ lr 6e-4, seed 1337 fixný (izolácia jedinej premennej), inštrumentácia (grad_norm + step-tagged ckpt), AP-24 aktívna, scoped strop 30 EUR; **chat, ktorý číta tento dokument, superviduje CC exekúciu zadania `docs/cc/M2-phase3b-retune.md`** (lokálna príprava → „ready for pod" → deploy → beh → report); oversight review výsledného reportu pôjde opäť novým chatom.

## 2. Čo sa stalo (2026-07-15, po HO-15)

1. **Oversight review** `results/M2-core-pair.md`: VERIFIED — všetky G1 vstupy, ledger, tok/s a spike timeline nezávisle prepočítané z raw logov; 1 vecná korekcia: **druhá epizóda nestability** (155 train záznamov > 9 natov v 16 675–22 100, train max 10.351, val max 10.088 @ 19 800; usadenie 7.52–7.59 až od ~35k) — report doplnený append-only sekciou **§xi C1–C6** (commit `c102998`), vrátane AP-24 retro-testu (fire @ 7 600, úspora ≈ 21.9 EUR, 0 false positives na flat).
2. **Spike diagnostika** (`docs/cc/M2-spike-diagnostics.md`, CC lokálne, 0 EUR, commit `ba4ca12`): **T0 — per-step checkpointy NEEXISTUJÚ** (`save_checkpoint` mirroruje výhradne `latest.pt` pod tým istým menom; bucket versioning OFF, 1 generácia/objekt) ⇒ T2–T4/T6 nevykonateľné; HO-15 tvrdenie o ckpt 6000/7000 bola neoverená domnienka (falzifikovaná; addendum A1, vlastníctvo Claude, Pravidlo W lekcia). **T5 vykonané:** okno 6 450–6 550 exaktný replay cez harnessovú `batches()` — korpusovo typické (jediný flag v spike okne: eot z = 3.27 @ 6 487, per-window v baseline rozsahu); oversight prepočítal všetkých 404 z-skóre z CSV, 0 nezhôd. **AP-20 file-list check `ba4ca12` [OVERENÉ z GitHub]: 8 súborov, všetky `added` (results/ + scripts/diagnostics/), 0 modifikácií — model/harness/testy nedotknuté.**
3. **Rozhodnutia (Daniel, D-log 2026-07-15 druhý zápis):** G1 = FAIL (stabilita + ±5 % pásmo); AP-24 prijaté: val_loss > running-best + 2.0 natov na ≥ 6 po sebe idúcich val evaloch ⇒ ckpt + GCS upload + ABORTED-instability + push + AP-23 self-terminate, človek rozhodne resume vs uzavrieť (AP-11 vratnosť), NaN/inf trigger nadradený, symetrické, nie retroaktívne; **Retune GO** so scoped stropom 30 EUR na SSRA rameno.
4. **Zadanie `docs/cc/M2-phase3b-retune.md` v1** vytvorené (commit `eb5ea8f`); odovzdanie CC = schválenie (pre-flight veto konvencia).

## 3. Kľúčové fakty [záväzné — nere-merať]

| artefakt | hodnota |
|---|---|
| G1 Phase 3 | **FAIL** (flat ppl 24.829 vs SSRA 1 917.639, pomer 77.23×; spike 6 475→6 500 + druhá epizóda 16 675–22 100, bez zotavenia) |
| retune parametre | symetrický pár @ **lr 6e-4**, dropout 0.0, **seed 1337 fixný** (jediná zmenená premenná = lr), steps 51 880, b16, warmup 778; mená `m2-core-{flat,ssra}-s2-850m-lr6e4` (AP-21) |
| inštrumentácia | `grad_norm` do JSONL + `step-<N>.pt` GCS mirror popri `latest.pt`; brána: §14 green (64/1 známy fail) + AP-11 resume test + **bit-identický 60-krokový CPU smoke** pred/po commite |
| AP-24 (ostrá) | trigger 6× po sebe val > best + 2.0; akcia ckpt→upload→ABORTED-instability→push→AP-23; retro na Phase 3: fire @ 7 600 |
| náklady | kumulatív M2 **40.53 EUR ≈ 13.5 %**; Phase 3b očakávanie ~28.5 EUR stabilný / ~7–8 EUR recídiva [ODHAD]; **strop 30 EUR scoped len na SSRA rameno**; ECB 1.1430 carry |
| repo | main `eb5ea8f` (D-log + HO-15 A1 + zadanie) ← `ba4ca12` (diagnostika) ← `c102998` (§xi + diag zadanie) |
| diagnostická pripravenosť | `scripts/diagnostics/` T2–T4 hotové a smoke-tested (mapovanie optimizer↔mená shape-overené 57/57); pri AP-24 stope okamžite vykonateľné na step-tagged ckptoch |

## 4. Otvorené položky (poradie záväzné)

| # | položka | kto |
|---|---|---|
| 1 | CC lokálna príprava per zadanie §2–§4 (configy + inštrumentácia + verifikačná brána + AP-24 implementácia) → signál „ready for pod" | CC |
| 2 | Deploy (kredit ≥ $40, AP-19 krok 0, booked rate + break-even prepočet, Pravidlo W) | Daniel |
| 3 | Beh: flat lr6e4 → SSRA lr6e4 (early cost gate, AP-24 ostrá) → AP-23 terminate | CC/pod |
| 4 | Report `results/M2-core-pair-lr6e4.md`; pri faile navyše `results/M2-spike-diagnostics-lr6e4.md` (T2–T4 + grad_norm timeline) a STOP | CC |
| 5 | Oversight review → **finálny G1 verdikt (Daniel)** → HO-17 + uzavretie M2 | Claude + Daniel, nový chat |
| 6 | Project files výmena po každom pushi | Daniel |
| 7 | GCS step-tagged ckpty zmazať po analýze (post-run rozhodnutie) | Daniel |
| 8 | Zenodo erratum/v1.1 — odložené; spike + retune výsledok = ďalší kandidát obsahu | neskôr |
| 9 | fla/transformers pin (Phase-4-only fail) | pred Phase 4 |
| 10 | ⚠ Revert záväzok: SA kľúč + org-policy enforcement | po M2/M3 |

## 5. Čoho sa nedotýkať

Uzavreté rozhodnutia (D1–D6, Q1–Q5, MD-1…MD-13, **AP-1…AP-24**) sa neotvárajú. Spec v1.2 normatívna; testy sa nemodifikujú. **Seed 1337 a single-variable izolácia sú binding** (zadanie §1) — žiadne ďalšie hyperparametrové zmeny, žiadny tretí pár; pri faile fallback (b)/(c) per `03`, bez ďalšej iterácie. Žiadne architektonické závery z ppl pomeru (spec §16). Committed logy a GCS objekty sa neprepisujú ani nemažú (AP-21; mazanie step-tagged ckptov = výhradne Danielovo post-run rozhodnutie). Negatívny výsledok je publikovateľný výsledok — žiadne „záchranné" asymetrické zásahy.

## 6. Otvárací prompt pre nový chat (copy-paste)

```
Vstup: docs/handover/HO-16-2026-07-15-g1-fail-retune-go.md + docs/00 (D-log
2026-07-15, druhy zapis) + docs/cc/M2-phase3b-retune.md (cele).
Kontext: G1 Phase 3 = FAIL; AP-24 prijate (auto-STOP); retune GO — jedina
povolena iteracia, symetricky par @ lr 6e-4, seed 1337 fixny, scoped strop
30 EUR na SSRA rameno.
Ulohy: supervizia CC exekucie zadania (lokalna priprava -> ready for pod ->
deploy -> beh -> report); interpretacia CC outputov; flagovanie nakladov;
pri AP-24 stope alebo faile enumerovana diagnostika per zadanie §6 a STOP.
Ziadne architektonicke zavery z ppl (spec §16); G1 verdikt po behu je
Danielov; oversight review reportu pojde dalsim novym chatom.
```
