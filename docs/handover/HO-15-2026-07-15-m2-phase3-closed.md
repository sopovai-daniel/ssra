# HO-15 — M2 Phase 3 behy UZAVRETÉ: vecný výsledok NEGATÍVNY (SSRA spike bez zotavenia), G1 verdikt PENDING; ďalej = oversight review + spike diagnostika + rozhodnutie o 1 retune iterácii

**Dátum:** 2026-07-15 · **Autor:** Claude (Fable 5) + Daniel (rozhodnutia) · **Predchádzajúci:** HO-14
**Vstupný bod nového chatu:** tento dokument + `docs/00-stav-a-triaz.md` (D-log 2026-07-15, jeden riadok) + `results/M2-core-pair.md` (celý, vrátane spike diagnostiky a §x open questions)

---

## 1. Stav jednou vetou

Phase 3 core pair dobehol infraštruktúrne bezchybne (pod self-terminated cez AP-23, nulový idle tail, SSRA run 25.69 EUR ≤ 30 EUR scoped cap), ale vecne negatívne — **SSRA utrpel konečný loss spike na kroku 6 475→6 500 a nezotavil sa** (final ppl 1 917.6 vs flat 24.829, pomer 77.23×), oba G1 vstupy čítajú FAIL; **chat, ktorý číta tento dokument, robí oversight review reportu, spike diagnostiku z checkpointov a pripravuje podklad pre Danielov G1 verdikt + rozhodnutie o max 1 retune iterácii per `03`.**

## 2. Čo sa stalo (2026-07-14 večer → 2026-07-15)

1. **Zadanie** `docs/cc/M2-phase3-core-pair.md` v1 (`391b671`): S2 850M core pair, selekcie (lr 1e-3, do 0.0), seed 1337, 51 880 krokov = 850 001 920 tok; potvrdilo `val-eval-2M` ako G1 eval set; nové AP-23 (self-terminate); early cost gate (30 EUR in-flight vynútenie).
2. **CC lokálna príprava** (`76fc814`, `73a783c`): configy z jednej šablóny (diff = 4 riadky), dry-run params flat 84 301 440 / SSRA 84 647 040 (gap +0.41 % < 1 %, AP-8 OK), 4× sha256 brány z manifestu, `scripts/cost_gate.py` (read-only supervisory, AP-20 neaktivované), AP-23 check v bootstrape.
3. **Pod** `ssra-m2-core` (`bxwa0whm15v8mi`), A100 SXM 80 GB Secure, **EUR-IS-1** (EU preferencia splnená), $1.50/hr total; AP-19 krok 0: Community not capturable (5. výskyt); cgroup 27.2 vCPU → thready 27; bundle na `73a783c`; AP-17 brána PASSED; pytest 64/1 známy fail.
4. **flat run:** čistý end-to-end, 137.3k tok/s / 10.85 GiB, final_eval_loss **3.21201** (ppl 24.829).
5. **SSRA run:** early cost gate PASS (12 387 tok/s = +0.42 % vs recal kotva; projekcia 25.01 EUR); peak 41.2 GiB = recal; **spike krok 6 475→6 500, konečný (žiadny NaN/inf), bez zotavenia počas zvyšných 45 380 krokov**; beh legálne dobehol všetkých 51 880 krokov — in-flight abort bol v zadaní definovaný len na NaN/inf (nedostatok zadania, vlastníctvo Claude, priznané; ≈ 20 EUR post-spike behu). Final_eval_loss **7.55885** (ppl 1 917.6).
6. **P-C nález:** `p1_attn_entropy` ≈ ln(32) uniformná CEZ spike okno, de-uniformizácia až PO ňom — **symptóm, nie prekurzor** (vstup pre design analýzu v Claude.ai projekte, nie záver).
7. **AP-23 prvé použitie:** 1. invokácia zlyhala (pod id + API key len v PID-1 env — známy vzorec), retry uspel, terminácia 15:32 UTC potvrdená connection-refused, **nulový idle tail** (4. výskyt idle vzoru sa nekonal).
8. **Ledger FINAL:** $32.9000 ≈ **28.78 EUR** (ECB 1.1430; odpočet ≥ 2 h po terminácii per pravidlo 2026-07-14). Dekompozícia [ODHAD]: SSRA ≈ 25.69 ≤ 30 cap ✓ + flat ≈ 2.68 + réžia ≈ 0.4. **Kumulatív M2: 40.53 EUR ≈ 13.5 % z 300.**

## 3. Kľúčové fakty [záväzné — nere-merať]

| artefakt | hodnota |
|---|---|
| G1 vstup flat | final_eval_loss 3.21201 · **ppl 24.829** (val-eval-2M, ctx 1024) |
| G1 vstup SSRA | final_eval_loss 7.55885 · **ppl 1 917.639** · pomer 77.23× |
| stabilita | flat čistý; SSRA konečný spike 6 475→6 500 bez zotavenia; oba behy 51 880/51 880 krokov, identický token stream, byte-identický eval |
| výkon | SSRA 12 387 tok/s / 41.2 GiB (= recal kotvy); flat 137.3k tok/s / 10.85 GiB |
| checkpointy | GCS `gs://ssra-poc-ew3/m2/core/{run_name}/` — spike okno zabalené ckptmi **6 000 / 7 000** (ckpt_every 1 000) ⇒ diagnostika lokálne, bez GPU spendu |
| repo | main `8bde937` (report + ploty + logy + runs.md); zadanie `391b671`; prep `76fc814`/`73a783c` |
| spend | pod $32.9000 ≈ 28.78 EUR FINAL; kumulatív **40.53 EUR ≈ 13.5 %** |

## 4. Otvorené položky (poradie záväzné)

| # | položka | kto |
|---|---|---|
| 1 | **Oversight review** `results/M2-core-pair.md` — nezávislý prepočet ppl, pomeru, ledger aritmetiky, tok/s konzistencie, spike timeline z logov (štandard: reimplementovať odvodenia, nie čítať report) | Claude, nový chat |
| 2 | **Spike diagnostika** z ckpt 6 000/7 000 + logov (grad normy, per-vrstvové štatistiky, P-C okno) — lokálne, zadarmo, PRED akýmkoľvek rozhodnutím | Claude/CC lokálne |
| 3 | **G1 verdikt** — vstupy čítajú FAIL na oboch ramenách (stabilita aj ±5 % pásmo) | **Daniel** |
| 4 | **Rozhodnutie o max 1 retune iterácii per `03`** — kandidát: symetrický pár @ lr 6e-4 (sweep runner-up, pre-deklarovaný), ≈ 28 EUR [ODHAD]; alternatíva: bez iterácie, reportovať negatívny výsledok | **Daniel**, po #1–#2 |
| 5 | **AP-24 návrh** (enumerovaný in-flight STOP pri konečnom pretrvávajúcom val regrese; pauza + ping, rozhoduje človek — nie tichý auto-abort) | Daniel |
| 6 | Project files vymeniť: `00`, `M2-core-pair.md`, tento HO, prípadne zadanie | Daniel, po push |
| 7 | Zenodo erratum/v1.1 — odložené (zbierka M2/M3 nálezov rastie) | neskôr |
| 8 | fla/transformers pin (Phase-4-only fail) | pred Phase 4 |
| 9 | ⚠ Revert záväzok: SA kľúč + org-policy enforcement | po M2/M3 |

## 5. Čoho sa nedotýkať

Uzavreté rozhodnutia (D1–D6, Q1–Q5, MD-1…MD-13, AP-1…AP-23) sa neotvárajú. Spec v1.2 normatívna; testy sa nemodifikujú. Sweep selekcie boli mechanicky správne — spike na S2 ich spätne neinvaliduje (scale-transfer je [HYPOTÉZA] pre diagnostiku, nie dôvod prepísať sweep). **Žiadne architektonické závery z ppl pomeru — 77× je G1 vstup, nie evidencia mechanizmu (spec §16); interpretácia spiku patrí do design analýzy.** G1 verdikt aj retune rozhodnutie sú výhradne Danielove. Committed logy a GCS objekty sa neprepisujú (AP-21). Negatívny výsledok je publikovateľný výsledok (D-log 2026-06-09) — žiadne „záchranné" asymetrické zásahy.

## 6. Otvárací prompt pre nový chat (copy-paste)

```
Vstup: docs/handover/HO-15-2026-07-15-m2-phase3-closed.md + docs/00 (D-log
2026-07-15, jeden riadok) + results/M2-core-pair.md (cely, vratane spike
sekcie a §x open questions).
Ulohy v poradi: (1) oversight review reportu — nezavisly prepocet ppl,
pomeru, ledger aritmetiky, tok/s konzistencie, spike timeline z logov;
(2) spike diagnostika z GCS checkpointov 6000/7000 lokalne (bez GPU
spendu) ako vstup pre rozhodnutie; (3) podklad pre G1 verdikt (Daniel) +
rozhodnutie o max 1 retune iteracii per 03 (kandidat: symetricky par
@ lr 6e-4, ~28 EUR); (4) AP-24 navrh na D-log zapis. Ziadne
architektonicke zavery z ppl pomeru (spec §16); G1 verdikt a retune su
Danielove rozhodnutia.
```

---

## Addendum A1 (2026-07-15, po oversight review + spike diagnostike)

**Korekcia §1 (stav jednou vetou) a §3 (riadok „checkpointy“):** tvrdenie „spike okno zabalené ckptmi **6 000 / 7 000** (ckpt_every 1 000) ⇒ diagnostika lokálne, bez GPU spendu“ bolo **nesprávne** — neoverená domnienka odvodená z kadencie checkpointovania bez overenia retenčnej sémantiky (vlastníctvo: Claude; presne trieda chyby, ktorú Pravidlo W zakazuje). Skutočnosť [OVERENÉ, `results/M2-spike-diagnostics.md` §1 + `src/ssra/checkpoint.py`]: `save_checkpoint` zapisuje a mirroruje výhradne `latest.pt` pod tým istým menom objektu, bucket versioning OFF (1 generácia/objekt) ⇒ per-step checkpointy neexistujú a T2–T4/T6 diagnostiky boli pre Phase 3 beh nevykonateľné; vykonaný bol T5 (negatívny — okno korpusovo typické). Evidencia pre Phase 3 je definitívne log-only. Otvorené položky #1–#5 tohto HO sú uzavreté D-logom 2026-07-15 (druhý zápis): G1 = FAIL, AP-24 prijaté, retune GO per `docs/cc/M2-phase3b-retune.md` — ktorého inštrumentácia (grad-norm logging + step-tagged ckpt mirror) uzatvára presne túto evidenčnú medzeru.
