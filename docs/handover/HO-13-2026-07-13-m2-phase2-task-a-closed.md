# HO-13 — M2 Phase 2 spustená: zadanie v1.1 + Task A (dáta 913M) uzavretá; ďalej = Task B sweep launch

**Dátum:** 2026-07-13 · **Autor:** Claude (Fable 5) + Daniel (rozhodnutia) · **Predchádzajúci:** HO-12
**Vstupný bod nového chatu:** tento dokument + `docs/00-stav-a-triaz.md` (D-log 2026-07-13, tri riadky) + `docs/cc/M2-phase2-sweep.md` v1.1 + `results/M2-sweep.md` (Task A sekcia)

---

## 1. Stav jednou vetou

Phase 2 zadanie (`docs/cc/M2-phase2-sweep.md` v1.1: symetrický S1 lr/dropout sweep per AP-14 + nové AP-21/AP-22) je v repe a Task A je dodaná — 913.6M train tokenov v GCS (`m2/data/m2-data-900m/`) s plnou AP-9 provenance, vyrobené na dedikovanom CPU pode (~0.5 EUR, terminovaný), oversight commit check PASS; **chat, ktorý číta tento dokument, riadi Task B: sweep na GPU pode — poradie CC lokálna príprava → „ready for pod" signál → deploy.**

## 2. Čo sa v tomto sedení stalo

1. **Phase 2 zadanie napísané** (v1, potom v1.1 po rozhodnutí Daniela presunúť Task A na CPU pod). Kľúčové [ODHAD, veto — bez veta]: 60M tokenov/run, b16 (zhoda s S2 gate batchom), seed 1337 všade, selekcia = final val loss na `val-eval-2M`; divergencia = platný výsledok bez retry; A100 80 GB trieda odporúčaná (24 GB karta = len 23 % headroom + láme porovnateľnosť).
2. **Nové AP (prijaté odovzdaním CC):** AP-21 run identity (unikátny `run_name` → log + GCS cesty; re-exekúcia = `-r1`…; commitnuté artefakty sa neprepisujú), AP-22 D5 ×1.20 error bar (launch gate ≤ 76 GiB / 80 GB). Strop 30 EUR = VÝHRADNE Phase 3 S2 850M run.
3. **Task A na CPU pode `ssra-m2-data`** (16 vCPU / 32 GB / 60 GB, $0.568/hr, runpod-ubuntu template — deviácia bez torch, pytest gate sa presúva na Task B pod), ~55 min vrátane ~23 min idle, terminovaný. Dodávka: commit `e70f2b9`; dva import-fixy (`28268b4`, `7bb2d1a`) — oversight check z gitu: len `scripts/data_scale.py`, model package netknutý.
4. **AP-21 výklad prijatý (Daniel):** suffix povinný od momentu, keď run čokoľvek zapísal; import-time abort bez zápisu suffix nevyžaduje.
5. **Prevádzkové lekcie:** pod deployovať až na explicitný „ready for pod" signál CC (idle ~0.2 EUR z tohto sedenia); Filesystem MCP dnes 2× vypadol (4-min timeouty) — pri zlyhaní docs zápisov fallback = plný obsah do chatu + `pbpaste >`.

## 3. Kľúčové fakty [záväzné — nere-merať, nere-hashovať]

| artefakt | hodnota |
|---|---|
| `train.bin` | 913 605 620 tokenov, sha256 `6d0e47cd…` |
| `val.bin` | 48 050 671 tokenov, sha256 `03e0dd1a…` |
| `val-eval-2M.bin` | 2 000 000 tokenov, sha256 `bde526d2…` (re-hash proti prefixu val.bin ✓) |
| GCS | `gs://ssra-poc-ew3/m2/data/m2-data-900m/` |
| provenance | fineweb-edu `sample-10BT`, hub rev `87f09149` (= Phase 0, žiadny drift), odc-by, načítané 2026-07-13 |
| tokenizer | FROZEN, sha256 `019568a2…` |
| sweep referenčné tok/s | SSRA S1 b16 = 27 079 · flat S1 b16 = 319 945 (recal `819fcb2`) |
| kumulatívny spend | ≈ 4.4 EUR ≈ 1.5 % stropu [ODHAD do doplnenia konzoly] |

## 4. Otvorené položky

| # | položka | kedy |
|---|---|---|
| 1 | Fakturovaná suma CPU podu z konzoly (~1 h delay) → CC doplní 2 placeholdery (`results/M2-sweep.md` §A.5 + `results/runs.md`) + commit + push | Daniel, dnes |
| 2 | **Task B launch:** CC prompt = lokálna príprava YAML (AP-21 mená) → STOP → „ready for pod" → Daniel deploy (AP-19 krok 0: Community cena PRED bookovaním — 3. prenos!) → SSH string CC → sweep ~3–4 h → terminate → report | nový chat, prvá úloha |
| 3 | Danielov verdikt nad (lr, dropout) selekciami zo sweep reportu → potom Phase 3 zadanie (S2 core pair, strop 30 EUR) | po Task B |
| 4 | transformers pin kompatibilný s fla 0.5.0 | pred Phase 4 |
| 5 | ⚠ Revert záväzok: zmazať SA kľúč + vrátiť org-policy enforcement | po M2/M3 |
| 6 | Zenodo erratum/v1.1 | po zozbieraní M2 nálezov |
| 7 | Project files vymeniť: `00`, `docs/cc/M2-phase2-sweep.md`, `results/M2-sweep.md`, tento HO | po push |

## 5. Čoho sa nedotýkať

Uzavreté rozhodnutia (D1–D6, Q1–Q5, MD-1…MD-13, AP-1…AP-22) sa neotvárajú. Spec v1.2 normatívna; existujúce testy sa nemodifikujú. Tokenizer sa NIKDY nepretrénuje (sha gate v pipeline). Žiadne kvalitatívne závery zo sweep lossov — selekcia je mechanická a within-model; SSRA-vs-flat loss porovnania nie sú evidencia ničoho (spec §16). Strop 30 EUR ≠ Phase 2; všetko vo Phase 2 pod AP-12. `ssra/__init__` torch import = nit bez akcie (housekeeping najskôr po M2). Pipeline `data_scale.py` sa v Phase 2/3 už nespúšťa — dáta sú finálne.

## 6. Otvárací prompt pre Task B chat (copy-paste)

```
Vstup: docs/handover/HO-13-2026-07-13-m2-phase2-task-a-closed.md + docs/00
(D-log 2026-07-13, tri riadky) + docs/cc/M2-phase2-sweep.md v1.1.
Úloha: riadiť Task B (sweep). Krok 1: priprav CC prompt na lokálnu prípravu
(YAML configy 6 stage-1 runov, AP-21 mená, harness na nové shardy
m2/data/m2-data-900m/, STOP + „ready for pod" signál). Deploy podu až po
signále; AP-19 krok 0 pred bookovaním.
```

---

## Addendum (2026-07-13 večer) — billing korekcia

Konzolový večerný odpočet zmenil čísla: **`ssra-m2-cal` (Phase 1) = $4.6293 ≈ 4.05 EUR** (nie $3.4786 z 2026-07-12 — skorý odpočet nebol finálny; „25 % pod odhadom" mystérium tým vysvetlené). **`ssra-m2-data` = $0.5567 ≈ 0.49 EUR** (finálne pre dnešok). `ssra-m2-recal` $0.9700 = **provizórne, re-check T+1** (môže dorásť k ≈ $1.30). **Kumulatív ≈ 5.39 EUR ≈ 1.8 % stropu.** Nové pravidlo (D-log 2026-07-13, korekčný riadok): konzolové sumy sú provizórne do potvrdenia na T+1; ledger korekcie výhradne append. **Pre Task B chat: pri zápise fakturácie sweep podu označiť sumu ako provizórnu a naplánovať T+1 re-check; recal T+1 re-check vykonať tiež.**
