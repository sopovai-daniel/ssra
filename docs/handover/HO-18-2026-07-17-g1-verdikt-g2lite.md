# HO-18 — 2026-07-17 — G1 verdikt: FAIL (formulácia B); G2-lite autorizované; Phase 3b účtovne uzavretá

**Účel:** kontinuita do nového chatu = autorstvo pre-registrovaného „Zadanie pre CC: M2 G2-lite". Autoritatívny stav: `docs/00-stav-a-triaz.md` (D-log 2026-07-17, **druhý** zápis). Tento HO nič nerozhoduje — sumarizuje a odovzdáva.

## §1 Výsledok v jednej vete

Oversight review Phase 3b reportu = **VERIFIED, 0 vecných korekcií** (všetko nezávisle prepočítané z raw JSONL logov); **G1 verdikt (Daniel) = FAIL** — stabilita PASS, ±5 % pásmo FAIL (+10,22 %), druhý fail per `03`; **formulácia B**: mechanický zápis + amendment autorizujúci jedno pre-registrované, inference-only **G2-lite** meranie (strop 10 EUR) na existujúcich checkpointoch; fakturácia FINÁLNA 31,25 EUR, kumulatív **71,78 EUR ≈ 23,9 %**.

## §2 Kľúčové fakty

| položka | hodnota |
|---|---|
| G1 vstupy (verifikované z raw logov) | flat 3,19333 (ppl 24,369) · SSRA 3,29065 (ppl 26,860) · +10,22 % · Δ 0,09732 natu |
| G1 verdikt | **FAIL** (druhý fail; retune bola jediná povolená iterácia) — stabilita PASS (0 flagov, AP-24 counter 0, spike sa nezopakoval), pásmo FAIL |
| Formulácia B | amendment k `03` „prechod na report": **G2-lite** — inference-only, oba modely, finálne checkpointy, **scoped strop 10 EUR**, protokol commitnutý PRED spustením, výsledok sa publikuje bez ohľadu na smer; žiadny tréning/tuning |
| R1 ckpt retencia | variant (i): držať 106 objektov ≈ 100,1 GiB ≈ 2,01 EUR/mes; rozhodnúť pri zamrazení draftu paperu, najneskôr 2026-08-31 |
| R2 AP-24 tail | (a) fallback akceptovaný; (c) GCS-mirror precondition pre-schválená ako povinná pred behom > 4 h bez dozoru; (b) push credential zamietnuté |
| R3 paper invarianty | parity gap +10,22 % (AP-8 honesty note) · lr-stability nález (užší stabilný lr rozsah, mechanizmus neurčený) · konštanta 11,8× · P-C uniformita Q_φ · spec §9 erratum kandidát · Limitations per M4 · + G2-lite |
| Fakturácia | **FINAL $35,7192305624485 ≈ 31,25 EUR** (ECB 1,1430; odpočet ≥ 2 h po terminácii potvrdený); kumulatív M2 **71,78 EUR ≈ 23,9 % z 300** |
| Checkpointy pre G2-lite (GCS) | `gs://ssra-poc-ew3/m2/core/m2-core-flat-s2-850m-lr6e4/latest.pt` (1 011 848 651 B) · `.../m2-core-ssra-s2-850m-lr6e4/latest.pt` (1 016 124 393 B); configy commit `3db45ef` |
| Oversight metodika | JSONL replay (AP-24 counter, running best, max regresie 0,00279/0,00334 presne), grad_norm okno steps ≥ 40k reprodukuje reportované rozsahy presne, windowed 12 404 tok/s = pure-train po odpočítaní 2 val evalov, ledger krížové testy sedia na ≈ 14 s |

## §3 Vstupy pre G2-lite zadanie (dizajnové body, riešiť v novom chate PRED spustením)

1. **Pozičné kódovanie flat baselinu** — rozhoduje o férovosti extrapolácie za tréningovú dĺžku 1024; overiť zo `docs/spec.md` + harnessu (Pravidlo W, žiadne domnienky). Ak flat neextrapoluje štrukturálne, protokol to musí priznať a dizajnovať okolo toho (napr. reporting per-model limity), nie zamlčať.
2. **Existencia needle generátora** — `03` M3 ho plánuje ako „vlastný generátor v repe"; overiť, či existuje; ak nie, zadanie obsahuje implementáciu + testy (lokálne, 0 EUR).
3. **Rozsahy N** (návrh na diskusiu: 1k/2k/4k/8k, prípadne vyššie) — VRAM projekcie povinné s **AP-22 ×1,20** error barom; launch gate ≤ 76 GiB.
4. **Metriky + interpretačné kritériá napísané VOPRED** (ppl vs dĺžka, per-position loss ak lacné; needle exact-match); žiadne architektonické závery (spec §16) — výstupy sú vstup pre rozhodnutie o tvare paperu.
5. **Cost plan pod 10 EUR** + AP checklist (12/17/18/19 krok 0/21/23); beh krátky a pod dozorom ⇒ R2(c) redesign sa neaktivuje.

## §4 Otvorené položky pre ďalšie chaty

1. **Nový chat:** autorstvo „Zadanie pre CC: M2 G2-lite" (pre-registrované; §3 body).
2. CC exekúcia G2-lite: lokálna príprava → „ready for pod" → Daniel deploy → beh → report.
3. Oversight review G2-lite + podklad pre rozhodnutie o tvare paperu (negatívny/analytický vs zmiešaný) — s dátami.
4. M4 draft (Claude) → Danielov review; Zenodo erratum bundling (standing rozhodnutie z 2026-06-12).
5. Retencia review pri zamrazení draftu / 2026-08-31 (R1).
6. Standing: AP-19 pozastavené do znovuobjavenia Community tieru; revert záväzok GCP SA kľúč + org-policy override po M2/M3.

## §5 Ohraničenia (binding, nezmenené)

Spec v1.2 frozen; **G1 = FAIL je zapísaný a neotvára sa**; G2-lite je jediné autorizované meranie (strop 10 EUR, žiadny tréning, žiadny tuning, žiadna ďalšia iterácia); žiadne architektonické závery z ppl ani z lr-nálezu (spec §16); Pravidlo W na všetko mimo kontextu; ledger append-only.

## §6 Otvárací prompt pre nový chat (copy-paste)

```
Vstup: docs/handover/HO-18-2026-07-17-g1-verdikt-g2lite.md + docs/00
(D-log 2026-07-17, druhy zapis) + results/M2-core-pair-lr6e4.md (§iv,
§viii) + docs/03-poc-plan.md (M3 sekcia).
Kontext: G1 = FAIL (formulacia B); autorizovane G2-lite — pre-registrovane,
inference-only meranie na finalnych checkpointoch lr6e4 paru, scoped strop
10 EUR; ziadny trening, ziadny tuning, ziadna dalsia iteracia.
Uloha: napisat "Zadanie pre CC: M2 G2-lite" — protokol pre-registrovany
PRED spustenim: (1) overit pozicne kodovanie flat baselinu zo
spec/harnessu (fair extrapolacia; Pravidlo W); (2) overit existenciu
needle generatora v repe; (3) navrhnut rozsahy N + VRAM projekcie s
AP-22 x1.20; (4) metriky + interpretacne kriteria napisane VOPRED;
(5) cost plan pod 10 EUR + AP checklist (12/17/18/19 krok 0/21/23).
Vysledok sa reportuje bez ohladu na smer. Ziadne architektonicke zavery
(spec §16).
```
