# 03 — PoC plán (GCP / Vertex AI)
**Aktualizované 2026-06-10:** doplnené dôsledky uzavretého M0 dizajnu (completion test, micro-gate G1b-D3, multi-needle sada, rozšírený ablačný plán) a **dvojstupňová publikačná stratégia (D-log B2)**. Štruktúra míľnikov nezmenená.

**Definícia úspechu projektu:** jasná, publikovateľná odpoveď na H1–H2 (pozitívna ALEBO negatívna) pri matched compute. Nafúknuté pozitívum bez rigoru = zlyhanie projektu. Negatívny výsledok s poctivým setupom = legitímny publikovateľný výstup.

## Orientačná časová os (odhady; popri klientskej práci; gates sú podmienené prechody, nie dátumy)
```
            jún         júl         aug         sep
W →     24 25 26 27 | 28 29 30 31 | 32 33 34 35 | 36 37 38
M0 spec ██ ██
M1 impl       ██ ██
M2 runy             ██ ██ ██ ██
M3 eval                         ██ ██ ██ ██
M4 paper            ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ██ ██ ██   (░ = draft priebežne)
Pub S1:        ▲ Zenodo DOI po G0 (technical note)
Pub S2:                                          ▲ plný paper po G2
```
M0/M1 odhady z plánu; M2 2–4 týždne [ODHAD], M3 3–4 týždne [ODHAD]. Stop-loss (nižšie) platí; hard stop ~koniec novembra 2026.

## M0 — Formalizácia (Claude.ai projekt; ~1–3 týždne popri klientskej práci)
- ✔ Diery D1–D6 a otázky Q1–Q5 uzavreté (2026-06-09/10; `01-mechanizmus.md` §4–§6, D-log v `00`).
- ✔ Zložitosť SSRA v2 odvodená pre variant A: tréning Θ(N·(w + m·log N)·d)/vrstva, decode stav O(m·d·log N)/vrstva (`01` §7).
- Zostáva: novelty téza v1 (z v0.2 v `02`) → `docs/spec.md` (jediný zdroj pravdy pre implementáciu) → **Gate G0 check**.
- **Gate G0:** blok má kauzálny, pozičný a kompresný dizajn s odvodenou zložitosťou ≤ O(N log N) a definovaným dekódovacím stavom. Stav k 2026-06-10: kritériá splnené na papieri — formálne prejdenie gate až po schválení `spec.md` Danielom. Inak redesign, nie kód.
- **Publikácia stupeň 1 (po G0, D-log B2):** Zenodo DOI technical note „SSRA: design & complexity analysis" (spec + complexity + novelty téza + prior-art vymedzenie). Fixuje prioritu myšlienky pred behom experimentov. Pred uploadom: rýchly novelty sken arXiv.

## M1 — Implementácia (Claude Code, toto repo; ~1–2 týždne)
- SSRA v2: **level-wise batchovaná** implementácia (žiadna Python rekurzia) podľa `spec.md`.
- Pool_φ: P1 (default) + P2 (control) + P3 (challenger) za jednotným rozhraním; hybrid P1×P3 pripravený ako konfigurácia.
- Baselines v jednotnom harness-e: (a) flat Transformer compute-matched, (b) Log-Linear GatedDeltaNet z verejnej implementácie, (c) MEGABYTE-style 2-level.
- Testy pred tréningom: **(i) shift test** kauzality (zmena tokenu t nesmie ovplyvniť logity < t), **(ii) completion test** (logity z inkrementálneho dekódovania ≡ logity z plného forwardu; overuje Fenwick logiku aj kauzalitu naraz), (iii) gradient flow check, (iv) throughput/VRAM krivky vs. N.
- Smoke runy lokálne (MacBook M1 16 GB): ≤10M parametrov, char-level, malý korpus — **iba funkčnosť, žiadne závery**.
- **Gate G1a:** všetky testy zelené; SSRA throughput krivka rastie sub-kvadraticky (empirické potvrdenie D3/D6).
- **Micro-gate G1b-D3 (stabilizačný screen P3):** P3 musí na 10M smoke dosiahnuť loss v pásme X % od P1 bez divergencie po štandardnej stabilizácii (load-balance loss, temperature anneal); inak P3 → appendix a hybrid P1×P3 preberá challenger slot. **Hodnotu X stanoví Daniel pred M1.**

## M2 — Tréningové runy (Vertex AI; [ODHAD] 2–4 týždne)
- Škály: S1 ≈ 25M parametrov (ladenie), S2 ≈ 85M (hlavné porovnanie). Dáta: FineWeb-Edu alebo SlimPajama vzorka; orientačne ~20× parametrov tokenov (Chinchilla heuristika). Tréning ctx 1k–8k.
- HW: 1× A100 (40/80 GB) alebo L4 podľa pomeru cena/VRAM; spot/preemptible + checkpointy do GCS. **Ceny Vertex GPU overiť v deň objednávky (Pravidlo W) — neuvádzať z pamäte.** Rozpočtový strop stanoví Daniel; plán sa škáluje pod strop (S2 možno zmenšiť, nie vynechať baseline).
- Disciplína runov: 1 run = 1 YAML config v `experiments/` commitnutý PRED spustením + log artefakt v GCS + riadok v `results/runs.md`. Bez commitnutého configu run „neexistuje".
- Dropout sila (D5) sa ladí tu.
- **Gate G1:** SSRA trénuje stabilne (loss krivka bez divergencie) a ppl pri ctx 1k je v pásme ±5 % flat baseline pri matched compute. Fail → diagnostika, max 1 redesign iterácia; druhý fail = negatívny výsledok, prechod na report.

## M3 — Evaluácia ([ODHAD] 3–4 týždne; ablácie = ďalšie runy, najväčšia compute položka)
- ppl vs. dĺžka kontextu (1k → 32k, extrapolácia za tréningovú dĺžku), per-position loss.
- Needle-in-haystack / RULER-lite sada (syntetická, vlastný generátor v repe) **+ multi-needle generátor (P-B): k ∈ {1,2,4,8,16} needles, vzdialenosť ∈ {128..ctx}, metrika exact-match recall** — najostrejší diskriminátor Pool_φ operátorov.
- Throughput a pamäť vs. N (tréning aj autoregresívne dekódovanie; decode latency vrátane worst-case spike na pozíciách 2^k).
- Tréningový monitoring: P-C collapse diagnostika (entropia attention máp Q_φ, účasť queries).
- Ablácie (osi z M0): (a) zdieľanie váh naprieč úrovňami on/off [os A — jadro H1], (b) Pool_φ ∈ {P1, P2, P3, hybrid} [D3], (c) m: 16 fixné vs m₀+g·ℓ [D3], (d) w ∈ {32, 64, 128} [D4], (e) read-out: zdieľané θ vs oddelené ψ [D4], (f) level embedding on/off [D2], (g) k=2 vs k=4 [Q2], (h) hĺbka/arita podľa (g). Po G2 (nie v hlavnom behu): zdieľanie θ naprieč vrstvami [Q4], FFN v uzle.
- **Gate G2:** merateľná long-context výhoda (ppl tail alebo needle) pri ≥ rovnakej krátkej ppl ⇒ pozitívny paper. Inak ⇒ negatívny/analytický paper (kde a prečo to padá).

## M4 — Publikácia stupeň 2 (plný paper)
- arXiv cs.CL/cs.LG (endorsement overiť VOPRED — long-lead úloha T4, beží od júna) alebo Zenodo DOI okamžite + následne TMLR. Stupeň 1 (technical note po G0) sa v paperi cituje ako predchádzajúca verzia.
- Kód: Apache-2.0, signed commity (dodatočný timestamp; primárny dôkaz priority = DOI stupňa 1), reprodukčný skript jedným príkazom, flip repa private → public súčasne s publikáciou.
- Povinná „Limitations" sekcia: small-scale, jedna doména, žiadne škálovacie zákony, **H3/variant B netestovaný (follow-up SSRA-TD)**, wall-clock vs FLOP poctivosť (QKVO členy).
- Pred submisiou: novelty sken zopakovať (okno medzi skenom a uploadom < 1 týždeň).

## Follow-up vetvy (po G2, mimo PoC scope)
1. **SSRA-TD** (variant B, down-pass): nosič H3; väčšia novelty vzdialenosť od #2; retrieval bottleneck známy vopred (P-D).
2. **Os C** (obsahové hranice, H-Net referencia): nahradenie pozičného polenia učenou segmentáciou.
3. **UT × škála**: zdieľanie θ aj naprieč vrstvami (extrémny parameter-saver).

## Deľba práce
| Claude | Daniel |
|---|---|
| Formalizácia, complexity dôkazy | Dizajnové a scope rozhodnutia |
| Implementácia SSRA + baselines + harness (Claude Code, „Zadanie pre CC" pattern) | Spúšťanie a monitoring runov |
| Eval sady, analýza výsledkov | GCP budget a účty |
| Draft paperu, related work | Autorstvo, submisia, obhajoba |

## Riadenie projektu (minimálny framework)
- **Úlohy:** GitHub Issues, labels `design/impl/eval/paper`, milestones M0–M4. Issue = vec na >1 sedenie.
- **Rozhodnutia:** D-log v `00` — nič nie je rozhodnuté, kým tam nie je.
- **Kontinuita:** HO séria v `docs/handover/` pri každom prepnutí chatu/fázy; veto-based režim.
- **Kadencia:** piatok 30 min — stav vs gate, D-log update, potreba HO.
- **Kapacitný flag:** SSRA M1–M2 beží popri sopovai-agents, sai-sentry a klientskej práci; odhady platia pri ~2 sústredených blokoch týždenne, inak sa posúvajú dátumy, nie scope.

## Stop-loss
Ak M0–M2 prekročí 2× plánovaný čas alebo budget strop, projekt sa zmrazí a vydá sa technický report o aktuálnom stave (aj to je defensive publication; stupeň 1 DOI už vtedy existuje). Sunk-cost pokračovanie je zakázané týmto dokumentom.
