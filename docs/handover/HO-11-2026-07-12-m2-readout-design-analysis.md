# HO-11 — Design analýza read-outu uzavretá; zadanie odovzdané CC; ďalej = review CC reportu

**Dátum:** 2026-07-12 · **Autor:** Claude (Opus 4.8) + Daniel (verdikty) · **Predchádzajúci:** HO-10
**Vstupný bod nového chatu:** tento dokument + `docs/00-stav-a-triaz.md` (D-log 2026-07-12) + `results/M2-readout-optimization.md` (vznikne z CC) + `docs/cc/M2-readout-optimization.md`

---

## 1. Stav jednou vetou

Krok 1 rozhodnutia (a) z D-logu 2026-07-12 je hotový: design analýza aritmeticky lokalizovala root cause OOM/throughput problému (per-token materializácia Fenwick K/V kópií, `model.py:113–114`), vyriešila červenú vlajku z HO-10 (implementácia nemá exces oproti teórii — vlajka bola moja aritmetická chyba), našla štrukturálny kľúč k fixu (súvislé konzumné intervaly ⇒ read-out bez gatheru) a vyprodukovala **„Zadanie pre CC: M2 read-out optimization"** (`docs/cc/M2-readout-optimization.md` v1, commit **`b036288`** spolu s D-log riadkom); **chat, ktorý číta tento dokument, robí oversight review CC reportu (`results/M2-readout-optimization.md`) a pripravuje go/no-go + launch mini-zadanie pre re-kalibráciu.**

## 2. Čo sa v tomto sedení stalo

1. Načítané z repa: HO-10, `docs/00`, spec §8/§9/§10, `results/M2-calibration.md`, `model.py`, `fenwick.py`, `attention.py`, kalibračné YAMLy (S1: d=384/h=6/L=10; S2: d=640/h=10/L=15; obe ctx 1024, m=16, w=64, tied 16k).
2. Root cause analýza + aritmetická verifikácia proti nameranej zlyhanej alokácii (§3 nižšie).
3. Chyba z HO-10 §3 vlastnená a opravená: back-of-envelope použil stale d_h = 24 (M1 throughput config d=192/h=8) namiesto S1 d_h = 64 a zamiešal Fenwick *bloky* (≈ log₂N) s *riadkami* (m na blok) — po oprave meraný cover = 95 = presná teória.
4. Zadanie napísané a commitnuté (`b036288`); D-log riadok aplikovaný (Filesystem MCP, dryRun → apply, súčasť toho istého commitu).
5. Runbook Fáz A–D odovzdaný Danielovi; spustenie CC = krok A6 runbooku (nasleduje po tomto HO).

## 3. Kľúčové technické nálezy [záväzný kontext — nere-derivovať]

| nález | hodnota | status |
|---|---|---|
| k_max @ (N=1024, w=64, m=16) | **95** riadkov (1+2+4+8 z úrovní 0–3 + max 5 blokov × 16) | [OVERENÉ enumeráciou `fenwick_blocks`] |
| S1 b32 gather tenzor (B=32, h=6, d_h=64, bf16) | 32·6·1024·95·64·2 B = **2.226 GiB** = presne nameraná zlyhaná alokácia | [OVERENÉ aritmeticky vs meranie] |
| S2 b16 len gathery | 2 × 1.855 GiB/vrstva × 15 = **55.7 GiB** ⇒ štrukturálny OOM | [OVERENÉ aritmeticky] |
| padding waste | priemer ≈ 50 riadkov/token vs pad 95 ⇒ ≈ 1.9× | [ODHAD] |
| window `unfold` + einsum | pravdepodobne contiguous kópie `[B,h,N,d_h,w+1]` ≈ ďalších 38 GiB @ S2 b16 | [HYPOTÉZA — CC overí profilom, deliverable D2] |
| spec §10 pamäťové účtovníctvo („score matrices N·(w+m log N)") | platí; implementácia bola mimo neho, nie naopak — `01` §7 claims netreba meniť | [OVERENÉ] |

**Štrukturálna lema pre fix [OVERENÉ — dôsledok spec §9 v1.1 derivácie, completion-testom certifikovanej v M1]:** uzol (ℓ, j) s koncom e = j·2^ℓ je vo `fenwick_blocks(p)` ⟺ j nepárne ∧ p ∈ [e, e+2^ℓ−1]; ≤ 1 blok na úroveň. Tokeny konzumujúce uzol tvoria **súvislý beh t ∈ [j·2^ℓ+w+1, (j+1)·2^ℓ+w]** ⇒ read-out je vyjadriteľný ako per-level bloková cross-attention (posun o w+1, grupy 2^ℓ, nepárne grupy → bmm na views): nulový gather, nulový padding, identické logity, jeden softmax ostáva.

## 4. Zadanie v skratke (`docs/cc/M2-readout-optimization.md` v1)

- **Povinné:** R1 grouped restructure + R4 window path bez per-token K/V kópií; **SDPA v read-oute zakázané** (vlastný softmax láme §8 one-softmax nad heterogénnymi kľúčmi). R2 checkpointing = podmienený knob; R3 chunked gather = fallback len po explicitnom STOP; R5 SDPA v `node_attn` = voliteľné, timebox.
- `summary_pos: virtual` contingency musí ostať funkčná: gathered fallback branch, alebo constant-phase identita (relatívna RoPE fáza = konštanta w+1) s vlastným unit testom.
- **AP-20 (nové, veto prešlo):** každý perf rework spec-governed výpočtu = frozen-reference A/B test + analytický pamäťový model + §14 nezmenené zelené.
- Deliverables D1–D9 vrátane: D2 profil starého path (overí window hypotézu), D5 analytická peak-memory tabuľka pre všetky kalibračné configy (go/no-go vstup pred míňaním EUR), D6 CPU/MPS microbench; bundlované D7 (cu126 komentáre v requirements-gpu.txt + Dockerfile) a D8 (ledger delta $3.4786 vs $4.6 odhad — anotácia bez backfillu).
- **Akceptácia:** §14 zelené bez úprav existujúcich testov · A/B atol 1e−5 fp32 · žiadna `[·,·,N,k_max,d_h]` materializácia v defaultnom tréningovom pathe · analytický S2 b16 peak ≤ ~60 GiB · CPU no-regression.
- **Stop-loss (D-log):** táto 1 implementačná iterácia + 1 re-kalibrácia. Merateľný cieľ: S2 beží na 80 GB @ b16 ∧ projekcia S2 @ 850M tokenov ≤ 25 EUR (≈ ≥ 11.5k tok/s @ $1.39/hr, ECB 1.1430; plný 1.7B ⇒ ≈ 10×). Nesplnené → fallback (b) scale-down na S1-triedu / (c) reportovanie konštanty; konštanta sa v paperi reportuje vždy.

## 5. Protokol pre chat, ktorý toto číta (review CC reportu)

1. Prečítať `results/M2-readout-optimization.md` + `git log` CC commitov.
2. Gate check kritérií §6 zadania (1–6); **nezávisle prepočítať D5 budget** dosadením do closed-form vzorca (S2 b16 ≤ ~60 GiB) a spot-checkom A/B test coverage (ragged N, level_emb on/off, readout_params shared/separate, summary_pos módy).
3. Overiť hygienu: `git diff` na `tests/` (existujúce súbory nezmenené), žiadne spec/§13 zmeny, deviácie všetky explicitné, R3/R2 použitie zdôvodnené.
4. Návrh D-log zápisu + odporúčanie go/no-go pre re-kalibráciu (verdikt Daniel).
5. Pri go: krátke launch mini-zadanie pre CC — reuse `docs/cc/M2-runpod-launch.md` checklist; pre-flight = AP-19 **obe** konzolové ceny v deň deployu (minule „not captured", teraz povinné) + kredit (účet zdieľaný s cudzím projektom) + HW rebrík A100 PCIe → SXM → H100 PCIe; run set: S1 b16 (regres vs 9 457 tok/s / 54.7 GiB), S1 b32, **S2 b16 = brána**, pri fite S2 b32/b64; terminate (nie stop) ihneď po signále (AP-18).

## 6. Otvorené položky

| # | položka | kedy |
|---|---|---|
| 1 | CC exekúcia zadania → `results/M2-readout-optimization.md` | beží po odovzdaní (Fáza B) |
| 2 | Oversight review reportu + go/no-go re-kalibrácia | nový chat, hneď po CC signále |
| 3 | Re-kalibrácia (~1–2 EUR) → verdikt vs merateľný cieľ | po go |
| 4 | AP-19 Community cena — povinný pre-flight ďalšieho launchu | pri D3 |
| 5 | transformers pin kompatibilný s fla 0.5.0 | pred Phase 4 |
| 6 | ⚠ Revert záväzok: zmazať SA kľúč + vrátiť org-policy enforcement | po M2/M3 |
| 7 | Zenodo erratum/v1.1 (retenčné pravidlo + read-out konštanta nález) | po zozbieraní M2 nálezov |
| 8 | `p1_attn_entropy` monitoring | Phase 2+ tréningy |
| 9 | Project files vymeniť (00 @ `b036288`, zadanie, tento HO; po Fáze C + report) | priebežne po commitoch |

## 7. Čoho sa nedotýkať

Uzavreté rozhodnutia (D1–D6, Q1–Q5, MD-1…MD-13, AP-1…AP-20) sa neotvárajú. Spec v1.2 je normatívna — read-out fix je čisto implementačný (§8 realization freedom), sémantiku rozhodujú **nezmenené** §14 testy + frozen-reference A/B; akákoľvek normatívna zmena = nové rozhodnutie pre Daniela. Existujúce testy sa nemodifikujú. Žiadne kvalitatívne závery z lossov (spec §16). G1 kritérium (±5 % val ppl) sa nemení bez D-logu. Flat baseline sa nikdy nedropuje. Phase 2 sa nespúšťa pred verdiktom re-kalibrácie.
