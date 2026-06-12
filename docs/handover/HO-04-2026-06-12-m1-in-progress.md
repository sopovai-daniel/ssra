# HO-04 — Handover: M1 v plnom behu (jadro + testy hotové, G1a namerané, smoke chain spustený)
**Projekt:** DFKS / SSRA (Scale-Shared Recursive Attention)
**Dátum:** 2026-06-12 | **Autor HO:** Claude | **Schvaľuje:** Daniel Sopov
**Účel:** vstupný kontext pre nasledujúci čerstvý chat (interpretácia M1 reportu + gate rozhodnutia). Nadväzuje na HO-03 (neprepisuje sa).

---

## 1. Stav v jednej vete
M1 implementácia cez Claude Code je z väčšiny hotová: SSRA v2 jadro + korektnostné testy 1–3, 7 zelené (CPU fp32, 30 testov pass), G1a benchmark nameraný (SSRA log-log slope **0.983** vs flat **1.923** na čistom rozsahu 1k–8k — kritérium ≤ 1.5 a < flat splnené na číslach; formálny gate verdikt čaká na Daniela pri reporte), nočná reťaz piatich smoke runov beží; zostáva G1b-D3 pár, P-C diagnostika a `results/M1-report.md`.

## 2. Čo sa stalo v tomto chate (Chat 5, 2026-06-11 → 06-12)
- **Zadanie pre CC vytvorené a commitnuté:** `docs/cc/M1-assignment.md` v1 + D-log zápis (commit 02ed82b). Protokol AP-1…AP-7 (CPU fp32 rozhoduje korektnosť; G1b-D3 = final val loss na matched páre, seed pravidlo pri 4–6 %; AP-5 baseline (b) CUDA contingency; AP-6 swap-aware throughput config; AP-7 k=4 stub do M3).
- **Post-hoc korekcia publikácie:** finálna hlavička note v1.0 (ORCID, DOI, status) nebola v publikačnom commite — zdrojový .md zosynchronizovaný (commit 8aebc4f); publikovaný PDF nedotknutý (md5 overené pri publikácii); tag `note-v1.0` existuje a nechal sa na pôvodnom commite (rozpor zdokumentovaný v D-logu).
- **Korekcia retenčného pravidla dekódera — nález CC, nezávisle overený odvodením (Claude), certifikovaný completion testom:** bodové pravidlo Frontier ∪ Fenwick má re-entry gap pre uzly 2^ℓ ≤ w; oprava = intervalový obal t ∈ [end(u), end(u)+2^ℓ+w]. **Spec → v1.1** (§9/§10/§18-MD-2, commit 08f8939), D-log zápis 2026-06-12. Trieda decode stavu O(m·d·log N) drží (w-člen v N konštantný); explicitná konštanta ~1,5×.
- **Publikovaná nota = kandidát na erratum** (§2.6 + §3 uvádzajú bodové pravidlo a konštantu 2; headline trieda platí). Rozhodnutie odložené: zbierať M1 nálezy (vrátane NoPE hypotézy) → jedna Zenodo verzia v1.1 alebo erratum v stage-2. 45-dňové okno výmeny súborov (~2026-07-26) na vecné zmeny nepoužívať.
- **CC priebeh M1:** jadro `src/ssra/` + testy + flat baseline hotové a commitnuté (commit „Implement SSRA v2 core per spec v1.0 + verification tests 1-3, 7"). AP-5: `fla` import OK, jadro vyžaduje Triton (na macOS nie je) → „integration done, execution deferred to M2 GPU"; **NVlabs GatedDeltaNet zamietnutý pre NVIDIA non-commercial licenciu** (Apache-2.0 nekompatibilná). Benchmark: dva pokazené behy poctivo zdokumentované (swap-bound 8k bod; cross-kontaminácia pamäťových vzoriek cez allocator cache) → finálny beh d=192/h=8/L=2: **SSRA 0.983 / flat 1.923**, pamäť pri 8k 7.95 vs 11.4 GiB; 16k: SSRA swap-bound (vylúčené), flat OOM. Artefakty: `results/M1-throughput.{json,png}`, logy v `logs/` (commit „Test 4: throughput/VRAM benchmark…").
- **Smoke chain spustený:** 5 runov za sebou (p1, p3, p2, flat, megabyte) v jednom background príkaze, log `logs/M1-smoke-chain.log`; pri zlyhaní runu reťaz pokračuje ďalším. Configy v `experiments/M1-smoke-*.yaml` (run disciplína: commit pred spustením).

## 3. Kľúčové platné fakty pre nový chat
- `docs/spec.md` **v1.1** = jediný zdroj pravdy pre implementáciu; D-log v `docs/00` aktuálny k 2026-06-12. Uzavreté: D1–D6, Q1–Q5, MD-1…MD-10 (MD-2 v korigovanej forme).
- Gate verdikty **G1a a G1b-D3 robí Daniel** pri reporte; CC dodáva čísla + odporúčanie. G1b-D3: X = 5 % relatívne na final val loss, matched pár P1/P3, pri medzere 4–6 % +2 seedy (AP-3).
- Smoke runy = len funkčnosť + G1b-D3 screen; žiadne závery o kvalite (anti-goal).
- Sledovať v smoke: NoPE summary-key správanie (spec §6, [HYPOTÉZA]); `summary_pos: virtual` sa nesmie zapnúť potichu.
- Stupeň 1 publikovaný: DOI 10.5281/zenodo.20647034; repo private do stage-2.

## 4. Next steps pre nový chat (v poradí)
1. Prečítať z repa: `results/M1-report.md`, `results/runs.md`, `logs/M1-smoke-chain.log`, `results/M1-throughput.json` (Filesystem MCP).
2. Interpretovať voči predikciám P-A…P-C a pripraviť podklad pre Danielove gate rozhodnutia **G1a** a **G1b-D3** → D-log zápisy.
3. Rozhodnúť erratum stratégiu noty podľa súhrnu M1 nálezov (jedna Zenodo v1.1 vs erratum v stage-2).
4. Ak gates prejdú: príprava M2 (Daniel stanoví budget strop Vertex AI; ceny GPU overiť v deň objednávky — Pravidlo W).

## 5. Vstupy na Danielovej strane
- Gate rozhodnutia G1a + G1b-D3 (pri reporte).
- Pred M2: budget strop Vertex AI.
- Voliteľné: obnoviť GitHub MCP token v Claude nastaveniach (hlási bad credentials); arXiv endorsement až pre M4.

## 6. Inštrukcie pre nový chat
1. Project files: `00` (po 2026-06-12), `spec.md` v1.1, `01`–`03`, HO-04, `M1-assignment.md`; pri pochybnosti over v repe, repo vyhráva.
2. Začni krokom §4.1 (čítanie reportu). Uzavreté rozhodnutia neotváraj.
3. Veto režim, epistemická disciplína a Pravidlo W podľa systémových inštrukcií; slovenčina ASCII v chate, angličtina v kóde/CC.

## 7. Mapa artefaktov (delta voči HO-03)
| artefakt | umiestnenie | stav |
|---|---|---|
| Zadanie pre CC (M1) | `docs/cc/M1-assignment.md` | v1, commit 02ed82b |
| spec | `docs/spec.md` | **v1.1** (retenčná korekcia), commit 08f8939 |
| nota — zdroj zosynchronizovaný | `paper/technical-note.md` | commit 8aebc4f; erratum odložené |
| SSRA jadro + testy | `src/ssra/`, `tests/` | hotové, 30 testov pass (CPU fp32) |
| G1a benchmark | `results/M1-throughput.{json,png}`, `logs/M1-throughput*.log` | hotové; 0.983 vs 1.923 |
| smoke chain | `logs/M1-smoke-chain.log`, `experiments/M1-smoke-*.yaml` | beží (5 runov) |
| M1 report | `results/M1-report.md` | vznikne po dobehnutí chainu |
