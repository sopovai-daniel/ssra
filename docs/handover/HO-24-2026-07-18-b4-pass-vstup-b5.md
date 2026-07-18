# HO-24 — 2026-07-18 — B4 reviewer pass PASS, vstup do B5 (publikácia 19. 7.)

**Stav:** B4 reviewer-pass gate = **PASS** (verdikt autora, potvrdený commitom C4 `c711c30`). Paper `paper/results-paper.md` vo verzii **v0.5**, pripravený na B5 (publikačný deň, nedeľa 19. 7.).
**Nadväzuje na:** HO-23 (B3 hotové, vstup B4); HO-20 §2–§4 ostáva záväzným rámcom publikácie.

---

## 1. Čo sa stalo v B4 (sobota 18. 7. večer)

Kompletná krížová kontrola všetkých čísel v paperi proti zdrojovým artefaktom (checklist HO-23 §2, body 1–9):

- **T1:** 12/12 buniek proti raw `results/g2lite/m2-g2lite-{flat,ssra}-m1.json`; overené aj 3-dp zaokrúhlenie 224.3745 → 224.375.
- **T2:** proti `results/M2-g2lite.md` §M2 + oversight recount 720 trialov (D-log 2026-07-17); pooled 60 % @1,024 = (0+19+17)/60 potvrdené aj z raw m2 JSON-ov.
- **§4.1–§4.6:** proti M1-report, M2-sweep, M2-core-pair, M2-spike-diagnostics, M2-core-pair-lr6e4, M2-calibration, M2-recalibration.
- **Parametrové počty** 84 301 440 / 84 647 040: aritmeticky odvodené z kódu (`baselines/flat.py`, `src/ssra/model.py`) do posledného parametra; delta 345 600 = e_ℓ + node LN + φ presne.
- **§3.4 protokol:** overený priamo zo `scripts/train.py` (AdamW β = 0.9/0.95, wd 0.01, clip 1.0, lineárny warmup 778 + kosínus s floor 0.1×, val_every 200 / 8 batchov / fixný seed, AP-24 margin 2.0 / counter 6, NaN okamžitý abort).
- **§5.5 kvalitatívne charakterizácie:** overené z raw m2 JSON-ov (flat >1k: template loopy bez číslic, napr. „the sun is the sun is"; SSRA @1k: fluent template bez číslic; SSRA >1k: non-template token loopy).
- **§2 vs nota:** H1/H2 doslovne zhodné; erratum §2.8 cituje notu presne; bound 2+⌈w/2^ℓ⌉ a O((m·log N + w·log m)·d) nezávisle odvodené a sedia.
- **Slovník** (spec §16 + docs/00): čistý. **Binding formulácie** (HO-20 §4): splnené po fixoch nižšie.
- **Ref [29]:** doplnená akademická citácia — Penedo, G., Kydlíček, H., Ben Allal, L., et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* NeurIPS 2024 Datasets and Benchmarks Track, arXiv:2406.17557 — overená z arXiv abs stránky (načítané 2026-07-18); abstrakt explicitne zavádza FineWeb-Edu. TODO(B4) zmazané; retrieval-dates riadok doplnený.
- **[src] tagy:** 85 odstránených z tela (Abstract → References) po rezolúcii každého; §0 INTERNAL + INTERNAL inventár ostávajú do B5 (mažú sa pri exporte).

## 2. Nálezy a fixy (aplikované v C4)

| Fix | Miesto | Podstata |
|---|---|---|
| **F-2** (vecný, jediný) | §4.3 | „≈ the count expected by chance" → presné „against ≈ 1 expected under normality — two of them on a heavy-tailed run-length statistic whose z-scores overstate significance" (zdroj: spike-diag §3.2–3.3: ~1 očakávaný vs 4 pozorované; flagy 6 504 a 6 532 na longest_run) |
| F-1 | §5.1 | „4th–6th decimal" → „≈ 1.3e-3 over 30 steps" (raw pár 3.4657279 → 3.4644055) |
| F-3 | §8 + §1 contrib 2 | needle-lite 0 % veta doplnená o binding §4.8 caveat na oboch miestach |
| F-4 | §3.1 | „matched scale" ukotvené na „(matching protocol in §3.2)" |
| F-5 | §8 (2×) | „matched budget" → „matched parameter-and-token budget (§3.2)"; „larger matched pair" → „under the §3.2 protocol" |

Zvážené a ponechané bez zmeny: T1 bound „≤ 321 vs 225" (oba pravdivé horné odhady, verbatim zdroj g2lite §P.2); single-seed nie je v Abstracte (kostra uzavretá v B3, §6 kryje prominentne); [27] v arXiv-only formáte (konzervatívne per overený záznam).

**Vedľajší nález:** M1-report §(vi-c) sám obsahuje nepresný label „4th–6th decimal" hneď vedľa raw páru, ktorý je 1.3e-3 (3.–4. desatinné) — nepresnosť vznikla už v internom reporte, paper ju zdedil, F-1 ju odstránil. Report sa nemení (interný artefakt, append-only kultúra); poznámka patrí do súhrnného M4 D-log zápisu.

## 3. Poznámka: kompresia kontextu počas B4

Chat prešiel kompresiou kontextu počas overovacej fázy. Vykonaný audit nad zachovaným pred-kompresným transkriptom (cielené grep-y): čísla fixov F-1/F-2, T1 kotva a parametrové počty re-ukotvené na doslovné pred-kompresné čítania zdrojov — **žiadna odchýlka**. Zápisy do súboru boli od kompresie nezávislé: anchor stringy z čerstvého čítania celého papera po kompresii, dry-run diff pred zápisom, apply diff ako úplný dôkaz zmien.

## 4. Stav repa a účtovníctvo

- **C4 = `c711c30`** (main): paper v0.5; 43 insertions(+), 48 deletions(−) — konzistentné s aplikovaným diffom (43 modifikovaných odsekov + 5 čisto zmazaných tag-riadkov pod tabuľkami).
- Zostávajúce TODO v paperi: **DOI** (hlavička) a **git tag** (§3.6) — obe sú kroky B5.
- D-log/ledger: B bloky sa nezapisujú jednotlivo; **jeden súhrnný M4 zápis pri B5**. Spend B0–B4 = **0 EUR**.

## 5. B5 checklist (nedeľa 19. 7. — publikačný deň)

Poradie záväzné; každý krok potvrdiť pred ďalším:

1. **Zenodo draft + rezervácia DOI** — NOVÝ záznam s vlastným DOI (nie nová verzia stage-1 záznamu). Metadáta: titul papera, autor + ORCID 0009-0004-8584-5156, licencia CC BY 4.0, related identifier na DOI noty 10.5281/zenodo.20647034. Presný postup a typy vzťahov overiť podľa aktuálnych Zenodo docs v B5 chate (Pravidlo W) — nie z pamäte.
2. **DOI do hlavičky** papera (nahradiť TODO); Status bump.
3. **Export:** zmazať §0 INTERNAL + INTERNAL inventár, captions presunúť na finálne miesta (F2/F3 dvojpanel; párovanie F4b/c a F4d/e), PDF export, vizuálna kontrola PDF autorom.
4. **Repo príprava na flip:** LICENSE (Apache-2.0 kód; CC BY 4.0 text papera — overiť prítomnosť/formuláciu), secrets sweep (git log + working tree), čistý `git status`, README.
5. **Zenodo upload + Publish**; overiť, že DOI resolvuje.
6. **Repo public flip + git tag** (návrh: `paper-v1.0`); tag doplniť do §3.6 (nahradiť TODO(B5)) + finálny commit.
7. **Note ↔ paper linky:** na Zenodo zázname noty doplniť odkaz na paper (related identifiers).
8. **Uzávierka:** súhrnný M4 D-log zápis (B0–B5, B4 PASS verdikt, kompresná poznámka, M1-report label poznámka), ledger riadok (0 EUR), záverečný HO ak treba.

**Stop pravidlo:** ak krok 1–5 zlyhá (Zenodo výpadok, PDF problém), publikácia sa posúva o deň — žiadne skratky cez checklist.

## 6. Governance pre B5 chat

- Veto režim platí; `edit_file` vždy dry-run najprv; anchor stringy z čerstvého čítania.
- Daniel commituje v termináli (Claude pripraví príkazy; signed `-S`, EN messages).
- Žiadne nové merania, žiadny spend.
- Otvárací kontext nového chatu: **tento HO-24 + `docs/00-stav-a-triaz.md` (tail)**.
