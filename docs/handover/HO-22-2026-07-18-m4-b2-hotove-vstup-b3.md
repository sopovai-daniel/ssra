# HO-22 — 2026-07-18 — M4 blok B2 hotový; vstup pre B3 (Abstract/§1/§8/disclosure/captions)

**Účel:** kontinuita do B3 drafting chatu. Autoritatívny stav: `docs/00-stav-a-triaz.md` (D-log 2026-07-17, GO záznam = **piaty** zápis dňa; HO-20 Addendum A1) + HO-20 §2/§4 ako záväzný rámec publikácie. Tento HO nič nerozhoduje — sumarizuje a odovzdáva.

## §1 Stav

- **B0** (`4627638`) + **B1** (`40f0a0c`) hotové — pozri HO-21.
- **B2 hotové** (commit **`8fc7f15`**, 164+/32−): `paper/results-paper.md` → **v0.3** —
  - **§4.1–4.8** plný draft: G1a shape tabuľka + slopes 0,983/1,923; sweep tabuľka 8 buniek + within-model selekcie; Phase 3 (spike 6 475→6 500, druhá epizóda per §xi C1, T5 data due diligence, „log-only evidence“ priznaná); Phase 3b (+10,22 %, stabilita v oversight formulácii „final val = running best“ + 0,00279/0,00334 per rameno, grad_norm rozsahy); lr-stability observation (mechanism undetermined); §4.6 throughput/memory tabuľka **31,8× / 11,8× / 11,1×** + read-out restructuring odsek; §4.7 T1 + O2/O3 labels + binding formulácie (flat prior confirmed / SSRA prior violated; 21,35→19,06 bez interpretácie); §4.8 T2 + floor rule + pre-registrovaný caveat.
  - **§5.1–5.5** deskriptívne (~1,5 str., cap dodržaný): P-C uniformita naprieč všetkými runmi + sekvencia symptom-not-precursor; e_ℓ zero rows; V2b kvantizácia; bucket lokalita (O6 delty exaktne z raw); needle behaviorálne pozorovania (kvalitatívne, z per-trial JSONov).
  - **§6** Limitations v próze, 6 odsekov — vrátane mandátu HO-21 §2 (baseliny (b)/(c) + ablácie nebežali, lebo padla brána) a navyše „Instability evidence depth“ (log-only evidencia Phase 3).
  - INTERNAL inventár figúr aktualizovaný: **všetkých 10 referencovaných plotov existuje** (listing `results/` 2026-07-18) — žiadna medzera, eskalácia nebola potrebná; sweep/G1b krivky uvedené ako voliteľné.
- **Overenia vykonané v B2:** T1 postavená z raw `results/g2lite/m2-g2lite-{flat,ssra}-m1.json` (**12/12 buniek == report §M1**); O6 delty (+0,99…+4,11 / +2,87…+6,83) prepočítané exaktne z raw bucketov; T2 = report §M2 + oversight recount 720 trialov (D-log 2026-07-17); každé číslo v §4–§6 nesie `[src:]` tag.
- **Spresnenia voči B1 kostre (odovzdané v B2 chate, bez veta):**
  1. **11,8× ≠ S2.** 11,8× je S1 b16 pomer (re-kalibrácia); produkčný S2 b16 pomer = 137 500/12 383 = **11,1×**. Paper reportuje obe presne; R3 invariant (11,8×) zachovaný. B1 kostra „11.8× at S2 b16“ bola nepresná — kostra nie je zdroj čísel.
  2. §4.4: presné 0,00279/0,00334 per rameno namiesto súhrnného 0,0033.
  3. §4.5: derived scoping fakt — sweep horizont ≈ 60M tok (3 662 krokov) končí pred onsetom nestability (step 6 475 ≈ 106M tok = 6 475 × 16 384); tagované ako derived.
- **Pending:** **B3** = Abstract, §1, §8, AI Assistance Disclosure, figure/table captions. Potom **B4** = nedeľný reviewer pass (gate; krížová kontrola každého čísla proti reportom, rozpustenie/strip [src] tagov, slovník/spec §16 scan, TODO(B4) voliteľná akademická citácia [29]); **B5** = publikácia (rezervovaný DOI, PDF export, Zenodo upload+Publish, repo LICENSE/sweep/public flip/tag, súhrnný M4 D-log zápis + ledger).
- D-log/ledger tejto session: žiadny zápis (B bloky sa nezapisujú jednotlivo, D-log návrh v HO-21 §1; spend 0 EUR).

## §2 Záväzné pre B3

- HO-20 §2 (zákazy: žiadne nové merania/spend, spec §16, zakázaný slovník, žiadna re-litigácia) + HO-20 §4 (mechanické formulácie) + drafting rules v paperi §0 (INTERNAL) platia v plnom rozsahu.
- **Abstract** per kostra TODO v paperi: mechanism recap + complexity class (same as Log-Linear Attention, **no better-class claim**); matched parameters + tokens (AP-8 honesty); negatívne výsledky otvorene — parity +10,22 % mimo ±5 %; nestabilita @ 1e-3 so stabilným retune @ 6e-4 (narrower empirical lr range, mechanism undetermined); flat prior confirmed / SSRA prior violated, no crossover; needle 0 % SSRA vo všetkých bunkách; konštanta 11,8× (S1) / 11,1× (S2). Framing: pre-registered falsification plan noty v1.0 vykonaný; oba smery deklarované ako publikovateľné.
- **§1** per kostra TODO (motivácia z noty §1 + [1], H1/H2 konzistentne s notou, dvojstupňová publikácia, contributions i–iv). **§8**: future work = návrhy, nie záväzky (stabilization/schedule ablácie, scale, SSRA-TD, os C, trajektóriová analýza na step-tagged ckptoch). **AI Assistance Disclosure** per HO-20 §2: experimenty exekuoval Claude Code na infraštruktúre prevádzkovanej a platenej autorom pod dohľadom; verifikačná suita (causality/equivalence/gradient-flow); všetky rozhodnutia, verdikty a review autor; COPE (AI nie autor, autor plná zodpovednosť).
- **Captions** pre F1–F4e + T1/T2: texty pripraviť do INTERNAL inventára (finálne umiestnenie pri PDF exporte); výhradne existujúce commitnuté artefakty, žiadne nové figúry.
- Čísla v Abstract/§1/§8 výhradne prebrané zo §4–§6 draftu (s [src] tagmi); nič nové sa nemeria.

## §3 Lekcie tejto session

1. **Kostra draftu nie je zdroj čísel.** Každé číslo sa pri draftovaní berie z reportu, nie z predchádzajúceho bloku (prípad „11,8× at S2“ v B1 kostre); skeletné bullety čítať ako mapu obsahu, hodnoty vždy re-verifikovať zo zdrojov.
2. Raw-JSON cross-check tabuliek (T1 12/12, O6 delty exaktne) je lacný a definitívny — držať ako štandard pre každú tabuľku v paperi aj pri B4 reviewer passe.

## §4 Otvárací prompt pre B3 chat (copy-paste)

```
Vstup: docs/handover/HO-22-2026-07-18-m4-b2-hotove-vstup-b3.md + docs/00
(D-log 2026-07-17, GO = piaty zapis) + paper/results-paper.md (v0.3,
commit 8fc7f15) — paper citat CELY z repa cez Filesystem MCP.
Kontext: HO-20 §2/§4 zavazny ramec; drafting rules v paperi §0 (INTERNAL);
ziadne nove merania/spend; publikacia Ne 19.7., reviewer-pass gate Ne
doobeda = podmienka; spec §16 a zakazany slovnik platia.
Uloha (B3): draft Abstract + §1 + §8 + AI Assistance Disclosure +
captions (F1-F4e, T1/T2) do paper/results-paper.md (edit_file, dry-run
pred apply); cisla vyhradne zo §4-§6 s [src] tagmi; Abstract per kostra
TODO (negativne vysledky otvorene, no better-class claim); §8 future
work = navrhy, nie zavazky; disclosure per HO-20 §2; potom commit bod C3
+ navrh B4 (nedelny reviewer pass checklist).
```
