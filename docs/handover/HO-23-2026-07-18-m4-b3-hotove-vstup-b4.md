# HO-23 — 2026-07-18 — M4 blok B3 hotový; vstup pre B4 (nedeľný reviewer pass = gate)

**Účel:** kontinuita do B4 review chatu. Autoritatívny stav: `docs/00-stav-a-triaz.md` (D-log 2026-07-17, GO záznam = **piaty** zápis dňa; HO-20 Addendum A1) + HO-20 §2/§4 ako záväzný rámec publikácie. Tento HO nič nerozhoduje — sumarizuje a odovzdáva.

## §1 Stav

- **B0** (`4627638`) + **B1** (`40f0a0c`) + **B2** (`8fc7f15`) hotové — pozri HO-21/HO-22.
- **B3 hotové** (commit **`23acf1f`**, 42+/13−): `paper/results-paper.md` → **v0.4** — plný text draftnutý, žiadny TODO(B3) nezostáva:
  - **Abstract** — jeden odsek so všetkými povinnými prvkami kostry: mechanism recap + trieda (same as Log-Linear Attention, „explicitly not a better one“); matched parameters + tokens s AP-8 honesty; +10,22 % mimo ±5 % (26,860 vs 24,369); nestabilita @ 1e-3 / čistý retune @ 6e-4, mechanism undetermined; „flat prior confirmed; SSRA prior violated“, no crossover; needle 0 % SSRA vo všetkých bunkách + pre-registrovaný caveat; 11,8× (S1) / 11,1× (S2) + poznámka, že equal-token protokol zvýhodnil SSRA na compute osi; either-outcome-publishable framing. Bez číselných citácií [n] (Zenodo plain-text abstract); [src] tagy ostávajú do B4.
  - **§1** — motivácia z noty §1, H1/H2 **doslovne** z noty [26] §1 (znenie overené čerstvým čítaním noty z repa, nie z pamäti), dvojstupňová publikácia, contributions 1–4, erratum pointer na §2.8.
  - **§8** — negatívny záver bez architektonických záverov (spec §16), „what survives“ odsek, future work ako návrhy s explicitnou vetou „none is scheduled, and each would require its own pre-registered plan“.
  - **AI Assistance Disclosure** — binding formulácie HO-20 §2 doslovne (CC exekúcia na infraštruktúre prevádzkovanej a platenej autorom pod dohľadom; causality/equivalence/gradient-flow suita; všetky rozhodnutia, verdikty a review autor) + odsek machine-checkable guards + COPE.
  - **Captions** T1/T2 + F1–F4e v INTERNAL inventári (finálne umiestnenie = B5 pri PDF exporte); výhradne existujúce commitnuté artefakty, žiadna nová figúra.
  - Hlavička Status v0.3 → v0.4; source mapa doplnená o riadok „Abstract/§1/§8/captions = no independent sources — recap of §2–§6 tagged values“.
- **Overenia v B3:** všetky čísla v nových sekciách sú recapy hodnôt už tagovaných v §2–§6 draftu; žiadne nové číslo, žiadne meranie, spend 0 EUR.
- **Spresnenia voči kostre (odovzdané v B3 chate, prijaté bez veta — potvrdené exekúciou commitu C3):**
  1. §8 future work má navyše bullet „registered, unexecuted comparisons and ablations z §6“ nad rámec HO-22 vymenovania — konzistentné so §6, ktorý ich už označuje ako otvorené.
  2. Abstract bez číselných citácií [n]; nota označená ako „the stage-1 record of this work“.
  3. Status v0.4 + nový source-map riadok (B4 pomôcka).
  4. Disclosure menuje Claude aj pre non-CC prácu (triáž, formalizácia, drafting, verifikačné passy) — širšie než binding minimum, konzistentné s v1.0 record-level transparentnosťou.
  5. Captions F2/F3 dvojpanelové (2 PNG = 1 figúra); F4b/c a F4d/e párované; kompozícia = B5.
- **B4 checklist prijatý** (návrh v B3 chate, bez veta) — plné znenie v §2 nižšie; B4 = **gate publikácie**.
- **Pending:** **B4** = reviewer pass (gate; plánovaný slot Ne doobeda, ~3–4 h — **akcelerácia na So večer 18.7. potvrdená Danielom po C3**: záväzný je PASS gate, nie kalendárny slot; termín „do 19.7.“ skorší beh kryje). **B5** = publikácia (rezervovaný DOI, PDF export, Zenodo upload+Publish, repo LICENSE/secrets sweep/public flip/tag, note↔paper linky, súhrnný M4 D-log zápis + ledger + záverečný HO) — pri PASS môže bežať hneď po B4, aj dnes; akcelerácia sa zachytí v súhrnnom M4 D-log zápise.
- D-log/ledger tejto session: žiadny zápis (B bloky sa nezapisujú jednotlivo; súhrnný M4 zápis pri B5 per HO-21 §1; spend 0 EUR).

## §2 Záväzné pre B4 (prijatý checklist)

Rámec: HO-20 §2 (zákazy) + §4 (mechanické formulácie) + drafting rules v paperi §0 (INTERNAL) + spec §16 + zakázaný slovník platia v plnom rozsahu. B4 = podmienka publikácie; pri FAIL posun o dni, nie týždne.

1. **Číselná krížová kontrola:** každé číslo v celom texte (vrátane Abstract/§1/§8/captions) proti zdrojovému reportu; T1 znovu proti raw `results/g2lite/m2-g2lite-{flat,ssra}-m1.json`, T2 proti `results/M2-g2lite.md` §M2 + recount štandard z oversight.
2. **[src] tagy:** resolve + strip. Po stripe nesmie existovať číslo bez dohľadateľného zdroja ani zabudnutý tag.
3. **Slovník + spec §16 scan:** zakázaný slovník (docs/00) cez celý text; žiadne architektonické závery, žiadne cross-model čítanie sweep lossov.
4. **Binding formulácie (HO-20 §4):** „flat prior confirmed; SSRA prior violated“ (nikde „both violated“); 21,35 → 19,06 vždy bez interpretácie; H2 caveat veta pri každom needle závere; AP-8 honesty pri každom výskyte „matched“.
5. **Reviewer pass obsahu:** „čo by oponent zhodil ako prvé“ — prejazd celého textu; osobitne Abstract (najčítanejší), §4.5 (lr-stability formulácia) a §6 (či Limitations kryjú všetky slabiny, ktoré by oponent našiel sám).
6. **Konzistencia:** krížové referencie §/F/T čísel, erratum §2.8 vs nota §2.6/§3, terminológia identická s notou, References formáty.
7. **TODO(B4) ref [29]:** voliteľná akademická citácia fineweb-edu — primárny zdroj overiť per Pravidlo W; ak neexistuje, ostáva dataset citácia a TODO sa maže.
8. **INTERNAL sekcie:** §0 + inventár označené na odstránenie/presun pri exporte; jediné zostávajúce TODO po B4 = DOI a git tag (oba B5).
9. **Výstup:** PASS/FAIL verdikt gate (Daniel). PASS → B5; FAIL → zoznam náprav + posun o dni.

## §3 Lekcie tejto session

1. **Wording-consistency úlohy = čítanie zdroja, nie parafráza z pamäti.** Verbatim H1/H2 a motivácia vyžadovali čerstvé čítanie noty §1 z repa; rovnaký princíp ako „kostra nie je zdroj čísel“ (HO-22 §3), aplikovaný na text.
2. **Tagging disciplína B1/B2 sa vyplatila:** B3 prebehol celý ako montáž recapov bez jediného nového zdroja — Abstract/§1/§8 sa dajú písať rýchlo a bezpečne práve preto, že §2–§6 nesú [src] tagy.

## §4 Otvárací prompt pre B4 chat (copy-paste)

```
Vstup: docs/handover/HO-23-2026-07-18-m4-b3-hotove-vstup-b4.md + docs/00
(D-log 2026-07-17, GO = piaty zapis) + paper/results-paper.md (v0.4,
commit 23acf1f) — paper citat CELY z repa cez Filesystem MCP; zdrojove
reporty results/*.md + raw JSONy citat z repa podla potreby krizovej
kontroly.
Kontext: HO-20 §2/§4 zavazny ramec; B4 = reviewer-pass GATE (podmienka
publikacie; deadline "do 19.7.", akceleracia na So vecer 18.7.
potvrdena Danielom — gate nezmeneny); ziadne nove merania/spend;
spec §16 a zakazany slovnik platia; pri FAIL posun o dni, nie tyzdne.
Uloha (B4): checklist HO-23 §2 body 1-9 v poradi — krizova kontrola
kazdeho cisla proti reportom (T1 z raw JSON), resolve+strip [src] tagov
(edit_file, dry-run pred apply), slovnik/spec §16 scan, binding
formulacie, obsahovy reviewer pass ("co by oponent zhodil ako prve"),
konzistencia referencii, TODO(B4) ref [29] per Pravidlo W, INTERNAL
check; vystup = PASS/FAIL verdikt (Daniel). Pri PASS commit bod C4 +
navrh B5 krokov (DOI rezervacia, PDF export, Zenodo, repo flip, D-log
+ ledger + HO); pri FAIL zoznam naprav.
```
