# HO-20 — 2026-07-17 — Oversight VERIFIED + GO: stage-2 paper na Zenodo do 19.7.

**Účel:** kontinuita do M4 drafting chatu. Autoritatívny stav: `docs/00-stav-a-triaz.md` (D-log 2026-07-17, **štvrtý** zápis). Tento HO nič nerozhoduje — sumarizuje rozhodnutia a odovzdáva.

## §1 Stav v jednej vete

Oversight review G2-lite: **VERIFIED, 0 vecných korekcií** (nezávislý prepočet M0/M1/M2 z raw JSON, plný recount 720 needle trialov, O1–O7 mechanicky, „zero model-code diffs" overené file listami commitov). **Rozhodnutia (Daniel, GO):** stage-2 vysledkový paper na **Zenodo do 19.7.** (nový záznam s vlastným DOI + repo public flip, Apache-2.0), R4 = variant B-lite, R5 erratum in-paper, R6 kickoff ihneď, R7 SA revert po publikácii; post-publikačná exploatácia dát → v1.1 = samostatné zadanie po 19.7. Výskumná časť je hotová; zostáva výhradne písanie, montáž a publikácia.

## §2 Rozhodnutia a ohraničenia (binding)

| položka | stav |
|---|---|
| Rozsah paperu | R3 invarianty (D-log 17.7., 2. zápis) + G2-lite; **variant B-lite**: negative-results report + kompaktná diagnostická sekcia (~1,5–2 str.) výhradne z existujúcich dát; krátky future-work odsek (SSRA-TD, os C, ablácie — návrhy, nie záväzky) |
| Kvalitatívna brána | **reviewer pass v nedeľu doobeda = podmienka publikácie** („čo by oponent zhodil ako prvé"); pri faile posun o dni, nie týždne — žiadny externý deadline neexistuje |
| Zenodo | **nový záznam s vlastným DOI** (cituje notu v1.0 ako prior version); NIE „new version" existujúceho záznamu; workflow overený v júni: rezervovaný DOI → hlavička → PDF export → upload → Publish |
| Repo | public flip + git tag v deň publikácie; predtým: LICENSE (Apache-2.0 kód / CC BY 4.0 texty), secrets sweep, čistý `git status`; zverejnením sa publikuje aj D-log + pre-registrované zadania = dôkaz metodológie |
| Erratum | in-paper (paper koriguje bodové retenčné pravidlo §2.6/§3 noty explicitnou erratum vetou); Zenodo verzia noty = voliteľné po publikácii |
| AI disclosure | stage-2 paper ponesie sekciu ekvivalentnú v1.0 disclosure, so spresnením: experimenty exekuoval Claude Code na infraštruktúre prevádzkovanej a platenej autorom pod dohľadom; všetky rozhodnutia, verdikty a review autor. COPE-konformné (D-log 4. zápis) |
| ✔ Zenodo v1.0 disclosure | **LIVE** — potvrdené screenshotom z privátneho okna 17.7. ≈ 23:03 (sekcia pred Abstractom, Version 1.0, PDF nedotknutý md5 589b51d0…). Claude fetch cache servíroval starú kópiu — falošný poplach, vlastníctvo Claude. **v1.0 PDF sa NEMENÍ** (anachronizmus „added on 9 July" vo file z 11.6.; md5 provenance; D-log 12.6. „nová verzia > výmena súborov"); disclosure vo file = voliteľná nota v1.1 „New version" po 19.7. (MD+PDF vygeneruje Claude zo zdroja `paper/technical-note.md`, Daniel len upload) |
| Zákazy | žiadne nové merania ani spend bez nového rozhodnutia (platí aj pre 0-EUR lokálne merania, HO-19 §4); žiadne architektonické závery (spec §16); zakázaný slovník platí; uzavreté D-logy sa nere-litigujú; G1 = FAIL (×2) sa neotvára |
| Standing | R1 ckpty držať do v1.1 rozhodnutia (deadline 31.8.); AP-19 pozastavené; R7 SA revert 20.–21.7. (kľúč + org-policy override; postup v D-logu 2026-07-12) |

## §3 Kostra paperu (schválená ako plán, [ODHAD] ~16–18 h + rezerva)

§1 Úvod + H1/H2 · §2 Mechanizmus (z noty + spec v1.2; korigované §9 pravidlo + erratum veta) · §3 Setup (AP-8 matched params+tokens honesty note FLOPs/wall, AP-9 provenance, budget/ledger) · §4 Výsledky: G1a slope 0.983, sweep, Phase 3 spike + druhá epizóda, Phase 3b parity gap **+10,22 %**, lr-stability nález (mechanizmus neurčený), konštanta **11,8×**, G2-lite M1 tabuľka + buckety + M2 needle · §5 Diagnostika (deskriptívna): P-C uniformita Q_φ ≈ ln(32) naprieč všetkými škálami/runmi, e_ℓ ablation-OFF, V2b kvantizácia + zasiahnutie per N, bucket-pozičná lokalita, needle behaviorálne pozorovania · §6 Limitations (jedna škála, single pair @ seed 1337 by design, packed short-doc korpus = degradation robustness nie long-range benefit, V2b, e_ℓ) · §7 Related work (`02` #1–#25) · §8 Záver + future work · AI Assistance Disclosure.

## §4 Interpretačný podklad (mechanické čítania; formulácie záväzné)

- **H1 na tejto škále nepodporená:** SSRA prior (stable/mild) porušený od N = 2 048 (r 4,11 → collapse od 4k → 561,8 @ 32k); flat prior (Press et al.) potvrdený (r 1,57 → 32,7); O4 žiadny crossover, pomer 1,111 → 21,4 (peak @ 16k) → 19,1. Namerané usporiadanie opačné voči H1 predikcii. Formulácia: **„flat prior confirmed; SSRA prior violated"** — nikdy „both violated".
- **H2 bez pozitívnej evidencie, v M3 tvare netestovaná:** SSRA 0 % vo všetkých 18 bunkách vrátane vlastnej tréningovej dĺžky; flat copy behavior existuje (pooled 60 % @ 1 024) ⇒ suita informatívna; za 1 024 needle konfundovaný pozičným kolapsom z M1; pre-registrovaná veta (85M @ 850M tok nemusí mať copy behavior) kryje výsledok.
- **Diagnostické pozorovania [OVERENÉ z raw]:** (i) poškodenie oboch modelov výhradne na pozíciách > 1 024, pozície ≤ 1 024 držia baseline pri každom N (flat 3,087–3,393; SSRA 3,220–3,410); per-position penalta @ 32k SSRA +2,87…+6,83 natu vs flat +0,99…+4,11; (ii) flat copy je depth-lokálna aj na trénovanej dĺžke: 0,00/0,95/0,85 @ hĺbkach 0,1/0,5/0,9; (iii) kvalitatívne: flat degeneruje do template-tvarovaných slučiek bez číslic, SSRA @ 1 024 plynulo pokračuje šablónu bez číslic pri každej hĺbke, SSRA > 1 024 degeneruje do ne-template tokenových slučiek; (iv) gap @ 1 024 na E = 11,07 % vs +10,22 % na G1 sete (cross-region konzistencia).
- Pomer 21,35 → 19,06 na vrchole gridu sa reportuje bez interpretácie.

## §5 Plán blokov do 19.7. [ODHAD]

Pi večer: outline + kostra `paper/` (1–2 h) → So doobeda: §2/§3/§7 (3–4 h) → So poobede: §4/§5/§6 (4–5 h) → So večer: §1/abstrakt/§8 + captiony (2 h) → **Ne doobeda: reviewer pass + krížová kontrola každého čísla proti reportom + slovník/spec §16 (3–4 h) = GATE** → Ne poobede: PDF, Zenodo upload+Publish, repo LICENSE/sweep/public/tag, note↔paper linky, D-log + ledger + HO-21 (2–3 h). Deľba: Claude píše a verifikuje; Daniel rozhoduje, recenzuje, konzolové kroky.

## §6 Otvárací prompt pre nový chat (copy-paste)

```
Vstup: docs/handover/HO-20-2026-07-17-go-publikacia-19-7.md + docs/00
(D-log 2026-07-17, stvrty zapis) + paper/technical-note.md + docs/spec.md
+ results/{M1-report,M2-sweep,M2-core-pair,M2-spike-diagnostics,
M2-core-pair-lr6e4,M2-g2lite}.md + 02-prior-art-mapa.md.
Kontext: oversight G2-lite VERIFIED 0 korekcii; GO (Daniel): stage-2
paper na Zenodo do 19.7., variant B-lite, erratum in-paper, novy zaznam
s vlastnym DOI, repo public flip v den publikacie; reviewer-pass gate
v nedelu doobeda = podmienka; ziadne nove merania/spend; spec §16 a
zakazany slovnik platia; AI disclosure sekcia povinna (HO-20 §2).
Uloha: (1) M4 outline + kostra paper/ podla HO-20 §3; (2) po schvaleni
sekcie §2/§3/§7 draft (montaz z noty v1.2 spec + `02`); (3) postupne
§4/§5/§6 z reportov (kazde cislo s odkazom na zdrojovy report); (4)
priebezne pripominat commit bodov. Necakat na nic ine — HO-20 §2 je
zavazny ramec.
```

## §7 Lekcie tejto session

1. Externé zmeny záznamov (Zenodo a pod.) zachytávať v D-logu v deň vykonania (edit z 9.7. bol zachytený až 17.7.); rozhodujúca verifikácia verejného UI = privátne okno používateľa (screenshot) — Claude web_fetch môže servírovať cache a opakovaný fetch tej istej URL v jednej session nie je nezávislé overenie.
2. Nezávislý recount per-trial dát (720 flagov) je lacný a definitívny — držať ako štandard oversight pre klasifikačné metriky.
3. Commit file-list check (GitHub MCP) je efektívny mechanický dôkaz „zero model-code diffs" — držať v oversight šablóne.
