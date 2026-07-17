# HO-21 — 2026-07-18 — M4 bloky B0+B1 hotové; vstup pre B2 (§4/§5/§6)

**Účel:** kontinuita do B2 drafting chatu. Autoritatívny stav: `docs/00-stav-a-triaz.md` (D-log 2026-07-17, GO záznam = **piaty** zápis dňa; pozri HO-20 Addendum A1) + HO-20 §2 ako záväzný rámec publikácie. Tento HO nič nerozhoduje — sumarizuje a odovzdáva.

## §1 Stav

- **B0 hotové** (commit `4627638`): `paper/results-paper.md` kostra v0.1 + HO-20 Addendum A1 (oprava odkazu: GO záznam je piaty, nie štvrtý zápis D-logu z 2026-07-17).
- **B1 hotové** (commit `40f0a0c`): paper v0.2 —
  - **§2 Mechanism**: kondenzovaná montáž noty v1.0 + spec v1.2 (2.1–2.7); **§2.8 = erratum R5** (bodové retenčné pravidlo noty označené ako nesprávne, intervalová oprava, headline trieda nedotknutá, súbor noty sa nemení).
  - **§3 Setup**: d=640/h=10/L=15; params 84 301 440 / 84 647 040 (+0,41 %); AdamW β(0,9, 0,95)/wd 0,01/clip 1,0 [overené z `scripts/train.py`]; warmup 778 + cosine floor 0,1×; eval 1 953/1 999 872/127; AP-8 honesty veta; **AP-24 timeline priznaná** (pravidlo zavedené až po Phase 3, aktívne v Phase 3b); spend 72,37 EUR uvedený.
  - **§7 Related work**: prozaické delimitácie #1–#25 (GPST/MANO/PSM explicitne; HKT len abstraktové tvrdenia); veta o nebežanom porovnaní s [2]; záver „the outcome (§4) is negative".
  - **References [1]–[29]**: [1]–[25] verbatim z noty; [26] self-cite noty; [27] Press et al., [28] Mohtashami & Jaggi — overené v primárnych zdrojoch (arXiv abs) v tejto session; [29] FineWeb-Edu dataset-card citácia + TODO(B4) na voliteľnú akademickú citáciu.
- **Pending:** Abstract, §1, §4, §5, §6, §8, text AI disclosure; figúry/tabuľky.
- **Bez veta platí:** názov súboru `paper/results-paper.md`; pracovný titul „SSRA: Scale-Shared Recursive Attention — Empirical Evaluation and Negative Results"; figúry = výhradne existujúce commitnuté artefakty (nové kompozitné figúry = v1.1 scope, D-log 2026-07-17).
- D-log: bloky B0–B4 sa nezapisujú jednotlivo; jeden súhrnný M4 záznam pri publikácii (B5) — návrh vo veto režime.

## §2 Záväzné pre B2

- HO-20 §2 (zákazy: žiadne nové merania/spend, spec §16, zakázaný slovník, žiadna re-litigácia) + HO-20 §4 (mechanické formulácie: „flat prior confirmed; SSRA prior violated", H2 caveat veta, pomer 21,35→19,06 bez interpretácie) platia v plnom rozsahu.
- Drafting rules sú zapísané priamo v paperi (§0 INTERNAL): **číslo bez inline `[src:]` tagu neexistuje**; tagy sa rozpúšťajú až pri nedeľnom reviewer passe.
- §4 = 4.1–4.8 podľa kostry; §5 = 5.1–5.5 s tvrdým capom ~1,5–2 strany; §6 Limitations podľa kostry **+ explicitne: pre-registrované M3 baseliny (b)/(c) a ablácie nebežali, lebo padla brána pred nimi**.
- Zdroj §4.6 (konštanta 11,8×): `results/M2-recalibration.md` + `results/M2-calibration.md` — **nie sú v project files, čítať z repa**.
- Figúry: overiť skutočné názvy plotov v `results/` (M1-throughput.png; loss ploty Phase 3 / Phase 3b; G2-lite ploty); T1 (ppl vs N) a T2 (needle grid) ako tabuľky z raw CSV/JSON. Akákoľvek medzera → rozhodnutie Daniel.

## §3 Lekcie tejto session

1. **GitHub code search neindexuje privátne repo `ssra`** (0 hitov aj na isto existujúce symboly, napr. „fenwick") — obsahové verifikácie kódu výhradne cez Filesystem MCP; commit file-list checky cez `get_commit` fungujú ďalej.
2. HO-20 odkaz „štvrtý zápis" bol nepresný — opravené append-only Addendom A1; počítanie zápisov D-logu overovať pred citovaním.

## §4 Otvárací prompt pre B2 chat (copy-paste)

```
Vstup: docs/handover/HO-21-2026-07-18-m4-b1-hotove-vstup-b2.md + docs/00
(D-log 2026-07-17, GO = piaty zapis) + paper/results-paper.md (v0.2,
commit 40f0a0c) + results/{M1-report,M2-sweep,M2-core-pair,
M2-spike-diagnostics,M2-core-pair-lr6e4,M2-g2lite,M2-recalibration,
M2-calibration}.md (posledne dva LEN v repe, citat cez Filesystem MCP).
Kontext: HO-20 §2 zavazny ramec; drafting rules v paperi §0 (INTERNAL);
ziadne nove merania/spend; publikacia Ne 19.7., reviewer-pass gate Ne
doobeda = podmienka; spec §16 a zakazany slovnik platia.
Uloha (B2): draft §4.1-4.8 + §5.1-5.5 + §6 do paper/results-paper.md
(edit_file, dry-run pred apply), kazde cislo s inline [src] tagom
overenym z repo reportov; T1/T2 ako tabulky z raw dat; figury =
existujuce commitnute ploty (overit nazvy v results/, medzera ->
Daniel); potom commit bod C2 + navrh pokracovania B3.
```
