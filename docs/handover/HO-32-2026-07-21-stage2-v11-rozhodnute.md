# HO-32 — Stage-2 v1.1 rozhodnuté: rozsah B prijatý; vstup pre build session

Dátum: 2026-07-21 (rozhodovacie sedenie, 0 EUR, žiadny build, žiadny commit do paper/). Nadväzuje: HO-31. Detailný záznam: D-log 2026-07-21.

## 1. Stav po tomto sedení

- **Rozhodnutie (Daniel, explicitný súhlas):** stage-2 v1.1 = **rozsah B (#1–#8)** — jedna verzia vyčerpá celý routovaný V11 materiál. Publish ako Zenodo **„New version" na 21439493** (nový DOI; v1.0 md5 `a0177d2334b30adc552ed8d80f4a9509` nedotknutá; koncept-DOI spája — overiť v UI, necitovať z pamäte). Target build ≤ ~2026-08-10 [NÁVRH], publish tvrdo ≤ 2026-08-31 (pred teardownom) ⇒ interim metadata note na 21439493 odpadá (HO-31 podmienka).
- **Rozsah B (#1–#8):**
  1. Nový **§5.6** — trajektóriová analýza: C-T1 verdikt výhradne mechanicky **INCONCLUSIVE** s oboma pre-registrovanými prahmi (supported: max ρ(latq) < 0,276348 — nesplnené; refuted: min ≥ 2,763482 — nesplnené; merané 1,533105 @ L13 … 7,836802 @ L9) + observácie (a)–(c) ako observácie (spec §16; mapovanie písmen verbatim z §K1-analysis reportu).
  2. **Row-10 spresnenie §5.2** — klasifikácia: **precision refinement, NIE erratum** (žiadne publikované číslo nepravdivé — „rows 11–15 exactly 0.0" aj max 4,638 platia; nepresná len implikácia „exercises only levels ≤ 10"; trénované riadky e_ℓ sú 0–9, riadok 10 presne 0,0 vo všetkých 52 ckpt aj inite). **§4.7 sa NEedituje** (jeho veta ostáva pravdivá — diff purity).
  3. **§5.1** — P-C entropia overlay figúra (17 sérií; 15/17 ≤ 0,00024 nat od ln 32; odchody len core runy). Text §5.1 **bez korekcie** — konzistentný s K3-a sémantickým nálezom; os-note do caption (entropia nad 32 poolovanými kľúčmi; participácia nad 16 latent queries, uniform 1/16).
  4. **§5.3** — V2b kvantová figúra (8/8 binád nezávisle re-derivovaných; doplnená bunka [256,512)=33; anotácia t=8192, ULP=w=64).
  5. **§5.5** — needle kategorizácia 720 trialov ako tabuľka (flat 36/4/320, SSRA 0/0/360; definície kategórií **verbatim zo zadania V11 §5**; kvalitatívne vzory ostávajú).
  6. **T-A…T-E kurátorsky** — 2–3 load-bearing figúry v tele §5.6 (kandidáti: ρ per-layer, T-C e_ℓ, T-D update-norm), zvyšok repo-referenced (anti-dilúcia negative-results paperu). Finálny výber [OP] v builde po inventári figúr.
  7. **Cross-ref hygiena** — veta o note v1.1 (DOI 10.5281/zenodo.21462145) zrkadliacej erratum in-file; **[26] OSTÁVA citácia v1.0** (pre-registračný record); §8 bullet → „(executed in v1.1; §5.6)"; Version history + hlavička v1.1; Zenodo related-works check.
  8. **Pre-registrovaný 0-EUR code-read:** konzumuje read-out plánovač level-10 (root) súhrn @ N=1024? Binárna otázka, len čítanie kódu; ak zodpovedaná, §5.2 uvedie mechanický dôvod [OVERENÉ z kódu] namiesto holého faktu. Zero-diff pod `src/ssra/` a `baselines/` platí.
- **Overenia v paperi [OVERENÉ z `paper/results-paper.md`, 2026-07-21]:** §5.2 verbatim znenie (trieda spresnenia potvrdená); §5.1 konzistentné s K3-a (žiadna korekcia textu; chybná glosa 1/32 nikdy nebola verejná); §8 posledný bullet „candidate material for a v1.1 of this record" nájdený verbatim (r. 372 per HO-31); inventár „New composite figures remain v1.1 scope" — kompozitné figúry vopred sankcionované.
- **Záväzné formulačné mantinely v1.1:** žiadne zmeny §4.x záverov ani headline formulácií („flat prior confirmed; SSRA prior violated", +10,22 %, 11,8×/11,1×); changelog povinne **„no v1.0 conclusion modified; one precision refinement (§5.2)"**; §0 INTERNAL pravidlá zdroja (number provenance tagy, binding formulations, forbidden vocab) platia pre všetky nové vety; žiadne nové merania/spend; K2 ostáva descoped.
- Spend 0 EUR; kumulatív nezmenený **72,37 EUR ≈ 24,1 %** (+ pending K1 korekcia ≈ 0,06 [ODHAD]).

## 2. Build session — checklist (poradie záväzné)

1. Pre-flight: HEAD zhoda; tento HO + D-log tail.
2. **Verbatim čítania (podmienka autorstva zadania):** §K1-analysis report (presné znenie observácií (a)–(c), cesty všetkých 7 figúr), zadanie `docs/cc/V11-data-exploitation.md` §5 (definície needle kategórií, T-A…T-E metriky), K3 artefakty (overlay + V2b figúry).
3. Autor **„Zadanie pre CC: stage-2 v1.1 build"** (`docs/cc/stage2-v11-build.md`, EN): presný zoznam editov rozsahu B; hard asserty na existenciu artefaktov (NPZ, `v11-k1-rho.csv`, summary JSON, figúry); pre-registrovaný code-read (#8); anti-ciele (žiadne zmeny §4.x, žiadne nové merania, zero-diff `src/ssra/` + `baselines/` — G-V11-4 vzor); changelog veta. Odovzdanie CC = schválenie (veto režim).
4. Build (CC) → **oversight pass** (Claude: každé nové číslo nezávisle proti raw artefaktom; formulačný audit spec §16) → **reviewer-pass gate (Daniel) = podmienka publikácie** („Čo by oponent zhodil ako prvé?" — očakávané údery: „post-hoc dolepok" → obrana = commitnuté zadanie s prahmi pred analýzou + mechanický INCONCLUSIVE; „figúrová dilúcia" → kurácia; T-C architektonické čítanie → len meraný fakt + prípadný mechanický dôvod z plánovača).
5. **AP-25 blok (Publish):** Zenodo New version na 21439493 → „Get a DOI now!" → nový DOI do hlavičky → PDF kanonickým toolchainom (`paper/export/build.sh`, trieda podpisu v5); **(a)** identity pre-registrácia PRED uploadom: `ssra-results-paper-v1.1.pdf` — presný názov, veľkosť v bajtoch, md5; md5 exportného zdroja; source commit SHA + presný počet súborov; **(c)** jednosúborový staging adresár; **(b)** verifikácia v cieli PRED Publish: názov + veľkosť + md5 v Zenodo UI + preview titulnej strany (v1.1, nový DOI) — bez (a)+(b) je checklist neplatný, Publish sa nevykoná; Publish; **(d)** user-view z privátneho okna/screenshot (web_fetch nie je dôkaz): v1.1 live, **v1.0 nedotknutá** (md5 `a0177d23…`); metadata reciprocita (related works; nota-záznamy bez zmeny — ich pointre na 21439493 platia cez version chain, overiť v UI).
6. Close: podpísaný commit paper/ (substantívne oddelene od housekeepingu, čistý `git status` pred tagom), tag `paper-v1.1` na source commit, D-log riadok (DOI, md5, AP-25 (a)–(d) status), ledger 0 EUR, HO; **#2 arXiv endorsement odblokovaný** — žiadosť môže citovať obe DOI + oba koncept-DOI.

## 3. Pending mimo build

- Billing korekcia K1 okna (~0,06 EUR [ODHAD]; append-only riadok po objavení v konzole; nekritické).
- Teardown ≤ 2026-08-31 (checklist HO-29 §3 + ssh keypair položka; inventár konzumentov GCS zvyškov pred delete).
- #2 arXiv endorsement — po v1.1 publish.

## 4. Otvárací prompt pre build session (paste-ready)

```
Stage-2 v1.1 build session (0 EUR lokalne; publish s AP-25). Cez Filesystem MCP precitaj docs/handover/HO-32-2026-07-21-stage2-v11-rozhodnute.md a docs/00-stav-a-triaz.md (tail 3), over git log -1 (<HASH>) a zhodu remote HEAD == local HEAD. Kontext: rozsah B (#1-#8) prijaty (D-log 2026-07-21); publish = Zenodo New version na 21439493; v1.0 (md5 a0177d23...) sa nedotyka; tvrdo <= 2026-08-31. Dnes: (1) verbatim citania §K1-analysis reportu + V11 zadania §5 + inventar figur; (2) autoruj docs/cc/stage2-v11-build.md per HO-32 §2 (edity #1-#8, hard asserty, code-read level-10, anti-ciele, changelog veta); (3) handoff CC -> build -> oversight -> reviewer-pass gate; potom AP-25 blok. Ziadne zmeny §4.x zaverov a headline formulacii.
```

(`<HASH>` doplň po commite + pushi close batchu tohto sedenia.)
