# HO-27 — Post-publikačný cleanup uzavretý + v1.1 kickoff (AP-25 prijaté, zadanie V11 pre-registrované)

Dátum: 2026-07-19 (nedeľa večer). Nadväzuje: HO-26 + R7 D-log riadok. Detailný záznam: D-log 2026-07-19, tretí riadok.

## 1. Stav po tomto sedení

- **Zenodo kotva [OVERENÉ user-view]:** záznam 21439493 — md5 `a0177d2334b30adc552ed8d80f4a9509` = v5 presne, titul byte-zhodný so zdrojom `paper/ssra-results-paper-v1.0.md`, Version 1.0 / 19. 7. / Preprint / Open.
- **GitHub issues:** #3/#5/#6 boli zavreté už 19. 7. 10:19Z v závere B5-2 sedenia (Daniel + Claude, GitHub MCP) — HO-26 §2.2 bol voči realite stale (písaný pred exekúciou). Oversight komentárov OK; #5 nesie neškodný date-slip (rozhodnutie X = 5 % je z 06-11/MD-10, komentár píše 06-12) — bez akcie. #2 arXiv ostáva open.
- **AP-25 prijaté** — plné znenie v D-logu: brána identity pre ireverzibilné externé akcie (pre-registrácia identity zo zdroja: názov + veľkosť + md5; verifikácia v cieľovom systéme vrátane preview PRED potvrdzujúcim klikom; jednosúborový staging; user-view rozhodujúci, Claude fetch nestačí; checklist bez brány je neplatný).
- **Rozhodnutia V3–V5 (Daniel):** K1 GO — primárna cesta (ii) GCE CPU VM @ europe-west3 (attached SA read-only, žiadne kľúče, R7 nedotknuté), fallback (i) egress subset; jednotný scoped strop 3,00 EUR na celé K1; extrakcia zo VŠETKÝCH step-tagged ckptov; T-D povinné. Nota v1.1 GO. Plný rozsah K3 + K2 (conditional) + K1. **Ckpt delete = ihneď po oversight PASS §K1** (nie čakať na 31. 8.; deadline ostáva tvrdý strop).
- **Zadanie `docs/cc/V11-data-exploitation.md` v1** — pre-registrované: metriky T-A…T-E, kritérium C-T1 pre H-T1 (φ latentné queries ≈ referencia ako kandidátny mechanizmus ln(32) uniformity), gates G-V11-1…5, S0 inventáre + cenové verifikácie (Pravidlo W), commit reportu bez ohľadu na smer.
- Spend 0 EUR; kumulatív **72,37 EUR ≈ 24,1 %**.

## 2. Nota v1.1 — publikačný checklist (prvá ostrá aplikácia AP-25)

Vlastník klikov: Daniel. Obsah + build: Claude (container, kanonická build cesta per HO-26). Rozsah delta v1.0 → v1.1 = výhradne body (a)–(d) kroku 1.

1. **Delta obsahu (Claude), zdroj `paper/technical-note.md`:**
   (a) sekcia „AI Assistance Disclosure" — text ekvivalentný record-level disclosure (doplnenej 9. 7.) + veta, že do súboru je doplnená vo v1.1;
   (b) **Erratum spec §9 (decode retention):** overiť presné znenie pravidla v publikovanom v1.0 PDF (md5 `589b51d0…`) a uviesť old/new — opravené pravidlo: uzol u sa drží, ak u ∈ Frontier(t) ∪ Fenwick(t−w−1) (nález M1);
   (c) „Status of empirical follow-up" — 1 odsek s DOI 10.5281/zenodo.21439493, neutrálne (G1 nesplnené, H1 nepodporená), žiadna interpretácia navyše;
   (d) Version history (v1.0 2026-06-11 → v1.1 dátum publish).
   Diff v1.1 zdrojov proti v1.0 musí byť výhradne (a)–(d) — verifikovať diffom pred buildom.
2. **Build (Claude container):** MD → PDF identickým toolchainom (pandoc 3.1.3 + XeTeX TL2023 + GNU FreeFont). **AP-25(a): md5 + veľkosť PDF aj zdrojového MD zapísať do D-logu/checklistu PRED uploadom.**
3. **Zenodo (Daniel):** záznam noty 20647034 → „New version"; upload zo staging adresára obsahujúceho VÝHRADNE tento PDF (AP-25(c)); metadáta: Version 1.1, publication date = deň publish, Description = v1.0 + poznámka o v1.1 zmenách; Related works skontrolovať v UI (či sa IsCitedBy/IsContinuedBy → 21439493 dedia na novú verziu).
4. **AP-25(b) brána (Daniel, PRED Publish):** v Zenodo UI overiť názov + veľkosť + **md5 == krok 2** + preview (titulná strana: v1.1, disclosure sekcia viditeľná). Screenshot.
5. **Publish + post-check (AP-25(d)):** privátne okno — v1.1 live s vlastným DOI; v1.0 nedotknutá (md5 `589b51d0…` stále na v1.0); koncept-DOI spája verzie.
6. **Repo + D-log:** nový DOI do hlavičky `paper/technical-note.md`, D-log riadok, commit.

Anti-ciele: žiadne zásahy do v1.0 záznamu; žiadne obsahové zmeny technických sekcií noty; žiadny nový vedecký obsah.

## 3. Ďalší chat / poradie

1. Daniel: commit tohto batchu (00 + zadanie + HO-27) + výmena project files (min. `00-stav-a-triaz.md`; pridať HO-27 a `V11-data-exploitation.md`).
2. Nový chat: odovzdanie zadania CC (= schválenie per veto režim) → S0 → K3 → K2? → K1 (VM create/delete = Daniel, príkazy pripraví CC; wall cap 4 h).
3. Oversight review reportu (Claude) → Daniel: ckpt delete (pre-delete listing + kontrola úplnosti extraktov per AP-25 vzor) → D-log + ledger.
4. Nota v1.1 per §2 — nezávislá malá session, kedykoľvek (nemusí čakať na V11).
5. Standing: #2 arXiv endorsement; teardown ≤ 31. 8. (SA delete + bucket IAM cleanup + prípadné zvyšky GCS).

## 4. Otvárací prompt pre nový chat (paste-ready)

```
V11 exekucia. Cez Filesystem MCP precitaj docs/handover/HO-27-2026-07-19-post-publikacny-cleanup-v11-kickoff.md a docs/00-stav-a-triaz.md (tail 15), over git log -1 (<HASH>). Kontext: publikacia uzavreta (DOI 10.5281/zenodo.21439493), AP-25 prijate, zadanie docs/cc/V11-data-exploitation.md v1 pre-registrovane (K3 core; K2 conditional na S0-B; K1 cesta (ii) GCE CPU VM @ europe-west3, strop 3 EUR, vsetky ckpty, T-D povinne, fallback (i)); nota v1.1 GO (checklist HO-27 §2); ckpt delete az po oversight PASS §K1. Kumulativ 72,37 EUR ~ 24,1 %, 0 pods. Dnes: odovzdanie zadania CC + supervizia S0 (inventare + cenove verifikacie, 0 EUR).
```

(`<HASH>` doplň po commite.)
