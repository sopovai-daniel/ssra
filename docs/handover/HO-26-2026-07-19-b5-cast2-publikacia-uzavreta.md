# HO-26 — B5 časť 2 hotová: PUBLIKÁCIA UZAVRETÁ (s incidentom a same-day opravou)

Dátum: 2026-07-19 (nedeľa predpoludním). Nadväzuje: HO-25. Detailný záznam: D-log riadok 2026-07-19.

## 1. Stav po tomto sedení

- **Stage-2 paper LIVE: DOI 10.5281/zenodo.21439493** (Publish 11:37 CEST, overené v privátnom okne). Pôvodne rezervovaný DOI 10.5281/zenodo.21432587 **ZANIKOL** — tombstone (omylom publikovaná nota; owner-delete do 30 dní, dôvod „Duplicate of another record", 10:54 CEST). Na starý DOI nič neodkazuje.
- **Artefakt záznamu:** `paper/ssra-results-paper-v1.0.pdf` **md5 `a0177d2334b30adc552ed8d80f4a9509`** (v5; obsah = v4, zmena výhradne DOI riadok hlavičky); exportný zdroj md5 `db038ffb4bf1a9df0f238bed8b51becf`. **Kanonická build cesta: Claude container** (pandoc 3.1.3 + XeTeX TL2023 + GNU FreeFont) alebo `paper/export/build.sh` v prostredí s deps — lokálny pandoc na Macu neexistuje (v4 aj v5 vznikli v containeri).
- **Repo:** PUBLIC; DOI korekcia = commit `fc0f27b` (5 súborov: 2× paper md, README, build.sh, PDF); tag `paper-v1.0` presunutý na `fc0f27b` (pôvodný objekt `acf35bb` na `7c3ee52` mal starý DOI v správe); nezávisle overené z GitHubu.
- **Krok 7 hotový:** nota 20647034 Related works += IsCitedBy + IsContinuedBy → 21439493 (Preprint); súbor noty nedotknutý (md5 `589b51d0…` overené na zázname zo screenshotu).
- **Spend dnes 0 EUR; kumulatív 72,37 EUR ≈ 24,1 % z 300.**
- **Zenodo mechanika [OVERENÉ help.zenodo.org, 2026-07-19]:** owner-delete publikovaného záznamu do 30 dní (novinka 12/2025, tombstone drží citáciu); súbory po Publish nemenné (zmena len cez support); metadata editovateľné kedykoľvek.

## 2. Ďalší chat / otvorené

1. **R7 GCP SA revert (20.–21.7.):** zmazať SA kľúč `ssra-runpod` + vrátiť org-policy enforcement na ssra-poc (záväzok D-log 2026-07-12) — najbližšia operačná session.
2. **GitHub issues cleanup:** #3 (Zenodo/ORCID — hotové dávno), #5 (X% threshold — rozhodnuté 2026-06-12), #6 (Vertex budget cap — obsolete, RunPod pivot) zavrieť s odkazmi na D-log; #2 (arXiv endorsement) ostáva open ako post-publikačný kandidát.
3. **Nota v1.1** (voliteľná Zenodo „New version" s disclosure vo file) + **v1.1 exploatácia dát** — nové pre-registrované zadanie (0-EUR pravidlo platí; kandidáti per D-log 2026-07-17).
4. **Retencia step-tagged ckptov** (~100 GiB ≈ 2 EUR/mes): rozhodnutie do 2026-08-31 (viazané na v1.1 trajektóriovú analýzu).
5. **Návrh AP-25 (veto režim):** pred každou ireverzibilnou externou akciou (Publish, delete, transfer) sa identita artefaktu overuje v cieľovom systéme (názov + veľkosť + preview obsahu), nie iba v zdroji.
6. Announcement (LinkedIn a pod.) — mimo governance, Daniel.

## 3. Poznámky pre nový chat

- Incident + oprava kompletne v D-logu 2026-07-19 (vlastníctvo: klik Daniel; procesná diera checklistu Claude, priznaná in-flight). Žiadne prepisy histórie — korekcie append-only; git história obsahuje starý DOI v commite `7c3ee52` a D-log riadku 2026-07-18 zámerne (historický záznam).
- Nová tvrdá brána aplikovaná pri re-run Publish: artefakt overený v cieľovom Zenodo UI vrátane preview PRED klikom.
- Project files v Claude projekte vymeniť: minimálne `00-stav-a-triaz.md`; pridať HO-26.
