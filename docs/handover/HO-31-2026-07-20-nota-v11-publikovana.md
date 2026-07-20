# HO-31 — Nota v1.1 publikovaná (prvá ostrá aplikácia AP-25 čistá); publikačná hygiena stage-1 uzavretá

Dátum: 2026-07-20 (session 4, večer). Nadväzuje: HO-30. Detailný záznam: D-log 2026-07-20 (piaty riadok).

## 1. Stav po tomto sedení

- **Nota v1.1 LIVE:** DOI **10.5281/zenodo.21462145**; v1.0 nedotknutá (md5 `589b51d026ce08ec1421d29aa4786ae6` overené user-view); koncept-DOI **10.5281/zenodo.20647033** spája verzie. AP-25 (a)–(d) splnené, 0 incidentov, spend 0 EUR. Identity: PDF `ssra-technical-note-v1.1.pdf` md5 `63ae756bae7db282dc483e124970b7f5` (108 263 B); zdroj `paper/technical-note.md` md5 `99f0d1eb9815a04b836979170dddd762` (31 419 B). Source commit `e00d105` (presne 3 súbory; zero-diff pod `src/ssra/`, `baselines/`).
- **Rozsah v1.1 (Daniel, veto režim):** výhradne HO-27 §2 (a)–(d) — disclosure vo file (record-level text verbatim + 2 vyznačené odchýlky), erratum §2.6 (zrkadlo stage-2 §2.8), Status of empirical follow-up, Version history + hlavička. **V11 kandidáti do noty NEzaradení** — routing do stage-2 línie (results paper r. 372 sám avizuje trajektóriovú analýzu ako kandidáta „for a v1.1 of this record"); row-10 spresnenie patrí tam.
- **Nálezy sedenia:** (i) HO-27 §2(b) mal invertované old/new — „opravené pravidlo" v checklistoch citovať z governing artefaktov (spec §9 / paper §2.8), nie parafrázou; zachytené vstavaným verifikačným krokom (pdftotext publikovaného v1.0 PDF). (ii) v1.0 nota bola Qt/DejaVu export (5 str.), nie kanonický toolchain ⇒ v1.1 vizuálne iná (FreeSerif, 7 str.; akceptované). (iii) Build detaily fixované v `paper/export/build-note.sh` (lmodern stub pre orezané containery; monofont Scale=0,9 kvôli 106-znakovému komentárovému riadku §2.1; render-verifikované 0 zalomení + bbox stĺpce).
- Kumulatív nezmenený **72,37 EUR ≈ 24,1 %** (+ pending K1 korekcia ≈ 0,06 [ODHAD]).

## 2. Pending

- **Billing korekcia K1 okna** (~0,06 EUR [ODHAD]): append-only riadok po objavení v GCP konzole (report typicky T+1; nekritické).
- **Rozhodnutie: stage-2 v1.1** — triáž V11 materiálu (C-T1 inconclusive + observácie, row-10 spresnenie §5.2, T-A…T-E figúry, V2b kvantová figúra, needle kategorizácia, P-C entropia overlay) do results paper v1.1. Odporúčanie: samostatná session do ~2–3 týždňov, pred teardownom; interim metadata note na 21439493 len ak stage-2 v1.1 sklzne za august.
- Teardown ≤ 2026-08-31 (checklist HO-29 §3 + ssh keypair položka; pred delete GCS zvyškov doplniť inventár konzumentov).
- #2 arXiv endorsement open (riešiť po stage-2 v1.1 rozhodnutí; žiadosť môže citovať obe DOI + koncept-DOI).

## 3. Otvárací prompt pre nový chat (paste-ready)

```
Stage-2 v1.1 triaz (0 EUR, rozhodovaci balik). Cez Filesystem MCP precitaj docs/handover/HO-31-2026-07-20-nota-v11-publikovana.md a docs/00-stav-a-triaz.md (tail 3), over git log -1 (<HASH>) a zhodu remote HEAD == local HEAD. Kontext: nota v1.1 LIVE (10.5281/zenodo.21462145, AP-25 cisto, v1.0 nedotknuta), V11 kompletna, V11 material routovany do stage-2 linie (results paper r. 372). Dnes: triaz kandidatov (C-T1 + observacie (a)-(c), row-10 spresnenie §5.2, T-A..T-E figury, V2b kvantova figura, needle kategorizacia, P-C entropia overlay) -> navrh rozsahu stage-2 v1.1 + checklist s AP-25 blokom; bez buildu, bez commitov do paper/.
```

(`<HASH>` doplň po commite + pushi close batchu tohto sedenia.)
