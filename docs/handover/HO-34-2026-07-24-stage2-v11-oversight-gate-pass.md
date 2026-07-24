# HO-34 — Stage-2 v1.1: build + oversight PASS + reviewer-pass gate PASS; ďalej AP-25 publish session

Dátum: 2026-07-24 (oversight session, 0 EUR). Nadväzuje: HO-33. Detailný záznam: D-log 2026-07-24.

## 1. Stav po tomto sedení

- **Build commit `bbadd2b`** (CC per pre-registrované zadanie `6bf7be9`; push Daniel 2026-07-21): presne 2 súbory — `paper/results-paper.md` (+57/−9) + `results/stage2-v11-build-report.md` (+188, nový); 0 pod `src/ssra/` a `baselines/` [OVERENÉ GitHub file listom, G-V11-4 vzor].
- **Oversight PASS (Claude, 2026-07-24; 0 vecných korekcií)** v plnom rozsahu HO-33 §3: v1.0 md5 nezávisle byte-identické (PDF `a0177d2334b30adc552ed8d80f4a9509`, MD `db038ffb4bf1a9df0f238bed8b51becf`); každé nové číslo nezávisle prepočítané z artefaktov — CSV/JSON/flat NPZ/kód lokálne, ssra NPZ (50,4 MiB nad copy ceiling) snippetom exekuovaným Danielom, 9/9 hodnôt presná zhoda (vrátane T-C row 10 == 0,0 všetkých 52 ckpt + init a T-D 0,0538 → peak 0,0667 @ 33k → 0,0597); formulačný audit čistý (patch hunky výhradne v sankcionovaných lokáciách; headline kontexty byte-identické konštrukciou, grep 1×/5×/4×/4×; changelog veta verbatim 2×; [OP] 2 a K3-c definície verbatim; spec §16 čisté); code-read #8 = **NO** dvomi nezávislými cestami (fenwick.py + vlastná enumerácia + guard `l2 > p_max` v defaultnej vetve). Pre-registrácia silnejšia než tvrdené: zadanie V11 má jediný commit `ee173e7` (2026-07-19). Nity N1/N2 bez akcie (D-log).
- **Reviewer-pass gate (Daniel, 2026-07-24): PASS ⇒ publikačná podmienka splnená.**
- Paper drží `TODO(v1.1-DOI)` / `TODO(v1.1-date)` (2+2 výskyty: hlavička + Version history) — vyriešia sa výhradne v publish session.
- Spend 0 EUR; kumulatív nezmenený **72,37 EUR ≈ 24,1 %** (+ pending K1 korekcia ≈ 0,06 [ODHAD]).

## 2. Publish session — záväzný postup (AP-25 blok)

Deľba práce: konzolové/Zenodo akcie a git push = Daniel; placeholder resolve, export, PDF build, verifikácie = Claude; CC netreba.

Tvrdé mantinely: žiadne zmeny §4.x záverov a headline formulácií; export transformácie výhradne mechanické per B5 vzor; verzia v1.0 na zázname nedotknutá (md5 `a0177d2334b30adc552ed8d80f4a9509` drží); Zenodo súbory sú po Publish nemenné ⇒ PDF finálny PRED Publish; vzniknutý draft NEMAZAŤ (rezervovaný DOI zaniká nenávratne — lekcia 19.7.).

1. **Pre-flight:** local HEAD == remote HEAD (close-batch commit), čistý `git status`, D-log tail 2.
2. **Zenodo (Daniel):** záznam 21439493 → „New version" → „Get a DOI now!" → rezervovaný DOI vložiť do chatu. New version dedí metadáta z v1.0 — skontrolovať: Version **1.1**; publication date = deň Publish; Related works **Continues → 10.5281/zenodo.20647034** zachované; resource type Preprint; licencia CC BY 4.0; titul byte-zhodný s hlavičkou.
3. **Claude:** doplniť DOI + dátum do placeholderov (presne 2+2 výskyty); vygenerovať exportnú kópiu `paper/ssra-results-paper-v1.1.md` mechanicky per B5 vzor (§0 INTERNAL + inventár odstránené len v exporte; **11 figúr** + plné captions na miestach in-text pointrov — 6 z v1.0 + F5/F6/F7a–c; T1/T2/**T3** plné captions; nereferencované figúry nevkladať) + PDF `paper/ssra-results-paper-v1.1.pdf` kanonickým toolchainom (pandoc 3.1.3 + XeTeX TL2023 + GNU FreeFont; `paper/export/build.sh` aktualizovať na v1.1 mená/md5 komentár). Render verifikácia: nový DOI na s. 1; grep `TODO(` v exporte = 0; headline grep drží; komentárové stĺpce a URL-fix z v4/v5 vzoru držia.
4. **Commit + push (Daniel, `-S`, EN message):** `results-paper.md` (resolved placeholders) + export .md + PDF + build skript delta; grep `TODO(v1.1` vo forward-facing súboroch = 0.
5. **AP-25 (a) pre-registrácia identity (Claude → chat, PRED uploadom):** presný názov `ssra-results-paper-v1.1.pdf`, veľkosť v bajtoch, md5; Daniel md5 potvrdí lokálne.
6. **AP-25 (c) staging:** upload z jednosúborového adresára obsahujúceho výhradne cieľový PDF.
7. **Upload do draftu → AP-25 (b) verifikácia v cieli PRED Publish:** názov + veľkosť + md5 zobrazené Zenodo UI + preview titulnej strany (nový DOI viditeľný). Checklist bez bodov (a)–(b) je pre ireverzibilnú akciu neplatný — exekútor Publish nevykoná.
8. **Publish (Daniel).**
9. **AP-25 (d) user-view:** privátne okno/screenshot — v1.1 live; verzia v1.0 nedotknutá (md5 `a0177d23…` drží); koncept-DOI spája verzie. `web_fetch` nie je nezávislé overenie (cache; lekcia 2026-07-17).
10. **Metadata reciprocita [NÁVRH, rozhodnúť v session]:** nota 20647034 už nesie IsCitedBy/IsContinuedBy → 21439493; doplnenie odkazu na nové v1.1 DOI je voliteľné (verzie spája koncept-DOI) — metadata-only edit, súbory nedotknuté.
11. **Close:** tag `paper-v1.1` na finálnom commite + push tagu; D-log riadok; ledger (0 EUR); HO-35; výmena project files.

Target ≤ ~2026-08-10 [NÁVRH]; tvrdo ≤ 2026-08-31 (pred teardownom; pri publish ≤ 31.8. interim metadata note odpadá).

## 3. Pending mimo publish

- Billing korekcia K1 okna (~0,06 EUR [ODHAD]; append-only po objavení v konzole; nekritické).
- Teardown ≤ 2026-08-31 (checklist HO-29 §3 + ssh keypair a pubkey v projekt metadátach + finálny delete SA `ssra-runpod@…` + bucket IAM cleanup; inventár konzumentov GCS zvyškov pred delete).
- #2 arXiv endorsement — po v1.1 publish (žiadosť môže citovať obe DOI + oba koncept-DOI).

## 4. Otvárací prompt pre publish session (paste-ready)

```
Stage-2 v1.1 publish session (0 EUR, AP-25). Cez Filesystem MCP precitaj docs/handover/HO-34-2026-07-24-stage2-v11-oversight-gate-pass.md a docs/00-stav-a-triaz.md (tail 2), over git log -1 (<HASH>) a zhodu remote HEAD == local HEAD. Kontext: build bbadd2b, oversight PASS (0 vecnych korekcii), reviewer-pass gate PASS (D-log 2026-07-24); paper drzi TODO(v1.1-DOI)/TODO(v1.1-date) (2+2 vyskyty). Dnes: HO-34 §2 kroky 1-11 — Zenodo New version na 21439493 + Get a DOI now (ja), placeholder resolve + export kopia + PDF kanonickym toolchainom (ty), commit+push (ja), AP-25 identity pre-registracia PRED uploadom, verifikacia v cieli PRED Publish, Publish, user-view, volitelna metadata reciprocita, tag paper-v1.1, close. Ziadne zmeny §4.x zaverov a headline formulacii; draft nemazat.
```

(`<HASH>` doplň = SHA close-batch commitu po tvojom pushi.)
