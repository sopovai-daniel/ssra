# HO-35 — Stage-2 v1.1 publikovaná (AP-25 (a)–(d) splnené, 1 poradová deviácia bez incidentu); publikačná línia V11 kompletne uzavretá

Dátum: 2026-07-24 (publish session). Nadväzuje: HO-34. Detailný záznam: D-log 2026-07-24 (tretí riadok).

## 1. Stav po tomto sedení

- **Stage-2 v1.1 LIVE:** DOI **10.5281/zenodo.21530947**; v1.0 nedotknutá (21439493, verzný panel v1.0 + v1.1); koncept-DOI **10.5281/zenodo.21439492** resolvuje na najnovšiu verziu. Identity: PDF `ssra-results-paper-v1.1.pdf` md5 `f0d6dab4e5247c3a96568a2aec1d218b` (1 795 836 B, 20 strán A4); exportný zdroj `paper/ssra-results-paper-v1.1.md` md5 `5384cc4e6cc6239098fc5b66f3aec13f` (465 r.); `paper/results-paper.md` po placeholder resolve md5 `595eb96377388c01cdfdf4bb089ebb93`. Source commit **`a60beea`** (presne 4 súbory; zero-diff pod `src/ssra/`, `baselines/` — [OVERENÉ GitHub file listom]), tag **`paper-v1.1`** na `a60beea`.
- **Render:** v1.1-špecifický parameter **linestretch 0,97** (v1.0: 0,98; dôvod: sirota poslednej referencie na 21. strane — autorský vizuálny nález; zdôvodnenie zdokumentované v `paper/export/build.sh` vrátane lmodern advisory). Re-verifikačná batéria 10/10 PASS; figúrová mapa F5 s. 13, F6 s. 14, T3 s. 15, F7a s. 16, F7b+F7c s. 17.
- **Description záznamu:** zdedený Abstract + AI Assistance Disclosure nedotknuté (obe sekcie vo v1.1 byte-identické s v1.0); append sekcie „Version history“ (verbatim z papera, de-markdown) vykonaný PRED Publish.
- **Poradová deviácia (bez incidentu):** skutočná sekvencia = Zenodo blok (upload → metadata + description → AP-25 (b) → Publish) PRED git blokom (commit → push → tag); identity chain (a) → (b) intaktný, publikovaný PDF byte-identický s committnutým.
- Kumulatív nezmenený **72,42 EUR ≈ 24,1 %** (session 0 EUR).

## 2. Pending

- **Voliteľná metadata reciprocita** (mikro-rozhodnutie, bez termínu, metadata-only): kandidáti — nota v1.0 (20647034) a/alebo nota v1.1 (21462145) doplniť IsCitedBy → 21530947; paper v1.0 záznam (21439493) sa NEDOTÝKA (verzný panel linkuje automaticky). Zamietnutie je legitímna voľba — DataCite sémantika už drží cez Continues + verzie.
- **Teardown ≤ 2026-08-31** (checklist HO-29 §3): GCS zvyšky (g2lite + data prefixy; pred delete inventár konzumentov), SA `ssra-runpod` finálny delete + bucket IAM cleanup, lokálny `~/.ssh/google_compute_engine` keypair + pubkey v projekt metadátach; R1 ckpt položka splnená (core pair zmazaný 20.7.).
- **#2 arXiv endorsement** (open; žiadosť môže citovať všetky DOI + oba koncept-DOI).

## 3. Otvárací prompt pre nový chat (paste-ready)

```
Post-publish standing session (0 EUR). Cez Filesystem MCP precitaj docs/handover/HO-35-2026-07-24-stage2-v11-publikovana.md a docs/00-stav-a-triaz.md (tail 3), over git log -1 (<HASH>) a zhodu remote HEAD == local HEAD. Kontext: stage-2 v1.1 LIVE (10.5281/zenodo.21530947, AP-25 cisto s poradovou deviaciou bez incidentu), v1.0 nedotknuta, tag paper-v1.1 na a60beea, V11 publikacna linia kompletne uzavreta. Agenda podla vyberu: (a) teardown planing/exekucia (checklist HO-29 §3 + HO-35 §2, tvrdo <= 2026-08-31), (b) metadata reciprocita mikro-rozhodnutie, (c) arXiv endorsement (#2). Ziadny build, ziadne zmeny paper/.
```

(`<HASH>` doplň po commite + pushi close batchu tohto sedenia.)
