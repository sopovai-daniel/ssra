# HO-03 — Handover: Stupeň 1 publikovaný (Zenodo DOI), M0 kompletne uzavreté → M1
**Projekt:** DFKS / SSRA (Scale-Shared Recursive Attention)
**Dátum:** 2026-06-11 | **Autor HO:** Claude | **Schvaľuje:** Daniel Sopov
**Účel:** (1) dokumentácia vývoja naprieč chatmi, (2) vstupný kontext pre nasledujúci čerstvý chat (M1). Nadväzuje na HO-02 (neprepisuje sa; história G0/spec je tam).

---

## 1. Stav v jednej vete
M0 je kompletne uzavreté: po G0 prebehol T2-subset, novelty sken a dvojkolový review technical note, ktorá je **publikovaná na Zenodo — DOI 10.5281/zenodo.20647034** (https://zenodo.org/records/20647034, CC BY 4.0); priorita myšlienky je zafixovaná verejným DOI a ďalší krok je **M1 implementácia** cez „Zadanie pre CC".

## 2. Čo sa stalo v tomto chate (Chat 4, 2026-06-11)
- **T2-subset hotový (Pravidlo W):** #3–#5, #8–#13, #16–#20 overené v primárnych zdrojoch; `02` prepísaná s URL + dátumami. Nálezy: FMA (#4) má per-level učené váhy (delimitácia drží); StreamingLLM (#20) prideľuje pozície v rámci cache ⇒ čistý NoPE mix súhrnných kľúčov = [HYPOTÉZA] pre M1 smoke, contingency `summary_pos: virtual` v spec existuje.
- **Novelty sken arXiv:** pridané #21–#25 (GPST = najbližší sused, MANO, PSM, HKT, DTP). Verdikt: kombinácia z §17 (cross-scale sharing × učená m-slot kompresia × kauzálny LM nad textom) drží voči #1–#25. Spec §17 rozšírená + explicitné delimitačné vety.
- **Technical note:** draft → reviewer pass → review kolo 1 (V1–V4) → v1.0 schválená Danielom („note OK"). Doplnené: ME-1 (exhaustívnosť dôkazu kauzality, aj spec §7), ref #14 (HRM, uzatvára surveyed set), overení autori #22 (Colagrande et al.) a #24 (Cirrincione), ORCID a rezervované DOI v hlavičke.
- **Publikácia:** ORCID 0009-0004-8584-5156 zaregistrovaný; Zenodo postup overený v oficiálnych docs; DOI rezervované → vložené do hlavičky → PDF vygenerované Claudom (pandoc+wkhtmltopdf, DejaVu Sans, 5 strán A4, vizuálne skontrolované) → upload (md5 zhoda bit po bite) → **Publish 2026-06-11**. Záznam overený fetchom (autor + ORCID + afiliácia, súbor, OpenAIRE).
- Aktualizované v repe: `00` (D-log ×7, Fáza, T1/T2/T4), `02` (kompletný prepis + #21–#25 + autori), `spec.md` (3+2 mikro-edity, veto režim, bez normatívnych zmien), `paper/technical-note.md` (v1.0) + `paper/ssra-technical-note-v1.0.pdf`. Commity: 80c9804, 12ad73b + finálny commit s tagom `note-v1.0`.

## 3. Kľúčové platné fakty pre nový chat
- `docs/spec.md` v1.0 = jediný zdroj pravdy pre **implementáciu**; pri spore o **stave** platí D-log v `docs/00`. D1–D6, Q1–Q5, MD-1…MD-10 uzavreté — neotvárať bez nového dôkazu sporu.
- **Stupeň 1 publikovaný:** DOI 10.5281/zenodo.20647034. Súbory záznamu sú po publish nemenné (vlastná editácia len 45 dní; ďalej len nová verzia cez Manage versions — nový DOI vo verznej rodine); metadáta editovateľné kedykoľvek. Stage-2 paper cituje note ako prior version.
- Zložitosť [OVERENÉ]: tréning Θ(N·(w + m·log N)·d)/vrstva (trieda #2, nie lepšia); decode stav O(m·d·log N), konštanta 2 (retenčné pravidlo spec §9). Defaulty: m=16, w=64, k=2; P1 default / P3 challenger (**G1b-D3: X = 5 %**) / P2 control.
- Žiadny tréning neprebehol, žiadne meranie neexistuje — všetky empirické výroky v note sú pre-registrované predikcie, nie nálezy.
- IP režim po stupni 1: myšlienka je verejná, repo ostáva private do stage-2; expozičné okno sa publikáciou uzavrelo presne podľa plánu.

## 4. Next step pre nový chat (v poradí)
1. **„Zadanie pre CC" pre M1** (vzor zo systémových inštrukcií: cieľ, spec referencie, akceptačné kritériá, anti-ciele; žiadny riadkový mikromanažment). Akceptačné kritériá = M1 testy spec §14: shift test, completion test (atol 1e−4 fp32), gradient flow, throughput/VRAM krivky (G1a: log-log slope ≤ 1,5 a pod flat), P-C diagnostika, G1b-D3 (P3 v pásme 5 % od P1 na 10M smoke), P3 determinizmus. Anti-ciele zo spec §16.
2. Implementácia v Claude Code podľa `CLAUDE.md` + spec: level-wise batching (D6, žiadna Python rekurzia), P1/P2/P3 + hybrid za jedným rozhraním, config YAML podľa spec §13 vrátane validačných pravidiel.
3. Smoke runy lokálne (MacBook M1 16 GB, ≤10M parametrov, char-level) — iba funkčnosť, žiadne závery.
4. Disciplína: 1 run = 1 YAML config v `experiments/` commitnutý PRED spustením + riadok v `results/runs.md`.

## 5. Vstupy na Danielovej strane
- Spustenie nového chatu pre M1 (tento je po viacerých kompakciách na hranici kontextu).
- Z T4 zostáva len arXiv endorsement — potrebné až pre M4, nie urgentné.
- Pred M2: budget strop Vertex AI (zatiaľ nie urgentné).

## 6. Inštrukcie pre nový chat
1. Project files: `00`–`03` + `spec.md` (aktuálne po commite tohto HO — Daniel vymení kópie); pri pochybnosti over v repe, repo vyhráva.
2. Začni krokom §4.1 (Zadanie pre CC). Nič z uzavretých rozhodnutí neotváraj.
3. Veto režim platí; epistemická disciplína a Pravidlo W podľa systémových inštrukcií projektu.
4. Slovenčina ASCII v chate, diakritika v dokumentoch, angličtina v kóde/spec/CC zadaniach.

## 7. Mapa artefaktov (delta voči HO-02)
| artefakt | umiestnenie | stav |
|---|---|---|
| technical note v1.0 (zdroj) | `paper/technical-note.md` | publikovaná verzia; nemeniť bez novej Zenodo verzie |
| technical note v1.0 (PDF) | `paper/ssra-technical-note-v1.0.pdf` + Zenodo | publikované; md5 589b51d026ce08ec1421d29aa4786ae6 |
| Zenodo záznam | https://zenodo.org/records/20647034 (DOI 10.5281/zenodo.20647034) | živý, Open, CC BY 4.0 |
| prior-art mapa | `docs/02-prior-art-mapa.md` | #1–#25, T2-subset ✔ |
| tento HO | `docs/handover/HO-03-2026-06-11-stage1-published.md` | finálny |
| Zadanie pre CC (M1) | TBD (docs/ alebo GitHub issue) | neexistuje — next step §4.1 |
