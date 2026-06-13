# HO-06 — Handover: spec v1.2 + M2 zadanie committnuté, M2 čaká na GPU quotu
**Projekt:** DFKS / SSRA (Scale-Shared Recursive Attention)
**Dátum:** 2026-06-13 | **Autor HO:** Claude (Opus 4.8) | **Schvaľuje:** Daniel Sopov
**Účel:** vstupný kontext pre nový čistý chat pod projektom (CC fáza 0 + spustenie M2 po quote). Nadväzuje na HO-05 (neprepisuje sa). Dôvod nového HO: open items HO-05 §4.1–4.2 sú splnené; tento chat (Opus 4.8) bol dojazd po prerušenom Fable-5 sedení a iba zosúladil stav.

---

## 1. Stav v jednej vete
M1 uzavreté; **spec v1.2 + `docs/cc/M2-assignment.md` sú hotové a committnuté** (2026-06-12); jediný blocker M2 = **GPU quota** (fallback beží na Danielovej strane); **CC fáza 0 (no-GPU groundwork) môže bežať hneď**.

## 2. Čo sa stalo (Chat 7 dojazd, 2026-06-13, Opus 4.8)
- Pôvodný Chat 7 (Fable 5, 2026-06-12) mal spraviť HO-05 §4.1 (spec v1.2) + §4.2 (M2 zadanie). **Prerušený** (vypršalo 5-h okno). Fable 5 odvtedy nedostupný (podľa Daniela: US exportný príkaz na Fable 5 / Mythos 5, 2026-06-12 — **neoverené z primárnych zdrojov, environmentálny fakt**). Pokračuje sa na **Opus 4.8**; CC aj ďalšie chaty tiež na Opus 4.8.
- **Zistené pri dojazde:** práca §4.1+§4.2 sa medzičasom **dokončila v inom sedení** a je v repe — `spec.md` = **v1.2** (vrátane MD-11…MD-13 v §18, stack-level tabuľky §12, prepísanej §13 validácie); `docs/cc/M2-assignment.md` v1 committnutý; D-log má oba zápisy (2026-06-12). Fable-5 draft patchu (orphan artefakt mimo repa) je tým **superseded — neaplikoval sa, zahodiť**.
- **Housekeeping (tento chat):** stará „Fáza" hlavička v `docs/00` ešte hlásila `spec v1.1` (kým D-log aj T3 boli v1.2) → zosynchronizované na v1.2 + aktualizovaný „Beží" riadok. Jediná zmena tohto chatu. Commit čaká (over `git status`).
- **Project-files v Claude projekte boli stale** (v1.1) — zdroj zámeny; **treba refresh** (00, spec.md, M2-assignment.md) pred ďalšími chatmi.

## 3. Kľúčové platné fakty
- `docs/spec.md` **v1.2** = jediný zdroj pravdy pre implementáciu. Uzavreté: D1–D6, Q1–Q5, **MD-1…MD-13**, gates G0/G1a/G1b-D3. v1.2 je čisto editorial (kodifikácia M1-potvrdení (a)–(d); žiadne nové normatívne rozhodnutie).
- `docs/cc/M2-assignment.md` v1 = zadanie pre CC. Štruktúra: **fázy 0–4** (fáza 0 = data pipeline + tokenizer + harness + checkpoint/resume, **bez GPU, hneď**; fáza 1 = kalibračný run = prvý platený krok so **STOP gate** u Daniela; fázy 2–4 = S1 sweep → S2 core pair → baselines). Protokoly **AP-8…AP-16** (AP-1…AP-7 z M1 platia kde aplikovateľné).
- **M2 prostredie:** GCP `ssra-poc` @ **europe-west3** (org policy Frankfurt-only), **1× L4 24 GB**, **strop 300 EUR kumulatívne** (tvrdý). GCE Spot vs Vertex = otvorené, rozhodne sa pri kalibrácii na cenách overených v deň objednávky (Pravidlo W, AP-10). A100 len cez project-level výnimku pre ew4 (NEAPLIKOVANÉ); in-policy rýchlejšia alternatíva = H100 ew3.
- **Blocker:** GPU quota zamietnutá (NOT_ENOUGH_USAGE_HISTORY) → Console žiadosť + seed usage + re-file (Danielova strana). Blokuje fázy 1+; fáza 0 nie.
- **[ODHAD]/veto v zadaní:** §5 configs (tvary S1 d=384/h=6/L=10 ≈24M, S2 d=640/h=10/L=15 ≈84M; tokenizer byte-level BPE **16k** tied; ctx 1024; ~20× token budget) a časť AP-8…AP-16 nesú `[ODHAD]`/`[návrh]` — Danielovo veto okno, ak ešte vedome neuzavreté.
- **G1 kritérium** (verdikt Daniel): stabilný tréning + val ppl @ ctx 1024 v ±5 % flat pri matched compute (AP-8 = matched params + matched tokens; FLOPs/wall-clock reportované, nie matchované). Pri faile: žiadny autonómny redesign (max 1 iterácia = Danielovo rozhodnutie + D-log).
- Smoke závery zakázané; NoPE summary keys [HYPOTÉZA] stojí (`summary_pos: none`); contingency flagy off; baseline (b) = fla-org/flash-linear-attention v0.5.0 MIT (GPU exekúcia v M2 uzatvára M1 AP-5).

## 4. Next steps (nový čistý chat)
1. **Štart nového chatu** s refreshnutými project-files (00 v1.2, spec.md v1.2, M2-assignment.md) + tento HO-06.
2. **CC fáza 0 kickoff** — ukáž CC na `docs/cc/M2-assignment.md` „Phase 0". Quota-independent, najproduktívnejší krok teraz.
3. (Voliteľne pred fázou 0) Danielov review veto okna §5 configs + AP-8…AP-16, ak ešte neuzavreté.
4. Po grantnutí quoty: fáza 1 kalibrácia (STOP gate) → potvrdenie/eskalácia HW → S1 → S2 → baselines.

## 5. Vstupy na Danielovej strane
- **GPU quota fallback** (Console žiadosť + seed usage + re-file) — long-lead, čím skôr.
- **Commit** dojazdu: over `git status`; ak sú v1.2/M2 zmeny už committnuté (pravdepodobne), tak Fáza housekeeping fix + HO-06 = malý commit (`git add docs/00-stav-a-triaz.md docs/handover/HO-06-*.md && git commit -S -m "Housekeeping: sync Fáza header to v1.2 + HO-06"`). Ak neboli committnuté, over skôr.
- **Refresh project-files** v Claude projekte (00, spec.md, M2-assignment.md).
- Budget custom period v Console (~2026-11-30 stop-loss horizont; stále mesačný kalendárny default).
- Voliteľné: GitHub MCP token (predtým bad credentials).

## 6. Inštrukcie pre nový chat
1. Project files: `00` (v1.2, vrátane Fáza hlavičky), `spec.md` **v1.2**, `01`–`03`, **HO-06**, `M1-assignment.md` + `M2-assignment.md` (vzory zadaní). Pri pochybnosti over v repe — repo vyhráva (project-files môžu byť stale).
2. Začni §4.2 (CC fáza 0 kickoff). Uzavreté rozhodnutia (D/Q/MD/gates) neotváraj.
3. Veto režim, epistemická disciplína, Pravidlo W; slovenčina ASCII v chate, angličtina v kóde/CC zadaniach; modely: Fable 5 nedostupný → Opus 4.8.

## 7. Mapa artefaktov (delta voči HO-05)
| artefakt | umiestnenie | stav |
|---|---|---|
| spec | `docs/spec.md` | **v1.2** (editorial; MD-11…MD-13) |
| D-log + Fáza hlavička | `docs/00-stav-a-triaz.md` | v1.2/M2 zápisy + Fáza hlavička zosúladená (tento chat) |
| M2 zadanie pre CC | `docs/cc/M2-assignment.md` | **v1, committnuté** (fázy 0–4, AP-8…AP-16) |
| HO-06 | `docs/handover/HO-06-2026-06-13-m2-ready-quota-block.md` | tento dokument |
| Fable-5 orphan draft | mimo repa (`/mnt/.../outputs/`) | superseded — zahodiť |
| GPU quota | Console/CLI | ⏳ zamietnutá → re-file (Daniel) |
