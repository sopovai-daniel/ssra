# HO-05 — Handover: M1 uzavreté, M2 infra pripravená (čaká GPU quota)
**Projekt:** DFKS / SSRA (Scale-Shared Recursive Attention)
**Dátum:** 2026-06-12 | **Autor HO:** Claude | **Schvaľuje:** Daniel Sopov
**Účel:** vstupný kontext pre nasledujúci čerstvý chat (spec v1.2 + M2 zadanie pre CC + spustenie runov). Nadväzuje na HO-04 (neprepisuje sa).

---

## 1. Stav v jednej vete
M1 je formálne uzavreté (`results/M1-report.md` 7/7 kritérií PASS; gates **G1a aj G1b-D3 prejdené** — verdikty Daniel, D-log 2026-06-12), GCP infra pre M2 je pripravená (projekt `ssra-poc`, budget strop 300 EUR, HW plán 1× L4 @ europe-west3) a jediný blocker je GPU quota (zamietnutá pre čerstvý projekt — NOT_ENOUGH_USAGE_HISTORY; fallback beží na Danielovej strane).

## 2. Čo sa stalo v tomto chate (Chat 6, 2026-06-12)
- **Gate podklad + verdikty:** G1a (SSRA slope 0.983 vs flat 1.923; sklony nezávisle prepočítané z raw JSON) a G1b-D3 (P3 vs P1 medzera +0.44 % ≪ 5 %; bez divergencie, stabilizácia aktívna) — obe PASS, M1 uzavreté. Tri D-log zápisy (commit 1b51f8a).
- **Spec konfirmácie z M1 reportu (veto, prijaté):** ln_f pred unembedding; m_schedule=linear ⇒ pool=p1 + reject expanzívnych schedule; flag `summary_pos_override`; spec §12 inventár → **editorial batch spec v1.2 pred M2** (zostáva urobiť).
- **P-C nález:** P1 pooling attention ~uniformná počas celého smoke (entropia ≈ ln 32; žiadny collapse, žiadna špecializácia) → M2 štandardne loguje `p1_attn_entropy` (informatívne, nie gating).
- **Rozhodnutie budget:** strop 300 EUR na existujúcom GCP účte; alternatíva „nový účet + $300 trial" zamietnutá (free trial len pre nových zákazníkov, GPU počas trialu nedostupné — primárne zdroje overené 2026-06-12).
- **GCP ops session (CC, spend 0 EUR):** projekt `ssra-poc` vytvorený + billing nalinkovaný (slot uvoľnený odlinkovaním nepoužívaného auto-vytvoreného Gemini projektu, spend €0, súhlas Daniel) + APIs (compute, cloudquotas, billingbudgets; Vertex AI zámerne NIE) + 3 GPU quota requesty podané (okamžite zamietnuté — očakávané) + budget 300 EUR (prahy 17/50/83 %).
- **Org policy nález [OVERENÉ živým dotazom]:** `gcp.resourceLocations` = **europe-west3 only (Frankfurt-only)**; domnienka „EU" korigovaná. **V ew3 A100 neexistuje** (dostupné: L4 zóny a,b; H100/H100-mega a,c; T4 b) ⇒ HW plán M2 = **1× L4 (24 GB)**.
- Vedľajšie: cleanup starého nepoužívaného Google účtu (4 projekty zmazané, credentials revoknuté); gcloud SDK 561→572; erratum noty ostáva odložené (D-log 2026-06-12, nič nové).

## 3. Kľúčové platné fakty pre nový chat
- `docs/spec.md` **v1.1** = jediný zdroj pravdy pre implementáciu; D-log v `docs/00` aktuálny k 2026-06-12 (4 nové zápisy: G1a, G1b-D3, M1 closure, M2 infra). Uzavreté: D1–D6, Q1–Q5, MD-1…MD-10.
- **M2 prostredie:** projekt `ssra-poc` @ europe-west3 (jediný povolený región); 1× L4 24 GB; **GCE Spot vs Vertex custom training = otvorené** (Vertex účtuje ~20–30 % prirážku — rozhodnú ceny overené v deň objednávky, Pravidlo W); in-policy rýchlostná alternatíva = H100 v ew3 (bez zmeny policy); A100 len cez project-level výnimku pre europe-west4 (písomný návrh existuje, NEAPLIKOVANÉ, rozhodne sa po kalibračnom rune).
- **Quota:** 3 requesty (GPUS_ALL_REGIONS, L4, spot L4 @ ew3, všetky =1) zamietnuté NOT_ENOUGH_USAGE_HISTORY; fallback = Console manuálna žiadosť + seed usage na ssra-poc + re-file (`gcloud beta quotas preferences list` na stav); quota maily sleduje Daniel.
- **Budget:** 300 EUR scoped na ssra-poc, len alertuje (nevynucuje); zatiaľ mesačný kalendárny → TODO custom period do ~2026-11-30 (stop-loss horizont).
- **L4 dôsledok [ODHAD]:** ~3–4× nižší throughput než A100 ⇒ S2 run rádovo dni wall-clock; cena porovnateľná/nižšia; **prvý platený M2 krok = kalibračný run (~1 h) na reálne tok/s** — až podľa neho sa potvrdí L4 alebo eskaluje (H100 ew3 / w4 výnimka).
- Smoke závery zakázané (anti-goal); NoPE summary keys [HYPOTÉZA] stojí; baseline (b) = fla-org/flash-linear-attention v0.5.0 (MIT), exekúcia čaká na GPU.

## 4. Next steps pre nový chat (v poradí)
1. **Editorial spec v1.2** (malý batch): §12 inventár (ln_f, untied unembedding), flag `summary_pos_override`, linear-scope veta. Žiadna normatívna zmena.
2. **M2 „Zadanie pre CC"**: data pipeline (FineWeb-Edu / SlimPajama vzorka, tokenizer, GCS bucket v ew3), kalibračný run ako prvý platený krok, S1 (~25M) / S2 (~85M) configy, run disciplína (YAML pred štartom + `results/runs.md`), G1 kritérium (ppl ctx 1k ±5 % flat pri matched compute), checkpoint/preempcia handling (spot), monitoring (`p1_attn_entropy`, P-C, loss krivky), baseline (b) GPU exekúcia, GCE-vs-Vertex rozhodnutie s overenými cenami.
3. Po grantnutí quoty: kalibračný run → potvrdenie/eskalácia HW → S1 runy → S2.

## 5. Vstupy na Danielovej strane
- **Quota fallback** (Console žiadosť + seed usage; návod v chate 6) — long-lead, čím skôr.
- Budget custom period v Console (~2 min).
- Billing slot quota increase formulár (nezávislé od M2; účet je na 5-projektovom limite).
- Voliteľné: GitHub MCP token refresh (stále hlási bad credentials).

## 6. Inštrukcie pre nový chat
1. Project files: `00` (po 2026-06-12 vrátane M2 infra zápisu), `spec.md` v1.1, `01`–`03`, HO-05, `M1-assignment.md` (referenčný vzor zadania). Pri pochybnosti over v repe — repo vyhráva.
2. Začni krokom §4.1 (spec v1.2), potom §4.2 (M2 zadanie). Uzavreté rozhodnutia neotváraj.
3. Veto režim, epistemická disciplína a Pravidlo W podľa systémových inštrukcií; slovenčina ASCII v chate, angličtina v kóde/CC zadaniach.

## 7. Mapa artefaktov (delta voči HO-04)
| artefakt | umiestnenie | stav |
|---|---|---|
| D-log | `docs/00-stav-a-triaz.md` | 4 nové zápisy 2026-06-12 (G1a, G1b-D3, M1 closure, M2 infra) |
| M1 report | `results/M1-report.md` | FINAL, 7/7 PASS |
| GCP infra | projekt `ssra-poc`, budget, quota preferences | mimo repa (Console/CLI); podstatné fakty v D-logu |
| HO-05 | `docs/handover/HO-05-2026-06-12-m2-infra-ready.md` | tento dokument |
| M2 zadanie pre CC | `docs/cc/M2-assignment.md` | NEEXISTUJE — vznikne v ďalšom chate |
