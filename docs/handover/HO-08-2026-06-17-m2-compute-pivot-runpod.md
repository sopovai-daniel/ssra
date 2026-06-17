# HO-08 — Handover: M2 compute pivot na RunPod (GCP GPU quota slepá ulička)
**Projekt:** DFKS / SSRA (Scale-Shared Recursive Attention)
**Dátum:** 2026-06-17 | **Autor HO:** Claude (Opus 4.8) | **Schvaľuje:** Daniel Sopov
**Účel:** vstupný kontext pre nový čistý chat = **RunPod research + exekúcia** (over live ceny → naplánuj a spusti prvý GPU beh). Nadväzuje na HO-07 (neprepisuje sa).

---

## 1. Stav v jednej vete
M2 **Phase 0 hotové** (7/7, pushnuté); GCP GPU quota = **doc-potvrdená slepá ulička** (`NOT_ENOUGH_USAGE_HISTORY` naprieč VŠETKÝMI projektmi) ⇒ **compute sa presúva na RunPod** (rental GPU), **GCS `gs://ssra-poc-ew3` ostáva kanonický storage**. Ďalší krok = over RunPod ceny + spusti prvý beh.

## 2. Prečo pivot (doc-podložené, 2026-06-17)
- Console „Edit Quotas" potvrdil tvrdý zámok **„0 to 0 / not eligible"** na L4 aj Preemptible L4 @ ew3 — Submit zablokovaný.
- **Rovnaký zámok pod established projektmi `sopovai-prod` aj `sai-sentry-prod`** (limit 0, eligibility blokovaná naprieč celým billing účtom). Cize to nie je o čerstvosti ssra-poc — Google gejtuje GPU prístup pre celý účet.
- Oficiálne Google docs [OVERENE, načítané 2026-06-17]: dôvod = `NOT_ENOUGH_USAGE_HISTORY` (Cloud Quotas API enum); GPU quota sa auto-udelí len projektu s **established billing history** (Deep Learning VM troubleshooting doc); seed VM (5 dní, e2-small) prah nedosiahol a „sufficient usage" je opaque ⇒ pasívne čakanie nemá zmysel.
- ew3 lock = org policy (Frankfurt-only); GPU eligibility je **region-independent** — zmena regiónu by nepomohla.
- **Rozhodnutie (Daniel, veto režim):** pivot na **RunPod** (per-hour rental GPU, žiadny quota approval). Alternatívy zvážené: Modal (runner-up — serverless, lepší na ongoing ops/repro, ale viac setupu teraz), Scaleway/OVH (EU-residency — pre tento workload non-issue, FineWeb-Edu = verejný text, 0 PII), Vast.ai (najlacnejší, ale reliability/repro risk). RunPod = najmenší delta od pôvodného „GCE Spot + vlastný checkpoint/resume" plánu.

## 3. Čo sa NEMENÍ (pivot je len „kde beží compute")
- **GCS `gs://ssra-poc-ew3` = kanonický artifact/checkpoint store** — dosiahnuteľný z RunPod cez service account (kľúč **MIMO repa**, secrets hygiena). Pull dát z GCS → tréning na RunPod → push checkpointov do GCS. Izolácia ostáva.
- **Prenosné bez zmeny:** Docker image (`docker/Dockerfile`, CUDA 12.4 + torch cu124), `scripts/train.py`, **checkpoint/resume (AP-11 — postavený presne pre preemptovateľný Spot GPU)**, data pipeline, tokenizer (BPE 16k, sha256 `019568a2…`).
- **Metodológia drží:** gates (G1: stable training + val ppl @ ctx 1024 ±5 % flat, verdikt Daniel), AP-8 matched-compute (matched params + tokens), configs §5 (S1 ≈24M, S2 ≈84M, ctx 1024, ~20× tokeny, S2 floor 10× ≈ 850M), baselines, run discipline (1 run = 1 committed YAML + runs.md row), bf16 autocast (AP-16).
- **`docs/cc/M2-assignment.md` ostáva v platnosti** — len exekučný substrát sa mení z GCP na RunPod. Fázy: kalibrácia (tok/s + VRAM + kill/resume na RunPod) → S1 sweep → S2 core pair → baselines.

## 4. Čo je BEZPREDMETNÉ / na zmazanie
- **AP-10 (GCE Spot vs Vertex)** — nahradené RunPod, neriešiť.
- **Seed VM `ssra-usage-seed`** — stratil zmysel. **Zmazať:** `gcloud compute instances delete ssra-usage-seed --zone europe-west3-a`.
- GPU quota requesty na GCP — neriešiť (slepá ulička).
- Budget custom period na GCP Console — GCP scoped budget už nekryje compute (presun na RunPod billing); GCP ostáva len storage (centy).

## 5. Next steps (nový research/exekučný chat)
1. **Over live RunPod ceny (Pravidlo W, URL + dátum):** L4 / RTX 4090 / A100 per-hour (on-demand aj Spot/community), v EUR. Pri 84M modeli je VRAM non-issue ⇒ lacné GPU stačí; tradeoff = tok/s vs cena.
2. **EUR projekcia pod 300 EUR strop** na ~850M–1.7B token budget (kalibrácia dá reálne tok/s). Over, či sa zmestí + rezerva na S1 sweep + baselines + iterácie.
3. **GCS access pattern z RunPod:** service account (read dáta + write checkpointy do `gs://ssra-poc-ew3`), kľúč injektovaný ako secret/env, NIKDY v repe ani v image. Over najbezpečnejší vzor (HMAC key vs SA JSON vs workload identity).
4. **Launch postup:** RunPod pod/template z `docker/Dockerfile`, pull dát, kalibračný beh (pytest na boxe + tok/s + VRAM + 1× kill/resume = ekvivalent Phase 1 STOP-gate) → Daniel potvrdí → S1 → S2.
5. **Budget tracking** na RunPod (per-run EUR ledger v `results/runs.md`, ako AP-12 ale na RunPod cenách).

## 6. Vstupy na Danielovej strane
- **RunPod účet** (registrácia + payment method + kredit).
- **GCS service account** pre RunPod (vytvoriť SA s read/write na bucket, stiahnuť kľúč — držať lokálne, MIMO repa).
- **Zmazať seed VM** `ssra-usage-seed`.
- Voliteľne: GitHub MCP token (predtým bad credentials).

## 7. Inštrukcie pre nový chat
1. Project files refresh: `00` (2 nové D-log riadky 17.6. — pivot), `spec.md` v1.2, `01`–`03`, **HO-08**, `M2-assignment.md`. Pri pochybnosti repo > project-files.
2. Začni §5 (RunPod ceny → projekcia → launch). Uzavreté rozhodnutia (D/Q/MD/gates) neotváraj.
3. Veto režim, Pravidlo W (ceny/API/verzie over voči primárnym zdrojom, URL+dátum, necituj z pamäte), markery [OVERENE]/[HYPOTEZA]/[SPEKULACIA]. Slovenčina ASCII v chate, EN v kóde/CC; dokumenty s diakritikou. Modely: Fable/Mythos nedostupné → Opus 4.8.
4. **Deľba práce:** implementáciu/launch robí Claude Code (alebo Daniel manuálne na RunPode); tento chat = dozor/analýza/D-log/písanie. `docs/00` edituje len Claude.ai (s Danielovým schválením + jeho commitom).
5. **Governance:** pivot je vedome logovaný (D-log 17.6.); RunPod exekučné detaily (presný launch flow, secret handling) môžu vyžadovať drobný update `docs/cc/M2-assignment.md` alebo nový krátky CC dodatok — navrhnúť, nie unilaterálne.

## 8. Mapa artefaktov (delta voči HO-07)
| artefakt | umiestnenie | stav |
|---|---|---|
| D-log pivot zápis | `docs/00-stav-a-triaz.md` (riadok 17.6.) | committnuté (tento chat) |
| HO-08 | `docs/handover/HO-08-2026-06-17-*.md` | committnuté (tento chat) |
| Prenosné na RunPod | `docker/Dockerfile`, `requirements-gpu.txt`, `scripts/train.py`, `src/ssra/checkpoint.py`, data pipeline | bez zmeny (commity `e4fa16f`/`988cc18`) |
| GCS storage | `gs://ssra-poc-ew3` | ostáva kanonický artifact/checkpoint store |
| seed VM | GCE `ssra-usage-seed` | ⏳ NA ZMAZANIE (stratil zmysel) |
| RunPod účet + GCS SA | — | ⏳ Daniel vytvorí pred prvým behom |
| GCP GPU quota | — | ✗ slepá ulička, neriešiť |
