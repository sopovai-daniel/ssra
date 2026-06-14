# HO-07 — Handover: M2 Phase 0 hotové, čaká sa GPU quota → Phase 1 kalibrácia
**Projekt:** DFKS / SSRA (Scale-Shared Recursive Attention)
**Dátum:** 2026-06-14 | **Autor HO:** Claude (Opus 4.8) | **Schvaľuje:** Daniel Sopov
**Účel:** vstupný kontext pre nový čistý chat pod projektom (re-check quoty → Phase 1 kalibrácia po grante). Nadväzuje na HO-06 (neprepisuje sa).

---

## 1. Stav v jednej vete
M2 **Phase 0 UZAVRETÉ** (7/7, commity `e4fa16f` + `988cc18`, pushnuté); jediný blocker Phase 1 = **GPU quota** (L4 = 0, „not eligible" — re-check po víkende); Phase 1 = kalibračný STOP-gate (prvý platený GPU krok).

## 2. Čo sa stalo (tento chat, 2026-06-14, Opus 4.8)
- CC dostal Phase 0 zadanie (`docs/cc/M2-assignment.md` „Phase 0") a vykonal ho — **takmer celé lokálne na Macu** + GCS bucket. Žiadny GPU, spend ~14 MiB GCS.
- **Deliverables:** GCS bucket `gs://ssra-poc-ew3` (ew3, UBLA, PAP enforced); data pipeline FineWeb-Edu `sample-10BT` → document-disjoint split → tokenizer → packed uint16 shardy; tokenizer byte-level BPE 16k (sha256 `019568a2…`); harness rozšírený (bf16 autocast, checkpoint/resume).
- **AP-9 provenance overené živo z hubu** (Pravidlo W): licencia **odc-by**, hub sha `87f09149…`, retrieved 2026-06-14. SlimPajama fallback netreba.
- **Checkpoint/resume (AP-11): spojitá loss krivka OVERENÁ 2 certifikátmi** — unit test (atol < 1e-5 každý krok, obnova modelu+optimizera+oboch RNG streamov, atomický zápis) + real-harness preemption sim (kill po step-30 ckpt, resume **bit-for-bit**). 36 passed / 1 skip (M1 Triton).
- **CPU smoke:** SSRA-P1 2.70M, 60 krokov fp32, bez divergencie — len funkčnosť (spec §16, žiadny kvalitatívny záver). P-C `p1_attn_entropy` ≈ ln(32), M1-konzistentné.
- **`scripts/train.py` 2-riadkový meta-log bugfix** (precision logované 2× → TypeError; fix → raz). Len JSONL run-meta, **žiadna dizajn/spec zmena** (overené diffom).
- **Quota L4 stále 0** — Console aj `gcloud beta quotas preferences list` potvrdili tvrdý zámok „0 to 0" / „not eligible" (3 prefs podané 12.6., grantedValue 0, bez pre-evaluácie). **Seed VM `ssra-usage-seed` beží** (e2-small, ew3-a, od 12.6.) pre usage históriu.
- **D-log (2 riadky 2026-06-14) + tento HO** committnuté.
- **Incidenty:** (i) VS Code terminál spadol uprostred Phase 0 — pravdepodobne pamäťový tlak na 16 GB Air-e, lebo **paralelne bežal lokálny LLM tréning v Antigravity** súčasne s CPU-ťažkým smoke loopom. CC session obnovená cez `claude --continue`, **nič stratené** (groundwork commit `e4fa16f` bol už v git-e). **Ponaučenie: jeden ťažký tréning naraz na Macu.** (ii) CC nabehol na Fable 5 (starý default z M1) → prepnuté na Opus 4.8.

## 3. Kľúčové platné fakty
- `docs/spec.md` **v1.2** = jediný zdroj pravdy pre implementáciu. Uzavreté: D1–D6, Q1–Q5, **MD-1…MD-13**, gates G0/G1a/G1b-D3.
- **M2 prostredie:** `ssra-poc` @ **europe-west3** (org policy Frankfurt-only), **1× L4 24 GB**, strop **300 EUR kumulatívne**. GCS bucket `gs://ssra-poc-ew3`. GCP auth živé (`daniel@sopovai.com`); gcloud config izolovaný do `ssra` (project ssra-poc, ew3); ADC quota project = ssra-poc.
- **Quota:** L4 + spot L4 + GPUS_ALL_REGIONS = granted 0, „not eligible" (~2 dni usage nestačí). Re-check po víkende. Seed VM `ssra-usage-seed` nechať bežať do grantu, **potom zmazať** (`gcloud compute instances delete ssra-usage-seed --zone europe-west3-a` — boot disk auto-delete + efemerná IP ⇒ čistý delete).
- **Phase 1 = kalibrácia** (prvý platený GPU krok, ~1 h, **STOP gate u Daniela**): `pytest tests/` na GPU boxe (spec §14.1–.3, .7) + tok/s + peak VRAM (S1/S2 kandidáti, oba modely) + 1× GPU kill+resume + **AP-10 cenový check** (L4 + H100 ew3, URL+dátum, EUR projekcia, GCE Spot vs Vertex) → STOP, Daniel potvrdí plán + HW pred Phase 2.
- **GPU typ:** L4 = default cieľ (in-policy, fit 300 EUR, kalibrácia aj tak beží na L4). Voľba L4 vs H100 (ew3, in-policy, rýchlejší ale páli budget) vs A100 (len ew4 cez NEAPLIKOVANÚ org-policy výnimku) sa rozhodne **až s kalibračnými tok/s číslami**. **VRAM nie je bottleneck** pri 84M modeli — tradeoff je čisto tok/s vs budget.
- **Configs §5 ([ODHAD], kalibrácia potvrdí):** S1 d=384/h=6/L=10 ≈24M; S2 d=640/h=10/L=15 ≈84M; ctx 1024; ~20× token budget (S2 floor 10× ≈ 850M ak budget tlačí, AP-12). **Vocab 16k je zabetónovaný** (tokenizer natrénovaný + shardy spakované).
- Smoke závery zakázané (anti-goal); NoPE summary keys [HYPOTÉZA] stojí (`summary_pos: none`); contingency flagy off; baseline (b) = fla-org/flash-linear-attention v0.5.0 MIT (GPU exekúcia v M2 uzatvára M1 AP-5).

## 4. Otvorené pripomienky (vedomé, stoja vo veto režime)
- **AP-13 single-seed S2:** hlavné porovnanie pobeží na 1 seede (1337), **bez** near-boundary eskalácie (na rozdiel od M1 AP-3, kde gap v 4–6 % spustil extra seedy). Ak val-ppl gap padne k 5 % hranici, G1 verdikt stojí na jednom seede. Parita s M1 rigorom = budget na 1 extra S2 pár (najdrahšie runy) — alebo vedome akceptovať single-seed + priznať v paperi. **Nevyriešené, Danielovo rozhodnutie pred Phase 3.**
- **Forward note (M3, nie M2):** pri tréningu na ctx 1024 strom siaha po úroveň 10, takže `e_ℓ` pre ℓ=11..15 ostanú na inite (zeros) a pri M3 length-extrapolácii na 32k bežia netrénované (= efektívne level_emb OFF pre tie úrovne; graceful, nie bug). Treba vedieť pri interpretácii.
- **`val_bpc` M1 (char-level) NIE je porovnateľné s M2 (token-level)** — pomenovanie log poľa, flag pre M3.
- **Budget custom period** v Console (~2026-11-30 stop-loss horizont; stále mesačný kalendárny default) — TODO Daniel.

## 5. Next steps (nový čistý chat)
1. **Po víkende: re-check quota** — `gcloud beta quotas preferences list --project ssra-poc`. Ak `grantedValue: 1` → Phase 1. Ak `0` → re-submit cez Console (Edit Quotas na L4 rows) alebo počkať; seed VM nechať bežať.
2. **Po grante:** CC Phase 1 kalibrácia (STOP gate) → Danielov verdikt HW (stay L4 / H100 ew3 / w4 výnimka) + plán → Phase 2 (S1 sweep) → Phase 3 (S2 core pair) → Phase 4 (baselines).
3. **Voliteľne pred Phase 3:** Danielovo vedomé rozhodnutie k AP-13 (single seed vs extra S2 pár).
4. Po Phase 1 grante zvážiť **delete seed VM** `ssra-usage-seed` (už nebude treba).

## 6. Vstupy na Danielovej strane
- **Re-check / re-file GPU quota** po víkende (long-lead, blokuje Phase 1).
- **Seed VM `ssra-usage-seed`:** nechať bežať do grantu, potom zmazať.
- **Budget custom period** v Console.
- Voliteľne: **GitHub MCP token** (predtým bad credentials — Issue pre M2 ak treba).

## 7. Inštrukcie pre nový chat
1. Project files refresh v Claude projekte: `00` (v1.2, 2 nové D-log riadky 14.6.), `spec.md` v1.2, `01`–`03`, **HO-07**, `M1`/`M2-assignment.md`. Pri pochybnosti over v repe — **repo vyhráva** (project-files môžu byť stale).
2. Začni §5 (re-check quoty / Phase 1 po grante). Uzavreté rozhodnutia (D/Q/MD/gates) neotváraj.
3. Veto režim, epistemická disciplína, Pravidlo W (ceny/quoty/API/citácie overiť voči primárnym zdrojom, URL+dátum); slovenčina ASCII v chate, angličtina v kóde/CC zadaniach, dokumenty s diakritikou; modely: Fable 5 / Mythos 5 nedostupné → Opus 4.8 (vrátane CC).
4. **Deľba práce:** `docs/cc/M2-assignment.md` JE hotové zadanie pre CC — NEGENERUJ, NEPREPISUJ. Implementáciu robí CC; tento chat = dozor/analýza/D-log/písanie. `docs/00` edituje len Claude.ai (s Danielovým schválením + jeho commitom); CC docs nechytá (anti-goal).
5. **Mac disciplína:** jeden ťažký tréning naraz (Antigravity vs CC smoke loop sa bijú o CPU/MPS na 16 GB Air-e — príčina pádu terminálu 14.6.).

## 8. Mapa artefaktov (delta voči HO-06)
| artefakt | umiestnenie | stav |
|---|---|---|
| Phase 0 kód | `src/ssra/{data.py,checkpoint.py}`, `scripts/{data_pipeline.py,train.py}`, `tests/test_checkpoint_resume.py` | commity `e4fa16f` + `988cc18`, pushnuté |
| env pins + GPU image | `requirements.txt`, `requirements-gpu.txt`, `docker/Dockerfile` | commit `e4fa16f` (GPU build až Phase 1) |
| Phase 0 configs | `experiments/M2-phase0-{data,cpu-smoke}.yaml` | committnuté |
| tokenizer | `artifacts/tokenizer/fineweb-edu-bpe-16384.json` | sha256 `019568a2…`, committnutý |
| Phase 0 report + manifest | `results/M2-phase0-report.md`, `results/M2-phase0-data-manifest.json` | committnuté |
| GCS data | `gs://ssra-poc-ew3` (`phase0/data/…`) | 5 objektov, 14.17 MiB |
| seed VM | GCE `ssra-usage-seed` (e2-small, ew3-a) | ⏳ beží (zmazať po grante quoty) |
| D-log + HO-07 | `docs/00-stav-a-triaz.md`, `docs/handover/HO-07-*.md` | committnuté (tento chat) |
| GPU quota | Console / CLI | ⏳ L4=0 „not eligible" → re-check po víkende (Daniel) |
