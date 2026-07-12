# HO-09 — 2026-07-12 — M2: RunPod launch-ready (ceny + HW + GCS access uzavreté)

**Vstupný bod pre nový chat.** Čítať spolu s `docs/00-stav-a-triaz.md` (D-log riadky 2026-06-17 a 2× 2026-07-12 = kompletný stav pivotu). Predchádzajúci HO: HO-08 (compute pivot RunPod). Spec: v1.2. Zadanie: `docs/cc/M2-assignment.md` **v1.1**.

## 1. Čo sa stalo od HO-08

1. **RunPod GCS access hotový** (D-log 2026-07-12, commity `bfa0683` + `fd31cd3`): org-policy `iam.disableServiceAccountKeyCreation` exemptovaná len na `ssra-poc` (rozhodnutie (a), Daniel); SA `ssra-runpod@ssra-poc.iam.gserviceaccount.com` s jediným bindingom `roles/storage.objectAdmin` na `gs://ssra-poc-ew3`; kľúč `~/ssra-secrets/ssra-runpod-key.json` (0600, mimo repa); round-trip verifikovaný. ⚠ Otvorený záväzok: revert (zmazať kľúč + vrátiť enforcement) po M2/M3.
2. **RunPod ceny overené** (D-log 2026-07-12, Pravidlo W: runpod.io/pricing + docs.runpod.io/pods/pricing, 2026-07-12): A100 PCIe 80 GB $1.39/hr · A100 SXM $1.49 · H100 PCIe $2.89 · L4 $0.39 · RTX 4090 $0.69; per-second billing, egress $0; pod bez network volume sa pri balance 0 terminuje aj s dátami (kryté AP-11 — checkpointy v GCS).
3. **HW rozhodnutie (Daniel):** default pre Phase 1 kalibráciu = **1× A100 80 GB**; dostupnosť sa rieši live v RunPod konzole v deň deployu, fallback poradie **A100 PCIe → A100 SXM → H100 PCIe** (rýchlostná eskalácia, AP-12 EUR projekcia pred launchom) → **L40S / RTX 4090 / L4** (nízkonákladový fallback). Finálna voľba HW pre Phase 2+ až s kalibračnými tok/s. Community tier prípustný (AP-11); presné Community ceny až v deploy konzole.
4. **Seed VM `ssra-usage-seed` zmazaná** (Daniel, vlastný terminál). Voliteľný verify v najbližšej CC ops session: `gcloud compute instances list --project ssra-poc` → očakávaný prázdny výstup.
5. **`docs/cc/M2-assignment.md` → v1.1:** Environment odsek prepísaný na RunPod (fallback rebrík, ceny s dátumom, GCS SA, AP-10 void, „Phases 1+ unblocked"). Fázy, gates, AP-8…AP-16, configy nedotknuté.
6. Commit `c15c50c` (signed) pushnutý; working tree clean; main = origin/main.

## 2. Platný stav (skratka)

- M2 Phase 0 ✔ (2026-06-14); **Phase 1 (kalibrácia) je ďalší krok** — prvý platený GPU beh, ~1 h, STOP gate u Daniela pred Phase 2.
- Budget: 300 EUR kumulatívne; AP-12 gates (run > 25 EUR STOP; 50/80 % stropu STOP). RunPod účtuje USD per-second → EUR/USD kurz (ECB) zafixovať v deň nabitia kreditu, viesť v cost ledgeri.
- Nemenné: spec v1.2, gates G1, AP-8 matched params+tokens, seed 1337, configy §5 (S1 24M / S2 84M, ctx 1024, vocab 16k zabetónovaný), pool P1-only v M2, contingency flagy off.
- Docker image CUDA 12.4, `scripts/train.py`, checkpoint/resume (AP-11, bit-for-bit verifikované), data v GCS — všetko prenosné bez zmeny.

## 3. Ďalšia úloha = zadanie pre CC: „M2 RunPod launch flow + Phase 1 kalibrácia"

Nový chat vygeneruje CC zadanie (EN, `docs/cc/` — buď addendum k M2-assignment v1.1 alebo samostatný launch-flow doc; rozhodne sa tam). Musí pokryť:

1. **Key injection do podu** — kľúč `~/ssra-secrets/ssra-runpod-key.json` sa do podu dostane bezpečne (RunPod Secrets / env mechanizmus — overiť v docs.runpod.io v deň písania, Pravidlo W); NIKDY nie v Docker image, repe, YAML configu ani logu; v pode aktivácia cez `GOOGLE_APPLICATION_CREDENTIALS` alebo `gcloud auth activate-service-account`; sanity check = `gsutil ls gs://ssra-poc-ew3` pred akýmkoľvek tréningom.
2. **Pod launch checklist** — výber GPU podľa fallback rebríka (live dostupnosť + Community cena vs Secure v konzole), image (CUDA 12.4 cu124), network volume rozhodnutie (checkpointy idú do GCS ⇒ minimálny lokálny disk; pozor na storage billing stopnutých podov), región (EU preferovaný, nie tvrdá požiadavka — tréningové dáta = verejný FineWeb-Edu, 0 PII).
3. **Phase 1 kalibrácia per M2-assignment §4** — `pytest tests/` na boxe (fp32 correctness, spec §14.1–.3, .7), tok/s + peak VRAM pre S1 aj S2 kandidátov (oba modely), 1× GPU kill+resume, EUR projekcia Phases 2–4 z nameraných tok/s. Pôvodný AP-10 obsah (GCE vs Vertex) je void — nahrádza ho RunPod on-demand vs Community porovnanie zachytené v kalibračnom reporte.
4. **STOP gate:** po kalibrácii CC zastaví, report `results/M2-calibration.md`, Daniel potvrdí HW + plán pred Phase 2. Ukončenie podu ihneď po dobehnutí (per-second billing = shutdown disciplína je hlavná nákladová páka).
5. **Run disciplína:** kalibračný beh = tiež 1 YAML v `experiments/` commitnutý pred spustením + riadok v `results/runs.md`.

Pre-flight veto review zadania pred launchom CC (štandardný pattern).

## 4. Otvorené položky mimo tejto úlohy

- Revert záväzok: SA kľúč + org-policy enforcement po M2/M3 (D-log 2026-07-12).
- AP-13 single-seed S2 — Danielovo rozhodnutie pred Phase 3 (HO-07 §4).
- `p1_attn_entropy` logovať štandardne v M2 (M1 P-C nález).
- Zenodo erratum/v1.1 (retenčné pravidlo) — po zbere M1/M2 nálezov.
- Project files v Claude projekte: vymeniť kópie `00-stav-a-triaz.md`, `M2-assignment.md` + pridať tento HO-09.
