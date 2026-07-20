# HO-29 — V11: K1 okno + ckpt delete uzavreté (oversight PASS), K1-ANALÝZA pending

Dátum: 2026-07-20 (session 2, dopoludnia). Nadväzuje: HO-28. Detailný záznam: D-log 2026-07-20 (riadky 2–3).

## 1. Stav po tomto sedení

- **Commity (lokálne v čase písania; push = close batch tohto sedenia):** `e2a5f8c` (K1 Phase A: extrakčný skript + manifesty + runbook + IAM pre-flight), `98fe162` (oversight fixy F1 alias groups / F1b init persistencia / F2 trainable), `1ee168b` (VM okno: report §K1 + §Deviations + §Ledger, NPZ, log, D-log batch), `8bbc578` (N3 fixup: RNG hypotéza relabel na „open re-check item" + rozšírenie na 3 kandidátov). Po pushi: G-V11-4 re-verifikácia z GitHub file listov všetkých commitov batchu.
- **VM okno:** create 10:23:45 → delete 10:43:15 CEST = **0:19:30 ≤ 4,0 h (G-V11-3 PASS)**; 104/104 objektov z explicitných manifestov; ~6–13 s/ckpt; abort pravidlo nikdy aktívne; `weights_only=True` nikdy uvoľnené; do bucketa nič nezapísané. Exekútorský amendment (D-log): D3+D4 = CC. 5 operačných deviácií (pip index, pyyaml, ssh keygen, lingering ssh channel, init-drop vetva) — report §Deviations.
- **Oversight PASS §K1 (extrakcia), nezávisle:** flat NPZ plný prepočet na Claude strane (všetky polia a čísla zhodné); ssra NPZ cez snippet exekuovaný Danielom (bytes, 130 kľúčov, P=393, alias aritmetika 60×3 → trainable 273, drift 1,1359 @ emb.weight — všetko zhodné). Navyše **T-C predbežne overené: riadky 11–15 všetkých 15 e_ℓ tabuliek = presne 0,0 vo všetkých 52 ckpt aj v inite.**
- **Init validácia DROPPED v oboch ramenách** (max rel drift 1,1359 / 1,1354, driver `emb.weight` ≈ independent-draw signatúra) ⇒ K1-analýza beží výhradne na S_min = step-1000 referencii; delta/cos-init stĺpce gated; príčina = open re-check item (N3: pod-side init path vs Phase A čítanie train.py / RNG konzumpcia pred konštrukciou v pod prostredí / build-level rozdiel RNG — bez výberu víťaza).
- **Ckpt delete VYKONANÉ (Daniel, po oversight PASS, AP-25 vzor):** pre-delete listing 53 + 54 = **107 = očakávanie** (106 + stray); `rm -r` oba prefixy Completed 53/53 + 54/54 (stray `latest.pt/latest.pt` v ssra dávke viditeľne odstránený); opakovaný beh „matched no objects" na oboch prefixoch = dôkaz prázdnoty. **R1 dostal exekučnú formu a je splnený pre core pair**; bucket ponecháva g2lite + data prefixy do teardownu ≤ 2026-08-31. NPZ extrakty (52 855 593 B + 293 600 B) sú odteraz jediné trvalé trajektóriové dáta — commitnuté v repe.
- **Ledger:** K1 okno ≈ **0,06 EUR [ODHAD]** (0,325 h VM + disk + 0,05 GiB egress, S0-C SKUs); kumulatív [ODHAD] **72,43 EUR ≈ 24,1 %**. Console-authoritative korekcia pending (≥ 12:43; append-only riadok, ak sa líši).

## 2. Pending — K1-ANALÝZA (jediný zostávajúci krok V11; 0 EUR, lokálne)

Vstupy: `results/v11/v11-k1-extract-{ssra,flat}.npz`. Pre-registrované zadanie: `docs/cc/V11-data-exploitation.md` §5 (metriky T-A…T-E, kritérium C-T1 pre H-T1). Záväzné pre analýzu:

1. **Dedup podľa `meta_json.alias_groups`** (60 skupín × 3 mená v ssra; kanonický = prvý v sorted poradí) — C-T1 medián sa počíta nad tenzormi, nie nad kľúčmi.
2. **Filter „trainable parameter tensors" z `meta_json.trainable`** (ssra 273, flat 183).
3. **Referencia výhradne S_min** (init DROPPED; `init_l2` + `full_init/*` v NPZ slúžia len na dokumentáciu drop verdiktu, nie ako referencia).
4. Výstupy: figúry pod `results/figures/v11/` (prefix `v11-k1-`), skript(y) `scripts/v11_k1_*` headless, sekcia **§K1-analýza** v reporte (T-A, T-B + C-T1 verdikt, T-C formálne, T-D s lr overlay z committed configov/logov, T-E flat kontrola) → oversight → V11 close.

## 3. Standing (nezmenené + doplnky)

- Nota v1.1: GO, bez termínu (checklist HO-27 §2).
- #2 arXiv endorsement open.
- Teardown ≤ 2026-08-31, checklist doplnený: SA `ssra-runpod` final delete; bucket legacy IAM cleanup; GCS zvyšky (g2lite, data); **lokálny `~/.ssh/google_compute_engine` keypair + pubkey v projekt metadátach** (K1 deviácia 3).
- Billing korekcia K1 okna po ≥ 12:43 (append-only).

## 4. Otvárací prompt pre nový chat (paste-ready)

```
K1-analyza (V11, 0 EUR, lokalne). Cez Filesystem MCP precitaj docs/handover/HO-29-2026-07-20-v11-k1-okno-a-delete-uzavrete-k1-analyza-pending.md a docs/00-stav-a-triaz.md (tail 3), over git log -1 (<HASH>) a zhodu remote HEAD == local HEAD. Kontext: K1 extrakcia + ckpt delete uzavrete (oversight PASS, D-log 2026-07-20), init DROPPED oba ramena -> vyhradne S_min referencia, NPZ v results/v11/ (ssra 130 kluccov / P=393 / alias_groups 60 / trainable 273; flat 10 / 183 / 0 / 183). Zavazne: dedup podla alias_groups, filter trainable, C-T1 nad tenzormi. Dnes: CC zadanie na T-A..T-E figury + C-T1 verdikt -> report §K1-analyza -> oversight -> V11 close.
```

(`<HASH>` doplň po commite + pushi close batchu.)
