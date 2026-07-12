# HO-10 — M2 Phase 1 (kalibrácia) uzavretá; read-out blocker; vstup do design analýzy

**Dátum:** 2026-07-12 · **Autor:** Claude (Opus 4.8) + Daniel (verdikty) · **Predchádzajúci:** HO-09
**Vstupný bod nového chatu:** tento dokument + `docs/00-stav-a-triaz.md` (D-log riadky 2026-07-12) + `results/M2-calibration.md`

---

## 1. Stav jednou vetou

M2 Phase 1 kalibrácia na RunPod prebehla a je uzavretá (spend 3.04 EUR, 1.0 % stropu): **infraštruktúra je 100 % funkčná a odblokovaná**, ale centrálny nález je, že **G1 na S2 je so súčasnou SSRA implementáciou nevykonateľná** — read-out gather spôsobuje ≈32× throughput a ≈9× pamäťový hendikep voči flat, S2 OOM-uje na 80 GB pri každom batchi. Rozhodnutie (a): read-out optimalizácia s merateľným cieľom a stop-lossom; **ďalší krok = design analýza `src/ssra/model.py:114` vs spec Fenwick read-out — to je úloha chatu, ktorý číta tento dokument.**

## 2. Čo sa v tomto sedení stalo (chronologicky)

1. Zadanie `docs/cc/M2-runpod-launch.md` v1 vytvorené (AP-17 key injection, AP-18 launch/lifecycle checklist, AP-19 tier comparison) a odovzdané CC; commit `7d4bb78`.
2. CC lokálna príprava: `scripts/pod_bootstrap.sh`, 12 kalibračných YAMLov (S1/S2 × ssra/flat × b16/32/64), tok/s + VRAM logging v `scripts/train.py`; image deviácia (oficiálny RunPod image + bootstrap piny namiesto projektového Dockerfile) navrhnutá a akceptovaná runbookom.
3. Daniel: SSH kľúč, secret `gcp_ssra_runpod_sa` (base64 SA kľúča), kredit, deploy podu `ssra-m2-cal` (A100 PCIe 80 GB, Secure $1.39/hr, CA-MTL-3).
4. Exekúcia s incidentmi (všetky vyriešené a zaznamenané, report §8): start-command decode nezbehol → bootstrap `/proc/1/environ` fallback; **thread-thrash** (~2 h idle, 2.6 EUR — torch 252 vlákien vs 26 vCPU cgroup; fix OMP/MKL=26 zapečený do bootstrapu); torch pin korekcia **2.12.0+cu126** (cu124 index končí na 2.6.0); fla 0.5.0 × transformers 5.13.1 nekompatibilita (jediný červený test, Phase-4-only); AP-11 kill+resume našiel a opravil **GPU-only resume bug** (RNG ByteTensory musia ostať CPU; `124ee72`), potom bit-for-bit PASS.
5. Report `results/M2-calibration.md` commitnutý (finálne `1de8691`); pod terminovaný; fakturované **$3.4786 ≈ 3.04 EUR** (konzola autoritatívna; ≈25 % pod odhadom CC — delta na zosúladenie s timestampmi, bez backfill vysvetlenia).
6. D-log: Phase 1 closure + rozhodnutie §9.1 (a) + akceptované návrhy (transformers pin pred Phase 4; cu126 ratifikácia; RunPod image = standing launch path; AP-19 odklad).

## 3. Kľúčové čísla [OVERENÉ meraním, report §3/§5]

| | SSRA-P1 | flat | pomer |
|---|---|---|---|
| S1 (24M) b16 tok/s | 9 457 | 300 978 | ≈32× |
| S1 b16 peak VRAM | 54.7 GiB | 6.35 GiB | ≈9× |
| S1 b32/b64, S2 b16/32/64 | **OOM (80 GB)** | 129–342k tok/s, ≤40 GiB | — |
| S2 @ 1.7B tok projekcia | ≈253 EUR (nespustiteľné) | 4.1 EUR | — |

OOM lokalizácia: `src/ssra/model.py:114` — read-out gather materializuje B×h×N×cover×d_head tenzory, autograd ich drží. Hardvér neriešenie: H100 = tých istých 80 GB; 24 GB karty neudržia ani S1 b16. M1 G1a (slope 0.983 vs 1.923) meral asymptotický tvar — **platí ďalej**; kalibrácia odhalila konštantu.

**Aritmetická červená vlajka pre analýzu [HYPOTÉZA]:** 2.23 GiB alokácia @ S1 b32 (B=32, h=8, N=1024, d_head=24) implikuje efektívny cover ≈90–180 uzlov/token; teória hovorí ≈ log₂N + w-okno ≈ 40. Buď implementácia materializuje viac než musí (duplicity? padding na max cover? všetky vrstvy naraz?), alebo viac tenzorov súčasne (k aj v gather + medzivýsledky), alebo je konštanta architektúre vlastná. Toto je prvá otázka design analýzy.

## 4. Platné rozhodnutie (D-log 2026-07-12, veto režim)

**Možnosť (a) — read-out optimalizácia**, sekvencia:
1. **Design analýza** (Claude.ai, nový chat, zadarmo): rozbor `model.py:114` vs spec §Fenwick read-out (Variant A) + `docs/01` §7; kandidátne smery bez zmeny sémantiky: `torch.gather`/`index_select` namiesto advanced indexingu, chunkovanie po tokenoch/vrstvách, selektívna rekompuácia (checkpointing) gather segmentu, zlúčenie k/v gather. Výstup: „Zadanie pre CC — read-out optimization" s akceptačnými kritériami.
2. **CC implementácia + CPU testy** (zadarmo): spec §14 testy = rozhodca sémantiky (completion test atol 1e-4 fp32 musí ostať zelený); žiadna normatívna zmena spec.
3. **Re-kalibrácia** (~1–2 EUR, nový pod, standing launch path — secret aj bootstrap už existujú).

**Merateľný cieľ:** SSRA S2 beží na 80 GB aspoň @ b16 ∧ projekcia S2 @ 850M tokenov ≤ 25 EUR (≈5× zrýchlenie; plný 1.7B budget ⇒ ≈10×).
**Stop-loss:** 1 implementačná iterácia + 1 re-kalibrácia; pri nesplnení fallback (b) symetrický scale-down G1 na S1-triedu (flat baseline sa nikdy nedropuje) + (c) poctivé reportovanie konštanty. Konštanta sa reportuje v paperi vždy, aj po úspešnom fixe.
**Phase 2 sa nespúšťa** pred verdiktom re-kalibrácie.

## 5. Infra stav (odblokované, standing)

- **Launch path:** RunPod oficiálny image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` + `scripts/pod_bootstrap.sh` (self-servuje kľúč z `/proc/1/environ`, thread limity, piny, dáta) — start-command decode už netreba. Secret `gcp_ssra_runpod_sa` existuje v RunPod účte.
- **Piny:** torch **2.12.0+cu126** (ratifikované; `requirements-gpu.txt`/Dockerfile komentáre treba updatnúť — otvorená položka); transformers pin kompatibilný s fla 0.5.0 **pred Phase 4**.
- **AP-11 GPU-verified** bit-for-bit (po `124ee72`). AP-17 hygiena držala (žiadny secret nikde).
- **Ekonomika:** spend celkom ≈3.04 EUR / 300; A100 PCIe $1.39/hr Secure je správna trieda pre flat aj SSRA-S1 prácu; AP-19 Community porovnanie = pre-flight ďalšieho launchu (cena „not captured" 12.7., žiadny backfill).
- Na Danielovom RunPod účte beží aj **cudzí projekt** — kredit je zdieľaný (dimenzovať top-upy na oba), pody rozlišovať menom (`ssra-*`).

## 6. Otvorené položky

| # | Položka | Kedy |
|---|---|---|
| 1 | **Design analýza read-outu → zadanie pre CC** | hneď (nový chat) |
| 2 | Ledger delta $4.6 odhad vs $3.4786 konzola — CC zosúladiť s timestampmi | pri najbližšom CC kontakte |
| 3 | requirements-gpu.txt + Dockerfile komentáre → cu126 | s CC zadaním #1 alebo housekeeping |
| 4 | transformers pin (fla 0.5.0 kompatibilný) | pred Phase 4 |
| 5 | AP-19 Community cena | pre-flight ďalšieho launchu |
| 6 | ⚠ Revert záväzok: zmazať SA kľúč + vrátiť org-policy enforcement | po M2/M3 |
| 7 | Zenodo erratum/v1.1 (staré retenčné pravidlo; + prípadný read-out constant nález) | po zozbieraní M2 nálezov |
| 8 | `p1_attn_entropy` ~ uniform — sledovať v M2 tréningoch | Phase 2+ |
| 9 | Project files v Claude projekte vymeniť (00, M2-calibration report, M2-runpod-launch, HO-10) | po commite tohto HO |

## 7. Čoho sa nedotýkať

Uzavreté rozhodnutia (D1–D6, Q1–Q5, MD-1–MD-13, AP-1–AP-19) sa neotvárajú. Spec v1.2 je normatívna; read-out fix musí byť implementačný (sémantika = spec §14 testy), akákoľvek normatívna zmena spec by bola nové rozhodnutie pre Daniela. Žiadne kvalitatívne závery z kalibračných loss hodnôt (spec §16). G1 kritérium (±5 % val ppl) sa nemení bez D-log zápisu.
