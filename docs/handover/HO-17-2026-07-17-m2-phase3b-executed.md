# HO-17 — 2026-07-17 — M2 Phase 3b vykonaná (retune @ lr 6e-4); G1 verdikt PENDING

**Účel:** kontinuita do nového chatu = oversight review reportu Phase 3b + G1 verdikt (Daniel) + follow-up rozhodnutia. Autoritatívny stav: `docs/00-stav-a-triaz.md` (D-log 2026-07-17). Tento HO nič nerozhoduje — sumarizuje a odovzdáva.

## §1 Výsledok v jednej vete

Jediná povolená retune iterácia (symetrický pár @ lr 6e-4, seed 1337, jediná zmenená premenná = lr) prebehla **čisto na oboch ramenách** — Phase 3 spike sa nezopakoval; kvalitatívna medzera SSRA vs flat je **+10,22 %**, mimo pre-registrovaného ±5 % pásma; **G1 verdikt je otvorený a patrí Danielovi**.

## §2 Kľúčové fakty

| položka | hodnota |
|---|---|
| G1 vstupy (`val-eval-2M`, ctx 1024) | flat 3,19333 (ppl 24,369) · SSRA 3,29065 (ppl 26,860) · medzera +10,22 % |
| Eval protokol | byte-identický: 1 953 okien / 1 999 872 tok / 127 dropped, obe ramená |
| Stabilita | oba behy čisté 51 880/51 880 krokov; 0 divergence flagov; AP-24 counter nikdy > 0 (261 evalov/rameno, max val regresia 0,00334 natu); žiadny NaN/inf |
| Pre-registrovaná inferencia (zadanie §1) | zmiznutie spiku pri izolácii jedinej premennej ⇒ lr implikované ako príčina Phase 3 nestability (pozorovanie; mechanizmus neurčený) |
| P-C | `p1_attn_entropy` ≈ ln(32) celý beh (3,4657→3,4348, min 3,4287); participácia [0,047; 0,100] |
| Pod | `ne2w6airwb4401`, A100 SXM 80 GB Secure, US región (deviácia, prípustná per AP-18), $1,50/hr, 60 GB disk |
| AP-23 | prvá invokácia úspešná (PID-1 env v terminate príkaze — lekcia Phase 3), nulový idle tail |
| Náklady [ODHAD] | SSRA ≈ 26,90 EUR ≤ 30 scoped cap ✓ · flat ≈ 3,84 · session ≈ 32,1 · **kumulatív M2 ≈ 72,6 EUR ≈ 24 %** — konzolový odpočet FINÁLNY ≥ 2 h po terminácii 08:45 UTC, doplniť append riadkom |
| Artefakty | main `8fa4041`: `results/M2-core-pair-lr6e4.md` §0–§x, logy s grad_norm timeline, ploty (repo + GCS), runs.md; 53 step-tagged ckptov/rameno v GCS (~98 GiB ≈ 2 EUR/mes [ODHAD]) |

## §3 Prevádzkové nálezy (do §ix reportu už zapísané CC)

1. **US pod → EU bucket:** step-tagged uploady ~70 s/interval (~29 MB/s) ⇒ +~1 h wall/rameno ≈ +2,6 EUR/pár [ODHAD]; EU preferencia má kvantifikovanú hodnotu.
2. **AP-24 terminate tail nie je pod-autonómny:** pod bez git credentials nesplní push precondition AP-23 ⇒ pri fire tréning stopne (nákladový účel splnený), ale pod idluje do ľudského zásahu. Vlastníctvo mischarakterizácie „plne autonómne": Claude.
3. AP-19 krok 0: Community not shown in deploy flow (6. výskyt).
4. Konzolový vCPU nesúlad (16 listing / 24 detail); throughput nedotknutý (12 383 tok/s ≈ kotva +0,4 %).

## §4 Otvorené položky pre nový chat

1. **Oversight review** `results/M2-core-pair-lr6e4.md`: nezávislé prepočty G1 vstupov, ledger, tok/s, grad_norm/P-C konzistencia, AP-24 counter tvrdenia z raw logov.
2. **G1 verdikt (Daniel)** na čerstvom páre — mechanické čítanie pre-registrovaných kritérií: stabilita splnená, ±5 % pásmo nesplnené; interpretácia a verdikt = Daniel; žiadne architektonické závery z medzery (spec §16).
3. **Retencia step-tagged ckptov** (~98 GiB ≈ 2 EUR/mes): držať pre M3/trajektóriovú analýzu vs delete.
4. **AP-24 terminate tail:** akceptovať dokumentovaný fallback vs scoped push credential (napr. fine-grained PAT ako RunPod secret) pred ďalšími dlhými behmi.
5. **Fakturačný odpočet** do ledgera (append riadok). Konzolový údaj odčítaný 2026-07-17: **$35,7192305624485 ≈ 31,25 EUR** (ECB 1,1430); krížovo sedí s wall-clock 23,81 h × $1,50 na minútu presne (start 08:56 UTC 16.7. → terminate 08:45 UTC 17.7.); FINÁLNY, ak odčítaný ≥ 10:45 UTC (≥ 2 h po terminácii, pravidlo 2026-07-14) — inak potvrdiť re-checkom. Kumulatív M2 po zápise ≈ 71,8 EUR ≈ 23,9 %.
6. **Ďalší krok per `03`:** dosah verdiktu na fallback (b)/(c), rozsah paper reportingu (vrátane nálezu „užší stabilný lr rozsah"), prípadný vstup do M3 plánovania.

## §5 Ohraničenia (binding, nezmenené)

Spec v1.2 frozen; žiadne architektonické závery z ppl (spec §16); uzavreté rozhodnutia sa neotvárajú; retune bola JEDINÁ povolená iterácia — žiadny tretí pár; Pravidlo W na všetko mimo kontextu; ledger append-only.

## §6 Otvárací prompt pre nový chat (copy-paste)

```
Vstup: docs/handover/HO-17-2026-07-17-m2-phase3b-executed.md + docs/00
(D-log 2026-07-17) + results/M2-core-pair-lr6e4.md (cely) +
docs/cc/M2-phase3b-retune.md (§1, §6, §7).
Kontext: Phase 3b vykonana; obe ramena stabilne end-to-end (spike sa
nezopakoval); medzera +10,22 % mimo pre-registrovaneho ±5 % pasma;
G1 verdikt PENDING.
Ulohy: (1) oversight review reportu — nezavisle prepocty G1 vstupov,
ledgera, tok/s a AP-24/grad_norm tvrdeni z raw logov; (2) podklad pre
G1 verdikt: mechanicke citanie pre-registrovanych kriterii + tradeoffy
formulacie verdiktu, verdikt je moj; (3) po verdikte D-log zapis +
rozhodnutia: retencia step-tagged ckptov, AP-24 terminate tail
(fallback vs scoped push credential), dalsi krok per 03 a rozsah
paper reportingu — navrhy s tradeoffmi; (4) fakturacny odpocet append
do ledgera. Ziadne architektonicke zavery z ppl (spec §16).
```
