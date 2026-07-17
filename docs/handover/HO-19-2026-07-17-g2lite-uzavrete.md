# HO-19 — 2026-07-17 — M2 G2-lite uzavreté: H1 nepodporená; vstup pre oversight review + rozhodnutie o tvare paperu

**Účel:** kontinuita do nového chatu = oversight review G2-lite reportu + interpretácia O1–O7 + Danielovo rozhodnutie o tvare paperu. Autoritatívny stav: `docs/00-stav-a-triaz.md` (D-log 2026-07-17, **tretí** zápis). Tento HO nič nerozhoduje — sumarizuje a odovzdáva.

## §1 Výsledok v jednej vete

G2-lite vykonané **presne per pre-registrovaný protokol** (`docs/cc/M2-g2lite.md`, commitnutý pred spustením; 0 zakázaných operácií, 0 zásahov do model kódu, deviácie D1/D2 disclosed pred meraním): M0 kotva exaktne zreplikovala G1 čísla (Δ 0,0 / −1e-5) ⇒ meraná bola presne trénovaná funkcia; **M1: oba modely za tréningovou dĺžkou degradujú, SSRA rádovo tvrdšie (collapse od 4 096), žiadny crossover — flat prior potvrdený, SSRA prior porušený; M2 needle: flat 60 % @ 1 024 inak 0, SSRA 0 % všade vrátane vlastnej tréningovej dĺžky** ⇒ **H1 na tejto škále nepodporená; formulácia B vyčerpaná**. Fakturácia FINÁLNA **0,59 EUR** (~17× pod scoped cap 10 EUR); **kumulatív M2 = 72,37 EUR ≈ 24,1 % z 300**.

## §2 Kľúčové fakty

| položka | hodnota |
|---|---|
| M0 kotva | PASS oba: flat 3,19333 (Δ exaktne 0,0), SSRA 3,29064 (Δ −1e-5); window identita 1 953/1 999 872/127; `final_eval` importovaný z `train.py` = identický code path konštrukciou |
| M1 ppl(N), flat | 23,68 → 37,13 → 96,07 → 224,4 → 443,8 → 775,4 · r(N) 1 → 1,57 → 4,06 → 9,47 → 18,7 → 32,7 |
| M1 ppl(N), SSRA | 26,31 → 108,2 → 1 154,9 → 4 179,0 → 9 476,2 → 14 778,0 · r(N) 1 → 4,11 → 43,9 → 158,9 → 360,2 → 561,8 |
| O-čítania (mechanické) | O4 žiadny crossover (pomer SSRA/flat 1,111 → 19,1–21,4); flat prior (Press et al.) POTVRDENÝ, SSRA stable/mild prior PORUŠENÝ od 2 048; O6 ani jeden positionally graceful (poškodenie výhradne na pozíciách > 1 024); O5 floor rule NEaktivovaný (flat pooled 60 % @ 1 024); O7 pri každej SSRA bunke N ≥ 2 048 (e_ℓ riadky 11–15 exaktne 0,0 — ablation-OFF stav úrovní 11–15, vlastnosť dizajnu 1k tréningu, zapísaná vopred) |
| M2 needle | flat @ 1 024 = 0,00/0,95/0,85 (hĺbky 0,1/0,5/0,9), 0 % za 1 024; SSRA 0 % vo všetkých bunkách vrátane 1 024 (pre-registrovaná veta §4: 85M @ 850M tok nemusí mať copy behavior — validný výsledok) |
| Deviácie | D1 = fp64 akumulácia M1 súčtov (mimo model forwardu); D2 = pre-flight batch cap `e305ad0` (B·N/2 ≤ 32 768, CUDA kernel-launch limit; wall-clock-only, batch-invariancia test-certifikovaná; povýšená z „note“ commitom `71918bd`) — obe disclosed pred meraním |
| V2b (do Limitations) | bf16 kvantum pozícií = 2^(⌊log₂ t⌋−7) (erratum zadania −8, vlastníctvo Claude; tabuľka governing); CUDA angle tensor fp32, pozície aj tak bf16-kvantované castom; @ 32k celé read-out okno [32704..32768] → jediná hodnota 32768,0; od N ≈ 8k ULP ≥ 64 ≈ w ⇒ pozičná diskriminácia v okne zničená — symetrický artefakt oboch modelov, kód nemenený |
| Pod + náklady | `pktqlt4jys3uiz` (ssra-m2-g2lite), RTX A6000 48 GB Secure, **EU-SE-1 (prvýkrát EU preferencia splnená)**, $0,50/hr total; billed **$0,6707691364 ≈ 0,59 EUR FINAL** (krížový test Δ < 1 s); AP-23 terminate 16:44:59Z, idle tail ≈ 11 s; AP-19 krok 0: not shown (7. výskyt) |
| Artefakty | `results/M2-g2lite.md` (§S/§M0/§M1/§M2/§O/§L + §Deviations); raw JSON/CSV + ploty; GCS mirror `gs://ssra-poc-ew3/m2/g2lite/` (12 objektov, listing verified); commity `f8a988c`/`5bcbe39`/`e305ad0`/`1b3b386`/`71918bd` + close-out (ledger + D-log + tento HO) |

## §3 Úlohy nového chatu

1. **Oversight review `results/M2-g2lite.md`:** nezávislý prepočet M0/M1/M2 z raw JSON/CSV artefaktov (repo `results/g2lite/` alebo GCS mirror) — ppl a r(N) hodnoty, bucket NLL, needle exact-match počty; konzistencia §Deviations (práve D1+D2, nič tiché); O1–O7 mechanická aplikácia proti §4 zadania; korekcie výhradne append-only.
2. **Interpretačný podklad pre paper** (žiadne architektonické závery, spec §16): čo čísla mechanicky hovoria o H1/H2; V2b + O7 ako pre-registrované limitácie; formulácia „flat prior confirmed; SSRA prior violated" (nie „both violated" — opravené in-flight).
3. **Danielovo rozhodnutie o tvare paperu:** R3 invarianty (D-log 2026-07-17, druhý zápis) + G2-lite výsledky; varianty na stôl: čistý negative-results report vs + diagnostická sekcia (P-C uniformita, e_ℓ ablation-OFF, V2b) vs rozsah budúcej práce; potom M4 draft plán.
4. **Standing položky:** R1 retencia step-tagged ckptov — rozhodnúť pri zamrazení draftu, najneskôr 2026-08-31; R2(c) GCS-mirror precondition povinná pred akýmkoľvek budúcim behom > 4 h bez dozoru; AP-19 pozastavené do znovuobjavenia Community tieru; revert záväzok GCP SA kľúč + org-policy override po M2/M3.

## §4 Ohraničenia (binding, nezmenené)

Spec v1.2 frozen; G1 = FAIL (×2) zapísaný a neotvára sa; **formulácia B vyčerpaná — žiadne ďalšie meranie ani spend bez nového Danielovho rozhodnutia**; žiadne architektonické závery z ppl/needle čísel (spec §16); Pravidlo W na všetko mimo kontextu; ledger append-only; uzavreté D-logy sa nere-litigujú.

## §5 Lekcie tejto session (pre budúce zadania)

1. V2b vzorec −8: parenteticky vzorec v pre-registrovanom texte mal byť buď odvodený, alebo vynechaný — empirická tabuľka bola správne governing (vlastníctvo: Claude).
2. Monitor formulácia „both priors violated" — prior-labely čítať mechanicky proti §4 zneniu, nie parafrázovať.
3. Deviácia ≠ „note": každá zmena pre-registrovaného obsahu (vrátane YAML batch tabuliek) ide do §Deviations, aj keď je benígna.
4. Dead-peer ssh: launcher session bez ServerAlive keepalive visí po self-terminate podu donekonečna — dlhé remote pipelines púšťať detached (`nohup … & + exit`) alebo so `ServerAliveInterval`.

## §6 Otvárací prompt pre nový chat (copy-paste)

```
Vstup: docs/handover/HO-19-2026-07-17-g2lite-uzavrete.md + docs/00
(D-log 2026-07-17, treti zapis) + docs/cc/M2-g2lite.md (§4 kriteria
O1-O7) + results/M2-g2lite.md + results/runs.md (tail).
Kontext: G2-lite vykonane presne per pre-registrovany protokol; M0 kotva
PASS (Δ 0,0 / -1e-5); M1 r(N): flat 1->32,7, SSRA 1->561,8, ziadny
crossover; needle: flat 60% @1k inak 0, SSRA 0 vsade; H1 na tejto skale
nepodporena; formulacia B vycerpana; kumulativ 72,37 EUR (24,1 %).
Uloha: (1) oversight review results/M2-g2lite.md — nezavisly prepocet
M0/M1/M2 z raw JSON/CSV (repo results/g2lite/ alebo GCS mirror
m2/g2lite/), konzistencia §Deviations (prave D1+D2), O1-O7 mechanicky;
korekcie append-only; (2) interpretacny podklad pre paper bez
architektonickych zaverov (spec §16); (3) podklad pre Danielovo
rozhodnutie o tvare paperu (R3 invarianty + G2-lite; varianty: cisty
negative-results report vs + diagnosticka sekcia) a plan M4 draftu;
(4) standing: R1 ckpt retencia (deadline 2026-08-31), AP-19 suspendovane,
GCP SA revert po M2/M3. Ziadny novy spend bez noveho rozhodnutia.
```
