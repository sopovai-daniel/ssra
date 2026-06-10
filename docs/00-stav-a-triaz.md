# 00 — Stav projektu a epistemická triáž
**Aktualizované:** 2026-06-10 | **Tento súbor je jediný zdroj pravdy o stave projektu.**

## Identita
- **DFKS** — interný kódový názov. **SSRA (Scale-Shared Recursive Attention)** — pracovný technický názov pre publikáciu (finálny TBD; vyhnúť sa kolízii s „Fractal Generative Models", Li et al. 2025).
- Autor myšlienky: Daniel Sopov. Asistencia: Gemini (exploračná fáza, jún 2026), Claude (triáž, formalizácia, PoC).
- Fáza / TRL: **M0 dizajn uzavretý (D1–D6, Q1–Q5)**. Žiadny tréning neprebehol. Žiadne meranie neexistuje. Ďalej: novelty téza v1 → `spec.md` → Gate G0 → Zenodo DOI (stupeň 1 publikácie).
- **Domov projektu:** toto repo (`ssra`, private GitHub do publikácie). Dizajn/analýza/písanie = Claude.ai projekt; implementácia = Claude Code v tomto repe.

## [OVERENÉ]
- Existuje v1.0 PyTorch skript: `scripts/v1_legacy.py` (~120 riadkov, MPS target s CPU fallbackom). Obsahuje rekurzívny blok `DFKS_Core`, **plnú tréningovú slučku** (AdamW, CE loss, grad clipping, 5 epoch, dummy korpus 1000 náhodných tokenov, seq_len 8, vocab 100) a nevyužitú funkciu `generate_next_token`. **Korekcia 2026-06-10 po inšpekcii kódu:** pôvodný popis „~80 riadkov, inference-only, očakávaný výstup náhodné slovo" bol nepresný. Beh zatiaľ nepotvrdený logom → úloha T0; očakávaný výstup = 5 riadkov klesajúceho loss. **Pozor na interpretáciu:** pokles loss je artefakt target leakage (bez kauzálnej masky pozícia t vidí svoj target x[t+1] vo vstupe pre 7 z 8 pozícií), nie dôkaz učenia — priamy motivátor diery D1.
- Mechanizmus v kóde v1.0: rekurzívne pozičné polenie sekvencie (midpoint, threshold=2), **jeden zdieľaný {Q,K,V} softmax attention blok na každej úrovni**, concat pozdĺž sekvenčnej osi, plná attention nad celou spojenou sekvenciou, aditívny šum 0.01 pri každom návrate. Bez kauzálnej masky, bez pozičného kódovania, bez FFN/residual/LN, jedna vrstva.
- Odvodená zložitosť v1.0: `cost(N) = 2·cost(N/2) + c·N²` ⇒ **Θ(N²), konkrétne ≈ 2cN²** — horšie než flat Transformer; navyše O(log N) sekvenčných závislých krokov. Claim O(N log N)/O(N) je pre v1.0 **falzifikovaný**. (Odvodenie: `01-mechanizmus.md` §3.)
- SSRA v2 up-pass pri pass-through schedule s_ℓ = min(2^ℓ, m): **Θ(N·m·d)** score-ops na vrstvu; pôvodný bound O(N·m²) v D3 bol voľný. (Odvodenie: `01-mechanizmus.md` §7.)
- SSRA v2 celkový tréning (up-pass + Fenwick read-out): **Θ(N·(w + m·log N)·d)** na vrstvu — rovnaká trieda ako Log-Linear Attention (#2), nie lepšia. Decode stav O(m·d·log N) na vrstvu. (Odvodenie: `01-mechanizmus.md` §7.)
- Klasifikačné osi v2: hierarchia **pozičná**, tok **bottom-up + priamy Fenwick read-out** (bez down-pass kaskády), zdieľanie pravidla naprieč škálami **áno** (θ, φ, vrátane read-outu).

## [HYPOTÉZA] — jadro projektu
- **H1:** Zdieľanie váh naprieč škálami je induktívny bias zodpovedajúci samopodobnosti jazyka (Hurst ≈ 0.7, Alabdulmohsin et al. 2024) → lepšia length-extrapolation pri rovnakom parameter budgete než flat baseline.
- **H2:** Kompresia v uzloch (m súhrnných vektorov, P1/P3 operátory) zachová retrieval schopnosť merateľnú needle-in-haystack testom pri zložitosti Θ(N·(w + m·log N)·d).
- **H3:** Top-down prechod (makro→mikro) zlepší konzistenciu dlhých generácií. **Presunuté do follow-up vetvy SSRA-TD (variant B, po Gate G2); v PoC sa netestuje — metriky konzistencie sú mäkké. V paperi: Limitations.**
- **P-A až P-E:** testovateľné predikcie pre M3 (needle vs vzdialenosť, multi-needle cliff, collapse diagnostika, A vs B retrieval, konzistencia) — definície v `01-mechanizmus.md` §8.

## [FIKCIA] — vymazané; nikdy necitovať ako fakt, nikdy v externej komunikácii
- „Funkčná softvérová implementácia s príkonom 20 W" — 20 W je metafora ľudského mozgu; nič nebolo merané.
- Protokol „Assassin" §3.1 (numerická erózia / stochastická rezonancia), §3.2 (pasca pevného bodu / w_i(t)), §3.4 (Zénonov paradox / α(t) ľudský stres) — naratív; premenné α(t), w_i(t), γ ani Lyapunov monitoring v systéme neexistujú. §3.3 bol reálny triviálny bug pri concat osi (opravený voľbou axis=1).
- Sekcia „matematický model" (IFS, Hausdorffova dimenzia): rovnice v pôvodnom dokumente fyzicky chýbajú; framing nemá operačné napojenie na kód. Status: dekorácia, nie model. Smie sa vrátiť len ako odvodená, do kódu napojená matematika.
- Pôvodný Gemini dokument v1.0 sa do project files ani do tohto repa nevkladá (kontaminácia kontextu); archív lokálne.

## Rozhodnutia (D-log)
| Dátum | Rozhodnutie |
|---|---|
| 2026-06-09 | Tréning na dátach v nastaveniach Claude účtu vypnutý; žiadny thumbs feedback v projekte. |
| 2026-06-09 | Publikačná stratégia: defensive publication — arXiv (overiť endorsement) alebo Zenodo DOI + Apache-2.0 kód; minimalizovať čas do publikácie. |
| 2026-06-09 | Cieľová definícia úspechu: jasná publikovateľná odpoveď na H1–H2 (pozitívna ALEBO negatívna) pri matched compute. |
| 2026-06-09 | Implementácia v lokálnom git repe cez Claude Code; Claude.ai projekt = dizajn/analýza/písanie. |
| 2026-06-09 | **D3 uzavreté:** Pool_φ default **P1** (latent-query/PMA), challenger **P3** (top-m select + ctx residual, micro-gate G1b-D3), control **P2** (strided merge). |
| 2026-06-09 | **D3 schedule:** m = 16 fixné (pass-through s_ℓ = min(2^ℓ, m)) ⇒ up-pass Θ(N·m·d); ablácia m_ℓ = m₀ + g·ℓ; schedule b·2^(ℓ/2) zamietnutý (konflikt s decode cieľom O(log N) stavu). |
| 2026-06-09 | Otvorená otázka §6.4 (residual/LN) povýšená: **pre-norm residual + LN v každom uzle je tvrdá požiadavka v2 kostry** (bez identity cesty log N prepisov ničí verbatim prenos aj gradient flow). |
| 2026-06-10 | **D4 uzavreté:** výstupná cesta = **variant A** (Fenwick read-out, okno w=64, read-out zdieľa θ s uzlovými blokmi; ablácia: oddelené ψ; ablácia w ∈ {32,128}). Variant B → follow-up vetva SSRA-TD po G2, v paperi Limitations. Completion test (inkrementálne ≡ plný forward) pridaný do M1 suity. |
| 2026-06-10 | **D1 uzavreté:** kauzalita = pásová kauzálna maska v lokálnom okne + štrukturálne hradlovanie súhrnov (uzol konzumovateľný až po konci svojho spanu) + bidirekčná attention VNÚTRI uzla povolená (legálna, expresívny zisk). Verifikácia: shift test + completion test. |
| 2026-06-10 | **D2 uzavreté:** pozícia = relatívny slot-RoPE vnútri uzla (rovnaká geometria na každej škále, sloty 1..2m) + level embedding e_ℓ ako vstupná súradnica; read-out kľúče heterogénne (okno: RoPE, súhrny: NoPE + level emb ≈ log-vzdialenostný kód). Ablácia: level emb OFF (čistá škálová ekvivariancia). |
| 2026-06-10 | **D5 uzavreté:** aditívny šum v1.0 odstránený úplne (inferencia aj dizajn); regularizácia = štandardný dropout len v tréningu, sila sa ladí v M2. |
| 2026-06-10 | **Q2 uzavreté:** k=2 default (štruktúrna zhoda s Log-Linear ⇒ čistá izolácia mechanizmu), k=4 ablácia. |
| 2026-06-10 | **Q4 uzavreté:** SSRA = attention-sublayer v štandardnom pre-norm stacku; θ_i, φ_i per-layer, zdieľané naprieč škálami v rámci vrstvy; FFN v uzle NIE (ablácia pri kapacitnom probléme); zdieľanie naprieč vrstvami = ablácia po G2. |
| 2026-06-10 | **Q5 uzavreté** cez D4: štrukturálne Fenwick rodina ako #2 (priznať v paperi), mechanizmus odlišný (softmax + učená Pool_φ + zdieľanie θ,φ naprieč škálami). Empirický dôkaz odlíšenia = needle/recall pri matched compute. |
| 2026-06-10 | Q3 (obsahová segmentácia, os C) **odložené** — v2 zostáva pozičná; os C = samostatný follow-up, nevťahovať pred G1. |
| 2026-06-10 | Inovačná stratégia potvrdená Danielom: maximálna os A v jadre (jedno pravidlo všade: uzly, škály, read-out; bidirekčná intra-uzlová attention), každá ďalšia odvážna voľba ako ablácia s predikciou. |
| 2026-06-10 | **Repo:** samostatné private GitHub repo `ssra` (lokálne `/Users/ds/Developer/ssra`); docs presunuté zo `sopovaidoc/DFKS` (rozhodnutie A1 — všetko v jednom repe). História písaná ako publikovateľná od prvého commitu: žiadne secrets, žiadne klientske referencie, signed commits, commit messages anglicky. Private repo nie je dôkaz priority — prioritu fixuje až verejný DOI. |
| 2026-06-10 | **Publikačná stratégia rozšírená na dvojstupňovú (rozhodnutie B2):** stupeň 1 = po Gate G0 Zenodo DOI technical note (spec + complexity analýza + novelty téza) — fixácia priority myšlienky (~koniec júna); stupeň 2 = po Gate G2 plný paper s výsledkami (arXiv/Zenodo, ~september). Vedomý trade-off: nápad verejný skôr. |

## Otvorené úlohy (top)
- ✔ **T0 hotové (2026-06-10):** beh zaznamenaný v `logs/T0-v1-sanity.log` (commit 7a9800d). Loss 4.476 → 2.884 za 5 epoch na náhodnom korpuse = empirické potvrdenie target leakage (na uniformne náhodných dátach niet čo učiť; štartovací loss ≈ ln(100) = 4.605 sedí s teóriou). Podotázka verzie zavretá: `v1_legacy.py` je jediná existujúca verzia — prvý Gemini návrh nebežal, opravený stav = tento súbor; Gemini archív ostáva lokálne mimo repa (FIKCIA pravidlo).
- **T1 (= M0, zostávajúce):** novelty téza v1 → `docs/spec.md` → **Gate G0 check** → Zenodo technical note (stupeň 1).
- **T2:** Overiť [K]-položky v `02-prior-art-mapa.md` v primárnych zdrojoch pred citovaním (rozšírené o #16–20).
- **T3:** PoC podľa `03-poc-plan.md`.
- **T4 (long-lead, hneď):** arXiv endorsement overiť; Zenodo účet + ORCID; GPG/SSH commit signing setup.
