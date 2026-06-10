# 01 — Mechanizmus: v1.0 destilácia, v2 dizajn (M0 uzavretý)
**Aktualizované:** 2026-06-10. Sekcie §1–§3 sú historický záznam v1.0 (nemeniť). Sekcie §4–§8 = uzavretý v2 dizajn; plné technické znenie sa prenáša do `spec.md` (vznikne ako ďalší krok, potom je `spec.md` jediný zdroj pravdy pre implementáciu).

## 1. Čo v1.0 skutočne robí (bez metafor) [historické]
```
forward(x), x ∈ R^[B, N, d]:
  ak N ≤ 2:               vráť Attn_θ(x)            # zdieľaný {Q,K,V} softmax blok
  inak:
    L = forward(x[:, :N/2]) ; R = forward(x[:, N/2:])   # pozičné polenie
    C = concat(L, R, axis=seq)
    out = Attn_θ(C)                                  # PLNÁ attention nad celým N
    vráť out + 0.01·ε                                # šum na každej úrovni
```
Vlastnosti: jedna vrstva; **rovnaké θ na všetkých úrovniach** (jediný samopodobný prvok); bez kauzálnej masky; bez pozičného kódovania; bez FFN, residual prepojení a normalizácie; Python rekurzia (vetvy sa nespracúvajú batchovane).

**Korekcia 2026-06-10 (inšpekcia `scripts/v1_legacy.py` po prenose do repa):** (i) skript NIE JE inference-only — `__main__` spúšťa plnú tréningovú slučku (AdamW, CE, grad clipping, 5 epoch, dummy korpus; `generate_next_token` existuje, ale sa nevolá); (ii) šum 0.01 sa pridáva len na interných úrovniach rekurzie, nie na base-case; (iii) bez kauzálnej masky trpí tréningový loss target leakage (pozícia t vidí x[t+1] vo vstupe) — klesajúci loss v T0 logu preto nie je signál učenia. Pseudokód vyššie ostáva platný pre forward mechanizmus.

## 2. Čo je na tom potenciálne cenné [historické, stále platí]
- **Jedno zdieľané pravidlo naprieč škálami** — málo obsadená os: MEGABYTE/Hourglass majú per-level parametre a 2–3 fixné úrovne; Universal Transformer/MoR zdieľajú v hĺbke, nie v škále sekvencie.
- Priame napojenie na publikovanú premisu samopodobnosti jazyka (Hurst ≈ 0.7) — motivácia s citáciou, nie mystika.

## 3. Odvodenie zložitosti v1.0 [OVERENÉ — falzifikácia claimu; historické]
Rekurencia: `cost(N) = 2·cost(N/2) + c·N²`. Súčet cez úrovne: `c·N²·(1 + 1/2 + 1/4 + …) = 2c·N²` ⇒ **Θ(N²)**, asymptoticky 2× drahšie než flat Transformer. Master theorem: a=2, b=2, f(N)=N² dominuje N^(log₂2)=N¹ ⇒ Θ(N²). Navyše: hĺbka log N **sekvenčne závislých** attention volaní. Záver: claims O(N log N)/O(N) platili až po vyriešení D3 — vyriešené v §5.

## 4. Dizajnové rozhodnutia D1–D6 [VŠETKY UZAVRETÉ 2026-06-09/10]

- **D1 Kauzalita — uzavreté.** Tri vrstvy mechanizmu: (i) lokálne okno tokenov = štandardná pásová kauzálna maska (token t vidí t−w..t); (ii) súhrny = **štrukturálne hradlovanie**, žiadna maska — uzol je konzumovateľný (rodičom aj read-outom) až keď jeho span celý leží ≤ t; kauzalita je vlastnosť dostupnosti, nie masky [OVERENÉ konštrukciou]; (iii) **vnútri uzla plná bidirekčná attention + Pool sú legálne**, lebo každý konzument uzla je striktne za koncom spanu — expresívna výhoda, ktorú flat kauzálny Transformer nemá. Verifikácia v M1: shift test (zmena tokenu t nesmie zmeniť logity pozícií < t) + completion test (inkrementálne dekódovanie ≡ plný forward; chytá Fenwick bugy).

- **D2 Pozičná informácia — uzavreté.** Princíp „zdieľaná geometria, súradnica ako vstup": (i) vnútri uzla **relatívny slot-RoPE** cez 2m vstupných slotov (pozície 1..2m), identický na každej úrovni ⇒ pravidlo je škálovo prenosné (presne stávka H1); ľavý/pravý potomok kódovaný poradím slotov; (ii) **level embedding e_ℓ** aditívne na vstupe uzla — vstupná súradnica, neláme zdieľanie váh; default ON, ablácia OFF (test čistej škálovej ekvivariancie — nech rozhodnú dáta); (iii) read-out heterogénne kľúče: okno tokenov RoPE relatívne k query, súhrny NoPE + level embedding. Bonus: vo Fenwick dekompozícii úroveň bloku ~ log vzdialenosti od query ⇒ level embedding funguje aj ako log-škálovaný kód vzdialenosti („nedávne jemne, staré hrubo"). Zmiešané typy kľúčov v jednej attention známe z praxe (memory/sink tokeny) [K — overiť v T2].

- **D3 Kompresia v uzloch — uzavreté; hlavný novelty lever.** Operátory: **P1 latent-query attention pool** (PMA/Perceiver štýl; m naučených zdieľaných queries) = **default**; **P3 learned top-(m−1) selekcia + kontextový residual** (verbatim pass-through sloty; STE/Gumbel len v tréningu) = **challenger** s micro-gate G1b-D3; **P2 strided pairwise merge** (Hourglass štýl) = **control** (izoluje otázku, či selekcia musí žiť v Pool, alebo ju zvládne zdieľaný Attn_θ sám). Hybrid (k slotov P3 + m−k slotov P1) = záložná ablačná vetva, ak P3 neprejde stabilizačným screenom. Schedule: **m = 16 fixné**, pass-through s_ℓ = min(2^ℓ, m); ablácia m_ℓ = m₀ + g·ℓ (Θ(N) compute, log-rastúca kapacita koreňa); schedule b·2^(ℓ/2) zamietnutý (decode stav O(√N)). Cena: **Θ(N·m·d)** na vrstvu — pôvodný bound O(N·m²) bol voľný (nad prahom je len ~N/m uzlov; padding variant „vždy m aj z 2 vstupov" zavrhnutý ako informačne aj výpočtovo horší). Riziká: P1 query collapse (mitigácia: diversity regularizácia [K], monitoring P-C); P3 rich-get-richer + STE nestabilita (mitigácia: load-balance loss, temperature anneal, G1b-D3).

- **D4 Výstupná cesta — uzavreté: variant A (Fenwick read-out).** Token t pri predikcii attenduje: (i) presné okno tokenov t−w..t, **w = 64** (ablácia {32, 128}); (ii) súhrny S_u dokončených uzlov vo Fenwick(t) — minimálna množina ≤ log₂t uzlov pokrývajúcich prefix [1, t−w) (binárna dekompozícia prefixu, rovnaká konštrukcia ako #2). **Read-out zdieľa θ s uzlovými blokmi** (token = uzol úrovne 0; maximálna os A); ablácia: oddelené ψ. Variant B (plný down-pass kaskáda) **zamietnutý pre v2**: ~8,5× drahší tréning pri reprezentatívnych hyperparametroch (N=8192, m=16, w=64, w_n=8), staleness makro stavu 2^ℓ tokenov na úrovni ℓ, a štrukturálny retrieval bottleneck (celý makro kontext cez m vektorov na uzol vs m vektorov NA BLOK v A) — presne na needle osi, kde sa hrá empirický súboj s #2. B = follow-up vetva **SSRA-TD** po G2 (nosič H3; väčšia novelty vzdialenosť od #2), v paperi 1 veta v Limitations.

- **D5 Šum — uzavreté.** Aditívny šum v1.0 (0.01·ε, kumulácia ~log N) odstránený úplne — z inferencie aj z dizajnu. Regularizácia v2 = štandardný dropout (attention + residual), len v tréningu, sila hyperparameter pre M2. „Stochastická rezonancia" zostáva vo FIKCII (`00` §FIKCIA).

- **D6 Implementácia — uzavreté (realizácia v M1).** Iteratívne **level-wise** spracovanie: uzly jednej úrovne batchované do jedného tensoru, žiadna Python rekurzia; inak GPU ostáva nevyužité a meranie throughputu je bezcenné.

## 5. SSRA v2 — finálna kostra [uzavretá; technický detail → spec.md]
Vrstva i ∈ 1..L štandardného pre-norm Transformer stacku = [SSRA-mix; FFN]. Parametre θ_i, φ_i, e_ℓ **per vrstva, zdieľané naprieč všetkými škálami a read-outom v rámci vrstvy**. Každá vrstva si stavia strom nad aktuálnymi aktiváciami (súhrny = aktivácie per vrstva, nie perzistentný stav).

```
# Up-pass (per layer; binary tree, k=2; level-wise batched)
leaf:  S_tok = x_tok                                    # leaf summary = token state, s_0 = 1
node u (children c1, c2, level ℓ):
  X   = concat(S_c1, S_c2) + e_ℓ                        # ≤ 2m vstupných slotov
  H_u = X + Attn_θ(LN(X))     # pre-norm residual; BIDIREKČNÁ vnútri uzla (D1); slot-RoPE 1..2m (D2)
  S_u = Pool_φ(H_u) ∈ R^[s_ℓ, d],  s_ℓ = min(2^ℓ, m)    # P1 default / P3 challenger / P2 control (D3)

# Read-out (per layer; variant A)
K_t = [x_{t-w} .. x_t]  ∪  [S_u : u ∈ Fenwick(t)]       # ≤ w + m·log t kľúčov
y_t = t-tý výstup SSRA-mix = Attn_θ(q_t, K_t)           # pásová maska v okne; súhrny hradlované štrukturálne
```

## 6. Otvorené otázky M0 [VŠETKY VYRIEŠENÉ]
1. **m a rast s úrovňou** → m = 16 fixné, ablácia m₀ + g·ℓ (D3).
2. **Binárny vs k-árny strom** → k=2 default (identický Fenwick skelet ako #2 ⇒ porovnanie izoluje mechanizmus, nie štruktúru), k=4 ablácia. Odvodenie [OVERENÉ]: dominantný člen up-passu Θ(N·m·d·k²/(k−1)); k=4 = 1,33× compute, 1,5× decode stav, 0,5× lossy hopov. Poctivá poznámka: k=4 znižuje POČET selekčných udalostí na polovicu, ale každá je 2× tvrdšia (top-m z 4m) — čistý efekt na needle teoreticky nerozhodnuteľný ⇒ [HYPOTÉZA], rozhodne ablácia.
3. **Pozičná vs obsahová segmentácia (os C)** → odložené; v2 pozičná; H-Net štýl = samostatný follow-up, nevťahovať pred G1.
4. **Viac vrstiev** → SSRA = attention-sublayer v štandardnom stacku (atribúcia H1 čistá; parameter-match s flat takmer zadarmo, φ je malé). FFN v uzle: NIE v defaulte, ablácia pri kapacitnom probléme na G1. Zdieľanie θ naprieč vrstvami (UT × škála): ablácia po G2 — nesmie kontaminovať hlavný test.
5. **Streaming kauzalita / vzťah k Fenwick** → vyriešené cez D4: štrukturálne tá istá Fenwick rodina ako #2 (v paperi priznať), mechanizmus odlišný — softmax + učená Pool_φ + zdieľanie θ,φ naprieč škálami. Dôkaz odlíšenia = needle/recall pri matched compute; ak ablácie rozdiel neukážu ⇒ legitímny negatívny výsledok („učená kompresia nad Fenwick hierarchiou nepridáva nad λ-decay").

## 7. Zložitosť v2 — konsolidácia [OVERENÉ odvodením; per vrstva, ×L pre celý model]

Predpoklady: binárny strom, N = 2^L_tree, pass-through schedule, score-ops účtovníctvo (QKVO projekcie N·d² pridávajú členy lineárne v N — triedu nemenia, ale pri d=512, N=8k sú rádovo porovnateľné ⇒ wall-clock úsporu vs flat rozhodne až M1 throughput krivka, v paperi nesľubovať viac).

| veličina | SSRA v2 (variant A) | odvodenie (skratka) |
|---|---|---|
| up-pass tréning | Θ(N·m·d) | pod prahom Σ c·N·2^ℓ·d < 2cNmd; nad prahom ~N/m uzlov à 4cm²d ⇒ < 4cNmd |
| read-out tréning | Θ(N·(w + m·log N)·d) | token t: w + m·log t kľúčov; Σ_t |
| **tréning total** | **Θ(N·(w + m·log N)·d)** | read-out dominuje; trieda = #2 (O(T log T)), **nie lepšia** |
| pamäť aktivácií (strom) | O(N·d·log m) | pod prahom N vektorov/úroveň × log₂m úrovní; nad prahom Σ ≈ N — ~5× flat pri m=16; score matice však N·(w+m log N) vs N² flat |
| decode stav | **O(m·d·log N)** | Fenwick invariant: ≤ 1 aktívny dokončený uzol na úroveň × m vektorov — **cieľ z `03` splnený** |
| decode compute/token (amort.) | O((w + m·log N + m²)·d) | read-out + amortizovane 1 uzlové dokončenie/token (N−1 interných uzlov / N tokenov) |
| decode worst-case latency | spike O(log N) sekvenčných dokončení | token na pozícii 2^k dokončí k uzlov, rodič čaká na deti; pre PoC OK, flag v spec |

Porovnanie tried: flat Transformer Θ(N²·d) tréning, O(N·d) KV stav; Log-Linear (#2) O(T log T) tréning, O(log T) stav — **rovnaká trieda ako my, súboj je o mechanizmus, nie o asymptotiku**; MEGABYTE (#8) 2 úrovne, Θ(N²/P²) global člen. Variant B (zamietnutý): Θ(N·m²·w_n·d) tréning — Θ(N) trieda, ale break-even proti A až pri astronomickom N.

## 8. Testovateľné predikcie pre M3 [všetko HYPOTÉZA]
- **P-A (needle vs vzdialenosť):** P2 recall klesá ~log(vzdialenosť) (počet kompresných úrovní; útlm ~2^−Δℓ v lineárnom režime [OVERENÉ len v lin. režime]); P1/P3 približne ploché po kapacitný strop.
- **P-B (multi-needle cliff):** k needles v podstrome → recall cliff pri k ≈ m (P3: m−1; P1: skôr < m kvôli collapse); P2 cliff pri k ≈ 1. Najostrejší diskriminátor operátorov — M3 multi-needle generátor: k ∈ {1,2,4,8,16}, vzdialenosť ∈ {128..ctx}, exact-match recall.
- **P-C (collapse diagnostika):** entropia attention máp Q_φ + účasť jednotlivých queries — lacný tréningový monitoring.
- **P-D (A vs B retrieval):** needle recall A ≫ B pre vzdialenosti > w_n·w; multi-needle strop A ~m·log N agregátne vs B ~m globálne. (Testovateľné až vo follow-up SSRA-TD.)
- **P-E (= H3, konzistencia):** B ≥ A na konzistencii dlhých generácií — mäkké metriky, v PoC sa netestuje, Limitations.
