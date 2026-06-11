# 02 — Prior-art mapa
**Stav overenia:** položky s ✔ overené v primárnych zdrojoch 2026-06-09 (URL platné k tomuto dátumu). Položky **[K]** sú z knowledge-base modelu — pred citovaním v paperi povinne overiť (Pravidlo W). **Aktualizované 2026-06-10:** pridané #16–#19 (kotvy pre D3 operátory a D2 heterogénne kľúče) — všetky [K], overiť v T2.

| # | Práca | Rok / venue | Mechanizmus | Vzťah k SSRA |
|---|---|---|---|---|
| 1 | Fractal Patterns May Illuminate the Success of Next-Token Prediction (Alabdulmohsin et al.) ✔ [arXiv:2402.01825](https://arxiv.org/abs/2402.01825) | NeurIPS 2024 | Jazyk je samopodobný (H ≈ 0.7), long-range dependent; mikro vzorce zrkadlia makro | **Naša premisa — je publikovaná.** Citovať ako motiváciu; premisa nie je novelty |
| 2 | Log-Linear Attention (Guo et al.) ✔ [arXiv:2506.04761](https://arxiv.org/abs/2506.04761) | ICLR 2026 | Fenwick-tree hierarchia stavov; O(T log T) tréning, O(log T) pamäť; nedávne jemne, staré hrubo; formálne spojenie s H-maticami | **Najbližší konkurent claimov — po D4 (variant A) zdieľame aj Fenwick skelet read-outu; v paperi explicitne priznať.** Diferenciácia: softmax blok + UČENÁ uzlová kompresia Pool_φ (diskrétne m sloty) + zdieľanie θ,φ naprieč škálami vs linear-attention kernel + štrukturálny maticový stav s λ-decay. Empirický súboj: recall/needle pri matched compute. Ak ablácie rozdiel neukážu ⇒ negatívny publikovateľný výsledok |
| 3 | H-Transformer-1D (Zhu & Soricut) **[K]** | ACL 2021 | Hierarchické (H-) matice, multiresolution aproximácia attention, lineárna zložitosť | Matematický aparát „fraktálnych matíc" existuje a má meno. Pred paperom overiť detaily poolingu a parametrizácie |
| 4 | Fast Multipole Attention **[K]** | — | Near-field presne, far-field komprimovane, multilevel | Rovnaká rodina multiresolution aproximácií. **Pozn. po D4: náš read-out (okno presne + súhrny hrubo) je near/far-field vzor — related work** |
| 5 | MRA — Multi-Resolution Attention **[K]** | 2022 | Adaptívne multiresolution zjemňovanie attention | Rodina; spomenutá aj v LLSA blogu ✔ ([zdroj](https://zhouyifan.net/blog-en/2025/12/19/20251211-llsa-1/)) |
| 6 | Native Sparse Attention (DeepSeek) ✔ [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) | ACL 2025 | Hierarchicky: komprimované bloky + selekcia tokenov + okno; natívne trénovateľné | SOTA efficiency kontext; iný cieľ (sparse selection). Related work; **kotva pre P3 (natívne trénovateľná selekcia)** |
| 7 | Infini-attention (Munkhdalai et al.) ✔ [arXiv:2404.07143](https://arxiv.org/abs/2404.07143) | 2024 | Kompresívna pamäť vo vanilla attention; bounded memory, infinite context | Pokrýva claim „kompresia minulosti bez mazania". Segmentová lineárna pamäť vs. náš strom |
| 8 | MEGABYTE (Yu et al., Meta) **[K]** | 2023 | 2 úrovne: global model nad patchmi + local model | Vertikálny tok, ale **2 fixné úrovne, nezdieľané váhy** — hlavný ablačný protivník pre os A. **Staleness down-cesty (patch granularita) = dôvod zamietnutia nášho variantu B pre v2** |
| 9 | Hourglass (Nawrot et al.) **[K]** | 2021/22 | Down/upsampling hierarchický LM (U-Net štýl) | Referenčný dizajn pre zamietnutý variant B; **kotva pre P2 (strided merge)**. Nesamopodobný, per-level parametre |
| 10 | Universal Transformer (Dehghani et al.) **[K]** | ICLR 2019 | Zdieľanie váh v hĺbke + ACT | „Rovnaké pravidlo rekurzívne" — ale v hĺbke, nie v škále sekvencie. Ablácia „zdieľanie naprieč vrstvami" (po G2) = UT × škála kombinácia |
| 11 | Mixture-of-Recursions **[K]** | 2025 | Adaptívna rekurzia zdieľaných blokov | Parameter sharing príbuzný osi A; bez škálovej hierarchie sekvencie |
| 12 | R2D2 / Fast-R2D2 (Hu et al.) **[K]** | 2021+ | Učený binárny strom, rekurzívna kompozícia rovnakou funkciou | Strom + zdieľaná funkcia — ale encoder/parsing, nie kauzálny LM |
| 13 | H-Net (Hwang, Gu et al.) **[K]** | 2025 | Dynamic chunking, učené obsahové hranice, vnoriteľné úrovne | Os C (obsahová hierarchia) — **odložená, follow-up po PoC**. Hranice áno, plná samopodobná attention rekurzia nie |
| 14 | HRM (Wang et al.) ✔ [arXiv:2506.21734](https://arxiv.org/abs/2506.21734) + TRM | 2025 | Mozgom inšpirovaná rekurencia, 2 časové škály, 27M/7M parametrov | Framing „brain-inspired tiny" je obsadený; puzzle domény, bez sekvenčného stromu |
| 15 | Fractal Generative Models (Li et al.) ✔ [arXiv:2502.17437](https://arxiv.org/abs/2502.17437) | 2025 | Rekurzívne generatívne moduly, samopodobné architektúry (obrazy) | **Názvová kolízia „fractal".** Iná doména; názov papera voliť tak, aby nedošlo k zámene |
| 16 | Set Transformer / PMA (Lee et al.) **[K]** | ICML 2019 | Pooling by Multihead Attention: m naučených seed/latent queries agreguje množinu | **Priama kotva pre P1 (default Pool_φ).** Naša odlišnosť: queries zdieľané naprieč VŠETKÝMI uzlami a škálami stromu |
| 17 | Perceiver / Perceiver IO (Jaegle et al.) **[K]** | ICML 2021 | Latent-query cross-attention bottleneck nad veľkým vstupom | Rodina P1; kotva pre latent-query kompresiu |
| 18 | Gist tokens (Mu et al.) **[K]** | NeurIPS 2023 | Naučené tokeny komprimujúce prompt do krátkych „gist" slotov cez attention masku | Kotva pre P1/koncept m súhrnných slotov; iný cieľ (prompt compression) |
| 19 | ToMe — Token Merging (Bolya et al.) **[K]** | ICLR 2023 | Bezgradientové zlučovanie podobných tokenov | Kotva pre P2/P3 rodinu (token redukcia); ViT doména |
| 20 | Memory / attention sink tokeny (napr. StreamingLLM, Xiao et al.) **[K]** | 2023+ | Špeciálne perzistentné kľúče vedľa bežných tokenov v jednej attention | Kotva pre D2 heterogénne kľúče (okno RoPE + súhrny NoPE+level v jednom read-oute) |

## Novelty téza v0.2 (v1 finalizovaná 2026-06-10 v `docs/spec.md` §17 — tam je záväzné znenie)
SSRA = kauzálny jazykový blok, v ktorom **jedno zdieľané (θ, φ) attention+pooling pravidlo generuje celú škálovú hierarchiu vrátane token-level read-outu** (samopodobnosť parametrov; token = uzol úrovne 0), s **učenou kompresiou v uzloch** (P1/P3) a **bidirekčnou attention vnútri uzlov legálnou v kauzálnom LM** (štrukturálne hradlovanie spanov). Žiadna z prác 1–20 nekombinuje súčasne: zdieľanie naprieč škálami × učená uzlová kompresia × kauzálne LM nad textom.

**Poctivé vymedzenie voči #2 (z D4):** pozičná Fenwick štruktúra prefixu je prevzatá (citovať); novelty nesie výhradne mechanizmus (softmax + Pool_φ + zdieľanie) a jeho empirické dôsledky na recall.

**Riziková veta (čo musí ablácia rozhodnúť):** ak zdieľanie naprieč škálami nepomáha, SSRA degeneruje na pomalší H-Transformer; ak uzlová kompresia stráca detail, prehráva s Log-Linear Attention na needle teste. Obe osi preto majú vyhradené ablácie v M3.

## Diferenciačné experimenty
1. vs. flat Transformer: ppl pri matched compute + length extrapolation (1k→32k).
2. vs. Log-Linear GatedDeltaNet: needle-in-haystack / RULER-lite **+ multi-needle sada (P-B)**, per-position loss.
3. vs. MEGABYTE-style 2-level: ablácia počtu úrovní a zdieľania váh (on/off).

## Must-cite v paperi
1, 2, 3, 6, 7, 8, 9, 10, 12, 16, 17 (+13 pri osi C, +18–20 podľa finálneho textu). Pred submisiou: nový novelty sken arXiv (cs.CL, cs.LG) od 2026-06 — pole sa hýbe mesačne; okno sken→upload < 1 týždeň.
