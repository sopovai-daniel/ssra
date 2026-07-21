# HO-33 — Stage-2 v1.1 build zadanie committnuté pred exekúciou; handoff CC; oversight v novom chate

Dátum: 2026-07-21 (build session, časť 1 — autorstvo zadania; 0 EUR). Nadväzuje: HO-32. Detailný záznam: D-log 2026-07-21 (druhý riadok).

## 1. Stav po tomto sedení

- **Pre-registrované zadanie `docs/cc/stage2-v11-build.md` v1 committnuté a pushnuté PRED exekúciou:** commit **`6bf7be9`** (GitHub file list overený: presne 1 súbor, +114, 0 pod `src/ssra/` a `baselines/`). Verejný timestamp edit-listu #1–#8 aj code-read otázky = obrana proti reviewer-úderu „post-hoc dolepok" (spolu s V11 zadaním committnutým 2026-07-19 pred akoukoľvek analýzou).
- **Verbatim čítania (HO-32 §2 krok 2) vykonané:** §K1-analysis report, zadanie V11 (§5 + §3 K3-c), inventár 11 figúr (3 in-body kandidáti vizuálne overené), kotvy v `paper/results-paper.md`, vzor hlavičky/Version history z noty v1.1.
- **Nálezy:** (i) definície needle kategórií = zadanie V11 **§3 K3-c**, nie §5 (HO-32 nepresnosť; build zadanie cituje governing lokáciu); (ii) písmená (a)–(c) pochádzajú z HO-30 §1 / D-log 2026-07-20, nie z reportu — mapovanie zakódované v zadaní s provenance.
- **[OP] rozhodnutia (veto režim, bez veta = platia):** in-body §5.6 = F7a `v11-k1-rho-ssra` / F7b `v11-k1-tc-levelemb-ssra` / F7c `v11-k1-td-ssra`, 6 figúr repo-referenced; §2.8 nota v1.1 inline DOI bez novej bib položky ([26] ostáva v1.0); §5.1 participácia repo-referenced (sémantika v caption os-note); Version history medzi Disclosure a References; build report `results/stage2-v11-build-report.md`; `TODO(v1.1-DOI)`/`TODO(v1.1-date)` placeholdery do AP-25 publish session.
- Spend 0 EUR; kumulatív nezmenený **72,37 EUR ≈ 24,1 %** (+ pending K1 korekcia ≈ 0,06 [ODHAD]).

## 2. Handoff prompt pre CC (paste-ready; odovzdanie = schválenie)

```
Execute the pre-registered assignment docs/cc/stage2-v11-build.md (committed at 6bf7be9, pushed).
Authority chain: spec v1.2 > docs/cc/V11-data-exploitation.md §3/§5 (pre-registered, unchanged) > the assignment.
Order: Gate B0 input asserts first — abort and report on any failure, no edits. Then edit #8 (read-only
code read), then edits #1-#7. Deliverables per assignment §6: edited paper/results-paper.md + new
results/stage2-v11-build-report.md — exactly these two files; one signed commit, message
"Stage-2 v1.1 build: integrate V11 material into results paper (#1-#8)"; clean git status before
commit; NO push. Hard constraints: zero diffs under src/ssra/ and baselines/; v1.0 artifacts
byte-identical (md5 pre/post in the report); every number sourced programmatically per Gate B3;
headline formulations grep-verified unchanged.
```

Po dobehnutí CC: rýchly pohľad na `results/stage2-v11-build-report.md` → **`git push` (human checkpoint)** → nový chat per §5.

## 3. Oversight checklist (nový chat; verdikt PASS/FAIL pred reviewer-pass gate)

1. **Pre-flight:** local HEAD == remote HEAD == CC build commit; prečítať tento HO + D-log tail 2 + `results/stage2-v11-build-report.md`.
2. **Diff audit z GitHubu (G-V11-4 vzor):** build commit file list = presne 2 súbory (`paper/results-paper.md`, `results/stage2-v11-build-report.md`); 0 pod `src/ssra/` a `baselines/`.
3. **v1.0 nedotknutá:** md5 `paper/ssra-results-paper-v1.0.pdf` = `a0177d2334b30adc552ed8d80f4a9509`, `paper/ssra-results-paper-v1.0.md` = `db038ffb4bf1a9df0f238bed8b51becf` (nezávisle, nie len z CC reportu).
4. **Nezávislý prepočet každého nového čísla** (Claude kód nezávislý od CC skriptov): mediány 2,763482 / 2,864481 + všetkých 15 ρ(latent_q) z `v11-k1-rho.csv`; needle totály a distribúcia z `v11-needle-category-counts.csv` (+ krížovo `v11-needle-categorized.csv`); entropy pokrytie z `v11-pc-entropy-summary.json`; V2b z `v11-v2b-quantum.json`; provenance skaláre (drift 1,1359/1,1354, step list, byte sizes) z NPZ `meta_json` — flat NPZ lokálne (293 KB prenosné), **ssra NPZ 50,4 MiB nad copy ceiling → snippet exekuovaný Danielom (vzor D-log 2026-07-20)**.
5. **Formulačný audit:** grep headline formulácií („flat prior confirmed; SSRA prior violated", „+10.22 %", „11.8×", „11.1×") — kontexty byte-identické s v1.0; changelog veta verbatim „no v1.0 conclusion modified; one precision refinement (§5.2)"; Abstract/§1/§3/§4/§6/§7 nedotknuté (jediné sankcionované dotyky mimo §5: §2.8 jedna veta, §8 bullet parentéza, hlavička, Version history, §0 bookkeeping); C-T1 výhradne mechanicky; observácie (a)–(c) ako observácie; [OP] citáty verbatim; spec §16 + forbidden vocab na každej novej vete.
6. **Code-read verifikácia (edit #8):** prečítať citované file:line, nezávisle overiť odpoveď a mechanický dôvod; skontrolovať, že §5.2 nesie len to, čo kód dokazuje.
7. Po PASS → **reviewer-pass gate (Daniel) = podmienka publikácie.** Očakávané údery a obrany: „post-hoc dolepok" → V11 zadanie committnuté 2026-07-19 + build zadanie `6bf7be9` pred exekúciou + mechanický INCONCLUSIVE; „figúrová dilúcia" → kurácia 3 in-body / 6 repo-referenced; T-C architektonické čítanie → len meraný fakt + prípadný mechanický dôvod [OVERENÉ z kódu].
8. Po reviewer-pass PASS → **AP-25 blok** (samostatná publish session per HO-32 §2 krok 5: New version na 21439493 → „Get a DOI now!" → DOI/date do placeholderov → export kópia + PDF kanonickým toolchainom → identity pre-registrácia → verifikácia v cieli → Publish → user-view → metadata reciprocita) → close (tag `paper-v1.1`, D-log, ledger, HO). Tvrdo ≤ 2026-08-31.

## 4. Pending mimo build

- Billing korekcia K1 okna (~0,06 EUR [ODHAD]; append-only po objavení v konzole; nekritické).
- Teardown ≤ 2026-08-31 (checklist HO-29 §3 + ssh keypair; inventár konzumentov GCS zvyškov pred delete).
- #2 arXiv endorsement — po v1.1 publish (žiadosť môže citovať obe DOI + oba koncept-DOI).

## 5. Otvárací prompt pre oversight chat (paste-ready)

```
Stage-2 v1.1 oversight (0 EUR, lokalne). Cez Filesystem MCP precitaj docs/handover/HO-33-2026-07-21-stage2-v11-zadanie-committnute.md a docs/00-stav-a-triaz.md (tail 2), over git log -1 (<HASH>) a zhodu remote HEAD == local HEAD. Kontext: zadanie committnute 6bf7be9 PRED exekuciou; CC build commit = <HASH> (presne 2 subory: paper/results-paper.md + results/stage2-v11-build-report.md, bez pushu CC — pushol Daniel). Dnes: oversight per HO-33 §3 — diff audit z GitHubu, md5 v1.0 artefaktov, nezavisly prepocet vsetkych novych cisel (CSV/JSON lokalne; ssra NPZ cez snippet ktory exekuujem ja), grep headline formulacii + changelog veta verbatim, formulacny audit spec §16, code-read verifikacia #8. Po PASS: podklady pre reviewer-pass gate. Ziadne zmeny §4.x zaverov a headline formulacii.
```

(`<HASH>` doplň = SHA CC build commitu po tvojom pushi.)
