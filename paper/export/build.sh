#!/usr/bin/env bash
# Rebuild the published results-paper PDF from the committed export source.
# Requires: pandoc >= 3.1, xelatex (TeX Live 2023), GNU FreeFont (FreeSerif/FreeSans/FreeMono).
# Run from the repository root; figures resolve via --resource-path=. (results/*.png).
# Rebuilds are content-identical but NOT byte-identical (embedded PDF timestamps);
# the published artifact of record is paper/ssra-results-paper-v1.1.pdf,
# md5 f0d6dab4e5247c3a96568a2aec1d218b (v1.1, built 2026-07-24; source
# paper/ssra-results-paper-v1.1.md md5 5384cc4e6cc6239098fc5b66f3aec13f).
# v1.0 artifact of record: paper/ssra-results-paper-v1.0.pdf,
# md5 a0177d2334b30adc552ed8d80f4a9509 (built 2026-07-19; linestretch 0.98).
#
# v1.1-specific parameter vs the v1.0 build:
#  - linestretch 0.97 (v1.0: 0.98): with the v1.1 additions the final
#    reference wrapped onto an orphaned 21st page; 0.97 pulls it back
#    (20 pages, author visual pass 2026-07-24). Line breaking is
#    independent of linestretch, so all horizontal-layout verifications
#    (comment-column alignment, URL breaking) are unaffected.
# If the TeX install lacks lmodern.sty (minimal containers), place an
# output-neutral stub (\ProvidesPackage{lmodern}\endinput) on TEXINPUTS;
# all document fonts are set explicitly via fontspec, so the stub does
# not affect output.
set -euo pipefail
TEXINPUTS="paper/export:" pandoc -f markdown+autolink_bare_uris \
  paper/ssra-results-paper-v1.1.md \
  -o paper/ssra-results-paper-v1.1.pdf \
  --pdf-engine=xelatex -H paper/export/header.tex \
  -V mainfont="FreeSerif" -V sansfont="FreeSans" -V monofont="FreeMono" \
  -V fontsize=10pt -V "geometry:margin=2.3cm" -V papersize=a4 \
  -V linestretch=0.97 -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
  --resource-path=.
