#!/usr/bin/env bash
# Rebuild the published results-paper PDF from the committed export source.
# Requires: pandoc >= 3.1, xelatex (TeX Live), GNU FreeFont (FreeSerif/FreeSans/FreeMono).
# Run from the repository root; figures resolve via --resource-path=. (results/*.png).
# Rebuilds are content-identical but NOT byte-identical (embedded PDF timestamps);
# the published artifact of record is paper/ssra-results-paper-v1.0.pdf,
# md5 7e7d243b0447babe664885a9729d295e (v1.0, built 2026-07-18).
set -euo pipefail
TEXINPUTS="paper/export:" pandoc -f markdown+autolink_bare_uris \
  paper/ssra-results-paper-v1.0.md \
  -o paper/ssra-results-paper-v1.0.pdf \
  --pdf-engine=xelatex -H paper/export/header.tex \
  -V mainfont="FreeSerif" -V sansfont="FreeSans" -V monofont="FreeMono" \
  -V fontsize=10pt -V "geometry:margin=2.3cm" -V papersize=a4 \
  -V linestretch=0.98 -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
  --resource-path=.
