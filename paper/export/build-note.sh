#!/usr/bin/env bash
# Rebuild the technical-note v1.1 PDF from the committed source.
# Requires: pandoc >= 3.1, xelatex (TeX Live 2023), GNU FreeFont.
# Run from the repository root.
# Rebuilds are content-identical but NOT byte-identical (embedded PDF
# timestamps); the artifact of record is paper/ssra-technical-note-v1.1.pdf,
# md5 63ae756bae7db282dc483e124970b7f5 (v1.1, built 2026-07-20; source
# paper/technical-note.md md5 99f0d1eb9815a04b836979170dddd762).
#
# Note-specific parameters vs export/build.sh (results paper):
#  - monofontoptions Scale=0.9: the longest code-comment line of note §2.1
#    (106 monospace cells) exceeds the footnotesize verbatim line capacity
#    (~100 cells at margin 2.3cm); scaling avoids an fvextra line break.
#    Render-verified for v1.1: zero wrap glyphs, comment columns x-aligned
#    (pdftotext -bbox), fonts exclusively GNU FreeFont.
#  - If the TeX install lacks lmodern.sty (minimal containers), place an
#    output-neutral stub (\ProvidesPackage{lmodern}\endinput) on TEXINPUTS;
#    all document fonts are set explicitly via fontspec, so the stub does
#    not affect output.
set -euo pipefail
TEXINPUTS="paper/export:" pandoc -f markdown+autolink_bare_uris \
  paper/technical-note.md \
  -o paper/ssra-technical-note-v1.1.pdf \
  --pdf-engine=xelatex -H paper/export/header.tex \
  -V mainfont="FreeSerif" -V sansfont="FreeSans" -V monofont="FreeMono" \
  -V monofontoptions="Scale=0.9" \
  -V fontsize=10pt -V "geometry:margin=2.3cm" -V papersize=a4 \
  -V linestretch=0.98 -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
  --resource-path=.
