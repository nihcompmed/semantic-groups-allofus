#!/usr/bin/env bash
# run_groupings.sh -- the Discussion's comparison with the other groupings (eMethods 5): eTable 13 and eFigure 4, drawn
# from the aggregate file outputs/workbench/R2025Q4R6/outcome_deciles/, which holds all 12 groupings (the 6 encoders and
# the 6 other groupings) and also feeds Figure 2. No participant-level data. Under a minute. README.md, "The other
# groupings".
#
#   PY=/path/to/env/bin/python bash scripts/run_groupings.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
cd "$HERE"
step() { echo; echo "=== $*"; }
step "eTable 13, decile 10 under all 12 groupings";   "$PY" etable_groupings_jno.py
step "eFigure 4, every decile under all 12 groupings"; "$PY" fig_groupings_deciles.py
step "check against the files of record";             "$PY" check_expected.py groupings
