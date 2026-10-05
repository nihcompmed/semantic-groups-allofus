#!/usr/bin/env bash
# run_contributions.sh -- Results, "Groups Contributing Most to the Distance": Figure 3 and eTable 8, drawn from the
# aggregate files in outputs/workbench/R2025Q4R6/direction_own/, written on the Researcher Workbench by direction_own.py
# with and without --invisible (../workbench/README.md, step 17). No participant-level data. Under a minute.
# README.md, "Groups contributing most to the distance".
#
#   PY=/path/to/env/bin/python bash scripts/run_contributions.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
cd "$HERE"
step() { echo; echo "=== $*"; }
step "Figure 3, groups among the 3 largest parts of the distance"; "$PY" fig_content_bars_jno.py
step "eTable 8, the same under all 6 encoders";                     "$PY" etable_content_jno.py
step "check against the files of record";                           "$PY" check_expected.py contributions
