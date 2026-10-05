#!/usr/bin/env bash
# run_deciles.sh -- Results, "Outcomes by Distance From the Cohort's Typical Scores": Figure 2, eFigure 2 and eTables 5
# to 7, drawn from the aggregate files in outputs/workbench/R2025Q4R6/ (outcome_deciles, ed_check,
# conditions_deciles_jno, sleep_device_deciles, cutoffs), written on the Researcher Workbench (../workbench/README.md).
# No participant-level data. About 1 minute. README.md, "Outcomes by decile of the distance".
#
#   PY=/path/to/env/bin/python bash scripts/run_deciles.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
cd "$HERE"
step() { echo; echo "=== $*"; }
step "Figure 2, outcomes by decile";                               "$PY" fig_outcome_deciles.py
step "eFigure 2, the 8 recorded conditions by decile, 6 encoders"; "$PY" fig_conditions_deciles.py
step "eTable 5, the 6 screening thresholds";                       "$PY" etable_thresholds_jno.py
step "eTable 6, sleep by decile within each device source";        "$PY" etable_sleep_device_jno.py
step "eTable 7, every outcome by decile, 6 encoders";              "$PY" etable_deciles_jno.py
step "check against the files of record";                          "$PY" check_expected.py deciles
