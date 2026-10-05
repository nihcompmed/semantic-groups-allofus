#!/usr/bin/env bash
# run_cohort.sh -- eTables 2 and 3 (the cohort, the outliers under each encoder, and the participants each outcome is
# measured on), drawn from the aggregate file outputs/workbench/R2025Q4R6/table1/, written on the Researcher Workbench by
# covariates.py (../workbench/README.md, step 15). No participant-level data. Under a minute. README.md, "Cohort".
#
#   PY=/path/to/env/bin/python bash scripts/run_cohort.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
cd "$HERE"
step() { echo; echo "=== $*"; }
step "eTables 2 and 3";                     "$PY" table1_jno.py
step "check against the files of record";   "$PY" check_expected.py cohort
