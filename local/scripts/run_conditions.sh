#!/usr/bin/env bash
# run_conditions.sh -- Results, "Semantic Groups and Recorded Conditions": Figure 4, eFigure 3 and eTables 9 to 11,
# drawn from the aggregate files in outputs/workbench/R2025Q4R6/ (conditions_assoc_jno, conditions_assoc_adjusted_jno),
# written on the Researcher Workbench (../workbench/README.md, step 18). The breast cancer timing quoted in the Results
# is in breast_timing_jno/ (not drawn). No participant-level data. About a minute. README.md, "Semantic groups and
# recorded conditions".
#
#   PY=/path/to/env/bin/python bash scripts/run_conditions.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
cd "$HERE"
step() { echo; echo "=== $*"; }
step "Figure 4, the prespecified encoder";   "$PY" fig_conditions_assoc.py --encoder gte-large-en-v1.5
for e in bge-m3 bge-large-en-v1.5 all-mpnet-base-v2 e5-large-v2 qwen3-embedding-0.6b; do
  step "eFigure 3, $e";                      "$PY" fig_conditions_assoc.py --encoder "$e"
done
step "eTables 9 to 11, all 6 encoders";      "$PY" etable_conditions_jno.py
step "check against the files of record";    "$PY" check_expected.py conditions
