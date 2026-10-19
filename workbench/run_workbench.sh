#!/usr/bin/env bash
# run_workbench.sh -- every Workbench step in order, one log per step in logs/; stops at the first failure.
# Step 4 runs in PheTK's own environment (PHETK_PY, by default ~/phetk_env/bin/python; README.md, Setup).
#   bash run_workbench.sh            all steps
#   bash run_workbench.sh 7          from step 7 on
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
PHETK_PY="${PHETK_PY:-$HOME/phetk_env/bin/python}"
FROM="${1:-0}"
mkdir -p logs
"$PY" preflight.py 2>&1 | tee logs/preflight.log
for n in 1 2 3 4 5 6 7 8 9 10; do
  [ "$n" -lt "$FROM" ] && continue
  script=$(ls step${n}_*.py)
  py="$PY"
  [ "$n" = 4 ] && py="$PHETK_PY"
  echo "== step $n: $script"
  "$py" "$script" 2>&1 | tee "logs/step${n}.log"
done
echo "done: the downloads are in downloads/*.zip"
