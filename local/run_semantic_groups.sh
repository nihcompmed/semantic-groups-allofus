#!/usr/bin/env bash
# run_semantic_groups.sh -- the semantic groups from the item text, every step in order, on CPU.
#
# Local only: it reads the item text and the stored item embeddings, never a participant's answers, and needs no
# access to the All of Us Researcher Workbench. README.md gives the environment, the inputs, the outputs and the results.
#
#   PY=env/bin/python bash run_semantic_groups.sh
#
# Writes data/items_151.csv, data/sources/items_text_of_record.csv and outputs/ (k_selection, fits, random_starts,
# basis_fit, cores, basis, xenc, tables), then runs check.py groups, which compares the result with the files of record
# in expected/.
# About 30 to 40 minutes on a 48-core CPU: step 3 about 7 minutes (multithreaded SVDs), step 5 about 20 minutes
# (600 fits of about 10 seconds, 6 jobs in parallel).
# CPU only: the fit is device dependent at the level of floating-point arithmetic, and the basis of record is the CPU fit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"           # local/: semantic_groups/, paper/, data/, outputs/, ross_proj/
PY="${PY:-python}"
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
# the fitting library: ROOT/ross_proj, or a copy installed in the environment
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ENCODERS="bge-m3 bge-large-en-v1.5 all-mpnet-base-v2 e5-large-v2 gte-large-en-v1.5 qwen3-embedding-0.6b"
LOGS="$ROOT/outputs/logs"
mkdir -p "$LOGS"
cd "$ROOT/semantic_groups"
"$PY" -c "import ross_proj, jax; print('ross_proj from', ross_proj.__file__, '| jax', jax.__version__, jax.devices())"

step() { echo; echo "=== $*"; }

step "1  text of record (declared stems removed from the administered wording)"
"$PY" step1_text.py > "$LOGS/step1_text.log"; tail -1 "$LOGS/step1_text.log"
step "2  item table of record, data/items_151.csv"
"$PY" step2_items.py
step "3  number of groups per encoder, Horn's parallel analysis (200 permutations)"
"$PY" step3_k.py
step "4  sparse fit started from the principal components"
"$PY" step4_fit.py
step "5  100 random orthonormal starts per encoder, 6 jobs in parallel (logs in outputs/logs/)"
pids=()
for e in $ENCODERS; do
    "$PY" step5_random_starts.py --encoder "$e" --lambda-score 1.0 --seeds 0-99 > "$LOGS/step5_$e.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done       # stops here if any job failed
for e in $ENCODERS; do tail -1 "$LOGS/step5_$e.log"; done
step "6  selection rule (orthonormal to 0.01, 99% of PCA's explained variance, largest median core energy)"
"$PY" step6_select.py --lambda-score 1.0
step "7  the basis of record, outputs/basis_fit/"
"$PY" step7_basis.py
step "8  core items of every group and core groups of every item"
"$PY" step8_cores.py
step "9  the basis package the Workbench code reads, outputs/basis/"
"$PY" step9_package.py
step "10 agreement of the groups across encoders"
"$PY" step10_agreement.py
step "11 Tables S1 and S2, with their values in outputs/tables/"
cd "$ROOT/paper"
"$PY" appendix2_table1.py
"$PY" appendix2_table2.py
step "12 check against the files of record"
cd "$ROOT"
"$PY" check.py groups
