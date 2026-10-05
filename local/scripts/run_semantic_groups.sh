#!/usr/bin/env bash
# run_semantic_groups.sh -- the semantic groups from the item text, every step in order, on CPU.
#
# Local only: it reads the item text and the stored item embeddings, never a participant's answers, and needs no
# access to the All of Us Researcher Workbench. README.md gives the environment, the inputs, the outputs and the results.
#
#   bash scripts/run_semantic_groups.sh                 # python on PATH
#   PY=/path/to/env/bin/python bash scripts/run_semantic_groups.sh
#
# Writes data/items_151.csv, data/sources/items_text_of_record.csv and outputs/ (k_selection, fits, random_starts,
# basis_fit, cores, basis, xenc, tables, supplementary_data_jno, method), then runs check_expected.py semantic_groups, which
# compares the result with the files of record in expected/ when that folder is present.
# About 30 to 40 minutes on a 48-core CPU: step 3 about 7 minutes (multithreaded SVDs), step 5 about 20 minutes
# (600 fits of about 10 seconds, 6 jobs in parallel).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"          # scripts/
ROOT="$(dirname "$HERE")"                       # the folder holding scripts/, data/, outputs/ (and ross_proj/ in the submission copy)
PY="${PY:-python}"
# CPU only: the fit is device dependent at the level of floating-point arithmetic, and the basis of record is the CPU fit.
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
# the fitting library: ROOT/ross_proj, or a copy installed in the environment
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ENCODERS="bge-m3 bge-large-en-v1.5 all-mpnet-base-v2 e5-large-v2 gte-large-en-v1.5 qwen3-embedding-0.6b"
LOGS="$ROOT/outputs/logs"
mkdir -p "$LOGS"
cd "$HERE"
"$PY" -c "import ross_proj, jax; print('ross_proj from', ross_proj.__file__, '| jax', jax.__version__, jax.devices())"

step() { echo; echo "=== $*"; }

step "1  text of record (declared stems removed from the administered wording)"
"$PY" build_text_of_record.py > "$LOGS/1_text_of_record.log"; tail -1 "$LOGS/1_text_of_record.log"
step "2  item table of record, data/items_151.csv"
"$PY" build_items_table.py
step "3  number of groups per encoder, Horn's parallel analysis (200 permutations)"
"$PY" 1_select_k.py
step "4  sparse fit started from the principal components"
"$PY" 2_fit.py
step "5  100 random orthonormal starts per encoder, 6 jobs in parallel (logs in outputs/logs/)"
pids=()
for e in $ENCODERS; do
    "$PY" random_starts.py --encoder "$e" --lambda-score 1.0 --seeds 0-99 > "$LOGS/5_random_starts_$e.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done       # stops here if any job failed
for e in $ENCODERS; do tail -1 "$LOGS/5_random_starts_$e.log"; done
step "6  selection rule (orthonormal to 0.01, 99% of PCA's explained variance, largest median core energy)"
"$PY" select_projection.py --lambda-score 1.0
step "7  the basis of record, outputs/basis_fit/"
"$PY" materialize_selection.py
step "8  cores of every group and every item"
"$PY" cores.py
step "9  the basis package the Workbench stage reads, outputs/basis/"
"$PY" build_basis_package.py
step "10 agreement of the groups across encoders"
"$PY" xenc.py
step "11 tables and figures: eTable 1 (items), eTable 4 (groups), eTable 12 (items crossing), Supplement 2, Figure 1, eFigure 1"
"$PY" etable1_items.py
"$PY" etable_groups_jno.py
"$PY" etable_switch_items.py
"$PY" supp_items_jno.py
"$PY" fig_method_jno.py
"$PY" fig_variance_explained.py
step "12 check against the files of record"
"$PY" check_expected.py semantic_groups
