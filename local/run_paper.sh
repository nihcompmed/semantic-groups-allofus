#!/usr/bin/env bash
# run_paper.sh -- every figure, appendix table, supplementary file and figure source data of the paper, drawn from the
# files in outputs/ (the semantic groups of record and the Researcher Workbench downloads), then the check against
# expected/. Writes into ../output/ beside this folder. Each script runs as its own process.
#   PY=env/bin/python bash run_paper.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu MPLBACKEND=Agg
cd paper
for s in figure1 figure2 figure2_supplement1 figure2_supplement2 figure3 figure4 figure5 figure5_supplement1 \
         figure6 figure7 appendix2_figure1 appendix2_figure2 \
         appendix2_table1 appendix2_table2 appendix3_table1 appendix4_table1 appendix5_table1 \
         supplementary_file1 supplementary_file2 supplementary_file3 \
         figure2_source_data figure3_source_data figure4_source_data figure5_source_data figure6_source_data \
         figure7_source_data; do
  echo "=== $s.py"
  "$PY" "$s.py"
done
cd ..
echo "=== check against the files of record"
"$PY" check.py paper
