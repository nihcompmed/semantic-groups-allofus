#!/usr/bin/env bash
# Stage 0 — regenerate all six core-151 embeddings, each in the env it needs.
#
# Five models run in the transformers-5.x env, gte-large-en-v1.5 in the
# transformers-4.x env (its custom remote code breaks on transformers 5.x).
# Set the folder that holds both envs and the models with SF=...:
#
#   SF=<folder holding env/, env_gte_tf4/ and models/> bash 0_embed_all.sh
#
# See the README ("Encoding the item text again") for creating these envs and
# downloading the models from scratch.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SF="${SF:?set SF to the folder holding env/, env_gte_tf4/ and models/}"
ENV_TF5="$SF/env"               # transformers 5.x  (sentence-transformers 5.x)
ENV_TF4="$SF/env_gte_tf4"       # transformers 4.x  (for gte's remote code)
MODELS="$SF/models"
export HF_HOME="$SF/hf_cache"   # gte's custom code (Alibaba-NLP/new-impl) is cached here

for m in bge-m3 bge-large-en-v1.5 all-mpnet-base-v2 e5-large-v2 qwen3-embedding-0.6b; do
    "$ENV_TF5/bin/python" "$HERE/0_embed.py" --model "$m" --models-dir "$MODELS"
done

# gte: transformers-4.x env + bundled custom code
"$ENV_TF4/bin/python" "$HERE/0_embed.py" \
    --model gte-large-en-v1.5 --models-dir "$MODELS" --trust-remote-code

echo "All embeddings written to $HERE/data/  (now run 1_select_k.py → 2_fit.py)"
