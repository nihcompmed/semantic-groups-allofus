"""Shared config for the All of Us paper's own-basis pipeline (paths, encoder list, loader).

The 151 items are described by one table of record, `../data/items_151.csv` (item, concept id,
survey, instrument, construct, the text embedded and where it came from, scale, encoding rule).
Each is embedded by six sentence encoders. Each `../data/embeddings/<prefix>_<model>.npz` holds
`items` (151 labels), `vectors` (151 x D, L2-normalized rows) and `meta`. The embeddings of
record are `admin151_items_*` (text_of_record).

The basis is fitted from the 151 items' own text, every component is cored by battery items by
construction, and all k are used.

Overridable without editing this file:
    ROSS_NPZ_PREFIX=<prefix>   ->  load ../data/embeddings/<prefix>_<model>.npz   (default: admin151_items)
    ROSS_OUT_DIR=<dir>         ->  write outputs there                            (default: ../outputs)
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))          # the folder holding scripts/, data/ and outputs/
DATA_DIR = os.path.join(PROJECT, "data", "embeddings")
OUT_DIR = os.environ.get("ROSS_OUT_DIR") or os.path.join(PROJECT, "outputs")
FIG_DIR = os.path.join(PROJECT, "figures")
# The table of record for the 151 items (build_items_table.py): item order, concept id, survey,
# instrument, construct, the embedded text (`text_of_record`), the response scale. Every script
# reads items and constructs from it. The response-level encoding map is the one other data file.
ITEMS_CSV = os.path.join(PROJECT, "data", "items_151.csv")
CONSTRUCTS_CSV = ITEMS_CSV                       # columns item, construct
TEXT_COL = "text_of_record"
ENCODING_CSV = os.path.join(PROJECT, "data", "encoding_map_all151.csv")

# The six encoders, in canonical order. gte-large-en-v1.5 is the exemplar for single-encoder figures.
_MODELS = ["bge-m3", "bge-large-en-v1.5", "all-mpnet-base-v2",
           "e5-large-v2", "gte-large-en-v1.5", "qwen3-embedding-0.6b"]
EXEMPLAR = "gte-large-en-v1.5"

# ROSS_MODELS=<a,b,...> replaces the model list for one run, so the TF-IDF comparator
# bases (`tfidf-subtlex-k4`, `tfidf-subtlex-k25`) go through this same pipeline into their own
# ROSS_OUT_DIR without touching the six encoders' outputs. Unset, the list is the six encoders.
if os.environ.get("ROSS_MODELS"):
    _MODELS = [m.strip() for m in os.environ["ROSS_MODELS"].split(",") if m.strip()]

# The default is the embeddings of record, admin151_items (text_of_record). The basis of record was fitted on them
# (its mu equals the admin151 column mean exactly).
_PREFIX = os.environ.get("ROSS_NPZ_PREFIX", "admin151_items")
EMBEDDERS = {m: f"{_PREFIX}_{m}.npz" for m in _MODELS}


def load_embedding(name):
    """Return (items, X) for an encoder.

    X = `vectors` exactly as stored: (151 items x D dims), items as samples, the orientation the
    sparse projection needs so that a sparse column is a few items per component. Rows are
    already L2-normalized by 0_embed.py. No further normalization here.
    """
    d = np.load(os.path.join(DATA_DIR, EMBEDDERS[name]), allow_pickle=True)
    return np.asarray(d["items"]).astype(str), np.asarray(d["vectors"], dtype=np.float64)
