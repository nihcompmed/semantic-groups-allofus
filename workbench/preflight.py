#!/usr/bin/env python3
"""preflight.py -- check this folder and the environment before step 1. It reads nothing from the data repository and
writes nothing.

Checks: the item table, the answer coding and the stand-alone sources are here and cover the 151 items; the basis
packages of the 6 encoders and of the 2 TF-IDF groupings are complete, with item embeddings for every item; the Python
packages are installed; WORKSPACE_CDR is set.

Run:  python3 preflight.py      (prints "preflight clean." when nothing is wrong)
"""
import importlib
import os
import sys

from settings import BASIS, BASIS_TFIDF, CDR, CODING_CSV, EMB_PREFIX, ENCODERS, ITEMS_CSV, SOURCES_CSV

problems = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


print("1. the items")
for f in (ITEMS_CSV, CODING_CSV, SOURCES_CSV):
    check(os.path.exists(f), os.path.basename(f))
try:
    import numpy as np
    import pandas as pd
    T = pd.read_csv(ITEMS_CSV)
    items = T["item"].astype(str).tolist()
    check(len(items) == 151, f"{len(items)} items in the item table")
    E = pd.read_csv(CODING_CSV)
    check(set(E["concept_id"]) == set(T["concept_id"]), "the coding covers every item")
    S = pd.read_csv(SOURCES_CSV)
    single = set(T.loc[T["instrument_code"] == "single item", "item"])
    check(set(S["item"]) == single, "a source study for every stand-alone question")
except Exception as e:                                       # reported, then the rest still runs
    check(False, f"the item files could not be read ({type(e).__name__}: {e})")
    items = []

print("2. the groupings formed from the item text")
for folder, names in ((BASIS, ENCODERS), (BASIS_TFIDF, ["tfidf-subtlex-k4", "tfidf-subtlex-k25"])):
    for name in names:
        b = os.path.join(folder, f"{name}.npz")
        e = os.path.join(folder, "item_embeddings", f"{EMB_PREFIX}_{name}.npz")
        check(os.path.exists(b) and os.path.exists(e), f"{os.path.basename(folder)}/{name}.npz and its embeddings")
        if items and os.path.exists(e):
            z = np.load(e, allow_pickle=True)
            check(set(map(str, z["items"])) >= set(items), f"{name}: an embedding for every item")
check(os.path.exists(os.path.join(BASIS, "components.csv")), "basis/components.csv (the groups' names)")

print("3. packages")
for mod in ("numpy", "pandas", "sklearn", "scipy", "statsmodels", "google.cloud.bigquery"):
    try:
        importlib.import_module(mod)
        check(True, mod)
    except ImportError:
        check(False, f"{mod} is not installed")

print("4. environment")
check(bool(CDR), f"WORKSPACE_CDR={CDR or '(not set)'}")

print("=" * 60)
if problems:
    print(f"{len(problems)} problem(s) to fix before step 1")
    sys.exit(1)
print("preflight clean.")
