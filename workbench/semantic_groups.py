#!/usr/bin/env python3
"""semantic_groups.py -- the semantic groups of every direction, for every encoder. THE one place the
grouping is defined, so nothing downstream recomputes it.

Runs wherever the basis package is, on the Workbench or locally. Needs nothing but the basis, the item
table and numpy/pandas. Both outputs are aggregates and leave the Workbench.

WHAT A SEMANTIC GROUP IS.
  * A direction's CORE ITEMS are its fixed-point participation-ratio core, listed per direction in
    basis/components.csv. The core is what names a direction; the full loading column is what
    attributes a person, and the two are not the same thing.
  * THE SPLIT. A direction whose core items do not all load in the same sense sets one group of items
    against another, so it carries TWO semantic groups, one per side. A direction whose core items load
    alike carries ONE. At most two, always.
  * NO DIRECTION IS READ FROM THE SPLIT. A direction of the fit has no canonical orientation, the same
    content loads positive under some encoders and negative under others, and the responses are not
    keyed so that a higher value means more of a construct. The two sides are reported as two groups.
    Which is written first follows the loading mass and means nothing.
  * THE LABEL. A group is named by the construct labels of ALL its items, in descending count, never by
    the most common one alone. A majority rule fails here in three ways: ties, collisions where two
    different groups take the same name, and erasure of the cross-instrument links the grouping exists
    to show.
  * Groups OVERLAP and do not cover. An item can be in the core of one direction, of two, or of none.

INPUTS   basis/components.csv, basis/<encoder>.npz, basis/item_embeddings/, the item table
OUTPUTS  <out>/semantic_groups.csv       one row per item per encoder: which groups it is in
         <out>/semantic_group_defs.csv   one row per group: its items, constructs and instruments

Run:  python3 semantic_groups.py                 (writes into screen_out/)
      python3 semantic_groups.py --out <dir>
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_config import CFG   # noqa: E402
try:
    from construct_labels import display          # the reader-facing construct names
except Exception:                                  # internal labels are readable too
    def display(x, strict=True):
        return str(x)

OUT_DIR = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else CFG.out_dir

T = pd.read_csv(CFG.items_csv)
order = T["item"].astype(str).tolist()
row = {it: i for i, it in enumerate(order)}
CON = {it: display(c) for it, c in zip(T["item"].astype(str), T["construct"].astype(str))}
INS = dict(zip(T["item"].astype(str), T["instrument"].astype(str))) if "instrument" in T else {}
C = pd.read_csv(os.path.join(CFG.basis_dir, "components.csv"))


def profile(items):
    c = Counter(CON[i] for i in items)
    return "; ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


defs, member = [], []
for enc in sorted(set(C.encoder)):
    b = np.load(os.path.join(CFG.basis_dir, f"{enc}.npz"), allow_pickle=True)
    e = np.load(os.path.join(CFG.emb_dir, f"{CFG.emb_prefix}_{enc}.npz"), allow_pickle=True)
    pos = {str(x): j for j, x in enumerate(e["items"])}
    E = np.asarray(e["vectors"], float)[[pos[it] for it in order]]
    P = (E - b["mu"]) @ b["W"].T
    holds = {}
    for _, r in C[C.encoder == enc].iterrows():
        j = int(r["component"])
        core = str(r["core_items"]).split()
        sides = {}
        for it in core:
            sides.setdefault(bool(P[row[it], j] >= 0), []).append(it)
        # the side carrying more loading mass is written first. A drawing and ordering convention only.
        keys = sorted(sides, key=lambda k: -sum(abs(P[row[i], j]) for i in sides[k]))
        two = len(keys) > 1
        for rank, k in enumerate(keys):
            its = sides[k]
            name = f"C{j}" if not two else f"C{j}{'+' if rank == 0 else '-'}"
            defs.append({"encoder": enc, "group": name, "component": j, "n_groups_on_direction": len(keys),
                         "n_items": len(its), "construct_profile": profile(its),
                         "instruments": "; ".join(sorted({INS.get(i, "") for i in its})),
                         "items": " ".join(its)})
            for i in its:
                holds.setdefault(i, []).append(name)
    for it in order:
        g = holds.get(it, [])
        member.append({"encoder": enc, "item": it, "instrument": INS.get(it, ""),
                       "construct": CON[it], "n_groups": len(g), "groups": " ".join(g)})

D = pd.DataFrame(defs).sort_values(["encoder", "n_items"], ascending=[True, False])
M = pd.DataFrame(member)
os.makedirs(OUT_DIR, exist_ok=True)
D.to_csv(os.path.join(OUT_DIR, "semantic_group_defs.csv"), index=False)
M.to_csv(os.path.join(OUT_DIR, "semantic_groups.csv"), index=False)

print(f"{len(D)} groups over {D.encoder.nunique()} encoders, {len(M)} item rows")
print(f"\n{'encoder':24s} {'directions':>10s} {'2-sided':>8s} {'groups':>7s} {'1 construct':>12s} "
      f"{'items in 1':>11s} {'2+':>4s} {'none':>5s}")
for enc, d in D.groupby("encoder"):
    m = M[M.encoder == enc]
    n1 = sum(1 for _, r in d.iterrows() if ";" not in r.construct_profile)
    print(f"{enc:24s} {d.component.nunique():>10d} "
          f"{int((d.n_groups_on_direction > 1).sum() / 2):>8d} {len(d):>7d} {n1:>12d} "
          f"{int((m.n_groups == 1).sum()):>11d} {int((m.n_groups >= 2).sum()):>4d} "
          f"{int((m.n_groups == 0).sum()):>5d}")
print(f"\nwrote -> {OUT_DIR}/semantic_group_defs.csv and semantic_groups.csv")
