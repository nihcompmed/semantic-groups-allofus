#!/usr/bin/env python
"""check.py -- compare a fresh run with the files of record in expected/.

  groups   after run_semantic_groups.sh: the item table, k, the retained starts, the basis of every encoder, the core
           items of every group, the agreement across encoders, and the values of Appendix 2-tables 1 and 2
  paper    after run_paper.sh: every figure (PNG, byte for byte), every appendix table (.tex, as text, with the comment
           line that names the script set aside), and every supplementary workbook (cell by cell), written into
           ../elife/ beside this folder

Run:  python check.py groups|paper
Exit: 0 when everything matches, 1 when anything differs.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))                       # local/
EXP = os.path.join(ROOT, "expected")
ELIFE = os.path.join(os.path.dirname(ROOT), "elife")                     # ../elife/
ENCODERS = ["bge-m3", "bge-large-en-v1.5", "all-mpnet-base-v2", "e5-large-v2", "gte-large-en-v1.5",
            "qwen3-embedding-0.6b"]
TOL = 1e-9
GROUPS_EXACT = ["data/items_151.csv", "outputs/k_selection/summary.csv", "outputs/basis/components.csv",
                "outputs/cores/summary.csv", "outputs/tables/groups.csv", "outputs/tables/switch_items.csv"]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def numbers(a, b):
    """Largest absolute difference between 2 JSON-like objects; None if their structure or any string differs."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return None
        ds = [numbers(a[k], b[k]) for k in a]
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return None
        ds = [numbers(x, y) for x, y in zip(a, b)]
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    else:
        return 0.0 if a == b else None
    return None if any(d is None for d in ds) else max(ds, default=0.0)


def groups():
    rows = []
    for rel in GROUPS_EXACT:
        a, b = os.path.join(ROOT, rel), os.path.join(EXP, rel)
        rows.append((rel, os.path.exists(a) and md5(a) == md5(b), "exact"))
    rel = "outputs/random_starts/selection.json"
    a, b = json.load(open(os.path.join(ROOT, rel))), json.load(open(os.path.join(EXP, rel)))
    same_seed = all(a[e]["selected"]["seed"] == b[e]["selected"]["seed"] for e in ENCODERS)
    d = numbers({e: {k: v for k, v in a[e]["selected"].items() if k != "path"} for e in ENCODERS},
                {e: {k: v for k, v in b[e]["selected"].items() if k != "path"} for e in ENCODERS})
    rows.append((rel, same_seed and d is not None and d <= TOL, f"seeds {'identical' if same_seed else 'differ'}, "
                                                                 f"metrics max |difference| {d}"))
    for e in ENCODERS:
        rel = f"outputs/basis/{e}.npz"
        A, B = np.load(os.path.join(ROOT, rel)), np.load(os.path.join(EXP, rel))
        d = max(float(np.abs(A[k] - B[k]).max()) for k in ("W", "mu", "sigma"))
        rows.append((rel, d <= TOL, f"W, mu, sigma max |difference| {d:.1e}"))
    rel = "outputs/xenc/summary.json"
    d = numbers(json.load(open(os.path.join(ROOT, rel))), json.load(open(os.path.join(EXP, rel))))
    rows.append((rel, d is not None and d <= TOL, f"max |difference| {d}"))
    return rows


def tex(p):
    """The lines of a table, without its LaTeX comment lines (the first names the script that wrote it)."""
    return [line for line in open(p).read().splitlines() if not line.startswith("%")]


def paper():
    rows = []
    for sub, ext in (("figures", ".png"), ("tables", ".tex"), ("supplementary", ".xlsx")):
        for f in sorted(os.listdir(os.path.join(EXP, "elife", sub))):
            if not f.endswith(ext):
                continue
            a, b = os.path.join(ELIFE, sub, f), os.path.join(EXP, "elife", sub, f)
            if not os.path.exists(a):
                rows.append((f"elife/{sub}/{f}", False, "not produced"))
            elif ext == ".png":
                rows.append((f"elife/{sub}/{f}", md5(a) == md5(b), "byte for byte"))
            elif ext == ".tex":
                rows.append((f"elife/{sub}/{f}", tex(a) == tex(b), "text"))
            else:
                A, B = pd.read_excel(a, sheet_name=None, header=None), pd.read_excel(b, sheet_name=None, header=None)
                ok = A.keys() == B.keys() and all(A[k].fillna("").astype(str).equals(B[k].fillna("").astype(str))
                                                  for k in A)
                rows.append((f"elife/{sub}/{f}", ok, "cell by cell"))
    return rows


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    assert which in ("groups", "paper"), "usage: check.py groups|paper"
    rows = groups() if which == "groups" else paper()
    for rel, ok, what in rows:
        print(f"{'match' if ok else 'DIFFERS':8s} {rel}  ({what})")
    n_ok = sum(ok for _, ok, _ in rows)
    print(f"{n_ok} of {len(rows)} match the files of record")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
