#!/usr/bin/env python
"""Stage 3 -- cores of the own basis.

For each encoder's fit (P = scores.npy, 151 items x k components, from 2_fit.py):

  core items of a component     fixed-point participation-ratio core DOWN its column of P.
                                 Iterate keep-top-round(PR) until the set stops changing. No free
                                 parameter. Scale-invariant, so P and Z give the same core.
  core components of an item    the same rule ACROSS its row of Z = (P - mean) / sd, where sd is
                                 the column SD over the 151 items. Standardization is load-bearing
                                 here because each entry of a row carries a different divisor.
  construct profile             the documented constructs of a component's core items, as a set
                                 with counts. No modal label, no purity.
  faithful item                 an item whose documented construct appears in the union of the
                                 profiles of its core components, with the item's own membership
                                 of any core excluded (leave-one-out). Without the exclusion every
                                 anchoring item is faithful by construction.

Nesting between the two readings (a core item of c usually has c among its core components) is
definitional, and is printed only so that it is never mistaken for evidence.

Writes  ../outputs/cores/<enc>_components.csv, <enc>_items.csv, summary.csv
Run:    CUDA_VISIBLE_DEVICES="" python cores.py
"""
import csv
import glob
import os
from collections import Counter

import numpy as np

from _common import CONSTRUCTS_CSV, EMBEDDERS, OUT_DIR, load_embedding


def pr(v):
    s2 = float((v ** 2).sum())
    s4 = float((v ** 4).sum())
    return s2 * s2 / s4 if s4 > 0 else 0.0


def fixed_point_core(v):
    """Indices of the fixed-point PR core of a vector, in index order."""
    idx = np.arange(len(v))
    while True:
        n = max(1, min(int(round(pr(v[idx]))), len(idx)))
        top = np.sort(idx[np.argsort(-np.abs(v[idx]))[:n]])
        if len(top) == len(idx) and (top == idx).all():
            return top
        idx = top


def fit_dir(enc):
    """The basis of record: ../outputs/basis_fit/<enc>/, written by materialize_selection.py."""
    d = os.path.join(OUT_DIR, "basis_fit", enc)
    if not os.path.exists(os.path.join(d, "scores.npy")):
        raise SystemExit(f"{enc}: no basis of record at {d}. Run select_projection.py then "
                         f"materialize_selection.py first.")
    return d


def profile_str(counter):
    return "; ".join(f"{k}:{v}" for k, v in counter.most_common())


def main():
    con = {r["item"]: r["construct"] for r in csv.DictReader(open(CONSTRUCTS_CSV))}
    outdir = os.path.join(OUT_DIR, "cores")
    os.makedirs(outdir, exist_ok=True)
    summary = []
    for enc in EMBEDDERS:
        fd = fit_dir(enc)
        P = np.load(os.path.join(fd, "scores.npy"))                 # 151 x k
        items, _ = load_embedding(enc)
        n_item, k = P.shape
        assert len(items) == n_item
        mu_s, sd = P.mean(0), P.std(0)
        Z = (P - mu_s) / sd

        # --- component side: core items and construct profiles
        core_items, comp_rows = {}, []
        for c in range(k):
            core = fixed_point_core(P[:, c])
            core = core[np.argsort(-np.abs(P[core, c]))]
            core_items[c] = [int(i) for i in core]
            prof = Counter(con[items[i]] for i in core)
            comp_rows.append({
                "component": c, "size": len(core), "n_constructs": len(prof),
                "construct_profile": profile_str(prof),
                "items": " ".join(items[i] for i in core),
                "abs_scores": " ".join(f"{abs(P[i, c]):.4f}" for i in core)})
        with open(os.path.join(outdir, f"{enc}_components.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(comp_rows[0].keys()))
            w.writeheader(); w.writerows(comp_rows)

        # --- item side: core components, faithfulness (leave-one-out)
        item_rows, faithful = [], 0
        core_cols = {}
        for i in range(n_item):
            cols = fixed_point_core(Z[i, :])
            cols = cols[np.argsort(-np.abs(Z[i, cols]))]
            core_cols[i] = [int(c) for c in cols]
            loo_union = set()
            profs = []
            for c in cols:
                others = [j for j in core_items[int(c)] if j != i]
                pc = Counter(con[items[j]] for j in others)
                loo_union |= set(pc)
                profs.append(profile_str(pc) if pc else "(only this item)")
            is_faithful = con[items[i]] in loo_union
            faithful += is_faithful
            item_rows.append({
                "item": items[i], "construct": con[items[i]],
                "n_core_components": len(cols),
                "core_components": " ".join(f"C{c}" for c in cols),
                "abs_z": " ".join(f"{abs(Z[i, c]):.2f}" for c in cols),
                "core_component_profiles_loo": " | ".join(profs),
                "anchors_a_core": int(any(i in core_items[c] for c in range(k))),
                "faithful_loo": int(is_faithful)})
        with open(os.path.join(outdir, f"{enc}_items.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(item_rows[0].keys()))
            w.writeheader(); w.writerows(item_rows)

        # --- nesting (definitional) and the summary
        e_cols = {(i, c) for i in range(n_item) for c in core_cols[i]}
        e_items = {(i, c) for c in range(k) for i in core_items[c]}
        inter = e_cols & e_items
        sizes = sorted(r["size"] for r in comp_rows)
        ncon = sorted(r["n_constructs"] for r in comp_rows)
        ncc = sorted(len(core_cols[i]) for i in range(n_item))
        covered = len({i for c in range(k) for i in core_items[c]})
        summary.append({
            "encoder": enc, "k": k,
            "core_size_min": sizes[0], "core_size_median": sizes[len(sizes) // 2], "core_size_max": sizes[-1],
            "items_in_some_core": covered, "items_in_no_core": n_item - covered,
            "components_single_construct": sum(1 for x in ncon if x == 1),
            "components_le2_constructs": sum(1 for x in ncon if x <= 2),
            "n_constructs_median": ncon[len(ncon) // 2],
            "core_components_per_item_median": ncc[len(ncc) // 2],
            "faithful_loo_frac": round(faithful / n_item, 3),
            "nesting_item_to_component": round(len(inter) / len(e_items), 3),
            "nesting_component_to_item": round(len(inter) / len(e_cols), 3),
            "max_abs_column_mean": float(np.abs(mu_s).max())})
        s = summary[-1]
        print(f"{enc:22s} k={k:2d} core size {s['core_size_min']}/{s['core_size_median']}/{s['core_size_max']}  "
              f"single-construct {s['components_single_construct']:2d}  <=2 {s['components_le2_constructs']:2d}  "
              f"items in no core {s['items_in_no_core']:2d}  faithful {s['faithful_loo_frac']:.2f}  "
              f"nesting {s['nesting_item_to_component']:.2f}", flush=True)

    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print("->", os.path.join(outdir, "summary.csv"))


if __name__ == "__main__":
    main()
