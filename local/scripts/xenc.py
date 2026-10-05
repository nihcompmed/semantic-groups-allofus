#!/usr/bin/env python
"""Cross-encoder reproducibility of the components, read from both sides (eMethods 1).

COMPONENT SIDE (label-free). For each pair of encoders, every pair of components is scored by the
Jaccard overlap of their fixed-point cores (item sets), and the optimal one-to-one matching is the
Hungarian assignment on 1 - Jaccard. k differs across encoders, so the assignment is rectangular:
the smaller set is fully matched, the surplus in the larger set stays unmatched. Null: permute item
identities in the second encoder's cores before matching, NPERM_MATCH times per pair. tau = the
null's 95th percentile. A component REPRODUCES in another encoder when its matched Jaccard there
is >= tau. Anchoring on one encoder, each component's count of other encoders it reproduces in
(0 to 5) is stratified afterwards by the number of documented constructs in its core, which plays
no part in the matching. Every encoder is used as the anchor in turn; the exemplar is reported
in the main text.

ITEM SIDE. Under each encoder an item has core components (cores.py, <enc>_items.csv), and each
of those has a construct profile. The item's construct set under that encoder is the union of the
profiles. Agreement = mean pairwise Jaccard of the sets over the 15 encoder pairs, against a null
that permutes item identities (NPERM_ITEM draws). Reported overall, by whether the item anchors a
core, and by the item's mean number of core components. A share-weighted form (each construct
weighted by its share of the profile, averaged over the item's core components, Ruzicka
similarity) is reported alongside.

Reads  ../outputs/cores/<enc>_components.csv, <enc>_items.csv, ../data/items_151.csv (item, construct)
Writes ../outputs/xenc/component_reproduction_<enc>.csv   one row per anchor component
       ../outputs/xenc/component_reproduction_summary.csv  per anchor, by constructs in core
       ../outputs/xenc/pairwise_median_jaccard.csv         encoder x encoder
       ../outputs/xenc/matched_jaccard_real.npy, matched_jaccard_null.npy, tau
       ../outputs/xenc/item_agreement.csv                  one row per item
       ../outputs/xenc/summary.json
Run:   python xenc.py
"""
import csv
import json
import os
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from _common import CONSTRUCTS_CSV, EMBEDDERS, EXEMPLAR, ITEMS_CSV, OUT_DIR

NPERM_MATCH = 5
NPERM_ITEM = 100
RNG = np.random.RandomState(0)


def jac(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def ruzicka(a, b):
    keys = sorted(set(a) | set(b))   # fixed order, so the sums do not depend on string hashing
    num = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    den = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return num / den if den else 1.0


def match(A, B):
    J = np.array([[jac(a, b) for b in B] for a in A])
    ri, ci = linear_sum_assignment(1.0 - J)
    return {int(i): float(J[i, j]) for i, j in zip(ri, ci)}


def main():
    con = {r["item"]: r["construct"] for r in csv.DictReader(open(CONSTRUCTS_CSV))}
    items = [r["item"] for r in csv.DictReader(open(ITEMS_CSV))]
    cores_dir = os.path.join(OUT_DIR, "cores")
    out = os.path.join(OUT_DIR, "xenc")
    os.makedirs(out, exist_ok=True)
    ENCS = list(EMBEDDERS)

    cores, profiles, item_cc = {}, {}, {}
    for enc in ENCS:
        rows = list(csv.DictReader(open(os.path.join(cores_dir, f"{enc}_components.csv"))))
        cores[enc] = [set(r["items"].split()) for r in rows]
        profiles[enc] = [frozenset(con[i] for i in c) for c in cores[enc]]
        item_cc[enc] = {r["item"]: [int(c[1:]) for c in r["core_components"].split()]
                        for r in csv.DictReader(open(os.path.join(cores_dir, f"{enc}_items.csv")))}

    # ---------------- component side
    pairs = list(combinations(ENCS, 2))
    real = np.array([v for A, B in pairs for v in match(cores[A], cores[B]).values()])
    null = []
    for A, B in pairs:
        for _ in range(NPERM_MATCH):
            perm = dict(zip(items, RNG.permutation(items)))
            null += list(match(cores[A], [{perm[x] for x in c} for c in cores[B]]).values())
    null = np.array(null)
    tau = float(np.percentile(null, 95))
    np.save(os.path.join(out, "matched_jaccard_real.npy"), real)
    np.save(os.path.join(out, "matched_jaccard_null.npy"), null)

    pm = pd.DataFrame(index=ENCS, columns=ENCS, dtype=float)
    for A, B in pairs:
        med = float(np.median(list(match(cores[A], cores[B]).values())))
        pm.loc[A, B] = pm.loc[B, A] = med
    pm.to_csv(os.path.join(out, "pairwise_median_jaccard.csv"))

    summary_rows, per_anchor = [], {}
    for anchor in ENCS:
        others = [e for e in ENCS if e != anchor]
        Jby = {E: match(cores[anchor], cores[E]) for E in others}
        rep = np.array([sum(1 for E in others if Jby[E].get(i, 0.0) >= tau)
                        for i in range(len(cores[anchor]))])
        nprof = np.array([len(p) for p in profiles[anchor]])
        rows = []
        for i, c in enumerate(cores[anchor]):
            prof = Counter(con[h] for h in c)
            rows.append({"component": i, "core_items": len(c), "n_constructs": nprof[i],
                         # ties alphabetically; most_common() would keep set order, which follows Python's
                         # per-run string hashing, so tied constructs would come out in a different order on every run
                         "construct_profile": "; ".join(f"{k}:{v}" for k, v in
                                                        sorted(prof.items(), key=lambda t: (-t[1], t[0]))),
                         "reproduces_in_encoders": int(rep[i]) + 1,
                         **{f"jaccard_{E}": round(Jby[E].get(i, 0.0), 3) for E in others},
                         "items": " ".join(sorted(c))})
        pd.DataFrame(rows).to_csv(os.path.join(out, f"component_reproduction_{anchor}.csv"), index=False)
        for m, lab in [(nprof == 1, "1"), (nprof == 2, "2"), (nprof == 3, "3"), (nprof == 4, "4"),
                       (nprof >= 5, "5+"), (nprof <= 2, "1 or 2"), (nprof >= 1, "all")]:
            summary_rows.append({"anchor": anchor, "constructs_in_core": lab, "components": int(m.sum()),
                                 "reproduce_ge4_of_6": int((m & (rep >= 3)).sum()),
                                 "reproduce_all_6": int((m & (rep == 5)).sum())})
        per_anchor[anchor] = {"k": len(rep), "all_6": int((rep == 5).sum()), "ge4_of_6": int((rep >= 3).sum())}
    pd.DataFrame(summary_rows).to_csv(os.path.join(out, "component_reproduction_summary.csv"), index=False)

    print(f"matched Jaccard: real median {np.median(real):.3f}, null median {np.median(null):.3f}, "
          f"tau {tau:.3f}, share of matched pairs >= tau {(real >= tau).mean():.2f}")
    print("pairwise median matched Jaccard, mean over the other five, per encoder:")
    for e in ENCS:
        print(f"  {e:22s} {pm.loc[e].drop(e).mean():.3f}")
    print("per anchor: components reproducing in >=4 of 6 / all 6:")
    for e, d in per_anchor.items():
        print(f"  {e:22s} {d['ge4_of_6']:2d} / {d['all_6']:2d}  of {d['k']}")
    ex = pd.DataFrame(summary_rows)
    ex = ex[ex.anchor == EXEMPLAR]
    print(f"exemplar {EXEMPLAR}, by constructs in the core:")
    print(ex.drop(columns="anchor").to_string(index=False))

    # ---------------- item side
    sets, wsets, ncc, anchored = [], [], [], []
    for enc in ENCS:
        anch = {i for c in cores[enc] for i in c}
        s, w, n = [], [], []
        for g in items:
            F = item_cc[enc][g]
            s.append(frozenset().union(*[profiles[enc][c] for c in F]))
            wd = {}
            for c in F:
                for h in cores[enc][c]:
                    wd[con[h]] = wd.get(con[h], 0.0) + 1.0 / (len(cores[enc][c]) * len(F))
            w.append(wd)
            n.append(len(F))
        sets.append(s); wsets.append(w); ncc.append(n); anchored.append([g in anch for g in items])
    ncc = np.array(ncc).mean(0)
    anchored = np.array(anchored).mean(0) >= 0.5
    n = len(items)

    def pairwise(S, sim, perm=False):
        vals = []
        for A, B in combinations(range(len(ENCS)), 2):
            pb = RNG.permutation(n) if perm else np.arange(n)
            vals.append(np.mean([sim(S[A][i], S[B][pb[i]]) for i in range(n)]))
        return float(np.mean(vals))

    per_item = np.array([np.mean([jac(sets[A][i], sets[B][i]) for A, B in combinations(range(len(ENCS)), 2)])
                         for i in range(n)])
    per_item_w = np.array([np.mean([ruzicka(wsets[A][i], wsets[B][i]) for A, B in combinations(range(len(ENCS)), 2)])
                           for i in range(n)])
    real_item, chance_item = pairwise(sets, jac), float(np.mean([pairwise(sets, jac, True) for _ in range(NPERM_ITEM)]))
    real_w, chance_w = pairwise(wsets, ruzicka), float(np.mean([pairwise(wsets, ruzicka, True) for _ in range(NPERM_ITEM)]))
    pd.DataFrame({"item": items, "construct": [con[g] for g in items],
                  "mean_core_components": ncc.round(2), "anchors_a_core": anchored.astype(int),
                  "agreement_profile": per_item.round(3), "agreement_weighted": per_item_w.round(3)}
                 ).to_csv(os.path.join(out, "item_agreement.csv"), index=False)
    strata = {}
    for lo, hi, lab in [(0, 1.5, "1"), (1.5, 2.5, "2"), (2.5, 4.5, "3-4"), (4.5, 99, "5+")]:
        m = (ncc >= lo) & (ncc < hi)
        strata[lab] = {"items": int(m.sum()), "profile": float(per_item[m].mean()) if m.sum() else None,
                       "weighted": float(per_item_w[m].mean()) if m.sum() else None}
    print(f"\nitem side: agreement {real_item:.3f} against chance {chance_item:.3f} "
          f"(weighted {real_w:.3f} against {chance_w:.3f})")
    print(f"  anchors a core n={int(anchored.sum())} {per_item[anchored].mean():.3f}; "
          f"anchors none n={int((~anchored).sum())} {per_item[~anchored].mean():.3f}")
    for lab, d in strata.items():
        if d["items"]:
            print(f"  core components {lab:4s} n={d['items']:3d}  profile {d['profile']:.3f}  weighted {d['weighted']:.3f}")
        else:
            print(f"  core components {lab:4s} n=  0  (no items)")

    json.dump({"tau": tau, "matched_real_median": float(np.median(real)), "matched_null_median": float(np.median(null)),
               "share_matched_ge_tau": float((real >= tau).mean()), "nperm_match": NPERM_MATCH,
               "per_anchor": per_anchor, "exemplar": EXEMPLAR,
               "item_agreement": real_item, "item_chance": chance_item,
               "item_agreement_weighted": real_w, "item_chance_weighted": chance_w,
               "item_agreement_anchoring": float(per_item[anchored].mean()),
               "item_agreement_nonanchoring": float(per_item[~anchored].mean()),
               "item_strata": strata, "nperm_item": NPERM_ITEM},
              open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("->", out)


if __name__ == "__main__":
    main()
