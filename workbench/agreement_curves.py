#!/usr/bin/env python3
"""agreement_curves.py -- how much the encoders (and the comparators) agree on WHO is far out, as a function
of how many people are taken.

STANDALONE on the Workbench (numpy, pandas, scipy). Reads only the per-person score files that
screen_battery.py and comparators.py already wrote. Writes aggregates only.

WHY. The 95th percentile is a convention. A mean pairwise Jaccard such as 0.58 between the encoders'
top-5% sets reads as weak agreement until one remembers that a cut on a continuous score turns small
ranking differences into set differences near the cut: distances correlated at 0.9 give an expected
Jaccard of about 0.5 at a 5% cut. So three things are reported instead of one number.

  1. Jaccard of the top-k sets against k, PAIRWISE: the 15 encoder pairs, each encoder against the
     factor model, the scale-score model and PCA, and the comparators against each other, with the
     chance value k / (2N - k) at every k, and the size of the 6-encoder intersection at every k.

     ★ THE INTERSECTION CURVE DEFINES NO STUDY POPULATION. It measures HOW FAR THE SIX OVERLAP at
     every cut, computed from the rank vectors (`rank` in main) and reading nothing from
     consensus_outliers.csv. That is a description of agreement. Nothing here selects anyone for
     analysis.

     ★ AND IT IS WHAT MAKES A JACCARD SUCH AS 0.58 READABLE. Against the chance value and the
     correlation benchmark in the paragraph above it is evidence that the encoders AGREE, since
     distances correlated at 0.9 give an expected Jaccard near 0.5 at a 5% cut. Report the curve and
     the benchmark, never the single number at a single cut.
     k runs on a log grid from K_MIN to N/2 plus the percentile points 90 to 99.
  2. Correlations of the distances themselves (Spearman on distance, Pearson on log distance) for the
     same pairs, the number behind the Jaccard.
  3. For people flagged at the 95th percentile by m of the 6 encoders (m = 1..5): where they sit under
     the encoders that did NOT flag them (percentile quantiles and the share in 90-95, 80-90, below 80).
     If most sit in 90-95, the disagreement is threshold noise on one picture; if many sit far below,
     some encoders see something the others do not.

INPUTS   screen_out/scores_<enc>.csv (id, distance, distance2, pctile, flagged) for the 6 encoders,
         scores_factor.csv, scores_scale.csv, scores_pca.csv (comparators.py)
         ★ consensus_outliers.csv is NOT read by this script. Every number below is computed from
         the score ranks directly.
OUTPUTS  screen_out/agreement_curves.csv        pair, k, n_A, n_B, intersection, jaccard, chance, x_chance
         screen_out/agreement_consensus_curve.csv k, n_all6, share_of_k, chance_all6
         screen_out/agreement_correlations.csv  pair, spearman_distance, pearson_log_distance
         screen_out/partial_flag_percentiles.csv m_of_6, n_people, quantiles of the percentile under the
                                                 non-flagging encoders, shares in bands
         every count under the dissemination policy (disclosure.py): 1 to 20 as "<=20", complementary
         cells as "suppressed", and the statistics that give them back blanked
         screen_out/agreement_summary.json

Run:  python3 agreement_curves.py         (--cohort k0 for the complete-case screen_out_k0/)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disclosure import MARK, SUPP, pair_hidden, partition_mask, small   # noqa: E402  the policy

HERE = os.path.dirname(os.path.abspath(__file__))
COHORT = sys.argv[sys.argv.index("--cohort") + 1] if "--cohort" in sys.argv else "k5"
OUT_DIR = os.path.join(HERE, "screen_out" if COHORT == "k5" else "screen_out_k0")
ID_COL = "id"
ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5", "all-mpnet-base-v2", "e5-large-v2", "qwen3-embedding-0.6b"]
COMPARATORS = ["factor", "scale", "pca"]
PCTILE = 95.0
K_MIN = 50
N_GRID = 40
# ★ THE DISSEMINATION POLICY (disclosure.py). An intersection of two top-k
# sets is hidden when it or k minus it is 1 to 20, with the Jaccard and its ratio to chance, which
# give it back. The same for the six-way intersection and share_of_k. The m-of-6 counts partition the
# cohort AND satisfy sum(m * n_m) = 6 x k95, two equations, so two hidden cells would be solvable:
# if any cell of that distribution is hidden, all of its counts are. Rows are kept.


def read_scores(name):
    p = os.path.join(OUT_DIR, f"scores_{name}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d[ID_COL] = d[ID_COL].astype(str)
    return d.set_index(ID_COL)["distance"].astype(float)


def main():
    D = {e: read_scores(e) for e in ENCODERS}
    assert all(v is not None for v in D.values()), "an encoder score file is missing -- run screen_battery.py"
    for c in COMPARATORS:
        s = read_scores(c)
        if s is not None:
            D[c] = s
        else:
            print(f"(scores_{c}.csv not found -- {c} skipped)")
    names = list(D)
    ids = D[ENCODERS[0]].index
    for n in names:
        D[n] = D[n].reindex(ids)
        assert D[n].notna().all(), f"{n}: ids do not match the first encoder's"
    N = len(ids)
    # rank positions: order[n][r] = index of the person with the r-th largest distance under n
    order = {n: np.argsort(-D[n].values, kind="mergesort") for n in names}
    rank = {n: np.empty(N, int) for n in names}
    for n in names:
        rank[n][order[n]] = np.arange(N)          # rank 0 = farthest

    pct_points = [int(round(N * (1 - p / 100))) for p in range(90, 100)]
    grid = sorted(set(np.unique(np.logspace(np.log10(K_MIN), np.log10(N // 2), N_GRID).astype(int)).tolist() + pct_points))
    pairs = [(a, b) for i, a in enumerate(ENCODERS) for b in ENCODERS[i + 1:]]
    pairs += [(e, c) for c in COMPARATORS if c in D for e in ENCODERS]
    pairs += [(a, b) for i, a in enumerate([c for c in COMPARATORS if c in D]) for b in [c for c in COMPARATORS if c in D][i + 1:]]

    rows = []
    for a, b in pairs:
        ra, rb = rank[a], rank[b]
        for k in grid:
            inter = int(((ra < k) & (rb < k)).sum())
            j = inter / (2 * k - inter)
            chance = k / (2 * N - k)
            rows.append({"pair": f"{a} | {b}", "A": a, "B": b, "k": k, "share_of_cohort": k / N, "n_A": k, "n_B": k,
                         "intersection": inter, "jaccard": j, "chance": chance, "x_chance": j / chance})
    C = pd.DataFrame(rows)
    Co = C.copy()
    hid = pd.Series([pair_hidden(i, k) for i, k in zip(C.intersection, C.k)], index=C.index)
    Co[["jaccard", "x_chance"]] = Co[["jaccard", "x_chance"]].mask(hid)
    Co["intersection"] = [(MARK if small(i) else SUPP) if h else int(i)
                          for i, h in zip(C.intersection, hid)]
    Co.to_csv(os.path.join(OUT_DIR, "agreement_curves.csv"), index=False)
    print(f"agreement_curves.csv: {int(hid.sum())} of {len(C)} rows suppressed (dissemination policy)")

    cons_rows = []
    R6 = np.column_stack([rank[e] for e in ENCODERS])
    for k in grid:
        n_all = int((R6 < k).all(1).sum())
        h = pair_hidden(n_all, k)
        cons_rows.append({"k": k, "share_of_cohort": k / N,
                          "n_all6": ((MARK if small(n_all) else SUPP) if h else n_all),
                          "share_of_k": np.nan if h else n_all / k,
                          "chance_all6": N * (k / N) ** 6})
    pd.DataFrame(cons_rows).to_csv(os.path.join(OUT_DIR, "agreement_consensus_curve.csv"), index=False)

    corr_rows = []
    for a, b in pairs:
        sp = stats.spearmanr(D[a].values, D[b].values).correlation
        pe = np.corrcoef(np.log(D[a].values + 1e-12), np.log(D[b].values + 1e-12))[0, 1]
        corr_rows.append({"pair": f"{a} | {b}", "A": a, "B": b, "spearman_distance": float(sp), "pearson_log_distance": float(pe)})
    K = pd.DataFrame(corr_rows)
    K.to_csv(os.path.join(OUT_DIR, "agreement_correlations.csv"), index=False)

    # partially flagged people at the operating point
    k95 = int(round(N * (1 - PCTILE / 100)))
    flagged6 = R6 < k95                                    # N x 6
    m = flagged6.sum(1)
    pct = {e: 100.0 * (1 - rank[e] / N) for e in ENCODERS}   # percentile of distance under e (100 = farthest)
    P6 = np.column_stack([pct[e] for e in ENCODERS])
    by_m = np.array([int((m == mm).sum()) for mm in range(0, 7)])
    m_hide, m_prim = partition_mask(by_m)
    m_all_hidden = bool(m_hide.any())                     # two equations: all or nothing
    part_rows = []
    for mm in range(1, 6):
        who = np.where(m == mm)[0]
        if len(who) == 0:
            continue
        if m_prim[mm]:                                    # the row rests on 1 to 20 people
            part_rows.append({"m_of_6": mm, "n_people": MARK, "n_person_encoder_cells": SUPP})
            continue
        vals = P6[who][~flagged6[who]]                    # percentiles under the encoders that did not flag them
        q = np.percentile(vals, [10, 25, 50, 75, 90])
        bands = {"share_90_95": float((vals >= 90).mean()), "share_80_90": float(((vals >= 80) & (vals < 90)).mean()),
                 "share_below_80": float((vals < 80).mean())}
        part_rows.append({"m_of_6": mm, "n_people": SUPP if m_all_hidden else int(len(who)),
                          "n_person_encoder_cells": SUPP if m_all_hidden else int(len(vals)),
                          "pct_q10": q[0], "pct_q25": q[1], "pct_median": q[2], "pct_q75": q[3], "pct_q90": q[4], **bands})
    Pp = pd.DataFrame(part_rows)
    Pp.to_csv(os.path.join(OUT_DIR, "partial_flag_percentiles.csv"), index=False)

    at95 = C[(C.k == k95)]
    enc_pairs = at95[at95.A.isin(ENCODERS) & at95.B.isin(ENCODERS)]
    summary = {"n": int(N), "k95": k95, "grid": grid,
               "mean_pairwise_jaccard_encoders_at_95": float(enc_pairs.jaccard.mean()),
               "range_pairwise_jaccard_encoders_at_95": [float(enc_pairs.jaccard.min()), float(enc_pairs.jaccard.max())],
               "chance_at_95": float(k95 / (2 * N - k95)),
               "mean_spearman_encoders": float(K[K.A.isin(ENCODERS) & K.B.isin(ENCODERS)].spearman_distance.mean()),
               "n_all6_at_95": (MARK if m_prim[6] else SUPP) if m_all_hidden else int(by_m[6]),
               "n_flagged_by_m": {int(mm): ((MARK if m_prim[mm] else SUPP) if m_all_hidden else int(by_m[mm]))
                                  for mm in range(0, 7)},
               "disclosure": "counts 1-20 written <=20, complementary cells 'suppressed'",
               "comparators_present": [c for c in COMPARATORS if c in D]}
    json.dump(summary, open(os.path.join(OUT_DIR, "agreement_summary.json"), "w"), indent=2)

    print(f"cohort {N:,} | k at the 95th percentile {k95:,} | all 6 flagged {int((m == 6).sum()):,}")
    print(f"encoder pairs at the 95th: Jaccard mean {enc_pairs.jaccard.mean():.3f} (range {enc_pairs.jaccard.min():.3f} to {enc_pairs.jaccard.max():.3f}), "
          f"chance {k95 / (2 * N - k95):.3f}; Spearman of distances mean {summary['mean_spearman_encoders']:.3f}")
    for c in [c for c in COMPARATORS if c in D]:
        sub = at95[at95.B == c]
        print(f"encoders vs {c:6s} at the 95th: Jaccard mean {sub.jaccard.mean():.3f} (range {sub.jaccard.min():.3f} to {sub.jaccard.max():.3f})")
    print("\nJaccard against k (mean over the 15 encoder pairs):")
    for k in [g for g in grid if g in pct_points or g in (grid[0], grid[len(grid) // 2], grid[-1])]:
        sub = C[(C.k == k) & C.A.isin(ENCODERS) & C.B.isin(ENCODERS)]
        print(f"  k {k:6d} ({100 * k / N:5.1f}%)  Jaccard {sub.jaccard.mean():.3f}  chance {sub.chance.iloc[0]:.3f}  x{sub.x_chance.mean():.0f}")
    print("\npeople flagged by m of 6 at the 95th, and where they sit under the encoders that did not flag them:")
    for r in Pp.itertuples():
        if isinstance(r.n_people, str):
            print(f"  m={r.m_of_6}: n {r.n_people}, suppressed")
            continue
        print(f"  m={r.m_of_6}: n {int(r.n_people)}, median percentile {r.pct_median:.1f} "
              f"(q25 {r.pct_q25:.1f}, q75 {r.pct_q75:.1f}); in 90-95 {100 * r.share_90_95:.0f}%, 80-90 {100 * r.share_80_90:.0f}%, below 80 {100 * r.share_below_80:.0f}%")
    print(f"\nwrote -> {OUT_DIR}: agreement_curves.csv, agreement_consensus_curve.csv, agreement_correlations.csv, "
          f"partial_flag_percentiles.csv, agreement_summary.json (aggregates only)")


if __name__ == "__main__":
    main()
