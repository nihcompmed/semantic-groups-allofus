#!/usr/bin/env python3
"""matched_breadth.py -- breadth and criterion-free share with the SET SIZES MATCHED.

WHY THE SIZES ARE MATCHED. A stricter cut raises mean breadth on its own, because it keeps the people
who are extreme under every encoder at once. Sets of different sizes therefore cannot be compared on
breadth.

WHAT MATCHING MEANS HERE. One stratum, like for like, with the cohort as a reference row.

  5% stratum (n = 4,518 each).  Each encoder ALONE flags 4,518 at its own 95th percentile, which is
    exactly what each comparator flags. So encoder-against-comparator is a fair contrast.

DEFINITIONS, TAKEN VERBATIM FROM comparators.py SO THE ROWS ARE COMPARABLE.
  * breadth is the number of the six criteria met, counted over a person's
    evaluable criteria
  * mean_breadth, share_none, share_ge2, share_ge5 are computed ONLY on people with all six
    evaluable, so a withheld criterion can never hide a met one
  * share_any_met_everyone is over the whole set, withheld or not (from the column `any_met`)
  Nothing here recomputes a criterion. It reads the flags cutoffs.py already wrote.

★ THE SCALE COMPARATOR has 34 units: 16 instrument totals, 5 BFI-2-XS trait scores and 13 standalone
questions, so every instrument is totaled (the WMH-CIDI and the MHQ included).

INPUTS   screen_out/cutoff_flags.csv          (from cutoffs.py: breadth, all_evaluable, any_met)
         screen_out/scores_<encoder>.csv      (from screen_battery.py: pctile, flagged)
         screen_out/scores_{scale,factor,pca,mca}.csv   (from comparators.py, mca90 with MCA_MODEL=mca90)
OUTPUTS  screen_out/matched_breadth.csv       1 row per set, plus the cohort as reference (LEAVES,
                                              aggregates only, suffix _<MCA_MODEL> when it is not mca)

Run:  python3 matched_breadth.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_config import CFG   # noqa: E402

OUT_DIR = CFG.out_dir
ID_COL = "id"
ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5",
            "all-mpnet-base-v2", "e5-large-v2", "qwen3-embedding-0.6b"]
# ★ MCA_MODEL picks WHICH mca run: "mca" is the primary, at the k matched
# to factor and pca; "mca90" is the sensitivity at MCA's own Horn k, written by
# `comparators.py --mca-k 90`. One or the other, never both -- 2 depths of the same
# representation in 1 comparison would split each other's exclusive picks.
MCA_MODEL = os.environ.get("MCA_MODEL", "mca")
COMPARATORS = ["scale", "factor", "pca", MCA_MODEL]

# ★ A tagged run must not be able to touch the primary's outputs. When MCA_MODEL
# is anything but "mca", every aggregate this script writes gains that suffix, so the sensitivity's
# tables and figures sit beside the primary's instead of on top of them. Per-model files already carry
# the model in their name and are left alone.
TAG = "" if MCA_MODEL == "mca" else f"_{MCA_MODEL}"


def tagged(name):
    stem, dot, ext = name.rpartition(".")
    return f"{stem}{TAG}{dot}{ext}"



def stats(ids, cf, label, stratum, n_target=None):
    """comparators.py's breadth block over an arbitrary id set, under the dissemination policy
    (disclosure.py): each share is a count over its denominator, and it is withheld
    where that count or its complement is 1 to 20."""
    from disclosure import SUPP, pair_hidden
    idx = cf.index.intersection(pd.Index(sorted(ids)))
    sub = cf.loc[idx]
    ev = sub[sub["all_evaluable"].astype(bool)]
    b = ev["breadth"].astype(float)
    n_s, n_b = len(idx), len(b)
    row = {"stratum": stratum, "flag_set": label, "n": n_s,
           "n_target": n_target if n_target is not None else n_s,
           "n_all_evaluable": n_b,
           "mean_breadth": round(float(b.mean()), 3) if n_b else np.nan}
    for col, k_ in (("share_none", int((b == 0).sum())), ("share_ge2", int((b >= 2).sum())),
                    ("share_ge5", int((b >= 5).sum()))):
        row[col] = np.nan if (not n_b or pair_hidden(k_, n_b)) else round(k_ / n_b, 4)
    k_any = int(sub["any_met"].astype(bool).sum())
    row["share_any_met_everyone"] = np.nan if (not n_s or pair_hidden(k_any, n_s)) else round(k_any / n_s, 4)
    if pair_hidden(n_b, n_s):
        row.update({"n_all_evaluable": SUPP, "mean_breadth": np.nan})
    return row


def load_scores(name):
    p = os.path.join(OUT_DIR, f"scores_{name}.csv")
    if not os.path.exists(p):
        print(f"  [skip] {os.path.basename(p)} not found")
        return None
    S = pd.read_csv(p)
    S[ID_COL] = S[ID_COL].astype(str)
    return S.set_index(ID_COL)


def top_n(S, n):
    """The n most extreme by percentile, ties broken by the squared distance."""
    by = [c for c in ("pctile", "distance2", "distance") if c in S.columns]
    assert by, "scores file has none of pctile/distance2/distance"
    return set(S.sort_values(by, ascending=False).index[:n])


def main():
    CFG.describe("matched_breadth")
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"))
    cf[ID_COL] = cf[ID_COL].astype(str)
    cf = cf.set_index(ID_COL)
    N = len(cf)
    print(f"[matched_breadth] cohort {N:,} | all criteria evaluable "
          f"{int(cf['all_evaluable'].astype(bool).sum()):,}")
    # ★ One stratum: the 5% operating point, where every encoder and every comparator flags the same
    # number by construction.

    S = {m: load_scores(m) for m in ENCODERS + COMPARATORS}
    S = {k: v for k, v in S.items() if v is not None}

    # the 5% operating point, as the comparators were built
    n5 = None
    for e in ENCODERS:
        if e in S and "flagged" in S[e].columns:
            n5 = int(S[e]["flagged"].astype(bool).sum())
            break
    if n5 is None:
        n5 = int(round(0.05 * N))
    S5 = f"single cut at 5% (n={n5})"
    print(f"  stratum: {S5}\n")

    rows = [stats(set(cf.index), cf, "cohort", "reference")]

    # ---- 5% stratum: every encoder alone against every comparator -------------------------------
    for m in ENCODERS + COMPARATORS:
        if m not in S:
            continue
        ids = (set(S[m].index[S[m]["flagged"].astype(bool)])
               if "flagged" in S[m].columns else top_n(S[m], n5))
        kind = "encoder" if m in ENCODERS else "comparator"
        rows.append(stats(ids, cf, f"{kind}: {m}", S5, n5))

    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(OUT_DIR, tagged("matched_breadth.csv")), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for s in B["stratum"].unique():
            print(f"--- {s} ---")
            print(B[B.stratum == s][["flag_set", "n", "n_all_evaluable", "mean_breadth",
                                     "share_none", "share_ge2", "share_ge5"]].to_string(index=False))
            print()
    print("-> matched_breadth.csv")


if __name__ == "__main__":
    main()
