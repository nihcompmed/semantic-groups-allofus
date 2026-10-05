#!/usr/bin/env python3
"""breadth_exclusive.py -- the exclusive outliers: whom each grouping alone flags, and how many of the
six criteria they meet.

THE DEFINITION
  7 groupings: 1 encoder, PCA, the factor model, MCA at k = 25 (PCA's Horn's k), MCA at its own
  Horn's k, TF-IDF at its own Horn's k (4) and at 25 (the encoders' Horn's k).
  A person is EXCLUSIVE to a grouping at cut p when they are in that grouping's top (100 - p)% of the
  cohort and in none of the other 6 groupings' top (100 - p)%.
  - Each depth is its own grouping: MCA at 25 and at 90, and TF-IDF at 4 and at 25, select different
    people (the sweep: MCA 3.38 against 2.23 criteria at the 95th).
  - Only 1 encoder at a time. The 6 encoders are near-copies (their sweep lines sit within about 0.1
    of each other), and near-copies inside one exclusive comparison erase each other's picks. The
    prespecified gte-large-en-v1.5 comes first. Each of the other 5 encoders can take its place,
    and then all 7 exclusive sets change, because the other 6 groupings exclude a different encoder's
    picks.
  - Cuts p = 80 to 99, as in breadth_sweep.py.

WHAT IS COMPUTED, per encoder slot, grouping and cut
  n_set                  the size of each top set, round(N (100 - p) / 100), the same for all 7
  n_exclusive            the people exclusive to this grouping
  n_exclusive_evaluable  those with all six criteria evaluable
  mean_breadth, sd_breadth, se, ci_lo, ci_hi   the number of the six met, over those people, with
                         breadth_sweep.py's own stats()
Ties at a cut are broken by the same stable sort as in breadth_sweep.py.

THE DISSEMINATION POLICY (disclosure.py)
  - n_exclusive of 1 to 20 is written "<=20", and every statistic on the row is blanked.
  - n_exclusive and its complement in the top set (n_set - n_exclusive, the people another grouping
    also flags) are a pair: when the complement is 1 to 20, n_exclusive is written "suppressed".
  - n_exclusive_evaluable and its complement in the exclusive set are a pair in the same way. A
    statistic over 1 to 20 evaluable people is blanked.
  - ★ when n_exclusive_evaluable is "suppressed", its statistics are blanked too: se =
    sd / sqrt(n) gives n back exactly.

INPUTS   everything breadth_sweep.py reads (scores for the 7 groupings, cutoff_flags.csv, the configs)
OUTPUTS  screen_out/breadth_exclusive/breadth_exclusive.csv       encoder slot x grouping x cut
         screen_out/breadth_exclusive/breadth_exclusive_meta.json  the groupings and their k per slot
         screen_out/breadth_exclusive_<CDR>.zip  those 2 files, checked by verdict() in zip_screen_aggregates.py

Run:  python3 breadth_exclusive.py                        (all 6 encoder slots, gte first)
      python3 breadth_exclusive.py --encoder bge-m3       (1 slot)
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from breadth_sweep import CUTS, ENCODERS, groupings, read_json, stats   # noqa: E402  the sweep's own pieces
from disclosure import MARK, SUPP, pair_hidden, small                   # noqa: E402  the dissemination policy
from zip_screen_aggregates import verdict                               # noqa: E402  the download check

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "breadth_exclusive")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
SLOTS = ([sys.argv[sys.argv.index("--encoder") + 1]] if "--encoder" in sys.argv else list(ENCODERS))
assert all(s in ENCODERS for s in SLOTS), f"--encoder must be one of {ENCODERS}"
STATS = ["mean_breadth", "sd_breadth", "se", "ci_lo", "ci_hi"]


def rank_orders(names, cf_index):
    """Each grouping's order of the cohort, most extreme first, by the sweep's stable sort. Shared with
    care_sweep.py, so both scripts select exactly the same people."""
    N = len(cf_index)
    order = {}
    for name in names:
        S = pd.read_csv(os.path.join(OUT_DIR, f"scores_{name}.csv"))
        S[ID_COL] = S[ID_COL].astype(str)
        assert len(S) == N and S[ID_COL].isin(cf_index).all(), f"{name}: ids do not match cutoff_flags.csv"
        S = S.set_index(ID_COL).reindex(cf_index)
        order[name] = np.argsort(-S["distance"].values, kind="mergesort")
    return order


def top_and_exclusive(order, names, N, cut):
    """For 1 cut: each grouping's top-set mask, and its exclusive mask (in its top set and in no other
    grouping's among `names`). Shared with care_sweep.py."""
    n_top = int(round(N * (100 - cut) / 100.0))
    member = {}
    for name in names:
        m = np.zeros(N, bool)
        m[order[name][:n_top]] = True
        member[name] = m
    count = np.sum([member[n] for n in names], axis=0)          # how many of the groupings flag each person
    return n_top, member, {n: member[n] & (count == 1) for n in names}


def policy(D):
    """The dissemination policy on the exclusive table, at write time (see the docstring)."""
    D = D.copy()
    for c in ("n_exclusive", "n_exclusive_evaluable"):
        D[c] = D[c].astype(object)
    for i in D.index:
        n_set = int(D.at[i, "n_set"])
        n_x, n_ev = int(D.at[i, "n_exclusive"]), int(D.at[i, "n_exclusive_evaluable"])
        if small(n_x):
            D.loc[i, STATS] = np.nan
            D.at[i, "n_exclusive"] = MARK
            D.at[i, "n_exclusive_evaluable"] = SUPP
            continue
        if pair_hidden(n_x, n_set):
            D.at[i, "n_exclusive"] = SUPP
        if small(n_ev):
            D.loc[i, STATS] = np.nan
        if pair_hidden(n_ev, n_x):
            D.at[i, "n_exclusive_evaluable"] = SUPP
            D.loc[i, STATS] = np.nan                 # ★ a statistic over a suppressed count would return it (disclosure.py): n = (sd / se)^2
    return D


def main():
    os.makedirs(OUT, exist_ok=True)
    NB = int(read_json("cutoff_summary.json")["n_criteria"])
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"))
    cf[ID_COL] = cf[ID_COL].astype(str)
    cf = cf.set_index(ID_COL)
    N = len(cf)
    ev = cf["all_evaluable"].astype(bool).values
    br = cf["breadth"].astype(float).values

    G12 = groupings()                                   # the sweep's 12, with their k and k rule
    others = [g for g in G12 if g[0] not in ENCODERS]   # the 6 that are not encoders
    assert len(others) == 6, f"expected 6 non-encoder groupings, got {[g[0] for g in others]}"

    order = rank_orders([g[0] for g in G12], cf.index)         # each grouping's rank order, once

    print(f"[breadth_exclusive] cohort {N:,} | all {NB} criteria evaluable {int(ev.sum()):,} | "
          f"encoder slots {len(SLOTS)} x 7 groupings x {len(CUTS)} cuts")
    rows, meta_slots = [], {}
    for slot in SLOTS:
        G7 = [g for g in G12 if g[0] == slot] + others
        meta_slots[slot] = [{"grouping": g, "family": f, "k": k, "k_rule": r} for g, f, k, r in G7]
        for cut in CUTS:
            n_top, _, exclusive = top_and_exclusive(order, [g[0] for g in G7], N, cut)
            for name, fam, k, rule in G7:
                excl = exclusive[name]
                keep = excl & ev
                rows.append({"encoder_slot": slot, "grouping": name, "family": fam, "k": k,
                             "cut_pctile": cut, "n_set": n_top, "n_exclusive": int(excl.sum()),
                             "n_exclusive_evaluable": int(keep.sum()), **stats(br[keep])})
        at = [r for r in rows if r["encoder_slot"] == slot and r["cut_pctile"] == 95]
        print(f"  slot {slot}: at the 95th (top set {at[0]['n_set']:,})")
        for r in at:
            print(f"    {r['grouping']:22s} exclusive {r['n_exclusive']:6,}  "
                  f"({r['n_exclusive_evaluable']:,} evaluable)  mean {r['mean_breadth']:.2f}, SD {r['sd_breadth']:.2f}")
    D = pd.DataFrame(rows)
    policy(D).to_csv(os.path.join(OUT, "breadth_exclusive.csv"), index=False)
    json.dump({"cdr": CDR, "n_criteria": NB, "cuts": CUTS, "slots": SLOTS, "groupings": meta_slots,
               "definition": "in this grouping's top (100 - p)% and in none of the other 6 groupings' top (100 - p)%",
               "denominator": "people with all criteria evaluable",
               "disclosure": "All of Us dissemination policy (disclosure.py)"},
              open(os.path.join(OUT, "breadth_exclusive_meta.json"), "w"), indent=2)

    files = ["breadth_exclusive.csv", "breadth_exclusive_meta.json"]
    for f in files:
        ok, why = verdict(os.path.join(OUT, f))
        assert ok, f"{f} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"breadth_exclusive_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(OUT, f), arcname=f"breadth_exclusive/{f}")
    print(f"\n[breadth_exclusive] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
