#!/usr/bin/env python3
"""breadth_sweep.py -- the all-outliers breadth sweep.

For each of the 12 groupings and each cut p = 80, 81, ..., 99: take the top (100 - p)% of the cohort
by that grouping's own Mahalanobis distance, and report the mean and standard deviation of the number
of the six criteria those people meet. The top stops at the 99th. The sets are cumulative:
the 80th holds everyone above the 80th, so each point is a set, not a band.

THE 12 GROUPINGS, each at the k stated
  item text   the 6 sentence encoders, each at its own Horn's k (23 to 27)     screen_battery.py
              tfidf4    TF-IDF at its own Horn's k, 4                          screen_tfidf.py
              tfidf25   TF-IDF at 25, referenced to the encoders' Horn's k      screen_tfidf.py
  responses   pca, factor   at Horn's k on the response correlations, 25       comparators.py
              mca       MCA at 25, referenced to PCA's Horn's k                comparators.py
              mca<h>    MCA at its own Horn's k, h (90 on this CDR)            comparators.py --mca-k horn
Not included: instrument totals, because there is no principled way to standardize them, although
comparators.py writes scores_scale.csv. The exclusive outliers are not part of this sweep.

WHAT IS COMPUTED, per grouping and cut
  n                the size of the top set, round(N (100 - p) / 100), the same for every grouping
  n_all_evaluable  the people in it with all six criteria evaluable
  mean_breadth, sd_breadth   the number of the six met, over those people (SD with ddof = 1)
  se, ci_lo, ci_hi           the standard error of the mean and its normal 95% interval
plus 1 row with grouping "cohort", cut 0, over the whole cohort, as the reference line.
★ ON THE PEOPLE WITH ALL SIX EVALUABLE: a criterion withheld for
missing items must never count as a criterion not met. So the denominator can differ a little by
grouping, and n_all_evaluable is in the file at every point.
★ TIES at a cut are broken by a stable sort on the file order, which is the same for every grouping.

THE DISSEMINATION POLICY (disclosure.py). n is at least 1% of the cohort. A statistic
over 1 to 20 evaluable people is blanked. n_all_evaluable is written "suppressed" when it or its
complement in the set (n minus it) is 1 to 20, and ★ the statistics over it are then blanked too,
because se = sd / sqrt(n) gives n back exactly, and the mean is a sum over n.

INPUTS   screen_out/cutoff_flags.csv, cutoff_summary.json         (cutoffs.py)
         screen_out/scores_<grouping>.csv                         (the 3 scripts above)
         screen_out/run_config.json, screen_tfidf_config.json,
         comparator_summary.json, comparator_summary_mca<h>.json    (each grouping's k, for the record)
         screen_out/mca_eigenvalues.csv                           (MCA's Horn's k; mca_basis.py)
OUTPUTS  screen_out/breadth_sweep/breadth_sweep.csv      grouping x cut, under the policy
         screen_out/breadth_sweep/breadth_sweep_meta.json each grouping's k, family and source, the cuts
         screen_out/breadth_sweep_<CDR>.zip               those 2 files, checked by verdict() of
                                                          zip_screen_aggregates.py: the only thing to
                                                          download

Run:  python3 breadth_sweep.py
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from disclosure import SUPP, pair_hidden, small                  # noqa: E402  the dissemination policy
from zip_screen_aggregates import verdict                        # noqa: E402  the check of every download

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "breadth_sweep")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
CUTS = list(range(80, 100))                 # 80 to 99 inclusive: the top stops at the 99th
ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5",
            "all-mpnet-base-v2", "e5-large-v2", "qwen3-embedding-0.6b"]
SECOND_K = 25                                # the second k of MCA and of TF-IDF


def mca_horn_k():
    """MCA's own Horn's k, by comparators.py's rule, so the file name matches what it wrote."""
    E = pd.read_csv(os.path.join(OUT_DIR, "mca_eigenvalues.csv")).sort_values("axis")
    fails = np.where(E["eigenvalue"].values <= E["null_95th"].values)[0]
    assert fails.size, "MCA's Horn's k reaches the axes computed; rerun mca_basis.py with more --axes"
    return int(fails[0])


def read_json(name):
    p = os.path.join(OUT_DIR, name)
    assert os.path.exists(p), f"{p} is missing"
    return json.load(open(p))


def groupings():
    """(name, family, k, how the k was chosen), in the order the figure draws them."""
    run = read_json("run_config.json")
    tf = read_json("screen_tfidf_config.json")["models"]
    cmp_ = read_json("comparator_summary.json")
    h = mca_horn_k()
    cmp_h = read_json(f"comparator_summary_mca{h}.json")
    k_resp = int(cmp_["k"])
    k_mca = int(cmp_["mca"]["k"])
    assert k_mca == k_resp, f"the default MCA run is at k = {k_mca}, not PCA's Horn's k {k_resp}"
    if k_resp != SECOND_K:
        print(f"★ PCA's Horn's k is {k_resp} on this CDR, not {SECOND_K}: MCA's second k follows it, "
              f"TF-IDF's stays at {SECOND_K}")
    G = [(e, "item text", int(run["k_per_encoder"][e]), "own Horn's k") for e in ENCODERS]
    G += [("tfidf4", "item text", int(tf["tfidf4"]["k"]), "own Horn's k"),
          ("tfidf25", "item text", int(tf["tfidf25"]["k"]), "encoders' Horn's k"),
          ("pca", "responses", k_resp, "own Horn's k"),
          ("factor", "responses", k_resp, "Horn's k on the response correlations"),
          (f"mca{h}", "responses", int(cmp_h["mca" + str(h)]["k"]), "own Horn's k"),
          ("mca", "responses", k_mca, "PCA's Horn's k")]
    return G


def stats(b):
    """Mean, SD, SE and the normal 95% interval of the criteria count over the evaluable people."""
    n = len(b)
    if n < 2:
        return dict(mean_breadth=np.nan, sd_breadth=np.nan, se=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    m, s = float(b.mean()), float(b.std(ddof=1))
    se = s / np.sqrt(n)
    return dict(mean_breadth=m, sd_breadth=s, se=se, ci_lo=m - 1.96 * se, ci_hi=m + 1.96 * se)


def policy(D):
    """The dissemination policy on the sweep table, at write time (see the docstring)."""
    D = D.copy()
    S = ["mean_breadth", "sd_breadth", "se", "ci_lo", "ci_hi"]
    D["n_all_evaluable"] = D["n_all_evaluable"].astype(object)
    for i in D.index:
        n_i, ev_i = int(D.at[i, "n"]), int(D.at[i, "n_all_evaluable"])
        assert not small(n_i), "a top set of 1 to 20 people cannot occur at a 1% cut of this cohort"
        if small(ev_i):
            D.loc[i, S] = np.nan
        if pair_hidden(ev_i, n_i):
            D.at[i, "n_all_evaluable"] = SUPP
            # ★ a statistic over a suppressed count would return it (disclosure.py):
            # mean = sum / n, and n = (sd / se)^2
            D.loc[i, S] = np.nan
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
    G = groupings()
    print(f"[breadth_sweep] cohort {N:,} | all {NB} criteria evaluable {int(ev.sum()):,} | "
          f"{len(G)} groupings x {len(CUTS)} cuts ({CUTS[0]} to {CUTS[-1]})")

    rows = [{"grouping": "cohort", "family": "reference", "k": np.nan, "cut_pctile": 0, "n": N,
             "n_all_evaluable": int(ev.sum()), **stats(br[ev])}]
    for name, fam, k, how in G:
        S = pd.read_csv(os.path.join(OUT_DIR, f"scores_{name}.csv"))
        S[ID_COL] = S[ID_COL].astype(str)
        assert len(S) == N and S[ID_COL].isin(cf.index).all(), f"{name}: ids do not match cutoff_flags.csv"
        S = S.set_index(ID_COL).reindex(cf.index)             # one row order for every grouping
        order = np.argsort(-S["distance"].values, kind="mergesort")   # most extreme first, stable
        for cut in CUTS:
            top = order[: int(round(N * (100 - cut) / 100.0))]
            keep = top[ev[top]]
            rows.append({"grouping": name, "family": fam, "k": k, "cut_pctile": cut, "n": len(top),
                         "n_all_evaluable": len(keep), **stats(br[keep])})
        at95 = next(r for r in rows if r["grouping"] == name and r["cut_pctile"] == 95)
        print(f"  {name:22s} k={k:3d} ({how})  at the 95th: mean {at95['mean_breadth']:.2f}, "
              f"SD {at95['sd_breadth']:.2f}, over {at95['n_all_evaluable']:,} evaluable")
    D = pd.DataFrame(rows)
    policy(D).to_csv(os.path.join(OUT, "breadth_sweep.csv"), index=False)
    json.dump({"cdr": CDR, "n_criteria": NB, "cuts": CUTS, "second_k": SECOND_K,
               "denominator": "people with all criteria evaluable",
               "groupings": [{"grouping": g, "family": f, "k": k, "k_rule": r} for g, f, k, r in G],
               "disclosure": "All of Us dissemination policy (disclosure.py)"},
              open(os.path.join(OUT, "breadth_sweep_meta.json"), "w"), indent=2)

    files = ["breadth_sweep.csv", "breadth_sweep_meta.json"]
    for f in files:
        ok, why = verdict(os.path.join(OUT, f))
        assert ok, f"{f} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"breadth_sweep_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(OUT, f), arcname=f"breadth_sweep/{f}")
    print(f"\n[breadth_sweep] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
