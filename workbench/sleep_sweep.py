#!/usr/bin/env python3
"""sleep_sweep.py -- Fitbit sleep among the outliers and the exclusive outliers: mean sleep time and
sleep variability, with coverage, age and device source written beside them.

THE SETS, exactly those of care_sweep.py and the 2 breadth scripts (their own code, imported)
  all        the 12 groupings' top (100 - p)% of the cohort, p = 80 to 99
  exclusive  the 7 groupings' exclusive sets, all 6 encoder slots, gte first

THE NIGHTS AND THE PEOPLE (explore_other_data.py cell 9's rules). A night counts when it is
the main sleep of the day, falls within 180 days either side of the person's last answer date on the
151 items, and lasts 60 to 960 minutes. A person enters with at least 14 such nights (MIN_NIGHTS, the manuscript's
rule; sleep_reliability.py reports what 14 nights buy and who has them).
  mean sleep time     the person's mean minutes asleep over those nights
  sleep variability   the person's SD of minutes asleep from night to night

PER SET
  n_set                   the people in the set
  n_sleep, coverage       those with at least 14 nights, and their share of the set
  sleep_mean, _lo, _hi    the mean of the personal means, normal 95% interval
  sleep_var_median, _lo, _hi   the MEDIAN of the personal SDs, with a distribution-free 95% interval
                          (order statistics). A median, because an SD is right skewed, as in cells 9-12.
  sleep_age               the mean age of the people with sleep data
  share_wear              their share whose nights are from the program's wear study (loaned devices),
                          rather than their own device (wear_enrollees_local.csv, cell 10)
plus the cohort row, on the same rules, as the reference line.
★ THE SLEEP SAMPLE IS SELECTED. Owning and wearing a device is associated with the distance (the
coverage gradient), so every set's sleep figures are read on those who wear one.
coverage, sleep_age and share_wear show how that sample differs from set to set. They do not remove it.
Unadjusted, as the care figures are.

THE DISSEMINATION POLICY (disclosure.py)
  - a set of 1 to 20: the row is blanked. An exclusive set whose complement in its top set is 1 to 20:
    n_set "suppressed".
  - n_sleep and coverage: hidden when the people with sleep data, or without, number 1 to 20; every
    sleep statistic is blanked when n_sleep is 1 to 20.
  - share_wear: blanked when the wear-study people, or the others, among n_sleep are 1 to 20.
  - ★ a statistic over a SUPPRESSED count is blanked too: coverage when n_set is "suppressed";
    every sleep statistic when n_sleep is (share_wear and sleep_age would otherwise return it exactly).
  Expected: exclusive sets at high cuts are 113 to 250 people and coverage is about 18%, so many of
  their sleep points are withheld or wide.

INPUTS   sleep_person_local.csv, wear_enrollees_local.csv   (beside the scripts; per person, stay)
         the responses file (age), and everything breadth_sweep.py reads
OUTPUTS  screen_out/sleep_sweep/sleep_sweep.csv, sleep_sweep_meta.json
         screen_out/sleep_sweep_<CDR>.zip    those 2, checked by verdict() in zip_screen_aggregates.py

Run:  python3 sleep_sweep.py
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from breadth_exclusive import rank_orders, top_and_exclusive           # noqa: E402  the same people
from breadth_sweep import CUTS, ENCODERS, groupings                      # noqa: E402
from disclosure import MARK, SUPP, pair_hidden, small                   # noqa: E402  the dissemination policy
from sleep_reliability import MIN_NIGHTS, sleep_cache                   # noqa: E402  one rule, one file
from wb_config import CFG                                               # noqa: E402
from zip_screen_aggregates import verdict                               # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "sleep_sweep")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
Z = 1.959964
STATS = ["coverage", "sleep_mean", "sleep_lo", "sleep_hi", "sleep_var_median", "sleep_var_lo",
         "sleep_var_hi", "sleep_age", "share_wear"]


def median_ci(x):
    """The median and its distribution-free 95% interval from order statistics (normal approximation
    to the binomial ranks, adequate above 20 people, which the policy guarantees)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    lo = max(int(np.floor(n / 2 - Z * np.sqrt(n) / 2)), 1)            # 1-indexed ranks
    hi = min(int(np.ceil(n / 2 + 1 + Z * np.sqrt(n) / 2)), n)
    return float(np.median(x)), float(x[lo - 1]), float(x[hi - 1])


def describe(mask, A, n_top=None):
    n = int(mask.sum())
    r = {"n_set": n}
    if n == 0:
        return r
    if small(n):
        r["n_set"] = MARK
        return r
    if n_top is not None and pair_hidden(n, n_top):
        r["n_set"] = SUPP
    s = mask & A["has_sleep"]
    k = int(s.sum())
    hidden = pair_hidden(k, n)
    r["n_sleep"] = (MARK if small(k) else SUPP) if hidden else k
    r["coverage"] = np.nan if hidden or r["n_set"] == SUPP else k / n
    if hidden or k < 2:                          # ★ a statistic over a suppressed count would return it (disclosure.py)
        return r
    m = A["mean_min"][s]
    se = float(np.std(m, ddof=1) / np.sqrt(k))
    r.update({"sleep_mean": float(m.mean()), "sleep_lo": float(m.mean()) - Z * se,
              "sleep_hi": float(m.mean()) + Z * se})
    med, lo, hi = median_ci(A["sd_min"][s])
    r.update({"sleep_var_median": med, "sleep_var_lo": lo, "sleep_var_hi": hi,
              "sleep_age": float(A["age"][s].mean())})
    if A["wear"] is not None:
        kw = int((s & A["wear"]).sum())
        r["share_wear"] = np.nan if pair_hidden(kw, k) else kw / k
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)         # the order every breadth script uses
    N = len(idx)
    R = pd.read_csv(CFG.responses_csv, usecols=[ID_COL, "age"])
    R[ID_COL] = R[ID_COL].astype(str)
    age = R.set_index(ID_COL)["age"].reindex(idx).values.astype(float)

    P = pd.read_csv(sleep_cache())
    P["person_id"] = P["person_id"].astype(str)
    P = P.set_index("person_id").reindex(idx)
    has = (P["nights"].fillna(0).values >= MIN_NIGHTS) & P["sd_min"].notna().values
    wf = os.path.join(HERE, "wear_enrollees_local.csv")
    wear = None
    if os.path.exists(wf):
        W = set(pd.read_csv(wf)["person_id"].astype(str))
        wear = idx.isin(W)
    else:
        print("  wear_enrollees_local.csv not found: share_wear is left out "
              "(copy it, or run `python3 explore_other_data.py --cells 10`)")
    A = {"age": age, "has_sleep": has, "mean_min": P["mean_min"].values.astype(float),
         "sd_min": P["sd_min"].values.astype(float), "wear": wear}
    print(f"[sleep_sweep] cohort {N:,} | at least {MIN_NIGHTS} nights {int(has.sum()):,} "
          f"({100 * has.mean():.1f}%)" + (f" | of them on wear-study devices {int((has & wear).sum()):,}"
                                          if wear is not None else ""))

    G12 = groupings()
    order = rank_orders([g[0] for g in G12], idx)
    info = {g[0]: g for g in G12}
    names12 = [g[0] for g in G12]
    others = [g[0] for g in G12 if g[0] not in ENCODERS]
    rows = [{"analysis": "cohort", "encoder_slot": "", "grouping": "cohort", "family": "reference",
             "k": np.nan, "cut_pctile": 0, **describe(np.ones(N, bool), A)}]
    for cut in CUTS:
        n_top, member, _ = top_and_exclusive(order, names12, N, cut)
        for name in names12:
            _, fam, k, _ = info[name]
            rows.append({"analysis": "all", "encoder_slot": "", "grouping": name, "family": fam, "k": k,
                         "cut_pctile": cut, **describe(member[name], A)})
        for slot in ENCODERS:
            names7 = [slot] + others
            _, _, excl = top_and_exclusive(order, names7, N, cut)
            for name in names7:
                _, fam, k, _ = info[name]
                rows.append({"analysis": "exclusive", "encoder_slot": slot, "grouping": name, "family": fam,
                             "k": k, "cut_pctile": cut, "n_top": n_top, **describe(excl[name], A, n_top)})
    D = pd.DataFrame(rows)
    for c in STATS:
        if c not in D.columns:
            D[c] = np.nan
    D.to_csv(os.path.join(OUT, "sleep_sweep.csv"), index=False)

    c = rows[0]
    print(f"  cohort: coverage {100 * c['coverage']:.1f}% | mean sleep {c['sleep_mean']:.1f} min | "
          f"variability (median SD) {c['sleep_var_median']:.1f} min | age of sleep sample {c['sleep_age']:.1f}"
          + (f" | wear study {100 * c['share_wear']:.1f}%" if "share_wear" in c else ""))
    for analysis, slot in (("all", ""), ("exclusive", ENCODERS[0])):
        print(f"  at the 95th, {analysis}{' (slot ' + slot + ')' if slot else ''}:")
        at = D[(D.analysis == analysis) & (D.encoder_slot == slot) & (D.cut_pctile == 95)]
        for r in at.itertuples():
            print(f"    {r.grouping:22s} n {r.n_set!s:>6} | with sleep {r.n_sleep!s:>5} "
                  f"({100 * r.coverage:4.1f}%) | mean {r.sleep_mean:6.1f} min | variability {r.sleep_var_median:5.1f} "
                  f"min | age {r.sleep_age:5.1f} | wear {100 * r.share_wear:4.1f}%")
    print(f"  sleep means withheld under the policy: {int(D.sleep_mean.isna().sum())} of {len(D)} rows")

    json.dump({"cdr": CDR, "cuts": CUTS, "slots": ENCODERS, "min_nights": MIN_NIGHTS,
               "nights": "main sleep, within 180 days of the last answer date on the 151 items, 60 to 960 minutes (cell 9)",
               "measures": {"sleep_mean": "mean of personal mean minutes asleep, normal 95% CI",
                            "sleep_var_median": "median of personal SDs of minutes asleep, order-statistic 95% CI"},
               "groupings_all": [{"grouping": g, "family": f, "k": k, "k_rule": r} for g, f, k, r in G12],
               "exclusive_groupings": "the slot's encoder plus " + ", ".join(others),
               "adjustment": "none (unadjusted); coverage, sleep_age and share_wear written beside",
               "disclosure": "All of Us dissemination policy (disclosure.py)"},
              open(os.path.join(OUT, "sleep_sweep_meta.json"), "w"), indent=2)
    files = ["sleep_sweep.csv", "sleep_sweep_meta.json"]
    for f in files:
        ok, why = verdict(os.path.join(OUT, f))
        assert ok, f"{f} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"sleep_sweep_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(OUT, f), arcname=f"sleep_sweep/{f}")
    print(f"\n[sleep_sweep] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
