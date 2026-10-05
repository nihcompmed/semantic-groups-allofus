#!/usr/bin/env python3
"""sleep_reliability.py -- why a person needs 14 nights: the reliability of a personal sleep mean, for
the supplement (eMethods 3).

This writes out, as aggregates, the analysis explore_other_data.py cell 9 prints. Same nights, same
people, same arithmetic, so the supplement can state how the minimum of 14 nights was chosen.

THE NIGHTS (cell 9, declared before its first run). A night counts when it is the main sleep of the
day, falls within 180 days either side of the person's last answer date on the 151 items, and lasts
60 to 960 minutes.
The per-person file sleep_person_local.csv holds, per person, the number of such nights and their
mean and SD of minutes asleep.

THE ARITHMETIC (cell 9's, unchanged), on people with at least 7 nights (NIGHT_FLOOR, the fewest at
which a person's own SD is read):
  within-person variance   s2w = pooled over people, each person's variance weighted by (nights - 1)
  between-person variance  s2b = the variance of the person means MINUS the sampling noise in them,
                           the mean of (person SD^2 / nights). Without the correction the spread of
                           person means includes noise and reliability is overstated.
  reliability of a mean of n nights   R(n) = s2b / (s2b + s2w / n)
  nights needed for R = 0.90          ceil( (0.90 / 0.10) x s2w / s2b )
Then, by decile of the prespecified encoder's distance (gte-large-en-v1.5, the whole cohort ranked,
cell 9's rank-first rule), how many people have 14 nights and how many have the nights needed: the
rule's cost falls on the burdened if they wear the device less.

★ 14 IS THE MANUSCRIPT'S RULE (MIN_NIGHTS). This script does not choose it. It reports R(14), the
nights needed for 0.90, and who clears both, so the supplement can say whether 14 meets the target
and what it costs, by decile.

THE DISSEMINATION POLICY (disclosure.py). Every count is large (hundreds or more).
A share is blanked when the people clearing, or not clearing, a threshold are 1 to 20.

INPUTS   sleep_person_local.csv   (explore_other_data.py cell 9; beside the scripts, per person, stays)
         screen_out/scores_gte-large-en-v1.5.csv, the responses file (the cohort's ids)
OUTPUTS  screen_out/sleep_reliability/sleep_reliability_summary.csv   the variances, R(14), nights needed
         screen_out/sleep_reliability/sleep_reliability_curve.csv     n = 1 to 60: SE of a mean, R(n)
         screen_out/sleep_reliability/sleep_nights_by_decile.csv      per decile: people, median nights,
                                                                       median SD, share with 14 and with
                                                                       the nights needed
         screen_out/sleep_reliability_<CDR>.zip    those 3, checked by verdict() in zip_screen_aggregates.py

Run:  python3 sleep_reliability.py
"""
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from disclosure import pair_hidden                 # noqa: E402  the dissemination policy
from wb_config import CFG                           # noqa: E402
from zip_screen_aggregates import verdict           # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "sleep_reliability")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
ENCODER = "gte-large-en-v1.5"
NIGHT_FLOOR = 7                     # cell 9: nights before a person's own SD is read
MIN_NIGHTS = 14                     # the manuscript's minimum, which this reports on
TARGET = 0.90                       # cell 9: measurement noise under a tenth of the true variance
CURVE = range(1, 61)


def sleep_cache():
    for p in (os.path.join(HERE, "sleep_person_local.csv"), os.path.join(OUT_DIR, "sleep_person_local.csv")):
        if os.path.exists(p):
            return p
    raise SystemExit("sleep_person_local.csv is missing: run "
                     "`python3 explore_other_data.py --cells 9`")


def deciles(ids):
    """cell 9's burden_deciles: the encoder's percentile over the whole cohort, ranked first, 10 equal groups."""
    M = pd.read_csv(os.path.join(OUT_DIR, f"scores_{ENCODER}.csv"), usecols=[ID_COL, "pctile"])
    M[ID_COL] = M[ID_COL].astype(str)
    M = M.set_index(ID_COL)["pctile"].reindex(ids).dropna()
    return (M.rank(method="first").sub(1) * 10 // len(M) + 1).astype(int)


def main():
    os.makedirs(OUT, exist_ok=True)
    ids = pd.read_csv(CFG.responses_csv, usecols=[ID_COL])[ID_COL].astype(str)
    N = len(ids)
    P = pd.read_csv(sleep_cache())
    P["person_id"] = P["person_id"].astype(str)
    P = P.set_index("person_id").reindex(ids).dropna(subset=["nights"])       # the cohort only
    n_any = int((P.nights >= 1).sum())
    K = P[(P.nights >= NIGHT_FLOOR) & P.sd_min.notna()]
    print(f"[sleep_reliability] cohort {N:,} | a counted night {n_any:,} | at least {NIGHT_FLOOR} nights "
          f"{len(K):,} | at least {MIN_NIGHTS} nights {int((P.nights >= MIN_NIGHTS).sum()):,}")

    w = (K.nights - 1).clip(lower=0)
    s2w = float((w * K.sd_min ** 2).sum() / max(w.sum(), 1))
    obs = float(K.mean_min.var(ddof=1))
    noise = float((K.sd_min ** 2 / K.nights).mean())
    s2b = max(obs - noise, 1e-9)
    rel = lambda n: s2b / (s2b + s2w / n)
    need = int(np.ceil((TARGET / (1 - TARGET)) * s2w / s2b))
    print(f"  within-person SD {np.sqrt(s2w):.1f} min | between-person SD {np.sqrt(s2b):.1f} min "
          f"(observed {np.sqrt(obs):.1f}, noise removed)")
    print(f"  R({MIN_NIGHTS}) = {rel(MIN_NIGHTS):.3f} | nights needed for R = {TARGET:.2f}: {need}")

    S = pd.DataFrame([{
        "n_people_cohort": N, "n_people_any_night": n_any, "n_people_floor": len(K),
        "n_people_min_nights": int((P.nights >= MIN_NIGHTS).sum()),
        "night_floor": NIGHT_FLOOR, "min_nights": MIN_NIGHTS, "target_reliability": TARGET,
        "within_sd_median": float(K.sd_min.median()), "within_sd_median_unfiltered": float(K.sd_all.median()),
        "pooled_within_var": s2w, "observed_between_var": obs, "sampling_noise_var": noise,
        "true_between_var": s2b, "pooled_within_sd": np.sqrt(s2w), "true_between_sd": np.sqrt(s2b),
        "reliability_at_min_nights": rel(MIN_NIGHTS), "se_at_min_nights": np.sqrt(s2w / MIN_NIGHTS),
        "nights_needed": need, "reliability_at_nights_needed": rel(need),
        "encoder_for_deciles": ENCODER, "cdr": CDR}])
    S.to_csv(os.path.join(OUT, "sleep_reliability_summary.csv"), index=False)
    pd.DataFrame({"nights": list(CURVE), "se_of_mean_min": [np.sqrt(s2w / n) for n in CURVE],
                  "reliability": [rel(n) for n in CURVE]}).to_csv(
        os.path.join(OUT, "sleep_reliability_curve.csv"), index=False)

    dec = deciles(ids)
    T = K.assign(decile=dec.reindex(K.index).values).dropna(subset=["decile"])
    rows = []
    for d, g in T.groupby("decile"):
        n = len(g)
        r = {"decile": int(d), "n_people": n, "median_nights": float(g.nights.median()),
             "median_within_sd": float(g.sd_min.median())}
        for thr, col in ((MIN_NIGHTS, "share_with_min_nights"), (need, "share_with_nights_needed")):
            k = int((g.nights >= thr).sum())
            r[col] = np.nan if pair_hidden(k, n) else k / n
        rows.append(r)
    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(OUT, "sleep_nights_by_decile.csv"), index=False)
    print(f"  by decile of {ENCODER}'s distance, among people with at least {NIGHT_FLOOR} nights:")
    print(B.round(3).to_string(index=False))

    files = ["sleep_reliability_summary.csv", "sleep_reliability_curve.csv", "sleep_nights_by_decile.csv"]
    for f in files:
        ok, why = verdict(os.path.join(OUT, f))
        assert ok, f"{f} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"sleep_reliability_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(OUT, f), arcname=f"sleep_reliability/{f}")
    print(f"\n  download -> {zpath}")


if __name__ == "__main__":
    main()
