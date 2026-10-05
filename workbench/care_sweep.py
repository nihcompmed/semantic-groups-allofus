#!/usr/bin/env python3
"""care_sweep.py -- care and access barriers among the outliers and the exclusive outliers, with each
set's age and sex written beside them.

THE SETS, exactly those of the 2 breadth scripts (their own code, imported, not copied)
  all        breadth_sweep.py: the 12 groupings, cuts p = 80 to 99, the top (100 - p)% of the cohort by
             each grouping's own distance.
  exclusive  breadth_exclusive.py: 7 groupings (1 encoder, PCA, factor, MCA at 25 and at its Horn's
             k, TF-IDF at 4 and 25); in this grouping's top (100 - p)% and in none of the other 6's.
             All 6 encoder slots, gte first.

THE OUTCOMES, none of which enters the scoring: the access survey contributes 0 of the 151
items, and the ED share comes from the records. Built by care.py --outcomes-only, read here.
  access   any access barrier: no usual source of care, a cost barrier, delayed care, or cutting back
           on medication for cost                                  over those who answered the survey
  cost     any cost barrier                                        over those who answered the survey
  avoid    delayed or avoided care because of a provider's race or religion, some of the time or more
                                                                   over those who answered the survey
  ed       the ED share of visits (concepts 9203, 262 over all visits), a mean of per-person shares
                                                                   over those with at least 1 visit
UNADJUSTED, with 95% intervals: Wilson for the 3 rates, normal for the ED mean.

AGE AND SEX, WRITTEN BESIDE THE RATES. The distance is computed after each
score is residualized on age and sex, which removes their LINEAR effect on each score's average. It
does not remove differences in spread, curvature in age, or age-specific co-movement of items, so a
top set can still be older or younger than the cohort, and barriers vary with age. mean_age, sd_age
and share_female (sex at birth, female = 0 in the response file) show each set's makeup directly.

THE DISSEMINATION POLICY (disclosure.py). The counts behind each statistic are
checked even though only n_set, n_survey and n_ehr are written, because a rate times its denominator
gives a count back:
  - a set of 1 to 20 people: the whole row is blanked (n_set "<=20").
  - an exclusive set whose complement in the top set is 1 to 20: n_set "suppressed".
  - share_female: blanked when the number of women or of men in the set is 1 to 20.
  - n_survey, n_ehr: "suppressed" when they or the people in the set without them are 1 to 20; the
    statistics over them are blanked when they are 1 to 20.
  - ★ a statistic over a SUPPRESSED count is blanked too: over n_set, mean_age, sd_age and
    share_female; over n_survey, the three rates and intervals; over n_ehr, the ED share. A rate and its
    interval can be solved exactly for their denominator, and rounding them does not prevent it.
  - a rate and its interval: blanked when the people with the outcome, or without it, are 1 to 20.
    Expected at high cuts for "avoid", whose cohort rate is about 10%.

INPUTS   screen_out/care_outcomes_per_person.csv   (care.py --outcomes-only; per person, stays)
         the responses file (age, sex), and everything breadth_sweep.py reads
OUTPUTS  screen_out/care_sweep/care_sweep.csv       1 row per analysis x encoder slot x grouping x cut,
                                                   plus the cohort row
         screen_out/care_sweep/care_sweep_meta.json
         screen_out/care_sweep_<CDR>.zip            those 2 files, checked by verdict() in zip_screen_aggregates.py

Run:  python3 care.py --outcomes-only     (only if care_outcomes_per_person.csv is not in screen_out/)
      python3 care_sweep.py
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
from wb_config import CFG                                               # noqa: E402
from zip_screen_aggregates import verdict                               # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "care_sweep")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
RATES = {"access": "any_access_barrier", "cost": "any_cost_barrier", "avoid": "care_avoid_provider"}
FEMALE = 0.0                        # build_responses_v9.py: SEX_AT_BIRTH_ENC = {"male": 1.0, "female": 0.0}
Z = 1.959964


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / d
    return p, c - h, c + h


def describe(mask, A, n_top=None):
    """1 row of statistics for the people in `mask`, with the policy applied (see the docstring)."""
    n = int(mask.sum())
    r = {"n_set": n}
    if n == 0:                                   # an empty exclusive set: nothing to describe
        return r
    if small(n):
        r["n_set"] = MARK
        return r
    if n_top is not None and pair_hidden(n, n_top):
        r["n_set"] = SUPP
    if r["n_set"] == SUPP:                       # ★ a statistic over a suppressed count would return it
        r.update({"mean_age": np.nan, "sd_age": np.nan, "share_female": np.nan})
    else:
        age, sex = A["age"][mask], A["sex"][mask]
        r["mean_age"], r["sd_age"] = float(age.mean()), float(age.std(ddof=1))
        n_f = int((sex == FEMALE).sum())
        r["share_female"] = np.nan if pair_hidden(n_f, n) else n_f / n

    s = mask & A["has_survey"]
    n_s = int(s.sum())
    r["n_survey"] = SUPP if pair_hidden(n_s, n) else n_s
    for short, col in RATES.items():
        k = int(np.nansum(A[col][s]))
        if small(n_s) or pair_hidden(k, n_s) or r["n_survey"] == SUPP:
            r.update({f"{short}_rate": np.nan, f"{short}_lo": np.nan, f"{short}_hi": np.nan})
        else:
            p, lo, hi = wilson(k, n_s)
            r.update({f"{short}_rate": p, f"{short}_lo": lo, f"{short}_hi": hi})

    e = mask & A["has_ehr"]
    n_e = int(e.sum())
    r["n_ehr"] = SUPP if pair_hidden(n_e, n) else n_e
    if small(n_e) or n_e < 2 or r["n_ehr"] == SUPP:
        r.update({"ed_mean": np.nan, "ed_lo": np.nan, "ed_hi": np.nan})
    else:
        x = A["ed_share"][e]
        m, se = float(np.nanmean(x)), float(np.nanstd(x, ddof=1) / np.sqrt(n_e))
        r.update({"ed_mean": m, "ed_lo": m - Z * se, "ed_hi": m + Z * se})
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)         # the order every breadth script uses
    N = len(idx)

    R = pd.read_csv(CFG.responses_csv, usecols=[ID_COL, "age", "sex"])
    R[ID_COL] = R[ID_COL].astype(str)
    R = R.set_index(ID_COL).reindex(idx)
    assert R.notna().all().all(), "age or sex missing for someone in the cohort"
    assert set(np.unique(R["sex"])) <= {0.0, 1.0}, "sex is not coded 0/1 as build_responses_v9.py writes it"

    p = os.path.join(OUT_DIR, "care_outcomes_per_person.csv")
    assert os.path.exists(p), f"{p} is missing: run `python3 care.py --outcomes-only`"
    O = pd.read_csv(p)
    O[ID_COL] = O[ID_COL].astype(str)
    O = O.set_index(ID_COL).reindex(idx)
    A = {"age": R["age"].values.astype(float), "sex": R["sex"].values.astype(float),
         "has_survey": O["has_survey"].fillna(False).astype(bool).values,      # absent from the file = no survey
         "has_ehr": O["has_ehr"].fillna(False).astype(bool).values,
         "ed_share": O["ed_share"].values.astype(float)}
    for col in RATES.values():
        A[col] = O[col].values.astype(float)
    print(f"[care_sweep] cohort {N:,} | answered the access survey {int(A['has_survey'].sum()):,} | "
          f"at least 1 visit {int(A['has_ehr'].sum()):,}")

    G12 = groupings()
    order = rank_orders([g[0] for g in G12], idx)
    info = {g[0]: g for g in G12}
    rows = [{"analysis": "cohort", "encoder_slot": "", "grouping": "cohort", "family": "reference",
             "k": np.nan, "cut_pctile": 0, **describe(np.ones(N, bool), A)}]

    names12 = [g[0] for g in G12]
    others = [g[0] for g in G12 if g[0] not in ENCODERS]
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
    D.to_csv(os.path.join(OUT, "care_sweep.csv"), index=False)

    coh = rows[0]
    print(f"  cohort: age {coh['mean_age']:.1f}, female {100 * coh['share_female']:.1f}% | access "
          f"{100 * coh['access_rate']:.1f}%, cost {100 * coh['cost_rate']:.1f}%, avoid {100 * coh['avoid_rate']:.1f}%, "
          f"ED share {100 * coh['ed_mean']:.2f}%")
    for analysis, slot in (("all", ""), ("exclusive", ENCODERS[0])):
        print(f"  at the 95th, {analysis}{' (slot ' + slot + ')' if slot else ''}:")
        at = D[(D.analysis == analysis) & (D.encoder_slot == slot) & (D.cut_pctile == 95)]
        for r in at.itertuples():
            print(f"    {r.grouping:22s} n {r.n_set!s:>6} | age {r.mean_age:5.1f}, female {100 * r.share_female:5.1f}% | "
                  f"access {100 * r.access_rate:5.1f}%, cost {100 * r.cost_rate:5.1f}%, avoid {100 * r.avoid_rate:5.1f}%, "
                  f"ED {100 * r.ed_mean:5.2f}%")
    withheld = int(D[[c for c in D.columns if c.endswith("_rate")]].isna().sum().sum())
    print(f"  rates withheld under the policy: {withheld} of {3 * (len(D))}")

    json.dump({"cdr": CDR, "cuts": CUTS, "slots": ENCODERS, "outcomes": {**RATES, "ed": "ed_share"},
               "groupings_all": [{"grouping": g, "family": f, "k": k, "k_rule": r} for g, f, k, r in G12],
               "exclusive_groupings": "the slot's encoder plus " + ", ".join(others),
               "adjustment": "none (unadjusted); mean_age, sd_age, share_female written beside the rates",
               "intervals": "Wilson 95% for the rates, normal 95% for the ED mean",
               "sex_coding": "female = 0, male = 1", "disclosure": "All of Us dissemination policy (disclosure.py)"},
              open(os.path.join(OUT, "care_sweep_meta.json"), "w"), indent=2)
    files = ["care_sweep.csv", "care_sweep_meta.json"]
    for f in files:
        ok, why = verdict(os.path.join(OUT, f))
        assert ok, f"{f} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"care_sweep_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(OUT, f), arcname=f"care_sweep/{f}")
    print(f"\n[care_sweep] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
