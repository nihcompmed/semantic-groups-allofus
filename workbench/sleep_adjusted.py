#!/usr/bin/env python3
"""sleep_adjusted.py -- mean sleep time and sleep variability among the outliers and the exclusive
outliers, adjusted for age and sex against a reference curve fitted once on the whole sleep cohort.

WHY THIS WAY. The sets differ in age (their sleep samples average 42 to 58 years against the cohort's
55) and in sex, and both move sleep. Stratifying inside each set fails where the sets are small (44 to
316 people with sleep data among the exclusive sets at the 95th), because strata go empty. So the
expected value for a person's age and sex is estimated ONCE, on everyone with sleep data, and each
set contributes only its people's gaps from that expectation.

THE REFERENCE CURVES, fitted on everyone in the cohort with at least 14 nights
  mean sleep time     least squares:    mean_min ~ 1 + age + age^2 + sex
  sleep variability   median regression: sd_min ~ 1 + age + age^2 + sex + log(nights)
                      log(nights) for variability only: a person's SD is estimated low
                      from few nights, and the sets differ in how many nights their people have.
                      A median regression because an SD is right skewed (explore_other_data.py, cells 9 to 12).
  age is centered and scaled before squaring, for numerical stability only; sex is 0 female, 1 male.
  A person's GAP = their value minus the curve's value for them.

PER SET (the sets of breadth_sweep.py and breadth_exclusive.py, through their own code)
  sleep_mean (_lo, _hi)         the cohort's mean + the set's mean gap; normal 95% interval of the gaps
  sleep_var_median (_lo, _hi)   the cohort's median SD + (the set's median gap - the cohort's median
                                gap); order-statistic 95% interval of the gaps, shifted the same way
  n_set, n_sleep                as in sleep_sweep.py
The cohort row therefore equals the cohort's own unadjusted values, so the reference line does not
move. The interval treats the curve as known; with 16,732 people its own uncertainty is negligible.
The column names are sleep_sweep.py's.

THE DISSEMINATION POLICY (disclosure.py): as in sleep_sweep.py. The fitted
coefficients are aggregates over 16,732 people.

INPUTS   sleep_person_local.csv (cell 9), the responses file (age, sex), everything breadth_sweep.py reads
OUTPUTS  screen_out/sleep_adjusted/sleep_adjusted.csv         the same layout as sleep_sweep.csv
         screen_out/sleep_adjusted/sleep_reference_curves.csv  both models' coefficients and SEs
         screen_out/sleep_adjusted/sleep_adjusted_meta.json
         screen_out/sleep_adjusted_<CDR>.zip                    those 3, checked by verdict() in
                                                                zip_screen_aggregates.py

Run:  python3 sleep_adjusted.py
"""
import json
import os
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from breadth_exclusive import rank_orders, top_and_exclusive           # noqa: E402  the same people
from breadth_sweep import CUTS, ENCODERS, groupings                      # noqa: E402
from disclosure import MARK, SUPP, pair_hidden, small                   # noqa: E402  the dissemination policy
from sleep_reliability import MIN_NIGHTS, sleep_cache                   # noqa: E402  one rule, one file
from sleep_sweep import median_ci                                       # noqa: E402
from wb_config import CFG                                               # noqa: E402
from zip_screen_aggregates import verdict                               # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "sleep_adjusted")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
Z = 1.959964
STATS = ["sleep_mean", "sleep_lo", "sleep_hi", "sleep_var_median", "sleep_var_lo", "sleep_var_hi"]


def fit_curves(age, sex, nights, mean_min, sd_min):
    """The 2 reference curves on the sleep cohort. Returns the gaps and a coefficient table."""
    a = (age - age.mean()) / age.std()
    X1 = sm.add_constant(np.column_stack([a, a ** 2, sex]))
    ols = sm.OLS(mean_min, X1).fit()
    X2 = sm.add_constant(np.column_stack([a, a ** 2, sex, np.log(nights)]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qr = sm.QuantReg(sd_min, X2).fit(q=0.5, max_iter=5000)
    converged = not any("iteration" in str(w.message).lower() for w in caught)
    names1 = ["intercept", "age_z", "age_z_squared", "sex_male"]
    names2 = names1 + ["log_nights"]
    C = pd.DataFrame(
        [{"model": "mean sleep time (OLS)", "term": t, "coef": b, "se": s} for t, b, s in zip(names1, ols.params, ols.bse)]
        + [{"model": "sleep variability (median regression)", "term": t, "coef": b, "se": s}
           for t, b, s in zip(names2, qr.params, qr.bse)])
    C["age_mean"], C["age_sd"], C["n_people"], C["converged"] = age.mean(), age.std(), len(age), converged
    return mean_min - ols.fittedvalues, sd_min - qr.fittedvalues, C, converged


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
    if hidden or k < 2:                          # ★ a statistic over a suppressed count would return it (disclosure.py)
        return r
    g1 = A["gap_mean"][s]
    m, se = float(g1.mean()), float(np.std(g1, ddof=1) / np.sqrt(k))
    r.update({"sleep_mean": A["coh_mean"] + m, "sleep_lo": A["coh_mean"] + m - Z * se,
              "sleep_hi": A["coh_mean"] + m + Z * se})
    med, lo, hi = median_ci(A["gap_sd"][s])
    shift = A["coh_sd_median"] - A["coh_gap_sd_median"]
    r.update({"sleep_var_median": shift + med, "sleep_var_lo": shift + lo, "sleep_var_hi": shift + hi})
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)         # the order every breadth script uses
    N = len(idx)
    R = pd.read_csv(CFG.responses_csv, usecols=[ID_COL, "age", "sex"])
    R[ID_COL] = R[ID_COL].astype(str)
    R = R.set_index(ID_COL).reindex(idx)
    assert set(np.unique(R["sex"])) <= {0.0, 1.0}, "sex is not coded 0/1 as build_responses_v9.py writes it"
    P = pd.read_csv(sleep_cache())
    P["person_id"] = P["person_id"].astype(str)
    P = P.set_index("person_id").reindex(idx)
    has = (P["nights"].fillna(0).values >= MIN_NIGHTS) & P["sd_min"].notna().values

    age, sex = R["age"].values.astype(float), R["sex"].values.astype(float)
    nights, mean_min, sd_min = (P[c].values.astype(float) for c in ("nights", "mean_min", "sd_min"))
    g1, g2, C, converged = fit_curves(age[has], sex[has], nights[has], mean_min[has], sd_min[has])
    gap_mean, gap_sd = np.full(N, np.nan), np.full(N, np.nan)
    gap_mean[has], gap_sd[has] = g1, g2
    A = {"has_sleep": has, "gap_mean": gap_mean, "gap_sd": gap_sd,
         "coh_mean": float(mean_min[has].mean()), "coh_sd_median": float(np.median(sd_min[has])),
         "coh_gap_sd_median": float(np.median(g2))}
    print(f"[sleep_adjusted] reference curves on {int(has.sum()):,} people with at least {MIN_NIGHTS} nights "
          f"| median regression converged: {converged}")
    print(C[["model", "term", "coef", "se"]].round(3).to_string(index=False))

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
    D.to_csv(os.path.join(OUT, "sleep_adjusted.csv"), index=False)
    C.to_csv(os.path.join(OUT, "sleep_reference_curves.csv"), index=False)

    c = rows[0]
    print(f"  cohort (unchanged by construction): mean sleep {c['sleep_mean']:.1f} min, variability {c['sleep_var_median']:.1f} min")
    for analysis, slot in (("all", ""), ("exclusive", ENCODERS[0])):
        print(f"  at the 95th, {analysis}{' (slot ' + slot + ')' if slot else ''}, age- and sex-adjusted:")
        at = D[(D.analysis == analysis) & (D.encoder_slot == slot) & (D.cut_pctile == 95)]
        for r in at.itertuples():
            print(f"    {r.grouping:22s} with sleep {r.n_sleep!s:>5} | mean {r.sleep_mean:6.1f} min "
                  f"[{r.sleep_lo:6.1f}, {r.sleep_hi:6.1f}] | variability {r.sleep_var_median:5.1f} min "
                  f"[{r.sleep_var_lo:5.1f}, {r.sleep_var_hi:5.1f}]")

    json.dump({"cdr": CDR, "cuts": CUTS, "slots": ENCODERS, "min_nights": MIN_NIGHTS,
               "adjustment": {"mean sleep time": "OLS on age, age^2, sex, fitted on the whole sleep cohort",
                              "sleep variability": "median regression on age, age^2, sex, log(nights), same cohort"},
               "median_regression_converged": converged,
               "groupings_all": [{"grouping": g, "family": f, "k": k, "k_rule": r} for g, f, k, r in G12],
               "exclusive_groupings": "the slot's encoder plus " + ", ".join(others),
               "disclosure": "All of Us dissemination policy (disclosure.py)"},
              open(os.path.join(OUT, "sleep_adjusted_meta.json"), "w"), indent=2)
    files = ["sleep_adjusted.csv", "sleep_reference_curves.csv", "sleep_adjusted_meta.json"]
    for f in files:
        ok, why = verdict(os.path.join(OUT, f))
        assert ok, f"{f} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"sleep_adjusted_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(OUT, f), arcname=f"sleep_adjusted/{f}")
    print(f"\n[sleep_adjusted] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
