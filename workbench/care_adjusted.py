#!/usr/bin/env python3
"""care_adjusted.py -- care and access barriers among the outliers and the exclusive outliers,
adjusted for age and sex against a reference curve fitted once on the whole cohort.

WHY THIS WAY. Direct standardization (each set's own rate in 8 sex x age cells, reweighted to the
cohort) needs people of every age and sex in every set, and the small sets do not have them (exclusive
sets of 113 to 340 people at the 99th; MCA 90's at the 95th averages 42 years), so cells go empty and a direct rate cannot be formed. So, as in sleep_adjusted.py, the
expected value for a person's age and sex is estimated ONCE, on everyone the outcome is measured on,
and each set contributes only its people's gaps from that expectation. A curve rather than age bands,
so that care and sleep are adjusted the same way.

THE REFERENCE CURVES, one per outcome, by least squares as for mean sleep time:
  outcome ~ 1 + age + age^2 + sex
  access, cost, avoid   0 or 1                 fitted on everyone who answered the access survey
  ed                    the person's ED share  fitted on everyone with at least 1 visit
  age is centered and scaled on the fitting sample before squaring, for numerical stability only;
  sex is 0 female, 1 male. A person's GAP = their value minus the curve's value for them.

PER SET (the sets of breadth_sweep.py and breadth_exclusive.py, through their own code)
  access_rate, cost_rate, avoid_rate, ed_mean (each with _lo, _hi)
      the cohort's value + the set's mean gap, with a normal 95% interval of the gaps. The curve has
      an intercept, so the gaps average 0 over the cohort and the cohort row equals the unadjusted
      cohort value: the reference line does not move (checked below). The interval treats the curve
      as known; with more than 86,000 people its own uncertainty is negligible.
  n_set, mean_age, sd_age, share_female, n_survey, n_ehr      as in care_sweep.py, unchanged
The column names are care_sweep.py's.

THE DISSEMINATION POLICY (disclosure.py). Every row goes first through care_sweep.py's
own describe(), which applies the policy to the unadjusted statistics. An adjusted value is written only
where the unadjusted one survived, so a statistic withheld there is withheld here, for the same people.
The fitted coefficients are aggregates over more than 86,000 people.

THE FIT AT 65, PRINTED ONLY. Medicare begins at 65, and a smooth curve in age cannot follow a step.
The run prints, for each outcome, the cohort's observed value and the curve's mean value by 5-year age
band and sex, so the fit can be judged on the Workbench. Printed output stays there, and this table
is not in the download.

INPUTS   screen_out/care_outcomes_per_person.csv   (care.py --outcomes-only; per person, stays)
         the responses file (age, sex), and everything breadth_sweep.py reads
OUTPUTS  screen_out/care_adjusted/care_adjusted.csv            the same layout as care_sweep.csv
         screen_out/care_adjusted/care_reference_curves.csv    the 4 curves' coefficients and SEs
         screen_out/care_adjusted/care_adjusted_meta.json
         screen_out/care_adjusted_<CDR>.zip                     those 3, checked by verdict() in
                                                                zip_screen_aggregates.py

Run:  python3 care_adjusted.py
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from breadth_exclusive import rank_orders, top_and_exclusive           # noqa: E402  the same people
from breadth_sweep import CUTS, ENCODERS, groupings                      # noqa: E402
from care_sweep import FEMALE, RATES, describe as describe_unadjusted   # noqa: E402  the same statistics and policy
from disclosure import MARK                                             # noqa: E402  the dissemination policy
from wb_config import CFG                                               # noqa: E402
from zip_screen_aggregates import verdict                               # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "care_adjusted")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
Z = 1.959964
# outcome -> (column in the outcomes file, who it is measured on, the value / low / high columns written)
OUTCOMES = {**{s: (c, "has_survey", (f"{s}_rate", f"{s}_lo", f"{s}_hi")) for s, c in RATES.items()},
            "ed": ("ed_share", "has_ehr", ("ed_mean", "ed_lo", "ed_hi"))}
BAND_EDGES = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, np.inf]   # the printed fit check only


def fit_curve(age, sex, y):
    """1 reference curve, y ~ 1 + age + age^2 + sex by least squares. Returns the fitted values and
    the coefficient rows."""
    a = (age - age.mean()) / age.std()
    fit = sm.OLS(y, sm.add_constant(np.column_stack([a, a ** 2, sex]))).fit()
    terms = ["intercept", "age_z", "age_z_squared", "sex_male"]
    rows = [{"term": t, "coef": b, "se": s, "age_mean": age.mean(), "age_sd": age.std(), "n_people": len(y)}
            for t, b, s in zip(terms, fit.params, fit.bse)]
    return np.asarray(fit.fittedvalues), rows


def describe(mask, A, n_top=None):
    """care_sweep.py's row, with each surviving statistic replaced by its adjusted value."""
    r = describe_unadjusted(mask, A, n_top)
    for short, (_, has, (v, lo, hi)) in OUTCOMES.items():
        if pd.isna(r.get(v, np.nan)):           # withheld or not reached unadjusted: withheld here too
            continue
        g = A[f"gap_{short}"][mask & A[has]]
        m, se = float(g.mean()), float(np.std(g, ddof=1) / np.sqrt(len(g)))
        c = A[f"coh_{short}"]
        r.update({v: c + m, lo: c + m - Z * se, hi: c + m + Z * se})
    return r


def print_band_fit(A):
    """The cohort's observed value beside the curve's, by 5-year age band and sex. Printed only."""
    for short, (col, has, _) in OUTCOMES.items():
        m = A[has]
        d = pd.DataFrame({"band": pd.cut(A["age"][m], BAND_EDGES, right=False), "female": A["sex"][m] == FEMALE,
                          "obs": 100 * A[col][m], "fit": 100 * A[f"fit_{short}"][m]})
        print(f"  {short} (%): observed and curve, by age band | female obs, fit | male obs, fit")
        for band, g in d.groupby("band", observed=True):
            f, mm = g[g.female], g[~g.female]
            print(f"    {str(band):>12}  {f.obs.mean():6.2f} {f.fit.mean():6.2f} | {mm.obs.mean():6.2f} {mm.fit.mean():6.2f}")


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

    coef = []
    for short, (col, has, _) in OUTCOMES.items():
        m = A[has]
        y = A[col][m]
        assert np.isfinite(y).all(), f"{col} is missing for someone it is measured on"
        fitted, rows = fit_curve(A["age"][m], A["sex"][m], y)
        A[f"fit_{short}"], A[f"gap_{short}"] = np.full(N, np.nan), np.full(N, np.nan)
        A[f"fit_{short}"][m], A[f"gap_{short}"][m] = fitted, y - fitted
        A[f"coh_{short}"] = float(y.mean())
        coef += [{"outcome": short, **r} for r in rows]
        out = int(((fitted < 0) | (fitted > 1)).sum())
        print(f"[care_adjusted] {short}: curve fitted on {int(m.sum()):,} | fitted values outside [0, 1]: {out}")
    C = pd.DataFrame(coef)
    print(C[["outcome", "term", "coef", "se"]].round(4).to_string(index=False))
    print_band_fit(A)

    G12 = groupings()
    order = rank_orders([g[0] for g in G12], idx)
    info = {g[0]: g for g in G12}
    names12 = [g[0] for g in G12]
    others = [g[0] for g in G12 if g[0] not in ENCODERS]
    rows = [{"analysis": "cohort", "encoder_slot": "", "grouping": "cohort", "family": "reference",
             "k": np.nan, "cut_pctile": 0, **describe(np.ones(N, bool), A)}]
    # the gaps average 0 over the cohort, so the adjusted cohort row is the unadjusted one
    unadj = describe_unadjusted(np.ones(N, bool), A)
    for short, (_, _, (v, _, _)) in OUTCOMES.items():
        assert abs(rows[0][v] - unadj[v]) < 1e-9, f"the cohort's adjusted {short} moved from its unadjusted value"
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
    for _, _, cols in OUTCOMES.values():
        for c in cols:
            if c not in D.columns:
                D[c] = np.nan
    D.to_csv(os.path.join(OUT, "care_adjusted.csv"), index=False)
    C.to_csv(os.path.join(OUT, "care_reference_curves.csv"), index=False)

    coh = rows[0]
    print(f"  cohort (unchanged by construction): access {100 * coh['access_rate']:.1f}%, cost "
          f"{100 * coh['cost_rate']:.1f}%, avoid {100 * coh['avoid_rate']:.1f}%, ED share {100 * coh['ed_mean']:.2f}%")
    for analysis, slot in (("all", ""), ("exclusive", ENCODERS[0])):
        print(f"  at the 95th, {analysis}{' (slot ' + slot + ')' if slot else ''}, age- and sex-adjusted:")
        at = D[(D.analysis == analysis) & (D.encoder_slot == slot) & (D.cut_pctile == 95)]
        for r in at.itertuples():
            print(f"    {r.grouping:22s} n {r.n_set!s:>6} | access {100 * r.access_rate:5.1f}%, cost "
                  f"{100 * r.cost_rate:5.1f}%, avoid {100 * r.avoid_rate:5.1f}%, ED {100 * r.ed_mean:5.2f}%")
    withheld = int(D[[c for c in D.columns if c.endswith("_rate")]].isna().sum().sum())
    print(f"  rates withheld under the policy: {withheld} of {3 * len(D)}")

    json.dump({"cdr": CDR, "cuts": CUTS, "slots": ENCODERS, "outcomes": {**RATES, "ed": "ed_share"},
               "groupings_all": [{"grouping": g, "family": f, "k": k, "k_rule": r} for g, f, k, r in G12],
               "exclusive_groupings": "the slot's encoder plus " + ", ".join(others),
               "adjustment": {"access, cost, avoid": "OLS on age, age^2, sex, fitted on everyone who answered the access survey",
                              "ed": "OLS on age, age^2, sex, fitted on everyone with at least 1 visit",
                              "adjusted value": "the cohort's value + the set's mean gap from the curve"},
               "n_fit_survey": int(A["has_survey"].sum()), "n_fit_ehr": int(A["has_ehr"].sum()),
               "intervals": "normal 95% of the set's gaps, the curve treated as known",
               "withheld": f"wherever care_sweep.py withholds the unadjusted value ({MARK} sets included)",
               "sex_coding": "female = 0, male = 1", "disclosure": "All of Us dissemination policy (disclosure.py)"},
              open(os.path.join(OUT, "care_adjusted_meta.json"), "w"), indent=2)
    files = ["care_adjusted.csv", "care_reference_curves.csv", "care_adjusted_meta.json"]
    for f in files:
        ok, why = verdict(os.path.join(OUT, f))
        assert ok, f"{f} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"care_adjusted_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(OUT, f), arcname=f"care_adjusted/{f}")
    print(f"\n[care_adjusted] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
