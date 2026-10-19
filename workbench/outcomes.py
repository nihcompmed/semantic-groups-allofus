"""outcomes.py -- the per-person outcomes read back from work/, and the statistics of one set of participants.

Loaders, in the cohort's order (the order of work/thresholds.csv):
  load(idx)            B thresholds met, C barriers to care and the emergency department share, S Fitbit sleep, SA the
                       sleep outcomes adjusted by their reference curves; each outcome's reference curve fitted once
  load_conditions(idx) the 8 recorded conditions, each 0/1 inside its population and missing outside it

The recorded conditions: 1 = at least 2 coded events mapped to the phecode (step 4), so a single event counts as not
recorded. Every label is the same in every coding year (no code moves in or out under any official change since
October 2016, after the 3 rows step 4 removes from the map).
  A  matching a threshold   generalized anxiety disorder MB_288.3 (GAD-7), ADHD MB_304 (ASRS)
  B  named disorders        posttraumatic stress disorder MB_290.1, panic disorder MB_288.2
  C  physical               hypercholesterolemia EM_239.1, asthma RE_475, breast cancer CA_105, prostate cancer CA_107.2
Populations: participants with at least 1 ICD-coded event; breast cancer women only, prostate cancer men only; for the 3
conditions with a test both cases and non-cases are restricted to participants whose record shows the test, so a 0
means "tested, no diagnosis recorded".

Statistics of a set (describe functions), each applying the dissemination policy to what it writes: unadjusted rates
with Wilson 95% intervals and means with normal intervals; adjusted values = the cohort's value + the set's mean gap
from the reference curve, with a normal 95% interval of the gaps (the curve treated as known).
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from release import MARK, SUPP, pair_hidden, small
from settings import COHORT_CSV, FEMALE, ID, MALE, MIN_NIGHTS, Z, work

RATES = {"access": "any_access_barrier", "cost": "any_cost_barrier", "avoid": "care_avoid_provider"}
CARE_OUTCOMES = {**{s: (c, "has_survey", (f"{s}_rate", f"{s}_lo", f"{s}_hi")) for s, c in RATES.items()},
                 "ed": ("ed_share", "has_ehr", ("ed_mean", "ed_lo", "ed_hi"))}
MIN_COUNT = 2                                   # a recorded condition needs 2 coded events
# key, phecode, label, list, men only
CONDITIONS = [
    ("gad", "MB_288.3", "Generalized anxiety disorder", "A", False),
    ("adhd", "MB_304", "Attention-deficit hyperactivity disorder", "A", False),
    ("ptsd", "MB_290.1", "Posttraumatic stress disorder", "B", False),
    ("panic", "MB_288.2", "Panic disorder", "B", False),
    ("hyperchol", "EM_239.1", "Hypercholesterolemia", "C", False),
    ("asthma", "RE_475", "Asthma", "C", False),
    ("breast", "CA_105", "Breast cancer", "C", False),
    ("prostate", "CA_107.2", "Prostate cancer", "C", True),
]
CODE = {c[0]: c[1] for c in CONDITIONS}
SEX_ONLY = {"breast": FEMALE, "prostate": MALE}
TESTED_BY = {"hyperchol": "lipid", "breast": "breast", "prostate": "prostate"}
TESTS = ["lipid", "breast", "prostate"]


def with_sex(key):
    """Whether a condition's models carry a sex term (not when its population is one sex)."""
    return key not in SEX_ONLY


# ------------------------------------------------------------------------------------------------- statistics
def wilson(k, n):
    """A rate and its Wilson 95% interval."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / d
    return p, c - h, c + h


def wilson_share(k, n):
    """A share and its Wilson 95% interval, the bounds kept on the right side of the share (at k = 0 the lower bound
    is 0 up to rounding)."""
    p = k / n
    d = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / d
    return p, min(c - h, p), max(c + h, p)


def stats(b):
    """Mean, SD, SE and the normal 95% interval of the thresholds met over the evaluable people."""
    n = len(b)
    if n < 2:
        return dict(mean_breadth=np.nan, sd_breadth=np.nan, se=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    m, s = float(b.mean()), float(b.std(ddof=1))
    se = s / np.sqrt(n)
    return dict(mean_breadth=m, sd_breadth=s, se=se, ci_lo=m - 1.96 * se, ci_hi=m + 1.96 * se)


def median_ci(x):
    """The median and its distribution-free 95% interval from order statistics."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    lo = max(int(np.floor(n / 2 - Z * np.sqrt(n) / 2)), 1)
    hi = min(int(np.ceil(n / 2 + 1 + Z * np.sqrt(n) / 2)), n)
    return float(np.median(x)), float(x[lo - 1]), float(x[hi - 1])


def fit_curve(age, sex, y, with_sex=True):
    """A reference curve, y ~ 1 + age + age^2 (+ sex) by least squares, age centered and scaled on the fitting
    sample. Returns the fitted values and the coefficient rows."""
    a = (age - age.mean()) / age.std()
    cols = [a, a ** 2] + ([sex] if with_sex else [])
    fit = sm.OLS(y, sm.add_constant(np.column_stack(cols))).fit()
    terms = ["intercept", "age_z", "age_z_squared"] + (["sex_male"] if with_sex else [])
    rows = [{"term": t, "coef": b, "se": s, "age_mean": age.mean(), "age_sd": age.std(), "n_people": len(y)}
            for t, b, s in zip(terms, fit.params, fit.bse)]
    return np.asarray(fit.fittedvalues), rows


def fit_sleep_curves(age, sex, nights, mean_min, sd_min):
    """The 2 sleep reference curves: mean sleep ~ age + age^2 + sex by least squares; night-to-night SD ~ age + age^2 +
    sex + log(nights) by median regression. Returns the 2 gaps, the coefficients and whether the median regression
    converged."""
    a = (age - age.mean()) / age.std()
    X1 = sm.add_constant(np.column_stack([a, a ** 2, sex]))
    ols = sm.OLS(mean_min, X1).fit()
    X2 = sm.add_constant(np.column_stack([a, a ** 2, sex, np.log(nights)]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qr = sm.QuantReg(sd_min, X2).fit(q=0.5, max_iter=5000)
    converged = not any("iteration" in str(w.message).lower() for w in caught)
    names1 = ["intercept", "age_z", "age_z_squared", "sex_male"]
    C = pd.DataFrame(
        [{"model": "mean sleep time (OLS)", "term": t, "coef": b, "se": s} for t, b, s in zip(names1, ols.params, ols.bse)]
        + [{"model": "sleep variability (median regression)", "term": t, "coef": b, "se": s}
           for t, b, s in zip(names1 + ["log_nights"], qr.params, qr.bse)])
    C["age_mean"], C["age_sd"], C["n_people"], C["converged"] = age.mean(), age.std(), len(age), converged
    return mean_min - ols.fittedvalues, sd_min - qr.fittedvalues, C, converged


# ------------------------------------------------------------------------------------------------- one set
def describe_care(mask, A, n_top=None):
    """Age, sex, the 3 barrier rates and the emergency department share of one set, unadjusted, under the policy."""
    n = int(mask.sum())
    r = {"n_set": n}
    if n == 0:
        return r
    if small(n):
        r["n_set"] = MARK
        return r
    if n_top is not None and pair_hidden(n, n_top):
        r["n_set"] = SUPP
    if r["n_set"] == SUPP:
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


def describe_care_adjusted(mask, A, n_top=None):
    """describe_care, each written statistic replaced by its value adjusted for age and sex."""
    r = describe_care(mask, A, n_top)
    for short, (_, has, (v, lo, hi)) in CARE_OUTCOMES.items():
        if pd.isna(r.get(v, np.nan)):
            continue
        g = A[f"gap_{short}"][mask & A[has]]
        m, se = float(g.mean()), float(np.std(g, ddof=1) / np.sqrt(len(g)))
        c = A[f"coh_{short}"]
        r.update({v: c + m, lo: c + m - Z * se, hi: c + m + Z * se})
    return r


def describe_sleep(mask, A, n_top=None):
    """Sleep in one set: coverage, mean sleep with its interval, the median night-to-night SD with its interval, the
    age of the sleep sample and its share on program-provided devices, under the policy."""
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
    if hidden or k < 2:
        return r
    m = A["mean_min"][s]
    se = float(np.std(m, ddof=1) / np.sqrt(k))
    r.update({"sleep_mean": float(m.mean()), "sleep_lo": float(m.mean()) - Z * se, "sleep_hi": float(m.mean()) + Z * se})
    med, lo, hi = median_ci(A["sd_min"][s])
    r.update({"sleep_var_median": med, "sleep_var_lo": lo, "sleep_var_hi": hi, "sleep_age": float(A["age"][s].mean())})
    if A["wear"] is not None:
        kw = int((s & A["wear"]).sum())
        r["share_wear"] = np.nan if pair_hidden(kw, k) else kw / k
    return r


def describe_sleep_adjusted(mask, A, n_top=None):
    """Sleep in one set adjusted for age and sex (and nights, for the SD), under describe_sleep's rules."""
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
    if hidden or k < 2:
        return r
    g1 = A["gap_mean"][s]
    m, se = float(g1.mean()), float(np.std(g1, ddof=1) / np.sqrt(k))
    r.update({"sleep_mean": A["coh_mean"] + m, "sleep_lo": A["coh_mean"] + m - Z * se,
              "sleep_hi": A["coh_mean"] + m + Z * se})
    med, lo, hi = median_ci(A["gap_sd"][s])
    shift = A["coh_sd_median"] - A["coh_gap_sd_median"]
    r.update({"sleep_var_median": shift + med, "sleep_var_lo": shift + lo, "sleep_var_hi": shift + hi})
    return r


# ------------------------------------------------------------------------------------------------- loaders
def cohort_order():
    """The cohort's ids, in the order every step uses."""
    cf = pd.read_csv(work("thresholds.csv"), usecols=[ID])
    return pd.Index(cf[ID].astype(str), name=ID)


def read_sleep(idx):
    P = pd.read_csv(work("sleep_person.csv"))
    P["person_id"] = P["person_id"].astype(str)
    return P.set_index("person_id").reindex(idx)


def wear_mask(idx):
    return idx.isin(set(pd.read_csv(work("wear_enrollees.csv"))["person_id"].astype(str)))


def load(idx):
    """The per-person inputs of the decile analyses, in idx order, and their reference curves fitted once:
    B (thresholds met), C (care), S (sleep), SA (sleep, adjusted), and whether the median regression converged."""
    N = len(idx)
    cf = pd.read_csv(work("thresholds.csv"))
    cf[ID] = cf[ID].astype(str)
    cf = cf.set_index(ID).reindex(idx)
    R = pd.read_csv(COHORT_CSV, usecols=[ID, "age", "sex"])
    R[ID] = R[ID].astype(str)
    R = R.set_index(ID).reindex(idx)
    assert R.notna().all().all(), "age or sex missing for someone in the cohort"
    assert set(np.unique(R["sex"])) <= {0.0, 1.0}, "sex is not coded 0/1"
    age, sex = R["age"].values.astype(float), R["sex"].values.astype(float)

    O = pd.read_csv(work("care_outcomes.csv"))
    O[ID] = O[ID].astype(str)
    O = O.set_index(ID).reindex(idx)
    C = {"age": age, "sex": sex,
         "has_survey": O["has_survey"].fillna(False).astype(bool).values,
         "has_ehr": O["has_ehr"].fillna(False).astype(bool).values,
         "ed_share": O["ed_share"].values.astype(float)}
    for col in RATES.values():
        C[col] = O[col].values.astype(float)
    for short, (col, has, _) in CARE_OUTCOMES.items():
        m = C[has]
        y = C[col][m]
        assert np.isfinite(y).all(), f"{col} is missing for someone it is measured on"
        fitted, _ = fit_curve(age[m], sex[m], y)
        C[f"gap_{short}"] = np.full(N, np.nan)
        C[f"gap_{short}"][m] = y - fitted
        C[f"coh_{short}"] = float(y.mean())

    P = read_sleep(idx)
    has = (P["nights"].fillna(0).values >= MIN_NIGHTS) & P["sd_min"].notna().values
    wear = wear_mask(idx)
    nights, mean_min, sd_min = (P[c].values.astype(float) for c in ("nights", "mean_min", "sd_min"))
    S = {"age": age, "has_sleep": has, "mean_min": mean_min, "sd_min": sd_min, "wear": wear}
    g1, g2, _, converged = fit_sleep_curves(age[has], sex[has], nights[has], mean_min[has], sd_min[has])
    gap_mean, gap_sd = np.full(N, np.nan), np.full(N, np.nan)
    gap_mean[has], gap_sd[has] = g1, g2
    SA = {"has_sleep": has, "gap_mean": gap_mean, "gap_sd": gap_sd,
          "coh_mean": float(mean_min[has].mean()), "coh_sd_median": float(np.median(sd_min[has])),
          "coh_gap_sd_median": float(np.median(g2))}
    ev, br = cf["all_evaluable"].astype(bool).values, cf["breadth"].astype(float).values
    fitted, _ = fit_curve(age[ev], sex[ev], br[ev])
    gap_br = np.full(N, np.nan)
    gap_br[ev] = br[ev] - fitted
    B = {"ev": ev, "br": br, "gap": gap_br, "coh": float(br[ev].mean())}
    return B, C, S, SA, converged


def load_conditions(idx):
    """Per person, in idx order: age, sex, has_rec (at least 1 ICD event), tested_<test>, and per condition den_<key>
    (its population) and y_<key> (0/1 inside the population, NaN outside)."""
    R = pd.read_csv(COHORT_CSV, usecols=[ID, "age", "sex"])
    R[ID] = R[ID].astype(str)
    R = R.set_index(ID).reindex(idx)
    assert R.notna().all().all(), "age or sex missing for someone in the cohort"
    P = pd.read_csv(work("phetk", "icd_person_summary.csv"), dtype={"person_id": str}).set_index("person_id").reindex(idx)
    has_rec = P["n_events"].fillna(0).values > 0
    C = pd.read_csv(work("phetk", "phecode_counts.tsv"), sep="\t", dtype={"person_id": str, "phecode": str},
                    usecols=["person_id", "phecode", "count"])
    C = C[C["phecode"].isin(CODE.values()) & C["person_id"].isin(idx)]
    T = pd.read_csv(work("test_checks.csv"), dtype={ID: str})
    A = {"age": R["age"].values.astype(float), "sex": R["sex"].values.astype(float), "has_rec": has_rec}
    for kind in TESTS:
        A[f"tested_{kind}"] = idx.isin(T.loc[T["kind"] == kind, ID])
    for key, code, *_ in CONDITIONS:
        cnt = C[C["phecode"] == code].set_index("person_id")["count"].reindex(idx).fillna(0).values
        den = has_rec.copy()
        if key in SEX_ONLY:
            den &= A["sex"] == SEX_ONLY[key]
        if key in TESTED_BY:
            den &= A[f"tested_{TESTED_BY[key]}"]
        A[f"den_{key}"] = den
        A[f"y_{key}"] = np.where(den, (cnt >= MIN_COUNT).astype(float), np.nan)
    return A
