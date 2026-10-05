#!/usr/bin/env python3
"""care_records.py -- the AMOUNT half: does recorded care track the amount of burden?

WHAT THIS IS. Three of the four care outcomes are answers the participant gave on the same
platform as the 151 items. This one is not. It counts outpatient visits in the record in a window that
opens when the participant finished answering, so nothing about the outcome passes through the
participant's own reporting.

WHY THE OUTCOME IS A COUNT WITH AN OFFSET. The follow-up is short and uneven: median 215 days to the
end of the observation period, and 36.6% of the cohort have a record that ends on or before their
answer date. Of the four candidate outcomes declared before the run, the three that need a full 365-day
window to be observed were defined for only 556 to 605 of the 2,177 most burdened and failed the
coverage rule. The count with follow-up as the offset was defined for 1,314 and passed. The offset is
what absorbs uneven observation: a person watched for 200 days is credited with 200 days of
opportunity, not a year.

  outcome    DISTINCT DAYS with an outpatient contact in the 365 days after the answer date, over
             OP_CONCEPTS below. Days rather than rows because one encounter can be recorded under more
             than one visit concept (9202 Outpatient Visit and 581477 Office Visit on the same day),
             and a row count would count it twice. The run prints rows per day so the size of that is
             visible rather than assumed.
  offset     log(days observed inside that window) = log(min(365, follow-up)), follow-up being days
             from the answer date to the end of the observation period
  eligible   follow-up > 0. Everyone else has no window at all, and WHO they are is reported here
             rather than passed over, because it runs with burden
  exposure   burden, as ONE ENCODER'S OWN percentile from scores_<encoder>.csv, cut into deciles.
             --encoder picks the encoder; gte-large-en-v1.5, the prespecified one, is the default, and
             any other writes files suffixed _<encoder>.
  model      negative binomial, log link, with a Poisson fallback under robust errors if it will not
             converge. Adjusted visits per 365 days by decile come from marginal standardization with
             delta-method intervals, and the headline is the rate ratio of decile 10 against decile 1.

WHAT THIS CANNOT SAY. A lower count of outpatient visits can mean worse access or less need, and the
count does not separate them. The coverage rule chose this outcome on coverage rather than on how
cleanly it reads, on purpose, so that the outcome could not be chosen by its association. The
interpretation is correspondingly careful.

NOT THE SOURCE HALF. Whether recorded care differs by WHAT a participant's burden consists of needs a
per-person group-level exposure, which does not exist yet. This script does the amount question only.

INPUTS   responses_out_<stem>/responses_k5.csv, the item table (concept ids, for the answer dates),
         screen_out/scores_<enc>.csv, the Repository (observation, observation_period, visit_occurrence),
         screen_out/covariates_per_person.csv (from covariates.py; without it the model falls back to
         age and sex alone and says so, and that fallback is not for the paper)
OUTPUTS  (suffix _<encoder> for any encoder but gte-large-en-v1.5)
         screen_out/care_records_by_decile.csv      adjusted contact days per window per decile (LEAVES)
         screen_out/care_records_eligibility.csv    who has a window at all, per decile (LEAVES)
         screen_out/care_records_coefs.csv          coefficients (LEAVES)
         screen_out/care_records_summary.json       the model, its rate ratio, and the eligibility split
                                                    (LEAVES)

★ THE DISSEMINATION POLICY (disclosure.py). The decile counts partition the model n
and the eligibility counts are checked in pairs. A coefficient for a covariate level held by 1 to 20
people in the model is blanked, since it is a statistic about those people. The flagged set's
eligibility ("had a record ending on or before their answer date") is written under the same rule.

Run:  python3 care_records.py
      python3 care_records.py --encoder bge-m3  (another encoder)
      python3 care_records.py --window 180      (a shorter window; 365 is the declared one)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from google.cloud import bigquery

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_config import CFG, ENCODERS   # noqa: E402
from disclosure import SUPP, marked, pair_hidden, partition_mask   # noqa: E402  the dissemination policy

CDR = os.environ.get("WORKSPACE_CDR", "")
OUT_DIR = CFG.out_dir
ID_COL = "id"
WINDOW = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 365
# Matching explore_other_data.py, settled from the concept distribution this CDR carries before any
# outcome was computed on it. Telehealth counts as a contact; laboratory and pharmacy visits do not.
OP_CONCEPTS = [9202, 581477, 722455]
COVARIATES = ["race_ethnicity", "income", "education", "employment", "insurance"]
ENCODER = sys.argv[sys.argv.index("--encoder") + 1] if "--encoder" in sys.argv else "gte-large-en-v1.5"
assert ENCODER in ENCODERS, f"--encoder must be one of {ENCODERS}"
SUF = "" if ENCODER == "gte-large-en-v1.5" else f"_{ENCODER}"

client = bigquery.Client()


def Q(sql, label, params):
    cfg = bigquery.QueryJobConfig(query_parameters=params, dry_run=True, use_query_cache=False)
    gb = client.query(sql, job_config=cfg).total_bytes_processed / 1e9
    print(f"  [{label}: scanning {gb:,.1f} GB]")
    return client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()


def rank_z(v):
    r = pd.Series(np.asarray(v, float)).rank(pct=True).values
    return (r - r.mean()) / r.std()


def burden_percentile(ids):
    """One encoder's own percentile and flag."""
    f = os.path.join(OUT_DIR, f"scores_{ENCODER}.csv")
    assert os.path.exists(f), f"{f} not found -- run screen_battery.py first"
    S = pd.read_csv(f, usecols=[ID_COL, "pctile", "flagged"]).set_index(ID_COL)
    S.index = S.index.astype("int64")
    S = S.reindex(ids)
    return S["pctile"], S["flagged"].fillna(False).astype(bool)


def design(d, decile_ref=1):
    """Intercept, burden deciles against decile 1, age, sex, and the 5 categorical covariates."""
    X = pd.DataFrame(index=d.index)
    X["const"] = 1.0
    gcols = []
    for dd in sorted(d.decile.unique()):
        if dd == decile_ref:
            continue
        X[f"D{dd}"] = (d.decile == dd).astype(float)
        gcols.append(f"D{dd}")
    X["age_z"] = rank_z(d["age"])
    X["sex"] = d["sex"].astype(float)
    for c in [c for c in COVARIATES if c in d.columns]:
        s = d[c].astype(str)
        ref = s.value_counts().idxmax()
        for lv in sorted(s.unique()):
            if lv != ref:
                X[f"{c}={lv}"] = (s == lv).astype(float)
    return X, gcols


def fit_count(y, X, logt):
    """Negative binomial with a log offset, falling back to Poisson with robust errors."""
    try:
        res = sm.NegativeBinomial(y.values, X.values, offset=logt, loglike_method="nb2").fit(disp=0, maxiter=200)
        if np.all(np.isfinite(res.params)) and np.isfinite(res.llf):
            # NegativeBinomial carries alpha as the last parameter; drop it for the linear predictor
            k = X.shape[1]
            return res, res.params[:k], np.asarray(res.cov_params())[:k, :k], "negative binomial"
    except Exception as e:
        print(f"  [negative binomial did not converge ({type(e).__name__}); Poisson with robust errors]")
    res = sm.GLM(y.values, X.values, family=sm.families.Poisson(), offset=logt).fit(cov_type="HC0")
    return res, np.asarray(res.params), np.asarray(res.cov_params()), "Poisson (robust)"


def standardized_rate(beta, cov, X, gcols, level, window):
    """Everyone's predicted visits per `window` days with the decile set to `level`. Delta-method SE."""
    Xl = X.copy()
    for c in gcols:
        Xl[c] = 0.0
    if f"D{level}" in Xl.columns:
        Xl[f"D{level}"] = 1.0
    eta = Xl.values @ beta + np.log(window)
    mu = np.exp(eta)
    g = (mu[:, None] * Xl.values).mean(axis=0)
    m = float(mu.mean())
    se = float(np.sqrt(max(g @ cov @ g, 0.0)))
    return m, se, g


def main():
    assert CDR, "WORKSPACE_CDR is not set"
    ids = pd.read_csv(CFG.responses_csv, usecols=[ID_COL])[ID_COL].astype("int64")
    cids = pd.read_csv(CFG.items_csv, usecols=["concept_id"])["concept_id"].astype("int64").tolist()
    print(f"[care_records] cohort {len(ids):,} | window {WINDOW} days | "
          f"outpatient concepts {OP_CONCEPTS}")

    A = Q(f"""SELECT person_id, MAX(observation_datetime) AS last_answer
              FROM `{CDR}.observation`
              WHERE person_id IN UNNEST(@ids) AND observation_source_concept_id IN UNNEST(@cids)
              GROUP BY person_id""", "answer dates",
          [bigquery.ArrayQueryParameter("ids", "INT64", [int(i) for i in ids]),
           bigquery.ArrayQueryParameter("cids", "INT64", cids)]).set_index("person_id")
    adate = pd.to_datetime(A.last_answer).dt.tz_localize(None).reindex(ids.values)

    P = Q(f"""SELECT person_id, observation_period_end_date AS e
              FROM `{CDR}.observation_period` WHERE person_id IN UNNEST(@ids)""", "observation_period",
          [bigquery.ArrayQueryParameter("ids", "INT64", [int(i) for i in ids])]).set_index("person_id")
    fu = (pd.to_datetime(P.e, errors="coerce").reindex(ids.values) - adate).dt.days

    # The window closes at the declared length OR at the end of the observation period, whichever comes
    # first, so the numerator counts visits over exactly the days the offset credits. Without the cap a
    # visit recorded after the observation period ends would inflate the rate for short-observed people,
    # who are disproportionately the most burdened.
    ok = adate.notna().values & pd.to_datetime(P.e, errors="coerce").reindex(ids.values).notna().values
    edate = pd.to_datetime(P.e, errors="coerce").reindex(ids.values)
    V = Q(f"""WITH anchor AS (
                SELECT pid, adt, edt
                FROM UNNEST(@pids) AS pid WITH OFFSET i
                JOIN (SELECT adt, o2 FROM UNNEST(@adates) AS adt WITH OFFSET o2) ON i = o2
                JOIN (SELECT edt, o3 FROM UNNEST(@edates) AS edt WITH OFFSET o3) ON i = o3)
              SELECT a.pid AS person_id,
                     COUNT(DISTINCT IF(v.visit_concept_id IN UNNEST(@op)
                             AND v.visit_start_date BETWEEN a.adt
                             AND LEAST(DATE_ADD(a.adt, INTERVAL @w DAY), a.edt),
                             v.visit_start_date, NULL)) AS n_op,
                     COUNTIF(v.visit_concept_id IN UNNEST(@op)
                             AND v.visit_start_date BETWEEN a.adt
                             AND LEAST(DATE_ADD(a.adt, INTERVAL @w DAY), a.edt)) AS n_op_rows
              FROM anchor a JOIN `{CDR}.visit_occurrence` v ON v.person_id = a.pid
              GROUP BY a.pid""", "outpatient contact days in the window",
          [bigquery.ArrayQueryParameter("pids", "INT64", [int(i) for i in ids.values[ok]]),
           bigquery.ArrayQueryParameter("adates", "DATE", list(adate.dt.date.values[ok])),
           bigquery.ArrayQueryParameter("edates", "DATE", list(edate.dt.date.values[ok])),
           bigquery.ArrayQueryParameter("op", "INT64", OP_CONCEPTS),
           bigquery.ScalarQueryParameter("w", "INT64", WINDOW)]).set_index("person_id")
    n_op = V.n_op.reindex(ids.values).fillna(0).astype("int64")
    n_rows = V.n_op_rows.reindex(ids.values).fillna(0).astype("int64")
    # One clinical encounter can be recorded under more than one visit concept, so a row count
    # double-counts it. The outcome is DISTINCT DAYS with an outpatient contact. The ratio below says
    # how much that mattered: at 1.00 the two agree and rows would have been safe.
    tot_rows, tot_days = int(n_rows.sum()), int(n_op.sum())
    print(f"  rows {tot_rows:,} over {tot_days:,} distinct contact days "
          f"({tot_rows / max(tot_days, 1):.2f} rows per day). The outcome is DAYS.")

    pct, flg = burden_percentile(ids)
    print(f"  exposure: {ENCODER}'s own percentile (scores_{ENCODER}.csv)")
    D = pd.DataFrame({ID_COL: ids.values, "n_op": n_op.values, "fu": fu.values,
                      "pct": pct.values, "flagged": flg.values}).set_index(ID_COL)
    D["decile"] = (pd.Series(D.pct).rank(method="first").sub(1) * 10 // len(D) + 1).astype(int).values

    # --- who has a window at all, reported before the model ---------------------------------------
    D["eligible"] = D.fu > 0
    el = D.groupby("decile").eligible.agg(["size", "sum"])
    el["pct"] = (100 * el["sum"] / el["size"]).round(1)
    print(f"\nELIGIBLE (a record that runs past the answer date): {int(D.eligible.sum()):,} of "
          f"{len(D):,} ({100 * D.eligible.mean():.1f}%)")
    print("  by decile of burden: " + "  ".join(f"d{int(i)} {r.pct:.1f}%" for i, r in el.iterrows()))
    n_fl, n_fl_el = int(D.flagged.sum()), int((D.flagged & D.eligible).sum())
    n_fl_end = int((D.flagged & (D.fu <= 0)).sum())          # a record ending on or before the answer date
    n_end = int((D.fu <= 0).sum())
    print(f"  flagged under {ENCODER}: {n_fl_el:,} of {n_fl:,} eligible; a record ending on or before the "
          f"answer date for {n_fl_end:,} ({100 * n_fl_end / max(n_fl, 1):.1f}%) against "
          f"{100 * n_end / len(D):.1f}% of the cohort")
    print("  Absence is not neutral here. It runs with burden, and it is reported whatever the model")
    print("  below shows, because who has a usable record is itself a result.\n")

    E = D[D.eligible].copy()
    E["t"] = np.minimum(WINDOW, E.fu).astype(float)

    # Age and sex come from the response build. covariates_per_person.csv carries ONLY the 5
    # categorical covariates, so joining it for age and sex silently produces neither.
    R = pd.read_csv(CFG.responses_csv, usecols=[ID_COL, "age", "sex"])
    R[ID_COL] = pd.to_numeric(R[ID_COL]).astype("int64")
    E = E.join(R.set_index(ID_COL), how="inner")

    cpath = os.path.join(OUT_DIR, "covariates_per_person.csv")
    if os.path.exists(cpath):
        C = pd.read_csv(cpath)
        C[ID_COL] = pd.to_numeric(C[ID_COL]).astype("int64")
        C = C.set_index(ID_COL)
        keep = [c for c in COVARIATES if c in C.columns]
        missing = [c for c in COVARIATES if c not in C.columns]
        if missing:
            print(f"  [covariates_per_person.csv has no {missing}; those are left out of the model]")
        E = E.join(C[keep], how="left")
        # covariates.py maps every non-answer to its own level, so a gap here means a person absent
        # from that file rather than an unanswered question. Filled the same way and counted.
        n_gap = int(E[keep].isna().any(axis=1).sum()) if keep else 0
        if n_gap:
            print(f"  [{n_gap:,} participants absent from covariates_per_person.csv, "
                  f"levelled as 'Prefer not to answer']")
            E[keep] = E[keep].fillna("Prefer not to answer")
        adjusted = f"age, sex and {len(keep)} sociodemographic covariates"
    else:
        print(f"  [{cpath} not found: falling back to age and sex only. NOT for the paper --")
        print("   run covariates.py first, because age drives both burden and health care use.]")
        adjusted = "age and sex only (PROVISIONAL)"
    E = E.dropna(subset=["age", "sex"])
    print(f"model on {len(E):,} participants, adjusted for {adjusted}")

    X, gcols = design(E)
    res, beta, cov, family = fit_count(E.n_op, X, np.log(E["t"].values))
    print(f"  family: {family}")

    rows, grads = [], {}
    for lv in sorted(E.decile.unique()):
        m, se, g = standardized_rate(beta, cov, X, gcols, lv, WINDOW)
        grads[lv] = g
        rows.append({"decile": int(lv), "n": int((E.decile == lv).sum()),
                     "adjusted_contact_days_per_window": round(m, 3),
                     "lo": round(m - 1.96 * se, 3), "hi": round(m + 1.96 * se, 3),
                     "raw_contact_days_per_window": round(float(
                         (E.loc[E.decile == lv, "n_op"] * WINDOW / E.loc[E.decile == lv, "t"]).mean()), 3)})
    B = pd.DataFrame(rows)

    # rate ratio, top decile against the bottom, on the log scale for the interval
    lo_, hi_ = int(B.decile.min()), int(B.decile.max())
    m_lo = B.loc[B.decile == lo_, "adjusted_contact_days_per_window"].iloc[0]
    m_hi = B.loc[B.decile == hi_, "adjusted_contact_days_per_window"].iloc[0]
    gl = grads[hi_] / m_hi - grads[lo_] / m_lo        # d log(mu_hi / mu_lo) with respect to the params
    se_log = float(np.sqrt(max(gl @ cov @ gl, 0.0)))
    rr = m_hi / m_lo
    print(f"\nADJUSTED OUTPATIENT CONTACT DAYS PER {WINDOW} DAYS, by decile of burden")
    for r in B.itertuples():
        print(f"  d{r.decile:<2d} n {r.n:6,}   adjusted {r.adjusted_contact_days_per_window:6.2f} "
              f"({r.lo:.2f} to {r.hi:.2f})   raw {r.raw_contact_days_per_window:6.2f}")
    print(f"\n  decile {hi_} vs decile {lo_}: rate ratio {rr:.2f} "
          f"(95% CI {rr * np.exp(-1.96 * se_log):.2f} to {rr * np.exp(1.96 * se_log):.2f})")
    print("  A rate ratio above 1 means more days with recorded outpatient contact per observed day")
    print("  among the most burdened. It does not say whether that is more need met or more need.")

    # ---- the dissemination policy, at write time
    Bw = B.copy()
    hide, prim = partition_mask(B["n"].values)
    Bw["n"] = marked(B["n"].values, hide, prim)
    Bw.loc[prim, ["adjusted_contact_days_per_window", "lo", "hi", "raw_contact_days_per_window"]] = np.nan
    Bw.to_csv(os.path.join(OUT_DIR, f"care_records_by_decile{SUF}.csv"), index=False)
    Ew = el.reset_index().rename(columns={"size": "n", "sum": "n_eligible", "pct": "eligible_pct"})
    Ew = Ew.astype(object)
    for i in Ew.index:
        if pair_hidden(int(Ew.at[i, "n_eligible"]), int(Ew.at[i, "n"])):
            Ew.loc[i, ["n_eligible", "eligible_pct"]] = [SUPP, np.nan]
    Ew.to_csv(os.path.join(OUT_DIR, f"care_records_eligibility{SUF}.csv"), index=False)
    se_all = np.sqrt(np.diag(cov))
    CO = pd.DataFrame({"term": list(X.columns), "beta": beta, "se": se_all})
    held = [c for c in X.columns if c != "const" and set(np.unique(X[c].values)) <= {0.0, 1.0}
            and 0 < int(X[c].sum()) <= 20]
    CO.loc[CO.term.isin(held), ["beta", "se"]] = np.nan
    CO.to_csv(os.path.join(OUT_DIR, f"care_records_coefs{SUF}.csv"), index=False)
    fl_ok = not (pair_hidden(n_fl_el, n_fl) or pair_hidden(n_fl_end, n_fl))
    json.dump({"encoder": ENCODER, "window_days": WINDOW, "family": family, "adjusted_for": adjusted,
               "disclosure": "counts 1-20 suppressed", "coefs_withheld": held,
               "n_flagged": n_fl, "n_flagged_eligible": n_fl_el if fl_ok else SUPP,
               "pct_flagged_record_ends_by_answer": round(100 * n_fl_end / max(n_fl, 1), 2) if fl_ok else None,
               "pct_cohort_record_ends_by_answer": round(100 * n_end / len(D), 2),
               "n_model": int(len(E)), "n_eligible": int(D.eligible.sum()), "n_cohort": int(len(D)),
               "rate_ratio_top_vs_bottom": round(float(rr), 4),
               "rr_lo": round(float(rr * np.exp(-1.96 * se_log)), 4),
               "rr_hi": round(float(rr * np.exp(1.96 * se_log)), 4),
               "op_concepts": OP_CONCEPTS},
              open(os.path.join(OUT_DIR, f"care_records_summary{SUF}.json"), "w"), indent=2)
    print(f"\nwrote -> {OUT_DIR}/care_records_{{by_decile,eligibility,coefs}}{SUF}.csv and _summary{SUF}.json")


if __name__ == "__main__":
    main()
