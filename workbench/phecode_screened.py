#!/usr/bin/env python3
"""phecode_screened.py -- the SCREENED COMPARISON for the two cancers.
A 0 for breast (prostate) cancer means "no diagnosis recorded", not "checked and negative". Here the non-cases
are split by whether the record shows they WERE CHECKED, and the cancer combinations are applied again.

WHAT "CHECKED" MEANS (any time in the record; built from the OMOP concept table by code, and the concepts found
are written to the meta file):
  breast, women   any mammogram, screening or diagnostic: CPT 77067 77065 77066 77057 77056 77055 77063 77061
                  77062; HCPCS G0202 G0204 G0206; or a breast-screening encounter: ICD-10-CM Z12.31 Z12.39;
                  ICD-9-CM V76.10 V76.11 V76.12 V76.19
  prostate, men   any PSA test: CPT 84153 84152 84154; HCPCS G0103; LOINC measurements whose name contains
                  "Prostate specific Ag"; or a prostate-screening encounter: ICD-10-CM Z12.5; ICD-9-CM V76.44
  The events are searched in procedure_occurrence, measurement, observation and condition_occurrence, by both
  source and standard concept ids. Pulled once for the cohort and cached, person level, in
  screen_out/screening_checks_local.csv (stays on the Workbench).

THE SCORES are phecode_cross.py's own (fit_combinations: each diagnosis's group combination, out of fold on the
same 10-fold split, seed 0), for breast and prostate cancer, and the distance. Nothing is refitted here.

THREE COMPARISONS per 10-year age band, women for breast cancer and men for prostate cancer (the target's
denominator: at least 1 ICD-coded event), AUC with DeLong 95%, no averaging over bands:
  cases_vs_checked      cases against CHECKED non-cases (the main test: is it more than who gets checked?)
  cases_vs_unchecked    cases against non-cases with no check in the record
  checked_vs_unchecked  among non-cases only: checked against unchecked (does the combination mark who gets
                        checked?)
A cell is written only with more than 20 people on each side ("<=20" otherwise).
Also, per band: the SHARE of non-cases, and of cases, who were checked (shares only, no counts), in bands where
the cases and the non-cases each exceed 20; each share is written only where BOTH its checked and its unchecked
people exceed 20 (cell_noncases, cell_cases; "<=20" otherwise).

AGE vs AGE + GROUPS, AMONG THE CHECKED. Per 10-year band, over CHECKED cases against CHECKED
non-cases only (women for breast, men for prostate), two out-of-fold scores, both from models fitted on the whole
denominator outside each fold of the same split:
  age          age, age^2 (+ sex for breast cancer, constant among women)
  age_groups   the same + the k groups (phecode_cross.py's model; its group part must equal fit_combinations'
               score, checked)
  AUC of each within the band, and the paired DeLong difference (age_groups - age) with its 95% interval and p.
  Written only with more than 20 checked cases and more than 20 checked non-cases in the band.
  -> phecode_screened_age_<enc>.csv  (the models fitted once over all ages; kept for the record)

ONE MODEL PER BAND, AGE AND THE GROUPS TOGETHER. The model is fitted within each age band, and the train/test split
checks it for overfitting.
Per 10-year band, on the CHECKED cases and CHECKED non-cases of that band only, ONE unpenalized logistic model:
cancer ~ age (years) + the k groups.
  counts: checked cases and checked non-cases in the band (1 to 20 written "<=20"; 0 permitted).
  AUC: fitted on 9 folds and scored on the 10th (10-fold CV inside the band, stratified on the label, seed 0); the AUC
       over the test-fold scores, DeLong 95% interval; the share of the 10 fits that converged.
  what matters: the same model fitted on the whole band; odds ratio of age (per year) and of each group (per SD),
       95% Wald interval, p; each group's odds ratio is net of age.
  RIDGE: the same model with an L2 penalty, on the same outer folds;
       predictors standardized on each training part; the penalty strength chosen by an inner 5-fold split of the
       training part (mean log-loss over 13 strengths); AUC over the test-fold scores, DeLong 95%.
  BANDS: three schemes side by
       side, "10-year", "18-39, then 10-year", "18-39, 40-59, 60+".
  AUC and odds ratios are written only with more than 20 checked cases and more than 20 checked non-cases.
  SCREENING BANDS, breast cancer only: under 40 (mammograms for a
       symptom or high risk), 40-49 (routine screening starts), 50-74 (core screening years), 75+ (optional).
  ASSOCIATION, breast cancer, screening bands, NO folds (an age-band specific association instead of a prediction,
       and which collection of the k groups matters): per band, ONE model on all its women, cancer ~ age + k groups:
       the joint Wald test of the k; age's odds ratio; each group's odds ratio per SD (collection = p < 0.05/k, each
       encoder's own k: 27 for gte).
       -> phecode_screened_assoc_<enc>.csv (per band), phecode_screened_assocor_<enc>.csv (per band x group).
  JOINT vs SINGLE (does scoring the groups together reveal an association that each group alone does not?):
       per band, each group ALONE (cancer ~ age + group j) next to the joint model; columns or_alone, p_alone,
       passes_alone in phecode_screened_assocor_<enc>.csv (band_alone).
  TIMING, breast cancer, screening bands: the survey
       anchor (median date of the 151 answers, as for age; pulled once, cached in survey_anchor_local.csv) against the
       first breast cancer code (PheTK first_event_date); "before" = code on or before the anchor. Per band and timing:
       counts, years apart (percentiles), and the same one model vs the same women without breast cancer.
       -> phecode_screened_timing_<enc>.csv, phecode_screened_timingor_<enc>.csv.
  ★ COUNTS ARE WRITTEN AS THEY ARE in the band tables. The only check run on the download is pii_check: no
  person-level column in any table. Bands with 20 or fewer cases or non-cases are not fitted ("too few to fit").
  The All of Us dissemination policy forbids publishing a participant count of 1 to 20, or anything from which
  one can be derived.
  -> phecode_screened_band_<enc>.csv (1 row per target x band), phecode_screened_bandor_<enc>.csv (x term)
  Disclosure of the counts: the band totals of checked cases / non-cases are published nowhere else and do not
  partition any published total; where a case share is written (coverage table) its unchecked side exceeds 20.

THE DISSEMINATION POLICY (disclosure.py). No count is written or printed except the checked counts per band
(1 to 20 as "<=20"). Every AUC and share rests on more than 20 people on each side of it, and is written to 3
decimals (disclosure.coarsen).
★ THE PRECISION MATTERS. A share written to 16 digits is an exact fraction and returns the counts behind it, and an
AUC at 16 digits is an exact fraction U / (n1 x n0) too.

INPUTS   as phecode_cross.py; BigQuery (concept, procedure_occurrence, measurement, observation,
         condition_occurrence) once, then the cache
OUTPUTS  screen_out/phecode_screened/phecode_screened_band_<enc>.csv     1 row per target x band (counts, test-fold AUC)
         screen_out/phecode_screened/phecode_screened_bandor_<enc>.csv   1 row per target x band x term (odds ratios)
         screen_out/phecode_screened/phecode_screened_age_<enc>.csv      1 row per target x band (age vs age + groups)
         screen_out/phecode_screened/phecode_screened_<enc>.csv          1 row per target x comparison x source x band
         screen_out/phecode_screened/phecode_screened_coverage_<enc>.csv 1 row per target x band
         screen_out/phecode_screened/phecode_screened_<enc>_meta.json    (with the concepts found)
         screen_out/phecode_screened_<enc>_<CDR>.zip                     those files, checked by pii_check()

Run:  python3 phecode_screened.py                  (--refresh pulls the checks again; --max-gb N stops a scan
                                                    larger than N GB, default 400)
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from disclosure import MARK, SUPP, coarsen, coarsen_sig, small           # noqa: E402  the dissemination policy
from phecode_block import delong                                         # noqa: E402
from phecode_cross import BAND_EDGES, BAND_LABELS, FEMALE, FOLDS, MIN_CELL, SEED, fit_combinations  # noqa: E402
from phecode_directions import direction_scores                          # noqa: E402
from phecode_sweep import COUNTS, Z, load                                # noqa: E402
from wb_config import CFG                                                # noqa: E402
from screen_battery import ENCODERS                                      # noqa: E402
from zip_screen_aggregates import ID_COLS                                # noqa: E402  person-level column names
import statsmodels.api as sm                                             # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "phecode_screened")
CACHE = os.path.join(OUT_DIR, "screening_checks_local.csv")              # person level. Stays on the Workbench.
CONCEPTS = os.path.join(OUT_DIR, "screening_check_concepts.csv")         # the concepts found (no people)
CDR = os.environ.get("WORKSPACE_CDR", "")
ANCHOR = os.path.join(OUT_DIR, "survey_anchor_local.csv")                # person level. Stays on the Workbench.
ID_COL = "id"
ENC = sys.argv[sys.argv.index("--encoder") + 1] if "--encoder" in sys.argv else ENCODERS[0]
MAX_GB = float(sys.argv[sys.argv.index("--max-gb") + 1]) if "--max-gb" in sys.argv else 400.0
CHECKS = {
    "breast": {"proc": ["77067", "77065", "77066", "77057", "77056", "77055", "77063", "77061", "77062",
                        "G0202", "G0204", "G0206"],
               "icd": ["Z12.31", "Z12.39", "V76.10", "V76.11", "V76.12", "V76.19"], "loinc_like": None},
    "prostate": {"proc": ["84153", "84152", "84154", "G0103"], "icd": ["Z12.5", "V76.44"],
                 "loinc_like": "%prostate specific ag%"},
}
TARGETS = [("breast", "women", FEMALE), ("prostate", "men", 1.0)]
COMPARISONS = ["cases_vs_checked", "cases_vs_unchecked", "checked_vs_unchecked"]


def pull_checks(idx):
    """Per person and kind (breast, prostate): whether any check is in the record. Cached."""
    if os.path.exists(CACHE) and "--refresh" not in sys.argv:
        C = pd.read_csv(CACHE, dtype={ID_COL: str})
        print(f"  [checks: read from {CACHE}, nothing pulled]")
        return C, (pd.read_csv(CONCEPTS, dtype=str) if os.path.exists(CONCEPTS) else None)
    from google.cloud import bigquery
    client = bigquery.Client()
    code_sql = []
    for kind, c in CHECKS.items():
        part = (f"SELECT concept_id, concept_code, vocabulary_id, concept_name, '{kind}' AS kind FROM `{CDR}.concept` "
                f"WHERE (vocabulary_id IN ('CPT4', 'HCPCS') AND concept_code IN UNNEST(@{kind}_proc)) "
                f"OR (vocabulary_id IN ('ICD10CM', 'ICD9CM') AND concept_code IN UNNEST(@{kind}_icd))")
        if c["loinc_like"]:
            part += f" OR (vocabulary_id = 'LOINC' AND LOWER(concept_name) LIKE '{c['loinc_like']}')"
        code_sql.append(part)
    p_codes = []
    for kind, c in CHECKS.items():
        p_codes += [bigquery.ArrayQueryParameter(f"{kind}_proc", "STRING", c["proc"]),
                    bigquery.ArrayQueryParameter(f"{kind}_icd", "STRING", c["icd"])]
    p_all = p_codes + [bigquery.ArrayQueryParameter("ids", "INT64", [int(i) for i in idx])]
    codes_q = " UNION ALL ".join(code_sql)
    cfg = lambda params, dry: bigquery.QueryJobConfig(query_parameters=params, dry_run=dry, use_query_cache=not dry)
    concepts = client.query(codes_q, job_config=cfg(p_codes, False)).to_dataframe()
    concepts[["kind", "vocabulary_id", "concept_code", "concept_name"]].drop_duplicates().to_csv(CONCEPTS, index=False)
    print(f"  [checks: {len(concepts):,} concepts found: " +
          ", ".join(f"{k} {int((concepts.kind == k).sum())}" for k in CHECKS) + "]", flush=True)
    ev_sql = f"""
      WITH codes AS ({codes_q}),
      ev AS (
        SELECT person_id, procedure_date AS d, procedure_source_concept_id AS c FROM `{CDR}.procedure_occurrence` WHERE person_id IN UNNEST(@ids)
        UNION ALL SELECT person_id, procedure_date, procedure_concept_id FROM `{CDR}.procedure_occurrence` WHERE person_id IN UNNEST(@ids)
        UNION ALL SELECT person_id, measurement_date, measurement_source_concept_id FROM `{CDR}.measurement` WHERE person_id IN UNNEST(@ids)
        UNION ALL SELECT person_id, measurement_date, measurement_concept_id FROM `{CDR}.measurement` WHERE person_id IN UNNEST(@ids)
        UNION ALL SELECT person_id, observation_date, observation_source_concept_id FROM `{CDR}.observation` WHERE person_id IN UNNEST(@ids)
        UNION ALL SELECT person_id, condition_start_date, condition_source_concept_id FROM `{CDR}.condition_occurrence` WHERE person_id IN UNNEST(@ids)
      )
      SELECT ev.person_id, codes.kind, MIN(ev.d) AS first_date, COUNT(DISTINCT ev.d) AS n_dates
      FROM ev JOIN (SELECT DISTINCT concept_id, kind FROM codes) AS codes ON ev.c = codes.concept_id
      GROUP BY ev.person_id, codes.kind"""
    gb = client.query(ev_sql, job_config=cfg(p_all, True)).total_bytes_processed / 1e9
    print(f"  [checks: scanning {gb:,.1f} GB]", flush=True)
    assert gb <= MAX_GB, f"the scan would read {gb:,.0f} GB, above --max-gb {MAX_GB:,.0f}; rerun with a larger --max-gb"
    C = client.query(ev_sql, job_config=cfg(p_all, False)).to_dataframe()
    C = C.rename(columns={"person_id": ID_COL})
    C[ID_COL] = C[ID_COL].astype(str)
    C.to_csv(CACHE, index=False)
    print(f"  [checks: cached to {CACHE} -- person level, do not download]", flush=True)
    return C, concepts


def fit_age_models(A, Yz, fold, scores):
    """Out-of-fold linear predictors of the age model and of the age + groups model, for the two cancers, fitted on
    each cancer's denominator outside each fold (the split and the design of phecode_cross.fit_combinations)."""
    from phecode_sweep import CONDITIONS
    N = len(fold)
    age, sex = A["age"], A["sex"]
    out = {}
    for key, code, label, lst, men_only in CONDITIONS:
        if key not in ("breast", "prostate"):
            continue
        d = A[f"den_{key}"]
        y = np.nan_to_num(A[f"y_{key}"], nan=0).astype(int)
        az = (age - age[d].mean()) / age[d].std()
        base = np.column_stack([az, az ** 2] + ([] if men_only else [sex]))
        lp_age, lp_full, grp = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
        for f in range(FOLDS):
            tr, te = d & (fold != f), fold == f
            r0 = sm.Logit(y[tr], sm.add_constant(base[tr], has_constant="add")).fit(disp=0, maxiter=200)
            lp_age[te] = sm.add_constant(base[te], has_constant="add") @ np.asarray(r0.params)
            X1 = np.column_stack([base, Yz])
            r1 = sm.Logit(y[tr], sm.add_constant(X1[tr], has_constant="add")).fit(disp=0, maxiter=200)
            b1 = np.asarray(r1.params)
            lp_full[te] = sm.add_constant(X1[te], has_constant="add") @ b1
            grp[te] = Yz[te] @ b1[-Yz.shape[1]:]
        worst = float(np.nanmax(np.abs(grp - scores[key])))
        assert worst < 1e-9, f"{key}: the age + groups model's group part differs from fit_combinations by {worst}"
        out[key] = {"age": lp_age, "age_groups": lp_full}
    return out


def band_cv(y, X):
    """Within one band: out-of-fold scores of the one model, cancer ~ age + groups (10 folds, stratified on the label,
    seed 0; fitted on 9 folds, scored on the 10th), and the share of the 10 fits that converged."""
    from sklearn.model_selection import StratifiedKFold
    n = len(y)
    s = np.full(n, np.nan)
    conv = []
    Xc = sm.add_constant(X, has_constant="add")
    for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(n), y):
        try:
            res = sm.Logit(y[tr], Xc[tr]).fit(disp=0, maxiter=500)
            conv.append(bool(res.mle_retvals.get("converged", False)))
            s[te] = Xc[te] @ np.asarray(res.params)
        except Exception:                                       # e.g. perfect separation in a small band
            conv.append(False)
    return s, float(np.mean(conv))


def band_full(y, X):
    """The same model fitted on the whole band: odds ratio (age per year, each group per SD), 95% Wald interval, p."""
    Xc = sm.add_constant(X, has_constant="add")
    try:
        f = sm.Logit(y, Xc).fit(disp=0, maxiter=500)
    except Exception:
        return None
    b, se, p = np.asarray(f.params)[1:], np.asarray(f.bse)[1:], np.asarray(f.pvalues)[1:]
    return {"converged": bool(f.mle_retvals.get("converged", False)),
            "ors": [{"or": float(np.exp(b[j])), "or_lo": float(np.exp(b[j] - Z * se[j])),
                     "or_hi": float(np.exp(b[j] + Z * se[j])), "p": float(p[j])} for j in range(len(b))]}


SCHEMES = {"10-year": BAND_EDGES,                                         # the bands
           "18-39, then 10-year": [18, 40, 50, 60, 70, 80, np.inf],
           "18-39, 40-59, 60+": [18, 40, 60, np.inf],
           "screening (breast)": [18, 40, 50, 75, np.inf]}             # breast screening practice
SCHEME_TARGETS = {"screening (breast)": ("breast",)}                    # other schemes: both cancers
C_GRID = np.logspace(-4, 2, 13)                                           # ridge strengths tried (C = 1 / penalty)


def band_labels(edges):
    return [f"{int(lo)}-{int(hi) - 1}" if np.isfinite(hi) else f"{int(lo)}+" for lo, hi in zip(edges[:-1], edges[1:])]


def band_ridge_cv(y, X):
    """The same model with a ridge (L2) penalty: out-of-fold scores on the SAME 10 outer folds as band_cv. In each
    outer training part the predictors are standardized on that part, and the penalty strength is chosen by an
    inner 5-fold split of that part (mean log-loss over C_GRID); the model is refitted on the whole training part
    with the chosen strength and scores the held-out fold. Returns the scores and the median chosen C."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    n = len(y)
    s, chosen = np.full(n, np.nan), []
    for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(n), y):
        loss = np.zeros(len(C_GRID))
        for itr, ite in StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(tr)), y[tr]):
            sc = StandardScaler().fit(X[tr][itr])
            for i, c in enumerate(C_GRID):
                mdl = LogisticRegression(C=c, max_iter=5000).fit(sc.transform(X[tr][itr]), y[tr][itr])
                loss[i] += log_loss(y[tr][ite], mdl.predict_proba(sc.transform(X[tr][ite]))[:, 1], labels=[0, 1])
        c = C_GRID[int(np.argmin(loss))]
        chosen.append(c)
        sc = StandardScaler().fit(X[tr])
        s[te] = LogisticRegression(C=c, max_iter=5000).fit(sc.transform(X[tr]), y[tr]).decision_function(sc.transform(X[te]))
    return s, float(np.median(chosen))



def bonf(k):
    """The correction for k tests in a band: 0.05 / k, each encoder's own k (27 for gte), as phecode_fifteen.py uses
    each encoder's own k."""
    return 0.05 / k


def band_assoc(y, age, G):
    """Association within one band, NO folds: an age-band specific association instead of a prediction, so there are
    no training folds. ONE model on all women in the band,
    cancer ~ age + the k groups: whether the groups are associated beyond age = the joint Wald test of the k group
    terms (k df); which groups = each group's odds ratio per SD, 95% Wald interval, p (the collection: p < 0.05/k);
    and age's odds ratio per year. Returns (row fields, per-term rows)."""
    k = G.shape[1]
    X = sm.add_constant(np.column_stack([age, G]), has_constant="add")     # const, age, k groups
    f = sm.Logit(y, X).fit(disp=0, maxiter=500)
    b, V, se, p = np.asarray(f.params), np.asarray(f.cov_params()), np.asarray(f.bse), np.asarray(f.pvalues)
    g = slice(2, 2 + k)
    w = float(b[g] @ np.linalg.solve(V[g, g], b[g]))
    row = {"groups_wald": w, "groups_df": k, "groups_p": float(chi2.sf(w, k)),
           "age_or": float(np.exp(b[1])), "age_or_lo": float(np.exp(b[1] - Z * se[1])),
           "age_or_hi": float(np.exp(b[1] + Z * se[1])), "converged": bool(f.mle_retvals.get("converged", False))}
    terms = [{"group": j, "or": float(np.exp(b[2 + j])), "or_lo": float(np.exp(b[2 + j] - Z * se[2 + j])),
              "or_hi": float(np.exp(b[2 + j] + Z * se[2 + j])), "p": float(p[2 + j]),
              "in_collection": bool(p[2 + j] < bonf(k))} for j in range(k)]
    return row, terms


def band_alone(y, age, G):
    """Each group ALONE (joint vs single): per group j, cancer ~ age + group j on all women in the
    band; odds ratio per SD, 95% Wald interval, p; passes = p < 0.05/k (the same correction as together)."""
    out, k = [], G.shape[1]
    for j in range(k):
        f = sm.Logit(y, sm.add_constant(np.column_stack([age, G[:, j]]), has_constant="add")).fit(disp=0, maxiter=500)
        b, se, p = float(f.params[2]), float(f.bse[2]), float(f.pvalues[2])
        out.append({"group": j, "or_alone": float(np.exp(b)), "or_alone_lo": float(np.exp(b - Z * se)),
                    "or_alone_hi": float(np.exp(b + Z * se)), "p_alone": p, "passes_alone": bool(p < bonf(k))})
    return out


def survey_anchor(idx):
    """Per person: the survey anchor date = the median, over the 151 items, of the date on which each item's kept
    (latest) answer was given; the anchor build_responses_v9.py uses for age (it was not saved there). Pulled once
    from BigQuery (the median is taken in the query, so one row per person comes back), then read from the cache."""
    if os.path.exists(ANCHOR) and "--refresh-anchor" not in sys.argv:
        D = pd.read_csv(ANCHOR, dtype={ID_COL: str})
        print(f"  [survey anchor: read from {ANCHOR}, nothing pulled]", flush=True)
    else:
        from google.cloud import bigquery
        client = bigquery.Client()
        cids = sorted(pd.read_csv(CFG.items_csv, usecols=["concept_id"])["concept_id"].astype("int64").unique().tolist())
        sql = f"""WITH t AS (SELECT person_id, observation_source_concept_id AS c, MAX(observation_date) AS d
                             FROM `{CDR}.observation`
                             WHERE person_id IN UNNEST(@ids) AND observation_source_concept_id IN UNNEST(@cids)
                             GROUP BY person_id, c)
                  SELECT DISTINCT person_id,
                         PERCENTILE_CONT(UNIX_DATE(d), 0.5) OVER (PARTITION BY person_id) AS anchor_day,
                         COUNT(*) OVER (PARTITION BY person_id) AS n_items_dated
                  FROM t"""
        params = [bigquery.ArrayQueryParameter("ids", "INT64", [int(i) for i in idx]),
                  bigquery.ArrayQueryParameter("cids", "INT64", cids)]
        dry = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params, dry_run=True,
                                                                    use_query_cache=False))
        gb = dry.total_bytes_processed / 1e9
        print(f"  [survey anchor: scanning {gb:,.1f} GB]", flush=True)
        assert gb <= MAX_GB, f"the scan would read {gb:,.0f} GB, above --max-gb {MAX_GB:,.0f}"
        D = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()
        D["person_id"] = D["person_id"].astype(str)
        D["anchor"] = pd.to_datetime(np.round(D["anchor_day"].astype(float)).astype("int64"), unit="D", origin="unix")
        D = D.rename(columns={"person_id": ID_COL})[[ID_COL, "anchor", "n_items_dated"]]
        D.to_csv(ANCHOR, index=False)
        print(f"  [survey anchor: cached to {ANCHOR} -- person level, do not download]", flush=True)
    D["anchor"] = pd.to_datetime(D["anchor"])
    return D.set_index(ID_COL)["anchor"].reindex(idx)


def first_code_date(idx, phecode):
    """PheTK's first_event_date for one phecode, per person in idx order (NaT without an event)."""
    P = pd.read_csv(COUNTS, sep="\t", dtype={"person_id": str, "phecode": str},
                    usecols=["person_id", "phecode", "count", "first_event_date"])
    p = P[P["phecode"] == phecode].set_index("person_id")
    assert p.index.is_unique, f"{phecode}: more than 1 row per person in {COUNTS}"
    return pd.to_datetime(p["first_event_date"]).reindex(idx).values


def pii_check(path):
    """(ok, reason). The only check run on these tables: no person-level column and no written index in any table;
    counts are written as they are."""
    if path.endswith(".json"):
        return True, ""
    d = pd.read_csv(path)
    hit = ID_COLS.intersection(str(c).strip().lower() for c in d.columns)
    if hit:
        return False, f"carries a person column ({', '.join(sorted(hit))})"
    unnamed = [str(c) for c in d.columns if not str(c).strip() or str(c).startswith("Unnamed:")]
    if unnamed:
        return False, f"unnamed column {unnamed[0]!r} -- an index written out, possibly person ids"
    return True, ""


def main():
    assert ENC in ENCODERS, f"--encoder must be one of {ENCODERS}"
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)
    N = len(idx)
    A = load(idx)
    Yz, names, worst = direction_scores(ENC, idx)
    assert worst < 1e-6, f"{ENC}: the recomputed distance differs from scores_{ENC}.csv by {worst}"
    k = Yz.shape[1]
    S = pd.read_csv(os.path.join(OUT_DIR, f"scores_{ENC}.csv"))
    S[ID_COL] = S[ID_COL].astype(str)
    dist = S.set_index(ID_COL).reindex(idx)["distance"].values.astype(float)
    distz = (dist - dist.mean()) / dist.std(ddof=1)
    fold = np.random.default_rng(SEED).permutation(np.arange(N) % FOLDS)   # phecode_cross.py's split
    C, concepts = pull_checks(idx)
    print(f"[phecode_screened] {ENC}: k = {k} | fitting the combinations (phecode_cross.py's)", flush=True)
    scores, _, _ = fit_combinations(A, Yz, fold, k, verbose=False)
    sources = {"breast": scores["breast"], "prostate": scores["prostate"], "distance": distz}
    band = np.digitize(A["age"], BAND_EDGES[1:-1])
    sex = A["sex"]
    LP = fit_age_models(A, Yz, fold, scores)
    arows = []
    for tgt, sname, sval in TARGETS:
        checked = idx.isin(C.loc[C["kind"] == tgt, ID_COL]).astype(bool)
        d = A[f"den_{tgt}"] & (sex == sval) & checked                        # ★ both sides checked
        y = np.nan_to_num(A[f"y_{tgt}"], nan=0).astype(int)
        for b, blab in enumerate(BAND_LABELS):
            m = d & (band == b)
            n1, n0 = int((m & (y == 1)).sum()), int((m & (y == 0)).sum())
            ok = n1 >= MIN_CELL and n0 >= MIN_CELL
            r = {"encoder": ENC, "target": tgt, "sex": sname, "band": blab, "population": "checked cases vs checked non-cases",
                 "cell": "written" if ok else MARK}
            if ok:
                auc, Sg = delong(y[m], np.vstack([LP[tgt]["age"][m], LP[tgt]["age_groups"][m]]))
                se = np.sqrt(np.diag(Sg))
                dlt = auc[1] - auc[0]
                sed = float(np.sqrt(max(Sg[0, 0] + Sg[1, 1] - 2 * Sg[0, 1], 0.0)))
                r.update({"auc_age": auc[0], "auc_age_lo": auc[0] - Z * se[0], "auc_age_hi": auc[0] + Z * se[0],
                          "auc_age_groups": auc[1], "auc_age_groups_lo": auc[1] - Z * se[1],
                          "auc_age_groups_hi": auc[1] + Z * se[1],
                          "dauc": dlt, "dauc_lo": dlt - Z * sed, "dauc_hi": dlt + Z * sed,
                          "dauc_p": float(2 * norm.sf(abs(dlt) / sed)) if sed > 0 else np.nan})
            arows.append(r)
    Ag = pd.DataFrame(arows)
    Ag = coarsen(Ag, [c for c in Ag.columns if c.startswith(("auc_", "dauc")) and c != "dauc_p"])   # ★ precision
    Ag = coarsen_sig(Ag, ["dauc_p"])
    Ag.to_csv(os.path.join(OUT, f"phecode_screened_age_{ENC}.csv"), index=False)

    brows, orows = [], []                                                # ONE model per band: age and the groups together
    for scheme, edges in SCHEMES.items():
        labels = band_labels(edges)
        bnd = np.digitize(A["age"], edges[1:-1])
        for tgt, sname, sval in TARGETS:
            if tgt not in SCHEME_TARGETS.get(scheme, (tgt,)):
                continue
            checked = idx.isin(C.loc[C["kind"] == tgt, ID_COL]).astype(bool)
            d = A[f"den_{tgt}"] & (sex == sval) & checked                    # ★ both sides checked
            y_all = np.nan_to_num(A[f"y_{tgt}"], nan=0).astype(int)
            for b, blab in enumerate(labels):
                m = d & (bnd == b)
                y = y_all[m]
                n1, n0 = int(y.sum()), int((y == 0).sum())
                ok = n1 >= MIN_CELL and n0 >= MIN_CELL
                r = {"encoder": ENC, "scheme": scheme, "target": tgt, "sex": sname, "band": blab,
                     "population": "checked cases vs checked non-cases", "model": "cancer ~ age + groups, fitted within the band",
                     "n_cases_checked": n1, "n_noncases_checked": n0,         # ★ counts as they are
                     "cell": "written" if ok else "too few to fit"}
                if ok:
                    X = np.column_stack([A["age"][m], Yz[m]])                # columns: age, then the k groups
                    s_oof, conv = band_cv(y, X)
                    s_rdg, c_med = band_ridge_cv(y, X)
                    r["fits_converged_share"] = conv
                    full = band_full(y, X)
                    if np.isnan(s_oof).any() or full is None:
                        r["cell"] = "fit failed"
                    else:
                        auc, Sg = delong(y, np.vstack([s_oof, s_rdg]))
                        se = np.sqrt(np.diag(np.atleast_2d(Sg)))
                        r.update({"auc": float(auc[0]), "auc_lo": float(auc[0] - Z * se[0]), "auc_hi": float(auc[0] + Z * se[0]),
                                  "auc_ridge": float(auc[1]), "auc_ridge_lo": float(auc[1] - Z * se[1]),
                                  "auc_ridge_hi": float(auc[1] + Z * se[1]), "ridge_C_median": c_med,
                                  "converged_full": full["converged"]})
                        for j, o in enumerate(full["ors"]):
                            cp = "" if j == 0 or "construct_profile" not in names.columns else names.iloc[j - 1]["construct_profile"]
                            orows.append({"encoder": ENC, "scheme": scheme, "target": tgt, "sex": sname, "band": blab,
                                          "term": "age (per year)" if j == 0 else f"group {j - 1}", "group": j - 1, **o,
                                          "construct_profile": cp})
                brows.append(r)
    Bd = coarsen(pd.DataFrame(brows), ["auc", "auc_lo", "auc_hi", "auc_ridge", "auc_ridge_lo", "auc_ridge_hi",
                                       "fits_converged_share"])          # ★ precision (disclosure.py)
    Bd = coarsen_sig(Bd, ["ridge_C_median"])
    Bd.to_csv(os.path.join(OUT, f"phecode_screened_band_{ENC}.csv"), index=False)
    Od = coarsen_sig(pd.DataFrame(orows), ["or", "or_lo", "or_hi", "p"])
    Od.to_csv(os.path.join(OUT, f"phecode_screened_bandor_{ENC}.csv"), index=False)

    arow, srow = [], []                                                  # ASSOCIATION, breast, screening bands
    edges = SCHEMES["screening (breast)"]
    bnd = np.digitize(A["age"], edges[1:-1])
    checked = idx.isin(C.loc[C["kind"] == "breast", ID_COL]).astype(bool)
    d = A["den_breast"] & (sex == FEMALE) & checked                          # women with a mammogram in the record
    y_all = np.nan_to_num(A["y_breast"], nan=0).astype(int)
    for b, blab in enumerate(band_labels(edges)):
        m = d & (bnd == b)
        y = y_all[m]
        n1, n0 = int(y.sum()), int((y == 0).sum())
        r = {"encoder": ENC, "target": "breast", "sex": "women", "bands": "screening (breast)", "band": blab,
             "n_cases_checked": n1, "n_noncases_checked": n0,
             "cell": "written" if (n1 >= MIN_CELL and n0 >= MIN_CELL) else "too few to fit"}
        if r["cell"] == "written":
            res, terms = band_assoc(y, A["age"][m], Yz[m])
            alone = band_alone(y, A["age"][m], Yz[m])
            r.update(res)
            r["n_groups_together"] = int(sum(t["in_collection"] for t in terms))
            r["n_groups_alone"] = int(sum(a["passes_alone"] for a in alone))
            for t, a in zip(terms, alone):
                cp = names.iloc[t["group"]]["construct_profile"] if "construct_profile" in names.columns else ""
                srow.append({"encoder": ENC, "target": "breast", "band": blab, **t,
                             **{k_: v for k_, v in a.items() if k_ != "group"}, "construct_profile": cp})
        arow.append(r)
    As = coarsen_sig(pd.DataFrame(arow), ["groups_wald", "groups_p", "age_or", "age_or_lo", "age_or_hi"])
    As.to_csv(os.path.join(OUT, f"phecode_screened_assoc_{ENC}.csv"), index=False)
    Ss = coarsen_sig(pd.DataFrame(srow), ["or", "or_lo", "or_hi", "p", "or_alone", "or_alone_lo", "or_alone_hi", "p_alone"])
    Ss.to_csv(os.path.join(OUT, f"phecode_screened_assocor_{ENC}.csv"), index=False)

    # TIMING: the first breast cancer code on or
    # before the survey anchor ("before") or after it ("after"); the same one model per band for each
    anchor = survey_anchor(idx).values
    fe = first_code_date(idx, "CA_105")
    no_anchor = d & pd.isna(anchor)
    print(f"  [timing: women with a mammogram and no dated survey answer, left out: {int(no_anchor.sum())}]", flush=True)
    dt = d & ~pd.isna(anchor)
    case = dt & (y_all == 1)
    assert not pd.isna(fe[case]).any(), "a breast cancer case without a first_event_date"
    gap = np.full(len(idx), np.nan)
    gap[case] = (pd.to_datetime(anchor[case]) - pd.to_datetime(fe[case])).days / 365.25   # > 0: code before survey
    before, after = case & (gap >= 0), case & (gap < 0)
    trow, tor = [], []
    for b, blab in enumerate(band_labels(edges)):
        inb = dt & (bnd == b)
        non = inb & (y_all == 0)
        for when, grp in (("before", before & inb), ("after", after & inb)):
            yrs = np.abs(gap[grp])
            r = {"encoder": ENC, "target": "breast", "bands": "screening (breast)", "band": blab, "timing": when,
                 "n_cases": int(grp.sum()), "n_noncases": int(non.sum())}
            if grp.sum() > 0:
                q = np.percentile(yrs, [10, 25, 50, 75, 90])
                r.update({f"years_p{p_}": float(v) for p_, v in zip((10, 25, 50, 75, 90), q)})
            r["cell"] = "written" if (grp.sum() >= MIN_CELL and non.sum() >= MIN_CELL) else "too few to fit"
            if r["cell"] == "written":
                m = grp | non
                res, terms = band_assoc(y_all[m], A["age"][m], Yz[m])
                r.update(res)
                for t in terms:
                    cp = names.iloc[t["group"]]["construct_profile"] if "construct_profile" in names.columns else ""
                    tor.append({"encoder": ENC, "target": "breast", "band": blab, "timing": when, **t, "construct_profile": cp})
            trow.append(r)
    Tm = coarsen_sig(pd.DataFrame(trow), ["groups_wald", "groups_p", "age_or", "age_or_lo", "age_or_hi"] +
                     [f"years_p{p_}" for p_ in (10, 25, 50, 75, 90)])
    Tm.to_csv(os.path.join(OUT, f"phecode_screened_timing_{ENC}.csv"), index=False)
    To = coarsen_sig(pd.DataFrame(tor), ["or", "or_lo", "or_hi", "p"]) if tor else pd.DataFrame()
    To.to_csv(os.path.join(OUT, f"phecode_screened_timingor_{ENC}.csv"), index=False)

    rows, cov = [], []
    for tgt, sname, sval in TARGETS:
        checked = idx.isin(C.loc[C["kind"] == tgt, ID_COL]).astype(bool)
        d = A[f"den_{tgt}"] & (sex == sval)
        y = np.nan_to_num(A[f"y_{tgt}"], nan=0).astype(int)
        case, non = d & (y == 1), d & (y == 0)
        for b, blab in enumerate(BAND_LABELS):
            inb = band == b
            groups = {"cases_vs_checked": (case & inb, non & checked & inb),
                      "cases_vs_unchecked": (case & inb, non & ~checked & inb),
                      "checked_vs_unchecked": (non & checked & inb, non & ~checked & inb)}
            for comp, (g1, g0) in groups.items():
                ok = int(g1.sum()) >= MIN_CELL and int(g0.sum()) >= MIN_CELL
                for src, sc in sources.items():
                    r = {"encoder": ENC, "target": tgt, "sex": sname, "band": blab, "comparison": comp,
                         "source": src, "cell": "written" if ok else MARK}
                    if ok:
                        m = g1 | g0
                        auc, Sg = delong(g1[m].astype(int), sc[m][None, :])
                        se = float(np.sqrt(np.atleast_2d(Sg)[0, 0]))
                        r.update({"auc": float(auc[0]), "auc_lo": float(auc[0] - Z * se), "auc_hi": float(auc[0] + Z * se)})
                    rows.append(r)
            ok = int((case & inb).sum()) >= MIN_CELL and int((non & inb).sum()) >= MIN_CELL
            c = {"encoder": ENC, "target": tgt, "sex": sname, "band": blab, "cell": "written" if ok else MARK}
            for who, g in (("noncases", non & inb), ("cases", case & inb)):
                # ★ a share is written only where its checked AND unchecked people each exceed 20
                both = ok and int((g & checked).sum()) >= MIN_CELL and int((g & ~checked).sum()) >= MIN_CELL
                c[f"cell_{who}"] = "written" if both else MARK
                c[f"share_checked_{who}"] = float(checked[g].mean()) if both else np.nan
            cov.append(c)

    R = coarsen(pd.DataFrame(rows), ["auc", "auc_lo", "auc_hi"])      # ★ precision (disclosure.py)
    Cv = coarsen(pd.DataFrame(cov), ["share_checked_noncases", "share_checked_cases"])
    R.to_csv(os.path.join(OUT, f"phecode_screened_{ENC}.csv"), index=False)
    Cv.to_csv(os.path.join(OUT, f"phecode_screened_coverage_{ENC}.csv"), index=False)
    meta = {"cdr": CDR, "encoder": ENC, "bands": BAND_LABELS, "comparisons": COMPARISONS,
            "checked": {k2: {kk: v for kk, v in c2.items()} for k2, c2 in CHECKS.items()},
            "tables": "procedure_occurrence, measurement, observation, condition_occurrence; source and standard ids",
            "scores": "phecode_cross.fit_combinations (out of fold, the same split) for breast and prostate cancer, "
                      "and the distance",
            "disclosure": "All of Us dissemination policy (disclosure.py): no count written; cells need more than 20 on each side; "
                          "a share needs more than 20 checked and more than 20 unchecked",
            "precision": "AUCs, intervals and shares to 3 decimals (disclosure.py)",
            "age_vs_age_groups": "checked cases vs checked non-cases per band; out-of-fold linear predictors of the age "
                                 "model (age, age^2, + sex for breast) and of the age + groups model, the same split; "
                                 "paired DeLong difference",
            "one_model_per_band": "checked cases vs checked non-cases per band; ONE unpenalized logistic model, cancer ~ age "
                                  "+ k groups, within the band; AUC over test-fold scores of 10-fold stratified CV (seed 0), "
                                  "DeLong 95%; odds ratios from the whole-band fit (age per year, groups per SD), 95% Wald; "
                                  "counts of checked cases and non-cases per band (1 to 20 written <=20); ridge (L2) AUC on the "
                                  "same folds, penalty chosen by an inner 5-fold split; three band schemes; a merged band's "
                                  "count that would return a hidden 10-year count by subtraction is written 'suppressed'"}
    if concepts is not None:
        meta["concepts_found"] = concepts[["kind", "vocabulary_id", "concept_code", "concept_name"]] \
            .drop_duplicates().sort_values(["kind", "vocabulary_id", "concept_code"]).astype(str).to_dict("records")
    json.dump(meta, open(os.path.join(OUT, f"phecode_screened_{ENC}_meta.json"), "w"), indent=2)

    print("\n  BREAST CANCER, women with a mammogram, screening bands. ONE model per band on all its women:")
    print(f"  breast cancer ~ age + the {k} groups. Joint test of the {k} groups; age OR per year; the groups whose OR passes")
    print(f"  the correction for {k} tests (fold from 1 per SD, net of age and the other {k - 1}).")
    for r in As.itertuples():
        head = f"  {r.band:6s}  with {r.n_cases_checked:>5} / without {r.n_noncases_checked:<6}"
        if r.cell != "written":
            print(head + "  too few to fit")
            continue
        print(head + f"  joint test p {r.groups_p:.2g} | age OR {r.age_or:.3f} [{r.age_or_lo:.3f}, {r.age_or_hi:.3f}]")
        coll = Ss[(Ss.band == r.band) & Ss.in_collection].sort_values("p")
        for _, g in coll.iterrows():
            up = g["or"] >= 1                                            # size only: fold from 1 either way
            fold, lo, hi = (g["or"], g["or_lo"], g["or_hi"]) if up else (1 / g["or"], 1 / g["or_hi"], 1 / g["or_lo"])
            print(f"          group {int(g['group']):2d} {str(g['construct_profile']).split(':')[0][:40]:40s} {fold:.2f}-fold "
                  f"[{lo:.2f}, {hi:.2f}]")
        if coll.empty:
            print("          no single group passes the correction")
    print("\n  JOINT vs SINGLE, same bands: each group ALONE (breast cancer ~ age + that group) next to ALL TOGETHER")
    print(f"  (fold from 1 per SD; passes = p < 0.05/{k} in each).")
    for r in As[As.cell == "written"].itertuples():
        t = Ss[Ss.band == r.band]
        both = t[t.in_collection & t.passes_alone]
        tog = t[t.in_collection & ~t.passes_alone]
        alo = t[~t.in_collection & t.passes_alone]
        f = lambda g, c: (g[c] if g[c] >= 1 else 1 / g[c])
        lab = lambda D, c: ", ".join(f"{int(g['group'])} {str(g['construct_profile']).split(':')[0]} {f(g, c):.2f}"
                                     for _, g in D.iterrows()) or "none"
        print(f"  {r.band:6s} passes together {r.n_groups_together}, alone {r.n_groups_alone}")
        print(f"          both:          {lab(both, 'or')}")
        print(f"          together only: {lab(tog, 'or')}")
        print(f"          alone only:    {lab(alo, 'or_alone')}")
    print("\n  TIMING: first breast cancer code ON OR BEFORE the survey anchor ('before') or AFTER it ('after').")
    print("  Years between them: median [25th, 75th percentile]. The same one model per band for each, vs the same women without.")
    for r in Tm.itertuples():
        head = f"  {r.band:6s} {r.timing:6s} cases {r.n_cases:>5} (without {r.n_noncases})"
        yrs = f" | years {r.years_p50:.1f} [{r.years_p25:.1f}, {r.years_p75:.1f}]" if r.n_cases > 0 else ""
        if r.cell != "written":
            print(head + yrs + " | too few to fit")
            continue
        coll = To[(To.band == r.band) & (To.timing == r.timing) & To.in_collection].sort_values("p")
        names_ = ", ".join(f"{int(g['group'])} {str(g['construct_profile']).split(':')[0]} "
                           f"{(g['or'] if g['or'] >= 1 else 1 / g['or']):.2f}" for _, g in coll.iterrows()) or "none individually"
        print(head + yrs + f" | joint test p {r.groups_p:.2g} | {names_}")
    print("\n  (the other tables are written as before; not printed)")
    files = [f"phecode_screened_timing_{ENC}.csv", f"phecode_screened_timingor_{ENC}.csv",
             f"phecode_screened_assoc_{ENC}.csv", f"phecode_screened_assocor_{ENC}.csv",
             f"phecode_screened_band_{ENC}.csv", f"phecode_screened_bandor_{ENC}.csv", f"phecode_screened_age_{ENC}.csv",
             f"phecode_screened_{ENC}.csv", f"phecode_screened_coverage_{ENC}.csv",
             f"phecode_screened_{ENC}_meta.json"]
    for fn in files:
        ok, why = pii_check(os.path.join(OUT, fn))
        assert ok, f"{fn} refused: {why}"
    zpath = os.path.join(OUT_DIR, f"phecode_screened_{ENC}_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"phecode_screened/{fn}")
    print(f"\n[phecode_screened] {len(R)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
