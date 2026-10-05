#!/usr/bin/env python3
"""care.py -- care outcomes that never entered the scoring, and each flagged set's care rates against the
rest of the cohort. Care by what carries a person is the --legacy path, which is not part of the paper.

STANDALONE Workbench script (numpy, pandas, scikit-learn, statsmodels, matplotlib, BigQuery).
The --legacy path adds the comparators entered alongside, the construct contributions for
the whole cohort, care by leading construct, care by construct contribution, and the circumstance side.

FRAMING. The flagged group is derived from the survey items and the responses alone. Health records
and the access survey are separate streams that never touched the screen. If they show the group is
under-served, that is a gap in CARE, not a validation of the screen.

OUTCOMES
  survey    any_access_barrier  = no usual source of care | any cost barrier | any structural delay | any medication rationing
            any_cost_barrier
            care_avoid_provider = delayed or no care because of a provider's race or religion, some of the time or more
  records   ed_share            = ED visits (9203, 262) / all visits, among people with >= 1 visit (window-free ratio)
            n_visits, visit_rate  supplement only, with the completeness caveat
            n_conditions          printed diagnostic only

WHAT IS COMPUTED BY DEFAULT
  the outcomes per person, pulled once and cached, then the group rates: EACH ENCODER'S OWN flagged set
  from scores_<encoder>.csv (the prespecified gte-large-en-v1.5 first), and each comparator's where
  scores_{scale,factor,pca}.csv exist, against the rest of the cohort, unadjusted (Wilson, Newcombe,
  Katz), under the dissemination policy. Then it stops.

WHAT --legacy COMPUTES (not part of the paper: it reads the consensus set, the participants flagged by all
6 encoders, works at the construct level, needs a file that no script in this folder writes, and writes its
counts without the dissemination policy)
  1. group rates: consensus outliers, and each comparator's flagged set, against the rest (Wilson, Newcombe, Katz)
  2. cohort gradient, rank-OLS: framework alone, comparator alone, both together; age, sex, and visits for the ED share
  3. care by leading construct among the consensus outliers (every construct leading at least one; no minimum count)
  4. care by construct contribution across the cohort: the 30 signed a_c per person (exact item partition,
     in distance units) under each encoder and each comparator, rank-standardized, one OLS per outcome
  5. the circumstance side (circumstance share > 1/2) among the consensus outliers, and outcomes by side
  6. the decile gradient on the framework percentile, with the two figures

INPUTS   responses_k5.csv, weights_k5.csv, the item table, screen_out/scores_*.csv
         --legacy also: basis/, screen_out/consensus_outliers.csv (screen_battery.py),
         screen_out/outlier_leaders_outliers.csv (written by no script in this folder),
         screen_out/comparator_weights_*.npz (comparators.py)
OUTPUTS  aggregates: care_group_rates.csv, care_group_rates.json
         --legacy instead: care_group_rates.csv, care_distance_ols.csv, care_by_leader.csv,
         care_by_construct_ols.csv, care_by_side.csv, care_gradient.csv, care_stats.json,
         fig_care_gradient.png, fig_care_visits_supp.png
         per person (stays): care_outcomes_per_person.csv  (reused by reruns; --repull forces a fresh pull)

Run:  python3 care.py --outcomes-only     (just the 4 outcomes per person, what the models read)
      python3 care.py                     (the outcomes, then the group rates on each own flagged set)
      python3 care.py --repull            (pull the outcomes again)
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_config import CFG   # noqa: E402  battery and responses folder, resolved once, no defaults
OUT_DIR       = os.path.join(HERE, "screen_out")
BASIS_DIR     = CFG.basis_dir
EMB_DIR       = CFG.emb_dir
EMB_PREFIX    = CFG.emb_prefix
RESPONSES_CSV = os.path.join(CFG.responses_dir, "responses_k5.csv")
WEIGHTS_CSV   = os.path.join(CFG.responses_dir, "weights_k5.csv")
ITEMS_CSV     = CFG.items_csv
OUTCOMES_CSV  = os.path.join(OUT_DIR, "care_outcomes_per_person.csv")
CDR           = os.environ.get("WORKSPACE_CDR", "")
ID_COL        = "id"
COVARIATES    = ["age", "sex"]
MIN_COUNT     = 0      # --legacy section 3 only, and NOT the dissemination policy: n >= 20 would still let
                       # a group of exactly 20 out, and the policy forbids 1 to 20 inclusive. The default
                       # path applies the policy through disclosure.py.
ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5",
            "all-mpnet-base-v2", "e5-large-v2", "qwen3-embedding-0.6b"]
COMPARATORS = ["scale", "factor", "pca"]
CIRCUMSTANCES = ["Everyday discrimination", "Healthcare discrimination", "Neighborhood disorder & safety",
                 "Neighborhood social cohesion", "Perceived social support", "Food insecurity",
                 "Housing instability", "Childhood adversity", "Interpersonal & sexual violence",
                 "Life-threatening events"]                      # the ten circumstance constructs
BINARY   = ["any_access_barrier", "any_cost_barrier", "care_avoid_provider"]
OUTCOMES = BINARY + ["ed_share"]
OKABE = {"red": "#D55E00", "orange": "#E69F00", "purple": "#CC79A7", "blue": "#0072B2", "green": "#009E73"}

USUAL = ["HealthAdvice_PlaceforHealthAdvice"]
COST = ["CantAffordCare_HealthcareProvider", "CantAffordCare_PrescriptionMedicines",
        "CantAffordCare_Specialist", "CantAffordCare_FollowupCare", "CantAffordCare_EmergencyCare",
        "CantAffordCare_DentalCare", "CantAffordCare_Eyeglasses",
        "DelayedMedicalCare_CantAffordCoPay", "DelayedMedicalCare_DeductibleTooHigh",
        "DelayedMedicalCare_HadToPayOutOfPocket"]
DELAYED = ["DelayedMedicalCare_ChildCare", "DelayedMedicalCare_ElderlyCare",
           "DelayedMedicalCare_RuralArea", "DelayedMedicalCare_TimeOffWork",
           "DelayedMedicalCare_Transportation"]
COPING = ["CantAffordCare_DelayedFillingRxToSaveMoney", "CantAffordCare_LowerCostRxToSaveMoney",
          "CantAffordCare_SkippedMedToSaveMoney", "CantAffordCare_TookLessMedToSaveMoney"]
MECH = ["HealthProviderRaceReligion_DelayedOrNoCare"]


# ----------------------------------------------------------------------------- statistics
def wilson(k, n, z=1.959963985):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2)
    return ((p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def rr_katz(ko, no, kr, nr, z=1.959963985):
    if min(ko, kr, no, nr) == 0:
        return float("nan"), float("nan"), float("nan")
    rr = (ko / no) / (kr / nr)
    se = math.sqrt(1.0 / ko - 1.0 / no + 1.0 / kr - 1.0 / nr)
    return rr, rr * math.exp(-z * se), rr * math.exp(z * se)


def rank_z(v):
    """Percentile rank, then standardized: a coefficient is per SD of rank."""
    r = pd.Series(np.asarray(v, float)).rank(pct=True).values
    return (r - r.mean()) / r.std()


def ols(y, Xcols, names):
    """OLS with intercept; returns params, conf_int, pvalues (Series by name), r2, n."""
    X = sm.add_constant(np.column_stack(Xcols), has_constant="add")
    fit = sm.OLS(np.asarray(y, float), X).fit()
    idx = ["const"] + list(names)
    return (pd.Series(fit.params, index=idx), pd.DataFrame(fit.conf_int(), index=idx),
            pd.Series(fit.pvalues, index=idx), float(fit.rsquared), int(fit.nobs))


# ----------------------------------------------------------------------------- the battery (as comparators.py)
def residualizer(C, w):
    A = np.column_stack([np.ones(len(w))] + ([C] if C is not None and C.shape[1] else []))
    sw = np.sqrt(w)[:, None]

    def apply(V):
        beta, *_ = np.linalg.lstsq(A * sw, V * sw, rcond=None)
        return V - A @ beta
    return apply


def load_battery():
    T = pd.read_csv(ITEMS_CSV)
    items = T["item"].astype(str).tolist()
    T = T.set_index("item")
    R = pd.read_csv(RESPONSES_CSV)
    R[ID_COL] = R[ID_COL].astype(str)
    R = R.set_index(ID_COL)
    raw = R[items].values.astype(float)
    assert not np.isnan(raw).any()
    lo, hi = T["scale_min"].values.astype(float), T["scale_max"].values.astype(float)
    rev = T["reverse_coded"].astype(str).str.lower().eq("true").values
    X = np.clip(2.0 * (raw - lo) / (hi - lo) - 1.0, -1.0, 1.0)
    X[:, rev] *= -1.0
    cov = R[COVARIATES].values.astype(float)
    w = (pd.read_csv(WEIGHTS_CSV).iloc[:, 0].values.astype(float)
         if WEIGHTS_CSV and os.path.exists(WEIGHTS_CSV) else np.ones(len(R)))
    con = T.loc[items, "construct"].astype(str).tolist()
    return R.index, items, con, X, cov, w


def framework_P(enc, items):
    b = np.load(os.path.join(BASIS_DIR, f"{enc}.npz"), allow_pickle=True)
    e = np.load(os.path.join(EMB_DIR, f"{EMB_PREFIX}_{enc}.npz"), allow_pickle=True)
    pos = {str(x): j for j, x in enumerate(e["items"])}
    E = np.asarray(e["vectors"], float)[[pos[it] for it in items]]
    return (E - b["mu"]) @ b["W"].T


def construct_contributions(Xr, S, w, con, Om=None):
    """a_c for every person, exact: T2 = xr' Q xr with Q = S Om S'. Returns (A: n x constructs, T2, cons)."""
    if Om is None:
        Yc = Xr @ S
        Yc = Yc - np.average(Yc, axis=0, weights=w)
        Om = LedoitWolf().fit(Yc).precision_
    Q = S @ Om @ S.T
    ai = Xr * (Xr @ Q)
    T2 = ai.sum(1)
    cons = sorted(set(con))
    A = np.column_stack([ai[:, [i for i, c in enumerate(con) if c == k]].sum(1) for k in cons])
    assert np.allclose(A.sum(1), T2, rtol=1e-6)
    return A, T2, cons


# ----------------------------------------------------------------------------- outcomes
def pull_outcomes(ids):
    from google.cloud import bigquery
    if not CDR:
        raise SystemExit("no CDR dataset: set WORKSPACE_CDR")
    client = bigquery.Client()
    Q = lambda sql: client.query(sql).to_dataframe()
    obs = Q(f"""SELECT person_id,
                DATE_DIFF(MAX(observation_period_end_date), MIN(observation_period_start_date), DAY)/365.25 AS obs_years
                FROM `{CDR}.observation_period` GROUP BY person_id""")
    util = Q(f"""SELECT person_id, COUNT(DISTINCT visit_occurrence_id) AS n_visits,
                 COUNTIF(visit_concept_id IN (9203, 262)) AS n_ed_visits
                 FROM `{CDR}.visit_occurrence` GROUP BY person_id""")
    multimorb = Q(f"""
      WITH psych AS (
        SELECT DISTINCT ca.descendant_concept_id AS cid
        FROM `{CDR}.concept_ancestor` ca JOIN `{CDR}.concept` c ON ca.ancestor_concept_id = c.concept_id
        WHERE c.vocabulary_id='SNOMED' AND c.standard_concept='S'
          AND c.concept_name IN ('Mental disorder','Substance-related disorder','Disorder due to psychoactive substance'))
      SELECT co.person_id, COUNT(DISTINCT co.condition_concept_id) AS n_conditions
      FROM `{CDR}.condition_occurrence` co
      WHERE co.condition_concept_id != 0 AND co.condition_concept_id NOT IN (SELECT cid FROM psych)
      GROUP BY co.person_id""")
    codes = ",".join("'" + x + "'" for x in USUAL + COST + DELAYED + COPING + MECH)
    s = Q(f"""SELECT s.person_id, c.concept_code, s.answer
              FROM `{CDR}.ds_survey` s JOIN `{CDR}.concept` c ON s.question_concept_id = c.concept_id
              WHERE c.concept_code IN ({codes})""")
    for df in (obs, util, multimorb, s):
        df["person_id"] = df["person_id"].astype(str)
    s["val"] = s["answer"].astype(str).str.lower().str.rsplit(": ", n=1).str[-1].str.strip()
    yes = lambda cc: set(s[s.concept_code.isin(cc) & (s.val == "yes")].person_id)
    F = pd.DataFrame({"person_id": sorted(set(s.person_id))})
    F["no_usual_source"] = F.person_id.isin(set(s[(s.concept_code == USUAL[0]) & (s.val == "no")].person_id)).astype(int)
    F["any_cost_barrier"] = F.person_id.isin(yes(COST)).astype(int)
    F["any_delayed_care"] = F.person_id.isin(yes(DELAYED)).astype(int)
    F["any_cost_coping"] = F.person_id.isin(yes(COPING)).astype(int)
    F["any_access_barrier"] = F[["no_usual_source", "any_cost_barrier", "any_delayed_care", "any_cost_coping"]].max(axis=1)
    F["care_avoid_provider"] = F.person_id.isin(set(
        s[(s.concept_code == MECH[0]) & (s.val.isin({"some of the time", "most of the time", "always"}))].person_id)).astype(int)
    O = pd.DataFrame({"person_id": [str(i) for i in ids]})
    O = (O.merge(F, on="person_id", how="left").merge(util, on="person_id", how="left")
          .merge(obs, on="person_id", how="left").merge(multimorb, on="person_id", how="left"))
    O["has_survey"] = O["any_access_barrier"].notna()
    O["has_ehr"] = O["n_visits"].fillna(0) >= 1
    O["ed_share"] = np.where(O["has_ehr"], O["n_ed_visits"] / O["n_visits"].replace(0, np.nan), np.nan)
    O["visit_rate"] = O["n_visits"] / O["obs_years"].replace(0, np.nan)
    O["n_conditions"] = O["n_conditions"].fillna(0)
    return O.rename(columns={"person_id": ID_COL}).set_index(ID_COL)


# ----------------------------------------------------------------------------- the JNO group rates
def group_rates_single(ids, O, avail):
    """The flagged set against the rest of the cohort, per outcome, UNADJUSTED (Wilson intervals,
    Newcombe difference, Katz risk ratio) -- the same statistics as section 1 of --legacy, on each
    encoder's own flagged set from scores_<encoder>.csv, and on each comparator's when its scores exist.
    The prespecified encoder is first; the other five are a consistency check.

    ★ The dissemination policy (disclosure.py). A rate times its n gives the count back,
    so a row is withheld when any count it implies is 1 to 20: the set's and the rest's n, the number
    with the outcome and without it in each, and the set's members lacking the outcome's data. The ED
    share is a mean of per-person ratios and a ratio of visit sums, and its row is checked on its n's."""
    from disclosure import SUPP, pair_hidden
    sets = {}
    for name in ENCODERS + COMPARATORS:
        p = os.path.join(OUT_DIR, f"scores_{name}.csv")
        if not os.path.exists(p):
            if name in ENCODERS:
                raise SystemExit(f"{p} not found -- run screen_battery.py first")
            continue
        s = pd.read_csv(p, usecols=[ID_COL, "flagged"]); s[ID_COL] = s[ID_COL].astype(str)
        sets[name] = s.set_index(ID_COL)["flagged"].reindex(ids).fillna(False).astype(bool).values
    rows, n_withheld = [], 0
    print(f"\ngroup rates: each flagged set against the rest of the cohort (unadjusted), "
          f"{ENCODERS[0]} first")
    for set_name, sel in sets.items():
        n_sel = int(sel.sum())
        for o in OUTCOMES:
            a = avail[o]
            y = O[o].values.astype(float)
            g, r = sel & a, (~sel) & a
            no, nr = int(g.sum()), int(r.sum())
            if o in BINARY:
                ko, kr = int(y[g].sum()), int(y[r].sum())
                hide = (pair_hidden(ko, no) or pair_hidden(kr, nr) or pair_hidden(no, n_sel)
                        or pair_hidden(no, no + nr))
                row = {"flag_set": set_name, "outcome": o}
                if hide:
                    row.update({"n_set": SUPP, "n_rest": SUPP})
                    n_withheld += 1
                else:
                    dlo, dhi = newcombe(ko, no, kr, nr)
                    rr, rlo, rhi = rr_katz(ko, no, kr, nr)
                    wl, wh = wilson(ko, no)
                    row.update({"n_set": no, "n_rest": nr, "rate_set": ko / no, "set_lo": wl, "set_hi": wh,
                                "rate_rest": kr / nr, "diff": ko / no - kr / nr, "diff_lo": dlo, "diff_hi": dhi,
                                "rr": rr, "rr_lo": rlo, "rr_hi": rhi})
                rows.append(row)
                if set_name == ENCODERS[0]:
                    print(f"  {o:22s} flagged {100*ko/no:5.1f}% vs rest {100*kr/nr:5.1f}%  RR "
                          f"{(ko/no)/(kr/nr):.2f}  (n {no:,} vs {nr:,})" + ("  [withheld from the file]" if hide else ""))
            else:
                nv, ne = O["n_visits"].values.astype(float), O["n_ed_visits"].values.astype(float)
                hide = pair_hidden(no, n_sel) or pair_hidden(nr, int((~sel).sum())) or no <= 20 or nr <= 20
                row = {"flag_set": set_name, "outcome": o}
                if hide:
                    row.update({"n_set": SUPP, "n_rest": SUPP})
                    n_withheld += 1
                else:
                    row.update({"n_set": no, "n_rest": nr,
                                "rate_set": float(np.nanmean(y[g])), "rate_rest": float(np.nanmean(y[r])),
                                "pooled_set": float(ne[g].sum() / nv[g].sum()), "pooled_rest": float(ne[r].sum() / nv[r].sum()),
                                "diff": float(np.nanmean(y[g]) - np.nanmean(y[r]))})
                rows.append(row)
                if set_name == ENCODERS[0]:
                    print(f"  {o:22s} mean per-person share flagged {100*np.nanmean(y[g]):.1f}% vs rest "
                          f"{100*np.nanmean(y[r]):.1f}%  (n {no:,} vs {nr:,})")
    GR = pd.DataFrame(rows)
    GR.to_csv(os.path.join(OUT_DIR, "care_group_rates.csv"), index=False)
    json.dump({"cdr": CDR, "n_cohort": int(len(ids)), "flag_sets": list(sets),
               "n_flagged": {k: int(v.sum()) for k, v in sets.items()},
               "rows_withheld_by_policy": n_withheld,
               "disclosure": "rows implying a count of 1 to 20 withheld"},
              open(os.path.join(OUT_DIR, "care_group_rates.json"), "w"), indent=2)
    print(f"\nwrote -> {OUT_DIR}: care_group_rates.csv, care_group_rates.json "
          f"({n_withheld} rows withheld under the dissemination policy)")


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ids, items, con, X, cov, w = load_battery()
    N = len(ids)
    resid = residualizer(cov, w)
    Xr = resid(X)
    age, sex = cov[:, 0], cov[:, 1]
    print(f"cohort {N:,} x {len(items)} items | CDR {CDR}")

    # outcomes
    if os.path.exists(OUTCOMES_CSV) and "--repull" not in sys.argv:
        O = pd.read_csv(OUTCOMES_CSV)
        O[ID_COL] = O[ID_COL].astype(str)
        O = O.set_index(ID_COL).reindex(ids)
        print(f"outcomes read from {OUTCOMES_CSV} (--repull to pull again)")
    else:
        O = pull_outcomes(ids)
        O.reset_index().to_csv(OUTCOMES_CSV, index=False)
        print(f"outcomes pulled and written to {OUTCOMES_CSV} (per person, stays)")
    O = O.reindex(ids)
    has_survey, has_ehr = O["has_survey"].astype(bool).values, O["has_ehr"].astype(bool).values
    print(f"answered the access survey {int(has_survey.sum()):,} ({100*has_survey.mean():.0f}%) | "
          f"EHR-linked (>= 1 visit) {int(has_ehr.sum()):,} ({100*has_ehr.mean():.0f}%)")

    # --outcomes-only stops here, having written the file covariates.py, care_covariates.py and
    # care_exclusive.py read. The default goes on to the group rates on each encoder's own flagged set
    # and stops after them. Everything after that block is the --legacy path: the consensus set, the
    # construct level and the comparators, which need consensus_outliers.csv, comparator_weights_*.npz
    # and outlier_leaders_outliers.csv.
    if "--outcomes-only" in sys.argv:
        print(f"\n--outcomes-only: stopping after {OUTCOMES_CSV}.")
        print("The group rates and the --legacy sections are not run. The outcomes file is what")
        print("covariates.py, care_covariates.py and care_exclusive.py read.")
        return
    avail = {o: (has_survey if o in BINARY else has_ehr) for o in OUTCOMES}

    # ★ THE DEFAULT: the flagged set against the rest, on EACH ENCODER'S OWN flagged set,
    # the prespecified one first, and nothing after it. Everything below this block reads the consensus
    # set or the construct level and runs only with --legacy, which needs a file no script here writes.
    if "--legacy" not in sys.argv:
        group_rates_single(ids, O, avail)
        return

    # flag sets and distances
    co = pd.read_csv(os.path.join(OUT_DIR, "consensus_outliers.csv"))
    co[ID_COL] = co[ID_COL].astype(str)
    co = co.set_index(ID_COL).reindex(ids)
    consensus = co["flagged_by_all"].astype(bool).values
    fw_pct = co["outlier_score"].values.astype(float)
    pct, flagged = {"framework": fw_pct}, {"consensus": consensus}
    for enc in ENCODERS:
        s = pd.read_csv(os.path.join(OUT_DIR, f"scores_{enc}.csv")); s[ID_COL] = s[ID_COL].astype(str)
        s = s.set_index(ID_COL).reindex(ids)
        pct[enc] = s["pctile"].values.astype(float)
    for m in COMPARATORS:
        s = pd.read_csv(os.path.join(OUT_DIR, f"scores_{m}.csv")); s[ID_COL] = s[ID_COL].astype(str)
        s = s.set_index(ID_COL).reindex(ids)
        pct[m] = s["pctile"].values.astype(float)
        flagged[m] = s["flagged"].astype(bool).values
    n_out = int(consensus.sum())
    print(f"consensus outliers {n_out:,}; comparator flag sets {[int(flagged[m].sum()) for m in COMPARATORS]}")
    stats = {"cdr": CDR, "n_cohort": N, "n_consensus": n_out,
             "n_survey": int(has_survey.sum()), "n_ehr": int(has_ehr.sum()),
             "n_consensus_survey": int((consensus & has_survey).sum()), "n_consensus_ehr": int((consensus & has_ehr).sum())}

    # ---- 1. group rates
    rows = []
    print("\n1. group rates against the rest of the cohort")
    for set_name, sel in flagged.items():
        for o in OUTCOMES:
            a = avail[o]
            y = O[o].values.astype(float)
            g, r = sel & a, (~sel) & a
            if o in BINARY:
                ko, kr = int(y[g].sum()), int(y[r].sum())
                no, nr = int(g.sum()), int(r.sum())
                dlo, dhi = newcombe(ko, no, kr, nr)
                rr, rlo, rhi = rr_katz(ko, no, kr, nr)
                wl, wh = wilson(ko, no)
                rows.append({"flag_set": set_name, "outcome": o, "n_set": no, "n_rest": nr,
                             "rate_set": ko / no, "set_lo": wl, "set_hi": wh, "rate_rest": kr / nr,
                             "diff": ko / no - kr / nr, "diff_lo": dlo, "diff_hi": dhi, "rr": rr, "rr_lo": rlo, "rr_hi": rhi})
                if set_name == "consensus":
                    print(f"  {o:22s} outliers {100*ko/no:5.1f}% vs rest {100*kr/nr:5.1f}%  diff {100*(ko/no-kr/nr):+5.1f} pp "
                          f"[{100*dlo:+.1f}, {100*dhi:+.1f}]  RR {rr:.2f} [{rlo:.2f}, {rhi:.2f}]  (n {no:,} vs {nr:,})")
            else:
                nv, ne = O["n_visits"].values.astype(float), O["n_ed_visits"].values.astype(float)
                rows.append({"flag_set": set_name, "outcome": o, "n_set": int(g.sum()), "n_rest": int(r.sum()),
                             "rate_set": float(np.nanmean(y[g])), "rate_rest": float(np.nanmean(y[r])),
                             "pooled_set": float(ne[g].sum() / nv[g].sum()), "pooled_rest": float(ne[r].sum() / nv[r].sum()),
                             "diff": float(np.nanmean(y[g]) - np.nanmean(y[r]))})
                if set_name == "consensus":
                    print(f"  {o:22s} mean per-person share outliers {100*np.nanmean(y[g]):.1f}% vs rest {100*np.nanmean(y[r]):.1f}%; "
                          f"pooled {100*ne[g].sum()/nv[g].sum():.1f}% vs {100*ne[r].sum()/nv[r].sum():.1f}%  (n {int(g.sum()):,} vs {int(r.sum()):,})")
    GR = pd.DataFrame(rows)
    GR.to_csv(os.path.join(OUT_DIR, "care_group_rates.csv"), index=False)
    for m in COMPARATORS:
        sub = GR[(GR.flag_set == m) & GR.outcome.isin(BINARY)]
        print(f"  {m:8s} flagged set: " + ", ".join(f"{r.outcome} RR {r.rr:.2f}" for r in sub.itertuples()))

    # ---- 2. cohort gradient: framework, comparator, both
    rows = []
    print("\n2. cohort gradient, rank-OLS (coefficient per SD of rank; binary outcomes as probability)")
    nvis_r = rank_z(O["n_visits"].fillna(0).values)
    age_r = rank_z(age)
    for o in OUTCOMES:
        a = avail[o]
        y = O[o].values.astype(float)[a]
        y = y if o in BINARY else rank_z(y)
        base = [age_r[a], sex[a]] + ([nvis_r[a]] if o == "ed_share" else [])
        bnames = ["age", "sex"] + (["visits"] if o == "ed_share" else [])
        p, ci, pv, r2, n = ols(y, [rank_z(pct["framework"][a])] + base, ["framework"] + bnames)
        rows.append({"outcome": o, "model": "framework alone", "term": "framework", "beta": p["framework"],
                     "lo": ci.loc["framework", 0], "hi": ci.loc["framework", 1], "p": pv["framework"], "r2": r2, "n": n})
        encs = [ols(y, [rank_z(pct[e][a])] + base, ["d"] + bnames)[0]["d"] for e in ENCODERS]
        print(f"  {o:22s} framework beta {p['framework']:+.4f} [{ci.loc['framework',0]:+.4f}, {ci.loc['framework',1]:+.4f}]  "
              f"p {pv['framework']:.1e}  n {n:,}  (per encoder {min(encs):+.4f} to {max(encs):+.4f})")
        for m in COMPARATORS:
            p1, ci1, pv1, r21, _ = ols(y, [rank_z(pct[m][a])] + base, [m] + bnames)
            p2, ci2, pv2, r22, _ = ols(y, [rank_z(pct["framework"][a]), rank_z(pct[m][a])] + base, ["framework", m] + bnames)
            rows.append({"outcome": o, "model": f"{m} alone", "term": m, "beta": p1[m], "lo": ci1.loc[m, 0], "hi": ci1.loc[m, 1], "p": pv1[m], "r2": r21, "n": n})
            for t in ("framework", m):
                rows.append({"outcome": o, "model": f"framework + {m}", "term": t, "beta": p2[t], "lo": ci2.loc[t, 0], "hi": ci2.loc[t, 1], "p": pv2[t], "r2": r22, "n": n})
            print(f"      {m:7s} alone {p1[m]:+.4f}; together: framework {p2['framework']:+.4f} [{ci2.loc['framework',0]:+.4f}, {ci2.loc['framework',1]:+.4f}], "
                  f"{m} {p2[m]:+.4f} [{ci2.loc[m,0]:+.4f}, {ci2.loc[m,1]:+.4f}]")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "care_distance_ols.csv"), index=False)

    # ---- construct contributions for everyone: framework per encoder, and each comparator
    A_fw, T2_fw = {}, {}
    cons = None
    for enc in ENCODERS:
        A_fw[enc], T2_fw[enc], cons = construct_contributions(Xr, framework_P(enc, items), w, con)
    A_cmp = {}
    for m in COMPARATORS:
        wz = np.load(os.path.join(OUT_DIR, f"comparator_weights_{m}.npz"), allow_pickle=True)
        assert [str(x) for x in wz["items"]] == items
        A_cmp[m], _, _ = construct_contributions(Xr, wz["S_eff"], w, con, Om=wz["Om"])
    ci_idx = [i for i, c in enumerate(cons) if c in CIRCUMSTANCES]
    assert len(ci_idx) == len(CIRCUMSTANCES), "a circumstance construct name does not match the items table"

    # ---- 3. care by leading construct among the consensus outliers
    L = pd.read_csv(os.path.join(OUT_DIR, "outlier_leaders_outliers.csv"))
    L[ID_COL] = L[ID_COL].astype(str)
    leader = L.set_index(ID_COL)["leader"].reindex(ids)
    lead_enc = {enc: pd.Series(np.array(cons)[A_fw[enc].argmax(1)], index=ids) for enc in ENCODERS}
    counts = leader[consensus].value_counts()
    big = [c for c, n in counts.items() if n >= MIN_COUNT]           # at 0: every construct that leads anyone
    rows = []
    print(f"\n3. care by leading construct among the consensus outliers (constructs leading >= {max(MIN_COUNT, 1)}; rest = cohort minus outliers)")
    for o in OUTCOMES:
        a = avail[o]
        y = O[o].values.astype(float)
        r = (~consensus) & a
        for c in big:
            g = consensus & a & (leader.values == c)
            if o in BINARY:
                ko, kr, no, nr = int(y[g].sum()), int(y[r].sum()), int(g.sum()), int(r.sum())
                wl, wh = wilson(ko, no); rr, rlo, rhi = rr_katz(ko, no, kr, nr)
                per_enc = [float(np.nanmean(y[consensus & a & (lead_enc[e].values == c)])) for e in ENCODERS]
                rows.append({"outcome": o, "leading_construct": c, "n": no, "rate": ko / no, "lo": wl, "hi": wh,
                             "rate_rest": kr / nr, "rr": rr, "rr_lo": rlo, "rr_hi": rhi,
                             "rate_by_encoder_min": min(per_enc), "rate_by_encoder_max": max(per_enc)})
            else:
                per_enc = [float(np.nanmean(y[consensus & a & (lead_enc[e].values == c)])) for e in ENCODERS]
                rows.append({"outcome": o, "leading_construct": c, "n": int(g.sum()), "rate": float(np.nanmean(y[g])),
                             "rate_rest": float(np.nanmean(y[r])), "rate_by_encoder_min": min(per_enc), "rate_by_encoder_max": max(per_enc)})
    BL = pd.DataFrame(rows)
    BL.to_csv(os.path.join(OUT_DIR, "care_by_leader.csv"), index=False)
    if len(BL) == 0:
        print(f"  no construct leads {max(MIN_COUNT, 1)} or more of the consensus outliers")
    for o in (OUTCOMES if len(BL) else []):
        sub = BL[BL.outcome == o].sort_values("rate", ascending=False)
        print(f"  {o}: " + "; ".join(f"{r.leading_construct} {100*r.rate:.1f}% (n {r.n})" for r in sub.itertuples())
              + f"; rest {100*sub.rate_rest.iloc[0]:.1f}%")

    # ---- 4. care by construct contribution across the cohort
    rows = []
    print("\n4. care by construct contribution across the cohort (OLS on the 30 rank-standardized a_c; framework = mean over encoders)")
    for o in OUTCOMES:
        a = avail[o]
        y = O[o].values.astype(float)[a]
        y = y if o in BINARY else rank_z(y)
        base = [age_r[a], sex[a]] + ([nvis_r[a]] if o == "ed_share" else [])
        bnames = ["age", "sex"] + (["visits"] if o == "ed_share" else [])
        r2_base = ols(y, base, bnames)[3]
        betas, los, his, r2s = [], [], [], []
        for enc in ENCODERS:
            Z = [rank_z(A_fw[enc][a, j]) for j in range(len(cons))]
            p, ci, pv, r2, n = ols(y, Z + base, cons + bnames)
            betas.append(p[cons].values); los.append(ci.loc[cons, 0].values); his.append(ci.loc[cons, 1].values); r2s.append(r2)
        betas, los, his = np.array(betas), np.array(los), np.array(his)
        for j, c in enumerate(cons):
            rows.append({"outcome": o, "model": "framework", "construct": c, "beta_mean": betas[:, j].mean(),
                         "beta_min": betas[:, j].min(), "beta_max": betas[:, j].max(),
                         "lo_mean": los[:, j].mean(), "hi_mean": his[:, j].mean(),
                         "r2_model_mean": float(np.mean(r2s)), "r2_covariates": r2_base, "n": n})
        order = np.argsort(-betas.mean(0))
        print(f"  {o:22s} R2 covariates {r2_base:.4f} -> with the 30 contributions {np.mean(r2s):.4f} (n {n:,})")
        print("      largest positive: " + "; ".join(f"{cons[j]} {betas[:, j].mean():+.4f} [{betas[:, j].min():+.4f}, {betas[:, j].max():+.4f}]" for j in order[:6]))
        print("      largest negative: " + "; ".join(f"{cons[j]} {betas[:, j].mean():+.4f}" for j in order[-3:]))
        for m in COMPARATORS:
            Z = [rank_z(A_cmp[m][a, j]) for j in range(len(cons))]
            p, ci, pv, r2, n = ols(y, Z + base, cons + bnames)
            for j, c in enumerate(cons):
                rows.append({"outcome": o, "model": m, "construct": c, "beta_mean": p[c], "beta_min": p[c], "beta_max": p[c],
                             "lo_mean": ci.loc[c, 0], "hi_mean": ci.loc[c, 1], "r2_model_mean": r2, "r2_covariates": r2_base, "n": n})
            top = p[cons].sort_values(ascending=False)
            print(f"      {m:7s} R2 {r2:.4f}; largest positive: " + "; ".join(f"{c} {v:+.4f}" for c, v in top.head(4).items()))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "care_by_construct_ols.csv"), index=False)

    # ---- 5. the circumstance side, among the consensus outliers
    circ_share = np.column_stack([A_fw[e][:, ci_idx].sum(1) / T2_fw[e] for e in ENCODERS])   # n x 6
    circ_mean = circ_share.mean(1)
    n_circ_enc = [int((circ_share[consensus, k] > 0.5).sum()) for k in range(len(ENCODERS))]
    side = np.where(circ_mean > 0.5, "circumstance", "state")
    rows = []
    print(f"\n5. D1 side among the consensus outliers: circumstance-carried (circumstance share > 1/2) "
          f"{int((circ_mean[consensus] > 0.5).sum())} of {n_out} by the encoder mean; per encoder {min(n_circ_enc)} to {max(n_circ_enc)}; "
          f"mean circumstance share {100*circ_mean[consensus].mean():.1f}%")
    for o in OUTCOMES:
        a = avail[o]; y = O[o].values.astype(float)
        gc, gs = consensus & a & (side == "circumstance"), consensus & a & (side == "state")
        if o in BINARY:
            kc, ks, nc, ns = int(y[gc].sum()), int(y[gs].sum()), int(gc.sum()), int(gs.sum())
            dlo, dhi = newcombe(kc, nc, ks, ns)
            rows.append({"outcome": o, "n_circumstance": nc, "rate_circumstance": kc / nc, "n_state": ns, "rate_state": ks / ns,
                         "diff": kc / nc - ks / ns, "diff_lo": dlo, "diff_hi": dhi})
            print(f"  {o:22s} circumstance-carried {100*kc/nc:.1f}% (n {nc}) vs state-carried {100*ks/ns:.1f}% (n {ns}); diff {100*(kc/nc-ks/ns):+.1f} pp [{100*dlo:+.1f}, {100*dhi:+.1f}]")
        else:
            rows.append({"outcome": o, "n_circumstance": int(gc.sum()), "rate_circumstance": float(np.nanmean(y[gc])),
                         "n_state": int(gs.sum()), "rate_state": float(np.nanmean(y[gs])), "diff": float(np.nanmean(y[gc]) - np.nanmean(y[gs]))})
            print(f"  {o:22s} mean share circumstance-carried {100*np.nanmean(y[gc]):.1f}% (n {int(gc.sum())}) vs state-carried {100*np.nanmean(y[gs]):.1f}% (n {int(gs.sum())})")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "care_by_side.csv"), index=False)
    stats["circumstance_carried_consensus"] = {"by_mean": int((circ_mean[consensus] > 0.5).sum()), "per_encoder": n_circ_enc,
                                               "mean_circumstance_share": float(circ_mean[consensus].mean())}

    # ---- 6. decile gradient and figures (framework percentile)
    d = O.copy()
    d["pct"] = fw_pct
    d["decile"] = pd.qcut(d["pct"], 10, labels=False) + 1
    g = d.groupby("decile").agg(n=("pct", "size"), mean_pctile=("pct", "mean"),
                                access_barrier_pct=("any_access_barrier", lambda x: 100 * np.nanmean(x)),
                                cost_barrier_pct=("any_cost_barrier", lambda x: 100 * np.nanmean(x)),
                                care_avoid_pct=("care_avoid_provider", lambda x: 100 * np.nanmean(x)),
                                median_visits=("n_visits", "median"), median_rate=("visit_rate", "median"),
                                ed_share_pct=("ed_share", lambda x: 100 * np.nanmean(x)),
                                median_conditions=("n_conditions", "median"))
    g.to_csv(os.path.join(OUT_DIR, "care_gradient.csv"))
    print("\n6. decile gradient on the framework percentile")
    print(g.round(2).to_string())
    print(f"[diagnostic, not reported] multimorbidity: median conditions outliers "
          f"{np.nanmedian(O['n_conditions'].values[consensus & has_ehr]):.0f} vs rest {np.nanmedian(O['n_conditions'].values[(~consensus) & has_ehr]):.0f}")

    fig, axL = plt.subplots(figsize=(6.9, 3.4), facecolor="white")
    axL.plot(g.mean_pctile, g.access_barrier_pct, "-o", ms=3.5, color=OKABE["red"], label="any access barrier")
    axL.plot(g.mean_pctile, g.cost_barrier_pct, "-s", ms=3.5, color=OKABE["orange"], label="cost barrier")
    axL.plot(g.mean_pctile, g.care_avoid_pct, "-^", ms=3.5, color=OKABE["purple"], label="care avoidance, provider race or religion")
    axL.set_xlabel("mean distance percentile", fontsize=8); axL.set_ylabel("% reporting (survey)", fontsize=8)
    axR = axL.twinx()
    axR.plot(g.mean_pctile, g.ed_share_pct, "--D", ms=3.5, color=OKABE["blue"], label="ED share of visits (records)")
    axR.set_ylabel("ED share of visits (%)", fontsize=8, color=OKABE["blue"])
    hl, ll = axL.get_legend_handles_labels(); hr, lr = axR.get_legend_handles_labels()
    axL.legend(hl + hr, ll + lr, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=7.5)
    fig.subplots_adjust(left=0.09, right=0.90, top=0.96, bottom=0.36)
    fig.savefig(os.path.join(OUT_DIR, "fig_care_gradient.png"), dpi=300, facecolor="white"); plt.close(fig)
    figs, axv = plt.subplots(figsize=(6.9, 3.4), facecolor="white")
    axv.plot(g.mean_pctile, g.median_visits, "--D", ms=3.5, color=OKABE["blue"], label="median total visits")
    axr = axv.twinx()
    axr.plot(g.mean_pctile, g.median_rate, "-o", ms=3.5, color=OKABE["green"], label="median visits per observation-year")
    axv.set_xlabel("mean distance percentile", fontsize=8)
    hv, lv = axv.get_legend_handles_labels(); hr2, lr2 = axr.get_legend_handles_labels()
    axv.legend(hv + hr2, lv + lr2, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=7.5)
    figs.subplots_adjust(left=0.09, right=0.90, top=0.96, bottom=0.30)
    figs.savefig(os.path.join(OUT_DIR, "fig_care_visits_supp.png"), dpi=300, facecolor="white"); plt.close(figs)

    stats["gradient"] = json.loads(g.reset_index().to_json(orient="records"))
    json.dump(stats, open(os.path.join(OUT_DIR, "care_stats.json"), "w"), indent=2)
    print(f"\nwrote -> {OUT_DIR}: care_group_rates, care_distance_ols, care_by_leader, care_by_construct_ols, care_by_side, care_gradient, care_stats, two figures")


if __name__ == "__main__":
    main()
