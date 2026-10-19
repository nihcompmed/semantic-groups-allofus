#!/usr/bin/env python3
"""step5_outcomes.py -- the outcomes recorded outside the 151 items, per participant (Methods, Outcomes; Appendix 3).

None of these enters the scores. Each is pulled once and kept in work/ (person level, never downloaded); a rerun reuses
the files (--refresh pulls again).

  care_outcomes.csv   the Health Care Access and Utilization survey and the visits (care_outcomes)
                        any_access_barrier   no usual source of care, any cost barrier, any delay of care for a
                                             practical reason, or cutting back on medication to save money
                        any_cost_barrier, care_avoid_provider (delayed or no care because of a provider's race or
                                             religion, some of the time or more)
                        has_survey           any stored answer to the 21 questions (the barriers' denominator)
                        ed_share             emergency department visits (concepts 9203, 262) over all visits, among
                                             those with at least 1 visit (has_ehr)
  answer_dates.csv, sleep_person.csv, wear_enrollees.csv   Fitbit sleep (sleep): a night counts when it is the main
                      sleep, falls within 180 days of the participant's last answer to the 151 items and lasts 60 to
                      960 minutes; per person the number of such nights and their mean and SD of minutes asleep; and the
                      participants of the program's wear study (devices the program provided)
  test_checks.csv     whether the record shows a cholesterol test, a mammogram or a blood PSA test (the populations of
                      3 recorded conditions), searched by concept in 4 tables, standard and source columns (condition_tests)
  visits_person.csv   visits per person, type and source (visits), for the emergency department check of step 7

DOWNLOADS
  sleep_reliability   the reliability of a personal sleep mean and why 14 nights: within- and between-person variance,
                      R(n) = s2b / (s2b + s2w / n) for n = 1 to 60, the nights needed for 0.90, and by decile of the
                      encoder chosen in advance how many have 14 nights
  care_answers        how the stored answers to the 21 barrier questions are read: every stored answer with its count
                      and reading, the participants with no usable answer, and the library versions (Methods)

Run:  python3 step5_outcomes.py
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from release import MARK, SUPP, aggregate_only, marked, pair_hidden, partition_mask, versions, write_download
from settings import CDR, COHORT_CSV, ID, ITEMS_CSV, MIN_NIGHTS, PRIMARY, download_dir, work

REFRESH = "--refresh" in sys.argv
DRY_RUN_LIMIT_GB = 50
MAX_GB = 400.0

# the 21 barrier questions
USUAL = ["HealthAdvice_PlaceforHealthAdvice"]
COST = ["CantAffordCare_HealthcareProvider", "CantAffordCare_PrescriptionMedicines", "CantAffordCare_Specialist",
        "CantAffordCare_FollowupCare", "CantAffordCare_EmergencyCare", "CantAffordCare_DentalCare",
        "CantAffordCare_Eyeglasses", "DelayedMedicalCare_CantAffordCoPay", "DelayedMedicalCare_DeductibleTooHigh",
        "DelayedMedicalCare_HadToPayOutOfPocket"]
DELAYED = ["DelayedMedicalCare_ChildCare", "DelayedMedicalCare_ElderlyCare", "DelayedMedicalCare_RuralArea",
           "DelayedMedicalCare_TimeOffWork", "DelayedMedicalCare_Transportation"]
COPING = ["CantAffordCare_DelayedFillingRxToSaveMoney", "CantAffordCare_LowerCostRxToSaveMoney",
          "CantAffordCare_SkippedMedToSaveMoney", "CantAffordCare_TookLessMedToSaveMoney"]
MECH = ["HealthProviderRaceReligion_DelayedOrNoCare"]
PARTS = {"no usual source": USUAL, "cost": COST, "delayed care": DELAYED, "medication coping": COPING,
         "clinician differs": MECH}
COLUMN = {"no usual source": "no_usual_source", "cost": "any_cost_barrier", "delayed care": "any_delayed_care",
          "medication coping": "any_cost_coping", "clinician differs": "care_avoid_provider"}
FREQ_BARRIER = {"some of the time", "most of the time", "always"}
FREQ_ANSWERS = FREQ_BARRIER | {"none of the time"}
NONANSWER = re.compile(r"^(skip|prefer\s*not.*|don'?t\s*know|dontknow)$")

# sleep
SLEEP_MIN_MIN, SLEEP_MAX_MIN = 60, 960
NIGHT_FLOOR, TARGET, CURVE = 7, 0.90, range(1, 61)

# the tests behind the populations of 3 recorded conditions
TESTS = {
    "lipid": {"loinc": ["2093-3", "2085-9", "13457-7", "18262-6", "2089-1", "43396-1", "9830-1", "24331-1", "57698-3"],
              "proc": ["80061", "82465", "83718", "83721"], "icd": []},
    "breast": {"loinc": [],
               "proc": ["77067", "77065", "77066", "77057", "77056", "77055", "77063", "77061", "77062", "G0202",
                        "G0204", "G0206"],
               "icd": ["Z12.31", "Z12.39", "V76.10", "V76.11", "V76.12", "V76.19"]},
    "prostate": {"loinc": ["2857-1", "19195-7", "19197-3", "35741-8", "83112-3", "10886-0", "19201-3", "19203-9",
                           "83113-1", "12841-3", "14120-0", "72576-2", "33667-7", "53764-7", "100716-0"],
                 "proc": ["84153", "84152", "84154", "G0103"], "icd": ["Z12.5", "V76.44"]},
}


class Repository:
    """BigQuery, with the cost of each query checked by a dry run first."""

    def __init__(self):
        from google.cloud import bigquery
        if not CDR:
            raise SystemExit("WORKSPACE_CDR is not set")
        self.bq = bigquery
        self.client = bigquery.Client()

    def query(self, sql, params=(), limit_gb=DRY_RUN_LIMIT_GB):
        cfg = self.bq.QueryJobConfig(query_parameters=list(params), dry_run=True, use_query_cache=False)
        gb = self.client.query(sql, job_config=cfg).total_bytes_processed / 1e9
        assert gb <= limit_gb, f"the query would scan {gb:,.0f} GB, above the {limit_gb:,.0f} GB limit"
        return self.client.query(sql, job_config=self.bq.QueryJobConfig(query_parameters=list(params))).to_dataframe()

    def ids(self, name, values):
        return self.bq.ArrayQueryParameter(name, "INT64", [int(x) for x in values])

    def array(self, name, kind, values):
        return self.bq.ArrayQueryParameter(name, kind, list(values))


# ------------------------------------------------------------------------------------------------- care
def survey_answers(db):
    """Every stored answer to the 21 barrier questions, with the survey it belongs to."""
    codes = ",".join("'" + x + "'" for v in PARTS.values() for x in v)
    s = db.query(f"""SELECT s.person_id, c.concept_code, s.answer, s.survey
                     FROM `{CDR}.ds_survey` s JOIN `{CDR}.concept` c ON s.question_concept_id = c.concept_id
                     WHERE c.concept_code IN ({codes})""")
    s["person_id"] = s["person_id"].astype(str)
    return s


def care_outcomes(db, ids, s):
    """The barriers per person (from the stored answers s), the visits and the emergency department share."""
    obs = db.query(f"""SELECT person_id,
                DATE_DIFF(MAX(observation_period_end_date), MIN(observation_period_start_date), DAY)/365.25 AS obs_years
                FROM `{CDR}.observation_period` GROUP BY person_id""")
    util = db.query(f"""SELECT person_id, COUNT(DISTINCT visit_occurrence_id) AS n_visits,
                 COUNTIF(visit_concept_id IN (9203, 262)) AS n_ed_visits
                 FROM `{CDR}.visit_occurrence` GROUP BY person_id""")
    multimorb = db.query(f"""
      WITH psych AS (
        SELECT DISTINCT ca.descendant_concept_id AS cid
        FROM `{CDR}.concept_ancestor` ca JOIN `{CDR}.concept` c ON ca.ancestor_concept_id = c.concept_id
        WHERE c.vocabulary_id='SNOMED' AND c.standard_concept='S'
          AND c.concept_name IN ('Mental disorder','Substance-related disorder','Disorder due to psychoactive substance'))
      SELECT co.person_id, COUNT(DISTINCT co.condition_concept_id) AS n_conditions
      FROM `{CDR}.condition_occurrence` co
      WHERE co.condition_concept_id != 0 AND co.condition_concept_id NOT IN (SELECT cid FROM psych)
      GROUP BY co.person_id""")
    for df in (obs, util, multimorb):
        df["person_id"] = df["person_id"].astype(str)
    s = s[["person_id", "concept_code", "answer"]].copy()
    s["val"] = s["answer"].astype(str).str.lower().str.rsplit(": ", n=1).str[-1].str.strip()
    yes = lambda cc: set(s[s.concept_code.isin(cc) & (s.val == "yes")].person_id)
    F = pd.DataFrame({"person_id": sorted(set(s.person_id))})
    F["no_usual_source"] = F.person_id.isin(set(s[(s.concept_code == USUAL[0]) & (s.val == "no")].person_id)).astype(int)
    F["any_cost_barrier"] = F.person_id.isin(yes(COST)).astype(int)
    F["any_delayed_care"] = F.person_id.isin(yes(DELAYED)).astype(int)
    F["any_cost_coping"] = F.person_id.isin(yes(COPING)).astype(int)
    F["any_access_barrier"] = F[["no_usual_source", "any_cost_barrier", "any_delayed_care", "any_cost_coping"]].max(axis=1)
    F["care_avoid_provider"] = F.person_id.isin(set(
        s[(s.concept_code == MECH[0]) & (s.val.isin(FREQ_BARRIER))].person_id)).astype(int)
    O = pd.DataFrame({"person_id": [str(i) for i in ids]})
    O = (O.merge(F, on="person_id", how="left").merge(util, on="person_id", how="left")
          .merge(obs, on="person_id", how="left").merge(multimorb, on="person_id", how="left"))
    O["has_survey"] = O["any_access_barrier"].notna()
    O["has_ehr"] = O["n_visits"].fillna(0) >= 1
    O["ed_share"] = np.where(O["has_ehr"], O["n_ed_visits"] / O["n_visits"].replace(0, np.nan), np.nan)
    O["visit_rate"] = O["n_visits"] / O["obs_years"].replace(0, np.nan)
    O["n_conditions"] = O["n_conditions"].fillna(0)
    return O.rename(columns={"person_id": ID}).set_index(ID)


def parse(answer):
    """The stored answer as read: the text after the last ': ', lowercased and stripped."""
    return str(answer).lower().rsplit(": ", 1)[-1].strip()


def reading(part, answer):
    p = parse(answer)
    hit = p == "no" if part == "no usual source" else (p in FREQ_BARRIER if part == "clinician differs" else p == "yes")
    if hit:
        return "barrier"
    return "no answer" if NONANSWER.match(p) else "not a barrier"


def care_answers(db, ids, s, O):
    """The download care_answers: every stored answer with its count and reading; the participants with no usable
    answer. The per-person barriers recomputed here must equal care_outcomes.csv."""
    s = s[s["person_id"].isin(set(ids))].copy()
    names = sorted(set(s["survey"].dropna().astype(str)))
    quoted = ",".join("'" + n.replace("'", "\\'") + "'" for n in names)
    m = db.query(f"""SELECT DISTINCT person_id FROM `{CDR}.ds_survey` WHERE survey IN ({quoted})""")
    in_modules = set(m["person_id"].astype(str)) & set(ids)
    s["val"] = s["answer"].astype(str).str.lower().str.rsplit(": ", n=1).str[-1].str.strip()
    yes = lambda cc: set(s[s.concept_code.isin(cc) & (s.val == "yes")].person_id)
    F = pd.DataFrame(index=pd.Index(ids, name=ID))
    F["has_survey"] = F.index.isin(set(s.person_id))
    F["no_usual_source"] = F.index.isin(set(s[(s.concept_code == USUAL[0]) & (s.val == "no")].person_id))
    F["any_cost_barrier"] = F.index.isin(yes(COST))
    F["any_delayed_care"] = F.index.isin(yes(DELAYED))
    F["any_cost_coping"] = F.index.isin(yes(COPING))
    F["any_access_barrier"] = F[["no_usual_source", "any_cost_barrier", "any_delayed_care", "any_cost_coping"]].any(axis=1)
    F["care_avoid_provider"] = F.index.isin(set(s[(s.concept_code == MECH[0]) & (s.val.isin(FREQ_BARRIER))].person_id))
    Oi = O.reindex(ids)
    for c in ["has_survey"] + list(COLUMN.values()) + ["any_access_barrier"]:
        theirs = (Oi[c].fillna(False).astype(bool) if c == "has_survey" else Oi[c].fillna(0).astype(float).astype(bool)).values
        assert (F[c].values.astype(bool) == theirs).all(), f"the stored answers do not reproduce care_outcomes.csv ({c})"
    has = F["has_survey"].values
    n_has, N = int(has.sum()), len(ids)
    rows, per_q = [], {}
    part_of = {q: p for p, qs in PARTS.items() for q in qs}
    for q in [x for v in PARTS.values() for x in v]:
        d = s[s.concept_code == q]
        k = d.groupby("person_id").size()
        per_q[q] = {"n_people_answered": int(len(k)), "n_people_more_than_1_answer": int((k > 1).sum())}
        g = d.groupby("answer")["person_id"].nunique().sort_values(ascending=False)
        counts = g.values.astype(int)
        hide, prim = partition_mask(counts)
        for (a, _), n in zip(g.items(), marked(counts, hide, prim)):
            rows.append({"question": q, "part": part_of[q], "answer": a, "parsed": parse(a),
                         "reading": reading(part_of[q], a), "n_people": n})
    A = pd.DataFrame(rows)
    hidden_q = {q for q in A.question.unique() if A[A.question == q].n_people.isin([MARK, SUPP]).any()}
    subst_clin = set(s[(s.concept_code == MECH[0]) & (s["answer"].map(parse).isin(FREQ_ANSWERS))].person_id)
    barrier_qs = [x for p, v in PARTS.items() if p != "clinician differs" for x in v]
    b = s[s.concept_code.isin(barrier_qs)]
    subst_items = set(b[~b["answer"].map(parse).str.match(NONANSWER)].person_id)
    idx = pd.Index(ids)
    n_no_clin = int((has & ~idx.isin(subst_clin)).sum())
    n_no_items = int((has & ~idx.isin(subst_items)).sum())
    shown = lambda c, total: (MARK if 0 < c <= 20 else SUPP) if pair_hidden(c, total) else int(c)
    summary = {
        "cdr": CDR, "n_cohort": N, "n_has_survey": shown(n_has, N), "n_in_survey_modules": shown(len(in_modules), N),
        "survey_modules": names,
        "n_part": {p: (SUPP if len(PARTS[p]) == 1 and PARTS[p][0] in hidden_q else shown(int(F[c].sum()), n_has))
                   for p, c in COLUMN.items()},
        "n_any_access_barrier": shown(int(F["any_access_barrier"].sum()), n_has),
        "n_no_substantive_clinician": SUPP if MECH[0] in hidden_q else shown(n_no_clin, n_has),
        "n_no_substantive_barrier_items": shown(n_no_items, n_has),
        "per_question": {q: {k: shown(v, n_has) for k, v in d.items()} for q, d in per_q.items()},
        "checked_against_care_outcomes": True,
        "parsing": "the stored answer after its last ': ', lowercased; barrier = 'yes' (cost, delayed care, medication "
                   "coping), 'no' (no usual source), some/most of the time or always (clinician differs)",
        "versions": versions()}
    d = download_dir("care_answers")
    A.to_csv(os.path.join(d, "care_answer_strings.csv"), index=False)
    json.dump(summary, open(os.path.join(d, "care_answer_summary.json"), "w"), indent=2,
              default=lambda o: int(o) if isinstance(o, np.integer) else str(o))
    write_download("care_answers", ["care_answer_strings.csv", "care_answer_summary.json"], aggregate_only)


# ------------------------------------------------------------------------------------------------- sleep
def answer_dates(db, ids):
    """Each participant's last answer date on the 151 items."""
    cids = pd.read_csv(ITEMS_CSV, usecols=["concept_id"])["concept_id"].astype("int64").tolist()
    d = db.query(f"""SELECT person_id, MAX(observation_datetime) AS last_answer
                     FROM `{CDR}.observation`
                     WHERE person_id IN UNNEST(@ids) AND observation_source_concept_id IN UNNEST(@cids)
                     GROUP BY person_id""", [db.ids("ids", ids), db.ids("cids", cids)])
    d["last_answer"] = pd.to_datetime(d.last_answer)
    if getattr(d.last_answer.dt, "tz", None) is not None:
        d["last_answer"] = d.last_answer.dt.tz_localize(None)
    return d


def sleep_person(db, ids, adate):
    """Per participant, the counted nights: their number, mean and SD of minutes asleep (and the same unfiltered)."""
    adate = adate.reindex(ids.values)
    ok = adate.notna().values
    main = "LOWER(CAST(v.is_main_sleep AS STRING)) IN ('true', '1', 't', 'yes')"
    win = "DATE(v.sleep_date) BETWEEN DATE_SUB(a.adt, INTERVAL 180 DAY) AND DATE_ADD(a.adt, INTERVAL 180 DAY)"
    params = [db.ids("pids", ids.values[ok]), db.array("adates", "DATE", adate.dt.date.values[ok])]
    return db.query(f"""WITH anchor AS (
                  SELECT pid, adt FROM UNNEST(@pids) AS pid WITH OFFSET i
                  JOIN (SELECT adt, o2 FROM UNNEST(@adates) AS adt WITH OFFSET o2) ON i = o2)
                  SELECT a.pid AS person_id,
                         COUNT(*) AS nights_all,
                         AVG(v.minute_asleep) AS mean_all,
                         STDDEV_SAMP(v.minute_asleep) AS sd_all,
                         COUNTIF(v.minute_asleep BETWEEN {SLEEP_MIN_MIN} AND {SLEEP_MAX_MIN}) AS nights,
                         AVG(IF(v.minute_asleep BETWEEN {SLEEP_MIN_MIN} AND {SLEEP_MAX_MIN},
                                v.minute_asleep, NULL)) AS mean_min,
                         STDDEV_SAMP(IF(v.minute_asleep BETWEEN {SLEEP_MIN_MIN} AND {SLEEP_MAX_MIN},
                                        v.minute_asleep, NULL)) AS sd_min
                  FROM anchor a JOIN `{CDR}.sleep_daily_summary` v ON v.person_id = a.pid
                  WHERE {main} AND {win}
                  GROUP BY a.pid""", params)


def wear_enrollees(db, ids):
    return db.query(f"""SELECT DISTINCT person_id FROM `{CDR}.wear_study`
                        WHERE person_id IN UNNEST(@ids) AND wear_consent_start_date IS NOT NULL""",
                    [db.ids("ids", ids)])


def sleep_reliability(ids):
    """The download sleep_reliability (see the docstring)."""
    P = pd.read_csv(work("sleep_person.csv"))
    P["person_id"] = P["person_id"].astype(str)
    P = P.set_index("person_id").reindex(ids).dropna(subset=["nights"])
    N = len(ids)
    n_any = int((P.nights >= 1).sum())
    K = P[(P.nights >= NIGHT_FLOOR) & P.sd_min.notna()]
    w = (K.nights - 1).clip(lower=0)
    s2w = float((w * K.sd_min ** 2).sum() / max(w.sum(), 1))
    obs = float(K.mean_min.var(ddof=1))
    noise = float((K.sd_min ** 2 / K.nights).mean())
    s2b = max(obs - noise, 1e-9)
    rel = lambda n: s2b / (s2b + s2w / n)
    need = int(np.ceil((TARGET / (1 - TARGET)) * s2w / s2b))
    d = download_dir("sleep_reliability")
    pd.DataFrame([{
        "n_people_cohort": N, "n_people_any_night": n_any, "n_people_floor": len(K),
        "n_people_min_nights": int((P.nights >= MIN_NIGHTS).sum()),
        "night_floor": NIGHT_FLOOR, "min_nights": MIN_NIGHTS, "target_reliability": TARGET,
        "within_sd_median": float(K.sd_min.median()), "within_sd_median_unfiltered": float(K.sd_all.median()),
        "pooled_within_var": s2w, "observed_between_var": obs, "sampling_noise_var": noise,
        "true_between_var": s2b, "pooled_within_sd": np.sqrt(s2w), "true_between_sd": np.sqrt(s2b),
        "reliability_at_min_nights": rel(MIN_NIGHTS), "se_at_min_nights": np.sqrt(s2w / MIN_NIGHTS),
        "nights_needed": need, "reliability_at_nights_needed": rel(need),
        "encoder_for_deciles": PRIMARY, "cdr": CDR}]).to_csv(os.path.join(d, "sleep_reliability_summary.csv"), index=False)
    pd.DataFrame({"nights": list(CURVE), "se_of_mean_min": [np.sqrt(s2w / n) for n in CURVE],
                  "reliability": [rel(n) for n in CURVE]}).to_csv(os.path.join(d, "sleep_reliability_curve.csv"),
                                                                  index=False)
    M = pd.read_csv(work(f"scores_{PRIMARY}.csv"), usecols=[ID, "pctile"])
    M[ID] = M[ID].astype(str)
    M = M.set_index(ID)["pctile"].reindex(ids).dropna()
    dec = (M.rank(method="first").sub(1) * 10 // len(M) + 1).astype(int)
    T = K.assign(decile=dec.reindex(K.index).values).dropna(subset=["decile"])
    rows = []
    for dd, g in T.groupby("decile"):
        n = len(g)
        r = {"decile": int(dd), "n_people": n, "median_nights": float(g.nights.median()),
             "median_within_sd": float(g.sd_min.median())}
        for thr, col in ((MIN_NIGHTS, "share_with_min_nights"), (need, "share_with_nights_needed")):
            k = int((g.nights >= thr).sum())
            r[col] = np.nan if pair_hidden(k, n) else k / n
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(d, "sleep_nights_by_decile.csv"), index=False)
    print(f"  sleep: 14-night reliability {rel(MIN_NIGHTS):.3f}; nights needed for {TARGET:.2f}: {need}", flush=True)
    write_download("sleep_reliability", ["sleep_reliability_summary.csv", "sleep_reliability_curve.csv",
                                         "sleep_nights_by_decile.csv"], aggregate_only)


# ------------------------------------------------------------------------------------------------- tests and visits
def test_checks(db, ids):
    """Per participant and test (lipid, breast, prostate): its first date and number of dates in the record."""
    parts, params = [], []
    for kind, c in TESTS.items():
        where = []
        if c["loinc"]:
            where.append(f"(vocabulary_id = 'LOINC' AND concept_code IN UNNEST(@{kind}_loinc))")
            params.append(db.array(f"{kind}_loinc", "STRING", c["loinc"]))
        if c["proc"]:
            where.append(f"(vocabulary_id IN ('CPT4', 'HCPCS') AND concept_code IN UNNEST(@{kind}_proc))")
            params.append(db.array(f"{kind}_proc", "STRING", c["proc"]))
        if c["icd"]:
            where.append(f"(vocabulary_id IN ('ICD10CM', 'ICD9CM') AND concept_code IN UNNEST(@{kind}_icd))")
            params.append(db.array(f"{kind}_icd", "STRING", c["icd"]))
        parts.append(f"SELECT concept_id, concept_code, vocabulary_id, concept_name, '{kind}' AS kind "
                     f"FROM `{CDR}.concept` WHERE " + " OR ".join(where))
    codes_q = " UNION ALL ".join(parts)
    tables = [("procedure_occurrence", "procedure_date", "procedure"), ("measurement", "measurement_date", "measurement"),
              ("observation", "observation_date", "observation"),
              ("condition_occurrence", "condition_start_date", "condition")]
    ev = " UNION ALL ".join(
        f"SELECT person_id, {d} AS d, {p}_{col}concept_id AS c FROM `{CDR}.{t}` WHERE person_id IN UNNEST(@ids)"
        for t, d, p in tables for col in ("", "source_"))
    T = db.query(f"""
      WITH codes AS ({codes_q}),
      ev AS ({ev})
      SELECT ev.person_id, codes.kind, MIN(ev.d) AS first_date, COUNT(DISTINCT ev.d) AS n_dates
      FROM ev JOIN (SELECT DISTINCT concept_id, kind FROM codes) AS codes ON ev.c = codes.concept_id
      GROUP BY ev.person_id, codes.kind""", params + [db.ids("ids", ids)], MAX_GB).rename(columns={"person_id": ID})
    T[ID] = T[ID].astype(str)
    return T


def visits(db, ids):
    """Visits per participant x visit type x source (visit_occurrence_ext.src_id with its digits removed)."""
    sql = """SELECT v.person_id, v.visit_concept_id, IFNULL(c.concept_name, 'No matching concept') AS visit_type,
                    {src} AS source, COUNT(DISTINCT v.visit_occurrence_id) AS n_visits
             FROM `{cdr}.visit_occurrence` v {join}
             LEFT JOIN `{cdr}.concept` c ON c.concept_id = v.visit_concept_id
             WHERE v.person_id IN UNNEST(@ids)
             GROUP BY 1, 2, 3, 4"""
    p = [db.ids("ids", ids)]
    try:
        V = db.query(sql.format(cdr=CDR, src="IFNULL(TRIM(REGEXP_REPLACE(e.src_id, r'[0-9]+', '')), 'not recorded')",
                                join=f"LEFT JOIN `{CDR}.visit_occurrence_ext` e ON e.visit_occurrence_id = v.visit_occurrence_id"),
                     p)
    except Exception as ex:                                                  # the source is optional
        print(f"  visit_occurrence_ext could not be joined ({type(ex).__name__}): source written 'not available'")
        V = db.query(sql.format(cdr=CDR, src="'not available'", join=""), p)
    V["person_id"] = V["person_id"].astype(str)
    return V


def main():
    cohort = pd.read_csv(COHORT_CSV, usecols=[ID])[ID]
    ids = cohort.astype(str).tolist()
    ids64 = cohort.astype("int64")
    need = lambda f: REFRESH or not os.path.exists(work(f))

    db = Repository()
    if need("care_outcomes.csv") or need("care_answers_rows.csv"):
        s = survey_answers(db)
        s.to_csv(work("care_answers_rows.csv"), index=False)
        care_outcomes(db, ids, s).reset_index().to_csv(work("care_outcomes.csv"), index=False)
        print("  care outcomes pulled", flush=True)
    else:
        s = pd.read_csv(work("care_answers_rows.csv"), dtype=str, keep_default_na=False, na_values=[""])
    O = pd.read_csv(work("care_outcomes.csv"))
    O[ID] = O[ID].astype(str)
    O = O.set_index(ID)
    care_answers(db, ids, s, O)

    if need("answer_dates.csv"):
        answer_dates(db, ids64).to_csv(work("answer_dates.csv"), index=False)
    if need("sleep_person.csv"):
        A = pd.read_csv(work("answer_dates.csv"))
        A["last_answer"] = pd.to_datetime(A.last_answer)
        sleep_person(db, ids64, A.set_index("person_id")["last_answer"]).to_csv(work("sleep_person.csv"), index=False)
    if need("wear_enrollees.csv"):
        wear_enrollees(db, ids64).to_csv(work("wear_enrollees.csv"), index=False)
    sleep_reliability(ids)

    if need("test_checks.csv"):
        test_checks(db, ids).to_csv(work("test_checks.csv"), index=False)
    if need("visits_person.csv"):
        visits(db, ids).to_csv(work("visits_person.csv"), index=False)
    print("[step 5] done", flush=True)


if __name__ == "__main__":
    main()
