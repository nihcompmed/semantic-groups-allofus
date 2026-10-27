#!/usr/bin/env python3
"""step6_cohort.py -- who the participants are: the cohort table (Supplementary File 2) and the participants against
the rest of the release (Supplementary File 3).

THE COVARIATES, self-reported in The Basics, coded as the All of Us papers in JAMA journals code them, every non-answer
kept as a level and nothing imputed; the text of the stored answer after its last ": " is mapped, so the mapping does
not depend on answer concept ids:
  race_ethnicity   Hispanic or Latino (any Hispanic answer), else the single race selected, else "More than one",
                   "None of these" or "Prefer not to answer" (a skip, prefer-not or no answer)
  income           <25k / 25k to <50k / 50k to <100k / 100k to <150k / 150k or more / Prefer not to answer
  education        Less than high school / High school or GED / Some college / College graduate or more / Prefer not
  employment       Employed / Not employed / Retired / Other (homemaker, student, unable to work) / Prefer not
                   (several answers resolved in that order)
  insurance        Yes / No / Prefer not to answer
Age and sex at birth come from step 1.

THE COHORT TABLE (download cohort_table): age, sex, the thresholds and each covariate in the cohort, in the subsets each
outcome is measured on (answered the access survey; at least 1 visit; at least 1 ICD event; at least 14 Fitbit nights),
and in each encoder's outliers. A level holding 1 to 20 people in any population, or in the complement of any subset in
the cohort, joins "Other (pooled)" in every population, and the pool grows until it too is 0 or more than 20.

THE PARTICIPANTS AGAINST THE RELEASE (download selection). The release's person table in 3 parts: included (the cohort),
excluded_record (a record for all 151 items but not in the cohort: sex at birth not female or male, age missing, or more
than 5 items missing) and rest. Rows: age on the data cutoff date (January 1, 2025), sex at birth, income, education,
employment, insurance, and the program's flags for linked records and Fitbit data (cb_search_person). The standardized
difference of each part against the included (no P values). Levels pooled as in the cohort table, also against the
release and every part's complement.

OUTPUTS  work/covariates.csv (per person)
DOWNLOADS cohort_table (cohort_table.csv, cohort_answer_map.csv), selection (selection_table.csv, selection_meta.json)

Run:  python3 step6_cohort.py
"""
import json
import os
import re

import numpy as np
import pandas as pd

from release import MAX_SUPPRESSED, SUPP, aggregate_only, marked, pair_hidden, partition_mask, versions, write_download
from settings import CDR, COHORT_CSV, ENCODERS, ID, MIN_NIGHTS, RESPONSES, download_dir, work

TABLE_FLOOR = MAX_SUPPRESSED + 1
PNA = "Prefer not to answer"
QUESTIONS = {1586140: "race_ethnicity", 1585375: "income", 1585940: "education", 1585952: "employment",
             1585386: "insurance"}
COVARIATES = ["race_ethnicity", "income", "education", "employment", "insurance"]
LEVELS = {
    "race_ethnicity": ["White", "Black or African American", "Hispanic or Latino", "Asian", "More than one",
                       "Another single population", "Middle Eastern or North African",
                       "Native Hawaiian or other Pacific Islander", "American Indian or Alaska Native", "None of these",
                       PNA],
    "income": ["<25k", "25k to <50k", "50k to <100k", "100k to <150k", "150k or more", PNA],
    "education": ["Less than high school", "High school or GED", "Some college", "College graduate or more", PNA],
    "employment": ["Employed", "Not employed", "Retired", "Other", PNA],
    "insurance": ["Yes", "No", PNA]}
POOLED = "Other (pooled)"
CUTOFF = pd.Timestamp("2025-01-01")
POPS = ["included", "excluded_record", "rest"]
BASICS = {q: c for q, c in QUESTIONS.items() if c != "race_ethnicity"}
FLAGS = {"ehr": "has_ehr_data", "fitbit": "has_fitbit"}
SEX_LEVELS = ["Female", "Male", "Other or no answer"]
SEX_SOURCE_PREFIX_RE = r"^(sexatbirth|biologicalsexatbirth|sex|gender|genderidentity)_"
SEX_AT_BIRTH_ENC = {"male": "Male", "female": "Female"}


# ------------------------------------------------------------------------------------------------- answer -> level
def tail(answer):
    return str(answer).lower().rsplit(": ", 1)[-1].strip()


def is_pna(t):
    return t.startswith("pmi") or "prefer not" in t or t in {"skip", "dont know", "don't know", "none", "nan", ""}


def map_race(t):
    if is_pna(t): return PNA
    if "more than one" in t: return "More than one"
    if "another single" in t: return "Another single population"
    if "hispanic" in t or "latino" in t: return "Hispanic or Latino"
    if "white" in t: return "White"
    if "black" in t or "african american" in t: return "Black or African American"
    if "asian" in t: return "Asian"
    if "middle eastern" in t or "north african" in t: return "Middle Eastern or North African"
    if "hawaiian" in t or "pacific" in t: return "Native Hawaiian or other Pacific Islander"
    if "american indian" in t or "alaska" in t: return "American Indian or Alaska Native"
    if "none of these" in t: return "None of these"
    return None


def map_income(t):
    if is_pna(t): return PNA
    nums = [int(x) for x in re.findall(r"(\d+)\s*k", t)]
    if "less" in t and nums: lo = 0
    elif "more" in t and nums: lo = nums[-1]
    elif nums: lo = nums[0]
    else: return None
    if lo < 25: return "<25k"
    if lo < 50: return "25k to <50k"
    if lo < 100: return "50k to <100k"
    if lo < 150: return "100k to <150k"
    return "150k or more"


def map_education(t):
    if is_pna(t): return PNA
    if "less than" in t or "never" in t or re.search(r"\b(one|1)\b.*\b(four|4)\b", t) \
            or re.search(r"\b(five|5)\b.*\b(eight|8)\b", t) or re.search(r"\b(nine|9)\b.*\b(eleven|11)\b", t):
        return "Less than high school"
    if "twelve" in t or "12" in t or "ged" in t: return "High school or GED"
    if "college graduate" in t or "advanced" in t: return "College graduate or more"
    if "college" in t: return "Some college"
    return None


def map_employment(t):
    if is_pna(t): return PNA
    if "not currently employed" in t or "not employed" in t or "out of work" in t: return "Not employed"
    if "employed for wages" in t or "self employed" in t or "self-employed" in t: return "Employed"
    if "retired" in t: return "Retired"
    if "homemaker" in t or "student" in t or "unable" in t: return "Other"
    return None


def map_insurance(t):
    if is_pna(t): return PNA
    if t == "yes" or t.endswith(" yes"): return "Yes"
    if t == "no" or t.endswith(" no"): return "No"
    return None


MAPPERS = {"race_ethnicity": map_race, "income": map_income, "education": map_education,
           "employment": map_employment, "insurance": map_insurance}
PRIORITY = {"employment": ["Employed", "Not employed", "Retired", "Other", PNA]}


def build(rows, ids):
    """rows: person_id, qid, answer (one row per selected answer). Returns (per-person levels, answer map, unmapped)."""
    rows = rows.copy()
    rows["person_id"] = rows["person_id"].astype(str)
    rows["covariate"] = rows["qid"].map(QUESTIONS)
    rows["tail"] = rows["answer"].map(tail)
    rows["level"] = [MAPPERS[c](t) for c, t in zip(rows["covariate"], rows["tail"])]
    amap = rows.groupby(["covariate", "tail", "level"], dropna=False).size().rename("n").reset_index()
    amap = amap[amap.n >= 0].sort_values(["covariate", "n"], ascending=[True, False])
    unmapped = amap[amap.level.isna()]
    P = pd.DataFrame(index=pd.Index(ids, name=ID))
    for c in COVARIATES:
        r = rows[rows.covariate == c].dropna(subset=["level"])
        if c == "race_ethnicity":
            g = r.groupby("person_id")["level"].agg(set)

            def resolve(s):
                s = set(s)
                if "Hispanic or Latino" in s: return "Hispanic or Latino"
                races = s - {PNA, "None of these"}
                if len(races) == 1: return races.pop()
                if len(races) > 1: return "More than one"
                if "None of these" in s: return "None of these"
                return PNA
            v = g.map(resolve)
        elif c in PRIORITY:
            order = {lv: i for i, lv in enumerate(PRIORITY[c])}
            v = r.groupby("person_id")["level"].agg(lambda s: sorted(set(s), key=lambda x: order[x])[0])
        else:
            v = r.sort_values("person_id").groupby("person_id")["level"].last()
        P[c] = v.reindex(P.index).fillna(PNA)
    return P, amap, unmapped


def pool_levels(series_list, floor, label=POOLED):
    """The levels to pool so that, in every series given, every remaining level and the pool itself hold 0 or at
    least `floor` members: a level under the floor joins the pool; if the pool is under the floor, the smallest
    remaining level joins it."""
    small = set()
    while True:
        bad = False
        for s in series_list:
            p = s.where(~s.isin(small), label)
            vc = p.value_counts()
            under = [lv for lv in vc.index if 0 < vc[lv] < floor]
            if not under:
                continue
            bad = True
            if under == [label] or (label in under and len(under) == 1):
                rest = vc.drop(labels=[label], errors="ignore")
                if len(rest) == 0:
                    return sorted(small)
                small.add(rest.idxmin())
            else:
                small |= {lv for lv in under if lv != label}
        if not bad:
            return sorted(small)


def apply_pool(series, small, label=POOLED):
    return series.where(~series.isin(set(small)), label)


# ------------------------------------------------------------------------------------------------- the cohort table
def cohort_table(P, R):
    O = pd.read_csv(work("care_outcomes.csv"), usecols=[ID, "has_survey", "has_ehr"])
    O[ID] = O[ID].astype(str)
    O = O.set_index(ID).reindex(P.index)
    cf = pd.read_csv(work("thresholds.csv"), usecols=[ID, "breadth", "all_evaluable", "any_met", "is_flagged"])
    cf[ID] = cf[ID].astype(str)
    cf = cf.set_index(ID).reindex(P.index)
    NB = 6
    R = R.set_index(ID).reindex(P.index)
    pops = {"cohort": pd.Series(True, index=P.index), "survey_linked": O["has_survey"].fillna(False).astype(bool),
            "records_linked": O["has_ehr"].fillna(False).astype(bool), "flagged": cf["is_flagged"].fillna(False).astype(bool)}
    I = pd.read_csv(work("phetk", "icd_person_summary.csv"), dtype={"person_id": str}, usecols=["person_id", "n_events"])
    I = I.set_index("person_id").reindex(P.index)
    pops["icd_linked"] = pd.Series(I["n_events"].fillna(0).values > 0, index=P.index)
    S = pd.read_csv(work("sleep_person.csv"))
    S["person_id"] = S["person_id"].astype(str)
    S = S.set_index("person_id").reindex(P.index)
    pops["sleep"] = pd.Series((S["nights"].fillna(0).values >= MIN_NIGHTS) & S["sd_min"].notna().values, index=P.index)
    for enc in ENCODERS[1:]:
        Sc = pd.read_csv(work(f"scores_{enc}.csv"), usecols=[ID, "flagged"])
        Sc[ID] = Sc[ID].astype(str)
        Sc = Sc.set_index(ID).reindex(P.index)
        assert Sc["flagged"].notna().all(), f"scores_{enc}.csv does not cover the cohort"
        pops[f"flagged_{enc}"] = Sc["flagged"].astype(bool)
    n_coh = int(len(P))
    rows = []
    for pop, m in pops.items():
        n = int(m.sum())
        n_ok = not pair_hidden(n, n_coh)
        rows.append({"population": pop, "variable": "n", "level": "", "n": n if n_ok else SUPP, "percent": 100.0})
        if not n_ok:
            continue
        a = R.loc[m, "age"]
        nf = int((R.loc[m, "sex"] == 0).sum())
        sex_ok = not pair_hidden(nf, n)
        rows += [{"population": pop, "variable": "age", "level": "mean (SD)", "n": n,
                  "value": f"{a.mean():.1f} ({a.std():.1f})"},
                 {"population": pop, "variable": "age", "level": "median (IQR)", "n": n,
                  "value": f"{a.median():.0f} ({a.quantile(.25):.0f} to {a.quantile(.75):.0f})"},
                 {"population": pop, "variable": "sex", "level": "female (sex at birth)", "n": nf if sex_ok else SUPP,
                  "percent": 100 * nf / n if sex_ok else np.nan}]
        b = cf.loc[m]
        ev = b["all_evaluable"].fillna(False).astype(bool)
        n_any = int(b["any_met"].fillna(False).astype(bool).sum())
        n_ev, n_none = int(ev.sum()), int((b.loc[ev, "breadth"] == 0).sum())
        any_ok, ev_ok = not pair_hidden(n_any, n), not pair_hidden(n_ev, n)
        none_ok = ev_ok and not pair_hidden(n_none, n_ev)
        rows += [{"population": pop, "variable": "criteria", "level": f"any of {NB} met", "n": n_any if any_ok else SUPP,
                  "percent": 100 * n_any / n if any_ok else np.nan},
                 {"population": pop, "variable": "criteria", "level": f"mean met, all {NB} evaluable",
                  "n": n_ev if ev_ok else SUPP, "value": f"{b.loc[ev, 'breadth'].mean():.2f}" if ev_ok else ""},
                 {"population": pop, "variable": "criteria", "level": f"none met, all {NB} evaluable",
                  "n": n_none if none_ok else SUPP, "percent": 100 * n_none / n_ev if none_ok and n_ev else np.nan}]
    for c in COVARIATES:
        series = [P.loc[m, c] for m in pops.values()] + [P.loc[~m, c] for k, m in pops.items() if k != "cohort"]
        small = pool_levels(series, floor=TABLE_FLOOR)
        present = set(P[c].unique())
        for pop, m in pops.items():
            if pair_hidden(int(m.sum()), n_coh):
                continue
            vc = apply_pool(P.loc[m, c], small).value_counts()
            for lv in [lv for lv in LEVELS[c] if lv not in small and lv in present] + ([POOLED] if small else []):
                k = int(vc.get(lv, 0))
                rows.append({"population": pop, "variable": c, "level": lv, "n": k, "percent": 100 * k / m.sum()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- the selection table
def as_flag(s):
    if s.dtype == bool:
        return s.fillna(False)
    return pd.to_numeric(s, errors="coerce").fillna(0) > 0


def sex_level(source):
    low = source.astype(str).str.strip().str.lower()
    tok = low.str.replace(SEX_SOURCE_PREFIX_RE, "", regex=True)
    return tok.map(SEX_AT_BIRTH_ENC).fillna("Other or no answer")


def populations(person_ids):
    """included / excluded_record / rest from step 1's files; they partition the person table."""
    inc = set(pd.read_csv(COHORT_CSV, usecols=[ID])[ID].astype(str))
    master = set(pd.read_csv(os.path.join(RESPONSES, "master.csv"), usecols=[ID])[ID].astype(str))
    dr = pd.read_csv(os.path.join(RESPONSES, "dropped_demographics.csv"))
    dropped = set(dr["person_id" if "person_id" in dr.columns else ID].astype(str))
    n_full = int(json.load(open(os.path.join(RESPONSES, "pull_summary.json")))["n_full_record"])
    full = master | dropped
    assert not (master & dropped) and len(full) == n_full and inc <= master and full <= set(person_ids)
    pop = pd.Series("rest", index=pd.Index(person_ids, name=ID))
    pop[pop.index.isin(full)] = "excluded_record"
    pop[pop.index.isin(inc)] = "included"
    sizes = pop.value_counts().reindex(POPS).fillna(0).astype(int)
    assert sizes.sum() == len(set(person_ids)) and sizes["included"] == len(inc)
    return pop, {"n_release": int(len(set(person_ids))), "n_full_record": n_full,
                 **{f"n_{k}": int(v) for k, v in sizes.items()}}


def std_diff_mean(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


def std_diff_prop(p, q):
    v = (p * (1 - p) + q * (1 - q)) / 2
    return (p - q) / np.sqrt(v) if v > 0 else 0.0


def selection_table(D, pop):
    rows, pooled = [], {}
    masks = {p: (pop == p).values for p in POPS}
    inc = masks["included"]
    for p in POPS:
        m = masks[p]
        rows.append({"population": p, "variable": "n", "level": "", "n": int(m.sum()), "percent": 100.0})
        a = D.loc[m, "age"].dropna()
        rows += [{"population": p, "variable": "age_at_cutoff", "level": "mean (SD)", "n": int(len(a)),
                  "value": f"{a.mean():.1f} ({a.std():.1f})",
                  "std_diff": np.nan if p == "included" else std_diff_mean(a, D.loc[inc, "age"].dropna())},
                 {"population": p, "variable": "age_at_cutoff", "level": "median (IQR)", "n": int(len(a)),
                  "value": f"{a.median():.0f} ({a.quantile(.25):.0f} to {a.quantile(.75):.0f})"}]
    for var in ["sex"] + list(BASICS.values()):
        series = [D.loc[masks[p], var] for p in POPS] + [D.loc[~masks[p], var] for p in POPS] + [D[var]]
        small = pool_levels(series, floor=TABLE_FLOOR)
        pooled[var] = small
        order = SEX_LEVELS if var == "sex" else LEVELS[var]
        present = set(D[var].unique())
        levels = [lv for lv in order if lv not in small and lv in present] + ([POOLED] if small else [])
        col = apply_pool(D[var], small)
        inc_share = {lv: float((col[inc] == lv).mean()) for lv in levels}
        for p in POPS:
            m = masks[p]
            vc = col[m].value_counts()
            for lv in levels:
                k = int(vc.get(lv, 0))
                share = k / m.sum()
                rows.append({"population": p, "variable": var, "level": lv, "n": k, "percent": 100 * share,
                             "std_diff": np.nan if p == "included" else std_diff_prop(share, inc_share[lv])})
    for key in FLAGS:
        for p in POPS:
            m = masks[p]
            k, n = int(D.loc[m, key].sum()), int(m.sum())
            hidden = pair_hidden(k, n)
            share = k / n
            rows.append({"population": p, "variable": key, "level": "yes", "n": SUPP if hidden else k,
                         "percent": np.nan if hidden else 100 * share,
                         "std_diff": np.nan if (p == "included" or hidden or pair_hidden(int(D.loc[inc, key].sum()),
                                                                                         int(inc.sum())))
                         else std_diff_prop(share, float(D.loc[inc, key].mean()))})
    T = pd.DataFrame(rows, columns=["population", "variable", "level", "n", "percent", "value", "std_diff"])
    for c in ["percent", "std_diff"]:
        T[c] = T[c].map(lambda v: round(float(v), 3) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else v)
    return T, pooled


def main():
    from google.cloud import bigquery
    if not CDR:
        raise SystemExit("WORKSPACE_CDR is not set")
    client = bigquery.Client()
    query = lambda sql, cfg=None: client.query(sql, job_config=cfg).result().to_dataframe()

    # the covariates and the cohort table
    R = pd.read_csv(COHORT_CSV, usecols=[ID, "age", "sex"])
    R[ID] = R[ID].astype(str)
    ids = R[ID].tolist()
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("ids", "INT64", [int(i) for i in ids])])
    rows = query(f"""SELECT s.person_id, s.question_concept_id AS qid, s.answer
                     FROM `{CDR}.ds_survey` s
                     WHERE s.question_concept_id IN ({",".join(str(q) for q in QUESTIONS)})
                       AND s.person_id IN UNNEST(@ids)""", cfg)
    P, amap, unmapped = build(rows, ids)
    if len(unmapped):
        print("  answers that map to no level (counted as Prefer not to answer):\n" + unmapped.to_string(index=False))
    P.reset_index().to_csv(work("covariates.csv"), index=False)
    am = amap.copy()
    am["n"] = am["n"].astype(object)
    for _, ix in amap.groupby("covariate").groups.items():
        ix = list(ix)
        hide, prim = partition_mask(amap.loc[ix, "n"].values)
        am.loc[ix, "n"] = marked(amap.loc[ix, "n"].values, hide, prim)
    d = download_dir("cohort_table")
    am.to_csv(os.path.join(d, "cohort_answer_map.csv"), index=False)
    cohort_table(P, R).to_csv(os.path.join(d, "cohort_table.csv"), index=False)
    write_download("cohort_table", ["cohort_table.csv", "cohort_answer_map.csv"], aggregate_only)

    # the participants against the release
    head = query(f"SELECT * FROM `{CDR}.cb_search_person` LIMIT 1")
    absent = [c for c in FLAGS.values() if c not in head.columns]
    if absent:
        raise SystemExit(f"cb_search_person has no column {absent}")
    Pp = query(f"""SELECT person_id, birth_datetime, year_of_birth,
                          CAST(sex_at_birth_source_value AS STRING) AS sex_source FROM `{CDR}.person`""")
    Pp["person_id"] = Pp["person_id"].astype(str)
    Pp = Pp.drop_duplicates("person_id").set_index("person_id")
    pids = Pp.index.tolist()
    pop, sizes = populations(pids)
    F = query(f"SELECT person_id, {', '.join(FLAGS.values())} FROM `{CDR}.cb_search_person`")
    F["person_id"] = F["person_id"].astype(str)
    F = F.drop_duplicates("person_id").set_index("person_id")
    if int((~pop.index.isin(F.index)).sum()):
        raise SystemExit("cb_search_person lacks a row for some persons of the person table")
    rows = query(f"""SELECT person_id, question_concept_id AS qid, answer FROM `{CDR}.ds_survey`
                     WHERE question_concept_id IN ({",".join(str(q) for q in BASICS)})""")
    rows = rows[rows["person_id"].astype(str).isin(set(pids))]
    B, amap, unmapped = build(rows, pids)
    birth = pd.to_datetime(Pp["birth_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    yob = pd.to_datetime(Pp["year_of_birth"].astype("Int64").astype(str).where(Pp["year_of_birth"].notna()) + "-07-01",
                         errors="coerce")
    D = pd.DataFrame(index=pop.index)
    D["age"] = ((CUTOFF - birth.fillna(yob).astype("datetime64[ns]")).dt.days / 365.25).round(1).reindex(D.index)
    D["sex"] = sex_level(Pp["sex_source"]).reindex(D.index)
    for c in BASICS.values():
        D[c] = B[c].reindex(D.index).fillna(PNA)
    for k, c in FLAGS.items():
        D[k] = as_flag(F[c]).reindex(D.index).fillna(False).astype(bool)
    T, pooled = selection_table(D, pop)
    d = download_dir("selection")
    T.to_csv(os.path.join(d, "selection_table.csv"), index=False)
    json.dump({"cdr": CDR, "cutoff_date": str(CUTOFF.date()), **sizes,
               "populations": {"included": "the cohort", "excluded_record": "a record for all 151 items, not in the "
                               "cohort (sex at birth not female or male, age missing, or more than 5 items missing)",
                               "rest": "the rest of the person table"},
               "std_diff": "against included; age: mean difference / sqrt((var1 + var2) / 2); a level or flag: "
                           "(p1 - p2) / sqrt((p1(1 - p1) + p2(1 - p2)) / 2)",
               "pooled_levels": pooled, "pool_floor": TABLE_FLOOR,
               "unmapped_answer_strings": int(unmapped[["covariate", "tail"]].drop_duplicates().shape[0]) if len(unmapped) else 0,
               "versions": versions()},
              open(os.path.join(d, "selection_meta.json"), "w"), indent=2, default=str)
    write_download("selection", ["selection_table.csv", "selection_meta.json"], aggregate_only)


if __name__ == "__main__":
    main()
