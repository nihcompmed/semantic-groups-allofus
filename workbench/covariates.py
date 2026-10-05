#!/usr/bin/env python3
"""covariates.py -- the five sociodemographic covariates from The Basics (seven with age and sex), and the Table 1
aggregates (eTables 2 and 3).

STANDALONE on the Workbench (pandas, google-cloud-bigquery). Runs after care.py --outcomes-only (it reads the outcome
cache for the survey- and records-linked subsets) and cutoffs.py (breadth and the flagged set).

THE COVARIATES, self-reported in The Basics, coded as the All of Us papers in JAMA journals code them
(PMC11780479), with every non-answer kept as a level and nothing imputed:
    race_ethnicity   Hispanic or Latino (any Hispanic answer), else the single race selected, else "More than one",
                     "None of these", or "Prefer not to answer" (PMI skip / prefer not / no row)
    income           <25k / 25k to <50k / 50k to <100k / 100k to <150k / 150k or more / Prefer not to answer
    education        Less than high school / High school or GED / Some college / College graduate or more / Prefer not to answer
    employment       Employed (for wages or self-employed) / Not employed / Retired / Other (homemaker, student,
                     unable to work) / Prefer not to answer   (multi-select resolved in that priority)
    insurance        Yes / No / Prefer not to answer
Age (at the reference date) and sex at birth come from the response build (build_responses_v9.py) and are not
pulled again.

Answers are mapped from the text of `ds_survey.answer` after its last ": " (the same field care.py reads), so the
mapping does not depend on answer concept ids. Every distinct answer (with at least MIN_COUNT respondents, so at
MIN_COUNT = 0 every one) is printed with its mapped level, and any answer that maps to nothing is printed loudly,
so the mapping can be checked on the Workbench before anything downstream runs.

WRITES (-> screen_out/)
    covariates_per_person.csv   id, the five covariates                                    (per person, STAYS)
    table1_covariates.csv       level counts and percents for the cohort, the survey-linked subset, the
                                records-linked subset and the prespecified encoder's flagged set, with age, sex
                                and criteria summaries                                          (aggregate)
    covariates_answer_map.csv   the distinct raw answers and the level each maps to                (aggregate)

★ THE DISSEMINATION POLICY (disclosure.py). Table 1 COLLAPSES rather than
blanks, which the policy names as a remedy: a level holding 1 to 20 people in ANY population, or in the
COMPLEMENT of any subpopulation within the cohort (the cohort minus the flagged set is a group a reader
can compute), joins "Other (pooled)" in every population, and the pool grows until it too is 0 or over
20. TABLE1_FLOOR = 21 does this. MIN_COUNT is 0, because care_covariates.py imports pool_levels for
its MODEL design and the models are not changed by this. The answer map is written with counts of 1 to
20 as "<=20" and complementary cells as "suppressed", per covariate.

★ THE FLAGGED SET IS THE PRESPECIFIED ENCODER'S OWN. It is read from cutoff_flags.csv's
`is_flagged`, which cutoffs.py takes from scores_<encoder>.csv.

★ eTABLES 2 AND 3. The table of participant characteristics is called Table 1 here and in the file names; in the
paper it is eTables 2 and 3. eTable 3 shows the subset each outcome is measured on, so two populations are
added to the four above:
    survey_linked    answered the Health Care Access and Utilization survey (the barriers)       care.py
    records_linked   at least 1 visit in the linked records (the ED share)                          care.py
    icd_linked       at least 1 ICD event (the 8 recorded conditions)                   phecode_coverage.py
    sleep            at least MIN_NIGHTS nights with a usable SD (sleep_sweep.py and sleep_adjusted.py's
                     own rule, imported from sleep_reliability.py)
The last two are read from the person-level files those steps leave on the Workbench
(screen_out/phetk/icd_person_summary_local.csv, sleep_person_local.csv). A missing file skips that population
with a printed note and changes nothing else. The pooling rule below also covers them and their complements.
eTable 2 shows the OTHER 5 ENCODERS' own 95th-percentile sets (`flagged_<encoder>`, from each
scores_<encoder>.csv `flagged` column, the rule cutoffs.py uses for gte), so their characteristics can be read
side by side with gte's (no test, no overlap statistic).
The sex row counts sex == 0 (female, as build_responses_v9.py codes it).

WRITES ALSO screen_out/table1_<CDR>.zip: table1_covariates.csv and covariates_answer_map.csv, each
checked by verdict() of zip_screen_aggregates.py first.

Run:  python3 covariates.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_config import CFG   # noqa: E402  battery and responses folder, resolved once, no defaults
from disclosure import MAX_SUPPRESSED, marked, one, pair_hidden, partition_mask, SUPP   # noqa: E402  the policy
OUT_DIR       = os.path.join(HERE, "screen_out")
RESPONSES_CSV = os.path.join(CFG.responses_dir, "responses_k5.csv")
CDR           = os.environ.get("WORKSPACE_CDR", "")
ID_COL        = "id"
MIN_COUNT     = 0      # pooling floor for the MODEL design care_covariates.py imports
OTHER_ENCODERS = ["bge-m3", "bge-large-en-v1.5", "all-mpnet-base-v2", "e5-large-v2",   # eTable 2's
                  "qwen3-embedding-0.6b"]                                             # other 5 flagged sets
TABLE1_FLOOR  = MAX_SUPPRESSED + 1   # Table 1 pools any level of 1 to 20 (dissemination policy)
PNA           = "Prefer not to answer"
QUESTIONS = {1586140: "race_ethnicity", 1585375: "income", 1585940: "education", 1585952: "employment", 1585386: "insurance"}
COVARIATES = ["race_ethnicity", "income", "education", "employment", "insurance"]
LEVELS = {  # display order
    "race_ethnicity": ["White", "Black or African American", "Hispanic or Latino", "Asian", "More than one", "Another single population",
                       "Middle Eastern or North African", "Native Hawaiian or other Pacific Islander", "American Indian or Alaska Native",
                       "None of these", PNA],
    "income": ["<25k", "25k to <50k", "50k to <100k", "100k to <150k", "150k or more", PNA],
    "education": ["Less than high school", "High school or GED", "Some college", "College graduate or more", PNA],
    "employment": ["Employed", "Not employed", "Retired", "Other", PNA],
    "insurance": ["Yes", "No", PNA]}


# ----------------------------------------------------------------------------- answer text -> level
def tail(answer):
    return str(answer).lower().rsplit(": ", 1)[-1].strip()


def is_pna(t):
    return t.startswith("pmi") or "prefer not" in t or t in {"skip", "dont know", "don't know", "none", "nan", ""}


def map_race(t):
    # CDR v9 Registered Tier strings: white, hispanic, black, asian, "more than one population",
    # "another single population" (the tier's generalization of the smaller groups), "race ethnicity none of these"
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
    # CDR v9 strings: "less than a high school degree or equivalent", "twelve or ged", "college one to three",
    # "college graduate or advanced degree"
    if is_pna(t): return PNA
    if "less than" in t or "never" in t or re.search(r"\b(one|1)\b.*\b(four|4)\b", t) or re.search(r"\b(five|5)\b.*\b(eight|8)\b", t) \
            or re.search(r"\b(nine|9)\b.*\b(eleven|11)\b", t): return "Less than high school"
    if "twelve" in t or "12" in t or "ged" in t: return "High school or GED"
    if "college graduate" in t or "advanced" in t: return "College graduate or more"
    if "college" in t: return "Some college"
    return None


def map_employment(t):
    # CDR v9 strings: "employed for wages or self-employed", "not currently employed for wages". The negative
    # form contains the positive one, so it is tested first.
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
    """rows: DataFrame person_id, qid, answer (one row per selected answer). Returns (per-person frame, answer map)."""
    rows = rows.copy()
    rows["person_id"] = rows["person_id"].astype(str)
    rows["covariate"] = rows["qid"].map(QUESTIONS)
    rows["tail"] = rows["answer"].map(tail)
    rows["level"] = [MAPPERS[c](t) for c, t in zip(rows["covariate"], rows["tail"])]
    amap = (rows.groupby(["covariate", "tail", "level"], dropna=False).size().rename("n").reset_index())
    amap = amap[amap.n >= MIN_COUNT].sort_values(["covariate", "n"], ascending=[True, False])
    unmapped = amap[amap.level.isna()]
    P = pd.DataFrame(index=pd.Index(ids, name=ID_COL))
    for c in COVARIATES:
        r = rows[rows.covariate == c].dropna(subset=["level"])
        if c == "race_ethnicity":
            # multi-select: Hispanic first, then a single race, then "More than one"
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


POOLED = "Other (pooled)"


def pool_levels(series_list, floor=MIN_COUNT, label=POOLED):
    """The set of levels to pool so that, in every series given, every remaining level and the pooled level itself
    have either zero or at least `floor` members. Iterates: a level under the floor joins the pool, and if the pool
    is still under the floor the smallest remaining level joins it. Returns the sorted set of pooled levels.
    At floor = 0 (MIN_COUNT = 0) no level is under the floor, so the empty set is returned and every level stays."""
    small = set()
    while True:
        bad = False
        for s in series_list:
            p = s.where(~s.isin(small), label)
            vc = p.value_counts()
            under = [l for l in vc.index if 0 < vc[l] < floor]
            if not under:
                continue
            bad = True
            if under == [label] or (label in under and len(under) == 1):
                rest = vc.drop(labels=[label], errors="ignore")
                if len(rest) == 0:
                    return sorted(small)
                small.add(rest.idxmin())
            else:
                small |= {l for l in under if l != label}
        if not bad:
            return sorted(small)


def apply_pool(series, small, label=POOLED):
    return series.where(~series.isin(set(small)), label)


# ----------------------------------------------------------------------------- main
def main():
    from google.cloud import bigquery
    if not CDR:
        raise SystemExit("no CDR dataset: set WORKSPACE_CDR")
    R = pd.read_csv(RESPONSES_CSV, usecols=[ID_COL, "age", "sex"])
    R[ID_COL] = R[ID_COL].astype(str)
    ids = R[ID_COL].tolist()
    client = bigquery.Client()
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("ids", "INT64", [int(i) for i in ids])])
    rows = client.query(f"""SELECT s.person_id, s.question_concept_id AS qid, s.answer
                            FROM `{CDR}.ds_survey` s
                            WHERE s.question_concept_id IN ({",".join(str(q) for q in QUESTIONS)})
                              AND s.person_id IN UNNEST(@ids)""", job_config=cfg).result().to_dataframe()
    print(f"cohort {len(ids):,} | Basics rows {len(rows):,} for {rows.person_id.nunique():,} people | CDR {CDR}")
    P, amap, unmapped = build(rows, ids)
    print(f"\ndistinct answers (n >= {MIN_COUNT}) and their level:")
    for r in amap.itertuples():
        print(f"  {r.covariate:15s} {r.n:7,d}  {str(r.tail)[:60]:60s} -> {r.level}")
    if len(unmapped):
        print(f"\n!! UNMAPPED answers with n >= {MIN_COUNT} (treated as no answer -> Prefer not to answer). Fix the mapper before running care_covariates.py:")
        print(unmapped.to_string(index=False))
    os.makedirs(OUT_DIR, exist_ok=True)
    P.reset_index().to_csv(os.path.join(OUT_DIR, "covariates_per_person.csv"), index=False)
    am = amap.copy()                                     # the policy, per covariate: its answers partition its rows
    am["n"] = am["n"].astype(object)
    for _, ix in amap.groupby("covariate").groups.items():
        ix = list(ix)
        hide, prim = partition_mask(amap.loc[ix, "n"].values)
        am.loc[ix, "n"] = marked(amap.loc[ix, "n"].values, hide, prim)
    am.to_csv(os.path.join(OUT_DIR, "covariates_answer_map.csv"), index=False)

    # ---- Table 1 aggregates
    # The per-person covariate file above is the thing most downstream scripts need, and it is already
    # written. Table 1 needs three more files that come from other steps: care.py --outcomes-only for
    # the linked subsets, and cutoffs.py for breadth and the flagged set (cutoff_flags.csv and
    # cutoff_summary.json). A missing one skips Table 1 with a note instead of raising, because the
    # useful output is already written and a raise would read as a failed run when it is not.
    need = {"care_outcomes_per_person.csv": "care.py --outcomes-only",
            "cutoff_flags.csv": "cutoffs.py",
            "cutoff_summary.json": "cutoffs.py"}
    absent = {f: who for f, who in need.items() if not os.path.exists(os.path.join(OUT_DIR, f))}
    if absent:
        print("\ncovariates_per_person.csv and covariates_answer_map.csv are written.")
        print("Table 1 is SKIPPED, because it needs files that later steps produce:")
        for f, who in absent.items():
            print(f"  {f:34s} from {who}")
        print("Everything that needs only the covariates can run now. Rerun this script after those")
        print("steps to get Table 1.")
        return
    O = pd.read_csv(os.path.join(OUT_DIR, "care_outcomes_per_person.csv"), usecols=[ID_COL, "has_survey", "has_ehr"])
    O[ID_COL] = O[ID_COL].astype(str); O = O.set_index(ID_COL).reindex(P.index)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL, "breadth", "all_evaluable", "any_met", "is_flagged"])
    cf[ID_COL] = cf[ID_COL].astype(str); cf = cf.set_index(ID_COL).reindex(P.index)
    import json
    js = json.load(open(os.path.join(OUT_DIR, "cutoff_summary.json")))
    NB, ENC = int(js["n_criteria"]), js.get("encoder", "?")
    R = R.set_index(ID_COL).reindex(P.index)
    pops = {"cohort": pd.Series(True, index=P.index), "survey_linked": O["has_survey"].fillna(False).astype(bool),
            "records_linked": O["has_ehr"].fillna(False).astype(bool), "flagged": cf["is_flagged"].fillna(False).astype(bool)}
    print(f"flagged set: {ENC}'s own, {int(pops['flagged'].sum()):,} participants (cutoff_flags.csv is_flagged)")
    # 2 more outcome subsets of eTable 3; each is skipped, with a note, if its person-level file is absent
    icd_file = os.path.join(OUT_DIR, "phetk", "icd_person_summary_local.csv")
    if os.path.exists(icd_file):
        I = pd.read_csv(icd_file, dtype={"person_id": str}, usecols=["person_id", "n_events"])
        I = I.set_index("person_id").reindex(P.index)
        pops["icd_linked"] = pd.Series(I["n_events"].fillna(0).values > 0, index=P.index)
    else:
        print(f"icd_linked SKIPPED: {icd_file} not found (phecode_coverage.py, in ~/phetk_env, writes it)")
    try:
        from sleep_reliability import MIN_NIGHTS, sleep_cache   # one rule, one file (sleep_sweep.py, sleep_adjusted.py)
        S = pd.read_csv(sleep_cache())
        S["person_id"] = S["person_id"].astype(str)
        S = S.set_index("person_id").reindex(P.index)
        pops["sleep"] = pd.Series((S["nights"].fillna(0).values >= MIN_NIGHTS) & S["sd_min"].notna().values,
                                  index=P.index)
    except SystemExit as e:
        print(f"sleep SKIPPED: {e}")
    # the other 5 encoders' own 95th-percentile sets for eTable 2 (the characteristics side by side, no test),
    # each from its own scores file's `flagged` column, the rule cutoffs.py applies to gte's. A missing scores
    # file stops the run: eTable 2 needs all 5.
    for enc in OTHER_ENCODERS:
        sf = os.path.join(OUT_DIR, f"scores_{enc}.csv")
        assert os.path.exists(sf), f"{sf} is missing; run screen_battery.py for {enc} first"
        Sc = pd.read_csv(sf, usecols=[ID_COL, "flagged"])
        Sc[ID_COL] = Sc[ID_COL].astype(str)
        Sc = Sc.set_index(ID_COL).reindex(P.index)
        assert Sc["flagged"].notna().all(), f"scores_{enc}.csv does not cover the cohort"
        pops[f"flagged_{enc}"] = Sc["flagged"].astype(bool)
    for k, m in pops.items():
        print(f"  population {k:32s} {int(m.sum()):,}")
    n_coh = int(len(P))
    rows_out = []
    for pop, m in pops.items():
        n = int(m.sum())
        # a subpopulation's size and its complement in the cohort are both computable from Table 1
        n_ok = not pair_hidden(n, n_coh)
        rows_out.append({"population": pop, "variable": "n", "level": "", "n": n if n_ok else SUPP, "percent": 100.0})
        if not n_ok:
            continue
        a = R.loc[m, "age"]
        nf = int((R.loc[m, "sex"] == 0).sum())
        sex_ok = not pair_hidden(nf, n)
        rows_out += [{"population": pop, "variable": "age", "level": "mean (SD)", "n": n, "value": f"{a.mean():.1f} ({a.std():.1f})"},
                     {"population": pop, "variable": "age", "level": "median (IQR)", "n": n, "value": f"{a.median():.0f} ({a.quantile(.25):.0f} to {a.quantile(.75):.0f})"},
                     # sex at birth is coded male = 1, female = 0 by build_responses_v9.py (SEX_AT_BIRTH_ENC)
                     {"population": pop, "variable": "sex", "level": "female (sex at birth)", "n": nf if sex_ok else SUPP,
                      "percent": 100 * nf / n if sex_ok else np.nan}]
        b = cf.loc[m]
        ev = b["all_evaluable"].fillna(False).astype(bool)
        n_any = int(b["any_met"].fillna(False).astype(bool).sum())
        n_ev, n_none = int(ev.sum()), int((b.loc[ev, "breadth"] == 0).sum())
        any_ok, ev_ok = not pair_hidden(n_any, n), not pair_hidden(n_ev, n)
        none_ok = ev_ok and not pair_hidden(n_none, n_ev)
        rows_out += [{"population": pop, "variable": "criteria", "level": f"any of {NB} met", "n": n_any if any_ok else SUPP,
                      "percent": 100 * n_any / n if any_ok else np.nan},
                     {"population": pop, "variable": "criteria", "level": f"mean met, all {NB} evaluable", "n": n_ev if ev_ok else SUPP,
                      "value": f"{b.loc[ev, 'breadth'].mean():.2f}" if ev_ok else ""},
                     {"population": pop, "variable": "criteria", "level": f"none met, all {NB} evaluable", "n": n_none if none_ok else SUPP,
                      "percent": 100 * n_none / n_ev if none_ok and n_ev else np.nan}]
    # pooling rule (the dissemination policy): the same levels are pooled in every population, and in the
    # complement of every subpopulation, so that no count a reader can compute is 1 to 20
    pooled_note = {}
    for c in COVARIATES:
        series = [P.loc[m, c] for m in pops.values()] + [P.loc[~m, c] for k, m in pops.items() if k != "cohort"]
        small = pool_levels(series, floor=TABLE1_FLOOR)
        pooled_note[c] = small
        present = set(P[c].unique())                      # levels the release does not use are not listed
        for pop, m in pops.items():
            if pair_hidden(int(m.sum()), n_coh):          # its size is suppressed, so are its levels
                continue
            vc = apply_pool(P.loc[m, c], small).value_counts()
            for lv in [l for l in LEVELS[c] if l not in small and l in present] + ([POOLED] if small else []):
                k = int(vc.get(lv, 0))
                rows_out.append({"population": pop, "variable": c, "level": lv, "n": k, "percent": 100 * k / m.sum()})
    T = pd.DataFrame(rows_out)
    T.to_csv(os.path.join(OUT_DIR, "table1_covariates.csv"), index=False)
    print(f"\nTable 1 aggregates written; levels pooled (1 to 20 in some population or complement):")
    for c, s in pooled_note.items():
        print(f"  {c}: {', '.join(s) if s else 'none'}")
    for c in COVARIATES:
        sub = T[(T.variable == c) & (T.population == "flagged")]
        print(f"  flagged, {c}: " + ", ".join(f"{r.level} {r.n} ({r.percent:.0f}%)" for r in sub.itertuples()))
    print(f"\nwrote -> {OUT_DIR}: covariates_per_person.csv (stays), table1_covariates.csv, covariates_answer_map.csv")
    # the download: the two aggregates, each checked by verdict() of zip_screen_aggregates.py first
    import zipfile
    from zip_screen_aggregates import verdict
    files = ["table1_covariates.csv", "covariates_answer_map.csv"]
    for f in files:
        ok, why = verdict(os.path.join(OUT_DIR, f))
        assert ok, f"{f} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"table1_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(OUT_DIR, f), arcname=f"table1/{f}")
    print(f"  download -> {zpath}")


if __name__ == "__main__":
    main()
