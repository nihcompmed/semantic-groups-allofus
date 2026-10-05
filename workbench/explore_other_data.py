#!/usr/bin/env python3
"""explore_other_data.py -- what else the Curated Data Repository holds for OUR 90,351, and for how many
of them.

WHY. Every outcome in the paper except the emergency department share is something a participant said
about themselves, on the same survey platform as the 151 items. Anything recorded rather than reported
is a different kind of evidence. Before proposing one, we need to know what exists and, far more
important, HOW MANY OF OUR COHORT IT COVERS. A table with a billion rows is useless if it covers 2% of
the 90,351, and a small table is valuable if it covers most of them.

COST. The Repository is BigQuery and the workspace pays per byte scanned. Cells 1 to 3 get
progressively more expensive, and the script prints the bytes each query will scan BEFORE running it.
Cells 4 and 9 to 13 are described in their own headers below and in the Run lines.
  Cell 1  metadata only, free: every table, its row count, and whether it has a person_id column.
  Cell 2  cheap: for the tables you choose, who of OUR cohort appears, split three ways -- overall, by
          decile of burden, and the most burdened against the rest. The split is the point. An overall
          figure can look adequate while the top of the burden distribution is empty, and selection
          into a wearable or an online task battery runs opposite to burden.
  Cell 3  targeted: for one table, the most common concepts among our cohort.
  Cell 6  the cognitive-task and Fitbit tables: what columns they hold and under what person key, the
          task measures profiled among our cohort, Fitbit days per person relative to the answer date
          (presence is not density), and whether the Fitbit coverage gradient survives age. No design
          is proposed; these are the three things that have to be known before one can be.
  Cell 8  the last coverage question, and the reason it is the last one: the wear study hands devices
          out and cell 7 showed that is the whole burden gradient, so the shape of a loaned record is
          compared with the shape of an owned one. DATES ONLY, no behavioral value, because burden
          against a recorded measure is the analysis and it is not run as reconnaissance.
  Cell 7  the two things cell 6 leaves open: the parsed task measures in the ds_*_outcomes tables, which
          cell 6 lists but does not profile (they spell the person key PERSON_ID), and whether the
          Fitbit coverage gradient is the program distributing devices
          through the wear study, which was the second candidate explanation after age.
  Cell 5  when each person finished the 151 items, how long their record runs after
          that, which of the four outcome candidates they are defined for, and the visit concepts this
          CDR actually carries. Decides whether the window design exists, against a rule declared
          before the run.
Run cell 1, read it, then edit TABLES in cell 2 to the handful worth costing.

★ THE STEPS OF THE PAPER RUN CELLS 9 AND 10 ONLY, for the person-level caches that later steps read
(sleep_person_local.csv, wear_enrollees_local.csv, answer_dates_local.csv). The other cells are
exploration. Cells 2 and 5 take "the most burdened" from consensus_outliers.csv (the participants
flagged by all 6 encoders), and cell 13 reads leading_group_per_person_*.csv, which no script in this
folder writes, so those cells are reconnaissance and not part of the analysis.

OUTPUT. The files written through _agg_csv (cells 7 to 12) are aggregates under the dissemination
policy, and they are the ones zip_screen_aggregates.py zips. The other cells write plain aggregate
CSVs for reading on the Workbench, which are not zipped. The per-person caches (*_local.csv) are
person level, stay on the Workbench, and are never zipped.

Run:  python3 explore_other_data.py                                    (cell 1 only, by default)
      python3 explore_other_data.py --cells 1,2
      python3 explore_other_data.py --responses-dir <folder>           (to profile a build elsewhere)
      python3 explore_other_data.py --cells 3 --table condition_occurrence --top 40
      python3 explore_other_data.py --cells 4       (measurement in detail: what is measured, and on whom)
      python3 explore_other_data.py --cells 5       (answer date, follow-up, visit windows)
      python3 explore_other_data.py --cells 6       (what the task and Fitbit tables hold, and who they reach)
      python3 explore_other_data.py --cells 7       (the parsed task measures; is the Fitbit gradient the devices)
      python3 explore_other_data.py --cells 8       (loaned device against owned: record shape, dates only)
      python3 explore_other_data.py --cells 9       (night-to-night SD of sleep: how many nights a mean needs)
      python3 explore_other_data.py --cells 10      (the same association where the nights were not chosen)
      python3 explore_other_data.py --cells 11      (THE sleep design: mean duration against burden)
      python3 explore_other_data.py --cells 12      (are duration and variability one finding or two)
      python3 explore_other_data.py --cells 13      (does the task design exist on the chosen population;
                                                     reads leading_group_per_person_*.csv)
      --encoder <name> reads that encoder's percentile instead of gte-large-en-v1.5's, and suffixes
      every _agg_csv file with _<name> (the supplement)
"""
import os
import sys

import numpy as np
import pandas as pd
from google.cloud import bigquery

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_config import CFG, ENCODERS   # noqa: E402  battery, folders and the 6 encoders, resolved once
CDR = os.environ.get("WORKSPACE_CDR", "")
RESP_DIR = CFG.responses_dir                              # --responses-dir still wins, wb_config reads it
COHORT_CSV = os.path.join(RESP_DIR, "responses_k5.csv")   # the cohort of record for that build
ID_COL = "id"
DRY_RUN_LIMIT_GB = 50                    # refuse any query that would scan more than this
CELLS = [c.strip() for c in (sys.argv[sys.argv.index("--cells") + 1] if "--cells" in sys.argv else "1").split(",")]
TABLE = sys.argv[sys.argv.index("--table") + 1] if "--table" in sys.argv else "condition_occurrence"
TOP = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 30
ENCODER = sys.argv[sys.argv.index("--encoder") + 1] if "--encoder" in sys.argv else "gte-large-en-v1.5"
assert ENCODER in ENCODERS, f"--encoder must be one of {ENCODERS}"
SUF = "" if ENCODER == "gte-large-en-v1.5" else f"_{ENCODER}"


def _agg_csv(df, name, index=False):
    """Write an aggregate that leaves the Workbench under the dissemination policy (disclosure.py).
    Count columns are `n`, `people` and `n_*`. A row whose count is 1 to 20 has it
    written "<=20" and every other number in the row blanked, since those numbers rest on its people.
    Where any row of a multi-row table is hidden, the column is treated as a partition and
    complementary cells are hidden too. Every file gets the encoder suffix for a non-default encoder."""
    from disclosure import MARK, SUPP, partition_mask
    d = df.reset_index() if index else df.copy()
    cnt = [c for c in d.columns if str(c) in ("n", "people") or str(c).startswith("n_")]
    stat = [c for c in d.columns if c not in cnt and pd.api.types.is_numeric_dtype(d[c])
            and str(c) not in ("decile", "band", "model", "kind")]
    for c in cnt:
        v = pd.to_numeric(d[c], errors="coerce")
        if not ((v > 0) & (v <= 20)).any():
            continue
        hide, prim = (partition_mask(v.fillna(0).values) if len(d) > 1
                      else (((v > 0) & (v <= 20)).values, ((v > 0) & (v <= 20)).values))
        d[c] = d[c].astype(object)
        d.loc[hide, c] = [MARK if p else SUPP for p in prim[hide]]
        d.loc[prim, stat] = np.nan
    root, ext = os.path.splitext(name)
    d.to_csv(f"{root}{SUF}{ext}", index=False)

# Cell 2 looks at these unless you edit the list. Chosen as the tables most likely to hold something
# that is recorded rather than reported. Add or remove after reading cell 1.
TABLES = [
    # the records, for the window design of cell 5
    "visit_occurrence", "observation_period", "condition_occurrence", "procedure_occurrence",
    "drug_exposure", "measurement", "death",
    # the cognitive task battery. The bare tables are the person-keyed ones; the ds_*_outcomes
    # siblings hold the parsed measures but did not report a person_id column in cell 1.
    "flanker", "gradcpt", "emorecog", "delaydiscounting",
    # Fitbit. The daily summaries, never the intraday tables: heart_rate_intraday is 6.8 TB and
    # steps_intraday 6.1 TB, and a coverage count needs neither.
    "ds_fitbit_device", "activity_summary", "sleep_daily_summary", "heart_rate_summary",
    "wear_study",
]

client = bigquery.Client()


def cost_of(sql):
    """Bytes this query would scan, without running it."""
    cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    return client.query(sql, job_config=cfg).total_bytes_processed


def Q(sql, label="", ids=None, extra=None):
    """ids, when given, are sent as the array parameter @ids (no temporary table, no write access needed).
    extra is a list of further query parameters, for the cells that anchor on a per-person date."""
    params = [bigquery.ArrayQueryParameter("ids", "INT64", [int(x) for x in ids])] if ids is not None else []
    params += list(extra or [])
    cfg = bigquery.QueryJobConfig(query_parameters=params, dry_run=True, use_query_cache=False)
    gb = client.query(sql, job_config=cfg).total_bytes_processed / 1e9
    if gb > DRY_RUN_LIMIT_GB:
        print(f"  [skipped {label}: would scan {gb:,.1f} GB, over the {DRY_RUN_LIMIT_GB} GB limit]")
        return None
    if gb > 1:
        print(f"  [{label}: scanning {gb:,.1f} GB]")
    return client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()


# ---- cell 1: what tables exist, how big, and do they key on a person -------------------------------------
def cell1():
    print(f"CDR = {CDR}\n")
    t = Q(f"""SELECT table_id AS table_name, row_count, ROUND(size_bytes/1e9, 2) AS size_gb
              FROM `{CDR}.__TABLES__` ORDER BY row_count DESC""", "table list")
    cols = Q(f"""SELECT table_name, COUNT(*) AS n_columns,
                        MAX(CASE WHEN column_name = 'person_id' THEN 1 ELSE 0 END) AS has_person_id
                 FROM `{CDR}.INFORMATION_SCHEMA.COLUMNS` GROUP BY table_name""", "column metadata")
    m = t.merge(cols, on="table_name", how="left")
    m["has_person_id"] = m["has_person_id"].fillna(0).astype(int)
    pd.set_option("display.max_rows", 300, "display.width", 200)
    print("every table in the CDR, largest first (row_count and size are metadata, free to read):\n")
    print(m.to_string(index=False))
    print("\nTables keyed on a person are the ones that can say anything about our cohort.")
    print("Pick from these for cell 2, then run:  python3 explore_other_data.py --cells 2")
    m.to_csv("other_data_tables.csv", index=False)
    print("-> other_data_tables.csv")


# ---- cell 2: for OUR cohort, how many are covered and how densely ----------------------------------------
def burden_deciles(ids):
    """Each cohort member's burden percentile under ONE ENCODER, cut into deciles.

    ★ The percentile is scores_<ENCODER>.csv's own `pctile`, the prespecified gte-large-en-v1.5 unless
    --encoder names another, which is the same exposure `care_covariates.py --gradient` uses, so the
    sleep gradient and the care gradient are read on one axis. The per-person caches (*_local.csv) hold
    no exposure, since the decile is attached here at run time, so caches from an earlier run are safe
    to reuse. Returns a Series indexed by id with values 1 (least burdened) to 10, or None when the
    score file is not beside us.
    """
    f = os.path.join(CFG.out_dir, f"scores_{ENCODER}.csv")
    if not os.path.exists(f):
        return None
    M = pd.read_csv(f, usecols=[ID_COL, "pctile"]).set_index(ID_COL)["pctile"]
    M.index = M.index.astype("int64")
    M = M.reindex(ids).dropna()
    # rank first, so the deciles are equal-sized whatever ties the percentile scale carries
    return (M.rank(method="first").sub(1) * 10 // len(M) + 1).astype(int)


def cell2():
    ids = pd.read_csv(COHORT_CSV, usecols=[ID_COL])[ID_COL].astype("int64")
    n = len(ids)
    dec = burden_deciles(ids)
    cpath = os.path.join(CFG.out_dir, "consensus_outliers.csv")
    out = set()
    if os.path.exists(cpath):
        C = pd.read_csv(cpath)
        out = set(C.loc[C["flagged_by_all"].astype(bool), ID_COL].astype("int64"))
    print(f"cohort of record: {n:,} participants | most burdened: {len(out):,} | "
          f"burden deciles: {'yes' if dec is not None else 'NO (scores_<enc>.csv not found)'}\n")
    print("Coverage is asked of the WHOLE COHORT and then split, because the question is whether a table")
    print("reaches people at every level of burden. An overall figure can look adequate while the top of")
    print("the distribution is empty, and selection into a wearable or an online task battery runs")
    print("opposite to burden.\n")

    summary, gradient = [], []
    for t in TABLES:
        # DISTINCT ids rather than a count, so one scan answers overall, by decile, and outliers vs
        # rest. The ids stay on the Workbench; only counts are written.
        sql = f"""SELECT DISTINCT person_id FROM `{CDR}.{t}` WHERE person_id IN UNNEST(@ids)"""
        d = Q(sql, t, ids)
        if d is None:
            continue
        cov = set(d.person_id.astype("int64")) if len(d) else set()
        n_cov = len(cov)
        row = {"table": t, "n_people": n_cov, "coverage_pct": round(100 * n_cov / n, 1)}
        line = f"  {t:24s} {n_cov:7,} of {n:,} ({100 * n_cov / n:5.1f}%)"
        if out:
            nb = len(cov & out)
            rest = n - len(out)
            nr = n_cov - nb
            row["n_burdened"] = nb
            row["burdened_pct"] = round(100 * nb / len(out), 1)
            row["rest_pct"] = round(100 * nr / rest, 1) if rest else None
            row["burdened_vs_rest"] = (round((nb / len(out)) / (nr / rest), 2)
                                       if rest and nr else None)
            line += (f" | burdened {100 * nb / len(out):5.1f}%  rest {100 * nr / rest:5.1f}%"
                     f"  ratio {row['burdened_vs_rest']}")
        if dec is not None:
            per = dec.groupby(dec).size()
            hit = dec[dec.index.isin(cov)].groupby(lambda i: dec[i]).size().reindex(per.index).fillna(0)
            pct = (100 * hit / per).round(1)
            for dd in per.index:
                gradient.append({"table": t, "decile": int(dd), "n_in_decile": int(per[dd]),
                                 "n_covered": int(hit[dd]), "coverage_pct": float(pct[dd])})
            row["decile1_pct"], row["decile10_pct"] = float(pct.iloc[0]), float(pct.iloc[-1])
            line += f" | d1 {pct.iloc[0]:5.1f}%  d10 {pct.iloc[-1]:5.1f}%"
        summary.append(row)
        print(line)

    R = pd.DataFrame(summary).sort_values("coverage_pct", ascending=False)
    R.to_csv("other_data_coverage.csv", index=False)
    print("\n-> other_data_coverage.csv")
    if gradient:
        pd.DataFrame(gradient).to_csv("other_data_coverage_by_decile.csv", index=False)
        print("-> other_data_coverage_by_decile.csv  (table x decile, the transition across percentiles)")
    print("\nREAD IT THIS WAY: coverage decides whether a table can carry an outcome at all. Below about")
    print("half the cohort, a null result means nothing and a positive one is a selected subgroup. Then")
    print("read d1 against d10 and the burdened-to-rest ratio: a flat gradient means the table reaches")
    print("every level of burden alike, and a falling one means the analysis would rest on the people")
    print("least like the ones it is about.")



# ---- cell 5: when they answered, how long the record runs after, and what it holds -------------------
# Nothing here proposes an outcome. It measures whether the window design exists, against the rule
# declared BEFORE the run: take the candidate defined for the most of the most
# burdened, requiring at least 1,200 of them evaluable and a median follow-up of at least 180 days, and
# if no candidate clears both, abandon the window design and report it as abandoned.

# The concepts a visit is recorded under. Printed in full by this cell, because a list written from
# memory is not to be used. These are the starting definitions, taken from
# care.py, and they are revised from the distribution this cell prints if the CDR disagrees.
ED_CONCEPTS = [9203, 262]          # Emergency Room Visit; Emergency Room and Inpatient Visit
# Outpatient Visit; Office Visit; Telehealth. Telehealth is included because the concept distribution
# shows it reaching 15,368 of the cohort: for answers given between
# 2022 and 2024 a video appointment is a care contact, and leaving it out would count someone with
# several as having none. Laboratory Visit (21,323 people) and Pharmacy visit (6,934) are deliberately
# NOT here -- neither is a clinician encounter. The choice was made from the concept counts, before
# any outcome was computed on them.
OP_CONCEPTS = [9202, 581477, 722455]


def _dist(x, label, unit="days"):
    x = pd.Series(x).dropna()
    if not len(x):
        return f"  {label:34s} (none)"
    q = x.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return (f"  {label:34s} n {len(x):6,} | median {q[0.5]:8.0f} {unit} | "
            f"p10 {q[0.1]:7.0f}  p25 {q[0.25]:7.0f}  p75 {q[0.75]:7.0f}  p90 {q[0.9]:7.0f}")


def _split(mask, dec, out, ids, label):
    """One line: how a per-person boolean falls over the cohort, the deciles, and the most burdened."""
    n = len(ids)
    s = f"  {label:34s} {mask.sum():6,} of {n:,} ({100 * mask.mean():5.1f}%)"
    if out:
        inb = ids.isin(out).values
        b, r = mask[inb], mask[~inb]
        s += f" | burdened {100 * b.mean():5.1f}%  rest {100 * r.mean():5.1f}%"
    if dec is not None:
        d = mask.groupby(dec.reindex(ids.values).values).mean() * 100
        s += f" | d1 {d.iloc[0]:5.1f}%  d10 {d.iloc[-1]:5.1f}%"
    return s


def cell5():
    ids = pd.read_csv(COHORT_CSV, usecols=[ID_COL])[ID_COL].astype("int64")
    n = len(ids)
    dec = burden_deciles(ids)
    cpath = os.path.join(CFG.out_dir, "consensus_outliers.csv")
    out = set()
    if os.path.exists(cpath):
        C = pd.read_csv(cpath)
        out = set(C.loc[C["flagged_by_all"].astype(bool), ID_COL].astype("int64"))
    cids = pd.read_csv(CFG.items_csv, usecols=["concept_id"])["concept_id"].astype("int64").tolist()
    print(f"cohort {n:,} | most burdened {len(out):,} | {len(cids)} item concepts\n")

    # --- when they answered ------------------------------------------------------------------------
    A = Q(f"""SELECT person_id, MIN(observation_datetime) AS first_answer,
                     MAX(observation_datetime) AS last_answer
              FROM `{CDR}.observation`
              WHERE person_id IN UNNEST(@ids) AND observation_source_concept_id IN UNNEST(@cids)
              GROUP BY person_id""", "answer dates", ids,
          [bigquery.ArrayQueryParameter("cids", "INT64", cids)])
    if A is None or not len(A):
        print("no answer dates returned -- stopping"); return
    A = A.set_index("person_id")
    A["first_answer"] = pd.to_datetime(A.first_answer).dt.tz_localize(None)
    A["last_answer"] = pd.to_datetime(A.last_answer).dt.tz_localize(None)
    A = A.reindex(ids.values)
    span = (A.last_answer - A.first_answer).dt.days
    print("WHEN THEY ANSWERED (the answer date is the LAST of the 151, because the 6 surveys are")
    print("months apart and the exposure is complete only at the last of them)")
    print(_dist(span, "span, first to last answer"))
    print(f"  {'answer date':34s} median {A.last_answer.median():%Y-%m-%d} | "
          f"range {A.last_answer.min():%Y-%m-%d} to {A.last_answer.max():%Y-%m-%d}")
    print(f"  {'missing an answer date':34s} {int(A.last_answer.isna().sum()):,}\n")

    adate = A.last_answer.dt.date

    # --- how long the record runs after ------------------------------------------------------------
    P = Q(f"""SELECT person_id, observation_period_start_date AS s, observation_period_end_date AS e
              FROM `{CDR}.observation_period` WHERE person_id IN UNNEST(@ids)""", "observation_period", ids)
    P = P.set_index("person_id").reindex(ids.values)
    fu_obs = (pd.to_datetime(P.e, errors="coerce") - pd.to_datetime(A.last_answer)).dt.days

    # --- what the visits look like around that date ------------------------------------------------
    ok = adate.notna().values
    zip_params = [
        bigquery.ArrayQueryParameter("pids", "INT64", [int(i) for i in ids.values[ok]]),
        bigquery.ArrayQueryParameter("adates", "DATE", list(adate.values[ok])),
        bigquery.ArrayQueryParameter("ed", "INT64", ED_CONCEPTS),
        bigquery.ArrayQueryParameter("op", "INT64", OP_CONCEPTS),
    ]
    W = Q(f"""WITH anchor AS (
                SELECT pid, adt FROM UNNEST(@pids) AS pid WITH OFFSET i
                JOIN (SELECT adt, o2 FROM UNNEST(@adates) AS adt WITH OFFSET o2) ON i = o2)
              SELECT a.pid AS person_id,
                     MAX(v.visit_start_date) AS last_visit,
                     COUNTIF(v.visit_start_date >= a.adt) AS n_after,
                     COUNTIF(v.visit_start_date BETWEEN DATE_SUB(a.adt, INTERVAL 365 DAY)
                                                    AND DATE_SUB(a.adt, INTERVAL 1 DAY)) AS n_365_before,
                     COUNTIF(v.visit_start_date BETWEEN a.adt
                                                    AND DATE_ADD(a.adt, INTERVAL 365 DAY)) AS n_365_after,
                     COUNTIF(v.visit_concept_id IN UNNEST(@op)
                             AND v.visit_start_date BETWEEN DATE_SUB(a.adt, INTERVAL 365 DAY)
                                                        AND DATE_SUB(a.adt, INTERVAL 1 DAY)) AS n_op_before,
                     COUNTIF(v.visit_concept_id IN UNNEST(@op)
                             AND v.visit_start_date BETWEEN a.adt
                                                        AND DATE_ADD(a.adt, INTERVAL 365 DAY)) AS n_op_after,
                     COUNTIF(v.visit_concept_id IN UNNEST(@ed)
                             AND v.visit_start_date BETWEEN a.adt
                                                        AND DATE_ADD(a.adt, INTERVAL 365 DAY)) AS n_ed_after
              FROM anchor a JOIN `{CDR}.visit_occurrence` v ON v.person_id = a.pid
              GROUP BY a.pid""", "visits around the answer date", None, zip_params)
    if W is None:
        print("visit window query skipped -- stopping"); return
    # Fill the COUNTS with zero and leave last_visit as a missing date. A blanket .fillna(0) puts an
    # integer into a DATE column, which BigQuery hands back as a db_dtypes array and which raises.
    # People absent from visit_occurrence are absent on purpose: no visits means zero visits and no
    # last visit, not a visit on day zero.
    W = W.set_index("person_id").reindex(ids.values)
    counts = [c for c in W.columns if c != "last_visit"]
    W[counts] = W[counts].fillna(0).astype("int64")
    last_visit = pd.to_datetime(W.last_visit, errors="coerce")
    fu_visit = (last_visit - pd.to_datetime(A.last_answer)).dt.days

    print("HOW LONG THE RECORD RUNS AFTER THE ANSWER DATE")
    print(_dist(fu_obs, "to the observation period end"))
    print(_dist(fu_visit[fu_visit > 0], "to the last visit, when after"))
    # The record can end BEFORE a person finished answering, in which case there is no window at all.
    # This is reported before anything else, because it bounds every candidate below.
    print(_split(pd.Series((fu_obs <= 0).values, index=ids.values), dec, out, ids,
                 "record ends on or before the answer"))
    for d in (180, 365):
        m = pd.Series((fu_obs >= d).values, index=ids.values)
        print(_split(m, dec, out, ids, f"follow-up >= {d} days"))
    print()

    # --- the four candidates, by how many of the most burdened they are DEFINED for -----------------
    ib = ids.isin(out).values
    cand = {
        # (a) needs the 365 days AFTER to be observed, not only an outpatient visit in the 365 before.
        # "No visit in the window" cannot be scored on a window that was never watched, and leaving the
        # follow-up condition out counts short-observed people as having no visit. That error runs with
        # burden, because follow-up is shorter for the most burdened.
        "a": ("no outpatient visit in the 365 days after, among those with one in the 365 before "
              "AND 365 days observed",
              pd.Series((W.n_op_before.values > 0) & (fu_obs >= 365).values, index=ids.values)),
        "b": ("any emergency visit in the 365 days after",
              pd.Series((fu_obs >= 365).values, index=ids.values)),
        "c": ("an emergency visit in that window with no outpatient visit in the prior 90 days",
              pd.Series((fu_obs >= 365).values, index=ids.values)),
        "d": ("outpatient visits in the window as a count, with follow-up as the offset",
              pd.Series((fu_obs > 0).values, index=ids.values)),
    }
    print("WHO EACH CANDIDATE IS DEFINED FOR (rule (i): the one defined for the most of the most")
    print("burdened, needing >= 1,200 of them and a median follow-up >= 180 days)")
    for k, (desc, _) in cand.items():
        print(f"    ({k}) {desc}")
    print()
    rows = []
    for k, (desc, m) in cand.items():
        nb = int(m.values[ib].sum())
        med = float(pd.Series(fu_obs.values[ib & m.values]).median())
        clears = bool(nb >= 1200 and med >= 180)
        rows.append({"candidate": k, "definition": desc, "n_cohort": int(m.sum()), "n_burdened": nb,
                     "median_followup_days_burdened": med, "clears_rule_i": clears})
        print(f"  ({k})  cohort {int(m.sum()):6,}   burdened {nb:5,}   "
              f"median follow-up {med:6.0f} d   {'CLEARS' if clears else 'fails'}")
    pd.DataFrame(rows).to_csv("step22_candidates.csv", index=False)
    print("\n  Candidates (b), (c) and (a) all need a full 365-day window to be observed. (d) does not:")
    print("  the count of visits with follow-up as the model's offset is what absorbs variable")
    print("  observation, which is why it can be defined for anyone observed at all.")

    # --- the visit concepts this CDR actually carries -----------------------------------------------
    V = Q(f"""SELECT v.visit_concept_id, c.concept_name, COUNT(*) AS n_rows,
                     COUNT(DISTINCT v.person_id) AS n_people
              FROM `{CDR}.visit_occurrence` v
              LEFT JOIN `{CDR}.concept` c ON c.concept_id = v.visit_concept_id
              WHERE v.person_id IN UNNEST(@ids)
              GROUP BY 1, 2 ORDER BY n_rows DESC""", "visit concepts", ids)
    if V is not None:
        print("\nVISIT CONCEPTS AMONG OUR COHORT (revise ED_CONCEPTS and OP_CONCEPTS from this)")
        print(V.head(20).to_string(index=False))
        V.to_csv("step22_visit_concepts.csv", index=False)

    pd.DataFrame({"followup_obs_days": fu_obs.values, "span_days": span.values,
                  "decile": (dec.reindex(ids.values).values if dec is not None else None),
                  "most_burdened": ib}).groupby(["decile", "most_burdened"]).agg(
        n=("followup_obs_days", "size"), median_followup=("followup_obs_days", "median"),
        median_span=("span_days", "median")).reset_index().to_csv("step22_followup_by_decile.csv", index=False)
    print("\n-> step22_candidates.csv, step22_visit_concepts.csv, step22_followup_by_decile.csv")



# ---- cell 6: what the cognitive-task and Fitbit tables actually hold, and who they reach ------------
# No design is proposed here. Three things have to be known first and none of them is known yet:
# what the columns ARE (cell 1 gave counts, not names), how much Fitbit data each covered person has
# (presence is not density), and WHY Fitbit coverage rises with burden when it was predicted to fall.
# That last one matters most: the most burdened are about 8 years younger, and if the gradient is age
# then it is confounded with the exposure and any Fitbit result inherits the confound.

TASK_TABLES = ["flanker", "gradcpt", "emorecog", "delaydiscounting"]
TASK_OUTCOMES = ["ds_flanker_outcomes", "ds_gradcpt_outcomes", "ds_emorecog_outcomes",
                 "ds_delaydiscounting_outcomes"]
FITBIT_TABLES = ["ds_fitbit_device", "activity_summary", "sleep_daily_summary", "heart_rate_summary",
                 "wear_study"]
ID_CANDIDATES = ["person_id", "research_id", "participant_id"]
NUMERIC_TYPES = ["INT64", "FLOAT64", "NUMERIC", "BIGNUMERIC"]


def _key_of(columns):
    """The person key of a table, matched WITHOUT REGARD TO CASE.

    The ds_*_outcomes tables spell it PERSON_ID, so a literal match would report all four as
    `key NONE FOUND` and skip them. Returns the column as the
    table spells it, so it can go straight into SQL.
    """
    low = {c.lower(): c for c in columns}
    return next((low[k] for k in ID_CANDIDATES if k in low), None)


def _burden_logit(has, decile, age, indent="    "):
    """Log-odds of `has` per SD of burden rank, before and after age. Returns a dict, prints nothing
    on success so that each caller words its own verdict.

    Why a logistic and not a table read within age bands: a fixture where coverage depended on age
    ALONE, with age and burden correlated as in the real cohort, still showed an 18-point within-band
    gradient in the youngest third. Banding a continuous confounder does not control it when the
    confounder and the exposure track each other. Age enters as its rank percentile and that squared.

    This is the same estimand cell 6 used for Fitbit coverage, factored out so that a gradient in task
    reach and a gradient in Fitbit coverage are comparable numbers rather than two similar recipes.
    """
    try:
        import statsmodels.api as sm
    except Exception as e:
        print(f"{indent}[not run ({type(e).__name__}: {e})]")
        return None
    T = pd.DataFrame({"has": pd.to_numeric(pd.Series(np.asarray(has)), errors="coerce"),
                      "decile": pd.to_numeric(pd.Series(np.asarray(decile)), errors="coerce"),
                      "age": pd.to_numeric(pd.Series(np.asarray(age)), errors="coerce")})
    T = T[T.has.notna() & T.decile.notna() & T.age.notna()]
    if len(T) < 100 or T.has.nunique() < 2:
        print(f"{indent}[not run: {len(T):,} rows with all three known, "
              f"{T.has.nunique()} distinct outcome values]")
        return None
    try:
        b = pd.Series(T.decile.astype(float)).rank(pct=True)
        b = ((b - b.mean()) / b.std()).values
        a = pd.Series(T.age.astype(float)).rank(pct=True)
        a = ((a - a.mean()) / a.std()).values
        y = T.has.astype(float).values
        m1 = sm.GLM(y, sm.add_constant(b.reshape(-1, 1)), family=sm.families.Binomial()).fit()
        m2 = sm.GLM(y, sm.add_constant(np.column_stack([b, a, a ** 2])),
                    family=sm.families.Binomial()).fit()
    except Exception as e:
        print(f"{indent}[not run ({type(e).__name__}: {e})]")
        return None
    c1, c2, se2 = float(m1.params[1]), float(m2.params[1]), float(m2.bse[1])
    # Unrounded. Callers round when they write, so a printed interval is never rounded twice.
    return {"n": int(len(T)), "rate": float(T.has.mean()), "unadjusted": c1, "age_adjusted": c2,
            "se": se2, "lo": c2 - 1.96 * se2, "hi": c2 + 1.96 * se2,
            "share_explained_by_age_pct": 100 * (1 - abs(c2) / abs(c1)) if c1 else float("nan"),
            "survives_age": bool(abs(c2) >= 1.96 * se2)}


def _burden_linear(y, decile, age, kind="mean", extra=None):
    """Change in a CONTINUOUS measure per SD of burden rank, before and after age. Returns a dict.

    `extra`, when given, is a DataFrame of further numeric columns to adjust for alongside age, aligned
    positionally with y. It changes nothing when omitted, so the cell 8 and cell 9 figures stand.

    The continuous analogue of `_burden_logit`, and it exists for the same reason: a table read by
    decile is the shape and not the answer, because age tracks burden and took most of the task-reach
    gradient in cell 7. Burden and age enter exactly as they do there, as rank percentiles standardized
    to unit variance, with age squared as well, so the two coefficients mean the same thing on the same
    scale and only the outcome differs.

    kind="mean" is least squares. kind="median" is a median regression, which is the model that matches
    a table of medians and is the one to trust when the outcome is as skewed as days of wear, where a
    handful of people with years of data pull the mean around.
    """
    try:
        import statsmodels.api as sm
    except Exception as e:
        print(f"    [not run ({type(e).__name__}: {e})]")
        return None
    T = pd.DataFrame({"y": pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce"),
                      "decile": pd.to_numeric(pd.Series(np.asarray(decile)), errors="coerce"),
                      "age": pd.to_numeric(pd.Series(np.asarray(age)), errors="coerce")})
    E = None
    if extra is not None and len(extra.columns):
        E = pd.DataFrame(np.asarray(extra, dtype=float), columns=list(extra.columns))
        # a constant column inside a subsample would make the design singular
        E = E.loc[:, E.std(axis=0).fillna(0) > 0]
        T = pd.concat([T.reset_index(drop=True), E.reset_index(drop=True)], axis=1)
    keep = T.y.notna() & T.decile.notna() & T.age.notna()
    if E is not None and len(E.columns):
        keep &= T[list(E.columns)].notna().all(axis=1)
    T = T[keep]
    if len(T) < 100 or T.y.nunique() < 3:
        print(f"    [not run: {len(T):,} rows with everything known]")
        return None
    b = pd.Series(T.decile.astype(float)).rank(pct=True)
    b = ((b - b.mean()) / b.std()).values
    a = pd.Series(T.age.astype(float)).rank(pct=True)
    a = ((a - a.mean()) / a.std()).values
    yy = T.y.astype(float).values
    Z = T[list(E.columns)].to_numpy(dtype=float) if (E is not None and len(E.columns)) else None
    # ★ A median regression stops at an iteration limit and returns the last iterate ANYWAY, with no
    # error and a perfectly ordinary-looking coefficient, and sleep reaches the default limit of
    # 1,000. The limit is raised and the warning is CAUGHT, so a number that did not converge is
    # reported as one rather than quoted: record what the estimator actually did.
    import warnings as _w
    converged = True
    X1 = sm.add_constant(b.reshape(-1, 1))
    cols = [b, a, a ** 2] + ([Z] if Z is not None else [])
    X2 = sm.add_constant(np.column_stack(cols))
    try:
        if kind == "median":
            with _w.catch_warnings(record=True) as caught:
                _w.simplefilter("always")
                m1 = sm.QuantReg(yy, X1).fit(q=0.5, max_iter=20000)
                m2 = sm.QuantReg(yy, X2).fit(q=0.5, max_iter=20000)
            converged = not any("iteration" in str(c.message).lower() for c in caught)
        else:
            m1 = sm.OLS(yy, X1).fit()
            m2 = sm.OLS(yy, X2).fit()
    except Exception as e:
        print(f"    [not run ({type(e).__name__}: {e})]")
        return None
    c1, c2, se2 = float(m1.params[1]), float(m2.params[1]), float(m2.bse[1])
    center = float(np.median(yy)) if kind == "median" else float(np.mean(yy))
    return {"kind": kind, "n": int(len(T)), "center": center, "unadjusted": c1, "age_adjusted": c2,
            "se": se2, "lo": c2 - 1.96 * se2, "hi": c2 + 1.96 * se2,
            "share_explained_by_age_pct": 100 * (1 - abs(c2) / abs(c1)) if c1 else float("nan"),
            "survives_age": bool(abs(c2) >= 1.96 * se2), "converged": converged}


def _columns(tables):
    """Column names and types, from INFORMATION_SCHEMA. Metadata, free."""
    lst = ", ".join(f"'{t}'" for t in tables)
    return Q(f"""SELECT table_name, column_name, data_type, ordinal_position
                 FROM `{CDR}.INFORMATION_SCHEMA.COLUMNS`
                 WHERE table_name IN ({lst}) ORDER BY table_name, ordinal_position""", "columns")


def cell6():
    ids = pd.read_csv(COHORT_CSV, usecols=[ID_COL])[ID_COL].astype("int64")
    n = len(ids)
    dec = burden_deciles(ids)
    tables = TASK_TABLES + TASK_OUTCOMES + FITBIT_TABLES
    C = _columns(tables)
    if C is None or not len(C):
        print("no column metadata returned"); return
    C.to_csv("task_fitbit_columns.csv", index=False)

    print("=" * 96)
    print("WHAT THESE TABLES HOLD (metadata, free). The parsed measures are what a design would use,")
    print("so the question is which table carries them and under what person key.\n")
    keyed = {}
    for t in tables:
        d = C[C.table_name == t]
        if not len(d):
            print(f"  {t:32s} ABSENT from this CDR")
            continue
        key = _key_of(d.column_name)
        keyed[t] = key
        num = d[d.data_type.isin(NUMERIC_TYPES)].column_name.tolist()
        dts = d[d.data_type.isin(["DATE", "DATETIME", "TIMESTAMP"])].column_name.tolist()
        print(f"  {t:32s} {len(d):2d} cols | key {key or 'NONE FOUND'}")
        print(f"      numeric: {', '.join(num) if num else '(none)'}")
        if dts:
            print(f"      dates  : {', '.join(dts)}")
        other = [c for c in d.column_name if c not in num + dts + ([key] if key else [])]
        if other:
            print(f"      other  : {', '.join(other[:8])}{' ...' if len(other) > 8 else ''}")
    print()

    # ---- where the measures are profiled ------------------------------------------------------------
    # Not here: on the raw task tables the first numeric column is `sitting_id`, a row identifier.
    # Cell 7 profiles them: every measure, how many people have one, what the quality flags remove,
    # and whether the battery reaches the burdened end of the cohort.
    print("=" * 96)
    print("THE PARSED TASK MEASURES are the ds_*_outcomes columns above, one row per sitting. What")
    print("they hold among our cohort is cell 7, which also reports reach after the quality flags.\n")

    # ---- Fitbit: density, not presence -------------------------------------------------------------
    print("\n" + "=" * 96)
    print("FITBIT DENSITY. Presence in a table means at least 1 day. A sleep or step measure needs days")
    print("NEAR the survey, so days are counted relative to each person's answer date.\n")
    A = Q(f"""SELECT person_id, MAX(observation_datetime) AS last_answer
              FROM `{CDR}.observation`
              WHERE person_id IN UNNEST(@ids) AND observation_source_concept_id IN UNNEST(@cids)
              GROUP BY person_id""", "answer dates", ids,
          [bigquery.ArrayQueryParameter(
              "cids", "INT64",
              pd.read_csv(CFG.items_csv, usecols=["concept_id"])["concept_id"].astype("int64").tolist())])
    adate = None
    if A is not None and len(A):
        adate = pd.to_datetime(A.set_index("person_id").last_answer).dt.tz_localize(None).reindex(ids.values)

    for t in ["activity_summary", "sleep_daily_summary", "heart_rate_summary"]:
        key, d = keyed.get(t), C[C.table_name == t]
        dts = d[d.data_type.isin(["DATE", "DATETIME", "TIMESTAMP"])].column_name.tolist()
        if key is None or not dts:
            print(f"  {t}: key {key}, date columns {dts or 'none'} -- skipped")
            continue
        dc = dts[0]
        if adate is None:
            continue
        ok = adate.notna().values
        r = Q(f"""WITH anchor AS (
                    SELECT pid, adt FROM UNNEST(@pids) AS pid WITH OFFSET i
                    JOIN (SELECT adt, o2 FROM UNNEST(@adates) AS adt WITH OFFSET o2) ON i = o2)
                  SELECT COUNT(DISTINCT a.pid) AS n_people,
                         APPROX_QUANTILES(d_all, 5) AS q_all,
                         APPROX_QUANTILES(d_win, 5) AS q_win
                  FROM (SELECT a.pid,
                               COUNT(DISTINCT DATE(v.{dc})) AS d_all,
                               COUNT(DISTINCT IF(DATE(v.{dc}) BETWEEN DATE_SUB(a.adt, INTERVAL 180 DAY)
                                                 AND DATE_ADD(a.adt, INTERVAL 180 DAY),
                                                 DATE(v.{dc}), NULL)) AS d_win
                        FROM anchor a JOIN `{CDR}.{t}` v ON v.{key} = a.pid
                        GROUP BY a.pid) a""", f"{t} density",
              None,
              [bigquery.ArrayQueryParameter("pids", "INT64", [int(i) for i in ids.values[ok]]),
               bigquery.ArrayQueryParameter("adates", "DATE", list(adate.dt.date.values[ok]))])
        if r is None or not len(r):
            continue
        row = r.iloc[0]
        qa, qw = list(row.q_all), list(row.q_win)
        print(f"  {t:24s} {int(row.n_people):6,} people | days ANY TIME "
              f"min {qa[0]:.0f} q1 {qa[1]:.0f} med {qa[2]:.0f} q3 {qa[3]:.0f} max {qa[4]:.0f}")
        print(f"  {'':24s} {'':6s}   days WITHIN 180 OF THE ANSWER "
              f"min {qw[0]:.0f} q1 {qw[1]:.0f} med {qw[2]:.0f} q3 {qw[3]:.0f} max {qw[4]:.0f}")

    # ---- is the Fitbit coverage gradient age? ------------------------------------------------------
    print("\n" + "=" * 96)
    print("IS THE FITBIT COVERAGE GRADIENT AGE? Coverage ROSE with burden in cell 2,\n"
          "against the prediction that it would fall. The most burdened")
    print("were younger in that run. If the gradient is age it is confounded with the")
    print("exposure, so it is read within age bands here.\n")
    cov = Q(f"""SELECT DISTINCT person_id FROM `{CDR}.ds_fitbit_device`
                WHERE person_id IN UNNEST(@ids)""", "fitbit roster", ids)
    if cov is None or dec is None:
        print("  cannot check: roster or deciles unavailable"); return
    has = pd.Series(ids.isin(set(cov.person_id.astype("int64"))).values, index=ids.values)
    age = pd.read_csv(COHORT_CSV, usecols=[ID_COL, "age"]).set_index(ID_COL)["age"].reindex(ids.values)
    band = pd.qcut(age, 3, labels=["youngest third", "middle third", "oldest third"])
    T = pd.DataFrame({"has": has.values, "decile": dec.reindex(ids.values).values,
                      "band": band.values, "age": age.values})
    g = T.groupby(["band", "decile"], observed=True).has.mean().unstack() * 100
    print("  Fitbit coverage (%) by burden decile, WITHIN age bands:")
    print(g.round(1).to_string())
    print(f"\n  mean age: has Fitbit {T.loc[T.has, 'age'].mean():.1f}, does not "
          f"{T.loc[~T.has, 'age'].mean():.1f}")
    within = (g.iloc[:, -1] - g.iloc[:, 0]).round(1)
    print(f"  decile 10 minus decile 1, within each band: {within.to_dict()}")
    print("  Read that table for shape, NOT as the answer. Thirds are too coarse to separate the two:")
    print("  age still varies inside a band and still tracks burden there, so the confound survives")
    print("  the banding. The adjustment below is what answers it.\n")

    # Burden against Fitbit coverage, before and after age. If the burden coefficient collapses once
    # age is in, the gradient WAS age and any Fitbit design inherits the confound.
    res = _burden_logit(T.has.values, T.decile.values, T.age.values, indent="  ")
    if res is None:
        print("  [the band table is all there is]")
    else:
        c1, c2, se2 = res["unadjusted"], res["age_adjusted"], res["se"]
        print("  Fitbit coverage on burden, log-odds per SD of burden rank:")
        print(f"    unadjusted        {c1:+.3f}")
        print(f"    with age adjusted {c2:+.3f}  (95% CI {res['lo']:+.3f} to {res['hi']:+.3f})")
        print(f"    age accounts for {res['share_explained_by_age_pct']:.0f}% of the unadjusted association")
        verdict = ("THE GRADIENT SURVIVES AGE. A Fitbit design can proceed with age as a covariate."
                   if res["survives_age"] else
                   "THE GRADIENT WAS AGE. A Fitbit design is confounded with the exposure.")
        print(f"    -> {verdict}")
        print("  Age is one of the two candidate explanations. The other, that the program HANDS OUT")
        print("  devices through the wear study, is cell 7 part B.")
        pd.DataFrame([{"burden_logodds_unadjusted": round(c1, 4),
                       "burden_logodds_age_adjusted": round(c2, 4), "se": round(se2, 4),
                       "lo": round(res["lo"], 4), "hi": round(res["hi"], 4),
                       "share_explained_by_age_pct": round(res["share_explained_by_age_pct"], 1),
                       "gradient_survives_age": res["survives_age"],
                       "verdict": verdict}]).to_csv("fitbit_age_check.csv", index=False)
    T.groupby(["band", "decile"], observed=True).has.agg(["size", "mean"]).reset_index().to_csv(
        "fitbit_coverage_by_age_band.csv", index=False)
    print("\n-> task_fitbit_columns.csv, fitbit_coverage_by_age_band.csv, fitbit_age_check.csv")


# ---- cell 3: what is actually in one table, for our cohort -----------------------------------------------
def cell3():
    ids = pd.read_csv(COHORT_CSV, usecols=[ID_COL])[ID_COL].astype("int64")
    concept_col = {"condition_occurrence": "condition_concept_id", "drug_exposure": "drug_concept_id",
                   "procedure_occurrence": "procedure_concept_id", "measurement": "measurement_concept_id",
                   "observation": "observation_concept_id", "device_exposure": "device_concept_id"}[TABLE]
    sql = f"""SELECT a.{concept_col} AS concept_id, c2.concept_name, c2.vocabulary_id,
                     COUNT(DISTINCT a.person_id) AS n_people
              FROM `{CDR}.{TABLE}` a
              LEFT JOIN `{CDR}.concept` c2 ON c2.concept_id = a.{concept_col}
              WHERE a.person_id IN UNNEST(@ids)
              GROUP BY 1, 2, 3 ORDER BY n_people DESC LIMIT {TOP}"""
    d = Q(sql, f"{TABLE} concepts", ids)
    if d is None:
        return
    n = len(ids)
    d["coverage_pct"] = (100 * d.n_people / n).round(1)
    print(f"\nmost common concepts in {TABLE} among the cohort of record:\n")
    print(d.to_string(index=False))
    d.to_csv(f"other_data_concepts_{TABLE}.csv", index=False)
    print(f"\n-> other_data_concepts_{TABLE}.csv")


# ---- cell 4: the measurement table in detail ------------------------------------------------------------
def cell4():
    """`measurement` holds two different things and only the split tells you which you have.

    The program takes physical measurements at enrollment (height, weight, blood pressure, waist, pulse)
    on nearly every participant. Those are MEASURED by program staff, on a standard protocol, and are
    about as independent of a self-report questionnaire as anything in the Repository. The same table
    also carries clinical laboratory results copied from health records, which exist only for people with
    a linked record and differ from person to person according to what their clinicians happened to order.
    The first can carry an analysis of the whole cohort. The second cannot, and selects on health care
    use, which is part of what this paper studies.

    ★ CLINICAL LABORATORY RESULTS ARE COUNTED HERE TOO, AND NOTHING IS DROPPED BEFORE YOU SEE IT.
    Pooling every recording type together and cutting the concept list at --top would answer neither
    question: the program's physical measurements cover most of the cohort and would fill the whole
    list, pushing every laboratory concept below the cut, and the tail would be gone before anyone could
    judge it. So:
      - the concept table is grouped BY RECORDING TYPE as well as by concept, and the top of EACH type
        is printed separately, so the labs are read on their own terms rather than against measurements
        they cannot compete with on coverage,
      - the types are taken from the data and never from a list of names written here, because which
        type concept the program's measurements carry is a fact about this CDR,
      - EVERY concept is written to the CSV whatever its coverage. --top governs printing only.
    The judgment of what is usable is then made by reading the file, not by this script.

    This cell prints the split, then the measurements with the share of our cohort that has a NUMERIC
    value, which is the only kind that can enter an analysis.
    """
    ids = pd.read_csv(COHORT_CSV, usecols=[ID_COL])[ID_COL].astype("int64")
    n = len(ids)

    d = Q(f"""SELECT a.measurement_type_concept_id AS type_id, c.concept_name AS type_name,
                     COUNT(DISTINCT a.person_id) AS n_people, COUNT(*) AS n_rows,
                     COUNT(DISTINCT a.measurement_concept_id) AS n_concepts,
                     COUNT(DISTINCT IF(a.value_as_number IS NOT NULL, a.person_id, NULL)) AS n_people_numeric
              FROM `{CDR}.measurement` a
              LEFT JOIN `{CDR}.concept` c ON c.concept_id = a.measurement_type_concept_id
              WHERE a.person_id IN UNNEST(@ids)
              GROUP BY 1, 2 ORDER BY n_people DESC""", "measurement by type", ids)
    if d is not None:
        d["coverage_pct"] = (100 * d.n_people / n).round(1)
        d["coverage_numeric_pct"] = (100 * d.n_people_numeric / n).round(1)
        print(f"\nmeasurement rows by how they were recorded -- the split that matters (cohort {n:,}):\n")
        print(d.to_string(index=False))
        d.to_csv("other_data_measurement_types.csv", index=False)

    d2 = Q(f"""SELECT a.measurement_type_concept_id AS type_id,
                      t.concept_name AS type_name,
                      a.measurement_concept_id AS concept_id,
                      c.concept_name, c.vocabulary_id,
                      COUNT(DISTINCT a.person_id) AS n_people,
                      COUNT(DISTINCT IF(a.value_as_number IS NOT NULL, a.person_id, NULL)) AS n_people_numeric,
                      COUNT(*) AS n_rows,
                      ROUND(APPROX_QUANTILES(a.value_as_number, 2)[OFFSET(1)], 2) AS median_value,
                      APPROX_TOP_COUNT(u.concept_name, 1)[SAFE_OFFSET(0)].value AS top_unit
               FROM `{CDR}.measurement` a
               LEFT JOIN `{CDR}.concept` c ON c.concept_id = a.measurement_concept_id
               LEFT JOIN `{CDR}.concept` t ON t.concept_id = a.measurement_type_concept_id
               LEFT JOIN `{CDR}.concept` u ON u.concept_id = a.unit_concept_id
               WHERE a.person_id IN UNNEST(@ids)
               GROUP BY 1, 2, 3, 4, 5
               ORDER BY n_people_numeric DESC""", "measurement concepts by type", ids)
    if d2 is not None:
        d2["coverage_numeric_pct"] = (100 * d2.n_people_numeric / n).round(1)
        d2.to_csv("other_data_measurement_concepts.csv", index=False)     # ★ everything, unfiltered
        print(f"\n{len(d2):,} distinct concept x recording-type pairs. ALL of them are in the CSV. "
              f"Printing the top {TOP} within each type:")
        for tid, g in d2.groupby("type_id", sort=False):
            tname = str(g["type_name"].iloc[0])
            print(f"\n  ---- recording type {tid} ({tname}): {len(g):,} concepts, "
                  f"{int(g.n_people_numeric.max()):,} people at best numeric coverage")
            cols = ["concept_id", "concept_name", "vocabulary_id", "n_people", "n_people_numeric",
                    "coverage_numeric_pct", "median_value", "top_unit"]
            print(g.sort_values("n_people_numeric", ascending=False).head(TOP)[cols].to_string(index=False))
        for thr in (50, 20, 10, 5, 1):
            k = int((d2.coverage_numeric_pct >= thr).sum())
            print(f"  concepts with a numeric value for >= {thr:>2}% of the cohort: {k:,}")
        print("\n-> other_data_measurement_types.csv, other_data_measurement_concepts.csv")
    print("\nREAD IT THIS WAY: a measurement is usable for this paper only if a large share of the 90,351")
    print("have a numeric value AND it was taken on a protocol rather than ordered by a clinician.")
    print("Anything present only for people with many health care visits selects on the thing we study.")
    print("Both halves are counted and written. Nothing is discarded here -- that call is made by")
    print("reading other_data_measurement_concepts.csv, which holds every concept at every coverage.")


# ---- cell 7: the parsed task measures, and whether the Fitbit gradient is the program's devices ----
# Two checks. Neither proposes a design, and neither is a result.
#
# PART A. Cell 6 lists the ds_*_outcomes columns and profiles none of them, so it gives the names of
# the parsed measures and nothing else. A measure is usable only if enough of the cohort have one, it varies, and
# it survives the table's own quality flags, and the flags are the part that is easy to forget: a
# battery that reaches half the cohort reaches rather less than half once the flagged sittings go.
# Reach is then read by burden decile and adjusted for age, because an online task battery is
# volunteered for and selection into it can run opposite to burden, which is the thing under study.
#
# PART B. Fitbit coverage RISES with burden and cell 6 ruled out age. The second candidate explanation
# was never checked: the program DISTRIBUTES devices through the wear study, so the gradient could be
# who was given one rather than anything about the participants. The test is the same estimand among
# the people the program did NOT hand a device to. Declared before the run: if the burden coefficient
# among non-enrollees has an interval covering zero, distribution explains the gradient; if it holds
# but at under half the whole-cohort coefficient, distribution explains part of it; otherwise it does
# not explain it. Enrollment is a row in `wear_study` with a consent start date, and the value counts
# of `resultsconsent_wear` are printed beside it so that definition can be seen to be the right one.

USABLE_SHARE = 0.20      # a measure reaching under this share of the cohort is marked thin. Declared here.


def cell7():
    ids = pd.read_csv(COHORT_CSV, usecols=[ID_COL])[ID_COL].astype("int64")
    n = len(ids)
    dec = burden_deciles(ids)
    age = pd.read_csv(COHORT_CSV, usecols=[ID_COL, "age"]).set_index(ID_COL)["age"].reindex(ids.values)
    C = _columns(TASK_OUTCOMES + ["wear_study", "ds_fitbit_device"])
    if C is None or not len(C):
        print("no column metadata returned"); return
    dvec = dec.reindex(ids.values).values if dec is not None else None

    # ---- part A: the parsed measures ---------------------------------------------------------------
    print("=" * 96)
    print("PART A -- THE PARSED TASK MEASURES. Cell 6 skipped these four tables because it matched the")
    print("person key case-sensitively and they spell it PERSON_ID. Rows are sittings, not people.")
    print(f"A measure reaching under {USABLE_SHARE:.0%} of the {n:,} is marked thin, a measure that does")
    print("not vary is marked constant. Quartiles are for reading, not for a result.\n")

    raw = None
    if os.path.exists("other_data_coverage.csv"):
        d = pd.read_csv("other_data_coverage.csv")
        raw = d.set_index("table")["n_people"] if "table" in d and "n_people" in d else None

    prof, usable = [], {}
    for t in TASK_OUTCOMES:
        d = C[C.table_name == t]
        if not len(d):
            print(f"  {t} ABSENT from this CDR\n"); continue
        key = _key_of(d.column_name)
        if key is None:
            print(f"  {t}: no person key among {ID_CANDIDATES} -- skipped\n"); continue
        cols = list(d.column_name)
        # Flags are chosen by NAME, not by type, so a flag stored as a boolean is still caught.
        flags = [c for c in cols if c.upper().startswith("FLAG_") or c.upper() == "ANY_TIMEOUTS"]
        skip = {k.lower() for k in ID_CANDIDATES} | {"sitting_id", "src_id"}
        meas = [c for c in d[d.data_type.isin(NUMERIC_TYPES)].column_name
                if c.lower() not in skip and c not in flags]
        if not meas:
            print(f"  {t}: no numeric measure outside the identifiers -- skipped\n"); continue

        sel = []
        for c in meas:
            sel.append(f"COUNT({c}) AS n_{c}, COUNT(DISTINCT IF({c} IS NOT NULL, {key}, NULL)) AS p_{c}, "
                       f"AVG({c}) AS mu_{c}, STDDEV({c}) AS sd_{c}, APPROX_QUANTILES({c}, 4) AS q_{c}")
        for c in flags:
            sel.append(f"AVG(IFNULL(CAST({c} AS INT64), 0)) AS fr_{c}, "
                       f"COUNT(DISTINCT IF(IFNULL(CAST({c} AS INT64), 0) != 0, {key}, NULL)) AS fp_{c}")
        # A null flag is read as not flagged. Stated so, because the alternative is to drop the sitting.
        clean = " + ".join(f"IFNULL(CAST({c} AS INT64), 0)" for c in flags) if flags else "0"
        r = Q(f"""SELECT COUNT(DISTINCT {key}) AS n_people, COUNT(*) AS n_rows,
                         COUNT(DISTINCT IF(({clean}) = 0, {key}, NULL)) AS n_clean_people,
                         COUNTIF(({clean}) = 0) AS n_clean_rows,
                         {', '.join(sel)}
                  FROM `{CDR}.{t}` WHERE {key} IN UNNEST(@ids)""", t, ids)
        if r is None or not len(r):
            continue
        row = r.iloc[0]
        npp, nrows = int(row.n_people), int(row.n_rows)
        print(f"  {t}   key {key}")
        line = (f"      {npp:,} of {n:,} ({100 * npp / n:.1f}%) | {nrows:,} sittings, "
                f"{nrows / max(npp, 1):.2f} per person")
        bare = t.replace("ds_", "").replace("_outcomes", "")
        if raw is not None and bare in raw.index:
            gap = int(raw[bare]) - npp
            line += (f" | `{bare}` reaches {int(raw[bare]):,}, so {gap:,} have a sitting with no "
                     f"parsed row" if gap else f" | matches `{bare}` exactly")
        print(line)
        if flags:
            ncp, ncr = int(row.n_clean_people), int(row.n_clean_rows)
            print(f"      after the {len(flags)} quality flags: {ncp:,} ({100 * ncp / n:.1f}% of the "
                  f"cohort) keep at least 1 unflagged sitting, {ncr:,} of {nrows:,} sittings survive")

        M = []
        for c in meas:
            q = row[f"q_{c}"]
            q = list(q) if q is not None and len(q) == 5 else []
            sd = float(row[f"sd_{c}"] or 0)
            pp, nn = int(row[f"p_{c}"]), int(row[f"n_{c}"])
            note = "CONSTANT" if sd == 0 else ("thin" if pp < USABLE_SHARE * n else "")
            M.append({"measure": c, "people": pp, "pct": round(100 * pp / n, 1), "sittings": nn,
                      "mean": round(float(row[f"mu_{c}"] or 0), 4), "sd": round(sd, 4),
                      "p25": round(float(q[1]), 4) if q else None,
                      "median": round(float(q[2]), 4) if q else None,
                      "p75": round(float(q[3]), 4) if q else None, "note": note})
            prof.append(dict(table=t, **M[-1]))
        print(pd.DataFrame(M).to_string(index=False, na_rep="-"))
        if flags:
            F = [{"flag": c, "share_of_sittings": round(float(row[f"fr_{c}"] or 0), 4),
                  "people_with_any": int(row[f"fp_{c}"])} for c in flags]
            print(pd.DataFrame(F).to_string(index=False))
        print()

        # The person list, for the reach gradient. Only the key column is scanned.
        u = Q(f"""SELECT DISTINCT {key} AS pid FROM `{CDR}.{t}`
                  WHERE {key} IN UNNEST(@ids) AND ({clean}) = 0""", f"{t} usable roster", ids)
        if u is not None:
            usable[t] = set(u.pid.astype("int64"))

    if prof:
        pd.DataFrame(prof).to_csv("task_measure_profile.csv", index=False)

    # ---- part A, continued: does the battery reach the burdened end? -------------------------------
    if usable and dvec is not None:
        print("=" * 96)
        print("WHO THE BATTERY REACHES. Counted on people with at least 1 UNFLAGGED sitting, which is")
        print("the set a design could use. The banded reading is for shape; the adjusted coefficient")
        print("below it is the answer, for the reason written into _burden_logit.\n")
        G, rows = {}, []
        for t, s in usable.items():
            has = pd.Series(ids.isin(s).values, index=ids.values)
            G[t] = (pd.DataFrame({"has": has.values, "decile": dvec})
                    .groupby("decile").has.mean() * 100).round(1)
        print("  usable reach (%) by burden decile:")
        print(pd.DataFrame(G).T.to_string())
        print("\n  reach on burden, log-odds per SD of burden rank:")
        for t, s in usable.items():
            res = _burden_logit(ids.isin(s).values, dvec, age.values, indent="    ")
            if res is None:
                continue
            print(f"    {t:32s} unadjusted {res['unadjusted']:+.3f} | age adjusted "
                  f"{res['age_adjusted']:+.3f} (95% CI {res['lo']:+.3f} to {res['hi']:+.3f})")
            rows.append({"table": t, "n": res["n"], "reach_pct": round(100 * res["rate"], 1),
                         "burden_logodds_unadjusted": round(res["unadjusted"], 4),
                         "burden_logodds_age_adjusted": round(res["age_adjusted"], 4),
                         "lo": round(res["lo"], 4), "hi": round(res["hi"], 4),
                         "gradient_survives_age": res["survives_age"]})
        if rows:
            pd.DataFrame(rows).to_csv("task_reach_gradient.csv", index=False)
        print("\n  A POSITIVE coefficient means the battery reaches the burdened MORE, as Fitbit did\n"
              "  in cell 6.")
        print("  A negative one means a task design would rest on the people least")
        print("  like the ones the paper is about. Either way it is read, not assumed.")

    # ---- part B: is the Fitbit gradient the program handing out devices? ---------------------------
    print("\n" + "=" * 96)
    print("PART B -- IS THE FITBIT GRADIENT THE PROGRAM'S DEVICES? Age is ruled out (cell 6). The wear")
    print("study DISTRIBUTES devices, so the gradient could be who was given one. The test is the same")
    print("estimand among the people who were NOT, with the rule declared before the run.\n")
    if not len(C[C.table_name == "wear_study"]):
        print("  wear_study ABSENT from this CDR -- the check cannot run"); return

    W = Q(f"""SELECT resultsconsent_wear AS consent,
                     COUNT(DISTINCT person_id) AS people,
                     COUNT(DISTINCT IF(wear_consent_start_date IS NOT NULL, person_id, NULL)) AS with_start,
                     COUNT(DISTINCT IF(wear_consent_end_date IS NOT NULL, person_id, NULL)) AS with_end
              FROM `{CDR}.wear_study` WHERE person_id IN UNNEST(@ids)
              GROUP BY 1 ORDER BY people DESC""", "wear_study consent", ids)
    if W is not None and len(W):
        print("  what wear_study says about our cohort, by the consent value it records:")
        print(W.to_string(index=False))
        print()

    E = Q(f"""SELECT DISTINCT person_id FROM `{CDR}.wear_study`
              WHERE person_id IN UNNEST(@ids) AND wear_consent_start_date IS NOT NULL""",
          "wear_study enrollees", ids)
    F = Q(f"""SELECT DISTINCT person_id FROM `{CDR}.ds_fitbit_device`
              WHERE person_id IN UNNEST(@ids)""", "fitbit roster", ids)
    if E is None or F is None:
        print("  roster unavailable -- the check cannot run"); return
    enrolled = pd.Series(ids.isin(set(E.person_id.astype("int64"))).values, index=ids.values)
    hasfit = pd.Series(ids.isin(set(F.person_id.astype("int64"))).values, index=ids.values)
    ne, nf = int(enrolled.sum()), int(hasfit.sum())
    both = int((enrolled & hasfit).sum())
    print(f"  enrolled in the wear study (a consent start date): {ne:,} of {n:,} ({100 * ne / n:.1f}%)")
    print(f"  with a Fitbit device record: {nf:,} ({100 * nf / n:.1f}%), of whom {both:,} "
          f"({100 * both / max(nf, 1):.1f}%) are wear-study enrollees")
    if nf and both / nf < 0.05:
        print("  Distribution reaches too few of the Fitbit havers to account for much, whatever the")
        print("  coefficients below say. Read them with that in mind.")

    if dvec is None:
        print("\n  burden deciles unavailable -- the gradient cannot be read"); return
    T = pd.DataFrame({"fit": hasfit.values, "enrolled": enrolled.values, "decile": dvec,
                      "age": age.values})
    tab = T.groupby("decile").agg(fitbit_pct=("fit", lambda s: round(100 * s.mean(), 1)),
                                  enrolled_pct=("enrolled", lambda s: round(100 * s.mean(), 1)))
    tab["enrollee_share_of_havers"] = T[T.fit].groupby("decile").enrolled.mean().mul(100).round(1)
    nonen = T[~T.enrolled]
    tab["fitbit_pct_non_enrollees"] = nonen.groupby("decile").fit.mean().mul(100).round(1)
    print("\n  by burden decile:")
    print(tab.to_string())
    _agg_csv(tab, "wear_study_by_decile.csv", index=True)

    print("\n  log-odds per SD of burden rank, age adjusted throughout:")
    out = []
    r_all = _burden_logit(T.fit.values, T.decile.values, T.age.values)
    r_non = _burden_logit(nonen.fit.values, nonen.decile.values, nonen.age.values)
    r_enr = _burden_logit(T.enrolled.values, T.decile.values, T.age.values)
    for label, res in [("Fitbit coverage, whole cohort", r_all),
                       ("Fitbit coverage, NON-ENROLLEES only", r_non),
                       ("wear-study enrollment itself", r_enr)]:
        if res is None:
            continue
        print(f"    {label:38s} {res['age_adjusted']:+.3f}  (95% CI {res['lo']:+.3f} to "
              f"{res['hi']:+.3f})  n {res['n']:,}, {res['rate']:.1%}")
        out.append({"outcome": label, "n": res["n"], "rate_pct": round(100 * res["rate"], 1),
                    "burden_logodds_unadjusted": round(res["unadjusted"], 4),
                    "burden_logodds_age_adjusted": round(res["age_adjusted"], 4),
                    "lo": round(res["lo"], 4), "hi": round(res["hi"], 4),
                    "survives": res["survives_age"]})

    if r_all is None or r_non is None:
        print("\n  one of the two models did not fit -- no verdict"); return
    c_all, c_non = r_all["age_adjusted"], r_non["age_adjusted"]
    if not r_non["survives_age"]:
        verdict = ("DISTRIBUTION EXPLAINS THE GRADIENT. Among the people the program did not hand a "
                   "device to, burden does not predict coverage.")
    elif abs(c_non) < 0.5 * abs(c_all):
        verdict = (f"DISTRIBUTION EXPLAINS PART OF IT. The gradient holds among non-enrollees but at "
                   f"{abs(c_non) / abs(c_all):.0%} of the whole-cohort coefficient.")
    else:
        verdict = ("DISTRIBUTION DOES NOT EXPLAIN IT. The gradient holds among the people the program "
                   "did not hand a device to, at the same size.")
    print(f"\n  -> {verdict}")
    if r_non["survives_age"] and abs(c_non) >= 0.5 * abs(c_all):
        print("  Both candidate explanations are now out. What is left is that more burdened")
        print("  participants took the wearable up more, which is a fact about the cohort and not a")
        print("  reason a design cannot proceed.")
    else:
        print("  A Fitbit design now has to say which people it is about. Coverage among the people the")
        print("  program did not equip is flat in burden and low, so the burdened people with Fitbit")
        print("  data are largely people the program equipped, and enrollment is itself graded on")
        print("  burden. Whether that is a confound depends on the design, and it is not settled here.")
    _agg_csv(pd.DataFrame(out + [{"outcome": "VERDICT", "verdict": verdict}]), "wear_study_check.csv")
    print("\n-> task_measure_profile.csv, task_reach_gradient.csv, wear_study_by_decile.csv,")
    print("   wear_study_check.csv")


# ---- cell 8: does a loaned device give the same kind of record as an owned one? --------------------
# THE LAST COVERAGE QUESTION, and it is here for one reason only. Cell 7 established that the whole
# burden gradient in Fitbit coverage is the wear study handing devices out, so provenance correlates
# with burden. Mean daily steps over a
# loaned study window and mean daily steps over three years of personal ownership are not the same
# quantity. If the two kinds of record have the same shape, provenance is a covariate and a
# whole-cohort design proceeds. If they do not, a design has to say which population it is about.
#
# ★ RECORD SHAPE ONLY. Not one behavioral value is read here -- no steps, no sleep minutes, no heart
# rate. Burden against a recorded measure IS the analysis, and it gets its objective and its
# definitions declared before it is run, not smuggled in as reconnaissance.
#
# Four things describe a record's shape and all four are dates: how many days carry data, the span
# from first day to last, the DUTY CYCLE (days with data divided by span, which is what separates a
# short dense loan from long sparse ownership), and where the record sits relative to the survey.
#
# ★ THE RULES, DECLARED HERE BEFORE THE RUN. Two kinds of record are ALIKE when their median days in
# the window and their median duty cycle each differ by less than SHAPE_TOL in relative terms, and the
# non-enrollee records are FLAT IN BURDEN when the median days in the window at decile 10 is within
# SHAPE_TOL of decile 1. Anything else is a difference, whatever it looks like on the eye.

SHAPE_TOL = 0.25
ANSWER_CACHE = "answer_dates_local.csv"          # person level. Stays on the Workbench. Never downloaded.


def _answer_dates(ids):
    """Each person's last answer date on the 151 items, cached because `observation` costs 7.8 GB.

    ★ PERSON LEVEL. The cache file stays on the Workbench and is never downloaded. It is
    written beside the other outputs so that a rerun of this cell is nearly free, which is the only
    reason it exists.
    """
    if os.path.exists(ANSWER_CACHE):
        d = pd.read_csv(ANSWER_CACHE)
        d["last_answer"] = pd.to_datetime(d.last_answer)
        print(f"  [answer dates: {len(d):,} read from {ANSWER_CACHE}, nothing scanned]")
    else:
        cids = pd.read_csv(CFG.items_csv, usecols=["concept_id"])["concept_id"].astype("int64").tolist()
        d = Q(f"""SELECT person_id, MAX(observation_datetime) AS last_answer
                  FROM `{CDR}.observation`
                  WHERE person_id IN UNNEST(@ids) AND observation_source_concept_id IN UNNEST(@cids)
                  GROUP BY person_id""", "answer dates", ids,
              [bigquery.ArrayQueryParameter("cids", "INT64", cids)])
        if d is None or not len(d):
            return None
        d["last_answer"] = pd.to_datetime(d.last_answer)
        if getattr(d.last_answer.dt, "tz", None) is not None:
            d["last_answer"] = d.last_answer.dt.tz_localize(None)
        d.to_csv(ANSWER_CACHE, index=False)
        print(f"  [answer dates: {len(d):,} cached to {ANSWER_CACHE} -- person level, do not download]")
    return d.set_index("person_id")["last_answer"]


def _shape_of(t, key, dc, ids, adate):
    """Per person, the shape of their record in table `t`. Dates only, no values."""
    ok = adate.notna().values
    return Q(f"""WITH anchor AS (
                   SELECT pid, adt FROM UNNEST(@pids) AS pid WITH OFFSET i
                   JOIN (SELECT adt, o2 FROM UNNEST(@adates) AS adt WITH OFFSET o2) ON i = o2)
                 SELECT a.pid AS person_id,
                        COUNT(DISTINCT DATE(v.{dc})) AS days,
                        COUNT(DISTINCT IF(DATE(v.{dc}) BETWEEN DATE_SUB(a.adt, INTERVAL 180 DAY)
                                          AND DATE_ADD(a.adt, INTERVAL 180 DAY),
                                          DATE(v.{dc}), NULL)) AS days_win,
                        DATE_DIFF(MAX(DATE(v.{dc})), MIN(DATE(v.{dc})), DAY) + 1 AS span,
                        DATE_DIFF(MIN(DATE(v.{dc})), a.adt, DAY) AS first_rel,
                        DATE_DIFF(MAX(DATE(v.{dc})), a.adt, DAY) AS last_rel
                 FROM anchor a JOIN `{CDR}.{t}` v ON v.{key} = a.pid
                 GROUP BY a.pid, a.adt""", f"{t} record shape", None,
             [bigquery.ArrayQueryParameter("pids", "INT64", [int(i) for i in ids.values[ok]]),
              bigquery.ArrayQueryParameter("adates", "DATE", list(adate.dt.date.values[ok]))])


def _shape_cached(t, key, dc, ids, adate):
    """The per-person shape frame, cached. ★ PERSON LEVEL: stays on the Workbench, never downloaded.

    Without the cache, re-reading the same dates to adjust for age would cost another 4.4 GB.
    The scan happens once.
    """
    f = f"shape_{t}_local.csv"
    if os.path.exists(f):
        print(f"  [{t}: shape read from {f}, nothing scanned]")
        return pd.read_csv(f)
    S = _shape_of(t, key, dc, ids, adate)
    if S is not None and len(S):
        S.to_csv(f, index=False)
        print(f"  [{t}: shape cached to {f} -- person level, do not download]")
    return S


def _reldiff(a, b):
    """Relative difference between two medians, against the larger. Nan-safe."""
    if not np.isfinite(a) or not np.isfinite(b) or max(abs(a), abs(b)) == 0:
        return float("nan")
    return abs(a - b) / max(abs(a), abs(b))


def cell8():
    ids = pd.read_csv(COHORT_CSV, usecols=[ID_COL])[ID_COL].astype("int64")
    dec = burden_deciles(ids)
    age = pd.read_csv(COHORT_CSV, usecols=[ID_COL, "age"]).set_index(ID_COL)["age"]
    C = _columns(["activity_summary", "sleep_daily_summary", "heart_rate_summary", "wear_study"])
    if C is None or not len(C):
        print("no column metadata returned"); return

    print("=" * 96)
    print("DOES A LOANED DEVICE GIVE THE SAME KIND OF RECORD AS AN OWNED ONE? Cell 7 found that the")
    print("whole burden gradient in Fitbit coverage is the wear study handing devices out, so who was")
    print("equipped now tracks burden. Dates only here. No steps, no sleep minutes, no heart rate:")
    print("burden against a recorded measure is the analysis, and it is not run as reconnaissance.\n")
    print(f"  ALIKE means both medians within {SHAPE_TOL:.0%}. FLAT means decile 10 within {SHAPE_TOL:.0%}")
    print("  of decile 1. Declared in the file above this function, before the run.\n")

    E = Q(f"""SELECT DISTINCT person_id FROM `{CDR}.wear_study`
              WHERE person_id IN UNNEST(@ids) AND wear_consent_start_date IS NOT NULL""",
          "wear_study enrollees", ids)
    if E is None:
        print("  wear_study roster unavailable -- the split cannot be made"); return
    enrolled = pd.Series(ids.isin(set(E.person_id.astype("int64"))).values, index=ids.values)
    adate = _answer_dates(ids)
    if adate is None:
        print("  answer dates unavailable -- the window cannot be defined"); return
    adate = adate.reindex(ids.values)

    rows, verdicts = [], []
    for t in ["activity_summary", "sleep_daily_summary", "heart_rate_summary"]:
        d = C[C.table_name == t]
        key = _key_of(d.column_name)
        dts = d[d.data_type.isin(["DATE", "DATETIME", "TIMESTAMP"])].column_name.tolist()
        if key is None or not dts:
            print(f"  {t}: key {key}, date columns {dts or 'none'} -- skipped\n"); continue
        S = _shape_cached(t, key, dts[0], ids, adate)
        if S is None or not len(S):
            continue
        S = S.set_index("person_id")
        S["duty"] = S.days / S.span.replace(0, np.nan)
        S["enrolled"] = enrolled.reindex(S.index).fillna(False).values
        S["decile"] = dec.reindex(S.index).values if dec is not None else np.nan
        S["age"] = age.reindex(S.index).values

        print("=" * 96)
        print(f"{t}   {len(S):,} people with a record and a known answer date\n")
        G = S.groupby("enrolled")[["days", "span", "duty", "days_win", "first_rel", "last_rel"]].median()
        G.insert(0, "people", S.groupby("enrolled").size())
        G.index = ["personal device" if not i else "wear study" for i in G.index]
        print("  medians by how the device was obtained:")
        print(G.round(2).to_string())
        print("  days = days carrying data, span = first to last day, duty = days/span,")
        print("  days_win = days within 180 of the answer, first_rel/last_rel = days from the answer")
        print("  to the first and last day of the record (negative is before the survey).")

        en, no = S[S.enrolled], S[~S.enrolled]
        if not len(en) or not len(no):
            print("  one side of the split is empty -- no verdict\n"); continue
        dw = _reldiff(en.days_win.median(), no.days_win.median())
        du = _reldiff(en.duty.median(), no.duty.median())
        alike = bool(dw < SHAPE_TOL and du < SHAPE_TOL)
        print(f"\n  days in window differ by {dw:.0%}, duty cycle by {du:.0%}  ->  "
              f"{'ALIKE' if alike else 'THEY DIFFER'}")

        flat, r10, adj = None, float("nan"), {}
        if dec is not None:
            byd = no.groupby("decile")[["days_win", "duty"]].median()
            byd.insert(0, "people", no.groupby("decile").size())
            print("\n  personal-device records by burden decile (the population with no program"
                  " selection):")
            print(byd.round(2).to_string())
            # written, not only printed: no number is copied off the Workbench by hand
            _agg_csv(byd.assign(table=t), f"fitbit_personal_by_decile_{t}.csv", index=True)
            if len(byd) >= 10 and byd.days_win.iloc[0]:
                r10 = _reldiff(byd.days_win.iloc[-1], byd.days_win.iloc[0])
                flat = bool(r10 < SHAPE_TOL)
                print(f"  decile 10 against decile 1: {r10:.0%}  ->  "
                      f"{'FLAT IN BURDEN' if flat else 'NOT FLAT'}")
                print("  Coverage being flat in burden does not make DENSITY flat. If the burdened own")
                print("  devices as often but wear them on fewer days, a measure is selected anyway.")

            # ★ That table is UNADJUSTED and age tracks burden, so it is the shape and not the answer,
            # exactly as the band table was for coverage. One outcome only, days in the window, which
            # is the quantity a design's precision rests on. Personal devices only, the population with
            # no program selection in it, which is the population the table above describes.
            print("\n  DAYS IN THE WINDOW on burden, per SD of burden rank, personal devices only:")
            for kind in ("mean", "median"):
                r = _burden_linear(no.days_win.values, no.decile.values, no.age.values, kind)
                if r is None:
                    continue
                print(f"    {kind:6s} {r['center']:6.1f} days | unadjusted {r['unadjusted']:+7.2f} | "
                      f"age adjusted {r['age_adjusted']:+7.2f} (95% CI {r['lo']:+.2f} to {r['hi']:+.2f})"
                      f" | age accounts for {r['share_explained_by_age_pct']:.0f}%"
                      f"{'' if r['converged'] else '  <- DID NOT CONVERGE, do not quote'}")
                adj[kind] = r
            if "median" in adj:
                said = ("THE DENSITY GRADIENT SURVIVES AGE" if adj["median"]["survives_age"]
                        else "THE DENSITY GRADIENT WAS AGE")
                print(f"    -> {said}, read on the median, which is the model")
                print("       that matches the table above.")
        for lab, sub in [("personal device", no), ("wear study", en)]:
            rows.append({"table": t, "provenance": lab, "people": int(len(sub)),
                         **{c: round(float(sub[c].median()), 3)
                            for c in ["days", "span", "duty", "days_win", "first_rel", "last_rel"]}})
        v = {"table": t, "days_win_reldiff": round(dw, 4), "duty_reldiff": round(du, 4),
             "alike": alike, "personal_d10_vs_d1_reldiff": round(r10, 4),
             "personal_flat_in_burden": flat}
        for kind, r in adj.items():
            v.update({f"days_win_per_sd_{kind}_unadjusted": round(r["unadjusted"], 3),
                      f"days_win_per_sd_{kind}_age_adjusted": round(r["age_adjusted"], 3),
                      f"days_win_per_sd_{kind}_lo": round(r["lo"], 3),
                      f"days_win_per_sd_{kind}_hi": round(r["hi"], 3),
                      f"days_win_per_sd_{kind}_age_share_pct": round(
                          r["share_explained_by_age_pct"], 1),
                      f"density_gradient_survives_age_{kind}": r["survives_age"],
                      f"converged_{kind}": r["converged"]})
        verdicts.append(v)
        print()

    if not rows:
        print("nothing to report"); return
    _agg_csv(pd.DataFrame(rows), "fitbit_record_shape.csv")
    _agg_csv(pd.DataFrame(verdicts), "fitbit_shape_check.csv")

    # TWO rules were declared and the summary answers BOTH, so a failed second rule is never hidden
    # behind a passed first one.
    print("=" * 96)
    ali = [v["table"] for v in verdicts if not v["alike"]]
    fla = [v["table"] for v in verdicts if v["personal_flat_in_burden"] is False]
    if not ali:
        print(f"PROVENANCE: all {len(verdicts)} tables ALIKE. A loaned record and an owned one have the")
        print("same shape, so provenance is a covariate and not a reason to split the population.")
    else:
        print(f"PROVENANCE: the two kinds of record DIFFER in {', '.join(ali)}. A measure read over a")
        print("loaned window and the same measure read over years of ownership are different")
        print("quantities, and provenance tracks burden, so a design has to say which population it is")
        print("about rather than pool them and adjust.")
    sur = [v["table"] for v in verdicts if v.get("density_gradient_survives_age_median")]
    if not fla:
        print("\nDENSITY: flat in burden among personal devices. Coverage and density agree.")
    elif not sur:
        print(f"\nDENSITY: the decile table falls in {', '.join(fla)}, but NOT ONCE AGE IS IN. The")
        print("gradient was age, the same way the banded coverage table was, and the raw table would")
        print("have declared a burden effect where the adjustment finds none. Nothing to answer.")
    else:
        print(f"\n★ DENSITY FALLS WITH BURDEN AND SURVIVES AGE in {', '.join(sur)}, among people the")
        print("program did not equip, so it is not program selection either. Coverage says the")
        print("burdened have a device as often. Density says they have far less data near the survey.")
        print("A measure averaged over the window is therefore estimated least precisely in exactly")
        print("the people the paper is about, and differentially. Wear time is itself behavior, so the")
        print("average runs over days they chose to wear it, which selects WITHIN a person as well as")
        print("between. A minimum-days rule fixes the first and worsens the second, because it removes")
        print("the burdened first. How much it removes at each threshold IS NOT COMPUTED HERE.")
        print("THIS, not provenance, is what a Fitbit design has to answer.")
    if sur and sur != fla:
        rest = [t for t in fla if t not in sur]
        if rest:
            print(f"  ({', '.join(rest)} failed the declared rule on raw medians but not after age.)")
    print("\nWHAT IS NOT IN THIS CELL, deliberately: any behavioral value. Burden against recorded")
    print("behavior is the analysis. It needs its outcome, its window and its adjustment set declared")
    print("first.")
    print("\n-> fitbit_record_shape.csv, fitbit_shape_check.csv  (aggregates only)")


# ---- cell 9: how variable is one person's sleep from night to night? ------------------------------
# WHY THIS. The argument that a thin Fitbit record can still carry a mean rests entirely on
# one quantity, the within-person night-to-night SD of sleep duration, and this cell measures it in
# this cohort rather than assuming it. Everything downstream follows from it: how many nights a
# person needs, whether the most burdened can clear that, and whether the minimum-nights rule costs the
# people the paper is about. It is measured here before any of that is fixed.
#
# ★ WHAT THIS CELL SPENDS. It reads sleep minutes, which no earlier cell did, and it reads the
# within-person SD against burden, because the question is whether precision is adequate FOR THE
# BURDENED and a pooled median cannot answer that. Sleep VARIABILITY is a psychiatric phenotype in its
# own right, so that regression is a design input here and CANNOT later be promoted to a finding on the
# strength of this printout. If it is ever to be reported, it is declared as an outcome with its own
# rule before a cell computes it again. The person MEAN against burden is the outcome of the sleep
# design and is deliberately NOT in this cell.
#
# ★ DECLARED BEFORE THE RUN. A night counts when it is the main sleep, falls within 180 days of the
# answer, and its duration lies between SLEEP_MIN_MIN and SLEEP_MAX_MIN. Per-person statistics are read
# on people with at least NIGHT_FLOOR such nights. The unfiltered figures are printed beside the
# filtered ones so the cost of the range is visible rather than assumed. A person mean is adequate at
# TARGET_RELIABILITY, and the cell reports the nights that takes and who has them, which answers the
# minimum-nights question from the data instead of from an assertion.

SLEEP_MIN_MIN, SLEEP_MAX_MIN = 60, 960          # a main sleep under 1 h or over 16 h is not one
NIGHT_FLOOR = 7                                 # nights needed before a person's own SD is read
TARGET_RELIABILITY = 0.90                       # measurement noise under a tenth of true variance


def cell9():
    ids = pd.read_csv(COHORT_CSV, usecols=[ID_COL])[ID_COL].astype("int64")
    dec = burden_deciles(ids)
    age = pd.read_csv(COHORT_CSV, usecols=[ID_COL, "age"]).set_index(ID_COL)["age"]
    print("=" * 96)
    print("HOW VARIABLE IS ONE PERSON'S SLEEP FROM NIGHT TO NIGHT? The case for reading a mean off a")
    print("thin record rests on this number and it has been my estimate, not this cohort's. Everything")
    print("downstream follows from it: nights needed, whether the burdened have them, and what a")
    print("minimum-nights rule costs. ★ The person MEAN against burden is the outcome of the sleep")
    print("design and is NOT in this cell.\n")
    print(f"  A night counts when it is the main sleep, falls within 180 days of the answer, and lasts")
    print(f"  {SLEEP_MIN_MIN} to {SLEEP_MAX_MIN} minutes. Per-person figures need {NIGHT_FLOOR} such")
    print(f"  nights. Adequate means reliability {TARGET_RELIABILITY:.2f}. All declared in the file.\n")

    M = Q(f"""SELECT CAST(is_main_sleep AS STRING) AS value, COUNT(*) AS rows_
              FROM `{CDR}.sleep_daily_summary` WHERE person_id IN UNNEST(@ids)
              GROUP BY 1 ORDER BY rows_ DESC""", "is_main_sleep values", ids)
    if M is not None and len(M):
        print("  what is_main_sleep holds, so the filter is taken from the data and not from memory:")
        print(M.to_string(index=False))
        print()
    MAIN = "LOWER(CAST(v.is_main_sleep AS STRING)) IN ('true', '1', 't', 'yes')"

    adate = _answer_dates(ids)
    if adate is None:
        print("  answer dates unavailable"); return
    adate = adate.reindex(ids.values)
    ok = adate.notna().values
    win = (f"DATE(v.sleep_date) BETWEEN DATE_SUB(a.adt, INTERVAL 180 DAY) "
           f"AND DATE_ADD(a.adt, INTERVAL 180 DAY)")
    anchor = """WITH anchor AS (
                  SELECT pid, adt FROM UNNEST(@pids) AS pid WITH OFFSET i
                  JOIN (SELECT adt, o2 FROM UNNEST(@adates) AS adt WITH OFFSET o2) ON i = o2)"""
    params = [bigquery.ArrayQueryParameter("pids", "INT64", [int(i) for i in ids.values[ok]]),
              bigquery.ArrayQueryParameter("adates", "DATE", list(adate.dt.date.values[ok]))]

    D = Q(f"""{anchor}
              SELECT COUNT(*) AS nights, COUNT(DISTINCT a.pid) AS people,
                     COUNTIF(v.minute_asleep IS NULL) AS n_null,
                     COUNTIF(v.minute_asleep < {SLEEP_MIN_MIN}) AS n_short,
                     COUNTIF(v.minute_asleep > {SLEEP_MAX_MIN}) AS n_long,
                     APPROX_QUANTILES(v.minute_asleep, 20) AS q
              FROM anchor a JOIN `{CDR}.sleep_daily_summary` v ON v.person_id = a.pid
              WHERE {MAIN} AND {win}""", "nightly distribution", None, params)
    if D is not None and len(D):
        r = D.iloc[0]
        q = list(r.q) if r.q is not None and len(r.q) == 21 else []
        print(f"  {int(r.nights):,} main-sleep nights in the window, {int(r.people):,} people")
        if q:
            print(f"  minutes asleep: min {q[0]:.0f} | p5 {q[1]:.0f} | p25 {q[5]:.0f} | "
                  f"median {q[10]:.0f} | p75 {q[15]:.0f} | p95 {q[19]:.0f} | max {q[20]:.0f}")
        tot = max(int(r.nights), 1)
        print(f"  outside the declared range: {int(r.n_short):,} under {SLEEP_MIN_MIN} min "
              f"({100 * int(r.n_short) / tot:.1f}%), {int(r.n_long):,} over {SLEEP_MAX_MIN} "
              f"({100 * int(r.n_long) / tot:.2f}%), {int(r.n_null):,} null\n")

    cache = "sleep_person_local.csv"             # person level. Workbench only. Never downloaded.
    if os.path.exists(cache):
        P = pd.read_csv(cache)
        print(f"  [per-person sleep read from {cache}, nothing scanned]")
    else:
        P = Q(f"""{anchor}
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
                  WHERE {MAIN} AND {win}
                  GROUP BY a.pid""", "per-person sleep", None, params)
        if P is None or not len(P):
            print("  no per-person rows returned"); return
        P.to_csv(cache, index=False)
        print(f"  [per-person sleep cached to {cache} -- person level, do not download]")

    P = P.set_index("person_id")
    P["decile"] = dec.reindex(P.index).values if dec is not None else np.nan
    P["age"] = age.reindex(P.index).values
    K = P[(P.nights >= NIGHT_FLOOR) & P.sd_min.notna()]
    print(f"\n  {len(P):,} people have a main-sleep night in the window, {len(K):,} have at least "
          f"{NIGHT_FLOOR}")
    if len(K) < 100:
        print("  too few to read"); return

    sd_q = K.sd_min.quantile([0.25, 0.5, 0.75])
    sda_q = K.sd_all.dropna().quantile([0.25, 0.5, 0.75])
    print(f"\n  WITHIN-PERSON SD of minutes asleep: median {sd_q[0.5]:.1f} min "
          f"(q1 {sd_q[0.25]:.1f}, q3 {sd_q[0.75]:.1f})")
    print(f"  the same without the duration range: median {sda_q[0.5]:.1f} min "
          f"(q1 {sda_q[0.25]:.1f}, q3 {sda_q[0.75]:.1f})   <- what the range costs")

    # Pooled within variance, and between variance CORRECTED for sampling noise. Without the
    # correction the observed spread of person means includes the noise and reliability is overstated.
    w = (K.nights - 1).clip(lower=0)
    s2w = float((w * K.sd_min ** 2).sum() / max(w.sum(), 1))
    obs = float(K.mean_min.var(ddof=1))
    noise = float((K.sd_min ** 2 / K.nights).mean())
    s2b = max(obs - noise, 1e-9)
    print(f"\n  pooled within-person variance {s2w:8.1f}  (SD {np.sqrt(s2w):.1f} min)")
    print(f"  spread of person means         {obs:8.1f}  (SD {np.sqrt(obs):.1f} min), of which")
    print(f"  sampling noise                 {noise:8.1f}, leaving true between-person variance")
    print(f"                                 {s2b:8.1f}  (SD {np.sqrt(s2b):.1f} min)")

    print("\n  what a person mean costs, at the pooled within-person SD:")
    print(f"  {'nights':>8} {'SE of mean':>12} {'reliability':>12}")
    for nn in [7, 14, 30, 60, 120, 240]:
        rel = s2b / (s2b + s2w / nn)
        print(f"  {nn:>8} {np.sqrt(s2w / nn):>9.1f} min {rel:>12.3f}")
    need = int(np.ceil((TARGET_RELIABILITY / (1 - TARGET_RELIABILITY)) * s2w / s2b))
    print(f"\n  -> reliability {TARGET_RELIABILITY:.2f} needs {need} nights.")

    if dec is not None:
        T = K.dropna(subset=["decile"])
        tab = T.groupby("decile").agg(people=("nights", "size"),
                                      median_nights=("nights", "median"),
                                      median_sd=("sd_min", "median"))
        tab["pct_with_enough"] = T.groupby("decile").nights.apply(
            lambda s: round(100 * float((s >= need).mean()), 1))
        print(f"\n  by burden decile, among people with at least {NIGHT_FLOOR} nights:")
        print(tab.round(1).to_string())
        lo, hi = tab.pct_with_enough.iloc[0], tab.pct_with_enough.iloc[-1]
        print(f"  share clearing {need} nights: {lo:.1f}% at decile 1 against {hi:.1f}% at decile 10")
        print("  ★ That last line is the minimum-nights rule's real cost, measured rather than guessed.")

        print("\n  WITHIN-PERSON SD on burden, per SD of burden rank (a DESIGN INPUT, see the header:")
        print("  it may not be promoted to a finding on the strength of this printout):")
        for kind in ("mean", "median"):
            r = _burden_linear(T.sd_min.values, T.decile.values, T.age.values, kind)
            if r is None:
                continue
            print(f"    {kind:6s} {r['center']:6.1f} min | unadjusted {r['unadjusted']:+7.2f} | "
                  f"age adjusted {r['age_adjusted']:+7.2f} (95% CI {r['lo']:+.2f} to {r['hi']:+.2f})"
                  f"{'' if r['converged'] else '  <- DID NOT CONVERGE, do not quote'}")
        _agg_csv(tab, "sleep_nights_by_decile.csv", index=True)

    _agg_csv(pd.DataFrame([{"n_people": len(K), "night_floor": NIGHT_FLOOR,
                   "within_sd_median": round(float(sd_q[0.5]), 2),
                   "within_sd_median_unfiltered": round(float(sda_q[0.5]), 2),
                   "pooled_within_var": round(s2w, 2), "observed_between_var": round(obs, 2),
                   "sampling_noise_var": round(noise, 2), "true_between_var": round(s2b, 2),
                   "target_reliability": TARGET_RELIABILITY, "nights_needed": need}]), "sleep_reliability.csv")
    print("\n-> sleep_reliability.csv, sleep_nights_by_decile.csv  (aggregates only)")


# ---- cell 10: the sleep-variability association where the nights were not the participant's choice --
# WHY. Cell 9 reads within-person sleep variability against burden while establishing a measurement
# property rather than while testing it.
# To be reported, the association has to survive the objection cell
# 8 already named: a person wearing an optional device chooses which nights we see, and if the burdened
# skip their worst nights, or only their worst, an observed SD is a biased sample of their nights.
#
# THE TEST. The wear study EQUIPPED about 12,400 of these people for a defined study period, so their
# observed nights are far less their own choice than an owner's are. The same estimand is run there.
# It is not a held-out sample, since cell 9's pooled figure included them, and it is not called one: it
# answers the observation objection, not the exploratory one. Cell 9's result is exploratory and stays
# labeled as exploratory whatever this returns.
#
# ★ THE RULE, DECLARED HERE BEFORE THE RUN. The association CONFIRMS in the protocol-observed subsample
# when the wear-study estimate carries the same sign, its 95% interval excludes zero, and its point
# estimate is at least CONFIRM_FRACTION of the pooled estimate. Otherwise it does not confirm, and that
# is reported instead. The median model is primary, as it was in cells 8 and 9, because an SD is itself
# right skewed.
#
# ★ TWO SENSITIVITIES, DECLARED WITH IT.
#   (1) NIGHTS. A sample SD is biased downward at small n and the burdened have fewer nights, so the
#       bias runs against the finding rather than making it. log(nights) is adjusted anyway, because a
#       reviewer will ask and the answer should already be in the table.
#   (2) THE COVARIATE SET. Race and ethnicity, income, education, employment and insurance, the set of
#       covariates.py. Employment is the one that matters here: shift work produces irregular sleep and tracks
#       burden. ★ This model is a SENSITIVITY AND NOT THE PRIMARY, because income and employment may lie
#       ON THE PATH from burden rather than confound it, and adjusting away a mediator understates the
#       thing being estimated. It is reported as what survives the most hostile adjustment available,
#       not as the better estimate.

CONFIRM_FRACTION = 0.5
# The 5 categorical covariates live in covariates_per_person.csv. SEX IS NOT AMONG THEM: it comes from
# the response build already coded male 1 / female 0, and it is added to the design separately, or the
# model would adjust for only 5 of the 6 declared covariates.
COVARIATE_SET = ["race_ethnicity", "income", "education", "employment", "insurance"]


def _enrollees(ids):
    """The wear-study roster, cached. Person level, Workbench only, never downloaded."""
    f = "wear_enrollees_local.csv"
    if os.path.exists(f):
        d = pd.read_csv(f)
        print(f"  [wear-study roster read from {f}, nothing scanned]")
    else:
        d = Q(f"""SELECT DISTINCT person_id FROM `{CDR}.wear_study`
                  WHERE person_id IN UNNEST(@ids) AND wear_consent_start_date IS NOT NULL""",
              "wear_study enrollees", ids)
        if d is None or not len(d):
            return None
        d.to_csv(f, index=False)
        print(f"  [wear-study roster cached to {f} -- person level, do not download]")
    return set(d.person_id.astype("int64"))


def _dummies(d, columns):
    """Categorical columns to dummies against the modal level."""
    X = pd.DataFrame(index=d.index)
    for c in [c for c in columns if c in d.columns]:
        s = d[c].astype(str)
        if s.nunique() < 2:
            continue
        ref = s.value_counts().idxmax()
        for lv in sorted(s.unique()):
            if lv != ref:
                X[f"{c}={lv}"] = (s == lv).astype(float)
    return X


def cell10():
    cache = "sleep_person_local.csv"
    if not os.path.exists(cache):
        print(f"{cache} not found. Run --cells 9 first: this cell adds no scan of its own."); return
    ids = pd.read_csv(COHORT_CSV, usecols=[ID_COL])[ID_COL].astype("int64")
    dec = burden_deciles(ids)
    B = pd.read_csv(COHORT_CSV, usecols=[ID_COL, "age", "sex"]).set_index(ID_COL)
    age = B["age"]

    print("=" * 96)
    print("DOES THE SLEEP-VARIABILITY ASSOCIATION HOLD WHERE THE NIGHTS WERE NOT THE PARTICIPANT'S")
    print("CHOICE? Cell 9 found it while establishing a measurement property, so it is exploratory and")
    print("stays labeled so. What it can be defended against is the observation objection: an owner")
    print("chooses which nights we see. The wear study equipped people for a study period, so theirs")
    print("are far less chosen, and the same estimand is run there.\n")
    print(f"  CONFIRMS when the wear-study estimate has the same sign, its interval excludes zero, and")
    print(f"  it is at least {CONFIRM_FRACTION:.0%} of the pooled estimate. Declared in the file.\n")

    enrolled = _enrollees(ids)
    if enrolled is None:
        print("  wear-study roster unavailable"); return
    P = pd.read_csv(cache).set_index("person_id")
    P = P[(P.nights >= NIGHT_FLOOR) & P.sd_min.notna()].copy()
    P["decile"] = dec.reindex(P.index).values if dec is not None else np.nan
    P["age"] = age.reindex(P.index).values
    P["wear"] = P.index.isin(enrolled)
    P["log_nights"] = np.log(P.nights.clip(lower=1))
    P = P.dropna(subset=["decile", "age"])

    cov = os.path.join(CFG.out_dir, "covariates_per_person.csv")
    CV = None
    if os.path.exists(cov):
        C = pd.read_csv(cov)
        C[ID_COL] = pd.to_numeric(C[ID_COL], errors="coerce")
        C = C.dropna(subset=[ID_COL]).set_index(C[ID_COL].astype("int64"))
        CV = _dummies(C.reindex(P.index), COVARIATE_SET)
        CV["sex"] = pd.to_numeric(B["sex"].reindex(P.index), errors="coerce").values
        have = sorted({c.split("=")[0] for c in CV.columns})
        missing = [c for c in COVARIATE_SET if c not in have]
        print(f"  covariates adjusted: {', '.join(have) if have else 'none'} "
              f"({len(CV.columns)} terms)")
        if missing:
            print(f"  ★ DECLARED BUT NOT FOUND, so NOT adjusted: {', '.join(missing)}")
    else:
        print(f"  [covariates_per_person.csv not beside us: the covariate sensitivity is skipped]")

    W, O = P[P.wear], P[~P.wear]
    print(f"\n  {len(P):,} people with at least {NIGHT_FLOOR} nights | wear study {len(W):,} | "
          f"personal device {len(O):,}")
    tab = P.groupby(["wear", "decile"]).agg(people=("nights", "size"),
                                            median_nights=("nights", "median"),
                                            median_sd=("sd_min", "median"))
    print("\n  nights and within-person SD by decile, so neither subsample is read blind:")
    print(tab.round(1).to_string())

    rows, est = [], {}
    for label, sub, ex in [("pooled, age adjusted", P, None),
                           ("WEAR STUDY, age adjusted", W, None),
                           ("personal device, age adjusted", O, None),
                           ("pooled, + log nights", P, ["log_nights"]),
                           ("WEAR STUDY, + log nights", W, ["log_nights"])]:
        extra = sub[ex] if ex else None
        for kind in ("median", "mean"):
            r = _burden_linear(sub.sd_min.values, sub.decile.values, sub.age.values, kind, extra)
            if r is None:
                continue
            if kind == "median":
                est[label] = r
            rows.append({"model": label, "kind": kind, "n": r["n"],
                         "center": round(r["center"], 2),
                         "unadjusted": round(r["unadjusted"], 3),
                         "adjusted": round(r["age_adjusted"], 3),
                         "lo": round(r["lo"], 3), "hi": round(r["hi"], 3),
                         "converged": r["converged"]})

    if CV is not None and len(CV.columns):
        for label, sub in [("pooled, + covariate set", P), ("WEAR STUDY, + covariate set", W)]:
            extra = pd.concat([sub[["log_nights"]], CV.reindex(sub.index)], axis=1)
            for kind in ("median", "mean"):
                r = _burden_linear(sub.sd_min.values, sub.decile.values, sub.age.values, kind, extra)
                if r is None:
                    continue
                if kind == "median":
                    est[label] = r
                rows.append({"model": label, "kind": kind, "n": r["n"],
                             "center": round(r["center"], 2),
                             "unadjusted": round(r["unadjusted"], 3),
                             "adjusted": round(r["age_adjusted"], 3),
                             "lo": round(r["lo"], 3), "hi": round(r["hi"], 3),
                             "converged": r["converged"]})

    print("\n  minutes of within-person SD per SD of burden rank. Median model, the declared primary:")
    for label, r in est.items():
        flag = "" if r["converged"] else "  <- DID NOT CONVERGE, do not quote"
        print(f"    {label:32s} n {r['n']:6,} | {r['age_adjusted']:+7.2f} "
              f"(95% CI {r['lo']:+.2f} to {r['hi']:+.2f}){flag}")
    if rows:
        _agg_csv(pd.DataFrame(rows), "sleep_variability_replication.csv")

    pooled = est.get("pooled, age adjusted")
    wear = est.get("WEAR STUDY, age adjusted")
    if pooled is None or wear is None:
        print("\n  one of the two primary models did not fit -- no verdict"); return
    same = np.sign(wear["age_adjusted"]) == np.sign(pooled["age_adjusted"])
    excl = wear["lo"] > 0 or wear["hi"] < 0
    big = abs(wear["age_adjusted"]) >= CONFIRM_FRACTION * abs(pooled["age_adjusted"])
    if same and excl and big:
        verdict = ("CONFIRMED IN THE PROTOCOL-OBSERVED SUBSAMPLE. The association is not an artifact of "
                   "owners choosing which nights we see.")
    elif same and excl:
        verdict = (f"PRESENT BUT SMALLER among the people the program equipped, "
                   f"{abs(wear['age_adjusted']) / abs(pooled['age_adjusted']):.0%} of the pooled "
                   f"estimate, so part of the pooled figure is which nights owners show us.")
    else:
        verdict = ("NOT CONFIRMED. Where the nights were not the participant's choice the association "
                   "does not hold, so the pooled figure is about what owners choose to record.")
    print(f"\n  -> {verdict}")
    print("  ★ Whatever this says, cell 9's result was found in a measurement check and is reported as")
    print("  exploratory. This answers the observation objection, not that one.")
    if "WEAR STUDY, + covariate set" in est:
        c = est["WEAR STUDY, + covariate set"]
        print(f"\n  Under the most hostile adjustment available, employment and income included, the")
        print(f"  wear-study estimate is {c['age_adjusted']:+.2f} ({c['lo']:+.2f} to {c['hi']:+.2f}).")
        print("  ★ Read that as what survives, NOT as the better estimate: income and employment may")
        print("  lie on the path from burden rather than confound it, and adjusting away a mediator")
        print("  understates what is being estimated.")
    _agg_csv(pd.DataFrame([{"pooled": round(pooled["age_adjusted"], 3),
                   "wear_study": round(wear["age_adjusted"], 3),
                   "wear_lo": round(wear["lo"], 3), "wear_hi": round(wear["hi"], 3),
                   "fraction_of_pooled": round(abs(wear["age_adjusted"]) /
                                               abs(pooled["age_adjusted"]), 3),
                   "confirm_fraction": CONFIRM_FRACTION, "confirmed": bool(same and excl and big),
                   "verdict": verdict}]), "sleep_variability_check.csv")
    print("\n-> sleep_variability_replication.csv, sleep_variability_check.csv  (aggregates only)")


# ---- cell 11: the sleep design itself -- mean sleep duration against burden -----------------------
# THIS IS THE ANALYSIS THE WHOLE FITBIT THREAD WAS FOR, and it is the one part of it that was specified
# before any sleep value was read. Every definition below was fixed in advance: the outcome, the
# window, the night floor, the covariate set, the provenance adjustment and the wear-study replication.
# Cell 9 deliberately leaves the person MEAN out for exactly this reason. Cell 9's variability result is
# exploratory and stays so. THIS ONE IS NOT.
#
# ★ THE NIGHT FLOOR IS 14, NOT 7. Cell 9 measures what a person mean costs: reliability 0.90 takes 14
# nights, so the floor is set where the measurement says rather than where a round number would.
#
# ★ THE PRIMARY MODEL IS LEAST SQUARES HERE, WHICH DEPARTS FROM CELLS 8 TO 10, and the reason is
# declared rather than discovered: those outcomes were bounded or right skewed (days of wear, an SD), so
# a median regression was the honest summary. A person's mean sleep duration is approximately symmetric
# across people, and the mean is the quantity anyone would want. The median model is reported beside it
# as a check, and if the two disagree that disagreement is the result.
#
# ★ NO DIRECTION IS PREDICTED. Short sleep and long sleep are both clinically plausible under high
# burden, so a one-sided prediction would be false precision. The test is two-sided.
#
# ★ WHAT COUNTS AS A FINDING, declared here before the run. An association is present when the
# provenance-adjusted estimate's 95 percent interval excludes zero AND it replicates in the
# protocol-observed wear-study subsample under the same CONFIRM_FRACTION rule cell 10 used. Anything
# else is reported as what it is, including a null, which is a perfectly good answer to a question that
# was asked properly.

MEAN_NIGHT_FLOOR = 14


def cell11():
    cache = "sleep_person_local.csv"
    if not os.path.exists(cache):
        print(f"{cache} not found. Run --cells 9 first: this cell adds no scan of its own."); return
    ids = pd.read_csv(COHORT_CSV, usecols=[ID_COL])[ID_COL].astype("int64")
    dec = burden_deciles(ids)
    B = pd.read_csv(COHORT_CSV, usecols=[ID_COL, "age", "sex"]).set_index(ID_COL)

    print("=" * 96)
    print("MEAN SLEEP DURATION AGAINST BURDEN. The analysis the Fitbit thread was for, and the part of")
    print("it specified before any sleep value was read. Cell 9's variability finding is exploratory")
    print("because it was born in a measurement check. THIS ONE IS NOT, and the difference is the")
    print("whole reason the mean was kept out of cell 9.\n")
    print(f"  Declared: mean minutes asleep on main-sleep nights within 180 days of the answer, people")
    print(f"  with at least {MEAN_NIGHT_FLOOR} such nights (reliability {TARGET_RELIABILITY:.2f} takes")
    print(f"  that many), least squares primary, two-sided, provenance adjusted, replicated in the")
    print(f"  wear study at {CONFIRM_FRACTION:.0%} of the pooled estimate.\n")

    enrolled = _enrollees(ids)
    if enrolled is None:
        print("  wear-study roster unavailable"); return
    P = pd.read_csv(cache).set_index("person_id")
    P = P[(P.nights >= MEAN_NIGHT_FLOOR) & P.mean_min.notna()].copy()
    P["decile"] = dec.reindex(P.index).values if dec is not None else np.nan
    P["age"] = B["age"].reindex(P.index).values
    P["wear"] = P.index.isin(enrolled).astype(float)
    P["log_nights"] = np.log(P.nights.clip(lower=1))
    P = P.dropna(subset=["decile", "age"])

    CV = None
    cov = os.path.join(CFG.out_dir, "covariates_per_person.csv")
    if os.path.exists(cov):
        C = pd.read_csv(cov)
        C[ID_COL] = pd.to_numeric(C[ID_COL], errors="coerce")
        C = C.dropna(subset=[ID_COL]).set_index(C[ID_COL].astype("int64"))
        CV = _dummies(C.reindex(P.index), COVARIATE_SET)
        CV["sex"] = pd.to_numeric(B["sex"].reindex(P.index), errors="coerce").values
        have = sorted({c.split("=")[0] for c in CV.columns})
        miss = [c for c in COVARIATE_SET if c not in have]
        print(f"  covariates adjusted: {', '.join(have)} ({len(CV.columns)} terms)")
        if miss:
            print(f"  ★ DECLARED BUT NOT FOUND, so NOT adjusted: {', '.join(miss)}")
    W = P[P.wear > 0]
    print(f"\n  {len(P):,} people with at least {MEAN_NIGHT_FLOOR} nights | wear study {len(W):,} | "
          f"personal device {len(P) - len(W):,}")

    if dec is not None:
        tab = P.groupby("decile").agg(people=("nights", "size"),
                                      median_nights=("nights", "median"),
                                      mean_minutes=("mean_min", "mean"),
                                      median_minutes=("mean_min", "median"))
        print("\n  mean minutes asleep by burden decile, unadjusted, for shape only:")
        print(tab.round(1).to_string())
        _agg_csv(tab, "sleep_duration_by_decile.csv", index=True)

    rows, est = [], {}
    plans = [("pooled, age + provenance", P, ["wear"]),
             ("WEAR STUDY, age adjusted", W, None),
             ("pooled, + log nights", P, ["wear", "log_nights"]),
             ("WEAR STUDY, + log nights", W, ["log_nights"])]
    if CV is not None and len(CV.columns):
        plans += [("pooled, + covariate set", P, ["wear", "log_nights"]),
                  ("WEAR STUDY, + covariate set", W, ["log_nights"])]
    for i, (label, sub, ex) in enumerate(plans):
        extra = sub[ex].copy() if ex else None
        if "covariate set" in label:
            extra = pd.concat([extra, CV.reindex(sub.index)], axis=1)
        for kind in ("mean", "median"):
            r = _burden_linear(sub.mean_min.values, sub.decile.values, sub.age.values, kind, extra)
            if r is None:
                continue
            if kind == "mean":
                est[label] = r
            rows.append({"model": label, "kind": kind, "n": r["n"],
                         "center": round(r["center"], 2), "unadjusted": round(r["unadjusted"], 3),
                         "adjusted": round(r["age_adjusted"], 3), "lo": round(r["lo"], 3),
                         "hi": round(r["hi"], 3), "converged": r["converged"]})

    print("\n  minutes of mean sleep per SD of burden rank. Least squares, the declared primary:")
    for label, r in est.items():
        flag = "" if r["converged"] else "  <- DID NOT CONVERGE, do not quote"
        print(f"    {label:32s} n {r['n']:6,} | {r['age_adjusted']:+7.2f} "
              f"(95% CI {r['lo']:+.2f} to {r['hi']:+.2f}){flag}")
    med = {r["model"]: r for r in rows if r["kind"] == "median"}
    if "pooled, age + provenance" in med:
        m = med["pooled, age + provenance"]
        print(f"    {'(median model, the check)':32s} {'':8s} | {m['adjusted']:+7.2f} "
              f"(95% CI {m['lo']:+.2f} to {m['hi']:+.2f})")
    if rows:
        _agg_csv(pd.DataFrame(rows), "sleep_duration_models.csv")

    pooled, wear = est.get("pooled, age + provenance"), est.get("WEAR STUDY, age adjusted")
    if pooled is None or wear is None:
        print("\n  one of the two primary models did not fit -- no verdict"); return
    present = pooled["lo"] > 0 or pooled["hi"] < 0
    same = np.sign(wear["age_adjusted"]) == np.sign(pooled["age_adjusted"])
    excl = wear["lo"] > 0 or wear["hi"] < 0
    big = abs(wear["age_adjusted"]) >= CONFIRM_FRACTION * abs(pooled["age_adjusted"])
    if present and same and excl and big:
        verdict = (f"BURDEN IS ASSOCIATED WITH MEAN SLEEP DURATION, {pooled['age_adjusted']:+.2f} "
                   f"minutes per SD, and it replicates where the nights were not chosen.")
    elif present:
        verdict = ("PRESENT IN THE WHOLE SAMPLE BUT NOT REPLICATED under the declared rule, so it is "
                   "not reported as established.")
    else:
        verdict = ("NO ASSOCIATION WITH MEAN SLEEP DURATION. The interval covers zero. This was asked "
                   "properly and a null is the answer, not a failure.")
    print(f"\n  -> {verdict}")
    span = 3.1                                   # decile 1 to decile 10 is about this many SDs of rank
    print(f"  Across the burden range, about {pooled['age_adjusted'] * span:+.0f} minutes a night.")
    print("  ★ Compare with the variability result, which is EXPLORATORY. If duration is")
    print("  null and variability is not, that is a finding about WHICH aspect of sleep tracks burden,")
    print("  and it is a more interesting paper than either number alone.")
    _agg_csv(pd.DataFrame([{"pooled": round(pooled["age_adjusted"], 3), "lo": round(pooled["lo"], 3),
                   "hi": round(pooled["hi"], 3), "wear_study": round(wear["age_adjusted"], 3),
                   "wear_lo": round(wear["lo"], 3), "wear_hi": round(wear["hi"], 3),
                   "night_floor": MEAN_NIGHT_FLOOR, "prespecified": True,
                   "replicated": bool(same and excl and big), "verdict": verdict}]), "sleep_duration_check.csv")
    print("\n-> sleep_duration_models.csv, sleep_duration_by_decile.csv, sleep_duration_check.csv")



# ---- cell 12: are the two sleep findings one thing or two? ----------------------------------------
# Cell 11 reads mean sleep duration against burden (prespecified) and cell 9 reads within-person
# night-to-night SD against burden (exploratory), in the same window, on the same nights of the same
# people. Before either is written up, they have to be shown to be two findings rather than one seen
# twice.
#
# WHY IT IS NOT OBVIOUS. The mean and the SD of a bounded positive quantity are usually correlated, and
# our own filter bounds it at SLEEP_MIN_MIN and SLEEP_MAX_MIN. Someone who sleeps briefly has less room
# to vary downward. Someone whose sleep is wildly variable cannot have a high mean without hitting the
# ceiling. So a single underlying shift could produce both numbers, and a reviewer will ask whether the
# variability finding is short sleep wearing a different hat.
#
# ★ THE RULE, DECLARED BEFORE THE RUN. The two are INDEPENDENT FINDINGS when each survives adjustment
# for the other: interval still excluding zero AND the point estimate at least INDEPENDENT_FRACTION of
# its own unadjusted value. If exactly one collapses, the surviving one is the finding and the other is
# its shadow. If both collapse, they are one thing measured two ways and neither is reported alone.
#
# ★ A PRECONDITION THE RULE NEEDS, found by building a world where the SD
# was a deterministic function of the mean. Mutual adjustment can only attribute primacy when the two
# measures are not collinear. Under near-perfect correlation BOTH coefficients collapse, because each
# adjustment removes the same information twice, and "both collapsed" then means the question is
# UNANSWERABLE rather than that the answer is one construct. So the correlation is tested first at
# COLLINEAR_MAX and the cell refuses to issue a substantive verdict above it.
# Both are adjusted for age, provenance and the other measure, standardized so the coefficients are
# comparable, and the wear-study subsample is read beside the pooled one as everywhere else.

INDEPENDENT_FRACTION = 0.5
COLLINEAR_MAX = 0.80        # above this, mutual adjustment cannot attribute primacy at all


def cell12():
    cache = "sleep_person_local.csv"
    if not os.path.exists(cache):
        print(f"{cache} not found. Run --cells 9 first."); return
    ids = pd.read_csv(COHORT_CSV, usecols=[ID_COL])[ID_COL].astype("int64")
    dec = burden_deciles(ids)
    B = pd.read_csv(COHORT_CSV, usecols=[ID_COL, "age", "sex"]).set_index(ID_COL)

    print("=" * 96)
    print("ARE THE TWO SLEEP FINDINGS ONE THING OR TWO? Duration and variability are both read against")
    print("burden, on the same nights of the same people. The mean and the SD of a bounded quantity")
    print("are usually correlated, so a single underlying shift could produce both. Each is therefore")
    print("adjusted for the other.\n")
    print(f"  INDEPENDENT when each survives the other: interval still excluding zero AND at least")
    print(f"  {INDEPENDENT_FRACTION:.0%} of its own unadjusted estimate. Declared in the file.\n")

    enrolled = _enrollees(ids)
    if enrolled is None:
        print("  wear-study roster unavailable"); return
    P = pd.read_csv(cache).set_index("person_id")
    P = P[(P.nights >= MEAN_NIGHT_FLOOR) & P.mean_min.notna() & P.sd_min.notna()].copy()
    P["decile"] = dec.reindex(P.index).values if dec is not None else np.nan
    P["age"] = B["age"].reindex(P.index).values
    P["wear"] = P.index.isin(enrolled).astype(float)
    P = P.dropna(subset=["decile", "age"])
    P["mean_z"] = (P.mean_min - P.mean_min.mean()) / P.mean_min.std()
    P["sd_z"] = (P.sd_min - P.sd_min.mean()) / P.sd_min.std()

    r = float(np.corrcoef(P.mean_min, P.sd_min)[0, 1])
    print(f"  {len(P):,} people. Correlation between a person's mean sleep and their own night-to-night")
    print(f"  SD: r = {r:+.3f}. That is the number the whole question turns on.\n")

    W = P[P.wear > 0]
    out, est = [], {}
    for pop, sub in [("pooled", P), ("WEAR STUDY", W)]:
        base = ["wear"] if pop == "pooled" else []
        for name, ycol, other in [("duration", "mean_min", "sd_z"), ("variability", "sd_min", "mean_z")]:
            for adj in (False, True):
                cols = base + ([other] if adj else [])
                extra = sub[cols] if cols else None
                rr = _burden_linear(sub[ycol].values, sub.decile.values, sub.age.values,
                                    "mean" if name == "duration" else "median", extra)
                if rr is None:
                    continue
                est[(pop, name, adj)] = rr
                out.append({"population": pop, "outcome": name, "adjusted_for_other": adj,
                            "n": rr["n"], "estimate": round(rr["age_adjusted"], 3),
                            "lo": round(rr["lo"], 3), "hi": round(rr["hi"], 3),
                            "converged": rr["converged"]})

    print("  minutes per SD of burden rank, before and after adjusting for the other measure:")
    for pop in ("pooled", "WEAR STUDY"):
        for name in ("duration", "variability"):
            a, b = est.get((pop, name, False)), est.get((pop, name, True))
            if a is None or b is None:
                continue
            frac = abs(b["age_adjusted"]) / abs(a["age_adjusted"]) if a["age_adjusted"] else float("nan")
            print(f"    {pop:11s} {name:12s} {a['age_adjusted']:+7.2f} -> {b['age_adjusted']:+7.2f} "
                  f"({b['lo']:+.2f} to {b['hi']:+.2f})  keeps {frac:.0%}")
    if out:
        _agg_csv(pd.DataFrame(out), "sleep_two_findings.csv")

    def survives(pop, name):
        a, b = est.get((pop, name, False)), est.get((pop, name, True))
        if a is None or b is None:
            return None
        return bool((b["lo"] > 0 or b["hi"] < 0)
                    and abs(b["age_adjusted"]) >= INDEPENDENT_FRACTION * abs(a["age_adjusted"]))

    d, v = survives("pooled", "duration"), survives("pooled", "variability")
    if abs(r) >= COLLINEAR_MAX:
        verdict = (f"NOT SEPARABLE. The two measures correlate at {r:+.2f}, past the {COLLINEAR_MAX:.2f} "
                   f"declared limit, so adjusting each for the other removes the same information twice "
                   f"and the data cannot say which is primary. Report them as two views of one "
                   f"quantity, or break the collinearity with a different design.")
        d = v = None
    elif d and v:
        verdict = ("TWO FINDINGS. Each survives adjustment for the other, so shorter sleep and more "
                   "variable sleep are separate facts about burden and both can be reported.")
    elif v and not d:
        verdict = ("ONE FINDING: VARIABILITY. Duration does not survive adjustment for it, so the "
                   "duration result is the shadow of the variability one.")
    elif d and not v:
        verdict = ("ONE FINDING: DURATION. Variability does not survive adjustment for it, so the "
                   "variability result is the shadow of the duration one.")
    else:
        verdict = ("ONE THING MEASURED TWICE. Neither survives the other, so they are not two results "
                   "and neither should be reported as independent of the other.")
    print(f"\n  -> {verdict}")
    if abs(r) >= COLLINEAR_MAX:
        print("  ★ Note what this is NOT saying. It is not saying they are one construct. It is saying")
        print("  the comparison cannot be made at this correlation, which is a different claim and a")
        print("  weaker one. Nothing above it in the printout is invalidated.")
    print("  ★ This says nothing about which is pre-specified. Duration is, variability is not, and")
    print("  that stays true whichever survives here.")
    _agg_csv(pd.DataFrame([{"n": len(P), "corr_mean_sd": round(r, 4),
                   "duration_independent": d, "variability_independent": v,
                   "independent_fraction": INDEPENDENT_FRACTION, "verdict": verdict}]), "sleep_two_findings_check.csv")
    print("\n-> sleep_two_findings.csv, sleep_two_findings_check.csv  (aggregates only)")



# ---- cell 13: does the task design exist on the chosen population? ---------------------------------
# ★ NOT PART OF THE ANALYSIS. The population below is the consensus set, and the leading group is read
# from leading_group_per_person_<tag>.csv, which no script in this folder writes.
# THE POPULATION: everyone flagged by ALL SIX encoders, the 2,177
# consensus outliers. They already carry a leading group, so no new attribution step is
# needed. What is NOT known is how many of them have a usable task sitting, because task reach falls
# with burden and these people are the burdened end of it. That is a counting question and it decides
# whether the design exists at all.
#
# ★ NO TASK VALUE IS READ AGAINST THE EXPOSURE HERE. Group means on d-prime are the OUTCOME and are not
# in this cell, the same discipline cell 9 kept for the person mean. What IS read is the SPREAD of
# d-prime over everyone usable, pooled across groups, because a minimum detectable difference cannot be
# computed without it, and a spread is a measurement property rather than a comparison.
#
# ★ DECLARED BEFORE THE RUN. A group counts as usable when at least MIN_GROUP of its members have a
# task sitting with no quality flag, the definition cell 7 used. The design is VIABLE when at least 3
# groups clear MIN_GROUP under every one of the six encoders, since a comparison needs a reference and
# at least two things to compare it with, under every repetition rather than under the best one.

MIN_GROUP = 60
TASK_PRIMARY = ("ds_gradcpt_outcomes", "DPRIME")      # provisional: the measure the prediction is about


def cell13():
    lg = None
    for tag in ("outliers", "invisible"):
        f = os.path.join(CFG.out_dir, f"leading_group_per_person_{tag}.csv")
        if os.path.exists(f):
            lg = pd.read_csv(f)
            print(f"  [leading group read from leading_group_per_person_{tag}.csv]")
            break
    if lg is None:
        print("leading_group_per_person_<tag>.csv not found (no script in this folder writes it)."); return
    idc = ID_COL if ID_COL in lg.columns else lg.columns[0]
    lg[idc] = pd.to_numeric(lg[idc], errors="coerce")
    lg = lg.dropna(subset=[idc])
    lg[idc] = lg[idc].astype("int64")
    gcol = next((c for c in ("group_name", "group", "leading_group") if c in lg.columns), None)
    ecol = next((c for c in ("encoder", "enc") if c in lg.columns), None)
    if gcol is None or ecol is None:
        print(f"  cannot find the group and encoder columns in {list(lg.columns)}"); return
    people = sorted(set(lg[idc]))
    ids = pd.Series(people, dtype="int64")

    print("=" * 96)
    print("DOES THE TASK DESIGN EXIST ON THE CHOSEN POPULATION? Everyone flagged by all six encoders,")
    print("who already carry a leading group. Task reach falls with burden and these are the burdened")
    print("end, so the counts are measured here rather than estimated. ★ Group means on the task are")
    print("the OUTCOME and are not in this cell.\n")
    print(f"  {len(people):,} people | VIABLE when at least 3 groups clear {MIN_GROUP} usable members")
    print(f"  under EVERY encoder, not under the best one. Declared in the file.\n")

    C = _columns(TASK_OUTCOMES)
    if C is None or not len(C):
        print("no column metadata"); return
    usable = {}
    for t in TASK_OUTCOMES:
        d = C[C.table_name == t]
        key = _key_of(d.column_name)
        if key is None:
            continue
        flags = [c for c in d.column_name if c.upper().startswith("FLAG_") or c.upper() == "ANY_TIMEOUTS"]
        clean = " + ".join(f"IFNULL(CAST({c} AS INT64), 0)" for c in flags) if flags else "0"
        f = f"usable_{t}_local.csv"
        if os.path.exists(f):
            u = pd.read_csv(f)
            print(f"  [{t}: usable roster read from {f}, nothing scanned]")
        else:
            u = Q(f"""SELECT DISTINCT {key} AS pid FROM `{CDR}.{t}`
                      WHERE {key} IN UNNEST(@ids) AND ({clean}) = 0""", f"{t} usable", ids)
            if u is None:
                continue
            u.to_csv(f, index=False)
            print(f"  [{t}: usable roster cached to {f} -- person level, do not download]")
        usable[t] = set(u.pid.astype("int64"))
        print(f"      {t:32s} {len(usable[t]):,} of {len(people):,} "
              f"({100 * len(usable[t]) / len(people):.1f}%)")

    prim, meas = TASK_PRIMARY
    if prim not in usable:
        print(f"\n  {prim} has no usable roster -- cannot go further"); return
    print(f"\n  usable members per leading group, per encoder, on {prim}:")
    T = lg.assign(usable=lg[idc].isin(usable[prim]))
    tab = T.pivot_table(index=gcol, columns=ecol, values="usable", aggfunc="sum").fillna(0).astype(int)
    print(tab.to_string())
    tab.to_csv("task_feasibility_by_group.csv")
    ok = {e: int((tab[e] >= MIN_GROUP).sum()) for e in tab.columns}
    print(f"\n  groups clearing {MIN_GROUP}: " + ", ".join(f"{e} {n}" for e, n in ok.items()))
    viable = all(n >= 3 for n in ok.values())

    # the spread, pooled over groups, which is a measurement property and not a comparison
    sd = Q(f"""SELECT STDDEV_SAMP({meas}) AS sd, COUNT(DISTINCT PERSON_ID) AS n, AVG({meas}) AS mu
               FROM `{CDR}.{prim}` WHERE PERSON_ID IN UNNEST(@ids)""", f"{prim} spread",
           pd.Series(sorted(usable[prim]), dtype="int64"))
    if sd is not None and len(sd):
        r = sd.iloc[0]
        s = float(r.sd)
        print(f"\n  {meas} over all usable members, pooled across groups: mean {float(r.mu):.3f}, "
              f"SD {s:.3f}, n {int(r.n):,}")
        print(f"  smallest difference detectable at 80% power against a group of {MIN_GROUP}:")
        for n2 in sorted({MIN_GROUP, 100, 200, 400}):
            mdd = 2.8 * s * np.sqrt(1 / MIN_GROUP + 1 / n2)
            print(f"    reference of {n2:4d} against {MIN_GROUP}: {mdd:.3f} {meas} units "
                  f"({mdd / s:.2f} SD)")
    print(f"\n  -> {'THE DESIGN EXISTS' if viable else 'THE DESIGN DOES NOT EXIST ON THIS POPULATION'}"
          f" under the declared rule.")
    if not viable:
        print("  Fewer than 3 groups clear the floor under at least one encoder. The choices are a")
        print("  larger population, a coarser grouping, or abandoning the task anchor. None of them is")
        print("  chosen here.")
    pd.DataFrame([{"people": len(people), "min_group": MIN_GROUP, "viable": viable,
                   **{f"groups_clearing_{e}": n for e, n in ok.items()}}]).to_csv(
        "task_feasibility_check.csv", index=False)
    print("\n-> task_feasibility_by_group.csv, task_feasibility_check.csv  (aggregates only)")


if __name__ == "__main__":
    assert CDR, "WORKSPACE_CDR is not set"
    if "1" in CELLS:
        cell1()
    if "2" in CELLS:
        cell2()
    if "3" in CELLS:
        cell3()
    if "4" in CELLS:
        cell4()
    if "5" in CELLS:
        cell5()
    if "6" in CELLS:
        cell6()
    if "7" in CELLS:
        cell7()
    if "8" in CELLS:
        cell8()
    if "9" in CELLS:
        cell9()
    if "10" in CELLS:
        cell10()
    if "11" in CELLS:
        cell11()
    if "12" in CELLS:
        cell12()
    if "13" in CELLS:
        cell13()
