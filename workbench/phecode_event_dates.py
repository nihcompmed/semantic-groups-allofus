#!/usr/bin/env python3
"""phecode_event_dates.py -- the dated diagnosis events behind the four screen pairs. The saved PheTK counts hold
only each person's FIRST event date; the timing of "on or before the survey" also needs the most recent code on or
before it.

RUNS IN THE PheTK ENVIRONMENT, like phecode_coverage.py:   ~/phetk_env/bin/python phecode_event_dates.py

WHAT IT PULLS. PheTK's own ICD event query (phetk._queries.phecode_icd_query, PheTK 0.3.6), so an event is
exactly what PheTK counted: one distinct (person, date, ICD code, vocabulary) from condition_occurrence and
observation. The query is restricted to the cohort (cutoff_flags.csv) and to the ICD codes that map, in
PheTK's rolled-up phecodeX X1.0 map, to the four diagnoses of the pairs: MB_286.2, MB_288.3, MB_304,
MB_284. The events are mapped to those phecodes on (ICD, flag) as PheTK maps them.

THE CHECK. For every person and each of the four phecodes, the number of mapped events and the first
event date must equal PheTK's saved counts (screen_out/phetk/phecode_counts_cohort_local.tsv, written by
phecode_coverage.py). It stops otherwise, so the dates below belong to exactly the events counted.

OUTPUT, PERSON LEVEL, which stays on the Workbench (never downloaded; no zip is written):
  screen_out/phetk/phecode_event_dates_local.csv    person_id, phecode, date  (distinct dates)
Only checks are printed.

Run:  ~/phetk_env/bin/python phecode_event_dates.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "screen_out")
CACHE = os.path.join(OUT_DIR, "phetk")
COUNTS = os.path.join(CACHE, "phecode_counts_cohort_local.tsv")
DATES = os.path.join(CACHE, "phecode_event_dates_local.csv")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
PHECODE_VERSION = "X1.0"                       # as phecode_coverage.py
CODES = ["MB_286.2", "MB_288.3", "MB_304", "MB_284"]


def main():
    import polars as pl
    from google.cloud import bigquery
    from phetk import _queries
    from phetk.phecode import get_phecode_map

    assert CDR, "WORKSPACE_CDR is not set"
    assert os.path.exists(COUNTS), f"{COUNTS} is missing: run phecode_coverage.py first"
    ids = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])[ID_COL].astype(str)
    M = get_phecode_map(PHECODE_VERSION, keep_all_columns=False).filter(pl.col("phecode").is_in(CODES))
    icd = sorted(set(M["ICD"].to_list()))
    print(f"[phecode_event_dates] cohort {len(ids):,} | {len(icd):,} ICD codes map to {', '.join(CODES)}", flush=True)

    sql = f"""SELECT DISTINCT person_id, date, ICD, vocabulary_id
              FROM ({_queries.phecode_icd_query(CDR)})
              WHERE person_id IN UNNEST(@ids) AND ICD IN UNNEST(@icd)"""
    params = [bigquery.ArrayQueryParameter("ids", "INT64", [int(i) for i in ids]),
              bigquery.ArrayQueryParameter("icd", "STRING", icd)]
    client = bigquery.Client()
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params, dry_run=True,
                                                                use_query_cache=False))
    print(f"  [events: scanning {dry.total_bytes_processed / 1e9:,.1f} GB]", flush=True)
    ev = pl.from_arrow(client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
                       .result().to_arrow())
    dt = ev.schema["date"]
    if dt == pl.Utf8:
        ev = ev.with_columns(pl.col("date").str.to_date())
    elif isinstance(dt, pl.Datetime):
        ev = ev.with_columns(pl.col("date").dt.date())
    ev = ev.with_columns(                                            # PheTK's own flag rule
        pl.when(pl.col("vocabulary_id").is_in(["ICD9", "ICD9CM"])).then(9)
        .when(pl.col("vocabulary_id").is_in(["ICD10", "ICD10CM"])).then(10)
        .otherwise(0).alias("flag").cast(pl.Int8),
        pl.col("person_id").cast(pl.Utf8))
    mapped = ev.join(M.with_columns(pl.col("flag").cast(pl.Int8)), on=["ICD", "flag"], how="inner")
    print(f"  {ev.height:,} events pulled, {mapped.height:,} mapped rows", flush=True)

    # the check: the same counts and first dates as PheTK's saved counts
    mine = mapped.group_by(["person_id", "phecode"]).agg(pl.len().alias("count"),
                                                          pl.col("date").min().alias("first_event_date")).to_pandas()
    C = pd.read_csv(COUNTS, sep="\t", dtype={"person_id": str, "phecode": str},
                    usecols=["person_id", "phecode", "count", "first_event_date"])
    C = C[C["phecode"].isin(CODES) & C["person_id"].isin(set(ids))]
    mine["first_event_date"] = pd.to_datetime(mine["first_event_date"])
    C["first_event_date"] = pd.to_datetime(C["first_event_date"])
    J = C.merge(mine, on=["person_id", "phecode"], how="outer", suffixes=("_phetk", "_here"), indicator=True)
    only = {k: int(v) for k, v in J["_merge"].value_counts().to_dict().items()}
    bad_n = int((J["count_phetk"] != J["count_here"]).sum())
    bad_d = int((J["first_event_date_phetk"] != J["first_event_date_here"]).sum())
    # the log leaves the Workbench (the dissemination policy): a count of 1 to 20 is printed as <=20
    sh = lambda n: "<=20" if 0 < n <= 20 else f"{n:,}"
    print(f"  against PheTK's saved counts: person-phecode pairs "
          f"{ {k: sh(v) for k, v in only.items()} }; count differs {sh(bad_n)}, first date differs {sh(bad_d)}",
          flush=True)
    assert only.get("left_only", 0) == 0 and only.get("right_only", 0) == 0 and bad_n == 0 and bad_d == 0, \
        "the dated events are not the events PheTK counted; nothing is written"

    out = mapped.select(["person_id", "phecode", "date"]).unique().sort(["person_id", "phecode", "date"])
    out.write_csv(DATES)
    print(f"  {out.height:,} person x phecode x date rows -> {DATES} (person level, do not download)")
    print("[phecode_event_dates] done; next: python3 phecode_screen_negative_timing.py")


if __name__ == "__main__":
    sys.exit(main())
