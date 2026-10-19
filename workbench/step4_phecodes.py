#!/usr/bin/env python3
"""step4_phecodes.py -- each participant's diagnosis codes, counted by phecode with PheTK (the recorded conditions).

RUNS IN ITS OWN ENVIRONMENT. PheTK 0.3.6 needs numpy < 2.0, so it lives in a separate virtual environment:
    python3 -m venv ~/phetk_env
    ~/phetk_env/bin/pip install phetk==0.3.6 google-cloud-bigquery-storage
    ~/phetk_env/bin/python step4_phecodes.py
google-cloud-bigquery-storage is not a PheTK dependency; without it the pull of the whole repository's ICD events falls
back to a much slower endpoint (the same rows). This script imports only pandas, numpy, polars and PheTK.

WHAT IT DOES
  map      phecodeX X1.0 as PheTK ships it, minus 3 rows, each the one code that made its condition's label change with
           the coding year (MAP_REMOVALS): RE_475 asthma, ICD-9-CM 493.21 (chronic obstructive asthma with status
           asthmaticus); CA_105 breast cancer, ICD-10-CM Z86.000 (personal history of in-situ neoplasm of breast);
           CA_107.2 prostate cancer, ICD-10-CM R97.21 (rising PSA after treatment). Each must be there exactly once.
  events   Phecode(platform="aou") pulls the ICD-9-CM and ICD-10-CM events of the whole repository, from
           condition_occurrence and observation; they are restricted to the cohort before counting (a person's counts
           depend only on their own events). One event = one distinct (person, date, code).
  count    per person and phecode, the number of events mapped to it (Phecode.count_phecode with the map above). A
           condition is recorded when the count is at least 2; a single event counts as not recorded (step 5 on).

OUTPUTS (work/phetk/, person level, never downloaded)
  phecodeX_map.csv          the map used
  icd_person_summary.csv    per person: events, coded dates, first and last date, years of record
  phecode_counts.tsv        per person and phecode: the count
Reused on a rerun (--refresh pulls again).

Run:  ~/phetk_env/bin/python step4_phecodes.py
"""
import os
import sys
from importlib.metadata import version

import pandas as pd

from settings import COHORT_CSV, ID, work

PHETK_VERSION = "0.3.6"
PHECODE_VERSION = "X1.0"
MAP_REMOVALS = [("RE_475", "493.21", 9), ("CA_105", "Z86.000", 10), ("CA_107.2", "R97.21", 10)]   # (phecode, ICD, flag)
MAP = work("phetk", "phecodeX_map.csv")
PERSON = work("phetk", "icd_person_summary.csv")
COUNTS = work("phetk", "phecode_counts.tsv")


def build_map():
    """X1.0 as PheTK ships it, minus MAP_REMOVALS; each removed row must be there exactly once."""
    import polars as pl
    from phetk.phecode import get_phecode_map
    M = get_phecode_map(PHECODE_VERSION, "US")
    drop = pl.lit(False)
    for code, icd, flag in MAP_REMOVALS:
        hit = (pl.col("phecode") == code) & (pl.col("ICD") == icd) & (pl.col("flag") == flag)
        n = M.filter(hit).height
        assert n == 1, f"({code}, {icd}, flag {flag}) is in X1.0 {n} times, expected exactly once"
        drop = drop | hit
    kept = M.filter(~drop)
    assert kept.height == M.height - len(MAP_REMOVALS), "the removal took other rows with it"
    return kept


def events_summary(ev):
    """Per person: events, coded dates, first and last date, years of record."""
    import polars as pl
    dt = ev.schema["date"]
    if dt == pl.Utf8:
        ev = ev.with_columns(pl.col("date").str.to_date())
    elif isinstance(dt, pl.Datetime):
        ev = ev.with_columns(pl.col("date").dt.date())
    ps = ev.group_by("person_id").agg(
        pl.len().alias("n_events"), pl.col("date").n_unique().alias("n_dates"),
        pl.col("date").min().alias("first_date"), pl.col("date").max().alias("last_date"))
    return ps.with_columns(
        (((pl.col("last_date") - pl.col("first_date")).dt.total_days() + 1) / 365.2425).alias("years_of_record"))


def main():
    installed = version("phetk")
    assert installed == PHETK_VERSION, f"PheTK {installed} is installed; the analysis uses {PHETK_VERSION}"
    ids = set(pd.read_csv(COHORT_CSV, usecols=[ID])[ID].astype(str))
    kept = build_map()
    kept.write_csv(MAP)
    from phetk.phecode import get_phecode_map
    back = get_phecode_map(PHECODE_VERSION, "US", phecode_map_file_path=MAP, keep_all_columns=False)
    assert back.height == kept.height, "the written map does not read back whole"
    print(f"[step 4] map: X1.0 minus {len(MAP_REMOVALS)} rows -> {MAP}", flush=True)
    if os.path.exists(COUNTS) and os.path.exists(PERSON) and "--refresh" not in sys.argv:
        print(f"[step 4] {COUNTS} and {PERSON} exist: nothing pulled (--refresh pulls again)")
        return
    import importlib.util
    assert importlib.util.find_spec("google.cloud.bigquery_storage") is not None, (
        "google-cloud-bigquery-storage is missing: ~/phetk_env/bin/pip install google-cloud-bigquery-storage")
    import polars as pl
    from phetk.phecode import Phecode
    ph = Phecode(platform="aou")                                          # the whole repository, as PheTK queries it
    ph.icd_events = ph.icd_events.filter(pl.col("person_id").cast(pl.Utf8).is_in(list(ids)))
    print(f"[step 4] ICD events of {ph.icd_events['person_id'].n_unique():,} cohort participants", flush=True)
    events_summary(ph.icd_events).with_columns(pl.col("person_id").cast(pl.Utf8)).write_csv(PERSON)
    ph.count_phecode(phecode_version=PHECODE_VERSION, icd_version="US", phecode_map_file_path=MAP,
                     output_file_path=COUNTS)
    print(f"[step 4] wrote {PERSON} and {COUNTS}", flush=True)


if __name__ == "__main__":
    main()
