#!/usr/bin/env python3
"""phecode_counts_jno.py -- PheTK's phecode counts for the cohort, with phecodeX X1.0 minus the 3 rows of
conditions_jno.MAP_REMOVALS.

RUNS IN THE PheTK ENVIRONMENT, as phecode_coverage.py (PheTK 0.3.6 needs numpy < 2.0):
    ~/phetk_env/bin/python phecode_counts_jno.py
It imports only pandas, numpy and polars (PheTK's own dependencies) and the constants of conditions_jno.py.

WHAT IT DOES
  map      PheTK's own phecodeX X1.0 table (get_phecode_map("X1.0"), all columns), minus the 3 (phecode, ICD, flag)
           rows of MAP_REMOVALS. Each row must be there exactly once, and nothing else is removed (checked).
           Written to screen_out/phetk/phecodeX_jno.csv and handed to PheTK through count_phecode's own
           phecode_map_file_path argument, which parses it with the X1.0 schema. Nothing else about PheTK changes.
  events   as phecode_coverage.py: Phecode(platform="aou") pulls the whole CDR's ICD events, restricted to the
           cohort of record (cutoff_flags.csv) before counting. The per-person summary of those events must equal
           phecode_coverage.py's icd_person_summary_local.csv, when present (the same events in both runs).
  count    Phecode.count_phecode(phecode_version="X1.0", icd_version="US", phecode_map_file_path=the map above).

OUTPUTS
  screen_out/phetk/phecodeX_jno.csv                      the map (no people)
  screen_out/phetk/phecode_counts_cohort_jno_local.tsv   person level, stays on the Workbench
  screen_out/phecode_counts_jno/removed_rows.csv         the 3 rows removed, as they were in X1.0
  screen_out/phecode_counts_jno/condition_counts.csv     the 8 phecodes of conditions_jno.py: people with any event
                                                         and cases (count >= 2) among the cohort, with this map and,
                                                         where phecode_coverage.py's counts are present, with X1.0
  screen_out/phecode_counts_jno/phecode_counts_jno_meta.json
  screen_out/phecode_counts_jno_<CDR>.zip                those 3 (no person-level column; counts as they are)

Run:  ~/phetk_env/bin/python phecode_counts_jno.py        (--refresh counts again when the cache exists)
"""
import json
import os
import sys
import zipfile
from importlib.metadata import version

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conditions_jno import (CDR, CONDITIONS, COUNTS_JNO, ID_COL, MAP_JNO, MAP_REMOVALS, MIN_COUNT,   # noqa: E402
                            OUT_DIR, PERSON, PHETK)

PHETK_VERSION = "0.3.6"
PHECODE_VERSION = "X1.0"
OUT = os.path.join(OUT_DIR, "phecode_counts_jno")
COUNTS_X10 = os.path.join(PHETK, "phecode_counts_cohort_local.tsv")      # phecode_coverage.py's, with X1.0 as shipped
PERSON_NEW = os.path.join(PHETK, "icd_person_summary_jno_local.csv")     # this run's, compared with PERSON


def build_map():
    """X1.0 as PheTK ships it, minus MAP_REMOVALS. Returns (the map, the removed rows)."""
    import polars as pl
    from phetk.phecode import get_phecode_map
    M = get_phecode_map(PHECODE_VERSION, "US")                            # all columns, PheTK's own schema
    drop = pl.lit(False)
    for code, icd, flag in MAP_REMOVALS:
        hit = (pl.col("phecode") == code) & (pl.col("ICD") == icd) & (pl.col("flag") == flag)
        n = M.filter(hit).height
        assert n == 1, f"({code}, {icd}, flag {flag}) is in X1.0 {n} times, expected exactly once"
        drop = drop | hit
    removed = M.filter(drop)
    kept = M.filter(~drop)
    assert kept.height == M.height - len(MAP_REMOVALS), "the removal took other rows with it"
    return kept, removed


def events_summary(ev):
    """Per person: events, coded dates, first and last date, years of record (phecode_coverage.py's form)."""
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


def count(ids, map_path):
    """PheTK's events for the cohort, counted with the map at map_path. Writes COUNTS_JNO and PERSON_NEW."""
    import polars as pl
    from phetk.phecode import Phecode
    ph = Phecode(platform="aou")                                          # the whole CDR, as PheTK queries it
    n_all = ph.icd_events["person_id"].n_unique()
    ph.icd_events = ph.icd_events.filter(pl.col("person_id").cast(pl.Utf8).is_in(list(ids)))
    print(f"[phecode_counts_jno] PheTK events: {n_all:,} people in the CDR, "
          f"{ph.icd_events['person_id'].n_unique():,} of them in the cohort", flush=True)
    events_summary(ph.icd_events).with_columns(pl.col("person_id").cast(pl.Utf8)).write_csv(PERSON_NEW)
    ph.count_phecode(phecode_version=PHECODE_VERSION, icd_version="US", phecode_map_file_path=map_path,
                     output_file_path=COUNTS_JNO)


def per_phecode(path, ids, has_rec_ids, codes):
    """n_any and n_case per phecode among the cohort (people with records; an event implies a record)."""
    C = pd.read_csv(path, sep="\t", dtype={"person_id": str, "phecode": str}, usecols=["person_id", "phecode", "count"])
    C = C[C["phecode"].isin(codes) & C["person_id"].isin(ids)]
    assert C["person_id"].isin(has_rec_ids).all(), "a counted person has no ICD event in the summary"
    any_n = C.groupby("phecode")["person_id"].nunique()
    case_n = C[C["count"] >= MIN_COUNT].groupby("phecode")["person_id"].nunique()
    return {p: (int(any_n.get(p, 0)), int(case_n.get(p, 0))) for p in codes}


def main():
    installed = version("phetk")
    assert installed == PHETK_VERSION, f"PheTK {installed} is installed, the analysis pins {PHETK_VERSION}"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PHETK, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    ids = pd.Index(cf[ID_COL].astype(str), name=ID_COL)

    kept, removed = build_map()
    kept.write_csv(MAP_JNO)
    from phetk.phecode import get_phecode_map                              # the file reads back as PheTK will read it
    back = get_phecode_map(PHECODE_VERSION, "US", phecode_map_file_path=MAP_JNO, keep_all_columns=False)
    assert back.height == kept.height, "the written map does not read back whole"
    for code, icd, flag in MAP_REMOVALS:
        assert back.filter((back["phecode"] == code) & (back["ICD"] == icd) & (back["flag"] == flag)).height == 0
    print(f"[phecode_counts_jno] map: X1.0 {kept.height + removed.height:,} rows, {removed.height} removed -> {MAP_JNO}")

    if "--refresh" in sys.argv or not os.path.exists(COUNTS_JNO):
        import importlib.util
        assert importlib.util.find_spec("google.cloud.bigquery_storage") is not None, (
            "google-cloud-bigquery-storage is missing, so the ICD pull would use the slow REST endpoint. "
            "Run: ~/phetk_env/bin/pip install google-cloud-bigquery-storage")
        count(set(ids), MAP_JNO)
    else:
        print(f"[phecode_counts_jno] reusing {COUNTS_JNO} (--refresh counts again)")

    same_events = None
    if os.path.exists(PERSON) and os.path.exists(PERSON_NEW):             # the same events as phecode_coverage.py's
        a = pd.read_csv(PERSON, dtype={"person_id": str}).set_index("person_id").sort_index()
        b = pd.read_csv(PERSON_NEW, dtype={"person_id": str}).set_index("person_id").sort_index()
        same_events = bool(a.index.equals(b.index) and (a["n_events"].values == b["n_events"].values).all()
                           and (a["n_dates"].values == b["n_dates"].values).all())
        assert same_events, "the ICD events differ from phecode_coverage.py's run: stop and look before going on"
        print("  the cohort's ICD events equal phecode_coverage.py's (people, events and dates per person)")
    person = pd.read_csv(PERSON_NEW if os.path.exists(PERSON_NEW) else PERSON, dtype={"person_id": str})
    has_rec_ids = set(person.loc[person["n_events"] > 0, "person_id"])

    codes = [c[1] for c in CONDITIONS]
    new = per_phecode(COUNTS_JNO, ids, has_rec_ids, codes)
    old = per_phecode(COUNTS_X10, ids, has_rec_ids, codes) if os.path.exists(COUNTS_X10) else None
    changed = {c for c, *_ in MAP_REMOVALS}
    rows = []
    for key, code, label, lst, _ in CONDITIONS:
        r = {"condition": key, "phecode": code, "label": label, "list": lst, "map": "X1.0 minus 1 row" if code in changed
             else "X1.0", "n_any": new[code][0], "n_case": new[code][1]}
        if old is not None:
            r.update({"n_any_x10": old[code][0], "n_case_x10": old[code][1]})
            if code not in changed:                                   # untouched phecodes must not move
                assert new[code] == old[code], f"{code}: counts moved although its rows did not"
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "condition_counts.csv"), index=False)
    removed.to_pandas()[["phecode", "ICD", "flag", "phecode_string"]].to_csv(os.path.join(OUT, "removed_rows.csv"),
                                                                              index=False)
    json.dump({"cdr": CDR, "phetk_version": installed, "phecode_version": PHECODE_VERSION,
               "map": "X1.0 as shipped with PheTK, minus the rows in removed_rows.csv, passed as phecode_map_file_path",
               "case_rule": f"count >= {MIN_COUNT}", "same_events_as_phecode_coverage": same_events,
               "counts": "written as they are"},
              open(os.path.join(OUT, "phecode_counts_jno_meta.json"), "w"), indent=2)

    print("  the 8 phecodes, cases (count >= 2) among the cohort" + (" [X1.0 as shipped]" if old else "") + ":")
    for r in rows:
        print(f"    {r['phecode']:9s} {r['label'][:32]:32s} {r['n_case']:>7,}" +
              (f"  [{r['n_case_x10']:,}]" if old is not None else ""))
    files = ["removed_rows.csv", "condition_counts.csv", "phecode_counts_jno_meta.json"]
    for fn in files:                                                  # no person-level column (pii_check's rule)
        if fn.endswith(".csv"):
            cols = {c.strip().lower() for c in pd.read_csv(os.path.join(OUT, fn)).columns}
            assert not cols & {"id", "person_id", "research_id"}, f"{fn} carries a person column"
    zpath = os.path.join(OUT_DIR, f"phecode_counts_jno_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"phecode_counts_jno/{fn}")
    print(f"\n[phecode_counts_jno] done\n  download -> {zpath}")


if __name__ == "__main__":
    main()
