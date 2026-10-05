#!/usr/bin/env python3
"""phecode_coverage.py -- how many of the 90,351 have diagnosis records PheTK can see, and how many
people each phecode reaches. A first, descriptive count, made before the denominator and the
diagnoses are chosen.

RUNS IN ITS OWN ENVIRONMENT. PheTK 0.3.6 requires numpy < 2.0, so it is installed in a separate
virtual environment and this script runs with that environment's python. The main environment, which
every other step uses, is not touched:
    python3 -m venv ~/phetk_env
    ~/phetk_env/bin/pip install phetk==0.3.6
    ~/phetk_env/bin/pip install google-cloud-bigquery-storage
    ~/phetk_env/bin/python phecode_coverage.py
The storage module is not one of PheTK's dependencies. Without it BigQuery falls back to its REST
endpoint, which is far slower for the whole CDR's ICD events. It changes how the rows are fetched, not
which rows, and the script stops before a pull if it is missing.
It imports nothing from the main environment, only pandas, numpy and polars (PheTK's own
dependencies) and disclosure.py and zip_screen_aggregates.py from this folder.

WHAT PheTK DOES (read from its source, v0.3.6)
  events   ICD-9-CM and ICD-10-CM codes from condition_occurrence AND observation, matched by source
           value or source concept. One event = one distinct (person, date, ICD code).
  map      phecodeX release X1.0, pinned. It is already rolled up: a code counts toward its phecode
           and every parent above it. 3,612 phecodes in 18 categories.
  count    per person and phecode, the number of mapped events (Phecode.count_phecode).
  case     count >= 2, PheTK's default min_phecode_count. A person with exactly 1 event counts as NOT
           having the diagnosis.
Phecode(platform="aou") pulls the ICD events of the whole CDR, as PheTK is written to. They are then
restricted to the 90,351 before count_phecode runs. A person's counts depend only on that person's own
events, so this gives each of our people exactly the counts a whole-CDR run would, with less time and disk.

WHAT IT REPORTS, over the cohort of record (the 90,351 of cutoff_flags.csv)
  phecode_coverage.json
    records     4 cells: at least 1 ICD event, and at least 1 visit in visit_occurrence (care.py's
                has_ehr, the ED-share denominator), crossed. Plus people with ICD events on at least
                2 distinct dates.
    among those with at least 1 ICD event: the 25th, 50th and 75th percentiles of
                years of record   (last - first coded date + 1 day) / 365.2425, PheTK's ehr_length
                coded dates       distinct dates with an ICD event
                events            distinct (date, ICD code) pairs
                phecodes as case  phecodes the person has a count >= 2 for
                and the number who are a case for no phecode.
  phecode_counts.csv   1 row per phecode of X1.0: its name, category and sex restriction, then
                n_any    people with at least 1 event
                n_case   people with count >= 2 (PheTK's case)

THE DISSEMINATION POLICY (disclosure.py). A count of 1 to 20 is written "<=20". The
records table is a partition of the cohort and goes through partition_mask. In the phecode table, a
count is also hidden ("suppressed") when subtracting it from a larger count it is a subset of leaves 1
to 20 people. The subsets are:
  - the cases of a phecode, within its people with any event;
  - ★ phecode X within phecode Y whenever EVERY ICD code of X also maps to Y in PheTK's rolled-up map
    (then every event of X is an event of Y, so X's people with any event, and its cases, are subsets
    of Y's). This covers a phecode's named parents, and also the 955 pairs where Y is not X's parent by
    name;
  - any phecode's people, within the people with at least 1 ICD event.
The check repeats until nothing changes. Percentiles are written; no minimum or maximum is.

PERSON-LEVEL FILES, which stay on the Workbench (never zipped), reused on a re-run (--refresh pulls again)
  screen_out/phetk/phecode_counts_cohort_local.tsv   PheTK's counts, the 90,351 only (the next steps read it)
  screen_out/phetk/icd_person_summary_local.csv      per person: events, coded dates, years of record
OUTPUTS  screen_out/phecode_coverage/phecode_coverage.json, phecode_counts.csv
         screen_out/phecode_coverage_<CDR>.zip          those 2, checked by zip_screen_aggregates.verdict()

Run:  ~/phetk_env/bin/python phecode_coverage.py
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
from disclosure import MARK, SUPP, marked, one, partition_mask, small    # noqa: E402  the policy
from zip_screen_aggregates import verdict                                 # noqa: E402

PHETK_VERSION = "0.3.6"
PHECODE_VERSION = "X1.0"
MIN_COUNT = 2                         # PheTK's default min_phecode_count: a case needs 2 events
OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "phecode_coverage")
CACHE = os.path.join(OUT_DIR, "phetk")
COUNTS = os.path.join(CACHE, "phecode_counts_cohort_local.tsv")
PERSON = os.path.join(CACHE, "icd_person_summary_local.csv")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"


def pull(ids):
    """PheTK's events and counts for the cohort, written to the 2 person-level caches."""
    import polars as pl
    from phetk.phecode import Phecode

    ph = Phecode(platform="aou")                                   # the whole CDR, as PheTK queries it
    n_all = ph.icd_events["person_id"].n_unique()
    ph.icd_events = ph.icd_events.filter(pl.col("person_id").cast(pl.Utf8).is_in(list(ids)))
    print(f"[phecode_coverage] PheTK events: {n_all:,} people in the CDR, "
          f"{ph.icd_events['person_id'].n_unique():,} of them in the cohort", flush=True)

    ev = ph.icd_events
    dt = ev.schema["date"]                                         # a date from BigQuery; text from a file
    if dt == pl.Utf8:
        ev = ev.with_columns(pl.col("date").str.to_date())
    elif isinstance(dt, pl.Datetime):
        ev = ev.with_columns(pl.col("date").dt.date())
    ps = ev.group_by("person_id").agg(
        pl.len().alias("n_events"), pl.col("date").n_unique().alias("n_dates"),
        pl.col("date").min().alias("first_date"), pl.col("date").max().alias("last_date"))
    ps = ps.with_columns(
        (((pl.col("last_date") - pl.col("first_date")).dt.total_days() + 1) / 365.2425).alias("years_of_record"))
    ps.with_columns(pl.col("person_id").cast(pl.Utf8)).write_csv(PERSON)

    ph.count_phecode(phecode_version=PHECODE_VERSION, icd_version="US", output_file_path=COUNTS)


def protect(values, relations):
    """The written form of each count: MARK for 1 to 20, SUPP where a larger count it is a subset of
    would give 1 to 20 back by subtraction. relations = (larger key, subset key). Repeats until stable."""
    out = {k: (MARK if small(v) else int(v)) for k, v in values.items()}
    hidden = {k for k, v in values.items() if small(v)}
    changed = True
    while changed:
        changed = False
        for big, sub in relations:
            if big in hidden or sub in hidden:
                continue
            if small(values[big] - values[sub]):
                hidden.add(sub)
                out[sub] = SUPP
                changed = True
    return out


def containing(Mfull):
    """For each phecode X, the phecodes Y (Y != X) that every ICD code of X also maps to. Then X's people
    are a subset of Y's, for any event and for cases alike."""
    codes = {p: frozenset(zip(d["ICD"], d["flag"])) for p, d in Mfull.groupby("phecode")}
    by_code = {}
    for p, cs in codes.items():
        for c in cs:
            by_code.setdefault(c, set()).add(p)
    return {x: set.intersection(*[by_code[c] for c in cs]) - {x} for x, cs in codes.items()}


def pct(x):
    q = np.percentile(np.asarray(x, float), [25, 50, 75])
    return {"p25": float(q[0]), "median": float(q[1]), "p75": float(q[2])}


def main():
    installed = version("phetk")
    assert installed == PHETK_VERSION, f"PheTK {installed} is installed, the analysis pins {PHETK_VERSION}"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)

    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    ids = pd.Index(cf[ID_COL].astype(str), name=ID_COL)            # the cohort of record, 90,351
    N = len(ids)
    if "--refresh" in sys.argv or not (os.path.exists(COUNTS) and os.path.exists(PERSON)):
        import importlib.util
        assert importlib.util.find_spec("google.cloud.bigquery_storage") is not None, (
            "google-cloud-bigquery-storage is missing, so the ICD pull would use the slow REST endpoint. "
            "Run: ~/phetk_env/bin/pip install google-cloud-bigquery-storage")
        pull(set(ids))
    else:
        print(f"[phecode_coverage] reusing {COUNTS} and {PERSON} (--refresh pulls again)")

    P = pd.read_csv(PERSON, dtype={"person_id": str}).set_index("person_id").reindex(ids)
    C = pd.read_csv(COUNTS, sep="\t", dtype={"person_id": str, "phecode": str})
    C = C[C["person_id"].isin(ids)]
    from phetk.phecode import get_phecode_map
    Mfull = get_phecode_map(PHECODE_VERSION).to_pandas()
    M = Mfull[["phecode", "phecode_string", "phecode_category", "sex"]].drop_duplicates("phecode")
    within = containing(Mfull)
    assert C["phecode"].isin(M["phecode"]).all(), "a counted phecode is not in the X1.0 map"

    # records: at least 1 ICD event, crossed with at least 1 visit (care.py's has_ehr)
    icd = P["n_events"].fillna(0).values > 0
    O = pd.read_csv(os.path.join(OUT_DIR, "care_outcomes_per_person.csv"), usecols=[ID_COL, "has_ehr"])
    O[ID_COL] = O[ID_COL].astype(str)
    visit = O.set_index(ID_COL).reindex(ids)["has_ehr"].fillna(False).astype(bool).values
    cells = {"icd_and_visit": int((icd & visit).sum()), "icd_no_visit": int((icd & ~visit).sum()),
             "visit_no_icd": int((~icd & visit).sum()), "neither": int((~icd & ~visit).sum())}
    hide, primary = partition_mask(list(cells.values()))
    cells_w = dict(zip(cells, marked(list(cells.values()), hide, primary)))
    n_icd, n_visit = int(icd.sum()), int(visit.sum())
    margins_ok = not hide.any()
    two = int((P["n_dates"].fillna(0).values >= 2).sum())

    cases = C[C["count"] >= MIN_COUNT]
    per_person = cases.groupby("person_id").size().reindex(ids[icd]).fillna(0).values
    Pi = P[icd]
    n_zero = int((per_person == 0).sum())
    coverage = {
        "cdr": CDR, "phetk_version": installed, "phecode_version": PHECODE_VERSION, "case_rule": f"count >= {MIN_COUNT}",
        "one_event": "counts as not having the diagnosis",
        "n_cohort": N,
        "records_cells": cells_w,
        "n_icd": one(n_icd) if margins_ok else SUPP,
        "n_visit": one(n_visit) if margins_ok else SUPP,
        "n_icd_2dates": SUPP if (small(two) or small(n_icd - two)) else two,
        "among_icd": {
            "years_of_record": pct(Pi["years_of_record"]),
            "coded_dates": pct(Pi["n_dates"]),
            "events": pct(Pi["n_events"]),
            "phecodes_as_case": pct(per_person),
            "n_case_for_no_phecode": SUPP if (small(n_zero) or small(n_icd - n_zero)) else n_zero,
        },
    }

    # per phecode: people with any event and cases, over the cohort
    any_n = C.groupby("phecode")["person_id"].nunique()
    case_n = cases.groupby("phecode")["person_id"].nunique()
    codes = M["phecode"].tolist()
    codeset = set(codes)
    vals, rel = {"icd": n_icd}, []
    for p in codes:
        vals[("any", p)] = int(any_n.get(p, 0))
        vals[("case", p)] = int(case_n.get(p, 0))
        rel += [(("any", p), ("case", p)), ("icd", ("any", p))]
        for a in within[p]:
            if a in codeset:
                rel += [(("any", a), ("any", p)), (("any", a), ("case", p)), (("case", a), ("case", p))]
    w = protect(vals, rel)
    T = M.copy()
    T["n_any"] = [w[("any", p)] for p in codes]
    T["n_case"] = [w[("case", p)] for p in codes]
    T.to_csv(os.path.join(OUT, "phecode_counts.csv"), index=False)
    n_hidden = int(sum(isinstance(v, str) for k, v in w.items() if k != "icd"))
    coverage["phecode_table"] = {"n_phecodes": len(codes), "counts_hidden_under_policy": n_hidden,
                                 "relations_checked": len(rel)}
    json.dump(coverage, open(os.path.join(OUT, "phecode_coverage.json"), "w"), indent=2)

    print(f"  cohort {N:,} | at least 1 ICD event {n_icd:,} | at least 1 visit {n_visit:,} | "
          f"events on at least 2 dates {two:,}")
    print(f"  records cells: {cells}")
    for k, v in coverage["among_icd"].items():
        print(f"  among those with events, {k}: {v}")
    top = pd.DataFrame({"phecode": codes, "n_case": [vals[("case", p)] for p in codes]}).merge(M, on="phecode")
    top = top.sort_values("n_case", ascending=False)
    print("  the 25 phecodes with the most cases (printed only):")
    for r in top.head(25).itertuples():
        print(f"    {r.phecode:12s} {r.n_case:>7,}  {r.phecode_string} [{r.phecode_category}]")
    for k in (100, 1000):
        print(f"  phecodes with at least {k:,} cases: {int((top.n_case >= k).sum()):,} of {len(codes):,}")
    print(f"  counts hidden under the policy: {n_hidden:,} of {2 * len(codes):,}")

    files = ["phecode_coverage.json", "phecode_counts.csv"]
    for f in files:
        ok, why = verdict(os.path.join(OUT, f))
        assert ok, f"{f} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"phecode_coverage_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(OUT, f), arcname=f"phecode_coverage/{f}")
    print(f"\n[phecode_coverage] done\n  download -> {zpath}")


if __name__ == "__main__":
    main()
