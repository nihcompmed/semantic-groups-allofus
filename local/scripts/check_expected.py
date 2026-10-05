#!/usr/bin/env python
"""check_expected.py -- compare a fresh run with the files of record in expected/.

Sets (each run script runs its own set last):
  semantic_groups   run_semantic_groups.sh   the semantic groups, eTables 1, 4 and 12, Supplement 2
  cohort            run_cohort.sh            eTables 2 and 3
  deciles           run_deciles.sh           eTables 5 to 7 (outcomes by decile of the distance)
  contributions     run_contributions.sh     eTable 8 (groups contributing most to the distance)
  conditions        run_conditions.sh        eTables 9 to 11 (semantic groups and recorded conditions)
  groupings         run_groupings.sh         eTable 13 (the other groupings, eMethods 5)
  all               every set

It compares the run's outputs with expected/, file by file, and says which match exactly, which match to a stated
tolerance, and which differ. Without an expected/ folder beside scripts/ it says so and stops.

What is compared:
  data/items_151.csv                          the item table of record                         exact
  outputs/k_selection/summary.csv             k per encoder (Horn's parallel analysis)          exact
  outputs/random_starts/selection.json        the retained start per encoder and its metrics    seed exact, metrics 1e-9
  outputs/basis/<encoder>.npz                 W, mu, sigma of the basis of record               max |difference|, 1e-9
  outputs/basis/components.csv                every group's core items and constructs           exact
  outputs/cores/summary.csv                   core sizes, single-construct groups, faithfulness exact
  outputs/xenc/summary.json                   agreement across encoders                          1e-9
  outputs/tables/etable_groups_jno.csv        eTable 4                                          exact
  outputs/tables/switch_items.csv             eTable 12 and its Supplement 2 file               exact
  set cohort:
  outputs/tables/etable_encoders_jno.{tex,csv}     eTable 2                                   exact
  outputs/tables/etable_subsets_jno.{tex,csv}      eTable 3                                   exact
  set deciles:
  outputs/tables/etable_thresholds_jno.{tex,csv}   eTable 5                                   exact
  outputs/tables/etable_sleep_device_jno.{tex,csv} eTable 6                                   exact
  outputs/tables/etable_deciles_jno.{tex,csv}      eTable 7                                   exact
  set contributions:
  outputs/tables/etable_content_jno.{tex,csv}      eTable 8                                   exact
  set conditions:
  outputs/tables/etable_control_jno.{tex,csv}      eTable 9                                   exact
  outputs/tables/etable_assoc_jno.{tex,csv}        eTable 10                                  exact
  outputs/tables/etable_breast_jno.{tex,csv}       eTable 11                                  exact
  set groupings:
  outputs/tables/etable_groupings_jno.{tex,csv}    eTable 13                                  exact

The fit is device dependent at the level of floating-point arithmetic (README, "Device dependence"). On another CPU
or other library versions, W can differ in the last digits; the groups themselves (components.csv) are what the paper
reports, so check them first.

Run:  python check_expected.py semantic_groups|cohort|deciles|contributions|conditions|groupings|all
Exit: 0 when everything matches, 1 when anything differs, 0 with a note when there is no expected/ folder.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "expected")
ENCODERS = ["bge-m3", "bge-large-en-v1.5", "all-mpnet-base-v2", "e5-large-v2", "gte-large-en-v1.5",
            "qwen3-embedding-0.6b"]
TOL = 1e-9
SEMANTIC_GROUPS_EXACT = ["data/items_151.csv", "outputs/k_selection/summary.csv", "outputs/basis/components.csv",
                         "outputs/cores/summary.csv", "outputs/tables/etable_groups_jno.csv",
                         "outputs/tables/switch_items.csv"]
R2_EXACT = [f"outputs/tables/{t}.{x}" for t in ("etable_thresholds_jno", "etable_sleep_device_jno", "etable_deciles_jno")
            for x in ("tex", "csv")]
GROUPINGS_EXACT = [f"outputs/tables/etable_groupings_jno.{x}" for x in ("tex", "csv")]
COHORT_EXACT = [f"outputs/tables/{t}.{x}" for t in ("etable_encoders_jno", "etable_subsets_jno") for x in ("tex", "csv")]
R3_EXACT = [f"outputs/tables/etable_content_jno.{x}" for x in ("tex", "csv")]
R5_EXACT = [f"outputs/tables/{t}.{x}" for t in ("etable_control_jno", "etable_assoc_jno", "etable_breast_jno")
            for x in ("tex", "csv")]


def exact(rel):
    a, b = os.path.join(ROOT, rel), os.path.join(EXP, rel)
    if not os.path.exists(a):
        return False, "not produced"
    same = open(a, "rb").read() == open(b, "rb").read()
    return same, "identical" if same else "differs"


def numbers(x, y, path=""):
    """Largest absolute difference between 2 JSON-like objects; None if their structure or any string differs."""
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) != set(y):
            return None
        ds = [numbers(x[k], y[k]) for k in x]
        return None if any(d is None for d in ds) else max(ds, default=0.0)
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return None
        ds = [numbers(a, b) for a, b in zip(x, y)]
        return None if any(d is None for d in ds) else max(ds, default=0.0)
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        return abs(float(x) - float(y))
    return 0.0 if x == y else None


def main():
    if not os.path.isdir(EXP):
        print(f"no expected/ folder beside scripts/ ({EXP}): nothing to compare. In the repository, outputs/ is the "
              "record itself.")
        return 0
    rows = []

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    which = {"r2": "deciles", "r3": "contributions", "r5": "conditions"}.get(which, which)   # short aliases
    sets = {"semantic_groups": SEMANTIC_GROUPS_EXACT, "cohort": COHORT_EXACT, "deciles": R2_EXACT,
            "contributions": R3_EXACT, "conditions": R5_EXACT, "groupings": GROUPINGS_EXACT}
    assert which in list(sets) + ["all"], f"unknown set {which}: {', '.join(sets)} or all"
    exact_files = [f for name, files in sets.items() if which in (name, "all") for f in files]
    for rel in exact_files:
        ok, what = exact(rel)
        rows.append((rel, ok, what))
    if which not in ("semantic_groups", "all"):
        return report(rows)

    rel = "outputs/random_starts/selection.json"
    try:
        a = json.load(open(os.path.join(ROOT, rel)))
        b = json.load(open(os.path.join(EXP, rel)))
        seeds_a = {e: a[e]["selected"]["seed"] for e in ENCODERS}
        seeds_b = {e: b[e]["selected"]["seed"] for e in ENCODERS}
        d = numbers({e: {k: v for k, v in a[e]["selected"].items() if k != "path"} for e in ENCODERS},
                    {e: {k: v for k, v in b[e]["selected"].items() if k != "path"} for e in ENCODERS})
        ok = seeds_a == seeds_b and d is not None and d <= TOL
        rows.append((rel, ok, f"seeds {'identical' if seeds_a == seeds_b else f'differ: {seeds_a} vs {seeds_b}'}, "
                              f"metrics max |difference| {d if d is not None else 'n/a (structure differs)'}"))
    except FileNotFoundError:
        rows.append((rel, False, "not produced"))

    for e in ENCODERS:
        rel = f"outputs/basis/{e}.npz"
        try:
            a, b = np.load(os.path.join(ROOT, rel)), np.load(os.path.join(EXP, rel))
            d = max(float(np.abs(a[k] - b[k]).max()) for k in ("W", "mu", "sigma"))
            rows.append((rel, d <= TOL, f"W, mu, sigma max |difference| {d:.1e}"))
        except FileNotFoundError:
            rows.append((rel, False, "not produced"))

    rel = "outputs/xenc/summary.json"
    try:
        d = numbers(json.load(open(os.path.join(ROOT, rel))), json.load(open(os.path.join(EXP, rel))))
        rows.append((rel, d is not None and d <= TOL, f"max |difference| {d if d is not None else 'n/a (structure differs)'}"))
    except FileNotFoundError:
        rows.append((rel, False, "not produced"))

    return report(rows)


def report(rows):
    w = max(len(r[0]) for r in rows)
    for rel, ok, what in rows:
        print(f"  {'MATCH' if ok else 'DIFF '}  {rel:{w}s}  {what}")
    n_bad = sum(1 for _, ok, _ in rows if not ok)
    print(f"\n{len(rows) - n_bad} of {len(rows)} match the files of record" +
          ("" if not n_bad else ". Where only W differs in the last digits, the fit ran on another CPU or other "
                                "library versions (README, 'Device dependence')."))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
