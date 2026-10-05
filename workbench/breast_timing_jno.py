#!/usr/bin/env python3
"""breast_timing_jno.py -- when the first breast cancer code was recorded, relative to the survey, among the women of
the breast cancer population of Figure 4 (quoted in the Results). It applies the timing split of phecode_screened.py
to the breast cancer label (CA_105 without Z86.000) and population (women with at least 1 ICD event and a mammogram
in the record; conditions_jno.py).

  anchor   the survey date: the median date of the participant's 151 kept answers, the anchor of age
           (phecode_screened.survey_anchor, cached in survey_anchor_local.csv)
  first    the first breast cancer code: PheTK's first_event_date for CA_105, counted with conditions_jno.py's map
  before   first code on or before the anchor; after = later
Written: the cases, the cases without a dated survey answer (left out), before and after, the share before, and the
years between first code and anchor (10th, 25th, 50th, 75th, 90th percentiles) in each group.

OUTPUTS  screen_out/breast_timing_jno/breast_timing_jno.csv      1 row per timing group
         screen_out/breast_timing_jno/breast_timing_jno_meta.json
         screen_out/breast_timing_jno_<CDR>.zip                  those 2 (pii_check; counts as they are)

Run (after conditions_jno.py):  python3 breast_timing_jno.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conditions_jno import CDR, CODE, COUNTS_JNO, ID_COL, OUT_DIR, load, zip_out   # noqa: E402
from phecode_screened import survey_anchor                                # noqa: E402  the same anchor

OUT = os.path.join(OUT_DIR, "breast_timing_jno")
PCT = (10, 25, 50, 75, 90)


def first_code_date(idx, phecode):
    """PheTK's first_event_date for one phecode, per person in idx order (NaT without an event)."""
    P = pd.read_csv(COUNTS_JNO, sep="\t", dtype={"person_id": str, "phecode": str},
                    usecols=["person_id", "phecode", "first_event_date"])
    p = P[P["phecode"] == phecode].set_index("person_id")
    assert p.index.is_unique, f"{phecode}: more than 1 row per person in {COUNTS_JNO}"
    return pd.to_datetime(p["first_event_date"]).reindex(idx).values


def main():
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)
    A = load(idx)
    d = A["den_breast"]
    case = d & (np.nan_to_num(A["y_breast"], nan=0) == 1)
    anchor = survey_anchor(idx).values
    fe = first_code_date(idx, CODE["breast"])
    assert not pd.isna(fe[case]).any(), "a breast cancer case without a first_event_date"
    no_anchor = case & pd.isna(anchor)
    dated = case & ~pd.isna(anchor)
    gap = np.full(len(idx), np.nan)
    gap[dated] = (pd.to_datetime(anchor[dated]) - pd.to_datetime(fe[dated])).days / 365.25   # > 0: code before survey
    before, after = dated & (gap >= 0), dated & (gap < 0)
    rows = []
    for when, m in (("before", before), ("after", after)):
        r = {"timing": when, "n_cases": int(m.sum()), "n_cases_dated": int(dated.sum()),
             "share": float(m.sum() / dated.sum()) if dated.sum() else np.nan}
        if m.sum():
            r.update({f"years_p{p}": float(v) for p, v in zip(PCT, np.percentile(np.abs(gap[m]), PCT))})
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "breast_timing_jno.csv"), index=False)
    json.dump({"cdr": CDR, "population": "women with at least 1 ICD event and a mammogram in the record (conditions_jno)",
               "label": "CA_105 without Z86.000, count >= 2", "n_cases": int(case.sum()),
               "n_cases_without_dated_answer": int(no_anchor.sum()),
               "anchor": "median date of the 151 kept answers (phecode_screened.survey_anchor)",
               "before": "first CA_105 code on or before the anchor", "counts": "written as they are; pii_check"},
              open(os.path.join(OUT, "breast_timing_jno_meta.json"), "w"), indent=2)
    print(f"[breast_timing_jno] breast cancer cases {int(case.sum()):,} | without a dated survey answer "
          f"{int(no_anchor.sum()):,}")
    for r in D.itertuples():
        print(f"  {r.timing:6s} {r.n_cases:>6,} of {r.n_cases_dated:,} ({100 * r.share:.1f}%)" +
              (f" | years apart, median {r.years_p50:.1f} (25th {r.years_p25:.1f}, 75th {r.years_p75:.1f})"
               if r.n_cases else ""))
    zpath = zip_out(OUT, ["breast_timing_jno.csv", "breast_timing_jno_meta.json"], "breast_timing_jno")
    print(f"\n[breast_timing_jno] done\n  download -> {zpath}")


if __name__ == "__main__":
    main()
