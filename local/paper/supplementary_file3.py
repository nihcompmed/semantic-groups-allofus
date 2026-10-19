#!/usr/bin/env python
"""supplementary_file3.py -- Supplementary file 3: the participants included against the rest of the All of Us data
release.

Populations: included (the cohort); a full record for all 151 items but not included (sex at birth other than female or
male, or fewer than 146 items answered); the rest of the release. Each characteristic is No. (%) or the stated summary,
with the standardized difference of each other population from the included participants (no P values).

Reads   the download selection (selection_table.csv)
Writes  elife/supplementary/Supplementary_file_3.xlsx (+ csv/)
Run     python supplementary_file3.py
"""
import os

import pandas as pd

from common import WB
from supplementary_file2 import LLAB, VLAB
from workbook import npct, write

PLAB = {"included": "Included (cohort)", "excluded_record": "Full record, not included", "rest": "Rest of the release"}
README = [("Supplementary file 3", "Participants included vs the rest of the All of Us data release."),
          ("Populations", "Included = the 90,351 analyzed; full record, not included = a record for all 151 items "
                          "but excluded (sex at birth other than female or male, or fewer than 146 items answered); "
                          "rest = everyone else in the release."),
          ("Standardized difference", "Difference from the included participants in SD units (no p values)."),
          ("Characteristics", "Age on January 1, 2025; sex at birth; income, education, employment, and health "
                              "insurance from The Basics survey, nonresponse as its own category; any health record "
                              "and any Fitbit data.")]


def main():
    S = pd.read_csv(os.path.join(WB, "selection", "selection_table.csv"))
    rows = []
    for (var, lvl), d in S.groupby(["variable", "level"], sort=False, dropna=False):
        row = {"characteristic": VLAB[var], "level": "" if pd.isna(lvl) else LLAB.get(lvl, lvl)}
        for p, lab in PLAB.items():
            x = d[d.population == p]
            if not len(x):
                continue
            x = x.iloc[0]
            row[lab] = x.value if pd.notna(x.value) else (npct(x.n, x.percent) if var != "n" else f"{int(x.n):,}")
            if p != "included":
                row[f"std_diff_{p}"] = round(float(x.std_diff), 3) if pd.notna(x.std_diff) else ""
        rows.append(row)
    write("Supplementary_file_3", README, [("Selection", pd.DataFrame(rows))])


if __name__ == "__main__":
    main()
