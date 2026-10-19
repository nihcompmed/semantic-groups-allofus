#!/usr/bin/env python
"""supplementary_file2.py -- Supplementary file 2: the characteristics of the cohort, of the populations each outcome
is measured on, and of the outliers under each encoder; and the codes of the tests that define the populations of 3
conditions.

SHEETS
  Participants   one row per characteristic and level, one column per population: No. (%) or the stated summary. Race
                 and ethnicity were not analyzed and are not reported.
  Test codes     the codes of the recorded tests (lipid panel, mammogram, PSA test) that define the populations for
                 hypercholesterolemia, breast cancer, and prostate cancer, as ../workbench/step5_outcomes.py lists them,
                 with their concept names (data/condition_test_codes.csv)

Reads   the download cohort_table (cohort_table.csv), data/condition_test_codes.csv
Writes  elife/supplementary/Supplementary_file_2.xlsx (+ csv/)
Run     python supplementary_file2.py
"""
import os

import pandas as pd

from common import DATA, ENCODERS, WB
from workbook import GLAB, npct, write

VLAB = {"n": "Participants", "age": "Age, years", "age_at_cutoff": "Age on January 1, 2025, years", "sex": "Sex at birth",
        "criteria": "Screening thresholds", "income": "Household income, US$", "education": "Education",
        "employment": "Employment", "insurance": "Health insurance", "ehr": "Any health record", "fitbit": "Any Fitbit data"}
LLAB = {"<25k": "<25,000", "25k to <50k": "25,000 to <50,000", "50k to <100k": "50,000 to <100,000",
        "100k to <150k": "100,000 to <150,000", "150k or more": "150,000 or more", "any of 6 met": "Any of the six met",
        "mean met, all 6 evaluable": "Mean number met, all six evaluable",
        "none met, all 6 evaluable": "None met, all six evaluable", "female (sex at birth)": "Female", "yes": "Yes"}
POPS = ["cohort", "survey_linked", "records_linked", "icd_linked", "sleep", "flagged"] + [f"flagged_{e}" for e in ENCODERS[1:]]
PLAB = {"cohort": "Cohort", "survey_linked": "Answered the Health Care Access and Utilization survey",
        "records_linked": "Any health record", "icd_linked": "Coded care (at least 1 visit and 1 diagnosis code)",
        "sleep": "At least 14 nights of Fitbit sleep", "flagged": "Outliers, gte-large-en-v1.5 (prespecified)",
        **{f"flagged_{e}": f"Outliers, {GLAB[e]}" for e in ENCODERS[1:]}}
README = [("Supplementary file 2", "Characteristics of the cohort, of the populations on which each outcome was "
                                   "measured, and of the outliers under each encoder."),
          ("Sheet Participants", "No. (%) or the stated summary. Income, education, employment, and health insurance from "
                                 "The Basics survey, nonresponse as its own category. Race and ethnicity were not "
                                 "analyzed and are not reported. Screening thresholds: the 6 published cutoffs (Methods)."),
          ("Sheet Test codes", "The codes of the recorded tests that define the populations for hypercholesterolemia, "
                               "breast cancer, and prostate cancer.")]


def participants():
    T = pd.read_csv(os.path.join(WB, "cohort_table", "cohort_table.csv"))
    T = T[T.variable != "race_ethnicity"]
    rows = []
    for var, lvl in T[["variable", "level"]].drop_duplicates().itertuples(index=False):
        row = {"characteristic": VLAB[var], "level": "" if pd.isna(lvl) else LLAB.get(lvl, lvl)}
        for p in POPS:
            x = T[(T.population == p) & (T.variable == var) & ((T.level == lvl) | (T.level.isna() & pd.isna(lvl)))]
            if not len(x):
                row[PLAB[p]] = ""
                continue
            x = x.iloc[0]
            if pd.notna(x.value):
                row[PLAB[p]] = x.value
            elif pd.notna(x.percent) and var != "n":
                row[PLAB[p]] = npct(x.n, x.percent)
            else:
                row[PLAB[p]] = f"{int(float(x.n)):,}" if str(x.n).replace(".", "").isdigit() else str(x.n)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    codes = pd.read_csv(os.path.join(DATA, "condition_test_codes.csv"))
    write("Supplementary_file_2", README, [("Participants", participants()), ("Test codes", codes)])


if __name__ == "__main__":
    main()
