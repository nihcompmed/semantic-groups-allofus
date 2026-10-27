#!/usr/bin/env python
"""figure5_source_data.py -- Figure 5 Source Data: every outcome by decile among the participants who could be
evaluated on all 6 screening thresholds and met none, deciles formed within this group, under the 6 encoders; the
conditions by decile; every trend below every threshold. The values, trends and Holm families are those of Figure
2 Source Data (figure2_source_data.py): the below-threshold trends share each encoder's family with the whole-cohort
trends.

Reads   the downloads deciles (below_threshold_outcomes.csv, below_threshold_conditions.csv, trends.csv) and deciles_ed
        (ed_deciles.csv, ed_trend.csv)
Writes  output/supplementary/Figure5_source_data.xlsx (+ csv/)
Run     python figure5_source_data.py
"""
import os

import pandas as pd

from common import ENCODERS, WB
from figure2_source_data import COMMON, condition_sheet, outcome_sheet, trend_sheet, trends
from workbook import write

README = [("Figure 5 Source Data", "Outcomes by decile among the 26,812 screen-negative participants, who could be "
                                   "evaluated on all 6 screening thresholds and met none, deciles formed within this "
                                   "group (Figure 5 "
                                   "and Figure S7)."), *COMMON[:4],
          ("Sheet Outcomes by decile", "Barriers to care, emergency department share (coded care), and sleep, 6 encoders."),
          ("Sheet Conditions by decile", "The 8 recorded conditions, 6 encoders."),
          ("Sheet Trends", "Every trend among screen-negative participants; Holm families shared with the whole-cohort "
                           "trends.")]


def main():
    O = pd.read_csv(os.path.join(WB, "deciles", "below_threshold_outcomes.csv"))
    E = pd.read_csv(os.path.join(WB, "deciles_ed", "ed_deciles.csv"))
    C = pd.read_csv(os.path.join(WB, "deciles", "below_threshold_conditions.csv"))
    main_trends, _, fam = trends()
    write("Figure5_source_data", README,
          [("Outcomes by decile", outcome_sheet(O[O.grouping.isin(ENCODERS)], E, "below_threshold")),
           ("Conditions by decile", condition_sheet(C[C.grouping.isin(ENCODERS + ["population"])])),
           ("Trends", trend_sheet(main_trends, "below_threshold", fam))])


if __name__ == "__main__":
    main()
