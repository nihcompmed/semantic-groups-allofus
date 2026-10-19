#!/usr/bin/env python
"""figure3_source_data.py -- Figure 3-source data 1: the groups that contribute most to the outliers' distance under
each of the 6 encoders, among all outliers and among the outliers below every threshold.

Per encoder and group: the outliers with the group among the 3 largest terms of their squared distance (No. and %), and
the group's term divided by the squared distance, averaged over the set. Counts of 1 to 20 are written <=20, as
downloaded.

Reads   the download contributors (contributors_top3_all.csv, contributors_all.csv, contributors_top3_below.csv,
        contributors_below.csv)
Writes  elife/supplementary/Figure3_source_data_1.xlsx (+ csv/)
Run     python figure3_source_data.py
"""
import os

import numpy as np
import pandas as pd

from common import ENCODERS, WB
from workbook import GLAB, names, write

SRC = os.path.join(WB, "contributors")
README = [("Figure 3-source data 1", "Groups contributing most to the distance under each of the 6 encoders (Figure 3)."),
          ("Outliers", "The 5% of participants with the largest distances under the encoder (4518); 'Screen-negative "
                       "outliers' = those among them who could be evaluated on all 6 screening thresholds and met none."),
          ("Top-3 contributor", "A group among the 3 largest terms of the participant's squared distance (Methods)."),
          ("Mean share", "The group's term divided by the squared distance, averaged over the set."),
          ("<=20", "Counts of 1 to 20 are written <=20, as downloaded.")]


def main():
    sheets = []
    for name, top, summary in (("All outliers", "contributors_top3_all.csv", "contributors_all.csv"),
                               ("Screen-negative outliers", "contributors_top3_below.csv", "contributors_below.csv")):
        T = pd.read_csv(os.path.join(SRC, top))
        S = pd.read_csv(os.path.join(SRC, summary)).set_index(["encoder", "component"])
        rows = []
        for e in ENCODERS:
            for t in T[T.encoder == e].sort_values("component").itertuples():
                share = S.loc[(e, t.component), "mean_share"]
                rows.append({"encoder": GLAB[e], "group": f"G{t.component + 1}", "core_constructs": names(t.semantic_groups),
                             "outliers": int(t.n_set), "top3_contributor_n": t.n_top, "top3_contributor_pct": t.pct_top,
                             "mean_share_of_squared_distance_pct": round(100 * share, 2) if pd.notna(share) else np.nan})
        sheets.append((name, pd.DataFrame(rows)))
    write("Figure3_source_data_1", README, sheets)


if __name__ == "__main__":
    main()
