#!/usr/bin/env python
"""appendix4_table1.py -- Appendix 4-table 1, the 25 factors of the factor model under oblimin and varimax rotation,
one row per factor, named as the semantic groups are.

A factor's core items are the fixed-point participation-ratio core of its column of the rotated loadings (the oblimin
pattern loadings; the varimax loadings), the rule of the semantic groups, as the Workbench applied it (step 9). A
factor is named by the constructs of all its core items, with the number of core items per construct, largest first.
Factors are numbered in the order the rotation returns them, so the same number means different factors under the 2
rotations; the paper's F numbers are the oblimin numbers. A core spans questionnaires when its items come from 2 or
more, each stand-alone question counted under the study it was taken from (data/standalone_sources.csv).

Reads   the download factor_decile10 (decile10_factors.csv), data/items_151.csv, data/standalone_sources.csv
Writes  elife/tables/elife_app_factors.tex (longtable, caption inside); outputs/tables/factors.csv (with the core items,
        read by appendix5_table1.py and supplementary_file1.py)
Run     python appendix4_table1.py
"""
import os
from collections import Counter

import pandas as pd

from common import ITEMS_CSV, SOURCES_CSV, TABLE_DATA, WB, display_listed, esc, write_table

SRC = os.path.join(WB, "factor_decile10", "decile10_factors.csv")
ROTATIONS = [("oblimin", "A. Oblimin rotation (the factors may correlate), the paper's F numbers"),
             ("varimax", "B. Varimax rotation (the factors are uncorrelated), a check")]
CAPTION = (r"\textbf{\AppTabFactors{}. The factors of the factor model under oblimin and varimax rotation.} One row per "
           r"factor of the factor model with 25 factors, fitted to the answers. A factor's core items are the items "
           r"with its largest loadings, found by the participation-ratio rule used for the semantic groups "
           r"(Appendix~\AppGroups{}), and each factor is named by the constructs of all its core items, with the number "
           r"of core items per construct, largest first. Factors are numbered in the order the rotation returns them, "
           r"so a number refers to different factors under the two rotations, and the factor numbers in the text are "
           r"those of the oblimin rotation. Each stand-alone question is counted under the study it was taken from. The "
           r"questionnaire and source abbreviations are spelled out "
           r"in Appendix~\AppGroups{}, and the core items of every factor are listed in Supplementary file~\SuppItems{}.")
HEAD = "Factor & Core items & Constructs of the core items & Questionnaires of the core items"


def factor_rows():
    """One row per rotation and factor: its core items, their constructs and questionnaires; and per rotation the
    counts the table's headings give."""
    d = pd.read_csv(SRC)
    d = d[(d.factor_model == "factor") & (d.k == 25)]
    it = pd.read_csv(ITEMS_CSV).set_index("item")
    abbr = it["instrument"].str.extract(r"\(([^()]*)\)\s*$")[0]
    single = it["instrument"] == "single item"
    study = pd.read_csv(SOURCES_CSV).set_index("item")["source"]
    assert abbr[~single].notna().all(), f"no abbreviation for {sorted(set(it.instrument[~single & abbr.isna()]))}"
    rows, head = [], {}
    for rot, _ in ROTATIONS:
        x = d[d.rotation == rot].sort_values("factor")
        assert len(x) == 25 and list(x.factor) == list(range(1, 26)), f"{rot}: factors are not numbered 1 to 25"
        n_span = 0
        for r in x.itertuples():
            core = r.core_items.split()
            qs = sorted({abbr[i] if not single[i] else study[i] for i in core})
            spans = len(qs) >= 2
            n_span += spans
            rows.append({"rotation": rot, "factor": int(r.factor), "n_core_items": len(core),
                         "constructs": r.name.replace(":", " ").replace("; ", ", "), "questionnaires": ", ".join(qs),
                         "spans": spans, "core_items": r.core_items})
        nc = Counter(len(n.split("; ")) for n in x.name)
        head[rot] = {"one": nc.get(1, 0), "two": nc.get(2, 0), "more": sum(v for k, v in nc.items() if k >= 3),
                     "span": n_span}
    return pd.DataFrame(rows), head


def main():
    R, head = factor_rows()
    os.makedirs(TABLE_DATA, exist_ok=True)
    R.to_csv(os.path.join(TABLE_DATA, "factors.csv"), index=False)
    L = ["\\begingroup\\footnotesize\\setlength{\\tabcolsep}{4pt}",
         "\\begin{longtable}{@{}r r >{\\raggedright\\arraybackslash}p{0.44\\linewidth} "
         ">{\\raggedright\\arraybackslash}p{0.30\\linewidth}@{}}",
         f"\\caption{{{CAPTION}}}\\label{{apptab:factors}}\\\\",
         f"\\toprule {HEAD} \\\\ \\midrule",
         f"\\endfirsthead \\toprule {HEAD} \\\\ \\midrule \\endhead",
         "\\bottomrule \\endfoot"]
    for rot, title in ROTATIONS:
        h = head[rot]
        L.append(f"\\multicolumn{{4}}{{@{{}}p{{0.97\\linewidth}}}}{{\\textbf{{{title}}} \\quad 25 factors, {h['one']} on "
                 f"1 construct, {h['two']} on 2, {h['more']} on 3 or more, {h['span']} with core items from more than 1 "
                 f"questionnaire}} \\\\[2pt]")
        for r in R[R.rotation == rot].itertuples():
            L.append(f"F{r.factor} & {r.n_core_items} & {esc(display_listed(r.constructs))} & {esc(r.questionnaires)} \\\\")
        L.append("\\addlinespace[4pt]")
    L += ["\\end{longtable}", "\\endgroup"]
    write_table("elife_app_factors", L, "appendix4_table1.py")
    print(head)


if __name__ == "__main__":
    main()
