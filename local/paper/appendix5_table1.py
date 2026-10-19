#!/usr/bin/env python
"""appendix5_table1.py -- Appendix 5-table 1, the 8 recorded conditions under the semantic groups and under the factor
model.

The units are the k semantic groups of each of the 6 encoders (download conditions) and the 25 factors of the factor
model under oblimin and varimax (download factor_conditions, fitted with the same models: each condition in its
population, age and age squared, and sex where both sexes are in it, the unit scores standardized over the cohort, the
item on a diagnosed life-threatening illness at weight 0, the *_without columns). A log odds ratio per SD is reported
as its absolute value (the sign of a unit is arbitrary), with its CI as sizes (the absolute limits in increasing order,
lower limit 0 when the CI includes 0).

PER GROUPING AND CONDITION
  largest alone      the largest |log OR| of a unit on its own (k regressions with 1 unit), and its unit
  largest with all   the largest |log OR| of a unit adjusted for the other units (1 regression with all k), and its unit
  share              the largest squared log OR with all units / the sum over the k units
  units to half      the smallest number of units, largest squared log OR first, whose sum reaches half of the total
CHECKS  the factor numbers equal Appendix 4-table 1's (the construct names of every factor, both rotations); 8 conditions
        per grouping; k = 25 for the factor model.

Reads   the downloads conditions (condition_groups.csv) and factor_conditions (factor_condition_groups.csv); the factors
        of appendix4_table1.py
Writes  elife/tables/elife_app_factor_conditions.tex (longtable, caption inside); outputs/tables/factor_conditions.csv
        (every value, full precision, with CIs)
Run     python appendix5_table1.py
"""
import os

import numpy as np
import pandas as pd

from appendix4_table1 import factor_rows
from common import CONDITIONS, ENCODERS, PRIMARY, TABLE_DATA, WB, write_table

SEM = os.path.join(WB, "conditions", "condition_groups.csv")
FAC = os.path.join(WB, "factor_conditions", "factor_condition_groups.csv")
FACTOR = {"oblimin": "factor k25 oblimin", "varimax": "factor k25 varimax"}
CONDS = [(c[0], c[2] if c[0] in ("breast", "prostate") else c[1]) for c in CONDITIONS]   # the cancers without "(women)"
CAPTION = (r"\textbf{\AppTabFactorConditions{}. The recorded conditions under the semantic groups and under the factor "
           r"model.} The units are the semantic groups of each encoder and the 25 factors of the factor model under each "
           r"rotation. Each condition was related to each unit on its own and to each unit adjusted for the other units of "
           r"its grouping, by the regressions of Appendix~\AppConditions{}. A, the largest absolute log odds ratio per SD "
           r"of a unit on its own, and B, of a unit adjusted for the other units, with the unit in parentheses for the "
           r"prespecified encoder and for the factor model. C, the largest unit's squared log odds ratio divided by the sum "
           r"over all units of the grouping, in the regression with all units. The six-encoder column gives the range over the six encoders. Factor numbers are those "
           r"of \AppTabFactors{}, part A for oblimin and part B for varimax, so the same number refers to different "
           r"factors under the two rotations.")
HEAD = ("\\toprule & \\multicolumn{2}{c}{Semantic groups} & \\multicolumn{2}{c}{Factor model} \\\\ "
        "\\cmidrule(lr){2-3} \\cmidrule(l){4-5} Condition & Prespecified encoder & Six encoders & Oblimin & Varimax "
        "\\\\ \\midrule")
PANELS = [("alone", "A. Largest absolute log odds ratio per SD, each unit on its own", True),
          ("with_all", "B. Largest absolute log odds ratio per SD, adjusted for the other units", True),
          ("share", "C. Share of the largest unit, in the regression with all units", False)]


def size_ci(lo, hi):
    """The CI of an absolute log odds ratio: the absolute limits in increasing order, 0 when the CI includes 0."""
    lo, hi = np.log(lo), np.log(hi)
    return (0.0, max(abs(lo), abs(hi))) if lo <= 0 <= hi else tuple(sorted((abs(lo), abs(hi))))


def measures(x):
    """x = the rows of 1 grouping and 1 condition (1 per unit)."""
    b_all = np.log(x.or_without.to_numpy())
    b_one = np.log(x.or_alone_without.to_numpy())
    unit = x.group.to_numpy() + 1                                    # G and F numbers = 0-based index + 1
    ia, iw = int(np.argmax(np.abs(b_one))), int(np.argmax(np.abs(b_all)))
    sq = np.sort(b_all ** 2)[::-1]
    r = x.iloc[ia], x.iloc[iw]
    a_lo, a_hi = size_ci(r[0].or_alone_lo_without, r[0].or_alone_hi_without)
    w_lo, w_hi = size_ci(r[1].or_lo_without, r[1].or_hi_without)
    return {"k": len(x),
            "alone": abs(b_one[ia]), "alone_lo": a_lo, "alone_hi": a_hi, "alone_unit": int(unit[ia]),
            "with_all": abs(b_all[iw]), "with_all_lo": w_lo, "with_all_hi": w_hi, "with_all_unit": int(unit[iw]),
            "share": sq[0] / sq.sum(), "units_to_half": int(np.argmax(np.cumsum(sq) >= sq.sum() / 2)) + 1}


def main():
    s = pd.read_csv(SEM)
    f = pd.read_csv(FAC)
    ft, _ = factor_rows()
    for rot, name in FACTOR.items():                                 # the F numbers are Appendix 4-table 1's
        got = (f[(f.encoder == name) & (f.condition == "gad")].sort_values("group")
               .construct_profile.str.replace(":", " ").str.replace("; ", ", ").tolist())
        want = ft[ft.rotation == rot].sort_values("factor").constructs.tolist()
        assert got == want, f"{rot}: the factor numbers differ from Appendix 4-table 1"
    rows = []
    for grouping, d, src in ([(e, s, e) for e in ENCODERS] + [(f"factor {r}", f, n) for r, n in FACTOR.items()]):
        g = d[d.encoder == src]
        assert sorted(g.condition.unique()) == sorted(c for c, _ in CONDS), f"{grouping}: conditions"
        for c, _ in CONDS:
            x = g[g.condition == c].sort_values("group")
            assert list(x.group) == list(range(len(x))), f"{grouping} {c}: units not numbered 0..k-1"
            if grouping.startswith("factor"):
                assert len(x) == 25, f"{grouping} {c}: k = {len(x)}"
            rows.append({"grouping": grouping, "condition": c, **measures(x)})
    R = pd.DataFrame(rows)
    os.makedirs(TABLE_DATA, exist_ok=True)
    R.to_csv(os.path.join(TABLE_DATA, "factor_conditions.csv"), index=False)

    f2 = lambda v: f"{v:.2f}"

    def rng(v):
        lo, hi = f2(v.min()), f2(v.max())
        return lo if lo == hi else f"{lo}--{hi}"

    sem = R[R.grouping.isin(ENCODERS)]
    L = ["\\begingroup\\footnotesize\\setlength{\\tabcolsep}{4pt}",
         "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{0.30\\linewidth} l l l l@{}}",
         f"\\caption{{{CAPTION}}}\\label{{apptab:factorconditions}}\\\\",
         HEAD, "\\endfirsthead", HEAD, "\\endhead", "\\bottomrule \\endfoot"]
    for col, title, unit in PANELS:
        L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{{title}}}}} \\\\[2pt]")
        for c, label in CONDS:
            def cell(grouping, prefix):
                r = R[(R.grouping == grouping) & (R.condition == c)].iloc[0]
                return f2(r[col]) + (f" ({prefix}{r[col + '_unit']})" if unit else "")
            L.append(f"{label} & {cell(PRIMARY, 'G')} & {rng(sem[sem.condition == c][col])} & "
                     f"{cell('factor oblimin', 'F')} & {cell('factor varimax', 'F')} \\\\")
        L.append("\\addlinespace[4pt]")
    L += ["\\end{longtable}", "\\endgroup"]
    write_table("elife_app_factor_conditions", L, "appendix5_table1.py")


if __name__ == "__main__":
    main()
