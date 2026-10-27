#!/usr/bin/env python
"""figure4_source_data.py -- Figure 4 Source Data: the outliers under the semantic groups and under the factor model
with 25 factors fitted to the answers, under each of the 6 encoders.

SHEETS
  Sets       per encoder and set (semantic groups only, both, factor model only): the participants, and the % for whom one
             questionnaire holds at least half of the squared distance under each distance (stand-alone questions
             counted under their source study); blank where not computed or withheld
  Outcomes   per outcome and encoder: each set's value adjusted for age and sex (one model of the 3 sets, standardized to
             their pooled age and sex) and the difference semantic groups only minus factor model only from the same
             model, with 95% CI, p and the Holm-adjusted p (figure4.holm: per encoder, over every outcome whose
             difference is written), as Figure 4 prints them. % for yes/no outcomes and their difference (percentage
             points), % of visits for the emergency department share, the mean number for thresholds met, minutes for
             sleep (the median for sleep variation)
  Factors    per factor (both rotations) and set: the participants with the factor among the 3 largest terms of their
             squared distance under the factor model, as released
  Groups     the same for every semantic group of each encoder under its own distance
Rows and cells withheld under the dissemination policy are blank (with a note in Outcomes).

Reads   the downloads factor_outliers (outliers_sets.csv, outliers_outcomes.csv, outliers_units.csv), factor_outliers_ed,
        factor_sources (sources_outlier_sets.csv)
Writes  output/supplementary/Figure4_source_data.xlsx (+ csv/)
Run     python figure4_source_data.py
"""
import os

import numpy as np
import pandas as pd

import figure4 as fo
from common import WB
from workbook import GLAB, names, write

FSET = {"enc_only": "Semantic groups only", "both": "Both", "fac_only": "Factor model only"}
OUTCOMES = [("thresholds_met", "Screening thresholds met", "mean No.", 1, 2),
            ("access", "Any barrier to care", "%", 100, 1),
            ("cost", "Cost barrier to care", "%", 100, 1),
            ("avoid", "Delayed care over clinicians' race, religion, or language", "%", 100, 1),
            ("ed_share_coded", "Emergency department share of visits (coded care)", "% of visits", 100, 1),
            ("sleep_mean_min", "Sleep per night", "min", 1, 1),
            ("sleep_variation_min", "Sleep variation (median)", "min", 1, 1),
            ("gad", "Generalized anxiety disorder", "%", 100, 1),
            ("adhd", "Attention-deficit/hyperactivity disorder", "%", 100, 1),
            ("ptsd", "Posttraumatic stress disorder", "%", 100, 1),
            ("panic", "Panic disorder", "%", 100, 1),
            ("hyperchol", "Hypercholesterolemia", "%", 100, 1),
            ("asthma", "Asthma", "%", 100, 1),
            ("breast", "Breast cancer", "%", 100, 1),
            ("prostate", "Prostate cancer", "%", 100, 1)]
README = [("Figure 4 Source Data", "Outliers under the semantic groups and under the factor model with 25 factors "
                                   "fitted to the answers, under each of the 6 encoders (Figure 4 and Figures "
                                   "S5 and S6)."),
          ("Outliers", "The 4518 participants (5%) with the largest distances under a grouping. Sets: outliers under "
                       "the semantic groups only, under both groupings, and under the factor model only."),
          ("Sheet Sets", "Participants in each set (Figure A) and the percentage for whom one questionnaire holds at "
                         "least half of the squared distance, under the semantic groups' and the factor model's "
                         "distance, each stand-alone question counted under its source study (Figure B). Blank: not "
                         "computed (both, factor model's distance) or withheld (small counts)."),
          ("Sheet Outcomes", "Every outcome adjusted for age and sex in each set, and the difference semantic groups "
                             "only minus factor model only from the same model, with p and the Holm-adjusted p over "
                             "the encoder's outcomes (Figure C-I)."),
          ("Sheet Factors", "Per factor (both rotations; F numbers of Table S4, oblimin = part A, varimax = "
                            "part B) and set: participants with the factor among the 3 largest terms of their squared "
                            "distance under the factor model (Figure S5: oblimin)."),
          ("Sheet Groups", "The same for every semantic group of each encoder under its own distance (Figure "
                           "S6: the prespecified encoder)."),
          ("Blank", "Withheld: a count of 1 to 20 participants, or a value that would let one be derived.")]


def fmt_p(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    if p > 0.99:
        return ">0.99"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def ci(v, lo, hi, sc, d):
    """A value with its 95% CI, scaled by sc, d decimals; blank when withheld."""
    if not np.isfinite(v):
        return ""
    return f"{sc * v:.{d}f} ({sc * lo:.{d}f} to {sc * hi:.{d}f})"


def unit_sheet(U, labels, cut=False):
    """Top-3 rows: every factor (both rotations) or every group (each encoder), per set, as released."""
    small = U.n_top3.dropna()
    assert not ((small >= 1) & (small <= 20)).any(), "a top-3 count of 1 to 20 would be published"
    rows = []
    for r in U.itertuples():
        fac = r.grouping == "factor"
        rows.append({**({"cut": "Outliers (top 5%)" if r.cut == "top 5%" else "Decile 10 (top 10%)"} if cut else {}),
                     "encoder": GLAB[r.encoder], "set": labels[r.set],
                     "rotation": r.rotation if fac else "", "unit": f"{'F' if fac else 'G'}{int(r.unit)}",
                     "core_constructs": names(r.name), "participants_in_set": int(r.n_set),
                     "top3_contributor_n": r.n_top3, "top3_contributor_pct": r.pct_top3})
    return pd.DataFrame(rows)


def sets_sheet():
    S = pd.read_csv(os.path.join(fo.SRC, "outliers_sets.csv"))
    S = S[S.factor_model == fo.FM]
    H = pd.read_csv(os.path.join(fo.SRC_SH, "sources_outlier_sets.csv")).set_index(["encoder", "set", "distance"])
    return pd.DataFrame([{"encoder": GLAB[r.encoder], "set": FSET[r.set], "participants": int(r.n_set),
                          "one_questionnaire_half_semantic_distance_pct":
                              H.one_source_half_pct.get((r.encoder, r.set, "encoder"), np.nan),
                          "one_questionnaire_half_factor_distance_pct":
                              H.one_source_half_pct.get((r.encoder, r.set, "factor"), np.nan)}
                         for r in S.itertuples()])


def outcomes_sheet():
    O = fo.load_outcomes()
    O = O[O.factor_model == fo.FM]
    H = fo.holm(O).set_index(["outcome", "encoder"])
    rows = []
    for oc, label, unit, sc, d in OUTCOMES:
        for e in fo.ENCODERS:
            r = O[(O.outcome == oc) & (O.encoder == e)].set_index("set")
            c = H.loc[(oc, e)]
            withheld = not np.isfinite(c["adj"])
            row = {"Outcome": label, "Unit": unit if not unit.startswith("%") else f"{unit} (difference: percentage points)",
                   "Encoder": e}
            for st, name in FSET.items():
                row[f"{name}, n"] = "" if not np.isfinite(r.loc[st, "n"]) else int(r.loc[st, "n"])
                row[f"{name}, adjusted (95% CI)"] = ci(r.loc[st, "adj"], r.loc[st, "adj_lo"], r.loc[st, "adj_hi"], sc, d)
            row["Difference, semantic only minus factor only (95% CI)"] = ci(c["adj"], c["adj_lo"], c["adj_hi"], sc, d)
            row["p"] = fmt_p(c["p"])
            row["p, Holm-adjusted"] = fmt_p(c["p_holm"])
            row["Holm-adjusted p < 0.05"] = "" if withheld else ("yes" if c["p_holm"] < 0.05 else "no")
            row["Note"] = "withheld (small counts)" if withheld else ""
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    U = pd.read_csv(os.path.join(fo.SRC, "outliers_units.csv"))
    U = U[(U.factor_model == fo.FM) & U.grouping.isin(["factor", "semantic groups"])]
    write("Figure4_source_data", README,
          [("Sets", sets_sheet()), ("Outcomes", outcomes_sheet()),
           ("Factors", unit_sheet(U[U.grouping == "factor"], FSET)),
           ("Groups", unit_sheet(U[U.grouping == "semantic groups"].assign(rotation=""), FSET))])


if __name__ == "__main__":
    main()
