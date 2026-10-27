#!/usr/bin/env python
"""figure6_source_data.py -- Figure 6 Source Data: the participants below every screening threshold under the
semantic groups and under the factor model with 25 factors, under each of the 6 encoders, at both cuts over the whole
cohort, the outliers (top 5%) and decile 10 (top 10%).

SETS, among the participants below every threshold, per encoder and cut: found (flagged under the encoder and the factor
model), missed (the encoder only), factor-only.
SHEETS
  Sets       per cut and encoder: the 3 sizes, the share of the encoder's set found by the factor model, mean age and %
             women (unadjusted), the % for whom one questionnaire holds at least half of the squared distance under the
             semantic groups (stand-alone questions counted under their source study)
  Outcomes   per outcome, cut and encoder: each set's value adjusted for age and sex (one model of the 3 sets), the 2
             differences missed minus found and missed minus factor-only from the same model, with 95% CI, p and the
             Holm-adjusted p (figure6.holm: per encoder and cut, over both differences of every outcome written), as
             Figure 6 prints them; units as in Figure 4 Source Data
  Factors    per factor (both rotations), cut and set: the participants with the factor among the 3 largest terms of
             their squared distance under the factor model, as released
  Groups     the same for every semantic group of each encoder under its own distance
Values withheld under the dissemination policy are blank, with a note.

Reads   the downloads factor_below (below_sets.csv, below_outcomes.csv, below_units.csv) and factor_sources
        (sources_below_sets.csv)
Writes  output/supplementary/Figure6_source_data.xlsx (+ csv/)
Run     python figure6_source_data.py
"""
import os

import numpy as np
import pandas as pd

import figure4 as fo
import figure6 as fb
from figure4_source_data import ci, fmt_p, unit_sheet
from workbook import write

BSET = {"enc_only": "Missed", "both": "Found", "fac_only": "Factor-only"}
SETS = list(BSET.items())
PAIRS = [("enc_only - both", "Missed minus found"), ("enc_only - fac_only", "Missed minus factor-only")]
CUTS = [("top 5%", "Outliers (top 5%)"), ("top 10%", "Decile 10 (top 10%)")]
OUTCOMES = [("access", "Any barrier to care", "%", 100, 1),
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
README = [("Figure 6 Source Data", "Screen-negative participants under the semantic groups and "
                                   "under the factor model with 25 factors, under each of the 6 encoders (Figure "
                                   "6 and Figures S8 and S9)."),
          ("Participants", "The 26,812 screen-negative participants, who could be evaluated on all 6 screening thresholds "
                           "and met none. Sets: found (flagged under both the semantic groups and the factor model), "
                           "missed (under the semantic groups only), factor-only (under "
                           "the factor model only), at 2 cuts over the whole cohort: the outliers (top 5%) and decile 10 "
                           "(top 10%)."),
          ("Sheet Sets", "Participants in each set, the percentage found, mean age and percentage of women "
                         "(unadjusted), and the percentage for whom one questionnaire holds at least half of the squared "
                         "distance under the semantic groups, each stand-alone question counted under its source study "
                         "(Figure A, B)."),
          ("Sheet Outcomes", "Every outcome adjusted for age and sex in each set, and the differences missed minus "
                             "found and missed minus factor-only from the same model, with p and the Holm-adjusted p "
                             "over the encoder's differences at that cut (Figure C)."),
          ("Sheet Factors", "Per factor (both rotations) and set: participants with the factor among the 3 largest terms "
                            "of their squared distance under the factor model (Figure S9: oblimin, decile 10)."),
          ("Sheet Groups", "The same for every semantic group of each encoder under its own distance (Figure "
                           "S8: the prespecified encoder, decile 10)."),
          ("Blank", "Withheld: a count of 1 to 20 participants, or a value that would let one be derived.")]
r1 = lambda x: round(float(x), 1) if np.isfinite(x) else np.nan


def sets_sheet():
    S = fo.sh_below(pd.read_csv(os.path.join(fb.SRC, "below_sets.csv"))).set_index(["cut", "encoder", "set"])
    rows = []
    for cut, cname in CUTS:
        for e in fo.ENCODERS:
            row = {"Cut": cname, "Encoder": e}
            for st, name in SETS:
                n = S.loc[(cut, e, st), "n_set"]
                row[f"{name}, n"] = "" if not np.isfinite(n) else int(n)
            row["Found by the factor model, % of the encoder's set"] = r1(S.loc[(cut, e, "both"), "found_by_factor_pct"])
            for col, lab in (("mean_age", "mean age, years"), ("female_pct", "women, %"),
                             ("one_source_half_enc_pct", "one questionnaire carries half, %")):
                for st, name in SETS:
                    row[f"{name}, {lab}"] = r1(S.loc[(cut, e, st), col])
            rows.append(row)
    return pd.DataFrame(rows)


def outcomes_sheet():
    O = pd.read_csv(os.path.join(fb.SRC, "below_outcomes.csv"))
    rows = []
    for cut, cname in CUTS:
        H = fb.holm(O, cut).set_index(["outcome", "set", "encoder"])
        for oc, label, unit, sc, d in OUTCOMES:
            for e in fo.ENCODERS:
                r = O[(O.cut == cut) & (O.outcome == oc) & (O.encoder == e)].set_index("set")
                withheld = not np.isfinite(r.loc["enc_only", "adj"]) if "enc_only" in r.index else True
                row = {"Outcome": label, "Unit": unit if not unit.startswith("%") else f"{unit} (difference: percentage points)",
                       "Cut": cname, "Encoder": e}
                for st, name in SETS:
                    n = r.loc[st, "n"] if st in r.index else np.nan
                    row[f"{name}, n"] = "" if not np.isfinite(n) else int(n)
                    row[f"{name}, adjusted (95% CI)"] = (ci(r.loc[st, "adj"], r.loc[st, "adj_lo"], r.loc[st, "adj_hi"], sc, d)
                                                         if st in r.index else "")
                for pair, name in PAIRS:
                    c = H.loc[(oc, pair, e)] if (oc, pair, e) in H.index else None
                    ok = c is not None and np.isfinite(c["adj"])
                    row[f"{name} (95% CI)"] = ci(c["adj"], c["adj_lo"], c["adj_hi"], sc, d) if ok else ""
                    row[f"{name}, p"] = fmt_p(c["p"]) if ok else ""
                    row[f"{name}, p Holm-adjusted"] = fmt_p(c["p_holm"]) if ok else ""
                    row[f"{name}, Holm-adjusted p < 0.05"] = ("yes" if c["p_holm"] < 0.05 else "no") if ok else ""
                row["Note"] = "withheld (small counts)" if withheld else ""
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    U = pd.read_csv(os.path.join(fb.SRC, "below_units.csv"))
    write("Figure6_source_data", README,
          [("Sets", sets_sheet()), ("Outcomes", outcomes_sheet()),
           ("Factors", unit_sheet(U[U.grouping == "factor"], BSET, cut=True)),
           ("Groups", unit_sheet(U[U.grouping == "semantic groups"], BSET, cut=True))])


if __name__ == "__main__":
    main()
