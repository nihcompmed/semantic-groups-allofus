#!/usr/bin/env python
"""figure2_source_data.py -- Figure 2 Source Data: every outcome by decile of the distance from the cohort's mean
scores, under the 6 encoders and the 6 other groupings; the conditions by decile; every trend with its Holm-adjusted p
and agreement; sleep by device source.

VALUES are the joint model's adjusted values (decile as 10 categories, age, age squared and sex, sex left out for the 2
cancers, averaged over the outcome's population), rates and shares in percent, 95% CIs by the delta method.
TRENDS, Holm families as the Methods: each encoder's 29 trends (15 in the whole cohort and 14 below every threshold, the
emergency department share on coded care), each other grouping's 7, each encoder's 4 device trends. Agreement = the
prespecified encoder's direction and Holm-adjusted p < 0.05 within the grouping's own family. Cells marked "<=20" in a
download are kept as they are.

The sheets of the deciles, the conditions and the trends are shared with Figure 5 Source Data
(figure5_source_data.py), which gives them below every threshold.

Reads   the downloads deciles (outcomes.csv, conditions.csv, sleep_device.csv, trends.csv) and deciles_ed (ed_deciles.csv,
        ed_trend.csv)
Writes  output/supplementary/Figure2_source_data.xlsx (+ csv/)
Run     python figure2_source_data.py
"""
import os

import numpy as np
import pandas as pd

from common import CONDITIONS, ENCODERS, PRIMARY, WB
from workbook import GLAB, GTYPE, OTHER, holm, pv, write

CLAB = {c[0]: c[1] for c in CONDITIONS}
OLAB = {"thresholds_met": "Screening thresholds met (mean number)", "access": "Any barrier to care",
        "cost": "Cost barrier", "avoid": "Delayed care over clinicians' race, religion, or language",
        "ed_share_pct_coded": "Emergency department share of visits (coded care)", "sleep_mean_min": "Mean minutes asleep",
        "sleep_variation_min": "Sleep variation (night-to-night SD, minutes)", **CLAB}
DEVICE = {"program": "program-provided", "own": "personal"}
COMMON = [("Values", "Adjusted values from the joint model of each outcome (decile as 10 categories, age, age squared, and "
                     "sex, sex left out for the 2 cancers), averaged over the outcome's population (Materials and methods, Appendix S3). "
                     "Rates and shares in percent. 95% CIs by the delta method."),
          ("Trends", "Each outcome refitted with the decile number as 1 slope: per_decile = odds ratio (logistic) or "
                     "difference in the outcome's units per decile; decile_1_to_10 = the same over the 9 steps. p_holm = "
                     "Holm-adjusted p within the grouping's family (29 per encoder: 15 whole cohort + 14 among "
                     "screen-negative participants; 7 per other grouping; 4 device trends per encoder)."),
          ("Agreement", "same_direction_and_holm_p_below_0.05: the trend has the prespecified encoder's direction and "
                        "Holm-adjusted p < 0.05 within its own family (Materials and methods)."),
          ("Pooling", "A decile with 1 to 20 participants with or without a condition was pooled with its neighbor "
                      "(decile_from to decile_to; pooled = TRUE); pooled deciles share 1 value."),
          ("Groupings", "The 6 sentence encoders and the 6 groupings that do not use a sentence encoder (Appendix S4); "
                        "each forms its own groups, distance, and deciles.")]


def trends():
    """(the whole-cohort and below-threshold trends with Holm-adjusted p, the device trends with theirs, the size of each
    grouping's family)"""
    TR = pd.read_csv(os.path.join(WB, "deciles", "trends.csv"))
    ET = pd.read_csv(os.path.join(WB, "deciles_ed", "ed_trend.csv"))
    TT = pd.concat([TR[TR.outcome != "ed_share_pct"], ET], ignore_index=True)   # the emergency department share on coded care
    main = TT[TT.scope.isin(["cohort", "below_threshold"])].copy()
    main["p_holm"] = np.nan
    for g, d in main.groupby("grouping"):
        main.loc[d.index, "p_holm"] = holm(d.p.values)
    fam = main.groupby("grouping").size()
    assert all(fam[e] == 29 for e in ENCODERS) and all(fam[g] == 7 for g in OTHER), fam
    dev = TT[TT.scope == "sleep_device"].copy()
    dev["p_holm"] = np.nan
    for g, d in dev.groupby("grouping"):
        assert len(d) == 4, (g, len(d))
        dev.loc[d.index, "p_holm"] = holm(d.p.values)
    return main, dev, fam


def trend_sheet(T, scope, fam):
    x = T[T.scope == scope].copy()
    x["key"] = x.outcome + "|" + x.source.fillna("")                     # device trends: 1 per source and outcome
    ref = x[x.grouping == PRIMARY].set_index("key")
    assert ref.index.is_unique
    rows = []
    for t in x.itertuples():
        neutral = 1.0 if t.scale == "odds ratio" else 0.0
        same = np.sign(t.per_decile - neutral) == np.sign(ref.loc[t.key, "per_decile"] - neutral)
        rows.append({"grouping": GLAB[t.grouping], "grouping_type": GTYPE[t.grouping], "outcome": OLAB[t.outcome],
                     "model": t.model, "participants": int(t.n), "scale": t.scale,
                     "per_decile": round(t.per_decile, 4), "per_decile_ci_low": round(t.lo, 4), "per_decile_ci_high": round(t.hi, 4),
                     "decile_1_to_10": round(t.over_9, 4), "decile_1_to_10_ci_low": round(t.over_9_lo, 4),
                     "decile_1_to_10_ci_high": round(t.over_9_hi, 4), "p": pv(t.p), "p_holm": pv(t.p_holm),
                     "holm_family_size": int(fam[t.grouping]) if scope != "sleep_device" else 4,
                     "same_direction_and_holm_p_below_0.05": "" if t.grouping == PRIMARY else bool(same and t.p_holm < 0.05)})
    return pd.DataFrame(rows)


def outcome_sheet(O, E, scope):
    rows = []
    groupings = [g for g in ENCODERS + OTHER if g in set(O.grouping)]
    for g in groupings:
        o = O[O.grouping == g].set_index("decile")
        e = E[(E.scope == scope) & (E.grouping == g)].set_index("decile")
        for dcl in range(1, 11):
            a = o.loc[dcl]
            row = {"grouping": GLAB[g], "grouping_type": GTYPE[g], "decile": dcl, "participants": int(a.n_set),
                   "mean_age_years": round(a.mean_age, 1), "female_pct": round(100 * a.share_female, 1)}
            if "mean_breadth_adj" in o.columns:
                row.update({"thresholds_met_participants": int(a.n_all_evaluable),
                            "thresholds_met_mean": round(a.mean_breadth_adj, 3), "thresholds_met_ci_low": round(a.mean_breadth_adj_lo, 3),
                            "thresholds_met_ci_high": round(a.mean_breadth_adj_hi, 3)})
            row["barriers_participants"] = int(a.n_survey)
            for k, lab in (("access", "any_barrier"), ("cost", "cost_barrier"), ("avoid", "clinician_race_religion_language")):
                row.update({f"{lab}_pct": round(100 * a[f"{k}_adj"], 2), f"{lab}_ci_low": round(100 * a[f"{k}_adj_lo"], 2),
                            f"{lab}_ci_high": round(100 * a[f"{k}_adj_hi"], 2)})
            if dcl in e.index:
                b = e.loc[dcl]
                row.update({"ed_participants": int(b.n_icd_visit), "ed_share_pct": round(100 * b.ed_adj, 3),
                            "ed_share_ci_low": round(100 * b.ed_adj_lo, 3), "ed_share_ci_high": round(100 * b.ed_adj_hi, 3)})
            row.update({"sleep_participants": int(a.n_sleep), "sleep_mean_min": round(a.sleep_mean_adj, 1),
                        "sleep_mean_ci_low": round(a.sleep_mean_adj_lo, 1), "sleep_mean_ci_high": round(a.sleep_mean_adj_hi, 1),
                        "sleep_variation_min": round(a.sleep_var_median_adj, 1),
                        "sleep_variation_ci_low": round(a.sleep_var_median_adj_lo, 1),
                        "sleep_variation_ci_high": round(a.sleep_var_median_adj_hi, 1)})
            rows.append(row)
    return pd.DataFrame(rows)


def condition_sheet(C):
    rows = []
    for t in C.itertuples():
        pooled = t.decile_from != t.decile_to
        rows.append({"grouping": GLAB[t.grouping], "condition": CLAB[t.condition], "phecode": t.phecode,
                     "decile_from": int(t.decile_from), "decile_to": int(t.decile_to), "pooled": bool(pooled),
                     "participants": t.n, "with_condition": t.k,
                     "pct": round(100 * t.adj, 2), "ci_low": round(100 * t.adj_lo, 2) if pd.notna(t.adj_lo) else np.nan,
                     "ci_high": round(100 * t.adj_hi, 2) if pd.notna(t.adj_hi) else np.nan})
    return pd.DataFrame(rows)


def sleep_sheet(S):
    rows = []
    for t in S[S.grouping.isin(ENCODERS)].itertuples():
        rows.append({"device_source": DEVICE[t.source],
                     "grouping": GLAB[t.grouping], "decile": int(t.decile), "sleep_participants": int(t.n_sleep),
                     "sleep_mean_min": round(t.sleep_mean_adj, 1), "sleep_mean_ci_low": round(t.sleep_mean_adj_lo, 1),
                     "sleep_mean_ci_high": round(t.sleep_mean_adj_hi, 1), "sleep_variation_min": round(t.sleep_var_median_adj, 1),
                     "sleep_variation_ci_low": round(t.sleep_var_median_adj_lo, 1),
                     "sleep_variation_ci_high": round(t.sleep_var_median_adj_hi, 1)})
    return pd.DataFrame(rows)


def main():
    O = pd.read_csv(os.path.join(WB, "deciles", "outcomes.csv"))
    E = pd.read_csv(os.path.join(WB, "deciles_ed", "ed_deciles.csv"))
    C = pd.read_csv(os.path.join(WB, "deciles", "conditions.csv"))
    S = pd.read_csv(os.path.join(WB, "deciles", "sleep_device.csv"))
    main_trends, dev, fam = trends()
    dev_sheet = trend_sheet(dev, "sleep_device", fam)
    dev_sheet.insert(0, "device_source", [DEVICE[s] for s in dev.source])
    readme = [("Figure 2 Source Data", "Outcomes by decile of the distance from the cohort's mean scores (Figure 2 and "
                                       "Figure S3)."), *COMMON,
              ("Sheet Outcomes by decile", "Screening thresholds met, barriers to care, emergency department share (coded "
                                           "care), and sleep, under the 6 encoders and 6 other groupings."),
              ("Sheet Conditions by decile", "The 8 recorded conditions under the 6 encoders; 'Whole population' rows give "
                                             "the percentage in each condition's population."),
              ("Sheet Trends", "Every trend in the whole cohort."),
              ("Sheet Sleep by device", "Sleep by decile within each device source, 6 encoders, with the 4 device trends "
                                        "of each encoder in Sheet Sleep device trends.")]
    write("Figure2_source_data", readme,
          [("Outcomes by decile", outcome_sheet(O[O.grouping.isin(ENCODERS + OTHER)], E, "cohort")),
           ("Conditions by decile", condition_sheet(C)), ("Trends", trend_sheet(main_trends, "cohort", fam)),
           ("Sleep by device", sleep_sheet(S)), ("Sleep device trends", dev_sheet)])


if __name__ == "__main__":
    main()
