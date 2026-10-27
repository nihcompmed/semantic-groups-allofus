#!/usr/bin/env python3
"""step3_thresholds.py -- the six screening thresholds on the cohort (Materials and methods; Table S3).

A threshold is a determination the instrument itself publishes: a cut-off on its total, a published rule over several
items, or a single item whose endorsement is the finding. Each is applied to the answers on the instrument's own scale
and in its own direction, never to the rescaled answers the scores use.
  PHQ-9     sum of the 9 items (0-3) >= 10                       at least moderate depression
  GAD-7     sum of the 7 items (0-3) >= 10                       at least moderate anxiety
  ACE       >= 4 of 7 exposure categories (the module scores categories: the 3 sexual-abuse items are 1 category, problem
            drinking and drug use in the household 1 category; household incarceration, ace_4, is not released in the
            Registered Tier, so 7 of the module's 8 categories); a binary item counts when Yes, a frequency item when
            Once or More than once
  ASRS      >= 4 of the 6 Part-A items at the clinical frequency: Sometimes or more for items 1-3, Often or more for 4-6
  HVS       either vigilance item Sometimes true or Often true    at risk for food insecurity
  Suicidal ideation or attempt   ss_2 (lifetime ideation) or ss_3 (lifetime attempt) answered Yes
A threshold is evaluated for a person only when every item it rests on was answered; otherwise it is withheld for them,
so an imputed answer never enters a threshold. The number of thresholds met (breadth) counts the evaluable ones; "met
none" requires all six evaluable.

The flagged set beside the cohort in the summary is the outliers of the encoder chosen in advance (gte-large-en-v1.5).

OUTPUTS   work/thresholds.csv   per person: each threshold (1 met, 0 not, empty withheld), n_withheld, breadth,
                                all_evaluable, any_met, none_met, is_flagged
DOWNLOAD  thresholds:  thresholds.csv (per threshold, evaluable and met, cohort and flagged, Wilson 95% CI),
          thresholds_met.csv (0 to 6 met, on those with all six evaluable), thresholds.json (the counts)

Run:  python3 step3_thresholds.py
"""
import json
import math

import numpy as np
import pandas as pd

from release import SUPP, marked, one, pair_hidden, partition_mask, write_download
from settings import CDR, COHORT_CSV, ID, MASK_CSV, PRIMARY, download_dir, work

PHQ = [f"phq9_{i}" for i in range(1, 10)]
GAD = [f"gad7_{i}" for i in range(1, 8)]
ASRS_INATT, ASRS_HYPER = ["asrs_1", "asrs_2", "asrs_3"], ["asrs_4", "asrs_5", "asrs_6"]
SUICIDE_ITEMS = ["ss_2", "ss_3"]
HVS = ["hvs_1", "hvs_2"]
ACE_CATEGORIES = {
    "household mental illness": ["ace_1"],
    "household substance abuse": ["ace_2", "ace_3"],
    "parental separation or divorce": ["ace_5"],
    "witnessed domestic violence": ["ace_6"],
    "physical abuse": ["ace_7"],
    "emotional abuse": ["ace_8"],
    "sexual abuse": ["ace_9", "ace_10", "ace_11"],
}
ACE_ITEMS = [i for v in ACE_CATEGORIES.values() for i in v]
ACE_ENDORSED = lambda v: v >= 0              # binary -1 No / 1 Yes; frequency -1 Never / 0 Once / 1 More than once
CRITERIA = {"PHQ-9": PHQ, "GAD-7": GAD, "ACE": ACE_ITEMS, "ASRS": ASRS_INATT + ASRS_HYPER, "HVS": HVS,
            "Suicidal ideation or attempt": SUICIDE_ITEMS}
NB = len(CRITERIA)


def wilson_pct(k, n, z=1.959963985):
    """Wilson 95% interval, in percent."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def evaluate(R):
    """The six thresholds on the answers, before withholding: a boolean frame."""
    ace_count = np.sum([np.any([ACE_ENDORSED(R[it].values.astype(float)) for it in its], axis=0)
                        for its in ACE_CATEGORIES.values()], axis=0)
    asrs_count = sum((R[i].values.astype(float) >= 2).astype(int) for i in ASRS_INATT) \
        + sum((R[i].values.astype(float) >= 3).astype(int) for i in ASRS_HYPER)
    F = pd.DataFrame(index=R.index)
    F["PHQ-9"] = R[PHQ].sum(axis=1).values >= 10
    F["GAD-7"] = R[GAD].sum(axis=1).values >= 10
    F["ACE"] = ace_count >= 4
    F["ASRS"] = asrs_count >= 4
    F["HVS"] = (R[HVS].values >= 2).any(axis=1)
    F["Suicidal ideation or attempt"] = (R[SUICIDE_ITEMS].values == 1).any(axis=1)
    return F


def main():
    R = pd.read_csv(COHORT_CSV)
    R[ID] = R[ID].astype(str)
    R = R.set_index(ID)
    M = pd.read_csv(MASK_CSV)
    M[ID] = M[ID].astype(str)
    M = M.set_index(ID).reindex(R.index)
    assert not M.isna().any().any(), "the imputation mask does not cover the cohort"
    need = sorted(set(i for v in CRITERIA.values() for i in v))
    names = list(CRITERIA)

    met = evaluate(R)
    withheld = pd.DataFrame({n: M[its].astype(bool).any(axis=1).values for n, its in CRITERIA.items()}, index=R.index)
    F = met.astype(float).where(~withheld)                                  # 1 met, 0 not, NaN withheld
    F["n_withheld"] = withheld.sum(axis=1).values
    F["breadth"] = F[names].sum(axis=1, min_count=1).fillna(0).astype(int)
    F["all_evaluable"] = F["n_withheld"] == 0
    F["any_met"] = (F[names] == 1).any(axis=1)
    F["none_met"] = F["all_evaluable"] & (F["breadth"] == 0)
    S0 = pd.read_csv(work(f"scores_{PRIMARY}.csv"))
    S0[ID] = S0[ID].astype(str)
    S0 = S0.set_index(ID)
    flag_ids = set(S0.index[S0["flagged"].astype(bool)])
    assert flag_ids <= set(F.index), "a flagged id is absent from the cohort"
    F["is_flagged"] = F.index.isin(flag_ids)
    n_coh, n_flag = len(F), int(F["is_flagged"].sum())
    O = F[F["is_flagged"]]
    print(f"[step 3] cohort {n_coh:,} | all six evaluable {int(F.all_evaluable.sum()):,} | met none "
          f"{int(F.none_met.sum()):,}", flush=True)

    rows = []
    for name in names:
        ec, eo = F[name].notna(), O[name].notna()
        kc, ko = int((F[name] == 1).sum()), int((O[name] == 1).sum())
        cl, ch = wilson_pct(kc, int(ec.sum()))
        ol, oh = wilson_pct(ko, int(eo.sum()))
        rows.append({"criterion": name, "n_items": len(CRITERIA[name]),
                     "n_cohort_evaluable": int(ec.sum()), "n_cohort": kc, "pct_cohort": round(100 * kc / ec.sum(), 2),
                     "cohort_lo": round(cl, 2), "cohort_hi": round(ch, 2),
                     "n_flagged_evaluable": int(eo.sum()), "n_flagged": ko, "pct_flagged": round(100 * ko / eo.sum(), 1),
                     "flagged_lo": round(ol, 1), "flagged_hi": round(oh, 1)})
    kc, ko = int(F["any_met"].sum()), int(O["any_met"].sum())
    cl, ch = wilson_pct(kc, n_coh)
    ol, oh = wilson_pct(ko, n_flag)
    rows.append({"criterion": "any_met", "n_items": len(need), "n_cohort_evaluable": n_coh, "n_cohort": kc,
                 "pct_cohort": round(100 * kc / n_coh, 2), "cohort_lo": round(cl, 2), "cohort_hi": round(ch, 2),
                 "n_flagged_evaluable": n_flag, "n_flagged": ko, "pct_flagged": round(100 * ko / n_flag, 1),
                 "flagged_lo": round(ol, 1), "flagged_hi": round(oh, 1)})
    S = pd.DataFrame(rows)
    Fa, Oa = F[F["all_evaluable"]], O[O["all_evaluable"]]
    B = pd.DataFrame({"breadth": range(NB + 1),
                      "n_cohort": [int((Fa.breadth == k).sum()) for k in range(NB + 1)],
                      "pct_cohort": [round(100 * (Fa.breadth == k).mean(), 2) for k in range(NB + 1)],
                      "n_flagged": [int((Oa.breadth == k).sum()) for k in range(NB + 1)],
                      "pct_flagged": [round(100 * (Oa.breadth == k).mean(), 1) for k in range(NB + 1)]})
    n_undet = int(((O.breadth == 0) & ~O.all_evaluable).sum())

    # the policy on what leaves: a side of the summary is hidden when evaluable, withheld, met or not met is 1 to 20
    Sw = S.astype(object).copy()
    side_hidden = {}
    for side, total in (("cohort", n_coh), ("flagged", n_flag)):
        n_ev, n_met = f"n_{side}_evaluable", f"n_{side}"
        dep = [f"pct_{side}", f"{side}_lo", f"{side}_hi"]
        for i in S.index:
            ev_i, met_i = int(S.at[i, n_ev]), int(S.at[i, n_met])
            h = pair_hidden(ev_i, total) or pair_hidden(met_i, ev_i)
            side_hidden[(side, S.at[i, "criterion"])] = h
            if h:
                Sw.loc[i, dep] = np.nan
                Sw.loc[i, [n_ev, n_met]] = SUPP
    # the 0 to 6 distribution partitions the all-evaluable count; a mean beside a hidden cell would solve it, so it goes
    Bw = B.astype(object).copy()
    margins, means = {}, {"cohort": float(Fa.breadth.mean()), "flagged": float(Oa.breadth.mean())}
    for side, total, margin in (("cohort", n_coh, len(Fa)), ("flagged", n_flag, len(Oa))):
        n_col, p_col = f"n_{side}", f"pct_{side}"
        if pair_hidden(margin, total):
            Bw[n_col], Bw[p_col] = SUPP, np.nan
            margins[side], means[side] = SUPP, None
            continue
        hide, prim = partition_mask(B[n_col].values, avoid=[0])
        Bw[n_col] = marked(B[n_col].values, hide, prim)
        Bw.loc[hide, p_col] = np.nan
        margins[side] = int(margin)
        if hide.any():
            means[side] = None
    none_f = Bw.loc[Bw.breadth == 0, "n_flagged"].iloc[0]
    none_c = Bw.loc[Bw.breadth == 0, "n_cohort"].iloc[0]

    F.rename_axis(ID).reset_index().to_csv(work("thresholds.csv"), index=False)
    d = download_dir("thresholds")
    Sw.to_csv(f"{d}/thresholds.csv", index=False)
    Bw.to_csv(f"{d}/thresholds_met.csv", index=False)
    json.dump({"cdr": CDR, "encoder": PRIMARY, "n_criteria": NB, "criteria": names,
               "n_cohort": n_coh, "n_flagged": one(n_flag), "n_criterion_items": len(need),
               "n_ace_categories": len(ACE_CATEGORIES), "disclosure": "counts of 1 to 20 suppressed (release.py)",
               "n_cohort_all_evaluable": margins["cohort"], "n_flagged_all_evaluable": margins["flagged"],
               "withheld_per_criterion_cohort": {n: (SUPP if side_hidden[("cohort", n)] else int(withheld[n].sum()))
                                                 for n in names},
               "n_flagged_none_met": none_f, "n_flagged_undetermined": one(n_undet), "n_cohort_none_met": none_c,
               "mean_breadth_flagged_all_evaluable": means["flagged"],
               "mean_breadth_cohort_all_evaluable": means["cohort"]},
              open(f"{d}/thresholds.json", "w"), indent=2,
              default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
    write_download("thresholds", ["thresholds.csv", "thresholds.json", "thresholds_met.csv"])


if __name__ == "__main__":
    main()
