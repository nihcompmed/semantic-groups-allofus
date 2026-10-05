#!/usr/bin/env python3
"""cutoffs.py -- the six published criteria on the cohort of record, and how the screen's flagged set
sits against them.

★ FOR THE PAPER (steps 5 and 6 of README.md) the later steps read the per-person columns `breadth` and
`all_evaluable` in cutoff_flags.csv, and `n_criteria` in cutoff_summary.json. zip_cutoffs.py zips
cutoff_summary.csv, cutoff_summary.json and breadth_table.csv, for the cohort and for gte's own 95th
percentile as "the flagged set" (eTable 5).

STANDALONE (numpy, pandas). Reads the cohort of record built by build_responses_v9 (responses_k5,
at most five missing, k-NN imputed) with its imputation mask, and the flagged column of the
prespecified encoder's own screen.

Criteria are evaluated on the CODEBOOK CODING, on each instrument's own response scale and in its
original direction, never on the rescaled and reverse-coded matrix the screen uses. The published
scoring is defined on that scale, so applying it to the rescaled matrix would not be the published
criterion.

★★ THE RULE FOR A CRITERION: A CRITERION IS A DETERMINATION THE INSTRUMENT ITSELF PUBLISHES -- a
threshold on its total, a published multi-item rule, or a single item whose endorsement is the
finding. Not a rule we build across items.

  NOT USED  WMH-CIDI psychosis  "any of three psychotic-experience items endorsed" is a compound
                                across items that neither the instrument nor All of Us publishes.
  NOT USED  PROMIS mental health  a single-item cut would be OUR choice. The codebook names "PROMIS
                                Mental Health" as a measure and gives it no threshold.
  NOT USED  the suicide screen  "any of ss_1/ss_2/ss_3" pools non-suicidal self-injury ideation,
                                suicidal ideation and a suicide attempt into one flag. No scoring
                                rule for it exists anywhere in the All of Us codebook.
  PHQ-9                         the band only, with no "OR phq9_9 endorsed" clause. The codebook
                                documents the bands at 5/10/15/20 and says nothing about item 9, and
                                this project assigns item 9 to suicidality rather than depression.
  ★ criterion 6                 SUICIDAL IDEATION OR ATTEMPT, met when ss_2 (lifetime ideation) OR
                                ss_3 (lifetime attempt) is Yes. Each Yes is itself a finding, so no
                                scoring rule is required, and the criterion is met when either is
                                present. It is pooled so that it corresponds to the record diagnosis
                                MB_284 (suicidal ideation, attempt or self-harm), which pools the same
                                two. ss_1 (thoughts of self-harm WITHOUT wanting to die) stays out: it
                                is neither ideation nor attempt. Evaluable only when both ss_2 and
                                ss_3 were answered (the withholding rule below).
  ACE                           7 of the module's 8 categories, threshold 4 (see below).
  NOT USED  PSS-10              it has bands in the codebook, but the developer published no
                                cut-offs and two incompatible conventions circulate under Cohen's
                                name.

★ ACE SCORES 7 OF THE MODULE'S 8 CATEGORIES, AND THE REASON IS THE DATA TIER.
`ace_4` (household incarceration) is marked REGISTERED_QUESTION_SUPPRESSED in the program codebook
(All_of_Us_Survey_Data_Codebooks.xlsx, Emotional Health History sheet, Registered Tier Rules), so
it is not released in the Registered Tier this analysis runs in. The published threshold of 4 is
applied to the 7 categories we hold, which makes the criterion strictly harder to meet than the
published one: anyone whose fourth category would have been incarceration is scored as not meeting
it. This is a restriction of the data tier.

WITHHOLDING. An imputed item enters the screen and the attribution like any other. It never enters
a published criterion. A criterion is evaluated for a person only when every item that
carries it was answered by that person; otherwise it is WITHHELD (NaN) for them. Consequences:
  * a criterion's rate is over the people for whom it is evaluable
  * breadth (number of the six met) is counted over the person's evaluable criteria and reported
    with the number withheld. The breadth DISTRIBUTION and "meets none" are reported on the people
    with all six evaluable, so a withheld criterion can never hide a met one
  * any_met is true when at least one evaluable criterion is met, whatever is withheld

THE SIX CRITERIA, each traced to the items that carry it and to what meeting it asserts

  PHQ-9            sum of the 9 items (0-3) >= 10
                   -> "at least moderate depression" (codebook: 5/10/15/20 = mild, moderate,
                      moderately severe, severe)
  GAD-7            sum of the 7 items (0-3) >= 10
                   -> "at least moderate anxiety" (codebook: 0-4 minimal, 5-9 mild, 10-14 moderate,
                      15-21 severe)
  ACE              >= 4 of 7 exposure CATEGORIES. The BRFSS module scores categories, not items: the
                   three sexual-abuse items collapse to one category and problem drinking and drug
                   use to one household-substance category. The module's 8 categories come from 11
                   items; household incarceration (ace_4) is suppressed in the Registered Tier, so
                   7 categories come from the 10 items in the battery.
                   A binary item counts when Yes, a frequency item when Once or More than once.
                   -> "increased risk for social, mental, or other wellbeing problems compared to
                      those who endorsed none" -- COMPARATIVE, not a statement about the person now
  ASRS             >= 4 of the 6 Part-A items at clinical frequency: Sometimes or worse (>= 2) for
                   items 1-3, Often or worse (>= 3) for items 4-6
                   -> "symptoms highly consistent with ADHD in adults and further investigation is
                      warranted" -- a referral trigger, not a determination
  HVS              either vigilance item Sometimes true or Often true (>= 2)
                   -> "at risk for food insecurity". The unit is the HOUSEHOLD ("we", "our food")
  Suicidal ideation or attempt   ss_2 or ss_3 endorsed
                   -> lifetime thoughts of killing oneself, or a lifetime suicide attempt defined by
                      intent to die. Each Yes is itself the finding.

★ THE SIX ARE NOT INTERCHANGEABLE. Two name a disorder severity, one defers to assessment, two name
a risk over different windows and units, one names lifetime suicidal thoughts or an attempt. Breadth sums across all six and
whatever reports it must say so.

OUTPUT (-> OUT_DIR)
  cutoff_flags.csv     id, one column per criterion (1 met, 0 not met, empty = withheld), n_withheld,
                       breadth, all_evaluable, any_met, none_met, is_flagged      (per person, stays)
  cutoff_summary.csv   per criterion: evaluable n and % met with Wilson 95% intervals, cohort and flagged
  breadth_table.csv    the 0-6 breadth distribution on the all-evaluable people, cohort against flagged
  cutoff_summary.json  the counts, for the record
  The three that leave are written under the dissemination policy (disclosure.py):
  counts of 1 to 20 as "<=20", complementary cells as "suppressed", derived values blanked.

Run:  python3 cutoffs.py
      python3 cutoffs.py --encoder bge-m3        (any of the six, for the supplement)
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_config import CFG   # noqa: E402  resolved once, no defaults
from disclosure import SUPP, one, pair_hidden, partition_mask, marked   # noqa: E402  the policy

ENCODER       = sys.argv[sys.argv.index("--encoder") + 1] if "--encoder" in sys.argv else "gte-large-en-v1.5"
COHORT        = CFG.cohort
RESPONSES_CSV = CFG.responses_csv
MASK_CSV      = CFG.mask_csv                     # True where a value was imputed; nothing is imputed under k0
SCORES_CSV    = os.path.join(CFG.out_dir, f"scores_{ENCODER}.csv")    # id, distance, distance2, pctile, flagged
OUT_DIR       = CFG.out_dir
N_ITEMS       = CFG.n_items
CDR           = os.environ.get("WORKSPACE_CDR", "")
ID_COL        = "id"

PHQ = [f"phq9_{i}" for i in range(1, 10)]
GAD = [f"gad7_{i}" for i in range(1, 8)]
ASRS_INATT, ASRS_HYPER = ["asrs_1", "asrs_2", "asrs_3"], ["asrs_4", "asrs_5", "asrs_6"]
SUICIDE_ITEMS = ["ss_2", "ss_3"]     # lifetime ideation, lifetime attempt (criterion 6)
HVS = ["hvs_1", "hvs_2"]
ACE_CATEGORIES = {
    "household mental illness": ["ace_1"],
    "household substance abuse": ["ace_2", "ace_3"],    # ONE category: Felitti asks alcohol OR drugs
    # "household incarceration": ace_4 is REGISTERED_QUESTION_SUPPRESSED, so absent in this tier
    "parental separation or divorce": ["ace_5"],
    "witnessed domestic violence": ["ace_6"],
    "physical abuse": ["ace_7"],
    "emotional abuse": ["ace_8"],
    "sexual abuse": ["ace_9", "ace_10", "ace_11"],      # ONE category from three questions
}
assert len(ACE_CATEGORIES) == 7, "7 of the module's 8 categories; ace_4 is suppressed in the Registered Tier"
ACE_ITEMS = [i for v in ACE_CATEGORIES.values() for i in v]
# Endorsement is >= 0 for every ACE item, which covers all three codings the module uses:
#   binary -1 No / 1 Yes; frequency -1 Never / 0 Once / 1 More than once; ace_5 -1 No or Parents not married / 1 Yes
ACE_ENDORSED = lambda v: v >= 0

CRITERIA = {          # name -> the items that carry it (all must be answered for it to be evaluable)
    "PHQ-9": PHQ,
    "GAD-7": GAD,
    "ACE": ACE_ITEMS,
    "ASRS": ASRS_INATT + ASRS_HYPER,
    "HVS": HVS,
    "Suicidal ideation or attempt": SUICIDE_ITEMS,
}
assert len(CRITERIA) == 6, "the screening thresholds are six criteria"
NB = len(CRITERIA)            # breadth runs 0..NB


def wilson(k, n, z=1.959963985):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def evaluate(R):
    """The six criteria on the codebook coding, before withholding. Returns a bool DataFrame."""
    ace_count = np.sum([np.any([ACE_ENDORSED(R[it].values.astype(float)) for it in its], axis=0)
                        for its in ACE_CATEGORIES.values()], axis=0)
    asrs_count = sum((R[i].values.astype(float) >= 2).astype(int) for i in ASRS_INATT) \
        + sum((R[i].values.astype(float) >= 3).astype(int) for i in ASRS_HYPER)
    F = pd.DataFrame(index=R.index)
    F["PHQ-9"] = R[PHQ].sum(axis=1).values >= 10          # the published band; NO item-9 clause
    F["GAD-7"] = R[GAD].sum(axis=1).values >= 10
    F["ACE"] = ace_count >= 4                             # of the 7 categories held (ace_4 suppressed)
    F["ASRS"] = asrs_count >= 4
    F["HVS"] = (R[HVS].values >= 2).any(axis=1)
    F["Suicidal ideation or attempt"] = (R[SUICIDE_ITEMS].values == 1).any(axis=1)   # either Yes
    return F


def main():
    CFG.describe("cutoffs")
    print(f"criteria: {NB} | encoder for the flagged set: {ENCODER}")
    R = pd.read_csv(RESPONSES_CSV)
    R[ID_COL] = R[ID_COL].astype(str)
    R = R.set_index(ID_COL)
    if COHORT == "k5":
        M = pd.read_csv(MASK_CSV)
        M[ID_COL] = M[ID_COL].astype(str)
        M = M.set_index(ID_COL).reindex(R.index)
        assert not M.isna().any().any(), "the imputation mask does not cover the cohort"
    else:                                              # complete cases: every item answered, nothing withheld
        M = pd.DataFrame(False, index=R.index, columns=[c for c in R.columns if c not in ("age", "sex")])

    need = sorted(set(i for v in CRITERIA.values() for i in v))
    absent = [c for c in need if c not in R.columns or c not in M.columns]
    assert not absent, f"items absent from the response file or the mask: {absent}"
    names = list(CRITERIA)

    met = evaluate(R)
    withheld = pd.DataFrame({n: M[its].astype(bool).any(axis=1).values for n, its in CRITERIA.items()}, index=R.index)
    F = met.astype(float).where(~withheld)                       # 1 met, 0 not, NaN withheld
    F["n_withheld"] = withheld.sum(axis=1).values
    F["breadth"] = F[names].sum(axis=1, min_count=1).fillna(0).astype(int)   # over evaluable criteria
    F["all_evaluable"] = F["n_withheld"] == 0
    F["any_met"] = (F[names] == 1).any(axis=1)
    F["none_met"] = F["all_evaluable"] & (F["breadth"] == 0)

    # ---- the flagged set is the prespecified encoder's own, NOT a consensus across encoders
    assert os.path.exists(SCORES_CSV), f"{SCORES_CSV} is missing; run screen_battery.py first"
    S0 = pd.read_csv(SCORES_CSV)
    S0[ID_COL] = S0[ID_COL].astype(str)
    S0 = S0.set_index(ID_COL)
    flag_ids = set(S0.index[S0["flagged"].astype(bool)])
    assert flag_ids <= set(F.index), "a flagged id is absent from the response file"
    F["is_flagged"] = F.index.isin(flag_ids)
    n_coh, n_flag = len(F), int(F["is_flagged"].sum())
    O = F[F["is_flagged"]]
    print(f"cohort {n_coh:,} | flagged under {ENCODER} {n_flag:,} ({100 * n_flag / n_coh:.2f}%)")
    print(f"criterion items {len(need)} of {N_ITEMS} | ACE on {len(ACE_CATEGORIES)} categories (ace_4 suppressed) | "
          f"people with every criterion evaluable: cohort {int(F.all_evaluable.sum()):,} "
          f"({100 * F.all_evaluable.mean():.1f}%), flagged {int(O.all_evaluable.sum()):,} "
          f"({100 * O.all_evaluable.mean():.1f}%)\n")

    rows = []
    for name in names:
        ec, eo = F[name].notna(), O[name].notna()
        kc, ko = int((F[name] == 1).sum()), int((O[name] == 1).sum())
        cl, ch = wilson(kc, int(ec.sum()))
        ol, oh = wilson(ko, int(eo.sum()))
        rows.append({"criterion": name, "n_items": len(CRITERIA[name]),
                     "n_cohort_evaluable": int(ec.sum()), "n_cohort": kc, "pct_cohort": round(100 * kc / ec.sum(), 2),
                     "cohort_lo": round(cl, 2), "cohort_hi": round(ch, 2),
                     "n_flagged_evaluable": int(eo.sum()), "n_flagged": ko, "pct_flagged": round(100 * ko / eo.sum(), 1),
                     "flagged_lo": round(ol, 1), "flagged_hi": round(oh, 1)})
    kc, ko = int(F["any_met"].sum()), int(O["any_met"].sum())
    cl, ch = wilson(kc, n_coh); ol, oh = wilson(ko, n_flag)
    rows.append({"criterion": "any_met", "n_items": len(need), "n_cohort_evaluable": n_coh, "n_cohort": kc,
                 "pct_cohort": round(100 * kc / n_coh, 2), "cohort_lo": round(cl, 2), "cohort_hi": round(ch, 2),
                 "n_flagged_evaluable": n_flag, "n_flagged": ko, "pct_flagged": round(100 * ko / n_flag, 1),
                 "flagged_lo": round(ol, 1), "flagged_hi": round(oh, 1)})
    S = pd.DataFrame(rows)
    print("criteria (% met among the people for whom the criterion is evaluable)")
    print(S.to_string(index=False))

    Fa, Oa = F[F["all_evaluable"]], O[O["all_evaluable"]]
    B = pd.DataFrame({"breadth": range(NB + 1),
                      "n_cohort": [int((Fa.breadth == k).sum()) for k in range(NB + 1)],
                      "pct_cohort": [round(100 * (Fa.breadth == k).mean(), 2) for k in range(NB + 1)],
                      "n_flagged": [int((Oa.breadth == k).sum()) for k in range(NB + 1)],
                      "pct_flagged": [round(100 * (Oa.breadth == k).mean(), 1) for k in range(NB + 1)]})
    print(f"\nbreadth (number of the {NB} met), on people with all {NB} evaluable: "
          f"cohort {len(Fa):,}, flagged {len(Oa):,}")
    print(B.to_string(index=False))
    print(f"mean breadth: flagged {Oa.breadth.mean():.2f}  cohort {Fa.breadth.mean():.2f}   mode {int(Oa.breadth.mode()[0])}")
    print(f"mean breadth over evaluable criteria, everyone: flagged {O.breadth.mean():.2f}  cohort {F.breadth.mean():.2f}")
    for t in (2, 4, NB):
        k = int((Oa.breadth >= t).sum())
        lo, hi = wilson(k, len(Oa))
        print(f"  flagged meeting >= {t}: {k} ({100 * k / len(Oa):.0f}%, 95% CI {lo:.0f}-{hi:.0f})")
    n_inv = int(O["none_met"].sum())
    lo, hi = wilson(n_inv, len(Oa))
    n_undet = int(((O.breadth == 0) & ~O.all_evaluable).sum())
    # Neutral wording on purpose. These are participants above the 95th percentile
    # on the battery scoring who met none of the published criteria. Nothing is claimed about the
    # criteria failing to do something, so no "invisible", "missed" or "undetected" anywhere.
    print(f"\nflagged meeting none of the {NB}, all {NB} evaluable: {n_inv} of {len(Oa)} "
          f"({100 * n_inv / len(Oa):.1f}%, 95% CI {lo:.1f}-{hi:.1f}); "
          f"a further {n_undet} flagged participants meet none of their evaluable criteria with at least one withheld")
    n_inv_c = int(F["none_met"].sum())
    print(f"cohort: none of the {NB} (all evaluable) {n_inv_c:,} of {len(Fa):,} ({100 * n_inv_c / len(Fa):.1f}%)")

    # ---- the dissemination policy, applied at WRITE time to the three files that
    # leave. disclosure.py states the rule. The printed tables above stay unsuppressed, because
    # stdout stays on the Workbench. cutoff_flags.csv is per person and never leaves.
    n_hidden = 0

    # cutoff_summary.csv: per criterion and side, evaluable of the side's total and met of evaluable.
    # A side is hidden when any of evaluable, withheld (total - evaluable), met or not met is 1 to 20,
    # since each is derivable from the others.
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
                n_hidden += 1

    # breadth_table.csv: each column is a partition of the all-evaluable count, which the json
    # reports, so complementary suppression applies. Breadth 0 is avoided as a complementary cell
    # because it is the no-criterion count that exemplar_encoder.py and direction_own.py report.
    # If the margin itself is hidden against its side's total, the whole column goes.
    # ★ The mean breadth over a column is Σ b·n_b / margin, a second equation beside the margin, so
    # with two hidden cells it solves both. The mean is blanked wherever its column hides a cell.
    Bw = B.astype(object).copy()
    margins, means = {}, {"cohort": float(Fa.breadth.mean()), "flagged": float(Oa.breadth.mean())}
    for side, total, margin in (("cohort", n_coh, len(Fa)), ("flagged", n_flag, len(Oa))):
        n_col, p_col = f"n_{side}", f"pct_{side}"
        if pair_hidden(margin, total):
            Bw[n_col], Bw[p_col] = SUPP, np.nan
            margins[side], means[side] = SUPP, None
            n_hidden += len(B)
            continue
        hide, prim = partition_mask(B[n_col].values, avoid=[0])
        Bw[n_col] = marked(B[n_col].values, hide, prim)
        Bw.loc[hide, p_col] = np.nan
        margins[side] = int(margin)
        if hide.any():
            means[side] = None
        n_hidden += int(hide.sum())
    none_f = Bw.loc[Bw.breadth == 0, "n_flagged"].iloc[0]       # the no-criterion count, as written
    none_c = Bw.loc[Bw.breadth == 0, "n_cohort"].iloc[0]

    os.makedirs(OUT_DIR, exist_ok=True)
    F.rename_axis(ID_COL).reset_index().to_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), index=False)
    Sw.to_csv(os.path.join(OUT_DIR, "cutoff_summary.csv"), index=False)
    Bw.to_csv(os.path.join(OUT_DIR, "breadth_table.csv"), index=False)
    json.dump({"cdr": CDR, "encoder": ENCODER, "n_criteria": NB, "criteria": names,
               "n_cohort": n_coh, "n_flagged": one(n_flag), "n_criterion_items": len(need),
               "n_ace_categories": len(ACE_CATEGORIES), "disclosure": "All of Us dissemination policy, counts 1-20 suppressed",
               "n_cohort_all_evaluable": margins["cohort"], "n_flagged_all_evaluable": margins["flagged"],
               "withheld_per_criterion_cohort": {n: (SUPP if side_hidden[("cohort", n)] else int(withheld[n].sum()))
                                                 for n in names},
               "n_flagged_none_met": none_f, "n_flagged_undetermined": one(n_undet), "n_cohort_none_met": none_c,
               "mean_breadth_flagged_all_evaluable": means["flagged"],
               "mean_breadth_cohort_all_evaluable": means["cohort"]},
              open(os.path.join(OUT_DIR, "cutoff_summary.json"), "w"), indent=2,
              default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
    print(f"dissemination policy applied to cutoff_summary.csv, breadth_table.csv and cutoff_summary.json: "
          f"{n_hidden} cells or rows suppressed")
    print(f"\nwrote -> {OUT_DIR}")


if __name__ == "__main__":
    main()
