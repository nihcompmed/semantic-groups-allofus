#!/usr/bin/env python
"""figure7_source_data.py -- Figure 7-source data 1: the semantic groups and the 8 recorded conditions under each of
the 6 encoders.

The models are those without the item on a diagnosed life-threatening illness (weight 0 in every group; the *_without
columns): each condition on age, age squared and sex (sex left out for the 2 cancers) and the group scores, each group on
its own or adjusted for the other groups (all groups in 1 regression). A log odds ratio per SD is given as its absolute value (a group's sign is arbitrary),
its 95% CI as absolute values (lower limit 0 when the CI includes 0).

COUNTERPARTS, the rule of the agreement across encoders (semantic_groups/step10_agreement.py): for each other encoder, a
prespecified-encoder group's counterpart is its partner in the optimal one-to-one assignment on the Jaccard overlap of
core items, kept when the overlap is at least tau (outputs/xenc/summary.json).

SHEETS
  Joint tests            per encoder and condition: cases, non-cases, groups, the Wald chi-square of all groups in the
                         regression with all groups, its df, the P value Holm-adjusted over the encoder's 8 conditions; the
                         same with the social-position covariates
  Groups                 per encoder, condition and group: its counterpart among the prespecified encoder's groups, the
                         constructs of its core items; on its own and adjusted for the other groups, the absolute log odds
                         ratio with its CI and its rank among the encoder's groups; adjusted, the P value Holm-adjusted over the
                         condition's groups
  Ranks across encoders  per prespecified-encoder group and condition: the rank of the group and of its counterpart under
                         each encoder, on its own and adjusted for the other groups ("-" = no counterpart)
  Social position        the same models with income, education, health insurance, and any access or cost barrier to
                         care added as covariates: the absolute log odds ratio, CI, rank, the change from the model
                         without them, and the Holm-adjusted p with all groups
CHECKS  the models without the covariates in conditions_adjusted equal those of conditions; the prespecified encoder's
        rows give the values the text reports for posttraumatic stress disorder.

Reads   the downloads conditions (condition_models.csv, condition_groups.csv) and conditions_adjusted
        (adjusted_<encoder>.csv, adjusted_groups_<encoder>.csv); outputs/cores/<encoder>_components.csv,
        outputs/xenc/summary.json
Writes  elife/supplementary/Figure7_source_data_1.xlsx (+ csv/)
Run     python figure7_source_data.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from common import CONDITIONS, ENCODERS, OUTPUTS, PRIMARY, WB
from workbook import holm, names, pv, write

CONDS = [c[0] for c in CONDITIONS]
LABEL = {c[0]: c[1] for c in CONDITIONS}
README = [
    ("Figure 7-source data 1", "Semantic groups and recorded conditions under each of the 6 encoders (Figure 7 and "
                               "Figure 7-figure supplements 1-5)."),
    ("Models", "Logistic regression of each recorded condition on age, age squared, and sex (sex left out for breast "
               "and prostate cancer) and the group scores, each group on its own or adjusted for the other groups (all groups "
               "in 1 regression) "
               "(Appendix 5). The illness item had weight 0 in every group. Scores are standardized over the cohort."),
    ("Absolute log odds ratio", "The absolute value of the log odds ratio per SD of a group's score. A group's sign "
                                "is arbitrary. The 95% CI is given as absolute values, with lower limit 0 when the CI "
                                "includes 0."),
    ("Rank", "Rank of the absolute log odds ratio among the encoder's groups for the condition, on its own or adjusted (1 = "
             "largest, ties share the smaller rank)."),
    ("Counterpart", "The prespecified encoder's (gte-large-en-v1.5) group matched by the overlap of core items "
                    "(Appendix 2); empty when the group has no counterpart."),
    ("p values", "Joint tests: Holm-adjusted over the 8 conditions of an encoder. Groups: Holm-adjusted over the "
                 "condition's groups adjusted for the other groups."),
    ("Sheet Joint tests", "Wald test of all groups together, in the regression with all groups."),
    ("Sheet Groups", "Every group, condition, and encoder, on its own and adjusted for the other groups."),
    ("Sheet Ranks across encoders", "For each group of the prespecified encoder, its rank and its counterparts' "
                                    "ranks under the other 5 encoders ('-' = no counterpart)."),
    ("Sheet Social position", "The same models with income, education, and health insurance (The Basics) and any "
                              "access or cost barrier to care (Health Care Access and Utilization survey) added as "
                              "covariates, nonresponse kept as a category (Appendix 5). Change = absolute log odds "
                              "ratio with these covariates minus without them. The Joint tests sheet gives the Wald "
                              "statistic and Holm-adjusted p with these covariates in its last 2 columns."),
]


def partners(tau):
    """For each other encoder: prespecified-encoder group -> its counterpart (both 0-based)."""
    sets = {e: [set(s.split()) for s in pd.read_csv(os.path.join(OUTPUTS, "cores", f"{e}_components.csv"))
                .sort_values("component")["items"]] for e in ENCODERS}
    P = {}
    for e in ENCODERS[1:]:
        J = np.array([[len(a & b) / len(a | b) for b in sets[e]] for a in sets[PRIMARY]])
        ri, ci = linear_sum_assignment(1.0 - J)
        P[e] = {int(i): int(j) for i, j in zip(ri, ci) if J[i, j] >= tau}
    return P


def size_ci(o, lo, hi):
    """|log OR| and its CI as sizes."""
    a, b = np.log(lo), np.log(hi)
    s = np.abs(np.log(o))
    slo = np.where(np.sign(a) == np.sign(b), np.minimum(np.abs(a), np.abs(b)), 0.0)
    return s, slo, np.maximum(np.abs(a), np.abs(b))


def ranks(s, index):
    return pd.Series(s, index=index).rank(ascending=False, method="min").astype(int)


def main():
    J = pd.read_csv(os.path.join(WB, "conditions", "condition_models.csv"))
    G = pd.read_csv(os.path.join(WB, "conditions", "condition_groups.csv"))
    tau = json.load(open(os.path.join(OUTPUTS, "xenc", "summary.json")))["tau"]
    adj = os.path.join(WB, "conditions_adjusted")
    AJ = {e: pd.read_csv(os.path.join(adj, f"adjusted_{e}.csv")).set_index("condition") for e in ENCODERS}
    AG = {e: pd.read_csv(os.path.join(adj, f"adjusted_groups_{e}.csv")) for e in ENCODERS}
    P = partners(tau)
    inv = {e: {j: i for i, j in P[e].items()} for e in ENCODERS[1:]}       # other encoder's group -> prespecified group
    prof = {e: pd.read_csv(os.path.join(OUTPUTS, "cores", f"{e}_components.csv")).set_index("component").construct_profile
            for e in ENCODERS}
    counterpart = lambda e, j: (j if e == PRIMARY else inv[e].get(j))

    jrows = []
    for e in ENCODERS:
        x = J[J.encoder == e].set_index("condition").loc[CONDS]
        h = holm(x.p_joint_without.values)
        aj = AJ[e].loc[CONDS]
        assert np.allclose(aj.groups_wald_before.values, x.wald_without.values), f"{e}: conditions_adjusted 'before' != conditions"
        ha = holm(aj.groups_p_after.values)
        assert (x.k == x.k.iloc[0]).all()
        for c, r, ph, wa, pa in zip(CONDS, x.itertuples(), h, aj.groups_wald_after.values, ha):
            jrows.append({"encoder": e, "condition": LABEL[c], "cases": int(r.n_cases), "non_cases": int(r.n_noncases),
                          "groups": int(r.k), "wald_chi_square": round(float(r.wald_without), 1), "df": int(r.k),
                          "p_holm": pv(ph), "wald_chi_square_social_position": round(float(wa), 1),
                          "p_holm_social_position": pv(pa)})
    JT = pd.DataFrame(jrows)

    grows = []
    for e in ENCODERS:
        for c in CONDS:
            g = G[(G.encoder == e) & (G.condition == c)].set_index("group").sort_index()
            sa, la, ha = size_ci(g.or_alone_without, g.or_alone_lo_without, g.or_alone_hi_without)
            sw, lw, hw = size_ci(g.or_without, g.or_lo_without, g.or_hi_without)
            ra, rw = ranks(sa, g.index), ranks(sw, g.index)
            ph = holm(g.p_without.values)
            for i, j in enumerate(g.index):
                cp = counterpart(e, j)
                grows.append({"encoder": e, "condition": LABEL[c], "group": f"G{j + 1}",
                              "counterpart_in_gte": f"G{cp + 1}" if cp is not None else "",
                              "core_constructs": names(prof[e][j]),
                              "alone_abs_log_or": round(float(sa[i]), 3), "alone_ci_low": round(float(la[i]), 3),
                              "alone_ci_high": round(float(ha[i]), 3), "alone_rank": int(ra[j]),
                              "all_groups_abs_log_or": round(float(sw[i]), 3), "all_groups_ci_low": round(float(lw[i]), 3),
                              "all_groups_ci_high": round(float(hw[i]), 3), "all_groups_rank": int(rw[j]),
                              "all_groups_p_holm": pv(ph[i])})
    GR = pd.DataFrame(grows)

    arows = []
    for e in ENCODERS:
        A = AG[e]
        for c in CONDS:
            a = A[A.condition == c].set_index("group").sort_index()
            g = G[(G.encoder == e) & (G.condition == c)].set_index("group").sort_index()
            assert np.allclose(a.or_before.values, g.or_without.values), f"{e} {c}: conditions_adjusted 'before' != conditions"
            sa, la, ha = size_ci(a.or_alone_after, a.or_alone_after_lo, a.or_alone_after_hi)
            sw, lw, hw = size_ci(a.or_after, a.or_after_lo, a.or_after_hi)
            ra, rw = ranks(sa, a.index), ranks(sw, a.index)
            da = sa - np.abs(np.log(g.or_alone_without.values))
            dw = sw - np.abs(np.log(g.or_without.values))
            ph = holm(a.p_after.values)
            for i, j in enumerate(a.index):
                cp = counterpart(e, j)
                arows.append({"encoder": e, "condition": LABEL[c], "group": f"G{j + 1}",
                              "counterpart_in_gte": f"G{cp + 1}" if cp is not None else "",
                              "core_constructs": names(prof[e][j]),
                              "alone_abs_log_or": round(float(sa[i]), 3), "alone_ci_low": round(float(la[i]), 3),
                              "alone_ci_high": round(float(ha[i]), 3), "alone_rank": int(ra[j]),
                              "alone_change": round(float(da[i]), 3),
                              "all_groups_abs_log_or": round(float(sw[i]), 3), "all_groups_ci_low": round(float(lw[i]), 3),
                              "all_groups_ci_high": round(float(hw[i]), 3), "all_groups_rank": int(rw[j]),
                              "all_groups_change": round(float(dw[i]), 3), "all_groups_p_holm": pv(ph[i])})
    SP = pd.DataFrame(arows)

    rrows = []
    for c in CONDS:
        for i in range(len(prof[PRIMARY])):
            row = {"gte_group": f"G{i + 1}", "core_constructs": names(prof[PRIMARY][i]), "condition": LABEL[c]}
            for model, col in (("alone", "alone_rank"), ("all_groups", "all_groups_rank")):
                for e in ENCODERS:
                    j = i if e == PRIMARY else P[e].get(i)
                    if j is None:
                        row[f"{model}_rank_{e}"] = "-"
                    else:
                        v = GR[(GR.encoder == e) & (GR.condition == LABEL[c]) & (GR.group == f"G{j + 1}")][col]
                        row[f"{model}_rank_{e}"] = int(v.iloc[0])
            rrows.append(row)
    RK = pd.DataFrame(rrows)

    # the prespecified encoder's rows give the values the text reports
    g = GR[(GR.encoder == PRIMARY) & (GR.condition == LABEL["ptsd"])].set_index("group")
    assert abs(g.loc["G16", "alone_abs_log_or"] - 0.929) < 6e-4 and g.loc["G16", "alone_rank"] == 1
    assert abs(g.loc["G6", "all_groups_abs_log_or"] - 0.551) < 6e-4 and g.loc["G6", "all_groups_rank"] == 1
    assert len(GR) == sum(len(prof[e]) for e in ENCODERS) * len(CONDS)
    write("Figure7_source_data_1", README, [("Joint tests", JT), ("Groups", GR), ("Ranks across encoders", RK),
                                            ("Social position", SP)], widths=(28, 120))


if __name__ == "__main__":
    main()
