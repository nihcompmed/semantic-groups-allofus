#!/usr/bin/env python
"""figure5.py -- Figure 5: the outcomes of Figure 2 by decile of the distance within the participants who met none of the
6 screening thresholds, under the prespecified encoder, drawn the way Figure 2 is (figure2.py's style and helpers).

Points are adjusted values with 95% CIs (the joint model); the dashed gray line is the value among all participants
below every threshold; a pooled bin is drawn at its middle with a line over the deciles it pools.
  A  barriers to care, %
  B  emergency department visits, % of visits (coded care)
  C  top: sleep per night, min; bottom: night-to-night variation, min (median)
  D  top: the 4 mental and behavioral conditions, %; bottom: the 4 physical, %

Reads   the downloads deciles (below_threshold_outcomes.csv, below_threshold_conditions.csv) and deciles_ed
Writes  elife/figures/below_threshold.{pdf,png}
Run     python figure5.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

import figure2 as f2                                                    # noqa: E402  Figure 2's style and helpers
from common import PRIMARY, WB, save                                     # noqa: E402

ENC = PRIMARY
X = np.arange(1, 11)


def main():
    O = pd.read_csv(os.path.join(WB, "deciles", "below_threshold_outcomes.csv"))
    C = pd.read_csv(os.path.join(WB, "deciles", "below_threshold_conditions.csv"))
    E = pd.read_csv(os.path.join(WB, "deciles_ed", "ed_deciles.csv"))
    E = E[E.scope == "below_threshold"]
    g = O[O.grouping == ENC].set_index("decile").loc[X]
    grp = O[O.grouping == "group"].iloc[0]
    e = E[E.grouping == ENC].set_index("decile").loc[X]
    eg = E[E.grouping == "group"].iloc[0]

    fig = plt.figure(figsize=(7.0, 6.6))
    outer = fig.add_gridspec(2, 2, left=0.115, right=0.825, top=0.955, bottom=0.085, wspace=0.45, hspace=0.30)
    A = fig.add_subplot(outer[0, 0])
    B = fig.add_subplot(outer[0, 1])
    gC = outer[1, 0].subgridspec(2, 1, hspace=0.14)
    Ct = fig.add_subplot(gC[0]); Cb = fig.add_subplot(gC[1], sharex=Ct)
    gD = outer[1, 1].subgridspec(2, 1, hspace=0.14)
    Dt = fig.add_subplot(gD[0]); Db = fig.add_subplot(gD[1], sharex=Dt)
    axes = [A, B, Ct, Cb, Dt, Db]

    for key, lab, col, mk in f2.BARRIERS:
        s = 100
        f2.series(A, X, s * g[f"{key}_adj"].values, s * g[f"{key}_adj_lo"].values, s * g[f"{key}_adj_hi"].values, col, mk)
        f2.cohort_line(A, s * grp[f"{key}_rate"])
    A.set_ylabel("Reporting a barrier, %")
    A.set_ylim(0, 45)
    A.legend([Line2D([], [], color=col, marker=mk, ms=3.6, lw=1.1) for _, _, col, mk in f2.BARRIERS],
             [lab for _, lab, _, _ in f2.BARRIERS], loc="upper left", frameon=False, fontsize=6.5, handlelength=1.6,
             borderaxespad=0.1, labelspacing=0.2)

    f2.series(B, X, 100 * e.ed_adj.values, 100 * e.ed_adj_lo.values, 100 * e.ed_adj_hi.values, f2.BLACK, "o")
    f2.cohort_line(B, 100 * eg.ed_adj)
    B.set_ylabel("ED share, %")
    B.set_ylim(0, 2)

    f2.series(Ct, X, g.sleep_mean_adj.values, g.sleep_mean_adj_lo.values, g.sleep_mean_adj_hi.values, f2.BLACK, "o")
    f2.cohort_line(Ct, grp.sleep_mean)
    Ct.set_ylabel("Sleep per\nnight, min")
    Ct.set_ylim(360, 400)
    f2.series(Cb, X, g.sleep_var_median_adj.values, g.sleep_var_median_adj_lo.values, g.sleep_var_median_adj_hi.values,
              f2.BLACK, "o")
    f2.cohort_line(Cb, grp.sleep_var_median)
    Cb.set_ylabel("Sleep\nvariation, min")
    Cb.set_ylim(65, 90)

    DODGE = [-0.18, -0.06, 0.06, 0.18]                   # as Figure 2: the 4 series of a plot side by side
    for ax, conds, ylab, ylim, gap, tint in ((Dt, f2.MENTAL, "Mental and behavioral\nconditions, %", (0, 8), 0.9, True),
                                             (Db, f2.PHYSICAL, "Physical\nconditions, %", (0, 40), 3.6, False)):
        lab_items = []
        for (key, lab, col, mk, ls), dx in zip(conds, DODGE):
            x = C[(C.grouping == ENC) & (C.condition == key)].sort_values("decile_from")
            xm = (x.decile_from.values + x.decile_to.values) / 2.0 + dx
            y, lo, hi = 100 * x.adj.values, 100 * x.adj_lo.values, 100 * x.adj_hi.values
            f2.series(ax, xm, y, lo, hi, col, mk, ls=ls, ms=3.0)
            for r in x.itertuples():                     # a pooled bin shows its span
                if r.decile_to != r.decile_from:
                    ax.plot([r.decile_from + dx, r.decile_to + dx], [100 * r.adj] * 2, color=col, lw=0.8, ls=ls)
            lab_items.append((y[-1], lab, col, xm[-1]))
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        f2.labels_right(ax, lab_items, gap, text_in_color=tint)

    for ax in axes:
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(X)
        ax.grid(axis="y", color="#E6E6E6", lw=0.5, zorder=0)
    for ax in (Ct, Dt):
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in (A, B):
        ax.set_xlabel("Decile of distance within the set", labelpad=2)
    for ax in (Cb, Db):
        ax.set_xlabel("Decile of distance within the set\n(1 = most typical, 10 = most atypical)", labelpad=2)
    fig.canvas.draw()
    xl = {0: A.get_position().x0 - 0.085, 1: B.get_position().x0 - 0.085}
    for ax, tag, col in ((A, "A", 0), (B, "B", 1), (Ct, "C", 0), (Dt, "D", 1)):
        fig.text(xl[col], ax.get_position().y1 + 0.012, tag, fontsize=10, fontweight="bold", va="bottom")
    fig.align_ylabels([Ct, Cb])
    fig.align_ylabels([Dt, Db])

    save(fig, "below_threshold")


if __name__ == "__main__":
    main()
