#!/usr/bin/env python
"""figure2.py -- Figure 2: the outcomes recorded outside the 151 items, by decile of the distance from the cohort's mean
scores, under the prespecified encoder (gte-large-en-v1.5), each in its own units.

x = decile 1 (most typical) to 10 (most atypical); every value adjusted for age and sex by the joint model (Workbench
step 7); points with 95% CIs; the dashed gray line is the cohort's value.
  A  screening thresholds met, mean number of the 6 (participants with all 6 evaluable)
  B  top: barriers to care, %: any barrier, cost, a clinician's race, religion, or language
     bottom: emergency department visits, % of visits (participants with at least 1 visit and 1 ICD-coded event)
  C  top: minutes of sleep per night (at least 14 Fitbit nights); bottom: night-to-night variation, minutes (median of
     the personal SDs, also adjusted for the number of nights)
  D  recorded diagnosis, %: top the 4 mental and behavioral conditions, bottom the 4 physical (populations in
     outcomes.py of the Workbench code); a pooled bin is drawn at its middle with a line over the deciles it pools

Figure 5 draws its panels with this script's style and helpers (series, cohort_line, labels_right).

Drawn at its printed size, the 2-column width of 180 mm, with no text under 8 pt and no line under 2 pt (Frontiers
figure guidelines); no gridlines.

Reads   the downloads deciles (outcomes.csv, conditions.csv) and deciles_ed (ed_deciles.csv)
Writes  output/figures/deciles.{pdf,png}
Run     python figure2.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

from common import PRIMARY, WB, save   # noqa: E402

ENC = PRIMARY

BLACK, GRAY = "#000000", "#8C8C8C"
# barriers: one blue lightness ramp, marker shape as the 2nd channel, so no barrier color comes close to a
# condition's hue; the conditions keep distinct hues; labels in the text's terms
BARRIERS = [("access", "Any barrier", "#08306B", "o"),
            ("cost", "Cost", "#2171B5", "s"),
            ("avoid", "Clinician race,\nreligion, or language", "#6BAED6", "^")]
# the 8 conditions of record: mental and behavioral in Okabe-Ito hues, solid; physical
# in grays, dashed, told apart by marker and direct label
MENTAL = [("gad", "GAD", "#D55E00", "o", "-"),
          ("adhd", "ADHD", "#CC79A7", "^", "-"),
          ("ptsd", "PTSD", "#009E73", "s", "-"),
          ("panic", "Panic disorder", "#0072B2", "D", "-")]
PHYSICAL = [("hyperchol", "Hypercholesterolemia", "#4D4D4D", "D", "--"),
            ("asthma", "Asthma", "#7F7F7F", "s", "--"),
            ("prostate", "Prostate cancer", "#4D4D4D", "v", "--"),
            ("breast", "Breast cancer", "#A6A6A6", "o", "--")]
X = np.arange(1, 11)
FS, FS_TAG = 8, 10                  # pt: text, panel letters
LW = 2.0                            # pt: every line
MS = 5.0                            # pt: marker size
W_IN = 180 / 25.4                   # the 2-column width
# the 4 series of a conditions plot are dodged sideways by DODGE (Figures 2 and 5)
DODGE = [-0.315, -0.105, 0.105, 0.315]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
                     "xtick.labelsize": FS, "ytick.labelsize": FS, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": LW, "xtick.major.width": LW,
                     "ytick.major.width": LW, "xtick.major.size": 4, "ytick.major.size": 4, "lines.linewidth": LW})


def series(ax, x, y, lo, hi, color, marker, ls="-", ms=MS):
    ax.errorbar(x, y, yerr=[y - lo, hi - y], color=color, marker=marker, ms=ms, lw=LW, ls=ls, elinewidth=LW,
                capsize=0, mec=color, mfc=color, mew=0, zorder=3)


def cohort_line(ax, v):
    ax.axhline(v, color=GRAY, lw=LW, ls=(0, (2.5, 2)), zorder=1)


def labels_right(ax, items, gap, text_in_color=False):
    """Direct labels just right of decile 10, spread apart symmetrically (at least `gap` data units) so a crowded pair
    moves half each way, with a short leader line when a label moved."""
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for _ in range(200):
        moved = False
        for i in range(1, len(ys)):
            d = ys[i] - ys[i - 1]
            if d < gap:
                ys[i - 1] -= (gap - d) / 2
                ys[i] += (gap - d) / 2
                moved = True
        if not moved:
            break
    for (y0, text, color, *x0), y in zip(items, ys):
        ax.annotate(text, xy=(x0[0] if x0 else 10, y0), xytext=(10.6, y), textcoords="data", va="center", ha="left", fontsize=FS,
                    color=color if text_in_color else "#222222", fontweight="bold" if text_in_color else "normal",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, lw=LW, shrinkA=0, shrinkB=2) if abs(y - y0) > gap * 0.2 else None)


def main():
    od = pd.read_csv(os.path.join(WB, "deciles", "outcomes.csv"))
    ed = pd.read_csv(os.path.join(WB, "deciles_ed", "ed_deciles.csv"))
    ed = ed[ed.scope == "cohort"]                      # coded care, the cohort's deciles
    ph = pd.read_csv(os.path.join(WB, "deciles", "conditions.csv"))
    g = od[od.grouping == ENC].set_index("decile").loc[X]
    c = od[od.grouping == "cohort"].iloc[0]
    e = ed[ed.grouping == ENC].set_index("decile").loc[X]
    ec = ed[ed.grouping == "cohort"].iloc[0]

    # 4 panel slots: B and C each stack 2 measures that share the decile axis,
    # each on its own y-axis in its own units (no twin axes)
    fig = plt.figure(figsize=(W_IN, 7.6))
    outer = fig.add_gridspec(2, 2, left=0.115, right=0.795, top=0.955, bottom=0.085, wspace=0.50, hspace=0.30)
    A = fig.add_subplot(outer[0, 0])
    gB = outer[0, 1].subgridspec(2, 1, hspace=0.14)
    Bt = fig.add_subplot(gB[0]); Bb = fig.add_subplot(gB[1], sharex=Bt)
    gC = outer[1, 0].subgridspec(2, 1, hspace=0.14)
    Ct = fig.add_subplot(gC[0]); Cb = fig.add_subplot(gC[1], sharex=Ct)
    gD = outer[1, 1].subgridspec(2, 1, hspace=0.14)
    Dt = fig.add_subplot(gD[0]); Db = fig.add_subplot(gD[1], sharex=Dt)
    axes = [A, Bt, Bb, Ct, Cb, Dt, Db]

    series(A, X, g.mean_breadth_adj.values, g.mean_breadth_adj_lo.values, g.mean_breadth_adj_hi.values, BLACK, "o")
    cohort_line(A, c.mean_breadth)
    A.set_ylabel("Screening thresholds met, mean No.")
    A.set_ylim(0, 3)

    for key, lab, col, mk in BARRIERS:
        s = 100
        series(Bt, X, s * g[f"{key}_adj"].values, s * g[f"{key}_adj_lo"].values, s * g[f"{key}_adj_hi"].values, col, mk)
        cohort_line(Bt, s * c[f"{key}_rate"])
    Bt.set_ylabel("Reporting a\nbarrier, %")
    Bt.set_ylim(0, 80)
    Bt.set_yticks([0, 20, 40, 60, 80])
    from matplotlib.lines import Line2D                    # the legend in the empty space right of the plot
    Bt.legend([Line2D([], [], color=col, marker=mk, ms=MS, lw=LW, mew=0) for _, _, col, mk in BARRIERS],
              [lab for _, lab, _, _ in BARRIERS], loc="upper left", bbox_to_anchor=(0.98, 1.05), frameon=False,
              fontsize=FS, handlelength=1.3, handletextpad=0.5, borderaxespad=0.1, labelspacing=0.35)

    series(Bb, X, 100 * e.ed_adj.values, 100 * e.ed_adj_lo.values, 100 * e.ed_adj_hi.values, BLACK, "o")
    cohort_line(Bb, 100 * ec.ed_adj)
    Bb.set_ylabel("ED share, %")
    Bb.set_ylim(0, 4)

    series(Ct, X, g.sleep_mean_adj.values, g.sleep_mean_adj_lo.values, g.sleep_mean_adj_hi.values, BLACK, "o")
    cohort_line(Ct, c.sleep_mean)
    Ct.set_ylabel("Sleep per\nnight, min")
    Ct.set_ylim(350, 395)

    series(Cb, X, g.sleep_var_median_adj.values, g.sleep_var_median_adj_lo.values, g.sleep_var_median_adj_hi.values,
           BLACK, "o")
    cohort_line(Cb, c.sleep_var_median)
    Cb.set_ylabel("Sleep\nvariation, min")
    Cb.set_ylim(70, 105)

    # the 4 series of each plot are dodged sideways by DODGE: PTSD and GAD end at 13.4% and 13.5% in decile 10, so
    # their points and intervals would otherwise sit on top of each other
    for ax, conds, ylab, ylim, gap, tint in ((Dt, MENTAL, "Mental and behavioral\nconditions, %", (0, 16), 2.2, True),
                                             (Db, PHYSICAL, "Physical\nconditions, %", (0, 33), 3.2, False)):
        lab_items = []
        for (key, lab, col, mk, ls), dx in zip(conds, DODGE):
            x = ph[(ph.grouping == ENC) & (ph.condition == key)].sort_values("decile_from")
            xm = (x.decile_from.values + x.decile_to.values) / 2.0 + dx
            y, lo, hi = 100 * x.adj.values, 100 * x.adj_lo.values, 100 * x.adj_hi.values
            series(ax, xm, y, lo, hi, col, mk, ls=ls, ms=4.4)
            for r in x.itertuples():                     # a pooled bin shows its span
                if r.decile_to != r.decile_from:
                    ax.plot([r.decile_from + dx, r.decile_to + dx], [100 * r.adj] * 2, color=col, lw=LW, ls=ls)
            lab_items.append((y[-1], lab, col, xm[-1]))
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        labels_right(ax, lab_items, gap, text_in_color=tint)   # colored labels name the colored lines

    for ax in axes:
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(X)
    for ax in (Bt, Ct, Dt):                              # the upper plot of a stack shares the lower one's x-axis
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in (A, Bb):
        ax.set_xlabel("Decile of distance", labelpad=2)
    for ax in (Cb, Db):
        ax.set_xlabel("Decile of distance\n(1 = most typical, 10 = most atypical)", labelpad=2)
    fig.canvas.draw()                                     # panel letters at one x per column, level with the slot tops
    xl = {0: A.get_position().x0 - 0.095, 1: Bt.get_position().x0 - 0.095}
    for ax, tag, col in ((A, "(A)", 0), (Bt, "(B)", 1), (Ct, "(C)", 0), (Dt, "(D)", 1)):
        fig.text(xl[col], ax.get_position().y1 + 0.016, tag, fontsize=FS_TAG, fontweight="bold", va="bottom")
    fig.align_ylabels([Bt, Bb])
    fig.align_ylabels([Ct, Cb])
    fig.align_ylabels([Dt, Db])

    save(fig, "deciles")


if __name__ == "__main__":
    main()
