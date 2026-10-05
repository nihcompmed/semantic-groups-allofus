#!/usr/bin/env python
"""fig_outcome_deciles.py -- Figure 2 as ONE figure: measures outside the answers' own scoring, by decile of the
distance from the cohort's typical scores, prespecified encoder (gte-large-en-v1.5), each measure in its own units.

LOCAL. Draws from downloaded aggregates only:
  outcome_deciles/outcome_deciles.csv          thresholds met, barriers, sleep     (workbench/outcome_deciles.py)
  ed_check/ed_check_deciles.csv                ED share on coded care               (workbench/ed_check.py)
  conditions_deciles_jno/conditions_deciles_jno.csv  recorded conditions            (workbench/conditions_deciles_jno.py,
                                                     the 8 conditions of record)

4 PANELS (B and C each stack 2 measures on the shared decile axis, each on its
own y-axis; no twin axes). x = decile 1, closest to the cohort's typical scores, to 10, furthest; every value
adjusted for age and sex; points with 95% CIs; the dashed gray line is the cohort's value.
  A  screening thresholds met, mean number of 6 (people with all 6 evaluable)
  B  top: barriers to care, %: any access barrier, any cost barrier, provider bias
     bottom: emergency department visits, % of visits (people with at least 1 visit and 1 ICD event)
  C  top: mean sleep, minutes (people with at least 14 nights)
     bottom: night-to-night sleep variation, minutes (median of personal SDs; also adjusted for number of nights)
  D  recorded diagnosis, %, the 8 conditions of record (conditions_jno.py), stacked like B and C:
     top = the 4 mental and behavioral (solid), bottom = the 4 physical (dashed), each on its
     own y-axis. Populations: people with at least 1 ICD event; the tested for hypercholesterolemia (cholesterol
     measurement or lipid panel), breast cancer (women with a mammogram) and prostate cancer (men with a blood PSA)

OUTPUTS  ../outputs/outcome_deciles/fig_r2_deciles.{pdf,png}, and a PNG preview in ../../jno/previews/ when
         ../../jno/ exists

Run:  python fig_outcome_deciles.py [--encoder gte-large-en-v1.5]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WB = os.path.join(HERE, "..", "outputs", "workbench", "R2025Q4R6")
OUT = os.path.join(HERE, "..", "outputs", "outcome_deciles")
PREVIEW = os.path.join(HERE, "..", "..", "jno", "previews")
ENC = sys.argv[sys.argv.index("--encoder") + 1] if "--encoder" in sys.argv else "gte-large-en-v1.5"

BLACK, GRAY = "#000000", "#8C8C8C"
# barriers: one blue lightness ramp, marker shape as the 2nd channel, so no barrier color comes close to a
# condition's hue; the conditions keep distinct hues; labels in the text's terms
BARRIERS = [("access", "Any barrier", "#08306B", "o"),
            ("cost", "Cost", "#2171B5", "s"),
            ("avoid", "Clinician differs", "#6BAED6", "^")]
# the 8 conditions of record: mental and behavioral in Okabe-Ito hues, solid; physical
# in grays, dashed, told apart by marker and direct label
MENTAL = [("gad", "GAD", "#D55E00", "o", "-"),
          ("adhd", "ADHD", "#CC79A7", "^", "-"),
          ("ptsd", "PTSD", "#009E73", "s", "-"),
          ("panic", "Panic disorder", "#0072B2", "D", "-")]
PHYSICAL = [("hyperchol", "High cholesterol", "#4D4D4D", "D", "--"),
            ("asthma", "Asthma", "#7F7F7F", "s", "--"),
            ("prostate", "Prostate cancer", "#4D4D4D", "v", "--"),
            ("breast", "Breast cancer", "#A6A6A6", "o", "--")]
X = np.arange(1, 11)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6})


def series(ax, x, y, lo, hi, color, marker, ls="-", ms=3.6):
    ax.errorbar(x, y, yerr=[y - lo, hi - y], color=color, marker=marker, ms=ms, lw=1.1, ls=ls, elinewidth=0.8,
                capsize=0, mec=color, mfc=color, zorder=3)


def cohort_line(ax, v):
    ax.axhline(v, color=GRAY, lw=0.8, ls=(0, (4, 3)), zorder=1)


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
        ax.annotate(text, xy=(x0[0] if x0 else 10, y0), xytext=(10.55, y), textcoords="data", va="center", ha="left", fontsize=7,
                    color=color if text_in_color else "#222222", fontweight="bold" if text_in_color else "normal",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.6, shrinkA=0, shrinkB=2) if abs(y - y0) > gap * 0.2 else None)


def main():
    od = pd.read_csv(os.path.join(WB, "outcome_deciles", "outcome_deciles.csv"))
    ed = pd.read_csv(os.path.join(WB, "ed_check", "ed_check_deciles.csv"))
    ph = pd.read_csv(os.path.join(WB, "conditions_deciles_jno", "conditions_deciles_jno.csv"))
    g = od[od.grouping == ENC].set_index("decile").loc[X]
    c = od[od.grouping == "cohort"].iloc[0]
    e = ed[ed.grouping == ENC].set_index("decile").loc[X]
    ec = ed[ed.grouping == "cohort"].iloc[0]

    # 4 panel slots: B and C each stack 2 measures that share the decile axis,
    # each on its own y-axis in its own units (no twin axes)
    fig = plt.figure(figsize=(7.0, 6.6))
    outer = fig.add_gridspec(2, 2, left=0.115, right=0.855, top=0.955, bottom=0.085, wspace=0.45, hspace=0.30)
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
    Bt.set_ylim(0, 75)
    from matplotlib.lines import Line2D                    # a legend in the empty upper left (checked against the marks)
    Bt.legend([Line2D([], [], color=col, marker=mk, ms=3.6, lw=1.1) for _, _, col, mk in BARRIERS],
              [lab for _, lab, _, _ in BARRIERS], loc="upper left", frameon=False, fontsize=6.5, handlelength=1.6,
              borderaxespad=0.1, labelspacing=0.2)

    series(Bb, X, 100 * e.ed_adj.values, 100 * e.ed_adj_lo.values, 100 * e.ed_adj_hi.values, BLACK, "o")
    cohort_line(Bb, 100 * ec.ed_adj)
    Bb.set_ylabel("ED visits,\n% of visits")
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
    DODGE = [-0.18, -0.06, 0.06, 0.18]
    for ax, conds, ylab, ylim, gap, tint in ((Dt, MENTAL, "Mental and behavioral\nconditions, %", (0, 16), 2.2, True),
                                             (Db, PHYSICAL, "Physical\nconditions, %", (0, 33), 3.2, False)):
        lab_items = []
        for (key, lab, col, mk, ls), dx in zip(conds, DODGE):
            x = ph[(ph.grouping == ENC) & (ph.condition == key)].sort_values("decile_from")
            xm = (x.decile_from.values + x.decile_to.values) / 2.0 + dx
            y, lo, hi = 100 * x.adj.values, 100 * x.adj_lo.values, 100 * x.adj_hi.values
            series(ax, xm, y, lo, hi, col, mk, ls=ls, ms=3.0)
            for r in x.itertuples():                     # a pooled bin shows its span
                if r.decile_to != r.decile_from:
                    ax.plot([r.decile_from + dx, r.decile_to + dx], [100 * r.adj] * 2, color=col, lw=0.8, ls=ls)
            lab_items.append((y[-1], lab, col, xm[-1]))
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        labels_right(ax, lab_items, gap, text_in_color=tint)   # colored labels name the colored lines

    for ax in axes:
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(X)
        ax.grid(axis="y", color="#E6E6E6", lw=0.5, zorder=0)
    for ax in (Bt, Ct, Dt):                              # the upper plot of a stack shares the lower one's x-axis
        plt.setp(ax.get_xticklabels(), visible=False)
    for ax in (A, Bb):
        ax.set_xlabel("Decile of distance", labelpad=2)
    for ax in (Cb, Db):
        ax.set_xlabel("Decile of distance\n(1 = most typical, 10 = most atypical)", labelpad=2)
    fig.canvas.draw()                                     # panel letters at one x per column, level with the slot tops
    xl = {0: A.get_position().x0 - 0.085, 1: Bt.get_position().x0 - 0.085}
    for ax, tag, col in ((A, "A", 0), (Bt, "B", 1), (Ct, "C", 0), (Dt, "D", 1)):
        fig.text(xl[col], ax.get_position().y1 + 0.012, tag, fontsize=10, fontweight="bold", va="bottom")
    fig.align_ylabels([Bt, Bb])
    fig.align_ylabels([Ct, Cb])
    fig.align_ylabels([Dt, Db])

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_r2_deciles.{ext}"), dpi=300)
    # the preview goes to the manuscript folder only when it exists, so a copy of the code elsewhere writes only
    # inside its own outputs/
    if os.path.isdir(os.path.dirname(PREVIEW)):
        os.makedirs(PREVIEW, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW, "fig_r2_deciles.png"), dpi=200)
    print("wrote", os.path.join(OUT, "fig_r2_deciles.pdf"))


if __name__ == "__main__":
    main()
