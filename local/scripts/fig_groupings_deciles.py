#!/usr/bin/env python
"""fig_groupings_deciles.py -- eFigure 4: the measures of Figure 2 by decile of the distance under every
grouping, the 6 sentence encoders and the 6 other groupings of eMethods 5.

LOCAL. Draws from downloaded aggregates only:
  outcome_deciles/outcome_deciles.csv   every grouping by decile, adjusted for age and sex
                                        (workbench/outcome_deciles.py), the same file as Figure 2's panels A to C
The emergency department share is left out: its coded-care version exists for the 6 encoders only (eMethods 5).

6 PANELS, each measure in its own units on its own y-axis (no twin axes); x = decile 1 (closest to the cohort's
typical answers) to 10 (furthest), each grouping with its own deciles; the dotted gray line is the cohort's value.
95% CIs on the prespecified encoder and the 6 other groupings, each
series nudged sideways by a fixed offset (DODGE) so the bars do not sit on top of each other; the lines join the
unshifted decile positions' values drawn at the shifted x. The other 5 encoders stay without bars (their CIs at every
decile are in eTable 7).
ENCODING. The prespecified encoder black; the other 5 encoders thin light gray (one family, identity not needed);
the 6 other groupings in 6 Okabe-Ito hues (yellow left out: too light on white) AND a marker + line style each, so no
line is identified by color alone: word counts dotted with diamonds, principal components and the factor model solid
with a circle and a square, MCA dashed with triangles; filled = the grouping's own Horn's k, open = the second k.
(Okabe-Ito is a published colorblind-safe set, and every hue has a second channel.)

OUTPUTS  ../outputs/outcome_deciles/fig_groupings_deciles.{pdf,png}, and a PNG preview in ../../jno/previews/ when
         ../../jno/ exists

Run:  python fig_groupings_deciles.py
"""
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D   # noqa: E402
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WB = os.path.join(HERE, "..", "outputs", "workbench", "R2025Q4R6")
OUT = os.path.join(HERE, "..", "outputs", "outcome_deciles")
PREVIEW = os.path.join(HERE, "..", "..", "jno", "previews")
X = np.arange(1, 11)
GTE = "gte-large-en-v1.5"
OTHER_ENC = ["bge-m3", "bge-large-en-v1.5", "all-mpnet-base-v2", "e5-large-v2", "qwen3-embedding-0.6b"]
# (grouping, legend label, color, marker, filled, linestyle)
OTHERS = [("tfidf4", "Word counts (TF-IDF), k = 4", "#E69F00", "D", True, ":"),
          ("tfidf25", "Word counts (TF-IDF), k = 25", "#E69F00", "D", False, ":"),
          ("pca", "Principal components, k = 25", "#0072B2", "o", True, "-"),
          ("factor", "Factor model, k = 25", "#009E73", "s", True, "-"),
          ("mca90", "Multiple correspondence analysis, k = 90", "#D55E00", "^", True, "--"),
          ("mca", "Multiple correspondence analysis, k = 25", "#CC79A7", "^", False, "--")]
# (column, panel title, scale)
PANELS = [("mean_breadth", "Screening thresholds met,\nmean No.", 1),
          ("access", "Any barrier to care, %", 100),
          ("cost", "Cost barrier, %", 100),
          ("avoid", "Delayed care because a\nclinician differs, %", 100),
          ("sleep_mean", "Mean sleep per night, min", 1),
          ("sleep_var_median", "Sleep variation, min", 1)]
COHORT_COL = {"mean_breadth": "mean_breadth", "access": "access_rate", "cost": "cost_rate", "avoid": "avoid_rate",
              "sleep_mean": "sleep_mean", "sleep_var_median": "sleep_var_median"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6})


# sideways offset per series with CI bars, in decile units: the 6 other groupings, then the prespecified encoder
DODGE = {"tfidf4": -0.27, "tfidf25": -0.18, "pca": -0.09, GTE: 0.0, "factor": 0.09, "mca90": 0.18, "mca": 0.27}


def series(od, g, col, sc, ci=False):
    t = od[od.grouping == g].copy()
    t["decile"] = t["decile"].astype(int)
    t = t.set_index("decile").loc[X]
    v = sc * t[f"{col}_adj"].values
    if not ci:
        return v
    return v, sc * t[f"{col}_adj_lo"].values, sc * t[f"{col}_adj_hi"].values


def main():
    od = pd.read_csv(os.path.join(WB, "outcome_deciles", "outcome_deciles.csv"))
    coh = od[od.grouping == "cohort"].iloc[0]
    need = {GTE, *OTHER_ENC, *[o[0] for o in OTHERS]}
    assert need <= set(od.grouping), f"missing groupings: {need - set(od.grouping)}"
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.2))
    for i, (ax, (col, title, sc)) in enumerate(zip(axes.ravel(), PANELS)):
        ax.axhline(sc * coh[COHORT_COL[col]], color="#8C8C8C", lw=0.8, ls=(0, (1, 2)), zorder=1)
        for g in OTHER_ENC:
            ax.plot(X, series(od, g, col, sc), color="#C8C8C8", lw=0.9, zorder=2)
        for g, _, c, m, filled, ls in OTHERS:
            v, lo, hi = series(od, g, col, sc, ci=True)
            xs = X + DODGE[g]
            ax.vlines(xs, lo, hi, color=c, lw=0.6, zorder=3)
            ax.plot(xs, v, color=c, lw=1.1, ls=ls, marker=m, ms=3.2,
                    mfc=c if filled else "white", mec=c, mew=0.8, zorder=3)
        v, lo, hi = series(od, GTE, col, sc, ci=True)
        ax.vlines(X + DODGE[GTE], lo, hi, color="#000000", lw=0.7, zorder=4)
        ax.plot(X + DODGE[GTE], v, color="#000000", lw=1.5, marker="o", ms=2.6, zorder=4)
        ax.set_title(f"{'ABCDEF'[i]}  " + title.replace("\n", "\n     "), loc="left", fontweight="normal")
        ax.set_xticks(X)
        ax.set_xlim(0.6, 10.4)
        ax.grid(axis="y", color="#E6E6E6", lw=0.5)
        ax.set_axisbelow(True)
        if i >= 3:
            ax.set_xlabel("Decile of the distance")
    handles = [Line2D([], [], color="#000000", lw=1.5, marker="o", ms=2.6,
                      label="gte-large-en-v1.5 (prespecified), k = 27"),
               Line2D([], [], color="#C8C8C8", lw=0.9, label="Other 5 sentence encoders, k = 23 to 27")]
    handles += [Line2D([], [], color=c, lw=1.1, ls=ls, marker=m, ms=3.2, mfc=c if f else "white", mec=c, mew=0.8,
                       label=lab) for _, lab, c, m, f, ls in OTHERS]
    handles.append(Line2D([], [], color="#8C8C8C", lw=0.8, ls=(0, (1, 2)), label="Cohort"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7, handlelength=2.6,
               columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1), h_pad=1.4, w_pad=1.2)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_groupings_deciles.{ext}"), dpi=200)
    # the preview only where the folder ../../jno exists, as in fig_outcome_deciles.py; the outputs do not depend on it
    if os.path.isdir(os.path.dirname(PREVIEW)):
        os.makedirs(PREVIEW, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW, "fig_groupings_deciles.png"), dpi=200)
    print("wrote", os.path.join(OUT, "fig_groupings_deciles.pdf"))


if __name__ == "__main__":
    main()
