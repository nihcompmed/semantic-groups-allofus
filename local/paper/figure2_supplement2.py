#!/usr/bin/env python
"""figure2_supplement2.py -- Figure 2-figure supplement 2: mean age and the percentage of women by decile of the
distance, under every grouping.

Panels  A  mean age (years), B  women (%); x = decile 1 (most typical) to 10 (most atypical); one line per grouping: the
6 sentence encoders in color, the 6 groupings formed without a sentence encoder in black with markers. Error bars are
95% CIs (normal for the mean age, Wilson for the percentage). The dashed gray line is the whole cohort.

Reads   the download deciles (outcomes.csv: grouping, decile, n_set, mean_age, sd_age, share_female, unadjusted)
Writes  elife/figures/deciles_age_sex.{pdf,png}
Run     python figure2_supplement2.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

from common import WB, save   # noqa: E402

SRC = os.path.join(WB, "deciles", "outcomes.csv")
Z = 1.959964
X = np.arange(1, 11)

# names as printed in the paper and its source data
ENCODERS = [("gte-large-en-v1.5", "gte-large-en-v1.5 (prespecified)", "#0072B2"),
            ("bge-m3", "bge-m3", "#D55E00"),
            ("bge-large-en-v1.5", "bge-large-en-v1.5", "#009E73"),
            ("all-mpnet-base-v2", "all-mpnet-base-v2", "#CC79A7"),
            ("e5-large-v2", "e5-large-v2", "#E69F00"),
            ("qwen3-embedding-0.6b", "Qwen3-Embedding-0.6B", "#56B4E9")]
INK, GRAY = "#1a1a1a", "#8c8c8c"
OTHERS = [("tfidf4", "Word counts (TF-IDF), Horn's k", (0, (1.2, 1.5)), "D", False),
          ("tfidf25", "Word counts (TF-IDF), 25 groups", (0, (1.2, 1.5)), "D", True),
          ("pca", "Principal components of the answers", "-", "o", True),
          ("factor", "Factor model of the answers", "-", "s", True),
          ("mca90", "Multiple correspondence analysis, Horn's k", (0, (5, 2)), "^", False),
          ("mca", "Multiple correspondence analysis, 25 groups", (0, (5, 2)), "^", True)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6})


def wilson(p, n):
    c = p + Z * Z / (2 * n)
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    d = 1 + Z * Z / n
    return (c - h) / d, (c + h) / d


def main():
    od = pd.read_csv(SRC)
    coh = od[od.grouping == "cohort"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.1))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.36, wspace=0.28)
    handles = []
    for name, label, *style in [(n, l, c, "-", None, True) for n, l, c in ENCODERS] + \
                               [(n, l, INK, ls, mk, filled) for n, l, ls, mk, filled in OTHERS]:
        color, ls, mk, filled = style
        g = od[od.grouping == name].set_index("decile").loc[X]
        n = g.n_set.values.astype(float)
        age, se = g.mean_age.values, g.sd_age.values / np.sqrt(n)
        fem = g.share_female.values
        flo, fhi = wilson(fem, n)
        kw = dict(color=color, ls=ls, lw=1.1 if mk is None else 0.9, marker=mk, ms=3.2,
                  mfc=(color if filled else "white"), mec=color, mew=0.8, elinewidth=0.7, capsize=0, zorder=3)
        h = axes[0].errorbar(X, age, yerr=Z * se, label=label, **kw)
        axes[1].errorbar(X, 100 * fem, yerr=[100 * (fem - flo), 100 * (fhi - fem)], **kw)
        handles.append(h)
    for ax, v, title, ylab in [(axes[0], coh.mean_age, "A  Mean age", "Mean age (years)"),
                               (axes[1], 100 * coh.share_female, "B  Women", "Women (%)")]:
        ax.axhline(v, color=GRAY, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylab)
        ax.set_xticks(X)
        ax.set_xlabel("Decile of the distance")
        ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
    cl, = axes[0].plot([], [], color=GRAY, lw=0.8, ls=(0, (4, 3)), label="Whole cohort")
    fig.legend(handles + [cl], [h.get_label() for h in handles] + ["Whole cohort"], loc="lower center", ncol=3,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.4)
    save(fig, "deciles_age_sex")
    for name, label, *_ in ENCODERS + OTHERS:
        g = od[od.grouping == name].set_index("decile")
        print(f"{label:46s} age {g.mean_age[1]:.1f} -> {g.mean_age[10]:.1f}   women {100 * g.share_female[1]:.1f} -> "
              f"{100 * g.share_female[10]:.1f}")
    print("cohort", coh.mean_age, coh.share_female)


if __name__ == "__main__":
    main()
