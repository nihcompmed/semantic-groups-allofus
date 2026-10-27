#!/usr/bin/env python
"""figure2_supplement1.py -- Figure S3: the 8 recorded conditions by decile of the distance under all 6
encoders, adjusted for age and sex (Figure 2D shows the prespecified encoder alone).

Row 1 the 4 mental and behavioral conditions, row 2 the 4 physical. In each panel 1 line per encoder: the prespecified
encoder in black with its 95% CIs, the other 5 as thin colored lines. A pooled bin would be drawn at the middle of its
deciles with a line across them. The dashed gray line is the share in the condition's whole population.

Reads   the download deciles (conditions.csv)
Writes  output/figures/deciles_conditions_encoders.{pdf,png}
Run     python figure2_supplement1.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import matplotlib.ticker           # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

from common import CONDITIONS, WB, save   # noqa: E402

# the prespecified encoder first, black; the other 5 in Okabe-Ito hues
ENCODERS = [("gte-large-en-v1.5", "gte-large-en-v1.5 (prespecified)", "#000000"), ("bge-m3", "bge-m3", "#E69F00"),
            ("bge-large-en-v1.5", "bge-large-en-v1.5", "#56B4E9"), ("all-mpnet-base-v2", "all-mpnet-base-v2", "#009E73"),
            ("e5-large-v2", "e5-large-v2", "#CC79A7"), ("qwen3-embedding-0.6b", "Qwen3-Embedding-0.6B", "#0072B2")]
WRAP = {"Generalized anxiety disorder": "Generalized anxiety\ndisorder",
        "Attention-deficit/hyperactivity disorder": "Attention-deficit/\nhyperactivity disorder",
        "Posttraumatic stress disorder": "Posttraumatic stress\ndisorder"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.5})


def main():
    ph = pd.read_csv(os.path.join(WB, "deciles", "conditions.csv"))
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 4.4))
    fig.subplots_adjust(left=0.08, right=0.975, top=0.91, bottom=0.215, wspace=0.42, hspace=0.62)
    handles = []
    for ax, (key, label, _, _) in zip(axs.ravel(), CONDITIONS):
        c0 = ph[(ph.grouping == "population") & (ph.condition == key)].iloc[0]
        ax.axhline(100 * c0.rate, color="#8C8C8C", lw=0.7, ls=(0, (4, 3)), zorder=1)
        top = 100 * c0.rate
        for i, (enc, _, col) in enumerate(ENCODERS[::-1]):        # the prespecified encoder drawn last, on top
            x = ph[(ph.grouping == enc) & (ph.condition == key)].sort_values("decile_from")
            xm = (x.decile_from.values + x.decile_to.values) / 2.0
            y, lo, hi = 100 * x.adj.values, 100 * x.adj_lo.values, 100 * x.adj_hi.values
            if enc == ENCODERS[0][0]:
                h = ax.errorbar(xm, y, yerr=[y - lo, hi - y], color=col, marker="o", ms=2.4, lw=1.0, elinewidth=0.6,
                                capsize=0, zorder=4)
                top = max(top, np.nanmax(hi))
            else:
                h, = ax.plot(xm, y, color=col, marker="o", ms=1.6, lw=0.7, zorder=3)
                top = max(top, np.nanmax(y))
            for r in x.itertuples():
                if r.decile_to != r.decile_from:
                    ax.plot([r.decile_from, r.decile_to], [100 * r.adj] * 2, color=col, lw=0.6)
            if key == CONDITIONS[0][0]:
                handles.append((enc, h))
        ax.set_title(WRAP.get(label, label), loc="left")
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks([1, 5, 10])
        ax.set_ylim(0, top * 1.15)
        ax.grid(axis="y", color="#E6E6E6", lw=0.4, zorder=0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))   # whole percentages
    for ax in axs[:, 0]:
        ax.set_ylabel("Recorded diagnosis, %")
    for ax in axs[-1]:
        ax.set_xlabel("Decile of distance")
    h = dict(handles)
    fig.legend([h[e] for e, _, _ in ENCODERS], [lab for _, lab, _ in ENCODERS], loc="lower center", ncol=3,
               frameon=False, fontsize=6.5, bbox_to_anchor=(0.52, 0.0), handlelength=1.8, columnspacing=1.6)
    save(fig, "deciles_conditions_encoders")


if __name__ == "__main__":
    main()
