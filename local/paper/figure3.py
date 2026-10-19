#!/usr/bin/env python
"""figure3.py -- Figure 3: the semantic groups contributing most to the outliers' distance.

Per group: the percentage of the set with the group among its 3 largest shares of the squared distance (the download
contributors). The 10 most frequent groups of each panel are drawn as bars, largest at the top, each labeled by its group
number, then every construct of its core items with its number of core items. A group that is a top-3 contributor for
20 or fewer participants is blank in the download and is not drawn.
  A  the outliers (distance above the 95th percentile under the prespecified encoder)
  B  those of them who met none of the 6 screening thresholds

Reads   the download contributors (contributors_top3_all.csv, contributors_top3_below.csv)
Writes  elife/figures/contributors.{pdf,png}
Run     python figure3.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import pandas as pd               # noqa: E402
from matplotlib.transforms import blended_transform_factory   # noqa: E402

from common import PRIMARY, WB, display_profile, save   # noqa: E402

ENC = PRIMARY
TOP = 10
TITLE = {"A": "Groups contributing most to the distance of outliers (n = {n})",   # outliers defined in the Methods
         "B": "Groups contributing most to the distance of screen-negative outliers (n = {n})"}
BAR = "#0072B2"                    # Okabe-Ito blue, one color (no second grouping is encoded)
WRAP = 60                          # characters per label line; wraps between constructs only

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6})


def row_label(cp, j):
    """The row label: group number, then each construct (display name) with its number of core items."""
    names = [p.replace(":", " ") for p in display_profile(cp).split("; ")] if isinstance(cp, str) and cp else []
    lines, cur = [], ""
    for k, n in enumerate(names):
        piece = n + ("," if k < len(names) - 1 else "")
        if cur and len(cur) + 1 + len(piece) > WRAP:
            lines.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    lines.append(cur)
    return f"G{j + 1}  " + "\n".join(lines)  # displayed group ID = G + (component index + 1), G1-G27


def load(tag):
    f = "contributors_top3_below.csv" if tag == "B" else "contributors_top3_all.csv"
    d = pd.read_csv(os.path.join(WB, "contributors", f))
    d = d[(d.encoder == ENC) & (d.ranking == "share")].copy()
    d["pct"] = pd.to_numeric(d.pct_top, errors="coerce")
    d = d[d.pct.notna()]                                   # blank = 20 or fewer participants: not drawn
    d = d.sort_values(["pct", "component"], ascending=[False, True]).head(TOP)
    return d


def main():
    A, B = load("A"), load("B")
    shown = pd.concat([A, B]).drop_duplicates("component")
    label = {r.component: row_label(r.semantic_groups, int(r.component)) for r in shown.itertuples()}
    xmax = 10 * (int(max(A.pct.max(), B.pct.max()) // 10) + 1)
    # panels stacked (A over B) so each has the full label column at print width (7.0 in); shared percentage axis
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
    fig.subplots_adjust(left=0.475, right=0.955, top=0.945, bottom=0.085, hspace=0.36)
    for ax, d, tag in ((axs[0], A, "A"), (axs[1], B, "B")):
        y = list(range(len(d)))[::-1]
        ax.barh(y, d.pct, color=BAR, height=0.68, zorder=2)
        for yy, v, nn in zip(y, d.pct, d.n_top.astype(int)):
            ax.text(v + xmax * 0.012, yy, f"{v:.1f} (n = {nn})", va="center", ha="left", fontsize=7, color="#222222")
        ax.set_yticks(y)
        ax.set_yticklabels([label[c] for c in d.component], fontsize=7, linespacing=1.0)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, xmax + 13)                          # room for the value label "47.0 (n = 119)" of the longest bar
        ax.set_xticks(range(0, xmax + 1, 10))
        ax.set_ylim(-0.6, len(d) - 0.4)
        ax.grid(axis="x", color="#E6E6E6", lw=0.6, zorder=0)
        ax.tick_params(axis="x", labelbottom=True)
        n = int(d.n_set.iloc[0])
        # full-width titles from the left edge of the figure, in the paper's own terms: what the panel shows and
        # for whom
        title = TITLE[tag].format(n=n)
        ax.text(0.01, 1.035, f"{tag}  {title}", transform=blended_transform_factory(fig.transFigure, ax.transAxes),
                ha="left", va="bottom", fontsize=8)
    axs[1].set_xlabel("Participants for whom the group was a top-3 contributor, %")
    save(fig, "contributors")
    for tag, d in (("A", A), ("B", B)):
        print(tag, [(label[c].replace(chr(10), " "), p) for c, p in zip(d.component, d.pct)])


if __name__ == "__main__":
    main()
