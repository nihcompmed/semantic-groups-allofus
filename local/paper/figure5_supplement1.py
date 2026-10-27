#!/usr/bin/env python
"""figure5_supplement1.py -- Figure S7: the outcomes of Figure 5 under all 6 encoders, one panel per
outcome, each with its own y axis.

The prespecified encoder with 95% CIs, the other 5 encoders as thin gray lines. Percentages start at 0; minutes of sleep
are not forced to 0. The conditions are drawn per bin (adjacent deciles pooled where a bin had 1 to 20 cases or
non-cases), at the bin's middle with a line over its deciles. All values adjusted for age and sex (the joint model).

Reads   the downloads deciles (below_threshold_outcomes.csv, below_threshold_conditions.csv) and deciles_ed (ed_deciles.csv)
Writes  output/figures/below_threshold_encoders.{png,pdf}
Run     python figure5_supplement1.py
"""
import os

import numpy as np
import pandas as pd

from common import CONDITIONS as _C, FIGURES, PRIMARY, WB

OUT = os.path.join(FIGURES, "below_threshold_encoders.png")
MAIN = PRIMARY
BLUE, GRAY = "#0072B2", "#9a9a9a"
SHORT = {c[0]: c[2] for c in _C}
# panel titles in the text's terms
CARE = [("access", "Any barrier to care, %", 100, True), ("cost", "Cost barrier, %", 100, True),
        ("avoid", "Delayed care over clinicians'\nrace, religion, or language, %", 100, True), ("ed", "ED share, %", 100, True),
        ("sleep_mean", "Sleep per night, min", 1, False), ("sleep_var_median", "Sleep variation, min", 1, False)]
CONDS = ["gad", "adhd", "ptsd", "panic", "hyperchol", "asthma", "breast", "prostate"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    O = pd.read_csv(os.path.join(WB, "deciles", "below_threshold_outcomes.csv"))
    C = pd.read_csv(os.path.join(WB, "deciles", "below_threshold_conditions.csv"))
    E = pd.read_csv(os.path.join(WB, "deciles_ed", "ed_deciles.csv"))   # the emergency share on coded care
    E = E[E.scope == "below_threshold"].set_index(["grouping", "decile"])
    for c in ("ed_adj", "ed_adj_lo", "ed_adj_hi"):
        O[c] = [E[c].get((g, d), np.nan) for g, d in zip(O.grouping, O.decile)]
    encs = [e for e in O.grouping.unique() if e != "group"]
    assert MAIN in encs
    num = lambda s: pd.to_numeric(s, errors="coerce")
    fig, axes = plt.subplots(4, 4, figsize=(10.5, 9.6))
    panels = [("care", x) for x in CARE] + [("cond", x) for x in CONDS]
    for ax, (kind, spec) in zip(axes.ravel(), panels):
        if kind == "care":
            key, title, scale, zero = spec
            col = "ed_adj" if key == "ed" else f"{key}_adj"
            for e in encs:
                g = O[O.grouping == e].sort_values("decile")
                x, v = g.decile.values, scale * num(g[col]).values
                if e == MAIN:
                    lo, hi = scale * num(g[f"{col}_lo"]).values, scale * num(g[f"{col}_hi"]).values
                    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.15, lw=0)
                    ax.plot(x, v, "-o", color=BLUE, lw=2, ms=4, zorder=3)
                else:
                    ax.plot(x, v, "-", color=GRAY, lw=0.8, zorder=2)
        else:
            key = spec
            title, zero = f"{SHORT[key]}, %", True
            for e in encs:
                g = C[(C.grouping == e) & (C.condition == key)].sort_values("decile_from")
                mid = (g.decile_from.values + g.decile_to.values) / 2
                v = 100 * g.adj.values
                if e == MAIN:
                    ax.errorbar(mid, v, yerr=[v - 100 * g.adj_lo.values, 100 * g.adj_hi.values - v], color=BLUE,
                                fmt="o", ms=4, lw=1.2, capsize=0, zorder=3)
                    ax.plot(mid, v, "-", color=BLUE, lw=2, zorder=3)
                    for a, b, vv in zip(g.decile_from, g.decile_to, v):
                        if b > a:
                            ax.plot([a, b], [vv, vv], color=BLUE, lw=1, alpha=0.6, zorder=3)
                else:
                    ax.plot(mid, v, "-", color=GRAY, lw=0.8, zorder=2)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xticks([1, 5, 10])
        ax.set_xlim(0.5, 10.5)
        if zero:
            ax.set_ylim(0, None)
        ax.grid(axis="y", color="0.9", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    nrow, ncol = axes.shape                          # the lowest VISIBLE panel of each column carries the x label
    for j in range(ncol):
        i = max(i for i in range(nrow) if i * ncol + j < len(panels))
        axes[i, j].set_xlabel("Decile of the distance within the set\n(1 = most typical, 10 = most atypical)", fontsize=8)
    # every mark named
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=BLUE, lw=2, marker="o", ms=4, label="Prespecified encoder"),
               Patch(facecolor=BLUE, alpha=0.15, lw=0, label="95% CI (barriers, ED share, sleep)"),
               Line2D([], [], color=BLUE, lw=0, marker="|", ms=9, mew=1.2, label="95% CI (conditions)"),
               Line2D([], [], color=BLUE, lw=1, alpha=0.6, label="Deciles pooled together"),
               Line2D([], [], color=GRAY, lw=0.8, label="Other five sentence encoders")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.replace(".png", ".pdf"))
    print(f"-> {OUT} (+ .pdf)")


if __name__ == "__main__":
    main()
