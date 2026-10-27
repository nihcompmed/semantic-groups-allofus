#!/usr/bin/env python
"""figure6.py -- Figure 6 and Figures S8 and S9: outliers below every screening threshold, under the semantic
groups and under the factor model (25 factors).

THE SETS, inside the participants meeting none of the 6 thresholds: missed (flagged under the encoder only), found
(flagged under both), factor-only. Who these people are uses the outlier definition (top 5%); their outcomes are compared
in the most atypical tenth (decile 10), because below every threshold the top-5% sets are too small for outcome counts to
be released, and deciles are the scale of every other outcome in the paper. What carries their distance is described in
the same decile-10 people. Association only, no tests on the top-3 frequencies.
  Figure 6               A  top 5%: per encoder, the 3 sets as one stacked bar, with the share found by the factor model
                         B  decile 10: % of each set with half of the squared distance from 1 questionnaire (on the
                            semantic groups' distance; each stand-alone question counted under the study it was taken from)
                         C  decile 10: barriers, the emergency department share on coded care, sleep, adjusted for age and
                            sex; brackets missed vs found and missed vs factor-only, with the prespecified encoder's
                            Holm-adjusted p and the number of encoders with a Holm-adjusted p < 0.05 in the same direction
  Figure S8              decile 10, the prespecified encoder's semantic groups among the 3 largest contributors
  Figure S9              decile 10, the factors (oblimin) among the 3 largest contributors of at least 15% of one set
Encoding as Figure 4 (figure4.py's helpers): missed black circles, found orange diamonds, factor-only green squares.
Drawn at its printed size, the 2-column width of 180 mm, with no text under 8 pt and no line under 2 pt (Frontiers figure
guidelines), in figure4.py's style: no gridlines, no white edges, each quantity named on its own axis with its unit, the
panel letters above the panels.

Reads   the downloads factor_below (below_sets.csv, below_outcomes.csv, below_units.csv), factor_sources
        (sources_below_sets.csv), factor_decile10 (decile10_factors.csv), and data/items_151.csv
Writes  output/figures/below_outliers_factor.{png,pdf}, below_outliers_factor_groups.{png,pdf},
        below_outliers_factor_factors.{png,pdf}
Run     python figure6.py
"""
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D    # noqa: E402
from matplotlib.patches import Patch   # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
from matplotlib.ticker import MaxNLocator   # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
from statsmodels.stats.multitest import multipletests   # noqa: E402

import figure4 as fo                    # noqa: E402  Figure 4's style, colors and helpers
from common import FIGURES, ITEMS_CSV, WB   # noqa: E402

SRC = os.path.join(WB, "factor_below")
GTE, ENCODERS = fo.GTE, fo.ENCODERS
SETS = {"enc_only": ("Missed (flagged under the semantic groups only)", "#000000", "#9A9A9A", "o", -0.22),
        "both": ("Found (flagged under both)", "#E69F00", "#F3CF80", "D", 0.0),
        "fac_only": ("Factor-only (flagged under the factor model only)", "#009E73", "#8FD3BE", "s", 0.22)}
SHORT = {"enc_only": "Missed", "both": "Found", "fac_only": "Factor-only"}
PAIRS = [("enc_only - both", "enc_only", "both"), ("enc_only - fac_only", "enc_only", "fac_only")]
# (the quantity and its unit, on the y axis; the outcomes on the x axis; the scale)
OUTS = [("Barriers to care, %", [("access", "Any"), ("cost", "Cost")], 100),
        ("ED share, %", [("ed_share_coded", "")], 100),
        ("Sleep per night, min", [("sleep_mean_min", "")], 1),
        ("Sleep variation, min", [("sleep_variation_min", "")], 1)]
FS, LW, MS, S_OTHER = fo.FS, fo.LW, fo.MS, fo.S_OTHER
W_IN = fo.W_IN
ROW = 0.32                              # supplements: inches per row
D_MIN = 15.0                            # panel D: factors among the 3 largest contributors of >= 15% of 1 set
CUT10 = "top 10%"


def holm(O, cut):
    c = O[(O.cut == cut) & O.set.isin([p[0] for p in PAIRS])].copy()
    c["p_holm"] = np.nan
    for e, g in c.groupby("encoder"):
        ok = g.p.notna()
        if ok.any():
            c.loc[g.index[ok], "p_holm"] = multipletests(g.p[ok], method="holm")[1]
    return c


def unit_rows(u, keep_min=None):
    """Per unit: the 3 sets' top-3 % for every encoder in u, the medians over encoders, and the sort key (missed
    minus the larger of the other 2 sets; units with the missed set withheld everywhere last, by their largest set)."""
    w = u.pivot_table(index=["unit", "name", "encoder"], columns="set", values="pct_top3").reset_index()
    for st in SETS:
        if st not in w:
            w[st] = np.nan
    med = w.groupby(["unit", "name"])[list(SETS)].median().reset_index()
    med["largest"] = med[list(SETS)].max(axis=1)
    if keep_min is not None:
        med = med[med.largest >= keep_min]
    else:
        med = med[med.largest.notna()]
    d = med["enc_only"] - med[["both", "fac_only"]].max(axis=1)
    med["key"] = np.where(d.notna(), d, -1000 + med["largest"])
    med = med.sort_values("key", ascending=False)
    return med, w


def unit_dots(ax, med, w, label_fn, fs=FS, x_fig=None, xmax=60):
    """One row per unit (top to bottom as med), the 3 sets dodged inside the row: gte large, the other encoders small."""
    n = len(med)
    dy = {"enc_only": 0.25, "both": 0.0, "fac_only": -0.25}
    for i, r in enumerate(med.itertuples()):
        y = n - 1 - i
        wu = w[w.unit == r.unit].set_index("encoder")
        gone = []
        for st, (_, col, colL, mk, _) in SETS.items():
            oth = [e for e in wu.index if e != GTE and np.isfinite(wu.loc[e, st])]
            ax.scatter(wu.loc[oth, st].values, np.full(len(oth), y + dy[st]), s=S_OTHER, marker=mk, color=colL, lw=0,
                       zorder=2)
            if GTE in wu.index and np.isfinite(wu.loc[GTE, st]):
                ax.plot(wu.loc[GTE, st], y + dy[st], marker=mk, color=col, ms=MS, lw=0, zorder=3)
            elif not oth:
                gone.append(st)
        if gone:                                # 1 note per row for every set withheld under every encoder, on its
            yn = np.mean([dy[st] for st in gone])   # own sub-row(s), at the right unless a neighbouring sub-row has
            near = [st for st in SETS if st not in gone and abs(dy[st] - yn) < 0.3]   # markers there
            left = bool(near) and np.nanmax(wu[near].values) > 0.45 * xmax
            ax.text(0.6 if left else xmax - 0.6, y + yn,
                    f"{', '.join(SHORT[st].lower() for st in gone)}: withheld (small counts)".capitalize(),
                    fontsize=FS, ha="left" if left else "right", va="center", color="#666666")
    ax.set_yticks(range(n))
    ax.set_yticklabels([label_fn(r) for r in med.itertuples()][::-1], fontsize=fs)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlim(0, xmax)
    fo.bands_y(ax, [n - 1 - k for k in range(1, n, 2)], x_fig)


def legend(fig, y, others="Other five sentence encoders (B\u2013D)"):
    h = [Line2D([], [], color=SETS[s][1], marker=SETS[s][3], lw=0, ms=MS + 1, label=SETS[s][0]) for s in SETS]
    h.append(Line2D([], [], color="#9A9A9A", marker="o", lw=0, ms=MS - 1, label=others))
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=FS, bbox_to_anchor=(0.5, y),
               handletextpad=0.4, handlelength=1.2, columnspacing=1.6)


def main():
    S = fo.sh_below(pd.read_csv(os.path.join(SRC, "below_sets.csv")))   # carries half: 25 sources
    O = pd.read_csv(os.path.join(SRC, "below_outcomes.csv"))
    U = pd.read_csv(os.path.join(SRC, "below_units.csv"))
    core = pd.read_csv(os.path.join(WB, "factor_decile10", "decile10_factors.csv"))
    core = core[(core.factor_model == fo.FM) & (core.rotation == fo.ROT)].set_index("factor")
    items = pd.read_csv(ITEMS_CSV).set_index("item")
    # laid out from the bottom up in inches; each panel letter sits in the space above its panel
    W = W_IN
    yC, hC = 0.32, 2.10                         # C, above the tick labels of its barriers panel
    y_note = yC + hC + 0.60                     # top of the 3-line note
    y_c = y_note + 0.08                         # the C heading
    yA, hA = y_c + 0.70, 1.12                   # A and B, above A's x label and B's 2-line tick labels
    y_ab = yA + hA + 0.10
    HF = y_ab + 0.62                            # the 2-row legend above (A) and (B)
    fig = plt.figure(figsize=(W, HF))

    # A: who the factor model finds, top 5%
    ax = fo.inch(fig, 1.46, yA, 3.20, hA)
    s5 = S[S.cut == "top 5%"].set_index(["encoder", "set"])
    for y, e in enumerate(ENCODERS[::-1]):
        left = 0
        for st in ("enc_only", "both", "fac_only"):
            w = s5.loc[(e, st), "n_set"]
            ax.barh(y, w, left=left, height=0.66, color=SETS[st][1], edgecolor="white", lw=0,
                    hatch="//" if st == "both" else None)
            left += w
        ax.text(left + 10, y, f"{s5.loc[(e, 'both'), 'found_by_factor_pct']:.0f}% found", va="center", fontsize=FS,
                color="#444444")
    ax.set_yticks(range(len(ENCODERS)))
    ax.set_yticklabels(ENCODERS[::-1], fontsize=FS)
    ax.set_ylim(-0.5, len(ENCODERS) - 0.5)
    ax.set_xlabel("Outliers meeting none of the six thresholds")
    ax.set_xlim(0, 760)
    ax.set_axisbelow(True)
    fo.tag(fig, 0.04, y_ab, "(A)", "Screen-negative outliers (top 5%)")

    # B: questionnaires, decile 10, on the semantic groups' distance
    ax = fo.inch(fig, 5.36, yA, W - 5.36 - 0.14, hA)
    s10 = S[S.cut == CUT10].set_index(["encoder", "set"])
    for x, st in enumerate(("enc_only", "both", "fac_only")):
        _, c, cL, mk, _ = SETS[st]
        v = s10.xs(st, level="set")["one_source_half_enc_pct"]
        oth = v[ENCODERS[1:]].dropna()
        ax.scatter(np.full(len(oth), x + 0.2), oth.values, s=S_OTHER, marker=mk, color=cL, lw=0, zorder=2)
        if np.isfinite(v[GTE]):
            ax.plot(x, v[GTE], marker=mk, color=c, ms=MS + 0.5, lw=0, zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Missed", "Found", "Factor-\nonly"], fontsize=FS)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(0, 70)
    fo.bands_x(ax, 3)
    ax.set_ylabel("At least half of the\nsquared distance, %", fontsize=FS)
    fo.tag(fig, 4.78, y_ab, "(B)", "One questionnaire (decile 10)")

    # C: outcomes, decile 10
    H = holm(O, CUT10)
    o10 = O[O.cut == CUT10]
    # each outcome panel: (left edge of its y label, room for the y label and tick labels, plot width), in inches
    slots = [(0.04, 0.50, 1.74), (2.62, 0.38, 0.80), (4.12, 0.48, 0.80), (5.74, 0.48, 0.80)]
    for (title, outs, sc), (l, m, wd) in zip(OUTS, slots):
        ax = fo.inch(fig, l + m, yC, wd, hC)
        tops = []
        for x, (oc, lab) in enumerate(outs):
            r = o10[o10.outcome == oc]
            for st, (_, col, colL, mk, dx) in SETS.items():
                rs = r[r.set == st].set_index("encoder")
                oth = [e for e in ENCODERS[1:] if e in rs.index and np.isfinite(rs.loc[e, "adj"])]
                ax.scatter(np.full(len(oth), x + dx + 0.09), sc * rs.loc[oth, "adj"].values, s=S_OTHER, marker=mk,
                           color=colL, lw=0, zorder=2)
                v, lo, hi = (sc * rs.loc[GTE, c] for c in ("adj", "adj_lo", "adj_hi"))
                ax.vlines(x + dx, lo, hi, color=col, lw=LW, zorder=3)
                ax.plot(x + dx, v, marker=mk, ms=MS, color=col, zorder=4, lw=0)
            top = sc * np.nanmax(r[r.set.isin(list(SETS))].adj_hi.values)
            for lev, (lab_p, a, b) in enumerate(PAIRS):
                c = H[(H.outcome == oc) & (H.set == lab_p)].set_index("encoder")
                sign = np.sign(c.loc[GTE, "adj"])
                n_sig = int(((c.p_holm < 0.05) & (np.sign(c.adj) == sign)).sum())
                tops.append((x + SETS[a][4], x + SETS[b][4], lev, top,
                             f"{fo.p_text(c.loc[GTE, 'p_holm'])}\n{n_sig} of {int(c.adj.notna().sum())}"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.margins(y=0.1)
        per_in = fo.headroom(ax, max(t[3] for t in tops), 0.80)     # 2 brackets, each with a 2-line label
        for xl, xr, lev, top, txt in tops:
            yb, tick = top + (0.07 + 0.38 * lev) * per_in, 0.04 * per_in
            ax.plot([xl, xl, xr, xr], [yb - tick, yb, yb, yb - tick], color="#444444", lw=LW, zorder=5,
                    solid_joinstyle="miter")
            ax.text((xl + xr) / 2, yb + 0.03 * per_in, txt, ha="center", va="bottom", fontsize=FS, color="#444444",
                    linespacing=1.05)
        ax.set_xticks(range(len(outs)))
        ax.set_xticklabels([lab for _, lab in outs])
        ax.set_xlim(-0.6, len(outs) - 0.4)
        if len(outs) == 1:
            ax.set_xticks([])
        elif len(outs) > 1:
            fo.bands_x(ax, len(outs))
        ax.set_ylabel(title, labelpad=3)
    fo.tag(fig, 0.04, y_c, "(C)", "Outcomes in decile 10, adjusted for age and sex")
    fig.text(0.04 / W, y_note / HF, "Brackets: missed vs found (lower) and missed vs factor-only (upper).\n$p$: "
             "gte-large-en-v1.5, Holm-adjusted over its contrasts.\nn of N: encoders with Holm-adjusted $p$ < 0.05 in "
             "the same direction.", fontsize=FS, color="#444444", ha="left", va="top", linespacing=1.25)

    legend(fig, 1 - 0.03 / HF, others="Other five sentence encoders (B, C)")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"below_outliers_factor.{ext}"), dpi=300)
    plt.close(fig)
    print(f"-> {os.path.join(FIGURES, 'below_outliers_factor.png')}")

    # Figure S9: factors, decile 10, all six encoders
    u = U[(U.cut == CUT10) & (U.grouping == "factor") & (U.rotation == fo.ROT)]
    med, w = unit_rows(u, keep_min=D_MIN)
    h_in = ROW * len(med)
    fig = plt.figure(figsize=(W_IN, 0.55 + h_in + 0.86))
    ax = fo.inch(fig, 3.60, 0.55, W_IN - 3.60 - 0.14, h_in)
    unit_dots(ax, med, w, lambda r: fo.wrap(fo.profile_label("F", r.unit, r.name)
                                            + fo.stress_wording(core, items, r.unit), 52), x_fig=0.01)
    ax.set_xlabel("Participants with the factor as a top-3 contributor, %")
    fo.title_at(ax, "Factors contributing most to the distance of screen-negative participants\nunder the factor model "
                "(25 factors) in decile 10", 0.01)
    legend(fig, 1 - 0.02 / fig.get_figheight(), others="Other five sentence encoders")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"below_outliers_factor_factors.{ext}"), dpi=300)
    plt.close(fig)
    print(f"-> {os.path.join(FIGURES, 'below_outliers_factor_factors.png')}")
    print(med[["unit", "name", "enc_only", "both", "fac_only", "key"]].round(1).to_string(index=False))

    # Figure S8: the prespecified encoder's semantic groups, decile 10
    u = U[(U.cut == CUT10) & (U.grouping == "semantic groups") & (U.encoder == GTE)]
    med, w = unit_rows(u)
    h_in = ROW * len(med)
    fig = plt.figure(figsize=(W_IN, 0.72 + h_in + 0.86))
    ax = fo.inch(fig, 4.45, 0.72, W_IN - 4.45 - 0.14, h_in)
    unit_dots(ax, med, w, lambda r: fo.wrap(fo.profile_label("G", r.unit, r.name), 56), x_fig=0.01)
    ax.set_xlabel("Participants with the group as a\ntop-3 contributor, %")
    fo.title_at(ax, "Semantic groups (gte-large-en-v1.5) contributing most to its distance,\nscreen-negative participants "
                "in decile 10", 0.01)
    h = [Line2D([], [], color=SETS[s][1], marker=SETS[s][3], lw=0, ms=MS + 1, label=SETS[s][0]) for s in SETS]
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=FS,
               bbox_to_anchor=(0.5, 1 - 0.03 / fig.get_figheight()), handletextpad=0.4, handlelength=1.2,
               columnspacing=1.6)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"below_outliers_factor_groups.{ext}"), dpi=300)
    plt.close(fig)
    print(f"-> {os.path.join(FIGURES, 'below_outliers_factor_groups.png')}")


if __name__ == "__main__":
    main()
