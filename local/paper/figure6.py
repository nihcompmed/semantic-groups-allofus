#!/usr/bin/env python
"""figure6.py -- Figure 6 and its figure supplements 1 and 2: outliers below every screening threshold, under the semantic
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
  figure supplement 1    decile 10, the prespecified encoder's semantic groups among the 3 largest contributors
  figure supplement 2    decile 10, the factors (oblimin) among the 3 largest contributors of at least 15% of one set
Encoding as Figure 4 (figure4.py's helpers): missed black circles, found orange diamonds, factor-only green squares.

Reads   the downloads factor_below (below_sets.csv, below_outcomes.csv, below_units.csv), factor_sources
        (sources_below_sets.csv), factor_decile10 (decile10_factors.csv), and data/items_151.csv
Writes  elife/figures/below_outliers_factor.{png,pdf}, below_outliers_factor_groups.{png,pdf},
        below_outliers_factor_factors.{png,pdf}
Run     python figure6.py
"""
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D    # noqa: E402
from matplotlib.patches import Patch   # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
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
OUTS = [("Barriers to care, %", [("access", "Any"), ("cost", "Cost")], 100),
        ("ED share, %", [("ed_share_coded", "")], 100),
        ("Sleep per\nnight, min", [("sleep_mean_min", "")], 1),
        ("Sleep\nvariation, min", [("sleep_variation_min", "")], 1)]
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


def unit_dots(ax, med, w, label_fn, fs=6.2, x_fig=None, xmax=60):
    """One row per unit (top to bottom as med), the 3 sets dodged inside the row: gte large, the other encoders small."""
    n = len(med)
    dy = {"enc_only": 0.25, "both": 0.0, "fac_only": -0.25}
    for i, r in enumerate(med.itertuples()):
        y = n - 1 - i
        wu = w[w.unit == r.unit].set_index("encoder")
        gone = []
        for st, (_, col, colL, mk, _) in SETS.items():
            oth = [e for e in wu.index if e != GTE and np.isfinite(wu.loc[e, st])]
            ax.scatter(wu.loc[oth, st].values, np.full(len(oth), y + dy[st]), s=6, marker=mk, color=colL, lw=0,
                       zorder=2)
            if GTE in wu.index and np.isfinite(wu.loc[GTE, st]):
                ax.plot(wu.loc[GTE, st], y + dy[st], marker=mk, color=col, ms=4.0, mec="white", mew=0.4, lw=0,
                        zorder=3)
            elif not oth:
                gone.append(st)
        if gone:                                # 1 note per row for every set withheld under every encoder, on its
            yn = np.mean([dy[st] for st in gone])   # own sub-row(s), at the right unless a neighbouring sub-row has
            near = [st for st in SETS if st not in gone and abs(dy[st] - yn) < 0.3]   # markers there
            left = bool(near) and np.nanmax(wu[near].values) > 0.45 * xmax
            ax.text(0.6 if left else xmax - 0.6, y + yn,
                    f"{', '.join(SHORT[st].lower() for st in gone)}: withheld (small counts)".capitalize(),
                    fontsize=5.4, ha="left" if left else "right", va="center", color="#666666")
    ax.set_yticks(range(n))
    ax.set_yticklabels([label_fn(r) for r in med.itertuples()][::-1], fontsize=fs)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlim(0, xmax)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    fo.bands_y(ax, [n - 1 - k for k in range(1, n, 2)], x_fig)


def legend(fig, y, others="Other five sentence encoders (B\u2013D)"):
    h = [Line2D([], [], color=SETS[s][1], marker=SETS[s][3], lw=0, ms=4.5, label=SETS[s][0]) for s in SETS]
    h.append(Line2D([], [], color="#9A9A9A", marker="o", lw=0, ms=3, label=others))
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, y),
               handletextpad=0.4, columnspacing=1.4)


def main():
    S = fo.sh_below(pd.read_csv(os.path.join(SRC, "below_sets.csv")))   # carries half: 25 sources
    O = pd.read_csv(os.path.join(SRC, "below_outcomes.csv"))
    U = pd.read_csv(os.path.join(SRC, "below_units.csv"))
    core = pd.read_csv(os.path.join(WB, "factor_decile10", "decile10_factors.csv"))
    core = core[(core.factor_model == fo.FM) & (core.rotation == fo.ROT)].set_index("factor")
    items = pd.read_csv(ITEMS_CSV).set_index("item")
    # the panels are placed on a 9.6 in layout whose part below panel C is cut, the positions mapped onto the shorter figure
    H0 = 9.6
    OFF = 0.43 * H0 - 0.45                      # inches removed below panel C (0.45 in kept for its tick labels)
    HF = H0 - OFF
    Y = lambda y: (y * H0 - OFF) / HF           # old figure fraction -> new
    SZ = lambda h: h * H0 / HF
    fig = plt.figure(figsize=(7.0, HF))

    # A: who the factor model finds, top 5%
    ax = fig.add_axes([0.2, Y(0.79), 0.42, SZ(0.125)])
    s5 = S[S.cut == "top 5%"].set_index(["encoder", "set"])
    for y, e in enumerate(ENCODERS[::-1]):
        left = 0
        for st in ("enc_only", "both", "fac_only"):
            w = s5.loc[(e, st), "n_set"]
            ax.barh(y, w, left=left, height=0.62, color=SETS[st][1], edgecolor="white", lw=1.0,
                    hatch="////" if st == "both" else None)
            left += w
        ax.text(left + 10, y, f"{s5.loc[(e, 'both'), 'found_by_factor_pct']:.0f}% found", va="center", fontsize=6.3,
                color="#444444")
    ax.set_yticks(range(len(ENCODERS)))
    ax.set_yticklabels(ENCODERS[::-1], fontsize=6.6)
    ax.set_xlabel("Outliers meeting none of the six thresholds")
    ax.set_xlim(0, 760)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax.set_axisbelow(True)
    fo.title_at(ax, "A  Screen-negative outliers (top 5%)", 0.02)

    # B: questionnaires, decile 10, on the semantic groups' distance
    ax = fig.add_axes([0.76, Y(0.79), 0.225, SZ(0.125)])
    s10 = S[S.cut == CUT10].set_index(["encoder", "set"])
    for x, st in enumerate(("enc_only", "both", "fac_only")):
        _, c, cL, mk, _ = SETS[st]
        v = s10.xs(st, level="set")["one_source_half_enc_pct"]
        oth = v[ENCODERS[1:]].dropna()
        ax.scatter(np.full(len(oth), x + 0.2), oth.values, s=6, marker=mk, color=cL, lw=0, zorder=2)
        if np.isfinite(v[GTE]):
            ax.plot(x, v[GTE], marker=mk, color=c, ms=4.5, lw=0, zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Missed", "Found", "Factor-\nonly"], fontsize=6.6)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(0, 70)
    fo.bands_x(ax, 3)
    ax.set_ylabel("At least half of the\nsquared distance, %", fontsize=6.8)
    ax.grid(axis="y", color="#E6E6E6", lw=0.5)
    fo.title_at(ax, "B  One questionnaire (decile 10)", 0.66)

    # C: outcomes, decile 10
    H = holm(O, CUT10)
    o10 = O[O.cut == CUT10]
    boxes = [[0.085, Y(0.43), 0.30, SZ(0.205)], [0.475, Y(0.43), 0.12, SZ(0.205)], [0.67, Y(0.43), 0.12, SZ(0.205)],
             [0.865, Y(0.43), 0.12, SZ(0.205)]]
    for (title, outs, sc), box in zip(OUTS, boxes):
        ax = fig.add_axes(box)
        tops = []
        for x, (oc, lab) in enumerate(outs):
            r = o10[o10.outcome == oc]
            for st, (_, col, colL, mk, dx) in SETS.items():
                rs = r[r.set == st].set_index("encoder")
                oth = [e for e in ENCODERS[1:] if e in rs.index and np.isfinite(rs.loc[e, "adj"])]
                ax.scatter(np.full(len(oth), x + dx + 0.09), sc * rs.loc[oth, "adj"].values, s=6, marker=mk,
                           color=colL, lw=0, zorder=2)
                v, lo, hi = (sc * rs.loc[GTE, c] for c in ("adj", "adj_lo", "adj_hi"))
                ax.vlines(x + dx, lo, hi, color=col, lw=0.8, zorder=3)
                ax.plot(x + dx, v, marker=mk, ms=4.0, color=col, zorder=4, lw=0)
            top = sc * np.nanmax(r[r.set.isin(list(SETS))].adj_hi.values)
            for lev, (lab_p, a, b) in enumerate(PAIRS):
                c = H[(H.outcome == oc) & (H.set == lab_p)].set_index("encoder")
                sign = np.sign(c.loc[GTE, "adj"])
                n_sig = int(((c.p_holm < 0.05) & (np.sign(c.adj) == sign)).sum())
                tops.append((x + SETS[a][4], x + SETS[b][4], lev, top,
                             f"{fo.p_text(c.loc[GTE, 'p_holm'])}, {n_sig} of {int(c.adj.notna().sum())}"))
        ax.margins(y=0.1)
        ylo, yhi = ax.get_ylim()
        rng = yhi - ylo
        yhi = max(yhi, max(t[3] for t in tops) + 0.55 * rng)
        ax.set_ylim(ylo, yhi)
        rng = yhi - ylo
        for xl, xr, lev, top, txt in tops:
            yb = top + (0.05 + 0.2 * lev) * rng
            ax.plot([xl, xl, xr, xr], [yb - 0.025 * rng, yb, yb, yb - 0.025 * rng], color="#444444", lw=0.6, zorder=5)
            ax.text((xl + xr) / 2, yb + 0.008 * rng, txt, ha="center", va="bottom", fontsize=5.6, color="#444444")
        ax.set_xticks(range(len(outs)))
        ax.set_xticklabels([lab for _, lab in outs])
        ax.set_xlim(-0.6, len(outs) - 0.4)
        if len(outs) == 1:
            ax.set_xticks([])
        elif len(outs) > 1:
            fo.bands_x(ax, len(outs))
        ax.set_title(title, loc="left", fontweight="normal", fontsize=7.5)
        ax.grid(axis="y", color="#E6E6E6", lw=0.5)
    fig.text(0.02, Y(0.712), "C  Outcomes in decile 10, adjusted for age and sex", fontsize=8)
    fig.text(0.02, Y(0.705), "Brackets: missed vs found (lower) and missed vs factor-only (upper).  $p$: "
             "gte-large-en-v1.5, Holm-adjusted over its contrasts.\nn of N: encoders with Holm-adjusted $p$ < 0.05 in "
             "the same direction.", fontsize=6.0, color="#444444", va="top", linespacing=1.3)

    legend(fig, Y(0.998), others="Other five sentence encoders (B, C)")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"below_outliers_factor.{ext}"), dpi=300)
    plt.close(fig)
    print(f"-> {os.path.join(FIGURES, 'below_outliers_factor.png')}")

    # figure supplement 2: factors, decile 10, all six encoders
    u = U[(U.cut == CUT10) & (U.grouping == "factor") & (U.rotation == fo.ROT)]
    med, w = unit_rows(u, keep_min=D_MIN)
    h_in = 0.26 * len(med)
    fig = plt.figure(figsize=(7.0, 0.55 + h_in + 0.85))
    ax = fig.add_axes([0.5, 0.55 / fig.get_figheight(), 0.47, h_in / fig.get_figheight()])
    unit_dots(ax, med, w, lambda r: fo.wrap(fo.profile_label("F", r.unit, r.name)
                                            + fo.stress_wording(core, items, r.unit), 52), x_fig=0.02)
    ax.set_xlabel("Participants with the factor as a top-3 contributor, %")
    fo.title_at(ax, "Factors contributing most to the distance of screen-negative participants under the factor model "
                "(25 factors) in decile 10", 0.02)
    legend(fig, 1 - 0.02 / fig.get_figheight(), others="Other five sentence encoders")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"below_outliers_factor_factors.{ext}"), dpi=300)
    plt.close(fig)
    print(f"-> {os.path.join(FIGURES, 'below_outliers_factor_factors.png')}")
    print(med[["unit", "name", "enc_only", "both", "fac_only", "key"]].round(1).to_string(index=False))

    # figure supplement 1: the prespecified encoder's semantic groups, decile 10
    u = U[(U.cut == CUT10) & (U.grouping == "semantic groups") & (U.encoder == GTE)]
    med, w = unit_rows(u)
    fig = plt.figure(figsize=(7.0, 0.3 + 0.26 * len(med) + 0.9))
    h_in = 0.26 * len(med)
    ax = fig.add_axes([0.5, 0.55 / fig.get_figheight(), 0.47, h_in / fig.get_figheight()])
    unit_dots(ax, med, w, lambda r: fo.wrap(fo.profile_label("G", r.unit, r.name), 56), x_fig=0.02)
    ax.set_xlabel("Participants with the group as a top-3 contributor, %")
    fo.title_at(ax, "Semantic groups (gte-large-en-v1.5) contributing most to its distance, screen-negative participants in "
                "decile 10", 0.02)
    h = [Line2D([], [], color=SETS[s][1], marker=SETS[s][3], lw=0, ms=4.5, label=SETS[s][0]) for s in SETS]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, 1 - 0.02 / fig.get_figheight()), handletextpad=0.3, columnspacing=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"below_outliers_factor_groups.{ext}"), dpi=300)
    plt.close(fig)
    print(f"-> {os.path.join(FIGURES, 'below_outliers_factor_groups.png')}")


if __name__ == "__main__":
    main()
