#!/usr/bin/env python
"""figure4.py -- Figure 4 and its figure supplements 1 and 2: the outliers found only by the semantic groups against those
found only by the factor model with 25 factors.

Outliers = each grouping's own 5% with the largest distances. The 3 sets, per encoder: outliers under both the encoder
and the factor model, under the encoder only (semantic groups only), under the factor model only.
  Figure 4                A  the 3 sets under each encoder; B  the outliers with at least half of their squared distance
                          from one questionnaire (each stand-alone question counted under the study it was taken from),
                          on the distance that flagged them and on the same distance; C-I  every outcome of the paper
                          adjusted for age and sex in each set, with the prespecified encoder's Holm-adjusted p for the
                          semantic-only minus factor-only contrast (over its 14 outcomes) and the number of encoders with a
                          Holm-adjusted p < 0.05 in the same direction
  figure supplement 1     the factors among the 3 largest contributors, semantic-only vs factor-only outliers
  figure supplement 2     the prespecified encoder's semantic groups among the 3 largest contributors
Semantic groups only black circles, factor model only green squares, both orange diamonds (Okabe-Ito); the prespecified
encoder large with 95% CIs, the other 5 encoders small light markers. Alternate light gray bands behind every second
category or row. Factors are named by the constructs of their core items, the 2 perceived stress factors also by their
core items' wording.

Figure 6 and the figure source data reuse this script's helpers.

Reads   the downloads factor_outliers (outliers_sets.csv, outliers_outcomes.csv, outliers_units.csv), factor_outliers_ed
        (outliers_ed.csv), factor_sources (sources_outlier_sets.csv), factor_decile10 (decile10_factors.csv), and
        data/items_151.csv
Writes  elife/figures/outliers_factor.{png,pdf}, outliers_factor_factors.{png,pdf}, outliers_factor_groups.{png,pdf}
Run     python figure4.py
"""
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D    # noqa: E402
from matplotlib.patches import Patch, Rectangle   # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
from statsmodels.stats.multitest import multipletests   # noqa: E402

import common as cl                     # noqa: E402
from common import FIGURES, ITEMS_CSV, WB   # noqa: E402

SRC = os.path.join(WB, "factor_outliers")
SRC_ED = os.path.join(WB, "factor_outliers_ed")         # the emergency department share, same sets and model
GTE = "gte-large-en-v1.5"
ENCODERS = [GTE, "bge-m3", "bge-large-en-v1.5", "all-mpnet-base-v2", "e5-large-v2", "qwen3-embedding-0.6b"]
FM, ROT = "factor", "oblimin"
Z = 1.959963984540054
# set: (legend label, color, light color for the other 5 encoders, marker, x offset)
SETS = {"enc_only": ("Semantic groups only", "#000000", "#9A9A9A", "o", -0.22),
        "both": ("Both", "#E69F00", "#F3CF80", "D", 0.0),
        "fac_only": ("Factor model only (25 factors)", "#009E73", "#8FD3BE", "s", 0.22)}
SEM, FAC = SETS["enc_only"][1], SETS["fac_only"][1]
BAND = "#F2F2F2"                         # alternate category bands

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6})


# ------------------------------------------------------------------------------------------------- helpers
def profile_label(prefix, unit, name):
    """<prefix><n>  <construct> <count>, ... in the paper's display names."""
    parts = [p.rsplit(":", 1) for p in name.split("; ")]
    return f"{prefix}{unit}  " + ", ".join(f"{cl.display(c)} {n}" for c, n in parts)


def wrap(label, width=62):
    """Break a long name after a comma so it fits beside its row."""
    if len(label) <= width:
        return label
    cut = label.rfind(", ", 0, width)
    return label if cut < 0 else label[:cut + 1] + "\n      " + label[cut + 2:]


def title_at(ax, text, x_fig):
    """A panel title starting at figure x x_fig (the left edge of the panel's tick labels)."""
    pos = ax.get_position()
    ax.set_title(text, loc="left", fontweight="normal", x=(x_fig - pos.x0) / pos.width)


def set_legend(fig, y, extra=True, sets=("enc_only", "both", "fac_only"), ci=True):
    h = [Line2D([], [], color=SETS[s][1], marker=SETS[s][3], lw=0, ms=4.5,
                label=SETS[s][0] + ((" (gte-large-en-v1.5, 95% CI)" if ci else " (gte-large-en-v1.5)")
                                    if s == "enc_only" else ""))
         for s in sets]
    if extra:
        h.append(Line2D([], [], color="#9A9A9A", marker="o", lw=0, ms=3, label="Other five sentence encoders"))
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, y),
               handletextpad=0.3, columnspacing=1.2)


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def stress_wording(core, items, unit):
    ci = core.loc[unit, "core_items"].split()
    if not all(i.startswith("sdoh_cpss") for i in ci):
        return ""
    rev = items.loc[ci, "reverse_coded"].astype(str).str.lower().eq("true")
    return " (positively worded)" if rev.all() else " (negatively worded)" if (~rev).all() else ""


def bands_x(ax, n):
    """Light gray bands behind every second category on the x axis (category i spans i - 0.5 to i + 0.5)."""
    for i in range(1, n, 2):
        ax.axvspan(i - 0.5, i + 0.5, color=BAND, lw=0, zorder=0)
    ax.set_axisbelow(True)


def bands_y(ax, centers, x_fig=None):
    """Light gray bands behind the rows at the given y centers, from figure x x_fig (under the tick labels) or from
    the axis's left edge to its right edge."""
    pos = ax.get_position()
    x0 = 0.0 if x_fig is None else (x_fig - pos.x0) / pos.width
    for c in centers:
        ax.add_patch(Rectangle((x0, c - 0.5), 1 - x0, 1.0, transform=ax.get_yaxis_transform(), color=BAND, lw=0,
                               zorder=0, clip_on=False))
    ax.set_axisbelow(True)


def dumbbell(ax, rows, labels, fs=6.4, x_fig=None):
    """rows: list of (semantic-only %, factor-only %), drawn top to bottom."""
    n = len(rows)
    for y, (a, b) in enumerate(rows):
        yy = n - 1 - y
        if np.isfinite(a) and np.isfinite(b):
            ax.hlines(yy, a, b, color="#C8C8C8", lw=1.0, zorder=1)
        if np.isfinite(b):
            ax.plot(b, yy, "s", color=FAC, ms=4.6, mec="white", mew=0.5, zorder=3)
        if np.isfinite(a):
            ax.plot(a, yy, "o", color=SEM, ms=4.6, mec="white", mew=0.5, zorder=4)
        if np.isfinite(a) != np.isfinite(b):
            x0 = a if np.isfinite(a) else b
            ax.text(x0 + 0.8, yy, "other set withheld (small count)", fontsize=5.8, va="center", color="#666666")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1], fontsize=fs)
    ax.set_ylim(-0.5, n - 0.5)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    bands_y(ax, [n - 1 - k for k in range(1, n, 2)], x_fig)


SRC_SH = os.path.join(WB, "factor_sources")             # one questionnaire carries half, 25 sources


def sh_sets(S):
    """The set rows with the carries-half columns from factor_sources (stand-alone questions counted by their source
    study). A cell factor_sources did not write (withheld, or not computed: the both set on the factor model's distance)
    is blank."""
    N = pd.read_csv(os.path.join(SRC_SH, "sources_outlier_sets.csv")).set_index(["encoder", "set", "distance"])
    S = S.copy()
    for col, dist in (("one_source_half_enc_pct", "encoder"), ("one_source_half_fac_pct", "factor")):
        S[col] = [N.one_source_half_pct.get((e, st, dist), np.nan) if fm == FM else np.nan
                  for e, st, fm in zip(S.encoder, S.set, S.factor_model)]
    return S


def sh_groupings(G):
    """The grouping rows with one_source_half_pct from factor_sources (blank where it is withheld)."""
    N = pd.read_csv(os.path.join(SRC_SH, "sources_groupings.csv")).set_index(["grouping", "set"])
    G = G.copy()
    G["one_source_half_pct"] = [N.one_source_half_pct.get((m, st), np.nan) for m, st in zip(G.model, G.set)]
    return G


def sh_below(S):
    """The below-threshold set rows with one_source_half_enc_pct from factor_sources; the factor-distance column is
    not computed there and is blanked."""
    N = pd.read_csv(os.path.join(SRC_SH, "sources_below_sets.csv")).set_index(["cut", "encoder", "set"])
    S = S.copy()
    S["one_source_half_enc_pct"] = [N.one_source_half_pct.get((c, e, st), np.nan)
                                    for c, e, st in zip(S.cut, S.encoder, S.set)]
    S["one_source_half_fac_pct"] = np.nan
    return S


def load_outcomes():
    """The outcome rows with the emergency department rows of factor_outliers_ed joined (same sets, model, columns)."""
    O = pd.read_csv(os.path.join(SRC, "outliers_outcomes.csv"))
    E = pd.read_csv(os.path.join(SRC_ED, "outliers_ed.csv"))
    assert set(E.columns) == set(O.columns) and not set(E.outcome) & set(O.outcome)
    return pd.concat([O, E[O.columns]], ignore_index=True)


def holm(O):
    """The semantic-only minus factor-only contrasts with Holm-adjusted p: per encoder, over every outcome whose
    contrast is written (14, or 13 where breast cancer is withheld), as the paper adjusts its decile trends."""
    c = O[O.set == "enc_only - fac_only"].copy()
    c["p_holm"] = np.nan
    for e, g in c.groupby("encoder"):
        ok = g.p.notna()
        c.loc[g.index[ok], "p_holm"] = multipletests(g.p[ok], method="holm")[1]
    return c


def p_text(p):
    """The paper's format: p < 0.001; 3 decimals below 0.01; else 2 decimals; p > 0.99 at the ceiling."""
    if p < 0.001:
        return "$p$ < 0.001"
    if p > 0.99:
        return "$p$ > 0.99"
    return f"$p$ = {p:.3f}" if p < 0.01 else f"$p$ = {p:.2f}"


NAMES = {"main": "outliers_factor", "supp_factors": "outliers_factor_factors", "supp_groups": "outliers_factor_groups"}


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"{NAMES[name]}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"-> {os.path.join(FIGURES, NAMES[name] + '.png')}")


# ------------------------------------------------------------------------------------------------- panels
# The panels are drawn into given axes, so the main figure and the supplements share one drawing code.
def inch(fig, left, bottom, width, height):
    """An axes placed in inches from the figure's lower left corner."""
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def draw_overlap(ax, S):
    """Stacked bars per encoder: semantic-only, both, factor-only outliers."""
    s = S[S.factor_model == FM].set_index(["encoder", "set"])["n_set"]
    for y, e in enumerate(ENCODERS[::-1]):
        left = 0
        for st in ("enc_only", "both", "fac_only"):
            w = s[(e, st)]
            ax.barh(y, w, left=left, height=0.62, color=SETS[st][1], edgecolor="white", lw=1.0,
                    hatch="////" if st == "both" else None)
            left += w
    ax.set_yticks(range(len(ENCODERS)))
    ax.set_yticklabels(ENCODERS[::-1])
    ax.set_xlabel("Participants")
    ax.set_xlim(0, 6600)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax.set_axisbelow(True)


def draw_half_own(ax, S):
    """% with 1 questionnaire carrying half, each set on the distance that flagged it."""
    s = S[S.factor_model == FM]
    own = [("enc_only", "one_source_half_enc_pct", "Semantic groups only,\non the semantic groups' distance"),
           ("fac_only", "one_source_half_fac_pct", "Factor model only,\non the factor model's distance")]
    for y, (st, col, lab) in enumerate(own[::-1]):
        v = s[s.set == st].set_index("encoder")[col]
        oth = v[ENCODERS[1:]].dropna()
        ax.barh(y, v[GTE], height=0.55, color=SETS[st][1], edgecolor="white", zorder=2)
        ax.hlines(y, oth.min(), oth.max(), color="#666666", lw=1.0, zorder=3)
        ax.vlines([oth.min(), oth.max()], y - 0.12, y + 0.12, color="#666666", lw=1.0, zorder=3)
        ax.text(v[GTE] + 1.0 if v[GTE] < 20 else v[GTE] - 1.0, y + 0.36, f"{v[GTE]:.1f}%", fontsize=6.5,
                ha="left" if v[GTE] < 20 else "right", va="bottom", color="#222222")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([o[2] for o in own[::-1]], fontsize=6.6)
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlim(0, 36)
    ax.set_xlabel("Outliers, %")
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax.set_axisbelow(True)


def draw_half_same(ax, S):
    """The same measure with both sets on the same distance."""
    s = S[S.factor_model == FM]
    rows = [("one_source_half_fac_pct", "Factor model's\ndistance"), ("one_source_half_enc_pct", "Semantic groups'\ndistance")]
    for y, (col, lab) in enumerate(rows):
        for st, dy in (("enc_only", -0.14), ("fac_only", 0.14)):
            _, c, cL, mk, _ = SETS[st]
            v = s[s.set == st].set_index("encoder")[col]
            oth = v[ENCODERS[1:]].dropna()
            ax.scatter(oth.values, np.full(len(oth), y + dy), s=6, marker=mk, color=cL, lw=0, zorder=2)
            ax.plot(v[GTE], y + dy, marker=mk, color=c, ms=4.2, lw=0, zorder=3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[1] for r in rows], fontsize=6.6)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(0, 36)
    bands_y(ax, [1])
    ax.set_xlabel("Outliers, %")
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)


FAMILIES = [("Screening thresholds\nmet, mean No.", [("thresholds_met", "")], 1),
            ("Barriers to care, %", [("access", "Any"), ("cost", "Cost"), ("avoid", "Clinician race,\nreligion, or language")], 100),
            ("ED share, %", [("ed_share_coded", "")], 100),
            ("Sleep per night,\nmin", [("sleep_mean_min", "")], 1),
            ("Sleep\nvariation, min", [("sleep_variation_min", "")], 1),
            ("Mental and behavioral conditions, %", [("gad", "GAD"), ("adhd", "ADHD"), ("ptsd", "PTSD"),
                                             ("panic", "Panic\ndisorder")], 100),
            ("Physical conditions, %", [("hyperchol", "Hyper-\ncholesterolemia"), ("asthma", "Asthma"),
                                        ("breast", "Breast\ncancer"), ("prostate", "Prostate\ncancer")], 100)]


def draw_outcomes(axes, O, letters):
    """The 7 outcome families, 1 axes each, with the semantic-only vs factor-only bracket, the prespecified encoder's
    Holm-adjusted p and the number of encoders with Holm-adjusted p < 0.05 in its direction."""
    O = O[O.factor_model == FM]
    H = holm(O)
    for (title, outs, sc), ax, tag in zip(FAMILIES, axes, letters):
        tops = []
        for x, (oc, lab) in enumerate(outs):
            r = O[O.outcome == oc]
            if not np.isfinite(r[(r.encoder == GTE) & (r.set == "enc_only")].adj).any():
                ax.text(x, 0.5, "withheld\n(small\ncounts)", ha="center", va="center", fontsize=6, color="#666666",
                        transform=ax.get_xaxis_transform())
                continue
            for st, (_, col, colL, mk, dx) in SETS.items():
                rs = r[r.set == st].set_index("encoder")
                others = [e for e in ENCODERS[1:] if e in rs.index and np.isfinite(rs.loc[e, "adj"])]
                ax.scatter(np.full(len(others), x + dx + 0.09), sc * rs.loc[others, "adj"].values, s=6, marker=mk,
                           color=colL, lw=0, zorder=2)
                v, lo, hi = (sc * rs.loc[GTE, c] for c in ("adj", "adj_lo", "adj_hi"))
                ax.vlines(x + dx, lo, hi, color=col, lw=0.8, zorder=3)
                ax.plot(x + dx, v, marker=mk, ms=4.0, color=col, zorder=4, lw=0)
            c = H[H.outcome == oc].set_index("encoder")
            sign = np.sign(c.loc[GTE, "adj"])
            n_sig = int(((c.p_holm < 0.05) & (np.sign(c.adj) == sign)).sum())
            n_ok = int(c.adj.notna().sum())
            tops.append((x, f"{p_text(c.loc[GTE, 'p_holm'])}\n{n_sig} of {n_ok}",
                         sc * np.nanmax(r[r.set.isin(list(SETS))].adj_hi.values)))
        ax.margins(y=0.1)
        ylo, yhi = ax.get_ylim()
        if tops:                                # headroom for the bracket and the 2-line label
            yhi = max(yhi, max(y for _, _, y in tops) + 0.36 * (yhi - ylo))
            ax.set_ylim(ylo, yhi)
        for x, t, y in tops:                    # bracket: semantic groups only <-> factor model only, the tested pair
            yb, tick = y + 0.04 * (yhi - ylo), 0.025 * (yhi - ylo)
            xl, xr = x + SETS["enc_only"][4], x + SETS["fac_only"][4]
            ax.plot([xl, xl, xr, xr], [yb - tick, yb, yb, yb - tick], color="#444444", lw=0.6, zorder=5)
            ax.text(x, yb + 0.01 * (yhi - ylo), t, ha="center", va="bottom", fontsize=6.0, color="#444444",
                    linespacing=1.1)
        ax.set_xticks(range(len(outs)))
        ax.set_xticklabels([lab for _, lab in outs])
        ax.set_xlim(-0.6, len(outs) - 0.4)
        if len(outs) > 1:
            bands_x(ax, len(outs))
        if len(outs) == 1:
            ax.set_xticks([])
        ax.set_title(f"{tag}  {title}", loc="left", fontweight="normal")
        ax.grid(axis="y", color="#E6E6E6", lw=0.5)


def factor_rows(U, core, items):
    u = U[(U.factor_model == FM) & (U.grouping == FM) & (U.rotation == ROT)]
    w = u.pivot_table(index=["unit", "name", "encoder"], columns="set", values="pct_top3").reset_index()
    w["d"] = w["fac_only"] - w["enc_only"]
    cons = w.groupby("unit")["d"].agg(lambda x: bool(x.notna().all() and np.sign(x).nunique() == 1))
    g = w[w.encoder == GTE].copy()
    g["key"] = g["d"].fillna(-99)
    g = g.sort_values("key", ascending=False)
    labels = [profile_label("F", r.unit, r.name) + stress_wording(core, items, r.unit) + (" *" if cons.loc[r.unit] else "")
              for r in g.itertuples()]
    return list(zip(g.enc_only, g.fac_only)), labels


def group_rows(U):
    u = U[(U.factor_model == FM) & (U.grouping == "semantic groups") & (U.encoder == GTE)]
    g = u.pivot_table(index=["unit", "name"], columns="set", values="pct_top3").reset_index()
    g["key"] = (g["fac_only"] - g["enc_only"]).fillna(-99)
    g = g.sort_values("key", ascending=False)
    return list(zip(g.enc_only, g.fac_only)), [profile_label("G", r.unit, r.name) for r in g.itertuples()]


# ------------------------------------------------------------------------------------------------- figures
def fig_main(S, O):
    """Figure 4: A overlap, B one questionnaire carries half (own distance | same distance), C-I outcomes.
    Counts one to nine in words and 4-digit numbers without a separator, as the eLife text writes them."""
    H_ = 8.7
    fig = plt.figure(figsize=(7.0, H_))
    set_legend(fig, 0.998)
    # A
    fig.text(0.02, 8.28 / H_, "A  Outliers under each sentence encoder and under the factor model (25 factors), 4518 under "
             "each grouping", fontsize=8, ha="left", va="bottom")
    ax = inch(fig, 1.40, 7.16, 5.40, 0.98)
    draw_overlap(ax, S)
    # B
    fig.text(0.02, 6.60 / H_, "B  Outliers with at least half of their squared distance from one questionnaire", fontsize=8, ha="left",
             va="bottom")
    fig.text(0.02, 6.40 / H_, "On the distance that flagged them", fontsize=6.6, ha="left", va="bottom")
    fig.text(4.62 / 7.0, 6.40 / H_, "On the same distance", fontsize=6.6, ha="left", va="bottom")
    ax = inch(fig, 2.10, 5.76, 1.80, 0.56)
    draw_half_own(ax, S)
    ax = inch(fig, 5.60, 5.76, 1.22, 0.56)
    draw_half_same(ax, S)
    # C-I
    fig.text(0.5, 5.18 / H_, "C–I  Outcomes, adjusted for age and sex. Brackets join the tested pair, semantic groups "
             "only vs factor model only.\n$p$: gte-large-en-v1.5, Holm-adjusted over "
             "its 14 outcomes.  n of N: encoders with Holm-adjusted $p$ < 0.05 in the same direction.", ha="center",
             va="top", fontsize=6.5, color="#444444", linespacing=1.3)
    W = 7.0
    top = [(0.07 * W, 0.095 * W), (0.235 * W, 0.26 * W), (0.56 * W, 0.095 * W), (0.72 * W, 0.095 * W), (0.88 * W, 0.095 * W)]
    bot = [(0.075 * W, 0.40 * W), (0.575 * W, 0.40 * W)]
    axes = [inch(fig, l, 3.02, w, 1.45) for l, w in top] + [inch(fig, l, 0.55, w, 1.62) for l, w in bot]
    draw_outcomes(axes, O, list("CDEFGHI"))
    save(fig, "main")


def fig_supp_factors(U, core, items):
    """Figure supplement: the factors among the 3 largest contributors, semantic-only vs factor-only (factor distance)."""
    fig = plt.figure(figsize=(7.0, 4.3))
    set_legend(fig, 0.995, sets=("enc_only", "fac_only"), ci=False, extra=False)
    ax = inch(fig, 4.58, 0.62, 2.24, 3.25)
    rows, labels = factor_rows(U, core, items)
    dumbbell(ax, rows, labels, fs=6.0, x_fig=0.03)
    ax.set_xlim(0, 30)
    ax.set_xlabel("Outliers with the factor as a\ntop-3 contributor, %")
    save(fig, "supp_factors")


def fig_supp_groups(U):
    """Figure supplement: the prespecified encoder's semantic groups among the 3 largest contributors (its distance)."""
    fig = plt.figure(figsize=(7.0, 4.6))
    set_legend(fig, 0.995, sets=("enc_only", "fac_only"), ci=False, extra=False)
    ax = inch(fig, 4.90, 0.62, 1.92, 3.55)
    rows, labels = group_rows(U)
    dumbbell(ax, rows, labels, fs=6.0, x_fig=0.03)
    ax.set_xlim(0, 36)
    ax.set_xlabel("Outliers with the group as a\ntop-3 contributor, %")
    save(fig, "supp_groups")


def main():
    S = sh_sets(pd.read_csv(os.path.join(SRC, "outliers_sets.csv")))     # carries half: 25 sources
    O = load_outcomes()
    U = pd.read_csv(os.path.join(SRC, "outliers_units.csv"))
    core = pd.read_csv(os.path.join(WB, "factor_decile10", "decile10_factors.csv"))
    core = core[(core.factor_model == FM) & (core.rotation == ROT)].set_index("factor")
    items = pd.read_csv(ITEMS_CSV).set_index("item")
    fig_main(S, O)
    fig_supp_factors(U, core, items)
    fig_supp_groups(U)


if __name__ == "__main__":
    main()
