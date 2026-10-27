#!/usr/bin/env python
"""figure1.py -- Figure 1: the method, in 3 panels.

  A  how the groups are formed (schematic): item text -> sentence encoder -> each item a point in semantic space ->
     PCA and the sparsity-penalized projection -> the groups, 3 drawn as directions with their core items at the far end.
     General: the panel title gives no number of groups (that number belongs to one encoder; panel B is the
     prespecified encoder's).
     The 3 items are shown by topic, not wording (the owners' terms of 2 of the questionnaires do not cover reprinting).
  B  what a semantic group is (data): a ribbon from each questionnaire, with its construct, to the group of the prespecified
     encoder whose core the item belongs to, for 5 groups that show questionnaires cut and joined. Items are assigned by
     core membership (outputs/basis/components.csv), never by the largest weight; no item is in 2 of the 5 cores.
  C  how participants are scored (schematic): answers -> scores on every group -> one distance each from the cohort's
     mean scores, with the top 5% marked.

Drawn at its printed size, the 2-column width of 180 mm, with no text under 8 pt and no line under 2 pt (Frontiers
figure guidelines). Separators that would be lines (white edges around dots, bars and cells) are left out.

Reads   data/items_151.csv, outputs/basis/components.csv
Writes  output/figures/method.{pdf,png}
Run     python figure1.py
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path                                   # noqa: E402

ENC = "gte-large-en-v1.5"
FEATURED = [(12, "sdoh_ucla_ls8_14"), (3, "ace_5"), (16, "sdoh_dms_2")]     # (group, 1 of its core items)
# each featured item is shown by its topic, not its wording: the ULS-8 and DMS owners' terms do not cover reprinting
TOPIC = {"sdoh_ucla_ls8_14": "Feeling isolated", "ace_5": "Parents separated or divorced",
         "sdoh_dms_2": "Less respect at health care visits"}
RIBBON = [12, 3, 6, 16, 2]   # no item in 2 of these cores
INK, RULE, STAGE, GREY, FLAG = "#222222", "#8C8C8C", "#16325C", "#C8C8C8", "#D55E00"
EDGE = "#ABABAB"   # box outlines, lighter than the arrows
GCOL = {12: "#0072B2", 3: "#E69F00", 16: "#009E73", 22: "#009E73"}          # Okabe-Ito; others neutral
NEUTRAL = "#9AA9B8"
FS, FS_HEAD, FS_TAG = 8, 9, 10      # pt: text, stage titles, panel letters
LW = 2.0                            # pt: every line
W_IN = 180 / 25.4                   # the 2-column width

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS})
from common import ITEMS_CSV, OUTPUTS, display, save   # noqa: E402

spell = display


def load():
    it = pd.read_csv(ITEMS_CSV).set_index("item")
    it["abbr"] = it["instrument"].str.extract(r"\(([^()]*)\)\s*$")[0]
    comp = pd.read_csv(os.path.join(OUTPUTS, "basis", "components.csv"))
    comp = comp[comp.encoder == ENC].set_index("component")
    for g, item in FEATURED:
        assert item in comp.at[g, "core_items"].split(), f"{item} is not a core item of group {g}"
    return it, comp


def lead(comp, g):
    return spell(comp.at[g, "construct_profile"].split(";")[0].split(":")[0])


def chip(ax, x, y, w, h, label, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015", fc=fc, ec=EDGE,
                                lw=LW, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=FS, color=INK, zorder=4,
            linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, label=None, dy=0.05, color=RULE, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=LW, color=color,
                                                                     mutation_scale=14, linestyle=ls), zorder=2)
    if label:
        ax.text((x0 + x1) / 2, max(y0, y1) + dy, label, ha="center", va="bottom", fontsize=FS, style="italic",
                color=INK, linespacing=1.1)


def ray_fn(ax):
    bb = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    w_in, h_in = fw * bb.width, fh * bb.height

    def ray(x0, y0, deg, length):
        t = np.deg2rad(deg)
        return x0 + length * np.cos(t) / w_in, y0 + length * np.sin(t) / h_in
    return ray


def panel_a(ax, it, comp):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.115, 0.95, "Items", ha="center", va="bottom", fontsize=FS_HEAD, fontweight="bold", color=STAGE)
    for k, (g, item) in enumerate(FEATURED):
        y = 0.72 - 0.23 * k
        txt = "\n".join(textwrap.wrap(TOPIC[item], 20))
        chip(ax, 0.005, y, 0.22, 0.19, txt)
        ax.text(0.236, y + 0.095, it.at[item, "abbr"], ha="left", va="center", fontsize=FS, fontweight="bold",
                color=GCOL[g])
    # terms as in the text
    ax.text(0.115, 0.0, "151 items from\n17 questionnaires and\n13 stand-alone questions", ha="center", va="bottom",
            fontsize=FS, color=INK, linespacing=1.15)

    arrow(ax, 0.300, 0.52, 0.360, 0.52, "sentence\nencoder")
    ax.text(0.450, 0.95, "Semantic space", ha="center", va="bottom", fontsize=FS_HEAD, fontweight="bold", color=STAGE)
    rng = np.random.RandomState(3)
    ax.scatter(0.375 + 0.15 * rng.rand(90), 0.26 + 0.62 * rng.rand(90), s=5, color=GREY, zorder=2, lw=0)
    for (g, _), (x, y) in zip(FEATURED, [(0.402, 0.78), (0.500, 0.33), (0.480, 0.70)]):
        ax.scatter(x, y, s=30, color=GCOL[g], zorder=4, lw=0)
    ax.text(0.450, 0.0, "each item is a point,\nplaced by what it asks", ha="center", va="bottom", fontsize=FS,
            color=INK, linespacing=1.15)

    # "PCA" above the arrow and the Methods' term below it
    arrow(ax, 0.545, 0.52, 0.615, 0.52, "PCA")
    ax.text(0.580, 0.52 - 0.05, "sparsity-\npenalized\nprojection", ha="center", va="top", fontsize=FS, style="italic",
            color=INK, linespacing=1.1)
    ax.text(0.80, 0.95, "Semantic groups", ha="center", va="bottom", fontsize=FS_HEAD, fontweight="bold",
            color=STAGE)
    ax.add_patch(FancyBboxPatch((0.635, 0.20), 0.335, 0.71, boxstyle="round,pad=0.004,rounding_size=0.015",
                                fc="#FAFAFA", ec=EDGE, lw=LW, zorder=0))
    ray = ray_fn(ax)
    ox, oy, length = 0.80, 0.52, 0.55
    for (g, _), deg, ha, va in zip(FEATURED, (90, 210, 330), ("center", "center", "center"), ("bottom", "top", "top")):
        xt, yt = ray(ox, oy, deg, length)
        ax.annotate("", xy=(xt, yt), xytext=(ox, oy), arrowprops=dict(arrowstyle="-|>", lw=LW, color=RULE,
                                                                      mutation_scale=12), zorder=2)
        for f in (0.12, 0.22, 0.32, 0.42):
            ax.scatter(*ray(ox, oy, deg, length * f), s=10, color=GREY, zorder=3, lw=0)
        for f in (0.54, 0.66, 0.78):
            ax.scatter(*ray(ox, oy, deg, length * f), s=22, color=GCOL[g], zorder=4, lw=0)
        lx, ly = ray(ox, oy, deg, length + (0.07 if deg == 90 else 0.10))
        name = lead(comp, g) if deg == 90 else "\n".join(textwrap.wrap(lead(comp, g), 14))
        ax.text(lx, ly, name, ha=ha, va=va, fontsize=FS, fontweight="bold", color=GCOL[g], linespacing=1.1)
    ax.scatter([ox], [oy], s=16, color=INK, zorder=5, lw=0)
    ax.annotate("its core\nitems", xy=ray(ox, oy, 90, length * 0.66), xytext=(0.860, 0.66), fontsize=FS,
                color=INK, ha="left", va="center", linespacing=1.1,
                arrowprops=dict(arrowstyle="-", lw=LW, color=RULE, shrinkA=3, shrinkB=5))
    ax.text(0.80, 0.0, "fixed from the item text\nbefore any answers are used", ha="center", va="bottom",
            fontsize=FS, style="italic", color=INK, linespacing=1.15)


def band(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha):
    cx = (x0 + x1) / 2.0
    verts = [(x0, y0a), (cx, y0a), (cx, y1a), (x1, y1a), (x1, y1b), (cx, y1b), (cx, y0b), (x0, y0b), (x0, y0a)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=1))


def panel_b(ax, it, comp):
    rows = [{"group": g, "item": i, "instr": it.at[i, "abbr"], "construct": spell(it.at[i, "construct"])}
            for g in RIBBON for i in comp.at[g, "core_items"].split()]
    D = pd.DataFrame(rows)
    assert D.instr.notna().all(), "an item without an instrument abbreviation"
    assert not D.item.duplicated().any(), "an item sits in 2 of the ribbon's cores"
    assert pd.Index([comp.at[g, "construct_profile"] for g in RIBBON]).is_unique, "2 ribbon labels repeat"
    # left order: first appearance down the group order, so bands cross as little as the data allow
    left = list(dict.fromkeys(D.instr))
    lsz = D.instr.value_counts().to_dict()
    gsz = D.group.value_counts().to_dict()
    gap = 1.1
    H = sum(gsz.values()) + gap * (len(RIBBON) - 1)

    def layout(order, sizes):
        tot = sum(sizes[o] for o in order)
        g = (H - tot) / max(len(order) - 1, 1)
        pos, y = {}, H
        for o in order:
            pos[o] = (y - sizes[o], y)
            y -= sizes[o] + g
        return pos
    L, R = layout(left, lsz), layout(RIBBON, gsz)
    xl, xr, bw = 0.0, 1.0, 0.014
    lcur = {k: v[1] for k, v in L.items()}
    rcur = {k: v[1] for k, v in R.items()}
    for g in RIBBON:
        for ins in left:
            n = int(((D.group == g) & (D.instr == ins)).sum())
            if not n:
                continue
            band(ax, xl + bw, lcur[ins], lcur[ins] - n, xr - bw, rcur[g], rcur[g] - n, GCOL.get(g, NEUTRAL), 0.55)
            lcur[ins] -= n
            rcur[g] -= n
    cons = D.drop_duplicates("instr").set_index("instr").construct
    for ins in left:
        b, t = L[ins]
        ax.add_patch(Rectangle((xl - bw, b), 2 * bw, t - b, facecolor=INK, edgecolor="none", lw=0, zorder=5))
        ax.text(xl - 0.04, (b + t) / 2 + 0.25, ins, ha="right", va="bottom", fontsize=FS, fontweight="bold", color=INK)
        ax.text(xl - 0.04, (b + t) / 2 - 0.15, cons[ins], ha="right", va="top", fontsize=FS, color="#555555")
    for g in RIBBON:
        b, t = R[g]
        col = GCOL.get(g, "#5F6F7F")
        ax.add_patch(Rectangle((xr - bw, b), 2 * bw, t - b, facecolor=col, edgecolor="none", lw=0, zorder=5))
        names = [spell(p.split(":")[0]) for p in comp.at[g, "construct_profile"].split("; ")]
        label = "\n".join(textwrap.wrap(f"G{g + 1}  " + ", ".join(names), 38, subsequent_indent="       "))
        ax.text(xr + 0.04, (b + t) / 2, label, ha="left", va="center", fontsize=FS, color=col, linespacing=1.1,
                fontweight="bold" if g in GCOL else "normal")
    # terms as in the text
    ax.text(xl - 0.04, H + 1.2, "Questionnaire\n(construct)", ha="right", va="bottom", fontsize=FS_HEAD,
            fontweight="bold", color=STAGE, linespacing=1.1)
    ax.text(xr + 0.04, H + 1.2, "Semantic group\n(constructs of its core items)", ha="left", va="bottom",
            fontsize=FS_HEAD, fontweight="bold", color=STAGE, linespacing=1.1)
    ax.set_xlim(-1.02, 2.25); ax.set_ylim(-0.6, H + 5.4); ax.axis("off")
    return D


def panel_c(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    rng = np.random.RandomState(11)
    top = 0.60
    ax.text(0.095, 0.95, "Answers", ha="center", va="bottom", fontsize=FS_HEAD, fontweight="bold", color=STAGE)
    for r in range(5):
        for c in range(8):
            ax.add_patch(Rectangle((0.030 + 0.0165 * c, top + 0.14 - 0.075 * r), 0.0148, 0.066,
                                   fc=plt.cm.Blues(0.2 + 0.55 * rng.rand()), ec="none", lw=0, zorder=3))
    ax.text(0.095, top - 0.27, "90,351 participants\n× 151 items", ha="center", va="top", fontsize=FS,
            color=INK, linespacing=1.15)
    px, py = 0.275, top
    arrow(ax, 0.170, py, px - 0.032, py)
    ax.scatter([px], [py], s=190, facecolor="white", edgecolor=STAGE, linewidth=LW, zorder=5)
    ax.plot([px - 0.011, px + 0.011], [py, py], color=STAGE, lw=LW, zorder=6, solid_capstyle="butt")
    ax.plot([px, px], [py - 0.055, py + 0.055], color=STAGE, lw=LW, zorder=6, solid_capstyle="butt")
    ax.text(px, 0.95, "the groups", ha="center", va="bottom", fontsize=FS, style="italic", color=INK)
    # name the operation: the circled symbol alone reads as plain addition
    ax.text(px, py - 0.11, "weighted\nsum", ha="center", va="top", fontsize=FS, style="italic", color=INK,
            linespacing=1.1)
    arrow(ax, px, 0.93, px, py + 0.09)
    arrow(ax, px + 0.032, py, 0.365, py)
    # the scatter shows each participant as a point
    ax.text(0.465, 0.95, "Participants placed\nby joint scores", ha="center", va="bottom", fontsize=FS_HEAD,
            fontweight="bold", color=STAGE, linespacing=1.1)
    pts = rng.randn(55, 2) * np.array([0.030, 0.085])
    ax.scatter(0.450 + pts[:, 0], top + pts[:, 1], s=5, color=GREY, zorder=3, lw=0)
    # the center cross: the cohort's mean scores, from which the distance is measured
    ax.plot([0.450], [top], marker="x", ms=7, mew=LW, color=STAGE, zorder=5)
    ax.text(0.458, top + 0.025, "mean scores", ha="left", va="bottom", fontsize=FS, color=STAGE, zorder=6,
            bbox=dict(fc="white", ec="none", pad=0.4))
    arrow(ax, 0.540, py, 0.600, py)
    ax.text(0.775, 0.95, "One distance each", ha="center", va="bottom", fontsize=FS_HEAD, fontweight="bold",
            color=STAGE)
    heights = [0.10, 0.32, 0.66, 0.90, 1.00, 0.88, 0.66, 0.46, 0.30, 0.19, 0.12, 0.08, 0.06, 0.05]
    for i, h in enumerate(heights):
        ax.add_patch(Rectangle((0.650 + 0.0158 * i, top - 0.16), 0.0140, 0.40 * h,
                               fc=FLAG if i >= len(heights) - 2 else "#BFBFBF", ec="none", zorder=3))
    ax.text(0.775, top - 0.27, "from the cohort's mean scores", ha="center", va="top", fontsize=FS,
            style="italic", color=INK)
    ax.text(0.650 + 0.0158 * 13, top - 0.02, "top 5%", ha="center", va="bottom", fontsize=FS, color=FLAG,
            fontweight="bold")
    # the panel is cropped to its content
    ax.set_ylim(0.16, 1.0)


def main():
    it, comp = load()
    # panel heights and the space above each panel (for its letter), in inches, from the top
    HA, HB, HC, TAG = 2.30, 2.95, 1.80, 0.30
    fh = TAG + HA + TAG + HB + TAG + HC + 0.04
    fig = plt.figure(figsize=(W_IN, fh))
    y, axes = fh, []
    for h in (HA, HB, HC):
        y -= TAG + h
        axes.append(fig.add_axes([0.01, y / fh, 0.98, h / fh]))
    axa, axb, axc = axes
    panel_a(axa, it, comp)
    D = panel_b(axb, it, comp)
    panel_c(axc)
    # panel letters in the space above each panel, outside the drawing
    for ax, tag in ((axa, "(A)"), (axb, "(B)"), (axc, "(C)")):
        p = ax.get_position()
        fig.text(p.x0, p.y1 + 0.10 / fh, tag, fontsize=FS_TAG, fontweight="bold", va="bottom", ha="left")
    save(fig, "method")
    print("ribbon memberships:", D.groupby("group").size().to_dict(), "| items:", D.item.nunique())


if __name__ == "__main__":
    main()
