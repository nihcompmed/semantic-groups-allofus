#!/usr/bin/env python
"""fig_method_jno.py -- Figure 1 of the JNO manuscript: the method, in 3 panels.

There is no separate item pool, and nothing is "placed" on a basis built elsewhere: the groups are "fixed from the item
text before any participant's answers were used", in the words of the Methods.

  A  How the groups are formed (SCHEMATIC): item text -> sentence encoder -> each item a point -> sparse projection ->
     groups, 3 drawn as directions with their core items at the far end. The 3 items are shown by topic, not wording
     (TOPIC), 1 core item of each featured group; the point cloud and the directions are schematic.
  B  What a semantic group is (DATA): a ribbon from each instrument (with its documented construct) to the gte group
     whose core the item sits in, for groups 12, 3, 6, 16, 2. Items are assigned by FIXED-POINT CORE membership
     (components.csv core_items), never by the largest loading. The 5 groups are simple examples that still show
     instruments cut and joined: pure 12 (ULS-8), 3 (ACE), 2 (EDS); ACE cut into 3 and 6, EDS cut into 16 and 2; 6 joins
     NDS and ACE (drug and alcohol use), 16 joins DMS and EDS (courtesy, respect). No item sits in 2 of these cores
     (asserted), no 2 labels repeat. 15 (NDS + SCNS) was left out: it shares the NDS drug-use item with 6.
  C  How participants are scored (SCHEMATIC): answers -> scores on every group -> one distance each, from the cohort's
     typical scores, with the top 5% marked.

Reads   data/items_151.csv, outputs/basis/components.csv (gte-large-en-v1.5)
Writes  outputs/method/fig_method_jno.{pdf,png}; preview ../../jno/previews/fig_method_jno.png when ../../jno/ exists
Run     python fig_method_jno.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs", "method")
PREVIEW = os.path.join(ROOT, "..", "jno", "previews")
ENC = "gte-large-en-v1.5"
FEATURED = [(12, "sdoh_ucla_ls8_14"), (3, "ace_5"), (16, "sdoh_dms_2")]     # (group, 1 of its core items)
# each featured item is shown by its topic, not its wording: the ULS-8 and DMS owners' terms do not cover reprinting
# (construct_labels.REPRINT, eMethods 1), and the 3 chips are labeled alike
TOPIC = {"sdoh_ucla_ls8_14": "Feeling isolated", "ace_5": "Parents separated or divorced",
         "sdoh_dms_2": "Less respect at health care visits"}
RIBBON = [12, 3, 6, 16, 2]   # no item in 2 of these cores
INK, RULE, STAGE, GREY, FLAG = "#222222", "#8C8C8C", "#16325C", "#C8C8C8", "#D55E00"
GCOL = {12: "#0072B2", 3: "#E69F00", 16: "#009E73", 22: "#009E73"}          # Okabe-Ito; others neutral
NEUTRAL = "#9AA9B8"
SPELL = [("Healthcare", "Health care"), (" & ", " and "), ("wellbeing", "well-being")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
from construct_labels import display   # noqa: E402


def spell(s):
    # the display name of construct_labels.DISPLAY (SPELL above is unused)
    return display(s)


def load():
    it = pd.read_csv(os.path.join(ROOT, "data", "items_151.csv")).set_index("item")
    it["abbr"] = it["instrument"].str.extract(r"\(([^()]*)\)\s*$")[0]
    comp = pd.read_csv(os.path.join(ROOT, "outputs", "basis", "components.csv"))
    comp = comp[comp.encoder == ENC].set_index("component")
    for g, item in FEATURED:
        assert item in comp.at[g, "core_items"].split(), f"{item} is not a core item of group {g}"
    return it, comp


def lead(comp, g):
    return spell(comp.at[g, "construct_profile"].split(";")[0].split(":")[0])


def chip(ax, x, y, w, h, label, fs, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015", fc=fc, ec=RULE,
                                lw=0.6, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4,
            linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, label=None, dy=0.05, color=RULE, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.9, color=color,
                                                                     mutation_scale=9, linestyle=ls), zorder=2)
    if label:
        ax.text((x0 + x1) / 2, max(y0, y1) + dy, label, ha="center", va="bottom", fontsize=6.6, style="italic",
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
    ax.text(0.115, 0.95, "Items", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=STAGE)
    for k, (g, item) in enumerate(FEATURED):
        y = 0.70 - 0.245 * k
        txt = "\n".join(textwrap.wrap(TOPIC[item], 22))
        chip(ax, 0.005, y, 0.22, 0.20, txt, fs=6.0)
        ax.text(0.232, y + 0.10, it.at[item, "abbr"], ha="left", va="center", fontsize=6.4, fontweight="bold",
                color=GCOL[g])
    # terms as in the text
    ax.text(0.115, 0.02, "151 items from 17 questionnaires\nand 13 stand-alone questions", ha="center", va="bottom",
            fontsize=6.4, color=INK, linespacing=1.2)

    arrow(ax, 0.285, 0.52, 0.345, 0.52, "sentence\nencoder")
    ax.text(0.445, 0.95, "Semantic space", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=STAGE)
    rng = np.random.RandomState(3)
    ax.scatter(0.365 + 0.16 * rng.rand(90), 0.22 + 0.66 * rng.rand(90), s=4, color=GREY, zorder=2, lw=0)
    for (g, _), (x, y) in zip(FEATURED, [(0.395, 0.78), (0.495, 0.30), (0.475, 0.70)]):
        ax.scatter(x, y, s=22, color=GCOL[g], zorder=4, edgecolor="white", linewidth=0.6)
    ax.text(0.445, 0.02, "each item is a point,\nplaced by what it asks", ha="center", va="bottom", fontsize=6.4,
            color=INK, linespacing=1.2)

    # "PCA" above the arrow and "sparse rotation" below it
    arrow(ax, 0.545, 0.52, 0.615, 0.52, "PCA")
    ax.text(0.580, 0.52 - 0.05, "sparse\nrotation", ha="center", va="top", fontsize=6.6, style="italic", color=INK,
            linespacing=1.1)
    k_groups = len(comp)
    ax.text(0.80, 0.95, f"{k_groups} semantic groups", ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            color=STAGE)
    ax.add_patch(FancyBboxPatch((0.63, 0.13), 0.34, 0.78, boxstyle="round,pad=0.004,rounding_size=0.015",
                                fc="#FAFAFA", ec=RULE, lw=0.6, zorder=0))
    ray = ray_fn(ax)
    ox, oy, length = 0.80, 0.50, 0.50
    for (g, _), deg, ha, va in zip(FEATURED, (90, 210, 330), ("center", "center", "center"), ("bottom", "top", "top")):
        xt, yt = ray(ox, oy, deg, length)
        ax.annotate("", xy=(xt, yt), xytext=(ox, oy), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=RULE,
                                                                      mutation_scale=8), zorder=2)
        for f in (0.12, 0.22, 0.32, 0.42):
            ax.scatter(*ray(ox, oy, deg, length * f), s=3, color=GREY, zorder=3, lw=0)
        for f in (0.62, 0.74, 0.86):
            ax.scatter(*ray(ox, oy, deg, length * f), s=14, color=GCOL[g], zorder=4, edgecolor="white", lw=0.5)
        lx, ly = ray(ox, oy, deg, length + (0.07 if deg == 90 else 0.10))
        name = lead(comp, g) if deg == 90 else "\n".join(textwrap.wrap(lead(comp, g), 14))
        ax.text(lx, ly, name, ha=ha, va=va, fontsize=6.6, fontweight="bold", color=GCOL[g], linespacing=1.1)
    ax.scatter([ox], [oy], s=6, color=INK, zorder=5)
    ax.annotate("its core items", xy=ray(ox, oy, 90, length * 0.74), xytext=(0.855, 0.80), fontsize=6.4,
                color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=RULE, connectionstyle="angle3,angleA=0,angleB=70"))
    ax.text(0.80, 0.02, "fixed from the item text before any answers are used", ha="center", va="bottom",
            fontsize=6.4, style="italic", color=INK)


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
    assert D.groupby("group").apply(lambda d: comp.at[d.name, "construct_profile"]).is_unique, "2 ribbon labels repeat"
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
    xl, xr, bw = 0.0, 1.0, 0.010
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
        ax.add_patch(Rectangle((xl - bw, b), 2 * bw, t - b, facecolor=INK, edgecolor="white", lw=0.5, zorder=5))
        ax.text(xl - 0.03, (b + t) / 2 + 0.25, ins, ha="right", va="bottom", fontsize=6.8, fontweight="bold", color=INK)
        ax.text(xl - 0.03, (b + t) / 2 - 0.15, cons[ins], ha="right", va="top", fontsize=6.3, color="#555555")
    for g in RIBBON:
        b, t = R[g]
        col = GCOL.get(g, "#5F6F7F")
        ax.add_patch(Rectangle((xr - bw, b), 2 * bw, t - b, facecolor=col, edgecolor="white", lw=0.5, zorder=5))
        names = [spell(p.split(":")[0]) for p in comp.at[g, "construct_profile"].split("; ")]
        label = "\n".join(textwrap.wrap(f"G{g + 1}  " + ", ".join(names), 40, subsequent_indent="      "))
        ax.text(xr + 0.03, (b + t) / 2, label, ha="left", va="center", fontsize=6.6, color=col, linespacing=1.1,
                fontweight="bold" if g in GCOL else "normal")
    # terms as in the text
    ax.text(xl - 0.03, H + 0.9, "Questionnaire\n(documented construct)", ha="right", va="bottom", fontsize=7.5,
            fontweight="bold", color=STAGE, linespacing=1.1)
    ax.text(xr + 0.03, H + 0.9, "Semantic group\n(constructs of its core items)", ha="left", va="bottom", fontsize=7.5,
            fontweight="bold", color=STAGE, linespacing=1.1)
    ax.set_xlim(-0.62, 1.78); ax.set_ylim(-0.6, H + 3.2); ax.axis("off")
    return D


def panel_c(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    rng = np.random.RandomState(11)
    top = 0.64
    ax.text(0.085, 0.95, "Answers", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=STAGE)
    for r in range(5):
        for c in range(8):
            ax.add_patch(Rectangle((0.020 + 0.0165 * c, top + 0.14 - 0.075 * r), 0.0148, 0.066,
                                   fc=plt.cm.Blues(0.2 + 0.55 * rng.rand()), ec="white", lw=0.3, zorder=3))
    ax.text(0.085, top - 0.29, "90 351 participants\n× 151 items", ha="center", va="top", fontsize=6.4,
            color=INK, linespacing=1.2)
    px, py = 0.265, top
    arrow(ax, 0.160, py, px - 0.030, py)
    ax.scatter([px], [py], s=150, facecolor="white", edgecolor=STAGE, linewidth=0.9, zorder=5)
    ax.plot([px - 0.010, px + 0.010], [py, py], color=STAGE, lw=0.9, zorder=6)
    ax.plot([px, px], [py - 0.05, py + 0.05], color=STAGE, lw=0.9, zorder=6)
    ax.text(px, 0.95, "the groups", ha="center", va="bottom", fontsize=6.6, style="italic", color=INK)
    # name the operation: the circled symbol alone reads as plain addition
    ax.text(px, py - 0.10, "weighted\nsum", ha="center", va="top", fontsize=6.4, style="italic", color=INK,
            linespacing=1.1)
    arrow(ax, px, 0.93, px, py + 0.08)
    arrow(ax, px + 0.030, py, 0.360, py)
    # the scatter shows each participant as a point
    ax.text(0.455, 0.95, "Participants placed\nby joint scores", ha="center", va="bottom", fontsize=7.5,
            fontweight="bold", color=STAGE, linespacing=1.1)
    pts = rng.randn(55, 2) * np.array([0.030, 0.085])
    ax.scatter(0.455 + pts[:, 0], top + pts[:, 1], s=3, color=GREY, zorder=3, lw=0)
    # the center cross: the cohort's typical scores, from which the distance is measured
    ax.plot([0.455], [top], marker="x", ms=6, mew=1.4, color=STAGE, zorder=5)
    ax.text(0.463, top + 0.018, "typical scores", ha="left", va="bottom", fontsize=5.8, color=STAGE, zorder=6,
            bbox=dict(fc="white", ec="none", pad=0.4))
    arrow(ax, 0.545, py, 0.600, py)
    ax.text(0.760, 0.95, "One distance each", ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            color=STAGE)
    heights = [0.10, 0.32, 0.66, 0.90, 1.00, 0.88, 0.66, 0.46, 0.30, 0.19, 0.12, 0.08, 0.06, 0.05]
    for i, h in enumerate(heights):
        ax.add_patch(Rectangle((0.650 + 0.0158 * i, top - 0.16), 0.0140, 0.40 * h,
                               fc=FLAG if i >= len(heights) - 2 else "#BFBFBF", ec="none", zorder=3))
    ax.text(0.760, top - 0.29, "from the cohort's typical scores", ha="center", va="top", fontsize=6.4,
            style="italic", color=INK)
    ax.text(0.650 + 0.0158 * 13, top + 0.00, "top 5%", ha="center", va="bottom", fontsize=6.2, color=FLAG,
            fontweight="bold")
    # the panel is cropped to its content
    ax.set_ylim(0.22, 1.0)


def main():
    it, comp = load()
    # the axes fractions give panels A and B fixed heights in inches; panel C is cropped to its content
    fig = plt.figure(figsize=(7.0, 6.83))
    axa = fig.add_axes([0.01, 0.6852, 0.98, 0.2826])
    axb = fig.add_axes([0.01, 0.2899, 0.98, 0.3529])
    axc = fig.add_axes([0.01, 0.0102, 0.98, 0.2372])
    panel_a(axa, it, comp)
    D = panel_b(axb, it, comp)
    panel_c(axc)
    for ax, tag in ((axa, "A"), (axb, "B"), (axc, "C")):
        p = ax.get_position()
        fig.text(p.x0, p.y1 + 0.004, tag, fontsize=10, fontweight="bold", va="bottom", ha="left")
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_method_jno.{ext}"), dpi=300)
    # the preview goes to the manuscript folder only when it exists, so a copy of the code elsewhere writes only
    # inside its own outputs/
    if os.path.isdir(os.path.dirname(PREVIEW)):
        os.makedirs(PREVIEW, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW, "fig_method_jno.png"), dpi=200)
    print("ribbon memberships:", D.groupby("group").size().to_dict(), "| items:", D.item.nunique())
    print("wrote", os.path.join(OUT, "fig_method_jno.pdf"))


if __name__ == "__main__":
    main()
