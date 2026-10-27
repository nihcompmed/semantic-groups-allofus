#!/usr/bin/env python
"""figure7.py -- Figure 7 and Figures S10 to S14: the semantic groups and the 8 recorded conditions as 2
heatmaps, A each group on its own, B each group adjusted for the other groups; Figure 7 for the prespecified encoder, the supplements for the
other 5 encoders.

The models are those without the item on a diagnosed life-threatening illness (weight 0 in every group), with age, age
squared and sex as covariates (sex left out for the 2 cancers). The cell value is the size of the log odds ratio per SD
of the group score, |log OR| (a group's sign is arbitrary). The 2 models answer different questions, so each panel has
its own color scale (blue for A, orange for B, light = 0 to dark = the panel's largest value) and they are not compared
cell by cell. No significance marks. Rows are the groups G1 to Gk, each labeled by every construct of its core items with
its number of core items (wrapped to at most 2 lines, or more where 2 would leave the heatmaps too narrow); columns are the 8 conditions, the 4 mental and behavioral first.
Drawn at its printed size, the 2-column width of 180 mm, with no text under 8 pt and no line under 2 pt (Frontiers
figure guidelines); the layout is set in inches, the label column as wide as the longest row label.

Reads   the download conditions (condition_groups.csv: the *_without columns)
Writes  output/figures/conditions.{png,pdf} (gte-large-en-v1.5) and conditions_<encoder>.{png,pdf} for the other 5
Run     python figure7.py
"""
import os
import textwrap

import numpy as np
import pandas as pd

from common import CONDITIONS as _C, ENCODERS, FIGURES, MENTAL as _MENTAL, PRIMARY, SHORT_CONDITION as _SHORT, WB
from common import display_profile

PANELS = [("alone", "A", "Each group on its own", "Blues"), ("with", "B", "Adjusted for the other groups", "Oranges")]
FS = {"base": 8.0, "x": 8.0, "y": 8.0, "title": 9.0, "cbar": 8.0, "tag": 10.0}
MAX_LINES = (2, 3, 4)   # every row label fits in 2 lines, or more if the heatmaps would be narrower than PW_MIN
PW_MIN = 1.4            # inches, the narrowest heatmap panel (8 columns of rotated names at 8 pt clear each other)
LW = 2.0                # pt: every line
W_IN = 180 / 25.4       # the 2-column width
LS = 1.0                # line spacing of the row labels
# inches: rotated condition names, gap, colorbar, its ticks and label, left and right edges, gap between the panels
XL, CBG, CBH, CBL, EDGE, EDGE_R, GAP = 1.02, 0.10, 0.11, 0.40, 0.04, 0.10, 0.12


def row_label(cp, j, width):
    """G<j + 1>, then each construct (display name) with its number of core items, wrapped by word to `width`."""
    names = [p.replace(":", " ") for p in display_profile(cp).split("; ")] if isinstance(cp, str) and cp else []
    words = []                                   # a construct's last word keeps its count
    for k, n in enumerate(names):
        parts = n.split(" ")
        if len(parts) >= 2:
            parts = parts[:-2] + [parts[-2] + " " + parts[-1]]
        if k < len(names) - 1:
            parts[-1] += ","
        words += parts
    head = f"G{j + 1}  "
    lines, cur = [], ""
    for w in words:
        room = width - (len(head) if not lines else 6)
        if cur and len(cur) + 1 + len(w) > room:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return head + "\n      ".join(lines)


def label_width(profiles, max_lines):
    """The narrowest wrap (characters) that keeps every row label to max_lines lines."""
    for w in range(30, 120):
        if max(row_label(cp, j, w).count("\n") + 1 for j, cp in profiles) <= max_lines:
            return w
    raise AssertionError("a row label does not fit in max_lines lines at any width")


def matrices(enc):
    G = pd.read_csv(os.path.join(WB, "conditions", "condition_groups.csv"))
    G = G[G.encoder == enc]
    assert len(G), f"no rows for {enc}"
    conds = [c[0] for c in _C]
    assert set(conds) == set(G.condition), "the conditions differ from common.CONDITIONS"
    groups = sorted(G.group.unique())
    M = {"alone": np.full((len(groups), len(conds)), np.nan)}
    M["with"] = M["alone"].copy()
    for jc, c in enumerate(conds):
        t = G[G.condition == c].set_index("group").loc[groups]
        M["alone"][:, jc] = np.abs(np.log(t.or_alone_without.values))
        M["with"][:, jc] = np.abs(np.log(t.or_without.values))
    assert not np.isnan(M["alone"]).any() and not np.isnan(M["with"]).any(), "a cell is missing"
    profiles = [(int(g), G[G.group == g].construct_profile.iloc[0]) for g in groups]
    return conds, groups, M, profiles


def draw(enc, plt):
    conds, groups, M, profiles = matrices(enc)
    k, n = len(groups), len(conds)
    n_m = sum(c in _MENTAL for c in conds)
    assert all(c in _MENTAL for c in conds[:n_m]), "mental columns not first"
    fig = plt.figure(figsize=(W_IN, 9.0))
    r = fig.canvas.get_renderer()

    def size(s, **kw):                           # width and height of a text, inches
        t = fig.text(0, 0, s, **kw)
        e = t.get_window_extent(r)
        t.remove()
        return e.width / fig.dpi, e.height / fig.dpi
    for max_lines in MAX_LINES:
        rows = [row_label(cp, j, label_width(profiles, max_lines)) for j, cp in profiles]
        lab = max(size(s, fontsize=FS["y"], linespacing=LS)[0] for s in rows)
        x0 = EDGE + lab + 0.07                   # the label column, then the tick pad
        pw = (W_IN - x0 - EDGE_R - GAP) / 2
        if pw >= PW_MIN:
            break
    # one row height for all, so that every label sits inside its own row
    ROW = max(size(s, fontsize=FS["y"], linespacing=LS)[1] for s in rows) + 0.012
    # each title in as few lines as fit the panel's width, the lines balanced; the panel letter on the line above
    titles = {}
    for _, tag, title, _ in PANELS:
        for n_lines in range(1, 4):
            t = "\n".join(textwrap.wrap(title, -(-len(title) // n_lines) + 3))
            if size(t, fontsize=FS["title"])[0] <= pw:
                break
        titles[tag] = t
    th = max(size(t, fontsize=FS["title"])[1] for t in titles.values())
    tag_h = size("(A)", fontsize=FS["tag"], fontweight="bold")[1]
    hh = ROW * k
    top = 0.04 + tag_h + 0.04 + th + 0.06
    fh = top + hh + XL + CBG + CBH + CBL
    fig.set_size_inches(W_IN, fh)
    for i, (key, tag, title, cmap) in enumerate(PANELS):
        xa = (x0 + i * (pw + GAP)) / W_IN
        ax = fig.add_axes([xa, (CBL + CBH + CBG + XL) / fh, pw / W_IN, hh / fh])
        vmax = float(M[key].max())
        im = ax.imshow(M[key], cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.axvline(n_m - 0.5, color="white", lw=2.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels([_SHORT[c] for c in conds], rotation=45, ha="right", rotation_mode="anchor",
                           fontsize=FS["x"])
        ax.set_yticks(range(k))
        ax.set_yticklabels(rows if i == 0 else [], fontsize=FS["y"], linespacing=LS)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_linewidth(LW)
        # the title above the panel, its top lines level across the panels, and the panel letter above the title
        yt = (fh - 0.04 - tag_h - 0.04) / fh
        fig.text(xa, yt, titles[tag], ha="left", va="top", fontsize=FS["title"])
        fig.text(xa, (fh - 0.04) / fh, f"({tag})", ha="left", va="top", fontsize=FS["tag"], fontweight="bold")
        cax = fig.add_axes([xa, CBL / fh, pw / W_IN, CBH / fh])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("Log odds ratio per SD, size", fontsize=FS["cbar"])
        cb.ax.tick_params(labelsize=FS["cbar"], width=LW)
        cb.outline.set_linewidth(LW)
    os.makedirs(FIGURES, exist_ok=True)
    out = os.path.join(FIGURES, "conditions.png" if enc == PRIMARY else f"conditions_{enc}.png")
    for path in (out, os.path.splitext(out)[0] + ".pdf"):
        fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"-> {out} (+ .pdf)")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS["base"]})
    for enc in ENCODERS:
        draw(enc, plt)


if __name__ == "__main__":
    main()
