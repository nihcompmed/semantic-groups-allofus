#!/usr/bin/env python
"""figure7.py -- Figure 7 and its figure supplements 1 to 5: the semantic groups and the 8 recorded conditions as 2
heatmaps, A each group on its own, B each group adjusted for the other groups; Figure 7 for the prespecified encoder, the supplements for the
other 5 encoders.

The models are those without the item on a diagnosed life-threatening illness (weight 0 in every group), with age, age
squared and sex as covariates (sex left out for the 2 cancers). The cell value is the size of the log odds ratio per SD
of the group score, |log OR| (a group's sign is arbitrary). The 2 models answer different questions, so each panel has
its own color scale (blue for A, orange for B, light = 0 to dark = the panel's largest value) and they are not compared
cell by cell. No significance marks. Rows are the groups G1 to Gk, each labeled by every construct of its core items with
its number of core items (wrapped to at most 2 lines); columns are the 8 conditions, the 4 mental and behavioral first.

Reads   the download conditions (condition_groups.csv: the *_without columns)
Writes  elife/figures/conditions.{png,pdf} (gte-large-en-v1.5) and conditions_<encoder>.{png,pdf} for the other 5
Run     python figure7.py
"""
import os

import numpy as np
import pandas as pd

from common import CONDITIONS as _C, ENCODERS, FIGURES, MENTAL as _MENTAL, PRIMARY, SHORT_CONDITION as _SHORT, WB
from common import display_profile

PANELS = [("alone", "A  Each group on its own", "Blues"), ("with", "B  Adjusted for the other groups", "Oranges")]
FS = {"base": 8.0, "x": 8.5, "y": 8.0, "title": 9.5, "cbar": 8.0}
MAX_LINES = 2           # every row label fits in 2 lines


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


def label_width(profiles):
    """The narrowest wrap (characters) that keeps every row label to MAX_LINES lines."""
    for w in range(30, 120):
        if max(row_label(cp, j, w).count("\n") + 1 for j, cp in profiles) <= MAX_LINES:
            return w
    raise AssertionError("a row label does not fit in MAX_LINES lines at any width")


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
    rows = [row_label(cp, j, label_width(profiles)) for j, cp in profiles]
    n_m = sum(c in _MENTAL for c in conds)
    assert all(c in _MENTAL for c in conds[:n_m]), "mental columns not first"
    fig = plt.figure(figsize=(7.0, 0.3 * k + 2.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.3 * k, 0.12], hspace=0.30, wspace=0.08)
    for i, (key, title, cmap) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i])
        vmax = float(M[key].max())
        im = ax.imshow(M[key], cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.axvline(n_m - 0.5, color="white", lw=2.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels([_SHORT[c] for c in conds], rotation=45, ha="right", rotation_mode="anchor",
                           fontsize=FS["x"])
        ax.set_yticks(range(k))
        ax.set_yticklabels(rows if i == 0 else [], fontsize=FS["y"], linespacing=1.05)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_linewidth(0.6)
        ax.set_title(title, loc="left", fontsize=FS["title"], pad=6)
        cax = fig.add_subplot(gs[1, i])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("Log odds ratio per SD, size", fontsize=FS["cbar"])
        cb.ax.tick_params(labelsize=FS["cbar"])
    os.makedirs(FIGURES, exist_ok=True)
    out = os.path.join(FIGURES, "conditions.png" if enc == PRIMARY else f"conditions_{enc}.png")
    for path in (out, os.path.splitext(out)[0] + ".pdf"):
        fig.savefig(path, dpi=300, bbox_inches="tight")
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
