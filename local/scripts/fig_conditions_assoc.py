#!/usr/bin/env python
"""fig_conditions_assoc.py -- Figure 4 (and eFigure 3 with --encoder): the semantic groups and the 8
recorded conditions of record, one heatmap per encoder.

LOCAL. Draws from the downloaded conditions_assoc_jno/conditions_assoc_jno_<enc>.csv and conditions_assoc_or_jno_<enc>.csv
(workbench/conditions_assoc_jno.py).

ONE HEATMAP, k groups (rows) x 8 conditions (columns): the odds ratio of each group per SD with all k groups in the
model, net of age and sex, by SIZE only (OR or 1/OR, whichever is >= 1). Markers: filled dot = significant together
AND alone (P < .05/k in each); ring = significant ONLY together; small cross = significant only alone. Above each
column: the joint test of the k groups (star = P < .05). The thresholds are not divided by the number of conditions.

LABELS: columns = the condition names of conditions_local.py (the same as eTable 7 and eFigure 2, same order), ADHD
abbreviated, superscript a on the 3 conditions whose participants were all tested; the columns grouped under "Mental
and behavioral" and "Physical". Rows = group number (as in eTable 4) and EVERY construct of the core items, commas,
AMA spelling (health care, and, well-being), wrapped. The key is a marker legend above the plot (no suptitle); the
colorbar says "Odds ratio per SD". Writes PNG and PDF.

Run:  python fig_conditions_assoc.py [--encoder <name>] [--wb <dir>] [--out <png>]
"""
import glob
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd

from conditions_local import CONDITIONS as _C, MENTAL as _MENTAL, SHORT as _SHORT, TESTED as _TESTED   # the 8 conditions
from construct_labels import display_profile   # printed construct names

HERE = os.path.dirname(os.path.abspath(__file__))
WB_ROOT = os.path.join(HERE, "..", "outputs", "workbench")
OUT_DIR = os.path.join(HERE, "..", "outputs", "conditions_assoc")
PREVIEW = os.path.join(HERE, "..", "..", "jno", "previews")
ENC = sys.argv[sys.argv.index("--encoder") + 1] if "--encoder" in sys.argv else "gte-large-en-v1.5"
SPELL = [("Healthcare", "Health care"), (" & ", " and "), ("wellbeing", "well-being")]
# text sizes, pt, at the figure's 7.0-inch width
# The figure is scaled to the page width, so text grows on paper only if the labels take less room: column labels are
# the short names of Figure 2D at 45 degrees (not wrapped full names at 90), and the row labels wrap by word (a
# construct's count stays with its name) at the narrowest width that keeps every label to MAX_LINES lines.
FS = {"base": 9.0, "x": 10.5, "y": 9.5, "head": 10.5, "star": 13, "legend": 9.5, "cbar": 9.5, "cbar_ticks": 9.0}
MAX_LINES = 2           # every row label fits in 2 lines, so rows of equal height never overlap (3-line labels would)
ROW_IN = 0.34           # height of 1 heatmap row = 2 label lines at FS["y"], inches
FIG_W = 5.4             # inches of the heatmap side; the row labels extend it to the left (bbox "tight")
CONDITIONS = [(k, lab) for k, lab, _, _ in _C]
COLS = dict(_SHORT)      # the short names of Figure 2D (GAD, ADHD, PTSD, Panic disorder, High cholesterol, ...)
TESTED = set(_TESTED)                                     # everyone in the population was tested (download meta "tested_by")
MENTAL = set(_MENTAL)


def wb_dir():
    if "--wb" in sys.argv:
        return sys.argv[sys.argv.index("--wb") + 1]
    hits = sorted(glob.glob(os.path.join(WB_ROOT, "*", "conditions_assoc_jno", f"conditions_assoc_jno_{ENC}.csv")),
                  key=os.path.getmtime)
    assert hits, f"no conditions_assoc_jno_{ENC}.csv under {WB_ROOT}; unpack its zip there"
    return os.path.dirname(os.path.dirname(hits[-1]))


def row_label(cp, j, width):
    # each construct with its number of core items, as eTable 4 names the groups, so rows with the same constructs
    # (gte 16 and 22, 2 and 8) read differently; display names (construct_labels)
    names = [p.replace(":", " ") for p in display_profile(cp).split("; ")] if isinstance(cp, str) and cp else []
    words = []                                   # wrap by word; a construct's last word keeps its count
    for k, n in enumerate(names):
        parts = n.split(" ")
        if len(parts) >= 2:
            parts = parts[:-2] + [parts[-2] + " " + parts[-1]]
        if k < len(names) - 1:
            parts[-1] += ","
        words += parts
    head = f"G{j + 1}  "                         # displayed group ID = G + (component index + 1)
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


def col_label(key):
    lab = COLS[key]
    return lab + ("$^{\\mathrm{a}}$" if key in TESTED else "")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS["base"]})
    wb = wb_dir()
    D = pd.read_csv(os.path.join(wb, "conditions_assoc_jno", f"conditions_assoc_jno_{ENC}.csv"))
    G = pd.read_csv(os.path.join(wb, "conditions_assoc_jno", f"conditions_assoc_or_jno_{ENC}.csv"))
    meta = json.load(open(os.path.join(wb, "conditions_assoc_jno", f"conditions_assoc_jno_{ENC}_meta.json")))
    assert set(meta["populations"]["tested_by"]) == TESTED, f"tested conditions in the meta != {TESTED}"
    assert abs(meta["p_joint"] - 0.05) < 1e-12 and abs(meta["p_group"] - 0.05 / meta["k"]) < 1e-12, "thresholds differ"
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(OUT_DIR, f"fig_conditions_assoc_{ENC}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    conds = [c for c, _ in CONDITIONS]
    assert set(conds) == set(D["condition"]), "the conditions differ from conditions_local.CONDITIONS"
    D = D.set_index("condition").loc[conds].reset_index()
    groups = sorted(G["group"].unique())
    k = len(groups)
    M = np.full((k, len(conds)), np.nan)
    tog = np.zeros_like(M, bool); alo = np.zeros_like(M, bool)
    for jc, c in enumerate(conds):
        t = G[G.condition == c].set_index("group").loc[groups]
        M[:, jc] = np.exp(np.abs(np.log(t["or"].values)))
        tog[:, jc], alo[:, jc] = t["passes_together"].values, t["passes_alone"].values
    profiles = [(int(g), G[G.group == g]["construct_profile"].iloc[0]) for g in groups]
    width = label_width(profiles)
    rows = [row_label(cp, j, width) for j, cp in profiles]

    fig, ax = plt.subplots(figsize=(FIG_W, ROW_IN * k + 2.6))
    vmax = float(np.nanpercentile(M, 99))
    im = ax.imshow(M, cmap="Reds", vmin=1, vmax=vmax, aspect="auto")
    # markers turn white on dark cells (the upper 40% of the color scale), where dark markers would disappear
    dark = (M - 1) / (vmax - 1) > 0.6
    for mask, kw in ((tog & alo, dict(s=16, marker="o", linewidths=0)),
                     (tog & ~alo, dict(s=34, facecolors="none", marker="o", linewidths=1.0)),
                     (~tog & alo, dict(s=18, marker="x", linewidths=0.8))):
        for on_dark, col in ((False, "0.15"), (True, "white")):
            yy, xx = np.nonzero(mask & (dark == on_dark))
            key = "edgecolors" if kw.get("facecolors") == "none" else "c"
            ax.scatter(xx, yy, **{**kw, key: col})
    for jc, r in enumerate(D.itertuples()):
        ax.text(jc, -0.75, "*" if r.passes_joint else "", ha="center", va="bottom", fontsize=FS["star"])
    # the 2 blocks of columns, with a white separator and a heading over each
    n_m = sum(c in MENTAL for c in conds)
    assert all(c in MENTAL for c in conds[:n_m]) and not any(c in MENTAL for c in conds[n_m:]), "mental columns not first"
    ax.axvline(n_m - 0.5, color="white", lw=2.5)
    for x0, x1, name in ((0, n_m - 1, "Mental and behavioral"), (n_m, len(conds) - 1, "Physical")):
        ax.plot([x0 - 0.4, x1 + 0.4], [-1.7, -1.7], color="0.3", lw=0.8, clip_on=False)
        ax.text((x0 + x1) / 2, -1.85, name, ha="center", va="bottom", fontsize=FS["head"], clip_on=False)
    ax.set_ylim(k - 0.5, -0.5)            # the headings and stars sit above the frame, outside the data limits
    ax.set_xlim(-0.5, len(conds) - 0.5)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([col_label(c) for c in conds], rotation=45, ha="right", rotation_mode="anchor", fontsize=FS["x"])
    ax.set_yticks(range(k))
    ax.set_yticklabels(rows, fontsize=FS["y"], linespacing=1.05)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    cb = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.015, aspect=25)
    cb.set_label("Odds ratio per SD", fontsize=FS["cbar"])
    cb.ax.tick_params(labelsize=FS["cbar_ticks"])
    # "Significant ...", to match the legend and Results
    handles = [Line2D([], [], ls="", marker="o", ms=4.2, color="0.15", label="Significant with all groups and alone"),
               Line2D([], [], ls="", marker="o", ms=6.0, mfc="none", mec="0.1", mew=1.0, label="Significant only with all groups"),
               Line2D([], [], ls="", marker="x", ms=4.6, color="0.15", mew=0.8, label="Significant only alone"),
               Line2D([], [], ls="", marker="$*$", ms=7, color="black", label="Joint test significant")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 1 + 3.0 / k), borderaxespad=0, ncol=2,
              frameon=False, fontsize=FS["legend"], handletextpad=0.3, columnspacing=1.6)
    for path in (out, os.path.splitext(out)[0] + ".pdf"):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    # the preview only where the folder ../../jno exists, as in fig_outcome_deciles.py; the outputs do not depend on it
    if ENC == "gte-large-en-v1.5" and "--out" not in sys.argv and os.path.isdir(os.path.dirname(PREVIEW)):
        os.makedirs(PREVIEW, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW, "fig_conditions_assoc.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out} (+ .pdf)")


if __name__ == "__main__":
    main()
