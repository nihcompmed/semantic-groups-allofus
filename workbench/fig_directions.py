#!/usr/bin/env python3
"""fig_directions.py -- which semantic groups account for the most burdened, one panel per encoder.
Runs on the Workbench or locally; the PNG is an aggregate and leaves.

EACH POINT IS A GROUP of that encoder's basis, labelled by the content of its core items. Averaged over
the people direction_own.py selected: with --tag own, that encoder's OWN flagged set at its 95th
percentile, and with --tag own_invisible, the part of it meeting none of the six criteria.

    x = mean |z_j|   MARGINAL, how far out they sit on that group alone, in cohort SDs
    y = mean |r_j|   CONDITIONAL, the standardized residual of that group given all the others, so how
                     surprising it is once the rest of their answers are accounted for

MAGNITUDES ONLY, AND NOTHING ABOUT MORE OR LESS. T^2(-y) = T^2(y), and a group of the fit has no
canonical orientation (the same content carries opposite signs under different encoders). The figure
says which groups account for the distance. It says nothing about anyone having more or less of
anything.

THE DIAGONAL is the reference. On it, marginal and conditional agree, so the group is effectively
independent of the rest of the profile. Above it, more surprising once the rest is accounted for; below
it, the rest already predicted it. Point area is the mean share of T^2, the exact additive partition.

TWO-SIDED GROUPS ARE NOT MARKED. Every group is labeled by the constructs of ALL its core items, listed
"A, B" (most core items first, the first 2 and "+n" for the rest), and drawn with the same filled marker.
A group whose core items carry opposite signs is not labeled "A vs B", consistent with reading groups by
size only.

THE TOP-3 FILE. Given --top3, the figure is drawn from direction_top3_<tag>.csv instead, where a group
is drawn when more than 20 people put it in their top 3 by share of T^2 (the dissemination policy),
and its |z|, |r| and share are averaged over those people alone rather than over everyone. Both axes are drawn either way: the
selection sets which people a group is averaged over, never what the axes show. The bare --ranking form
draws the abs_z and abs_r diagnostics, which are NOT for the paper, because selecting people by |z| or
|r| inflates the axis that selected them. One panel per encoder, and --encoders picks which to draw.

LABELS are placed greedily, largest share first, each taking the first candidate offset that clears
every label already placed, with a hairline leader when it lands far from its marker. The run prints
how many were placed clear per panel. Long names are truncated.

INPUTS   screen_out/direction_summary_<tag>.csv   (from direction_own.py)
         screen_out/direction_top3_<tag>.csv      (the same, restricted, with --top3 or --ranking)
         screen_out/semantic_group_defs.csv     (from semantic_groups.py, THE definition of a group)
OUTPUTS  screen_out/fig_directions_<tag>.png, or fig_directions_<tag>_<ranking>.png

Run:  python3 fig_directions.py --tag own              (each encoder's own flagged set, no selection)
      python3 fig_directions.py --tag own --top3       (the top-3 view, all encoders)
      python3 fig_directions.py --tag own --top3 --encoders gte-large-en-v1.5   (1 encoder's panel)
      python3 fig_directions.py --tag own_invisible    (the same on those meeting none of the six)
      python3 fig_directions.py --tag own --ranking abs_r   (a diagnostic, not for the paper)
      python3 fig_directions.py --summary <csv> --out <png>     (locally, on a file brought back)
      --tag is required in practice: no script in this folder writes the files that the default tag
      `outliers`, and `invisible` from --invisible, name.
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_config import CFG   # noqa: E402
try:
    from construct_labels import short            # display names, when the module travels with us
except Exception:                                  # pragma: no cover - internal labels are readable too
    def short(x, strict=True):
        return str(x)

ARG = lambda f, d=None: sys.argv[sys.argv.index(f) + 1] if f in sys.argv else d
TAG = "invisible" if "--invisible" in sys.argv else ARG("--tag", "outliers")
# --top3 draws the top-3 panel and needs no argument, because "top 3" means by share. The bare
# --ranking form stays for the abs_z and abs_r diagnostics. Both draw |z| against |r|: the ranking sets
# which people a group is averaged over, never what the axes show.
RANKING = "share" if "--top3" in sys.argv else ARG("--ranking")
DEFAULT_SRC = f"direction_top3_{TAG}.csv" if RANKING else f"direction_summary_{TAG}.csv"
SUMMARY = ARG("--summary", os.path.join(CFG.out_dir, DEFAULT_SRC))
OUT = ARG("--out", os.path.join(CFG.out_dir,
          f"fig_directions_{TAG}{'_' + RANKING if RANKING else ''}.png"))
DEFS = ARG("--groups", os.path.join(CFG.out_dir, "semantic_group_defs.csv"))
# a comparator's units carry published names, not construct labels: see read_labels
RAW_LABELS = "--raw-labels" in sys.argv
ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5", "all-mpnet-base-v2", "e5-large-v2",
            "qwen3-embedding-0.6b"]
ONLY = [e for e in (ARG("--encoders") or "").split(",") if e]
RANK_NAME = {"abs_z": "$|z|$", "abs_r": "$|r|$", "share": "share of $T^2$"}
MAXLAB = 34                                 # longer names are truncated
OKABE = {"orange": "#E69F00", "blue": "#0072B2", "grey": "0.45"}


def place_labels(fig, ax, items, fontsize):
    """Label every point, greedily, without stacking labels on one another.

    Points are placed in order of importance, so the groups holding the most of the distance get first
    choice of position and the small ones take what is left. Each label tries a ring of candidate
    offsets and takes the first whose text box misses every box already placed. A point that cannot be
    placed anywhere clear keeps the default position rather than losing its label, because a missing
    label is worse than a crowded one.

    items: (x, y, text, priority) with larger priority placed first.
    """
    CAND = [(0, 10), (0, -14), (12, 3), (-12, 3), (12, -10), (-12, -10),
            (0, 21), (0, -25), (22, 12), (-22, 12), (22, -20), (-22, -20)]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    # a label pushed clear of the crowd is no longer obviously attached to its point, so anything past
    # the near ring gets a hairline back to the marker
    leader = dict(arrowstyle="-", lw=0.4, color="0.6", shrinkA=0, shrinkB=3)
    placed, n_clear = [], 0
    for x, y, txt, _ in sorted(items, key=lambda t: -t[3]):
        chosen = None
        for i, (dx, dy) in enumerate(CAND):
            ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
            an = ax.annotate(txt, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha,
                             fontsize=fontsize, color="0.15", zorder=4,
                             arrowprops=leader if i >= 4 else None)
            bb = an.get_window_extent(renderer=rend).expanded(1.03, 1.12)
            if not any(bb.overlaps(p) for p in placed):
                chosen, n_clear = (an, bb), n_clear + 1
                break
            an.remove()
        if chosen is None:
            an = ax.annotate(txt, (x, y), textcoords="offset points", xytext=CAND[0], ha="center",
                             fontsize=fontsize, color="0.45", zorder=4)
            chosen = (an, an.get_window_extent(renderer=rend))
        placed.append(chosen[1])
    return n_clear, len(items)


def read_labels(path, raw=False):
    """(encoder, component) -> (label, is_two_sided), from semantic_group_defs.csv. The grouping is
    defined in semantic_groups.py and is not recomputed here.

    raw=True takes `construct_profile` VERBATIM instead of parsing it into construct names and
    passing them through construct_labels.short(). That parsing exists because a semantic group is
    named by the documented constructs of its core items, and an unknown name there is a mistake
    worth stopping for. A COMPARATOR's units are not named that way -- the scale model's units are
    published instruments, PHQ-9 and DMS and HVS, which carry their own names and are not construct
    labels at all -- so validating them against the construct vocabulary would reject correct names.
    Use --raw-labels only for a defs file whose labels are already what should be printed."""
    D = pd.read_csv(path)
    out = {}
    for (enc, j), d in D.groupby(["encoder", "component"]):
        d = d.sort_values("group")                       # C12 alone, or C12+ before C12-
        if raw:
            out[(enc, int(j))] = (", ".join(str(r.construct_profile) for _, r in d.iterrows()), False)
            continue
        # ★ the constructs of ALL the core items, both sides pooled, listed "A, B" --
        # most core items first, the first 2 and "+n" -- never "A vs B", and never marked two-sided
        count = {}
        for _, r in d.iterrows():
            for x in str(r.construct_profile).split("; "):
                name, _, n = x.rpartition(" ")
                name, n = (name, int(n)) if n.isdigit() else (x, 1)
                count[name] = count.get(name, 0) + n
        names = [short(n) for n, _ in sorted(count.items(), key=lambda t: -t[1])]
        out[(enc, int(j))] = (", ".join(names[:2]) + (f" +{len(names) - 2}" if len(names) > 2 else ""), False)
    return out


S = pd.read_csv(SUMMARY)
if "ranking" in S.columns:
    if not RANKING:
        raise SystemExit(f"{SUMMARY} holds three rankings ({', '.join(sorted(set(S.ranking)))}). "
                         "They answer different questions and must not be drawn together. "
                         "Pass --ranking to choose one.")
    S = S[S.ranking == RANKING]
    if not len(S):
        raise SystemExit(f"no rows with ranking={RANKING} in {SUMMARY}")
elif RANKING:
    raise SystemExit(f"--ranking given but {SUMMARY} has no ranking column. The top-3 file is "
                     f"direction_top3_{TAG}.csv, written by direction_own.py.")
# Rows with 1 to 20 people are kept in the top-3 file with n_top "<=20" and blank statistics
# (direction_own.py, dissemination policy). They have nothing to draw, so they are dropped
# here and counted. Complementary rows ("suppressed") keep their statistics and are drawn without n.
held = S.mean_abs_z.isna() | S.mean_abs_r.isna()
if held.any():
    print(f"  {int(held.sum())} group(s) with 1 to 20 people are not drawn")
    S = S[~held].copy()
LAB = read_labels(DEFS, raw=RAW_LABELS)
present = [e for e in ENCODERS if e in set(S.encoder)] or sorted(set(S.encoder))
if ONLY:
    present = [e for e in present if e in ONLY] or present
ncol = 3 if len(present) > 2 else len(present)
nrow = int(np.ceil(len(present) / ncol))
single = len(present) == 1                       # a single panel, built near print width
fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                         figsize=(6.5, 5.6) if single else (5.2 * ncol, 5.0 * nrow))
# a direction can carry a negative mean share (it lowers the distance), and an area cannot be negative,
# so the area floors at zero and such a direction is drawn at the minimum size.
smax = max(float(S.mean_share.max()), 1e-9)
placement = []
for ax, enc in zip(axes.ravel(), present):
    d = S[S.encoder == enc]
    xs, ys = d.mean_abs_z.values, d.mean_abs_r.values
    mx, my = (xs.max() - xs.min()) or 1.0, (ys.max() - ys.min()) or 1.0
    xlo, xhi = xs.min() - 0.16 * mx, xs.max() + 0.16 * mx
    ylo, yhi = ys.min() - 0.20 * my, ys.max() + 0.26 * my
    lo, hi = min(xlo, ylo), max(xhi, yhi)
    ax.plot([lo, hi], [lo, hi], color=OKABE["grey"], lw=0.8, zorder=1)
    todo = []
    for _, r in d.iterrows():
        lab, two = LAB.get((enc, int(r.component)), (f"C{int(r.component)}", False))
        if len(lab) > MAXLAB:                       # full names are in the eTable, not on the plot
            lab = lab[:MAXLAB - 1].rstrip(" /;,") + "…"
        if "n_top" in d.columns and str(r.n_top).strip().isdigit():
            lab = f"{lab} ({int(r.n_top)})"
        ax.scatter(r.mean_abs_z, r.mean_abs_r, s=40 + 900 * max(float(r.mean_share), 0.0) / smax,
                   facecolor="none" if two else OKABE["orange"],
                   edgecolor=OKABE["blue"] if two else "none",
                   linewidths=1.4, alpha=1.0 if two else 0.75, zorder=3)
        todo.append((r.mean_abs_z, r.mean_abs_r, lab, float(r.mean_share)))
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    clear, total = place_labels(fig, ax, todo, 6.5 if single else 5.5)
    placement.append((enc, clear, total))
    # the ranking picks WHICH PEOPLE each group is averaged over. Both axes are drawn whichever ranking
    # is used, so the title names the selection briefly and the caption carries the rest. Keep it to one
    # line: a wrapped title on a left-aligned panel runs off the right-hand column of the figure.
    ax.set_title(f"{enc}, {len(d)} groups, top 3 by {RANK_NAME[RANKING]}" if RANKING
                 else f"{enc}, {len(d)} groups",
                 loc="left", fontsize=9)
    ax.set_xlabel("mean $|z_j|$, marginal (cohort SDs)", fontsize=8)
    ax.set_ylabel("mean $|r_j|$, conditional (cohort SDs)", fontsize=8)
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
for ax in axes.ravel()[len(present):]:
    ax.set_visible(False)

handles = [plt.Line2D([], [], marker="o", ls="", color="0.6", ms=11, label="point area = mean share of $T^2$")]
if RANKING:
    handles.append(plt.Line2D([], [], ls="", label="(n) = people with it in their top 3"))
# one row of 4 entries runs off a 6.5in single panel, so the legend wraps when it is narrow
ncol_leg = 2 if (single and len(handles) > 3) else len(handles)
fig.legend(handles=handles, loc="lower center", ncol=ncol_leg, frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.004))
fig.tight_layout(rect=(0, 0.08 if ncol_leg < len(handles) else 0.04, 1, 1))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200)
print(f"{len(S)} groups over {len(present)} encoders -> {OUT}")
for enc, clear, total in placement:
    flag = "" if clear == total else f"   <-- {total - clear} could not be placed clear"
    print(f"  labels placed clear of one another: {clear:3d} of {total:3d}  {enc}{flag}")
miss = [(e, int(c)) for e, c in zip(S.encoder, S.component) if (e, int(c)) not in LAB]
print(f"labels resolved for {len(S) - len(miss)} of {len(S)}" + (f"; missing {miss[:5]}" if miss else ""))
