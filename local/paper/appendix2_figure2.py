#!/usr/bin/env python
"""appendix2_figure2.py -- Figure S2, what the sparsity-penalized projection does to each group's item weights,
against the dense principal components at the same number of groups.

FOR EACH ENCODER (k = Horn's number of groups, as in the basis package):
  groups of record   P = (E - mu) W^T, the item weights of the basis package (outputs/basis/<enc>.npz, the retained
                     fit), E = the 151 item vectors (admin151_items, the text of record)
  dense components   Q = (E - mean) V_k^T, the top-k right singular vectors of the centered item vectors (the exact
                     principal components; the "svd" reference row of the selection)
  per column         core = the fixed-point rule of the semantic groups (semantic_groups/step8_cores.py); core size =
                     its number of items; core energy = the core's share of the column's squared weights. A dense
                     component's core is found by the same rule on its own column; the 2 solutions' columns are not
                     paired.

CHECKS (the script stops if any fails): the groups' core sizes equal outputs/cores/<enc>_components.csv; the median
core energy of the groups equals the retained fit's coreE and that of the dense components the "svd" row's coreE in
outputs/random_starts/ (selection.json, candidates.csv), to 1e-9.

Writes  output/figures/sparsification.{png,pdf}
        outputs/sparsification/sparsification_points.csv   encoder, solution, component, core_size, core_energy
        outputs/sparsification/sparsification_summary.csv  per encoder x solution: k, medians, ranges
Run     python appendix2_figure2.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from common import ENCODERS, FIGURES, LOCAL, OUTPUTS                     # noqa: E402

sys.path.insert(0, os.path.join(LOCAL, "semantic_groups"))
from step8_cores import fixed_point_core                                 # noqa: E402  the paper's core rule

OUT = os.path.join(OUTPUTS, "sparsification")
PREFIX = "admin151_items"
GROUPS, DENSE = "semantic groups", "dense principal components"
COLOR = {GROUPS: "#D55E00", DENSE: "#0072B2"}                          # Okabe-Ito vermillion and blue
OFFSET = {GROUPS: -0.14, DENSE: 0.14}                                    # side by side at each core size
INK, MUTED, GRID = "#222222", "#555555", "#DDDDDD"


def columns(M):
    """(core size, core energy) of every column of M."""
    size, energy = [], []
    for j in range(M.shape[1]):
        v = M[:, j].astype(float)
        c = fixed_point_core(v)
        sq = v ** 2
        size.append(len(c))
        energy.append(float(sq[c].sum() / sq.sum()))
    return np.array(size), np.array(energy)


def points():
    sel = json.load(open(os.path.join(OUTPUTS, "random_starts", "selection.json")))
    cand = pd.read_csv(os.path.join(OUTPUTS, "random_starts", "candidates.csv"), dtype={"seed": str})
    rows = []
    for enc in ENCODERS:
        b = np.load(os.path.join(OUTPUTS, "basis", f"{enc}.npz"), allow_pickle=True)
        e = np.load(os.path.join(OUTPUTS, "basis", "item_embeddings", f"{PREFIX}_{enc}.npz"), allow_pickle=True)
        E = np.asarray(e["vectors"], float)
        W = np.asarray(b["W"], float)
        k = W.shape[0]
        P = (E - b["mu"]) @ W.T
        Xc = E - E.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        Q = Xc @ Vt[:k].T
        gs, ge = columns(P)
        ds, de = columns(Q)
        # checks
        comp = pd.read_csv(os.path.join(OUTPUTS, "cores", f"{enc}_components.csv")).sort_values("component")
        assert (comp["size"].values == gs).all(), f"{enc}: core sizes differ from outputs/cores"
        assert abs(np.median(ge) - sel[enc]["selected"]["coreE"]) < 1e-9, f"{enc}: median core energy != selection.json"
        svd = cand[(cand.encoder == enc) & (cand.seed == "svd") & (cand.k == k)]
        assert len(svd) == 1 and abs(np.median(de) - float(svd.coreE.iloc[0])) < 1e-9, f"{enc}: dense != candidates.csv svd"
        for sol, s, en in [(GROUPS, gs, ge), (DENSE, ds, de)]:
            rows += [{"encoder": enc, "solution": sol, "component": j, "core_size": int(s[j]),
                      "core_energy": round(float(en[j]), 6)} for j in range(k)]
    return pd.DataFrame(rows)


def summary(D):
    g = D.groupby(["encoder", "solution"], sort=False)
    return g.agg(k=("component", "size"), median_core_size=("core_size", "median"), min_core_size=("core_size", "min"),
                 max_core_size=("core_size", "max"), median_core_energy=("core_energy", "median"),
                 min_core_energy=("core_energy", "min"), max_core_energy=("core_energy", "max")).reset_index()


def draw(D):
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.6), sharex=True, sharey=True)
    xmax = int(D.core_size.max()) + 1
    for i, (ax, enc) in enumerate(zip(axes.flat, ENCODERS)):
        for sol in [DENSE, GROUPS]:
            d = D[(D.encoder == enc) & (D.solution == sol)]
            x = d.core_size.values + OFFSET[sol]
            if sol == GROUPS:
                ax.scatter(x, d.core_energy, marker="x", s=22, linewidths=1.1, color=COLOR[sol], zorder=3)
            else:
                ax.scatter(x, d.core_energy, marker="o", s=20, facecolors="none", edgecolors=COLOR[sol],
                           linewidths=1.0, zorder=2)
        name = enc + (" (prespecified)" if i == 0 else "")
        ax.set_title(f"{'ABCDEF'[i]}  {name}", loc="left", fontsize=8, color=INK, pad=4)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_xticks(range(0, xmax + 1, 4))
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        if i % 3 == 0:
            ax.set_ylabel("Share of squared weights\nheld by the core items")
        if i >= 3:
            ax.set_xlabel("Core size (items)")
    handles = [Line2D([], [], marker="x", linestyle="none", markersize=5.5, markeredgewidth=1.1, color=COLOR[GROUPS],
                      label="Semantic groups (sparsity-penalized projection)"),
               Line2D([], [], marker="o", linestyle="none", markersize=5, markerfacecolor="none",
                      markeredgecolor=COLOR[DENSE], label="Dense principal components")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.2, w_pad=0.8)
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    D = points()
    D.to_csv(os.path.join(OUT, "sparsification_points.csv"), index=False)
    S = summary(D)
    S.to_csv(os.path.join(OUT, "sparsification_summary.csv"), index=False)
    print(S.to_string(index=False))
    fig = draw(D)
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, "sparsification.png"), dpi=300)
    fig.savefig(os.path.join(FIGURES, "sparsification.pdf"))
    print(f"checks passed; wrote -> {OUT} and {FIGURES}")


if __name__ == "__main__":
    main()
