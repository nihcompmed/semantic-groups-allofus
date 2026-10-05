#!/usr/bin/env python
"""fig_variance_explained.py -- eFigure 1, the choice of k: cumulative variance explained against k for every
grouping, with Horn's k marked by a star. For the supplement.

LOCAL. Draws from saved files only, so it runs before and after the Workbench run without change.

THE PANELS (a shared y axis, 0 to 100%, and a shared x axis, k = 1 to 100)
  a  the 6 sentence encoders, one line each: the share of the 151 item embeddings' variance carried
     by the first k principal components (1_select_k.py). 151 items in 768 or 1024 dimensions, so
     the 151 eigenvalues hold all the variance.
  b  TF-IDF (SUBTLEX-US weights, stop words kept), the same on the 151 TF-IDF vectors
     (tfidf_horn.py). Star = its own Horn's k. Circle = its second k, 25, referenced to the
     encoders' Horn's k (23 to 27): TF-IDF, like the encoders, comes from the item text alone.
  c  PCA of the responses: correlation eigenvalues of the 151 standardized responses over the cohort,
     cumulative over their total, 151 (k_choice.py on the Workbench).
  d  the factor model at the same Horn's k: the COMMON variance its k factors carry, the sum of the
     communalities over 151, from 1 fit per k on a grid (k_choice.py). It sits below PCA by
     construction, because a factor model sets each item's unique variance aside.
  e  MCA of the responses (k_choice.py): the RAW share of the total inertia, (J - Q) / Q. RAW ONLY,
     for 2 reasons. It is the same quantity as PCA's (each eigenvalue's share of the total), and
     Horn's test is run on it. And it is the established use of MCA on these
     surveys: Biji et al. 2026 (Am J Hum Genet 113:1434-1447; ade4, default parameters) report the
     raw share as "variance explained", axis 1 10.5% and their 20 axes about 29.9%. The
     Benzecri-adjusted share stays in k_choice_mca.csv and is not drawn. Star = MCA's own Horn's k.
     Circle = its second k, 25, referenced to PCA's Horn's k: MCA, like PCA and the factor model,
     is fitted on the responses.
  star   Horn's k: the leading components whose eigenvalue clears the permutation null's 95th
         percentile, cut at the first that does not.

INPUTS   ../outputs/k_selection/<model>/parallel_analysis.json        (encoders, tfidf_subtlex)
         <wb>/k_choice/k_choice_{pca,factor,mca}.csv, k_choice_summary.json   (k_choice.py)
         <wb> = --wb <dir>, default the newest ../outputs/workbench/<CDR>/ holding k_choice/.
         Without k_choice/, PCA is drawn from <wb>/comparator_summary.json if there is one (the same
         eigenvalues and the same parallel analysis), and the rest is left empty.
OUTPUTS  ../outputs/k_selection/fig_variance_explained.png
         ../outputs/k_selection/variance_explained_summary.csv   grouping, Horn's k, cumulative % at
                                                                 Horn's k and at the second k

Run:  python fig_variance_explained.py
      python fig_variance_explained.py --wb ../outputs/workbench/<CDR>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KSEL = os.path.join(HERE, "..", "outputs", "k_selection")
WB_ROOT = os.path.join(HERE, "..", "outputs", "workbench")
OUT_PNG = os.path.join(KSEL, "fig_variance_explained.png")
OUT_CSV = os.path.join(KSEL, "variance_explained_summary.csv")

ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5",
            "all-mpnet-base-v2", "e5-large-v2", "qwen3-embedding-0.6b"]
# Okabe-Ito, minus the yellow, which is unreadable as a line on white. 1 color per encoder.
ENC_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
INK, MUTED = "#1a1a1a", "#8c8c8c"
SECOND_K = 25          # TF-IDF: the encoders' Horn's k; MCA: PCA's Horn's k
X_MAX = 100


def wb_dir():
    if "--wb" in sys.argv:
        return sys.argv[sys.argv.index("--wb") + 1]
    for probe in ("k_choice/k_choice_summary.json", "comparator_summary.json"):
        hits = sorted(glob.glob(os.path.join(WB_ROOT, "*", probe)), key=os.path.getmtime)
        if hits:
            return os.path.dirname(hits[-1]).replace(os.sep + "k_choice", "")
    return None


def first_crossing(real, null):
    """Horn's k: the leading components that clear the null before the first that does not."""
    below = np.where(np.asarray(real) <= np.asarray(null))[0]
    return int(below[0]) if below.size else len(real)


def curve(real, null, stored_k=None):
    real, null = np.asarray(real, float), np.asarray(null, float)
    k = first_crossing(real, null)
    assert stored_k is None or k == int(stored_k), f"stored Horn's k {stored_k} is not the first crossing ({k})"
    return {"k": np.arange(1, len(real) + 1), "cum": 100 * np.cumsum(real) / real.sum(), "horn": k}


def at(c, k):
    """Cumulative % at k, or NaN where the curve does not reach k."""
    hit = np.where(np.asarray(c["k"]) == k)[0]
    return float(c["cum"][hit[0]]) if hit.size else np.nan


def load_workbench(wb):
    pca = factor = mca = None
    kc = os.path.join(wb, "k_choice") if wb else None
    if kc and os.path.exists(os.path.join(kc, "k_choice_summary.json")):
        S = json.load(open(os.path.join(kc, "k_choice_summary.json")))
        P = pd.read_csv(os.path.join(kc, "k_choice_pca.csv")).sort_values("component")
        pca = curve(P["eigenvalue"], P["null_95th"], S["pca"]["horn_k"])
        pca["source"] = "k_choice.py"
        F = pd.read_csv(os.path.join(kc, "k_choice_factor.csv")).sort_values("k")
        factor = {"k": F["k"].values, "cum": F["common_cum_pct"].values, "horn": S["factor"]["horn_k"],
                  "source": "k_choice.py"}
        M = pd.read_csv(os.path.join(kc, "k_choice_mca.csv")).sort_values("axis")
        horn = first_crossing(M["eigenvalue"].values, M["null_95th"].values)
        assert horn == S["mca"]["horn_k"], "k_choice_mca.csv and k_choice_summary.json disagree on Horn's k"
        mca = {"k": M["axis"].values, "cum": M["cum_pct_inertia"].values, "horn": horn,
               "source": "k_choice.py"}
    elif wb and os.path.exists(os.path.join(wb, "comparator_summary.json")):
        pa = json.load(open(os.path.join(wb, "comparator_summary.json")))["parallel_analysis"]
        pca = curve(pa["eigenvalues"], pa["null_pctile"], pa["k"])
        pca["source"] = "comparator_summary.json"
    return pca, factor, mca


def main():
    wb = wb_dir()
    print(f"Workbench aggregates: {wb or 'none found'}")
    enc = {}
    for e in ENCODERS:
        d = json.load(open(os.path.join(KSEL, e, "parallel_analysis.json")))
        enc[e] = curve(d["real_eigs"], d["null_percentile"], d["k"])
    d = json.load(open(os.path.join(KSEL, "tfidf_subtlex", "parallel_analysis.json")))
    tfidf = curve(d["real_eigs"], d["null_percentile"], d["k"])
    pca, factor, mca = load_workbench(wb)

    rows = []
    for e in ENCODERS:
        rows.append({"grouping": e, "fitted_on": "item text", "horn_k": enc[e]["horn"],
                     "cum_pct_at_horn_k": at(enc[e], enc[e]["horn"])})
    rows.append({"grouping": "TF-IDF (SUBTLEX-US)", "fitted_on": "item text", "horn_k": tfidf["horn"],
                 "cum_pct_at_horn_k": at(tfidf, tfidf["horn"]), "second_k": SECOND_K,
                 "cum_pct_at_second_k": at(tfidf, SECOND_K)})
    for name, c in (("PCA", pca), ("factor model (common variance)", factor), ("MCA (raw inertia)", mca)):
        if c is None:
            rows.append({"grouping": name, "fitted_on": "responses", "source": "pending Workbench run"})
            continue
        r = {"grouping": name, "fitted_on": "responses", "horn_k": c["horn"],
             "cum_pct_at_horn_k": at(c, c["horn"]), "source": c.get("source")}
        if name.startswith("MCA"):
            r.update({"second_k": SECOND_K, "cum_pct_at_second_k": at(c, SECOND_K)})
        rows.append(r)
    S = pd.DataFrame(rows).round(1)
    S.to_csv(OUT_CSV, index=False)
    print(S.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.6), sharex=True, sharey=True, constrained_layout=True)
    ax_enc, ax_tf, ax_pca, ax_fa, ax_mca, ax_leg = axes.ravel()

    def style(ax, title):
        ax.set_title(title, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, X_MAX + 1)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="0.92", lw=0.6)
        ax.set_axisbelow(True)

    def pending(ax):
        ax.text(0.5, 0.5, "pending\nWorkbench run", transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color=MUTED)

    def line(ax, c, color=INK, ls="-", lw=1.2):
        n = np.asarray(c["k"]) <= X_MAX
        ax.plot(np.asarray(c["k"])[n], np.asarray(c["cum"])[n], color=color, lw=lw, ls=ls)

    def star(ax, c, color=INK, kk=None):
        kk = c["horn"] if kk is None else kk
        ax.plot(kk, at(c, kk), marker="*", ms=11, color=color, mec="white", mew=0.7, zorder=5)

    def circle(ax, c, color=INK, kk=SECOND_K):
        ax.plot(kk, at(c, kk), marker="o", ms=5.5, mfc="white", mec=color, mew=1.3, zorder=5)

    def note(ax, text, y=0.05, va="bottom"):
        ax.text(0.97, y, text, transform=ax.transAxes, ha="right", va=va, fontsize=7, color=INK,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

    style(ax_enc, "A  Sentence encoders (item text)")
    for e, col in zip(ENCODERS, ENC_COLORS):
        line(ax_enc, enc[e], col)
        star(ax_enc, enc[e], col)
    hk = [enc[e]["horn"] for e in ENCODERS]
    hv = [at(enc[e], enc[e]["horn"]) for e in ENCODERS]
    note(ax_enc, f"Horn's k {min(hk)} to {max(hk)}\n{min(hv):.0f}% to {max(hv):.0f}%")

    style(ax_tf, "B  TF-IDF (item text)")
    line(ax_tf, tfidf)
    star(ax_tf, tfidf)
    circle(ax_tf, tfidf)
    note(ax_tf, f"Horn's k = {tfidf['horn']}: {at(tfidf, tfidf['horn']):.0f}%\n"
                f"k = {SECOND_K}: {at(tfidf, SECOND_K):.0f}%")

    style(ax_pca, "C  PCA (responses)")
    if pca:
        line(ax_pca, pca)
        star(ax_pca, pca)
        note(ax_pca, f"Horn's k = {pca['horn']}: {at(pca, pca['horn']):.0f}%")
    else:
        pending(ax_pca)

    style(ax_fa, "D  Factor model (responses)")
    if factor:
        line(ax_fa, factor)
        star(ax_fa, factor)
        note(ax_fa, f"Horn's k = {factor['horn']}: {at(factor, factor['horn']):.0f}%\n(common variance)")
    else:
        pending(ax_fa)

    style(ax_mca, "E  MCA (responses)")
    if mca:
        line(ax_mca, mca)
        star(ax_mca, mca)
        circle(ax_mca, mca)
        note(ax_mca, f"Horn's k = {mca['horn']}: {at(mca, mca['horn']):.0f}%\n"
                     f"k = {SECOND_K}: {at(mca, SECOND_K):.0f}%")
    else:
        pending(ax_mca)

    for ax in (ax_fa, ax_mca, ax_pca):      # panel c sits above the legend cell, so it needs its own
        ax.set_xlabel("Number of components, k", fontsize=8)
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative variance explained (%)", fontsize=8)

    # the 6th cell holds the legend, outside every plotting area
    ax_leg.axis("off")
    handles = [Line2D([], [], color=c, lw=1.2, label=e) for e, c in zip(ENCODERS, ENC_COLORS)]
    handles += [Line2D([], [], color=INK, marker="*", ms=10, mec="white", lw=0, label="Horn's k"),
                Line2D([], [], color=INK, marker="o", mfc="white", ms=5.5, lw=0, label=f"second k = {SECOND_K}")]
    ax_leg.legend(handles=handles, loc="center", frameon=False, fontsize=7)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PNG[:-4] + ".pdf", bbox_inches="tight")   # vector copy for the supplement
    print(f"-> {OUT_PNG}\n-> {OUT_CSV}")


if __name__ == "__main__":
    main()
