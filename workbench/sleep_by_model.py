#!/usr/bin/env python3
"""sleep_by_model.py -- the wearable sleep gradient, run for EVERY grouping instead of 1.

WHAT THIS IS AND IS NOT. It is NOT an outlier comparison. The sleep result is a gradient across the
whole cohort, run in explore_other_data.py cells 9 to 12 on the prespecified encoder's own
percentile. This runs the same analysis, unchanged, for all 6 encoders and for the 4 comparator
groupings (scale, factor, pca and MCA), so the gradient can be read as a property of the scoring
rather than of the encoder that happened to be chosen.

THE SAMPLE is everyone with usable device nights, about 16,700 of the 90,351, NOT a selected subset.
Deciles are taken over the WHOLE COHORT from each grouping's own score and then attached to the sleep
subsample, which is how the reported decile figures were built.

THE ESTIMATOR IS COPIED FROM `explore_other_data.py` `_burden_linear`, not reimplemented, so every
number here is comparable with the ones already reported:
  * burden enters as the RANK PERCENTILE OF ITS DECILE, standardized to unit variance
  * age enters the same way, as a standardized rank percentile, WITH ITS SQUARE
  * log(nights) is adjusted, because a sample SD is biased downward at small n and the burdened have
    fewer nights
  * kind="median" is a median regression and is PRIMARY for variability, because an SD is itself
    right skewed. kind="mean" is least squares and is primary for duration.
  * ★ a median regression stops at an iteration limit and returns the last iterate with no error, so
    the warning is caught and `converged` is reported. A number that did not converge is reported as
    one rather than quoted.

★ THE COVERAGE GRADIENT IS PART OF THE RESULT. Device coverage changes with burden (it rises from
the bottom decile to the top), and whether it does so under every grouping or only under some is
exactly what this script can say. The share of each decile with usable nights is computed for every
grouping and is the third panel, not a footnote.
If coverage changes faster under 1 grouping than another, its sleep gradient is measured on a
differently selected subsample and the 2 are not directly comparable.

★ WHAT THIS CANNOT SETTLE. Owning and wearing a device is associated with burden, so the sleep sample
is conditioned on something the exposure affects, under every grouping alike. The comparison between
groupings is fair because they are all conditioned the same way. No causal reading survives it.

★ THE SCALE COMPARATOR has 34 units: 16 instrument totals, 5 BFI-2-XS trait scores and 13 standalone
questions, so every instrument is totaled (the WMH-CIDI and the MHQ included).

INPUTS   sleep_person_local.csv (explore_other_data.py cell 9, beside the scripts or in screen_out/)
         wear_enrollees_local.csv (cell 10, optional: without it the wear-study split is skipped)
         screen_out/scores_<encoder>.csv, scores_{scale,factor,pca,mca}.csv (mca90 with MCA_MODEL=mca90)
         the responses file, for age and sex
OUTPUTS  (suffix _<MCA_MODEL> when MCA_MODEL is not mca)
         screen_out/sleep_by_model.csv          grouping x outcome x kind: n, center, unadjusted,
                                                age-adjusted, interval, converged (LEAVES)
         screen_out/sleep_by_model_deciles.csv  grouping x decile: coverage, duration, variability
         screen_out/fig_sleep_by_model.png      3 panels, pooled (LEAVES, aggregates only)
         screen_out/fig_sleep_by_model_strata.png  the same 3, wear study over personal device

Run:  python3 sleep_by_model.py
      python3 sleep_by_model.py --min-nights 14   (the default, the manuscript's rule)
      python3 sleep_by_model.py --no-fig
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_config import CFG   # noqa: E402

ARG = lambda f, d: (type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d)
OUT_DIR = CFG.out_dir
ID_COL = "id"
ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5",
            "all-mpnet-base-v2", "e5-large-v2", "qwen3-embedding-0.6b"]
EXEMPLAR = "gte-large-en-v1.5"
# ★ MCA_MODEL picks WHICH mca run: "mca" is the primary, at the k matched
# to factor and pca; "mca90" is the sensitivity at MCA's own Horn k, written by
# `comparators.py --mca-k 90`. One or the other, never both -- 2 depths of the same
# representation in 1 comparison would split each other's exclusive picks.
MCA_MODEL = os.environ.get("MCA_MODEL", "mca")
COMPARATORS = ["scale", "factor", "pca", MCA_MODEL]

# ★ A tagged run must not be able to touch the primary's outputs. When MCA_MODEL
# is anything but "mca", every aggregate this script writes gains that suffix, so the sensitivity's
# tables and figures sit beside the primary's instead of on top of them. Per-model files already carry
# the model in their name and are left alone.
TAG = "" if MCA_MODEL == "mca" else f"_{MCA_MODEL}"


def tagged(name):
    stem, dot, ext = name.rpartition(".")
    return f"{stem}{TAG}{dot}{ext}"

MIN_NIGHTS = ARG("--min-nights", 14)
NO_FIG = "--no-fig" in sys.argv
ENC_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
CMP_STYLE = {"scale": "-", "factor": (0, (5, 2)), "pca": (0, (1.5, 1.5)),
             "mca": (0, (4, 1, 1, 1)), "mca90": (0, (4, 1, 1, 1))}   # mca90 shares mca's style, the runs never coexist
CMP_LAB = {"scale": "published scoring units", "factor": "factor model",
           "pca": "principal components", "mca": "multiple correspondence analysis",
           "mca90": "multiple correspondence analysis (Horn k)"}
# duration is a mean, variability is a median. Stated here rather than chosen per run.
PRIMARY = {"mean_min": "mean", "sd_min": "median"}
LABELS = {"mean_min": "Mean nightly sleep (min)", "sd_min": "Within-person SD of sleep (min)"}


def deciles(M):
    """explore_other_data.py's rule: rank first, so deciles are equal-sized whatever ties there are."""
    return (M.rank(method="first").sub(1) * 10 // len(M) + 1).astype(int)


def burden_linear(y, decile, age, kind, extra=None):
    """Copied from explore_other_data.py `_burden_linear`. Do not 'improve' it: its whole value is
    that these numbers sit beside the reported ones on the same scale."""
    try:
        import statsmodels.api as sm
    except Exception as e:
        print(f"    [not run ({type(e).__name__}: {e})]")
        return None
    T = pd.DataFrame({"y": pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce"),
                      "decile": pd.to_numeric(pd.Series(np.asarray(decile)), errors="coerce"),
                      "age": pd.to_numeric(pd.Series(np.asarray(age)), errors="coerce")})
    E = None
    if extra is not None and len(extra.columns):
        E = pd.DataFrame(np.asarray(extra, dtype=float), columns=list(extra.columns))
        E = E.loc[:, E.std(axis=0).fillna(0) > 0]
        T = pd.concat([T.reset_index(drop=True), E.reset_index(drop=True)], axis=1)
    keep = T.y.notna() & T.decile.notna() & T.age.notna()
    if E is not None and len(E.columns):
        keep &= T[list(E.columns)].notna().all(axis=1)
    T = T[keep]
    if len(T) < 100 or T.y.nunique() < 3:
        return None
    b = pd.Series(T.decile.astype(float)).rank(pct=True)
    b = ((b - b.mean()) / b.std()).values
    a = pd.Series(T.age.astype(float)).rank(pct=True)
    a = ((a - a.mean()) / a.std()).values
    yy = T.y.astype(float).values
    Z = T[list(E.columns)].to_numpy(dtype=float) if (E is not None and len(E.columns)) else None
    import warnings as _w
    converged = True
    X1 = sm.add_constant(b.reshape(-1, 1))
    X2 = sm.add_constant(np.column_stack([b, a, a ** 2] + ([Z] if Z is not None else [])))
    try:
        if kind == "median":
            with _w.catch_warnings(record=True) as caught:
                _w.simplefilter("always")
                m1 = sm.QuantReg(yy, X1).fit(q=0.5, max_iter=20000)
                m2 = sm.QuantReg(yy, X2).fit(q=0.5, max_iter=20000)
            converged = not any("iteration" in str(c.message).lower() for c in caught)
        else:
            m1, m2 = sm.OLS(yy, X1).fit(), sm.OLS(yy, X2).fit()
    except Exception as e:
        print(f"    [not run ({type(e).__name__}: {e})]")
        return None
    c1, c2, se2 = float(m1.params[1]), float(m2.params[1]), float(m2.bse[1])
    center = float(np.median(yy)) if kind == "median" else float(np.mean(yy))
    return {"kind": kind, "n": int(len(T)), "center": center, "unadjusted": c1,
            "age_adjusted": c2, "se": se2, "lo": c2 - 1.96 * se2, "hi": c2 + 1.96 * se2,
            "converged": converged}


def find_file(name, required=True, hint=""):
    for p in (os.path.join(HERE, name), os.path.join(OUT_DIR, name), name):
        if os.path.exists(p):
            return p
    if required:
        raise SystemExit(f"{name} not found. {hint}")
    return None


def main():
    CFG.describe("sleep_by_model")
    S = pd.read_csv(find_file("sleep_person_local.csv", True,
                              "Run cell 9 of explore_other_data.py first.")
                    ).rename(columns={"person_id": ID_COL})
    S[ID_COL] = S[ID_COL].astype(str)
    S = S.set_index(ID_COL)
    S = S[(S["nights"] >= MIN_NIGHTS) & S["sd_min"].notna()].copy()
    S["log_nights"] = np.log(S["nights"].clip(lower=1))
    R = pd.read_csv(CFG.responses_csv, usecols=[ID_COL, "age", "sex"])
    R[ID_COL] = R[ID_COL].astype(str)
    dem = R.set_index(ID_COL)
    N = len(dem)
    S = S.join(dem, how="inner")
    # ★ THE WEAR STUDY IS A DIFFERENT SAMPLE AND HAS TO BE SPLIT OUT. The program equipped people for
    # a fixed study period, so their nights are set by the design rather than by their own behavior,
    # and the program supplied devices MORE OFTEN at higher burden -- which is what makes the coverage
    # gradient rise. Pooling the 2 mixes a behavioral quantity with an administrative one.
    wf = find_file("wear_enrollees_local.csv", False)
    if wf is None:
        print("  [wear_enrollees_local.csv not found: the wear-study split is skipped, pooled only]")
        S["stratum_wear"] = False
        strata = [("pooled", None)]
    else:
        roster = pd.read_csv(wf)["person_id"].astype(str)
        S["stratum_wear"] = S.index.isin(set(roster))
        strata = [("pooled", None), ("wear study", True), ("personal device", False)]
    print(f"[sleep_by_model] cohort {N:,} | with >= {MIN_NIGHTS} usable nights {len(S):,} "
          f"({100 * len(S) / N:.1f}%) | median nights {S['nights'].median():.0f}")
    if wf is not None:
        w, o = int(S.stratum_wear.sum()), int((~S.stratum_wear).sum())
        print(f"  wear study {w:,} (median nights {S.loc[S.stratum_wear, 'nights'].median():.0f}) | "
              f"personal device {o:,} (median nights {S.loc[~S.stratum_wear, 'nights'].median():.0f})")

    rows, drows = [], []
    for m in ENCODERS + COMPARATORS:
        f = os.path.join(OUT_DIR, f"scores_{m}.csv")
        if not os.path.exists(f):
            print(f"  [skip] scores_{m}.csv not found")
            continue
        Sc = pd.read_csv(f)
        Sc[ID_COL] = Sc[ID_COL].astype(str)
        by = next(c for c in ("pctile", "distance2", "distance") if c in Sc.columns)
        sc = Sc.set_index(ID_COL)[by].reindex(dem.index)          # cohort-wide, as reported
        dec = deciles(sc)
        sub = S.copy()
        sub["decile"] = dec.reindex(sub.index).values
        sub = sub[sub["decile"].notna()]

        cov = pd.DataFrame({"decile": dec})
        cov["has_sleep"] = cov.index.isin(S.index)
        for d, g in cov.groupby("decile"):
            for sname, flag in strata:
                sd_ = sub[sub.decile == d] if flag is None else \
                    sub[(sub.decile == d) & (sub.stratum_wear == flag)]
                drows.append({"grouping": m, "kind": "encoder" if m in ENCODERS else "comparator",
                              "stratum": sname, "decile": int(d), "n_decile": len(g),
                              "n_sleep": len(sd_),
                              "pct_with_sleep": 100.0 * float(g.has_sleep.mean()) if flag is None
                              else 100.0 * len(sd_) / len(g) if len(g) else np.nan,
                              "median_nights": float(sd_["nights"].median()) if len(sd_) else np.nan,
                              "mean_min": float(sd_["mean_min"].mean()) if len(sd_) else np.nan,
                              "sd_min": float(sd_["sd_min"].median()) if len(sd_) else np.nan})

        for sname, flag in strata:
            ss = sub if flag is None else sub[sub.stratum_wear == flag]
            line = [f"  {m:24s} {sname:16s} n {len(ss):6,d}"]
            for o in ("mean_min", "sd_min"):
                for kind in ("mean", "median"):
                    r = burden_linear(ss[o].values, ss["decile"].values, ss["age"].values, kind,
                                      ss[["log_nights"]])
                    if r is None:
                        continue
                    r.update({"grouping": m, "stratum": sname, "outcome": o,
                              "kind_is_primary": kind == PRIMARY[o],
                              "model": "age, age^2, log nights"})
                    rows.append(r)
                    if kind == PRIMARY[o]:
                        line.append(f"{o} {r['age_adjusted']:+.2f} "
                                    f"[{r['lo']:+.2f}, {r['hi']:+.2f}]"
                                    f"{'' if r['converged'] else ' NOT CONVERGED'}")
            print(" | ".join(line))

    # ---- the dissemination policy (disclosure.py), at write time. In the decile file,
    # n_sleep partitions each grouping-and-stratum's sleep sample across the deciles, and a share with
    # sleep gives n_sleep back from n_decile, so both are checked. A model row resting on 1 to 20 people
    # is blanked. Rows are kept.
    from disclosure import MARK, SUPP, pair_hidden, partition_mask, small
    D = pd.DataFrame(rows)
    if "n" in D.columns:
        tiny = D["n"].apply(lambda v: small(v) if pd.notna(v) else False)
        D.loc[tiny, [c for c in D.columns if c not in ("grouping", "stratum", "outcome", "model", "n")
                     and pd.api.types.is_numeric_dtype(D[c])]] = np.nan
        D["n"] = D["n"].astype(object); D.loc[tiny, "n"] = MARK
    D.to_csv(os.path.join(OUT_DIR, tagged("sleep_by_model.csv")), index=False)
    DEC = pd.DataFrame(drows)
    DEC["n_sleep"] = DEC["n_sleep"].astype(object)
    stat = ["pct_with_sleep", "median_nights", "mean_min", "sd_min"]
    n_hid = 0
    for _, ix in DEC.groupby(["grouping", "stratum"]).groups.items():
        ix = list(ix)
        ns = [int(v) for v in DEC.loc[ix, "n_sleep"]]
        hide, prim = partition_mask(ns)
        for i, n_s, h, p_ in zip(ix, ns, hide, prim):
            if p_ or pair_hidden(n_s, int(DEC.at[i, "n_decile"])):
                DEC.loc[i, stat] = np.nan
                DEC.at[i, "n_sleep"] = MARK if small(n_s) else SUPP
                n_hid += 1
            elif h:
                DEC.at[i, "n_sleep"] = SUPP
                DEC.at[i, "pct_with_sleep"] = np.nan
                n_hid += 1
    DEC.to_csv(os.path.join(OUT_DIR, tagged("sleep_by_model_deciles.csv")), index=False)
    print(f"  dissemination policy: {n_hid} decile rows suppressed")
    print(f"  -> sleep_by_model.csv ({len(D)} rows), sleep_by_model_deciles.csv ({len(DEC)} rows)")
    if NO_FIG or not len(DEC):
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PANELS = [("mean_min", LABELS["mean_min"]), ("sd_min", LABELS["sd_min"]),
              ("pct_with_sleep", "Share of decile with\nusable device nights (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6), constrained_layout=True)
    P0 = DEC[DEC.stratum == "pooled"]
    for ax, (col, lab) in zip(axes, PANELS):
        for e, c in zip(ENCODERS, ENC_COLORS):
            d = P0[P0.grouping == e].sort_values("decile")
            if len(d):
                ax.plot(d.decile, d[col], color=c, lw=1.3, marker="o", ms=2.5, label=e)
        for g in COMPARATORS:
            d = P0[P0.grouping == g].sort_values("decile")
            if len(d):
                ax.plot(d.decile, d[col], color="black", ls=CMP_STYLE[g], lw=1.5, label=CMP_LAB[g])
        ax.set_xlabel("Decile of the score", fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        ax.set_xticks(range(1, 11))
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.34))
    out = os.path.join(OUT_DIR, tagged("fig_sleep_by_model.png"))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  -> {out}")

    # ---- both strata ON THE SAME AXES ------------------------------------------------------------
    # Stratum is the salient dimension here, not grouping: the groupings gave the same gradient, so
    # each stratum is drawn as a band from the lowest to the highest of the groupings present (10 with
    # MCA) at each decile, with their median as a line.
    # The band width is the disagreement BETWEEN groupings and the gap between bands is the difference
    # between wear study and personal device. A separate line per grouping and stratum would hide both.
    two = [s_ for s_ in ("wear study", "personal device") if (DEC.stratum == s_).any()]
    if len(two) < 2:
        return
    SCOL = {"wear study": "#D55E00", "personal device": "#0072B2"}
    fig2, ax2 = plt.subplots(1, 3, figsize=(6.5, 2.6), constrained_layout=True)
    for ax, (col, lab) in zip(ax2, PANELS):
        for sname in two:
            W = (DEC[DEC.stratum == sname].pivot_table(index="decile", columns="grouping",
                                                       values=col).sort_index())
            if not len(W):
                continue
            ax.fill_between(W.index, W.min(axis=1), W.max(axis=1), color=SCOL[sname],
                            alpha=0.22, lw=0)
            ax.plot(W.index, W.median(axis=1), color=SCOL[sname], lw=1.8, marker="o", ms=3,
                    label=f"{sname} (band = the {W.shape[1]} groupings)")
        P = DEC[DEC.stratum == "pooled"].pivot_table(index="decile", columns="grouping",
                                                     values=col).sort_index()
        if len(P):
            ax.plot(P.index, P.median(axis=1), color="0.35", lw=1.0, ls=(0, (4, 3)), label="pooled")
        ax.set_xlabel("Decile of the score", fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        ax.set_xticks(range(1, 11))
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    h2, l2 = ax2[0].get_legend_handles_labels()
    fig2.legend(h2, l2, loc="lower center", ncol=3, frameon=False, fontsize=8,
                bbox_to_anchor=(0.5, -0.16))
    out2 = os.path.join(OUT_DIR, tagged("fig_sleep_by_model_strata.png"))
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"  -> {out2}")


if __name__ == "__main__":
    main()
