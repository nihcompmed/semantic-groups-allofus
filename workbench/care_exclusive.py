#!/usr/bin/env python3
"""care_exclusive.py -- an exclusive-set comparison on the CARE outcomes instead of the criteria.

★ THE SETS HERE ARE NOT THOSE OF breadth_exclusive.py. That script defines the exclusive outliers
over 7 groupings (gte, PCA, factor, MCA at 25 and at its Horn's k, TF-IDF at 4 and 25), with no
instrument totals, each exclusive against all 6 others. This script builds its own groups of 5,
defined below.

WHY THIS AND NOT THE CRITERIA. The six criteria are the instruments' own thresholds, so a grouping that
keeps instrument boundaries is rewarded by them and a grouping that breaks those boundaries by design
is not. Breadth therefore DESCRIBES what each grouping selects and cannot say whether the people it
selects are in trouble. The care outcomes can, because nothing in the 151 items feeds them: every
barrier comes from the Health Care Access and Utilization survey, which contributes 0 of the 151, and
the ED share comes from records. The comparison runs across the whole cut range.

THE SETS. For each encoder the group is {that encoder, factor, pca, instrument totals, MCA}, a
quintuple. A person is EXCLUSIVE to 1 of the 5 when none of the other 4 in that group selected them.
Symmetric within each group, so no grouping is the reference, and 6 groups give the comparators 6
slightly different exclusive sets. The output column is named `quadruple_encoder`, although the
groups have 5 members.

★ EXCLUSIVITY IS DEFINED AGAINST THE OTHER MEMBERS, so a person exclusive among 4 need not be
exclusive among 5. Adding a member to the group is not an added row: EVERY NUMBER IN THIS TABLE
CHANGES.

THE OUTCOMES, care.py's, unchanged.
  any_access_barrier   no usual source of care, or any cost barrier, or any structural delay, or any
                       medication rationing. Survey answerers only.
  any_cost_barrier     survey answerers only.
  care_avoid_provider  care avoided because of a professional's race or religion, "some of the time"
                       or more. Survey answerers only.
  ed_share             ED visits (concepts 9203, 262) over all visits, among people with >= 1 visit.
                       A ratio, so free of the observation-window confound. EHR-linked only.

  ★ THE DENOMINATOR IS PER OUTCOME, NOT PER SET. A barrier rate is over the exclusive people who
  answered the access survey, the ED share over the exclusive people with a visit. Both ns are
  written out at every point.

  ★ RAW AND ADJUSTED, BOTH. The exclusive sets differ in composition -- at the 95th the encoder's
  people were about 5 years younger than the factor model's -- and access and cost barriers fall with
  age, so a raw gap is part selection and part composition. Every rate is therefore written twice:
  raw with a Wilson (or normal) interval, and age- and sex-adjusted. The adjusted figure comes from
  pooling the 5 exclusive sets and fitting outcome ~ set + age + sex, then marginally standardizing,
  which avoids extrapolating any set's model into an age range it does not occupy. Each comparator
  set also carries a contrast against the encoder set, an odds ratio for a binary outcome and a
  difference for the ED share, taken from the coefficient so it has an interval. The adjusted RATES
  have no interval, because the question they answer is whether an ordering survives adjustment.

  Verified on a fixture where the outcome depends only on age: a raw gap of +0.176
  between 2 groups differing by 13 years collapses to +0.013 adjusted, with the odds ratio bracketing
  1.0.

  These are still not interchangeable with care.py's estimates, which adjust for the full
  sociodemographic set on the whole cohort. Read the cohort line as a reference, not as a control.

  ★ CARE AVOIDED SHARES A CONSTRUCT WITH THE EXPOSURE. The battery carries 7 health care
  discrimination items while this outcome measures the same construct from a different survey with
  different items. It is reported because dropping it would hide the problem, not because it is
  clean.

★ THE SCALE COMPARATOR has 34 units: 16 instrument totals, 5 BFI-2-XS trait scores and 13 standalone
questions, so every instrument is totaled (the WMH-CIDI and the MHQ included).

INPUTS   screen_out/care_outcomes_per_person.csv   (from care.py)
         screen_out/scores_<encoder>.csv, scores_{scale,factor,pca,mca}.csv (mca90 with MCA_MODEL=mca90)
         the responses file, for age and sex only
OUTPUTS  (suffix _<MCA_MODEL> when it is not mca)
         screen_out/care_exclusive.csv             cut x group (column quadruple_encoder) x grouping x
                                                   outcome, raw and age- and sex-adjusted, with
                                                   contrasts, under the dissemination policy (LEAVES)
         screen_out/fig_care_exclusive.png         4 panels (LEAVES, aggregates only)

Run:  python3 care_exclusive.py
      python3 care_exclusive.py --from 50        (plot the whole range)
      python3 care_exclusive.py --no-fig
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_config import CFG   # noqa: E402

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

BINARY = ["any_access_barrier", "any_cost_barrier", "care_avoid_provider"]
OUTCOMES = BINARY + ["ed_share"]
LABELS = {"any_access_barrier": "Any access barrier", "any_cost_barrier": "Any cost barrier",
          "care_avoid_provider": "Care avoided (race or religion)", "ed_share": "ED share of visits"}
CUTS = [50, 60, 70, 75, 80, 85, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
PLOT_FROM = float(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 90.0
MIN_N = 30
NO_FIG = "--no-fig" in sys.argv
ENC_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
CMP_STYLE = {"scale": "-", "factor": (0, (5, 2)), "pca": (0, (1.5, 1.5)),
             "mca": (0, (4, 1, 1, 1)), "mca90": (0, (4, 1, 1, 1))}   # mca90 shares mca's style, the runs never coexist
CMP_LAB = {"scale": "published scoring units", "factor": "factor model",
           "pca": "principal components", "mca": "multiple correspondence analysis",
           "mca90": "multiple correspondence analysis (Horn k)"}


def irls(X, y, iters=60, tol=1e-9):
    """Plain logistic regression by IRLS. Returns (beta, standard errors), or None if it does not
    settle -- small exclusive sets can separate, and a silent garbage fit is worse than no fit."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        XtW = X.T * w
        H = XtW @ X
        try:
            step = np.linalg.solve(H, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return None
        b = b + step
        if np.max(np.abs(step)) < tol:
            cov = np.linalg.inv(H)
            if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) < 0):
                return None
            return b, np.sqrt(np.diag(cov))
    return None


def adjusted(sets_by_name, O, dem, outcome, enc_name):
    """Pool the exclusive sets of one group (5 sets), fit outcome ~ set + age + sex, and
    marginally standardize.

    The adjusted rate for a set is the model's prediction for EVERY pooled participant with the set
    indicator held at that set, averaged. The contrast reported is each comparator set against the
    encoder set, as an odds ratio for a binary outcome and as a difference for the ED share, taken
    from the coefficient so it carries an interval. No interval is given for the adjusted rates
    themselves -- marginal standardization needs a delta method or a bootstrap and neither is worth
    the cost here, where the question is whether an ORDERING survives adjustment.
    """
    names = [enc_name] + [c for c in COMPARATORS if c in sets_by_name]
    frames = []
    for i, nm in enumerate(names):
        idx = sets_by_name[nm]
        d = pd.DataFrame({"y": O[outcome].reindex(idx), "age": dem["age"].reindex(idx),
                          "sex": dem["sex"].reindex(idx)})
        d["set"] = nm
        frames.append(d.dropna())
    P = pd.concat(frames)
    if P["set"].nunique() < 2 or P.groupby("set").size().min() < MIN_N:
        return {}
    age = (P["age"].values - P["age"].mean()) / (P["age"].std(ddof=0) or 1.0)
    sexv = pd.factorize(P["sex"])[0].astype(float)
    D = [np.ones(len(P)), age, sexv]
    for nm in names[1:]:
        D.append((P["set"].values == nm).astype(float))
    X = np.column_stack(D)
    y = P["y"].values.astype(float)
    out = {}
    binary = outcome in BINARY
    if binary:
        if len(np.unique(y)) < 2:
            return {}
        fit = irls(X, y)
        if fit is None:
            return {}
        b, se = fit
        for j, nm in enumerate(names):
            Z = X.copy()
            Z[:, 3:] = 0.0
            if j > 0:
                Z[:, 2 + j] = 1.0
            out[f"{outcome}_adj__{nm}"] = float((1 / (1 + np.exp(-np.clip(Z @ b, -30, 30)))).mean())
        for j, nm in enumerate(names[1:], start=1):
            k = 2 + j
            out[f"{outcome}_or__{nm}"] = float(np.exp(b[k]))
            out[f"{outcome}_or_lo__{nm}"] = float(np.exp(b[k] - 1.96 * se[k]))
            out[f"{outcome}_or_hi__{nm}"] = float(np.exp(b[k] + 1.96 * se[k]))
    else:
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ b
        s2 = float(resid @ resid) / max(len(y) - X.shape[1], 1)
        try:
            cov = s2 * np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            return {}
        se = np.sqrt(np.diag(cov))
        for j, nm in enumerate(names):
            Z = X.copy()
            Z[:, 3:] = 0.0
            if j > 0:
                Z[:, 2 + j] = 1.0
            out[f"{outcome}_adj__{nm}"] = float((Z @ b).mean())
        for j, nm in enumerate(names[1:], start=1):
            k = 2 + j
            out[f"{outcome}_diff__{nm}"] = float(b[k])
            out[f"{outcome}_diff_lo__{nm}"] = float(b[k] - 1.96 * se[k])
            out[f"{outcome}_diff_hi__{nm}"] = float(b[k] + 1.96 * se[k])
    return out


def wilson(k, n):
    if not n:
        return (np.nan, np.nan)
    p, z = k / n, 1.959963985
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def xticks_for(lo):
    grid = [50, 60, 70, 80, 90, 99] if lo < 85 else [90, 92, 94, 95, 96, 97, 98, 99]
    return [t for t in grid if t >= lo]


def rates(idx, O, dem, label, cut, quad_enc, grouping):
    row = {"cut_pctile": cut, "quadruple_encoder": quad_enc, "grouping": grouping,
           "name": label, "n_exclusive": len(idx)}
    d = dem.reindex(idx)
    row["mean_age"] = float(d["age"].mean()) if len(d) else np.nan
    row["pct_female"] = float(100 * (d["sex"] == d["sex"].mode().iloc[0]).mean()) if len(d) else np.nan
    sub = O.reindex(idx)
    for o in OUTCOMES:
        v = sub[o].dropna()
        row[f"n_{o}"] = len(v)
        if len(v) < MIN_N:
            row[o] = row[f"{o}_lo"] = row[f"{o}_hi"] = np.nan
            continue
        if o in BINARY:
            k = float(v.sum())
            row[o] = k / len(v)
            row[f"{o}_lo"], row[f"{o}_hi"] = wilson(k, len(v))
        else:
            m, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
            row[o], row[f"{o}_lo"], row[f"{o}_hi"] = m, m - 1.96 * se, m + 1.96 * se
    return row


def main():
    CFG.describe("care_exclusive")
    p = os.path.join(OUT_DIR, "care_outcomes_per_person.csv")
    assert os.path.exists(p), f"{p} not found -- run care.py first"
    O = pd.read_csv(p)
    O[ID_COL] = O[ID_COL].astype(str)
    O = O.set_index(ID_COL)
    R = pd.read_csv(CFG.responses_csv, usecols=[ID_COL, "age", "sex"])
    R[ID_COL] = R[ID_COL].astype(str)
    dem = R.set_index(ID_COL)
    N = len(O)
    print(f"[care_exclusive] {N:,} people | survey {int(O['has_survey'].sum()):,} | "
          f"EHR {int(O['has_ehr'].sum()):,}")

    sets = {}
    for m in ENCODERS + COMPARATORS:
        f = os.path.join(OUT_DIR, f"scores_{m}.csv")
        if not os.path.exists(f):
            print(f"  [skip] scores_{m}.csv not found")
            continue
        S = pd.read_csv(f)
        S[ID_COL] = S[ID_COL].astype(str)
        by = next(c for c in ("pctile", "distance2", "distance") if c in S.columns)
        order = S.sort_values(by, ascending=False)[ID_COL].values
        for cut in CUTS:
            k = int(round(N * (100 - cut) / 100.0))
            if k >= 1:
                sets[(m, cut)] = O.index.intersection(pd.Index(order[:k]))

    rows = [rates(O.index, O, dem, "cohort", np.nan, "reference", "cohort")]
    for cut in CUTS:
        for e in ENCODERS:
            quad = {e: sets.get((e, cut))}
            for c in COMPARATORS:
                quad[c] = sets.get((c, cut))
            if any(v is None for v in quad.values()):
                continue
            excl = {}
            for name, S_ in quad.items():
                others = pd.Index([])
                for o_, T_ in quad.items():
                    if o_ != name:
                        others = others.union(T_)
                excl[name] = S_.difference(others)
            adj = {}
            for o in OUTCOMES:
                adj.update(adjusted(excl, O, dem, o, e))
            for name, idx in excl.items():
                r = rates(idx, O, dem, name, cut, e, "encoder" if name == e else name)
                for o in OUTCOMES:                       # this set's own adjusted value
                    r[f"{o}_adj"] = adj.get(f"{o}_adj__{name}", np.nan)
                    if name != e:                        # and its contrast against the encoder
                        key = "or" if o in BINARY else "diff"
                        for suf in ("", "_lo", "_hi"):
                            r[f"{o}_{key}{suf}_vs_encoder"] = adj.get(f"{o}_{key}{suf}__{name}", np.nan)
                rows.append(r)
    D = pd.DataFrame(rows)
    # ---- the dissemination policy (disclosure.py), at write time. An exclusive set
    # of 1 to 20 has every statistic blanked. A rate times its n gives the count back, and so does the
    # share female, so each is withheld where that count or its complement is 1 to 20, and a set's n
    # with an outcome is hidden when it or the set's remainder is 1 to 20.
    from disclosure import MARK, SUPP, pair_hidden, small
    Dw = D.copy().astype(object)
    for i in D.index:
        ne = int(D.at[i, "n_exclusive"])
        if small(ne):
            keep = ["cut_pctile", "quadruple_encoder", "grouping", "name"]
            Dw.loc[i, [c for c in D.columns if c not in keep]] = np.nan
            Dw.at[i, "n_exclusive"] = MARK
            continue
        if pd.notna(D.at[i, "pct_female"]) and pair_hidden(round(D.at[i, "pct_female"] * ne / 100), ne):
            Dw.at[i, "pct_female"] = np.nan
        for o in OUTCOMES:
            no = int(D.at[i, f"n_{o}"])
            cols = [c for c in D.columns if c == o or c.startswith(f"{o}_")]
            if small(no):
                Dw.loc[i, cols] = np.nan
                Dw.at[i, f"n_{o}"] = MARK
                continue
            if pair_hidden(no, ne):
                Dw.at[i, f"n_{o}"] = SUPP
            if o in BINARY and pd.notna(D.at[i, o]) and pair_hidden(round(float(D.at[i, o]) * no), no):
                Dw.loc[i, [o, f"{o}_lo", f"{o}_hi"]] = np.nan
    Dw.to_csv(os.path.join(OUT_DIR, tagged("care_exclusive.csv")), index=False)
    print(f"  -> care_exclusive.csv ({len(D)} rows)")
    at = D[(D.cut_pctile == 95) & (D.quadruple_encoder == EXEMPLAR)]
    coh = D[D.grouping == "cohort"].iloc[0]
    print(f"    cohort: access {100 * coh.any_access_barrier:.1f}%  cost "
          f"{100 * coh.any_cost_barrier:.1f}%  avoided {100 * coh.care_avoid_provider:.1f}%  "
          f"ED share {100 * coh.ed_share:.1f}%  age {coh.mean_age:.1f}")
    print(f"    95th, {EXEMPLAR} group, exclusive to (raw, then age- and sex-adjusted):")
    for _, r in at.iterrows():
        print(f"      {r['name']:22s} n {int(r.n_exclusive):5d} age {r.mean_age:.1f}")
        print(f"        raw  access {100 * r.any_access_barrier:5.1f}%  cost "
              f"{100 * r.any_cost_barrier:5.1f}%  avoided {100 * r.care_avoid_provider:5.1f}%  "
              f"ED {100 * r.ed_share:4.2f}%")
        print(f"        adj  access {100 * r.any_access_barrier_adj:5.1f}%  cost "
              f"{100 * r.any_cost_barrier_adj:5.1f}%  avoided {100 * r.care_avoid_provider_adj:5.1f}%"
              f"  ED {100 * r.ed_share_adj:4.2f}%")
        if r['name'] != EXEMPLAR:
            print(f"        vs encoder: access OR {r.any_access_barrier_or_vs_encoder:.2f} "
                  f"[{r.any_access_barrier_or_lo_vs_encoder:.2f}, "
                  f"{r.any_access_barrier_or_hi_vs_encoder:.2f}]  cost OR "
                  f"{r.any_cost_barrier_or_vs_encoder:.2f} "
                  f"[{r.any_cost_barrier_or_lo_vs_encoder:.2f}, "
                  f"{r.any_cost_barrier_or_hi_vs_encoder:.2f}]")
    if NO_FIG:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0), constrained_layout=True)
    for ax, o in zip(axes.ravel(), OUTCOMES):
        for e, c in zip(ENCODERS, ENC_COLORS):
            d = D[(D.grouping == "encoder") & (D.quadruple_encoder == e)].sort_values("cut_pctile")
            if len(d):
                ax.plot(d.cut_pctile, 100 * d[o], color=c, lw=1.3, label=e)
        for g in COMPARATORS:
            d = D[(D.grouping == g) & (D.quadruple_encoder == EXEMPLAR)].sort_values("cut_pctile")
            if len(d):
                ax.plot(d.cut_pctile, 100 * d[o], color="black", ls=CMP_STYLE[g], lw=1.5,
                        label=CMP_LAB[g])
        ax.axhline(100 * coh[o], color="0.45", lw=0.9, ls=(0, (1, 2)), label="whole cohort")
        ax.set_xlabel("Percentile cut", fontsize=8)
        ax.set_ylabel(LABELS[o] + " (%)", fontsize=8)
        ax.set_xlim(PLOT_FROM, max(CUTS))
        ax.set_xticks(xticks_for(PLOT_FROM))
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.09))
    out = os.path.join(OUT_DIR, tagged("fig_care_exclusive.png"))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
