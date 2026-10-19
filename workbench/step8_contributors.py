#!/usr/bin/env python3
"""step8_contributors.py -- which semantic groups carry each outlier's distance (Figure 3 and its source data).

For each encoder's own outliers (distance above its 95th percentile), and for those of them who met none of the six
thresholds with all six evaluable, each participant's squared distance divides exactly into one share per group
(a_j / T2 = y_j (Omega y)_j / T2, summing to 1, where y are the residualized centered scores and Omega the Ledoit-Wolf
precision). A participant's top 3 are the 3 groups with the largest shares. Per group: the percent of the set with the
group in its top 3, and the mean and median share, |z| (y_j over its SD) and |r| (the conditional score). Groups are
named by the construct profile of their core items (basis/components.csv). Sizes only: a group's sign has no meaning.

The policy: the top-3 counts of one encoder partition 3 x n slots, so they go through release.partition_mask (a count of
1 to 20 is written "<=20" with its statistics blank; cells hidden only to protect it are written "suppressed"). An
encoder whose set, or the rest of its outliers, is 1 to 20 writes nothing.

DOWNLOAD contributors: contributors_all.csv and contributors_top3_all.csv (all outliers), contributors_below.csv and
contributors_top3_below.csv (outliers below every threshold)

Run:  python3 step8_contributors.py
"""
import os

import numpy as np
import pandas as pd

import scoring as sc
from release import MARK, marked, pair_hidden, partition_mask, write_download
from settings import BASIS, ENCODERS, ID, download_dir, work

TOP_N = 3


def contributions(items, R, X, cov, w, below):
    """Per encoder: every outlier's share, z and r per group (restricted to `below` when given)."""
    frames = []
    for enc in ENCODERS:
        S = pd.read_csv(work(f"scores_{enc}.csv"))
        S[ID] = S[ID].astype(str)
        own = set(S.loc[S["flagged"].astype(bool), ID])
        n_own = len(own)
        if below is not None:
            own = own & below
        idx = np.array([i for i, pid in enumerate(R.index) if pid in own])
        assert len(idx), f"{enc}: no participant selected"
        if pair_hidden(len(idx), n_own):
            print(f"  {enc}: {len(idx)} selected of {n_own}, a count of 1 to 20 on one side: nothing written")
            continue
        Pk = sc.place(enc, items)[0]
        Y = sc.wls_residualize(X @ Pk, cov, w)
        T2, lw = sc.mahalanobis2(Y, w)
        Yc = Y - np.average(Y, axis=0, weights=w)
        Om, Sig = lw.precision_, lw.covariance_
        G = Yc @ Om
        d_saved = S.set_index(ID)["distance"].reindex(R.index).values.astype(float)
        worst = float(np.nanmax(np.abs(np.sqrt(np.maximum(T2, 0)) - d_saved)))
        assert worst < 1e-6, f"{enc}: the recomputed distance differs from scores_{enc}.csv by {worst}"
        a = (Yc[idx] * G[idx]) / T2[idx, None]
        z = Yc[idx] / np.sqrt(np.diag(Sig))
        r = G[idx] / np.sqrt(np.diag(Om))
        assert np.abs(a.sum(axis=1) - 1.0).max() < 1e-6, f"{enc}: shares do not sum to 1"
        k = Pk.shape[1]
        frames.append(pd.DataFrame({ID: np.repeat(R.index.values[idx], k), "encoder": enc,
                                    "component": np.tile(np.arange(k), len(idx)),
                                    "share": a.ravel(), "z": z.ravel(), "r": r.ravel()}))
    assert frames, "every encoder's set was withheld under the dissemination policy"
    D = pd.concat(frames, ignore_index=True)
    D["abs_z"], D["abs_r"] = D.z.abs(), D.r.abs()
    return D


def tables(D):
    cp = pd.read_csv(os.path.join(BASIS, "components.csv"))
    cp = cp[cp.encoder.isin(ENCODERS)]
    GD = pd.DataFrame({"encoder": cp.encoder, "group": cp.component, "component": cp.component,
                       "n_items": cp.n_core_items, "construct_profile": cp.construct_profile, "instruments": ""})
    rows = []
    for (enc, j), d in GD.groupby(["encoder", "component"]):
        rows.append({"encoder": enc, "component": int(j), "n_semantic_groups": len(d), "n_core_items": int(d.n_items.sum()),
                     "semantic_groups": " vs ".join(d.construct_profile),
                     "instruments": "; ".join(sorted({x for v in d.instruments for x in str(v).split("; ") if x}))})
    GROUPS = pd.DataFrame(rows)
    g = D.groupby(["encoder", "component"])
    S = pd.DataFrame({
        "n_people": g[ID].nunique(), "mean_abs_z": g.abs_z.mean(), "median_abs_z": g.abs_z.median(),
        "mean_abs_r": g.abs_r.mean(), "median_abs_r": g.abs_r.median(),
        "mean_share": g.share.mean(), "median_share": g["share"].median(),
    }).reset_index().merge(GROUPS, on=["encoder", "component"], how="left")
    S = S.sort_values(["encoder", "mean_abs_z"], ascending=[True, False])
    top = D.sort_values("share", ascending=False, kind="mergesort").groupby(["encoder", ID], sort=False).head(TOP_N)
    assert top.groupby("encoder").size().equals(D.groupby("encoder")[ID].nunique() * TOP_N), "top-3 slots do not add up"
    g2 = top.groupby(["encoder", "component"])
    B = pd.DataFrame({
        "n_top": g2[ID].nunique(), "mean_abs_z": g2.abs_z.mean(), "median_abs_z": g2.abs_z.median(),
        "mean_abs_r": g2.abs_r.mean(), "median_abs_r": g2.abs_r.median(),
        "mean_share": g2.share.mean(), "median_share": g2["share"].median(),
    }).reset_index()
    full = D[["encoder", "component"]].drop_duplicates()
    B = full.merge(B, on=["encoder", "component"], how="left").fillna({"n_top": 0})        # in nobody's top 3: 0
    B["n_top"] = B["n_top"].astype(int)
    B.insert(2, "ranking", "share")
    T3 = B.merge(GROUPS, on=["encoder", "component"], how="left")
    n_set = D.groupby("encoder")[ID].nunique()
    T3.insert(3, "n_set", T3["encoder"].map(n_set).astype(int))
    T3 = T3.sort_values(["ranking", "encoder", "n_top"], ascending=[True, True, False]).reset_index(drop=True)
    T3["n_top"] = T3["n_top"].astype(object)
    stat_cols = [c for c in T3.columns if c.startswith(("mean_", "median_"))]
    for _, ix in T3.groupby(["encoder", "ranking"]).groups.items():
        ix = list(ix)
        hide, prim = partition_mask(T3.loc[ix, "n_top"].values)
        T3.loc[ix, "n_top"] = marked(T3.loc[ix, "n_top"].values, hide, prim)
        T3.loc[[i for i, p in zip(ix, prim) if p], stat_cols] = np.nan
    num = pd.to_numeric(T3["n_top"], errors="coerce")
    T3.insert(4, "pct_top", (100.0 * num / T3["n_set"]).round(1))
    return S, T3


def main():
    T, items = sc.items_table()
    R, X, cov, w = sc.load_battery(items)
    cf = pd.read_csv(work("thresholds.csv"))
    cf[ID] = cf[ID].astype(str)
    below = set(cf.loc[cf["none_met"].astype(bool), ID])
    d = download_dir("contributors")
    for tag, sel in (("all", None), ("below", below)):
        S, T3 = tables(contributions(items, R, X, cov, w, sel))
        S.to_csv(os.path.join(d, f"contributors_{tag}.csv"), index=False)
        T3.to_csv(os.path.join(d, f"contributors_top3_{tag}.csv"), index=False)
        print(f"  {tag}: {len(T3)} rows, {int((T3.n_top == MARK).sum())} written {MARK}", flush=True)
    write_download("contributors", ["contributors_all.csv", "contributors_top3_all.csv", "contributors_below.csv",
                                    "contributors_top3_below.csv"])


if __name__ == "__main__":
    main()
