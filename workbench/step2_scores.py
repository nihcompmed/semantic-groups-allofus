#!/usr/bin/env python3
"""step2_scores.py -- every participant's distance under the 12 groupings, and the choice of k for the groupings fitted
on the answers (Figure S1; Appendix S4).

THE 12 GROUPINGS (scoring.py scores them all the same way)
  formed from the item text, built outside the Workbench and shipped in basis/ and basis_tfidf/
    the 6 sentence encoders, each at its own Horn's k (23 to 27)
    tfidf4, tfidf25     word counts (TF-IDF) at their own Horn's k (4) and at 25, the encoders' Horn's k
  fitted here on the answers (standardized over the cohort)
    pca                 principal components at Horn's k on the answers' correlation matrix (25)
    factor              maximum-likelihood factors with varimax at the same k
    mca                 multiple correspondence analysis of the answer categories at PCA's Horn's k
    mca<h>              MCA at its own Horn's k, h (90 on this release)
The factor model at k = 3 (Cattell-Nelson-Gorsuch) is scored too; Appendix S4 compares the outliers with it.

THE CHOICE OF k (download k_choice)
  PCA     Horn's parallel analysis on the correlation matrix of the 151 standardized answers (100 permutations, 95th
          percentile); cumulative variance explained
  factor  the same k; common variance (sum of communalities / 151) at k = 1 to 30 and 35 to 100 by 5, unrotated
  MCA     the indicator matrix of the answer categories (imputed answers snapped to the nearest answered level); Horn's
          k by permuting each item's answers (10 permutations), on 100 axes; raw and Benzecri-adjusted inertia

OUTPUTS (work/)  scores_<grouping>.csv (id, distance, distance2, pctile, flagged), scores_factor3.csv (id, distance,
                 distance2), groupings.csv (each grouping's family, k and how k was chosen), the MCA category coordinates
DOWNLOAD         downloads/k_choice_<release>.zip

Run:  python3 step2_scores.py
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

import scoring as sc
from release import write_download
from settings import BASIS_TFIDF, CDR, ENCODERS, ID, download_dir, work

N_PERM, PA_PCTILE = 100, 95.0
FACTOR_GRID = list(range(1, 31)) + list(range(35, 101, 5))
MCA_AXES, MCA_PERM = 100, 10
MCA_SECOND_K = 25                       # MCA's second k: PCA's Horn's k on the same answers (checked below)
TFIDF = {"tfidf4": ("tfidf-subtlex-k4", 4), "tfidf25": ("tfidf-subtlex-k25", 25)}
K3 = 3


def main():
    t0 = time.time()
    T, items = sc.items_table()
    resp, X, cov, w = sc.load_battery(items)
    ids = resp.index
    n, p = X.shape
    print(f"[step 2] cohort {n:,} x {p} items", flush=True)
    groupings = []

    # ---- the 6 encoders and the 2 TF-IDF groupings
    for enc in ENCODERS:
        P, _ = sc.place(enc, items)
        d, T2, pct, flag = sc.score_placed(P, X, cov, w)
        sc.scores_frame(ids, d, T2, pct, flag).to_csv(work(f"scores_{enc}.csv"), index=False)
        groupings.append((enc, "item text", int(P.shape[1]), "own Horn's k"))
        print(f"  {enc:22s} k = {P.shape[1]}", flush=True)
    for name, (basis, k_exp) in TFIDF.items():
        P, _ = sc.place(basis, items, BASIS_TFIDF)
        assert P.shape[1] == k_exp, f"{name}: the basis has k = {P.shape[1]}, expected {k_exp}"
        d, T2, pct, flag = sc.score_placed(P, X, cov, w)
        sc.scores_frame(ids, d, T2, pct, flag).to_csv(work(f"scores_{name}.csv"), index=False)
        print(f"  {name:22s} k = {P.shape[1]}", flush=True)

    # ---- PCA and the factor model: Horn's k, the factor grid
    sd = X.std(0)
    Xs = (X - X.mean(0)) / sd
    k, real, thr = sc.parallel_analysis_k(Xs, N_PERM, PA_PCTILE, sc.SEED)
    cum = 100 * np.cumsum(real) / real.sum()
    print(f"  PCA and factor: Horn's k = {k}", flush=True)
    kc = download_dir("k_choice")
    pd.DataFrame({"component": np.arange(1, p + 1), "eigenvalue": real, "null_95th": thr,
                  "share_pct": 100 * real / real.sum(), "cum_pct": cum}).to_csv(os.path.join(kc, "k_choice_pca.csv"),
                                                                              index=False)
    frows = []
    for kk in sorted(set(FACTOR_GRID) | {k}):
        fa = FactorAnalysis(n_components=kk, random_state=sc.SEED).fit(Xs)
        frows.append({"k": kk, "common_cum_pct": 100 * float((fa.components_ ** 2).sum()) / p,
                      "converged": bool(fa.n_iter_ < fa.max_iter)})
    F = pd.DataFrame(frows)
    F.to_csv(os.path.join(kc, "k_choice_factor.csv"), index=False)

    resid = sc.residualizer(cov, w)
    Xr = resid(X)
    S_fa, _, _ = sc.factor_model(Xs, k, sc.SEED)
    S_pc, _ = sc.pca_model(Xs, k)
    for name, S_eff in (("factor", S_fa / sd[:, None]), ("pca", S_pc / sd[:, None])):
        d, T2, pct, flag, _ = sc.score_linear(S_eff, Xr, w)
        sc.scores_frame(ids, d, T2, pct, flag).to_csv(work(f"scores_{name}.csv"), index=False)
    S3, _, _ = sc.factor_model(Xs, K3, sc.SEED)
    d3, T2_3, _, _, _ = sc.score_linear(S3 / sd[:, None], Xr, w)
    pd.DataFrame({ID: ids, "distance": d3, "distance2": T2_3}).to_csv(work("scores_factor3.csv"), index=False)

    # ---- MCA: one fit on 100 axes and its Horn's k; scored at PCA's k and at its own k
    resp_m, Z, cats, chosen, J, c, n_m, Q = sc.load_categories(items)
    assert n_m == n and Q == p, "MCA and the other groupings read different answers"
    k_ax = min(MCA_AXES, J - Q - 1)
    sig, U, V = sc.mca_fit(Z, c, n, Q, J, k_ax)
    lam = sig ** 2
    G = ((1.0 / np.sqrt(c))[:, None] * V) * sig                            # category principal coordinates
    null95, _ = sc.mca_horn(chosen, c, n, Q, J, k_ax, MCA_PERM, sc.SEED)
    passes = lam > null95
    h = int(np.argmax(~passes)) if (~passes).any() else k_ax
    assert h < k_ax, f"MCA's Horn's k reaches the {k_ax} axes computed: raise MCA_AXES"
    pct_i, ben_pct, keep = sc.inertia_shares(lam, J, Q)
    cc = pd.DataFrame(G, columns=[f"mca{a + 1}" for a in range(k_ax)]).assign(
        item=[t[0] for t in cats], value=[t[1] for t in cats], mass=c)
    cc.to_csv(work("mca_category_coords.csv"), index=False)
    ev = pd.DataFrame({"axis": np.arange(1, k_ax + 1), "eigenvalue": lam, "pct_inertia": pct_i,
                       "benzecri_pct": ben_pct, "above_1_over_Q": keep, "null_95th": null95, "keep_horn": passes})
    ev.to_csv(work("mca_eigenvalues.csv"), index=False)
    C = pd.read_csv(work("mca_category_coords.csv"))                    # read back, as the scores use the file
    E = pd.read_csv(work("mca_eigenvalues.csv"))
    Xraw = pd.read_csv(sc.COHORT_CSV)[items].values.astype(float)        # the answers on their own scales
    fails = np.where(E["eigenvalue"].values <= E["null_95th"].values)[0]
    h_file = int(fails[0])
    for name, kk in (("mca", k), (f"mca{h_file}", h_file)):
        Gm, ch = sc.mca_representation(items, Xraw, C, E, kk)
        d, T2, pct, flag = sc.score_mca(Gm, ch, w, resid)
        sc.scores_frame(ids, d, T2, pct, flag).to_csv(work(f"scores_{name}.csv"), index=False)
    print(f"  MCA: {J} answer categories | Horn's k = {h}", flush=True)
    M = pd.DataFrame({"axis": np.arange(1, k_ax + 1), "eigenvalue": lam, "null_95th": null95, "pct_inertia": pct_i,
                      "cum_pct_inertia": np.cumsum(pct_i), "benzecri_pct": ben_pct,
                      "cum_benzecri_pct": np.cumsum(ben_pct)})
    M.to_csv(os.path.join(kc, "k_choice_mca.csv"), index=False)

    groupings += [("tfidf4", "item text", 4, "own Horn's k"), ("tfidf25", "item text", 25, "encoders' Horn's k"),
                  ("pca", "responses", k, "own Horn's k"), ("factor", "responses", k, "Horn's k on the response correlations"),
                  (f"mca{h_file}", "responses", h_file, "own Horn's k"), ("mca", "responses", k, "PCA's Horn's k")]
    pd.DataFrame(groupings, columns=["grouping", "family", "k", "k_rule"]).to_csv(work("groupings.csv"), index=False)

    m_at = M.set_index("axis")
    fa_at = float(F.set_index("k").loc[k, "common_cum_pct"])
    summary = {
        "cdr": CDR, "cohort_n": int(n), "items_Q": int(Q), "categories_J": int(J),
        "pca": {"horn_k": k, "cum_pct_at_horn_k": round(float(cum[k - 1]), 2), "n_perm": N_PERM, "pctile": PA_PCTILE,
                "seed": sc.SEED},
        "factor": {"horn_k": k, "common_cum_pct_at_horn_k": round(fa_at, 2), "grid": sorted(set(FACTOR_GRID) | {k}),
                   "measure": "sum of communalities / Q (common variance), unrotated"},
        "mca": {"horn_k": h, "cum_pct_inertia_at_horn_k": round(float(m_at.loc[h, "cum_pct_inertia"]), 2),
                "cum_benzecri_pct_at_horn_k": round(float(m_at.loc[h, "cum_benzecri_pct"]), 2),
                "second_k": MCA_SECOND_K, "second_k_reference": "PCA's Horn's k on the same responses",
                "cum_pct_inertia_at_second_k": round(float(m_at.loc[MCA_SECOND_K, "cum_pct_inertia"]), 2),
                "cum_benzecri_pct_at_second_k": round(float(m_at.loc[MCA_SECOND_K, "cum_benzecri_pct"]), 2),
                "total_inertia": round((J - Q) / Q, 4), "axes_computed": int(k_ax), "n_perm": MCA_PERM,
                "benzecri_over_all_axes_above_1_over_Q": not bool(keep[-1])}}
    if k != MCA_SECOND_K:
        summary["mca"]["cum_pct_inertia_at_pca_horn_k"] = round(float(m_at.loc[k, "cum_pct_inertia"]), 2)
    json.dump(summary, open(os.path.join(kc, "k_choice_summary.json"), "w"), indent=2)
    write_download("k_choice", ["k_choice_pca.csv", "k_choice_factor.csv", "k_choice_mca.csv", "k_choice_summary.json"])
    print(f"[step 2] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
