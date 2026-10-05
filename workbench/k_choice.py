#!/usr/bin/env python3
"""k_choice.py -- the choice of k for the 3 groupings fitted on the RESPONSES: Horn's k and the
cumulative variance it explains. For the supplement's account of k (eFigure 1, panels C to E).

STANDALONE on the Workbench. Needs only the response files, not the screen. Aggregates only.

WHAT EACH GROUPING GETS
  PCA     Horn's parallel analysis on the correlation matrix of the 151 standardized responses,
          comparators.py's own function (parallel_analysis_k, 100 permutations, 95th percentile,
          seed 0), so this k is the one comparators.py fits PCA and the factor model at. Cumulative
          variance explained = the running sum of the correlation eigenvalues over their total, 151.
  factor  the same Horn's k (comparators.py fits both at it). Its variance explained is the COMMON
          variance: the sum of the communalities (squared loadings, summed over factors and items)
          over 151. A factor model splits each item's variance into a common and a unique part, so
          this is the model's own measure and it sits below PCA's at every k. One fit per k, so it
          is computed on a grid (FACTOR_GRID), not at every k. Unrotated: comparators.py rotates by
          varimax, and a rotation leaves the communalities unchanged.
  MCA     the indicator matrix of the answer categories, with imputed cells snapped to answered
          levels, and Horn's k by permuting each item's answers: mca_basis.py's own functions
          (load_categories, fit, horn) with the settings of step 4, 100 axes and 10 permutations, so
          this k is the one `mca_basis.py --axes 100 --parallel 10` gives. Variance explained is the
          RAW share of the total inertia, (J - Q) / Q, which is exact however few axes are computed,
          and is the quantity Horn's test is run on. The Benzecri-adjusted share is written beside it.
          ★ MCA is also reported at k = 25, PCA's Horn's k on the same responses:
          MCA and PCA are both fitted on the responses, so PCA's k is the reference for MCA's second k.

The groupings fitted on the ITEM TEXT (the 6 encoders and TF-IDF) are computed locally, not here
(1_select_k.py and tfidf_horn.py in the repository). TF-IDF's second k, 25, is referenced to the
encoders' Horn's k (23 to 27) in the same way.

OUTPUTS (-> screen_out/k_choice/, then zipped)
  k_choice_pca.csv       component, eigenvalue, null_95th, share_pct, cum_pct            (1 to 151)
  k_choice_factor.csv    k, common_cum_pct, converged, secs                              (the grid)
  k_choice_mca.csv       axis, eigenvalue, null_95th, pct_inertia, cum_pct_inertia,
                         benzecri_pct, cum_benzecri_pct                                    (1 to 100)
  k_choice_summary.json  each grouping's Horn's k and cumulative variance at it, and MCA at 25;
                         the cohort size, J, Q, the settings, the CDR
  screen_out/k_choice_<CDR>.zip   the 4 files above: the only thing to download from this step.
  No person-level output. The only participant count is the cohort size.

Run:  python3 k_choice.py            (the full factor grid, about 45 factor fits; the MCA permutations
                                      are the slowest part)
      python3 k_choice.py --quick    (the factor model at Horn's k only)
"""
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from comparators import CDR, N_PERM, PA_PCTILE, SEED, load_battery, parallel_analysis_k   # noqa: E402
from mca_basis import fit, horn, inertia_shares, load_categories                           # noqa: E402

OUT = os.path.join(HERE, "screen_out", "k_choice")
QUICK = "--quick" in sys.argv
FACTOR_GRID = list(range(1, 31)) + list(range(35, 101, 5))
MCA_AXES, MCA_PERM = 100, 10          # step 4's `mca_basis.py --axes 100 --parallel 10`
REF_K = 25                            # MCA's second k: PCA's Horn's k on the same responses (checked below)


def main():
    os.makedirs(OUT, exist_ok=True)
    t_all = time.time()

    # ---- PCA and the factor model -----------------------------------------------------------------
    ids, items, T, X, cov, w = load_battery()
    n, p = X.shape
    Xs = (X - X.mean(0)) / X.std(0)
    print(f"[k_choice] cohort {n:,} x {p} items | parallel analysis, {N_PERM} permutations", flush=True)
    k_pca, real, thr = parallel_analysis_k(Xs, N_PERM, PA_PCTILE, SEED)
    cum = 100 * np.cumsum(real) / real.sum()
    pd.DataFrame({"component": np.arange(1, p + 1), "eigenvalue": real, "null_95th": thr,
                  "share_pct": 100 * real / real.sum(), "cum_pct": cum}).to_csv(
        os.path.join(OUT, "k_choice_pca.csv"), index=False)
    print(f"  PCA: Horn's k = {k_pca}, cumulative variance {cum[k_pca - 1]:.1f}%", flush=True)
    if k_pca != REF_K:
        print(f"  ★ PCA's Horn's k is {k_pca}, not {REF_K}: MCA's second k is still reported at {REF_K}, "
              f"and at {k_pca} beside it", flush=True)

    grid = [k_pca] if QUICK else sorted(set(FACTOR_GRID) | {k_pca})
    frows = []
    for k in grid:
        t0 = time.time()
        fa = FactorAnalysis(n_components=k, random_state=SEED).fit(Xs)
        common = 100 * float((fa.components_ ** 2).sum()) / p
        frows.append({"k": k, "common_cum_pct": common, "converged": bool(fa.n_iter_ < fa.max_iter),
                      "secs": round(time.time() - t0, 1)})
        print(f"  factor k = {k:3d}: common variance {common:5.1f}% (PCA {cum[k - 1]:5.1f}%) "
              f"converged {fa.n_iter_ < fa.max_iter} [{time.time() - t0:.0f}s]", flush=True)
    F = pd.DataFrame(frows)
    F.to_csv(os.path.join(OUT, "k_choice_factor.csv"), index=False)
    fa_at = float(F.set_index("k").loc[k_pca, "common_cum_pct"])

    # ---- MCA --------------------------------------------------------------------------------------
    resp, Z, cats, chosen, J, c, n_m, Q = load_categories(items)
    assert n_m == n and Q == p, "MCA and the comparators read different response matrices"
    k_ax = min(MCA_AXES, J - Q - 1)
    print(f"[k_choice] MCA: truncated SVD for {k_ax} axes (total inertia {(J - Q) / Q:.3f}), "
          f"then {MCA_PERM} permutations", flush=True)
    sig, _, _ = fit(Z, c, n, Q, k_ax)
    lam = sig ** 2
    null95, _ = horn(chosen, c, n, Q, J, k_ax, MCA_PERM, SEED)
    passes = lam > null95
    k_mca = int(np.argmax(~passes)) if (~passes).any() else k_ax
    assert k_mca < k_ax, (f"MCA's Horn's k reaches the {k_ax} axes computed, so the cut is not seen; "
                          f"raise MCA_AXES")
    pct, ben_pct, keep = inertia_shares(lam, J, Q)
    ben_complete = not bool(keep[-1])       # False when axis k_ax is still above 1/Q
    if not ben_complete:
        print(f"  ★ axis {k_ax} is still above 1/Q, so the Benzecri shares are over the first {k_ax} axes only")
    M = pd.DataFrame({"axis": np.arange(1, k_ax + 1), "eigenvalue": lam, "null_95th": null95,
                      "pct_inertia": pct, "cum_pct_inertia": np.cumsum(pct),
                      "benzecri_pct": ben_pct, "cum_benzecri_pct": np.cumsum(ben_pct)})
    M.to_csv(os.path.join(OUT, "k_choice_mca.csv"), index=False)
    m_at = M.set_index("axis")
    print(f"  MCA: Horn's k = {k_mca}, raw inertia {m_at.loc[k_mca, 'cum_pct_inertia']:.1f}% "
          f"(Benzecri {m_at.loc[k_mca, 'cum_benzecri_pct']:.1f}%); at k = {REF_K}: "
          f"{m_at.loc[REF_K, 'cum_pct_inertia']:.1f}% (Benzecri {m_at.loc[REF_K, 'cum_benzecri_pct']:.1f}%)",
          flush=True)

    summary = {
        "cdr": CDR, "cohort_n": int(n), "items_Q": int(Q), "categories_J": int(J),
        "pca": {"horn_k": k_pca, "cum_pct_at_horn_k": round(float(cum[k_pca - 1]), 2),
                "n_perm": N_PERM, "pctile": PA_PCTILE, "seed": SEED},
        "factor": {"horn_k": k_pca, "common_cum_pct_at_horn_k": round(fa_at, 2), "grid": grid,
                   "measure": "sum of communalities / Q (common variance), unrotated"},
        "mca": {"horn_k": k_mca, "cum_pct_inertia_at_horn_k": round(float(m_at.loc[k_mca, "cum_pct_inertia"]), 2),
                "cum_benzecri_pct_at_horn_k": round(float(m_at.loc[k_mca, "cum_benzecri_pct"]), 2),
                "second_k": REF_K, "second_k_reference": "PCA's Horn's k on the same responses",
                "cum_pct_inertia_at_second_k": round(float(m_at.loc[REF_K, "cum_pct_inertia"]), 2),
                "cum_benzecri_pct_at_second_k": round(float(m_at.loc[REF_K, "cum_benzecri_pct"]), 2),
                "total_inertia": round((J - Q) / Q, 4), "axes_computed": int(k_ax), "n_perm": MCA_PERM,
                "benzecri_over_all_axes_above_1_over_Q": ben_complete},
        "runtime_min": round((time.time() - t_all) / 60, 1),
    }
    if k_pca != REF_K:
        summary["mca"]["cum_pct_inertia_at_pca_horn_k"] = round(float(m_at.loc[k_pca, "cum_pct_inertia"]), 2)
    json.dump(summary, open(os.path.join(OUT, "k_choice_summary.json"), "w"), indent=2)

    zpath = os.path.join(HERE, "screen_out", f"k_choice_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in ("k_choice_pca.csv", "k_choice_factor.csv", "k_choice_mca.csv", "k_choice_summary.json"):
            z.write(os.path.join(OUT, f), arcname=f"k_choice/{f}")
    print(f"\n[k_choice] done in {summary['runtime_min']} min\n  download -> {zpath}")


if __name__ == "__main__":
    main()
