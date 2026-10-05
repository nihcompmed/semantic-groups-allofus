#!/usr/bin/env python3
"""mca_basis.py -- multiple correspondence analysis of the 151 items, as a fifth grouping.

WHY. The 2 other response-fitted comparators the analysis carries, common-factor analysis and PCA,
both treat ordinal answers as continuous. A reviewer can fairly say that makes the covariance-defined
side of the comparison a strawman for categorical survey data. MCA is the reduction built for this
data type, so including it closes that objection instead of inviting it. It is also the method used at
scale on these same surveys (Biji et al., Am J Hum Genet 2026;113:1434-1447, n = 413,457, 142
variables, 50 axes, ade4).

★ THIS SCRIPT DOES NOT TOUCH `comparators.py`. It stands alone and writes its outputs beside the other
screen outputs. Scoring MCA as a fifth grouping is a separate step (comparators.py), taken only after
the check at the bottom of this script passes, because `comparators.py` states its contract as "each a
linear map S of the keyed, rescaled responses X, so Y = X S" and MCA is NOT linear in X.

★ WHAT MCA DOES, AND HOW IT DIFFERS FROM EVERY OTHER GROUPING HERE. It does not group items. It places
ANSWER CATEGORIES in a space: the input is an indicator matrix with one column per (item, answer) and
exactly one non-zero per item per person, and each axis is a contrast among categories. An item can
therefore enter an axis through only some of its levels, which no other grouping in this analysis can
express. The item-level analogue of a loading is the DISCRIMINATION MEASURE, the squared correlation
ratio between the axis and the item, written here to `mca_discrimination.csv` so the fixed-point
participation-ratio core rule can name axes exactly as it names components.

★ WHY THE DISTANCE AND THE ATTRIBUTION STILL WORK. Correspondence analysis has a transition formula:

        F[i,k] = (1 / (Q * sigma_k)) * sum over items q of G[category chosen by i on q, k]

with Q the number of items, sigma_k the singular value, F the row (person) principal coordinates and G
the column (category) principal coordinates. A person's coordinate is an EXACT SUM OF ONE TERM PER
ITEM. That is all the T^2 partition needs -- additivity over items, not linearity in the response
values. The only difference from `factor` and `pca` is that an item's weight depends on WHICH CATEGORY
the person endorsed rather than being a fixed column of S. The script verifies the identity numerically
before writing anything, and reports the largest absolute error.

DEFINITIONS, fixed before the numbers.
  categories    the distinct numeric values of each item AMONG CELLS THAT WERE ACTUALLY ANSWERED.
                `build_responses_v9.py` maps every answer code to a numeric value one to one, so those
                values are the answer categories.
                ★ IMPUTED CELLS ARE SNAPPED TO THE NEAREST REAL LEVEL, and this is not cosmetic.
                That script imputes with KNNImputer(k=5) and inverts the rescaling "so the files stay
                in codebook units (fractional where imputed)", so an imputed cell can hold 2.4 or 3.8.
                Taken literally each such value is its own category with a mass of a few people, and
                MCA weights a category by 1/mass, so imputation noise would carry the largest
                coordinates on every axis. Without snapping there are 2,400 categories over 151
                items, a mean of 15.9 and a maximum of 39, where the real items have 4 to 7 levels.
                `imputed_mask_k5.csv` says which cells were imputed, the real levels are read off the
                cells that were not, and each imputed cell is moved to the nearest one. --no-snap
                keeps imputed values as their own categories.
  missingness   the imputed matrix (`responses_k5.csv`), so MCA sees exactly what every other grouping
                here sees. ★ Biji did the opposite and let nonresponse be its own category, and report
                that their axis 1 tracks nonresponse. We cannot copy that: the cohort requires at least
                146 of 151 answered. This is a stated difference, not an oversight.
  keying        irrelevant, and this is a property worth knowing. Reversing an item's scoring direction
                permutes its category labels and leaves the indicator matrix unchanged, so MCA is
                invariant to the keying that `load_battery` applies for every other grouping.
  axes          ★ k COMES FROM HORN'S PARALLEL ANALYSIS, the same rule `comparators.py` uses for
                factor and pca, so the k's are comparable by construction rather than by coincidence.
                Each item's answers are permuted independently, which keeps every category mass and
                therefore the total inertia and destroys only the association between items, and an
                axis is retained when its eigenvalue exceeds the 95th percentile of the null for that
                axis, cutting at the first failure. Biji's 50, and the 20 they put in their models,
                carry no criterion at all -- they are stated without one. The number of axes COMPUTED
                is --axes (default 50). Step 4 of README.md uses 100, because MCA's Horn's k is 90 on
                this CDR and the cut must fall inside the axes computed. No script in this folder uses
                a 20-axis slice.

k_choice.py imports load_categories, fit, horn and inertia_shares from here, with the same settings,
so the Horn's k it reports is the one this script gives.

INPUTS   CFG.responses_csv (k5 by default), CFG.items_csv for the item order
OUTPUTS  screen_out/mca_axes.csv            person x axes principal coordinates. PERSON LEVEL, stays here.
         screen_out/mca_category_coords.csv category x axes, with each category's mass
         screen_out/mca_discrimination.csv  item x axes squared correlation ratios
         screen_out/mca_eigenvalues.csv     eigenvalue, % of inertia, Benzecri-adjusted %, and with
                                            --parallel the null's 95th percentile and keep_horn
         printed checks

Run:  python3 mca_basis.py --axes 100 --parallel 10   (step 4 of README.md: Horn's k inside 100 axes)
      python3 mca_basis.py                            (no parallel analysis, 50 axes written)
      python3 mca_basis.py --axes 50 --cohort k0
      python3 mca_basis.py --no-snap                  (imputed values kept as their own categories)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, svds

from wb_config import CFG

ID_COL = "id"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_out")
# ★ Reject flags this script does not know: a silently ignored --parallel would print a run that looks
# complete with no parallel analysis in it. Unknown flag, hard stop.
# Checked in main() rather than at import, so k_choice.py can import the functions below.
_KNOWN = {"--axes", "--parallel", "--no-snap",
          "--items", "--basis", "--emb-prefix", "--responses-dir", "--cohort"}   # the last 5 are wb_config's


def reject_unknown_flags():
    unknown = [a for a in sys.argv[1:] if a.startswith("--") and a not in _KNOWN]
    if unknown:
        raise SystemExit(f"unknown flag(s): {', '.join(unknown)}. Known: {', '.join(sorted(_KNOWN))}")

N_AXES = int(sys.argv[sys.argv.index("--axes") + 1]) if "--axes" in sys.argv else 50
SNAP = "--no-snap" not in sys.argv
N_PERM = int(sys.argv[sys.argv.index("--parallel") + 1]) if "--parallel" in sys.argv else 0
SEED = 0
J_GLOBAL = [0]


def snap_to_levels(raw, items, mask_csv):
    """Move every imputed cell to the nearest level that was actually answered for that item."""
    if not os.path.exists(mask_csv):
        print(f"[mca_basis] no imputed mask at {mask_csv} -- nothing to snap")
        return raw, 0
    mask = pd.read_csv(mask_csv)
    mask = mask[items].values.astype(bool) if all(c in mask.columns for c in items) else None
    assert mask is not None, "the imputed mask does not cover every item"
    assert mask.shape == raw.shape, f"mask {mask.shape} does not match responses {raw.shape}"
    out = raw.copy()
    moved = 0
    for j in range(raw.shape[1]):
        m = mask[:, j]
        if not m.any():
            continue
        levels = np.unique(raw[~m, j])
        assert levels.size, f"item {items[j]} has no answered cell to take levels from"
        idx = np.abs(raw[m, j][:, None] - levels[None, :]).argmin(axis=1)
        snapped = levels[idx]
        moved += int((snapped != raw[m, j]).sum())
        out[m, j] = snapped
    return out, moved


def build_indicator(raw, items):
    """Indicator matrix, plus the (item, category) index and each person's chosen column per item."""
    n, Q = raw.shape
    cols, offsets, cats = [], [], []
    start = 0
    for j, item in enumerate(items):
        vals = np.unique(raw[:, j])
        code = np.searchsorted(vals, raw[:, j])
        cols.append(code + start)
        offsets.append(start)
        cats += [(item, float(v)) for v in vals]
        start += len(vals)
    chosen = np.column_stack(cols)                       # n x Q, the column each person occupies
    J = start
    data = np.ones(n * Q, dtype=np.float64)
    indptr = np.arange(0, n * Q + 1, Q)
    Z = csr_matrix((data, chosen.ravel(), indptr), shape=(n, J))
    return Z, np.array(cats, dtype=object), chosen, J


def fit(Z, c, n, Q, k):
    """Truncated SVD of the standardised residual matrix. Returns singular values, U, V."""
    inv_sqrt_c, sqrt_c = 1.0 / np.sqrt(c), np.sqrt(c)
    scale = 1.0 / (np.sqrt(n) * Q)

    def mv(x):
        x = np.asarray(x).ravel()
        return scale * (Z @ (inv_sqrt_c * x)) - (sqrt_c @ x) / np.sqrt(n)

    def rmv(y):
        y = np.asarray(y).ravel()
        return scale * (inv_sqrt_c * (Z.T @ y)) - sqrt_c * (y.sum() / np.sqrt(n))

    def mm(X):
        X = np.asarray(X)
        return scale * (Z @ (inv_sqrt_c[:, None] * X)) - np.outer(np.ones(n), sqrt_c @ X) / np.sqrt(n)

    def rmm(Y):
        Y = np.asarray(Y)
        return scale * (inv_sqrt_c[:, None] * (Z.T @ Y)) - np.outer(sqrt_c, Y.sum(axis=0)) / np.sqrt(n)

    S = LinearOperator((n, J_GLOBAL[0]), matvec=mv, rmatvec=rmv, matmat=mm, rmatmat=rmm, dtype=np.float64)
    U, sig, Vt = svds(S, k=k, random_state=SEED)
    order = np.argsort(-sig)
    return sig[order], U[:, order], Vt[order].T


def indicator_from_chosen(chosen, J, n):
    Q = chosen.shape[1]
    return csr_matrix((np.ones(n * Q), chosen.ravel(), np.arange(0, n * Q + 1, Q)), shape=(n, J))


def horn(chosen, c, n, Q, J, k, n_perm, seed):
    """Horn's parallel analysis for MCA: permute each item's answers independently, which keeps every
    category mass and so the total inertia, and destroys only the association between items. Retain
    the leading axes whose eigenvalue exceeds the 95th percentile of the null for that axis, cutting
    at the first failure. This is the rule `comparators.py` already uses for factor and pca."""
    rng = np.random.default_rng(seed)
    null = np.zeros((n_perm, k))
    for p in range(n_perm):
        perm = np.column_stack([chosen[rng.permutation(n), j] for j in range(Q)])
        sig, _, _ = fit(indicator_from_chosen(perm, J, n), c, n, Q, k)
        null[p] = sig ** 2
        print(f"    permutation {p + 1}/{n_perm}: largest null eigenvalue {null[p, 0]:.5f}", flush=True)
    return np.percentile(null, 95, axis=0), null


def inertia_shares(lam, J, Q):
    """Raw % of the total inertia (J - Q) / Q, which is exact however few axes are computed, and the
    Benzecri-adjusted %, which keeps only axes above 1/Q and is computed over the axes given."""
    pct = 100.0 * lam / ((J - Q) / Q)
    keep = lam > 1.0 / Q                            # Benzecri keeps axes above the 1/Q threshold
    ben = np.where(keep, ((Q / (Q - 1.0)) * (lam - 1.0 / Q)) ** 2, 0.0)
    ben_pct = 100.0 * ben / ben.sum() if ben.sum() > 0 else np.zeros_like(ben)
    return pct, ben_pct, keep


def load_categories(items):
    """The response matrix with imputed cells snapped to answered levels (unless --no-snap), and the
    indicator matrix built from it. Shared with k_choice.py so both see the same categories."""
    resp = pd.read_csv(CFG.responses_csv)
    resp[ID_COL] = resp[ID_COL].astype(str)
    resp = resp.set_index(ID_COL)
    missing = [c for c in items if c not in resp.columns]
    assert not missing, f"{len(missing)} item columns absent from the responses: {missing[:5]}"
    raw = resp[items].values.astype(float)
    assert not np.isnan(raw).any(), "missing cells in the response matrix -- MCA here expects the imputed k5 file"
    n, Q = raw.shape
    if SNAP:
        before = sum(len(np.unique(raw[:, j])) for j in range(Q))
        raw, moved = snap_to_levels(raw, items, CFG.mask_csv)
        after = sum(len(np.unique(raw[:, j])) for j in range(Q))
        print(f"[mca_basis] snapped {moved:,} imputed cells to the nearest answered level; "
              f"categories {before:,} -> {after:,}")
    else:
        print("[mca_basis] --no-snap: imputed values keep their own categories (NOT recommended)")
    Z, cats, chosen, J = build_indicator(raw, items)
    per_item = np.diff(np.r_[[0], np.cumsum([len(np.unique(raw[:, j])) for j in range(Q)])])
    print(f"[mca_basis] {n:,} participants x {Q} items -> {J} answer categories "
          f"(mean {J / Q:.1f} per item, min {per_item.min()}, max {per_item.max()})")
    # P = Z/N with N = n*Q; row masses are all 1/n; column masses c_j = n_j/N.
    nj = np.asarray(Z.sum(axis=0)).ravel()
    assert (nj > 0).all()
    c = nj / float(n * Q)
    J_GLOBAL[0] = J
    return resp, Z, cats, chosen, J, c, n, Q


def main():
    reject_unknown_flags()
    os.makedirs(OUT, exist_ok=True)
    items = pd.read_csv(CFG.items_csv)["item"].astype(str).tolist()
    resp, Z, cats, chosen, J, c, n, Q = load_categories(items)

    # --- correspondence analysis of the indicator matrix -------------------------------------------
    # S = D_r^-1/2 (P - r c') D_c^-1/2 = A - (1/sqrt n) 1 sqrt(c)',  A = Z D_c^-1/2 / (sqrt(n) Q)
    inv_sqrt_c = 1.0 / np.sqrt(c)

    k = min(N_AXES, J - Q - 1)
    print(f"[mca_basis] truncated SVD for {k} axes (total inertia {(J - Q) / Q:.3f})")
    sig, U, V = fit(Z, c, n, Q, k)
    lam = sig ** 2
    F = np.sqrt(n) * U * sig                        # person principal coordinates, n x k
    G = (inv_sqrt_c[:, None] * V) * sig             # category principal coordinates, J x k

    pct, ben_pct, keep = inertia_shares(lam, J, Q)

    # --- the check the rest of the analysis depends on ---------------------------------------------
    # F[i,k] must equal (1/(Q sigma_k)) * sum_q G[chosen(i,q), k]. If this fails, the T^2 partition
    # cannot be carried over to MCA and nothing downstream should be wired up.
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n, size=min(2000, n), replace=False)
    recon = G[chosen[idx]].sum(axis=1) / (Q * sig)
    err = np.abs(recon - F[idx])
    rel = err.max() / max(np.abs(F[idx]).max(), 1e-12)
    print(f"[check] transition formula on {len(idx):,} participants: max abs error {err.max():.3e}, "
          f"relative {rel:.3e}")
    assert rel < 1e-8, "the per-item terms do not reproduce the person coordinate -- do NOT wire MCA in"

    # discrimination measures: eta^2 of each item with each axis, computed from the coordinates
    # themselves so no scaling convention can be got wrong. The identity mean_q eta^2_qk = lambda_k
    # is printed as a second check.
    var_k = F.var(axis=0)
    eta = np.zeros((Q, k))
    for j in range(Q):
        code = chosen[:, j] - chosen[:, j].min()
        cnt = np.bincount(code)
        means = np.zeros((len(cnt), k))
        np.add.at(means, code, F)
        means /= cnt[:, None]
        eta[j] = ((cnt[:, None] / n) * means ** 2).sum(axis=0) / var_k
    ident = np.abs(eta.mean(axis=0) - lam).max()
    print(f"[check] mean discrimination equals the eigenvalue: max |mean_q eta^2 - lambda| = {ident:.3e}")

    horn_k, null95 = None, None
    if N_PERM > 0:
        print(f"\n[horn] parallel analysis, {N_PERM} permutations of each item's answers")
        null95, null = horn(chosen, c, n, Q, J, k, N_PERM, SEED)
        passes = lam > null95
        horn_k = int(np.argmax(~passes)) if (~passes).any() else k
        print(f"  {'axis':>5} {'eigenvalue':>11} {'null 95th':>11} {'keep':>6}")
        lo, hi = max(0, horn_k - 5), min(k, horn_k + 5)
        for a in range(lo, hi):
            print(f"  {a + 1:>5} {lam[a]:>11.5f} {null95[a]:>11.5f} {'yes' if passes[a] else 'no':>6}")
        print(f"★ HORN'S k = {horn_k}" + ("" if horn_k < k else f"  -- the cut is at or beyond the {k} axes "
              f"computed, so rerun with a larger --axes before using it"))

    print("\n  axis  eigenvalue   % inertia   Benzecri %   top items by discrimination")
    for a in range(min(8, k)):
        top = ", ".join(np.array(items)[np.argsort(-eta[:, a])[:4]])
        print(f"  {a + 1:>4}  {lam[a]:>10.4f}  {pct[a]:>10.2f}  {ben_pct[a]:>10.2f}   {top}")

    pd.DataFrame(F, index=resp.index, columns=[f"mca{a + 1}" for a in range(k)]).rename_axis(ID_COL) \
        .reset_index().to_csv(os.path.join(OUT, "mca_axes.csv"), index=False)
    pd.DataFrame(G, columns=[f"mca{a + 1}" for a in range(k)]).assign(
        item=[t[0] for t in cats], value=[t[1] for t in cats], mass=c) \
        .to_csv(os.path.join(OUT, "mca_category_coords.csv"), index=False)
    pd.DataFrame(eta, index=items, columns=[f"mca{a + 1}" for a in range(k)]).rename_axis("item") \
        .reset_index().to_csv(os.path.join(OUT, "mca_discrimination.csv"), index=False)
    eig = pd.DataFrame({"axis": np.arange(1, k + 1), "eigenvalue": lam, "pct_inertia": pct,
                        "benzecri_pct": ben_pct, "above_1_over_Q": keep})
    if null95 is not None:
        eig["null_95th"] = null95
        eig["keep_horn"] = lam > null95
        eig.attrs["horn_k"] = horn_k
    eig.to_csv(os.path.join(OUT, "mca_eigenvalues.csv"), index=False)
    print(f"\nwrote mca_axes.csv (person level, stays on the Workbench), mca_category_coords.csv, "
          f"mca_discrimination.csv, mca_eigenvalues.csv to {OUT}")


if __name__ == "__main__":
    main()
