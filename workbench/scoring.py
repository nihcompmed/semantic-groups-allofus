"""scoring.py -- how a participant is scored on a grouping and how far from typical they are.

Every grouping is scored the same way; only the matrix that maps the answers to scores differs:
  X     the answers, each item rescaled to [-1, 1] on its own scale and keyed so that +1 is the instrument's keyed
        direction (load_battery)
  P     items x k: a semantic grouping's item weights, P = (E - mu) W' from the item embeddings E and the basis
        package (place); a factor model's regression-score weights; PCA's loadings; MCA's category coordinates
  Y     X P, the k scores; residualized on age and sex by least squares (wls_residualize)
  T2    the squared Mahalanobis distance of the residualized scores under a Ledoit-Wolf precision (mahalanobis2)
  distance = sqrt(T2); an outlier is above the grouping's own 95th percentile of distance (score_placed)
The squared distance divides exactly among the k scores (share of score j = y_j (Omega y)_j / T2) and among the 151
items (a_i = x_i (Q x)_i with Q = P Omega P'), which is how the steps attribute a participant's distance.
"""
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FactorAnalysis, PCA

from settings import BASIS, COHORT_CSV, COVARIATES, EMB_PREFIX, ID, ITEMS_CSV, MASK_CSV, PCTILE

SEED = 0


# ------------------------------------------------------------------------------------------------- the answers
def items_table():
    """The item table and the item order."""
    T = pd.read_csv(ITEMS_CSV)
    return T, T["item"].astype(str).tolist()


def load_battery(items=None):
    """(resp, X, cov, w): the cohort file indexed by id (as text), the keyed rescaled answers, age and sex, and unit
    weights."""
    T = pd.read_csv(ITEMS_CSV)
    items = items or T["item"].astype(str).tolist()
    scales = (T.set_index("item").reindex(items)
                .rename(columns={"scale_min": "min", "scale_max": "max", "reverse_coded": "reverse"}))
    assert scales["min"].notna().all(), "the item table does not cover every item"
    resp = pd.read_csv(COHORT_CSV)
    resp[ID] = resp[ID].astype(str)
    resp = resp.set_index(ID)
    missing = [c for c in items if c not in resp.columns]
    assert not missing, f"{len(missing)} item columns absent from the cohort file: {missing[:5]}"
    raw = resp[items].values.astype(float)
    assert not np.isnan(raw).any(), "missing answers in the cohort file"
    lo = scales["min"].values.astype(float)
    hi = scales["max"].values.astype(float)
    rev = scales["reverse"].astype(str).str.lower().eq("true").values
    X = np.clip(2.0 * (raw - lo) / (hi - lo) - 1.0, -1.0, 1.0)
    X[:, rev] *= -1.0                                                       # +1 = the instrument's keyed direction
    cov = resp[COVARIATES].values.astype(float)
    assert not np.isnan(cov).any(), "age or sex is missing for someone"
    w = np.ones(len(resp))
    return resp, X, cov, w


# ------------------------------------------------------------------------------------------------- the distance
def wls_residualize(Y, C, w):
    """Each column of Y residualized on C (with an intercept) by weighted least squares."""
    if C is None or C.shape[1] == 0:
        return Y - np.average(Y, axis=0, weights=w)
    A = np.column_stack([np.ones(len(Y)), C])
    sw = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(A * sw, Y * sw, rcond=None)
    return Y - A @ beta


def residualizer(C, w):
    """wls_residualize as a function of the matrix, the design built once."""
    A = np.column_stack([np.ones(len(w))] + ([C] if C is not None and C.shape[1] else []))
    sw = np.sqrt(w)[:, None]

    def apply(V):
        beta, *_ = np.linalg.lstsq(A * sw, V * sw, rcond=None)
        return V - A @ beta
    return apply


def mahalanobis2(Yr, w):
    """Squared Mahalanobis distance under a Ledoit-Wolf precision; returns (T2, the fitted LedoitWolf)."""
    ctr = Yr - np.average(Yr, axis=0, weights=w)
    lw = LedoitWolf().fit(ctr)
    return np.einsum("ij,ij->i", ctr, ctr @ lw.precision_), lw


def place(enc, items, basis_dir=None):
    """P = (E - mu) W' for the items in order, from basis_dir/<enc>.npz and its item embeddings; and the package."""
    basis_dir = basis_dir or BASIS
    b = np.load(os.path.join(basis_dir, f"{enc}.npz"), allow_pickle=True)
    e = np.load(os.path.join(basis_dir, "item_embeddings", f"{EMB_PREFIX}_{enc}.npz"), allow_pickle=True)
    pos = {str(x): j for j, x in enumerate(e["items"])}
    absent = [it for it in items if it not in pos]
    assert not absent, f"{enc}: {len(absent)} items have no embedding: {absent[:5]}"
    E = np.asarray(e["vectors"], float)[[pos[it] for it in items]]
    return (E - b["mu"]) @ b["W"].T, b


def score_placed(P, X, cov, w):
    """distance, squared distance, percentile and outlier flag for one placed grouping."""
    Yr = wls_residualize(X @ P, cov, w)
    T2, _ = mahalanobis2(Yr, w)
    d = np.sqrt(np.maximum(T2, 0.0))
    pct = pd.Series(d).rank(pct=True).values * 100.0
    return d, T2, pct, d > np.percentile(d, PCTILE)


def scores_frame(ids, d, T2, pct, flag):
    return pd.DataFrame({ID: ids, "distance": d, "distance2": T2, "pctile": pct, "flagged": flag})


# ------------------------------------------------------------------------------------------------- answer-based groupings
def parallel_analysis_k(Xs, n_perm, pctile, seed):
    """Horn's parallel analysis on the correlation matrix, with a permutation null (Buja and Eyuboglu)."""
    n = len(Xs)
    real = np.sort(np.linalg.eigvalsh(Xs.T @ Xs / (n - 1)))[::-1]
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, Xs.shape[1]))
    for b in range(n_perm):
        Xp = rng.permuted(Xs, axis=0)
        null[b] = np.sort(np.linalg.eigvalsh(Xp.T @ Xp / (n - 1)))[::-1]
    thr = np.percentile(null, pctile, axis=0)
    below = np.where(real <= thr)[0]
    k = int(below[0]) if below.size else Xs.shape[1]
    return k, real, thr


def factor_model(Xs, k, seed):
    """Maximum-likelihood common factors with varimax; the regression-score weights S (items x k), checked against
    transform()."""
    fa = FactorAnalysis(n_components=k, rotation="varimax", random_state=seed).fit(Xs)
    Wpsi = fa.components_ / fa.noise_variance_
    cov_z = np.linalg.inv(np.eye(k) + Wpsi @ fa.components_.T)
    S = Wpsi.T @ cov_z
    chk = Xs[:2000]
    assert np.allclose(fa.transform(chk), (chk - fa.mean_) @ S, atol=1e-8), "factor-score weights do not reproduce"
    return S, fa.components_.T, {"n_iter": int(fa.n_iter_), "loglike_final": float(fa.loglike_[-1]),
                                 "converged": bool(fa.n_iter_ < fa.max_iter)}


def pca_model(Xs, k):
    p = PCA(n_components=k, svd_solver="full").fit(Xs)
    return p.components_.T, {"explained_variance_ratio": float(p.explained_variance_ratio_.sum())}


def score_linear(S_eff, Xr, w):
    """The distance for a linear grouping Y = X S_eff, X already residualized."""
    Yc = Xr @ S_eff
    Yc = Yc - np.average(Yc, axis=0, weights=w)
    Om = LedoitWolf().fit(Yc).precision_
    T2 = np.einsum("ij,ij->i", Yc, Yc @ Om)
    d = np.sqrt(np.maximum(T2, 0.0))
    return d, T2, pd.Series(d).rank(pct=True).values * 100.0, d > np.percentile(d, PCTILE), Om


def pr(v):
    """The participation ratio of a vector: (sum v^2)^2 / sum v^4."""
    s2 = float((v ** 2).sum())
    s4 = float((v ** 4).sum())
    return s2 * s2 / s4 if s4 > 0 else 0.0


def fixed_point_core(v):
    """The core of a column of weights: its largest entries, as many as the participation ratio of those entries,
    repeated until the set no longer changes."""
    idx = np.arange(len(v))
    while True:
        n = max(1, min(int(round(pr(v[idx]))), len(idx)))
        top = np.sort(idx[np.argsort(-np.abs(v[idx]))[:n]])
        if len(top) == len(idx) and (top == idx).all():
            return top
        idx = top


# ------------------------------------------------------------------------------------------------- MCA
def snap_to_levels(raw, items, mask_csv):
    """Every imputed answer moved to the nearest level that was actually answered for that item."""
    mask = pd.read_csv(mask_csv)
    mask = mask[items].values.astype(bool)
    assert mask.shape == raw.shape, f"mask {mask.shape} does not match the answers {raw.shape}"
    out = raw.copy()
    for j in range(raw.shape[1]):
        m = mask[:, j]
        if not m.any():
            continue
        levels = np.unique(raw[~m, j])
        assert levels.size, f"item {items[j]} has no answered cell to take levels from"
        idx = np.abs(raw[m, j][:, None] - levels[None, :]).argmin(axis=1)
        out[m, j] = levels[idx]
    return out


def build_indicator(raw, items):
    """The indicator matrix (one column per answer category), the (item, value) of each column and each person's
    column per item."""
    n, Q = raw.shape
    cols, cats = [], []
    start = 0
    for j, item in enumerate(items):
        vals = np.unique(raw[:, j])
        cols.append(np.searchsorted(vals, raw[:, j]) + start)
        cats += [(item, float(v)) for v in vals]
        start += len(vals)
    chosen = np.column_stack(cols)
    J = start
    Z = csr_matrix((np.ones(n * Q, dtype=np.float64), chosen.ravel(), np.arange(0, n * Q + 1, Q)), shape=(n, J))
    return Z, np.array(cats, dtype=object), chosen, J


def mca_fit(Z, c, n, Q, J, k):
    """Truncated SVD of the standardized residual matrix of correspondence analysis: singular values, U, V."""
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

    S = LinearOperator((n, J), matvec=mv, rmatvec=rmv, matmat=mm, rmatmat=rmm, dtype=np.float64)
    U, sig, Vt = svds(S, k=k, random_state=SEED)
    order = np.argsort(-sig)
    return sig[order], U[:, order], Vt[order].T


def mca_horn(chosen, c, n, Q, J, k, n_perm, seed):
    """Horn's parallel analysis for MCA: each item's answers permuted independently (every category's mass, and so the
    total inertia, kept); the 95th percentile of the null eigenvalue of each axis."""
    rng = np.random.default_rng(seed)
    null = np.zeros((n_perm, k))
    for p in range(n_perm):
        perm = np.column_stack([chosen[rng.permutation(n), j] for j in range(Q)])
        Zp = csr_matrix((np.ones(n * Q), perm.ravel(), np.arange(0, n * Q + 1, Q)), shape=(n, J))
        sig, _, _ = mca_fit(Zp, c, n, Q, J, k)
        null[p] = sig ** 2
        print(f"    permutation {p + 1}/{n_perm}", flush=True)
    return np.percentile(null, 95, axis=0), null


def inertia_shares(lam, J, Q):
    """The raw % of the total inertia (J - Q) / Q, and the Benzecri-adjusted %, over the axes given."""
    pct = 100.0 * lam / ((J - Q) / Q)
    keep = lam > 1.0 / Q
    ben = np.where(keep, ((Q / (Q - 1.0)) * (lam - 1.0 / Q)) ** 2, 0.0)
    ben_pct = 100.0 * ben / ben.sum() if ben.sum() > 0 else np.zeros_like(ben)
    return pct, ben_pct, keep


def load_categories(items):
    """The answers with imputed cells snapped to answered levels, and the indicator matrix built from them."""
    resp = pd.read_csv(COHORT_CSV)
    resp[ID] = resp[ID].astype(str)
    resp = resp.set_index(ID)
    raw = resp[items].values.astype(float)
    n, Q = raw.shape
    raw = snap_to_levels(raw, items, MASK_CSV)
    Z, cats, chosen, J = build_indicator(raw, items)
    nj = np.asarray(Z.sum(axis=0)).ravel()
    assert (nj > 0).all()
    c = nj / float(n * Q)
    print(f"[mca] {n:,} participants x {Q} items -> {J} answer categories", flush=True)
    return resp, Z, cats, chosen, J, c, n, Q


def mca_representation(items, resp_raw, C, E, k):
    """MCA at k from its category coordinates C and eigenvalues E: Gs (category x k, divided by Q sigma) and each
    person's category per item, so that a person's coordinate is the sum over items of Gs[chosen]. The levels are
    those listed in C (imputed answers snapped to the nearest one)."""
    axes = [c for c in C.columns if c.startswith("mca")][:k]
    sigma = np.sqrt(E["eigenvalue"].values[:k])
    Q = len(items)
    G = C[axes].values / (Q * sigma[None, :])
    chosen = np.empty((len(resp_raw), Q), dtype=np.int64)
    for j, item in enumerate(items):
        rows = np.where(C["item"].values == item)[0]
        assert rows.size, f"{item} has no categories"
        levels = C["value"].values[rows].astype(float)
        chosen[:, j] = rows[np.abs(resp_raw[:, j][:, None] - levels[None, :]).argmin(axis=1)]
    return G, chosen


def score_mca(G, chosen, w, resid):
    """The distance for MCA: the per-item terms summed, residualized, centered, Ledoit-Wolf, T2."""
    Yr = resid(G[chosen].sum(axis=1))
    Yr = Yr - np.average(Yr, axis=0, weights=w)
    Om = LedoitWolf().fit(Yr).precision_
    T2 = np.einsum("ij,ij->i", Yr, Yr @ Om)
    d = np.sqrt(np.maximum(T2, 0.0))
    return d, T2, pd.Series(d).rank(pct=True).values * 100.0, d > np.percentile(d, PCTILE)
