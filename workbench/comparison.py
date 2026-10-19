"""comparison.py -- the pieces of the comparison between the semantic groups and the factor model.

THE FACTOR MODELS. Maximum likelihood on the standardized answers at k = 25 (Horn) and k = 3 (Cattell-Nelson-Gorsuch).
A participant's distance does not depend on the rotation: with A the unrotated loadings and Y0 the unrotated regression
scores, the varimax scores are Y0 R (R orthogonal) and the oblimin scores Y0 T, and the Ledoit-Wolf precision rotates with
them, so every participant keeps the same squared distance (checked against step 2's scores). The rotation decides only
how the distance divides among the factors, so factor-level results are written under both rotations. Oblimin: direct
oblimin with gamma = 0 (quartimin), the gradient projection algorithm for oblique rotations (Bernaards and Jennrich
2005, as GPArotation's GPFoblq), no Kaiser normalization, from the identity and 10 random orthonormal starts, the lowest
criterion kept. Factors are named as the semantic groups are, by the constructs of their core items.

WHAT CARRIES A DISTANCE. The squared distance divides exactly among the 151 items, a_i = x_i (Q x)_i with
Q = S Omega S'; summed by source (each published questionnaire, and each stand-alone question by the study it was taken
from: 17 questionnaires and 8 source studies), as a share of the squared distance. nsrc_half = the sources, largest
first, needed to carry half; "one questionnaire carries half" when nsrc_half = 1.

THE POLICY. Top-1 and top-3 counts of the units (groups or factors) partition a set's n or 3 x n slots, and a grouping's
own outliers = both + one-grouping-only, so the counts are hidden as linked systems (secure_masks): a hidden count of 1 to
20 that the written numbers determine forces one more hidden cell, until none is determined.
"""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FactorAnalysis

import scoring as sc
from release import pair_hidden, partition_mask, small
from settings import ID, PCTILE, SOURCES_CSV, work

SETS = ["both", "enc_only", "fac_only"]
ROTATIONS = ["oblimin", "varimax"]
N_STARTS, ROT_SEED = 10, 0
EPS, MAXIT = 1e-5, 20000


# ------------------------------------------------------------------------------------------------- oblimin
def vgq_oblimin(L, gam=0.0):
    """The oblimin criterion and its gradient with respect to the loadings (GPArotation's vgQ.oblimin)."""
    k = L.shape[1]
    X = (L ** 2) @ (np.ones((k, k)) - np.eye(k))
    if gam != 0:
        p = L.shape[0]
        X = (np.eye(p) - np.full((p, p), gam / p)) @ X
    return L * X, float((L ** 2 * X).sum() / 4.0)


def gpf_oblq(A, T0, eps=EPS, maxit=MAXIT):
    """Gradient projection for an oblique rotation of A from the start T0. Returns the pattern loadings, Phi, T, the
    criterion, the iterations and whether it converged."""
    T = T0.copy()
    al = 1.0
    L = A @ np.linalg.inv(T).T
    Gq, f = vgq_oblimin(L)
    G = -(L.T @ Gq @ np.linalg.inv(T)).T
    it, s = 0, np.inf
    for it in range(maxit + 1):
        Gp = G - T @ np.diag((T * G).sum(0))
        s = float(np.sqrt(np.trace(Gp.T @ Gp)))
        if s < eps:
            break
        al *= 2.0
        for _ in range(11):
            X = T - al * Gp
            v = 1.0 / np.sqrt((X ** 2).sum(0))
            Tt = X @ np.diag(v)
            Lt = A @ np.linalg.inv(Tt).T
            Gqt, ft = vgq_oblimin(Lt)
            if f - ft > 0.5 * s ** 2 * al:
                break
            al /= 2.0
        T, L, Gq, f = Tt, Lt, Gqt, ft
        G = -(L.T @ Gq @ np.linalg.inv(T)).T
    return L, T.T @ T, T, f, it, bool(s < eps)


def oblimin(A, n_starts=N_STARTS, seed=ROT_SEED):
    """The identity and n_starts random orthonormal starts; the lowest criterion kept, each factor's largest loading
    made positive."""
    k = A.shape[1]
    rng = np.random.default_rng(seed)
    starts = [np.eye(k)] + [np.linalg.qr(rng.standard_normal((k, k)))[0] for _ in range(n_starts)]
    runs = [gpf_oblq(A, T0) for T0 in starts]
    fs = np.array([r[3] for r in runs])
    best = int(np.argmin(fs))
    reached = int((fs <= fs[best] + 1e-6 * max(1.0, abs(fs[best]))).sum())
    L, Phi, T, f, it, conv = runs[best]
    sgn = np.sign(L[np.abs(L).argmax(0), np.arange(k)])
    sgn[sgn == 0] = 1.0
    L, T = L * sgn, T * sgn
    info = {"criterion": f, "iterations": int(it), "converged": conv,
            "start_kept": "identity" if best == 0 else f"random {best}", "starts_total": len(starts),
            "starts_reaching_lowest": reached}
    return L, T.T @ T, T, info


# ------------------------------------------------------------------------------------------------- distances divided
def item_division(Xc, S, Om):
    a = Xc * (Xc @ (S @ Om @ S.T))
    return a, a.sum(1)


def by_source(a, T2, src_idx, n_src):
    return np.column_stack([a[:, src_idx == j].sum(1) for j in range(n_src)]) / T2[:, None]


def n_half(sh):
    return (np.cumsum(-np.sort(-sh, axis=1), axis=1) < 0.5).sum(1) + 1


def nh_for(a, T2, labels):
    """nsrc_half per participant with the given source label per item."""
    src = list(dict.fromkeys(labels))
    idx = np.array([src.index(s) for s in labels])
    return n_half(by_source(a, T2, idx, len(src)))


def profile(core, con):
    return "; ".join(f"{k}:{v}" for k, v in Counter(con[i] for i in core).most_common())


def per_question_labels(T):
    """Each stand-alone question its own source (30 sources): the sources of the factors' and groups' cores."""
    return [f"single:{i}" if c == "single item" else str(c) for i, c in zip(T["item"].astype(str), T["instrument_code"])]


def source_labels(T):
    """Each stand-alone question under the study it was taken from (standalone_sources.csv; 25 sources)."""
    M = pd.read_csv(SOURCES_CSV).set_index("item")["source"]
    single = T["instrument"].astype(str).eq("single item") if "instrument" in T else T["instrument_code"].eq("single item")
    items = T["item"].astype(str)
    assert set(items[single.values]) == set(M.index), "standalone_sources.csv must cover exactly the stand-alone questions"
    assert not set(T["instrument_code"].astype(str)[~single.values]) & set(M.values), "a source study shares a label"
    return [M[i] if s else str(c) for i, s, c in zip(items, single, T["instrument_code"].astype(str))]


# ------------------------------------------------------------------------------------------------- the factor models
def factor_model(name, k, scores_file, Xs, sd, Xc, w, idx, items, con, src_name):
    """The unrotated fit, the 2 rotations, the rotation-free distance (checked against the scores file), each
    rotation's shares, factor names and core sources, and the item division."""
    f0 = FactorAnalysis(n_components=k, rotation=None, random_state=sc.SEED).fit(Xs)
    fv = FactorAnalysis(n_components=k, rotation="varimax", random_state=sc.SEED).fit(Xs)
    psi = f0.noise_variance_
    assert np.allclose(psi, fv.noise_variance_, rtol=1e-8), f"{name}: the 2 fits differ"
    A, Wv = f0.components_.T, fv.components_.T
    Rm = np.linalg.lstsq(A, Wv, rcond=None)[0]
    assert np.allclose(A @ Rm, Wv, atol=1e-8) and np.allclose(Rm.T @ Rm, np.eye(k), atol=1e-8), \
        f"{name}: R is not the varimax rotation"
    Wpsi = A.T / psi
    S0 = (Wpsi.T @ np.linalg.inv(np.eye(k) + Wpsi @ A)) / sd[:, None]
    Y0 = Xc @ S0
    Y0 = Y0 - np.average(Y0, axis=0, weights=w)
    Om0 = LedoitWolf().fit(Y0).precision_
    T2 = np.einsum("ij,ij->i", Y0, Y0 @ Om0)
    s = pd.read_csv(work(scores_file))
    s[ID] = s[ID].astype(str)
    worst = float(np.max(np.abs(np.sqrt(np.maximum(T2, 0)) - s.set_index(ID)["distance"].reindex(idx).values)))
    assert worst < 1e-6, f"{name}: the rotation-free distance differs from {scores_file} by {worst}"
    Lo, Phi, Tm, info = oblimin(A)
    Ti = np.linalg.inv(Tm)
    rot = {"oblimin": (Y0 @ Tm, Ti @ Om0 @ Ti.T, Lo), "varimax": (Y0 @ Rm, Rm.T @ Om0 @ Rm, Wv)}
    out = {}
    for r, (Y, Om, L) in rot.items():
        assert np.allclose(np.einsum("ij,ij->i", Y, Y @ Om), T2, rtol=1e-6), f"{name} {r}: the distance changed"
        u = (Y * (Y @ Om)) / T2[:, None]
        assert np.abs(u.sum(1) - 1).max() < 1e-6, f"{name} {r}: shares do not sum to 1"
        cores = [sc.fixed_point_core(L[:, j]) for j in range(k)]
        out[r] = {"u": u, "names": [profile(c, con) for c in cores],
                  "sources": ["; ".join(dict.fromkeys(src_name[i] for i in c)) for c in cores],
                  "core_items": [" ".join(items[i] for i in c) for c in cores],
                  "spans": [len(set(src_name[i] for i in c)) >= 2 for c in cores]}
    a, T2i = item_division(Xc, S0, Om0)
    assert np.allclose(T2i, T2, rtol=1e-6), f"{name}: the item division does not reproduce the squared distance"
    return {"T2": T2, "a": a, "rot": out, "oblimin": info, "check": f"max |difference| {worst:.1e}",
            "A": A, "Wv": Wv, "Rm": Rm, "Tm": Tm, "Lo": Lo, "psi": psi, "S0": S0}


def outliers(name, idx):
    """A grouping's outliers: distance above its own 95th percentile (equal to its flagged column where there is one)."""
    s = pd.read_csv(work(f"scores_{name}.csv"))
    s[ID] = s[ID].astype(str)
    s = s.set_index(ID).reindex(idx)
    d = s["distance"].values.astype(float)
    assert np.isfinite(d).all(), f"{name}: a distance is missing for someone in the cohort"
    m = d > np.percentile(d, PCTILE)
    flagged = "flagged" in s.columns
    if flagged:
        assert (s["flagged"].astype(bool).values == m).all(), f"{name}: the outliers differ from the flagged column"
    return m, flagged


# ------------------------------------------------------------------------------------------------- unit counts
def top_counts(U, m):
    """(n, top-1 counts, top-3 counts) of the units over one set."""
    n = int(m.sum())
    kk = U.shape[1]
    if n == 0:
        return 0, np.zeros(kk, np.int64), np.zeros(kk, np.int64)
    order = np.argsort(-U[m], axis=1)
    t1 = np.array([int((order[:, 0] == j).sum()) for j in range(kk)])
    t3 = np.array([int((order[:, :min(3, kk)] == j).any(1).sum()) for j in range(kk)])
    assert t1.sum() == n and t3.sum() == min(3, kk) * n, "top-1/top-3 counts do not partition the set"
    return n, t1, t3


def _identifiable(H, blocks, links):
    """The hidden cells the written numbers determine (unknowns: hidden cells; equations: each block's written sum
    and, unit by unit, total = part 1 + part 2 for each link): those where every null-space vector is 0."""
    var = [(b, j) for b in blocks for j in np.flatnonzero(H[b])]
    if not var:
        return []
    col = {v: i for i, v in enumerate(var)}
    rows = []
    for b in blocks:
        r = np.zeros(len(var))
        for j in np.flatnonzero(H[b]):
            r[col[(b, j)]] = 1.0
        rows.append(r)
    for t, p1, p2 in links:
        for j in range(len(blocks[t])):
            r = np.zeros(len(var))
            for b, sgn in ((t, 1.0), (p1, -1.0), (p2, -1.0)):
                if H[b][j]:
                    r[col[(b, j)]] = sgn
            if r.any():
                rows.append(r)
    M = np.array(rows)
    _, sv, Vt = np.linalg.svd(M, full_matrices=True)
    rank = int((sv > 1e-9 * max(1.0, sv.max())).sum())
    Nul = Vt[rank:]
    return [v for v in var if (Nul.shape[0] == 0 or np.abs(Nul[:, col[v]]).max() < 1e-9)]


def secure_masks(blocks, links, fixed):
    """Hide masks for linked count vectors: blocks {name: counts}, each a partition of a written total; links (total,
    part 1, part 2); fixed: blocks whose mask is written elsewhere. Start from partition_mask, then hide one more written
    cell next to any determined hidden count of 1 to 20, until none is determined."""
    H = {b: partition_mask(v)[0] for b, v in blocks.items()}
    linked = {b: set() for b in blocks}
    for t, p1, p2 in links:
        for a in (t, p1, p2):
            linked[a] |= {t, p1, p2} - {a}
    while True:
        bad = [(b, j) for b, j in _identifiable(H, blocks, links) if 0 < int(blocks[b][j]) <= 20]
        if not bad:
            return H
        b, j = bad[0]
        cand = [(o, j) for o in sorted(linked[b], key=str) if o not in fixed and not H[o][j]]
        if not cand and b not in fixed:
            cand = [(b, i) for i in np.flatnonzero(~H[b] & (blocks[b] > 0))]
        if not cand:
            cand = [(o, i) for o in sorted(linked[b], key=str) if o not in fixed
                    for i in np.flatnonzero(~H[o] & (blocks[o] > 0))]
        assert cand, f"no written cell left to hide next to {b} unit {j + 1}"
        o, i = min(cand, key=lambda c: int(blocks[c[0]][c[1]]))
        H[o][i] = True
        for x, v in blocks.items():
            while x not in fixed and 0 < int(v[H[x]].sum()) <= 20:
                c2 = np.flatnonzero(~H[x] & (v > 0))
                if not len(c2):
                    break
                H[x][c2[np.argmin(v[c2])]] = True


def unit_rows(n, t1, t3, ms, labels, srcs, spans, masks=None):
    """Per unit: mean share, top 1 and top 3 (hidden by masks, or by partition_mask). Empty for a set of 0 or 1 to 20."""
    if n == 0 or small(n):
        return []
    h1, h3 = masks if masks is not None else (partition_mask(t1)[0], partition_mask(t3)[0])
    return [{"unit": j + 1, "name": labels[j], "core_sources": srcs[j], "spans": bool(spans[j]), "n_set": n,
             "mean_share": float(ms[j]),
             "n_top1": np.nan if h1[j] else int(t1[j]), "pct_top1": np.nan if h1[j] else 100 * t1[j] / n,
             "n_top3": np.nan if h3[j] else int(t3[j]), "pct_top3": np.nan if h3[j] else 100 * t3[j] / n}
            for j in range(len(t1))]


def breadth_row(m, nh, span_share):
    """Sources carrying half and the share carried by groups spanning sources, over one set (blank for 1 to 20)."""
    n = int(m.sum())
    if n == 0 or small(n):
        return {"nsrc_half_mean": np.nan, "nsrc_half_q1": np.nan, "nsrc_half_median": np.nan, "nsrc_half_q3": np.nan,
                "one_source_half_pct": np.nan, "spanning_share_mean": np.nan}
    q = np.percentile(nh[m], [25, 50, 75])
    one = int((nh[m] == 1).sum())
    return {"nsrc_half_mean": float(nh[m].mean()), "nsrc_half_q1": q[0], "nsrc_half_median": q[1],
            "nsrc_half_q3": q[2], "one_source_half_pct": np.nan if pair_hidden(one, n) else 100 * one / n,
            "spanning_share_mean": float(span_share[m].mean()) if span_share is not None else np.nan}


# ------------------------------------------------------------------------------------------------- derivable counts
def identifiable(masks):
    """Every group of participants whose count of any indicator is determined by counts over the given sets: the atoms
    of the sets' memberships, and every union of atoms whose 0/1 vector lies in the row space of the set-by-atom
    matrix (complete: a 0/1 vector in the row space is fixed by its values on r pivot atoms, so the 2^r choices are all
    the candidates)."""
    from scipy.linalg import qr
    M = np.array(masks)
    keys, inv = np.unique(M.T, axis=0, return_inverse=True)
    inv = inv.ravel()
    atoms = np.flatnonzero(keys.any(1))
    A = keys[atoms].T.astype(float)
    _, Rr, pr = qr(A.T, pivoting=True, mode="economic")
    r = int((np.abs(np.diag(Rr)) > 1e-9).sum())
    assert r <= 20, f"rank {r}: too many linked sets for the complete check"
    B = A[np.sort(pr[:r])]
    _, Rc, pc = qr(B, pivoting=True, mode="economic")
    cols = pc[:r]
    Tm = np.linalg.solve(B[:, cols], B)
    bits = ((np.arange(1, 2 ** r)[:, None] >> np.arange(r)) & 1).astype(float)
    V = bits @ Tm
    ok = np.all((np.abs(V) < 1e-6) | (np.abs(V - 1) < 1e-6), axis=1)
    return inv, atoms, np.rint(V[ok]).astype(int)


def secure_cells(cells):
    """Hide written cells until no identifiable group has a count k, or complement n - k, of 1 to 20."""
    hidden = 0
    while True:
        W = [c for c in cells if c["write"]]
        if not W:
            return hidden
        inv, atoms, V = identifiable([c["mask"] for c in W])
        pos = np.searchsorted(atoms, inv)
        inside = (pos < len(atoms)) & (atoms[np.minimum(pos, len(atoms) - 1)] == inv)
        ind = W[0]["ind"]
        k_atom = np.bincount(pos[inside], weights=ind[inside], minlength=len(atoms))
        n_atom = np.bincount(pos[inside], minlength=len(atoms))
        k, n = V @ k_atom, V @ n_atom
        bad = np.flatnonzero([small(a) or small(b - a) for a, b in zip(k, n)])
        if not len(bad):
            return hidden
        u = V[bad[0]]
        um = np.isin(inv, atoms[u == 1])
        cand = [c for c in W if (c["mask"] & um).any()]
        min(cand, key=lambda c: int(c["mask"].sum()))["write"] = False
        hidden += 1
