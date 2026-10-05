#!/usr/bin/env python
"""Select the operating projection per encoder from the random-restart sweep.

Candidates: every seed under ../outputs/random_starts/<enc>__k{k}_ls{ls}/ for the lambda_score(s)
given, plus the PCA-initialized fit under ../outputs/fits/<enc>__k{k}_ls{ls}/ (labelled seed "pca").

Rule:
  among candidates whose W stays orthonormal (largest off-diagonal |cos| between rows, rounded to
  3 decimals, <= ORTHO_MAX) and whose explained variance is within 1% of the PCA maximum at the
  same k (VE >= VE_FRAC * VE_PCA), retain the one that MAXIMIZES the median core energy, the
  fraction of a component's squared weight that sits on its fixed-point PR core.

VE_PCA is computed here from the centered embeddings (top-k singular values), so no baseline file
is needed.

Writes  ../outputs/random_starts/candidates.csv    one row per candidate with all metrics
        ../outputs/random_starts/selection.json    the retained candidate per encoder, with the
                                                   PCA-initialized fit's metrics alongside
Run:    CUDA_VISIBLE_DEVICES="" python select_projection.py --lambda-score 1.0
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

from _common import EMBEDDERS, OUT_DIR, load_embedding

ORTHO_MAX = 0.01
VE_FRAC = 0.99


def participation_ratio(col):
    sq = col.astype(float) ** 2
    s1 = sq.sum()
    return 0.0 if s1 <= 0 else float(s1 * s1 / (sq * sq).sum())


def fixed_point_core_size(col):
    sq = np.sort(col.astype(float) ** 2)[::-1]
    k = max(1, min(int(round(participation_ratio(col))), len(sq)))
    for _ in range(1000):
        top = sq[:k]
        s1 = top.sum()
        prk = 1.0 if s1 <= 0 else float(s1 * s1 / (top * top).sum())
        kn = max(1, min(int(round(prk)), len(sq)))
        if kn == k:
            break
        k = kn
    return k


def core_metrics(P):
    """(median core energy, mean core size, median core size) over the columns of P."""
    ce, cs = [], []
    for c in range(P.shape[1]):
        sq = P[:, c].astype(float) ** 2
        tot = sq.sum()
        s = fixed_point_core_size(P[:, c])
        cs.append(s)
        ce.append(0.0 if tot <= 0 else float(np.sort(sq)[::-1][:s].sum() / tot))
    return float(np.median(ce)), float(np.mean(cs)), float(np.median(cs))


def ortho_maxcos(W):
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    C = Wn @ Wn.T
    return float(np.abs(C[~np.eye(W.shape[0], dtype=bool)]).max())


def variance_explained(X, mean, W, P):
    Xc = X - mean
    return 1.0 - np.sum((Xc - P @ W) ** 2) / np.sum(Xc ** 2)


def ve_pca(X, k):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    return float((s[:k] ** 2).sum() / (s ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-score", nargs="+", type=float, default=[1.0])
    args = ap.parse_args()
    rs_dir = os.path.join(OUT_DIR, "random_starts")
    rows, out = [], {}
    print(f"{'encoder':22s} {'k':>2s} {'sel':>8s} {'coreE':>6s} {'core':>5s} {'ve':>6s} {'ortho':>6s} "
          f"| {'pass/all':>8s} | {'pca-init coreE':>14s} {'core':>5s} {'ve':>6s}")
    for enc in EMBEDDERS:
        items, X = load_embedding(enc)
        cands = []
        for ls in args.lambda_score:
            fit_hits = glob.glob(os.path.join(OUT_DIR, "fits", f"{enc}__k*_ls{ls}"))
            if len(fit_hits) != 1:
                raise SystemExit(f"{enc}: expected one PCA-initialized fit at ls {ls}, found {fit_hits}")
            fd = fit_hits[0]
            k = int(re.search(r"__k(\d+)_", fd).group(1))
            vp = ve_pca(X, k)
            mean = np.load(os.path.join(fd, "mean.npy"))

            def evaluate(W, P, seed, path):
                ce, mc, md = core_metrics(P)
                ve = variance_explained(X, mean, W, P)
                orth = ortho_maxcos(W)
                ok = round(orth, 3) <= ORTHO_MAX and ve >= VE_FRAC * vp
                return dict(encoder=enc, k=k, lambda_score=ls, seed=seed, coreE=ce, mean_core=mc,
                            median_core=md, ve=ve, ve_pca=vp, ortho=orth, passes=int(ok),
                            path=os.path.relpath(path, OUT_DIR) if path else "")

            cands.append(evaluate(np.load(os.path.join(fd, "loadings.npy")),
                                  np.load(os.path.join(fd, "scores.npy")), "pca", fd))
            # the exact SVD (dense PCA) basis at the same k: a reference row, never selected.
            # Computed here rather than read from a lambda_score 0 fit, which is the PCA subspace
            # only approximately (principal angles up to 6 degrees).
            Xc = X - X.mean(0)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            row = evaluate(Vt[:k], Xc @ Vt[:k].T, "svd", None)
            row["passes"] = 0
            cands.append(row)
            for sd in sorted(glob.glob(os.path.join(rs_dir, f"{enc}__k{k}_ls{ls}", "seed*"))):
                if not (os.path.exists(os.path.join(sd, "W.npy")) and os.path.exists(os.path.join(sd, "P.npy"))):
                    continue
                seed = int(re.search(r"seed(\d+)", sd).group(1))
                cands.append(evaluate(np.load(os.path.join(sd, "W.npy")),
                                      np.load(os.path.join(sd, "P.npy")), seed, sd))
        rows.extend(cands)
        passers = [c for c in cands if c["passes"]]
        best = max(passers, key=lambda c: c["coreE"]) if passers else None
        pca = next(c for c in cands if c["seed"] == "pca")
        out[enc] = dict(selected=best, pca_init=pca, n_candidates=len(cands), n_pass=len(passers))
        b = best or pca
        print(f"{enc:22s} {b['k']:2d} {str(b['seed']):>8s} {b['coreE']:6.3f} {b['mean_core']:5.2f} {b['ve']:6.3f} "
              f"{b['ortho']:6.4f} | {len(passers):3d}/{len(cands):<4d} | {pca['coreE']:14.3f} "
              f"{pca['mean_core']:5.2f} {pca['ve']:6.3f}")

    with open(os.path.join(rs_dir, "candidates.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump(out, open(os.path.join(rs_dir, "selection.json"), "w"), indent=2)
    print(f"\n-> {rs_dir}/candidates.csv, selection.json")


if __name__ == "__main__":
    main()
