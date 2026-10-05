#!/usr/bin/env python
"""Materialize the selected basis per encoder as a fit folder (after select_projection.py).

Reads  ../outputs/random_starts/selection.json
Writes ../outputs/basis_fit/<enc>/{loadings.npy, scores.npy, mean.npy, fit.json}

loadings = W (k x D), scores = P (151 x k), mean = the column mean of the item embeddings (the
centering every fit in this pipeline uses, checked equal to the fits' saved mean to 0.0).
fit.json records which candidate this is (seed, lambda_score, k, core energy, explained variance,
orthogonality, source path). cores.py and build_basis_package.py read from here and nowhere else,
so the basis of record is a single folder.

--pca-init takes the PCA-initialized fit for every encoder instead of the selection. It serves a
restricted 41-item control, where the selection rule admits no candidate for 4 of the 6 encoders
because 41 items over about 10 components leave no fit that is both this concentrated and this close
to the PCA subspace. That control uses the PCA-initialized fit throughout.

--svd takes the exact SVD basis at the fit's k for every model instead: W = the top k right
singular vectors of the centered item matrix, the reference row select_projection.py computes and
never selects. Used for the 2 TF-IDF comparator groupings. At the encoders' lambda_score
1.0 no TF-IDF fit keeps 99% of PCA's explained variance (best 5.9% against 13.4% at k = 4, 35.9%
against 40.0% at k = 25), so the rule selects nothing. The SVD basis keeps all the variance of its
subspace. A Mahalanobis distance depends only on the subspace the k components span (up to the Ledoit-Wolf
shrinkage), so for the flagged sets this is the full-variance version of the same grouping. Each
column's sign is fixed so that its largest-magnitude item projection is positive.

Run:  python materialize_selection.py [--pca-init | --svd]
"""
import json
import os
import shutil
import sys

import numpy as np

from _common import EMBEDDERS, OUT_DIR, load_embedding

PCA_INIT = "--pca-init" in sys.argv
SVD = "--svd" in sys.argv
assert not (PCA_INIT and SVD), "--pca-init and --svd are alternatives"


def svd_candidate(X, k, ve_pca):
    """The exact SVD basis at k, sign fixed per column, with the metrics fit.json records."""
    mean = X.mean(0)
    Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:k].copy()
    P = Xc @ W.T
    flip = np.sign(P[np.abs(P).argmax(0), np.arange(k)])
    W *= flip[:, None]
    P *= flip[None, :]
    ve = 1.0 - np.sum((Xc - P @ W) ** 2) / np.sum(Xc ** 2)
    return W, P, mean, {"seed": "svd", "path": "", "coreE": None, "mean_core": None,
                        "median_core": None, "ve": float(ve), "ve_pca": ve_pca,
                        "ortho": float(np.abs(W @ W.T - np.eye(k)).max())}


def main():
    sel = json.load(open(os.path.join(OUT_DIR, "random_starts", "selection.json")))
    root = os.path.join(OUT_DIR, "basis_fit")
    for enc in EMBEDDERS:
        s = sel[enc]["pca_init"] if (PCA_INIT or SVD) else sel[enc]["selected"]
        if s is None:
            raise SystemExit(f"{enc}: no candidate passed the selection rule "
                             f"(--pca-init takes the PCA-initialized fit instead, --svd the SVD basis)")
        items, X = load_embedding(enc)
        if SVD:
            W, P, mean, s = svd_candidate(X, int(s["k"]), s["ve_pca"])
            s["lambda_score"] = None
        elif s["seed"] == "pca":
            src = os.path.join(OUT_DIR, s["path"])
            W = np.load(os.path.join(src, "loadings.npy"))
            P = np.load(os.path.join(src, "scores.npy"))
            mean = np.load(os.path.join(src, "mean.npy"))
        else:
            src = os.path.join(OUT_DIR, s["path"])
            W = np.load(os.path.join(src, "W.npy"))
            P = np.load(os.path.join(src, "P.npy"))
            mean = X.mean(0)
        err = float(np.abs((X - mean) @ W.T - P).max())
        if err > 1e-8:
            raise SystemExit(f"{enc}: (X - mean) W^T does not reproduce P (max abs err {err:.2e})")
        d = os.path.join(root, enc)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        np.save(os.path.join(d, "loadings.npy"), W)
        np.save(os.path.join(d, "scores.npy"), P)
        np.save(os.path.join(d, "mean.npy"), mean)
        json.dump({"encoder": enc, "n_components": int(W.shape[0]), "lambda_score": s["lambda_score"],
                   "seed": s["seed"], "coreE": s["coreE"], "mean_core": s["mean_core"],
                   "median_core": s.get("median_core"), "ve": s["ve"], "ve_pca": s["ve_pca"],
                   "ortho": s["ortho"], "source": s["path"], "reconstruction_max_abs_err": err,
                   "rule": ("pca-init" if PCA_INIT else
                            "svd (TF-IDF comparators)" if SVD else "selection rule")},
                  open(os.path.join(d, "fit.json"), "w"), indent=2)
        ce = "  n/a" if s["coreE"] is None else f"{s['coreE']:.3f}"
        print(f"{enc:22s} seed {str(s['seed']):>4s}  k={W.shape[0]:2d}  coreE {ce}  ve {s['ve']:.3f}  -> {d}")


if __name__ == "__main__":
    main()
