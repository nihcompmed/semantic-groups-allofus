#!/usr/bin/env python
"""Package the own basis so the Workbench stage needs nothing else from this folder.

Per encoder, ../outputs/basis/<enc>.npz holds
    W          (k, D)   component loadings of the fit
    mu         (D,)     mean of the 151 item embeddings, the centering for the projection
    sigma      (k,)     per-component SD of the item projections over the 151 items
    mu_scores  (k,)     per-component mean of the item projections (zero by construction)
    k, dim, encoder, fit_dir, lambda_score, n_items
and the folder also carries
    item_embeddings/<prefix>_<enc>.npz        the battery's item embeddings, copied (admin151_items = of record)
    components.csv                            every component's core items and construct profile
    manifest.csv

Placing the battery on its own basis, which is all the Workbench script does:
    P = (E - mu) @ W.T          # 151 x k, reproduces scores.npy of the fit exactly
    Z = (P - mu_scores) / sigma
All k components are scored.

Run:  CUDA_VISIBLE_DEVICES="" python build_basis_package.py
"""
import csv
import glob
import os
import shutil

import numpy as np

from _common import DATA_DIR, EMBEDDERS, OUT_DIR, load_embedding


def fit_dir(enc):
    """The basis of record: ../outputs/basis_fit/<enc>/, written by materialize_selection.py."""
    d = os.path.join(OUT_DIR, "basis_fit", enc)
    if not os.path.exists(os.path.join(d, "scores.npy")):
        raise SystemExit(f"{enc}: no basis of record at {d}. Run select_projection.py then "
                         f"materialize_selection.py first.")
    return d


def main():
    out = os.path.join(OUT_DIR, "basis")
    emb_out = os.path.join(out, "item_embeddings")
    os.makedirs(emb_out, exist_ok=True)
    cores_dir = os.path.join(OUT_DIR, "cores")

    comp_rows, manifest = [], []
    for enc, npz in EMBEDDERS.items():
        fd = fit_dir(enc)
        W = np.load(os.path.join(fd, "loadings.npy"))
        mu = np.load(os.path.join(fd, "mean.npy"))
        P = np.load(os.path.join(fd, "scores.npy"))
        items, E = load_embedding(enc)
        recon = (E - mu) @ W.T
        err = float(np.abs(recon - P).max())
        if err > 1e-8:
            raise SystemExit(f"{enc}: (E - mu) W^T does not reproduce scores.npy (max abs err {err:.2e})")
        mu_s, sigma = P.mean(0), P.std(0)

        np.savez_compressed(
            os.path.join(out, f"{enc}.npz"),
            W=W.astype(np.float64), mu=mu.astype(np.float64),
            sigma=sigma.astype(np.float64), mu_scores=mu_s.astype(np.float64),
            k=W.shape[0], dim=W.shape[1], encoder=enc,
            fit_dir=os.path.basename(fd), lambda_score=1.0, n_items=len(items))
        shutil.copy(os.path.join(DATA_DIR, npz), os.path.join(emb_out, npz))

        for r in csv.DictReader(open(os.path.join(cores_dir, f"{enc}_components.csv"))):
            comp_rows.append({"encoder": enc, "component": int(r["component"]),
                              "n_core_items": int(r["size"]), "n_constructs": int(r["n_constructs"]),
                              "core_items": r["items"], "construct_profile": r["construct_profile"]})
        manifest.append({"encoder": enc, "k": int(W.shape[0]), "embedding_dim": int(W.shape[1]),
                         "n_items": len(items), "fit_dir": os.path.basename(fd),
                         "sigma_min": round(float(sigma.min()), 4),
                         "sigma_max": round(float(sigma.max()), 4),
                         "max_abs_mu_scores": float(np.abs(mu_s).max()),
                         "reconstruction_max_abs_err": err})
        print(f"{enc:22s} k={W.shape[0]:2d} dim={W.shape[1]:4d} sigma {sigma.min():.3f}-{sigma.max():.3f} "
              f"recon err {err:.1e}", flush=True)

    for name, rows in [("components.csv", comp_rows), ("manifest.csv", manifest)]:
        with open(os.path.join(out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print("->", out)


if __name__ == "__main__":
    main()
