#!/usr/bin/env python
"""Random-restart sweep for one encoder at one lambda_score, the route to a sparser basis.

Fits the sparse projection from N random orthonormal initializations (seeds) at a fixed
lambda_score. Only the initialization varies across seeds: with batch_size = n (full batch) and
subsample_fraction = 1.0 the robust scale and the single minibatch are deterministic, so
random_state feeds only the random orthonormal start. For each seed the loadings W (k x D) and the
item projection P (151 x k) are saved, so every metric can be recomputed downstream.

Layout:
    ../outputs/random_starts/<encoder>__k{k}_ls{ls}/seed{seed:03d}/W.npy, P.npy
    ../outputs/random_starts/<encoder>__k{k}_ls{ls}/metrics.csv      one row per seed

Resumable: seeds whose W.npy and P.npy exist are skipped unless --overwrite.

Run (CPU, one encoder):
    CUDA_VISIBLE_DEVICES="" python random_starts.py \
        --encoder gte-large-en-v1.5 --lambda-score 1.0 --seeds 0-99
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from ross_proj import fit_projection
from ross_proj.defining_sets import participation_ratio  # (sum z^2)^2 / sum z^4 per component

from _common import EMBEDDERS, OUT_DIR, load_embedding


def _prk(sq, k):
    """PR over the top-k already-sorted (descending) squared loadings."""
    a = sq[:max(k, 1)]
    return a.sum() ** 2 / ((a ** 2).sum() + 1e-300)


def fixed_point_core_size(col):
    """Fixed-point PR core size of one component column (same rule as cores.py)."""
    sq = np.sort(np.asarray(col, float) ** 2)[::-1]
    k = int(round(_prk(sq, len(sq))))
    for _ in range(200):
        kn = int(round(_prk(sq, k)))
        if kn == k:
            break
        k = kn
    return max(k, 1)


def variance_explained(X, mean, W, Z):
    """1 - ||Xc - Z W||^2 / ||Xc||^2 with W k x D and Z n x k."""
    Xc = X - mean
    return 1.0 - np.sum((Xc - Z @ W) ** 2) / np.sum(Xc ** 2)


def parse_seeds(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def resolve_k(name, k_arg):
    if k_arg != "auto":
        return int(k_arg)
    path = os.path.join(OUT_DIR, "k_selection", name, "parallel_analysis.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: run 1_select_k.py first, or pass --k <int>")
    return int(json.load(open(path))["k"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(EMBEDDERS))
    ap.add_argument("--lambda-score", type=float, default=1.0)
    ap.add_argument("--lambda-ortho", type=float, default=0.1)
    ap.add_argument("--k", default="auto")
    ap.add_argument("--seeds", default="0-99")
    ap.add_argument("--total-steps", type=int, default=5000)
    ap.add_argument("--out-dir", default=os.path.join(OUT_DIR, "random_starts"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    name, ls = args.encoder, args.lambda_score
    items, X = load_embedding(name)
    n = X.shape[0]
    k = resolve_k(name, args.k)
    seeds = parse_seeds(args.seeds)

    tag = f"{name}__k{k}_ls{ls}"
    root = os.path.join(args.out_dir, tag)
    os.makedirs(root, exist_ok=True)
    metrics_path = os.path.join(root, "metrics.csv")
    new_file = not os.path.exists(metrics_path)
    mf = open(metrics_path, "a", newline="")
    writer = csv.writer(mf)
    if new_file:
        writer.writerow(["encoder", "k", "lambda_score", "seed", "ve",
                         "pr_onetime_median", "pr_onetime_mean",
                         "pr_fixed_median", "pr_fixed_mean", "n_steps", "secs"])
        mf.flush()

    print(f"[{tag}] {len(seeds)} seeds on {n}x{X.shape[1]}, k={k}, lambda_score={ls}", flush=True)
    for seed in seeds:
        sd = os.path.join(root, f"seed{seed:03d}")
        wnpy, pnpy = os.path.join(sd, "W.npy"), os.path.join(sd, "P.npy")
        if not args.overwrite and os.path.exists(wnpy) and os.path.exists(pnpy):
            print(f"  seed {seed:3d}  skip (exists)", flush=True)
            continue
        os.makedirs(sd, exist_ok=True)
        t0 = time.time()
        res = fit_projection(
            X, n_components=k, lambda_score=ls, lambda_ortho=args.lambda_ortho,
            init="random_normalized", random_state=seed,
            batch_size=n, subsample_fraction=1.0, total_steps=args.total_steps,
            verbose=False, log_interval=0)
        secs = time.time() - t0
        W, P = np.asarray(res.W), np.asarray(res.Z)
        np.save(wnpy, W)
        np.save(pnpy, P)
        ve = variance_explained(X, res.mean, W, P)
        pr1 = participation_ratio(P)
        prf = np.array([fixed_point_core_size(P[:, c]) for c in range(k)])
        n_steps = int(getattr(res.model, "n_steps_", -1))
        writer.writerow([name, k, ls, seed, f"{ve:.6f}",
                         f"{np.median(pr1):.4f}", f"{np.mean(pr1):.4f}",
                         f"{np.median(prf):.4f}", f"{np.mean(prf):.4f}", n_steps, f"{secs:.1f}"])
        mf.flush()
        print(f"  seed {seed:3d}  VE={ve:.4f}  PR1_med={np.median(pr1):5.1f}  "
              f"PRfix_med={np.median(prf):4.1f}  [{secs:.1f}s]", flush=True)
    mf.close()
    print(f"[{tag}] done -> {root}", flush=True)


if __name__ == "__main__":
    main()
