#!/usr/bin/env python3
"""screen_tfidf.py -- score the respondents on the 2 TF-IDF groupings, exactly as the encoders are
scored.

WHAT THE TF-IDF GROUPINGS ARE. Each item's text of record becomes a TF-IDF vector: stop words kept,
IDF from the SUBTLEX-US film-subtitle corpus, numerals written as number words
(../local/scripts/tfidf_ref.py in the repository). These vectors take the place of the sentence
embeddings. The basis is the SVD basis of the 151 centered vectors (materialize_selection.py --svd),
because at the encoders' lambda_score no sparse fit keeps 99% of PCA's variance, so the
encoders' selection rule admits nothing. basis_tfidf/manifest.csv records the build. Two depths:
  tfidf4    k = 4, TF-IDF's own Horn's k, which carries 13% of the item vectors' variance
  tfidf25   k = 25, referenced to the encoders' Horn's k (23 to 27), because
            TF-IDF, like the encoders, comes from the item text alone. It carries 40%.

HOW THEY ARE SCORED. With screen_battery.py's own functions, imported, not copied: load_battery,
place_battery (P = (E - mu) W^T) and score_placed (Y = X P, residualized on age and sex by weighted
least squares, T2 under a Ledoit-Wolf precision, the empirical 95th percentile). The only difference
from an encoder is the basis folder, basis_tfidf/.

INPUTS   basis_tfidf/<basis>.npz, basis_tfidf/item_embeddings/<prefix>_<basis>.npz
         the responses and weights screen_battery.py reads (wb_config.py)
OUTPUTS  screen_out/scores_tfidf4.csv, scores_tfidf25.csv   id, distance, distance2, pctile, flagged
                                                            (per person, stays on the Workbench)
         screen_out/screen_tfidf_config.json                 k, dimension, basis file checksum, n flagged

Run:  python3 screen_tfidf.py                 (cohort of record)
      python3 screen_tfidf.py --cohort k0     (complete cases)
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from screen_battery import (CFG, CDR, COHORT, COVARIATES, EMB_PREFIX, ID_COL, ITEMS_CSV, OUT_DIR,  # noqa: E402
                            PCTILE, load_battery, place_battery, score_placed)

TFIDF_DIR = os.path.join(HERE, "basis_tfidf")
MODELS = {"tfidf4": "tfidf-subtlex-k4", "tfidf25": "tfidf-subtlex-k25"}   # score name -> basis name
EXPECTED_K = {"tfidf4": 4, "tfidf25": 25}


def main():
    CFG.describe("screen_tfidf")
    assert os.path.isdir(TFIDF_DIR), f"{TFIDF_DIR} is missing; it ships in the Workbench folder"
    os.makedirs(OUT_DIR, exist_ok=True)
    items = pd.read_csv(ITEMS_CSV)["item"].astype(str).tolist()
    resp, X, cov, w = load_battery(items)
    print(f"cohort {COHORT}: {len(resp):,} respondents x {len(items)} items | covariates {COVARIATES}")

    meta = {"cohort": COHORT, "n_cohort": int(len(resp)), "pctile": PCTILE, "covariates": COVARIATES,
            "basis_dir": "basis_tfidf", "emb_prefix": EMB_PREFIX, "cdr": CDR, "models": {}}
    for name, basis in MODELS.items():
        P, b = place_battery(basis, items, TFIDF_DIR, os.path.join(TFIDF_DIR, "item_embeddings"))
        assert P.shape[1] == EXPECTED_K[name], f"{name}: basis has k = {P.shape[1]}, expected {EXPECTED_K[name]}"
        d, T2, pct, flag = score_placed(P, X, cov, w)
        pd.DataFrame({ID_COL: resp.index, "distance": d, "distance2": T2,
                      "pctile": pct, "flagged": flag}).to_csv(
            os.path.join(OUT_DIR, f"scores_{name}.csv"), index=False)
        f = os.path.join(TFIDF_DIR, f"{basis}.npz")
        meta["models"][name] = {"basis": basis, "k": int(P.shape[1]), "dim": int(b["W"].shape[1]),
                                "basis_md5": hashlib.md5(open(f, "rb").read()).hexdigest(),
                                "n_flagged": int(flag.sum())}
        print(f"  {name:8s} ({basis})  k={P.shape[1]:3d}  dim={b['W'].shape[1]}  flagged={int(flag.sum()):5d}")
    json.dump(meta, open(os.path.join(OUT_DIR, "screen_tfidf_config.json"), "w"), indent=2)
    print(f"\nwrote -> {OUT_DIR}")


if __name__ == "__main__":
    main()
