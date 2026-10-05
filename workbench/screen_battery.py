#!/usr/bin/env python3
"""screen_battery.py -- score the All of Us respondents on the battery's own basis (Methods; eMethods 2).

STANDALONE. Needs only numpy, pandas and scikit-learn. Nothing is refitted here: the basis is
fitted once from the battery's item text (build_basis_package.py) and shipped in basis/, and this
script only projects onto it. There is no pool and no keep rule: every component is cored by
battery items by construction, and all k components are scored.

WHAT IT DOES, per encoder

  1. PLACE the battery on its basis
        P = (E - mu) @ W.T                         items x k        E = the battery's item embeddings
     This reproduces the fit's item projection exactly (checked when the package was built).

  2. SCORE the respondents
        X    = responses rescaled to [-1, 1] per the scale file, reverse-coded where it says so
        Y    = X @ P                               respondents x k
        Yr   = Y residualized on the covariates (age, sex) by weighted least squares
        T2   = Mahalanobis distance^2 of Yr under a Ledoit-Wolf precision
     Responses are bounded and discrete, so the flag is the empirical upper tail of the cohort's own
     distance distribution, not a chi-square cutoff.

  3. CONSENSUS across encoders, WRITTEN BUT NOT READ. Each analysis reads one encoder's own flagged
     set from scores_<enc>.csv, gte-large-en-v1.5 prespecified, not the consensus set (flagged under
     every encoder). The consensus file, the p90-p99 sweep of pairwise Jaccard and the
     mean-percentile outlier score are written, and nothing in the JNO analysis reads them.

The TF-IDF groupings are scored by screen_tfidf.py, which imports this script's functions, so they
pass through exactly the steps above.

INPUTS (copied to the Workbench next to this script)
  basis/<enc>.npz                        W, mu, sigma, mu_scores          (build_basis_package.py)
  basis/item_embeddings/<prefix>_<enc>.npz   items, vectors
  basis/components.csv                   each component's core items and construct profile
  <the one *items*.csv beside the script>  the item table: item order, scale_min, scale_max, reverse_coded (wb_config.py)
  RESPONSES_CSV                          id, age, sex, then one column per item (build_responses_v9)
  WEIGHTS_CSV                            one weight per respondent, positional (optional)

OUTPUTS (-> OUT_DIR)
  scores_<enc>.csv        id, distance, distance2, pctile, flagged   (flagged = above the encoder's own 95th)
  consensus_outliers.csv  id, n_encoders_flagging, outlier_score, flagged_by_all   (not read, see 3)
  screen_summary.csv      the p90-p99 sweep                                       (not read, see 3)
  run_config.json         every setting and count, for the record, including the CDR version

Run:  %run screen_battery.py                (cohort of record, responses_k5 -> screen_out/)
      %run screen_battery.py --cohort k0   (complete cases, a sensitivity analysis -> screen_out_k0/)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

# ----------------------------------------------------------------------------- CONFIG
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_config import CFG   # noqa: E402  which battery, which embeddings, which responses: resolved once, no defaults
BASIS_DIR     = CFG.basis_dir
EMB_DIR       = CFG.emb_dir
EMB_PREFIX    = CFG.emb_prefix
COHORT        = CFG.cohort
RESPONSES_CSV = CFG.responses_csv
WEIGHTS_CSV   = CFG.weights_csv
ITEMS_CSV     = CFG.items_csv
OUT_DIR       = CFG.out_dir
CDR           = os.environ.get("WORKSPACE_CDR", "")            # recorded, never used here

ID_COL     = "id"
COVARIATES = ["age", "sex"]      # residualized out of the component scores. [] to skip.
PCTILE     = 95.0                # per-encoder flag threshold, empirical
SWEEP      = range(90, 100)

ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5",
            "all-mpnet-base-v2", "e5-large-v2", "qwen3-embedding-0.6b"]


def wls_residualize(Y, C, w):
    """Residualize each column of Y on the covariate matrix C (intercept added) by WLS."""
    if C is None or C.shape[1] == 0:
        return Y - np.average(Y, axis=0, weights=w)
    A = np.column_stack([np.ones(len(Y)), C])
    sw = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(A * sw, Y * sw, rcond=None)
    return Y - A @ beta


def mahalanobis2(Yr, w):
    """Squared Mahalanobis distance under a Ledoit-Wolf precision of the weighted covariance."""
    ctr = Yr - np.average(Yr, axis=0, weights=w)
    lw = LedoitWolf().fit(ctr)
    return np.einsum("ij,ij->i", ctr, ctr @ lw.precision_), lw


def load_battery(items):
    """Responses rescaled to [-1, 1] and keyed, plus covariates and weights. Shared with the
    attribution scripts so every stage sees the same X."""
    scales = (pd.read_csv(ITEMS_CSV).set_index("item").reindex(items)
                .rename(columns={"scale_min": "min", "scale_max": "max", "reverse_coded": "reverse"}))
    assert scales["min"].notna().all(), "the items table does not cover every item"
    resp = pd.read_csv(RESPONSES_CSV)
    resp[ID_COL] = resp[ID_COL].astype(str)
    resp = resp.set_index(ID_COL)
    missing = [c for c in items if c not in resp.columns]
    assert not missing, f"{len(missing)} item columns absent from the responses: {missing[:5]}"
    raw = resp[items].values.astype(float)
    n_missing = int(np.isnan(raw).sum())
    assert n_missing == 0, (f"{n_missing} missing response cells. This screen has no imputation "
                            f"path -- use the k0 (complete) response file or add one deliberately.")
    lo = scales["min"].values.astype(float)
    hi = scales["max"].values.astype(float)
    rev = scales["reverse"].astype(str).str.lower().eq("true").values
    X = np.clip(2.0 * (raw - lo) / (hi - lo) - 1.0, -1.0, 1.0)
    X[:, rev] *= -1.0                                     # +1 = the instrument's keyed direction
    cov = None
    if COVARIATES:
        absent = [c for c in COVARIATES if c not in resp.columns]
        assert not absent, f"covariates absent from the responses file: {absent}"
        cov = resp[COVARIATES].values.astype(float)
        assert not np.isnan(cov).any(), "a covariate is missing for some respondent"
    if WEIGHTS_CSV and os.path.exists(WEIGHTS_CSV):
        w = pd.read_csv(WEIGHTS_CSV).iloc[:, 0].values.astype(float)
        assert len(w) == len(resp), f"weights length {len(w)} != respondents {len(resp)}"
    else:
        w = np.ones(len(resp))
    return resp, X, cov, w


def place_battery(enc, items, basis_dir=None, emb_dir=None):
    """P = (E - mu) W^T for the battery's items in canonical order, plus the package fields.
    basis_dir and emb_dir default to the encoders' basis/; screen_tfidf.py passes basis_tfidf/."""
    basis_dir = basis_dir or BASIS_DIR
    emb_dir = emb_dir or EMB_DIR
    b = np.load(os.path.join(basis_dir, f"{enc}.npz"), allow_pickle=True)
    W, mu = b["W"], b["mu"]
    e = np.load(os.path.join(emb_dir, f"{EMB_PREFIX}_{enc}.npz"), allow_pickle=True)
    order = [str(x) for x in e["items"]]
    pos = {it: j for j, it in enumerate(order)}
    absent = [it for it in items if it not in pos]
    assert not absent, f"{enc}: {len(absent)} items have no embedding: {absent[:5]}"
    E = np.asarray(e["vectors"], float)[[pos[it] for it in items]]
    return (E - mu) @ W.T, b


def score_placed(P, X, cov, w):
    """Step 2 for one placed basis: Y = X P, residualized, Ledoit-Wolf T2, the empirical 95th.
    Returns distance, distance^2, percentile and the flag, one value per respondent."""
    Y = X @ P                                             # respondents x k
    Yr = wls_residualize(Y, cov, w)
    T2, _ = mahalanobis2(Yr, w)
    d = np.sqrt(np.maximum(T2, 0.0))
    pct = pd.Series(d).rank(pct=True).values * 100.0
    flag = d > np.percentile(d, PCTILE)
    return d, T2, pct, flag


# ----------------------------------------------------------------------------- main
def main():
    CFG.describe("screen")
    os.makedirs(OUT_DIR, exist_ok=True)
    items = pd.read_csv(ITEMS_CSV)["item"].astype(str).tolist()
    resp, X, cov, w = load_battery(items)
    print(f"cohort {COHORT} ({'of record, imputed' if COHORT == 'k5' else 'complete cases, sensitivity'}): "
          f"{len(resp):,} respondents x {len(items)} items | "
          f"covariates {COVARIATES or 'none'} | weights "
          f"{'unit' if np.allclose(w, 1) else 'supplied'}")

    dist, flags, k_of = {}, {}, {}
    for enc in ENCODERS:
        P, b = place_battery(enc, items)                  # items x k, all k scored
        d, T2, pct, flag = score_placed(P, X, cov, w)
        dist[enc] = pd.Series(d, index=resp.index)
        flags[enc] = pd.Series(flag, index=resp.index)
        k_of[enc] = int(P.shape[1])
        pd.DataFrame({ID_COL: resp.index, "distance": d, "distance2": T2,
                      "pctile": pct, "flagged": flag}).to_csv(
            os.path.join(OUT_DIR, f"scores_{enc}.csv"), index=False)
        print(f"  {enc:22s} k={P.shape[1]:3d}  flagged={int(flag.sum()):5d}")

    D = pd.DataFrame(dist)
    F = pd.DataFrame(flags)
    cons = F.all(axis=1)
    score = (D.rank(pct=True) * 100).mean(axis=1)
    pd.DataFrame({ID_COL: D.index, "n_encoders_flagging": F.sum(axis=1).values,
                  "outlier_score": score.values, "flagged_by_all": cons.values}).to_csv(
        os.path.join(OUT_DIR, "consensus_outliers.csv"), index=False)

    sweep = []
    for t in SWEEP:
        thr = D.quantile(t / 100.0)
        sets = {m: set(D.index[D[m] > thr[m]]) for m in ENCODERS}
        js = [len(sets[a] & sets[b]) / len(sets[a] | sets[b])
              for i, a in enumerate(ENCODERS) for b in ENCODERS[i + 1:]]
        f = (100 - t) / 100.0
        null = f / (2 - f)
        sweep.append({"pctile": t, "mean_flagged_per_encoder": int(np.mean([len(s) for s in sets.values()])),
                      "consensus_N": len(set.intersection(*sets.values())),
                      "union_N": len(set.union(*sets.values())),
                      "mean_jaccard": round(float(np.mean(js)), 3),
                      "null_jaccard": round(null, 4),
                      "x_null": round(float(np.mean(js)) / null, 1)})
    sw = pd.DataFrame(sweep)
    sw.to_csv(os.path.join(OUT_DIR, "screen_summary.csv"), index=False)
    print(f"\nconsensus outliers at p{PCTILE:.0f} (flagged by all {len(ENCODERS)}): "
          f"{int(cons.sum()):,} of {len(D):,}")
    print(sw.to_string(index=False))

    json.dump({"cohort": COHORT, "n_cohort": int(len(resp)), "n_items": len(items), "encoders": ENCODERS,
               "k_per_encoder": k_of, "pctile": PCTILE, "covariates": COVARIATES,
               "responses_csv": RESPONSES_CSV, "weights_csv": WEIGHTS_CSV,
               "unit_weights": bool(np.allclose(w, 1)), "cdr": CDR,
               "n_consensus_outliers": int(cons.sum())},
              open(os.path.join(OUT_DIR, "run_config.json"), "w"), indent=2)
    print(f"\nwrote -> {OUT_DIR}")


if __name__ == "__main__":
    main()
