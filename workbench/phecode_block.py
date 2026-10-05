#!/usr/bin/env python3
"""phecode_block.py -- the semantic groups TOGETHER: how much the block of all k groups adds to age and sex,
for each of the 15 diagnoses.

THE CLAIM IT SERVES: a MULTIVARIATE ASSOCIATION of each diagnosis with the semantic
groups taken together. Not causal, and not a prediction of who will be diagnosed. The cross-validated AUC
below measures the strength of that association on people the model was not fitted on (a guard against the
optimism of k fitted terms); it is worded as association, never as prediction.

THE PEOPLE, LABELS AND GROUP SCORES are Stage 2's (phecode_directions.py): each diagnosis's denominator (the
people with at least 1 ICD-coded event; prostate cancer: the men among them), the label PheTK count >= 2,
and the k group scores exactly as the screen computes them (Yr, age and sex residualized), standardized over
the cohort of record. It stops unless the distance recomputed from them equals scores_<enc>.csv.

THREE LOGISTIC REGRESSIONS per diagnosis, by maximum likelihood with no penalty:
  base      age, age^2, sex (prostate cancer: age, age^2)
  groups    base + all k group scores            (THE ANSWER; the same model as Stage 2's)
  distance  base + the screen's distance, standardized over the cohort (one term: how atypical overall)
IN SAMPLE: McFadden pseudo-R^2 of each; likelihood-ratio tests groups vs base (k df) and distance vs base
  (1 df). The groups model's pseudo-R^2 must equal Stage 2's (checked where Stage 2's file is present).
OUT OF SAMPLE: 10-fold cross-validation, stratified on the label, the same folds for the three models
  (seed 0). Each person's predicted probability comes from the model fitted without their fold, and the AUC
  is computed over all these out-of-fold predictions. Differences between the models' AUCs (groups - base,
  distance - base, groups - distance) have DeLong 95% intervals and p-values, paired on the same people.

WITHIN-AGE AUC (the AUC above is pooled, so pairs of different ages count; this one is not). Each person's
  GROUP COMBINATION is the k group terms of
  the groups model fitted without their fold (out of fold), and the pairs compared are only those in the same
  5-year age band (18-24, 25-29, ..., 80-84, 85+) and the same sex (prostate cancer: men only, so every pair
  is already man with man and the band alone is used).
  The AUC is pooled over all such pairs (each band weighted by its number of case / non-case pairs): among
  people of the same age (and sex), how often a person with the diagnosis has the higher group combination.
  0.5 = no separation. The same for the distance itself (no fitting; its sign is the screen's own: higher =
  more atypical). 95% intervals from 1,000 bootstrap resamples of people within each band (seed 0).

THE DISSEMINATION POLICY (disclosure.py). Every statistic rests on a whole denominator
(at least 61,631 people; prostate cancer: the men among them) and at least 688 cases. The only counts
written are each diagnosis's denominator and cases, as in Stage 2, checked against 1 to 20; for prostate
cancer neither is written NOR PRINTED (a one-sex denominator).
★ PRECISION: AUCs, differences and intervals are written to 3 decimals, pseudo-R^2, LR statistics
and p-values to 4 significant figures (disclosure.coarsen). At full precision an AUC is U / (cases x non-cases),
and two pseudo-R^2 with their LR statistic give the null log-likelihood; either could return prostate cancer's
men-only denominator.

INPUTS   as phecode_directions.py
OUTPUTS  screen_out/phecode_block/phecode_block_<enc>.csv         1 row per diagnosis
         screen_out/phecode_block/phecode_block_<enc>_meta.json
         screen_out/phecode_block_<enc>_<CDR>.zip                  those 2, checked by verdict()

Run:  python3 phecode_block.py                          (gte-large-en-v1.5)
      python3 phecode_block.py --encoder bge-m3         (one encoder at a time, for the supplement)
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, norm, rankdata
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from disclosure import coarsen, coarsen_sig, small                       # noqa: E402  the policy
from phecode_directions import direction_scores                          # noqa: E402  Stage 2's group scores
from phecode_sweep import CONDITIONS, Z, load                            # noqa: E402  Stage 1's labels
from screen_battery import ENCODERS                                      # noqa: E402
from zip_screen_aggregates import verdict                                # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "phecode_block")
STAGE2 = os.path.join(OUT_DIR, "phecode_directions")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
ENC = sys.argv[sys.argv.index("--encoder") + 1] if "--encoder" in sys.argv else ENCODERS[0]
FOLDS, SEED = 10, 0
AGE_EDGES = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, np.inf]
N_BOOT = 1000


def within_auc(score, y, strata):
    """AUC over case / non-case pairs in the same stratum only, pooled over strata (weights = pairs)."""
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        yy, ss = y[m], score[m]
        n1 = int(yy.sum())
        n0 = int(m.sum()) - n1
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(ss)
        num += r[yy == 1].sum() - n1 * (n1 + 1) / 2.0
        den += n1 * n0
    return num / den


def within_auc_ci(score, y, strata, rng):
    """Bootstrap 95% interval: people resampled with replacement within each stratum."""
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        out[b] = within_auc(score[idx], y[idx], strata[idx])
    return np.percentile(out, [2.5, 97.5])


def delong(y, P):
    """AUCs of the rows of P (models x people) against y, and their DeLong covariance (fast form, Sun & Xu
    2014), paired on the same people."""
    pos, neg = P[:, y == 1], P[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([rankdata(r) for r in pos])
    ty = np.array([rankdata(r) for r in neg])
    tz = np.array([rankdata(np.r_[a, b]) for a, b in zip(pos, neg)])
    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    return auc, S


def fit(y, X):
    res = sm.Logit(y, sm.add_constant(X)).fit(disp=0, maxiter=200)
    return res


def main():
    assert ENC in ENCODERS, f"--encoder must be one of {ENCODERS}"
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)
    A = load(idx)
    Yz, names, worst = direction_scores(ENC, idx)
    assert worst < 1e-6, f"{ENC}: the recomputed distance differs from scores_{ENC}.csv by {worst}"
    k = Yz.shape[1]
    S = pd.read_csv(os.path.join(OUT_DIR, f"scores_{ENC}.csv"))
    S[ID_COL] = S[ID_COL].astype(str)
    dist = S.set_index(ID_COL).reindex(idx)["distance"].values.astype(float)
    assert not np.isnan(dist).any(), "a cohort member has no distance"
    distz = (dist - dist.mean()) / dist.std(ddof=1)
    s2meta = os.path.join(STAGE2, f"phecode_directions_{ENC}_meta.json")
    s2 = json.load(open(s2meta))["conditions"] if os.path.exists(s2meta) else None
    print(f"[phecode_block] {ENC}: k = {k} | distance reproduced (largest difference {worst:.1e}) | "
          f"{FOLDS}-fold cross-validation, seed {SEED}", flush=True)

    rows, meta = [], {}
    for key, code, label, lst, men_only in CONDITIONS:
        d = A[f"den_{key}"]
        y = A[f"y_{key}"][d].astype(int)
        n, n_case = int(d.sum()), int(y.sum())
        assert not (small(n) or small(n_case) or small(n - n_case)), f"{key}: a count of 1 to 20"
        age = A["age"][d]
        az = (age - age.mean()) / age.std()
        base = np.column_stack([az, az ** 2] + ([] if men_only else [A["sex"][d]]))
        X = {"base": base, "groups": np.column_stack([base, Yz[d]]), "distance": np.column_stack([base, distz[d]])}
        fits = {m: fit(y, x) for m, x in X.items()}
        r = {"encoder": ENC, "condition": key, "phecode": code, "label": label, "list": lst, "k": k}
        for m, f in fits.items():
            r[f"pseudo_r2_{m}"] = float(f.prsquared)
            r[f"converged_{m}"] = bool(f.mle_retvals.get("converged", False))
        for m, df in (("groups", k), ("distance", 1)):
            lr = 2.0 * (fits[m].llf - fits["base"].llf)
            r[f"lr_{m}"], r[f"lr_df_{m}"], r[f"lr_p_{m}"] = float(lr), df, float(chi2.sf(lr, df))
        if s2 is not None:                             # the groups model is Stage 2's model
            assert abs(r["pseudo_r2_groups"] - s2[key]["pseudo_r2"]) < 1e-9, f"{key}: groups model differs from Stage 2's"

        # out of sample: the same folds for the three models
        P = {m: np.full(n, np.nan) for m in X}
        comb = np.full(n, np.nan)                      # the group combination, out of fold
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(n), y):
            for m, x in X.items():
                f = fit(y[tr], x[tr])
                P[m][te] = f.predict(sm.add_constant(x[te], has_constant="add"))
                if m == "groups":
                    comb[te] = Yz[d][te] @ np.asarray(f.params)[-k:]
        assert not any(np.isnan(p).any() for p in P.values())
        order = ["base", "groups", "distance"]
        auc, Sig = delong(y, np.vstack([P[m] for m in order]))
        for i, m in enumerate(order):
            se = np.sqrt(Sig[i, i])
            r.update({f"auc_{m}": auc[i], f"auc_{m}_lo": auc[i] - Z * se, f"auc_{m}_hi": auc[i] + Z * se})
        for a, b in (("groups", "base"), ("distance", "base"), ("groups", "distance")):
            i, j = order.index(a), order.index(b)
            dlt = auc[i] - auc[j]
            se = np.sqrt(max(Sig[i, i] + Sig[j, j] - 2 * Sig[i, j], 0.0))
            r.update({f"dauc_{a}_{b}": dlt, f"dauc_{a}_{b}_lo": dlt - Z * se, f"dauc_{a}_{b}_hi": dlt + Z * se,
                      f"dauc_{a}_{b}_p": float(2 * norm.sf(abs(dlt) / se)) if se > 0 else np.nan})
        # within-age (and sex) AUC of the group combination and of the distance
        band = np.digitize(age, AGE_EDGES[1:-1])
        strata = band if men_only else band * 2 + A["sex"][d].astype(int)
        rng = np.random.default_rng(SEED)
        for m, sc in (("groups", comb), ("distance", distz[d])):
            w = within_auc(sc, y, strata)
            lo, hi = within_auc_ci(sc, y, strata, rng)
            r.update({f"auc_within_{m}": w, f"auc_within_{m}_lo": lo, f"auc_within_{m}_hi": hi})
        r["within_strata"] = "5-year age band" if men_only else "5-year age band x sex"
        rows.append(r)
        meta[key] = {"phecode": code, "label": label, "list": lst}
        if not men_only:                               # ★ a one-sex denominator is never written or printed
            meta[key].update({"n_denominator": n, "n_cases": n_case})
        counts = "men only" if men_only else f"n {n:,} cases {n_case:,}"
        print(f"  {lst} {label[:38]:38s} {counts} | pseudo-R2 base {r['pseudo_r2_base']:.3f} groups "
              f"{r['pseudo_r2_groups']:.3f} distance {r['pseudo_r2_distance']:.3f} | AUC (out of fold) base "
              f"{auc[0]:.3f} groups {auc[1]:.3f} distance {auc[2]:.3f} | groups - base "
              f"{r['dauc_groups_base']:+.3f} [{r['dauc_groups_base_lo']:+.3f}, {r['dauc_groups_base_hi']:+.3f}] | "
              f"within-age AUC groups {r['auc_within_groups']:.3f} [{r['auc_within_groups_lo']:.3f}, "
              f"{r['auc_within_groups_hi']:.3f}] distance {r['auc_within_distance']:.3f}", flush=True)

    D = pd.DataFrame(rows)
    # ★ precision (disclosure.py): AUCs, their differences and intervals to 3 decimals; pseudo-R^2, LR
    # statistics and p-values to 4 significant figures (at full precision two pseudo-R^2 and their LR give
    # the null log-likelihood, and with it the prostate cancer men-only denominator)
    D = coarsen(D, [c for c in D.columns if c.startswith(("auc_", "dauc_")) and not c.endswith("_p")])
    D = coarsen_sig(D, [c for c in D.columns if c.startswith(("pseudo_r2_", "lr_")) and not c.startswith("lr_df_")]
                    + [c for c in D.columns if c.startswith("dauc_") and c.endswith("_p")])
    D.to_csv(os.path.join(OUT, f"phecode_block_{ENC}.csv"), index=False)
    json.dump({"cdr": CDR, "encoder": ENC, "k": k, "folds": FOLDS, "seed": SEED, "conditions": meta,
               "claim": "multivariate association of each diagnosis with the semantic groups together; not causal, "
                        "not a prediction of who will be diagnosed",
               "models": {"base": "age, age^2, sex (prostate: age, age^2)", "groups": "base + all k group scores "
                          "(Stage 2's model)", "distance": "base + the screen's distance, standardized"},
               "in_sample": "McFadden pseudo-R^2; likelihood-ratio tests against base",
               "out_of_sample": f"{FOLDS}-fold stratified cross-validation, the same folds for all models; AUC over "
                                "the out-of-fold predictions; DeLong 95% intervals, paired",
               "within_age_auc": "the out-of-fold group combination (and the distance), compared only within 5-year age "
                                 "bands (x sex except prostate cancer), pooled over pairs; bootstrap 95% CI, "
                                 f"{N_BOOT} resamples within bands",
               "age_bands": [a for a in AGE_EDGES[:-1]],
               "disclosure": "All of Us dissemination policy (disclosure.py)",
               "precision": "AUCs, differences and intervals to 3 decimals; pseudo-R^2, LR statistics and p-values "
                            "to 4 significant figures (disclosure.py)"},
              open(os.path.join(OUT, f"phecode_block_{ENC}_meta.json"), "w"), indent=2)
    files = [f"phecode_block_{ENC}.csv", f"phecode_block_{ENC}_meta.json"]
    for fn in files:
        ok, why = verdict(os.path.join(OUT, fn))
        assert ok, f"{fn} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"phecode_block_{ENC}_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"phecode_block/{fn}")
    print(f"\n[phecode_block] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
