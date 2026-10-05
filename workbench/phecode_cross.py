#!/usr/bin/env python3
"""phecode_cross.py -- the CROSS-APPLICATION MATRIX: is the combination of semantic groups that goes with a
diagnosis specific to it?

THE HYPOTHESIS: one combination of groups goes with breast cancer and with prostate cancer, and it is
not the combination that goes with the other 13 diagnoses. "Important" is RELATIVE: the combination fitted on
diagnosis i is applied to every diagnosis j, and it matters for j if it separates j's cases from non-cases of
the same age and sex.

THE COMBINATIONS. For each diagnosis i (Stage 2's people and labels; phecode_block.py's groups model: age,
age^2, sex (prostate: age, age^2) + all k group scores), the combination is the k group terms, w_i . group
scores (age and sex terms excluded). It is fitted OUT OF FOLD: one split of the whole cohort into 10 folds
(seed 0), the model for i fitted on i's denominator outside a fold, and EVERY cohort member in that fold
scored with it. So every person's score on every combination comes from a model fitted without them. The
distance (standardized) is added as a 16th source, the reference.

THE PER-BAND AUC. For source s and target j, within each
10-year age band (18-29, 30-39, ..., 70-79, 80+) and each sex separately (prostate cancer: men only), over j's
denominator: the AUC of score s for j (how often a person with j has the higher score than a person without it,
same band and sex; 0.5 = no separation; below 0.5 = the combination goes the other way). DeLong 95% interval.
A cell is written only where it holds more than 20 cases and more than 20 non-cases; otherwise it is marked
"<=20". Nothing is averaged over bands or sexes.

THE WEIGHTS. w_i from the fit on all of i's denominator (k values per diagnosis), and the similarity of every
pair (Pearson correlation of w_i and w_j over the k groups), with its minimum and maximum over the 10 fold fits.

THE DISSEMINATION POLICY (disclosure.py). No count is written. A per-band AUC is written only
over more than 20 cases and more than 20 non-cases; a withheld cell says only "<=20" (the policy's own display).
The weights and similarities are coefficients over whole denominators. The printed log follows the same rule.
★ PRECISION: AUCs, intervals and correlations are written to 3 decimals and weights to 4 significant
figures (disclosure.coarsen). At full precision an AUC is an exact fraction U / (cases x non-cases), which gives
the cell's counts; summed over bands among men they would return prostate cancer's men-only denominator.

INPUTS   as phecode_directions.py
OUTPUTS  screen_out/phecode_cross/phecode_cross_bands_<enc>.csv     1 row per source x target x band x sex
         screen_out/phecode_cross/phecode_cross_weights_<enc>.csv   1 row per diagnosis x group (full fit)
         screen_out/phecode_cross/phecode_cross_similarity_<enc>.csv 1 row per pair of diagnoses
         screen_out/phecode_cross/phecode_cross_<enc>_meta.json
         screen_out/phecode_cross_<enc>_<CDR>.zip                    those 4, checked by verdict()

Run:  python3 phecode_cross.py                          (gte-large-en-v1.5)
      python3 phecode_cross.py --encoder bge-m3         (one encoder at a time)
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from disclosure import MARK, coarsen, coarsen_sig                        # noqa: E402  the dissemination policy
from phecode_block import delong                                         # noqa: E402  the same DeLong
from phecode_directions import direction_scores                          # noqa: E402  Stage 2's group scores
from phecode_sweep import CONDITIONS, Z, load                            # noqa: E402  Stage 1's labels
from screen_battery import ENCODERS                                      # noqa: E402
from zip_screen_aggregates import verdict                                # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "phecode_cross")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
ENC = sys.argv[sys.argv.index("--encoder") + 1] if "--encoder" in sys.argv else ENCODERS[0]
FOLDS, SEED = 10, 0
BAND_EDGES = [18, 30, 40, 50, 60, 70, 80, np.inf]
BAND_LABELS = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
FEMALE = 0.0                                                             # build_responses_v9.py: female = 0, male = 1
MIN_CELL = 21                                                            # more than 20 cases and non-cases


def fit_combinations(A, Yz, fold, k, verbose=True):
    """Each diagnosis's group combination, out of fold for everyone (the model fitted on its denominator
    outside each fold, applied to everyone in the fold), with the full-fit and per-fold weights. Shared with
    phecode_screened.py, so both use exactly the same scores."""
    N = len(fold)
    age, sex = A["age"], A["sex"]
    scores, W_full, W_fold = {}, {}, {}
    for key, code, label, lst, men_only in CONDITIONS:
        d = A[f"den_{key}"]
        y = np.nan_to_num(A[f"y_{key}"], nan=0).astype(int)
        az_all = (age - age[d].mean()) / age[d].std()

        def design(m):
            cols = [az_all[m], az_all[m] ** 2] + ([] if men_only else [sex[m]])
            return sm.add_constant(np.column_stack(cols + [Yz[m]]), has_constant="add")
        sc = np.full(N, np.nan)
        W_fold[key] = []
        for f in range(FOLDS):
            tr = d & (fold != f)
            res = sm.Logit(y[tr], design(tr)).fit(disp=0, maxiter=200)
            w = np.asarray(res.params)[-k:]
            W_fold[key].append(w)
            te = fold == f
            sc[te] = Yz[te] @ w
        res = sm.Logit(y[d], design(d)).fit(disp=0, maxiter=200)
        W_full[key] = np.asarray(res.params)[-k:]
        scores[key] = sc
        if verbose:
            print(f"  combination fitted: {label}", flush=True)
    return scores, W_full, W_fold


def main():
    assert ENC in ENCODERS, f"--encoder must be one of {ENCODERS}"
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)
    N = len(idx)
    A = load(idx)
    Yz, names, worst = direction_scores(ENC, idx)
    assert worst < 1e-6, f"{ENC}: the recomputed distance differs from scores_{ENC}.csv by {worst}"
    k = Yz.shape[1]
    S = pd.read_csv(os.path.join(OUT_DIR, f"scores_{ENC}.csv"))
    S[ID_COL] = S[ID_COL].astype(str)
    dist = S.set_index(ID_COL).reindex(idx)["distance"].values.astype(float)
    distz = (dist - dist.mean()) / dist.std(ddof=1)
    fold = np.random.default_rng(SEED).permutation(np.arange(N) % FOLDS)
    age, sex = A["age"], A["sex"]
    band = np.digitize(age, BAND_EDGES[1:-1])
    print(f"[phecode_cross] {ENC}: k = {k} | distance reproduced (largest difference {worst:.1e}) | "
          f"{FOLDS} folds over the cohort, seed {SEED}", flush=True)

    # the combinations: out of fold for everyone, and the full fit
    scores, W_full, W_fold = fit_combinations(A, Yz, fold, k)
    scores["distance"] = distz

    # the per-band AUC, per sex, no averaging
    rows = []
    sources = [c[0] for c in CONDITIONS] + ["distance"]
    for key, code, label, lst, men_only in CONDITIONS:
        d = A[f"den_{key}"]
        y = np.nan_to_num(A[f"y_{key}"], nan=0).astype(int)
        sexes = [("men", 1.0)] if men_only else [("women", FEMALE), ("men", 1.0)]
        for b, blab in enumerate(BAND_LABELS):
            for sname, sval in sexes:
                m = d & (band == b) & (sex == sval)
                n1 = int(y[m].sum())
                n0 = int(m.sum()) - n1
                ok = n1 >= MIN_CELL and n0 >= MIN_CELL
                for src in sources:
                    r = {"encoder": ENC, "source": src, "target": key, "target_phecode": code, "band": blab,
                         "sex": sname, "cell": "written" if ok else MARK}
                    if ok:
                        auc, Sg = delong(y[m], scores[src][m][None, :])
                        se = float(np.sqrt(np.atleast_2d(Sg)[0, 0]))
                        r.update({"auc": float(auc[0]), "auc_lo": float(auc[0] - Z * se), "auc_hi": float(auc[0] + Z * se)})
                    rows.append(r)
    B = pd.DataFrame(rows)

    # the weights and their similarity
    wrows = []
    for key, *_ in CONDITIONS:
        for g in range(k):
            wrows.append({"encoder": ENC, "diagnosis": key, "group": g, "weight": float(W_full[key][g])})
    Wt = pd.DataFrame(wrows).merge(names.rename(columns={"direction": "group"}), on="group", how="left")
    srows = []
    keys = [c[0] for c in CONDITIONS]
    for i, a in enumerate(keys):
        for b_ in keys[i + 1:]:
            full = float(np.corrcoef(W_full[a], W_full[b_])[0, 1])
            per = [float(np.corrcoef(W_fold[a][f], W_fold[b_][f])[0, 1]) for f in range(FOLDS)]
            srows.append({"encoder": ENC, "a": a, "b": b_, "corr_full": full, "corr_fold_min": min(per),
                          "corr_fold_max": max(per)})
    Sm = pd.DataFrame(srows)

    B = coarsen(B, ["auc", "auc_lo", "auc_hi"])      # ★ precision (disclosure.py): an AUC is U / (cases x non-cases)
    Wt = coarsen_sig(Wt, ["weight"])
    Sm = coarsen(Sm, ["corr_full", "corr_fold_min", "corr_fold_max"])
    B.to_csv(os.path.join(OUT, f"phecode_cross_bands_{ENC}.csv"), index=False)
    Wt.to_csv(os.path.join(OUT, f"phecode_cross_weights_{ENC}.csv"), index=False)
    Sm.to_csv(os.path.join(OUT, f"phecode_cross_similarity_{ENC}.csv"), index=False)
    json.dump({"cdr": CDR, "encoder": ENC, "k": k, "folds": FOLDS, "seed": SEED, "bands": BAND_LABELS,
               "sources": sources, "targets": keys,
               "combination": "the k group terms of the groups model (age, age^2, sex + k groups; prostate: age, "
                              "age^2 + k groups), fitted outside each fold on the source's denominator, applied to "
                              "everyone in the fold",
               "auc": "per 10-year age band and per sex, over the target's denominator; DeLong 95%; written only "
                      f"with more than 20 cases and non-cases ({MARK} otherwise); no averaging over bands or sexes",
               "similarity": "Pearson correlation of two diagnoses' k weights (full fit), with the min and max over "
                             "the fold fits",
               "disclosure": "All of Us dissemination policy (disclosure.py): no count is written",
               "precision": "AUCs, intervals and correlations to 3 decimals; weights to 4 significant figures "
                            "(disclosure.py)"},
              open(os.path.join(OUT, f"phecode_cross_{ENC}_meta.json"), "w"), indent=2)

    # printed: the two cancers as source and target, and how many cells were written (no counts)
    w = B[B.cell == "written"]
    print(f"\n  cells written {len(w):,} of {len(B):,}")
    for tgt in ("breast", "prostate"):
        for src in ("breast", "prostate", "distance"):
            t = w[(w.target == tgt) & (w.source == src)]
            print(f"  target {tgt:9s} source {src:9s}: " +
                  "  ".join(f"{r.band} {r.sex[0]} {r.auc:.2f}" for r in t.itertuples()))
    print("  weight similarity (full fit), breast vs prostate: "
          f"{Sm[((Sm.a == 'breast') & (Sm.b == 'prostate')) | ((Sm.a == 'prostate') & (Sm.b == 'breast'))].corr_full.iloc[0]:.2f}")

    files = [f"phecode_cross_bands_{ENC}.csv", f"phecode_cross_weights_{ENC}.csv",
             f"phecode_cross_similarity_{ENC}.csv", f"phecode_cross_{ENC}_meta.json"]
    for fn in files:
        ok, why = verdict(os.path.join(OUT, fn))
        assert ok, f"{fn} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"phecode_cross_{ENC}_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"phecode_cross/{fn}")
    print(f"\n[phecode_cross] {len(B)} band rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
