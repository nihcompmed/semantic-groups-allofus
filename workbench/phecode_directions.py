#!/usr/bin/env python3
"""phecode_directions.py -- STAGE 2 of the phecodes: which semantic directions go with each diagnosis.
gte-large-en-v1.5 first.

THE PEOPLE. Each diagnosis's denominator: the people with at least 1 ICD-coded event
(61,631), and for prostate cancer the men among them. The labels are phecode_sweep.py's own (load()):
1 = PheTK count >= 2.

THE DIRECTION SCORES are exactly those that enter the distance, recomputed with
screen_battery.py's own functions:
  - Y = X P over the cohort of record, and Yr = Y residualized on age and sex;
  - checked: the distance recomputed from Yr equals scores_<enc>.csv (it stops otherwise);
  - each direction standardized to unit SD over the cohort of record, so every number is per SD.
  - DIRECTION LEVEL ONLY: nothing is linked to individual items, and the scores on the directions are
    the predictors. The scores are used exactly as the screen computes them, with no
    flip and no item-level reading. A direction's sign is the basis's own and has no meaning of its
    own, so a direction's association is read by its SIZE (|SMD|, |log OR|). Each direction is named
    by its construct profile (basis/components.csv). Orienting the directions by their core items'
    loadings would change no size and no p.

TWO ANALYSES, per diagnosis and direction:
  1. EACH DIRECTION ON ITS OWN: the standardized mean difference, the mean score of the people with the
     diagnosis minus that of the people without it, in SD units, with a normal 95% interval. The scores
     are already residualized on age and sex.
  2. ALL DIRECTIONS TOGETHER: a logistic regression of the diagnosis on all k scores plus age, age
     squared and sex (prostate cancer: age and age squared), by maximum likelihood with no penalty.
     Each direction's odds ratio per SD holds the other k - 1 fixed, with a Wald 95% interval and
     p-value. The largest variance inflation factor among the scores is written as a check.
  bonferroni = p below 0.05 / (k x 15), written as a reading aid.

THE DISSEMINATION POLICY (disclosure.py). Every statistic rests on a whole
denominator of at least 61,631 people (prostate: the men among them) and at least 688 cases. The only
counts written are each diagnosis's denominator and cases, which are checked against 1 to 20. ★ For
prostate cancer neither is written NOR PRINTED: against the coverage table's both-sex count they give
the women with a prostate cancer label.

INPUTS   as phecode_sweep.py, plus the responses, the items table and basis/ (screen_battery.py's inputs)
         and scores_<enc>.csv
OUTPUTS  screen_out/phecode_directions/phecode_directions_<enc>.csv       1 row per diagnosis x direction
         screen_out/phecode_directions/phecode_directions_<enc>_meta.json per diagnosis: n, cases,
                                                                           convergence, largest VIF
         screen_out/phecode_directions_<enc>_<CDR>.zip                     those 2, checked by verdict()

Run:  python3 phecode_directions.py                          (gte-large-en-v1.5)
      python3 phecode_directions.py --encoder bge-m3         (one encoder at a time)
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
from disclosure import small                                             # noqa: E402  the dissemination policy
from phecode_sweep import CONDITIONS, Z, load                            # noqa: E402  Stage 1's labels
from screen_battery import (BASIS_DIR, ENCODERS, ITEMS_CSV, load_battery,  # noqa: E402  the screen's own scores
                            mahalanobis2, place_battery, wls_residualize)
from zip_screen_aggregates import verdict                                # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "phecode_directions")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
ENC = sys.argv[sys.argv.index("--encoder") + 1] if "--encoder" in sys.argv else ENCODERS[0]


def direction_scores(enc, idx):
    """Yr for the cohort in `idx` order, standardized over the cohort, as the screen computes it.
    Returns the scores, the direction names, and the largest distance difference from scores_<enc>.csv."""
    items = pd.read_csv(ITEMS_CSV)["item"].astype(str).tolist()
    resp, X, cov, w = load_battery(items)
    P, _ = place_battery(enc, items)
    Yr = wls_residualize(X @ P, cov, w)
    T2, _ = mahalanobis2(Yr, w)
    S = pd.read_csv(os.path.join(OUT_DIR, f"scores_{enc}.csv"))
    S[ID_COL] = S[ID_COL].astype(str)
    d_saved = S.set_index(ID_COL).reindex(resp.index)["distance"].values
    worst = float(np.max(np.abs(np.sqrt(np.maximum(T2, 0)) - d_saved)))
    Yz = (Yr - Yr.mean(axis=0)) / Yr.std(axis=0, ddof=1)
    comp = pd.read_csv(os.path.join(BASIS_DIR, "components.csv"))
    comp = comp[comp.encoder == enc].sort_values("component")
    assert len(comp) == P.shape[1], f"{enc}: components.csv has {len(comp)} rows for k = {P.shape[1]}"
    assert list(comp["component"]) == list(range(P.shape[1])), f"{enc}: components are not 0..k-1 in order"
    rows = [{"direction": int(c.component), "construct_profile": c.construct_profile,
             "n_core_items": int(c.n_core_items)} for c in comp.itertuples()]
    Yz = pd.DataFrame(Yz, index=resp.index).reindex(idx).values
    return Yz, pd.DataFrame(rows), worst


def main():
    assert ENC in ENCODERS, f"--encoder must be one of {ENCODERS}"
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)
    A = load(idx)
    Yz, names, worst = direction_scores(ENC, idx)
    assert worst < 1e-6, f"{ENC}: the recomputed distance differs from scores_{ENC}.csv by {worst}"
    k = Yz.shape[1]
    bonf = 0.05 / (k * len(CONDITIONS))
    print(f"[phecode_directions] {ENC}: k = {k} | distance reproduced (largest difference {worst:.1e})")

    rows, meta = [], {}
    for key, code, label, lst, men_only in CONDITIONS:
        d = A[f"den_{key}"]
        y = A[f"y_{key}"][d]
        Y = Yz[d]
        n, n_case = int(d.sum()), int(y.sum())
        assert not (small(n) or small(n_case) or small(n - n_case)), f"{key}: a count of 1 to 20"
        # 1. each direction on its own
        c, nc = Y[y == 1], Y[y == 0]
        smd = c.mean(axis=0) - nc.mean(axis=0)
        se1 = np.sqrt(c.var(axis=0, ddof=1) / len(c) + nc.var(axis=0, ddof=1) / len(nc))
        # 2. all directions together
        age = A["age"][d]
        az = (age - age.mean()) / age.std()
        cov = [az, az ** 2] + ([] if men_only else [A["sex"][d]])
        Xd = sm.add_constant(np.column_stack(cov + [Y]))
        fit = sm.Logit(y, Xd).fit(disp=0, maxiter=200)
        b, se2, p = fit.params[-k:], fit.bse[-k:], fit.pvalues[-k:]
        R = np.corrcoef(Y, rowvar=False)
        vif = float(np.max(np.diag(np.linalg.inv(R))))
        meta[key] = {"phecode": code, "label": label, "list": lst,
                     "converged": bool(fit.mle_retvals.get("converged", False)), "max_vif": vif,
                     "pseudo_r2": float(fit.prsquared)}
        if not men_only:                             # ★ a sex-restricted denominator is never written or printed
            meta[key].update({"n_denominator": n, "n_cases": n_case})
        for j in range(k):
            rows.append({"encoder": ENC, "condition": key, "phecode": code, "direction": j,
                         "smd": smd[j], "smd_lo": smd[j] - Z * se1[j], "smd_hi": smd[j] + Z * se1[j],
                         "or": np.exp(b[j]), "or_lo": np.exp(b[j] - Z * se2[j]), "or_hi": np.exp(b[j] + Z * se2[j]),
                         "p": p[j], "bonferroni": bool(p[j] < bonf)})
        top = np.argsort(-np.abs(b))[:3]
        counts = "men only" if men_only else f"n {n:,} cases {n_case:,}"
        print(f"  {lst} {label[:38]:38s} {counts} | converged {meta[key]['converged']} | "
              f"max VIF {vif:.2f} | largest |log OR|: " +
              ", ".join(f"d{j} OR {np.exp(b[j]):.2f}" for j in top))
    D = pd.DataFrame(rows).merge(names, on="direction", how="left")
    D.to_csv(os.path.join(OUT, f"phecode_directions_{ENC}.csv"), index=False)
    json.dump({"cdr": CDR, "encoder": ENC, "k": k, "bonferroni_threshold": bonf, "conditions": meta,
               "scores": "Yr of the screen (age and sex residualized), standardized over the cohort of record; "
                         "direction level, read by size (the sign is the basis's own)",
               "analysis_1": "standardized mean difference, cases minus non-cases, normal 95% CI",
               "analysis_2": "logistic regression on all k scores + age, age^2, sex (prostate: age, age^2); OR per SD, Wald 95% CI",
               "disclosure": "All of Us dissemination policy (disclosure.py)"},
              open(os.path.join(OUT, f"phecode_directions_{ENC}_meta.json"), "w"), indent=2)
    files = [f"phecode_directions_{ENC}.csv", f"phecode_directions_{ENC}_meta.json"]
    for fn in files:
        ok, why = verdict(os.path.join(OUT, fn))
        assert ok, f"{fn} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"phecode_directions_{ENC}_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"phecode_directions/{fn}")
    print(f"\n[phecode_directions] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
