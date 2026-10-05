#!/usr/bin/env python3
"""phecode_screen_negative_directions.py -- the backtrack for the clinical question: among the people a
published screen does not flag, which semantic directions go with a recorded history of the matching
diagnosis, once the screen's own score is held fixed.

It follows phecode_screen_negative.py (the screen-negative people, before / after, the score), and it uses
Stage 2's direction scores and model (phecode_directions.py). gte-large-en-v1.5 only.

THE THREE PAIRS: PHQ-9 with MB_286.2 MDD, GAD-7 with MB_288.3 GAD, the ASRS with MB_304 ADHD.
Criterion 6 is not among them.

THE PEOPLE, per pair: those who screen negative (the criterion evaluable and not met, at least 1 ICD-coded
event), exactly phecode_screen_negative.py's.
THE OUTCOME: 1 = the diagnosis first coded on or before the answer date of that screen; 0 = no diagnosis
(PheTK count < 2). The people first coded AFTER the survey are LEFT OUT: they had no recorded history at
the survey and are not free of the diagnosis either.

THE DIRECTION SCORES are Stage 2's (phecode_directions.direction_scores): the screen's Yr, residualized on
age and sex, standardized over the cohort of record, so every odds ratio is per SD. It stops unless the
distance recomputed from them equals scores_gte-large-en-v1.5.csv. DIRECTION LEVEL ONLY, read by SIZE:
no item-level reading, no orientation; each direction is named by its construct profile
(basis/components.csv).

TWO LOGISTIC REGRESSIONS per pair, by maximum likelihood with no penalty:
  with_score     outcome ~ all 27 directions + age, age^2, sex + the screen's score as categories
                 (reference 0: PHQ-9 total 0-9, GAD-7 total 0-9, ASRS Part A count 0-3; the terms of
                 phecode_screen_negative.py's column c). THE ANSWER.
  without_score  the same without the score (the comparison: which directions lose their association
                 once the score is held fixed).
  Each direction: odds ratio per SD holding the rest fixed, Wald 95% interval, p;
  bonferroni = p below 0.05 / (27 x 3), a reading aid.
  Checks written per model: convergence, pseudo-R^2, the largest variance inflation factor among the 27
  directions alone and among them within the full design (with the score and covariates).

THE DISSEMINATION POLICY (disclosure.py). Every statistic rests on a whole screen-negative group (more
than 45,000 people and 700 cases). The only counts written are each pair's people in the model and its
cases. Both are already derivable from phecode_screen_negative.csv
(n minus the after cases, and the before cases), and both, with their difference, are checked against 1
to 20. The script also stops unless its own counts equal what phecode_screen_negative.csv publishes.

INPUTS   everything phecode_screen_negative.py reads (the answer-date cache is reused; nothing is pulled),
         everything phecode_directions.py reads, and screen_out/phecode_screen_negative/phecode_screen_negative.csv
OUTPUTS  screen_out/phecode_screen_negative_directions/phecode_screen_negative_directions.csv   1 row per pair x model x direction
         screen_out/phecode_screen_negative_directions/phecode_screen_negative_directions_meta.json
         screen_out/phecode_screen_negative_directions_<CDR>.zip                                those 2, checked by verdict()

Run:  python3 phecode_screen_negative_directions.py
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
from disclosure import small                                                   # noqa: E402  the policy
from phecode_directions import direction_scores                                # noqa: E402  Stage 2's scores
from phecode_screen_negative import (DATES, ENC, MIN_COUNT, PAIRS, PHECODE,    # noqa: E402  the same people
                                     SCORES, Z, answer_dates, first_event_dates, load)
from wb_config import CFG                                                      # noqa: E402
from zip_screen_aggregates import verdict                                      # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "phecode_screen_negative_directions")
PUBLISHED = os.path.join(OUT_DIR, "phecode_screen_negative", "phecode_screen_negative.csv")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
KEYS = ["mdd", "gad", "adhd"]                    # the three pairs of the backtrack


def max_vif(M):
    """The largest variance inflation factor among the columns of M (no constant)."""
    R = np.corrcoef(M, rowvar=False)
    return float(np.max(np.diag(np.linalg.inv(R))))


def main():
    os.makedirs(OUT, exist_ok=True)
    assert os.path.exists(DATES), f"{DATES} is missing: run phecode_screen_negative.py first"
    assert os.path.exists(PUBLISHED), f"{PUBLISHED} is missing: run phecode_screen_negative.py first"
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"))
    cf[ID_COL] = cf[ID_COL].astype(str)
    idx = pd.Index(cf[ID_COL], name=ID_COL)
    A = load(idx)
    has_rec = A["has_rec"]
    D = answer_dates(idx)
    FE = first_event_dates(idx)
    items = [i for key in KEYS for i in dict((p[0], p[3]) for p in PAIRS)[key]]
    Rsp = pd.read_csv(CFG.responses_csv, usecols=[ID_COL] + sorted(set(items)))
    Rsp[ID_COL] = Rsp[ID_COL].astype(str)
    Rsp = Rsp.set_index(ID_COL).reindex(idx)
    Yz, names, worst = direction_scores(ENC, idx)
    assert worst < 1e-6, f"{ENC}: the recomputed distance differs from scores_{ENC}.csv by {worst}"
    k = Yz.shape[1]
    bonf = 0.05 / (k * len(KEYS))
    pub = pd.read_csv(PUBLISHED)
    print(f"[phecode_screen_negative_directions] {ENC}: k = {k} | distance reproduced (largest difference {worst:.1e})")

    rows, meta = [], {}
    for key in KEYS:
        _, crit, rule, _ = next(p for p in PAIRS if p[0] == key)
        code, label = PHECODE[key]
        sname, sfun, top = SCORES[key]
        v = pd.to_numeric(cf[crit], errors="coerce").values
        neg = (v == 0) & has_rec
        adate = D[D["pair"] == key].set_index(ID_COL)["answer_date"].reindex(idx).values
        fe, cnt = FE[key]
        case = has_rec & (cnt >= MIN_COUNT)
        before = case & (fe <= adate)
        after = case & (fe > adate)
        # the same people and counts as the file already published
        t = pub[(pub.pair == key) & (pub.decile == 0)].iloc[0]
        assert int(t["n"]) == int(neg.sum()), f"{key}: {int(neg.sum())} screen-negative here, {t['n']} published"
        for kind, cell in (("before", before), ("after", after)):
            if pd.notna(t[f"{kind}_rate"]):
                assert round(t[f"{kind}_rate"] * t["n"]) == int((neg & cell).sum()), f"{key}: {kind} cases differ"
        m = neg & ~after                                 # the people first coded after are left out
        y = before[m].astype(float)
        n, n_case = int(m.sum()), int(y.sum())
        assert not (small(n) or small(n_case) or small(n - n_case)), f"{key}: a count of 1 to 20"
        score = np.asarray(sfun(Rsp), dtype=float)[m]
        assert not np.isnan(score).any() and score.min() >= 0 and score.max() <= top, f"{key}: score out of range"
        cats = list(range(1, top + 1))
        for c in [0] + cats:
            assert not small((score == c).sum()) and (score == c).sum() > 0, f"{key}: score {c} holds 1 to 20 people"
        age = A["age"][m]
        az = (age - age.mean()) / age.std()
        cov = [az, az ** 2, A["sex"][m]]
        dummies = [(score == c).astype(float) for c in cats]
        Y = Yz[m]
        meta[key] = {"phecode": code, "diagnosis": label, "criterion": crit, "screen_negative": rule,
                     "score": sname, "n_people": n, "n_cases": n_case,
                     "outcome": "first coded on or before the survey (1) vs no diagnosis (0); first coded after: left out",
                     "max_vif_directions": max_vif(Y), "models": {}}
        for model, extra in (("with_score", dummies), ("without_score", [])):
            X = np.column_stack(cov + extra + [Y])
            fit = sm.Logit(y, sm.add_constant(X)).fit(disp=0, maxiter=200)
            b, se, p = fit.params[-k:], fit.bse[-k:], fit.pvalues[-k:]
            vif_full = np.diag(np.linalg.inv(np.corrcoef(X, rowvar=False)))[-k:]
            meta[key]["models"][model] = {"converged": bool(fit.mle_retvals.get("converged", False)),
                                          "pseudo_r2": float(fit.prsquared),
                                          "max_vif_directions_in_design": float(np.max(vif_full))}
            for j in range(k):
                rows.append({"encoder": ENC, "pair": key, "criterion": crit, "phecode": code, "model": model,
                             "direction": j, "or": np.exp(b[j]), "or_lo": np.exp(b[j] - Z * se[j]),
                             "or_hi": np.exp(b[j] + Z * se[j]), "p": p[j], "bonferroni": bool(p[j] < bonf)})
            top5 = np.argsort(-np.abs(b))[:5]
            mm = meta[key]["models"][model]
            print(f"  {crit} <-> {code} [{model}] n {n:,} cases {n_case:,} | converged {mm['converged']} | "
                  f"pseudo-R2 {mm['pseudo_r2']:.3f} | max VIF {mm['max_vif_directions_in_design']:.2f} | "
                  f"largest |log OR|: " + ", ".join(f"d{j} {np.exp(b[j]):.2f}" for j in top5))
    R_ = pd.DataFrame(rows).merge(names, on="direction", how="left")
    R_.to_csv(os.path.join(OUT, "phecode_screen_negative_directions.csv"), index=False)
    json.dump({"cdr": CDR, "encoder": ENC, "k": k, "bonferroni_threshold": bonf, "pairs": meta,
               "scores": "Stage 2's: Yr of the screen (age and sex residualized), standardized over the cohort of "
                         "record; direction level, read by size (the sign is the basis's own)",
               "with_score": "logistic regression on all k directions + age, age^2, sex + the screen's score as "
                             "categories (reference 0); OR per SD, Wald 95% CI",
               "without_score": "the same without the score",
               "people": "each pair's screen-negative people (phecode_screen_negative.py), less those first coded after",
               "disclosure": "All of Us dissemination policy (disclosure.py)"},
              open(os.path.join(OUT, "phecode_screen_negative_directions_meta.json"), "w"), indent=2)
    files = ["phecode_screen_negative_directions.csv", "phecode_screen_negative_directions_meta.json"]
    for fn in files:
        ok, why = verdict(os.path.join(OUT, fn))
        assert ok, f"{fn} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"phecode_screen_negative_directions_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"phecode_screen_negative_directions/{fn}")
    print(f"\n[phecode_screen_negative_directions] {len(R_)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
