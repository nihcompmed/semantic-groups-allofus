#!/usr/bin/env python3
"""phecode_fifteen_adjusted.py -- the 15 phecode diagnoses with income, education, insurance and access held fixed.

THE PEOPLE AND LABELS are phecode_fifteen.py's (the 0 corrected for breast and prostate cancer, hypercholesterolemia
and chronic kidney disease: both sides restricted to people whose record shows the test; the other 11: 0 = no
diagnosis recorded).

THE CONTROLS, all self-reported, none among the 151 items, every non-answer kept as its own level (nobody dropped):
  income      <25k / 25k to <50k / 50k to <100k / 100k to <150k / 150k or more / Prefer not to answer (The Basics)
  education   Less than high school / High school or GED / Some college / College graduate or more / Prefer not to
              answer (The Basics)
  insurance   Yes / No / Prefer not to answer (The Basics)
  access      any access barrier: yes / no / not answered (Health Care Access survey)
  cost        any cost barrier: yes / no / not answered (the same survey)
  income, education and insurance from covariates.py (screen_out/covariates_per_person.csv); the two barriers from
  care.py (screen_out/care_outcomes_per_person.csv). "Avoided care because of how a provider treated me" is NOT used:
  it overlaps the healthcare-discrimination group. Each control enters as categories (reference = its most common
  level).

TWO MODELS PER CONDITION, all ages, no folds, unpenalized logistic:
  before   condition ~ age_z + age_z^2 (+ sex) + the k groups            (= phecode_fifteen.py's; checked)
  after    the same + income + education + insurance + access + cost
  per condition: the joint Wald test of the k groups before and after (p < 0.05/15); groups passing together before
  and after, and ALONE after (condition ~ age_z + age_z^2 (+ sex) + the controls + that group), all at p < 0.05/(k x 15);
  per group: the odds ratio before and after, and the share of its log odds ratio kept after the control.

COUNTS are written as they are; the only check is pii_check (no person-level column).

INPUTS   as phecode_fifteen.py (its caches: screening_checks_local.csv, lab_checks_local.csv), plus
         screen_out/covariates_per_person.csv (run covariates.py if missing) and screen_out/care_outcomes_per_person.csv
OUTPUTS  screen_out/phecode_fifteen_adjusted/phecode_fifteen_adjusted_<enc>.csv       1 row per condition
         screen_out/phecode_fifteen_adjusted/phecode_fifteen_adjusted_or_<enc>.csv    1 row per condition x group
         screen_out/phecode_fifteen_adjusted/phecode_fifteen_adjusted_<enc>_meta.json
         screen_out/phecode_fifteen_adjusted_<enc>_<CDR>.zip                          those 3

Run:  python3 phecode_fifteen_adjusted.py              (gte-large-en-v1.5)
      python3 phecode_fifteen_adjusted.py --encoder bge-m3
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phecode_directions import direction_scores                          # noqa: E402
from phecode_fifteen import CHECKED_BY, LABS, N_COND, P_JOINT, SCREEN_CHECKS, fit, pii_check, pull_labs   # noqa: E402
from phecode_sweep import CONDITIONS, FEMALE, Z, load                    # noqa: E402
from screen_battery import ENCODERS                                      # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "phecode_fifteen_adjusted")
BEFORE = os.path.join(OUT_DIR, "phecode_fifteen")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
ENC = sys.argv[sys.argv.index("--encoder") + 1] if "--encoder" in sys.argv else ENCODERS[0]
COVS = ["income", "education", "insurance"]
BARRIERS = {"access": "any_access_barrier", "cost": "any_cost_barrier"}


def controls(idx):
    """The 5 controls per person, each a categorical level (non-answers kept as levels)."""
    p = os.path.join(OUT_DIR, "covariates_per_person.csv")
    assert os.path.exists(p), f"{p} is missing: run `python3 covariates.py` first"
    Cv = pd.read_csv(p, dtype={ID_COL: str}).set_index(ID_COL).reindex(idx)
    q = os.path.join(OUT_DIR, "care_outcomes_per_person.csv")
    assert os.path.exists(q), f"{q} is missing: run `python3 care.py --outcomes-only` first"
    O = pd.read_csv(q, dtype={ID_COL: str}).set_index(ID_COL).reindex(idx)
    out = pd.DataFrame(index=idx)
    for c in COVS:
        out[c] = Cv[c].fillna("Prefer not to answer").astype(str)
    surv = O["has_survey"].fillna(False).astype(bool)
    for name, col in BARRIERS.items():
        v = O[col]
        out[name] = np.where(~surv | v.isna(), "not answered", np.where(v.astype(float) > 0, "yes", "no"))
    return out


def dummies(C, mask):
    """Indicator columns for the controls within a population: reference = the most common level; constant columns
    dropped. Returns (matrix, column names)."""
    cols, names = [], []
    for c in C.columns:
        v = C[c].values[mask]
        levels, counts = np.unique(v, return_counts=True)
        ref = levels[np.argmax(counts)]
        for lv in levels:
            if lv == ref:
                continue
            x = (v == lv).astype(float)
            # a level identical to one already in (the same people skipped both barrier questions) adds nothing
            if 0 < x.sum() < len(x) and not any(np.array_equal(x, y_) for y_ in cols):
                cols.append(x)
                names.append(f"{c}={lv}")
    if not cols:
        return np.zeros((int(mask.sum()), 0)), names
    X = np.column_stack(cols)
    keep = []                                                     # and no column that the others already determine
    for j in range(X.shape[1]):
        if np.linalg.matrix_rank(np.column_stack([np.ones(len(X))] + [X[:, i] for i in keep + [j]])) == len(keep) + 2:
            keep.append(j)
    return X[:, keep], [names[j] for j in keep]


def main():
    assert ENC in ENCODERS, f"--encoder must be one of {ENCODERS}"
    assert os.path.exists(SCREEN_CHECKS), f"{SCREEN_CHECKS} is missing: run phecode_screened.py first"
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)
    A = load(idx)
    Yz, names, worst = direction_scores(ENC, idx)
    assert worst < 1e-6, f"{ENC}: the recomputed distance differs from scores_{ENC}.csv by {worst}"
    k = Yz.shape[1]
    p_group = 0.05 / (k * N_COND)
    S = pd.read_csv(SCREEN_CHECKS, dtype={ID_COL: str})
    L, _ = pull_labs(idx)
    checks = {("screen", kind): idx.isin(S.loc[S["kind"] == kind, ID_COL]) for kind in ("breast", "prostate")}
    checks.update({("lab", kind): idx.isin(L.loc[L["kind"] == kind, ID_COL]) for kind in LABS})
    Ctl = controls(idx)
    bfile = os.path.join(BEFORE, f"phecode_fifteen_{ENC}.csv")
    before_ref = pd.read_csv(bfile).set_index("condition") if os.path.exists(bfile) else None
    cp = names["construct_profile"].values if "construct_profile" in names.columns else [""] * k
    print(f"[phecode_fifteen_adjusted] {ENC}: k = {k} | controls: income, education, insurance, access barrier, cost "
          f"barrier | joint test at p < 0.05/15, groups at p < 0.05/{k * N_COND}", flush=True)
    for c in Ctl.columns:
        print(f"  [{c}: " + ", ".join(f"{lv} {n:,}" for lv, n in Ctl[c].value_counts().items()) + "]", flush=True)

    rows, grows = [], []
    for key, code, label, lst, men_only in CONDITIONS:
        d = A[f"den_{key}"].copy()
        if key == "breast":
            d &= A["sex"] == FEMALE
        checked = key in CHECKED_BY
        if checked:
            d &= checks[CHECKED_BY[key]]
        y = A[f"y_{key}"][d].astype(int)
        age = A["age"][d]
        az = (age - age.mean()) / age.std()
        both_sexes = len(np.unique(A["sex"][d])) > 1
        base = np.column_stack([az, az ** 2] + ([A["sex"][d]] if both_sexes else []))
        Xc, cnames = dummies(Ctl, d)
        res = {}
        for when, B_ in (("before", base), ("after", np.column_stack([base, Xc]))):
            f = fit(y, np.column_stack([B_, Yz[d]]))
            b, V = np.asarray(f.params), np.asarray(f.cov_params())
            g = slice(1 + B_.shape[1], 1 + B_.shape[1] + k)
            w = float(b[g] @ np.linalg.solve(V[g, g], b[g]))
            res[when] = {"b": b[g], "se": np.asarray(f.bse)[g], "p": np.asarray(f.pvalues)[g], "w": w,
                         "pj": float(chi2.sf(w, k)), "conv": bool(f.mle_retvals.get("converged", False)), "B": B_}
        if before_ref is not None and key in before_ref.index:           # model "before" = phecode_fifteen.py's
            wr = float(before_ref.at[key, "groups_wald"])
            assert abs(res["before"]["w"] - wr) <= 1e-6 * max(1.0, wr), f"{key}: 'before' differs from phecode_fifteen.py"
        n_tb = int((res["before"]["p"] < p_group).sum())
        n_ta = int((res["after"]["p"] < p_group).sum())
        n_aa = 0
        for j in range(k):
            fa = fit(y, np.column_stack([res["after"]["B"], Yz[d][:, j]]))
            ba, sa, pa = float(fa.params[-1]), float(fa.bse[-1]), float(fa.pvalues[-1])
            n_aa += pa < p_group
            b0, b1 = res["before"]["b"][j], res["after"]["b"][j]
            grows.append({"encoder": ENC, "condition": key, "phecode": code, "label": label, "list": lst, "group": j,
                          "construct_profile": cp[j],
                          "or_before": float(np.exp(b0)), "p_before": float(res["before"]["p"][j]),
                          "passes_before": bool(res["before"]["p"][j] < p_group),
                          "or_after": float(np.exp(b1)), "or_after_lo": float(np.exp(b1 - Z * res["after"]["se"][j])),
                          "or_after_hi": float(np.exp(b1 + Z * res["after"]["se"][j])),
                          "p_after": float(res["after"]["p"][j]), "passes_after": bool(res["after"]["p"][j] < p_group),
                          "share_kept": float(b1 / b0) if b0 != 0 else np.nan,
                          "or_alone_after": float(np.exp(ba)), "p_alone_after": pa, "passes_alone_after": bool(pa < p_group)})
        rows.append({"encoder": ENC, "condition": key, "phecode": code, "label": label, "list": lst,
                     "zero_means": "checked, no diagnosis recorded" if checked else "no diagnosis recorded",
                     "n_cases": int(y.sum()), "n_noncases": int((y == 0).sum()), "n_control_columns": len(cnames),
                     "groups_wald_before": res["before"]["w"], "groups_p_before": res["before"]["pj"],
                     "groups_wald_after": res["after"]["w"], "groups_p_after": res["after"]["pj"],
                     "passes_joint_after": bool(res["after"]["pj"] < P_JOINT),
                     "n_groups_together_before": n_tb, "n_groups_together_after": n_ta,
                     "n_groups_alone_after": int(n_aa),
                     "converged": bool(res["before"]["conv"] and res["after"]["conv"])})
        print(f"  {lst} {label[:40]:40s} {'(checked)' if checked else '         '} joint p {res['before']['pj']:.2g} -> "
              f"{res['after']['pj']:.2g} | together {n_tb:2d} -> {n_ta:2d}, alone after {int(n_aa):2d}", flush=True)

    D = pd.DataFrame(rows)
    G = pd.DataFrame(grows)
    D.to_csv(os.path.join(OUT, f"phecode_fifteen_adjusted_{ENC}.csv"), index=False)
    G.to_csv(os.path.join(OUT, f"phecode_fifteen_adjusted_or_{ENC}.csv"), index=False)
    json.dump({"cdr": CDR, "encoder": ENC, "k": k, "p_joint": P_JOINT, "p_group": p_group,
               "controls": {"income": "The Basics", "education": "The Basics", "insurance": "The Basics",
                            "access": "any access barrier (Health Care Access survey)",
                            "cost": "any cost barrier (Health Care Access survey)"},
               "coding": "categories, non-answers as levels, reference = most common level in the population",
               "not_used": "avoided care because of provider treatment (overlaps the healthcare-discrimination group)",
               "counts": "written as they are; pii_check only"},
              open(os.path.join(OUT, f"phecode_fifteen_adjusted_{ENC}_meta.json"), "w"), indent=2)

    f_ = lambda x: x if x >= 1 else 1 / x
    for c in ("breast", "prostate"):
        t = G[(G.condition == c) & (G.passes_before | G.passes_after)].sort_values("p_before")
        print(f"\n  {c}: groups passing before or after (fold from 1 per SD, together; before -> after; share of the log "
              "odds ratio kept)")
        for _, x in t.iterrows():
            print(f"    {int(x['group']):2d} {str(x['construct_profile']).split(':')[0][:32]:32s} {f_(x['or_before']):.2f} -> "
                  f"{f_(x['or_after']):.2f} ({100 * x['share_kept']:.0f}% kept){'' if x['passes_after'] else '  no longer passes'}")
    files = [f"phecode_fifteen_adjusted_{ENC}.csv", f"phecode_fifteen_adjusted_or_{ENC}.csv",
             f"phecode_fifteen_adjusted_{ENC}_meta.json"]
    for fn in files:
        ok, why = pii_check(os.path.join(OUT, fn))
        assert ok, f"{fn} refused: {why}"
    zpath = os.path.join(OUT_DIR, f"phecode_fifteen_adjusted_{ENC}_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"phecode_fifteen_adjusted/{fn}")
    print(f"\n[phecode_fifteen_adjusted] {len(D)} conditions\n  download -> {zpath}")


if __name__ == "__main__":
    main()
