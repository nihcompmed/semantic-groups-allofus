#!/usr/bin/env python3
"""conditions_assoc_jno.py -- the semantic groups and the 8 recorded conditions of conditions_jno.py, one model per
condition (Figure 4, eFigure 3). It fits the model of phecode_fifteen.py on these conditions, with their populations
and their significance thresholds.

THE PEOPLE AND LABELS: conditions_jno.load(), the same populations as Figure 2D (conditions_deciles_jno.py).

ONE MODEL PER CONDITION, all ages, unpenalized logistic, on everyone in the population (no folds):
  condition ~ age_z + age_z^2 (+ sex where the population holds both sexes) + the k groups
  (the group scores of phecode_directions.direction_scores: residualized on age and sex and standardized, as the
  distance uses them; the distance is recomputed and must equal scores_<enc>.csv)
  joint    the Wald test of the k group terms together (k df), at P < .05 (1 test per condition; the conditions
           are separate questions, so no division by their number)
  together each group's odds ratio per SD, net of age, sex and the other groups, at P < .05/k (the search over the
           k groups within a condition; each encoder's own k, 27 for gte)
  alone    each group alone: condition ~ age_z + age_z^2 (+ sex) + that group, at the same P < .05/k

CHECK. GAD, ADHD and PTSD have the same labels and populations in phecode_fifteen.py, so their model must equal the
one there, where that download is on the Workbench: cases, non-cases and the joint Wald statistic, and every group's
odds ratio.

OUTPUTS  screen_out/conditions_assoc_jno/conditions_assoc_jno_<enc>.csv     1 row per condition
         screen_out/conditions_assoc_jno/conditions_assoc_or_jno_<enc>.csv  1 row per condition x group
         screen_out/conditions_assoc_jno/conditions_assoc_jno_<enc>_meta.json
         screen_out/conditions_assoc_jno_<enc>_<CDR>.zip                    those 3 (pii_check; counts as they are)

Run (after conditions_jno.py):  python3 conditions_assoc_jno.py --encoder <encoder>     (gte-large-en-v1.5 by default)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conditions_jno import (CDR, CONDITIONS, ID_COL, OUT_DIR, SEX_ONLY, TESTED_BY, UNCHANGED, load,   # noqa: E402
                            with_sex, zip_out)
from phecode_directions import direction_scores                           # noqa: E402  the group scores
from phecode_fifteen import fit                                           # noqa: E402  the same logistic fit
from phecode_sweep import Z                                               # noqa: E402
from screen_battery import ENCODERS                                       # noqa: E402

OUT = os.path.join(OUT_DIR, "conditions_assoc_jno")
OLD = os.path.join(OUT_DIR, "phecode_fifteen")
ENC = sys.argv[sys.argv.index("--encoder") + 1] if "--encoder" in sys.argv else ENCODERS[0]
P_JOINT = 0.05                                                            # 1 joint test per condition


def base_matrix(A, d, key):
    """age_z, age_z^2 (+ sex) over the population d, age standardized on d."""
    age = A["age"][d]
    az = (age - age.mean()) / age.std()
    return np.column_stack([az, az ** 2] + ([A["sex"][d]] if with_sex(key) else []))


def population_text(key):
    sex = {0.0: "women, ", 1.0: "men, "}.get(SEX_ONLY.get(key), "")
    return sex + (f"tested ({TESTED_BY[key]})" if key in TESTED_BY else "at least 1 ICD event")


def main():
    assert ENC in ENCODERS, f"--encoder must be one of {ENCODERS}"
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)
    A = load(idx)
    Yz, names, worst = direction_scores(ENC, idx)
    assert worst < 1e-6, f"{ENC}: the recomputed distance differs from scores_{ENC}.csv by {worst}"
    k = Yz.shape[1]
    p_group = 0.05 / k
    cp = names["construct_profile"].values if "construct_profile" in names.columns else [""] * k
    print(f"[conditions_assoc_jno] {ENC}: k = {k} | {len(CONDITIONS)} conditions | joint test at P < .05, "
          f"groups at P < .05/{k}", flush=True)

    rows, grows = [], []
    for key, code, label, lst, _ in CONDITIONS:
        d = A[f"den_{key}"]
        y = A[f"y_{key}"][d].astype(int)
        assert len(np.unique(A["sex"][d])) == (2 if with_sex(key) else 1), f"{key}: the sex term does not fit the population"
        base = base_matrix(A, d, key)
        f = fit(y, np.column_stack([base, Yz[d]]))
        b, V = np.asarray(f.params), np.asarray(f.cov_params())
        g = slice(1 + base.shape[1], 1 + base.shape[1] + k)
        w = float(b[g] @ np.linalg.solve(V[g, g], b[g]))
        pj = float(chi2.sf(w, k))
        se, pv = np.asarray(f.bse)[g], np.asarray(f.pvalues)[g]
        n_tog = n_alo = 0
        for j in range(k):
            fa = fit(y, np.column_stack([base, Yz[d][:, j]]))
            ba, sa, pa = float(fa.params[-1]), float(fa.bse[-1]), float(fa.pvalues[-1])
            tog, alo = bool(pv[j] < p_group), bool(pa < p_group)
            n_tog += tog
            n_alo += alo
            grows.append({"encoder": ENC, "condition": key, "phecode": code, "label": label, "list": lst, "group": j,
                          "construct_profile": cp[j],
                          "or": float(np.exp(b[g][j])), "or_lo": float(np.exp(b[g][j] - Z * se[j])),
                          "or_hi": float(np.exp(b[g][j] + Z * se[j])), "p": float(pv[j]), "passes_together": tog,
                          "or_alone": float(np.exp(ba)), "or_alone_lo": float(np.exp(ba - Z * sa)),
                          "or_alone_hi": float(np.exp(ba + Z * sa)), "p_alone": pa, "passes_alone": alo})
        rows.append({"encoder": ENC, "condition": key, "phecode": code, "label": label, "list": lst,
                     "population": population_text(key),
                     "zero_means": "tested, no diagnosis recorded" if key in TESTED_BY else "no diagnosis recorded",
                     "n_cases": int(y.sum()), "n_noncases": int((y == 0).sum()),
                     "groups_wald": w, "groups_df": k, "groups_p": pj, "passes_joint": bool(pj < P_JOINT),
                     "n_groups_together": int(n_tog), "n_groups_alone": int(n_alo),
                     "pseudo_r2": float(f.prsquared), "converged": bool(f.mle_retvals.get("converged", False))})
        print(f"  {lst} {label[:32]:32s} cases {int(y.sum()):>6} / {int((y == 0).sum()):<6} | joint p {pj:.2g} | "
              f"groups together {n_tog:2d}, alone {n_alo:2d}", flush=True)
    D = pd.DataFrame(rows)
    G = pd.DataFrame(grows)

    checked = "phecode_fifteen download not present"
    ofile, gfile = os.path.join(OLD, f"phecode_fifteen_{ENC}.csv"), os.path.join(OLD, f"phecode_fifteen_or_{ENC}.csv")
    if os.path.exists(ofile) and os.path.exists(gfile):                # the unchanged labels reproduce phecode_fifteen
        O = pd.read_csv(ofile).set_index("condition")
        OG = pd.read_csv(gfile)
        for key in UNCHANGED:
            r = D[D.condition == key].iloc[0]
            assert int(O.at[key, "n_cases"]) == r.n_cases and int(O.at[key, "n_noncases"]) == r.n_noncases, \
                f"{key}: the population moved"
            wr = float(O.at[key, "groups_wald"])
            assert abs(r.groups_wald - wr) <= 1e-6 * max(1.0, wr), f"{key}: the joint Wald statistic moved"
            a = G[G.condition == key].sort_values("group")["or"].values
            o = OG[OG.condition == key].sort_values("group")["or"].values
            assert np.allclose(a, o, rtol=1e-6, atol=0), f"{key}: a group's odds ratio moved"
        checked = f"{', '.join(UNCHANGED)}: cases, non-cases, joint Wald and every odds ratio equal phecode_fifteen_{ENC}"
        print(f"  check: {checked}")

    D.to_csv(os.path.join(OUT, f"conditions_assoc_jno_{ENC}.csv"), index=False)
    G.to_csv(os.path.join(OUT, f"conditions_assoc_or_jno_{ENC}.csv"), index=False)
    json.dump({"cdr": CDR, "encoder": ENC, "k": k, "p_joint": P_JOINT, "p_group": p_group,
               "model": "condition ~ age_z + age_z^2 (+ sex where both sexes) + k groups, unpenalized logistic, no folds",
               "thresholds": "joint P < .05 per condition; each group, together and alone, P < .05/k",
               "populations": {"all": "people with at least 1 ICD event", "sex_only": SEX_ONLY, "tested_by": TESTED_BY},
               "check_unchanged": checked, "counts": "written as they are; pii_check"},
              open(os.path.join(OUT, f"conditions_assoc_jno_{ENC}_meta.json"), "w"), indent=2)

    print("\n  groups that pass TOGETHER but not ALONE (size of the odds ratio per SD, together):")
    for r in D.itertuples():
        t = G[(G.condition == r.condition) & G.passes_together & ~G.passes_alone]
        lab = ", ".join(f"G{int(x['group']) + 1} {str(x['construct_profile']).split(':')[0]} "
                        f"{(x['or'] if x['or'] >= 1 else 1 / x['or']):.2f}" for _, x in t.iterrows())
        print(f"  {r.label[:32]:32s} {lab or 'none'}")
    files = [f"conditions_assoc_jno_{ENC}.csv", f"conditions_assoc_or_jno_{ENC}.csv", f"conditions_assoc_jno_{ENC}_meta.json"]
    zpath = zip_out(OUT, files, f"conditions_assoc_jno_{ENC}")
    print(f"\n[conditions_assoc_jno] {len(D)} conditions\n  download -> {zpath}")


if __name__ == "__main__":
    main()
