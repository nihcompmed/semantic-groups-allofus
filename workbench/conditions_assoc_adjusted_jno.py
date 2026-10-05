#!/usr/bin/env python3
"""conditions_assoc_adjusted_jno.py -- the semantic groups and the recorded conditions beside the main model
(eMethods 4): the 8 recorded conditions of conditions_jno.py with income, education, insurance and the access and cost
barriers added. The main model adjusts for age and sex only, and this model is shown beside it. It uses the controls
and coding of phecode_fifteen_adjusted.py (its own controls() and dummies()), with these conditions, populations and
thresholds.

TWO MODELS PER CONDITION, all ages, no folds, unpenalized logistic, on conditions_jno.load()'s population:
  before   condition ~ age_z + age_z^2 (+ sex) + the k groups      (= conditions_assoc_jno.py's model; checked)
  after    the same + income + education + insurance + access barrier + cost barrier (categories, non-answers kept as
           levels, reference = the most common level in the population)
  per condition: the joint Wald test of the k groups before and after (P < .05); groups passing together before and
  after, and alone after (condition ~ age_z + age_z^2 (+ sex) + the controls + that group), at P < .05/k;
  per group: the odds ratio before and after, and the share of its log odds ratio kept after the controls.

OUTPUTS  screen_out/conditions_assoc_adjusted_jno/conditions_assoc_adjusted_jno_<enc>.csv     1 row per condition
         screen_out/conditions_assoc_adjusted_jno/conditions_assoc_adjusted_or_jno_<enc>.csv  1 row per condition x group
         screen_out/conditions_assoc_adjusted_jno/conditions_assoc_adjusted_jno_<enc>_meta.json
         screen_out/conditions_assoc_adjusted_jno_<enc>_<CDR>.zip                             those 3 (pii_check)

Run (after conditions_assoc_jno.py for the same encoder):  python3 conditions_assoc_adjusted_jno.py --encoder <encoder>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conditions_assoc_jno import ENC, P_JOINT, base_matrix                # noqa: E402
from conditions_jno import CDR, CONDITIONS, ID_COL, OUT_DIR, TESTED_BY, load, zip_out   # noqa: E402
from phecode_directions import direction_scores                           # noqa: E402
from phecode_fifteen import fit                                           # noqa: E402
from phecode_fifteen_adjusted import controls, dummies                    # noqa: E402  the same controls and coding
from phecode_sweep import Z                                               # noqa: E402
from screen_battery import ENCODERS                                       # noqa: E402

OUT = os.path.join(OUT_DIR, "conditions_assoc_adjusted_jno")
BEFORE = os.path.join(OUT_DIR, "conditions_assoc_jno")


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
    Ctl = controls(idx)
    bfile = os.path.join(BEFORE, f"conditions_assoc_jno_{ENC}.csv")
    assert os.path.exists(bfile), f"{bfile} is missing: run `python3 conditions_assoc_jno.py --encoder {ENC}` first"
    before_ref = pd.read_csv(bfile).set_index("condition")
    cp = names["construct_profile"].values if "construct_profile" in names.columns else [""] * k
    print(f"[conditions_assoc_adjusted_jno] {ENC}: k = {k} | controls: income, education, insurance, access barrier, "
          f"cost barrier | joint test at P < .05, groups at P < .05/{k}", flush=True)
    for c in Ctl.columns:
        print(f"  [{c}: " + ", ".join(f"{lv} {n:,}" for lv, n in Ctl[c].value_counts().items()) + "]", flush=True)

    rows, grows = [], []
    for key, code, label, lst, _ in CONDITIONS:
        d = A[f"den_{key}"]
        y = A[f"y_{key}"][d].astype(int)
        base = base_matrix(A, d, key)
        Xc, cnames = dummies(Ctl, d)
        res = {}
        for when, B_ in (("before", base), ("after", np.column_stack([base, Xc]))):
            f = fit(y, np.column_stack([B_, Yz[d]]))
            b, V = np.asarray(f.params), np.asarray(f.cov_params())
            g = slice(1 + B_.shape[1], 1 + B_.shape[1] + k)
            w = float(b[g] @ np.linalg.solve(V[g, g], b[g]))
            res[when] = {"b": b[g], "se": np.asarray(f.bse)[g], "p": np.asarray(f.pvalues)[g], "w": w,
                         "pj": float(chi2.sf(w, k)), "conv": bool(f.mle_retvals.get("converged", False)), "B": B_}
        wr = float(before_ref.at[key, "groups_wald"])                    # "before" = conditions_assoc_jno.py's model
        assert abs(res["before"]["w"] - wr) <= 1e-6 * max(1.0, wr), f"{key}: 'before' differs from conditions_assoc_jno.py"
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
                          "or_alone_after": float(np.exp(ba)), "or_alone_after_lo": float(np.exp(ba - Z * sa)),
                          "or_alone_after_hi": float(np.exp(ba + Z * sa)), "p_alone_after": pa,
                          "passes_alone_after": bool(pa < p_group)})
        rows.append({"encoder": ENC, "condition": key, "phecode": code, "label": label, "list": lst,
                     "zero_means": "tested, no diagnosis recorded" if key in TESTED_BY else "no diagnosis recorded",
                     "n_cases": int(y.sum()), "n_noncases": int((y == 0).sum()), "n_control_columns": len(cnames),
                     "groups_wald_before": res["before"]["w"], "groups_p_before": res["before"]["pj"],
                     "groups_wald_after": res["after"]["w"], "groups_p_after": res["after"]["pj"],
                     "passes_joint_after": bool(res["after"]["pj"] < P_JOINT),
                     "n_groups_together_before": n_tb, "n_groups_together_after": n_ta,
                     "n_groups_alone_after": int(n_aa),
                     "converged": bool(res["before"]["conv"] and res["after"]["conv"])})
        print(f"  {lst} {label[:32]:32s} joint p {res['before']['pj']:.2g} -> {res['after']['pj']:.2g} | "
              f"together {n_tb:2d} -> {n_ta:2d}, alone after {int(n_aa):2d}", flush=True)

    D = pd.DataFrame(rows)
    G = pd.DataFrame(grows)
    D.to_csv(os.path.join(OUT, f"conditions_assoc_adjusted_jno_{ENC}.csv"), index=False)
    G.to_csv(os.path.join(OUT, f"conditions_assoc_adjusted_or_jno_{ENC}.csv"), index=False)
    json.dump({"cdr": CDR, "encoder": ENC, "k": k, "p_joint": P_JOINT, "p_group": p_group,
               "controls": {"income": "The Basics", "education": "The Basics", "insurance": "The Basics",
                            "access": "any access barrier (Health Care Access survey)",
                            "cost": "any cost barrier (Health Care Access survey)"},
               "coding": "categories, non-answers as levels, reference = most common level in the population",
               "not_used": "avoided care because of provider treatment (overlaps the healthcare-discrimination group)",
               "counts": "written as they are; pii_check"},
              open(os.path.join(OUT, f"conditions_assoc_adjusted_jno_{ENC}_meta.json"), "w"), indent=2)
    files = [f"conditions_assoc_adjusted_jno_{ENC}.csv", f"conditions_assoc_adjusted_or_jno_{ENC}.csv",
             f"conditions_assoc_adjusted_jno_{ENC}_meta.json"]
    zpath = zip_out(OUT, files, f"conditions_assoc_adjusted_jno_{ENC}")
    print(f"\n[conditions_assoc_adjusted_jno] {len(D)} conditions\n  download -> {zpath}")


if __name__ == "__main__":
    main()
