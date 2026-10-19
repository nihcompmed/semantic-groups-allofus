#!/usr/bin/env python3
"""step10_conditions.py -- the semantic groups and the 8 recorded conditions (Figure 7, its supplements and source
data; Appendix 5).

THE GROUP SCORES. Each encoder's scores X P, residualized on age and sex and standardized over the cohort, exactly as the
distance uses them (the distance recomputed from them must equal step 2's). The item on a diagnosed life-threatening
illness (mhqukb_41) gets weight 0 in every group ("without"): with it, the groups' association with the 2 cancers came
from that one item. The models with the item ("with") are written beside.

ONE MODEL PER CONDITION, in its population (outcomes.load_conditions), unpenalized logistic, no folds:
  condition ~ age_z + age_z^2 (+ sex where the population holds both sexes) + the k group scores
  joint     the Wald test of the k groups together (k df), P < .05
  together  each group's odds ratio per SD, net of age, sex and the other groups, P < .05/k
  alone     each group alone: condition ~ age_z + age_z^2 (+ sex) + that group, P < .05/k
DOWNLOADS
  conditions            condition_models.csv (per encoder and condition), condition_groups.csv (per group), and
                        condition_items.csv: the borderline items alone against each condition (odds ratio per SD)
  conditions_adjusted   the same models with income, education, insurance and the access and cost barriers added
                        (categories, non-answers kept as levels, reference = the most common level), per encoder
  factor_conditions     the same models on the answer-based factors at k = 25 and k = 3, under oblimin and varimax
                        (the rotation changes each factor's association, not the joint test, which is checked)

Run:  python3 step10_conditions.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2

import comparison as cp
import outcomes as oc
import scoring as sc
from models import logistic as fit
from release import versions, write_download
from settings import BASIS, CDR, ENCODERS, ID, Z, download_dir, work

ITEM = "mhqukb_41"                                       # "diagnosed with a life-threatening illness"
ITEMS_ALONE = [ITEM, "cidi5_5", "sdoh_mos_ss_1", "sdoh_mos_ss_2", "sdoh_mos_ss_3", "sdoh_mos_ss_4", "phq9_3", "phq9_4",
               "phq9_5"]
P_JOINT = 0.05
COVS = ["income", "education", "insurance"]
BARRIERS = {"access": "any_access_barrier", "cost": "any_cost_barrier"}


def scores(X, cov, w, P, resp_index, idx):
    """The group scores of a given P: X P residualized on age and sex, standardized, in idx order."""
    Yr = sc.wls_residualize(X @ P, cov, w)
    Yz = (Yr - Yr.mean(axis=0)) / Yr.std(axis=0, ddof=1)
    return pd.DataFrame(Yz, index=resp_index).reindex(idx).values


def group_names(enc, k):
    comp = pd.read_csv(os.path.join(BASIS, "components.csv"))
    comp = comp[comp.encoder == enc].sort_values("component")
    assert list(comp["component"]) == list(range(k)), f"{enc}: components.csv does not list 0..k-1"
    return comp["construct_profile"].values


def check_distance(enc, X, cov, w, P, resp_index):
    """The distance recomputed from the scores equals step 2's."""
    Yr = sc.wls_residualize(X @ P, cov, w)
    T2, _ = sc.mahalanobis2(Yr, w)
    S = pd.read_csv(work(f"scores_{enc}.csv"))
    S[ID] = S[ID].astype(str)
    worst = float(np.max(np.abs(np.sqrt(np.maximum(T2, 0)) - S.set_index(ID).reindex(resp_index)["distance"].values)))
    assert worst < 1e-6, f"{enc}: the recomputed distance differs from scores_{enc}.csv by {worst}"


def base_matrix(A, d, key):
    """age_z, age_z^2 (+ sex) over the population d, age standardized on d."""
    age = A["age"][d]
    az = (age - age.mean()) / age.std()
    return np.column_stack([az, az ** 2] + ([A["sex"][d]] if oc.with_sex(key) else []))


def models(A, Yz, key, k):
    """The joint Wald test of the k groups, and each group's odds ratio with all groups in the model and alone."""
    d = A[f"den_{key}"]
    y = A[f"y_{key}"][d].astype(int)
    base = base_matrix(A, d, key)
    f = fit(y, np.column_stack([base, Yz[d]]))
    nb = 1 + base.shape[1]
    g = slice(nb, nb + k)
    b, V = np.asarray(f.params), np.asarray(f.cov_params())
    wald = float(b[g] @ np.linalg.solve(V[g, g], b[g]))
    se, pv = np.asarray(f.bse)[g], np.asarray(f.pvalues)[g]
    rows = []
    for j in range(k):
        fa = fit(y, np.column_stack([base, Yz[d][:, j]]))
        ba, sa = float(fa.params[-1]), float(fa.bse[-1])
        rows.append({"group": j, "or": float(np.exp(b[g][j])), "or_lo": float(np.exp(b[g][j] - Z * se[j])),
                     "or_hi": float(np.exp(b[g][j] + Z * se[j])), "p": float(pv[j]),
                     "or_alone": float(np.exp(ba)), "or_alone_lo": float(np.exp(ba - Z * sa)),
                     "or_alone_hi": float(np.exp(ba + Z * sa)), "p_alone": float(fa.pvalues[-1])})
    return {"n_cases": int(y.sum()), "n_noncases": int((y == 0).sum()), "wald": wald, "p_joint": float(chi2.sf(wald, k)),
            "converged": bool(f.mle_retvals.get("converged", False))}, rows


def joint_row(name, key, label, k, jw, gw, jo, go, p_group):
    return {"encoder": name, "condition": key, "label": label, "k": k, "n_cases": jw["n_cases"],
            "n_noncases": jw["n_noncases"], "wald_with": jw["wald"], "p_joint_with": jw["p_joint"],
            "wald_without": jo["wald"], "p_joint_without": jo["p_joint"],
            "n_together_with": sum(r["p"] < p_group for r in gw), "n_together_without": sum(r["p"] < p_group for r in go),
            "n_alone_with": sum(r["p_alone"] < p_group for r in gw), "n_alone_without": sum(r["p_alone"] < p_group for r in go)}


def controls(idx):
    """The 5 controls per person, each a category (non-answers kept as levels)."""
    Cv = pd.read_csv(work("covariates.csv"), dtype={ID: str}).set_index(ID).reindex(idx)
    O = pd.read_csv(work("care_outcomes.csv"), dtype={ID: str}).set_index(ID).reindex(idx)
    out = pd.DataFrame(index=idx)
    for c in COVS:
        out[c] = Cv[c].fillna("Prefer not to answer").astype(str)
    surv = O["has_survey"].fillna(False).astype(bool)
    for name, col in BARRIERS.items():
        v = O[col]
        out[name] = np.where(~surv | v.isna(), "not answered", np.where(v.astype(float) > 0, "yes", "no"))
    return out


def dummies(C, mask):
    """Indicators for the controls within a population: reference = the most common level; a column identical to an
    earlier one, or determined by the others, is left out."""
    cols, names = [], []
    for c in C.columns:
        v = C[c].values[mask]
        levels, counts = np.unique(v, return_counts=True)
        ref = levels[np.argmax(counts)]
        for lv in levels:
            if lv == ref:
                continue
            x = (v == lv).astype(float)
            if 0 < x.sum() < len(x) and not any(np.array_equal(x, y_) for y_ in cols):
                cols.append(x)
                names.append(f"{c}={lv}")
    if not cols:
        return np.zeros((int(mask.sum()), 0)), names
    X = np.column_stack(cols)
    keep = []
    for j in range(X.shape[1]):
        if np.linalg.matrix_rank(np.column_stack([np.ones(len(X))] + [X[:, i] for i in keep + [j]])) == len(keep) + 2:
            keep.append(j)
    return X[:, keep], [names[j] for j in keep]


def main():
    idx = oc.cohort_order()
    A = oc.load_conditions(idx)
    T = pd.read_csv(sc.ITEMS_CSV)
    items = T["item"].astype(str).tolist()
    con = T["construct"].astype(str).tolist()
    j_item = items.index(ITEM)
    resp, X, cov, w = sc.load_battery(items)
    keys = [c[0] for c in oc.CONDITIONS]
    labels = {c[0]: c[2] for c in oc.CONDITIONS}

    # the borderline items alone
    irows = []
    for it in ITEMS_ALONE:
        x_all = pd.Series(X[:, items.index(it)], index=resp.index).reindex(idx).values
        for key in keys:
            d = A[f"den_{key}"]
            y = A[f"y_{key}"][d].astype(int)
            x = x_all[d]
            sd = float(np.std(x, ddof=1))
            f = fit(y, np.column_stack([base_matrix(A, d, key), x / sd]))
            b, se = float(f.params[-1]), float(f.bse[-1])
            irows.append({"item": it, "condition": key, "label": labels[key], "sd_item": sd,
                          "or_per_sd": float(np.exp(b)), "or_per_sd_lo": float(np.exp(b - Z * se)),
                          "or_per_sd_hi": float(np.exp(b + Z * se)), "p": float(f.pvalues[-1])})

    # the semantic groups, with and without the item; and with the controls (without the item)
    jrows, grows, without = [], [], {}
    Ctl = controls(idx)
    da = download_dir("conditions_adjusted")
    adjusted_files = []
    for enc in ENCODERS:
        P, _ = sc.place(enc, items)
        k = P.shape[1]
        check_distance(enc, X, cov, w, P, resp.index)
        cpf = group_names(enc, k)
        Yz_with = scores(X, cov, w, P, resp.index, idx)
        P0 = P.copy()
        P0[j_item, :] = 0.0
        Yz_without = scores(X, cov, w, P0, resp.index, idx)
        p_group = 0.05 / k
        for key in keys:
            jw, gw = models(A, Yz_with, key, k)
            jo, go = models(A, Yz_without, key, k)
            without[(enc, key)] = jo["wald"]
            jrows.append({**joint_row(enc, key, labels[key], k, jw, gw, jo, go, p_group),
                          "converged": jw["converged"] and jo["converged"]})
            for a, b in zip(gw, go):
                grows.append({"encoder": enc, "condition": key, "group": a["group"], "construct_profile": cpf[a["group"]],
                              **{f"{x}_with": v for x, v in a.items() if x != "group"},
                              **{f"{x}_without": v for x, v in b.items() if x != "group"},
                              "passes_with": a["p"] < p_group, "passes_without": b["p"] < p_group})
        rows, arows = [], []
        for key, code, label, lst, _ in oc.CONDITIONS:
            d = A[f"den_{key}"]
            y = A[f"y_{key}"][d].astype(int)
            base = base_matrix(A, d, key)
            Xc, cnames = dummies(Ctl, d)
            res = {}
            for when, B_ in (("before", base), ("after", np.column_stack([base, Xc]))):
                f = fit(y, np.column_stack([B_, Yz_without[d]]))
                b, V = np.asarray(f.params), np.asarray(f.cov_params())
                g = slice(1 + B_.shape[1], 1 + B_.shape[1] + k)
                wv = float(b[g] @ np.linalg.solve(V[g, g], b[g]))
                res[when] = {"b": b[g], "se": np.asarray(f.bse)[g], "p": np.asarray(f.pvalues)[g], "w": wv,
                             "pj": float(chi2.sf(wv, k)), "conv": bool(f.mle_retvals.get("converged", False)), "B": B_}
            wr = without[(enc, key)]
            assert abs(res["before"]["w"] - wr) <= 1e-6 * max(1.0, wr), f"{enc} {key}: 'before' is not the model without the item"
            n_aa = 0
            for j in range(k):
                fa = fit(y, np.column_stack([res["after"]["B"], Yz_without[d][:, j]]))
                ba, sa, pa = float(fa.params[-1]), float(fa.bse[-1]), float(fa.pvalues[-1])
                n_aa += pa < p_group
                b0, b1 = res["before"]["b"][j], res["after"]["b"][j]
                arows.append({"encoder": enc, "condition": key, "phecode": code, "label": label, "list": lst, "group": j,
                              "construct_profile": cpf[j],
                              "or_before": float(np.exp(b0)), "p_before": float(res["before"]["p"][j]),
                              "passes_before": bool(res["before"]["p"][j] < p_group),
                              "or_after": float(np.exp(b1)), "or_after_lo": float(np.exp(b1 - Z * res["after"]["se"][j])),
                              "or_after_hi": float(np.exp(b1 + Z * res["after"]["se"][j])),
                              "p_after": float(res["after"]["p"][j]), "passes_after": bool(res["after"]["p"][j] < p_group),
                              "share_kept": float(b1 / b0) if b0 != 0 else np.nan,
                              "or_alone_after": float(np.exp(ba)), "or_alone_after_lo": float(np.exp(ba - Z * sa)),
                              "or_alone_after_hi": float(np.exp(ba + Z * sa)), "p_alone_after": pa,
                              "passes_alone_after": bool(pa < p_group)})
            rows.append({"encoder": enc, "condition": key, "phecode": code, "label": label, "list": lst,
                         "zero_means": "tested, no diagnosis recorded" if key in oc.TESTED_BY else "no diagnosis recorded",
                         "n_cases": int(y.sum()), "n_noncases": int((y == 0).sum()), "n_control_columns": len(cnames),
                         "groups_wald_before": res["before"]["w"], "groups_p_before": res["before"]["pj"],
                         "groups_wald_after": res["after"]["w"], "groups_p_after": res["after"]["pj"],
                         "passes_joint_after": bool(res["after"]["pj"] < P_JOINT),
                         "n_groups_together_before": int((res["before"]["p"] < p_group).sum()),
                         "n_groups_together_after": int((res["after"]["p"] < p_group).sum()),
                         "n_groups_alone_after": int(n_aa),
                         "converged": bool(res["before"]["conv"] and res["after"]["conv"])})
        pd.DataFrame(rows).to_csv(os.path.join(da, f"adjusted_{enc}.csv"), index=False)
        pd.DataFrame(arows).to_csv(os.path.join(da, f"adjusted_groups_{enc}.csv"), index=False)
        adjusted_files += [f"adjusted_{enc}.csv", f"adjusted_groups_{enc}.csv"]
        print(f"  {enc}: k = {k}, 8 conditions", flush=True)
    d = download_dir("conditions")
    pd.DataFrame(jrows).to_csv(os.path.join(d, "condition_models.csv"), index=False)
    pd.DataFrame(grows).to_csv(os.path.join(d, "condition_groups.csv"), index=False)
    pd.DataFrame(irows).to_csv(os.path.join(d, "condition_items.csv"), index=False)
    json.dump({"cdr": CDR, "item_removed": ITEM, "items_alone": ITEMS_ALONE, "conditions": keys,
               "without": "the item's weight set to 0 in every group, then residualized and standardized",
               "covariates": "age_z, age_z^2, sex where the population holds both sexes",
               "threshold": "joint P < .05; groups P < .05/k", "versions": versions()},
              open(os.path.join(d, "conditions_meta.json"), "w"), indent=2)
    write_download("conditions", ["condition_models.csv", "condition_groups.csv", "condition_items.csv",
                                  "conditions_meta.json"])
    json.dump({"cdr": CDR, "scores": "illness item mhqukb_41 at weight 0 in every group", "p_joint": P_JOINT,
               "controls": {"income": "The Basics", "education": "The Basics", "insurance": "The Basics",
                            "access": "any access barrier (Health Care Access survey)",
                            "cost": "any cost barrier (Health Care Access survey)"},
               "coding": "categories, non-answers as levels, reference = most common level in the population",
               "not_used": "avoided care because of provider treatment (overlaps the healthcare-discrimination group)",
               "versions": versions()},
              open(os.path.join(da, "conditions_adjusted_meta.json"), "w"), indent=2)
    write_download("conditions_adjusted", adjusted_files + ["conditions_adjusted_meta.json"])

    # the answer-based factors
    Xr = sc.wls_residualize(X, cov, w)
    Xc = Xr - np.average(Xr, axis=0, weights=w)
    sd = X.std(0)
    Xs = (X - X.mean(0)) / sd
    src = cp.per_question_labels(T)
    jrows, grows = [], []
    for k, sfile in ((25, "scores_factor.csv"), (3, "scores_factor3.csv")):
        F = cp.factor_model(f"factor k{k}", k, sfile, Xs, sd, Xc, w, idx, items, con, src)
        W = {"oblimin": (F["S0"] @ F["Tm"], F["Lo"]), "varimax": (F["S0"] @ F["Rm"], F["Wv"])}
        p_group = 0.05 / k
        wald = {}
        for rot in cp.ROTATIONS:
            S, L = W[rot]
            name = f"factor k{k} {rot}"
            cores = [sc.fixed_point_core(L[:, j]) for j in range(k)]
            names = [cp.profile(cc, con) for cc in cores]
            srcs = ["; ".join(dict.fromkeys(src[i] for i in cc)) for cc in cores]
            S0 = S.copy()
            S0[j_item, :] = 0.0
            Yw = scores(X, cov, w, S, resp.index, idx)
            Yo = scores(X, cov, w, S0, resp.index, idx)
            for key in keys:
                jw, gw = models(A, Yw, key, k)
                jo, go = models(A, Yo, key, k)
                assert jw["converged"] and jo["converged"], f"{name} {key}: a fit did not converge"
                wald[(rot, key)] = (jw["wald"], jo["wald"])
                jrows.append({**joint_row(name, key, labels[key], k, jw, gw, jo, go, p_group), "converged": True})
                for a, b in zip(gw, go):
                    grows.append({"encoder": name, "condition": key, "group": a["group"],
                                  "construct_profile": names[a["group"]], "core_sources": srcs[a["group"]],
                                  **{f"{x}_with": v for x, v in a.items() if x != "group"},
                                  **{f"{x}_without": v for x, v in b.items() if x != "group"},
                                  "passes_with": a["p"] < p_group, "passes_without": b["p"] < p_group})
        for key in keys:                              # the joint test does not depend on the rotation
            for i in (0, 1):
                a_, b_ = wald[("oblimin", key)][i], wald[("varimax", key)][i]
                assert abs(a_ - b_) <= 1e-4 * max(1.0, abs(a_)), f"k={k} {key}: the joint Wald differs between rotations"
    d = download_dir("factor_conditions")
    pd.DataFrame(jrows).to_csv(os.path.join(d, "factor_condition_models.csv"), index=False)
    pd.DataFrame(grows).to_csv(os.path.join(d, "factor_condition_groups.csv"), index=False)
    json.dump({"cdr": CDR, "models": [f"factor k{k} {r}" for k in (25, 3) for r in cp.ROTATIONS], "item_removed": ITEM,
               "threshold": "joint P < .05; factors P < .05/k", "versions": versions()},
              open(os.path.join(d, "factor_conditions_meta.json"), "w"), indent=2, default=str)
    write_download("factor_conditions", ["factor_condition_models.csv", "factor_condition_groups.csv",
                                         "factor_conditions_meta.json"])


if __name__ == "__main__":
    main()
