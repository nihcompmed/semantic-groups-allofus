#!/usr/bin/env python3
"""step7_deciles.py -- the outcomes by decile of the distance (Figures 2 and 5 and their supplements; Appendix 3).

DOWNLOAD deciles. One model per grouping and outcome: outcome ~ decile + age + age^2 + sex (models.joint), a decile's
adjusted value standardized over the outcome's population; the trend = the decile number as one slope (models.trend_fit).
Four scopes, each with its own people:
  outcomes.csv                    the 12 groupings x thresholds met, the 3 barriers to care, the emergency department
                                  share of visits, minutes of sleep, the night-to-night SD of sleep (the cohort row,
                                  decile 0, carries the cohort's values)
  conditions.csv                  the 6 encoders x the 8 recorded conditions, each in its population, deciles pooled
                                  where a bin holds 1 to 20 cases or non-cases
  sleep_device.csv                sleep by decile separately for program-provided and own devices (the 6 encoders)
  below_threshold_outcomes.csv, below_threshold_conditions.csv
                                  the participants who met none of the six thresholds, deciles within them (6 encoders)
  trends.csv                      the trend of every scope, grouping and outcome
DOWNLOAD deciles_ed. The emergency department share on coded care (at least 1 visit and at least 1 ICD event) by decile,
  for the 12 groupings and below every threshold (6 encoders), with the joint model and the trend.
DOWNLOAD ed_visits. Who has a visit but no coded diagnosis, and what their visits are (by group, source and type); and the
  emergency department share by decile on coded care adjusted by a reference curve (the 6 encoders).

Run:  python3 step7_deciles.py
"""
import json
import os

import numpy as np
import pandas as pd

import deciles as dc
import models
import outcomes as oc
from release import (MARK, SUPP, aggregate_only, coarsen, coarsen_sig, hide_cells, pair_hidden, small, versions,
                     write_download)
from settings import CDR, ENCODERS, MIN_NIGHTS, download_dir, work

ED_CONCEPTS = (9203, 262)
TOP_TYPES = 12
GROUPS = ("icd", "visit_only")
ED_COLS = ["ed_mean", "ed_lo", "ed_hi", "ed_adj", "ed_adj_lo", "ed_adj_hi"]


# ------------------------------------------------------------------------------------------------- deciles
def sleep_device(idx, N, age, sex, nights, order, trends):
    P = oc.read_sleep(idx)
    has = (P["nights"].fillna(0).values >= MIN_NIGHTS) & P["sd_min"].notna().values
    mean_min, sd_min = P["mean_min"].values.astype(float), P["sd_min"].values.astype(float)
    wear = oc.wear_mask(idx)
    rows = []
    for src, h in {"program": has & wear, "own": has & ~wear}.items():
        S = {"age": age, "has_sleep": h, "mean_min": mean_min, "sd_min": sd_min, "wear": None}
        c = oc.describe_sleep(np.ones(N, bool), S)
        coh = {k: c.get(k, np.nan) for k in dc.SLEEP_UNADJ}
        coh.update({v: coh.get(k, np.nan) for k, v in dc.SLEEP_ADJ.items()})
        rows.append({"source": src, "grouping": "cohort", "decile": 0, **coh})
        for name in ENCODERS:
            M = dc.decile_masks(order[name], N)
            dec = np.zeros(N, int)
            for j in dc.DECILES:
                dec[M[j]] = j
            Cm = models.covs(age[h], sex[h], True)
            Cv = models.covs(age[h], sex[h], True, np.log(nights[h]))
            Jm = models.joint("ols", mean_min[h], dec[h], dc.DECILES, Cm, f"sleep_device {src} {name} mean")
            Jv = models.joint("median", sd_min[h], dec[h], dc.DECILES, Cv, f"sleep_device {src} {name} variation")
            trends.append(dc.trend_row("sleep_device", name, "sleep_mean_min", "ols", mean_min[h], dec[h], Cm, src))
            trends.append(dc.trend_row("sleep_device", name, "sleep_variation_min", "median", sd_min[h], dec[h], Cv, src))
            for j in dc.DECILES:
                d = oc.describe_sleep(M[j], S)
                r = {k: d.get(k, np.nan) for k in dc.SLEEP_UNADJ}
                for base, est in (("sleep_mean", Jm[j]), ("sleep_var_median", Jv[j])):
                    cols = (dc.SLEEP_ADJ[base], dc.SLEEP_ADJ["sleep_lo" if base == "sleep_mean" else "sleep_var_lo"],
                            dc.SLEEP_ADJ["sleep_hi" if base == "sleep_mean" else "sleep_var_hi"])
                    if dc.is_num(r.get(base, np.nan)):
                        r[cols[0]], r[cols[1]], r[cols[2]] = est
                    else:
                        r[cols[0]] = r[cols[1]] = r[cols[2]] = np.nan
                rows.append({"source": src, "grouping": name, "decile": j, **r})
    return pd.DataFrame(rows)


def below_threshold(N, grp, B, C, S, SA, A, age, sex, nights, order, trends):
    rows, masks = [], {}
    g_all = dc.row(grp, B, C, S, SA)
    for adj, raw in ((("access_adj", "access_adj_lo", "access_adj_hi"), ("access_rate", "access_lo", "access_hi")),
                     (("cost_adj", "cost_adj_lo", "cost_adj_hi"), ("cost_rate", "cost_lo", "cost_hi")),
                     (("avoid_adj", "avoid_adj_lo", "avoid_adj_hi"), ("avoid_rate", "avoid_lo", "avoid_hi")),
                     (("ed_adj", "ed_adj_lo", "ed_adj_hi"), ("ed_mean", "ed_lo", "ed_hi")),
                     (("sleep_mean_adj", "sleep_mean_adj_lo", "sleep_mean_adj_hi"), ("sleep_mean", "sleep_lo", "sleep_hi")),
                     (("sleep_var_median_adj", "sleep_var_median_adj_lo", "sleep_var_median_adj_hi"),
                      ("sleep_var_median", "sleep_var_lo", "sleep_var_hi"))):
        for a, r_ in zip(adj, raw):                                         # the group row is unadjusted, as the cohort's
            g_all[a] = g_all.get(r_, np.nan)
    rows.append({"grouping": "group", "decile": 0, **{k: v for k, v in g_all.items() if k not in dc.DROP_BELOW}})
    specs = dc.outcome_specs(B, C, S, nights, grp, False)
    for name in ENCODERS:
        M = dc.group_deciles(order[name], grp, N)
        masks[name] = M
        dec = np.zeros(N, int)
        for j in dc.DECILES:
            dec[M[j]] = j
        J = dc.fit_specs(specs, dec, age, sex, nights, f"below_threshold {name}", "below_threshold", name, trends,
                         trend_scale=True)
        for j in dc.DECILES:
            r = dc.row(M[j], B, C, S, SA)
            for key, _, _, _, _, _, cols, _ in specs:
                dc.put(r, cols, J[key][j])
            rows.append({"grouping": name, "decile": j, **{k: v for k, v in r.items() if k not in dc.DROP_BELOW}})
    O = pd.DataFrame(rows)
    O = coarsen_sig(coarsen(O, [c for c in dc.RATE_COLS if c in O.columns]), [c for c in dc.SIG_COLS if c in O.columns])
    Cd, pooled = dc.condition_rows("below_threshold", A, masks, grp, age, sex, N, trends, "group", "below_threshold")
    return O, Cd, pooled


def deciles(cf, idx, B, C, S, SA, A, nights):
    N = len(idx)
    grp = cf["none_met"].astype(bool).values
    age, sex = A["age"], A["sex"]
    assert np.allclose(age, C["age"]) and np.allclose(sex, C["sex"]), "age or sex differ between the 2 loaders"
    trends = []
    G = dc.groupings()
    order12 = dc.rank_orders([g[0] for g in G], idx)
    rows = [{"grouping": "cohort", "family": "reference", "k": np.nan, "decile": 0,
             **dc.row(np.ones(N, bool), B, C, S, SA)}]
    specs = dc.outcome_specs(B, C, S, nights, np.ones(N, bool), True)
    for i, (name, fam, kk, _) in enumerate(G):
        r_, _ = dc.grouping_rows(name, fam, kk, order12[name], N, B, C, S, SA, age, sex, nights, specs, trends,
                                 check_first=(i == 0))
        rows += r_
    D = dc.finish(rows)
    order = dc.rank_orders(ENCODERS, idx)
    masks = {name: dc.decile_masks(order[name], N) for name in ENCODERS}
    Cc, pooled_c = dc.condition_rows("cohort", A, masks, np.ones(N, bool), age, sex, N, trends, "population",
                                     "cohort_conditions")
    Sd = sleep_device(idx, N, age, sex, nights, order, trends)
    Ob, Cb, pooled_b = below_threshold(N, grp, B, C, S, SA, A, age, sex, nights, order, trends)
    T = pd.DataFrame(trends)
    bad = [k for k, v in models.CONVERGED.items() if not v and "median" not in k and "variation" not in k
           and "sleep_var" not in k]
    assert not bad, f"logistic fits that did not converge: {bad[:5]}"
    d = download_dir("deciles")
    files = {"outcomes.csv": D, "conditions.csv": Cc, "sleep_device.csv": Sd, "below_threshold_outcomes.csv": Ob,
             "below_threshold_conditions.csv": Cb, "trends.csv": T}
    for fn, X in files.items():
        X.to_csv(os.path.join(d, fn), index=False)
    ok, why = aggregate_only(os.path.join(d, "outcomes.csv"))
    assert ok, f"outcomes.csv refused: {why}"
    json.dump({"cdr": CDR,
               "model": "outcome ~ decile (categories; pooled bins for conditions) + age_z + age_z^2 (+ sex where both "
                        "sexes) (+ log(nights) for sleep variation); age_z within the population",
               "families": {"thresholds met, ED share, sleep mean": "least squares, HC1", "barriers, conditions":
                            "logistic", "sleep variation": "median regression"},
               "adjusted value": "standardized prediction over the outcome's population with the decile set; delta-method 95% CI",
               "test": "trend: decile number 1-10 as one slope, same covariates; P < .05",
               "pooled_condition_bins": {"cohort": pooled_c, "below_threshold": pooled_b},
               "median_regression_not_converged": [k for k, v in models.CONVERGED.items() if not v],
               "min_nights": MIN_NIGHTS, "versions": versions()},
              open(os.path.join(d, "deciles_meta.json"), "w"), indent=2, default=str)
    write_download("deciles", list(files) + ["deciles_meta.json"])


# ------------------------------------------------------------------------------------------------- the ED share on coded care
def ed_scope_rows(scope, masks, pop, visit, C2, age, sex, trends, check_first=False):
    rows = []
    for name, M in masks.items():
        N = len(pop)
        dec = np.zeros(N, int)
        for j in dc.DECILES:
            dec[M[j]] = j
        y = C2["ed_share"][pop].astype(float)
        Cv = models.covs(age[pop], sex[pop], True)
        est = models.joint("ols", y, dec[pop], dc.DECILES, Cv, f"{scope} {name} ed coded")
        if check_first:
            models.check_intercept_only("ols", y, dec[pop], dc.DECILES, f"{scope} {name} ed coded")
            check_first = False
        trends.append(dc.trend_row(scope, name, "ed_share_pct_coded", "ols", 100 * y, dec[pop], Cv))
        R, n_vis = {}, {}
        for j in dc.DECILES:
            r = oc.describe_care(M[j], C2)
            R[j] = {"n_set": int(M[j].sum()), "n": int((M[j] & pop).sum()),
                    **{c: r.get(c, np.nan) for c in ("ed_mean", "ed_lo", "ed_hi")}}
            R[j]["ed_adj"], R[j]["ed_adj_lo"], R[j]["ed_adj_hi"] = est[j]
            n_vis[j] = int((M[j] & visit).sum())
        cnt = [R[j]["n"] for j in dc.DECILES]
        forced = [pair_hidden(R[j]["n"], n_vis[j]) or pair_hidden(R[j]["n"], R[j]["n_set"]) for j in dc.DECILES]
        hid = hide_cells([cnt, [R[j]["n_set"] - R[j]["n"] for j in dc.DECILES],
                          [n_vis[j] - R[j]["n"] for j in dc.DECILES]], forced)
        for j, h in zip(dc.DECILES, hid):
            r = {"scope": scope, "grouping": name, "decile": j, "n_set": R[j]["n_set"], "n_visit": n_vis[j],
                 "n_icd_visit": (MARK if small(R[j]["n"]) else SUPP) if h else R[j]["n"]}
            r.update({c: (np.nan if h else R[j][c]) for c in ED_COLS})
            rows.append(r)
    return rows


def deciles_ed(cf, idx, C, A):
    N = len(idx)
    grp = cf["none_met"].astype(bool).values
    age, sex = A["age"], A["sex"]
    visit = C["has_ehr"].astype(bool)
    coded = visit & A["has_rec"].astype(bool) & np.isfinite(C["ed_share"])
    C2 = dict(C)
    C2["has_ehr"] = coded
    trends = []
    coh = oc.describe_care(np.ones(N, bool), C2)
    rows = [{"scope": "cohort", "grouping": "cohort", "decile": 0, "n_set": N, "n_visit": int(visit.sum()),
             "n_icd_visit": int(coded.sum()), **{c: coh.get(c, np.nan) for c in ("ed_mean", "ed_lo", "ed_hi")},
             "ed_adj": coh.get("ed_mean", np.nan), "ed_adj_lo": coh.get("ed_lo", np.nan),
             "ed_adj_hi": coh.get("ed_hi", np.nan)}]
    G = dc.groupings()
    order = dc.rank_orders([g[0] for g in G], idx)
    rows += ed_scope_rows("cohort", {g[0]: dc.decile_masks(order[g[0]], N) for g in G}, coded, visit, C2, age, sex,
                          trends, check_first=True)
    g_all = oc.describe_care(grp, C2)
    rows.append({"scope": "below_threshold", "grouping": "group", "decile": 0, "n_set": int(grp.sum()),
                 "n_visit": int((grp & visit).sum()), "n_icd_visit": int((grp & coded).sum()),
                 **{c: g_all.get(c, np.nan) for c in ("ed_mean", "ed_lo", "ed_hi")}, "ed_adj": g_all.get("ed_mean", np.nan),
                 "ed_adj_lo": g_all.get("ed_lo", np.nan), "ed_adj_hi": g_all.get("ed_hi", np.nan)})
    rows += ed_scope_rows("below_threshold", {e: dc.group_deciles(order[e], grp, N) for e in ENCODERS}, grp & coded,
                          grp & visit, C2, age, sex, trends)
    d = download_dir("deciles_ed")
    coarsen_sig(pd.DataFrame(rows), ED_COLS).to_csv(os.path.join(d, "ed_deciles.csv"), index=False)
    pd.DataFrame(trends).to_csv(os.path.join(d, "ed_trend.csv"), index=False)
    json.dump({"cdr": CDR, "people": "coded care: at least 1 visit and at least 1 ICD event",
               "model": "ED share ~ decile (categories) + age_z + age_z^2 + sex, least squares, HC1; standardized per "
                        "decile, delta-method 95% CI",
               "test": "trend: decile number as one slope, same covariates, ED share in points",
               "versions": versions()},
              open(os.path.join(d, "ed_deciles_meta.json"), "w"), indent=2, default=str)
    write_download("deciles_ed", ["ed_deciles.csv", "ed_trend.csv", "ed_deciles_meta.json"])


# ------------------------------------------------------------------------------------------------- the visits check
def cell(c, total):
    return MARK if small(c) else SUPP if pair_hidden(c, total) else int(c)


def visits_by_group(V, groups, n_cohort):
    """The visits of people with an ICD event and of people with visits only: per group, by source and by type."""
    rows = []
    tot = V.groupby("person_id")["n_visits"].sum()
    ed = V[V.visit_concept_id.isin(ED_CONCEPTS)].groupby("person_id")["n_visits"].sum()
    for g in GROUPS:
        p = sorted(groups[g])
        t = tot.reindex(p).fillna(0).values
        e = ed.reindex(p).fillna(0).values
        n_ed = int((e > 0).sum())
        q = np.percentile(t, [25, 50, 75])
        rows.append({"section": "group", "label": "all visits", "group": g, "people": cell(len(p), n_cohort),
                     "people_any_ed": cell(n_ed, len(p)), "visits_p25": q[0], "visits_median": q[1], "visits_p75": q[2],
                     "mean_ed_share": float(np.mean(e / t)) if not pair_hidden(n_ed, len(p)) else np.nan})
    reach = V.groupby("visit_type")["person_id"].nunique().sort_values(ascending=False)
    top = list(reach.index[:TOP_TYPES])
    V = V.assign(type_label=np.where(V.visit_type.isin(top), V.visit_type, "other types"))
    for section, col, labels, with_only in (("source", "source", sorted(V.source.unique()), True),
                                           ("type", "type_label", top + (["other types"] if len(reach) > TOP_TYPES else []),
                                            False)):
        for lab in labels:
            per = {}
            for g in GROUPS:
                W = V[V.person_id.isin(groups[g])]
                people = W[W[col] == lab].person_id.nunique()
                share = W.loc[W[col] == lab, "n_visits"].sum() / W["n_visits"].sum()
                r = {"people": people, "share_of_visits": share}
                if with_only:
                    s = W.groupby("person_id")[col].agg(lambda x: set(x) == {lab})
                    r["people_only"] = int(s.sum())
                per[g] = (r, len(groups[g]))
            hide = any(pair_hidden(r["people"], n) for r, n in per.values())
            hide_only = with_only and any(pair_hidden(r["people_only"], n) for r, n in per.values())
            for g, (r, n) in per.items():
                row = {"section": section, "label": lab, "group": g,
                       "people": (MARK if small(r["people"]) else SUPP) if hide else int(r["people"]),
                       "share_of_visits": np.nan if hide else float(r["share_of_visits"])}
                if with_only:
                    row["people_only"] = (MARK if small(r["people_only"]) else SUPP) if hide_only else int(r["people_only"])
                rows.append(row)
    return pd.DataFrame(rows)


def ed_deciles_curve(idx, C, icd):
    """The ED share by decile on people with a visit and an ICD event, adjusted by a reference curve refitted on them."""
    N = len(idx)
    visit = C["has_ehr"].copy()
    C2 = dict(C)
    C2["has_ehr"] = visit & icd
    m = C2["has_ehr"]
    y = C["ed_share"]
    fitted, _ = oc.fit_curve(C["age"][m], C["sex"][m], y[m])
    C2["gap_ed"] = np.full(N, np.nan)
    C2["gap_ed"][m] = y[m] - fitted
    C2["coh_ed"] = float(y[m].mean())

    def ed_row(mask, A_):
        r = oc.describe_care(mask, A_)
        a = oc.describe_care_adjusted(mask, A_)
        out = {"n_set": int(mask.sum()), "n": int((mask & A_["has_ehr"]).sum())}
        out.update({c: r.get(c, np.nan) for c in ("ed_mean", "ed_lo", "ed_hi")})
        out.update({"ed_adj": a.get("ed_mean", np.nan), "ed_adj_lo": a.get("ed_lo", np.nan),
                    "ed_adj_hi": a.get("ed_hi", np.nan)})
        return out

    coh = ed_row(np.ones(N, bool), C2)
    rows = [{"grouping": "cohort", "decile": 0, "n_set": N, "n_visit": int(visit.sum()), "n_icd_visit": coh["n"],
             **{c: coh[c] for c in ED_COLS}}]
    order = dc.rank_orders(ENCODERS, idx)
    for name in ENCODERS:
        M = dc.decile_masks(order[name], N)
        R = {j: ed_row(M[j], C2) for j in dc.DECILES}
        Rv = {j: ed_row(M[j], C) for j in dc.DECILES}
        cnt = [R[j]["n"] for j in dc.DECILES]
        forced = [pair_hidden(R[j]["n"], Rv[j]["n"]) or pair_hidden(R[j]["n"], R[j]["n_set"]) for j in dc.DECILES]
        hid = hide_cells([cnt, [R[j]["n_set"] - R[j]["n"] for j in dc.DECILES],
                          [Rv[j]["n"] - R[j]["n"] for j in dc.DECILES]], forced)
        for j, h in zip(dc.DECILES, hid):
            r = {"grouping": name, "decile": j, "n_set": R[j]["n_set"], "n_visit": Rv[j]["n"],
                 "n_icd_visit": (MARK if small(R[j]["n"]) else SUPP) if h else R[j]["n"]}
            r.update({c: (np.nan if h else R[j][c]) for c in ED_COLS})
            rows.append(r)
    return pd.DataFrame(rows)


def ed_visits(idx, C, A):
    N = len(idx)
    icd = A["has_rec"]
    visit = C["has_ehr"]
    V = pd.read_csv(work("visits_person.csv"), dtype={"person_id": str, "visit_type": str, "source": str})
    V = V[V.person_id.isin(set(idx))]
    tot = V.groupby("person_id")["n_visits"].sum().reindex(idx).fillna(0).values
    ed = V[V.visit_concept_id.isin(ED_CONCEPTS)].groupby("person_id")["n_visits"].sum().reindex(idx).fillna(0).values
    assert ((tot > 0) == visit).all(), "the people with a visit are not care_outcomes.csv's has_ehr"
    ok = np.isclose(np.where(visit, ed / np.where(tot > 0, tot, 1), np.nan)[visit], C["ed_share"][visit])
    assert ok.all(), "a person's ED share from the visits differs from care_outcomes.csv's"
    groups = {"icd": set(idx[icd & visit]), "visit_only": set(idx[~icd & visit])}
    Av = visits_by_group(V, groups, N)
    Av = coarsen_sig(coarsen(Av, ["share_of_visits"]), ["visits_p25", "visits_median", "visits_p75", "mean_ed_share"])
    for c in ("people", "people_any_ed", "people_only"):
        Av[c] = Av[c].map(lambda v: str(int(v)) if isinstance(v, (int, float, np.integer, np.floating))
                          and pd.notna(v) else v).astype(object)
    Bd = coarsen_sig(ed_deciles_curve(idx, C, icd), ED_COLS)
    d = download_dir("ed_visits")
    Av.to_csv(os.path.join(d, "ed_visits.csv"), index=False)
    Bd.to_csv(os.path.join(d, "ed_visits_deciles.csv"), index=False)
    json.dump({"cdr": CDR, "visit": "a distinct visit_occurrence_id", "ed_concepts": list(ED_CONCEPTS),
               "source": "visit_occurrence_ext.src_id with digits removed", "top_types": TOP_TYPES,
               "groups": {"icd": "at least 1 visit and at least 1 ICD event", "visit_only": "at least 1 visit and no ICD event"},
               "deciles": {"denominator": "at least 1 visit and at least 1 ICD event", "encoders": ENCODERS,
                           "adjustment": "OLS on age, age^2, sex, refitted once on that denominator"},
               "versions": versions()},
              open(os.path.join(d, "ed_visits_meta.json"), "w"), indent=2)
    write_download("ed_visits", ["ed_visits.csv", "ed_visits_deciles.csv", "ed_visits_meta.json"], aggregate_only)


def main():
    cf, idx = dc.cohort_frame()
    B, C, S, SA, _ = oc.load(idx)
    A = oc.load_conditions(idx)
    nights = dc.nights_of(idx)
    print(f"[step 7] cohort {len(idx):,} | met no threshold {int(cf['none_met'].astype(bool).sum()):,}", flush=True)
    ed_visits(idx, C, A)
    deciles(cf, idx, B, C, S, SA, A, nights)
    deciles_ed(cf, idx, C, A)


if __name__ == "__main__":
    main()
