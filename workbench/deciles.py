"""deciles.py -- participants ranked by distance, cut into tenths, and the outcomes of each tenth.

THE DECILES. Each grouping's own order of the cohort (most atypical first; ties broken by a stable sort on the cohort's
order, the same for every grouping) is cut at round(N j / 10), j = 0 to 10: decile 10 is the most atypical tenth, decile
1 the least. Below every screening threshold, the same order restricted to those participants is cut the same way
(group_deciles).

THE DISSEMINATION POLICY ACROSS DECILES. The deciles partition the cohort, whose totals are public, and deciles 9 and 10
are the top 20% and 10% of the cohort. For each count a statistic rests on (evaluable, women, answered the survey, each
barrier, at least 1 visit, at least 14 nights, wear study), with its complement: deciles 9 and 10 are hidden where the
statistic of the top 20% or the top 10% is withheld, or their own count is 1 to 20; deciles 1 to 8 partition a public
total and go through release.hide_cells, so a hidden cell is never recovered by subtraction. Hiding a count hides every
statistic resting on it. A recorded condition's deciles are pooled instead (pool): adjacent deciles are merged, toward
the nearer end, while a bin holds 1 to 20 cases or non-cases.
"""
import numpy as np
import pandas as pd

import models
import outcomes as oc
from release import SUPP, hide_cells, pair_hidden, small
from settings import ID, work

DECILES = list(range(1, 11))
BREADTH = ["mean_breadth", "sd_breadth", "se", "ci_lo", "ci_hi"]
BREADTH_ADJ = ["mean_breadth_adj", "mean_breadth_adj_lo", "mean_breadth_adj_hi"]
RATE_KEYS = list(oc.RATES)
CARE_ADJ = {"access": ("access_rate", "access_lo", "access_hi"), "cost": ("cost_rate", "cost_lo", "cost_hi"),
            "avoid": ("avoid_rate", "avoid_lo", "avoid_hi"), "ed": ("ed_mean", "ed_lo", "ed_hi")}
# each count a written statistic rests on: (family, the columns it hides, the family whose hiding forces it)
FAMILIES = [
    ("evaluable", ["n_all_evaluable"] + BREADTH + BREADTH_ADJ, None),
    ("female", ["share_female"], None),
    ("survey", ["n_survey"], None),
    ("ehr", ["n_ehr", "ed_mean", "ed_lo", "ed_hi", "ed_adj", "ed_adj_lo", "ed_adj_hi"], None),
    ("sleep", ["n_sleep", "coverage", "sleep_mean", "sleep_lo", "sleep_hi", "sleep_var_median", "sleep_var_lo",
               "sleep_var_hi", "sleep_age", "sleep_mean_adj", "sleep_mean_adj_lo", "sleep_mean_adj_hi",
               "sleep_var_median_adj", "sleep_var_median_adj_lo", "sleep_var_median_adj_hi"], None),
] + [(s, [f"{s}_rate", f"{s}_lo", f"{s}_hi", f"{s}_adj", f"{s}_adj_lo", f"{s}_adj_hi"], "survey")
     for s in RATE_KEYS] + [("wear", ["share_wear"], "sleep")]
RATE_COLS = ([f"{s}_{w}" for s in RATE_KEYS for w in ("rate", "lo", "hi", "adj", "adj_lo", "adj_hi")]
             + ["share_female", "coverage", "share_wear"])
SIG_COLS = (["mean_age", "sd_age"] + BREADTH + BREADTH_ADJ + ["ed_mean", "ed_lo", "ed_hi", "ed_adj", "ed_adj_lo",
            "ed_adj_hi", "sleep_mean", "sleep_lo", "sleep_hi", "sleep_var_median", "sleep_var_lo", "sleep_var_hi",
            "sleep_age", "sleep_mean_adj", "sleep_mean_adj_lo", "sleep_mean_adj_hi", "sleep_var_median_adj",
            "sleep_var_median_adj_lo", "sleep_var_median_adj_hi"])
DROP_BELOW = BREADTH + BREADTH_ADJ + ["n_all_evaluable"]      # thresholds met: 0 below every threshold
SLEEP_UNADJ = ["n_set", "n_sleep", "coverage", "sleep_age", "sleep_mean", "sleep_lo", "sleep_hi", "sleep_var_median",
               "sleep_var_lo", "sleep_var_hi"]
SLEEP_ADJ = {"sleep_mean": "sleep_mean_adj", "sleep_lo": "sleep_mean_adj_lo", "sleep_hi": "sleep_mean_adj_hi",
             "sleep_var_median": "sleep_var_median_adj", "sleep_var_lo": "sleep_var_median_adj_lo",
             "sleep_var_hi": "sleep_var_median_adj_hi"}


# ------------------------------------------------------------------------------------------------- orders and masks
def groupings():
    """(name, family, k, how k was chosen) of the 12 groupings, from step 2."""
    G = pd.read_csv(work("groupings.csv"))
    return [(r.grouping, r.family, int(r.k), r.k_rule) for r in G.itertuples()]


def rank_orders(names, idx):
    """Each grouping's order of the cohort, most atypical first, by a stable sort on the cohort's order."""
    order = {}
    for name in names:
        S = pd.read_csv(work(f"scores_{name}.csv"))
        S[ID] = S[ID].astype(str)
        assert len(S) == len(idx) and S[ID].isin(idx).all(), f"{name}: ids do not match the cohort"
        S = S.set_index(ID).reindex(idx)
        order[name] = np.argsort(-S["distance"].values, kind="mergesort")
    return order


def decile_masks(order_g, N):
    """Decile j (1 = least atypical, 10 = most) as a boolean mask, cut at round(N * i / 10)."""
    b = [int(round(N * i / 10.0)) for i in range(11)]
    out = {}
    for i in range(10):
        m = np.zeros(N, bool)
        m[order_g[b[i]:b[i + 1]]] = True
        out[10 - i] = m
    return out


def group_deciles(order_g, in_group, N):
    """Decile j of a group (below every threshold), in the grouping's order, cut at round(Ng * i / 10)."""
    pos = order_g[in_group[order_g]]
    ng = len(pos)
    b = [int(round(ng * i / 10.0)) for i in range(11)]
    out = {}
    for i in range(10):
        m = np.zeros(N, bool)
        m[pos[b[i]:b[i + 1]]] = True
        out[10 - i] = m
    return out


def pool(k, n):
    """Bins of deciles such that no bin has 1 to 20 cases or 1 to 20 non-cases: the lowest failing bin is merged with
    its neighbour toward the nearer end (k, n: dicts by decile)."""
    bins = [[j] for j in DECILES]
    fails = lambda b: small(sum(k[j] for j in b)) or small(sum(n[j] - k[j] for j in b))
    while len(bins) > 1:
        bad = [i for i, b in enumerate(bins) if fails(b)]
        if not bad:
            break
        i = bad[0]
        low_half = np.mean(bins[i]) <= 5.5
        other = (i - 1 if i > 0 else i + 1) if low_half else (i + 1 if i + 1 < len(bins) else i - 1)
        a, b = sorted((i, other))
        bins[a] = bins[a] + bins[b]
        del bins[b]
    return bins


# ------------------------------------------------------------------------------------------------- one decile
def row(mask, B, C, S, SA):
    """Every statistic of one set: care (unadjusted and adjusted by the curves), thresholds met, sleep."""
    r = oc.describe_care(mask, C)
    adj = oc.describe_care_adjusted(mask, C)
    for short, cols in CARE_ADJ.items():
        for c, a in zip(cols, (f"{short}_adj", f"{short}_adj_lo", f"{short}_adj_hi")):
            r[a] = adj.get(c, np.nan)
    keep = mask & B["ev"]
    n, ev = int(mask.sum()), int(keep.sum())
    st = oc.stats(B["br"][keep])
    r["n_all_evaluable"] = SUPP if pair_hidden(ev, n) else ev
    g = B["gap"][keep]
    if ev >= 2:
        m, se = float(g.mean()), float(np.std(g, ddof=1) / np.sqrt(ev))
        st.update({"mean_breadth_adj": B["coh"] + m, "mean_breadth_adj_lo": B["coh"] + m - oc.Z * se,
                   "mean_breadth_adj_hi": B["coh"] + m + oc.Z * se})
    if small(ev) or pair_hidden(ev, n):
        st = {k: np.nan for k in st}
    r.update(st)
    s = oc.describe_sleep(mask, S)
    r.update({k: v for k, v in s.items() if k != "n_set"})
    sa = oc.describe_sleep_adjusted(mask, SA)
    for c in ("sleep_mean", "sleep_var_median"):
        r[f"{c}_adj"] = sa.get(c, np.nan)
    for c, a in (("sleep_lo", "sleep_mean_adj_lo"), ("sleep_hi", "sleep_mean_adj_hi"),
                 ("sleep_var_lo", "sleep_var_median_adj_lo"), ("sleep_var_hi", "sleep_var_median_adj_hi")):
        r[a] = sa.get(c, np.nan)
    return r


def counts(mask, B, C, S):
    """The counts each family rests on, with the total its complement is taken from."""
    n = int(mask.sum())
    sv, e, sl = mask & C["has_survey"], mask & C["has_ehr"], mask & S["has_sleep"]
    out = {"evaluable": (int((mask & B["ev"]).sum()), n),
           "female": (int((C["sex"][mask] == oc.FEMALE).sum()), n),
           "survey": (int(sv.sum()), n), "ehr": (int(e.sum()), n), "sleep": (int(sl.sum()), n)}
    for s, col in oc.RATES.items():
        out[s] = (int(np.nansum(C[col][sv])), int(sv.sum()))
    out["wear"] = (int((sl & S["wear"]).sum()), int(sl.sum())) if S["wear"] is not None else (0, int(sl.sum()))
    return out


def written(r, fam):
    """Did the set's own statistics write this family?"""
    first = {"evaluable": "mean_breadth", "female": "share_female", "survey": "n_survey", "ehr": "n_ehr",
             "sleep": "n_sleep", "wear": "share_wear"}.get(fam, f"{fam}_rate")
    v = r.get(first, np.nan)
    if isinstance(v, str):
        return False
    return pd.notna(v)


def decile_hides(K, top80, top90):
    """The (decile, family) cells to hide across the partition. K[j][fam] = (count, total)."""
    hid, why = set(), {"sweep_withheld": 0, "own_count": 0, "partition": 0, "forced_by_parent": 0}
    for fam, _, parent in FAMILIES:
        w80, w90 = written(top80, fam), written(top90, fam)
        forced = set()
        if not (w80 and w90):
            forced |= {9, 10}
            why["sweep_withheld"] += 2
        for j in (9, 10):
            c, t = K[j][fam]
            if pair_hidden(c, t) and j not in forced:
                forced.add(j)
                why["own_count"] += 1
        if parent is not None:
            f = {j for j in DECILES if (j, parent) in hid}
            why["forced_by_parent"] += len(f - forced)
            forced |= f
        part = [1, 2, 3, 4, 5, 6, 7, 8] if w80 else DECILES
        cnt = [K[j][fam][0] for j in part]
        comp = [K[j][fam][1] - K[j][fam][0] for j in part]
        h = hide_cells([cnt, comp], [j in forced for j in part])
        new = {j for j, x in zip(part, h) if x} - forced
        why["partition"] += len(new)
        hid |= {(j, fam) for j in forced | new}
    return hid, why


# ------------------------------------------------------------------------------------------------- the joint model on deciles
def is_num(v):
    return not isinstance(v, str) and pd.notna(v)


def outcome_specs(B, C, S, nights, scope_mask, with_breadth):
    """(key, trend name, model, population, y, with_sex, (adj, lo, hi) columns, extra covariate) per outcome."""
    sp = []
    if with_breadth:
        sp.append(("breadth", "thresholds_met", "ols", scope_mask & B["ev"], B["br"], True, BREADTH_ADJ, None))
    for s, col in oc.RATES.items():
        m = scope_mask & C["has_survey"] & np.isfinite(C[col])
        sp.append((s, s, "logit", m, C[col], True, (f"{s}_adj", f"{s}_adj_lo", f"{s}_adj_hi"), None))
    m = scope_mask & C["has_ehr"] & np.isfinite(C["ed_share"])
    sp.append(("ed", "ed_share", "ols", m, C["ed_share"], True, ("ed_adj", "ed_adj_lo", "ed_adj_hi"), None))
    m = scope_mask & S["has_sleep"]
    sp.append(("sleep_mean", "sleep_mean_min", "ols", m, S["mean_min"], True,
               ("sleep_mean_adj", "sleep_mean_adj_lo", "sleep_mean_adj_hi"), None))
    sp.append(("sleep_var", "sleep_variation_min", "median", m, S["sd_min"], True,
               ("sleep_var_median_adj", "sleep_var_median_adj_lo", "sleep_var_median_adj_hi"), np.log(nights)))
    return sp


def put(r, cols, est):
    """Write the joint model's adjusted value only where the set's own statistic is written (else blank)."""
    if is_num(r.get(cols[0], np.nan)):
        r[cols[0]], r[cols[1]], r[cols[2]] = est
    else:
        r[cols[0]] = r[cols[1]] = r[cols[2]] = np.nan


def trend_row(scope, grouping, outcome, kind, y, dec, C, source=""):
    r = models.trend_fit(kind, y.astype(float), dec.astype(float), C)
    return {"scope": scope, "source": source, "grouping": grouping, "outcome": outcome, "model": kind,
            "n": int(len(y)), **r}


def fit_specs(specs, dec, age, sex, nights, tag, scope, grouping, trends, trend_scale=None):
    """The joint estimates per outcome, and the trend rows."""
    J = {}
    for key, tname, kind, m, y, ws, cols, extra in specs:
        C = models.covs(age[m], sex[m], ws, extra[m] if extra is not None else None)
        J[key] = models.joint(kind, y[m].astype(float), dec[m], DECILES, C, f"{tag} {key}")
        yt = 100 * y[m] if (trend_scale and key == "ed") else y[m]
        trends.append(trend_row(scope, grouping, tname + ("_pct" if (trend_scale and key == "ed") else ""), kind,
                                yt, dec[m], C))
    return J


def grouping_rows(name, fam, kk, order_g, N, B, C, S, SA, age, sex, nights, specs, trends, check_first=False):
    """One grouping's 10 decile rows: the set statistics, the joint model's adjusted values, the hiding across deciles."""
    M = decile_masks(order_g, N)
    top90 = np.zeros(N, bool)
    top90[order_g[:int(round(N * 0.1))]] = True
    top80 = np.zeros(N, bool)
    top80[order_g[:int(round(N * 0.2))]] = True
    assert (M[10] == top90).all() and ((M[9] | M[10]) == top80).all(), f"{name}: deciles 9 and 10 are not the top sets"
    dec = np.zeros(N, int)
    for j in DECILES:
        dec[M[j]] = j
    R = {j: row(M[j], B, C, S, SA) for j in DECILES}
    K = {j: counts(M[j], B, C, S) for j in DECILES}
    hid, _ = decile_hides(K, row(top80, B, C, S, SA), row(top90, B, C, S, SA))
    if check_first:
        for key, _, kind, m, y, *_ in specs:
            if kind in ("logit", "ols"):
                models.check_intercept_only(kind, y[m].astype(float), dec[m], DECILES, f"cohort {name} {key}")
    J = fit_specs(specs, dec, age, sex, nights, f"cohort {name}", "cohort", name, trends, trend_scale=True)
    rows = []
    for j in DECILES:
        r = R[j]
        for key, _, _, _, _, _, cols, _ in specs:
            put(r, cols, J[key][j])
        for fam_, cols, _ in FAMILIES:
            if (j, fam_) in hid:
                for c in cols:
                    r[c] = SUPP if c.startswith("n_") else np.nan
        rows.append({"grouping": name, "family": fam, "k": kk, "decile": j, **r})
    return rows, M


def condition_rows(scope_name, A, masks, pop_scope, age, sex, N, trends, label_pop, tag_scope):
    """Per grouping and recorded condition: the pooled bins as the categories of the joint logistic model, the
    unadjusted share of each bin (Wilson), the trend; plus 1 row per condition for its whole population."""
    rows, pooled = [], {}
    for key, code, label, lst, _ in oc.CONDITIONS:
        d = pop_scope & A[f"den_{key}"]
        k0, n0 = int(np.nansum(A[f"y_{key}"][d])), int(d.sum())
        p, lo, hi = oc.wilson_share(k0, n0)
        rows.append({"grouping": label_pop, "condition": key, "phecode": code, "label": label, "list": lst,
                     "decile_from": 0, "decile_to": 0, "n": n0, "k": k0, "rate": p, "lo": lo, "hi": hi,
                     "adj": p, "adj_lo": np.nan, "adj_hi": np.nan})
    for name, M in masks.items():
        dec = np.zeros(N, int)
        for j in DECILES:
            dec[M[j]] = j
        for key, code, label, lst, _ in oc.CONDITIONS:
            pop = pop_scope & A[f"den_{key}"]
            dm = {j: M[j] & pop for j in DECILES}
            n = {j: int(dm[j].sum()) for j in DECILES}
            k = {j: int(np.nansum(A[f"y_{key}"][dm[j]])) for j in DECILES}
            bins = pool(k, n)
            if len(bins) < 10:
                pooled[f"{name} {key}"] = [f"{b[0]}-{b[-1]}" for b in bins if len(b) > 1]
            binof = np.full(N, -1)
            for i, b in enumerate(bins):
                for j in b:
                    binof[dm[j]] = i
            y = A[f"y_{key}"][pop].astype(float)
            C = models.covs(age[pop], sex[pop], oc.with_sex(key))
            est = models.joint("logit", y, binof[pop], list(range(len(bins))), C, f"{tag_scope} {name} {key}")
            trends.append(trend_row(scope_name, name, key, "logit", y, dec[pop], C))
            for i, b in enumerate(bins):
                m = np.zeros(N, bool)
                for j in b:
                    m |= dm[j]
                nn, kk = int(m.sum()), int(np.nansum(A[f"y_{key}"][m]))
                assert not small(kk) and not small(nn - kk), f"{name} {key} {b}: a bin has 1 to 20 cases or non-cases"
                p, lo, hi = oc.wilson_share(kk, nn)
                v, vlo, vhi = est[i]
                rows.append({"grouping": name, "condition": key, "phecode": code, "label": label, "list": lst,
                             "decile_from": b[0], "decile_to": b[-1], "n": nn, "k": kk, "rate": p, "lo": lo, "hi": hi,
                             "adj": v, "adj_lo": vlo, "adj_hi": vhi})
    return pd.DataFrame(rows), pooled


def finish(rows):
    """The decile rows as a frame, every family's columns present, written at the policy's precision."""
    from release import coarsen, coarsen_sig
    D = pd.DataFrame(rows)
    for _, cols, _ in FAMILIES:
        for c in cols:
            if c not in D.columns:
                D[c] = np.nan
    return coarsen_sig(coarsen(D, RATE_COLS), SIG_COLS)


def nights_of(idx):
    return oc.read_sleep(idx)["nights"].values.astype(float)


def cohort_frame():
    """work/thresholds.csv indexed by id, with the cohort's order."""
    cf = pd.read_csv(work("thresholds.csv"))
    cf[ID] = cf[ID].astype(str)
    return cf, pd.Index(cf[ID], name=ID)

