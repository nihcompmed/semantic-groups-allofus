#!/usr/bin/env python3
"""care_covariates.py -- the cohort care gradient on one encoder's own percentile with the sociodemographic
covariates held fixed (--gradient), and the leader models: care by what carries a
person with total burden AND the covariates held fixed (--legacy).

STANDALONE on the Workbench (numpy, pandas, scipy, statsmodels). --gradient runs after screen_battery.py
(scores_<model>.csv), care.py --outcomes-only and covariates.py. --legacy also needs the outlier_leaders_*.csv
files, which no script in this folder writes.

THE LEADER MODELS (--legacy only). Among the members of a tail with the outcome:

    outcome ~ leading construct + burden + age + sex + race and ethnicity + income + education + employment
              + insurance (+ total visits for the ED share)

Leading construct as dummies (every construct leading at least MIN_COUNT people, reference psychosis; a leader
under MIN_COUNT would be pooled as "other", so at MIN_COUNT = 0 every leader present in the fitted subset is a
level and nothing is pooled); burden rank-standardized within the tail, one model per burden measure (the
distance on the components, the scale-score distance, content extremity, breadth, and for a comparator its own
distance); each categorical covariate as dummies against its largest level, every level kept as is (a level
under covariates.MIN_COUNT would be pooled into "Other (pooled)"; at 0 none is) and "Prefer not to answer" kept
as a level. Logistic regression for the binary outcomes, OLS for the ED share.

REPORTED by the leader models (--legacy), per outcome and burden measure: the adjusted prevalence
per leading construct by marginal standardization (every member's prediction with the leader set to
that construct, averaged), with a 95 percent interval by the delta method; the adjusted risk ratio
against the reference with its interval; the spread of adjusted prevalences; the likelihood-ratio (or F) test of the leader beyond burden and
covariates; the burden coefficient. The same again on the complete-covariate subset (no "Prefer not to
answer" on any covariate) as the sensitivity, flagged covariate_set = complete. The covariate coefficients
of the content-extremity model are written too.

--gradient: across the cohort with the outcome, (a) outcome ~ burden percentile (per SD of rank) +
covariates, the coefficient and, for binary outcomes, the odds ratio per SD; (b) outcome ~ decile of burden
+ covariates, the adjusted prevalence per decile by marginal standardization and the risk ratio of the top
decile against the bottom.

INPUTS   --gradient: screen_out/scores_<model>.csv (gte-large-en-v1.5 by default), care_outcomes_per_person.csv,
         covariates_per_person.csv, responses_k5.csv (age, sex, the 151 items), the item table,
         and scores_scale.csv when present (read, used only by --legacy)
         --legacy also: outlier_leaders_outliers.csv (or outlier_leaders_<model>_tail.csv), consensus_outliers.csv
OUTPUTS  (-> screen_out/, aggregates only, suffix _<model>, which the default makes _gte-large-en-v1.5)
         care_gradient_covariates_<model>.csv, care_gradient_covariates_<model>_summary.csv     (--gradient)
         --legacy:
         care_covariates.csv            outcome x burden x covariate_set x leader: n, raw rate, adjusted prevalence
                                        with interval, adjusted RR with interval
         care_covariates_summary.csv    outcome x burden x covariate_set: n, model, burden beta with interval,
                                        spread raw and adjusted, LR or F, p, pseudo-R2 without and with the leader
         care_covariates_coefs.csv      the covariate coefficients of the extremity model per outcome
         care_covariates.json           settings, leaders kept, pooled levels

★ THE EXPOSURE. The exposure of --gradient is ONE ENCODER'S
OWN percentile from scores_<model>.csv, and --model DEFAULTS to the prespecified gte-large-en-v1.5,
so the gradient never falls back to the six-encoder pooled `outlier_score` of consensus_outliers.csv.
consensus_outliers.csv is not read except under --legacy. scores_scale.csv is read only if present. The
leader models (no --gradient) read outlier_leaders_*.csv, which no script in this folder writes, and run
only with --legacy, and their files are written without the dissemination policy.
The dissemination policy (disclosure.py) is applied to the decile file: the decile
counts partition the fitted n, and a raw rate is blanked where the count it implies is 1 to 20.

Run:  python3 care_covariates.py --gradient                       (gte-large-en-v1.5)
      python3 care_covariates.py --gradient --model bge-m3        (any encoder)
      python3 care_covariates.py --gradient --model factor        (a comparator's own distance, once
                                                                   comparators.py has written scores_factor.csv)
   The leader models. Every one of these needs --legacy, or the script stops:
      python3 care_covariates.py --legacy                         (the 2,177, their leaders)
      python3 care_covariates.py --legacy --model factor          (the factor tail, its leaders, also --model scale)
      python3 care_covariates.py --legacy --reference "Healthcare discrimination"   (risk ratios against another
                                                                  leader, outputs get the suffix _ref_healthcare)
      python3 care_covariates.py --legacy --leaders rbc           (the 2,177 with the reconstruction-based leader,
                                                                  outputs get the suffix _rbc)
      python3 care_covariates.py --legacy --min-leads 3           (small tests only)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from covariates import COVARIATES, PNA, pool_levels, apply_pool   # covariate names, non-answer level, pooling rule (a no-op at covariates.MIN_COUNT = 0)

from wb_config import CFG   # noqa: E402  battery and responses folder, resolved once, no defaults
OUT_DIR       = os.path.join(HERE, "screen_out")
RESPONSES_CSV = os.path.join(CFG.responses_dir, "responses_k5.csv")
ITEMS_CSV     = CFG.items_csv
ID_COL        = "id"
LEGACY        = "--legacy" in sys.argv
MODEL         = (sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv
                 else None if LEGACY else "gte-large-en-v1.5")   # None only on the --legacy path
GRADIENT      = "--gradient" in sys.argv
MIN_COUNT     = int(sys.argv[sys.argv.index("--min-leads") + 1]) if "--min-leads" in sys.argv else 0
# used only by the --legacy leader models. It is NOT the dissemination policy, which forbids counts of 1 to 20
# inclusive and is applied to the --gradient output through disclosure.py
REFERENCE     = sys.argv[sys.argv.index("--reference") + 1] if "--reference" in sys.argv else "Psychosis"
REF_TAG       = "" if REFERENCE == "Psychosis" else "_ref_" + REFERENCE.split()[0].lower()   # file suffix for another reference
LEADERS       = sys.argv[sys.argv.index("--leaders") + 1] if "--leaders" in sys.argv else None   # e.g. rbc: outlier_leaders_rbc.csv
BINARY        = ["any_access_barrier", "any_cost_barrier", "care_avoid_provider"]
OUTCOMES      = BINARY + ["ed_share"]
SUF           = (f"_{MODEL}" if MODEL else "") + (f"_{LEADERS}" if LEADERS else "") + REF_TAG
Z95           = 1.959963985


def rank_z(v):
    r = pd.Series(np.asarray(v, float)).rank(pct=True).values
    return (r - r.mean()) / r.std()


def read(name):
    d = pd.read_csv(os.path.join(OUT_DIR, name))
    d[ID_COL] = d[ID_COL].astype(str)
    return d.set_index(ID_COL)


# ----------------------------------------------------------------------------- design and fitting
def covariate_columns(d):
    """Dummies for the five categorical covariates against their largest level (levels under covariates.MIN_COUNT pooled;
    none at 0). Returns (frame, pooled)."""
    X = pd.DataFrame(index=d.index)
    pooled = {}
    for c in COVARIATES:
        s = d[c].astype(str)
        small = pool_levels([s])
        if small:
            s = apply_pool(s, small)
            pooled[c] = small
        ref = s.value_counts().idxmax()
        for lv in sorted(s.unique()):
            if lv != ref:
                X[f"{c}={lv}"] = (s == lv).astype(float)
    return X, pooled


def base_columns(d, burden_col, extra_num):
    X = pd.DataFrame(index=d.index)
    X["const"] = 1.0
    if burden_col is not None:
        X["burden_z"] = rank_z(d[burden_col])
    X["age_z"] = rank_z(d["age"])
    X["sex"] = d["sex"].astype(float)
    for c in extra_num:
        X[c + "_z"] = rank_z(d[c])
    C, pooled = covariate_columns(d)
    return pd.concat([X, C], axis=1), pooled


def group_columns(values, levels, reference):
    """Dummies for a grouping variable against `reference` (values outside `levels` pooled as 'other')."""
    g = pd.Series(values).where(pd.Series(values).isin(levels), "other")
    X = pd.DataFrame(index=g.index)
    for L in [l for l in levels if l != reference] + (["other"] if (g == "other").any() else []):
        X[f"G_{L}"] = (g == L).astype(float).values
    return X, g


def fit(y, X, binary):
    """Returns (result, is_logit). Logistic with an OLS fallback on failure (separation)."""
    if binary:
        try:
            res = sm.Logit(y, X.values).fit(disp=0, maxiter=300)
            if np.isfinite(res.llf) and np.all(np.isfinite(res.params)):
                return res, True
        except Exception as e:
            print(f"    [logistic fit failed ({type(e).__name__}); linear probability model used]")
    return sm.OLS(y, X.values).fit(), False


def standardize(res, logit, X, group_cols, level):
    """Marginal standardization: every row's prediction with the group set to `level`. Returns (mean, gradient wrt params)."""
    Xl = X.copy()
    for c in group_cols:
        Xl[c] = 0.0
    if f"G_{level}" in Xl.columns:
        Xl[f"G_{level}"] = 1.0
    eta = Xl.values @ res.params
    if logit:
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = (p * (1 - p))[:, None] * Xl.values
    else:
        p = eta
        grad = Xl.values
    return float(p.mean()), grad.mean(axis=0)


def adjusted_by_group(res, logit, X, group_cols, levels, reference):
    """Adjusted prevalence (mean, lo, hi) per level and RR against the reference (rr, lo, hi), delta method."""
    cov = np.asarray(res.cov_params())
    est = {L: standardize(res, logit, X, group_cols, L) for L in levels}
    out = {}
    m_ref, g_ref = est[reference]
    for L, (m, g) in est.items():
        se = float(np.sqrt(g @ cov @ g))
        if m > 0 and m_ref > 0:
            gl = g / m - g_ref / m_ref
            se_l = float(np.sqrt(gl @ cov @ gl))
            rr = m / m_ref
            out[L] = (m, m - Z95 * se, m + Z95 * se, rr, rr * np.exp(-Z95 * se_l), rr * np.exp(Z95 * se_l))
        else:
            out[L] = (m, m - Z95 * se, m + Z95 * se, np.nan, np.nan, np.nan)
    return out


def joint_test(res_full, res_base, logit, k):
    if k == 0:
        return np.nan, np.nan
    if logit:
        stat = 2.0 * (res_full.llf - res_base.llf)
        return float(stat), float(stats.chi2.sf(stat, k))
    stat = ((res_base.ssr - res_full.ssr) / k) / (res_full.ssr / res_full.df_resid)
    return float(stat), float(stats.f.sf(stat, k, res_full.df_resid))


# ----------------------------------------------------------------------------- data
def load_common():
    R = pd.read_csv(RESPONSES_CSV)
    R[ID_COL] = R[ID_COL].astype(str)
    R = R.set_index(ID_COL)
    items = pd.read_csv(ITEMS_CSV)["item"].astype(str).tolist()
    Z = (R[items] - R[items].mean()) / R[items].std(ddof=0).replace(0, np.nan)
    R["extremity"] = Z.abs().mean(axis=1)
    O = read("care_outcomes_per_person.csv")
    C = read("covariates_per_person.csv")
    D = R[["age", "sex", "extremity"]].join(O[OUTCOMES + ["n_visits", "has_survey", "has_ehr"]]).join(C[COVARIATES])
    co = None
    if LEGACY:                                             # the consensus exposure, never the default
        co = read("consensus_outliers.csv")
        D["framework"] = co["outlier_score"]
    if os.path.exists(os.path.join(OUT_DIR, "scores_scale.csv")):
        D["scale"] = read("scores_scale.csv")["pctile"]
    if MODEL:
        D["own"] = read(f"scores_{MODEL}.csv")["pctile"]
        print(f"exposure: the percentile of {MODEL}'s own distance, scores_{MODEL}.csv")
    return D, co


def tail_frame(D, co):
    if LEADERS:
        assert not MODEL, "--leaders applies to the consensus outliers, not to a comparator's tail"
        L = read(f"outlier_leaders_{LEADERS}.csv")            # the same people, another leader rule
    elif MODEL:
        L = read(f"outlier_leaders_{MODEL}_tail.csv")
    else:
        L = read("outlier_leaders_outliers.csv")
    T = D.loc[L.index].copy()
    T["leader"] = L["leader"]
    T["breadth"] = L["breadth"]
    return T


# ----------------------------------------------------------------------------- the leader models
def run_leaders(D, co):
    global REFERENCE
    T = tail_frame(D, co)
    counts = T["leader"].value_counts()
    kept = [c for c, k in counts.items() if k >= MIN_COUNT]             # at 0: every leader, nothing pooled
    if REFERENCE not in kept:
        print(f"  reference {REFERENCE} leads fewer than {max(MIN_COUNT, 1)}; using {kept[0]}")
        REFERENCE = kept[0]
    burdens = (["own"] if MODEL else []) + ["framework"] + (["scale"] if MODEL != "scale" else []) + ["extremity", "breadth"]
    print(f"{'the ' + MODEL + ' tail' if MODEL else 'consensus outliers'} {len(T):,} | leaders kept ({MIN_COUNT}+) {len(kept)}, "
          f"pooled {int(counts[~counts.index.isin(kept)].sum())} | reference {REFERENCE}")
    complete = ~T[COVARIATES].eq(PNA).any(axis=1)
    print(f"  complete covariates for {int(complete.sum()):,} of {len(T):,}")
    rows, summ, coefs, pooled_all = [], [], [], {}
    for o in OUTCOMES:
        extra = ["n_visits"] if o == "ed_share" else []
        for b in burdens:
            for cset, mask in (("full", pd.Series(True, index=T.index)), ("complete", complete)):
                d = T[mask].dropna(subset=[o, b, "age", "sex"] + extra)
                if len(d) < 50:
                    continue
                y = d[o].astype(float).values
                Xb, pooled = base_columns(d, b, extra)
                # leaders kept are re-evaluated on the fitted subset (the breadth subset is smaller), so no
                # reported group falls under MIN_COUNT; a leader dropped here is pooled into "other". At 0 every
                # leader PRESENT in the subset is kept (max(MIN_COUNT, 1): a leader with nobody in the subset
                # would be an all-zero column, not a group)
                vc_d = d["leader"].value_counts()
                kept_d = [c for c in kept if vc_d.get(c, 0) >= max(MIN_COUNT, 1)]
                if REFERENCE not in kept_d:
                    kept_d = [REFERENCE] + kept_d
                Xg, lead = group_columns(d["leader"].values, kept_d, REFERENCE)
                Xg.index = d.index
                Xf = pd.concat([Xb, Xg], axis=1)
                gcols = list(Xg.columns)
                res_f, logit = fit(y, Xf, o in BINARY)
                res_b, _ = fit(y, Xb, o in BINARY) if logit else (sm.OLS(y, Xb.values).fit(), False)
                stat, p = joint_test(res_f, res_b, logit, len(gcols))
                params = pd.Series(res_f.params, index=Xf.columns)
                ci = pd.DataFrame(np.asarray(res_f.conf_int()), index=Xf.columns)
                adj = adjusted_by_group(res_f, logit, Xf, gcols, kept_d + (["other"] if "G_other" in gcols else []), REFERENCE)
                raw = {L: float(d.loc[lead.values == L, o].mean()) for L in kept_d}
                spread_raw = max(raw.values()) - min(raw.values())
                adj_k = {L: v[0] for L, v in adj.items() if L in kept_d}
                spread_adj = max(adj_k.values()) - min(adj_k.values())
                r2b = float(res_b.prsquared if logit else res_b.rsquared)
                r2f = float(res_f.prsquared if logit else res_f.rsquared)
                summ.append({"outcome": o, "burden": b, "covariate_set": cset, "n": int(len(d)), "model": "logistic" if logit else "ols",
                             "burden_beta": float(params["burden_z"]), "burden_lo": float(ci.loc["burden_z", 0]), "burden_hi": float(ci.loc["burden_z", 1]),
                             "spread_raw": spread_raw, "spread_adjusted": spread_adj, "stat_leader": stat, "p_leader": p,
                             "r2_without_leader": r2b, "r2_with_leader": r2f,
                             "top_leader": max(adj_k, key=adj_k.get), "bottom_leader": min(adj_k, key=adj_k.get)})
                for L, (m, lo, hi, rr, rlo, rhi) in adj.items():
                    nL = int((lead.values == L).sum())
                    rows.append({"outcome": o, "burden": b, "covariate_set": cset, "leader": L, "n": nL,
                                 "rate_raw": float(d.loc[lead.values == L, o].mean()) if nL else np.nan,
                                 "adj_prevalence": m, "adj_lo": lo, "adj_hi": hi, "rr_vs_reference": rr, "rr_lo": rlo, "rr_hi": rhi})
                if b == "extremity" and cset == "full":
                    pooled_all[o] = pooled
                    for term in Xf.columns:
                        if term == "const" or term.startswith("G_"):
                            continue
                        coefs.append({"outcome": o, "model": "logistic" if logit else "ols", "term": term, "coef": float(params[term]),
                                      "lo": float(ci.loc[term, 0]), "hi": float(ci.loc[term, 1]),
                                      "or": float(np.exp(params[term])) if logit else np.nan})
                if cset == "full":
                    f = lambda v: f"{100 * v:.1f}"
                    order = sorted(adj_k, key=lambda k_: -adj_k[k_])
                    print(f"\n{o} | burden = {b} | n {len(d):,} | burden {'log-odds' if logit else 'beta'} per SD {params['burden_z']:+.3f} "
                          f"[{ci.loc['burden_z', 0]:+.3f}, {ci.loc['burden_z', 1]:+.3f}] | {'LR' if logit else 'F'} {stat:.1f}, p {p:.1e}")
                    print("   adjusted prevalence by leader (RR vs " + REFERENCE + "): "
                          + "; ".join(f"{L} {f(adj[L][0])}% [{f(adj[L][1])}, {f(adj[L][2])}] RR {adj[L][3]:.2f}" for L in order))
                    print(f"   spread raw {f(spread_raw)}% -> adjusted {f(spread_adj)}%")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"care_covariates{SUF}.csv"), index=False)
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(OUT_DIR, f"care_covariates{SUF}_summary.csv"), index=False)
    pd.DataFrame(coefs).to_csv(os.path.join(OUT_DIR, f"care_covariates{SUF}_coefs.csv"), index=False)
    json.dump({"model": MODEL or "framework", "n_tail": int(len(T)), "n_complete_covariates": int(complete.sum()),
               "min_leads": MIN_COUNT, "leaders_kept": kept, "reference": REFERENCE, "burdens": burdens,
               "pooled_levels_extremity_model": pooled_all, "summary": json.loads(S.to_json(orient="records"))},
              open(os.path.join(OUT_DIR, f"care_covariates{SUF}.json"), "w"), indent=2)
    print(f"\nwrote -> {OUT_DIR}: care_covariates{SUF}.csv, care_covariates{SUF}_summary.csv, care_covariates{SUF}_coefs.csv, care_covariates{SUF}.json")


# ----------------------------------------------------------------------------- the cohort gradient
def run_gradient(D, co):
    from disclosure import SUPP, marked, pair_hidden, partition_mask
    pct_col = "own"
    rows, summ = [], []
    for o in OUTCOMES:
        extra = ["n_visits"] if o == "ed_share" else []
        d = D.dropna(subset=[o, pct_col, "age", "sex"] + extra).copy()
        y = d[o].astype(float).values
        # (a) per SD of rank
        Xa, pooled = base_columns(d, pct_col, extra)
        res_a, logit = fit(y, Xa, o in BINARY)
        pa = pd.Series(res_a.params, index=Xa.columns); ca = pd.DataFrame(np.asarray(res_a.conf_int()), index=Xa.columns)
        # (b) deciles
        dec = (pd.Series(d[pct_col].values).rank(pct=True).values * 10).clip(1e-9, 10 - 1e-9)
        d["decile"] = np.ceil(dec).astype(int)
        Xb, _ = base_columns(d, None, extra)
        levels = [str(k) for k in range(1, 11)]
        Xg, g = group_columns(d["decile"].astype(str).values, levels, "1")
        Xg.index = d.index
        Xf = pd.concat([Xb, Xg], axis=1)
        res_b, logit_b = fit(y, Xf, o in BINARY)
        adj = adjusted_by_group(res_b, logit_b, Xf, list(Xg.columns), levels, "1")
        for k in levels:
            m, lo, hi, rr, rlo, rhi = adj[k]
            rows.append({"outcome": o, "decile": int(k), "n": int((g.values == k).sum()),
                         "rate_raw": float(d.loc[g.values == k, o].mean()), "adj_prevalence": m, "adj_lo": lo, "adj_hi": hi,
                         "rr_vs_decile1": rr, "rr_lo": rlo, "rr_hi": rhi})
        summ.append({"outcome": o, "n": int(len(d)), "model": "logistic" if logit else "ols",
                     "beta_per_sd": float(pa["burden_z"]), "beta_lo": float(ca.loc["burden_z", 0]), "beta_hi": float(ca.loc["burden_z", 1]),
                     "or_per_sd": float(np.exp(pa["burden_z"])) if logit else np.nan,
                     "or_lo": float(np.exp(ca.loc["burden_z", 0])) if logit else np.nan, "or_hi": float(np.exp(ca.loc["burden_z", 1])) if logit else np.nan,
                     "rr_top_vs_bottom_decile": adj["10"][3], "rr_lo": adj["10"][4], "rr_hi": adj["10"][5],
                     "adj_prev_bottom": adj["1"][0], "adj_prev_top": adj["10"][0]})
        print(f"{o}: n {len(d):,} | per SD of rank {pa['burden_z']:+.4f} [{ca.loc['burden_z', 0]:+.4f}, {ca.loc['burden_z', 1]:+.4f}]"
              + (f" (OR {np.exp(pa['burden_z']):.2f})" if logit else "")
              + f" | adjusted prevalence decile 1 {100 * adj['1'][0]:.1f}% -> decile 10 {100 * adj['10'][0]:.1f}%, RR {adj['10'][3]:.2f} [{adj['10'][4]:.2f}, {adj['10'][5]:.2f}]")
    # the policy, per outcome: decile counts partition the fitted n; a raw rate gives its count back
    G = pd.DataFrame(rows)
    G["n"] = G["n"].astype(object)
    n_hidden = 0
    for o, ix in G.groupby("outcome").groups.items():
        ix = list(ix)
        nn = [int(x) for x in G.loc[ix, "n"]]
        hide, prim = partition_mask(nn)
        G.loc[ix, "n"] = marked(nn, hide, prim)
        for i, n_i, h in zip(ix, nn, hide):
            k_i = round(float(G.at[i, "rate_raw"]) * n_i) if o in BINARY and n_i else None
            if h or (k_i is not None and pair_hidden(k_i, n_i)):
                G.at[i, "rate_raw"] = np.nan
                n_hidden += 1
    G.to_csv(os.path.join(OUT_DIR, f"care_gradient_covariates{SUF}.csv"), index=False)
    print(f"dissemination policy: {n_hidden} decile rows with a raw rate withheld")
    pd.DataFrame(summ).to_csv(os.path.join(OUT_DIR, f"care_gradient_covariates{SUF}_summary.csv"), index=False)
    print(f"\nwrote -> {OUT_DIR}: care_gradient_covariates{SUF}.csv, care_gradient_covariates{SUF}_summary.csv")


def main():
    if not GRADIENT and not LEGACY:
        raise SystemExit("only --gradient is part of the paper. The leader models read files that no script in "
                         "this folder writes; pass --legacy to run them anyway.")
    D, co = load_common()
    if GRADIENT:
        run_gradient(D, co)
    else:
        run_leaders(D, co)


if __name__ == "__main__":
    main()
