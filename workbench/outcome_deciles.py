#!/usr/bin/env python3
"""outcome_deciles.py -- the outcomes by DECILE of the distance (Figure 2A to C and eTable 7; eFigure 4 and
eTable 13 for all 12 groupings): the screening thresholds met, the barriers to care and Fitbit sleep, from
the least atypical tenth of the cohort to the most.

WHY. The main text reports these outcomes for the semantic encoder alone. Non-overlapping deciles show
the whole range of the distance: where each measure rises, and whether the least atypical tenths sit at
the cohort or below it. The recorded conditions are reported the same way (phecode_deciles.py).

THE DECILES. phecode_deciles.decile_masks, imported: each grouping's own order of the cohort
(breadth_exclusive.rank_orders, most extreme first, the same stable sort as every set) is cut at
round(N * j / 10), j = 0 to 10. Decile 10 is the most atypical tenth, decile 1 the least. Decile 10 is
exactly the sweeps' top set at the 90th, and deciles 9 and 10 together exactly their top set at the 80th
(checked below).

PER GROUPING (the sweeps' 12) AND DECILE, by the sweeps' own functions, imported, not copied:
  n_set, mean_age, sd_age, share_female            care_sweep.describe
  n_all_evaluable, mean_breadth, sd_breadth, se, ci_lo, ci_hi
                                                   breadth_sweep.stats, over the people with all 6
                                                   criteria evaluable
  mean_breadth_adj, _adj_lo, _adj_hi               the same adjusted for age and sex: 1 reference curve,
                                                   thresholds met ~ 1 + age + age^2 + sex by least squares
                                                   (care_adjusted.fit_curve), fitted once on everyone with all
                                                   6 evaluable; the cohort's mean + the decile's mean gap,
                                                   normal 95% interval of the gaps
  n_survey; access_, cost_, avoid_ rate/lo/hi; n_ehr; ed_mean/lo/hi
                                                   care_sweep.describe, unadjusted
  access_adj, access_adj_lo, access_adj_hi, ... ed_adj ...
                                                   care_adjusted.describe: the 4 reference curves (age,
                                                   age^2, sex), fitted once, exactly as care_adjusted.py
  n_sleep, coverage, sleep_mean/lo/hi, sleep_var_median/lo/hi, sleep_age, share_wear
                                                   sleep_sweep.describe, unadjusted
  sleep_mean_adj ..., sleep_var_median_adj ...     sleep_adjusted.describe: its 2 reference curves,
                                                   fitted once, exactly as sleep_adjusted.py
plus the cohort row (decile 0). The adjusted cohort row equals the unadjusted one (checked).

THE DISSEMINATION POLICY (disclosure.py)
  - Within a decile: each describe function applies its sweep's own rules (a count of 1 to 20, a count
    whose complement in the decile is 1 to 20, a statistic over a suppressed count).
  - Across deciles, as phecode_deciles.py. The deciles partition the cohort, whose totals are public, and
    deciles 9 and 10 are sets the sweeps already published (the 80th minus the 90th, and the 90th). For
    each count a statistic rests on (evaluable, women, answered the survey, each barrier, at least 1
    visit, at least 14 nights, wear study), with its complement:
      * deciles 9 and 10 are hidden where the sweep's own function withholds the value at the 80th or
        at the 90th, and wherever their own count or complement is 1 to 20;
      * deciles 1 to 8 partition a public total (the cohort minus the top set at the 80th) and go
        through phecode_deciles.hide_cells, so a hidden cell is never recovered by subtraction. Where
        the 80th is withheld, the partition is over all 10 deciles.
    Hiding a count hides every statistic resting on it: the survey count hides the 3 barrier rates, the
    sleep count hides share_wear.
  - Precision (disclosure.py), applied last: rates, shares and their intervals to 3
    decimals; means, SDs and the ED share to 4 significant figures.
  - Printed output follows the policy too: only written values are printed.

INPUTS   screen_out/cutoff_flags.csv, cutoff_summary.json, scores_<grouping>.csv and the configs
         (everything breadth_sweep.py reads); screen_out/care_outcomes_per_person.csv (care.py
         --outcomes-only); sleep_person_local.csv, wear_enrollees_local.csv (beside the scripts); the
         responses file (age, sex). Per person, all of them stay on the Workbench.
         If present, screen_out/{breadth_sweep,care_sweep,care_adjusted,sleep_sweep,sleep_adjusted}/*.csv:
         decile 10 is checked against their 90th.
OUTPUTS  screen_out/outcome_deciles/outcome_deciles.csv        1 row per grouping x decile, plus the cohort row
         screen_out/outcome_deciles/outcome_deciles_meta.json
         screen_out/outcome_deciles_<CDR>.zip                    those 2, checked by
                                                                 zip_screen_aggregates.verdict()
The figures are drawn locally, by scripts/fig_outcome_deciles.py in the repository.

Run:  python3 outcome_deciles.py
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from breadth_exclusive import rank_orders                                  # noqa: E402  the same order
from breadth_sweep import ENCODERS, groupings, read_json, stats            # noqa: E402  the sweep's own pieces
import care_adjusted                                                       # noqa: E402  its curves and describe()
import care_sweep                                                          # noqa: E402  its describe()
from disclosure import SUPP, coarsen, coarsen_sig, pair_hidden, small     # noqa: E402  the policy
from phecode_deciles import DECILES, decile_masks, hide_cells             # noqa: E402  the same deciles, the same rule
import sleep_adjusted                                                      # noqa: E402  its curves and describe()
from sleep_reliability import MIN_NIGHTS, sleep_cache                      # noqa: E402  one rule, one file
import sleep_sweep                                                         # noqa: E402  its describe()
from env_versions import versions                                          # noqa: E402  library versions, for the Methods
from wb_config import CFG                                                  # noqa: E402
from zip_screen_aggregates import verdict                                  # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "outcome_deciles")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
BREADTH = ["mean_breadth", "sd_breadth", "se", "ci_lo", "ci_hi"]
BREADTH_ADJ = ["mean_breadth_adj", "mean_breadth_adj_lo", "mean_breadth_adj_hi"]
RATE_KEYS = list(care_sweep.RATES)                                         # access, cost, avoid
CARE_ADJ = {"access": ("access_rate", "access_lo", "access_hi"), "cost": ("cost_rate", "cost_lo", "cost_hi"),
            "avoid": ("avoid_rate", "avoid_lo", "avoid_hi"), "ed": ("ed_mean", "ed_lo", "ed_hi")}
SLEEP_ADJ = ["sleep_mean", "sleep_lo", "sleep_hi", "sleep_var_median", "sleep_var_lo", "sleep_var_hi"]

# Each count a written statistic rests on: (family, the columns it hides, the family whose hiding forces it)
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
SIG_COLS = (["mean_age", "sd_age"] + BREADTH + BREADTH_ADJ + ["ed_mean", "ed_lo", "ed_hi", "ed_adj", "ed_adj_lo", "ed_adj_hi",
            "sleep_mean", "sleep_lo", "sleep_hi", "sleep_var_median", "sleep_var_lo", "sleep_var_hi", "sleep_age",
            "sleep_mean_adj", "sleep_mean_adj_lo", "sleep_mean_adj_hi", "sleep_var_median_adj",
            "sleep_var_median_adj_lo", "sleep_var_median_adj_hi"])


def load(idx):
    """The per-person inputs, in the cohort order every breadth script uses, and the 6 reference curves."""
    N = len(idx)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"))
    cf[ID_COL] = cf[ID_COL].astype(str)
    cf = cf.set_index(ID_COL).reindex(idx)
    R = pd.read_csv(CFG.responses_csv, usecols=[ID_COL, "age", "sex"])
    R[ID_COL] = R[ID_COL].astype(str)
    R = R.set_index(ID_COL).reindex(idx)
    assert R.notna().all().all(), "age or sex missing for someone in the cohort"
    assert set(np.unique(R["sex"])) <= {0.0, 1.0}, "sex is not coded 0/1 as build_responses_v9.py writes it"
    age, sex = R["age"].values.astype(float), R["sex"].values.astype(float)

    # care: care_sweep.py's inputs, and care_adjusted.py's 4 curves, fitted once, as that script fits them
    p = os.path.join(OUT_DIR, "care_outcomes_per_person.csv")
    assert os.path.exists(p), f"{p} is missing: run `python3 care.py --outcomes-only`"
    O = pd.read_csv(p)
    O[ID_COL] = O[ID_COL].astype(str)
    O = O.set_index(ID_COL).reindex(idx)
    C = {"age": age, "sex": sex,
         "has_survey": O["has_survey"].fillna(False).astype(bool).values,      # absent from the file = no survey
         "has_ehr": O["has_ehr"].fillna(False).astype(bool).values,
         "ed_share": O["ed_share"].values.astype(float)}
    for col in care_sweep.RATES.values():
        C[col] = O[col].values.astype(float)
    for short, (col, has, _) in care_adjusted.OUTCOMES.items():
        m = C[has]
        y = C[col][m]
        assert np.isfinite(y).all(), f"{col} is missing for someone it is measured on"
        fitted, _ = care_adjusted.fit_curve(age[m], sex[m], y)
        C[f"gap_{short}"] = np.full(N, np.nan)
        C[f"gap_{short}"][m] = y - fitted
        C[f"coh_{short}"] = float(y.mean())

    # sleep: sleep_sweep.py's inputs, and sleep_adjusted.py's 2 curves, fitted once on the same people
    P = pd.read_csv(sleep_cache())
    P["person_id"] = P["person_id"].astype(str)
    P = P.set_index("person_id").reindex(idx)
    has = (P["nights"].fillna(0).values >= MIN_NIGHTS) & P["sd_min"].notna().values
    wf = os.path.join(HERE, "wear_enrollees_local.csv")
    wear = idx.isin(set(pd.read_csv(wf)["person_id"].astype(str))) if os.path.exists(wf) else None
    if wear is None:
        print("  wear_enrollees_local.csv not found: share_wear is left out")
    nights, mean_min, sd_min = (P[c].values.astype(float) for c in ("nights", "mean_min", "sd_min"))
    S = {"age": age, "has_sleep": has, "mean_min": mean_min, "sd_min": sd_min, "wear": wear}
    g1, g2, _, converged = sleep_adjusted.fit_curves(age[has], sex[has], nights[has], mean_min[has], sd_min[has])
    gap_mean, gap_sd = np.full(N, np.nan), np.full(N, np.nan)
    gap_mean[has], gap_sd[has] = g1, g2
    SA = {"has_sleep": has, "gap_mean": gap_mean, "gap_sd": gap_sd,
          "coh_mean": float(mean_min[has].mean()), "coh_sd_median": float(np.median(sd_min[has])),
          "coh_gap_sd_median": float(np.median(g2))}
    # thresholds met: 1 reference curve on everyone with all 6 evaluable, care_adjusted.fit_curve, adjusted
    # for age and sex like the other outcomes
    ev, br = cf["all_evaluable"].astype(bool).values, cf["breadth"].astype(float).values
    fitted, _ = care_adjusted.fit_curve(age[ev], sex[ev], br[ev])
    gap_br = np.full(N, np.nan)
    gap_br[ev] = br[ev] - fitted
    B = {"ev": ev, "br": br, "gap": gap_br, "coh": float(br[ev].mean())}
    return B, C, S, SA, converged


def row(mask, B, C, S, SA):
    """1 set, every statistic, each through its sweep's own function and within-set rules."""
    r = care_sweep.describe(mask, C)
    adj = care_adjusted.describe(mask, C)
    for short, cols in CARE_ADJ.items():
        for c, a in zip(cols, (f"{short}_adj", f"{short}_adj_lo", f"{short}_adj_hi")):
            r[a] = adj.get(c, np.nan)
    keep = mask & B["ev"]
    n, ev = int(mask.sum()), int(keep.sum())
    st = stats(B["br"][keep])
    r["n_all_evaluable"] = SUPP if pair_hidden(ev, n) else ev              # breadth_sweep.policy, on one set
    g = B["gap"][keep]
    if ev >= 2:
        m, se = float(g.mean()), float(np.std(g, ddof=1) / np.sqrt(ev))
        st.update({"mean_breadth_adj": B["coh"] + m, "mean_breadth_adj_lo": B["coh"] + m - care_adjusted.Z * se,
                   "mean_breadth_adj_hi": B["coh"] + m + care_adjusted.Z * se})
    if small(ev) or pair_hidden(ev, n):
        st = {k: np.nan for k in st}
    r.update(st)
    s = sleep_sweep.describe(mask, S)
    r.update({k: v for k, v in s.items() if k != "n_set"})
    sa = sleep_adjusted.describe(mask, SA)
    for c in ("sleep_mean", "sleep_var_median"):
        r[f"{c}_adj"] = sa.get(c, np.nan)
    for c, a in (("sleep_lo", "sleep_mean_adj_lo"), ("sleep_hi", "sleep_mean_adj_hi"),
                 ("sleep_var_lo", "sleep_var_median_adj_lo"), ("sleep_var_hi", "sleep_var_median_adj_hi")):
        r[a] = sa.get(c, np.nan)
    return r


def counts(mask, B, C, S):
    """The raw counts each family rests on, with the total its complement is taken from."""
    n = int(mask.sum())
    sv, e, sl = mask & C["has_survey"], mask & C["has_ehr"], mask & S["has_sleep"]
    out = {"evaluable": (int((mask & B["ev"]).sum()), n),
           "female": (int((C["sex"][mask] == care_sweep.FEMALE).sum()), n),
           "survey": (int(sv.sum()), n), "ehr": (int(e.sum()), n), "sleep": (int(sl.sum()), n)}
    for s, col in care_sweep.RATES.items():
        out[s] = (int(np.nansum(C[col][sv])), int(sv.sum()))
    out["wear"] = (int((sl & S["wear"]).sum()), int(sl.sum())) if S["wear"] is not None else (0, int(sl.sum()))
    return out


def written(r, fam):
    """Did the sweep's own function write this family's statistic for this set?"""
    first = {"evaluable": "mean_breadth", "female": "share_female", "survey": "n_survey", "ehr": "n_ehr",
             "sleep": "n_sleep", "wear": "share_wear"}.get(fam, f"{fam}_rate")
    v = r.get(first, np.nan)
    if isinstance(v, str):                                   # "<=20" or "suppressed"
        return False
    return pd.notna(v)


def decile_hides(K, top80, top90):
    """The (decile, family) cells to hide across the partition (see the docstring). K[j][fam] = (count, total)."""
    hid, why = set(), {"sweep_withheld": 0, "own_count": 0, "partition": 0, "forced_by_parent": 0}
    for fam, _, parent in FAMILIES:                          # parents come first in FAMILIES
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


def check_against_sweeps(D, name):
    """Decile 10 must carry the sweeps' 90th where both files wrote it (rounding allowed for)."""
    pairs = [("breadth_sweep", "breadth_sweep.csv", {"mean_breadth": "mean_breadth"}, "grouping == @name and cut_pctile == 90"),
             ("care_sweep", "care_sweep.csv", {"access_rate": "access_rate", "ed_mean": "ed_mean", "mean_age": "mean_age"},
              "analysis == 'all' and grouping == @name and cut_pctile == 90"),
             ("care_adjusted", "care_adjusted.csv", {"access_adj": "access_rate", "ed_adj": "ed_mean"},
              "analysis == 'all' and grouping == @name and cut_pctile == 90"),
             ("sleep_sweep", "sleep_sweep.csv", {"sleep_mean": "sleep_mean", "sleep_var_median": "sleep_var_median"},
              "analysis == 'all' and grouping == @name and cut_pctile == 90"),
             ("sleep_adjusted", "sleep_adjusted.csv", {"sleep_mean_adj": "sleep_mean", "sleep_var_median_adj": "sleep_var_median"},
              "analysis == 'all' and grouping == @name and cut_pctile == 90")]
    d10 = D[(D.grouping == name) & (D.decile == 10)].iloc[0]
    done = []
    for folder, fn, cols, q in pairs:
        p = os.path.join(OUT_DIR, folder, fn)
        if not os.path.exists(p):
            continue
        W = pd.read_csv(p).query(q)
        if len(W) != 1:
            continue
        for ours, theirs in cols.items():
            a, b = pd.to_numeric(d10.get(ours), errors="coerce"), pd.to_numeric(W.iloc[0][theirs], errors="coerce")
            if pd.notna(a) and pd.notna(b):
                assert abs(a - b) <= 1e-6 * max(1.0, abs(b)), f"{name} {ours}: decile 10 {a} is not {folder}'s 90th {b}"
        done.append(folder)
    return done


def main():
    os.makedirs(OUT, exist_ok=True)
    NB = int(read_json("cutoff_summary.json")["n_criteria"])
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)
    N = len(idx)
    B, C, S, SA, converged = load(idx)
    print(f"[outcome_deciles] cohort {N:,} | all {NB} criteria evaluable {int(B['ev'].sum()):,} | answered the access "
          f"survey {int(C['has_survey'].sum()):,} | at least 1 visit {int(C['has_ehr'].sum()):,} | at least "
          f"{MIN_NIGHTS} nights {int(S['has_sleep'].sum()):,} | median regression converged: {converged}")

    coh = row(np.ones(N, bool), B, C, S, SA)
    for short, (v, _, _) in CARE_ADJ.items():                  # the gaps average 0 over the cohort
        assert abs(coh[f"{short}_adj"] - coh[v]) < 1e-9, f"the cohort's adjusted {short} moved"
    assert abs(coh["sleep_mean_adj"] - coh["sleep_mean"]) < 1e-9, "the cohort's adjusted sleep mean moved"
    assert abs(coh["mean_breadth_adj"] - coh["mean_breadth"]) < 1e-9, "the cohort's adjusted thresholds met moved"
    assert abs(coh["sleep_var_median_adj"] - coh["sleep_var_median"]) < 1e-9, "the cohort's adjusted variability moved"
    rows = [{"grouping": "cohort", "family": "reference", "k": np.nan, "decile": 0, **coh}]

    G12 = groupings()
    order = rank_orders([g[0] for g in G12], idx)
    hidden_total = {}
    for name, fam, kk, _ in G12:
        M = decile_masks(order[name], N)
        top90 = np.zeros(N, bool); top90[order[name][:int(round(N * 0.1))]] = True
        top80 = np.zeros(N, bool); top80[order[name][:int(round(N * 0.2))]] = True
        assert (M[10] == top90).all() and ((M[9] | M[10]) == top80).all(), f"{name}: deciles are not the sweeps' sets"
        assert np.sum([M[j] for j in DECILES], axis=0).max() == 1 and sum(M[j].sum() for j in DECILES) == N
        R = {j: row(M[j], B, C, S, SA) for j in DECILES}
        K = {j: counts(M[j], B, C, S) for j in DECILES}
        hid, why = decile_hides(K, row(top80, B, C, S, SA), row(top90, B, C, S, SA))
        hidden_total[name] = why
        for j in DECILES:
            r = R[j]
            for fam_, cols, _ in FAMILIES:
                if (j, fam_) in hid:
                    for c in cols:
                        if c.startswith("n_"):
                            r[c] = SUPP
                        else:
                            r[c] = np.nan
            rows.append({"grouping": name, "family": fam, "k": kk, "decile": j, **r})

    D = pd.DataFrame(rows)
    for _, cols, _ in FAMILIES:
        for c in cols:
            if c not in D.columns:
                D[c] = np.nan
    checked = {name: check_against_sweeps(D, name) for name, *_ in G12}
    print(f"  decile 10 checked against the sweeps' 90th: {sorted(set(sum(checked.values(), [])))}")
    D = coarsen_sig(coarsen(D, RATE_COLS), SIG_COLS)          # ★ precision (disclosure.py): after every check
    D.to_csv(os.path.join(OUT, "outcome_deciles.csv"), index=False)

    c0 = D.iloc[0]
    g = D[D.grouping == ENCODERS[0]].set_index("decile")

    def line(label, col, scale=1.0, fmt="{:5.2f}"):
        vals = " ".join("    -" if pd.isna(pd.to_numeric(g.at[j, col], errors="coerce"))
                        else fmt.format(scale * float(g.at[j, col])) for j in DECILES)
        print(f"    {label:34s} [{fmt.format(scale * float(c0[col]))}] {vals}")
    print(f"  {ENCODERS[0]}, by decile (1 = least atypical ... 10 = most); cohort in []:")
    line("thresholds met, mean", "mean_breadth")
    line("thresholds met, adjusted", "mean_breadth_adj")
    line("mean age, y", "mean_age", fmt="{:5.1f}")
    for s in RATE_KEYS:
        line(f"{s} barrier, adjusted, %", f"{s}_adj", 100, "{:5.1f}")
    line("ED share, adjusted, %", "ed_adj", 100, "{:5.2f}")
    line("sleep, adjusted, min", "sleep_mean_adj", fmt="{:5.1f}")
    line("night-to-night variation, adj., min", "sleep_var_median_adj", fmt="{:5.1f}")
    tot = {k: sum(w[k] for w in hidden_total.values()) for k in next(iter(hidden_total.values()))}
    print(f"  cells hidden across deciles (12 groupings): {tot}")

    json.dump({"cdr": CDR, "deciles": "each grouping's own order, cut at round(N * j / 10); decile 10 = the most "
                                      "atypical tenth = the sweeps' top set at the 90th; deciles 9 + 10 = their top set at the 80th",
               "n_criteria": NB, "min_nights": MIN_NIGHTS,
               "groupings_all": [{"grouping": g_, "family": f_, "k": k, "k_rule": r} for g_, f_, k, r in G12],
               "statistics": "breadth_sweep.stats, care_sweep.describe, care_adjusted.describe, sleep_sweep.describe, "
                             "sleep_adjusted.describe (imported)",
               "adjustment": {"thresholds met": "OLS on age, age^2, sex, fitted once on everyone with all 6 evaluable",
                              "care": "OLS on age, age^2, sex (care_adjusted.py's 4 curves, fitted once)",
                              "sleep": "OLS (mean) and median regression with log(nights) (variability), "
                                       "sleep_adjusted.py's curves, fitted once", "median_regression_converged": converged},
               "checked_against": checked, "withheld": hidden_total,
               "disclosure": "All of Us dissemination policy (disclosure.py); see the docstring",
               "precision": "rates, shares, intervals to 3 decimals; means, SDs, ED share to 4 significant figures",
               "versions": versions()},
              open(os.path.join(OUT, "outcome_deciles_meta.json"), "w"), indent=2)
    files = ["outcome_deciles.csv", "outcome_deciles_meta.json"]
    for fn in files:
        ok, why = verdict(os.path.join(OUT, fn))
        assert ok, f"{fn} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"outcome_deciles_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"outcome_deciles/{fn}")
    print(f"\n[outcome_deciles] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
