#!/usr/bin/env python3
"""phecode_sweep.py -- STAGE 1 of the phecodes: is being an outlier associated with a higher chance of each
diagnosis, with age and sex standardized?

Runs in the MAIN environment (python3), like care_adjusted.py. It needs no PheTK: it reads the
person-level counts phecode_coverage.py wrote with PheTK 0.3.6 (phecodeX X1.0).

THE CONDITIONS (15). Each is 0/1 per person: 1 = PheTK count >= 2, and
a person with exactly 1 event counts as not having it.
  A  the record side of the criteria: MB_286.2 major depressive disorder, MB_288.3 generalized anxiety
     disorder, MB_304 ADHD, MB_284 suicidal ideation, attempt or self-harm
  B  other mental and behavioral codes with >= 1,000 cases: MB_282 nicotine dependence, MB_280
     psychoactive substance related disorders, MB_290.1 PTSD, MB_280.1 alcohol use disorders, MB_286.1
     bipolar disorder
  C  Biji et al. 2026's six: EM_239.1 hypercholesterolemia, RE_475 asthma, CV_404 ischemic heart disease,
     GU_582.2 chronic kidney disease, CA_105 breast cancer, CA_107.2 prostate cancer

THE DENOMINATOR. The people with at least 1 ICD-coded event in PheTK's extraction
(61,631 of the cohort). Prostate cancer: the men among them. Breast cancer keeps men.

PER SET (the sets of breadth_sweep.py and breadth_exclusive.py, through their own code: all outliers,
12 groupings; exclusive outliers, 7 groupings in each of the 6 encoder slots; cuts 80 to 99)
  n_set, n_rec          the set, and its people with at least 1 ICD event (the coverage, written, not drawn)
  <c>_rate, _lo, _hi    UNADJUSTED share with the condition over the set's denominator, Wilson 95%
  <c>_adj, _adj_lo, _adj_hi
                        ADJUSTED for age and sex exactly as care and sleep: one least-squares curve per
                        condition, condition ~ 1 + age + age^2 + sex, fitted once over its denominator (for
                        prostate cancer, age and age^2 only). A set's value = the cohort's share + the set's
                        mean gap from the curve, normal 95% interval of the gaps, the curve treated as known.
                        The gaps average 0 over the cohort, so the cohort row does not move (checked).

THE DISSEMINATION POLICY (disclosure.py), applied per condition:
  - a set of 1 to 20 is blanked (n_set "<=20"); an exclusive set whose complement in the top set is 1
    to 20 is written "suppressed" (as care_sweep.py);
  - a share is withheld when the set's cases, or its non-cases, or its denominator, number 1 to 20;
  - ★ NESTED SETS. Rare diagnoses make a count recoverable by subtraction between sets that contain one
    another, so a share is also withheld when that subtraction would leave 1 to 20 cases or non-cases:
      the top set at one cut contains the same grouping's top set at every higher cut;
      the top set contains the same grouping's exclusive set at the same cut;
      the cohort contains every set;
      within a set, psychoactive substance (MB_280) contains alcohol (MB_280.1), which rolls up to it;
      ★ and the two combined: (substance - alcohol) in a larger set minus the same in a set it contains
      counts people with substance but not alcohol between the two sets.
    The same checks guard n_rec. The smaller set is the one withheld, and the check repeats until
    nothing changes.
  - an adjusted value is written only where the unadjusted one is.
  - ★ A SEX-RESTRICTED DENOMINATOR IS NEVER WRITTEN. The coverage table counts prostate cancer over both
    sexes. The men-only denominator times the cohort's prostate share gives the men's count, and the
    difference is the women with a prostate cancer label. So the men-only n_people for prostate cancer
    in phecode_reference_curves.csv is not written.
  - ★ PRECISION. The same denominator can come back through the digits: a rate with its Wilson
    interval, written to 16 digits, solves exactly for n and k, and the curve's intercept + age^2
    coefficient is the cohort share. Shares and intervals are written to 3 decimals and the curve's
    coefficients, SEs, age mean and SD to 4 significant figures (disclosure.coarsen); at 3 decimals the
    cohort row fits at least 41 different prostate case counts.
  - ★ where n_rec is "suppressed", every share on the row is blanked: 14 shares over the same
    hidden n_rec, even rounded, would fix it (disclosure.py, the care_sweep finding).

INPUTS   screen_out/phetk/phecode_counts_cohort_local.tsv, screen_out/phetk/icd_person_summary_local.csv
         (phecode_coverage.py, person-level, stay); the responses file (age, sex); everything
         breadth_sweep.py reads; screen_out/phecode_coverage/phecode_counts.csv for a cross-check
OUTPUTS  screen_out/phecode_sweep/phecode_sweep.csv            1 row per set, plus the cohort row
         screen_out/phecode_sweep/phecode_reference_curves.csv  the 15 curves' coefficients and SEs
         screen_out/phecode_sweep/phecode_sweep_meta.json
         screen_out/phecode_sweep_<CDR>.zip                     those 3, checked by zip_screen_aggregates.verdict()

Run:  python3 phecode_sweep.py
"""
import itertools
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from breadth_exclusive import rank_orders, top_and_exclusive           # noqa: E402  the same people
from breadth_sweep import CUTS, ENCODERS, groupings                      # noqa: E402
from disclosure import MARK, SUPP, coarsen, coarsen_sig, pair_hidden, small  # noqa: E402  the policy
from wb_config import CFG                                               # noqa: E402
from zip_screen_aggregates import verdict                               # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "phecode_sweep")
COUNTS = os.path.join(OUT_DIR, "phetk", "phecode_counts_cohort_local.tsv")
PERSON = os.path.join(OUT_DIR, "phetk", "icd_person_summary_local.csv")
COVERAGE = os.path.join(OUT_DIR, "phecode_coverage", "phecode_counts.csv")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
Z = 1.959964
MIN_COUNT = 2                         # PheTK's case rule (phecode_coverage.py)
FEMALE = 0.0                          # build_responses_v9.py: female = 0, male = 1
# key, phecode, label, list, denominator restricted to men
CONDITIONS = [
    ("mdd", "MB_286.2", "Major depressive disorder", "A", False),
    ("gad", "MB_288.3", "Generalized anxiety disorder", "A", False),
    ("adhd", "MB_304", "Attention-deficit hyperactivity disorder", "A", False),
    ("suicide", "MB_284", "Suicidal ideation, attempt or self-harm", "A", False),
    ("nicotine", "MB_282", "Nicotine dependence", "B", False),
    ("substance", "MB_280", "Psychoactive substance related disorders", "B", False),
    ("ptsd", "MB_290.1", "Posttraumatic stress disorder", "B", False),
    ("alcohol", "MB_280.1", "Alcohol use disorders", "B", False),
    ("bipolar", "MB_286.1", "Bipolar disorder", "B", False),
    ("hyperchol", "EM_239.1", "Hypercholesterolemia", "C", False),
    ("asthma", "RE_475", "Asthma", "C", False),
    ("ihd", "CV_404", "Ischemic heart disease", "C", False),
    ("ckd", "GU_582.2", "Chronic kidney disease", "C", False),
    ("breast", "CA_105", "Breast cancer", "C", False),
    ("prostate", "CA_107.2", "Prostate cancer", "C", True),
]
CONTAINS = [("substance", "alcohol")]  # MB_280.1 rolls up to MB_280: an alcohol case is a substance case


def wilson(k, n):
    p = k / n
    d = 1 + Z ** 2 / n
    c = (p + Z ** 2 / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / d
    return p, min(c - h, p), max(c + h, p)      # at k = 0 the bound is 0 up to rounding; keep it at p


def fit_curve(age, sex, y, with_sex=True):
    """y ~ 1 + age + age^2 (+ sex) by least squares, age centered and scaled on the fitting sample."""
    a = (age - age.mean()) / age.std()
    cols = [a, a ** 2] + ([sex] if with_sex else [])
    fit = sm.OLS(y, sm.add_constant(np.column_stack(cols))).fit()
    terms = ["intercept", "age_z", "age_z_squared"] + (["sex_male"] if with_sex else [])
    rows = [{"term": t, "coef": b, "se": s, "age_mean": age.mean(), "age_sd": age.std(), "n_people": len(y)}
            for t, b, s in zip(terms, fit.params, fit.bse)]
    return np.asarray(fit.fittedvalues), rows


def load(idx):
    """Per person, in the cohort's order: age, sex, has_rec, and each condition's 0/1 label (NaN outside
    its denominator)."""
    N = len(idx)
    R = pd.read_csv(CFG.responses_csv, usecols=[ID_COL, "age", "sex"])
    R[ID_COL] = R[ID_COL].astype(str)
    R = R.set_index(ID_COL).reindex(idx)
    assert R.notna().all().all(), "age or sex missing for someone in the cohort"
    assert set(np.unique(R["sex"])) <= {0.0, 1.0}, "sex is not coded 0/1 as build_responses_v9.py writes it"
    for f in (COUNTS, PERSON):
        assert os.path.exists(f), f"{f} is missing: run phecode_coverage.py first (in ~/phetk_env)"
    P = pd.read_csv(PERSON, dtype={"person_id": str}).set_index("person_id").reindex(idx)
    has_rec = P["n_events"].fillna(0).values > 0
    C = pd.read_csv(COUNTS, sep="\t", dtype={"person_id": str, "phecode": str})
    C = C[C["phecode"].isin([c[1] for c in CONDITIONS]) & C["person_id"].isin(idx)]
    A = {"age": R["age"].values.astype(float), "sex": R["sex"].values.astype(float), "has_rec": has_rec}
    male = A["sex"] != FEMALE
    for key, code, _, _, men_only in CONDITIONS:
        cnt = C[C["phecode"] == code].set_index("person_id")["count"].reindex(idx).fillna(0).values
        den = has_rec & male if men_only else has_rec
        A[f"den_{key}"] = den
        A[f"cases_any_sex_{key}"] = int((has_rec & (cnt >= MIN_COUNT)).sum())   # phecode_coverage.py's count
        A[f"y_{key}"] = np.where(den, (cnt >= MIN_COUNT).astype(float), np.nan)
    return A


def set_stats(mask, A):
    """Raw per-condition numbers for 1 set: denominator, cases, and the adjusted mean gap with its SE."""
    s = {"n_set": int(mask.sum()), "n_rec": int((mask & A["has_rec"]).sum())}
    for key, *_ in CONDITIONS:
        d = mask & A[f"den_{key}"]
        n = int(d.sum())
        k = int(np.nansum(A[f"y_{key}"][d]))
        g = A[f"gap_{key}"][d]
        s[key] = {"n": n, "k": k, "gap": float(g.mean()) if n else np.nan,
                  "se": float(np.std(g, ddof=1) / np.sqrt(n)) if n > 1 else np.nan}
    return s


def protect(sets, relations):
    """Which (set, condition) shares, and which n_rec, may be written. sets: id -> set_stats. relations:
    (larger set id, contained set id). Returns (hidden shares, hidden n_rec, how many shares the
    primary rule hid before the nested-set checks)."""
    hid = {(sid, key) for sid, s in sets.items() for key, *_ in CONDITIONS
           if small(s[key]["n"]) or small(s[key]["k"]) or small(s[key]["n"] - s[key]["k"]) or s[key]["n"] < 2}
    n_primary = len(hid)
    hid_rec = {sid for sid, s in sets.items() if pair_hidden(s["n_rec"], s["n_set"])}
    changed = True
    while changed:
        changed = False
        for big, sub in relations:
            B, S = sets[big], sets[sub]
            if sub not in hid_rec and big not in hid_rec and small(B["n_rec"] - S["n_rec"]):
                hid_rec.add(sub)
                changed = True
            for key, *_ in CONDITIONS:
                if (sub, key) in hid or (big, key) in hid:
                    continue
                dk = B[key]["k"] - S[key]["k"]
                dm = (B[key]["n"] - B[key]["k"]) - (S[key]["n"] - S[key]["k"])
                if small(dk) or small(dm):
                    hid.add((sub, key))
                    changed = True
            for big_c, sub_c in CONTAINS:            # (substance - alcohol), across the two sets
                if {(big, big_c), (big, sub_c), (sub, big_c), (sub, sub_c)} & hid:
                    continue
                dd = (B[big_c]["k"] - B[sub_c]["k"]) - (S[big_c]["k"] - S[sub_c]["k"])
                dn = (B[big_c]["n"] - S[big_c]["n"]) - dd
                if small(dd) or small(dn):
                    hid.add((sub, sub_c))
                    changed = True
        for sid, s in sets.items():                  # within a set: substance contains alcohol
            for big_c, sub_c in CONTAINS:
                if (sid, big_c) in hid or (sid, sub_c) in hid:
                    continue
                if small(s[big_c]["k"] - s[sub_c]["k"]):
                    hid.add((sid, sub_c))
                    changed = True
    return hid, hid_rec, n_primary


def main():
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)         # the order every breadth script uses
    N = len(idx)
    A = load(idx)
    print(f"[phecode_sweep] cohort {N:,} | with at least 1 ICD event {int(A['has_rec'].sum()):,}")

    # the labels must reproduce phecode_coverage.py's case counts
    if os.path.exists(COVERAGE):
        T = pd.read_csv(COVERAGE, dtype=str).set_index("phecode")
        for key, code, *_ in CONDITIONS:
            k = A[f"cases_any_sex_{key}"]              # both sexes, as phecode_coverage.py counted
            w = T.at[code, "n_case"]
            if str(w).isdigit():
                assert int(w) == k, f"{key}: {k} cases here, {w} in phecode_coverage.py"
        print("  case counts match phecode_counts.csv")

    coef = []
    for key, code, _, _, men_only in CONDITIONS:
        d = A[f"den_{key}"]
        fitted, rows = fit_curve(A["age"][d], A["sex"][d], A[f"y_{key}"][d], with_sex=not men_only)
        A[f"gap_{key}"] = np.full(N, np.nan)
        A[f"gap_{key}"][d] = A[f"y_{key}"][d] - fitted
        A[f"coh_{key}"] = float(np.mean(A[f"y_{key}"][d]))
        if men_only:                                 # ★ never write a sex-restricted denominator (see the policy)
            rows = [{**r, "n_people": "not written"} for r in rows]
        coef += [{"condition": key, "phecode": code, **r} for r in rows]
    Cf = pd.DataFrame(coef)

    # the sets, through the breadth scripts' own code
    G12 = groupings()
    order = rank_orders([g[0] for g in G12], idx)
    info = {g[0]: g for g in G12}
    names12 = [g[0] for g in G12]
    others = [g[0] for g in G12 if g[0] not in ENCODERS]
    meta_rows, sets, rel = [], {}, []
    sets["cohort"] = set_stats(np.ones(N, bool), A)
    meta_rows.append(("cohort", {"analysis": "cohort", "encoder_slot": "", "grouping": "cohort",
                                 "family": "reference", "k": np.nan, "cut_pctile": 0}, None))
    for cut in CUTS:
        n_top, member, _ = top_and_exclusive(order, names12, N, cut)
        for name in names12:
            sid = ("all", name, cut)
            sets[sid] = set_stats(member[name], A)
            _, fam, k, _ = info[name]
            meta_rows.append((sid, {"analysis": "all", "encoder_slot": "", "grouping": name, "family": fam,
                                    "k": k, "cut_pctile": cut}, None))
            rel.append(("cohort", sid))
        for slot in ENCODERS:
            names7 = [slot] + others
            _, _, excl = top_and_exclusive(order, names7, N, cut)
            for name in names7:
                sid = ("exclusive", slot, name, cut)
                sets[sid] = set_stats(excl[name], A)
                _, fam, k, _ = info[name]
                meta_rows.append((sid, {"analysis": "exclusive", "encoder_slot": slot, "grouping": name,
                                        "family": fam, "k": k, "cut_pctile": cut, "n_top": n_top}, n_top))
                rel += [(("all", name, cut), sid), ("cohort", sid)]
    for name in names12:                             # the top sets are nested across cuts
        for p, q in itertools.combinations(CUTS, 2):
            rel.append((("all", name, p), ("all", name, q)))
    hid, hid_rec, n_primary = protect(sets, rel)

    rows = []
    for sid, head, n_top in meta_rows:
        s = sets[sid]
        r = dict(head)
        n = s["n_set"]
        if n == 0:
            r["n_set"] = 0
            rows.append(r)
            continue
        if small(n):
            r["n_set"] = MARK
            rows.append(r)
            continue
        r["n_set"] = SUPP if (n_top is not None and pair_hidden(n, n_top)) else n
        r["n_rec"] = (MARK if small(s["n_rec"]) else SUPP) if sid in hid_rec else s["n_rec"]
        for key, *_ in CONDITIONS:
            if (sid, key) in hid or sid in hid_rec:  # ★ shares over a suppressed n_rec would return it
                continue
            c = s[key]
            p, lo, hi = wilson(c["k"], c["n"])
            m = A[f"coh_{key}"] + c["gap"]
            r.update({f"{key}_rate": p, f"{key}_lo": lo, f"{key}_hi": hi,
                      f"{key}_adj": m, f"{key}_adj_lo": m - Z * c["se"], f"{key}_adj_hi": m + Z * c["se"]})
        rows.append(r)
    D = pd.DataFrame(rows)
    stat_cols = [f"{key}_{w}" for key, *_ in CONDITIONS for w in ("rate", "lo", "hi", "adj", "adj_lo", "adj_hi")]
    for c in stat_cols:
        if c not in D.columns:
            D[c] = np.nan
    for key, *_ in CONDITIONS:                       # the gaps average 0 over each denominator
        c0 = D.iloc[0]
        assert abs(c0[f"{key}_adj"] - c0[f"{key}_rate"]) < 1e-9, f"the cohort's adjusted {key} moved"
    D = coarsen(D, stat_cols)                        # ★ precision (disclosure.py): after every check above
    Cf = coarsen_sig(Cf, ["coef", "se", "age_mean", "age_sd"])
    D.to_csv(os.path.join(OUT, "phecode_sweep.csv"), index=False)
    Cf.to_csv(os.path.join(OUT, "phecode_reference_curves.csv"), index=False)

    c0 = D.iloc[0]
    print("  at the 95th, adjusted share, % (cohort | gte all | gte exclusive), withheld = -:")
    at = lambda a, g, s="": D[(D.analysis == a) & (D.grouping == g) & (D.encoder_slot == s) & (D.cut_pctile == 95)].iloc[0]
    ga, ge = at("all", ENCODERS[0]), at("exclusive", ENCODERS[0], ENCODERS[0])
    f = lambda v: "   -" if pd.isna(v) else f"{100 * v:5.1f}"
    for key, code, label, lst, _ in CONDITIONS:
        print(f"    {lst} {code:9s} {label:42s} {f(c0[key + '_rate'])} | {f(ga[key + '_adj'])} | {f(ge[key + '_adj'])}")
    withheld = int(D[[f"{key}_rate" for key, *_ in CONDITIONS]].iloc[1:].isna().sum().sum())
    print(f"  shares withheld under the policy: {withheld:,} of {len(CONDITIONS) * (len(D) - 1):,} "
          f"(1 to 20 in the set itself: {n_primary:,}; added by the nested-set checks: {len(hid) - n_primary:,}; "
          f"the rest are empty or blanked sets)")

    json.dump({"cdr": CDR, "cuts": CUTS, "slots": ENCODERS,
               "conditions": [{"key": k, "phecode": c, "label": l, "list": s, "denominator":
                               "men with at least 1 ICD event" if m else "people with at least 1 ICD event"}
                              for k, c, l, s, m in CONDITIONS],
               "case_rule": f"PheTK count >= {MIN_COUNT} (phecodeX X1.0, PheTK 0.3.6); 1 event = not a case",
               "groupings_all": [{"grouping": g, "family": f_, "k": k, "k_rule": r} for g, f_, k, r in G12],
               "exclusive_groupings": "the slot's encoder plus " + ", ".join(others),
               "adjustment": "OLS on age, age^2, sex (prostate: age, age^2), fitted once over each denominator; "
                             "set value = cohort share + the set's mean gap",
               "intervals": "Wilson 95% unadjusted; normal 95% of the gaps adjusted, the curve treated as known",
               "withheld": f"cases, non-cases or denominator of 1 to 20 ({MARK}), and nested-set subtraction checks",
               "sex_coding": "female = 0, male = 1", "disclosure": "All of Us dissemination policy (disclosure.py)",
               "precision": "shares and intervals to 3 decimals; curve coefficients, SEs, age mean and SD to 4 "
                            "significant figures (disclosure.py)"},
              open(os.path.join(OUT, "phecode_sweep_meta.json"), "w"), indent=2)
    files = ["phecode_sweep.csv", "phecode_reference_curves.csv", "phecode_sweep_meta.json"]
    for fn in files:
        ok, why = verdict(os.path.join(OUT, fn))
        assert ok, f"{fn} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"phecode_sweep_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"phecode_sweep/{fn}")
    print(f"\n[phecode_sweep] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
