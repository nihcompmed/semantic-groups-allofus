#!/usr/bin/env python3
"""phecode_fifteen.py -- the 15 phecode diagnoses side by side, one model per condition, with the 0s corrected where a
test exists. The 15 conditions are phecodes, not the survey instruments. The 0 is corrected for the 4 diseases with a
test (hypercholesterolemia, chronic kidney disease, breast cancer, prostate cancer), and everything else is kept the
same for the other conditions.

THE PEOPLE. Stage 2's denominators (people with at least 1 ICD-coded event; prostate cancer: men), with two changes:
  - breast cancer: women only;
  - for the 4 conditions with a test, BOTH cases and non-cases are restricted to people whose record shows the test,
    at any time ("checked": had the test):
      breast cancer        a mammogram or breast-screening encounter (phecode_screened.py's checks, cached)
      prostate cancer      a PSA test or prostate-screening encounter (the same cache)
      hypercholesterolemia any cholesterol measurement: total, LDL, HDL, triglycerides, or a lipid panel
      chronic kidney dis.  any serum creatinine or eGFR measurement, or a panel that includes creatinine
  The other 11 keep 0 = no diagnosis recorded. The label is Stage 2's (PheTK count >= 2).

ONE MODEL PER CONDITION, all ages: condition ~ age_z + age_z^2 (+ sex where both sexes are
in the population) + the 27 groups, unpenalized logistic, on everyone in the population (no folds):
  whether  the joint Wald test of the 27 group terms (27 df); passes at p < 0.05/15 (15 conditions)
  which    each group's odds ratio per SD, net of age, sex and the others; passes at p < 0.05/(k x 15), k = the
           encoder's number of groups (gte: 27, so 0.05/405)
  alone    each group ALONE: condition ~ age_z + age_z^2 (+ sex) + that group; the same threshold
For the 11 unchanged conditions the model is Stage 2's model on Stage 2's people: its pseudo-R^2 must equal Stage 2's
(checked where phecode_directions' meta file is present).

COUNTS are written as they are. The only check is pii_check: no person-level column and no written index in any table.

INPUTS   as phecode_directions.py; screen_out/screening_checks_local.csv (phecode_screened.py; run it first);
         BigQuery (concept, measurement, procedure_occurrence) once for the cholesterol and kidney tests, then the
         cache screen_out/lab_checks_local.csv (person level, stays on the Workbench)
OUTPUTS  screen_out/phecode_fifteen/phecode_fifteen_<enc>.csv       1 row per condition
         screen_out/phecode_fifteen/phecode_fifteen_or_<enc>.csv    1 row per condition x group (together and alone)
         screen_out/phecode_fifteen/phecode_fifteen_<enc>_meta.json (with the lab concepts found)
         screen_out/phecode_fifteen_<enc>_<CDR>.zip                 those 3

Run:  python3 phecode_fifteen.py        (--refresh pulls the lab tests again; --max-gb N, default 400)
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phecode_directions import direction_scores                          # noqa: E402  Stage 2's group scores
from phecode_sweep import CONDITIONS, FEMALE, Z, load                    # noqa: E402  Stage 1's labels
from screen_battery import ENCODERS                                      # noqa: E402
from zip_screen_aggregates import ID_COLS                                # noqa: E402  person-level column names

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "phecode_fifteen")
STAGE2 = os.path.join(OUT_DIR, "phecode_directions")
SCREEN_CHECKS = os.path.join(OUT_DIR, "screening_checks_local.csv")      # phecode_screened.py's cache (person level)
LAB_CHECKS = os.path.join(OUT_DIR, "lab_checks_local.csv")               # person level. Stays on the Workbench.
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
ENC = sys.argv[sys.argv.index("--encoder") + 1] if "--encoder" in sys.argv else ENCODERS[0]
MAX_GB = float(sys.argv[sys.argv.index("--max-gb") + 1]) if "--max-gb" in sys.argv else 400.0
N_COND = len(CONDITIONS)
P_JOINT = 0.05 / N_COND                                                   # 15 joint tests
LABS = {
    "lipid": {"loinc": ["2093-3", "2085-9", "13457-7", "18262-6", "2089-1", "2571-8", "43396-1", "9830-1",
                        "24331-1", "57698-3"],
              "cpt": ["80061", "82465", "83718", "83721", "84478"]},
    "kidney": {"loinc": ["2160-0", "38483-4", "14682-9", "33914-3", "48642-3", "48643-1", "62238-1", "88293-6",
                         "88294-4", "98979-8", "50044-7"],
               "cpt": ["82565", "80047", "80048", "80053", "80069"]},
}
CHECKED_BY = {"breast": ("screen", "breast"), "prostate": ("screen", "prostate"),
              "hyperchol": ("lab", "lipid"), "ckd": ("lab", "kidney")}


def pull_labs(idx):
    """Per person and kind (lipid, kidney): whether any such test is in the record. Pulled once, cached."""
    if os.path.exists(LAB_CHECKS) and "--refresh" not in sys.argv:
        L = pd.read_csv(LAB_CHECKS, dtype={ID_COL: str})
        print(f"  [lab tests: read from {LAB_CHECKS}, nothing pulled]", flush=True)
        return L, None
    from google.cloud import bigquery
    client = bigquery.Client()
    parts, params = [], []
    for kind, c in LABS.items():
        parts.append(f"SELECT concept_id, concept_code, vocabulary_id, concept_name, '{kind}' AS kind FROM `{CDR}.concept` "
                     f"WHERE (vocabulary_id = 'LOINC' AND concept_code IN UNNEST(@{kind}_loinc)) "
                     f"OR (vocabulary_id IN ('CPT4', 'HCPCS') AND concept_code IN UNNEST(@{kind}_cpt))")
        params += [bigquery.ArrayQueryParameter(f"{kind}_loinc", "STRING", c["loinc"]),
                   bigquery.ArrayQueryParameter(f"{kind}_cpt", "STRING", c["cpt"])]
    codes_q = " UNION ALL ".join(parts)
    cfg = lambda p, dry: bigquery.QueryJobConfig(query_parameters=p, dry_run=dry, use_query_cache=not dry)
    concepts = client.query(codes_q, job_config=cfg(params, False)).to_dataframe()
    print(f"  [lab tests: {len(concepts):,} concepts found: " +
          ", ".join(f"{k} {int((concepts.kind == k).sum())}" for k in LABS) + "]", flush=True)
    p_all = params + [bigquery.ArrayQueryParameter("ids", "INT64", [int(i) for i in idx])]
    ev_sql = f"""
      WITH codes AS ({codes_q}),
      ev AS (
        SELECT person_id, measurement_date AS d, measurement_concept_id AS c FROM `{CDR}.measurement` WHERE person_id IN UNNEST(@ids)
        UNION ALL SELECT person_id, measurement_date, measurement_source_concept_id FROM `{CDR}.measurement` WHERE person_id IN UNNEST(@ids)
        UNION ALL SELECT person_id, procedure_date, procedure_concept_id FROM `{CDR}.procedure_occurrence` WHERE person_id IN UNNEST(@ids)
        UNION ALL SELECT person_id, procedure_date, procedure_source_concept_id FROM `{CDR}.procedure_occurrence` WHERE person_id IN UNNEST(@ids)
      )
      SELECT ev.person_id, codes.kind, MIN(ev.d) AS first_date, COUNT(DISTINCT ev.d) AS n_dates
      FROM ev JOIN (SELECT DISTINCT concept_id, kind FROM codes) AS codes ON ev.c = codes.concept_id
      GROUP BY ev.person_id, codes.kind"""
    gb = client.query(ev_sql, job_config=cfg(p_all, True)).total_bytes_processed / 1e9
    print(f"  [lab tests: scanning {gb:,.1f} GB]", flush=True)
    assert gb <= MAX_GB, f"the scan would read {gb:,.0f} GB, above --max-gb {MAX_GB:,.0f}; rerun with a larger --max-gb"
    L = client.query(ev_sql, job_config=cfg(p_all, False)).to_dataframe().rename(columns={"person_id": ID_COL})
    L[ID_COL] = L[ID_COL].astype(str)
    L.to_csv(LAB_CHECKS, index=False)
    print(f"  [lab tests: cached to {LAB_CHECKS} -- person level, do not download]", flush=True)
    return L, concepts


def fit(y, X):
    return sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=500)


def pii_check(path):
    """(ok, reason): no person-level column and no written index (the only check)."""
    if path.endswith(".json"):
        return True, ""
    d = pd.read_csv(path)
    hit = ID_COLS.intersection(str(c).strip().lower() for c in d.columns)
    if hit:
        return False, f"carries a person column ({', '.join(sorted(hit))})"
    unnamed = [str(c) for c in d.columns if not str(c).strip() or str(c).startswith("Unnamed:")]
    if unnamed:
        return False, f"unnamed column {unnamed[0]!r} -- an index written out, possibly person ids"
    return True, ""


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
    P_GROUP = 0.05 / (k * N_COND)                                         # k groups x 15 conditions (gte: 405)
    S = pd.read_csv(SCREEN_CHECKS, dtype={ID_COL: str})
    L, concepts = pull_labs(idx)
    checks = {("screen", kind): idx.isin(S.loc[S["kind"] == kind, ID_COL]) for kind in ("breast", "prostate")}
    checks.update({("lab", kind): idx.isin(L.loc[L["kind"] == kind, ID_COL]) for kind in LABS})
    s2meta = os.path.join(STAGE2, f"phecode_directions_{ENC}_meta.json")
    s2 = json.load(open(s2meta))["conditions"] if os.path.exists(s2meta) else None
    cp = names["construct_profile"].values if "construct_profile" in names.columns else [""] * k
    print(f"[phecode_fifteen] {ENC}: k = {k} | 15 conditions | joint test at p < 0.05/15, groups at p < 0.05/{k * N_COND}",
          flush=True)

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
        f = fit(y, np.column_stack([base, Yz[d]]))
        b, V = np.asarray(f.params), np.asarray(f.cov_params())
        g = slice(1 + base.shape[1], 1 + base.shape[1] + k)
        w = float(b[g] @ np.linalg.solve(V[g, g], b[g]))
        pj = float(chi2.sf(w, k))
        if s2 is not None and not checked and key != "breast":           # the unchanged 11: Stage 2's model
            assert abs(f.prsquared - s2[key]["pseudo_r2"]) < 1e-9, f"{key}: the model differs from Stage 2's"
        se, pv = np.asarray(f.bse)[g], np.asarray(f.pvalues)[g]
        n_tog = n_alo = 0
        for j in range(k):
            fa = fit(y, np.column_stack([base, Yz[d][:, j]]))
            ba, sa, pa = float(fa.params[-1]), float(fa.bse[-1]), float(fa.pvalues[-1])
            tog, alo = bool(pv[j] < P_GROUP), bool(pa < P_GROUP)
            n_tog += tog
            n_alo += alo
            grows.append({"encoder": ENC, "condition": key, "phecode": code, "label": label, "list": lst, "group": j,
                          "construct_profile": cp[j],
                          "or": float(np.exp(b[g][j])), "or_lo": float(np.exp(b[g][j] - Z * se[j])),
                          "or_hi": float(np.exp(b[g][j] + Z * se[j])), "p": float(pv[j]), "passes_together": tog,
                          "or_alone": float(np.exp(ba)), "or_alone_lo": float(np.exp(ba - Z * sa)),
                          "or_alone_hi": float(np.exp(ba + Z * sa)), "p_alone": pa, "passes_alone": alo})
        rows.append({"encoder": ENC, "condition": key, "phecode": code, "label": label, "list": lst,
                     "population": ("women, " if key == "breast" else "men, " if men_only else "") +
                                   ("checked (" + CHECKED_BY[key][1] + " test)" if checked else "at least 1 ICD event"),
                     "zero_means": "checked, no diagnosis recorded" if checked else "no diagnosis recorded",
                     "n_cases": int(y.sum()), "n_noncases": int((y == 0).sum()),
                     "groups_wald": w, "groups_df": k, "groups_p": pj, "passes_joint": bool(pj < P_JOINT),
                     "n_groups_together": int(n_tog), "n_groups_alone": int(n_alo),
                     "pseudo_r2": float(f.prsquared), "converged": bool(f.mle_retvals.get("converged", False))})
        print(f"  {lst} {label[:40]:40s} {'(checked)' if checked else '         '} cases {int(y.sum()):>6} / "
              f"{int((y == 0).sum()):<6} | joint p {pj:.2g} | groups together {n_tog:2d}, alone {n_alo:2d}", flush=True)

    D = pd.DataFrame(rows)
    G = pd.DataFrame(grows)
    D.to_csv(os.path.join(OUT, f"phecode_fifteen_{ENC}.csv"), index=False)
    G.to_csv(os.path.join(OUT, f"phecode_fifteen_or_{ENC}.csv"), index=False)
    meta = {"cdr": CDR, "encoder": ENC, "k": k, "p_joint": P_JOINT, "p_group": P_GROUP,
            "model": "condition ~ age_z + age_z^2 (+ sex where both sexes) + k groups, unpenalized logistic, no folds",
            "checked": {k2: v[1] for k2, v in CHECKED_BY.items()}, "labs": LABS,
            "zero": "the 4 test conditions: checked and no diagnosis recorded; the other 11: no diagnosis recorded",
            "counts": "written as they are; pii_check only"}
    if concepts is not None:
        meta["lab_concepts_found"] = concepts[["kind", "vocabulary_id", "concept_code", "concept_name"]] \
            .drop_duplicates().sort_values(["kind", "vocabulary_id", "concept_code"]).astype(str).to_dict("records")
    json.dump(meta, open(os.path.join(OUT, f"phecode_fifteen_{ENC}_meta.json"), "w"), indent=2)

    print("\n  groups that pass TOGETHER but not ALONE (fold from 1 per SD, together):")
    for r in D.itertuples():
        t = G[(G.condition == r.condition) & G.passes_together & ~G.passes_alone]
        lab = ", ".join(f"{int(x['group'])} {str(x['construct_profile']).split(':')[0]} "
                        f"{(x['or'] if x['or'] >= 1 else 1 / x['or']):.2f}" for _, x in t.iterrows())
        print(f"  {r.label[:40]:40s} {lab or 'none'}")
    files = [f"phecode_fifteen_{ENC}.csv", f"phecode_fifteen_or_{ENC}.csv", f"phecode_fifteen_{ENC}_meta.json"]
    for fn in files:
        ok, why = pii_check(os.path.join(OUT, fn))
        assert ok, f"{fn} refused: {why}"
    zpath = os.path.join(OUT_DIR, f"phecode_fifteen_{ENC}_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"phecode_fifteen/{fn}")
    print(f"\n[phecode_fifteen] {len(D)} conditions\n  download -> {zpath}")


if __name__ == "__main__":
    main()
