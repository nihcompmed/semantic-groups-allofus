#!/usr/bin/env python3
"""conditions_jno.py -- the recorded conditions of Figure 2D and Figure 4, in one place. Imported by
phecode_counts_jno.py, conditions_deciles_jno.py, conditions_assoc_jno.py, conditions_assoc_adjusted_jno.py and
breast_timing_jno.py. Run on its own, it pulls the 3 tests once and writes their summary.

THE 8 CONDITIONS. Each is 0/1 per person: 1 = at least 2 coded events mapped to the phecode (PheTK 0.3.6's
default), and a single event counts as not recorded. Every label is the same in every coding year: only conditions
whose codes no official change moves in or out are kept, checked against the CDC cumulative ICD-10-CM conversion
table (FY2026, every change from October 2016) and the 2018 ICD-9-CM <-> ICD-10-CM GEMs, up to the data cutoff of
January 1, 2025.
  A  threshold matches   GAD MB_288.3 (GAD-7 >= 10), ADHD MB_304 (ASRS >= 4 of 6)
  B  named disorders     PTSD MB_290.1, panic disorder MB_288.2 (named in ICD-10-CM chapter 5, >= 1,000 cases)
  C  physical            hypercholesterolemia EM_239.1, asthma RE_475, breast cancer CA_105, prostate cancer
                         CA_107.2 (4 of the 6 of Biji et al. 2026)
  Left out, because their labels change with the coding year: MDD, suicidal ideation/attempt/self-harm,
  nicotine dependence, bipolar disorder, alcohol use disorder, dysthymic disorder, ischemic heart disease, chronic
  kidney disease; and every broad category.

THE MAP. phecodeX X1.0 as PheTK ships it, minus 3 rows (MAP_REMOVALS), each the one code that made its label change
with the coding year:
  RE_475   ICD-9-CM 493.21  chronic obstructive asthma with status asthmaticus (its 2 siblings and its ICD-10-CM
                            counterpart J44.0 are COPD only)
  CA_105   ICD-10-CM Z86.000  personal history of in-situ neoplasm of breast (no ICD-9-CM counterpart)
  CA_107.2 ICD-10-CM R97.21  rising PSA following treatment for prostate cancer (before October 2016, R97.2,
                            elevated PSA, outside the label)
phecode_counts_jno.py writes that map and has PheTK count with it.

THE POPULATIONS, the same for the deciles (Figure 2) and the semantic groups (Figure 4):
  every condition   participants with at least 1 ICD-coded event in PheTK's extraction
  breast cancer     women only (SEX_ONLY)       prostate cancer   men only
  3 conditions with a test (TESTED_BY): both the 1s and the 0s are restricted to participants whose record shows
  the test at any time, so a 0 means "tested, no diagnosis recorded":
  hypercholesterolemia  a cholesterol measurement or lipid panel (no triglyceride-only test)
  breast cancer         any mammogram, screening or diagnostic, under the codes before and after the 2017-2018
                        change (CPT 77055-77057 -> 77065-77067; Medicare G0202/G0204/G0206), tomosynthesis, or a
                        breast-screening encounter
  prostate cancer       a BLOOD PSA test only: explicit serum, plasma and dried-blood-spot
                        LOINC codes, the PSA CPT/HCPCS codes, or a prostate-screening encounter
The tests are searched by concept id in procedure_occurrence, measurement, observation and condition_occurrence,
standard and source columns alike, and the concepts found are written out.

OUTPUTS of `python3 conditions_jno.py`
  screen_out/test_checks_jno_local.csv            person level, stays on the Workbench (person, kind, first date,
                                                  number of dates)
  screen_out/condition_tests_jno/condition_tests_concepts.csv   the concepts found (no people)
  screen_out/condition_tests_jno/condition_populations.csv      per condition: people with records, the sex
                                                  restriction, tested, the population, cases and non-cases, and the
                                                  cases left out for having no test in the record
  screen_out/condition_tests_jno/condition_tests_meta.json
  screen_out/condition_tests_jno_<CDR>.zip        those 3, checked by pii_check (counts as they are)

Run (after phecode_counts_jno.py):  python3 conditions_jno.py        (--refresh pulls the tests again)
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "screen_out")
PHETK = os.path.join(OUT_DIR, "phetk")
MAP_JNO = os.path.join(PHETK, "phecodeX_jno.csv")                         # phecode_counts_jno.py writes it
COUNTS_JNO = os.path.join(PHETK, "phecode_counts_cohort_jno_local.tsv")   # person level, stays
PERSON = os.path.join(PHETK, "icd_person_summary_local.csv")              # phecode_coverage.py's, person level
TEST_CHECKS = os.path.join(OUT_DIR, "test_checks_jno_local.csv")          # person level, stays
OUT = os.path.join(OUT_DIR, "condition_tests_jno")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
MIN_COUNT = 2                         # PheTK's default min_phecode_count
FEMALE, MALE = 0.0, 1.0               # build_responses_v9.py: female = 0, male = 1
MAX_GB = float(sys.argv[sys.argv.index("--max-gb") + 1]) if "--max-gb" in sys.argv else 400.0

# key, phecode, label, list, men only (the 5-field form every phecode script unpacks)
CONDITIONS = [
    ("gad", "MB_288.3", "Generalized anxiety disorder", "A", False),
    ("adhd", "MB_304", "Attention-deficit hyperactivity disorder", "A", False),
    ("ptsd", "MB_290.1", "Posttraumatic stress disorder", "B", False),
    ("panic", "MB_288.2", "Panic disorder", "B", False),
    ("hyperchol", "EM_239.1", "Hypercholesterolemia", "C", False),
    ("asthma", "RE_475", "Asthma", "C", False),
    ("breast", "CA_105", "Breast cancer", "C", False),
    ("prostate", "CA_107.2", "Prostate cancer", "C", True),
]
KEYS = [c[0] for c in CONDITIONS]
CODE = {c[0]: c[1] for c in CONDITIONS}
SEX_ONLY = {"breast": FEMALE, "prostate": MALE}
TESTED_BY = {"hyperchol": "lipid", "breast": "breast", "prostate": "prostate"}
# (phecode, ICD, flag): the rows removed from phecodeX X1.0; each must be there exactly once
MAP_REMOVALS = [("RE_475", "493.21", 9), ("CA_105", "Z86.000", 10), ("CA_107.2", "R97.21", 10)]
# the labels that neither a removal nor a population change touches: for these, the conditions_* scripts must
# reproduce the phecode_* scripts' files exactly (a check that the pipeline is the same where the labels are)
UNCHANGED = ["gad", "adhd", "ptsd"]

TESTS = {
    "lipid": {                                                           # cholesterol measurement or lipid panel
        "loinc": ["2093-3",       # Cholesterol [Mass/volume] in Serum or Plasma
                  "2085-9",       # Cholesterol in HDL
                  "13457-7",      # Cholesterol in LDL, by calculation
                  "18262-6",      # Cholesterol in LDL, direct assay
                  "2089-1",       # Cholesterol in LDL
                  "43396-1",      # Cholesterol non HDL
                  "9830-1",       # Cholesterol.total/Cholesterol in HDL
                  "24331-1",      # Lipid 1996 panel
                  "57698-3"],     # Lipid panel with direct LDL
        "proc": ["80061",         # Lipid panel
                 "82465",         # Cholesterol, serum or whole blood, total
                 "83718",         # HDL cholesterol
                 "83721"],        # LDL cholesterol
        "icd": []},
    "breast": {                                                          # any mammogram, or a screening encounter
        "loinc": [],
        "proc": ["77067", "77065", "77066",            # 2017 on: screening, diagnostic unilateral, bilateral
                 "77057", "77056", "77055",            # to 2016
                 "77063", "77061", "77062",            # tomosynthesis
                 "G0202", "G0204", "G0206"],           # Medicare, through 2017
        "icd": ["Z12.31", "Z12.39", "V76.10", "V76.11", "V76.12", "V76.19"]},
    "prostate": {                                                        # a blood PSA test only
        "loinc": ["2857-1",       # Prostate specific Ag [Mass/volume] in Serum or Plasma
                  "19195-7",      # [Units/volume] in Serum or Plasma
                  "19197-3",      # [Moles/volume] in Serum or Plasma
                  "35741-8",      # in Serum or Plasma by detection limit <= 0.01 ng/mL
                  "83112-3",      # in Serum or Plasma by Immunoassay
                  "10886-0",      # Free, in Serum or Plasma
                  "19201-3",      # Free [Units/volume] in Serum or Plasma
                  "19203-9",      # Free [Moles/volume] in Serum or Plasma
                  "83113-1",      # Free, in Serum or Plasma by Immunoassay
                  "12841-3",      # Free/total in Serum or Plasma
                  "14120-0",      # Free/total in Serum or Plasma (deprecated code, kept for older records)
                  "72576-2",      # Free/total [Pure mass fraction] in Serum or Plasma
                  "33667-7",      # protein bound (complexed) in Serum or Plasma
                  "53764-7",      # Prostate specific Ag panel, Serum or Plasma
                  "100716-0"],    # in dried blood spot
        "proc": ["84153", "84152", "84154", "G0103"],  # PSA total, complexed, free; Medicare PSA screening
        "icd": ["Z12.5", "V76.44"]},
}


def with_sex(key):
    """Whether a condition's models carry a sex term (not where its population is one sex)."""
    return key not in SEX_ONLY


def pull_tests(idx):
    """Per person and kind (lipid, breast, prostate): whether the record shows the test, its first date and its
    number of dates. BigQuery once, cached person level. Returns (checks, concepts found or None)."""
    if os.path.exists(TEST_CHECKS) and "--refresh" not in sys.argv:
        T = pd.read_csv(TEST_CHECKS, dtype={ID_COL: str})
        print(f"  [tests: read from {TEST_CHECKS}, nothing pulled]", flush=True)
        return T, None
    from google.cloud import bigquery
    client = bigquery.Client()
    parts, params = [], []
    for kind, c in TESTS.items():
        where = []
        if c["loinc"]:
            where.append(f"(vocabulary_id = 'LOINC' AND concept_code IN UNNEST(@{kind}_loinc))")
            params.append(bigquery.ArrayQueryParameter(f"{kind}_loinc", "STRING", c["loinc"]))
        if c["proc"]:
            where.append(f"(vocabulary_id IN ('CPT4', 'HCPCS') AND concept_code IN UNNEST(@{kind}_proc))")
            params.append(bigquery.ArrayQueryParameter(f"{kind}_proc", "STRING", c["proc"]))
        if c["icd"]:
            where.append(f"(vocabulary_id IN ('ICD10CM', 'ICD9CM') AND concept_code IN UNNEST(@{kind}_icd))")
            params.append(bigquery.ArrayQueryParameter(f"{kind}_icd", "STRING", c["icd"]))
        parts.append(f"SELECT concept_id, concept_code, vocabulary_id, concept_name, '{kind}' AS kind "
                     f"FROM `{CDR}.concept` WHERE " + " OR ".join(where))
    codes_q = " UNION ALL ".join(parts)
    cfg = lambda p, dry: bigquery.QueryJobConfig(query_parameters=p, dry_run=dry, use_query_cache=not dry)
    concepts = client.query(codes_q, job_config=cfg(params, False)).to_dataframe()
    print("  [tests: concepts found: " + ", ".join(f"{k} {int((concepts.kind == k).sum())}" for k in TESTS) + "]",
          flush=True)
    for kind, c in TESTS.items():                   # every listed code must exist in the CDR's concept table
        found = set(concepts.loc[concepts.kind == kind, "concept_code"].astype(str))
        absent = [x for x in c["loinc"] + c["proc"] + c["icd"] if x not in found]
        print(f"  [tests: {kind}: listed codes not in the concept table: {absent or 'none'}]", flush=True)
    p_all = params + [bigquery.ArrayQueryParameter("ids", "INT64", [int(i) for i in idx])]
    tables = [("procedure_occurrence", "procedure_date", "procedure"), ("measurement", "measurement_date", "measurement"),
              ("observation", "observation_date", "observation"),
              ("condition_occurrence", "condition_start_date", "condition")]
    ev = " UNION ALL ".join(
        f"SELECT person_id, {d} AS d, {p}_{col}concept_id AS c FROM `{CDR}.{t}` WHERE person_id IN UNNEST(@ids)"
        for t, d, p in tables for col in ("", "source_"))
    ev_sql = f"""
      WITH codes AS ({codes_q}),
      ev AS ({ev})
      SELECT ev.person_id, codes.kind, MIN(ev.d) AS first_date, COUNT(DISTINCT ev.d) AS n_dates
      FROM ev JOIN (SELECT DISTINCT concept_id, kind FROM codes) AS codes ON ev.c = codes.concept_id
      GROUP BY ev.person_id, codes.kind"""
    gb = client.query(ev_sql, job_config=cfg(p_all, True)).total_bytes_processed / 1e9
    print(f"  [tests: scanning {gb:,.1f} GB]", flush=True)
    assert gb <= MAX_GB, f"the scan would read {gb:,.0f} GB, above --max-gb {MAX_GB:,.0f}; rerun with a larger --max-gb"
    T = client.query(ev_sql, job_config=cfg(p_all, False)).to_dataframe().rename(columns={"person_id": ID_COL})
    T[ID_COL] = T[ID_COL].astype(str)
    T.to_csv(TEST_CHECKS, index=False)
    print(f"  [tests: cached to {TEST_CHECKS} -- person level, do not download]", flush=True)
    return T, concepts


def load(idx):
    """Per person, in idx order: age, sex, has_rec, tested_<kind>, and per condition den_<key> (its population),
    y_<key> (0/1 inside the population, NaN outside) and cases_any_sex_<key> (count >= 2 among everyone with
    records, both sexes, tested or not: the form of phecode_counts_jno.py's table)."""
    from wb_config import CFG
    R = pd.read_csv(CFG.responses_csv, usecols=[ID_COL, "age", "sex"])
    R[ID_COL] = R[ID_COL].astype(str)
    R = R.set_index(ID_COL).reindex(idx)
    assert R.notna().all().all(), "age or sex missing for someone in the cohort"
    assert set(np.unique(R["sex"])) <= {FEMALE, MALE}, "sex is not coded 0/1 as build_responses_v9.py writes it"
    for f, how in ((COUNTS_JNO, "~/phetk_env/bin/python phecode_counts_jno.py"),
                   (PERSON, "~/phetk_env/bin/python phecode_coverage.py"),
                   (TEST_CHECKS, "python3 conditions_jno.py")):
        assert os.path.exists(f), f"{f} is missing: run `{how}` first"
    P = pd.read_csv(PERSON, dtype={"person_id": str}).set_index("person_id").reindex(idx)
    has_rec = P["n_events"].fillna(0).values > 0
    C = pd.read_csv(COUNTS_JNO, sep="\t", dtype={"person_id": str, "phecode": str},
                    usecols=["person_id", "phecode", "count"])
    C = C[C["phecode"].isin(CODE.values()) & C["person_id"].isin(idx)]
    T = pd.read_csv(TEST_CHECKS, dtype={ID_COL: str})
    A = {"age": R["age"].values.astype(float), "sex": R["sex"].values.astype(float), "has_rec": has_rec}
    for kind in TESTS:
        A[f"tested_{kind}"] = idx.isin(T.loc[T["kind"] == kind, ID_COL])
    for key, code, *_ in CONDITIONS:
        cnt = C[C["phecode"] == code].set_index("person_id")["count"].reindex(idx).fillna(0).values
        den = has_rec.copy()
        if key in SEX_ONLY:
            den &= A["sex"] == SEX_ONLY[key]
        if key in TESTED_BY:
            den &= A[f"tested_{TESTED_BY[key]}"]
        A[f"den_{key}"] = den
        A[f"cases_any_sex_{key}"] = int((has_rec & (cnt >= MIN_COUNT)).sum())
        A[f"y_{key}"] = np.where(den, (cnt >= MIN_COUNT).astype(float), np.nan)
        A[f"case_rec_sex_{key}"] = has_rec & (cnt >= MIN_COUNT) & (
            (A["sex"] == SEX_ONLY[key]) if key in SEX_ONLY else True)
    return A


def pii_check(path):
    """(ok, reason): no person-level column and no written index (counts are written as they are)."""
    from zip_screen_aggregates import ID_COLS
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


def zip_out(out_dir, files, name):
    for fn in files:
        ok, why = pii_check(os.path.join(out_dir, fn))
        assert ok, f"{fn} refused: {why}"
    zpath = os.path.join(OUT_DIR, f"{name}_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(out_dir, fn), arcname=f"{os.path.basename(out_dir)}/{fn}")
    return zpath


def main():
    sys.path.insert(0, HERE)
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)
    T, concepts = pull_tests(idx)
    A = load(idx)
    rows = []
    for key, code, label, lst, _ in CONDITIONS:
        sex_ok = (A["sex"] == SEX_ONLY[key]) if key in SEX_ONLY else np.ones(len(idx), bool)
        rec_sex = A["has_rec"] & sex_ok
        den = A[f"den_{key}"]
        y = A[f"y_{key}"]
        case_rec_sex = A[f"case_rec_sex_{key}"]
        rows.append({"condition": key, "phecode": code, "label": label, "list": lst,
                     "sex": {FEMALE: "women", MALE: "men"}.get(SEX_ONLY.get(key), "both"),
                     "test": TESTED_BY.get(key, ""),
                     "n_records_sex": int(rec_sex.sum()), "cases_records_sex": int(case_rec_sex.sum()),
                     "n_population": int(den.sum()), "n_cases": int(np.nansum(y[den])),
                     "n_noncases": int((y[den] == 0).sum()),
                     "cases_without_test": int((case_rec_sex & ~den).sum())})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "condition_populations.csv"), index=False)
    files = ["condition_populations.csv", "condition_tests_meta.json"]
    if concepts is not None:
        concepts[["kind", "vocabulary_id", "concept_code", "concept_name"]].drop_duplicates() \
            .sort_values(["kind", "vocabulary_id", "concept_code"]).to_csv(
            os.path.join(OUT, "condition_tests_concepts.csv"), index=False)
    if os.path.exists(os.path.join(OUT, "condition_tests_concepts.csv")):
        files.insert(0, "condition_tests_concepts.csv")
    tested = {k: {"cohort": int(A[f"tested_{k}"].sum()), "with_records": int((A[f"tested_{k}"] & A["has_rec"]).sum())}
              for k in TESTS}
    json.dump({"cdr": CDR, "tests": TESTS, "tested_people": tested, "sex_only": {k: v for k, v in SEX_ONLY.items()},
               "tested_by": TESTED_BY, "case_rule": f"PheTK count >= {MIN_COUNT} (phecodeX X1.0 minus MAP_REMOVALS)",
               "map_removals": MAP_REMOVALS, "sex_coding": "female = 0, male = 1",
               "tables_searched": "procedure_occurrence, measurement, observation, condition_occurrence; standard and "
                                  "source concept ids", "counts": "written as they are; pii_check"},
              open(os.path.join(OUT, "condition_tests_meta.json"), "w"), indent=2)
    print(f"[conditions_jno] cohort {len(idx):,} | with at least 1 ICD event {int(A['has_rec'].sum()):,}")
    for k, v in tested.items():
        print(f"  tested, {k}: {v['cohort']:,} in the cohort, {v['with_records']:,} with records")
    for r in D.itertuples():
        print(f"  {r.list} {r.label[:32]:32s} population {r.n_population:>6,} | cases {r.n_cases:>6,} | "
              f"non-cases {r.n_noncases:>6,}" + (f" | cases without the test {r.cases_without_test:,}" if r.test else ""))
    zpath = zip_out(OUT, files, "condition_tests_jno")
    print(f"\n[conditions_jno] done\n  download -> {zpath}")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
