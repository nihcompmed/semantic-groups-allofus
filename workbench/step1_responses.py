#!/usr/bin/env python3
"""step1_responses.py -- the cohort and its answers to the 151 items (Methods, Participants; Appendix 1).

WHAT IT DOES
  1. Pulls every answer to the 151 items from `observation` and keeps each person's latest answer per item. Skips,
     "Prefer not to answer" and "Don't know" are missing. Each answer code is mapped to its numeric value through
     encoding_map_all151.csv; the script stops if any answer code does not map.
  2. Keeps everyone with a record for all 151 items, with age (at the median date of their answers) and sex at birth
     recorded as female or male. A record that is a skip counts as a record with a missing answer.
  3. The cohort: those with at most K_MAX missing answers (3.3% of the items, 5 of 151). Missing answers are imputed by
     k-nearest neighbours (5 neighbours, nan-Euclidean distance) fitted on the complete cases, on each item rescaled to
     [-1, 1], then returned to the item's own scale. Imputed answers enter the scores; they never enter a screening
     threshold (step 3).

OUTPUTS (responses/)
  cohort.csv          id, age, sex (1 male, 0 female), the 151 items         the cohort, every step reads it
  imputed_mask.csv    id, one column per item, True where imputed
  master.csv          everyone with a record for all 151 items and binary sex and age, missing answers kept
  dropped_demographics.csv, demographics_all.csv, missingness_distribution.csv
  pull_summary.json, run_config.json, cohort_sizes.csv                       the counts quoted in the Methods
DOWNLOAD  downloads/responses_counts_<release>.zip: pull_summary.json, run_config.json, cohort_sizes.csv

If responses/master.csv exists, the pull is skipped and the cohort is rebuilt from it (--refresh pulls again).

Run:  python3 step1_responses.py
"""
import datetime
import gc
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.impute import KNNImputer

from release import write_download
from settings import CDR, CODING_CSV, COHORT_CSV, ID, ITEMS_CSV, MASK_CSV, RESPONSES, download_dir

MASTER = os.path.join(RESPONSES, "master.csv")
K_NEIGH = 5                 # KNNImputer's default, uniform weights, nan-Euclidean distance
MISSING_SHARE = 0.033       # the cohort's tolerance, as a share of the items
IMPUTE_CHUNK = 1000         # rows per transform call (memory only; the result is the same)
WORKING_MB = 256
SENT_RE = r"skip|prefernot|dont"
SENT_CID = 2000000010
SEX_VARIABLE = "sex_at_birth"
SEX_SOURCE_PREFIX_RE = r"^(sexatbirth|biologicalsexatbirth|sex|gender|genderidentity)_"
SEX_AT_BIRTH_ENC = {"male": 1.0, "female": 0.0}
DEMO_NONANSWER_EXACT = {"", "nan", "none", "no matching concept", "unknown", "not specified", "none of these",
                        "not specified or unknown"}
TEXT_ALIAS = {"excllent": "excellent"}
REC_CODE_TO_RESPONSE = {"ehhwb_64": "Yes, within the last 12 months", "ehhwb_65": "Yes, but not in the last 12 months",
                        "ehhwb_61": "Never"}
CODE_TO_NAN = {(40192470, "SDOH_58")}      # one stray "Some days" on religious attendance -> missing


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def pull(items, item_order, N_ITEMS):
    """Parts 1 and 2: the answers, the coding, everyone with a full record, age and sex. Writes master.csv."""
    from google.cloud import bigquery
    client = bigquery.Client()
    cid2item = dict(zip(items["concept_id"].astype(int), items["item"]))
    ids = items["concept_id"].astype(int).tolist()

    def Q(sql, ids=None):
        cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("ids", "INT64", ids)] if ids else [])
        return client.query(sql, job_config=cfg).result().to_dataframe()

    print(f"[pull] observation rows for the {N_ITEMS} items ...", flush=True)
    resp = Q(f"""SELECT o.person_id, o.observation_source_concept_id AS cid,
                        CAST(o.value_source_value AS STRING) AS vsv, o.value_as_concept_id AS vcid,
                        o.observation_datetime AS odt
                 FROM `{CDR}.observation` o WHERE o.observation_source_concept_id IN UNNEST(@ids)""", ids)
    s = resp["vsv"].astype(str).str.lower()
    resp["is_sentinel"] = s.str.startswith("pmi") | s.str.contains(SENT_RE, na=False, regex=True) \
        | (pd.to_numeric(resp["vcid"], errors="coerce") == SENT_CID)
    del s
    print(f"[pull] {len(resp):,} rows | {resp['cid'].nunique()} of {N_ITEMS} items", flush=True)
    distinct = Q(f"""SELECT DISTINCT o.observation_source_concept_id AS cid, CAST(o.value_source_value AS STRING) AS vsv,
                            c.concept_name AS answer_label
                     FROM `{CDR}.observation` o LEFT JOIN `{CDR}.concept` c ON c.concept_id = o.value_source_concept_id
                     WHERE o.observation_source_concept_id IN UNNEST(@ids)""", ids)

    # answer code -> numeric value
    emap = pd.read_csv(CODING_CSV)
    lut = {(int(r.concept_id), norm(r.response)): r.numeric_value for r in emap.itertuples()}
    rec3 = emap.loc[emap["rule"] == "REC3", "concept_id"].astype(int).unique()
    code_override = {(int(c), code): lut[(int(c), norm(resp_))] for c in rec3
                     for code, resp_ in REC_CODE_TO_RESPONSE.items() if (int(c), norm(resp_)) in lut}
    d = distinct.drop_duplicates(["cid", "vsv"]).copy()
    sl = d["vsv"].astype(str).str.lower()
    d["is_sentinel"] = sl.str.startswith("pmi") | sl.str.contains(SENT_RE, na=False, regex=True)
    d["route_nan"] = [(int(c), str(v)) in CODE_TO_NAN for c, v in zip(d["cid"], d["vsv"])]

    def resolve(c, v, a):
        key = (int(c), str(v))
        if key in code_override:
            return code_override[key]
        a = str(a)
        cands = [a] + ([a.rsplit(": ", 1)[1]] if ": " in a else [])
        for x in cands:
            val = lut.get((int(c), TEXT_ALIAS.get(norm(x), norm(x))))
            if val is not None:
                return val
        return np.nan
    d["numeric"] = [resolve(c, v, a) for c, v, a in zip(d["cid"], d["vsv"], d["answer_label"])]
    bad = d[(~d["is_sentinel"]) & (~d["route_nan"]) & d["numeric"].isna()]
    if len(bad):
        bad.to_csv(os.path.join(RESPONSES, "unmapped_answer_codes.csv"), index=False)
        raise SystemExit(f"{len(bad)} answer codes did not map:\n" + bad.to_string())
    code_value = {(int(r.cid), str(r.vsv)): float(r.numeric) for r in d.itertuples()
                  if pd.notna(r.numeric) and not r.route_nan}

    # everyone with a record for every item, their latest answer per item
    resp["item"] = resp["cid"].astype(int).map(cid2item)
    resp["numeric"] = [code_value.get((int(c), str(v)), np.nan) for c, v in zip(resp["cid"], resp["vsv"])]
    latest = resp[["person_id", "item", "numeric", "odt"]].sort_values("odt").drop_duplicates(["person_id", "item"],
                                                                                             keep="last")
    del resp, distinct, d
    gc.collect()
    n_rec = latest.groupby("person_id")["item"].nunique()
    full = set(n_rec.index[n_rec == N_ITEMS])
    n_full = len(full)
    print(f"[cohort] persons with a record for all {N_ITEMS}: {n_full:,}", flush=True)
    lat = latest[latest.person_id.isin(full)].copy()
    del latest
    gc.collect()
    mat = lat.pivot(index="person_id", columns="item", values="numeric").reindex(columns=item_order)
    mat["missing_count"] = mat[item_order].isna().sum(axis=1)
    vc = mat["missing_count"].value_counts().sort_index()
    cum = vc.cumsum()
    pd.DataFrame({"missing_count": vc.index, "n_persons": vc.values, "cumulative": cum.values}).to_csv(
        os.path.join(RESPONSES, "missingness_distribution.csv"), index=False)

    # age at the median answer date, sex at birth
    odt = pd.to_datetime(lat["odt"], errors="coerce", utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    o = pd.DataFrame({"person_id": lat["person_id"].values, "odt": odt.values})
    del lat, odt
    gc.collect()
    o["ns"] = o["odt"].astype("int64")
    anchor = pd.to_datetime(o.groupby("person_id")["ns"].median().round().astype("int64"))
    del o
    gc.collect()
    person = Q(f"""SELECT person_id, birth_datetime, year_of_birth, {SEX_VARIABLE}_concept_id AS sex_concept_id,
                          CAST({SEX_VARIABLE}_source_value AS STRING) AS sex_source FROM `{CDR}.person`""")
    p = person.drop_duplicates("person_id").set_index("person_id")
    birth = pd.to_datetime(p["birth_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    yob = pd.to_datetime(p["year_of_birth"].astype("Int64").astype(str).where(p["year_of_birth"].notna()) + "-07-01",
                         errors="coerce")
    p["birth"] = birth.fillna(yob).astype("datetime64[ns]")
    p["age"] = ((anchor.reindex(p.index) - p["birth"]).dt.days / 365.25).round(1)
    low = p["sex_source"].astype(str).str.strip().str.lower()
    tok = low.str.replace(SEX_SOURCE_PREFIX_RE, "", regex=True)
    p["sex"] = tok.map(SEX_AT_BIRTH_ENC)
    nonans = low.str.startswith("pmi") | low.str.contains(SENT_RE, na=False, regex=True) \
        | low.str.contains("prefer not", na=False) | tok.isin(DEMO_NONANSWER_EXACT)
    p["sex_reason"] = np.where(p["sex"].notna(), "ok", np.where(nonans, "nonanswer", "unencodable"))
    demo = p.reindex(mat.index)
    keep = demo["age"].notna() & demo["sex_reason"].eq("ok")
    demo[~keep].reset_index().to_csv(os.path.join(RESPONSES, "dropped_demographics.csv"), index=False)
    mat = mat[keep.values].copy()
    mat["age"] = demo.loc[mat.index, "age"].values
    mat["sex"] = demo.loc[mat.index, "sex"].values
    print(f"[demographics] kept {len(mat):,} with age and binary sex at birth", flush=True)
    demo.loc[mat.index, ["age", "sex_source", "sex", "sex_reason"]].rename_axis(ID).reset_index().to_csv(
        os.path.join(RESPONSES, "demographics_all.csv"), index=False)
    mat.rename_axis(ID).reset_index().to_csv(MASTER, index=False)
    json.dump({"cdr": CDR, "tier": "Registered", "date": datetime.date.today().isoformat(),
               "n_full_record": n_full, "n_full_record_with_demographics": int(len(mat))},
              open(os.path.join(RESPONSES, "pull_summary.json"), "w"), indent=2)
    return mat, n_full


def main():
    os.makedirs(RESPONSES, exist_ok=True)
    items = pd.read_csv(ITEMS_CSV)
    assert items["concept_id"].is_unique and len(items) > 0, ITEMS_CSV
    item_order = items["item"].tolist()
    N_ITEMS = len(items)
    K_MAX = max(1, round(MISSING_SHARE * N_ITEMS))
    if os.path.exists(MASTER) and "--refresh" not in sys.argv:
        mat = pd.read_csv(MASTER).set_index(ID)
        assert list(mat.columns[:N_ITEMS]) == item_order and {"missing_count", "age", "sex"} <= set(mat.columns)
        n_full = json.load(open(os.path.join(RESPONSES, "pull_summary.json")))["n_full_record"]
        print(f"[resume] {MASTER} exists: the pull is skipped (--refresh pulls again)", flush=True)
    else:
        mat, n_full = pull(items, item_order, N_ITEMS)

    # part 3: the cohort, imputed
    scales = items.set_index("item").reindex(item_order)
    lo, hi = scales["scale_min"].values.astype(float), scales["scale_max"].values.astype(float)
    X = mat[item_order].values.astype(float)
    U = 2.0 * (X - lo) / (hi - lo) - 1.0                                  # [-1, 1], NaN where missing
    complete_ = mat["missing_count"].values == 0
    target = (mat["missing_count"].values >= 1) & (mat["missing_count"].values <= K_MAX)
    rows = np.flatnonzero(target)
    print(f"[impute] donors (complete cases) {int(complete_.sum()):,} | receivers (1 to {K_MAX} missing) {len(rows):,}",
          flush=True)
    imp = KNNImputer(n_neighbors=K_NEIGH, weights="uniform", metric="nan_euclidean").fit(U[complete_])
    U_imp = U.copy()
    with config_context(working_memory=WORKING_MB):
        for i0 in range(0, len(rows), IMPUTE_CHUNK):
            r = rows[i0:i0 + IMPUTE_CHUNK]
            U_imp[r] = imp.transform(U[r])
    assert not np.isnan(U_imp[complete_ | target]).any()
    X_imp = lo + (U_imp + 1.0) / 2.0 * (hi - lo)                          # back to the item's own scale
    mask = np.isnan(X) & (complete_ | target)[:, None]
    sel = complete_ | target
    k5 = pd.DataFrame(X_imp[sel], index=mat.index[sel], columns=item_order)
    k5.insert(0, "sex", mat["sex"].values[sel])
    k5.insert(0, "age", mat["age"].values[sel])
    k5.rename_axis(ID).reset_index().to_csv(COHORT_CSV, index=False)
    pd.DataFrame(mask[sel], index=mat.index[sel], columns=item_order).rename_axis(ID).reset_index().to_csv(
        MASK_CSV, index=False)
    sizes = pd.DataFrame({"missing_at_most": list(range(0, 11)),
                          "n_persons": [int((mat.missing_count <= k).sum()) for k in range(0, 11)]})
    sizes.to_csv(os.path.join(RESPONSES, "cohort_sizes.csv"), index=False)
    json.dump({"cdr": CDR, "tier": "Registered", "date": datetime.date.today().isoformat(),
               "n_full_record": int(n_full), "n_full_record_with_demographics": int(len(mat)),
               "n_complete_cases": int(complete_.sum()), "n_cohort": int(sel.sum()), "max_missing": K_MAX,
               "k_neighbors": K_NEIGH, "imputed_cells": int(mask[sel].sum()),
               "imputed_cells_share": float(mask[sel].sum() / mask[sel].size), "sex_variable": SEX_VARIABLE},
              open(os.path.join(RESPONSES, "run_config.json"), "w"), indent=2)
    print(f"[done] cohort {int(sel.sum()):,} | imputed cells {int(mask[sel].sum()):,} "
          f"({100 * mask[sel].sum() / mask[sel].size:.3f}%)", flush=True)

    d = download_dir("responses_counts")
    for f in ("pull_summary.json", "run_config.json", "cohort_sizes.csv"):
        shutil.copy(os.path.join(RESPONSES, f), d)
    write_download("responses_counts", ["pull_summary.json", "run_config.json", "cohort_sizes.csv"])


if __name__ == "__main__":
    main()
