#!/usr/bin/env python3
"""build_responses_v9.py -- the response matrix for one battery of items on CDR v9 (Methods; eMethods 2).

The battery is whatever item table sits next to this script, or is named with --items. Nothing in the
script assumes how many items there are. Each battery's run writes to its own folder.

WHAT IT DOES
  A. pull every item's answers from `observation` (latest answer per person and item); flag sentinels
     (Skip / Prefer Not To Answer / Don't Know are missing); map answer codes to numeric values through
     the encoding map. Cohort = persons with a record for EVERY item of the battery. Print the whole
     missingness distribution over those persons (this is the number that sets the tolerance) and write
     it to missingness_distribution.csv. Age at each person's own anchor (median answer date); sex at
     birth binary; persons without age or binary sex dropped, with the accounting. Writes
     responses_master.csv (NaN kept, missing_count), demographics_all.csv, dropped_demographics.csv,
     pull_summary.json. If responses_master.csv already exists in the output folder, part A is skipped
     and the matrix is read from it (BUILD_RESUME=0 forces a fresh pull).
  B. impute: rescale each item to [-1, 1] on its scale range, fit KNNImputer(k=K_NEIGH) on the complete
     cases, transform the rows with 1..K_MAX missing, invert the rescaling so the files stay in codebook
     units (fractional where imputed).

TOLERANCE. K_MAX is the most missing items a person may have and still enter the cohort of record. It is
set as a share of the battery, 3.3%. Unless --kmax or BUILD_KMAX is given, that share is applied to
whatever battery is being built, rounded to the nearest whole item, so 41 items give 1. Read the
printed distribution before accepting it.

OUTPUT FILE NAMES are an interface, not a description: the downstream scripts (screen_battery.py,
cutoffs.py, exemplar_encoder.py, ...) read responses_k5.csv as "the cohort of record" and responses_k0.csv
as "the complete cases" regardless of the tolerance used. The tolerance actually applied is in
run_config.json and cohort_sizes.csv.
       responses_k0.csv        complete cases: id, age, sex, every item                (sensitivity)
       responses_k5.csv        cohort of record (at most K_MAX missing), imputed, no NaN   (the screen reads this)
       imputed_mask_k5.csv     id + one boolean per item, True where a value was imputed
       responses_master.csv    everyone with a record for every item, NaN kept, missing_count
       weights_k0/k5.csv       all ones, positional
       missingness_distribution.csv, demographics_all.csv, dropped_demographics.csv, cohort_sizes.csv,
       pull_summary.json, run_config.json

HOW TO POINT IT AT A BATTERY. Command line beats environment beats discovery:
       python3 build_responses_v9.py                                  # the one *items*.csv and encoding_map*.csv beside it
       python3 build_responses_v9.py --items X.csv --emap Y.csv --out DIR --kmax N
   There is deliberately NO default item table. If more than one candidate file is beside the script,
   it stops and lists them rather than guess.

MEMORY. The pull frames are freed before the imputer runs, the master matrix is written BEFORE the
imputation, and the imputer transforms the receivers in chunks of IMPUTE_CHUNK rows (identical result).
"""
import json, os, re, gc, sys, glob, datetime
import numpy as np, pandas as pd
from sklearn.impute import KNNImputer
from sklearn import config_context

# ----------------------------------------------------------------------------- CONFIG
CDR          = os.environ["WORKSPACE_CDR"]
HERE         = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def _arg(flag, env, default=None):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return os.environ.get(env) or default


def _only(pattern, what):
    """The one file beside this script matching the pattern; refuse when that is ambiguous."""
    hits = sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, pattern)))
    if len(hits) == 1:
        return os.path.join(HERE, hits[0])
    listing = ", ".join(sorted(x for x in os.listdir(HERE) if x.endswith(".csv"))) or "none"
    raise SystemExit(f"cannot tell which {what} to use: {len(hits)} files match {pattern} in {HERE} "
                     f"({', '.join(hits) or 'none'}).\n  Name it:  python3 build_responses_v9.py --items <items.csv> "
                     f"--emap <map.csv>\n  CSV files here: {listing}")


ITEMS_PATH   = _arg("--items", "BUILD_ITEMS") or _only("*items*.csv", "item table")
EMAP_PATH    = _arg("--emap",  "BUILD_EMAP")  or _only("encoding_map*.csv", "encoding map")
_stem        = os.path.basename(ITEMS_PATH).replace(".csv", "")
OUT          = _arg("--out", "BUILD_OUT") or os.path.join(HERE, f"responses_out_{_stem}")  # beside the scripts; wb_config.py uses the same rule
MASTER       = os.path.join(OUT, "responses_master.csv")
RESUME       = os.environ.get("BUILD_RESUME", "1") == "1" and os.path.exists(MASTER)
K_NEIGH      = 5            # KNNImputer default, uniform weights, nan-Euclidean
D4_MISSING_SHARE = 0.033    # the tolerance as a share of the battery; K_MAX defaults to this share of N_ITEMS
IMPUTE_CHUNK = 1000         # receivers per transform call (memory only, no effect on the result)
WORKING_MB   = 256          # sklearn working memory for the distance chunks
SENT_RE      = r"skip|prefernot|dont"
SENT_CID     = 2000000010
SEX_VARIABLE = "sex_at_birth"
AGE_PLAUSIBLE = (18, 100)
ID_OUT       = "id"
SEX_SOURCE_PREFIX_RE = r"^(sexatbirth|biologicalsexatbirth|sex|gender|genderidentity)_"
SEX_AT_BIRTH_ENC = {"male": 1.0, "female": 0.0}
DEMO_NONANSWER_EXACT = {"", "nan", "none", "no matching concept", "unknown",
                        "not specified", "none of these", "not specified or unknown"}
TEXT_ALIAS = {"excllent": "excellent"}
REC_CODE_TO_RESPONSE = {"ehhwb_64": "Yes, within the last 12 months",
                        "ehhwb_65": "Yes, but not in the last 12 months", "ehhwb_61": "Never"}
CODE_TO_NAN = {(40192470, "SDOH_58")}      # one stray "Some days" on religious attendance -> missing

def norm(s): return re.sub(r"\s+", " ", str(s).strip().lower())

items = pd.read_csv(ITEMS_PATH)
assert items["concept_id"].is_unique and len(items) > 0, ITEMS_PATH
N_ITEMS    = len(items)
cid2item   = dict(zip(items["concept_id"].astype(int), items["item"]))
item_order = items["item"].tolist()
ids        = items["concept_id"].astype(int).tolist()
K_MAX      = int(_arg("--kmax", "BUILD_KMAX") or max(1, round(D4_MISSING_SHARE * N_ITEMS)))
print(f"[config] battery  {ITEMS_PATH}  ({N_ITEMS} items)\n[config] encoding {EMAP_PATH}\n[config] out      {OUT}\n"
      f"[config] cohort of record = at most {K_MAX} missing of {N_ITEMS}"
      + ("" if _arg("--kmax", "BUILD_KMAX") else f"  (the cohort rule's missing-item share, {D4_MISSING_SHARE:.3f}, applied to this battery)"), flush=True)
os.makedirs(OUT, exist_ok=True)

# ============================================================================= PART A: pull -> master
if RESUME:
    print(f"[resume] {MASTER} exists; reading it and skipping the pull (BUILD_RESUME=0 to force a pull)")
    mat = pd.read_csv(MASTER).set_index(ID_OUT)
    assert list(mat.columns[:len(item_order)]) == item_order and {"missing_count", "age", "sex"} <= set(mat.columns)
    n_full = json.load(open(os.path.join(OUT, "pull_summary.json")))["n_full_record"]
    print(f"[resume] {len(mat):,} persons with a full record and demographics (full record before demographics: {n_full:,})")
else:
    from google.cloud import bigquery
    client = bigquery.Client()

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
    print(f"[pull] {len(resp):,} rows | {resp['cid'].nunique()} of {N_ITEMS} items | sentinel rows {int(resp.is_sentinel.sum()):,}", flush=True)

    distinct = Q(f"""SELECT DISTINCT o.observation_source_concept_id AS cid, CAST(o.value_source_value AS STRING) AS vsv,
                            c.concept_name AS answer_label
                     FROM `{CDR}.observation` o LEFT JOIN `{CDR}.concept` c ON c.concept_id = o.value_source_concept_id
                     WHERE o.observation_source_concept_id IN UNNEST(@ids)""", ids)

    # ------------------------------------------------------------------------- code -> value
    emap = pd.read_csv(EMAP_PATH)
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
        if key in code_override: return code_override[key]
        a = str(a); cands = [a] + ([a.rsplit(": ", 1)[1]] if ": " in a else [])
        for x in cands:
            val = lut.get((int(c), TEXT_ALIAS.get(norm(x), norm(x))))
            if val is not None: return val
        return np.nan
    d["numeric"] = [resolve(c, v, a) for c, v, a in zip(d["cid"], d["vsv"], d["answer_label"])]
    bad = d[(~d["is_sentinel"]) & (~d["route_nan"]) & d["numeric"].isna()]
    if len(bad):
        bad.to_csv(os.path.join(OUT, "unmapped_answer_codes.csv"), index=False)
        raise SystemExit(f"[ERROR] {len(bad)} non-sentinel answer codes did not map:\n" + bad.to_string())
    code_value = {(int(r.cid), str(r.vsv)): float(r.numeric) for r in d.itertuples() if pd.notna(r.numeric) and not r.route_nan}
    print(f"[map] {len(code_value)} substantive codes, {int(d.is_sentinel.sum())} sentinel codes, {int(d.route_nan.sum())} routed to missing", flush=True)

    # ------------------------------------------------------------------------- matrix, everyone with a record for every item
    resp["item"] = resp["cid"].astype(int).map(cid2item)
    resp["numeric"] = [code_value.get((int(c), str(v)), np.nan) for c, v in zip(resp["cid"], resp["vsv"])]
    latest = resp[["person_id", "item", "numeric", "odt"]].sort_values("odt").drop_duplicates(["person_id", "item"], keep="last")
    del resp, distinct, d; gc.collect()
    n_rec = latest.groupby("person_id")["item"].nunique()
    full = set(n_rec.index[n_rec == N_ITEMS])
    n_full = len(full)
    print(f"[cohort] persons with a record for all {len(item_order)}: {n_full:,}", flush=True)
    lat = latest[latest.person_id.isin(full)].copy()
    del latest; gc.collect()
    mat = lat.pivot(index="person_id", columns="item", values="numeric").reindex(columns=item_order)
    mat["missing_count"] = mat[item_order].isna().sum(axis=1)
    vc = mat["missing_count"].value_counts().sort_index()
    cum = vc.cumsum()
    print(f"[cohort] missingness over the {len(item_order)} items, among the {len(mat):,} with a record for all of them:", flush=True)
    for m_, c_ in vc.items():
        if m_ <= 10:
            print(f"           missing {int(m_):2d}: {int(c_):8,}   cumulative <= {int(m_):2d}: {int(cum[m_]):8,}", flush=True)
    print(f"           missing > 10: {int(vc[vc.index > 10].sum()):,}", flush=True)
    pd.DataFrame({"missing_count": vc.index, "n_persons": vc.values, "cumulative": cum.values}).to_csv(
        os.path.join(OUT, "missingness_distribution.csv"), index=False)
    print(f"[cohort] operating rule: at most {K_MAX} missing -> {int((mat.missing_count <= K_MAX).sum()):,}", flush=True)

    # ------------------------------------------------------------------------- demographics
    odt = pd.to_datetime(lat["odt"], errors="coerce", utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    o = pd.DataFrame({"person_id": lat["person_id"].values, "odt": odt.values})
    del lat, odt; gc.collect()
    o["ns"] = o["odt"].astype("int64")
    anchor = pd.to_datetime(o.groupby("person_id")["ns"].median().round().astype("int64"))
    span = (o.groupby("person_id")["odt"].max() - o.groupby("person_id")["odt"].min()).dt.days
    del o; gc.collect()
    print(f"[demo] span of a person's {N_ITEMS} answers, days: median {span.median():.0f}, p90 {span.quantile(.9):.0f}", flush=True)
    person = Q(f"""SELECT person_id, birth_datetime, year_of_birth, {SEX_VARIABLE}_concept_id AS sex_concept_id,
                          CAST({SEX_VARIABLE}_source_value AS STRING) AS sex_source FROM `{CDR}.person`""")
    p = person.drop_duplicates("person_id").set_index("person_id")
    birth = pd.to_datetime(p["birth_datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    yob = pd.to_datetime(p["year_of_birth"].astype("Int64").astype(str).where(p["year_of_birth"].notna()) + "-07-01", errors="coerce")
    p["birth"] = birth.fillna(yob).astype("datetime64[ns]")
    p["age"] = ((anchor.reindex(p.index) - p["birth"]).dt.days / 365.25).round(1)
    low = p["sex_source"].astype(str).str.strip().str.lower()
    tok = low.str.replace(SEX_SOURCE_PREFIX_RE, "", regex=True)
    p["sex"] = tok.map(SEX_AT_BIRTH_ENC)
    nonans = low.str.startswith("pmi") | low.str.contains(SENT_RE, na=False, regex=True) | low.str.contains("prefer not", na=False) | tok.isin(DEMO_NONANSWER_EXACT)
    p["sex_reason"] = np.where(p["sex"].notna(), "ok", np.where(nonans, "nonanswer", "unencodable"))
    demo = p.reindex(mat.index)
    keep = demo["age"].notna() & demo["sex_reason"].eq("ok")
    print("[demo] drop accounting on the full-record cohort:")
    print(f"   no person row {int(demo['age'].isna().sum() - (demo['age'].isna() & demo.index.isin(p.index)).sum()):,} | "
          f"age missing {int((demo['age'].isna() & demo.index.isin(p.index)).sum()):,} | "
          f"sex non-answer {int(demo['sex_reason'].eq('nonanswer').sum()):,} | sex unencodable {int(demo['sex_reason'].eq('unencodable').sum()):,}")
    demo[~keep].reset_index().to_csv(os.path.join(OUT, "dropped_demographics.csv"), index=False)
    mat = mat[keep.values].copy()
    mat["age"] = demo.loc[mat.index, "age"].values
    mat["sex"] = demo.loc[mat.index, "sex"].values
    print(f"[demo] kept {len(mat):,} with age and binary sex at birth; ages {mat.age.min():.1f} to {mat.age.max():.1f}", flush=True)
    demo.loc[mat.index, ["age", "sex_source", "sex", "sex_reason"]].rename_axis(ID_OUT).reset_index().to_csv(
        os.path.join(OUT, "demographics_all.csv"), index=False)
    del person, p, demo, span, anchor; gc.collect()

    # ------------------------------------------------------------------------- the master matrix, written BEFORE the imputation
    mat.rename_axis(ID_OUT).reset_index().to_csv(MASTER, index=False)
    json.dump({"cdr": CDR, "tier": "Registered", "date": datetime.date.today().isoformat(),
               "n_full_record": n_full, "n_full_record_with_demographics": int(len(mat))},
              open(os.path.join(OUT, "pull_summary.json"), "w"), indent=2)
    print(f"[master] wrote {MASTER} ({len(mat):,} rows); a rerun resumes from it", flush=True)

# ============================================================================= PART B: imputation and the cohort files
scales = items.set_index("item").reindex(item_order)
lo, hi = scales["scale_min"].values.astype(float), scales["scale_max"].values.astype(float)
X = mat[item_order].values.astype(float)
U = 2.0 * (X - lo) / (hi - lo) - 1.0                      # [-1, 1], NaN where missing
complete = mat["missing_count"].values == 0
target = (mat["missing_count"].values >= 1) & (mat["missing_count"].values <= K_MAX)
rows = np.flatnonzero(target)
print(f"[impute] donors (complete cases) {int(complete.sum()):,} | receivers (1 to {K_MAX} missing) {len(rows):,} | {K_NEIGH} nearest neighbours per receiver | "
      f"chunks of {IMPUTE_CHUNK:,}", flush=True)
imp = KNNImputer(n_neighbors=K_NEIGH, weights="uniform", metric="nan_euclidean").fit(U[complete])
U_imp = U.copy()
with config_context(working_memory=WORKING_MB):
    for i0 in range(0, len(rows), IMPUTE_CHUNK):
        r = rows[i0:i0 + IMPUTE_CHUNK]
        U_imp[r] = imp.transform(U[r])
        if (i0 // IMPUTE_CHUNK) % 10 == 0 or i0 + IMPUTE_CHUNK >= len(rows):
            print(f"[impute] {min(i0 + IMPUTE_CHUNK, len(rows)):,} of {len(rows):,}", flush=True)
assert not np.isnan(U_imp[complete | target]).any()
X_imp = lo + (U_imp + 1.0) / 2.0 * (hi - lo)               # back to codebook units, fractional where imputed
mask = np.isnan(X) & (complete | target)[:, None]

# ----------------------------------------------------------------------------- outputs
COV = ["age", "sex"]
k0 = mat[complete]
k0[COV + item_order].rename_axis(ID_OUT).reset_index().to_csv(os.path.join(OUT, "responses_k0.csv"), index=False)
pd.DataFrame({"weight": np.ones(len(k0), dtype=int)}).to_csv(os.path.join(OUT, "weights_k0.csv"), index=False)
sel = complete | target
k5 = pd.DataFrame(X_imp[sel], index=mat.index[sel], columns=item_order)
k5.insert(0, "sex", mat["sex"].values[sel]); k5.insert(0, "age", mat["age"].values[sel])
k5.rename_axis(ID_OUT).reset_index().to_csv(os.path.join(OUT, "responses_k5.csv"), index=False)
pd.DataFrame(mask[sel], index=mat.index[sel], columns=item_order).rename_axis(ID_OUT).reset_index().to_csv(
    os.path.join(OUT, "imputed_mask_k5.csv"), index=False)
pd.DataFrame({"weight": np.ones(int(sel.sum()), dtype=int)}).to_csv(os.path.join(OUT, "weights_k5.csv"), index=False)
sizes = pd.DataFrame({"missing_at_most": list(range(0, 11)),
                      "n_persons": [int((mat.missing_count <= k).sum()) for k in range(0, 11)]})
sizes.to_csv(os.path.join(OUT, "cohort_sizes.csv"), index=False)
json.dump({"cdr": CDR, "tier": "Registered", "date": datetime.date.today().isoformat(),
           "n_full_record": int(n_full), "n_full_record_with_demographics": int(len(mat)),
           "n_k0": int(complete.sum()), "n_k5": int(sel.sum()), "k_max": K_MAX, "k_neighbors": K_NEIGH,
           "imputed_cells": int(mask[sel].sum()), "imputed_cells_share": float(mask[sel].sum() / mask[sel].size),
           "sex_variable": SEX_VARIABLE},
          open(os.path.join(OUT, "run_config.json"), "w"), indent=2)
print(f"[done] responses_k0 {int(complete.sum()):,} | responses_k5 {int(sel.sum()):,} | imputed cells {int(mask[sel].sum()):,} "
      f"({100 * mask[sel].sum() / mask[sel].size:.3f}% of the k5 matrix) -> {OUT}")
print(sizes.to_string(index=False))
