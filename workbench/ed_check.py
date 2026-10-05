#!/usr/bin/env python3
"""ed_check.py -- a check of the emergency department (ED) share's denominator (Figure 2B, eMethods 3).

THE QUESTION. The ED share's denominator is everyone in the cohort with at least 1 visit in visit_occurrence
(care.py's has_ehr, 86,365). Of them, 24,807 have no ICD-coded diagnosis in condition_occurrence or observation
(phecode_coverage.py's records cells). So "at least 1 visit" is wider than coded care. What are those people's
visits, and does the ED share by decile change when the denominator is only people with coded care?

PART A, THE VISITS. The cohort's people with at least 1 visit, in 2 groups:
  icd         at least 1 ICD event (PheTK's person summary, as phecode_coverage.py reads it)
  visit_only  no ICD event
  per group:  people; visits per person (25th, 50th, 75th percentiles); people with at least 1 ED visit (visit
              concepts 9203, 262, as care.py); the mean per-person ED share
  by SOURCE   visit_occurrence_ext.src_id with its digits removed, so a site number is never written (an EHR
              site, or a program source): people with at least 1 visit from it, people whose visits ALL come from
              it, and its share of the group's visits
  by TYPE     visit_concept_id with its concept name, for the TOP_TYPES types reaching the most people across both
              groups, the rest pooled as "other types": people with at least 1 such visit, and the type's share of
              the group's visits
  A visit is a distinct visit_occurrence_id. If visit_occurrence_ext cannot be joined, the source is written
  "not available" and the rest runs.
PART B, THE ED SHARE BY DECILE ON CODED CARE. For the 6 encoders, outcome_deciles.py's deciles and its own
functions (care_sweep.describe, care_adjusted.describe), with the denominator narrowed to people with at least 1
visit AND at least 1 ICD event; the ED reference curve (age, age^2, sex) is refitted once on that denominator.
Written per decile: n_set, n_visit (at least 1 visit, as outcome_deciles.csv's n_ehr), n_icd_visit (of them,
with an ICD event), and the ED share unadjusted and adjusted on n_icd_visit. The current denominator is
recomputed too, as a check against outcome_deciles.csv, and not written again.

CHECKS, before anything is written (each stops the run; no count is printed)
  - the pull's people with at least 1 visit are exactly care_outcomes_per_person.csv's has_ehr, and each
    person's ED share equals that file's;
  - the 4 records cells equal phecode_coverage.json's, where that file is present;
  - part B's current denominator equals outcome_deciles.csv's ED values, where that file is present.

THE DISSEMINATION POLICY (disclosure.py)
  Part A: a people count is hidden when it, or its complement in its group, is 1 to 20; a row's cells are hidden
  in BOTH groups when either is (the 2 groups partition the people with a visit, a public total); a visit share
  or mean over a hidden count is blanked. Percentiles are written; no minimum or maximum is.
  Part B: n_icd_visit is hidden when it, or n_visit minus it, or n_set minus it, is 1 to 20; across the 10
  deciles, which partition a public total, through phecode_deciles.hide_cells; every ED statistic over a hidden
  count is blanked. Plus care_sweep.describe's own within-set rules.
  Precision (disclosure.py): shares to 3 decimals; means and percentiles to 4 significant figures.
  Printed output follows the policy: only written values are printed.

INPUTS   screen_out/care_outcomes_per_person.csv, screen_out/phetk/icd_person_summary_local.csv, and everything
         outcome_deciles.py reads. Pulled once and cached (per person, stays on the Workbench, never downloaded):
         screen_out/ed_check/visits_person_local.csv  (--refresh pulls again)
OUTPUTS  screen_out/ed_check/ed_check_visits.csv     part A, 1 row per group summary / source / type
         screen_out/ed_check/ed_check_deciles.csv    part B, 1 row per encoder x decile, plus the cohort row
         screen_out/ed_check/ed_check_meta.json
         screen_out/ed_check_<CDR>.zip               those 3, checked by verdict() in zip_screen_aggregates.py

Run:  python3 ed_check.py
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
from breadth_sweep import ENCODERS                                         # noqa: E402
import care_adjusted                                                       # noqa: E402  its curve and describe()
import care_sweep                                                          # noqa: E402  its describe()
from disclosure import MARK, SUPP, coarsen, coarsen_sig, pair_hidden, small   # noqa: E402  the dissemination policy
from outcome_deciles import load                                           # noqa: E402  the same inputs and curves
from phecode_deciles import DECILES, decile_masks, hide_cells             # noqa: E402  the same deciles, the same rule
from env_versions import versions                                          # noqa: E402  library versions, for the Methods
from zip_screen_aggregates import verdict                                  # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "ed_check")
CACHE = os.path.join(OUT, "visits_person_local.csv")
PERSON = os.path.join(OUT_DIR, "phetk", "icd_person_summary_local.csv")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
ED_CONCEPTS = (9203, 262)                    # care.py: Emergency Room Visit, Emergency Room and Inpatient Visit
TOP_TYPES = 12
GROUPS = ("icd", "visit_only")
ED_COLS = ["ed_mean", "ed_lo", "ed_hi", "ed_adj", "ed_adj_lo", "ed_adj_hi"]


def pull(ids):
    """Visits per person x visit type x source, over the cohort, from the CDR."""
    from google.cloud import bigquery
    if not CDR:
        raise SystemExit("no CDR dataset: set WORKSPACE_CDR")
    client = bigquery.Client()
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("ids", "INT64", [int(i) for i in ids])])
    sql = """SELECT v.person_id, v.visit_concept_id, IFNULL(c.concept_name, 'No matching concept') AS visit_type,
                    {src} AS source, COUNT(DISTINCT v.visit_occurrence_id) AS n_visits
             FROM `{cdr}.visit_occurrence` v {join}
             LEFT JOIN `{cdr}.concept` c ON c.concept_id = v.visit_concept_id
             WHERE v.person_id IN UNNEST(@ids)
             GROUP BY 1, 2, 3, 4"""
    try:
        V = client.query(sql.format(
            cdr=CDR, src="IFNULL(TRIM(REGEXP_REPLACE(e.src_id, r'[0-9]+', '')), 'not recorded')",
            join=f"LEFT JOIN `{CDR}.visit_occurrence_ext` e ON e.visit_occurrence_id = v.visit_occurrence_id"),
            job_config=cfg).to_dataframe()
    except Exception as ex:                                                 # the source is optional
        print(f"  visit_occurrence_ext could not be joined ({type(ex).__name__}): the source is written 'not available'")
        V = client.query(sql.format(cdr=CDR, src="'not available'", join=""), job_config=cfg).to_dataframe()
    V["person_id"] = V["person_id"].astype(str)
    V.to_csv(CACHE, index=False)


def cell(c, total):
    """A people count as written: the number, or hidden when it or its complement is 1 to 20."""
    return MARK if small(c) else SUPP if pair_hidden(c, total) else int(c)


def part_a(V, groups, sizes):
    """The visits of the 2 groups (see the docstring). `groups[g]` is the set of person ids in group g."""
    rows = []
    tot = V.groupby("person_id")["n_visits"].sum()
    ed = V[V.visit_concept_id.isin(ED_CONCEPTS)].groupby("person_id")["n_visits"].sum()
    for g in GROUPS:
        p = sorted(groups[g])
        t = tot.reindex(p).fillna(0).values
        e = ed.reindex(p).fillna(0).values
        n_ed = int((e > 0).sum())
        q = np.percentile(t, [25, 50, 75])
        rows.append({"section": "group", "label": "all visits", "group": g, "people": cell(len(p), sizes["cohort"]),
                     "people_any_ed": cell(n_ed, len(p)), "visits_p25": q[0], "visits_median": q[1], "visits_p75": q[2],
                     "mean_ed_share": float(np.mean(e / t)) if not pair_hidden(n_ed, len(p)) else np.nan})
    # sources and types: people with at least 1 such visit, people whose visits are all of it, share of visits
    reach = V.groupby("visit_type")["person_id"].nunique().sort_values(ascending=False)
    top = list(reach.index[:TOP_TYPES])
    V = V.assign(type_label=np.where(V.visit_type.isin(top), V.visit_type, "other types"))
    for section, col, labels, with_only in (("source", "source", sorted(V.source.unique()), True),
                                           ("type", "type_label", top + (["other types"] if len(reach) > TOP_TYPES else []), False)):
        for lab in labels:
            per = {}
            for g in GROUPS:
                W = V[V.person_id.isin(groups[g])]
                n_g = len(groups[g])
                people = W[W[col] == lab].person_id.nunique()
                share = W.loc[W[col] == lab, "n_visits"].sum() / W["n_visits"].sum()
                r = {"people": people, "share_of_visits": share}
                if with_only:
                    s = W.groupby("person_id")[col].agg(lambda x: set(x) == {lab})
                    r["people_only"] = int(s.sum())
                per[g] = (r, n_g)
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


def part_b(idx, C, icd, B, S, SA):
    """The ED share by decile on people with a visit and an ICD event; the current denominator as a check."""
    N = len(idx)
    visit = C["has_ehr"].copy()
    C2 = dict(C)
    C2["has_ehr"] = visit & icd
    m = C2["has_ehr"]
    y = C["ed_share"]
    fitted, _ = care_adjusted.fit_curve(C["age"][m], C["sex"][m], y[m])
    C2["gap_ed"] = np.full(N, np.nan)
    C2["gap_ed"][m] = y[m] - fitted
    C2["coh_ed"] = float(y[m].mean())

    def ed_row(mask, A):
        r = care_sweep.describe(mask, A)
        a = care_adjusted.describe(mask, A)
        out = {"n_set": int(mask.sum()), "n": int((mask & A["has_ehr"]).sum())}
        out.update({c: r.get(c, np.nan) for c in ("ed_mean", "ed_lo", "ed_hi")})
        out.update({"ed_adj": a.get("ed_mean", np.nan), "ed_adj_lo": a.get("ed_lo", np.nan), "ed_adj_hi": a.get("ed_hi", np.nan)})
        return out

    coh = ed_row(np.ones(N, bool), C2)
    assert abs(coh["ed_adj"] - coh["ed_mean"]) < 1e-9, "the adjusted ED share of the coded-care cohort moved"
    rows = [{"grouping": "cohort", "decile": 0, "n_set": N, "n_visit": int(visit.sum()), "n_icd_visit": coh["n"],
             **{c: coh[c] for c in ED_COLS}}]
    order = rank_orders(ENCODERS, idx)
    current = {}
    OD = os.path.join(OUT_DIR, "outcome_deciles", "outcome_deciles.csv")
    OD = pd.read_csv(OD) if os.path.exists(OD) else None
    for name in ENCODERS:
        M = decile_masks(order[name], N)
        R = {j: ed_row(M[j], C2) for j in DECILES}
        Rv = {j: ed_row(M[j], C) for j in DECILES}                         # the current denominator, a check only
        current[name] = Rv
        if OD is not None:
            for j in DECILES:
                w = OD[(OD.grouping == name) & (OD.decile == j)].iloc[0]
                for ours, theirs in (("ed_mean", "ed_mean"), ("ed_adj", "ed_adj")):
                    b = pd.to_numeric(w[theirs], errors="coerce")
                    if pd.notna(b):
                        assert abs(Rv[j][ours] - b) <= 5e-4 * abs(b) + 1e-12, f"{name} decile {j}: {ours} is not outcome_deciles.csv's"
                assert int(w["n_ehr"]) == Rv[j]["n"], f"{name} decile {j}: n_visit is not outcome_deciles.csv's n_ehr"
        cnt = [R[j]["n"] for j in DECILES]
        forced = [pair_hidden(R[j]["n"], Rv[j]["n"]) or pair_hidden(R[j]["n"], R[j]["n_set"]) for j in DECILES]
        hid = hide_cells([cnt, [R[j]["n_set"] - R[j]["n"] for j in DECILES], [Rv[j]["n"] - R[j]["n"] for j in DECILES]],
                         forced)
        for j, h in zip(DECILES, hid):
            r = {"grouping": name, "decile": j, "n_set": R[j]["n_set"], "n_visit": Rv[j]["n"],
                 "n_icd_visit": (MARK if small(R[j]["n"]) else SUPP) if h else R[j]["n"]}
            r.update({c: (np.nan if h else R[j][c]) for c in ED_COLS})
            rows.append(r)
    return pd.DataFrame(rows), OD is not None


def main():
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)
    N = len(idx)
    B, C, S, SA, _ = load(idx)                                                # outcome_deciles.py's inputs and curves
    assert os.path.exists(PERSON), f"{PERSON} is missing: it is written by phecode_coverage.py (PheTK env)"
    P = pd.read_csv(PERSON, dtype={"person_id": str}).set_index("person_id").reindex(idx)
    icd = P["n_events"].fillna(0).values > 0
    visit = C["has_ehr"]

    if "--refresh" in sys.argv or not os.path.exists(CACHE):
        pull(idx)
    else:
        print(f"[ed_check] reusing {CACHE} (--refresh pulls again)")
    V = pd.read_csv(CACHE, dtype={"person_id": str, "visit_type": str, "source": str})
    V = V[V.person_id.isin(set(idx))]

    # checks against care.py's outcomes and phecode_coverage.py's cells
    tot = V.groupby("person_id")["n_visits"].sum().reindex(idx).fillna(0).values
    ed = V[V.visit_concept_id.isin(ED_CONCEPTS)].groupby("person_id")["n_visits"].sum().reindex(idx).fillna(0).values
    assert ((tot > 0) == visit).all(), "the pull's people with a visit are not care_outcomes_per_person.csv's has_ehr"
    ok = np.isclose(np.where(visit, ed / np.where(tot > 0, tot, 1), np.nan)[visit], C["ed_share"][visit])
    assert ok.all(), "a person's ED share from the pull differs from care_outcomes_per_person.csv's"
    cells = {"icd_and_visit": int((icd & visit).sum()), "icd_no_visit": int((icd & ~visit).sum()),
             "visit_no_icd": int((~icd & visit).sum()), "neither": int((~icd & ~visit).sum())}
    pc = os.path.join(OUT_DIR, "phecode_coverage", "phecode_coverage.json")
    if os.path.exists(pc):
        written = json.load(open(pc))["records_cells"]
        for k, v in cells.items():
            if str(written[k]).isdigit():
                assert int(written[k]) == v, f"records cell {k} is not phecode_coverage.json's"
    print(f"[ed_check] cohort {N:,} | at least 1 visit {cell(int(visit.sum()), N)} | of them with an ICD event "
          f"{cell(cells['icd_and_visit'], int(visit.sum()))}, without {cell(cells['visit_no_icd'], int(visit.sum()))}")

    groups = {"icd": set(idx[icd & visit]), "visit_only": set(idx[~icd & visit])}
    A = part_a(V, groups, {"cohort": N})
    A = coarsen_sig(coarsen(A, ["share_of_visits"]), ["visits_p25", "visits_median", "visits_p75", "mean_ed_share"])
    for c in ("people", "people_any_ed", "people_only"):                       # counts written as whole numbers
        A[c] = A[c].map(lambda v: str(int(v)) if isinstance(v, (int, float, np.integer, np.floating))
                         and pd.notna(v) else v).astype(object)
    Bd, checked = part_b(idx, C, icd, B, S, SA)
    Bd = coarsen_sig(Bd, ED_COLS)
    A.to_csv(os.path.join(OUT, "ed_check_visits.csv"), index=False)
    Bd.to_csv(os.path.join(OUT, "ed_check_deciles.csv"), index=False)

    print("  part A, by group (icd | visit_only):")
    for g in GROUPS:
        r = A[(A.section == "group") & (A.group == g)].iloc[0]
        print(f"    {g:10s} people {r.people} | visits per person {r.visits_p25:g} / {r.visits_median:g} / {r.visits_p75:g} "
              f"| any ED {r.people_any_ed} | mean ED share {r.mean_ed_share}")
    for section in ("source", "type"):
        print(f"  part A, by {section}: label | icd people, share of visits | visit_only people, share of visits")
        X = A[A.section == section]
        for lab in X.label.unique():
            a, b = (X[(X.label == lab) & (X.group == g)].iloc[0] for g in GROUPS)
            only = f" (only this source: {a.people_only} | {b.people_only})" if section == "source" else ""
            print(f"    {str(lab)[:40]:40s} {a.people!s:>7} {a.share_of_visits!s:>6} | {b.people!s:>7} {b.share_of_visits!s:>6}{only}")
    g = Bd[Bd.grouping == ENCODERS[0]].set_index("decile")
    c0 = Bd.iloc[0]
    print(f"  part B, {ENCODERS[0]}, ED share adjusted, %, on people with a visit AND an ICD event "
          f"(cohort [{100 * c0.ed_adj:.2f}]):")
    print("    " + " ".join("    -" if pd.isna(g.at[j, "ed_adj"]) else f"{100 * g.at[j, 'ed_adj']:5.2f}" for j in DECILES))

    json.dump({"cdr": CDR, "question": "what the visits of people with a visit and no ICD "
                                       "event are, and the ED share by decile on people with coded care",
               "visit": "a distinct visit_occurrence_id", "ed_concepts": list(ED_CONCEPTS),
               "source": "visit_occurrence_ext.src_id with digits removed", "top_types": TOP_TYPES,
               "groups": {"icd": "at least 1 visit and at least 1 ICD event (PheTK person summary)",
                          "visit_only": "at least 1 visit and no ICD event"},
               "part_b": {"denominator": "at least 1 visit and at least 1 ICD event", "encoders": ENCODERS,
                          "adjustment": "OLS on age, age^2, sex, refitted once on that denominator (care_adjusted.fit_curve)",
                          "current_denominator_checked_against_outcome_deciles": checked},
               "disclosure": "All of Us dissemination policy (disclosure.py); see the docstring",
               "precision": "shares to 3 decimals; means and percentiles to 4 significant figures",
               "versions": versions()},
              open(os.path.join(OUT, "ed_check_meta.json"), "w"), indent=2)
    files = ["ed_check_visits.csv", "ed_check_deciles.csv", "ed_check_meta.json"]
    for fn in files:
        ok, why = verdict(os.path.join(OUT, fn))
        assert ok, f"{fn} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"ed_check_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"ed_check/{fn}")
    print(f"\n[ed_check] download -> {zpath}")


if __name__ == "__main__":
    main()
