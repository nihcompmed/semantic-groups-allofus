#!/usr/bin/env python3
"""care_band_check.py -- how much the curve's misfit moves each set's age- and sex-adjusted care value
A check on care_adjusted.py; it does not change care_adjusted.py's outputs.

WHY. care_adjusted.py's reference is a least-squares curve on age, age squared and sex. Its printed
band table shows the curve missing the cohort's own rates at the ends of the age range (above them
at 18 to 24, below them over 80, below zero for men over 80 on cost and provider bias). An estimate
outside the Workbench puts the effect on every set at the 95th at 0.2 points or less, but it has to
assume each set's ages are normal, because only their mean and SD leave the Workbench. This computes
it exactly, on everyone's own age, at every cut.

THE SECOND REFERENCE. For each outcome, the cohort's own value in each 5-year age band x sex cell
(18-24, 25-29, ..., 80-84, 85 and over; 28 cells), over everyone the outcome is measured on, the same
people the curve is fitted on. A person's band gap = their value minus their cell's value. A set's
band-adjusted value = the cohort's value + the set's mean band gap. Both references average to the
cohort's value over the cohort, so the two adjusted values differ only by the set's mean of
(curve value - cell value): the misfit, weighted by the set's own ages and sexes.

PER SET (the sets of breadth_sweep.py and breadth_exclusive.py, through their own code)
  <outcome>_curve   care_adjusted.py's value, recomputed with its own functions (checked against
                    screen_out/care_adjusted/care_adjusted.csv when that file is there)
  <outcome>_band    the band-adjusted value
  <outcome>_diff    band minus curve, in the outcome's units (a proportion; x 100 for points)
  outcome in access, cost, avoid, ed. n_set as care_sweep.py writes it.

THE DISSEMINATION POLICY (disclosure.py). Each row goes through care_sweep.py's
describe() first, and a value is written only where the unadjusted one survived, exactly as in
care_adjusted.py. The 28 cell values are never written. The printed cell sizes stay on the Workbench.

INPUTS   as care_adjusted.py
OUTPUTS  screen_out/care_band_check/care_band_check.csv        1 row per set, as care_adjusted.csv
         screen_out/care_band_check/care_band_check_meta.json  the largest |diff| per outcome, and where
         screen_out/care_band_check_<CDR>.zip                   those 2, checked by verdict() in
                                                                zip_screen_aggregates.py

Run:  python3 care_band_check.py
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from breadth_exclusive import rank_orders, top_and_exclusive           # noqa: E402  the same people
from breadth_sweep import CUTS, ENCODERS, groupings                      # noqa: E402
from care_adjusted import BAND_EDGES, OUTCOMES, fit_curve                # noqa: E402  the same curve
from care_sweep import RATES, describe as describe_unadjusted           # noqa: E402  the same policy
from disclosure import MARK                                             # noqa: E402  the dissemination policy
from wb_config import CFG                                               # noqa: E402
from zip_screen_aggregates import verdict                               # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "care_band_check")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"


def band_values(age, sex, y):
    """Each person's cell value (5-year age band x sex, over the people passed) and the cell sizes."""
    band = pd.cut(age, BAND_EDGES, right=False).codes
    assert (band >= 0).all(), "an age falls outside the bands"
    cell = band * 2 + sex.astype(int)
    d = pd.DataFrame({"cell": cell, "y": y})
    g = d.groupby("cell")["y"]
    return g.transform("mean").values, g.size()


def compare(mask, A, n_top=None):
    """1 row: n_set, and per outcome the curve-adjusted value, the band-adjusted value and their
    difference, written only where care_sweep.py writes the unadjusted value."""
    r = describe_unadjusted(mask, A, n_top)
    row = {"n_set": r["n_set"]}
    for short, (_, has, (v, _, _)) in OUTCOMES.items():
        if pd.isna(r.get(v, np.nan)):
            continue
        s = mask & A[has]
        c = A[f"coh_{short}"]
        cur = c + float(A[f"gap_curve_{short}"][s].mean())
        ban = c + float(A[f"gap_band_{short}"][s].mean())
        row.update({f"{short}_curve": cur, f"{short}_band": ban, f"{short}_diff": ban - cur})
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)         # the order every breadth script uses
    N = len(idx)

    R = pd.read_csv(CFG.responses_csv, usecols=[ID_COL, "age", "sex"])
    R[ID_COL] = R[ID_COL].astype(str)
    R = R.set_index(ID_COL).reindex(idx)
    assert R.notna().all().all(), "age or sex missing for someone in the cohort"
    assert set(np.unique(R["sex"])) <= {0.0, 1.0}, "sex is not coded 0/1 as build_responses_v9.py writes it"

    p = os.path.join(OUT_DIR, "care_outcomes_per_person.csv")
    assert os.path.exists(p), f"{p} is missing: run `python3 care.py --outcomes-only`"
    O = pd.read_csv(p)
    O[ID_COL] = O[ID_COL].astype(str)
    O = O.set_index(ID_COL).reindex(idx)
    A = {"age": R["age"].values.astype(float), "sex": R["sex"].values.astype(float),
         "has_survey": O["has_survey"].fillna(False).astype(bool).values,      # absent from the file = no survey
         "has_ehr": O["has_ehr"].fillna(False).astype(bool).values,
         "ed_share": O["ed_share"].values.astype(float)}
    for col in RATES.values():
        A[col] = O[col].values.astype(float)

    for short, (col, has, _) in OUTCOMES.items():
        m = A[has]
        y = A[col][m]
        assert np.isfinite(y).all(), f"{col} is missing for someone it is measured on"
        fitted, _ = fit_curve(A["age"][m], A["sex"][m], y)
        cellv, sizes = band_values(A["age"][m], A["sex"][m], y)
        A[f"gap_curve_{short}"], A[f"gap_band_{short}"] = np.full(N, np.nan), np.full(N, np.nan)
        A[f"gap_curve_{short}"][m], A[f"gap_band_{short}"][m] = y - fitted, y - cellv
        A[f"coh_{short}"] = float(y.mean())
        print(f"[care_band_check] {short}: {int(m.sum()):,} people in {len(sizes)} cells | "
              f"smallest cell {int(sizes.min()):,} (printed only)")

    G12 = groupings()
    order = rank_orders([g[0] for g in G12], idx)
    info = {g[0]: g for g in G12}
    names12 = [g[0] for g in G12]
    others = [g[0] for g in G12 if g[0] not in ENCODERS]
    rows = [{"analysis": "cohort", "encoder_slot": "", "grouping": "cohort", "family": "reference",
             "k": np.nan, "cut_pctile": 0, **compare(np.ones(N, bool), A)}]
    for short in OUTCOMES:                    # both references average to the cohort's value over the cohort
        assert abs(rows[0][f"{short}_diff"]) < 1e-9, f"the cohort's {short} differs between the 2 references"
    for cut in CUTS:
        n_top, member, _ = top_and_exclusive(order, names12, N, cut)
        for name in names12:
            _, fam, k, _ = info[name]
            rows.append({"analysis": "all", "encoder_slot": "", "grouping": name, "family": fam, "k": k,
                         "cut_pctile": cut, **compare(member[name], A)})
        for slot in ENCODERS:
            names7 = [slot] + others
            _, _, excl = top_and_exclusive(order, names7, N, cut)
            for name in names7:
                _, fam, k, _ = info[name]
                rows.append({"analysis": "exclusive", "encoder_slot": slot, "grouping": name, "family": fam,
                             "k": k, "cut_pctile": cut, "n_top": n_top, **compare(excl[name], A, n_top)})
    D = pd.DataFrame(rows)
    cols = [f"{s}_{w}" for s in OUTCOMES for w in ("curve", "band", "diff")]
    for c in cols:
        if c not in D.columns:
            D[c] = np.nan

    # the curve values must be care_adjusted.py's own, where that file is on disk
    prev = os.path.join(OUT_DIR, "care_adjusted", "care_adjusted.csv")
    if os.path.exists(prev):
        P = pd.read_csv(prev)
        assert len(P) == len(D), "care_adjusted.csv has a different number of rows"
        for s, (_, _, (v, _, _)) in OUTCOMES.items():
            assert (D[f"{s}_curve"].isna().values == P[v].isna().values).all(), \
                f"{s}: the values withheld differ from care_adjusted.csv"
        worst = max(float(np.nanmax(np.abs(D[f"{s}_curve"].values - P[v].values)))
                    for s, (_, _, (v, _, _)) in OUTCOMES.items())
        assert worst < 1e-9, f"the curve values differ from care_adjusted.csv by {worst}"
        print(f"  curve values match care_adjusted.csv (largest difference {worst:.1e})")
    else:
        print("  care_adjusted.csv not found: the curve values were not cross-checked")
    D.to_csv(os.path.join(OUT, "care_band_check.csv"), index=False)

    summary = {}
    print("  band minus curve, in percentage points (ED share also in points):")
    for s in OUTCOMES:
        d = D[f"{s}_diff"]
        i = d.abs().idxmax()
        r = D.loc[i]
        summary[s] = {"largest_abs_diff_points": 100 * float(abs(d[i])),
                      "at": {"analysis": r.analysis, "encoder_slot": r.encoder_slot, "grouping": r.grouping,
                             "cut_pctile": int(r.cut_pctile)},
                      "largest_abs_diff_points_at_95th": 100 * float(d[D.cut_pctile == 95].abs().max())}
        print(f"    {s:6s} largest |diff| {100 * abs(d[i]):.2f} ({r.analysis}, {r.encoder_slot or '-'}, {r.grouping}, "
              f"{int(r.cut_pctile)}th) | at the 95th {summary[s]['largest_abs_diff_points_at_95th']:.2f}")
    for analysis, slot in (("all", ""), ("exclusive", ENCODERS[0])):
        print(f"  at the 95th, {analysis}{' (slot ' + slot + ')' if slot else ''}: curve / band, %")
        at = D[(D.analysis == analysis) & (D.encoder_slot == slot) & (D.cut_pctile == 95)]
        for r in at.itertuples():
            print(f"    {r.grouping:22s} " + " | ".join(
                f"{s} {100 * getattr(r, s + '_curve'):5.1f} / {100 * getattr(r, s + '_band'):5.1f}" for s in OUTCOMES))

    json.dump({"cdr": CDR, "cuts": CUTS, "slots": ENCODERS, "outcomes": {**RATES, "ed": "ed_share"},
               "band_edges_years": [e for e in BAND_EDGES if np.isfinite(e)] + ["and over"],
               "references": {"curve": "care_adjusted.py: OLS on age, age^2, sex",
                              "band": "the cohort's own value in each 5-year age band x sex cell (28 cells)"},
               "diff": "band-adjusted minus curve-adjusted, a proportion in the csv; points in this summary",
               "summary": summary,
               "withheld": f"wherever care_sweep.py withholds the unadjusted value ({MARK} sets included)",
               "disclosure": "All of Us dissemination policy (disclosure.py); the cell values are not written"},
              open(os.path.join(OUT, "care_band_check_meta.json"), "w"), indent=2)
    files = ["care_band_check.csv", "care_band_check_meta.json"]
    for f in files:
        ok, why = verdict(os.path.join(OUT, f))
        assert ok, f"{f} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"care_band_check_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(OUT, f), arcname=f"care_band_check/{f}")
    print(f"\n[care_band_check] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
