#!/usr/bin/env python3
"""sleep_device_deciles.py -- Fitbit sleep by decile of the distance, separately for program-provided devices and
participants' own devices (eTable 6).

WHY. In outcome_deciles.py the share of the sleep sample whose nights come from the program's wear study rises from
53% in decile 1 to 75% in decile 10. The age and sex adjustment does not cover the device, so part of the sleep
gradient could be the device mix. Within one device source the mix cannot differ between deciles.

THE 2 DEVICE SOURCES, by the wear-study roster sleep_sweep.py already uses (wear_enrollees_local.csv: a row in
`wear_study` with a consent start date):
  program   the person is enrolled in the wear study, through which the program provides devices
  own       the person is not enrolled
The sleep rules are unchanged (sleep_reliability.MIN_NIGHTS = 14 nights, the nights of sleep_sweep.py).

PER DEVICE SOURCE, ENCODER (the 6) AND DECILE (outcome_deciles.py's deciles: phecode_deciles.decile_masks)
  sleep_sweep.describe:     n_sleep (people of that source with >= 14 nights), coverage (their share of the
                            decile), sleep_mean with its interval, sleep_var_median with its interval, sleep_age
  sleep_adjusted.describe:  the same 2 measures adjusted for age and sex (variability also for log nights), with
                            sleep_adjusted.fit_curves REFITTED ONCE ON EACH SOURCE'S OWN SLEEP SAMPLE, so each
                            source is compared with its own cohort value (the cohort row of that source)
plus 1 cohort row per source.

WHAT LEAVES THE WORKBENCH: aggregates only, checked for person-level columns (pii_check, as phecode_fifteen.py).
The describe functions keep their own within-set rules unchanged.

INPUTS   sleep_person_local.csv, wear_enrollees_local.csv (beside the scripts; per person, stay), the responses
         file (age, sex), screen_out/cutoff_flags.csv and scores_<encoder>.csv
OUTPUTS  screen_out/sleep_device_deciles/sleep_device_deciles.csv    1 row per source x encoder x decile, + cohort rows
         screen_out/sleep_device_deciles/sleep_device_deciles_meta.json
         screen_out/sleep_device_deciles_<CDR>.zip

Run:  python3 sleep_device_deciles.py
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
from env_versions import versions                                          # noqa: E402  library versions, for the Methods
from phecode_deciles import DECILES, decile_masks                          # noqa: E402  the same deciles
import sleep_adjusted                                                      # noqa: E402  its curves and describe()
from sleep_reliability import MIN_NIGHTS, sleep_cache                      # noqa: E402  one rule, one file
import sleep_sweep                                                         # noqa: E402  its describe()
from wb_config import CFG                                                  # noqa: E402
from zip_screen_aggregates import ID_COLS                                  # noqa: E402  person-level column names

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "sleep_device_deciles")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
UNADJ = ["n_set", "n_sleep", "coverage", "sleep_age", "sleep_mean", "sleep_lo", "sleep_hi",
         "sleep_var_median", "sleep_var_lo", "sleep_var_hi"]
ADJ = {"sleep_mean": "sleep_mean_adj", "sleep_lo": "sleep_mean_adj_lo", "sleep_hi": "sleep_mean_adj_hi",
       "sleep_var_median": "sleep_var_median_adj", "sleep_var_lo": "sleep_var_median_adj_lo",
       "sleep_var_hi": "sleep_var_median_adj_hi"}


def pii_check(path):
    """(ok, reason): no person-level column and no written index."""
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
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)
    N = len(idx)
    R = pd.read_csv(CFG.responses_csv, usecols=[ID_COL, "age", "sex"])
    R[ID_COL] = R[ID_COL].astype(str)
    R = R.set_index(ID_COL).reindex(idx)
    assert set(np.unique(R["sex"])) <= {0.0, 1.0}, "sex is not coded 0/1 as build_responses_v9.py writes it"
    age, sex = R["age"].values.astype(float), R["sex"].values.astype(float)
    P = pd.read_csv(sleep_cache())
    P["person_id"] = P["person_id"].astype(str)
    P = P.set_index("person_id").reindex(idx)
    has = (P["nights"].fillna(0).values >= MIN_NIGHTS) & P["sd_min"].notna().values
    nights, mean_min, sd_min = (P[c].values.astype(float) for c in ("nights", "mean_min", "sd_min"))
    wf = os.path.join(HERE, "wear_enrollees_local.csv")
    assert os.path.exists(wf), f"{wf} is missing: run `python3 explore_other_data.py --cells 10`"
    wear = idx.isin(set(pd.read_csv(wf)["person_id"].astype(str)))
    sources = {"program": has & wear, "own": has & ~wear}
    print(f"[sleep_device_deciles] cohort {N:,} | at least {MIN_NIGHTS} nights {int(has.sum()):,} | "
          f"program-provided {int(sources['program'].sum()):,} | own device {int(sources['own'].sum()):,}")

    order = rank_orders(ENCODERS, idx)
    rows, curves = [], []
    for src, h in sources.items():
        g1, g2, C, converged = sleep_adjusted.fit_curves(age[h], sex[h], nights[h], mean_min[h], sd_min[h])
        C.insert(0, "source", src)
        curves.append(C)
        gap_mean, gap_sd = np.full(N, np.nan), np.full(N, np.nan)
        gap_mean[h], gap_sd[h] = g1, g2
        S = {"age": age, "has_sleep": h, "mean_min": mean_min, "sd_min": sd_min, "wear": None}
        SA = {"has_sleep": h, "gap_mean": gap_mean, "gap_sd": gap_sd, "coh_mean": float(mean_min[h].mean()),
              "coh_sd_median": float(np.median(sd_min[h])), "coh_gap_sd_median": float(np.median(g2))}

        def one(mask):
            r = sleep_sweep.describe(mask, S)
            a = sleep_adjusted.describe(mask, SA)
            out = {c: r.get(c, np.nan) for c in UNADJ}
            out.update({v: a.get(k, np.nan) for k, v in ADJ.items()})
            return out

        coh = one(np.ones(N, bool))
        assert abs(coh["sleep_mean_adj"] - coh["sleep_mean"]) < 1e-9, f"{src}: the adjusted cohort mean moved"
        assert abs(coh["sleep_var_median_adj"] - coh["sleep_var_median"]) < 1e-9, f"{src}: the adjusted cohort variability moved"
        rows.append({"source": src, "grouping": "cohort", "decile": 0, "median_regression_converged": converged, **coh})
        for name in ENCODERS:
            M = decile_masks(order[name], N)
            for j in DECILES:
                rows.append({"source": src, "grouping": name, "decile": j, **one(M[j])})

    D = pd.DataFrame(rows)
    # the 2 sources partition the sleep sample: their people add up to outcome_deciles.py's n_sleep (checked
    # against its file where present)
    od = os.path.join(OUT_DIR, "outcome_deciles", "outcome_deciles.csv")
    if os.path.exists(od):
        O = pd.read_csv(od)
        for name in ENCODERS:
            M = decile_masks(order[name], N)
            assert all(int((M[j] & sources["program"]).sum()) + int((M[j] & sources["own"]).sum()) == int((M[j] & has).sum())
                       for j in DECILES), f"{name}: the 2 sources do not partition the sleep sample"
            for j in DECILES:
                tot = int((decile_masks(order[name], N)[j] & has).sum())             # the 2 sources together
                w = pd.to_numeric(O[(O.grouping == name) & (O.decile == j)]["n_sleep"].iloc[0], errors="coerce")
                assert pd.isna(w) or int(w) == tot, f"{name} decile {j}: the 2 sources do not add up to outcome_deciles.csv's n_sleep"
    D.to_csv(os.path.join(OUT, "sleep_device_deciles.csv"), index=False)
    pd.concat(curves).to_csv(os.path.join(OUT, "sleep_device_curves.csv"), index=False)

    for src in sources:
        c0 = D[(D.source == src) & (D.grouping == "cohort")].iloc[0]
        g = D[(D.source == src) & (D.grouping == ENCODERS[0])].set_index("decile")
        print(f"  {src}: {ENCODERS[0]}, adjusted, by decile (cohort of this source in [])")
        print(f"    mean sleep, min       [{c0.sleep_mean_adj:6.1f}] " + " ".join(f"{v:6.1f}" for v in g.sleep_mean_adj))
        print(f"    variation, min        [{c0.sleep_var_median_adj:6.1f}] " + " ".join(f"{v:6.1f}" for v in g.sleep_var_median_adj))
        print(f"    people with sleep     [{c0.n_sleep!s:>6}] " + " ".join(f"{v!s:>6}" for v in g.n_sleep))

    json.dump({"cdr": CDR, "sources": {"program": "enrolled in the wear study (wear_enrollees_local.csv)",
                                       "own": "not enrolled in the wear study"},
               "min_nights": MIN_NIGHTS, "encoders": ENCODERS,
               "deciles": "outcome_deciles.py's (phecode_deciles.decile_masks); decile 10 = the most atypical tenth",
               "adjustment": "sleep_adjusted.fit_curves refitted once on each source's own sleep sample",
               "leaves the Workbench": "aggregates only; pii_check",
               "versions": versions()},
              open(os.path.join(OUT, "sleep_device_deciles_meta.json"), "w"), indent=2)
    files = ["sleep_device_deciles.csv", "sleep_device_curves.csv", "sleep_device_deciles_meta.json"]
    for fn in files:
        ok, why = pii_check(os.path.join(OUT, fn))
        assert ok, f"{fn}: {why}"
    zpath = os.path.join(OUT_DIR, f"sleep_device_deciles_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"sleep_device_deciles/{fn}")
    print(f"\n[sleep_device_deciles] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
