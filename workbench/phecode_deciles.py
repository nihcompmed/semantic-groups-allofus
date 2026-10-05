#!/usr/bin/env python3
"""phecode_deciles.py -- the CONTROL for STAGE 1 of the phecodes: each diagnosis by decile of each
grouping's distance, from the least atypical tenth of the cohort to the most.

WHY. Stage 1's sets are cumulative, the top (100 - p)% of the cohort, so extending them to low p would
drift to the cohort line by construction. Non-overlapping deciles show whether the least atypical tenths
sit at or below the cohort, and where along the distance each diagnosis's share rises. This is a
control and a description. It does NOT change the outlier sets, which stay the top
(100 - p)% for p = 80 to 99, and their exclusive subsets.

THE DECILES. Each grouping's own order of the cohort (breadth_exclusive.rank_orders, most extreme first,
the same stable sort as every set) is cut at round(N * j / 10), j = 0 to 10. Decile 10 is the first
tenth, and decile 1 the last. So decile 10 is exactly Stage 1's top set at the 90th, and deciles 9 and
10 together are exactly its top set at the 80th (the same rounding as top_and_exclusive).

PER GROUPING (12) AND DECILE (1 to 10), per condition, as phecode_sweep.py (its own functions):
  n_rec, the unadjusted share (Wilson 95%), the share adjusted for age and sex (the same 15 curves,
  fitted once over each denominator; cohort share + the decile's mean gap, normal 95%).

THE DISSEMINATION POLICY (disclosure.py). The deciles partition the cohort, and two of
them are sets Stage 1 already published, so:
  - deciles 9 and 10 are written only where Stage 1 wrote both the top set at the 80th and at the 90th
    for that grouping and condition. Their values are then already derivable from Stage 1 and add
    nothing, and Stage 1's own checks cover them.
  - deciles 1 to 8 partition a total already public (the cohort minus Stage 1's top set at the 80th).
    Their cases, non-cases and n_rec go through partition_mask's rule (complete(), which starts from
    the cells already hidden), so a hidden cell can never be recovered by subtraction. If the 80th was not written in Stage 1, the partition is over deciles 1
    to 10 with the cohort's total, and 9 and 10 are hidden.
  - psychoactive substance contains alcohol: if subtracting alcohol from substance in any decile of a
    grouping leaves 1 to 20 cases, alcohol is withheld for all 10 deciles of that grouping.
  - an adjusted value is written only where the unadjusted one is.
  - ★ shares and intervals are written to 3 decimals (disclosure.coarsen): at full precision
    a prostate rate and its Wilson interval solve for the men-only denominator (phecode_sweep.py).
  - ★ where n_rec is "suppressed", every share on the row is blanked (disclosure.py).

INPUTS   as phecode_sweep.py, plus screen_out/phecode_sweep/phecode_sweep.csv (Stage 1, run first)
OUTPUTS  screen_out/phecode_deciles/phecode_deciles.csv        1 row per grouping x decile, plus the cohort row
         screen_out/phecode_deciles/phecode_deciles_meta.json
         screen_out/phecode_deciles_<CDR>.zip                    those 2, checked by verdict() in
                                                                 zip_screen_aggregates.py

Run:  python3 phecode_deciles.py
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from breadth_exclusive import rank_orders                              # noqa: E402  the same order
from breadth_sweep import ENCODERS, groupings                           # noqa: E402
from disclosure import DECIMALS, SUPP, coarsen, small   # noqa: E402  the dissemination policy
from phecode_sweep import CONDITIONS, CONTAINS, Z, fit_curve, load, wilson   # noqa: E402  Stage 1's own functions
from zip_screen_aggregates import verdict                               # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "phecode_deciles")
STAGE1 = os.path.join(OUT_DIR, "phecode_sweep", "phecode_sweep.csv")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
DECILES = list(range(1, 11))


def decile_masks(order_g, N):
    """Decile j (1 = least atypical, 10 = most) as a boolean mask, cut at round(N * i / 10)."""
    b = [int(round(N * i / 10.0)) for i in range(11)]
    out = {}
    for i in range(10):
        m = np.zeros(N, bool)
        m[order_g[b[i]:b[i + 1]]] = True
        out[10 - i] = m
    return out


def complete(counts, forced):
    """Cells of a partition of a public total to hide: the forced ones, every count of 1 to 20, then the
    smallest remaining nonzero cells until the hidden cells sum to 0 or to more than 20 (the rule of
    disclosure.partition_mask, started from cells that must be hidden for another reason)."""
    c = np.asarray(counts, dtype=np.int64)
    hide = np.array([small(x) for x in c], bool) | np.asarray(forced, bool)
    while True:
        s = int(c[hide].sum())
        if s == 0 or s > 20:
            return hide
        cand = np.flatnonzero(~hide & (c > 0))
        if len(cand) == 0:
            return hide
        hide[cand[np.argmin(c[cand])]] = True


def hide_cells(parts, forced):
    """Iterate complete() over several partitions (cases, non-cases, denominators) until stable."""
    hide = np.asarray(forced, bool)
    while True:
        new = hide.copy()
        for counts in parts:
            new |= complete(counts, new)
        if (new == hide).all():
            return hide
        hide = new


def main():
    os.makedirs(OUT, exist_ok=True)
    assert os.path.exists(STAGE1), f"{STAGE1} is missing: run phecode_sweep.py first"
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)
    N = len(idx)
    A = load(idx)
    for key, code, _, _, men_only in CONDITIONS:                 # the same 15 curves as Stage 1
        d = A[f"den_{key}"]
        fitted, _ = fit_curve(A["age"][d], A["sex"][d], A[f"y_{key}"][d], with_sex=not men_only)
        A[f"gap_{key}"] = np.full(N, np.nan)
        A[f"gap_{key}"][d] = A[f"y_{key}"][d] - fitted
        A[f"coh_{key}"] = float(np.mean(A[f"y_{key}"][d]))

    S1 = pd.read_csv(STAGE1)
    S1 = S1[S1.analysis == "all"].set_index(["grouping", "cut_pctile"])
    G12 = groupings()
    order = rank_orders([g[0] for g in G12], idx)

    rows = []
    coh = {"grouping": "cohort", "family": "reference", "k": np.nan, "decile": 0,
           "n_set": N, "n_rec": int(A["has_rec"].sum())}
    for key, *_ in CONDITIONS:
        d = A[f"den_{key}"]
        p, lo, hi = wilson(int(np.nansum(A[f"y_{key}"][d])), int(d.sum()))
        coh.update({f"{key}_rate": p, f"{key}_lo": lo, f"{key}_hi": hi,
                    f"{key}_adj": A[f"coh_{key}"], f"{key}_adj_lo": np.nan, f"{key}_adj_hi": np.nan})
    rows.append(coh)
    n_hidden = {"stage1_not_written": 0, "partition": 0, "alcohol_in_substance": 0}

    for name, fam, kk, _ in G12:
        M = decile_masks(order[name], N)
        # checks: the deciles are Stage 1's sets where they should be
        top90 = np.zeros(N, bool); top90[order[name][:int(round(N * 0.1))]] = True
        top80 = np.zeros(N, bool); top80[order[name][:int(round(N * 0.2))]] = True
        assert (M[10] == top90).all() and ((M[9] | M[10]) == top80).all(), f"{name}: deciles are not Stage 1's sets"
        assert np.sum([M[j] for j in DECILES], axis=0).max() == 1 and sum(M[j].sum() for j in DECILES) == N

        st = {}
        for j in DECILES:
            m = M[j]
            s = {"n_set": int(m.sum()), "n_rec": int((m & A["has_rec"]).sum())}
            for key, *_ in CONDITIONS:
                d = m & A[f"den_{key}"]
                n = int(d.sum())
                g = A[f"gap_{key}"][d]
                s[key] = {"n": n, "k": int(np.nansum(A[f"y_{key}"][d])), "gap": float(g.mean()),
                          "se": float(np.std(g, ddof=1) / np.sqrt(n))}
            st[j] = s

        hid = set()                                              # (decile, key)
        hid_rec = set()
        for key, *_ in CONDITIONS:
            w80 = pd.notna(S1.at[(name, 80), f"{key}_rate"])
            w90 = pd.notna(S1.at[(name, 90), f"{key}_rate"])
            if not (w80 and w90):
                hid |= {(9, key), (10, key)}
                n_hidden["stage1_not_written"] += 2
            part = [1, 2, 3, 4, 5, 6, 7, 8] if w80 else DECILES
            k = [st[j][key]["k"] for j in part]
            mm = [st[j][key]["n"] - st[j][key]["k"] for j in part]
            nn = [st[j][key]["n"] for j in part]
            h = hide_cells([k, mm, nn], [(j, key) in hid for j in part])
            for j, x in zip(part, h):
                if x and (j, key) not in hid:
                    hid.add((j, key))
                    n_hidden["partition"] += 1
        for big, sub in CONTAINS:                                 # substance contains alcohol
            if any(small(st[j][big]["k"] - st[j][sub]["k"]) for j in DECILES):
                n_hidden["alcohol_in_substance"] += sum((j, sub) not in hid for j in DECILES)
                hid |= {(j, sub) for j in DECILES}
        w80r = pd.notna(pd.to_numeric(S1.at[(name, 80), "n_rec"], errors="coerce"))
        w90r = pd.notna(pd.to_numeric(S1.at[(name, 90), "n_rec"], errors="coerce"))
        if not (w80r and w90r):
            hid_rec |= {9, 10}
        partr = [1, 2, 3, 4, 5, 6, 7, 8] if w80r else DECILES
        hr = hide_cells([[st[j]["n_rec"] for j in partr], [st[j]["n_set"] - st[j]["n_rec"] for j in partr]],
                        [j in hid_rec for j in partr])
        hid_rec |= {j for j, x in zip(partr, hr) if x}

        for j in DECILES:
            s = st[j]
            r = {"grouping": name, "family": fam, "k": kk, "decile": j, "n_set": s["n_set"],
                 "n_rec": SUPP if j in hid_rec else s["n_rec"]}
            for key, *_ in CONDITIONS:
                if (j, key) in hid or j in hid_rec:      # ★ shares over a suppressed n_rec would return it
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
    # the deciles written where Stage 1 wrote the same set must carry Stage 1's value (Stage 1 writes it
    # rounded to DECIMALS places, so the check allows half a unit of that rounding)
    for name, *_ in G12:
        for key, *_ in CONDITIONS:
            v = D[(D.grouping == name) & (D.decile == 10)][f"{key}_adj"].iloc[0]
            if pd.notna(v):
                assert abs(v - S1.at[(name, 90), f"{key}_adj"]) <= 0.5 * 10 ** -DECIMALS + 1e-12, \
                    f"{name} {key}: decile 10 is not Stage 1's 90th"
    D = coarsen(D, stat_cols)                  # ★ precision (disclosure.py): after every check above
    D.to_csv(os.path.join(OUT, "phecode_deciles.csv"), index=False)

    c0 = D.iloc[0]
    print(f"[phecode_deciles] cohort {N:,} | 12 groupings x 10 deciles")
    print("  gte-large-en-v1.5, adjusted share by decile (1 = least atypical ... 10 = most), %; cohort in []:")
    gte = D[D.grouping == ENCODERS[0]].set_index("decile")
    for key, code, label, lst, _ in CONDITIONS:
        vals = " ".join("   -" if pd.isna(gte.at[j, f"{key}_adj"]) else f"{100 * gte.at[j, f'{key}_adj']:4.1f}" for j in DECILES)
        print(f"    {lst} {label[:34]:34s} [{100 * c0[key + '_rate']:4.1f}] {vals}")
    tot = len(CONDITIONS) * 12 * 10
    withheld = int(D[[f"{key}_rate" for key, *_ in CONDITIONS]].iloc[1:].isna().sum().sum())
    print(f"  shares withheld: {withheld:,} of {tot:,} {n_hidden}")

    json.dump({"cdr": CDR, "deciles": "each grouping's own order, cut at round(N * j / 10); decile 10 = the most atypical "
                                      "tenth = Stage 1's top set at the 90th; deciles 9 + 10 = its top set at the 80th",
               "conditions": [{"key": k, "phecode": c, "label": l, "list": s, "denominator":
                               "men with at least 1 ICD event" if m else "people with at least 1 ICD event"}
                              for k, c, l, s, m in CONDITIONS],
               "groupings_all": [{"grouping": g, "family": f_, "k": k, "k_rule": r} for g, f_, k, r in G12],
               "adjustment": "the 15 curves of phecode_sweep.py (age, age^2, sex; prostate age, age^2)",
               "withheld": n_hidden, "disclosure": "All of Us dissemination policy (disclosure.py); see the docstring",
               "precision": "shares and intervals to 3 decimals (disclosure.py)"},
              open(os.path.join(OUT, "phecode_deciles_meta.json"), "w"), indent=2)
    files = ["phecode_deciles.csv", "phecode_deciles_meta.json"]
    for fn in files:
        ok, why = verdict(os.path.join(OUT, fn))
        assert ok, f"{fn} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"phecode_deciles_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"phecode_deciles/{fn}")
    print(f"\n[phecode_deciles] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
