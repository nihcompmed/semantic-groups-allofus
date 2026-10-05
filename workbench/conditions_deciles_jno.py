#!/usr/bin/env python3
"""conditions_deciles_jno.py -- the 8 recorded conditions of conditions_jno.py by decile of the distance, under
each of the 6 encoders (Figure 2D, eFigure 2, eTable 7). It uses the deciles, the pooling of sparse bins and the
adjustment of phecode_deciles_pooled.py, on these conditions and their populations (the tested population where a
condition has a test, the same as in Figure 4).

PER BIN (phecode_deciles_pooled.py's own pool(); phecode_sweep.py's fit_curve() and wilson()): the population n, the
cases k, the share with its Wilson 95% interval, and the share adjusted for age and sex: one least-squares curve
per condition, y ~ 1 + age + age^2 (+ sex where the population holds both sexes), fitted once over its population;
a bin's value = the population's share + the bin's mean gap from the curve, normal 95% interval. Adjacent deciles
are pooled toward the nearer end while a bin has 1 to 20 cases or non-cases. Plus 1 row per condition for the
whole population.

CHECK. GAD, ADHD and PTSD have the same labels and populations in phecode_deciles_pooled.py, so their rows must
equal phecode_deciles_pooled.csv exactly where that download is on the Workbench (n and k equal, shares to 1e-9).

OUTPUTS  screen_out/conditions_deciles_jno/conditions_deciles_jno.csv      1 row per encoder x condition x bin
         screen_out/conditions_deciles_jno/conditions_curves_jno.csv       the 8 curves' coefficients and SEs
         screen_out/conditions_deciles_jno/conditions_deciles_jno_meta.json
         screen_out/conditions_deciles_jno_<CDR>.zip                       those 3 (pii_check; counts as they are)

Run (after phecode_counts_jno.py and conditions_jno.py):  python3 conditions_deciles_jno.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from breadth_exclusive import rank_orders                                   # noqa: E402  the same order
from breadth_sweep import ENCODERS                                          # noqa: E402
from conditions_jno import CDR, CONDITIONS, ID_COL, OUT_DIR, SEX_ONLY, TESTED_BY, UNCHANGED, load, with_sex, zip_out  # noqa: E402
from disclosure import small                                                # noqa: E402  1 to 20 inclusive
from env_versions import versions                                           # noqa: E402
from phecode_deciles import DECILES, decile_masks                           # noqa: E402  the same deciles
from phecode_deciles_pooled import pool                                     # noqa: E402  the same pooling
from phecode_sweep import Z, fit_curve, wilson                              # noqa: E402  the same curve and interval

OUT = os.path.join(OUT_DIR, "conditions_deciles_jno")
OLD = os.path.join(OUT_DIR, "phecode_deciles_pooled", "phecode_deciles_pooled.csv")


def main():
    os.makedirs(OUT, exist_ok=True)
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"), usecols=[ID_COL])
    idx = pd.Index(cf[ID_COL].astype(str), name=ID_COL)
    N = len(idx)
    A = load(idx)
    coef = []
    for key, code, *_ in CONDITIONS:
        d = A[f"den_{key}"]
        fitted, rows = fit_curve(A["age"][d], A["sex"][d], A[f"y_{key}"][d], with_sex=with_sex(key))
        A[f"gap_{key}"] = np.full(N, np.nan)
        A[f"gap_{key}"][d] = A[f"y_{key}"][d] - fitted
        A[f"coh_{key}"] = float(np.mean(A[f"y_{key}"][d]))
        assert abs(np.mean(A[f"gap_{key}"][d])) < 1e-9, f"{key}: the gaps do not average 0 over the population"
        coef += [{"condition": key, "phecode": code, **r} for r in rows]
    order = rank_orders(ENCODERS, idx)

    rows = []
    for key, code, label, lst, _ in CONDITIONS:
        d = A[f"den_{key}"]
        k0, n0 = int(np.nansum(A[f"y_{key}"][d])), int(d.sum())
        p, lo, hi = wilson(k0, n0)
        rows.append({"grouping": "population", "condition": key, "phecode": code, "label": label, "list": lst,
                     "decile_from": 0, "decile_to": 0, "n": n0, "k": k0, "rate": p, "lo": lo, "hi": hi,
                     "adj": A[f"coh_{key}"], "adj_lo": np.nan, "adj_hi": np.nan})
    pooled = {}
    for name in ENCODERS:
        M = decile_masks(order[name], N)
        for key, code, label, lst, _ in CONDITIONS:
            dm = {j: M[j] & A[f"den_{key}"] for j in DECILES}
            n = {j: int(dm[j].sum()) for j in DECILES}
            k = {j: int(np.nansum(A[f"y_{key}"][dm[j]])) for j in DECILES}
            bins = pool(k, n)
            if len(bins) < 10:
                pooled[f"{name} {key}"] = [f"{b[0]}-{b[-1]}" for b in bins if len(b) > 1]
            for b in bins:
                m = np.zeros(N, bool)
                for j in b:
                    m |= dm[j]
                nn, kk = int(m.sum()), int(np.nansum(A[f"y_{key}"][m]))
                assert not small(kk) and not small(nn - kk), f"{name} {key} {b}: a bin still has 1 to 20 cases or non-cases"
                p, lo, hi = wilson(kk, nn)
                g = A[f"gap_{key}"][m]
                mg, se = float(g.mean()), float(np.std(g, ddof=1) / np.sqrt(nn))
                c = A[f"coh_{key}"]
                rows.append({"grouping": name, "condition": key, "phecode": code, "label": label, "list": lst,
                             "decile_from": b[0], "decile_to": b[-1], "n": nn, "k": kk, "rate": p, "lo": lo, "hi": hi,
                             "adj": c + mg, "adj_lo": c + mg - Z * se, "adj_hi": c + mg + Z * se})
    D = pd.DataFrame(rows)

    checked = "phecode_deciles_pooled.csv not present"
    if os.path.exists(OLD):                          # the unchanged labels reproduce phecode_deciles_pooled.csv exactly
        P = pd.read_csv(OLD)
        n_cmp = 0
        for r in D[D.condition.isin(UNCHANGED) & (D.grouping != "population")].itertuples():
            w = P[(P.grouping == r.grouping) & (P.condition == r.condition) & (P.decile_from == r.decile_from)
                  & (P.decile_to == r.decile_to)]
            assert len(w) == 1, f"{r.grouping} {r.condition} {r.decile_from}-{r.decile_to}: not in the download"
            w = w.iloc[0]
            assert int(w.n) == r.n and int(w.k) == r.k, f"{r.grouping} {r.condition} {r.decile_from}: n or k moved"
            assert abs(float(w.adj) - r.adj) < 1e-9 and abs(float(w.rate) - r.rate) < 1e-9, \
                f"{r.grouping} {r.condition} {r.decile_from}: the share moved"
            n_cmp += 1
        checked = f"{n_cmp} bins of {', '.join(UNCHANGED)} equal phecode_deciles_pooled.csv (n, k, shares)"
        print(f"  check: {checked}")

    D.to_csv(os.path.join(OUT, "conditions_deciles_jno.csv"), index=False)
    pd.DataFrame(coef).to_csv(os.path.join(OUT, "conditions_curves_jno.csv"), index=False)
    print(f"[conditions_deciles_jno] cohort {N:,} | {len(ENCODERS)} encoders x {len(CONDITIONS)} conditions")
    print(f"  pooled bins: {pooled if pooled else 'none'}")
    g = D[D.grouping == ENCODERS[0]]
    print(f"  {ENCODERS[0]}, adjusted share by bin, %; the whole population in []:")
    for key, code, label, *_ in CONDITIONS:
        x = g[g.condition == key].sort_values("decile_from")
        c0 = D[(D.grouping == "population") & (D.condition == key)].iloc[0]
        cells = " ".join(f"{r.decile_from}{'-' + str(r.decile_to) if r.decile_to != r.decile_from else ''}:{100 * r.adj:.1f}"
                         for r in x.itertuples())
        print(f"    {label[:30]:30s} n {int(c0.n):>6,} [{100 * c0.rate:4.1f}] {cells}")
    json.dump({"cdr": CDR, "encoders": ENCODERS, "conditions": [c[:4] for c in CONDITIONS],
               "populations": {"all": "people with at least 1 ICD event", "sex_only": SEX_ONLY, "tested_by": TESTED_BY},
               "bins": "the 10 deciles of phecode_deciles.py; a bin with 1 to 20 cases or non-cases is merged with its "
                       "neighbor toward the nearer end, lowest such bin first, until none is left",
               "pooled": pooled,
               "adjustment": "OLS y ~ 1 + age + age^2 (+ sex where both sexes), fitted once over each population; "
                             "bin value = population share + the bin's mean gap; normal 95% interval of the gaps",
               "check_unchanged": checked, "counts": "written as they are; pii_check",
               "versions": versions()},
              open(os.path.join(OUT, "conditions_deciles_jno_meta.json"), "w"), indent=2)
    zpath = zip_out(OUT, ["conditions_deciles_jno.csv", "conditions_curves_jno.csv", "conditions_deciles_jno_meta.json"],
                    "conditions_deciles_jno")
    print(f"\n[conditions_deciles_jno] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
