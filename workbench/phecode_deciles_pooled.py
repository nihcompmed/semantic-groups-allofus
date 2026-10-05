#!/usr/bin/env python3
"""phecode_deciles_pooled.py -- the 15 recorded conditions by decile of the distance, with adjacent deciles POOLED
where a decile holds 20 or fewer cases, so every bin can be published as it is.

WHY. In phecode_deciles.py some decile shares are blank: suicidal ideation, attempt or self-harm (deciles 1-4) and
bipolar disorder (deciles 1-3), where few people have the diagnosis, and prostate cancer (deciles 9-10), which
phecode_deciles.py's masking withholds at the 90th percentile although its cases are many. The All of Us Data and
Statistics Dissemination Policy (5/12/2020) permits "collapsing data across cells" for published figures. So each bin
here has more than 20 cases and more than 20 non-cases, and no cell is masked.

THE BINS, per encoder and condition. Start from the 10 deciles of outcome_deciles.py / phecode_deciles.py
(phecode_deciles.decile_masks). While a bin has 1 to 20 cases or 1 to 20 non-cases, take the lowest such bin and
merge it with its neighbor toward the NEARER END (a bin centered at decile 5.5 or below with the bin below it, or
above it if it is the first; otherwise with the bin above it, or below it if it is the last), so sparse low deciles
pool among themselves and the high deciles keep their resolution. A bin with 0 cases is kept (the policy permits 0).
Deciles that need no pooling stay single.

PER BIN, as phecode_deciles.py computes a decile (its own functions): the denominator n (people with at least 1 ICD
event; men for prostate cancer), the cases k (count >= 2, PheTK's rule), the share with its Wilson 95% interval, and
the share adjusted for age and sex (phecode_sweep's 15 curves, fitted once over each denominator: the cohort's share
+ the bin's mean gap, normal 95% interval). Plus 1 cohort row per condition. The 6 encoders.

WHAT LEAVES THE WORKBENCH: aggregates only, checked for person-level columns (pii_check).

INPUTS   as phecode_deciles.py (screen_out/phetk/*, the responses file, scores_<encoder>.csv, cutoff_flags.csv)
OUTPUTS  screen_out/phecode_deciles_pooled/phecode_deciles_pooled.csv       1 row per encoder x condition x bin
         screen_out/phecode_deciles_pooled/phecode_deciles_pooled_meta.json
         screen_out/phecode_deciles_pooled_<CDR>.zip

Run:  python3 phecode_deciles_pooled.py
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
from disclosure import small                                               # noqa: E402  1 to 20 inclusive
from phecode_deciles import DECILES, decile_masks                          # noqa: E402  the same deciles
from phecode_sweep import CONDITIONS, Z, fit_curve, load, wilson           # noqa: E402  Stage 1's own functions
from sleep_device_deciles import pii_check                                 # noqa: E402  the person-level check

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "phecode_deciles_pooled")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"


def pool(k, n):
    """Bins (lists of deciles) such that no bin has 1 to 20 cases or 1 to 20 non-cases. k, n: dicts by decile."""
    bins = [[j] for j in DECILES]
    fails = lambda b: small(sum(k[j] for j in b)) or small(sum(n[j] - k[j] for j in b))
    while len(bins) > 1:
        bad = [i for i, b in enumerate(bins) if fails(b)]
        if not bad:
            break
        i = bad[0]
        low_half = np.mean(bins[i]) <= 5.5                       # merge toward the nearer end, so the high
        other = (i - 1 if i > 0 else i + 1) if low_half else (i + 1 if i + 1 < len(bins) else i - 1)   # deciles stay apart
        a, b = sorted((i, other))
        bins[a] = bins[a] + bins[b]
        del bins[b]
    return bins


def main():
    os.makedirs(OUT, exist_ok=True)
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
    order = rank_orders(ENCODERS, idx)

    rows = []
    for key, code, label, lst, _ in CONDITIONS:
        d = A[f"den_{key}"]
        p, lo, hi = wilson(int(np.nansum(A[f"y_{key}"][d])), int(d.sum()))
        rows.append({"grouping": "cohort", "condition": key, "phecode": code, "label": label, "list": lst,
                     "decile_from": 0, "decile_to": 0, "n": int(d.sum()), "k": int(np.nansum(A[f"y_{key}"][d])),
                     "rate": p, "lo": lo, "hi": hi, "adj": A[f"coh_{key}"], "adj_lo": np.nan, "adj_hi": np.nan})
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
    # single deciles must equal phecode_deciles.csv where that file wrote them (3 decimals there)
    pd_path = os.path.join(OUT_DIR, "phecode_deciles", "phecode_deciles.csv")
    if os.path.exists(pd_path):
        P = pd.read_csv(pd_path)
        for r in D[(D.grouping != "cohort") & (D.decile_from == D.decile_to)].itertuples():
            w = P[(P.grouping == r.grouping) & (P.decile == r.decile_from)]
            v = pd.to_numeric(w[f"{r.condition}_adj"].iloc[0], errors="coerce") if len(w) else np.nan
            if pd.notna(v):
                assert abs(v - r.adj) <= 0.0005 + 1e-9, f"{r.grouping} {r.condition} decile {r.decile_from}: not phecode_deciles.csv's"
    D.to_csv(os.path.join(OUT, "phecode_deciles_pooled.csv"), index=False)

    print(f"[phecode_deciles_pooled] cohort {N:,} | 6 encoders x {len(CONDITIONS)} conditions")
    print(f"  pooled bins: {pooled if pooled else 'none'}")
    g = D[D.grouping == ENCODERS[0]]
    print(f"  {ENCODERS[0]}, adjusted share by bin, %; cohort in []:")
    for key, code, label, *_ in CONDITIONS:
        x = g[g.condition == key].sort_values("decile_from")
        c0 = D[(D.grouping == "cohort") & (D.condition == key)].iloc[0]
        cells = " ".join(f"{r.decile_from}{'-' + str(r.decile_to) if r.decile_to != r.decile_from else ''}:{100 * r.adj:.1f}"
                         for r in x.itertuples())
        print(f"    {label[:34]:34s} [{100 * c0.rate:4.1f}] {cells}")

    json.dump({"cdr": CDR, "encoders": ENCODERS,
               "bins": "the 10 deciles of phecode_deciles.py; a bin with 1 to 20 cases or non-cases is merged with its "
                       "neighbor toward the nearer end, lowest such bin first, until none is left",
               "pooled": pooled,
               "adjustment": "phecode_sweep.py's 15 curves (age, age^2, sex; prostate age, age^2), fitted once over each denominator",
               "denominator": "people with at least 1 ICD event (men for prostate cancer)",
               "leaves the Workbench": "aggregates only; pii_check",
               "versions": versions()},
              open(os.path.join(OUT, "phecode_deciles_pooled_meta.json"), "w"), indent=2)
    files = ["phecode_deciles_pooled.csv", "phecode_deciles_pooled_meta.json"]
    for fn in files:
        ok, why = pii_check(os.path.join(OUT, fn))
        assert ok, f"{fn}: {why}"
    zpath = os.path.join(OUT_DIR, f"phecode_deciles_pooled_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"phecode_deciles_pooled/{fn}")
    print(f"\n[phecode_deciles_pooled] {len(D)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
