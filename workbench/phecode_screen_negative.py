#!/usr/bin/env python3
"""phecode_screen_negative.py -- the clinical question. Among the people a published screen does NOT flag,
the share diagnosed with the matching condition, by decile of gte's distance, split by whether the diagnosis
was first coded before or after the person answered that screen.

Runs in the MAIN environment (python3), like phecode_sweep.py. It reads the person-level PheTK counts
phecode_coverage.py wrote (PheTK 0.3.6, phecodeX X1.0) and pulls one small thing from BigQuery, the
answer dates of the screen items.

THE FOUR PAIRS
  PHQ-9 total >= 10                                  <-> MB_286.2 major depressive disorder
  GAD-7 total >= 10                                  <-> MB_288.3 generalized anxiety disorder
  ASRS Part A, >= 4 of 6 at clinical frequency       <-> MB_304 ADHD
  ss_2 or ss_3 Yes (criterion 6)                     <-> MB_284 suicidal ideation, attempt or self-harm

SCREENS NEGATIVE, per pair:
  1. the criterion is evaluable: every item answered, none imputed (cutoff_flags.csv; withheld = empty);
  2. it is not met (cutoff_flags.csv = 0);
  3. the person has at least 1 ICD-coded event (phecode_sweep.py's denominator).
  ss_1 plays no part: it asks about thoughts, and the codes record acts.

THE DIAGNOSIS: PheTK count >= 2, split by its first_event_date (PheTK's earliest event of the
phecode, the children rolled up) against the person's ANSWER DATE for that pair:
  before   first coded on or before the answer date (the survey day counts as before)
  after    first coded after it, so nothing was recorded up to and including the day they answered
THE ANSWER DATE: the date the person answered THAT screen's own items, the latest of them if they span
more than one day. PHQ-9, GAD-7, ss_2 and ss_3 are in Emotional Health History and
Well-Being, the ASRS in Behavioral Health and Personality. Pulled once from `observation` (the screen
items' concept ids only) and cached, person level, in screen_out/screen_answer_dates_local.csv, which stays
on the Workbench. answer_dates_local.csv is NOT used: it holds the last answer on any of the 151 items.

THE DECILES: the cohort's deciles of gte-large-en-v1.5's distance, exactly phecode_deciles.py's
(decile_masks on breadth_exclusive.rank_orders; 1 = least atypical, 10 = most), kept to each pair's
screen-negative people. gte only: only the semantic grouping is interpreted here.

PER PAIR, the screen-negative total (decile 0) and each decile:
  n                               screen-negative people with at least 1 ICD event
  before_rate, _lo, _hi           share first coded before, Wilson 95%
  after_rate, _lo, _hi            share first coded after, Wilson 95%
  before_adj, after_adj (+ _lo, _hi)
                                  adjusted for age and sex as Stage 1: one least-squares curve per pair and
                                  kind (y ~ 1 + age + age^2 + sex), fitted once over the pair's screen-negative
                                  people; value = their share + the decile's mean gap, normal 95%, the curve
                                  treated as known. The gaps average 0, so the total row does not move.
  before_adjs, after_adjs (+ _lo, _hi)
                                  ALSO standardized for the screen's own score below its threshold: the
                                  same curve plus the score as categories (reference 0).
                                  PHQ-9 total 0-9, GAD-7 total 0-9, ASRS count of Part A items at clinical
                                  frequency 0-3. Criterion 6 has no score below its threshold (screening
                                  negative = both answers No), so its columns stay empty.
  score_mean, score_sd            that score's mean and SD over the row's people (empty for criterion 6)
  The scores are computed from the response file with cutoffs.py's own formulas, and every screen-negative
  person must fall below the threshold, or the script stops.

THE DISSEMINATION POLICY (disclosure.py). In one decile the screen-negative people split
into A (a case first coded before), B (a case first coded after) and C (not a case). The file shows n =
A + B + C, A (through the before share) and B (through the after share). A reader also holds, for the same
decile, the people with records R and the cases K (phecode_deciles.csv; A and B lie inside K, C inside
R - K), and the deciles sum to the total row. The counts a reader could then form, per kind, are
  before   A, n - A, K - A
  after    B, n - B, K - B
  both     C, A + B, K - A - B, (R - K) - C
Each is a partition of the 10 deciles with a known total, and phecode_deciles.complete is applied to
it: a count of 1 to 20 hides its share, and more shares are hidden, smallest first, until the hidden cells
sum to 0 or to more than 20. "both" is hidden in a decile whenever either share is, and a cell it adds
hides the share with the smaller count. The loop repeats until nothing changes.
  - The cohort's R and K are treated as public in every decile, whether or not phecode_deciles wrote them.
  - The total row is checked against the cohort (R, K) the same way. If it fails, that share is hidden in
    every row, since the deciles would add up to it.
  - n, R - n (the others with records in a decile) and, for the total row, S - n (the screen-negative
    people with no ICD event; S is public through cutoff_summary) must each exceed 20, or the script
    stops: the checks above rest on them.
  - An adjusted value is written only where the unadjusted one is.

INPUTS   screen_out/cutoff_flags.csv; screen_out/scores_gte-large-en-v1.5.csv; the PheTK caches of
         phecode_coverage.py (screen_out/phetk/); the responses file (age, sex); the battery's items file
         (the screen items' concept ids); BigQuery `observation` once, then the cache
OUTPUTS  screen_out/phecode_screen_negative/phecode_screen_negative.csv         1 row per pair x (total, 10 deciles)
         screen_out/phecode_screen_negative/phecode_screen_negative_curves.csv  the 8 + 6 curves' coefficients
         screen_out/phecode_screen_negative/phecode_screen_negative_meta.json
         screen_out/phecode_screen_negative_<CDR>.zip                            those 3, checked by verdict()

Run:  python3 phecode_screen_negative.py              (--refresh pulls the answer dates again)
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from breadth_exclusive import rank_orders                              # noqa: E402  the same order
from breadth_sweep import ENCODERS                                      # noqa: E402
from cutoffs import ASRS_HYPER, ASRS_INATT, GAD, PHQ, SUICIDE_ITEMS     # noqa: E402  the criteria's own items
from disclosure import small                                            # noqa: E402  the policy
from phecode_deciles import DECILES, decile_masks, hide_cells           # noqa: E402  the same deciles and partition rule
from phecode_sweep import CONDITIONS, COUNTS, MIN_COUNT, Z, fit_curve, load, wilson   # noqa: E402
from wb_config import CFG                                               # noqa: E402
from zip_screen_aggregates import verdict                               # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "phecode_screen_negative")
DATES = os.path.join(OUT_DIR, "screen_answer_dates_local.csv")        # person level. Stays on the Workbench.
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
ENC = "gte-large-en-v1.5"
KINDS = ("before", "after")
# key (= the diagnosis key in phecode_sweep.CONDITIONS), the criterion's column in cutoff_flags.csv,
# what screening negative means, the criterion's items
PAIRS = [
    ("mdd", "PHQ-9", "PHQ-9 total < 10", PHQ),
    ("gad", "GAD-7", "GAD-7 total < 10", GAD),
    ("adhd", "ASRS", "fewer than 4 of the 6 ASRS Part A items at clinical frequency", ASRS_INATT + ASRS_HYPER),
    ("suicide", "Suicidal ideation or attempt", "ss_2 and ss_3 both answered No", SUICIDE_ITEMS),
]
PHECODE = {k: (c, lab) for k, c, lab, *_ in CONDITIONS}
# the screen's own score below its threshold, by cutoffs.py's formulas (name, function of the responses,
# the highest value a screen-negative person can have); criterion 6 has none
SCORES = {
    "mdd": ("PHQ-9 total", lambda R: R[PHQ].sum(axis=1).values, 9),
    "gad": ("GAD-7 total", lambda R: R[GAD].sum(axis=1).values, 9),
    "adhd": ("ASRS Part A items at clinical frequency",
             lambda R: (sum((R[i].values.astype(float) >= 2).astype(int) for i in ASRS_INATT)
                        + sum((R[i].values.astype(float) >= 3).astype(int) for i in ASRS_HYPER)), 3),
}


def answer_dates(idx):
    """Per person and pair: the latest date on which the pair's screen items were answered, and the
    earliest (for the spread check). Pulled once, then read from the person-level cache."""
    items = pd.read_csv(CFG.items_csv, usecols=["item", "concept_id"])
    cid = dict(zip(items["item"], items["concept_id"].astype("int64")))
    need = {}
    for key, _, _, its in PAIRS:
        absent = [i for i in its if i not in cid]
        assert not absent, f"{key}: {absent} not in {CFG.items_csv}"
        need[key] = [cid[i] for i in its]
    if os.path.exists(DATES) and "--refresh" not in sys.argv:
        D = pd.read_csv(DATES, dtype={ID_COL: str})
        print(f"  [answer dates: {len(D):,} rows read from {DATES}, nothing pulled]")
    else:
        from google.cloud import bigquery
        client = bigquery.Client()
        sql = f"""SELECT person_id, observation_source_concept_id AS concept_id,
                         MIN(observation_date) AS first_date, MAX(observation_date) AS last_date
                  FROM `{CDR}.observation`
                  WHERE person_id IN UNNEST(@ids) AND observation_source_concept_id IN UNNEST(@cids)
                  GROUP BY person_id, concept_id"""
        params = [bigquery.ArrayQueryParameter("ids", "INT64", [int(i) for i in idx]),
                  bigquery.ArrayQueryParameter("cids", "INT64", sorted({c for v in need.values() for c in v}))]
        dry = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params, dry_run=True,
                                                                    use_query_cache=False))
        print(f"  [answer dates: scanning {dry.total_bytes_processed / 1e9:,.1f} GB]", flush=True)
        E = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()
        E["person_id"] = E["person_id"].astype(str)
        E["concept_id"] = E["concept_id"].astype("int64")
        for c in ("first_date", "last_date"):
            E[c] = pd.to_datetime(E[c])
        rows = []
        for key, cids in need.items():
            e = (E[E["concept_id"].isin(cids)].groupby("person_id")
                 .agg(answer_date=("last_date", "max"), earliest=("first_date", "min"),
                      n_items_dated=("concept_id", "nunique")).reset_index())
            e["pair"] = key
            rows.append(e)
        D = pd.concat(rows, ignore_index=True).rename(columns={"person_id": ID_COL})
        D.to_csv(DATES, index=False)
        print(f"  [answer dates: {len(D):,} rows cached to {DATES} -- person level, do not download]")
    for c in ("answer_date", "earliest"):
        D[c] = pd.to_datetime(D[c])
    return D


def first_event_dates(idx):
    """Per diagnosis of the 4 pairs: PheTK's first_event_date per person (NaT without an event)."""
    C = pd.read_csv(COUNTS, sep="\t", dtype={"person_id": str, "phecode": str},
                    usecols=["person_id", "phecode", "count", "first_event_date"])
    codes = {PHECODE[k][0]: k for k, *_ in PAIRS}
    C = C[C["phecode"].isin(codes) & C["person_id"].isin(idx)]
    out = {}
    for code, key in codes.items():
        c = C[C["phecode"] == code].set_index("person_id")
        assert c.index.is_unique, f"{code}: more than 1 row per person in {COUNTS}"
        out[key] = (pd.to_datetime(c["first_event_date"]).reindex(idx).values,
                    c["count"].reindex(idx).fillna(0).values)
    return out


def fit_curve_score(age, sex, score, y, top):
    """y ~ 1 + age + age^2 + sex + the score as categories 1..top (reference 0), least squares, age
    centered and scaled on the fitting sample. Every category must hold more than 20 people."""
    a = (age - age.mean()) / age.std()
    cats = list(range(1, top + 1))
    for c in [0] + cats:
        assert not small((score == c).sum()) and (score == c).sum() > 0, \
            f"score category {c} holds {(score == c).sum()} people; the policy needs more than 20 in each"
    cols = [a, a ** 2, sex] + [(score == c).astype(float) for c in cats]
    fit = sm.OLS(y, sm.add_constant(np.column_stack(cols))).fit()
    terms = ["intercept", "age_z", "age_z_squared", "sex_male"] + [f"score_{c}" for c in cats]
    rows = [{"term": t, "coef": b, "se": s, "age_mean": age.mean(), "age_sd": age.std(), "n_people": len(y)}
            for t, b, s in zip(terms, fit.params, fit.bse)]
    return np.asarray(fit.fittedvalues), rows


def quantities(s):
    """The counts a reader could form in one row, per kind (see the policy in the docstring)."""
    A, B, n, K, R = s["A"], s["B"], s["n"], s["K"], s["R"]
    C = n - A - B
    return {"before": [A, n - A, K - A],
            "after": [B, n - B, K - B],
            "both": [C, A + B, K - A - B, (R - K) - C]}


def protect(st, tot):
    """Which (row, kind) shares to hide. st: decile -> counts; tot: the total row's counts (row 0)."""
    for j, s in list(st.items()) + [(0, tot)]:
        assert not small(s["n"]) and not small(s["R"] - s["n"]), \
            f"decile {j}: n = {s['n']}, R - n = {s['R'] - s['n']}; the policy's checks rest on both exceeding 20"
    hid = set()
    qt = quantities(tot)
    for kind in KINDS:                                     # the total row, against the cohort
        if any(small(q) for q in qt[kind]):
            hid |= {(j, kind) for j in [0] + DECILES}
    if not hid and any(small(q) for q in qt["both"]):
        kind = "before" if tot["A"] <= tot["B"] else "after"
        hid |= {(j, kind) for j in [0] + DECILES}
    q = {j: quantities(st[j]) for j in DECILES}
    changed = True
    while changed:
        changed = False
        for kind in KINDS:
            forced = [(j, kind) in hid for j in DECILES]
            parts = [[q[j][kind][i] for j in DECILES] for i in range(len(q[DECILES[0]][kind]))]
            for j, x in zip(DECILES, hide_cells(parts, forced)):
                if x and (j, kind) not in hid:
                    hid.add((j, kind))
                    changed = True
        forced = [any((j, k) in hid for k in KINDS) for j in DECILES]
        parts = [[q[j]["both"][i] for j in DECILES] for i in range(len(q[DECILES[0]]["both"]))]
        for j, x, f in zip(DECILES, hide_cells(parts, forced), forced):
            if x and not f:
                hid.add((j, "before" if st[j]["A"] <= st[j]["B"] else "after"))
                changed = True
    return hid


def main():
    os.makedirs(OUT, exist_ok=True)
    assert ENCODERS[0] == ENC, f"breadth_sweep.ENCODERS[0] is {ENCODERS[0]}, expected {ENC}"
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"))
    cf[ID_COL] = cf[ID_COL].astype(str)
    idx = pd.Index(cf[ID_COL], name=ID_COL)                   # the order every breadth script uses
    N = len(idx)
    A = load(idx)
    has_rec = A["has_rec"]
    order = rank_orders([ENC], idx)[ENC]
    M = decile_masks(order, N)
    top90 = np.zeros(N, bool)
    top90[order[:int(round(N * 0.1))]] = True
    assert (M[10] == top90).all(), "decile 10 is not Stage 1's top set at the 90th"
    D = answer_dates(idx)
    FE = first_event_dates(idx)
    Rsp = pd.read_csv(CFG.responses_csv, usecols=[ID_COL] + PHQ + GAD + ASRS_INATT + ASRS_HYPER)
    Rsp[ID_COL] = Rsp[ID_COL].astype(str)
    Rsp = Rsp.set_index(ID_COL).reindex(idx)
    print(f"[phecode_screen_negative] cohort {N:,} | with at least 1 ICD event {int(has_rec.sum()):,} | {ENC}")

    rows, curves, pair_meta = [], [], []
    for key, crit, rule, items in PAIRS:
        code, label = PHECODE[key]
        v = pd.to_numeric(cf[crit], errors="coerce").values        # 1 met, 0 not met, NaN withheld
        neg_all = v == 0                                           # S: screens negative, records or not
        neg = neg_all & has_rec
        Dk = D[D["pair"] == key].set_index(ID_COL)
        assert Dk.index.is_unique, f"{key}: more than 1 answer date per person"
        adate = Dk["answer_date"].reindex(idx).values
        early = Dk["earliest"].reindex(idx).values
        assert not pd.isna(adate[neg_all]).any(), \
            f"{key}: {int(pd.isna(adate[neg_all]).sum())} screen-negative people have no answer date for the screen"
        fe, cnt = FE[key]
        case = has_rec & (cnt >= MIN_COUNT)
        y = A[f"y_{key}"]
        assert (case == (np.nan_to_num(y, nan=0) == 1)).all(), f"{key}: the case labels differ from phecode_sweep.load"
        assert not pd.isna(fe[case]).any(), f"{key}: a case without a first_event_date"
        before = case & (fe <= adate)                              # the survey day counts as before
        after = case & (fe > adate)
        assert ((before | after) == case)[neg].all() and not (before & after).any()

        spread = (adate[neg] - early[neg]).astype("timedelta64[D]").astype(float)
        print(f"\n  {crit} <-> {code} {label}: screen-negative {int(neg_all.sum()):,}, of them with records "
              f"{int(neg.sum()):,}; screen items answered over more than 1 day: {100 * np.mean(spread > 0):.2f}%, "
              f"over more than 30 days: {100 * np.mean(spread > 30):.2f}% (printed only)")

        score = None
        if key in SCORES:
            sname, sfun, top = SCORES[key]
            score = np.asarray(sfun(Rsp), dtype=float)
            sc = score[neg_all]
            assert not np.isnan(sc).any(), f"{key}: a screen-negative person has no {sname}"
            assert (sc == np.round(sc)).all() and sc.min() >= 0 and sc.max() <= top, \
                f"{key}: {sname} of a screen-negative person is outside 0 to {top}; cutoffs.py disagrees"
        gap, share, gaps = {}, {}, {}
        for kind, yk in (("before", before), ("after", after)):
            yf = yk.astype(float)
            fitted, cr = fit_curve(A["age"][neg], A["sex"][neg], yf[neg])
            gap[kind] = np.full(N, np.nan)
            gap[kind][neg] = yf[neg] - fitted
            share[kind] = float(yf[neg].mean())
            curves += [{"pair": key, "criterion": crit, "phecode": code, "kind": kind,
                        "adjustment": "age, age^2, sex", **r} for r in cr]
            if score is not None:
                fitted2, cr2 = fit_curve_score(A["age"][neg], A["sex"][neg], score[neg], yf[neg], top)
                gaps[kind] = np.full(N, np.nan)
                gaps[kind][neg] = yf[neg] - fitted2
                curves += [{"pair": key, "criterion": crit, "phecode": code, "kind": kind,
                            "adjustment": f"age, age^2, sex, {sname} (categories)", **r} for r in cr2]

        def counts(m, ref):
            s = {"n": int(m.sum()), "A": int(before[m].sum()), "B": int(after[m].sum()),
                 "R": int((ref & has_rec).sum()), "K": int((ref & case).sum())}
            for kind in KINDS:
                g = gap[kind][m]
                s[f"gap_{kind}"], s[f"se_{kind}"] = float(g.mean()), float(np.std(g, ddof=1) / np.sqrt(len(g)))
                if score is not None:
                    g = gaps[kind][m]
                    s[f"gaps_{kind}"], s[f"ses_{kind}"] = float(g.mean()), float(np.std(g, ddof=1) / np.sqrt(len(g)))
            if score is not None:
                s["score_mean"], s["score_sd"] = float(score[m].mean()), float(np.std(score[m], ddof=1))
            return s

        tot = counts(neg, np.ones(N, bool))
        assert not small(int(neg_all.sum()) - tot["n"]), f"{key}: S - n is 1 to 20"
        st = {j: counts(neg & M[j], M[j]) for j in DECILES}
        assert sum(st[j]["n"] for j in DECILES) == tot["n"] and sum(st[j]["A"] for j in DECILES) == tot["A"]
        hid = protect(st, tot)

        for j, s in [(0, tot)] + [(j, st[j]) for j in DECILES]:
            r = {"pair": key, "criterion": crit, "screen_negative": rule, "phecode": code, "diagnosis": label,
                 "decile": j, "n": s["n"], "score": SCORES[key][0] if score is not None else "",
                 "score_mean": s.get("score_mean", np.nan), "score_sd": s.get("score_sd", np.nan)}
            for kind, k in (("before", s["A"]), ("after", s["B"])):
                if (j, kind) in hid:
                    continue
                p, lo, hi = wilson(k, s["n"])
                m_ = share[kind] + s[f"gap_{kind}"]
                se = s[f"se_{kind}"]
                r.update({f"{kind}_rate": p, f"{kind}_lo": lo, f"{kind}_hi": hi,
                          f"{kind}_adj": m_, f"{kind}_adj_lo": m_ - Z * se, f"{kind}_adj_hi": m_ + Z * se})
                if score is not None:                  # written only where the unadjusted share is
                    m2, se2 = share[kind] + s[f"gaps_{kind}"], s[f"ses_{kind}"]
                    r.update({f"{kind}_adjs": m2, f"{kind}_adjs_lo": m2 - Z * se2, f"{kind}_adjs_hi": m2 + Z * se2})
            rows.append(r)
        n_hid = {kind: sum((j, kind) in hid for j in [0] + DECILES) for kind in KINDS}
        pair_meta.append({"pair": key, "criterion": crit, "screen_negative": rule, "items": items,
                          "phecode": code, "diagnosis": label, "shares_withheld_of_11": n_hid})

        def pct(kind, s, j):
            return "    -" if (j, kind) in hid else f"{100 * s['A' if kind == 'before' else 'B'] / s['n']:5.1f}"

        def pcts(kind, s, j):
            if score is None or (j, kind) in hid:
                return "    -"
            return f"{100 * (share[kind] + s[f'gaps_{kind}']):5.1f}"
        print(f"    decile        n   before%  after%   | standardized for age, sex and "
              f"{SCORES[key][0] if score is not None else '(no score)'}: before%  after%   score mean  (- = withheld)")
        for j, s in [(0, tot)] + [(j, st[j]) for j in DECILES]:
            sm_ = f"{s['score_mean']:5.2f}" if score is not None else "    -"
            print(f"    {'all' if j == 0 else j:>6} {s['n']:>8,}   {pct('before', s, j)}   {pct('after', s, j)}   |"
                  f"{'':>45}{pcts('before', s, j)}   {pcts('after', s, j)}   {sm_}")
        print(f"    withheld: before {n_hid['before']}, after {n_hid['after']} of 11 rows")

    R_ = pd.DataFrame(rows)
    for kind in KINDS:
        for w in ("rate", "lo", "hi", "adj", "adj_lo", "adj_hi", "adjs", "adjs_lo", "adjs_hi"):
            if f"{kind}_{w}" not in R_.columns:
                R_[f"{kind}_{w}"] = np.nan
    for key, *_ in PAIRS:                                  # the gaps average 0 over each pair's denominator
        t = R_[(R_.pair == key) & (R_.decile == 0)].iloc[0]
        for kind in KINDS:
            if pd.notna(t[f"{kind}_rate"]):
                assert abs(t[f"{kind}_adj"] - t[f"{kind}_rate"]) < 1e-9, f"{key} {kind}: the total's adjusted share moved"
                if key in SCORES:
                    assert abs(t[f"{kind}_adjs"] - t[f"{kind}_rate"]) < 1e-9, f"{key} {kind}: the total's standardized share moved"
    R_.to_csv(os.path.join(OUT, "phecode_screen_negative.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(OUT, "phecode_screen_negative_curves.csv"), index=False)
    json.dump({"cdr": CDR, "grouping": ENC, "pairs": pair_meta,
               "screens_negative": "the criterion evaluable (every item answered, none imputed) and not met, "
                                   "and at least 1 ICD-coded event",
               "case_rule": f"PheTK count >= {MIN_COUNT} (phecodeX X1.0, PheTK 0.3.6); 1 event = not a case",
               "timing": "before = first_event_date on or before the answer date (the survey day counts as before); "
                         "after = later",
               "answer_date": "per pair, the latest observation_date of the screen's own items",
               "deciles": "the cohort's deciles of gte's distance (phecode_deciles.py), kept to the screen-negative "
                          "people; decile 0 = all of them; 1 = least atypical, 10 = most",
               "adjustment": "per pair and kind, OLS on age, age^2, sex fitted once over the pair's screen-negative "
                             "people; value = their share + the decile's mean gap",
               "standardized_for_score": "the same, plus the screen's own score below its threshold as categories "
                                         "(reference 0): PHQ-9 total 0-9, GAD-7 total 0-9, ASRS Part A count 0-3; "
                                         "none for criterion 6 (screening negative = both answers No)",
               "intervals": "Wilson 95% unadjusted; normal 95% of the gaps adjusted, the curve treated as known",
               "withheld": "partitions of the deciles with known totals, against the cohort's R and K; see the "
                           "script's docstring", "disclosure": "All of Us dissemination policy (disclosure.py)"},
              open(os.path.join(OUT, "phecode_screen_negative_meta.json"), "w"), indent=2)
    files = ["phecode_screen_negative.csv", "phecode_screen_negative_curves.csv", "phecode_screen_negative_meta.json"]
    for fn in files:
        ok, why = verdict(os.path.join(OUT, fn))
        assert ok, f"{fn} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"phecode_screen_negative_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"phecode_screen_negative/{fn}")
    print(f"\n[phecode_screen_negative] {len(R_)} rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
