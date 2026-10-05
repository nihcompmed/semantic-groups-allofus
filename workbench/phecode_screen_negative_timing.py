#!/usr/bin/env python3
"""phecode_screen_negative_timing.py -- the timing of "on or before the survey". For the people each screen
does not flag and who carry the matching diagnosis, how long before (or after) the day they answered that
screen the diagnosis was coded.

Runs in the MAIN environment, after phecode_event_dates.py (PheTK environment) has written the dated
events, and after phecode_screen_negative.py (the same people, whose published counts it must reproduce).

THE PEOPLE AND LABELS are phecode_screen_negative.py's, for its four pairs: screen negative (the criterion
evaluable and not met, at least 1 ICD-coded event), a case = PheTK count >= 2, "before" = first coded on or
before the answer date of that screen (the survey day counts as before), "after" = later.

FIVE MEASURES, in days (both dates):
  last_before     before cases: answer date minus the MOST RECENT code on or before it
                  (whether the diagnosis was still being recorded around the survey)
  first_before    before cases: answer date minus the FIRST code (how long it has been in the record)
  first_after     after cases: the first code minus the answer date
  record_reach    all screen-negative people with records: answer date minus their first ICD event of
                  any kind (how far back the record goes; nobody can have a code older than their record)
  record_follow   the same people: their last ICD event of any kind minus the answer date (how far the
                  record runs after the survey)
Bands (days): before 0 | 1-30 | 31-182 | 183-365 | 366-730 | 731-1826 | 1827-3652 | > 3652;
  after 1-30 | 31-182 | 183-365 | 366-730 | > 730; reach < 0 (starts after the survey) | 0-365 | 366-730 |
  731-1826 | 1827-3652 | > 3652; follow < 0 (ends before the survey) | 0-182 | 183-365 | 366-730 | > 730.

WHAT IS WRITTEN
  phecode_screen_negative_timing_bands.csv     per pair and measure, over ALL the pair's screen-negative
                                               people: each band's count and share, and the percentiles
                                               (10, 25, 50, 75, 90) of the measure
  phecode_screen_negative_timing_deciles.csv   per pair, measure and cohort decile of gte's distance: the
                                               same percentiles, and for last_before the number and share
                                               of before cases with a code in the 365 days up to the survey;
                                               decile 0 = all the pair's screen-negative people
  phecode_screen_negative_timing_meta.json
  Percentiles are written; no minimum or maximum is (as phecode_coverage.py).

THE DISSEMINATION POLICY (disclosure.py)
  - Bands partition a total that phecode_screen_negative.csv already publishes (the before cases, the
    after cases, or n). A band of 1 to 20 is hidden, and more bands, smallest first, until the hidden
    bands sum to 0 or to more than 20 (phecode_deciles.complete). If that total is not published there, the
    whole measure is withheld.
  - ★ The two before measures cover the SAME people, and the first code is never later than the most recent
    one, so at every band edge the cumulative count of last_before contains that of first_before. Where
    both are derivable, their difference is a count, and a difference of 1 to 20 hides one more band.
  - Decile split (within 365 days / older) of last_before: each row partitions that decile's before cases,
    published only where phecode_screen_negative.csv wrote that decile's before share (else the row is
    withheld), and each column partitions the overall count. Rows and columns are completed in turn until
    nothing changes. The cohort's cases in the decile minus either cell (K - X, K - Y) must exceed 20.
  - Decile percentiles are written only where the decile's before (or after) share was written, and never
    for a group of 20 or fewer.
The files are checked by zip_screen_aggregates.verdict().

INPUTS   screen_out/phetk/phecode_event_dates_local.csv (phecode_event_dates.py); screen_out/phetk/
         icd_person_summary_local.csv; everything phecode_screen_negative.py reads; and its output
         screen_out/phecode_screen_negative/phecode_screen_negative.csv
OUTPUTS  screen_out/phecode_screen_negative_timing/  the 3 files above
         screen_out/phecode_screen_negative_timing_<CDR>.zip   those 3, checked by verdict()

Run:  python3 phecode_screen_negative_timing.py
"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from breadth_exclusive import rank_orders                                      # noqa: E402
from disclosure import MARK, SUPP, small                                       # noqa: E402  the policy
from phecode_deciles import DECILES, complete, decile_masks                    # noqa: E402
from phecode_screen_negative import (DATES, ENC, MIN_COUNT, PAIRS, PHECODE,    # noqa: E402  the same people
                                     answer_dates, first_event_dates, load)
from zip_screen_aggregates import verdict                                      # noqa: E402

OUT_DIR = os.path.join(HERE, "screen_out")
OUT = os.path.join(OUT_DIR, "phecode_screen_negative_timing")
EVENTS = os.path.join(OUT_DIR, "phetk", "phecode_event_dates_local.csv")
PERSON = os.path.join(OUT_DIR, "phetk", "icd_person_summary_local.csv")
PUBLISHED = os.path.join(OUT_DIR, "phecode_screen_negative", "phecode_screen_negative.csv")
CDR = os.environ.get("WORKSPACE_CDR", "")
ID_COL = "id"
INF = np.inf
BANDS = {
    "before": [(0, 0, "on the survey day"), (1, 30, "1-30 days"), (31, 182, "1-6 months"),
               (183, 365, "6-12 months"), (366, 730, "1-2 years"), (731, 1826, "2-5 years"),
               (1827, 3652, "5-10 years"), (3653, INF, "over 10 years")],
    "after": [(1, 30, "1-30 days"), (31, 182, "1-6 months"), (183, 365, "6-12 months"),
              (366, 730, "1-2 years"), (731, INF, "over 2 years")],
    "reach": [(-INF, -1, "record starts after the survey"), (0, 365, "0-1 year"), (366, 730, "1-2 years"),
              (731, 1826, "2-5 years"), (1827, 3652, "5-10 years"), (3653, INF, "over 10 years")],
    "follow": [(-INF, -1, "record ends before the survey"), (0, 182, "0-6 months"), (183, 365, "6-12 months"),
               (366, 730, "1-2 years"), (731, INF, "over 2 years")],
}
MEASURES = [("last_before", "before"), ("first_before", "before"), ("first_after", "after"),
            ("record_reach", "reach"), ("record_follow", "follow")]
PCTS = [10, 25, 50, 75, 90]
WITHIN = 365                                   # the decile split of last_before


def band_counts(x, bands):
    return np.array([int(((x >= lo) & (x <= hi)).sum()) for lo, hi, _ in bands], dtype=np.int64)


def pct(x):
    return {f"p{p}_days": float(np.percentile(x, p)) for p in PCTS}


def protect_bands(counts, total_published):
    """Hidden bands per measure. counts: measure -> band counts. total_published: measure -> bool."""
    hid = {m: (np.ones(len(c), bool) if not total_published[m] else np.zeros(len(c), bool))
           for m, c in counts.items()}
    changed = True
    while changed:
        changed = False
        for m, c in counts.items():
            h = complete(c, hid[m])
            if (h != hid[m]).any():
                hid[m], changed = h, True
        L, F = counts["last_before"], counts["first_before"]          # nested cumulative counts
        for e in range(len(L) - 1):
            def derivable(m):
                return (not hid[m][:e + 1].any()) or (not hid[m][e + 1:].any())
            if derivable("last_before") and derivable("first_before") and not hid["first_before"].all():
                d = int(L[:e + 1].sum() - F[:e + 1].sum())
                if small(d):
                    j = e if not hid["first_before"][e] else int(np.flatnonzero(~hid["first_before"])[0])
                    hid["first_before"][j] = True
                    changed = True
    return hid


def protect_split(X, Y, K, row_ok):
    """Hidden decile rows of the within-365 split. X, Y: per decile; K: the cohort's cases per decile;
    row_ok: the decile's before share was published (its total A = X + Y is known)."""
    hide = np.array([not ok for ok in row_ok], bool)
    hide |= np.array([small(x) or small(y) or small(k - x) or small(k - y) for x, y, k in zip(X, Y, K)])
    while True:
        new = hide | complete(X, hide) | complete(Y, hide)
        if (new == hide).all():
            return hide
        hide = new


def main():
    os.makedirs(OUT, exist_ok=True)
    for f, what in ((EVENTS, "phecode_event_dates.py (in ~/phetk_env)"), (PUBLISHED, "phecode_screen_negative.py"),
                    (DATES, "phecode_screen_negative.py")):
        assert os.path.exists(f), f"{f} is missing: run {what} first"
    cf = pd.read_csv(os.path.join(OUT_DIR, "cutoff_flags.csv"))
    cf[ID_COL] = cf[ID_COL].astype(str)
    idx = pd.Index(cf[ID_COL], name=ID_COL)
    N = len(idx)
    A = load(idx)
    has_rec = A["has_rec"]
    order = rank_orders([ENC], idx)[ENC]
    Mdec = decile_masks(order, N)
    dec = np.zeros(N, int)
    for j in DECILES:
        dec[Mdec[j]] = j
    D = answer_dates(idx)
    FE = first_event_dates(idx)
    Ps = pd.read_csv(PERSON, dtype={"person_id": str}, usecols=["person_id", "first_date", "last_date"])
    Ps = Ps.set_index("person_id").reindex(idx)
    rec_first = pd.to_datetime(Ps["first_date"]).values
    rec_last = pd.to_datetime(Ps["last_date"]).values
    E = pd.read_csv(EVENTS, dtype={"person_id": str, "phecode": str})
    E["date"] = pd.to_datetime(E["date"])
    pub = pd.read_csv(PUBLISHED)
    print(f"[phecode_screen_negative_timing] cohort {N:,} | dated events {len(E):,} | {ENC}")

    band_rows, dec_rows, meta = [], [], {}
    for key, crit, rule, _ in PAIRS:
        code, label = PHECODE[key]
        v = pd.to_numeric(cf[crit], errors="coerce").values
        neg = (v == 0) & has_rec
        adate = D[D["pair"] == key].set_index(ID_COL)["answer_date"].reindex(idx).values
        fe, cnt = FE[key]
        case = has_rec & (cnt >= MIN_COUNT)
        before = neg & case & (fe <= adate)
        after = neg & case & (fe > adate)
        # the dated events: the first date must be PheTK's; the most recent one on or before the survey
        e = E[E["phecode"] == code]
        e = e[e["person_id"].isin(idx[before | after])]
        first_here = e.groupby("person_id")["date"].min().reindex(idx).values
        assert (first_here[before | after] == fe[before | after]).all(), f"{key}: event dates disagree with PheTK's first date"
        ad = pd.Series(adate, index=idx)
        e = e.assign(adate=ad.reindex(e["person_id"]).values)
        last_on_before = e[e["date"] <= e["adate"]].groupby("person_id")["date"].max().reindex(idx).values
        assert not pd.isna(last_on_before[before]).any(), f"{key}: a before case with no code on or before the survey"
        days = lambda a, b: (a - b).astype("timedelta64[D]").astype(float)
        meas = {"last_before": (days(adate, last_on_before), before, "before"),
                "first_before": (days(adate, fe), before, "before"),
                "first_after": (days(fe, adate), after, "after"),
                "record_reach": (days(adate, rec_first), neg, "reach"),
                "record_follow": (days(rec_last, adate), neg, "follow")}
        assert (meas["last_before"][0][before] <= meas["first_before"][0][before]).all()
        assert (meas["first_after"][0][after] >= 1).all()

        # the totals, which phecode_screen_negative.csv publishes (or withholds)
        t = pub[(pub.pair == key) & (pub.decile == 0)].iloc[0]
        assert int(t["n"]) == int(neg.sum()), f"{key}: n differs from the published file"
        published = {"before": pd.notna(t["before_rate"]), "after": pd.notna(t["after_rate"]),
                     "reach": True, "follow": True}
        for kind, cell in (("before", before), ("after", after)):
            if published[kind]:
                assert round(t[f"{kind}_rate"] * t["n"]) == int(cell.sum()), f"{key}: {kind} cases differ"

        counts = {m: band_counts(meas[m][0][meas[m][1]], BANDS[b]) for m, b in MEASURES}
        hid = protect_bands(counts, {m: published[b] for m, b in MEASURES})
        meta[key] = {"criterion": crit, "phecode": code, "diagnosis": label, "screen_negative": rule, "measures": {}}
        for m, b in MEASURES:
            x = meas[m][0][meas[m][1]]
            n_grp = int(meas[m][1].sum())
            ok = published[b]
            meta[key]["measures"][m] = {"written": bool(ok), "bands_hidden": int(hid[m].sum())}
            for (lo, hi, lab), c, h in zip(BANDS[b], counts[m], hid[m]):
                r = {"pair": key, "criterion": crit, "phecode": code, "measure": m, "band": lab,
                     "band_lo_days": lo, "band_hi_days": hi, "n_group": n_grp if ok else np.nan}
                if ok:
                    r["n_band"] = (MARK if small(c) else SUPP) if h else int(c)
                    if not h and n_grp > 0:
                        r["share"] = c / n_grp
                band_rows.append(r)
            if ok and n_grp > 20:
                meta[key]["measures"][m].update(pct(x))

        # by decile: percentiles, and the within-365 split of last_before
        dec_pub = pub[(pub.pair == key) & (pub.decile > 0)].set_index("decile")
        K = np.array([int((case & (dec == j)).sum()) for j in DECILES])
        X = np.array([int((before & (dec == j) & (meas["last_before"][0] <= WITHIN)).sum()) for j in DECILES])
        Y = np.array([int((before & (dec == j)).sum()) for j in DECILES]) - X
        row_ok = [bool(published["before"] and pd.notna(dec_pub.at[j, "before_rate"])) for j in DECILES]
        split_hidden = protect_split(X, Y, K, row_ok)
        for m, b in MEASURES:                          # decile 0 = all the pair's screen-negative people
            grp = meas[m][1]
            r = {"pair": key, "criterion": crit, "phecode": code, "measure": m, "decile": 0,
                 "n_group": int(grp.sum()) if published[b] else np.nan}
            if published[b] and grp.sum() > 20:
                r.update(pct(meas[m][0][grp]))
            dec_rows.append(r)
        for i, j in enumerate(DECILES):
            for m, b in MEASURES:
                grp = meas[m][1] & (dec == j)
                n_grp = int(grp.sum())
                if b in ("before", "after"):
                    ok = published[b] and pd.notna(dec_pub.at[j, f"{b}_rate"])
                else:
                    ok = True
                r = {"pair": key, "criterion": crit, "phecode": code, "measure": m, "decile": j,
                     "n_group": n_grp if ok else np.nan}
                if ok and n_grp > 20:
                    r.update(pct(meas[m][0][grp]))
                if m == "last_before" and not split_hidden[i]:
                    r.update({"n_within_365": int(X[i]), "share_within_365": X[i] / (X[i] + Y[i])})
                dec_rows.append(r)
        meta[key]["split_rows_hidden"] = int(split_hidden.sum())

        # ★ the log leaves the Workbench with the zip, so it prints only what the files publish: a case
        # count is printed only where phecode_screen_negative.csv publishes its share
        shown = lambda kind, cell: f"{int(cell.sum()):,}" if published[kind] else "withheld"
        print(f"\n  {crit} <-> {code}: screen-negative with records {int(neg.sum()):,}; before cases "
              f"{shown('before', before)}, after cases {shown('after', after)}")
        for m, b in MEASURES:
            sh = [("   -" if h else f"{100 * c / max(1, meas[m][1].sum()):4.1f}") for c, h in zip(counts[m], hid[m])]
            print(f"    {m:14s} " + " ".join(f"{s:>5}" for s in sh) + "   % by band (- = withheld)")
        print("    within 365 days, % of before cases by decile: " +
              " ".join("   -" if split_hidden[i] else f"{100 * X[i] / (X[i] + Y[i]):4.1f}" for i in range(10)))

    B_ = pd.DataFrame(band_rows)
    Dd = pd.DataFrame(dec_rows)
    B_.to_csv(os.path.join(OUT, "phecode_screen_negative_timing_bands.csv"), index=False)
    Dd.to_csv(os.path.join(OUT, "phecode_screen_negative_timing_deciles.csv"), index=False)
    json.dump({"cdr": CDR, "grouping": ENC, "pairs": meta,
               "measures": {"last_before": "answer date minus the most recent code on or before it (before cases)",
                            "first_before": "answer date minus the first code (before cases)",
                            "first_after": "first code minus the answer date (after cases)",
                            "record_reach": "answer date minus the first ICD event of any kind (all screen-negative with records)",
                            "record_follow": "last ICD event of any kind minus the answer date (the same people)"},
               "bands_days": {k: [[lo, hi, lab] for lo, hi, lab in v] for k, v in
                              {k2: [(None if lo == -INF else lo, None if hi == INF else hi, lab) for lo, hi, lab in v2]
                               for k2, v2 in BANDS.items()}.items()},
               "decile_split_days": WITHIN, "percentiles": PCTS,
               "events": "PheTK's own ICD events (phecode_event_dates.py), checked against its saved counts",
               "disclosure": "All of Us dissemination policy (disclosure.py); see the docstring"},
              open(os.path.join(OUT, "phecode_screen_negative_timing_meta.json"), "w"), indent=2)
    files = ["phecode_screen_negative_timing_bands.csv", "phecode_screen_negative_timing_deciles.csv",
             "phecode_screen_negative_timing_meta.json"]
    for fn in files:
        ok, why = verdict(os.path.join(OUT, fn))
        assert ok, f"{fn} refused by the download check: {why}"
    zpath = os.path.join(OUT_DIR, f"phecode_screen_negative_timing_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(OUT, fn), arcname=f"phecode_screen_negative_timing/{fn}")
    print(f"\n[phecode_screen_negative_timing] {len(B_)} band rows, {len(Dd)} decile rows\n  download -> {zpath}")


if __name__ == "__main__":
    main()
