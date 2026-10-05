#!/usr/bin/env python3
"""exemplar_encoder.py -- criteria breadth and the attribution profiles for ONE encoder's own flagged set.

WHY THIS EXISTS. The analysis runs on a single prespecified encoder, `gte-large-en-v1.5`, not on a
consensus across encoders. This script computes everything downstream of the screen -- criteria
breadth, the attribution profiles, the no-criterion group -- on the exemplar's OWN flagged set.

WHAT IT DOES NOT CHANGE. The screen itself: the distance, the covariate residualization, the
Ledoit-Wolf precision and the 95th-percentile threshold are screen_battery.py's, copied here, and
the flagged set is checked against `scores_<encoder>.csv` before anything is reported. The
attribution math is copied here with its assertions.

★ THE CONSTRUCT LAYER IS COMPUTED HERE ON PURPOSE, AND ONLY AS A MEAN OVER A SUBSET. It is needed
because the claim is about what content carries the participants who meet no criterion. It is
reported as the MEAN SHARE OF T^2 OVER THE SUBSET and never as a per-person leading construct: the
per-person argmax moved for a few dozen people on a fresh pull of the data, because k-NN
imputation breaks ties differently and an argmax over close shares flips. A mean over a group does
not have that failure mode.

DISCLOSURE RULES THIS OBEYS. Every output is an aggregate. No person-level file is written. The
dissemination policy is applied through disclosure.py: a count of 1 to 20 is
written "<=20", cells that would let one be derived by subtraction are written "suppressed", and a
percent or mean that would give one back is blanked. Rows are kept so the suppression is visible.

Reads   responses_out_<stem>/responses_<cohort>.csv, weights_<cohort>.csv   (CFG)
        <items>.csv                       item, construct, scale_min, scale_max, reverse_coded
        basis/<encoder>.npz               W, mu
        basis/item_embeddings/<prefix>_<encoder>.npz
        screen_out/scores_<encoder>.csv   the screen's own flagged column, for the cross-check
        cutoff_flags.csv                  breadth, all_evaluable, none_met    (from cutoffs.py)
        cutoff_summary.json               n_criteria -- the size of the criteria set, read not assumed

Writes  exemplar_<encoder>_summary.json
        exemplar_<encoder>_breadth.csv          criteria met, flagged against cohort
        exemplar_<encoder>_groups.csv           group layer, flagged set, with each group's core items
        exemplar_<encoder>_constructs.csv       construct layer, flagged set
        exemplar_<encoder>_nocriterion_groups.csv
        exemplar_<encoder>_nocriterion_constructs.csv

Run     python3 exemplar_encoder.py
        python3 exemplar_encoder.py --encoder bge-large-en-v1.5      # any of the six
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_config import CFG   # noqa: E402
from disclosure import MARK, SUPP, marked, one, pair_hidden, partition_mask, small   # noqa: E402

ID_COL = "id"
COVARIATES = ["age", "sex"]
PCTILE = 95.0
DEFAULT_ENCODER = "gte-large-en-v1.5"


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def residualizer(C, w):
    """Project the covariates and an intercept out of any column block."""
    A = np.column_stack([np.ones(len(w))] + ([C] if C is not None and C.shape[1] else []))
    sw = np.sqrt(w)[:, None]

    def apply(V):
        beta, *_ = np.linalg.lstsq(A * sw, V * sw, rcond=None)
        return V - A @ beta
    return apply


def cores_of(P, items):
    """Core items of each group: keep the round(PR) largest entries by absolute value (eMethods 1)."""
    out = []
    for j in range(P.shape[1]):
        v = P[:, j]
        pr = (v ** 2).sum() ** 2 / (v ** 4).sum()
        k = max(1, int(round(pr)))
        idx = np.argsort(-np.abs(v))[:k]
        out.append([items[i] for i in sorted(idx, key=lambda i: -abs(v[i]))])
    return out


def main():
    enc = arg("--encoder", DEFAULT_ENCODER)
    CFG.describe("exemplar_encoder")
    print(f"[exemplar] encoder = {enc}\n")

    items = pd.read_csv(CFG.items_csv)["item"].astype(str).tolist()
    meta = pd.read_csv(CFG.items_csv).set_index("item")
    con = [str(meta["construct"].get(it, "unlabelled")) for it in items]
    scales = meta.reindex(items).rename(
        columns={"scale_min": "min", "scale_max": "max", "reverse_coded": "reverse"})

    R = pd.read_csv(CFG.responses_csv)
    R[ID_COL] = R[ID_COL].astype(str)
    R = R.set_index(ID_COL)
    raw = R[items].values.astype(float)
    assert not np.isnan(raw).any(), "missing responses -- use the k0 file"

    lo = scales["min"].values.astype(float)
    hi = scales["max"].values.astype(float)
    rev = scales["reverse"].astype(str).str.lower().eq("true").values
    X = np.clip(2.0 * (raw - lo) / (hi - lo) - 1.0, -1.0, 1.0)
    X[:, rev] *= -1.0

    cov = R[COVARIATES].values.astype(float) if COVARIATES else None
    w = (pd.read_csv(CFG.weights_csv).iloc[:, 0].values.astype(float)
         if CFG.weights_csv and os.path.exists(CFG.weights_csv) else np.ones(len(R)))
    resid = residualizer(cov, w)
    Xr = resid(X)

    # ---- the screen, for this encoder only. screen_battery.py's arithmetic, copied.
    b = np.load(os.path.join(CFG.basis_dir, f"{enc}.npz"), allow_pickle=True)
    W, mu = b["W"], b["mu"]
    e = np.load(os.path.join(CFG.emb_dir, f"{CFG.emb_prefix}_{enc}.npz"), allow_pickle=True)
    pos = {str(x): j for j, x in enumerate(e["items"])}
    E = np.asarray(e["vectors"], float)[[pos[it] for it in items]]
    P = (E - mu) @ W.T                                     # items x k
    k = P.shape[1]

    Y = resid(X @ P)
    Yc = Y - np.average(Y, axis=0, weights=w)
    lw = LedoitWolf().fit(Yc)
    Om, Sig = lw.precision_, lw.covariance_
    G = Yc @ Om
    T2 = np.einsum("ij,ij->i", Yc, G)
    d = np.sqrt(np.maximum(T2, 0.0))
    thr = float(np.percentile(d, PCTILE))
    flag = d > thr
    ids = R.index.to_numpy()

    # ---- cross-check against the screen's own file before anything is reported
    sp = os.path.join(CFG.out_dir, f"scores_{enc}.csv")
    if os.path.exists(sp):
        S = pd.read_csv(sp)
        S[ID_COL] = S[ID_COL].astype(str)
        S = S.set_index(ID_COL).reindex(R.index)
        dd = float(np.abs(S["distance"].values - d).max())
        same = int((S["flagged"].astype(bool).values == flag).sum())
        print(f"cross-check against scores_{enc}.csv: max |distance| diff {dd:.2e}, "
              f"flagged agrees on {same:,} of {len(flag):,}")
        assert dd < 1e-8, "recomputed distance does not match the screen"
        assert same == len(flag), "recomputed flag does not match the screen"
    else:
        print(f"[warn] {sp} not found -- the cross-check against the screen did not run")

    n_flag = int(flag.sum())
    print(f"\ncohort {len(R):,} | k = {k} | 95th percentile of the distance = {thr:.4f} "
          f"| flagged {n_flag:,} ({100 * n_flag / len(R):.2f}%)")

    # ---- criteria met, flagged against cohort, on the people with every criterion evaluable
    # ★ The size of the criteria set is READ from cutoff_summary.json rather than assumed here.
    # Nothing below hardcodes a count.
    cf = pd.read_csv(os.path.join(CFG.out_dir, "cutoff_flags.csv"))
    cf[ID_COL] = cf[ID_COL].astype(str)
    cf = cf.set_index(ID_COL).reindex(R.index)
    ev = cf["all_evaluable"].astype(bool).values
    br = cf["breadth"].astype(float).values
    none_met = cf["none_met"].astype(bool).values
    with open(os.path.join(CFG.out_dir, "cutoff_summary.json")) as fh:
        NB = int(json.load(fh)["n_criteria"])

    rows = []
    for nb in range(NB + 1):
        nc = int(((br == nb) & ev).sum())
        nf = int(((br == nb) & ev & flag).sum())
        rows.append(dict(breadth=nb, n_cohort=nc, pct_cohort=round(100 * nc / max(1, ev.sum()), 2),
                         n_flagged=nf, pct_flagged=round(100 * nf / max(1, (ev & flag).sum()), 1)))
    B = pd.DataFrame(rows)
    n_ev_f = int((ev & flag).sum())
    print(f"\ncriteria met, on the people with all {NB} evaluable: cohort {int(ev.sum()):,}, "
          f"flagged {n_ev_f:,}")
    print(B.to_string(index=False))                         # printed unsuppressed: stdout stays here
    print(f"\nmean criteria met, all {NB} evaluable: flagged {br[ev & flag].mean():.2f}  "
          f"cohort {br[ev].mean():.2f}")
    n_none = int((none_met & flag).sum())
    print(f"flagged and meeting none of the {NB} (all {NB} evaluable): {n_none:,} "
          f"({100 * n_none / max(1, n_ev_f):.1f}% of the evaluable flagged)")

    # ---- the breadth table as it leaves: each column partitions its all-evaluable margin, which the
    # json reports, so complementary suppression applies (disclosure.py). Breadth 0 is avoided as a
    # complementary cell, since it is the no-criterion count. A mean over a column with a hidden cell
    # is a second equation that solves two hidden cells, so it is blanked with them.
    Bw = B.astype(object).copy()
    margin_out, mean_out = {}, {"cohort": round(float(br[ev].mean()), 3),
                                "flagged": round(float(br[ev & flag].mean()), 3)}
    for side, total, margin in (("cohort", len(R), int(ev.sum())), ("flagged", n_flag, n_ev_f)):
        n_col, p_col = f"n_{side}", f"pct_{side}"
        if pair_hidden(margin, total):
            Bw[n_col], Bw[p_col] = SUPP, np.nan
            margin_out[side], mean_out[side] = SUPP, None
            continue
        hide, prim = partition_mask(B[n_col].values, avoid=[0])
        Bw[n_col] = marked(B[n_col].values, hide, prim)
        Bw.loc[hide, p_col] = np.nan
        margin_out[side] = margin
        if hide.any():
            mean_out[side] = None
    Bw.to_csv(os.path.join(CFG.out_dir, f"exemplar_{enc}_breadth.csv"), index=False)
    none_out = Bw.loc[Bw.breadth == 0, "n_flagged"].iloc[0]   # the no-criterion count, as written
    none_shown = not isinstance(none_out, str)

    # ---- the attribution, for the flagged set and for the no-criterion subset
    core = cores_of(P, items)
    Q = P @ Om @ P.T                                        # items x items
    QX = Xr @ Q

    def profile(mask, tag, label, n_shown=True):
        """n_shown=False when the subset's size is itself suppressed elsewhere: then neither n nor
        the top-3 counts are written, since those sum to 3n."""
        idx = np.flatnonzero(mask)
        n = len(idx)
        if n == 0:
            print(f"\n[{label}] nobody selected, skipped")
            return None
        if small(n):
            print(f"\n[{label}] {n} people -- 1 to 20, nothing written (dissemination policy)")
            return dict(n=MARK, suppressed=True)

        a = (Yc[idx] * G[idx]) / T2[idx, None]              # group share, sums to 1 per person
        assert np.allclose(a.sum(1), 1.0, atol=1e-8), \
            f"{label}: group shares do not sum to 1 (worst {np.abs(a.sum(1) - 1).max():.2e})"
        z = np.abs(Yc[idx] / np.sqrt(np.diag(Sig)))
        r = np.abs(G[idx] / np.sqrt(np.diag(Om)))
        top3 = np.zeros(k, dtype=int)
        for row in np.argsort(-a, axis=1)[:, :3]:
            top3[row] += 1
        # The top-3 counts partition 3n slots, so complementary suppression applies. The means are
        # over all n people of the subset, never over the top-3 people, so they stay.
        h3, p3 = partition_mask(top3)
        Gr = pd.DataFrame(dict(
            group=np.arange(k), mean_share=a.mean(0), median_share=np.median(a, 0),
            mean_abs_z=z.mean(0), mean_abs_r=r.mean(0),
            n_top3=marked(top3, h3, p3) if n_shown else [SUPP] * k,
            core_items=[" | ".join(c) for c in core],
            core_constructs=[" | ".join(sorted({con[items.index(i)] for i in c})) for c in core],
        )).sort_values("mean_share", ascending=False)
        Gr.to_csv(os.path.join(CFG.out_dir, f"exemplar_{enc}_{tag}groups.csv"), index=False)

        ai = Xr[idx] * QX[idx]                              # item share, exact partition of T^2
        t2i = ai.sum(1)
        assert np.allclose(t2i, T2[idx], rtol=1e-6), f"{label}: item partition does not reproduce T^2"
        si = ai / t2i[:, None]
        C = (pd.DataFrame(si, columns=items).T.groupby(con).sum().T)
        Cs = pd.DataFrame(dict(construct=C.columns, mean_share=C.mean(0).values,
                               median_share=C.median(0).values)).sort_values(
            "mean_share", ascending=False)
        Cs.to_csv(os.path.join(CFG.out_dir, f"exemplar_{enc}_{tag}constructs.csv"), index=False)

        # The PRINTED view is a reading convenience and is truncated. The RETURNED summary is not:
        # it carries every group and every construct in rank order. A truncated
        # summary is a trap for anything that reads the json instead of the csv, and the composition
        # result is read off exactly these ranks.
        print(f"\n[{label}] n = {n:,}")
        print(f"  groups holding the most (mean share of T^2), top 6 of {len(Gr)}, with their core items:")
        for _, g in Gr.head(6).iterrows():
            print(f"    C{int(g['group']):<3d} {g['mean_share']:.3f}  {g['core_constructs'][:60]}")
        print(f"  constructs holding the most (mean share of T^2), top 8 of {len(Cs)}:")
        for _, c in Cs.head(8).iterrows():
            print(f"    {c['construct']:<34s} {c['mean_share']:.3f}")
        print(f"  (full rankings in exemplar_{enc}_{tag}groups.csv, "
              f"exemplar_{enc}_{tag}constructs.csv and the json summary)")
        return dict(n=n if n_shown else SUPP,
                    n_groups=int(len(Gr)), n_constructs=int(len(Cs)),
                    groups_by_share=[int(x) for x in Gr["group"]],
                    group_shares=[round(float(x), 6) for x in Gr["mean_share"]],
                    constructs_by_share=list(Cs["construct"]),
                    construct_shares=[round(float(x), 6) for x in Cs["mean_share"]])

    prof_all = profile(flag, "", "flagged set")
    prof_none = profile(flag & none_met, "nocriterion_", f"flagged, meeting none of the {NB}",
                        n_shown=none_shown)

    json.dump(dict(
        encoder=enc, cohort=CFG.cohort, n_cohort=int(len(R)), k=int(k),
        pctile=PCTILE, threshold=thr, n_flagged=one(n_flag),
        n_all_evaluable_cohort=margin_out["cohort"], n_all_evaluable_flagged=margin_out["flagged"],
        mean_breadth_flagged=mean_out["flagged"],
        mean_breadth_cohort=mean_out["cohort"],
        n_flagged_none_met=none_out,
        flagged_profile=prof_all, nocriterion_profile=prof_none,
        disclosure="counts 1-20 written <=20, complementary cells 'suppressed'",
        note=("construct layer computed deliberately for Results 4 and reported only as a mean "
              "share over a subset, never as a per-person leading construct"),
    ), open(os.path.join(CFG.out_dir, f"exemplar_{enc}_summary.json"), "w"), indent=2,
        default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
    print(f"\nwrote -> {CFG.out_dir}")


if __name__ == "__main__":
    main()
