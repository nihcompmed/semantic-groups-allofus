#!/usr/bin/env python3
"""ceiling.py -- whether the composition ranking is a ranking of PEOPLE or of RULERS.

THE OBJECTION. The composition result ranks semantic groups against each other by mean |z|, and |z|
is in units of that group's own cohort SD. Those units do not reach equally far. An item whose
extreme response level is given by 2% of the cohort puts anyone who gives it far out in SD terms; an
item whose levels people spread across evenly cannot put anyone far out however badly off they are.
So a group resting on thin-tailed content has a SHORT ruler and one resting on skewed content has a
LONG one, and a gap in mean |z| between two groups could be a comparison between rulers rather than
between people.

  ★ WHAT DRIVES RULER LENGTH IS NOT THE NUMBER OF RESPONSE LEVELS, it is the observed mass in the
  end cell. A 50/50 binary has the SHORTEST ruler in the battery: an endorser sits 1.0 SD out. A
  five-level item where 97% answer at the floor has one of the longest. Verified on synthetic items:
  ceiling |z| of the end cell is 1.01 for a 50/50 binary, 1.43 for a five-level item
  people spread across, 5.66 for a 3% binary, 10.22 for a five-level item with 97% at the floor.
  Level count does not order them. Tail mass does. This is why ceiling_items.csv below reports the
  observed frequency of every level and not the published range.

  ★ IT BITES ONE CLAIM ONLY. Everywhere else the distance ORDERS PEOPLE, and every person-level
  result is checked against something that never entered the scoring (criteria met, care barriers,
  contact days). Ruler length cannot manufacture those. It bites the one place where groups are
  ranked against each other, which is the composition result.

HOW IT IS ANSWERED. Not by modeling how far out it was POSSIBLE to sit. An attainable-maximum
ceiling does not work: the attainable maximum sums |P_cj| over all
151 items while the SD grows like its square root, so such a ceiling measures how spread a
direction's loadings are and barely sees the response distributions it was built to see.

The answer used instead is distribution-free. Put every group on ONE ruler. Within each direction,
rank the cohort on |z| and map the rank through the same normal quantile function, so a participant's
score becomes how far out they sit RELATIVE TO THAT GROUP'S OWN DISTRIBUTION, expressed on a scale
shared by all groups. Ruler length cannot survive a rank transform.

  L_j   mean |z_j| over EACH ENCODER'S OWN FLAGGED SET among those who put
        j in their top 3 by SIGNED share, which is `direction_own.py`'s reporting rule.
        ★ The selection is `scores_<encoder>.csv`'s own `flagged` column, 4,518 per encoder at the
        95th percentile. A shallower cut lowers mean |z| on its own, so L_j depends on the depth
        of the cut. The spread RATIO, being a ratio of spreads, is far less sensitive to depth
        than the levels are.
  S_j   mean COMMON-RULER score over those same people: mean of Phi^-1(rank_i / (N+1)) where rank_i
        is that participant's rank on |z_j| within direction j. L_j with the ruler removed, in units
        comparable across groups.
  tail  p95, p99 and p99.9 of |z_j| over the cohort, and the skewness of z_j. These describe
        ruler length directly, and are what makes L_j and S_j diverge when they diverge.

★ THE READING, FIXED BEFORE THE NUMBERS ARE SEEN. The statistic is the SPREAD
  RATIO: the spread of S_j across groups divided by the spread of L_j across groups, over groups with
  at least MIN_LEAD leaders, reported per encoder as both an SD ratio and a range ratio.

  ★ MIN_LEAD IS 21. This statistic fits no model, so no floor for estimating a model applies
  here. The dissemination policy's bound is the one threshold that applies: a statistic
  resting on 1 to 20 people cannot leave the Workbench, so a group is ranked from 21 leaders.
  --min-lead can raise it and cannot lower it below 21.

    ratio near 1   the groups differ by as much on the common ruler as they do in their own SD
                   units. The composition ranking is a ranking of people.
    ratio near 0   the groups are alike on the common ruler and differ only in SD units, so the
                   spread in |z| is ruler length. The ranking is then read on S_j, with L_j
                   reported beside it.

  ★ WHY NOT SPEARMAN, AND WHY NOT THE TOP-5 OVERLAP. Both fail when tested against a simulated
  artifact. A rank correlation between L
  and S reads +1.000 whether the difference between groups is real or pure ruler length, because
  both quantities increase with how far out the leaders sit within their own column; and top-5
  overlap read 4/5 in the pure-artifact case, because when groups are alike on the common ruler
  their order is noise and noise still overlaps. The two controls separate ONLY on the size of the
  gap: with identical percentiles and eight long rulers the spread ratio was 0.000, and with one
  ruler and genuinely different leaders it was 1.056. The top 5 by each measure is written out
  per encoder, because the manuscript's sentence is about a top 5 and should be readable, but the
  ratio is what decides the question.

  Either way both numbers are reported. Nothing here is a pass/fail gate on the finding.

WHAT LEAVES. Three files, all aggregates. No per-person quantity is written. ★ The dissemination
policy (disclosure.py):
  * n_leaders partition 3 x n_flagged slots per encoder. A count of 1 to 20 is written "<=20" with
    its leader statistics blank; a complementary cell is written "suppressed" and keeps them.
  * an item's level counts partition the cohort. A level of 1 to 20 is written "<=20", complementary
    levels "suppressed", and their shares blanked. ★ For such an item z_of_level, item_ruler and
    thin_end_share are blanked too: z on two levels solves the item's mean and SD, and the mean with
    the known cohort total solves two hidden cells.
  * cohort_max is not written. It is one participant's value. The tail is carried by p95, p99 and
    p99.9, each of which has far more than 20 people above it at 90,351.
Rows are kept, so the suppression is visible.

★ LEVELS ARE THE ANSWERED LEVELS. The response file keeps imputed cells in codebook
units, fractional where imputed, so a raw count of distinct values makes every imputed value a
"level" held by a few people. Unsnapped, items with 2 to 5 levels read up to 15. The
item table therefore snaps each imputed cell to the nearest level answered for that item, from
imputed_mask_k5.csv, exactly as mca_basis.py does. ★ ONLY THE ITEM TABLE IS SNAPPED.
The scores, z, T^2 and the top-3 selection stay on the imputed matrix, which is what the screen,
exemplar_encoder.py and direction_own.py score, so L_j here is the |z| those scripts report.

INPUTS   the battery, basis and responses wb_config resolves (self-contained: the scores are rebuilt
         here), plus screen_out/scores_<encoder>.csv for each
         encoder's own flagged column, and semantic_group_defs.csv for the group names.
OUTPUTS  screen_out/ceiling_directions.csv   encoder x direction: L, S, tail, n_leaders (LEAVES)
         screen_out/ceiling_items.csv        item x level: observed count and share (LEAVES)
         screen_out/ceiling_summary.json     per encoder: the two declared statistics (LEAVES)

Run:  python3 ceiling.py                 (the ranking floor is 21 leaders, and --min-lead N raises it)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, skew, spearmanr
from sklearn.covariance import LedoitWolf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_config import CFG   # noqa: E402
from disclosure import marked, partition_mask   # noqa: E402  the dissemination policy

ARG = lambda f, d: (type(d)(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d)
BASIS_DIR     = CFG.basis_dir
EMB_DIR       = CFG.emb_dir
EMB_PREFIX    = CFG.emb_prefix
RESPONSES_CSV = CFG.responses_csv
WEIGHTS_CSV   = CFG.weights_csv
ITEMS_CSV     = CFG.items_csv
OUT_DIR       = CFG.out_dir
ID_COL        = "id"
COVARIATES    = ["age", "sex"]
TOP_N         = 3                     # a person's top 3 by SIGNED share, as direction_own.py has it
MIN_LEAD      = max(ARG("--min-lead", 21), 21)   # ranked from 21 leaders, the policy's bound
TOP_K         = 5                     # the manuscript's sentence is about the top 5
ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5",
            "all-mpnet-base-v2", "e5-large-v2", "qwen3-embedding-0.6b"]


def residualizer(C, w):
    """The operator that projects the covariates and the intercept out of any column block."""
    A = np.column_stack([np.ones(len(w))] + ([C] if C is not None and C.shape[1] else []))
    sw = np.sqrt(w)[:, None]

    def apply(V):
        beta, *_ = np.linalg.lstsq(A * sw, V * sw, rcond=None)
        return V - A @ beta
    return apply


def snap_to_levels(raw, items, mask_csv, ids):
    """Move every imputed cell to the nearest level that was actually answered for that item.
    mca_basis.py's rule, with the mask aligned on id rather than assumed to share the row order."""
    if not os.path.exists(mask_csv):
        print(f"[ceiling] no imputed mask at {mask_csv} -- nothing to snap")
        return raw, 0
    M = pd.read_csv(mask_csv)
    M[ID_COL] = M[ID_COL].astype(str)
    M = M.set_index(ID_COL).reindex(ids)
    assert all(c in M.columns for c in items), "the imputed mask does not cover every item"
    assert not M[items].isna().any().any(), "the imputed mask does not cover every respondent"
    mask = M[items].values.astype(bool)
    out, moved = raw.copy(), 0
    for j in range(raw.shape[1]):
        m = mask[:, j]
        if not m.any():
            continue
        levels = np.unique(raw[~m, j])
        assert levels.size, f"item {items[j]} has no answered cell to take levels from"
        snapped = levels[np.abs(raw[m, j][:, None] - levels[None, :]).argmin(axis=1)]
        moved += int((snapped != raw[m, j]).sum())
        out[m, j] = snapped
    return out, moved


def main():
    CFG.describe("ceiling")
    T = pd.read_csv(ITEMS_CSV)
    items = T["item"].astype(str).tolist()
    scales = (T.set_index("item").reindex(items)
               .rename(columns={"scale_min": "min", "scale_max": "max",
                                "reverse_coded": "reverse"}))

    R = pd.read_csv(RESPONSES_CSV)
    R[ID_COL] = R[ID_COL].astype(str)
    R = R.set_index(ID_COL)
    raw = R[items].values.astype(float)
    assert not np.isnan(raw).any(), "missing responses -- use the k0 file"

    lo_s = scales["min"].values.astype(float)
    hi_s = scales["max"].values.astype(float)
    rev = scales["reverse"].astype(str).str.lower().eq("true").values
    X = np.clip(2.0 * (raw - lo_s) / (hi_s - lo_s) - 1.0, -1.0, 1.0)
    X[:, rev] *= -1.0
    N = len(R)

    cov = R[COVARIATES].values.astype(float) if COVARIATES else None
    w = (pd.read_csv(WEIGHTS_CSV).iloc[:, 0].values.astype(float)
         if WEIGHTS_CSV and os.path.exists(WEIGHTS_CSV) else np.ones(N))
    resid = residualizer(cov, w)

    # ---- the item side: what each level actually got ------------------------------------------
    # The published range is not the ruler, the observed levels are. An item whose top level nobody
    # ever gives is, in this cohort, an item without that level. This table is the EXPLANATION of
    # any divergence between L and S, and it is worth reading on its own.
    # Snapped for this table only: see the docstring. X itself, which everything below scores, is not.
    raw_s, moved = snap_to_levels(raw, items, CFG.mask_csv, R.index) if CFG.cohort == "k5" else (raw, 0)
    Xs = np.clip(2.0 * (raw_s - lo_s) / (hi_s - lo_s) - 1.0, -1.0, 1.0)
    Xs[:, rev] *= -1.0
    print(f"[ceiling] item table: {moved:,} imputed cells snapped to the nearest answered level")
    rows = []
    for c, it in enumerate(items):
        vals, cnt = np.unique(Xs[:, c], return_counts=True)
        mu, sd_c = Xs[:, c].mean(), Xs[:, c].std(ddof=0)
        for v, n in zip(vals, cnt):
            rows.append({"item": it, "value_rescaled": float(v), "n": int(n), "share": float(n / N),
                         "z_of_level": float((v - mu) / sd_c) if sd_c > 0 else 0.0})
    IT = pd.DataFrame(rows)
    end = (IT.sort_values(["item", "value_rescaled"]).groupby("item")
             .agg(n_levels_used=("value_rescaled", "size"),
                  share_lowest=("share", "first"), share_highest=("share", "last"),
                  z_lowest=("z_of_level", "first"), z_highest=("z_of_level", "last")))
    end["thin_end_share"] = end[["share_lowest", "share_highest"]].min(axis=1)
    end["item_ruler"] = end[["z_lowest", "z_highest"]].abs().max(axis=1)
    IT = IT.merge(end.reset_index()[["item", "n_levels_used", "thin_end_share", "item_ruler"]],
                  on="item", how="left")
    # the policy, at write time: each item's level counts partition the cohort
    IT_out = IT.copy()
    IT_out["n"] = IT_out["n"].astype(object)
    n_lv = n_items_hit = 0
    for it_name, ix in IT.groupby("item").groups.items():
        ix = list(ix)
        hide, prim = partition_mask(IT.loc[ix, "n"].values)
        if not hide.any():
            continue
        IT_out.loc[ix, "n"] = marked(IT.loc[ix, "n"].values, hide, prim)
        IT_out.loc[[i for i, h in zip(ix, hide) if h], "share"] = np.nan
        IT_out.loc[ix, ["z_of_level", "item_ruler", "thin_end_share"]] = np.nan
        n_lv += int(hide.sum()); n_items_hit += 1
    IT_out.to_csv(os.path.join(OUT_DIR, "ceiling_items.csv"), index=False)
    print(f"  dissemination policy: {n_lv} item levels suppressed across {n_items_hit} items, whose "
          f"z, ruler and thin-end share are blanked")
    print(f"[ceiling] items {len(items)} | levels used: median {end.n_levels_used.median():.0f}, "
          f"range {end.n_levels_used.min():.0f}-{end.n_levels_used.max():.0f}")
    print(f"  item ruler (|z| of the end cell): median {end.item_ruler.median():.2f}, "
          f"range {end.item_ruler.min():.2f}-{end.item_ruler.max():.2f}")
    print(f"  Spearman(levels used, ruler) = {spearmanr(end.n_levels_used, end.item_ruler)[0]:+.3f}"
          f"   <- near zero means level count is not what sets ruler length")
    print(f"  -> ceiling_items.csv")

    # ---- the people the manuscript reports -----------------------------------------------------
    # ★ The selection is PER ENCODER and is made inside the loop below from scores_<enc>.csv's own
    # `flagged` column.
    pos_of = {pid: i for i, pid in enumerate(R.index)}

    gpath = os.path.join(OUT_DIR, "semantic_group_defs.csv")
    G = pd.read_csv(gpath) if os.path.exists(gpath) else None

    out, summary = [], {}
    for enc in ENCODERS:
        spath = os.path.join(OUT_DIR, f"scores_{enc}.csv")
        if not os.path.exists(spath):
            print(f"  [skip] scores_{enc}.csv not found")
            continue
        S0 = pd.read_csv(spath)
        S0[ID_COL] = S0[ID_COL].astype(str)
        own = set(S0.loc[S0["flagged"].astype(bool), ID_COL])
        idx = np.array([pos_of[pid] for pid in own if pid in pos_of])
        assert len(idx), f"no respondents selected for {enc}"
        print(f"  {enc:24s} own flagged set {len(idx):,} of {N:,} ({100 * len(idx) / N:.2f}%)")

        b = np.load(os.path.join(BASIS_DIR, f"{enc}.npz"), allow_pickle=True)
        W, mu_b = b["W"], b["mu"]
        e = np.load(os.path.join(EMB_DIR, f"{EMB_PREFIX}_{enc}.npz"), allow_pickle=True)
        pos = {str(x): j for j, x in enumerate(e["items"])}
        E = np.asarray(e["vectors"], float)[[pos[it] for it in items]]
        Pk = (E - mu_b) @ W.T
        k = Pk.shape[1]

        Y = resid(X @ Pk)
        Yc = Y - np.average(Y, axis=0, weights=w)
        lw = LedoitWolf().fit(Yc)
        Om, Sig = lw.precision_, lw.covariance_
        sd = np.sqrt(np.diag(Sig))
        Gm = Yc @ Om
        T2 = np.einsum("ij,ij->i", Yc, Gm)

        Z = Yc / sd                                   # N x k, the manuscript's z
        AZ = np.abs(Z)
        # ONE ruler for every direction: rank the cohort within the direction, then map the rank
        # through the same normal quantile function. (N + 1) keeps the top rank finite.
        PCT = np.column_stack([rankdata(AZ[:, j], method="average") / (N + 1) for j in range(k)])
        NS = norm.ppf(PCT)                            # the common-ruler score

        a = (Yc[idx] * Gm[idx]) / T2[idx, None]       # the signed, exact partition
        top = np.argsort(-a, axis=1)[:, :TOP_N]       # top 3 by SIGNED share
        is_top = np.zeros_like(a, dtype=bool)
        np.put_along_axis(is_top, top, True, axis=1)

        for j in range(k):
            who = is_top[:, j]
            n_lead = int(who.sum())
            p95, p99, p999 = np.percentile(AZ[:, j], [95, 99, 99.9])
            out.append({
                "encoder": enc, "component": j, "n_leaders": n_lead,
                "leader_mean_abs_z": float(AZ[idx][who, j].mean()) if n_lead else np.nan,
                "leader_mean_pct":   float(PCT[idx][who, j].mean()) if n_lead else np.nan,
                "leader_mean_common_ruler": float(NS[idx][who, j].mean()) if n_lead else np.nan,
                "cohort_p95": float(p95), "cohort_p99": float(p99), "cohort_p999": float(p999),
                "tail_ratio_p999_p95": float(p999 / p95) if p95 > 0 else np.nan,
                "skew_z": float(skew(Z[:, j])),
            })

        D = pd.DataFrame([o for o in out if o["encoder"] == enc])
        Dr = D[D.n_leaders >= MIN_LEAD].dropna(
            subset=["leader_mean_abs_z", "leader_mean_common_ruler"])
        L, S = Dr.leader_mean_abs_z.values, Dr.leader_mean_common_ruler.values
        if len(Dr) == 0:
            print(f"  {enc}: no direction has {MIN_LEAD} or more leaders -- nothing ranked")
            summary[enc] = {"k": int(k), "n_flagged": int(len(idx)), "ranked_directions": 0,
                            "min_lead": MIN_LEAD}
            continue
        sd_ratio = float(S.std(ddof=1) / L.std(ddof=1)) if len(Dr) > 1 and L.std(ddof=1) > 0 else np.nan
        rg_ratio = float(np.ptp(S) / np.ptp(L)) if len(Dr) > 1 and np.ptp(L) > 0 else np.nan
        # printed as a diagnostic only: it reads +1 whether the difference is real or ruler length
        rho = float(spearmanr(L, S)[0]) if len(Dr) > 2 else np.nan
        topL = list(Dr.sort_values("leader_mean_abs_z", ascending=False).component[:TOP_K])
        topS = list(Dr.sort_values("leader_mean_common_ruler", ascending=False).component[:TOP_K])
        summary[enc] = {
            "k": int(k), "n_flagged": int(len(idx)), "ranked_directions": int(len(Dr)), "min_lead": MIN_LEAD,
            "spread_ratio_sd": sd_ratio, "spread_ratio_range": rg_ratio,
            "L_range": [float(L.min()), float(L.max())],
            "S_range": [float(S.min()), float(S.max())],
            "top5_by_abs_z": [int(x) for x in topL],
            "top5_by_common_ruler": [int(x) for x in topS],
            "top5_overlap": int(len(set(topL) & set(topS))),
            "spearman_L_vs_S_diagnostic_only": rho,
            "tail_ratio_range": [float(Dr.tail_ratio_p999_p95.min()),
                                 float(Dr.tail_ratio_p999_p95.max())],
        }
        print(f"  {enc}: k={k}, {len(Dr)} ranked | ★ spread ratio {sd_ratio:.3f} (SD) "
              f"{rg_ratio:.3f} (range) | L {L.min():.2f}-{L.max():.2f}, S {S.min():.2f}-{S.max():.2f}"
              f" | top-5 overlap {len(set(topL) & set(topS))}/{TOP_K}")

    O = pd.DataFrame(out)
    # the policy, at write time: n_leaders partition 3 x n_flagged slots per encoder
    O = O.reset_index(drop=True)
    O["n_leaders"] = O["n_leaders"].astype(object)
    n_p = n_c = 0
    lead_cols = [c for c in O.columns if c.startswith("leader_")]
    for _, ix in O.groupby("encoder").groups.items():
        ix = list(ix)
        hide, prim = partition_mask(O.loc[ix, "n_leaders"].values)
        O.loc[ix, "n_leaders"] = marked(O.loc[ix, "n_leaders"].values, hide, prim)
        O.loc[[i for i, p in zip(ix, prim) if p], lead_cols] = np.nan
        n_p += int(prim.sum()); n_c += int((hide & ~prim).sum())
    print(f"  dissemination policy: {n_p} directions with 1 to 20 leaders, {n_c} complementary")
    if G is not None and {"encoder", "component"} <= set(G.columns):
        nm = next((c for c in ("group", "label", "name") if c in G.columns), None)
        if nm:
            O = O.merge(G[["encoder", "component", nm]].drop_duplicates(),
                        on=["encoder", "component"], how="left")
    O.to_csv(os.path.join(OUT_DIR, "ceiling_directions.csv"), index=False)
    with open(os.path.join(OUT_DIR, "ceiling_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> ceiling_directions.csv, ceiling_summary.json")


if __name__ == "__main__":
    main()
