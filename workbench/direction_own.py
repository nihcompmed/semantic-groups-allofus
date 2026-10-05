#!/usr/bin/env python3
"""direction_own.py -- |z| and |r| per direction for EACH ENCODER'S OWN OUTLIERS.

★★ THE CONTENT OF THE OUTLIERS (Figure 3, eTable 8). DESCRIPTION ONLY, no reference group: the aim is to show
what content drives these outliers, not to compare. Per person, the distance splits exactly into one part per
group (a_j / T^2, summing to 1); the person's top 3 are the 3 groups with the largest parts. Reported per group:
the percent of the set with that group in its top 3 (`pct_top`, over `n_set`). Sets: each encoder's own outliers
(its 95th percentile), and, with --invisible, those of them who met none of the 6 thresholds (all 6 evaluable,
none met).
  * scoring CALLS screen_battery.py (load_battery, place_battery, wls_residualize, mahalanobis2), and the
    recomputed distance is ASSERTED equal to scores_<enc>.csv (< 1e-6);
  * an encoder's groups are named from the basis package's components.csv (construct profile of the core
    items), the SAME names as the phecode heatmaps;
  * only the ranking by share is computed;
  * the top-3 file carries n_set and pct_top (blank wherever n_top is withheld).
The question is what each encoder says about THE PEOPLE IT ITSELF FLAGS.

★ THE CUT, DECIDED BEFORE THE NUMBERS: each encoder's own 95th percentile, which is 4,518 people per
encoder. That is the encoder's actual operating point: the screen flags at the 95th per encoder. It
also puts all 6 encoders at the SAME DEPTH, which is what makes the 6 panels comparable: selecting
further into a tail raises |z| on its own, so unequal depth would compare rulers rather than encoders.

THE SCORING AND THE STATISTICS:
  * the scoring is screen_battery.py's -- X rescaled to [-1,1], covariates projected out, Y = XP
    centered, Ledoit-Wolf covariance, T^2 = y' Omega y
  * z_j = y_j / sqrt(Sigma_jj) MARGINAL, r_j = (Omega y)_j / sqrt(Omega_jj) CONDITIONAL,
    share = a_j / T^2 the exact additive partition
  * in the top-3 view, TOP 3 MEANS BY SIGNED SHARE, so a suppressor never enters a top 3
  * MAGNITUDES ONLY. T^2(-y) = T^2(y) and a direction carries no canonical orientation, so nothing
    here says anyone has more or less of anything.

The output files carry the tag `own`, and `fig_directions.py --tag own` draws them.

★ THE SCALE COMPARATOR has 34 units: 16 instrument totals, 5 BFI-2-XS trait scores and 13 standalone
questions, so every instrument is totaled (the WMH-CIDI and the MHQ included).

INPUTS   the battery, basis and responses wb_config resolves
         screen_out/scores_<encoder>.csv        (each encoder's own flag at its 95th percentile)
         components.csv of the basis package    (the construct profile that names each group)
         screen_out/cutoff_flags.csv            (only with --invisible; column `none_met`)
         screen_out/cutoff_summary.json         (only with --invisible; n_criteria)
OUTPUTS  screen_out/direction_summary_<tag>.csv one row per encoder per direction (LEAVES)
         screen_out/direction_top3_<tag>.csv    one row per encoder per ranking per direction (LEAVES)
         where <tag> is `own`, or `own_invisible` with --invisible

★ --invisible RESTRICTS EACH ENCODER'S OWN SET TO THE PARTICIPANTS NO CRITERION REACHES, as
`cutoffs.py` defines them: `none_met` is EVERY criterion evaluable AND breadth 0, so a withheld
criterion can never pass as a criterion not met.

★ THE DISSEMINATION POLICY (disclosure.py). A top-3 row rests on the n_top
people who put that group in their top 3, and the n_top of one encoder and ranking partition 3 x n
slots. Where n_top is 1 to 20 the row is KEPT with n_top written "<=20" and every statistic blank.
Cells hidden only so that one cannot be recovered by subtraction are written "suppressed" and keep
their statistics, which rest on more than 20 people. An encoder whose selected set is 1 to 20, or
leaves 1 to 20 of its flagged set outside it, writes nothing. fig_directions.py draws only rows with
statistics.

Run:  python3 direction_own.py
      then  python3 fig_directions.py --tag own
            python3 fig_directions.py --tag own --top3

      python3 direction_own.py --invisible
      then  python3 fig_directions.py --tag own_invisible
            python3 fig_directions.py --tag own_invisible --top3

★ --model scale|factor|pca RUNS A COMPARATOR GROUPING INSTEAD OF THE 6 ENCODERS, on its own flagged
set, with everything else identical. For the SCALE model the units are the 34 published scoring
units, so a point is a named instrument -- PHQ-9, GAD-7, DMS, HVS -- with its documented construct
beside it, and nothing has to be labelled by us at all. 13 of the 34 are standalone questions
belonging to no instrument, so the instrument reading covers 21 of the points. That makes it the most directly
readable version of this figure and the one a reviewer can check against the instruments themselves.
The projection is loaded from `comparator_weights_<model>.npz`, which comparators.py already wrote,
so it is the grouping the comparator was actually scored with rather than a reconstruction, and the
saved item order is REORDERED to the battery's rather than assumed to match.

      python3 direction_own.py --model scale
      then  python3 fig_directions.py --tag own_scale \
                     --groups screen_out/comparator_group_defs_scale.csv --raw-labels
            (add --top3 for the other view, --invisible upstream for the criterion-free subset)

★ --raw-labels IS REQUIRED FOR A COMPARATOR and is not optional. fig_directions.py normally parses
construct_profile into documented construct names and validates each against
construct_labels.DISPLAY, which is right for a semantic group -- an unknown construct there is a
mistake worth stopping for. A comparator's units are PUBLISHED INSTRUMENTS, not construct labels, so
that validation rejects correct names. --raw-labels prints the unit name as written.
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
from disclosure import MARK, marked, pair_hidden, partition_mask   # noqa: E402  the dissemination policy
from screen_battery import load_battery, mahalanobis2, place_battery, wls_residualize   # noqa: E402

BASIS_DIR, EMB_DIR, EMB_PREFIX = CFG.basis_dir, CFG.emb_dir, CFG.emb_prefix
RESPONSES_CSV, WEIGHTS_CSV, ITEMS_CSV = CFG.responses_csv, CFG.weights_csv, CFG.items_csv
OUT_DIR = CFG.out_dir
ID_COL = "id"
COVARIATES = ["age", "sex"]
# --invisible restricts each encoder's own flagged set to the participants who meet NONE of the
# published criteria, as cutoffs.py defines it: `none_met` is EVERY criterion evaluable AND breadth 0,
# so a withheld criterion can never pass as a criterion not met. The set has SIX criteria.
INVISIBLE = "--invisible" in sys.argv
# --model runs a COMPARATOR grouping instead of the 6 encoders. Its "directions" are that grouping's
# own units, which for the scale model are the 34 published scoring units, so every point on the
# figure is a named instrument rather than a group we had to label.
MODEL = (sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else None)
COMPARATORS = ["scale", "factor", "pca"]
if MODEL is not None and MODEL not in COMPARATORS:
    raise SystemExit(f"--model must be one of {COMPARATORS}, got {MODEL!r}")
TAG = ("own" if MODEL is None else f"own_{MODEL}") + ("_invisible" if INVISIBLE else "")
TOP_N = 3
RANKINGS = ["share"]                       # top 3 by share
ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5",
            "all-mpnet-base-v2", "e5-large-v2", "qwen3-embedding-0.6b"]


def load_projection(name, items, BASIS_DIR, EMB_DIR, EMB_PREFIX, OUT_DIR):
    """(P, unit names). An encoder's P comes from its basis and its item embeddings. A comparator's
    comes from `comparator_weights_<model>.npz`, which comparators.py already saved, so the grouping
    here is byte for byte the one the comparator was scored with -- not a reconstruction."""
    if name not in COMPARATORS:
        b = np.load(os.path.join(BASIS_DIR, f"{name}.npz"), allow_pickle=True)
        e = np.load(os.path.join(EMB_DIR, f"{EMB_PREFIX}_{name}.npz"), allow_pickle=True)
        pos = {str(x): j for j, x in enumerate(e["items"])}
        E = np.asarray(e["vectors"], float)[[pos[it] for it in items]]
        P = (E - b["mu"]) @ b["W"].T
        return P, [str(j) for j in range(P.shape[1])]
    wp = os.path.join(OUT_DIR, f"comparator_weights_{name}.npz")
    if not os.path.exists(wp):
        raise SystemExit(f"{wp} not found -- run comparators.py first")
    z = np.load(wp, allow_pickle=True)
    # the saved row order is the comparator's own; reorder it to ours rather than assuming they match
    order = {str(x): i for i, x in enumerate(z["items"])}
    missing = [it for it in items if it not in order]
    assert not missing, f"{name}: {len(missing)} items in the battery are absent from {wp}"
    P = np.asarray(z["S_eff"], float)[[order[it] for it in items]]
    return P, [str(u) for u in z["units"]]


def residualizer(C, w):
    A = np.column_stack([np.ones(len(w))] + ([C] if C is not None and C.shape[1] else []))
    sw = np.sqrt(w)[:, None]

    def apply(V):
        beta, *_ = np.linalg.lstsq(A * sw, V * sw, rcond=None)
        return V - A @ beta
    return apply


def main():
    CFG.describe("direction_own")
    items = pd.read_csv(ITEMS_CSV)["item"].astype(str).tolist()
    R, X, cov, w = load_battery(items)                   # ★ the screen's own loader, not a copy
    print(f"[direction_own] cohort {len(R):,} x {len(items)} items")

    invisible, nb = None, "?"
    if INVISIBLE:
        cpath = os.path.join(OUT_DIR, "cutoff_flags.csv")
        if not os.path.exists(cpath):
            raise SystemExit(f"{cpath} not found -- run cutoffs.py first")
        cf = pd.read_csv(cpath)
        cf[ID_COL] = cf[ID_COL].astype(str)
        # ★ the column is `none_met`, and the number of criteria is READ from cutoff_summary.json
        # rather than assumed.
        invisible = set(cf.loc[cf["none_met"].astype(bool), ID_COL])
        jpath = os.path.join(OUT_DIR, "cutoff_summary.json")
        nb = json.load(open(jpath))["n_criteria"] if os.path.exists(jpath) else "?"
        print(f"  --invisible: {len(invisible):,} of the cohort meet none of the {nb} with all {nb} "
              f"evaluable")

    frames = []
    unit_names = {}
    for enc in (ENCODERS if MODEL is None else [MODEL]):
        sp = os.path.join(OUT_DIR, f"scores_{enc}.csv")
        if not os.path.exists(sp):
            print(f"  [skip] scores_{enc}.csv not found")
            continue
        S = pd.read_csv(sp)
        S[ID_COL] = S[ID_COL].astype(str)
        if "flagged" in S.columns:
            own = set(S.loc[S["flagged"].astype(bool), ID_COL])
        else:
            # ★ The depth is READ from the screen's own run_config.json, never hardcoded:
            # screen_battery.py's PCTILE is a setting, and a hardcoded 5% would silently re-cut a run
            # made at any other depth. If run_config.json is absent the script STOPS rather than
            # guessing -- a wrong depth is not a formatting problem, it changes every |z| reported.
            cfg = os.path.join(OUT_DIR, "run_config.json")
            if not os.path.exists(cfg):
                raise SystemExit(
                    f"scores_{enc}.csv has no `flagged` column and {cfg} is absent, so the cut depth "
                    f"is unknown. Rerun screen_battery.py, or add the flagged column. Refusing to "
                    f"assume a percentile.")
            pct = float(json.load(open(cfg))["pctile"])
            by = next(c for c in ("pctile", "distance2", "distance") if c in S.columns)
            n_take = int(round((1.0 - pct / 100.0) * len(R)))
            own = set(S.sort_values(by, ascending=False)[ID_COL].head(n_take))
            print(f"  [{enc}] no `flagged` column; cut at the {pct:g}th percentile from run_config.json "
                  f"-> {n_take:,} people")
        n_own = len(own)
        if invisible is not None:
            own = own & invisible
        idx = np.array([i for i, pid in enumerate(R.index) if pid in own])
        assert len(idx), f"{enc}: no respondents selected"
        if pair_hidden(len(idx), n_own):
            print(f"  {enc:24s} {len(idx)} selected of {n_own} -- a count of 1 to 20 on one side, "
                  f"nothing written for it (dissemination policy)")
            continue

        if MODEL is None:
            Pk = place_battery(enc, items)[0]              # ★ the screen's own placement
            units = [str(j) for j in range(Pk.shape[1])]
        else:
            Pk, units = load_projection(enc, items, BASIS_DIR, EMB_DIR, EMB_PREFIX, OUT_DIR)
        unit_names[enc] = units
        Y = wls_residualize(X @ Pk, cov, w)
        T2, lw = mahalanobis2(Y, w)
        Yc = Y - np.average(Y, axis=0, weights=w)
        Om, Sig = lw.precision_, lw.covariance_
        G = Yc @ Om
        assert np.allclose(np.einsum("ij,ij->i", Yc, G), T2), f"{enc}: T^2 does not reproduce"
        if "distance" in S.columns:                       # ★ the screen's own distance, person by person
            d_saved = S.set_index(ID_COL)["distance"].reindex(R.index).values.astype(float)
            worst = float(np.nanmax(np.abs(np.sqrt(np.maximum(T2, 0)) - d_saved)))
            assert worst < 1e-6, f"{enc}: the recomputed distance differs from scores_{enc}.csv by {worst}"
        a = (Yc[idx] * G[idx]) / T2[idx, None]
        z = Yc[idx] / np.sqrt(np.diag(Sig))
        r = G[idx] / np.sqrt(np.diag(Om))
        off = np.abs(a.sum(axis=1) - 1.0).max()
        assert off < 1e-6, f"{enc}: shares do not sum to 1, worst {off:.2e}"
        k = Pk.shape[1]
        frames.append(pd.DataFrame({
            ID_COL: np.repeat(R.index.values[idx], k), "encoder": enc,
            "component": np.tile(np.arange(k), len(idx)),
            "share": a.ravel(), "z": z.ravel(), "r": r.ravel()}))
        extra = (f"own flagged {n_own:,} -> meeting none of the {nb}: {len(idx):,} "
                 f"({100 * len(idx) / n_own:.1f}%)" if invisible is not None
                 else f"own flagged {len(idx):,}")
        print(f"  {enc:24s} {extra} | k = {k} | shares sum to 1 (worst {off:.1e})")

    if not frames:
        raise SystemExit("every encoder's selection was suppressed under the dissemination policy -- nothing to write")
    D = pd.concat(frames, ignore_index=True)
    D["abs_z"], D["abs_r"] = D.z.abs(), D.r.abs()

    if MODEL is None:
        # ★ named from the basis package's components.csv, the construct profile of each
        # direction's core items, so the names are the SAME as in the phecode heatmaps
        # (phecode_directions.direction_scores reads the same file).
        cp = pd.read_csv(os.path.join(BASIS_DIR, "components.csv"))
        cp = cp[cp.encoder.isin(ENCODERS)]
        GD = pd.DataFrame({"encoder": cp.encoder, "group": cp.component, "component": cp.component,
                           "n_items": cp.n_core_items, "construct_profile": cp.construct_profile,
                           "instruments": ""})
    else:
        # A comparator's units ARE its groups and they carry their own published names, so no
        # labelling step is needed and none is invented. The defs file is written in
        # semantic_group_defs.csv's shape so `fig_directions.py --groups` draws it unchanged.
        units = unit_names[MODEL]
        meta = {}
        up = os.path.join(OUT_DIR, "comparator_units_scale.csv")
        if MODEL == "scale" and os.path.exists(up):
            U = pd.read_csv(up)
            meta = {str(r.unit): (int(r.n_items), str(r.constructs), str(r.instrument_code))
                    for r in U.itertuples()}
        fp = os.path.join(OUT_DIR, f"comparator_factors_{MODEL}.csv")
        if MODEL in ("factor", "pca") and os.path.exists(fp):
            F = pd.read_csv(fp)
            key = "factor" if "factor" in F.columns else F.columns[0]
            meta = {str(r[key]): (int(r.get("size", 0)), str(r.get("construct_profile", "")), "")
                    for _, r in F.iterrows()}
        GD = pd.DataFrame([{
            "encoder": MODEL, "group": f"C{j}", "component": j, "n_groups_on_direction": 1,
            "n_items": meta.get(u, (0, "", ""))[0],
            "construct_profile": f"{u}" + (f" ({meta[u][1]})" if u in meta and meta[u][1] else ""),
            "instruments": meta.get(u, (0, "", ""))[2], "items": "",
        } for j, u in enumerate(units)])
        dp = os.path.join(OUT_DIR, f"comparator_group_defs_{MODEL}.csv")
        GD.to_csv(dp, index=False)
        print(f"  group names taken from the {MODEL} model's own units -> "
              f"{os.path.basename(dp)}")
    rows = []
    for (enc, j), d in GD.groupby(["encoder", "component"]):
        rows.append({"encoder": enc, "component": int(j), "n_semantic_groups": len(d),
                     "n_core_items": int(d.n_items.sum()),
                     "semantic_groups": " vs ".join(d.construct_profile),
                     "instruments": "; ".join(sorted({x for v in d.instruments
                                                      for x in str(v).split("; ") if x}))})
    GROUPS = pd.DataFrame(rows)

    g = D.groupby(["encoder", "component"])
    S = pd.DataFrame({
        "n_people": g[ID_COL].nunique(),
        "mean_abs_z": g.abs_z.mean(), "median_abs_z": g.abs_z.median(),
        "mean_abs_r": g.abs_r.mean(), "median_abs_r": g.abs_r.median(),
        "mean_share": g.share.mean(), "median_share": g["share"].median(),
    }).reset_index().merge(GROUPS, on=["encoder", "component"], how="left")
    S = S.sort_values(["encoder", "mean_abs_z"], ascending=[True, False])
    S.to_csv(os.path.join(OUT_DIR, f"direction_summary_{TAG}.csv"), index=False)

    T3 = []
    for rank in RANKINGS:
        top = (D.sort_values(rank, ascending=False, kind="mergesort")
                 .groupby(["encoder", ID_COL], sort=False).head(TOP_N))
        slots = top.groupby("encoder").size()
        want = D.groupby("encoder")[ID_COL].nunique() * TOP_N
        assert slots.equals(want), f"{rank}: top-{TOP_N} slots {slots.to_dict()} != {want.to_dict()}"
        g2 = top.groupby(["encoder", "component"])
        B = pd.DataFrame({
            "n_top": g2[ID_COL].nunique(),
            "mean_abs_z": g2.abs_z.mean(), "median_abs_z": g2.abs_z.median(),
            "mean_abs_r": g2.abs_r.mean(), "median_abs_r": g2.abs_r.median(),
            "mean_share": g2.share.mean(), "median_share": g2["share"].median(),
        }).reset_index()
        # ★ a group in NOBODY's top 3 is written with n_top 0 (0%), not left out
        full = D[["encoder", "component"]].drop_duplicates()
        B = full.merge(B, on=["encoder", "component"], how="left").fillna({"n_top": 0})
        B["n_top"] = B["n_top"].astype(int)
        B.insert(2, "ranking", rank)
        T3.append(B)
    T3 = pd.concat(T3, ignore_index=True).merge(GROUPS, on=["encoder", "component"], how="left")
    n_set = D.groupby("encoder")[ID_COL].nunique()
    T3.insert(3, "n_set", T3["encoder"].map(n_set).astype(int))
    T3 = T3.sort_values(["ranking", "encoder", "n_top"], ascending=[True, True, False])
    # the policy, per encoder and ranking: the n_top partition 3 x n slots
    T3 = T3.reset_index(drop=True)
    T3["n_top"] = T3["n_top"].astype(object)
    n_prim = n_comp = 0
    stat_cols = [c for c in T3.columns if c.startswith(("mean_", "median_"))]
    for _, ix in T3.groupby(["encoder", "ranking"]).groups.items():
        ix = list(ix)
        hide, prim = partition_mask(T3.loc[ix, "n_top"].values)
        T3.loc[ix, "n_top"] = marked(T3.loc[ix, "n_top"].values, hide, prim)
        T3.loc[[i for i, p in zip(ix, prim) if p], stat_cols] = np.nan
        n_prim += int(prim.sum()); n_comp += int((hide & ~prim).sum())
    num = pd.to_numeric(T3["n_top"], errors="coerce")        # withheld cells stay blank
    T3.insert(4, "pct_top", (100.0 * num / T3["n_set"]).round(1))
    T3.to_csv(os.path.join(OUT_DIR, f"direction_top3_{TAG}.csv"), index=False)

    print(f"\n  -> direction_summary_{TAG}.csv ({len(S)} rows), "
          f"direction_top3_{TAG}.csv ({len(T3)} rows: {n_prim} written {MARK} with statistics blank, "
          f"{n_comp} complementary)")
    # ★ one zip per run to download, checked for person-level columns first
    import zipfile
    from zip_screen_aggregates import ID_COLS
    outs = [f"direction_summary_{TAG}.csv", f"direction_top3_{TAG}.csv"]
    for fn in outs:
        cols = {str(c).strip().lower() for c in pd.read_csv(os.path.join(OUT_DIR, fn), nrows=0).columns}
        assert not (ID_COLS & cols), f"{fn} carries a person-level column: {ID_COLS & cols}"
    zpath = os.path.join(OUT_DIR, f"direction_{TAG}_{os.environ.get('WORKSPACE_CDR', '') or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in outs:
            zf.write(os.path.join(OUT_DIR, fn), arcname=f"direction_own/{fn}")
    print(f"  download -> {zpath}")
    g = ("" if MODEL is None
         else f" --groups {os.path.join(OUT_DIR, f'comparator_group_defs_{MODEL}.csv')}"
              f" --raw-labels")
    print(f"  draw with:  python3 fig_directions.py --tag {TAG}{g}")
    print(f"        and:  python3 fig_directions.py --tag {TAG}{g} --top3")


if __name__ == "__main__":
    main()
