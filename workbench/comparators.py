#!/usr/bin/env python3
"""comparators.py -- the two established comparators, scored and attributed exactly as the framework is.

★ FOR THE PAPER (step 4 of README.md, eMethods 5) this script supplies the scores of the groupings fitted
on the responses: scores_pca.csv, scores_factor.csv, scores_mca.csv (k = 25) and, with --mca-k horn,
scores_mca90.csv. k_choice.py imports its load_battery and parallel_analysis_k. Everything else it
writes -- the instrument-totals scores (scores_scale.csv), attribution files and the comparator_* tables
against each encoder's 95th-percentile set -- is not used for the results in the paper, and no step
reads it. The description below is of the script as a whole.

STANDALONE (numpy, pandas, scikit-learn). Runs after screen_battery.py. Reads the cohort of record
and the framework's flag sets, fits nothing but the comparators, and holds everything downstream of
the representation fixed: the same covariates, the same weighted residualization, the same
Ledoit-Wolf precision, the same distance, the same empirical 95th percentile, the same exact
partition of T^2 down to items and constructs. The only thing that differs between a comparator
and the framework is what a person is scored on.

THE COMPARATORS (each a linear map S of the keyed, rescaled responses X, so Y = X S)

  scale    the field's practice in All of Us: ONE SCORE PER INSTRUMENT. A unit is the
           instrument's total (16 instruments), one trait score per Big Five trait for the
           BFI-2-XS (5, because it publishes 5 scores and no total, so summing its 15 items would
           add extraversion to conscientiousness), and one score per item for the 13 single items,
           which belong to no instrument and so have nothing to total. 34 units.

           ★ The WMH-CIDI and the MHQ are totaled too. Scoring them 1 item at a time would make
           53 units of which 34 are single items -- so a model called instrument totals would,
           for 21 of its items, not total anything. Neither publishes a summed total, but SUMMING
           THE ITEMS IS THE PRE-ESTABLISHED GENERIC RULE when an instrument has no published one,
           and it is what our own criteria already do with these items: the psychosis screen
           counts endorsed MHQ items and the WMH-CIDI criterion pools CIDI items. So the sum is
           the scoring the benchmark itself uses, not a score invented here. It also gives every
           criterion's instrument a matching unit, psychosis and the CIDI included.

           A unit's score is the sum of the keyed rescaled values of
           its items, which for a unit whose items share a response range is the published total
           up to an affine constant. Scores are standardized over the cohort. (The ACE category
           count and the ASRS screener count are threshold versions of these sums and belong to
           the criteria, cutoffs.py.)
  factor   psychometric practice: common-factor analysis with a varimax rotation on the
           standardized responses, k by Horn's parallel analysis on the response correlation
           matrix (permutation null, 95th percentile). Regression factor scores, which are linear
           in the responses, so the weight matrix is recovered exactly and the item partition holds.
  pca      principal component analysis at the same k, the supplement's check.

Each factor or component is named by the construct profile of its core items, the fixed-point
participation-ratio core down its column of loadings, the rule the framework uses.

★ THE COMPARISONS. Each comparator is set against EACH ENCODER'S OWN flagged set, the prespecified
gte-large-en-v1.5 first. "The factor tail held X% of the highest-scoring participants" is share_B_in_A
of the factor | gte-large-en-v1.5 row of comparator_flag_overlap.csv. Criteria columns are `any_met`
and the criteria count is read from cutoff_summary.json. (iii) needs a file no script in this folder
writes, so it does not run. Every aggregate is written under the dissemination policy (disclosure.py):
an intersection or a share that gives a count of 1 to 20 back is withheld, and the leading-construct
counts are suppressed per model and flag set.

WHAT IS COMPARED (these tables are NOT used for the results in the paper), each against each encoder's
own flagged set, reported whichever way it goes
  (i)   flag overlap: Jaccard against the independence null f_A f_B / (f_A + f_B - f_A f_B), and
        the share of each set inside the other
  (ii)  against the criteria, when screen_out/cutoff_flags.csv exists: breadth of each flag
        set and the share meeting none of the six
  (iii) the leading construct on the consensus outliers, against the framework's. It runs only if
        screen_out/attribution_per_person_outliers.csv exists, which no script in this folder writes.
  (iv)  care is computed in care.py, which reads the scores_<model>.csv and comparator_weights_<model>.npz
        written here

OUTPUTS (-> screen_out/)
  scores_<model>.csv                 id, distance, distance2, pctile, flagged   (per person, stays)
  attribution_<model>.csv            id, flagged, leading unit and construct, shares, carrying count (per person, stays)
  comparator_units_scale.csv         the 34 units, their items and constructs
  comparator_factors_<model>.csv     factor, core size, construct profile, core items, loadings
  comparator_flag_overlap.csv        pairwise overlap among scale, factor, pca, mca, the six encoders
  comparator_leading_constructs.csv  leading-construct counts and mean construct shares, per model and flag set
  comparator_breadth.csv             (ii), when the criteria exist
  comparator_attribution_agreement.csv  (iii), so not written by a run in this folder
  comparator_summary.json            k, convergence, counts, settings, the CDR string
  A run with --mca-k writes its aggregates with the suffix _mca<k> (see MCA_NAME below).

k_choice.py imports load_battery and parallel_analysis_k from here, so the choice of k it reports is
the k this script fits at.

Run:  python3 comparators.py              (k by parallel analysis; MCA at that same k)
      python3 comparators.py --mca-k horn (MCA at its own Horn's k, read from mca_eigenvalues.csv,
                                           after the default run; 90 on this CDR, so scores_mca90.csv)
      python3 comparators.py --mca-k 90   (MCA at a stated k, after the default run)
      python3 comparators.py --k 20       (override k, sensitivity only)
      python3 comparators.py --no-mca     (skip MCA)
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FactorAnalysis, PCA

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wb_config import CFG   # noqa: E402  battery and responses folder, resolved once, no defaults
RESPONSES_CSV = os.path.join(CFG.responses_dir, "responses_k5.csv")   # the cohort of record
WEIGHTS_CSV   = os.path.join(CFG.responses_dir, "weights_k5.csv")
ITEMS_CSV     = CFG.items_csv
OUT_DIR       = os.path.join(HERE, "screen_out")
CDR           = os.environ.get("WORKSPACE_CDR", "")
ID_COL        = "id"
COVARIATES    = ["age", "sex"]
PCTILE        = 95.0
N_PERM        = 100          # parallel analysis draws
PA_PCTILE     = 95.0
SEED          = 0
K_OVERRIDE    = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else None
ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5",
            "all-mpnet-base-v2", "e5-large-v2", "qwen3-embedding-0.6b"]
# ★ Only items belonging to NO instrument are scored one at a time; everything else is totaled.
# See the scale paragraph above for why the WMH-CIDI and the MHQ are totaled.
PER_ITEM_INSTRUMENTS      = {"single item"}                     # no instrument, nothing to total
PER_CONSTRUCT_INSTRUMENTS = {"BFI-2-XS"}                        # publishes 5 trait scores, no total
MODELS = ["scale", "factor", "pca"]
# ★ MCA is a fifth grouping. It is scored at MCA_K, matched to the k that
# factor and pca get from parallel analysis, because a Mahalanobis distance on 90 dimensions is more
# diffuse than one on 25 and who clears the 95th percentile would then turn on the dimension count
# rather than on the representation. Truncating MCA from its own Horn k of 90 to 25 retains 98.5% of
# its Benzecri-adjusted inertia, so the matching costs it almost nothing. --mca-k 90 is the sensitivity.
USE_MCA = "--no-mca" not in sys.argv


def mca_horn_k(out_dir):
    """MCA's own Horn's k from mca_basis.py's eigenvalue file: the leading axes whose eigenvalue clears
    the permutation null's 95th percentile, cut at the first that does not (mca_basis.py's own rule).
    Shared with breadth_sweep.py so both name the Horn-k run the same way."""
    p = os.path.join(out_dir, "mca_eigenvalues.csv")
    assert os.path.exists(p), f"{p} is missing; run `mca_basis.py --axes 100 --parallel 10` first"
    E = pd.read_csv(p).sort_values("axis")
    assert "null_95th" in E.columns, f"{p} has no null; rerun mca_basis.py with --parallel 10"
    fails = np.where(E["eigenvalue"].values <= E["null_95th"].values)[0]
    assert fails.size, f"MCA's Horn's k reaches the {len(E)} axes computed; rerun mca_basis.py with more --axes"
    return int(fails[0])


# --mca-k <int> or --mca-k horn (MCA's own Horn's k, read from mca_eigenvalues.csv).
_MCA_ARG = sys.argv[sys.argv.index("--mca-k") + 1] if "--mca-k" in sys.argv else None
MCA_K = (None if _MCA_ARG is None                                                    # None = match k
         else mca_horn_k(os.path.join(HERE, "screen_out")) if _MCA_ARG == "horn" else int(_MCA_ARG))
# ★ The sensitivity gets its OWN model name, so `--mca-k 90` writes scores_mca90.csv and cannot
# overwrite the primary run's scores_mca.csv. The exclusive-set scripts (care_exclusive.py,
# matched_breadth.py) take one or the other through MCA_MODEL, never both at once: 2 depths of the
# same representation inside 1 exclusive-set comparison would cannibalize each other's exclusive
# picks. breadth_sweep.py reads BOTH, as 2 lines: it builds no exclusive sets.
MCA_NAME = "mca" if MCA_K is None else f"mca{MCA_K}"
# A tagged run writes its aggregates beside the primary's, never over them. Per-model files
# (scores_*, attribution_*, comparator_weights_*) already carry the model name.
TAG = "" if MCA_NAME == "mca" else f"_{MCA_NAME}"


def tagged(name):
    stem, dot, ext = name.rpartition(".")
    return f"{stem}{TAG}{dot}{ext}"


# ----------------------------------------------------------------------------- shared pieces (cores.py)
def residualizer(C, w):
    A = np.column_stack([np.ones(len(w))] + ([C] if C is not None and C.shape[1] else []))
    sw = np.sqrt(w)[:, None]

    def apply(V):
        beta, *_ = np.linalg.lstsq(A * sw, V * sw, rcond=None)
        return V - A @ beta
    return apply


def pr(v):
    s2 = float((v ** 2).sum()); s4 = float((v ** 4).sum())
    return s2 * s2 / s4 if s4 > 0 else 0.0


def fixed_point_core(v):
    idx = np.arange(len(v))
    while True:
        n = max(1, min(int(round(pr(v[idx]))), len(idx)))
        top = np.sort(idx[np.argsort(-np.abs(v[idx]))[:n]])
        if len(top) == len(idx) and (top == idx).all():
            return top
        idx = top


def profile_str(counter):
    return "; ".join(f"{k}:{v}" for k, v in counter.most_common())


def load_battery():
    T = pd.read_csv(ITEMS_CSV)
    items = T["item"].astype(str).tolist()
    T = T.set_index("item")
    R = pd.read_csv(RESPONSES_CSV)
    R[ID_COL] = R[ID_COL].astype(str)
    R = R.set_index(ID_COL)
    raw = R[items].values.astype(float)
    assert not np.isnan(raw).any(), "missing response cells"
    lo, hi = T["scale_min"].values.astype(float), T["scale_max"].values.astype(float)
    rev = T["reverse_coded"].astype(str).str.lower().eq("true").values
    X = np.clip(2.0 * (raw - lo) / (hi - lo) - 1.0, -1.0, 1.0)
    X[:, rev] *= -1.0                                    # +1 = the instrument's keyed direction
    cov = R[COVARIATES].values.astype(float)
    w = (pd.read_csv(WEIGHTS_CSV).iloc[:, 0].values.astype(float)
         if WEIGHTS_CSV and os.path.exists(WEIGHTS_CSV) else np.ones(len(R)))
    assert len(w) == len(R)
    return R.index, items, T, X, cov, w


# ----------------------------------------------------------------------------- the representations
def scale_units(items, T):
    """The 34 published scoring units -> (S: 151 x q indicator, unit table).

    16 instrument totals, 5 BFI-2-XS trait scores, 13 single items that belong to no instrument."""
    units, rows = [], []
    inst = T.loc[items, "instrument_code"].astype(str).values
    con = T.loc[items, "construct"].astype(str).values
    for code in dict.fromkeys(inst):                      # first-appearance order
        pos = [i for i, c in enumerate(inst) if c == code]
        if code in PER_ITEM_INSTRUMENTS:
            groups = [(items[i], [i]) for i in pos]
        elif code in PER_CONSTRUCT_INSTRUMENTS:
            groups = [(f"{code}:{c}", [i for i in pos if con[i] == c]) for c in dict.fromkeys(con[i] for i in pos)]
        else:
            groups = [(code, pos)]
        for name, idx in groups:
            units.append(idx)
            rows.append({"unit": name, "instrument_code": code, "n_items": len(idx),
                         "items": " ".join(items[i] for i in idx),
                         "constructs": "; ".join(sorted(set(con[i] for i in idx)))})
    S = np.zeros((len(items), len(units)))
    for j, idx in enumerate(units):
        S[idx, j] = 1.0
    return S, pd.DataFrame(rows)


def parallel_analysis_k(Xs, n_perm, pctile, seed):
    """Horn's parallel analysis on the correlation matrix, permutation null (Buja & Eyuboglu)."""
    n = len(Xs)
    real = np.sort(np.linalg.eigvalsh(Xs.T @ Xs / (n - 1)))[::-1]
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, Xs.shape[1]))
    for b in range(n_perm):
        Xp = rng.permuted(Xs, axis=0)
        null[b] = np.sort(np.linalg.eigvalsh(Xp.T @ Xp / (n - 1)))[::-1]
    thr = np.percentile(null, pctile, axis=0)
    below = np.where(real <= thr)[0]
    k = int(below[0]) if below.size else Xs.shape[1]
    return k, real, thr


def factor_model(Xs, k, seed):
    fa = FactorAnalysis(n_components=k, rotation="varimax", random_state=seed).fit(Xs)
    Wpsi = fa.components_ / fa.noise_variance_                       # k x d
    cov_z = np.linalg.inv(np.eye(k) + Wpsi @ fa.components_.T)
    S = Wpsi.T @ cov_z                                               # d x k, regression scores
    chk = Xs[:2000]
    assert np.allclose(fa.transform(chk), (chk - fa.mean_) @ S, atol=1e-8), "factor-score weights do not reproduce transform()"
    L = fa.components_.T                                             # d x k standardized loadings
    return S, L, {"n_iter": int(fa.n_iter_), "loglike_final": float(fa.loglike_[-1]), "converged": bool(fa.n_iter_ < fa.max_iter)}


def pca_model(Xs, k):
    p = PCA(n_components=k, svd_solver="full").fit(Xs)
    return p.components_.T, p.components_.T, {"explained_variance_ratio": float(p.explained_variance_ratio_.sum())}


def name_columns(L, items, con, value_name="abs_loadings"):
    rows = []
    for c in range(L.shape[1]):
        core = fixed_point_core(L[:, c])
        core = core[np.argsort(-np.abs(L[core, c]))]
        prof = Counter(con[i] for i in core)
        rows.append({"factor": c, "size": len(core), "n_constructs": len(prof),
                     "construct_profile": profile_str(prof),
                     "items": " ".join(items[i] for i in core),
                     value_name: " ".join(f"{abs(L[i, c]):.3f}" for i in core)})
    return pd.DataFrame(rows)


def mca_representation(items, resp_raw, k):
    """The MCA grouping, from what `mca_basis.py` wrote. Returns Gs (category x k, already divided by
    Q*sigma) and `chosen` (n x Q, the category row each person occupies), so that a person's coordinate
    is the plain SUM over items of Gs[chosen[i, q]].

    ★ MCA IS NOT A LINEAR MAP OF X, which is what every other representation here is. It is linear in
    the indicator matrix: correspondence analysis gives F[i, m] = (1/(Q*sigma_m)) * sum_q G[chosen, m].
    The partition of T^2 needs only additivity over items, not linearity in the response values, so
    everything downstream is unchanged -- but the weight an item carries depends on WHICH CATEGORY the
    person endorsed rather than being a fixed column of S.

    `chosen` is rebuilt here rather than imported, by snapping each response to the nearest level LISTED
    IN THE CATEGORY FILE. That file is mca_basis.py's own output, so the levels agree by construction,
    including its snapping of imputed cells."""
    cc = os.path.join(OUT_DIR, "mca_category_coords.csv")
    ev = os.path.join(OUT_DIR, "mca_eigenvalues.csv")
    if not (os.path.exists(cc) and os.path.exists(ev)):
        return None
    C = pd.read_csv(cc)
    E = pd.read_csv(ev)
    axes = [c for c in C.columns if c.startswith("mca")]
    assert k <= len(axes), f"--mca-k {k} exceeds the {len(axes)} axes in {cc}; rerun mca_basis.py with --axes {k}"
    axes = axes[:k]
    sigma = np.sqrt(E["eigenvalue"].values[:k])
    Q = len(items)
    G = C[axes].values / (Q * sigma[None, :])                     # per-item contribution, ready to sum
    chosen = np.empty((len(resp_raw), Q), dtype=np.int64)
    for j, item in enumerate(items):
        rows = np.where(C["item"].values == item)[0]
        assert rows.size, f"{item} has no categories in {cc}"
        levels = C["value"].values[rows].astype(float)
        idx = np.abs(resp_raw[:, j][:, None] - levels[None, :]).argmin(axis=1)
        chosen[:, j] = rows[idx]
    return G, chosen, [f"MCA{c}" for c in range(k)], E


def score_and_attribute_mca(name, G, chosen, w, con, unit_names, ids, resid):
    """The same scoring as score_and_attribute, on a representation that is additive over items rather
    than linear in X. Everything after Y is identical: residualize, Ledoit-Wolf, T^2, the 95th
    percentile, the exact partition over units and over items."""
    n, Q = chosen.shape
    Yr = resid(G[chosen].sum(axis=1))                             # sum of one term per item, then residualized
    Yr = Yr - np.average(Yr, axis=0, weights=w)
    lw = LedoitWolf().fit(Yr)
    Om = lw.precision_
    V = Yr @ Om
    T2 = np.einsum("ij,ij->i", Yr, V)
    d = np.sqrt(np.maximum(T2, 0.0))
    pct = pd.Series(d).rank(pct=True).values * 100.0
    flag = d > np.percentile(d, PCTILE)
    a_unit = (Yr * V) / T2[:, None]
    lead_unit = a_unit.argmax(1)
    # item layer, exact. Residualizing is linear, so the residualized per-item terms still sum to Yr.
    ai = np.empty((n, Q))
    for q in range(Q):
        ai[:, q] = np.einsum("ij,ij->i", resid(G[chosen[:, q]]), V)
    assert np.allclose(ai.sum(1), T2, rtol=1e-6), f"{name}: item partition does not reproduce T2"
    si = ai / T2[:, None]
    cons = sorted(set(con))
    cidx = {c: [i for i, x in enumerate(con) if x == c] for c in cons}
    CS = np.column_stack([si[:, cidx[c]].sum(1) for c in cons])
    order = np.argsort(-CS, axis=1)
    sorted_cs = np.take_along_axis(CS, order, axis=1)
    n_carry = (np.cumsum(sorted_cs, axis=1) < 0.5).sum(1) + 1
    scores = pd.DataFrame({ID_COL: ids, "distance": d, "distance2": T2, "pctile": pct, "flagged": flag})
    attr = pd.DataFrame({ID_COL: ids, "flagged": flag,
                         "leading_unit": [unit_names[j] for j in lead_unit],
                         "leading_unit_share": a_unit[np.arange(len(ids)), lead_unit],
                         "leading_construct": [cons[j] for j in order[:, 0]],
                         "leading_construct_share": sorted_cs[:, 0],
                         "second_construct": [cons[j] for j in order[:, 1]],
                         "second_construct_share": sorted_cs[:, 1],
                         "n_carrying": n_carry})
    return scores, attr, pd.DataFrame(CS, index=ids, columns=cons), float(lw.shrinkage_), Om


# ----------------------------------------------------------------------------- scoring, identical for every representation
def score_and_attribute(name, S_eff, Xr, w, con, unit_names, ids):
    """Y = X S_eff residualized -> LW -> T2 -> flags; exact partition over units and over items -> constructs."""
    Yc = Xr @ S_eff
    Yc = Yc - np.average(Yc, axis=0, weights=w)
    lw = LedoitWolf().fit(Yc)
    Om = lw.precision_
    G = Yc @ Om
    T2 = np.einsum("ij,ij->i", Yc, G)
    d = np.sqrt(np.maximum(T2, 0.0))
    pct = pd.Series(d).rank(pct=True).values * 100.0
    thr = np.percentile(d, PCTILE)
    flag = d > thr
    # unit layer
    a_unit = (Yc * G) / T2[:, None]
    lead_unit = a_unit.argmax(1)
    # item layer, exact
    Q = S_eff @ Om @ S_eff.T
    ai = Xr * (Xr @ Q)
    assert np.allclose(ai.sum(1), T2, rtol=1e-6), f"{name}: item partition does not reproduce T2"
    si = ai / T2[:, None]
    cons = sorted(set(con))
    cidx = {c: [i for i, x in enumerate(con) if x == c] for c in cons}
    CS = np.column_stack([si[:, cidx[c]].sum(1) for c in cons])       # n x 30 construct shares
    order = np.argsort(-CS, axis=1)
    lead_con = order[:, 0]
    sorted_cs = np.take_along_axis(CS, order, axis=1)
    n_carry = (np.cumsum(sorted_cs, axis=1) < 0.5).sum(1) + 1          # constructs needed to reach half
    scores = pd.DataFrame({ID_COL: ids, "distance": d, "distance2": T2, "pctile": pct, "flagged": flag})
    attr = pd.DataFrame({ID_COL: ids, "flagged": flag,
                         "leading_unit": [unit_names[j] for j in lead_unit],
                         "leading_unit_share": a_unit[np.arange(len(ids)), lead_unit],
                         "leading_construct": [cons[j] for j in lead_con],
                         "leading_construct_share": sorted_cs[:, 0],
                         "second_construct": [cons[j] for j in order[:, 1]],
                         "second_construct_share": sorted_cs[:, 1],
                         "n_carrying": n_carry})
    return scores, attr, pd.DataFrame(CS, index=ids, columns=cons), float(lw.shrinkage_), Om


def overlap_row(A, B, a, b, N):
    A, B = set(A), set(B)
    inter, union = len(A & B), len(A | B)
    fA, fB = len(A) / N, len(B) / N
    null = fA * fB / (fA + fB - fA * fB)
    j = inter / union if union else float("nan")
    return {"A": a, "B": b, "n_A": len(A), "n_B": len(B), "intersection": inter,
            "jaccard": round(j, 4), "null_jaccard": round(null, 4), "x_null": round(j / null, 1) if null else float("nan"),
            "share_A_in_B": round(inter / len(A), 3) if A else float("nan"),
            "share_B_in_A": round(inter / len(B), 3) if B else float("nan")}


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ids, items, T, X, cov, w = load_battery()
    con = T.loc[items, "construct"].astype(str).tolist()
    N = len(ids)
    print(f"cohort {N:,} x {len(items)} items | covariates {COVARIATES} | weights {'unit' if np.allclose(w, 1) else 'supplied'}")
    resid = residualizer(cov, w)
    Xr = resid(X)
    Xraw = pd.read_csv(RESPONSES_CSV)[items].values.astype(float)   # codebook units, for MCA's categories
    sd = X.std(0)
    Xs = (X - X.mean(0)) / sd

    # framework flag sets: each encoder's own
    from disclosure import MARK, SUPP, marked, pair_hidden, partition_mask, small
    sets = {}
    for enc in ENCODERS:
        s = pd.read_csv(os.path.join(OUT_DIR, f"scores_{enc}.csv"))
        s[ID_COL] = s[ID_COL].astype(str)
        sets[enc] = set(s.loc[s["flagged"].astype(bool), ID_COL])
        assert sets[enc] <= set(ids), f"{enc}: flagged ids absent from the response file"
    PRIMARY = ENCODERS[0]
    print(f"framework: each encoder's own flagged set, {[len(sets[e]) for e in ENCODERS]}; "
          f"prespecified {PRIMARY}\n")

    # ---- representations
    reps, meta = {}, {}
    S_scale, units = scale_units(items, T)
    assert (S_scale.sum(1) == 1).all(), "every item belongs to exactly one unit"
    sd_y = (X @ S_scale).std(0)
    reps["scale"] = (S_scale / sd_y, units["unit"].tolist())            # standardized unit scores
    units.to_csv(os.path.join(OUT_DIR, tagged("comparator_units_scale.csv")), index=False)
    print(f"scale model: {len(units)} published scoring units "
          f"({int((units.n_items > 1).sum())} multi-item, {int((units.n_items == 1).sum())} single-item)")

    if K_OVERRIDE is None:
        k, real, thr = parallel_analysis_k(Xs, N_PERM, PA_PCTILE, SEED)
        print(f"parallel analysis on the response correlation matrix: k = {k} "
              f"(eigenvalues {', '.join(f'{v:.2f}' for v in real[:k + 2])}; null 95th at k: {thr[k - 1]:.3f}, at k+1: {thr[k]:.3f})")
        meta["parallel_analysis"] = {"k": k, "n_perm": N_PERM, "pctile": PA_PCTILE,
                                     "eigenvalues": [float(v) for v in real], "null_pctile": [float(v) for v in thr]}
    else:
        k = K_OVERRIDE
        print(f"k overridden to {k} (sensitivity)")
        meta["parallel_analysis"] = {"k": k, "overridden": True}
    assert k >= 2, "k < 2, nothing to fit"

    S_fa, L_fa, m_fa = factor_model(Xs, k, SEED)
    reps["factor"] = (S_fa / sd[:, None], [f"F{c}" for c in range(k)])  # weights in X units
    meta["factor"] = m_fa
    print(f"factor model: k = {k}, varimax, EM iterations {m_fa['n_iter']}, converged {m_fa['converged']}")
    S_pc, L_pc, m_pc = pca_model(Xs, k)
    reps["pca"] = (S_pc / sd[:, None], [f"PC{c}" for c in range(k)])
    meta["pca"] = m_pc
    print(f"pca: k = {k}, explained variance {100 * m_pc['explained_variance_ratio']:.1f}%")
    # ---- MCA, the fifth grouping. Scored at the same k unless --mca-k says otherwise.
    models = list(MODELS)
    mca = None
    if USE_MCA:
        mk = MCA_K if MCA_K is not None else k
        mca = mca_representation(items, Xraw, mk)
        if mca is None:
            print(f"\nmca: SKIPPED, screen_out/mca_category_coords.csv not found -- run mca_basis.py first")
        else:
            G_mca, chosen_mca, mca_names, E_mca = mca
            models.append(MCA_NAME)
            horn = int(E_mca["keep_horn"].sum()) if "keep_horn" in E_mca.columns else None
            how = ("matched to the factor and pca k" if MCA_K is None else "set by --mca-k")
            if horn:
                how += f"; MCA's own Horn k is {horn}"
            print(f"\n{MCA_NAME}: k = {mk} ({how}), "
                  f"{G_mca.shape[0]} answer categories over {len(items)} items")
            meta[MCA_NAME] = {"k": mk, "matched_to_k": MCA_K is None, "horn_k": horn,
                           "n_categories": int(G_mca.shape[0]),
                           "benzecri_pct_retained": (round(float(E_mca["benzecri_pct"].values[:mk].sum()), 2)
                                                     if "benzecri_pct" in E_mca.columns else None)}

    naming = [("factor", L_fa, "abs_loadings"), ("pca", L_pc, "abs_loadings")]
    if mca is not None:
        # ★ MCA axes are named the same way, from the DISCRIMINATION MEASURES rather than loadings.
        # eta^2 is the share of an axis's variance an item's categories separate: non-negative, so
        # the core rule reads concentration exactly as it does for a loading column, but the sign is
        # gone. WHICH END of an axis an answer sits at lives in mca_category_coords.csv, not here,
        # because an MCA axis contrasts ANSWER CATEGORIES and an item can enter through only some of
        # its levels. A two-sided MCA axis therefore cannot be read off this table alone.
        DM = pd.read_csv(os.path.join(OUT_DIR, "mca_discrimination.csv")).set_index("item")
        DM = DM.reindex(items)
        assert DM.notna().all().all(), "mca_discrimination.csv does not cover every item"
        naming.append((MCA_NAME, DM.values[:, :G_mca.shape[1]], "discrimination"))
    for name, L, vname in naming:
        F = name_columns(L, items, con, vname)
        F.to_csv(os.path.join(OUT_DIR, tagged(f"comparator_factors_{name}.csv")), index=False)
        print(f"  {name}: core size median {F['size'].median():.0f}, single-construct {int((F.n_constructs == 1).sum())} of {k}, "
              f"at most two {int((F.n_constructs <= 2).sum())}")
        for r in F.itertuples():
            print(f"    {name} {r.factor:2d} [{r.size}] {r.construct_profile}")

    # ---- scoring and attribution, identical downstream
    CSd, lead_rows = {}, []
    for m in models:
        if m == MCA_NAME:
            unit_names = mca_names
            scores, attr, CS, shrink, Om = score_and_attribute_mca(m, G_mca, chosen_mca, w, con,
                                                                   unit_names, ids, resid)
            S_eff = None
        else:
            S_eff, unit_names = reps[m]
            scores, attr, CS, shrink, Om = score_and_attribute(m, S_eff, Xr, w, con, unit_names, ids)
        # ★ A tagged run touches NOTHING the primary owns. scale, factor and pca are recomputed here
        # because the overlap and breadth tables need their flag sets, and they come back byte
        # identical, but an identical rewrite is still a rewrite, so the tagged run refuses to make
        # one and requires the primary to have run first.
        if TAG and m != MCA_NAME:
            need = [os.path.join(OUT_DIR, f"{pfx}_{m}.{ext}")
                    for pfx, ext in (("scores", "csv"), ("attribution", "csv"), ("comparator_weights", "npz"))]
            absent = [f for f in need if not os.path.exists(f)]
            assert not absent, (f"{m}: this is a tagged run ({MCA_NAME}) and it will not write the primary's "
                                f"files. Run `python3 comparators.py` first. Missing: "
                                + ", ".join(os.path.basename(f) for f in absent))
            print(f"  ({m}: recomputed for the tables, files left as the primary wrote them)")
            sets[m] = set(scores.loc[scores["flagged"], ID_COL])
            CSd[m] = (CS, attr.set_index(ID_COL))
            meta[f"{m}_n_flagged"] = int(scores["flagged"].sum())
            meta[f"{m}_lw_shrinkage"] = shrink
            continue
        scores.to_csv(os.path.join(OUT_DIR, f"scores_{m}.csv"), index=False)
        attr.to_csv(os.path.join(OUT_DIR, f"attribution_{m}.csv"), index=False)
        if S_eff is None:
            # ★ MCA has no S_eff: its weights are per category, not per item, so `care.py` cannot read
            # this file the way it reads the others and `mca` is deliberately NOT in its COMPARATORS
            # list. What is saved is enough to rebuild the per-item terms: G is already divided by
            # Q*sigma, so a person's coordinate is the sum of G over the categories they endorsed.
            np.savez(os.path.join(OUT_DIR, f"comparator_weights_{m}.npz"), G=G_mca, Om=Om,
                     units=np.array(unit_names), items=np.array(items), additive_over_items=True)
        else:
            np.savez(os.path.join(OUT_DIR, f"comparator_weights_{m}.npz"), S_eff=S_eff, Om=Om,
                     units=np.array(unit_names), items=np.array(items))   # read by care.py's --legacy path
        sets[m] = set(scores.loc[scores["flagged"], ID_COL])
        CSd[m] = (CS, attr.set_index(ID_COL))
        meta[f"{m}_n_flagged"] = int(scores["flagged"].sum())
        meta[f"{m}_lw_shrinkage"] = shrink
        q_m = len(unit_names)
        print(f"\n{m}: q = {q_m}, flagged {int(scores['flagged'].sum()):,} at p{PCTILE:.0f}, LW shrinkage {shrink:.4f}")
        for set_name in (m, PRIMARY):
            sel = attr[ID_COL].isin(sets[set_name]).values
            cnt = Counter(attr.loc[sel, "leading_construct"])
            mean_cs = CS[sel].mean(0)
            for c in CS.columns:
                lead_rows.append({"model": m, "flag_set": set_name, "n": int(sel.sum()), "construct": c,
                                  "n_leading": int(cnt.get(c, 0)), "mean_share": round(float(mean_cs[c]), 4)})
            top = cnt.most_common(6)
            print(f"  leading construct among {set_name} ({int(sel.sum()):,}): "
                  + ", ".join(f"{c} {n} ({100 * n / sel.sum():.0f}%)" for c, n in top))
            print(f"  mean carrying count among {set_name}: {attr.loc[sel, 'n_carrying'].mean():.2f}")
    LR = pd.DataFrame(lead_rows)          # n_leading partitions each model-and-set's n across constructs
    LR["n_leading"] = LR["n_leading"].astype(object)
    for _, ix in LR.groupby(["model", "flag_set"]).groups.items():
        ix = list(ix)
        nl = [int(v) for v in LR.loc[ix, "n_leading"]]
        hide, prim = partition_mask(nl)
        LR.loc[ix, "n_leading"] = marked(nl, hide, prim)
    LR.to_csv(os.path.join(OUT_DIR, tagged("comparator_leading_constructs.csv")), index=False)

    # ---- (i) flag overlap
    names = models + ENCODERS
    rows = [overlap_row(sets[a], sets[b], a, b, N) for i, a in enumerate(names) for b in names[i + 1:]]
    OV = pd.DataFrame(rows)
    OVw = OV.copy()          # the policy: an intersection and each set's remainder must be 0 or over 20
    hid = [pair_hidden(r.intersection, r.n_A) or pair_hidden(r.intersection, r.n_B) for r in OV.itertuples()]
    OVw.loc[hid, ["jaccard", "x_null", "share_A_in_B", "share_B_in_A"]] = np.nan
    OVw["intersection"] = [(MARK if small(r.intersection) else SUPP) if h else int(r.intersection)
                           for r, h in zip(OV.itertuples(), hid)]
    OVw.to_csv(os.path.join(OUT_DIR, tagged("comparator_flag_overlap.csv")), index=False)
    print("\n(i) flag overlap (Jaccard against the independence null; shares of each set inside the other)")
    show = OV[OV["A"].isin(models) & OV["B"].isin(models + [PRIMARY])]
    print(show.to_string(index=False))
    for m in models:
        js = OV[(OV["A"] == m) & OV["B"].isin(ENCODERS)]["jaccard"]
        print(f"  {m} against the six encoders: Jaccard {js.min():.3f} to {js.max():.3f}, mean {js.mean():.3f}")

    # ---- (ii) breadth, when the criteria exist
    cf_path = os.path.join(OUT_DIR, "cutoff_flags.csv")
    if os.path.exists(cf_path):
        cf = pd.read_csv(cf_path)
        cf[ID_COL] = cf[ID_COL].astype(str)
        cf = cf.set_index(ID_COL)
        ok = cf["all_evaluable"].astype(bool)          # breadth read only where every criterion is evaluable
        jp = os.path.join(OUT_DIR, "cutoff_summary.json")
        NB = int(json.load(open(jp))["n_criteria"]) if os.path.exists(jp) else None
        rows = []
        for s in models + ENCODERS + ["cohort"]:
            idx = cf.index if s == "cohort" else cf.index.intersection(list(sets[s]))
            b = cf.loc[idx].loc[ok.reindex(idx).values, "breadth"]
            n_s, n_b = (N if s == "cohort" else len(sets[s])), len(b)
            k_any = int(cf.loc[idx, "any_met"].astype(bool).sum())
            row = {"flag_set": s, "n": n_s, "n_all_evaluable": n_b, "mean_breadth": round(float(b.mean()), 3)}
            # each share is a count over n_all_evaluable (or n); withheld where that count or its
            # complement is 1 to 20
            for col, k_ in (("share_none", int((b == 0).sum())), ("share_ge2", int((b >= 2).sum())),
                            ("share_ge5", int((b >= 5).sum()))):
                row[col] = np.nan if pair_hidden(k_, n_b) else round(k_ / n_b, 4)
            row["share_any_met_everyone"] = np.nan if pair_hidden(k_any, n_s) else round(k_any / n_s, 4)
            if pair_hidden(n_b, n_s):
                row.update({"n_all_evaluable": SUPP, "mean_breadth": np.nan})
            rows.append(row)
        BR = pd.DataFrame(rows)
        BR.to_csv(os.path.join(OUT_DIR, tagged("comparator_breadth.csv")), index=False)
        print(f"\n(ii) breadth of each flag set (number of the {NB} criteria met, on people with all evaluable)")
        print(BR.to_string(index=False))
    else:
        print("\n(ii) breadth: pending, screen_out/cutoff_flags.csv not found (run cutoffs.py, then rerun)")

    # ---- (iii) attribution agreement on the consensus outliers, when the framework's attribution exists
    ap = os.path.join(OUT_DIR, "attribution_per_person_outliers.csv")
    if os.path.exists(ap):
        fw = pd.read_csv(ap)
        fw[ID_COL] = fw[ID_COL].astype(str)
        fw = fw.set_index(ID_COL)["top_construct"]
        rng = np.random.default_rng(SEED)
        unit_con = {r.unit: set(r.constructs.split("; ")) for r in units.itertuples()}
        rows = []
        for m in models:
            at = CSd[m][1].reindex(fw.index)
            agree = (at["leading_construct"].values == fw.values)
            floors = [float((rng.permutation(at["leading_construct"].values) == fw.values).mean()) for _ in range(200)]
            row = {"model": m, "n": len(fw), "agree_construct": round(float(agree.mean()), 4),
                   "floor_construct": round(float(np.mean(floors)), 4)}
            if m == "scale":
                in_unit = [c in unit_con[u] for c, u in zip(fw.values, at["leading_unit"].values)]
                row["agree_construct_in_leading_unit"] = round(float(np.mean(in_unit)), 4)
            rows.append(row)
        AG = pd.DataFrame(rows)
        AG.to_csv(os.path.join(OUT_DIR, tagged("comparator_attribution_agreement.csv")), index=False)
        print("\n(iii) leading construct on the consensus outliers, framework against comparator")
        print(AG.to_string(index=False))
    else:
        print("\n(iii) attribution agreement: not run (it needs a file that no script in this folder writes).")

    meta.update({"n_cohort": N, "n_items": len(items), "k": k, "pctile": PCTILE, "covariates": COVARIATES,
                 "n_units_scale": int(S_scale.shape[1]), "responses_csv": RESPONSES_CSV, "cdr": CDR,
                 "flag_sets": "each encoder's own",
                 "disclosure": "All of Us dissemination policy (disclosure.py)"})
    json.dump(meta, open(os.path.join(OUT_DIR, tagged("comparator_summary.json")), "w"), indent=2)
    print(f"\nwrote -> {OUT_DIR}")


if __name__ == "__main__":
    main()
