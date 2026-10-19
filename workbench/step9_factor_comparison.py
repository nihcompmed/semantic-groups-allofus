#!/usr/bin/env python3
"""step9_factor_comparison.py -- the outliers under each encoder against the outliers under the factor model (Figures 4
and 6, their supplements and source data; Appendix 4).

THE SETS, per encoder and factor model (k = 25; k = 3 in Appendix 4):  both / enc_only / fac_only = in both groupings'
set / the encoder's only / the factor model's only, for three kinds of set:
  factor_outliers    the outliers (distance above the grouping's own 95th percentile)
  factor_decile10    the most atypical tenth (decile 10 of each grouping's order)
  factor_below       inside the participants who met none of the six thresholds: their top 5% and their decile 10
                     (k = 25)
PER SET  size, age, sex; outcomes (thresholds met, the 3 barriers, the emergency department share on coded care, sleep,
         the 8 recorded conditions): raw and adjusted for age and sex by the joint model of the 3 sets (models.joint),
         with the contrast enc_only - fac_only (and enc_only - both) from the same fit; which sources and units (groups,
         factors) carry the distance; whether one questionnaire carries half.
DOWNLOADS factor_decile10, factor_outliers, factor_below, factor_outliers_ed (the emergency department share in the
          outlier sets) and factor_sources (one questionnaire carries half, each stand-alone question counted under the
          study it was taken from)

Run:  python3 step9_factor_comparison.py
"""
import json
import os

import numpy as np
import pandas as pd

import comparison as cp
import deciles as dc
import models
import outcomes as oc
import scoring as sc
from release import coarsen_sig, pair_hidden, small, versions, write_download
from settings import BASIS, CDR, ENCODERS, ID, PCTILE, download_dir, work

FACTOR_MODELS = [("factor", 25, "scores_factor.csv"), ("factor3", 3, "scores_factor3.csv")]
CONTRAST = "enc_only - fac_only"
CONTRAST_2SET = "enc_only - fac_only, 2-set model"


class Inputs:
    """Everything the 5 analyses share: the cohort, the outcomes, the answers, the factor models."""

    def __init__(self):
        cf, idx = dc.cohort_frame()
        self.cf, self.idx, self.N = cf, idx, len(idx)
        self.A = oc.load_conditions(idx)
        self.age, self.sex = self.A["age"], self.A["sex"]
        self.B, self.C, self.S, self.SA, _ = oc.load(idx)
        assert np.allclose(self.age, self.C["age"]) and np.allclose(self.sex, self.C["sex"])
        self.nights = dc.nights_of(idx)
        self.ev = self.B["ev"]
        self.none = self.ev & (self.B["br"] == 0)
        assert (cf["none_met"].astype(bool).values == self.none).all(), "none_met is not all-evaluable-and-0-met"
        T = pd.read_csv(sc.ITEMS_CSV)
        self.T = T
        self.items = T["item"].astype(str).tolist()
        self.con = T["construct"].astype(str).tolist()
        self.src_name = cp.per_question_labels(T)
        self.sources = list(dict.fromkeys(self.src_name))
        self.src_idx = np.array([self.sources.index(s) for s in self.src_name])
        R, X, cov, w = sc.load_battery(self.items)
        pos = pd.Index(R.index.astype(str)).get_indexer(idx)
        assert (pos >= 0).all(), "the cohort file does not cover the cohort"
        self.X, self.cov, self.w = X[pos], cov[pos], w[pos]
        Xr = sc.wls_residualize(self.X, self.cov, self.w)
        self.Xc = Xr - np.average(Xr, axis=0, weights=self.w)
        self.sd = self.X.std(0)
        self.Xs = (self.X - self.X.mean(0)) / self.sd
        self.FM = {name: cp.factor_model(name, k, f, self.Xs, self.sd, self.Xc, self.w, idx, self.items, self.con,
                                         self.src_name) for name, k, f in FACTOR_MODELS}
        for name in self.FM:
            self.FM[name]["sh"] = cp.by_source(self.FM[name]["a"], self.FM[name]["T2"], self.src_idx, len(self.sources))
            self.FM[name]["nh"] = cp.n_half(self.FM[name]["sh"])
        self.comp = pd.read_csv(os.path.join(BASIS, "components.csv"))
        self.coded = self.C["has_ehr"].astype(bool) & self.A["has_rec"].astype(bool) & np.isfinite(self.C["ed_share"])
        self.enc = {}
        for enc in ENCODERS:
            P = sc.place(enc, self.items)[0]
            Y = sc.wls_residualize(self.X @ P, self.cov, self.w)
            T2, lw = sc.mahalanobis2(Y, self.w)
            Yc = Y - np.average(Y, axis=0, weights=self.w)
            Om = lw.precision_
            s_ = pd.read_csv(work(f"scores_{enc}.csv"))
            s_[ID] = s_[ID].astype(str)
            worst = float(np.max(np.abs(np.sqrt(np.maximum(T2, 0)) - s_.set_index(ID)["distance"].reindex(idx).values)))
            assert worst < 1e-6, f"{enc}: recomputed distance differs from scores_{enc}.csv by {worst}"
            a, T2i = cp.item_division(self.Xc, P, Om)
            assert np.allclose(T2i, T2, rtol=1e-6), f"{enc}: the item division does not reproduce the squared distance"
            u = (Yc * (Yc @ Om)) / T2[:, None]
            assert np.abs(u.sum(1) - 1).max() < 1e-6, f"{enc}: group shares do not sum to 1"
            sh = cp.by_source(a, T2, self.src_idx, len(self.sources))
            kE = P.shape[1]
            cores = [sc.fixed_point_core(P[:, j]) for j in range(kE)]
            spans = np.array([len(set(self.src_name[i] for i in c)) >= 2 for c in cores])
            cpr = self.comp[self.comp.encoder == enc].sort_values("component")
            assert len(cpr) == kE, f"{enc}: components.csv has {len(cpr)} groups, P has {kE}"
            self.enc[enc] = {"a": a, "T2": T2, "u": u, "sh": sh, "nh": cp.n_half(sh), "kE": kE, "spans": spans,
                             "span_share": u[:, spans].sum(1), "names": cpr["construct_profile"].tolist(),
                             "srcs": ["; ".join(dict.fromkeys(self.src_name[i] for i in c)) for c in cores]}

    def specs(self, ed=False):
        """The outcomes compared between the sets: (name, model, who it is measured on, y, with_sex, extra)."""
        B, C, S, A = self.B, self.C, self.S, self.A
        sp = [] if ed else [("thresholds_met", "ols", B["ev"], B["br"], True, None)]
        for s_, col in oc.RATES.items():
            sp.append((s_, "logit", C["has_survey"] & np.isfinite(C[col]), C[col], True, None))
        if ed:
            sp.append(("ed_share_coded", "ols", self.coded, C["ed_share"], True, None))
        sp.append(("sleep_mean_min", "ols", S["has_sleep"], S["mean_min"], True, None))
        sp.append(("sleep_variation_min", "median", S["has_sleep"], S["sd_min"], True,
                   np.log(np.where(self.nights > 0, self.nights, 1))))
        for key, *_ in oc.CONDITIONS:
            sp.append((key, "logit", A[f"den_{key}"], A[f"y_{key}"], oc.with_sex(key), None))
        return sp


# ------------------------------------------------------------------------------------------------- decile 10
def factor_decile10(I):
    N = I.N
    names = ENCODERS + [f[0] for f in FACTOR_MODELS]
    order = dc.rank_orders(names, I.idx)
    d10 = {g: dc.decile_masks(order[g], N)[10] for g in names}
    pctl = {}
    for g in names:
        r = np.empty(N)
        r[order[g]] = np.arange(N)
        pctl[g] = 100.0 * (1.0 - r / N)
    rows_fac = []
    for name, k, _ in FACTOR_MODELS:
        for r in cp.ROTATIONS:
            q = I.FM[name]["rot"][r]
            for j in range(k):
                rows_fac.append({"factor_model": name, "k": k, "rotation": r, "factor": j + 1, "name": q["names"][j],
                                 "core_sources": q["sources"][j], "core_items": q["core_items"][j], "spans": q["spans"][j]})
    rows_set, rows_out, rows_src, rows_unit = [], [], [], []
    specs_all = I.specs()
    for enc in ENCODERS:
        E = I.enc[enc]
        for fname, fk, _ in FACTOR_MODELS:
            F = I.FM[fname]
            sets = {"both": d10[enc] & d10[fname], "enc_only": d10[enc] & ~d10[fname], "fac_only": d10[fname] & ~d10[enc]}
            allm = sets["both"] | sets["enc_only"] | sets["fac_only"]
            cat = np.full(N, -1)
            for i, s in enumerate(cp.SETS):
                cat[sets[s]] = i
            fem = {s: int((I.sex[sets[s]] == oc.FEMALE).sum()) for s in cp.SETS}
            one_e = {s: int((E["nh"][sets[s]] == 1).sum()) for s in cp.SETS}
            one_f = {s: int((F["nh"][sets[s]] == 1).sum()) for s in cp.SETS}
            hide = lambda kk: any(small(kk[s]) or small(sets[s].sum() - kk[s]) for s in cp.SETS)
            hf, he, hff = hide(fem), hide(one_e), hide(one_f)
            for s in cp.SETS:
                m = sets[s]
                n = int(m.sum())
                qe, qf = np.percentile(E["nh"][m], [25, 50, 75]), np.percentile(F["nh"][m], [25, 50, 75])
                rows_set.append({"encoder": enc, "factor_model": fname, "set": s, "n_set": n,
                                 "mean_age": float(I.age[m].mean()), "female_pct": np.nan if hf else 100 * fem[s] / n,
                                 "pctile_encoder_mean": float(pctl[enc][m].mean()),
                                 "pctile_factor_mean": float(pctl[fname][m].mean()),
                                 "nsrc_half_enc_mean": float(E["nh"][m].mean()), "nsrc_half_enc_q1": qe[0],
                                 "nsrc_half_enc_median": qe[1], "nsrc_half_enc_q3": qe[2],
                                 "one_source_half_enc_pct": np.nan if he else 100 * one_e[s] / n,
                                 "nsrc_half_fac_mean": float(F["nh"][m].mean()), "nsrc_half_fac_q1": qf[0],
                                 "nsrc_half_fac_median": qf[1], "nsrc_half_fac_q3": qf[2],
                                 "one_source_half_fac_pct": np.nan if hff else 100 * one_f[s] / n,
                                 "spanning_share_mean": float(E["span_share"][m].mean()),
                                 "groups_spanning": int(E["spans"].sum()), "groups_total": E["kE"]})
                for j, sname in enumerate(I.sources):
                    rows_src.append({"encoder": enc, "factor_model": fname, "set": s, "distance": "encoder",
                                     "source": sname, "mean_share": float(E["sh"][m, j].mean())})
                    rows_src.append({"encoder": enc, "factor_model": fname, "set": s, "distance": fname,
                                     "source": sname, "mean_share": float(F["sh"][m, j].mean())})
                units = [("semantic groups", "", E["u"], E["names"], E["srcs"], E["spans"])]
                units += [(fname, r, F["rot"][r]["u"], F["rot"][r]["names"], F["rot"][r]["sources"], F["rot"][r]["spans"])
                          for r in cp.ROTATIONS]
                for gname, r, U, labels, srcs, sp in units:
                    kk = U.shape[1]
                    top = np.argsort(-U[m], axis=1)[:, :min(3, kk)]
                    for j in range(kk):
                        nt = int((top == j).any(1).sum())
                        rows_unit.append({"encoder": enc, "factor_model": fname, "set": s, "grouping": gname,
                                          "rotation": r, "unit": j + 1, "name": labels[j], "core_sources": srcs[j],
                                          "spans": bool(sp[j]), "mean_share": float(U[m, j].mean()),
                                          "pct_top3": np.nan if (small(nt) or small(n - nt)) else 100 * nt / n})
            for oname, kind, have, y, ws, extra in specs_all:
                pop = allm & have
                yy = np.asarray(y, float)
                binary = kind == "logit"
                ks = {s: int(np.nansum(yy[pop & sets[s]])) for s in cp.SETS}
                ns = {s: int((pop & sets[s]).sum()) for s in cp.SETS}
                hid = binary and any(small(ks[s]) or small(ns[s] - ks[s]) for s in cp.SETS)
                Cv = models.covs(I.age[pop], I.sex[pop], ws, extra[pop] if extra is not None else None)
                est = models.joint(kind, yy[pop], cat[pop], [0, 1, 2], Cv, f"decile10 {enc} {fname} {oname}")
                for i, s in enumerate(cp.SETS):
                    mm = pop & sets[s]
                    raw = float(np.median(yy[mm])) if kind == "median" else float(yy[mm].mean())
                    rows_out.append({"encoder": enc, "factor_model": fname, "outcome": oname, "model": kind, "set": s,
                                     "n": ns[s], "k": (np.nan if hid else ks[s]) if binary else np.nan,
                                     "raw": np.nan if hid else raw, "adj": np.nan if hid else est[i][0],
                                     "adj_lo": np.nan if hid else est[i][1], "adj_hi": np.nan if hid else est[i][2],
                                     "p": np.nan})
                m2 = pop & (sets["enc_only"] | sets["fac_only"])
                C2 = models.covs(I.age[m2], I.sex[m2], ws, extra[m2] if extra is not None else None)
                d, lo, hi, p = models.contrast_two_sets(kind, yy[m2], sets["enc_only"][m2].astype(float), C2)
                rows_out.append({"encoder": enc, "factor_model": fname, "outcome": oname, "model": kind,
                                 "set": CONTRAST, "n": int(m2.sum()), "k": np.nan, "raw": np.nan,
                                 "adj": np.nan if hid else d, "adj_lo": np.nan if hid else lo,
                                 "adj_hi": np.nan if hid else hi, "p": np.nan if hid else p})
    d = download_dir("factor_decile10")
    files = {"decile10_sets.csv": pd.DataFrame(rows_set), "decile10_outcomes.csv": pd.DataFrame(rows_out),
             "decile10_sources.csv": pd.DataFrame(rows_src), "decile10_units.csv": pd.DataFrame(rows_unit),
             "decile10_factors.csv": pd.DataFrame(rows_fac)}
    for fn, D in files.items():
        D.to_csv(os.path.join(d, fn), index=False)
    json.dump({"cdr": CDR, "sets": cp.SETS, "factor_models": [f[:2] for f in FACTOR_MODELS], "rotations": cp.ROTATIONS,
               "oblimin": {f[0]: I.FM[f[0]]["oblimin"] for f in FACTOR_MODELS}, "versions": versions()},
              open(os.path.join(d, "decile10_meta.json"), "w"), indent=2, default=str)
    write_download("factor_decile10", list(files) + ["decile10_meta.json"])


# ------------------------------------------------------------------------------------------------- the outliers
def factor_outliers(I):
    N = I.N
    names = ENCODERS + [f[0] for f in FACTOR_MODELS]
    out, pctl = {}, {}
    for g in names:
        out[g], _ = cp.outliers(g, I.idx)
        s = pd.read_csv(work(f"scores_{g}.csv"))
        s[ID] = s[ID].astype(str)
        r = s.set_index(ID).reindex(I.idx)["distance"].rank(method="first").values
        pctl[g] = 100.0 * r / N
    assert len({int(m.sum()) for m in out.values()}) == 1, "the groupings have different numbers of outliers"
    rows_grp, rows_gunit = [], []

    def grouping_rows(gname, model, rotation, U, labels, srcs, spans, nh, span_share):
        m_all = out[model]
        n_o, n_e, n_n = int(m_all.sum()), int((m_all & I.ev).sum()), int((m_all & I.none).sum())
        he, hn = pair_hidden(n_e, n_o), pair_hidden(n_n, n_e)
        for sname, m in (("all outliers", m_all), ("below every threshold", m_all & I.none)):
            n = int(m.sum())
            hidden = sname == "below every threshold" and hn
            base = {"grouping": gname, "model": model, "rotation": rotation, "set": sname,
                    "n_set": np.nan if hidden or small(n) else n}
            if sname == "all outliers":
                base.update({"n_evaluable": np.nan if he else n_e, "n_none_met": np.nan if (he or hn) else n_n,
                             "none_met_pct": np.nan if (he or hn) else 100 * n_n / n_e})
            rows_grp.append({**base, **(cp.breadth_row(m, nh, span_share) if not hidden
                                        else cp.breadth_row(m & False, nh, None))})
            if not hidden:
                n_, t1_, t3_ = cp.top_counts(U, m)
                ms_ = U[m].mean(0) if n_ else np.zeros(U.shape[1])
                for x in cp.unit_rows(n_, t1_, t3_, ms_, labels, srcs, spans):
                    rows_gunit.append({"grouping": gname, "model": model, "rotation": rotation, "set": sname, **x})

    for name, k, _ in FACTOR_MODELS:
        F = I.FM[name]
        for r in cp.ROTATIONS:
            q = F["rot"][r]
            sp = np.array(q["spans"])
            grouping_rows(f"factor k = {k}", name, r, q["u"], q["names"], q["sources"], sp, F["nh"], q["u"][:, sp].sum(1))
    rows_set, rows_out, rows_src, rows_unit = [], [], [], []
    SU, GU = {}, {}
    specs_all = I.specs()
    for enc in ENCODERS:
        E = I.enc[enc]
        grouping_rows("semantic groups", enc, "", E["u"], E["names"], E["srcs"], E["spans"], E["nh"], E["span_share"])
        for fname, fk, _ in FACTOR_MODELS:
            F = I.FM[fname]
            sets = {"both": out[enc] & out[fname], "enc_only": out[enc] & ~out[fname], "fac_only": out[fname] & ~out[enc]}
            allm = sets["both"] | sets["enc_only"] | sets["fac_only"]
            cat = np.full(N, -1)
            for i, s in enumerate(cp.SETS):
                cat[sets[s]] = i
            ns_ = {s: int(sets[s].sum()) for s in cp.SETS}
            tiny = any(small(ns_[s]) or ns_[s] == 0 for s in cp.SETS)
            fem = {s: int((I.sex[sets[s]] == oc.FEMALE).sum()) for s in cp.SETS}
            nev = {s: int((sets[s] & I.ev).sum()) for s in cp.SETS}
            nno = {s: int((sets[s] & I.none).sum()) for s in cp.SETS}
            hide = lambda kk, tot: any(small(kk[s]) or small(tot[s] - kk[s]) for s in cp.SETS)
            hf, hev = hide(fem, ns_), hide(nev, ns_)
            hno = hev or hide(nno, nev)
            for s in cp.SETS:
                m = sets[s]
                n = ns_[s]
                if tiny:
                    rows_set.append({"encoder": enc, "factor_model": fname, "set": s, "n_set": np.nan})
                    continue
                be, bf = cp.breadth_row(m, E["nh"], E["span_share"]), cp.breadth_row(m, F["nh"], None)
                rows_set.append({"encoder": enc, "factor_model": fname, "set": s, "n_set": n,
                                 "mean_age": float(I.age[m].mean()), "female_pct": np.nan if hf else 100 * fem[s] / n,
                                 "pctile_encoder_mean": float(pctl[enc][m].mean()),
                                 "pctile_factor_mean": float(pctl[fname][m].mean()),
                                 "n_evaluable": np.nan if hev else nev[s], "n_none_met": np.nan if hno else nno[s],
                                 "none_met_pct": np.nan if hno else 100 * nno[s] / nev[s],
                                 **{c.replace("nsrc_half", "nsrc_half_enc").replace("one_source_half",
                                                                                    "one_source_half_enc"): v
                                    for c, v in be.items() if c != "spanning_share_mean"},
                                 **{c.replace("nsrc_half", "nsrc_half_fac").replace("one_source_half",
                                                                                    "one_source_half_fac"): v
                                    for c, v in bf.items() if c != "spanning_share_mean"},
                                 "spanning_share_mean": be["spanning_share_mean"],
                                 "groups_spanning": int(E["spans"].sum()), "groups_total": E["kE"]})
                for j, sname in enumerate(I.sources):
                    rows_src.append({"encoder": enc, "factor_model": fname, "set": s, "distance": "encoder",
                                     "source": sname, "mean_share": float(E["sh"][m, j].mean())})
                    rows_src.append({"encoder": enc, "factor_model": fname, "set": s, "distance": fname,
                                     "source": sname, "mean_share": float(F["sh"][m, j].mean())})
            if not tiny:
                units = [(("semantic groups", enc, ""), E["u"], E["names"], E["srcs"], E["spans"])]
                units += [((fname, fname, r), F["rot"][r]["u"], F["rot"][r]["names"], F["rot"][r]["sources"],
                           F["rot"][r]["spans"]) for r in cp.ROTATIONS]
                for key, U, labels, srcs, sp in units:
                    for s in cp.SETS:
                        n_, t1_, t3_ = cp.top_counts(U, sets[s])
                        SU.setdefault(key, {})[(enc, fname, s)] = (n_, t1_, t3_, U[sets[s]].mean(0), labels, srcs, sp)
                    if key not in GU:
                        GU[key] = cp.top_counts(U, out[key[1]])[1:]
            for oname, kind, have, y, ws, extra in specs_all:
                pop = allm & have
                yy = np.asarray(y, float)
                binary = kind == "logit"
                ks = {s: int(np.nansum(yy[pop & sets[s]])) for s in cp.SETS}
                ns = {s: int((pop & sets[s]).sum()) for s in cp.SETS}
                hid_n = tiny or any(small(ns[s]) or ns[s] == 0 for s in cp.SETS)
                hid = hid_n or (binary and any(small(ks[s]) or small(ns[s] - ks[s]) for s in cp.SETS))
                if hid:
                    for s in cp.SETS + [CONTRAST, CONTRAST_2SET]:
                        n_w = int((pop & (sets["enc_only"] | sets["fac_only"])).sum()) if s not in cp.SETS else ns[s]
                        rows_out.append({"encoder": enc, "factor_model": fname, "outcome": oname, "model": kind,
                                         "set": s, "n": np.nan if hid_n else n_w, "k": np.nan, "raw": np.nan,
                                         "adj": np.nan, "adj_lo": np.nan, "adj_hi": np.nan, "p": np.nan})
                    continue
                Cv = models.covs(I.age[pop], I.sex[pop], ws, extra[pop] if extra is not None else None)
                est, con = models.joint_with_contrasts(kind, yy[pop], cat[pop], [0, 1, 2], Cv,
                                                       f"outliers {enc} {fname} {oname}", [(CONTRAST, 1, 2)])
                con = con[CONTRAST]
                for i, s in enumerate(cp.SETS):
                    mm = pop & sets[s]
                    raw = float(np.median(yy[mm])) if kind == "median" else float(yy[mm].mean())
                    rows_out.append({"encoder": enc, "factor_model": fname, "outcome": oname, "model": kind, "set": s,
                                     "n": ns[s], "k": ks[s] if binary else np.nan, "raw": raw, "adj": est[i][0],
                                     "adj_lo": est[i][1], "adj_hi": est[i][2], "p": np.nan})
                m2 = pop & (sets["enc_only"] | sets["fac_only"])
                C2 = models.covs(I.age[m2], I.sex[m2], ws, extra[m2] if extra is not None else None)
                d2 = models.contrast_two_sets(kind, yy[m2], sets["enc_only"][m2].astype(float), C2)
                for lab, (dd, ll, hh, pp) in ((CONTRAST, con), (CONTRAST_2SET, d2)):
                    rows_out.append({"encoder": enc, "factor_model": fname, "outcome": oname, "model": kind,
                                     "set": lab, "n": int(m2.sum()), "k": np.nan, "raw": np.nan, "adj": dd,
                                     "adj_lo": ll, "adj_hi": hh, "p": pp})
    for key, D in SU.items():                       # the set-level unit counts, hidden as linked systems
        semantic = key[0] == "semantic groups"
        other = "enc_only" if semantic else "fac_only"
        links = [("G", (e, f, "both"), (e, f, other)) for (e, f, s) in D if s == "both"]
        H1 = cp.secure_masks({"G": GU[key][0], **{b: v[1] for b, v in D.items()}}, links, {"G"})
        H3 = cp.secure_masks({"G": GU[key][1], **{b: v[2] for b, v in D.items()}}, links, {"G"})
        for (e, f, s), (n_, t1_, t3_, ms_, labels, srcs, sp) in D.items():
            for x in cp.unit_rows(n_, t1_, t3_, ms_, labels, srcs, sp, masks=(H1[(e, f, s)], H3[(e, f, s)])):
                rows_unit.append({"encoder": e, "factor_model": f, "set": s,
                                  "grouping": "semantic groups" if semantic else f, "rotation": key[2], **x})
    d = download_dir("factor_outliers")
    files = {"outliers_sets.csv": pd.DataFrame(rows_set), "outliers_outcomes.csv": pd.DataFrame(rows_out),
             "outliers_sources.csv": pd.DataFrame(rows_src), "outliers_units.csv": pd.DataFrame(rows_unit),
             "outliers_groupings.csv": pd.DataFrame(rows_grp), "outliers_grouping_units.csv": pd.DataFrame(rows_gunit)}
    for fn, D in files.items():
        fl = [c for c in D.columns if D[c].dtype.kind == "f" and not c.startswith("n_") and c not in ("n", "k")]
        coarsen_sig(D, fl).to_csv(os.path.join(d, fn), index=False)
    json.dump({"cdr": CDR, "pctile": PCTILE, "sets": cp.SETS, "factor_models": [f[:2] for f in FACTOR_MODELS],
               "rotations": cp.ROTATIONS, "contrast": "enc_only - fac_only from the same fit as the set values; the 2-set "
               "model's contrast written beside it", "versions": versions()},
              open(os.path.join(d, "outliers_meta.json"), "w"), indent=2, default=str)
    write_download("factor_outliers", list(files) + ["outliers_meta.json"])
    return out


# ------------------------------------------------------------------------------------------------- below every threshold
def factor_below(I):
    N = I.N
    FNAME = "factor"
    pairs = [("enc_only - fac_only", 1, 2), ("enc_only - both", 1, 0)]
    F = dict(I.FM[FNAME])
    names = ENCODERS + [FNAME]
    order = dc.rank_orders(names, I.idx)
    cuts = {"top 5%": {g: cp.outliers(g, I.idx)[0] for g in names},
            "top 10%": {g: dc.decile_masks(order[g], N)[10] for g in names}}
    specs = I.specs(ed=True)
    rows_set, rows_out, rows_unit = [], [], []
    SU, GU = {}, {}
    size = {}
    for enc in ENCODERS:
        E = I.enc[enc]
        for cut, m in cuts.items():
            none = I.none
            sets = {"both": none & m[enc] & m[FNAME], "enc_only": none & m[enc] & ~m[FNAME],
                    "fac_only": none & m[FNAME] & ~m[enc]}
            ns_ = {s: int(sets[s].sum()) for s in cp.SETS}
            size[(cut, enc)] = ns_
            tiny = any(small(ns_[s]) or ns_[s] == 0 for s in cp.SETS)
            if cut == "top 10%" and ("top 5%", enc) in size:
                n5 = size[("top 5%", enc)]
                tiny = tiny or any(small(ns_[s] - n5[s]) for s in cp.SETS)
            if tiny:
                rows_set += [{"cut": cut, "encoder": enc, "set": s, "n_set": np.nan} for s in cp.SETS]
                continue
            fem = {s: int((I.sex[sets[s]] == oc.FEMALE).sum()) for s in cp.SETS}
            hf = any(small(fem[s]) or small(ns_[s] - fem[s]) for s in cp.SETS)
            found = ns_["both"] / (ns_["both"] + ns_["enc_only"])
            for s in cp.SETS:
                mm = sets[s]
                be, bf = cp.breadth_row(mm, E["nh"], None), cp.breadth_row(mm, F["nh"], None)
                rows_set.append({"cut": cut, "encoder": enc, "set": s, "n_set": ns_[s],
                                 "mean_age": float(I.age[mm].mean()), "female_pct": np.nan if hf else 100 * fem[s] / ns_[s],
                                 "found_by_factor_pct": 100 * found if s != "fac_only" else np.nan,
                                 **{k.replace("nsrc_half", "nsrc_half_enc").replace("one_source_half", "one_source_half_enc"): v
                                    for k, v in be.items() if k != "spanning_share_mean"},
                                 **{k.replace("nsrc_half", "nsrc_half_fac").replace("one_source_half", "one_source_half_fac"): v
                                    for k, v in bf.items() if k != "spanning_share_mean"}})
            units = [(("semantic groups", enc, "", cut), E["u"], E["names"], E["srcs"], E["spans"])]
            units += [((FNAME, FNAME, r, cut), F["rot"][r]["u"], F["rot"][r]["names"], F["rot"][r]["sources"],
                       F["rot"][r]["spans"]) for r in cp.ROTATIONS]
            for key, U, labels, srcs, sp in units:
                for s in cp.SETS:
                    n_, t1_, t3_ = cp.top_counts(U, sets[s])
                    SU.setdefault(key, {})[(enc, s)] = (n_, t1_, t3_, U[sets[s]].mean(0), labels, srcs, sp)
                if key not in GU:
                    GU[key] = cp.top_counts(U, none & m[key[1]])[1:]
            allm = sets["both"] | sets["enc_only"] | sets["fac_only"]
            cat = np.full(N, -1)
            for i, s in enumerate(cp.SETS):
                cat[sets[s]] = i
            for oname, kind, have, y, ws, extra in specs:
                pop = allm & have
                yy = np.asarray(y, float)
                binary = kind == "logit"
                ks = {s: int(np.nansum(yy[pop & sets[s]])) for s in cp.SETS}
                ns = {s: int((pop & sets[s]).sum()) for s in cp.SETS}
                n_ok = {s: not (small(ns[s]) or small(ns_[s] - ns[s])) for s in cp.SETS}
                k_ok = {s: n_ok[s] and not (small(ks[s]) or small(ns[s] - ks[s])) for s in cp.SETS}
                fit_ok = all(ns[s] > 0 and not small(ns[s]) for s in cp.SETS) and \
                    (not binary or all(not (small(ks[s]) or small(ns[s] - ks[s])) for s in cp.SETS))
                est = con_ = None
                if fit_ok:
                    Cv = models.covs(I.age[pop], I.sex[pop], ws, extra[pop] if extra is not None else None)
                    est, con_ = models.joint_with_contrasts(kind, yy[pop], cat[pop], [0, 1, 2], Cv,
                                                            f"below {cut} {enc} {oname}", pairs)
                for i, s in enumerate(cp.SETS):
                    mm = pop & sets[s]
                    raw = (float(np.median(yy[mm])) if kind == "median" else float(yy[mm].mean())) if ns[s] else np.nan
                    rows_out.append({"cut": cut, "encoder": enc, "outcome": oname, "model": kind, "set": s,
                                     "n": ns[s] if n_ok[s] else np.nan,
                                     "k": (ks[s] if k_ok[s] else np.nan) if binary else np.nan,
                                     "raw": raw if (k_ok[s] if binary else n_ok[s]) else np.nan,
                                     "adj": est[i][0] if fit_ok else np.nan, "adj_lo": est[i][1] if fit_ok else np.nan,
                                     "adj_hi": est[i][2] if fit_ok else np.nan, "p": np.nan})
                for lab, i, j in pairs:
                    a_, b_ = cp.SETS[i], cp.SETS[j]
                    dd = con_[lab] if fit_ok else (np.nan,) * 4
                    rows_out.append({"cut": cut, "encoder": enc, "outcome": oname, "model": kind, "set": lab,
                                     "n": ns[a_] + ns[b_] if (n_ok[a_] and n_ok[b_]) else np.nan, "k": np.nan,
                                     "raw": np.nan, "adj": dd[0], "adj_lo": dd[1], "adj_hi": dd[2], "p": dd[3]})
    for key, D in SU.items():
        semantic = key[0] == "semantic groups"
        other = "enc_only" if semantic else "fac_only"
        links = [("G", (e, "both"), (e, other)) for (e, s) in D if s == "both"]
        H1 = cp.secure_masks({"G": GU[key][0], **{b: v[1] for b, v in D.items()}}, links, {"G"})
        H3 = cp.secure_masks({"G": GU[key][1], **{b: v[2] for b, v in D.items()}}, links, {"G"})
        for (e, s), (n_, t1_, t3_, ms_, labels, srcs, sp) in D.items():
            for x in cp.unit_rows(n_, t1_, t3_, ms_, labels, srcs, sp, masks=(H1[(e, s)], H3[(e, s)])):
                rows_unit.append({"cut": key[3], "encoder": e, "set": s,
                                  "grouping": "semantic groups" if semantic else FNAME, "rotation": key[2], **x})
    d = download_dir("factor_below")
    files = {"below_sets.csv": pd.DataFrame(rows_set), "below_outcomes.csv": pd.DataFrame(rows_out),
             "below_units.csv": pd.DataFrame(rows_unit)}
    for fn, D in files.items():
        fl = [c for c in D.columns if D[c].dtype.kind == "f" and not c.startswith("n_") and c not in ("n", "k")]
        coarsen_sig(D, fl).to_csv(os.path.join(d, fn), index=False)
    json.dump({"cdr": CDR, "cuts": list(cuts), "sets": cp.SETS, "contrasts": [c[0] for c in pairs],
               "factor_model": [FNAME, 25], "rotations": cp.ROTATIONS, "versions": versions()},
              open(os.path.join(d, "below_meta.json"), "w"), indent=2, default=str)
    write_download("factor_below", list(files) + ["below_meta.json"])


# ------------------------------------------------------------------------------------------------- the ED share
def factor_outliers_ed(I):
    N = I.N
    FNAME = "factor"
    visit = I.C["has_ehr"].astype(bool)
    y_all = np.asarray(I.C["ed_share"], float)
    out = {g: cp.outliers(g, I.idx)[0] for g in ENCODERS + [FNAME]}
    rows = []
    for enc in ENCODERS:
        sets = {"both": out[enc] & out[FNAME], "enc_only": out[enc] & ~out[FNAME], "fac_only": out[FNAME] & ~out[enc]}
        ns_ = {s: int(sets[s].sum()) for s in cp.SETS}
        cat = np.full(N, -1)
        for i, s in enumerate(cp.SETS):
            cat[sets[s]] = i
        pop = (cat >= 0) & I.coded
        ns = {s: int((pop & sets[s]).sum()) for s in cp.SETS}
        nv = {s: int((sets[s] & visit).sum()) for s in cp.SETS}
        n_ok = {s: not (pair_hidden(ns[s], ns_[s]) or pair_hidden(ns[s], nv[s])) for s in cp.SETS}
        fit_ok = all(ns[s] > 0 and n_ok[s] for s in cp.SETS)
        est = con = None
        if fit_ok:
            Cv = models.covs(I.age[pop], I.sex[pop], True)
            est, con = models.joint_with_contrasts("ols", y_all[pop], cat[pop], [0, 1, 2], Cv,
                                                   f"outliers {enc} ed_share_coded", [(CONTRAST, 1, 2)])
            con = con[CONTRAST]
        for i, s in enumerate(cp.SETS):
            mm = pop & sets[s]
            rows.append({"encoder": enc, "factor_model": FNAME, "outcome": "ed_share_coded", "model": "ols", "set": s,
                         "n": ns[s] if n_ok[s] else np.nan, "k": np.nan,
                         "raw": float(y_all[mm].mean()) if (n_ok[s] and ns[s]) else np.nan,
                         "adj": est[i][0] if fit_ok else np.nan, "adj_lo": est[i][1] if fit_ok else np.nan,
                         "adj_hi": est[i][2] if fit_ok else np.nan, "p": np.nan})
        dd = con if fit_ok else (np.nan,) * 4
        rows.append({"encoder": enc, "factor_model": FNAME, "outcome": "ed_share_coded", "model": "ols", "set": CONTRAST,
                     "n": ns["enc_only"] + ns["fac_only"] if (n_ok["enc_only"] and n_ok["fac_only"]) else np.nan,
                     "k": np.nan, "raw": np.nan, "adj": dd[0], "adj_lo": dd[1], "adj_hi": dd[2], "p": dd[3]})
    D = pd.DataFrame(rows)
    fl = [c for c in D.columns if D[c].dtype.kind == "f" and c not in ("n", "k")]
    d = download_dir("factor_outliers_ed")
    coarsen_sig(D, fl).to_csv(os.path.join(d, "outliers_ed.csv"), index=False)
    json.dump({"cdr": CDR, "outcome": "ed_share_coded", "sets": cp.SETS, "factor_model": [FNAME, 25],
               "contrast": "enc_only - fac_only (same fit as the set values)", "versions": versions()},
              open(os.path.join(d, "outliers_ed_meta.json"), "w"), indent=2, default=str)
    write_download("factor_outliers_ed", ["outliers_ed.csv", "outliers_ed_meta.json"])


# ------------------------------------------------------------------------------------------------- one questionnaire carries half
def factor_sources(I):
    N = I.N
    FNAME = "factor"
    lab25 = cp.source_labels(I.T)
    assert len(set(lab25)) == 25, f"{len(set(lab25))} sources, expected 25"
    F = I.FM[FNAME]
    NH = {FNAME: cp.nh_for(F["a"], F["T2"], lab25)}
    names = ENCODERS + [FNAME]
    out = {g: cp.outliers(g, I.idx)[0] for g in names}
    order = dc.rank_orders(names, I.idx)
    d10m = {g: dc.decile_masks(order[g], N)[10] for g in names}
    for enc in ENCODERS:
        NH[enc] = cp.nh_for(I.enc[enc]["a"], I.enc[enc]["T2"], lab25)
    cells = []
    for g in names:
        for sname, m in (("all outliers", out[g]), ("below every threshold", out[g] & I.none)):
            cells.append({"file": "groupings", "grouping": g, "set": sname, "distance": g, "mask": m})
    for enc in ENCODERS:
        sets = {"both": out[enc] & out[FNAME], "enc_only": out[enc] & ~out[FNAME], "fac_only": out[FNAME] & ~out[enc]}
        for s, m in sets.items():
            for dist in ((enc, FNAME) if s != "both" else (enc,)):
                cells.append({"file": "outlier_sets", "encoder": enc, "set": s, "distance": dist, "mask": m})
        for cut, mm in (("top 5%", out), ("top 10%", d10m)):
            sets = {"both": I.none & mm[enc] & mm[FNAME], "enc_only": I.none & mm[enc] & ~mm[FNAME],
                    "fac_only": I.none & mm[FNAME] & ~mm[enc]}
            for s, m in sets.items():
                cells.append({"file": "below_sets", "cut": cut, "encoder": enc, "set": s, "distance": enc, "mask": m})
    for dist in names:                                      # the policy, per distance
        fam = [c for c in cells if c["distance"] == dist]
        ind = (NH[dist] == 1).astype(int)
        for c in fam:
            c["ind"], c["nh"] = ind, NH[dist]
            n = int(c["mask"].sum())
            c["write"] = n > 0 and not small(n)
        cp.secure_cells(fam)

    def summary(c):
        m = c["mask"]
        n = int(m.sum())
        if not c["write"] or n == 0:
            return {"one_source_half_pct": np.nan, "nsrc_half_mean": np.nan, "nsrc_half_median": np.nan}
        nh = c["nh"][m]
        return {"one_source_half_pct": 100.0 * float((nh == 1).mean()), "nsrc_half_mean": float(nh.mean()),
                "nsrc_half_median": float(np.median(nh))}

    def frame(kind, keys):
        rows = []
        for c in cells:
            if c["file"] != kind:
                continue
            n = int(c["mask"].sum())
            rows.append({**{k: c[k] for k in keys}, "distance": "encoder" if c["distance"] in ENCODERS and kind != "groupings"
                         else ("own" if kind == "groupings" else "factor"),
                         "n_set": n if not small(n) else np.nan, **summary(c)})
        D = pd.DataFrame(rows)
        return coarsen_sig(D, [x for x in D.columns if D[x].dtype.kind == "f" and x != "n_set"])

    d = download_dir("factor_sources")
    files = {"sources_outlier_sets.csv": frame("outlier_sets", ["encoder", "set"]),
             "sources_groupings.csv": frame("groupings", ["grouping", "set"]),
             "sources_below_sets.csv": frame("below_sets", ["cut", "encoder", "set"])}
    for fn, D in files.items():
        D.to_csv(os.path.join(d, fn), index=False)
    json.dump({"cdr": CDR, "n_sources": 25, "sets": cp.SETS, "factor_model": [FNAME, 25], "versions": versions()},
              open(os.path.join(d, "sources_meta.json"), "w"), indent=2, default=str)
    write_download("factor_sources", list(files) + ["sources_meta.json"])


def main():
    I = Inputs()
    print(f"[step 9] cohort {I.N:,} | factor models fitted", flush=True)
    factor_decile10(I)
    factor_outliers(I)
    factor_below(I)
    factor_outliers_ed(I)
    factor_sources(I)
    bad = [t for t, v in models.CONVERGED.items() if not v and "median" not in t]
    assert not bad, f"logistic fits that did not converge: {bad[:5]}"


if __name__ == "__main__":
    main()
