#!/usr/bin/env python
"""etable_conditions_jno.py -- eTables 9 to 11, semantic groups and recorded conditions: the 8 conditions of record,
their populations and the thresholds P < .05 (joint) and P < .05/k (each group).

LOCAL. Reads the downloads in outputs/workbench/R2025Q4R6/:
  conditions_assoc_jno/conditions_assoc{,_or}_jno_<enc>.csv                     (workbench/conditions_assoc_jno.py)
  conditions_assoc_adjusted_jno/conditions_assoc_adjusted{,_or}_jno_<enc>.csv   (workbench/conditions_assoc_adjusted_jno.py)
Writes outputs/tables/:
  etable_assoc_jno.{tex,csv}     eTable 10   per condition: participants with and without it, and the number of
                                 groups significant together and alone under each encoder (joint test in the caption)
  etable_breast_jno.{tex,csv}    eTable 11   breast and prostate cancer: every group passing together or alone,
                                 under each encoder, with both odds ratios (sizes) and 95% CIs
  etable_control_jno.{tex,csv}   eTable 9    before and after the adjustment for income, education, insurance
                                 and the 2 barriers: groups passing together before and after, alone after, and the
                                 median share of each log odds ratio kept, per condition and encoder
  (bands(), breast cancer within screening age bands, is not in the paper and is not called)
Odds ratios are SIZES: OR or 1/OR, whichever is >= 1, the interval inverted with it (a group's sign is arbitrary).
P values in JAMA style (<.001; 3 decimals below .01; 2 decimals otherwise).

Run:  python etable_conditions_jno.py
"""
import os

import numpy as np
import pandas as pd

from construct_labels import display_listed   # display names in the printed table (CSV keeps the data labels)
from etable_deciles_jno import CONDITIONS, ENCODERS

HERE = os.path.dirname(os.path.abspath(__file__))
WB = os.path.join(HERE, "..", "outputs", "workbench", "R2025Q4R6")
OUT = os.path.join(HERE, "..", "outputs", "tables")
from conditions_local import MENTAL, PHYSICAL, TESTED as _TESTED   # the 8 conditions of record
LAB = dict(CONDITIONS)
TESTED = set(_TESTED)
assert set(MENTAL + PHYSICAL) == set(LAB)
SUB, STEM = "conditions_assoc_jno", "conditions_assoc"                            # the downloads
SUB_ADJ, STEM_ADJ = "conditions_assoc_adjusted_jno", "conditions_assoc_adjusted"
esc = lambda s: str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def num(n):
    n = int(n)
    return f"{n:,}".replace(",", "\\,") if n >= 10000 else str(n)


def pval(p):
    if p < 0.001:
        return "$<$.001"
    return f"{p:.3f}".lstrip("0") if p < 0.01 else f"{p:.2f}".lstrip("0")


def size(v, lo, hi):
    return (v, lo, hi) if v >= 1 else (1 / v, 1 / hi, 1 / lo)


def ci(v, lo, hi):
    v, lo, hi = size(v, lo, hi)
    return f"{v:.2f} ({lo:.2f}-{hi:.2f})"


def cond_label(c):
    return esc(LAB[c]) + ("\\textsuperscript{a}" if c in TESTED else "")


def read(sub, stem, enc):
    return pd.read_csv(os.path.join(WB, sub, f"{stem}_{enc}.csv"))


def read_jno(sub, stem, enc, or_file=False):
    """The downloads: <stem>_jno_<enc>.csv and <stem>_or_jno_<enc>.csv."""
    return pd.read_csv(os.path.join(WB, sub, f"{stem}{'_or' if or_file else ''}_jno_{enc}.csv"))


ROUNDING = ("Significance was judged on the unrounded \\textit{P} value, so a rounded value close to the threshold can "
            "appear on either side of it.")


def enc_head(enc_label, k):
    name = esc(enc_label).replace(" (prespecified)", "\\\\(prespecified)")
    return f"\\shortstack{{{name}\\\\$k$ = {k}}}"


def fifteen():
    D = {e: read_jno(SUB, STEM, e).set_index("condition") for e, _ in ENCODERS}
    K = {e: len(read_jno(SUB, STEM, e, or_file=True).group.unique()) for e, _ in ENCODERS}
    for e in D:
        assert D[e].passes_joint.all() and D[e].converged.all(), e
        assert (D[e].groups_p < 0.001).all(), e
        for c in ("n_cases", "n_noncases"):
            assert (D[e][c] == D[ENCODERS[0][0]][c]).all(), f"{e}: populations differ"
    base = D[ENCODERS[0][0]]
    lines, rows = [], []
    for block, conds in (("Mental and behavioral", MENTAL), ("Physical", PHYSICAL)):
        lines.append(f"\\multicolumn{{{3 + len(ENCODERS)}}}{{@{{}}l}}{{\\textit{{{block}}}}} \\\\")
        for c in conds:
            cells = [f"{int(D[e].at[c, 'n_groups_together'])} / {int(D[e].at[c, 'n_groups_alone'])}" for e, _ in ENCODERS]
            lines.append(f"\\hspace*{{1em}}{cond_label(c)} & {num(base.at[c, 'n_cases'])} & {num(base.at[c, 'n_noncases'])} & "
                         + " & ".join(cells) + " \\\\")
            rows += [{"condition": c, "encoder": e, "n_with": int(base.at[c, "n_cases"]),
                      "n_without": int(base.at[c, "n_noncases"]), "k": K[e], "joint_p": D[e].at[c, "groups_p"],
                      "together": int(D[e].at[c, "n_groups_together"]), "alone": int(D[e].at[c, "n_groups_alone"])}
                     for e, _ in ENCODERS]
    heads = " & ".join(enc_head(l, K[e]) for e, l in ENCODERS)
    # title: "Significant", the main text's word.
    cap = ("\\caption{\\textbf{eTable \\eTabFifteen. Semantic Groups Significant Together and Alone for Each Recorded Condition, "
           "Under Each of the 6 Encoders.} Participants with at least 1 diagnosis code, breast cancer among women and "
           "prostate cancer among men (eMethods \\eMethConditions{}). With and without give "
           "the number of participants with and without the recorded condition, the same under every encoder. Each "
           "encoder's cell gives the number of groups significant at \\textit{P} $<$ .05/$k$, together, with all $k$ groups, "
           "age, and sex in the model, and alone, with age and sex only. The groups considered together were significant "
           "in the joint test, \\textit{P} $<$ .05, for every condition under every encoder (\\textit{P} $<$ .001 for each). "
           "The prespecified encoder's values are shown in Figure \\FigConditions{}, and the other encoders' in eFigure "
           "\\eFigHeatmaps{}. \\textsuperscript{a}Participants with and without the diagnosis were limited to those with "
           "the relevant test (eMethods \\eMethConditions{}).}\\label{etab:fifteen}")
    col = "@{}>{\\raggedright\\arraybackslash}p{5.2cm}rr*{%d}{c}@{}" % len(ENCODERS)
    head = (" & & & \\multicolumn{%d}{c}{Groups significant together / alone} \\\\ \\cmidrule(l){4-%d}" % (len(ENCODERS), 3 + len(ENCODERS))
            + "\nCondition & With & Without & " + heads + " \\\\")
    write("etable_assoc_jno", cap, col, head, lines, rows, size_cmd="\\footnotesize", sep="4pt")


def breast():
    lines, rows = [], []
    N = read_jno(SUB, STEM, ENCODERS[0][0]).set_index("condition")
    for e, lab in ENCODERS:
        O = read_jno(SUB, STEM, e, or_file=True).rename(columns={"or": "or_t"})
        lines.append(f"\\midrule\\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{{esc(lab)}}}}} \\\\")
        for c in ("breast", "prostate"):
            x = O[(O.condition == c) & (O.passes_together | O.passes_alone)].copy()
            x["f"] = np.exp(np.abs(np.log(x["or_t"])))
            x["rank"] = np.where(x.passes_together & x.passes_alone, 0, np.where(x.passes_together, 1, 2))
            x = x.sort_values(["rank", "f"], ascending=[True, False])
            lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{esc(LAB[c])}}}}} \\\\")
            if not len(x):
                lines.append("\\multicolumn{7}{@{}l}{\\hspace*{1em}No group was significant together or alone} \\\\")
            for r in x.itertuples():
                passed = ("Together and alone" if r.passes_together and r.passes_alone
                          else "Together only" if r.passes_together else "Alone only")
                cons = r.construct_profile.replace(":", " ").replace("; ", ", ")
                # the Wald P of each OR against 1, from the same rows as the ORs
                # (conditions_assoc_or_jno_<enc>.csv: p, p_alone); significant = P < .05/k.
                lines.append(f"\\hspace*{{1em}}G{int(r.group) + 1} & {esc(display_listed(cons))} & {ci(r.or_t, r.or_lo, r.or_hi)} & "
                             f"{pval(r.p)} & {ci(r.or_alone, r.or_alone_lo, r.or_alone_hi)} & {pval(r.p_alone)} & "
                             f"{passed} \\\\")
                rows.append({"encoder": e, "condition": c, "group": r.group, "constructs": cons,
                             "or_size": size(r.or_t, r.or_lo, r.or_hi)[0],
                             "or_alone_size": size(r.or_alone, r.or_alone_lo, r.or_alone_hi)[0],
                             "p_together": r.p, "p_alone": r.p_alone,
                             "passes_together": r.passes_together, "passes_alone": r.passes_alone})
    # title: "Significant", the main text's word.
    cap = ("\\caption{\\textbf{eTable \\eTabBreast. Semantic Groups Significant Together or Alone for Breast and Prostate "
           "Cancer, Under Each of the 6 Encoders.} Women (breast cancer) and men (prostate cancer) with at least 1 diagnosis "
           "code whose records showed a mammogram or a blood prostate-specific antigen test (eMethods \\eMethConditions{}), "
           f"{num(N.at['breast', 'n_cases'])} women with breast cancer and {num(N.at['breast', 'n_noncases'])} without, and "
           f"{num(N.at['prostate', 'n_cases'])} men with prostate cancer and {num(N.at['prostate', 'n_noncases'])} without. "
           "Every group significant at \\textit{P} $<$ .05/$k$ together or alone is listed. Together is the "
           "odds ratio per SD with all groups, age, and age squared in the model, and alone the odds ratio with age and "
           "age squared only, each with its 95\\% CI and the \\textit{P} value of the Wald test of the odds ratio against 1. "
           + ROUNDING + " A group significant only together is associated with the cancer net of "
           "the other groups. Odds ratios are sizes, the odds ratio or its inverse, whichever is 1 "
           "or more, because the sign of a group is arbitrary (eMethods \\eMethDistance{}). Groups are numbered and named as in eTable "
           "\\eTabGroups{}, and the first construct listed is the group's main construct, by which groups were matched "
           "across encoders.}\\label{etab:breast}")
    col = "@{}r>{\\raggedright\\arraybackslash}p{4.6cm}ccccl@{}"
    head = ("Group & Constructs of the core items & Together, OR (95\\% CI) & \\textit{P} & Alone, OR (95\\% CI) & "
            "\\textit{P} & Significant \\\\")
    write("etable_breast_jno", cap, col, head, lines, rows, size_cmd="\\scriptsize", sep="3pt")


def control():
    lines, rows = [], []
    J = {e: read_jno(SUB_ADJ, STEM_ADJ, e).set_index("condition") for e, _ in ENCODERS}
    O = {e: read_jno(SUB_ADJ, STEM_ADJ, e, or_file=True) for e, _ in ENCODERS}
    B = {e: read_jno(SUB, STEM, e).set_index("condition") for e, _ in ENCODERS}
    for e in J:
        assert J[e].passes_joint_after.all() and J[e].converged.all() and (J[e].groups_p_after < 0.001).all(), e
        assert (J[e].n_groups_together_before == B[e].n_groups_together.reindex(J[e].index)).all(), f"{e}: before differs"
    for block, conds in (("Mental and behavioral", MENTAL), ("Physical", PHYSICAL)):
        lines.append(f"\\multicolumn{{{1 + 2 * len(ENCODERS)}}}{{@{{}}l}}{{\\textit{{{block}}}}} \\\\")
        for c in conds:
            cells = []
            for e, _ in ENCODERS:
                j = J[e].loc[c]
                o = O[e][(O[e].condition == c) & O[e].passes_before]
                med = float(o.share_kept.median()) if len(o) else np.nan
                cells += [f"{int(j.n_groups_together_before)} / {int(j.n_groups_together_after)} / "
                          f"{int(j.n_groups_alone_after)}", "---" if np.isnan(med) else f"{100 * med:.0f}"]
                rows.append({"condition": c, "encoder": e, "together_before": int(j.n_groups_together_before),
                             "together_after": int(j.n_groups_together_after), "alone_after": int(j.n_groups_alone_after),
                             "median_share_kept": med, "n_groups_before": len(o), "joint_p_after": j.groups_p_after})
            lines.append(f"\\hspace*{{1em}}{cond_label(c)} & " + " & ".join(cells) + " \\\\")
    heads = " & ".join(f"\\multicolumn{{2}}{{c}}{{{esc(l.replace(' (prespecified)', '$^b$'))}}}" for _, l in ENCODERS)
    rules = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ENCODERS)))
    sub = " & ".join(["Groups", "Kept, \\%"] * len(ENCODERS))
    cap = ("\\caption{\\textbf{eTable \\eTabControl. Semantic Groups and Recorded Conditions After Adjustment for Social and "
           "Economic Position, Under Each of the 6 Encoders.} The models of eTable \\eTabFifteen{} with household income, "
           "education, health insurance, any barrier to care, and a cost barrier added (eMethods \\eMethConditions{}). Groups gives the number "
           "of groups significant at \\textit{P} $<$ .05/$k$ together before the adjustment, together after it, "
           "and alone after it, the alone models carrying the same adjustment. Kept is the median, over the groups "
           "significant together before the adjustment, of the share of each group's log odds ratio that remained after it "
           "(--- where no group was significant together before). The groups considered together were significant in the "
           "joint test, \\textit{P} $<$ .05, after the adjustment for every condition under every encoder (\\textit{P} $<$ .001 "
           "for each). \\textsuperscript{a}Participants "
           "with and without the diagnosis were limited to those with the relevant test (eMethods \\eMethConditions{}). "
           "\\textsuperscript{b}Prespecified encoder.}\\label{etab:control}")
    col = "@{}>{\\raggedright\\arraybackslash}p{3.4cm}*{%d}{cc}@{}" % len(ENCODERS)
    head = f"Condition & {heads} \\\\ {rules}\n & {sub} \\\\"
    write("etable_control_jno", cap, col, head, lines, rows, size_cmd="\\scriptsize", sep="2.5pt")


def bands():
    lines, rows = [], []
    A = {e: read("phecode_screened", "phecode_screened_assoc", e) for e, _ in ENCODERS}
    order = ["18-39", "40-49", "50-74", "75+"]
    labels = {"18-39": "Younger than 40", "40-49": "40-49", "50-74": "50-74", "75+": "75 or older"}
    for e in A:
        A[e] = A[e][A[e].target == "breast"].set_index("band").loc[order]
        assert A[e].converged.all(), e
        assert (A[e].n_cases_checked == A[ENCODERS[0][0]].n_cases_checked).all(), e
    base = A[ENCODERS[0][0]]
    for b in order:
        cells = []
        for e, _ in ENCODERS:
            r = A[e].loc[b]
            cells += [pval(r.groups_p), f"{int(r.n_groups_together)} / {int(r.n_groups_alone)}"]
            rows.append({"band": b, "encoder": e, "n_with": int(r.n_cases_checked), "n_without": int(r.n_noncases_checked),
                         "joint_p": r.groups_p, "together": int(r.n_groups_together), "alone": int(r.n_groups_alone)})
        lines.append(f"{labels[b]} & {num(base.at[b, 'n_cases_checked'])} & {num(base.at[b, 'n_noncases_checked'])} & "
                     + " & ".join(cells) + " \\\\")
    heads = " & ".join(f"\\multicolumn{{2}}{{c}}{{{esc(l.replace(' (prespecified)', '$^a$'))}}}" for _, l in ENCODERS)
    rules = "".join(f"\\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(ENCODERS)))
    sub = " & ".join(["\\textit{P}", "Groups"] * len(ENCODERS))
    cap = ("\\caption{\\textbf{eTable \\eTabBands. Breast Cancer Within Screening Age Bands, Under Each of the 6 Encoders.} "
           "Women with at least 1 diagnosis code whose records showed a mammogram (eMethods \\eMethConditions{}), in 4 age bands. With and "
           "without give the number of women with and without recorded breast cancer. In each band, 1 logistic regression "
           "included age in years and all groups. \\textit{P} is the joint test of the groups considered together. Groups "
           "gives the number passing \\textit{P} $<$ .05/$k$ within the band together, with all groups and age in the "
           "model, and alone, with age only. \\textsuperscript{a}Prespecified encoder.}\\label{etab:bands}")
    col = "@{}lrr*{%d}{cc}@{}" % len(ENCODERS)
    head = f"Age, y & With & Without & {heads} \\\\ {rules}\n & & & {sub} \\\\"
    write("etable_bands_jno", cap, col, head, lines, rows, size_cmd="\\scriptsize", sep="3pt")


def write(stem, cap, col, head, lines, rows, size_cmd, sep):
    tex = [f"% generated by scripts/etable_conditions_jno.py -- do not edit by hand",
           f"\\begingroup{size_cmd}\\setlength{{\\tabcolsep}}{{{sep}}}\\renewcommand{{\\arraystretch}}{{1.25}}",
           f"\\begin{{longtable}}{{{col}}}", cap + " \\\\", "\\toprule", head, "\\midrule", "\\endfirsthead", "\\toprule",
           head, "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"] + lines + ["\\end{longtable}", "\\endgroup"]
    os.makedirs(OUT, exist_ok=True)   # a fresh copy has no outputs/tables yet
    open(os.path.join(OUT, f"{stem}.tex"), "w").write("\n".join(tex) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{stem}.csv"), index=False)
    print("wrote", os.path.join(OUT, f"{stem}.tex"), f"({len(rows)} rows)")


if __name__ == "__main__":
    fifteen()
    breast()
    control()
    # bands() is not called: it is not in the paper
