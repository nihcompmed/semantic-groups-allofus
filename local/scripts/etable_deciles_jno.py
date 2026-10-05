#!/usr/bin/env python
"""etable_deciles_jno.py -- eTable 7: every outcome of Figure 2 by decile of the distance, adjusted for age and sex,
for the prespecified encoder with 95% CIs and for the other 5 encoders as values. The Results cite it for the
agreement across encoders.

LOCAL. Reads downloaded aggregates only:
  outcome_deciles/outcome_deciles.csv                 thresholds met, barriers, sleep
  ed_check/ed_check_deciles.csv                       ED share on coded care
  conditions_deciles_jno/conditions_deciles_jno.csv   the 8 recorded conditions of record
                                                      (workbench/conditions_deciles_jno.py)
OUTPUTS  ../outputs/tables/etable_deciles_jno.{tex,csv}

Run:  python etable_deciles_jno.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WB = os.path.join(HERE, "..", "outputs", "workbench", "R2025Q4R6")
OUT = os.path.join(HERE, "..", "outputs", "tables")
ENCODERS = [("gte-large-en-v1.5", "gte-large-en-v1.5 (prespecified)"), ("bge-m3", "bge-m3"),
            ("bge-large-en-v1.5", "bge-large-en-v1.5"), ("all-mpnet-base-v2", "all-mpnet-base-v2"),
            ("e5-large-v2", "e5-large-v2"), ("qwen3-embedding-0.6b", "Qwen3-Embedding-0.6B")]
D = list(range(1, 11))
# (source, key, label, scale, decimals, cohort column)
MEASURES = [("od", "mean_breadth", "Screening thresholds met, mean No.", 1, 2, "mean_breadth"),
            ("od", "access", "Any barrier to care, \\%", 100, 1, "access_rate"),
            ("od", "cost", "Cost barrier, \\%", 100, 1, "cost_rate"),
            ("od", "avoid", "Delayed care, clinician differs, \\%", 100, 1, "avoid_rate"),
            ("ed", "ed", "ED visits, \\% of visits", 100, 2, "ed_adj"),
            ("od", "sleep_mean", "Mean sleep per night, min", 1, 1, "sleep_mean"),
            ("od", "sleep_var_median", "Night-to-night variation, min", 1, 1, "sleep_var_median")]
# the 8 conditions of record, their order and labels (conditions_local.py)
from conditions_local import CONDITIONS as _C   # noqa: E402
CONDITIONS = [(k, lab) for k, lab, _, _ in _C]


def fmt(v, dec):
    return "" if pd.isna(v) else f"{v:.{dec}f}"


def cell(v, lo, hi, dec, ci):
    if pd.isna(v):
        return ""
    s = fmt(v, dec)
    return f"\\shortstack{{{s}\\\\{{\\tiny ({fmt(lo, dec)}-{fmt(hi, dec)})}}}}" if ci else s


def main():
    od = pd.read_csv(os.path.join(WB, "outcome_deciles", "outcome_deciles.csv"))
    ed = pd.read_csv(os.path.join(WB, "ed_check", "ed_check_deciles.csv"))
    ph = pd.read_csv(os.path.join(WB, "conditions_deciles_jno", "conditions_deciles_jno.csv"))
    pop = ph[ph.grouping == "population"].set_index("condition")
    thin = lambda n: f"{int(n):,}".replace(",", "\\,") if int(n) >= 10000 else str(int(n))
    oc, ec = od[od.grouping == "cohort"].iloc[0], ed[ed.grouping == "cohort"].iloc[0]
    lines, rows = [], []
    for enc, enc_label in ENCODERS:
        ci = enc == ENCODERS[0][0]
        lines.append(f"\\midrule\\multicolumn{{12}}{{@{{}}l}}{{\\textbf{{{enc_label}}}}} \\\\")
        g = od[od.grouping == enc].set_index("decile")
        e = ed[ed.grouping == enc].set_index("decile")
        for src, key, label, sc, dec, ccol in MEASURES:
            T, C = (g, oc) if src == "od" else (e, ec)
            col = f"{key}_adj"
            vals = [cell(sc * T.at[j, col], sc * T.at[j, col + "_lo"], sc * T.at[j, col + "_hi"], dec, ci) for j in D]
            coh = fmt(sc * C[ccol], dec)
            lines.append(f"{label} & {coh} & " + " & ".join(vals) + " \\\\")
            rows += [{"encoder": enc, "measure": key, "decile": j, "value": sc * T.at[j, col],
                      "lo": sc * T.at[j, col + "_lo"], "hi": sc * T.at[j, col + "_hi"], "cohort": sc * C[ccol]} for j in D]
        lines.append("\\multicolumn{12}{@{}l}{\\textit{Recorded diagnosis, \\%}} \\\\")
        for key, label in CONDITIONS:
            x = ph[(ph.grouping == enc) & (ph.condition == key)].sort_values("decile_from")
            c0 = pop.loc[key]
            cells = []
            for r in x.itertuples():
                span = r.decile_to - r.decile_from + 1
                c = cell(100 * r.adj, 100 * r.adj_lo, 100 * r.adj_hi, 1, ci)
                cells.append(c if span == 1 else f"\\multicolumn{{{span}}}{{c}}{{{c}}}")
                rows.append({"encoder": enc, "measure": key, "decile": f"{r.decile_from}-{r.decile_to}" if span > 1 else r.decile_from,
                             "value": 100 * r.adj, "lo": 100 * r.adj_lo, "hi": 100 * r.adj_hi, "cohort": 100 * c0.rate})
            lines.append(f"\\hspace*{{1em}}{label} & {fmt(100 * c0.rate, 1)} & " + " & ".join(cells) + " \\\\")
    head = ("Outcome & Cohort & " + " & ".join(f"{j}" for j in D) + " \\\\")
    cap = ("\\caption{\\textbf{eTable \\eTabDeciles. Outcomes by Decile of the Distance From the Cohort's Typical "
           "Scores, Under Each Encoder.} Values are adjusted for age and sex (Methods and eMethods \\eMethMeasures{}). Decile 1 is the "
           "most typical tenth of the cohort and decile 10 the most atypical, each encoder with its own "
           "deciles. For the prespecified encoder, each value is followed by its 95\\% CI. The cohort column is the "
           "observed value, which the adjustment leaves unchanged. Each outcome is computed on the participants with the "
           "data it needs (eTable \\eTabSubsets). Screening thresholds, participants with all 6 thresholds evaluable "
           "(58\\,682). Barriers, participants who answered the Health Care Access and Utilization survey (87\\,257). "
           "Delayed care, clinician differs, delayed or did not see doctors or other clinicians because they differed "
           "from the participant in race, religion, or native language, some of the time or more. ED indicates emergency "
           "department, whose share of visits is computed on participants with at least 1 visit and 1 diagnosis code "
           "(61\\,558). Sleep, participants with at least 14 nights of Fitbit data (16\\,732). Recorded diagnoses, "
           f"participants with at least 1 diagnosis code ({thin(pop.at['gad', 'n'])}); for hypercholesterolemia, those "
           f"with a cholesterol measurement or lipid panel ({thin(pop.at['hyperchol', 'n'])}); for breast cancer, women "
           # 2026-10-05 (process pass, author: "yes, apply all 13"). Previous, VERBATIM:
           # f"with a mammogram ({thin(pop.at['breast', 'n'])}); for prostate cancer, men with a blood prostate-specific "
           f"with a mammogram or screening visit ({thin(pop.at['breast', 'n'])}); for prostate cancer, men with a blood prostate-specific "
           # 2026-10-05 (process pass, author: "yes, apply all 13"). Previous, VERBATIM:
           # f"antigen test ({thin(pop.at['prostate', 'n'])}). For a recorded diagnosis, the cohort column is the share in "
           f"antigen test or screening visit ({thin(pop.at['prostate', 'n'])}). For a recorded diagnosis, the cohort column is the share in "
           "that population.}\\label{etab:deciles}")
    tex = ["% generated by scripts/etable_deciles_jno.py -- do not edit by hand",
           "\\begingroup\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\\renewcommand{\\arraystretch}{1.9}",
           "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{5.0cm}r*{10}{c}@{}}",
           cap + " \\\\",
           "\\toprule", " & & \\multicolumn{10}{c}{Decile of the distance} \\\\", "\\cmidrule(l){3-12}",
           head, "\\endfirsthead",
           "\\toprule", head, "\\endhead", "\\bottomrule", "\\endfoot"] + lines + ["\\end{longtable}", "\\endgroup"]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "etable_deciles_jno.tex"), "w").write("\n".join(tex) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "etable_deciles_jno.csv"), index=False)
    print("wrote", os.path.join(OUT, "etable_deciles_jno.tex"), len(rows), "values")


if __name__ == "__main__":
    main()
