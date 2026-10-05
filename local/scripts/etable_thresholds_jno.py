#!/usr/bin/env python
"""etable_thresholds_jno.py -- eTable 5: the 6 screening thresholds, with the
number evaluable and met in the cohort and among the participants above the 95th percentile under the prespecified
encoder, and the number of thresholds met.

LOCAL. Reads the downloaded cutoffs/{cutoff_summary.csv, cutoff_summary.json, breadth_table.csv} (workbench/cutoffs.py,
packaged by workbench/zip_cutoffs.py).
OUTPUTS  ../outputs/tables/etable_thresholds_jno.{tex,csv}

Run:  python etable_thresholds_jno.py
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "outputs", "workbench", "R2025Q4R6", "cutoffs")
OUT = os.path.join(HERE, "..", "outputs", "tables")
# criterion (as cutoffs.py names it) -> (label, instrument with citation key, met when)
ROWS = [("PHQ-9", "Depression", "Patient Health Questionnaire-9\\cite{phq9_kroenke2001}", "Total score of 10 or more"),
        ("GAD-7", "Anxiety", "Generalized Anxiety Disorder-7\\cite{gad7_spitzer2006}", "Total score of 10 or more"),
        ("ASRS", "Attention-deficit/hyperactivity disorder", "Adult ADHD Self-Report Scale, Part A\\cite{asrs_kessler2005}",
         "4 or more of 6 items at the marked frequency (sometimes or more often for items 1-3, often or more often "
         "for items 4-6)"),
        ("ACE", "Adverse childhood experiences", "Adverse childhood experiences module\\cite{ace_felitti1998}",
         "4 or more of 7 categories\\textsuperscript{a}"),
        ("HVS", "Food insecurity", "Hunger Vital Sign\\cite{hvs_hager2010}",
         "Either item sometimes true or often true"),
        ("Suicidal ideation or attempt", "Suicidal ideation or attempt", "2 single items",
         "Yes to lifetime thoughts of killing oneself or to a lifetime suicide attempt")]


def num(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def big(n):
    return f"{n:,}".replace(",", "\\,") if n is not None and n >= 10000 else (str(n) if n is not None else "")


def npct(k, n):
    k, n = num(k), num(n)
    if k is None or n is None:
        return "\\textsuperscript{b}"
    return f"{big(k)} ({100 * k / n:.1f})"


def main():
    S = pd.read_csv(os.path.join(SRC, "cutoff_summary.csv")).set_index("criterion")
    J = json.load(open(os.path.join(SRC, "cutoff_summary.json")))
    B = pd.read_csv(os.path.join(SRC, "breadth_table.csv"))
    assert int(J["n_criteria"]) == 6 and "Suicidal ideation or attempt" in S.index, "not the six-threshold summary"
    N, NF = int(J["n_cohort"]), int(J["n_flagged"])
    lines, csv = [], []
    lines.append("\\multicolumn{7}{@{}l}{\\textbf{Each threshold}} \\\\")
    for key, label, inst, rule in ROWS:
        r = S.loc[key]
        ev_c, ev_f = num(r.n_cohort_evaluable), num(r.n_flagged_evaluable)
        lines.append(f"{label} & {inst} & {int(r.n_items)} & {rule} & {big(ev_c)} & {npct(r.n_cohort, r.n_cohort_evaluable)} & "
                     f"{npct(r.n_flagged, r.n_flagged_evaluable) if ev_f is not None else chr(92) + 'textsuperscript{b}'} \\\\")
        csv.append({"threshold": label, "items": int(r.n_items), "cohort_evaluable": ev_c, "cohort_met": num(r.n_cohort),
                    "flagged_evaluable": ev_f, "flagged_met": num(r.n_flagged)})
    ra = S.loc["any_met"]
    lines.append(f"Any of the 6 & & {int(ra.n_items)} & & {big(num(ra.n_cohort_evaluable))} & "
                 f"{npct(ra.n_cohort, ra.n_cohort_evaluable)} & {npct(ra.n_flagged, ra.n_flagged_evaluable)} \\\\")
    lines.append("\\midrule")
    nae, nfe = int(J["n_cohort_all_evaluable"]), int(J["n_flagged_all_evaluable"])
    lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textbf{{Number of thresholds met}}, participants with all 6 evaluable}} & "
                 f"{big(nae)} & & {big(nfe)} \\\\")
    for r in B.itertuples():
        lines.append(f"\\hspace*{{1em}}{int(r.breadth)} & & & & & {big(num(r.n_cohort))} ({r.pct_cohort:.1f}) & "
                     f"{big(num(r.n_flagged))} ({r.pct_flagged:.1f}) \\\\")
        csv.append({"threshold": f"met {int(r.breadth)}", "cohort_met": num(r.n_cohort), "flagged_met": num(r.n_flagged)})
    lines.append(f"\\hspace*{{1em}}Mean & & & & & {J['mean_breadth_cohort_all_evaluable']:.2f} & "
                 f"{J['mean_breadth_flagged_all_evaluable']:.2f} \\\\")
    cap = ("\\caption{\\textbf{eTable \\eTabThresholds. The 6 Screening Thresholds, and the Number Evaluable and Met.} Each threshold is "
           "a determination its instrument publishes, applied on the codebook coding. A threshold was evaluable for a "
           "participant who answered every item it uses, and imputed answers were not used (eMethods \\eMethMeasures{}). The 6 "
           f"thresholds use 36 of the 151 items. The cohort is {big(N)} participants. The 95th percentile set is the "
           f"{NF} participants above the 95th percentile of the distance under the prespecified encoder "
           "(gte-large-en-v1.5). Values are No. (\\%), with percentages of the participants evaluable.}\\label{etab:thresholds}")
    tex = ["% generated by scripts/etable_thresholds_jno.py -- do not edit by hand",
           "\\begin{table}[htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{3.5pt}", cap, "\\vspace{4pt}",
           "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{3.0cm}c"
           ">{\\raggedright\\arraybackslash}p{3.6cm}rrr@{}}",
           "\\toprule",
           " & & & & \\multicolumn{2}{c}{Cohort} & 95th percentile set \\\\",
           "\\cmidrule(lr){5-6}\\cmidrule(l){7-7}",
           "Threshold & Instrument & Items, No. & Met when & Evaluable, No. & Met, No. (\\%) & Met, No. (\\%) \\\\",
           "\\midrule"] + lines + [
           "\\bottomrule", "\\end{tabular}", "\\par\\vspace{3pt}\\raggedright",
           "\\textsuperscript{a} Household mental illness, household substance abuse, parental separation or divorce, "
           "witnessed domestic violence, physical abuse, emotional abuse, and sexual abuse. The module's eighth category, "
           "household incarceration, is withheld in the Registered Tier.\\par",
           "\\textsuperscript{b} Not shown, because the number of participants in the set who could not be evaluated is "
           "20 or fewer (All of Us Data and Statistics Dissemination Policy).\\par",
           "\\end{table}"]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "etable_thresholds_jno.tex"), "w").write("\n".join(tex) + "\n")
    pd.DataFrame(csv).to_csv(os.path.join(OUT, "etable_thresholds_jno.csv"), index=False)
    print("wrote", os.path.join(OUT, "etable_thresholds_jno.tex"))


if __name__ == "__main__":
    main()
