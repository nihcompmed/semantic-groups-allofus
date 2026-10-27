#!/usr/bin/env python
"""appendix3_table1.py -- Table S3, the 6 screening thresholds: the number of participants evaluable and the
number who met each, in the cohort and among the outliers under the prespecified encoder, and the number of thresholds
met by the participants evaluable on all 6.

A threshold was evaluable for a participant who answered every item it uses (imputed answers not used). Values are
No. (%), with percentages of the participants evaluable; "Any of the 6" counts the participants who met at least one of
the thresholds they could be evaluated on, with the whole cohort as the denominator. A cell the download withheld under
the dissemination policy is marked b.

Reads   the download thresholds (thresholds.csv, thresholds.json, thresholds_met.csv)
Writes  output/tables/app_thresholds.tex; outputs/tables/thresholds.csv (the counts)
Run     python appendix3_table1.py
"""
import json
import os

import pandas as pd

from common import TABLE_DATA, WB, write_table

SRC = os.path.join(WB, "thresholds")
# threshold (as the download names it) -> (label, instrument with citation key, met when)
ROWS = [("PHQ-9", "Depression", "Patient Health Questionnaire-9 \\citep{phq9_kroenke2001}", "Total score of 10 or more"),
        ("GAD-7", "Anxiety", "Generalized Anxiety Disorder-7 \\citep{gad7_spitzer2006}", "Total score of 10 or more"),
        ("ASRS", "Attention-deficit/hyperactivity disorder", "Adult ADHD Self-Report Scale, Part A \\citep{asrs_kessler2005}",
         "4 or more of 6 items at the marked frequency (sometimes or more often for items 1-3, often or more often "
         "for items 4-6)"),
        ("ACE", "Adverse childhood experiences", "Adverse childhood experiences module \\citep{ace_felitti1998}",
         "4 or more of 7 categories\\textsuperscript{a}"),
        ("HVS", "Food insecurity", "Hunger Vital Sign \\citep{hvs_hager2010}",
         "Either item sometimes true or often true"),
        ("Suicidal ideation or attempt", "Suicidal ideation or attempt", "2 single items",
         "Yes to lifetime thoughts of killing oneself or to a lifetime suicide attempt")]
WITHHELD = "\\textsuperscript{b}"


def num(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def big(n):
    """A count, with a thousands comma from 10,000."""
    return "" if n is None else (f"{n:,}" if n >= 10000 else str(n))


def npct(k, n):
    k, n = num(k), num(n)
    if k is None or n is None:
        return WITHHELD
    return f"{big(k)} ({100 * k / n:.1f})"


def caption(N, NF):
    return ("\\caption{\\textbf{\\AppTabThresholds{}. The six screening thresholds, and the number evaluable and met.} "
            "Each threshold is the published cutoff of its questionnaire, applied on the codebook coding. A threshold "
            "was evaluable for a participant who answered every item it uses, and imputed answers were not used "
            f"(Appendix~\\AppOutcomes{{}}). The six thresholds use 36 of the 151 items. The cohort is {big(N)} "
            f"participants, and the outliers are the {big(NF)} participants above the 95th percentile of the distance "
            "under the prespecified encoder (gte-large-en-v1.5). Values are No. (\\%), with percentages of the "
            "participants evaluable. Any of the 6 counts the participants who met at least one of the thresholds "
            f"they could be evaluated on, with all {big(N)} as the denominator.}}\\label{{apptab:thresholds}}")


def main():
    S = pd.read_csv(os.path.join(SRC, "thresholds.csv")).set_index("criterion")
    J = json.load(open(os.path.join(SRC, "thresholds.json")))
    B = pd.read_csv(os.path.join(SRC, "thresholds_met.csv"))
    assert int(J["n_criteria"]) == 6 and "Suicidal ideation or attempt" in S.index, "not the six-threshold summary"
    N, NF = int(J["n_cohort"]), int(J["n_flagged"])
    lines, csv = ["\\multicolumn{7}{@{}l}{\\textbf{Each threshold}} \\\\"], []
    for key, label, inst, rule in ROWS:
        r = S.loc[key]
        ev_c, ev_f = num(r.n_cohort_evaluable), num(r.n_flagged_evaluable)
        lines.append(f"{label} & {inst} & {int(r.n_items)} & {rule} & {big(ev_c)} & "
                     f"{npct(r.n_cohort, r.n_cohort_evaluable)} & "
                     f"{npct(r.n_flagged, r.n_flagged_evaluable) if ev_f is not None else WITHHELD} \\\\")
        csv.append({"threshold": label, "items": int(r.n_items), "cohort_evaluable": ev_c, "cohort_met": num(r.n_cohort),
                    "flagged_evaluable": ev_f, "flagged_met": num(r.n_flagged)})
    ra = S.loc["any_met"]
    lines.append(f"Any of the 6 & & {int(ra.n_items)} & & {big(num(ra.n_cohort_evaluable))} & "
                 f"{npct(ra.n_cohort, ra.n_cohort_evaluable)} & {npct(ra.n_flagged, ra.n_flagged_evaluable)} \\\\")
    lines.append("\\midrule")
    nae, nfe = int(J["n_cohort_all_evaluable"]), int(J["n_flagged_all_evaluable"])
    lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textbf{{Number of thresholds met}}, participants with all 6 "
                 f"evaluable}} & {big(nae)} & & {big(nfe)} \\\\")
    for r in B.itertuples():
        lines.append(f"\\hspace*{{1em}}{int(r.breadth)} & & & & & {big(num(r.n_cohort))} ({r.pct_cohort:.1f}) & "
                     f"{big(num(r.n_flagged))} ({r.pct_flagged:.1f}) \\\\")
        csv.append({"threshold": f"met {int(r.breadth)}", "cohort_met": num(r.n_cohort), "flagged_met": num(r.n_flagged)})
    lines.append(f"\\hspace*{{1em}}Mean & & & & & {J['mean_breadth_cohort_all_evaluable']:.2f} & "
                 f"{J['mean_breadth_flagged_all_evaluable']:.2f} \\\\")
    L = ["\\begin{table}[htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{3.5pt}", caption(N, NF), "\\vspace{4pt}",
         "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{3.0cm}c"
         ">{\\raggedright\\arraybackslash}p{3.6cm}rrr@{}}",
         "\\toprule",
         " & & & & \\multicolumn{2}{c}{Cohort} & Outliers \\\\",
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
    write_table("app_thresholds", L, "appendix3_table1.py")
    os.makedirs(TABLE_DATA, exist_ok=True)
    pd.DataFrame(csv).to_csv(os.path.join(TABLE_DATA, "thresholds.csv"), index=False)


if __name__ == "__main__":
    main()
