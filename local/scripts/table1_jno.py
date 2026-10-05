#!/usr/bin/env python
"""table1_jno.py -- eTables 2 and 3 of the JAMA Network Open supplement, and a 2-column Table 1.

  Table 1 (NOT IN THE PAPER: eTable 2 holds both of its columns; still written)
                        the cohort and the participants above the 95th percentile under the prespecified encoder
                        (gte-large-en-v1.5), No. (%)
  eTable 2, encoders    the cohort and each of the 6 encoders' own 95th-percentile sets, side by side, so the rows can
                        be read across to see whether the numbers are similar (no test, no overlap statistic)
  eTable 3, subsets     the cohort and the subset each outcome is measured on: the access survey (barriers), at least
                        1 visit (the ED share), at least 1 ICD event (the 8 recorded conditions), at least 14 nights
                        of sleep

Decisions: the screening-threshold rows are OUTCOMES and are left out of all 3 tables; age is
mean (SD) only; female and male are both shown (male = the column's n minus its female count; sex at birth is
answered and codable for everyone in the cohort); 2 eTables, not one of 11 columns.

Style (JAMA Network Open instructions for authors): frequency data as No. (%); footnotes lettered in order of
appearance; abbreviations in the footnotes; four-digit numbers without a separator, five or more digits with a thin
space.

RACE AND ETHNICITY are not shown in any of the 3 tables. This study did not use race or ethnicity in any analysis; the
Methods state that they are recorded in All of Us but were not used and are not reported. Any race_ethnicity rows in
the input file are simply not shown.

Reads   outputs/workbench/R2025Q4R6/table1/table1_covariates.csv   (workbench/covariates.py)
Writes  outputs/tables/table1_jno.tex, etable_encoders_jno.tex, etable_subsets_jno.tex (+ a .csv of each, for the
        Word conversion)
Run     python scripts/table1_jno.py

The eTable numbers come from LaTeX macros (\\eTabEncoders, \\eTabSubsets) defined in the manuscript source,
which the main text and the supplement both read, so renumbering the supplement is one edit and a main-text callout
cannot silently point at the wrong table.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "outputs", "workbench", "R2025Q4R6", "table1", "table1_covariates.csv")
OUT = os.path.join(HERE, "outputs", "tables")

ENCODERS = [("flagged", "gte-large-en-v1.5"), ("flagged_bge-m3", "bge-m3"),
            ("flagged_bge-large-en-v1.5", "bge-large-en-v1.5"), ("flagged_all-mpnet-base-v2", "all-mpnet-base-v2"),
            ("flagged_e5-large-v2", "e5-large-v2"), ("flagged_qwen3-embedding-0.6b", "qwen3-embedding-0.6b")]
SUBSETS = [("survey_linked", "Access survey"), ("records_linked", "At least 1 visit"),
           ("icd_linked", "At least 1 diagnosis code"), ("sleep", "At least 14 nights of sleep")]

# (variable, level in the file, label); None as the level = a heading row. Race and ethnicity are not shown
# (see the docstring).
ROWS = [("sex", None, "Sex at birth"),
        ("sex", "female (sex at birth)", "Female"),
        ("sex", "__male__", "Male"),
        ("income", None, "Annual household income, \\$"),
        ("income", "<25k", "<25\\,000"),
        ("income", "25k to <50k", "25\\,000-49\\,999"),
        ("income", "50k to <100k", "50\\,000-99\\,999"),
        ("income", "100k to <150k", "100\\,000-149\\,999"),
        ("income", "150k or more", "$\\geq$150\\,000"),
        ("income", "Prefer not to answer", "Prefer not to answer or skipped"),
        ("education", None, "Education"),
        ("education", "Less than high school", "Less than high school"),
        ("education", "High school or GED", "High school or GED"),
        ("education", "Some college", "Some college"),
        ("education", "College graduate or more", "College graduate or more"),
        ("education", "Prefer not to answer", "Prefer not to answer or skipped"),
        ("employment", None, "Employment"),
        ("employment", "Employed", "Employed"),
        ("employment", "Not employed", "Not employed"),
        ("employment", "Prefer not to answer", "Prefer not to answer or skipped"),
        ("insurance", None, "Health insurance"),
        ("insurance", "Yes", "Yes"),
        ("insurance", "No", "No"),
        ("insurance", "Prefer not to answer", "Prefer not to answer, don't know, or skipped")]

# footnote marks, lettered in order of appearance in each table
FN_SET = ("Participants whose distance from the cohort's typical scores, computed on the groups formed from the "
          "item text by {enc}, exceeded the cohort's 95th percentile.")
FN_SEX = "Self-reported in The Basics survey."
ABBR_GED = "Abbreviation: GED, General Educational Development."


HEAD2 = {e: "\\shortstack[r]{" + t + "}" for e, t in [      # encoder names on 2 lines, so 7 columns fit the page
    ("gte-large-en-v1.5", "gte-large-\\\\en-v1.5"), ("bge-large-en-v1.5", "bge-large-\\\\en-v1.5"),
    ("all-mpnet-base-v2", "all-mpnet-\\\\base-v2"), ("e5-large-v2", "e5-large-\\\\v2"),
    ("qwen3-embedding-0.6b", "qwen3-embed-\\\\ding-0.6b")]}


SUB2 = {"Access survey": "Access\\\\survey", "At least 1 visit": "At least\\\\1 visit",   # subset heads, 2 lines
        "At least 1 diagnosis code": "At least 1\\\\diagnosis code",
        "At least 14 nights of sleep": "At least 14\\\\nights of sleep"}


def thin(n):
    """JAMA style: four-digit numbers without a separator, five or more digits with a (thin) space."""
    n = int(n)
    return f"{n:,}".replace(",", "\\,") if n >= 10000 else str(n)


def load():
    T = pd.read_csv(SRC)
    assert "female (sex at birth)" in set(T.level.dropna()), \
        "not the covariates.py output this script expects (no 'female (sex at birth)' level)"
    return T


def column(T, pop):
    """{(variable, level): 'No. (%)' or 'mean (SD)'} for one population, with the male row derived."""
    S = T[T.population == pop]
    n = int(S[S.variable == "n"].n.iloc[0])
    out = {("n", ""): n}
    for r in S.itertuples():
        if r.variable == "age" and r.level == "mean (SD)":
            out[("age", "mean (SD)")] = str(r.value)
        elif r.variable not in ("n", "age", "criteria"):
            k = int(r.n)
            out[(r.variable, r.level)] = f"{thin(k)} ({100 * k / n:.1f})"
            out[(r.variable, r.level, "k")] = k
    nf = out[("sex", "female (sex at birth)", "k")]
    out[("sex", "__male__")] = f"{thin(n - nf)} ({100 * (n - nf) / n:.1f})"
    for v in ("income", "education", "employment", "insurance"):   # every level shown sums to n
        tot = sum(val for key, val in out.items() if len(key) == 3 and key[0] == v)
        assert tot == n, f"{pop}: {v} levels sum to {tot}, not {n} (a pooled level would need its own row)"
    return out


def body(cols, marks):
    """The LaTeX rows: age, then each heading and its levels. `marks` = {row label: footnote letter}."""
    L = ["Age, mean (SD), y & " + " & ".join(c[("age", "mean (SD)")] for c in cols) + " \\\\"]
    csv = [["Age, mean (SD), y"] + [c[("age", "mean (SD)")] for c in cols]]
    for var, level, label in ROWS:
        lab = label + (f"\\textsuperscript{{{marks[label]}}}" if label in marks else "")
        if level is None:
            L.append(f"{lab} & " + " & ".join("" for _ in cols) + " \\\\")
            csv.append([label.replace("\\$", "$")] + ["" for _ in cols])
        else:
            # a wrapped level keeps its indent: first line 1em, continuation lines 2em
            L.append(f"\\hangindent=2em\\hangafter=1\\hspace*{{1em}}{lab} & " + " & ".join(c[(var, level)] for c in cols) + " \\\\")
            csv.append(["  " + label.replace("\\,", " ").replace("$\\geq$", ">=")] +
                       [c[(var, level)].replace("\\,", " ") for c in cols])
    return L, csv


def tabular(cols, heads, marks, colspec):
    L = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
         f" & \\multicolumn{{{len(cols)}}}{{c}}{{No. (\\%)}} \\\\", f"\\cmidrule(l){{2-{len(cols) + 1}}}",
         " & " + " & ".join(heads) + " \\\\",
         "Characteristic & " + " & ".join(f"(n = {thin(c[('n', '')])})" for c in cols) + " \\\\",
         "\\midrule"]
    rows, csv = body(cols, marks)
    L += rows + ["\\bottomrule", "\\end{tabular}"]
    return L, csv


def notes(items):
    return ["\\par\\vspace{3pt}\\begin{minipage}{\\linewidth}\\raggedright\\scriptsize"] + \
           [f"\\textsuperscript{{{m}}} {t}\\par" for m, t in items if m] + \
           [t + "\\par" for m, t in items if not m] + ["\\end{minipage}"]


def write(name, lines, csv_rows, csv_head):
    os.makedirs(OUT, exist_ok=True)              # a fresh copy has no outputs/tables/ yet
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write(f"% generated by scripts/table1_jno.py -- do not edit by hand\n" + "\n".join(lines) + "\n")
    pd.DataFrame(csv_rows, columns=csv_head).to_csv(os.path.join(OUT, name + ".csv"), index=False)


def main():
    T = load()
    coh = column(T, "cohort")

    # ---- Table 1 (not in the paper): cohort and gte's set, without a float or caption
    gte = column(T, "flagged")
    marks = {"Sex at birth": "b"}
    tab, csv = tabular([coh, gte], ["Cohort", "Above 95th percentile\\textsuperscript{a}"], marks,
                       "@{}>{\\raggedright\\arraybackslash}p{6.2cm}rr@{}")
    lines = ["\\begingroup\\footnotesize\\setlength{\\tabcolsep}{6pt}"] + tab + notes(
        [("a", FN_SET.format(enc="the prespecified sentence encoder (gte-large-en-v1.5)")), ("b", FN_SEX),
         ("", ABBR_GED)]) + ["\\endgroup"]
    write("table1_jno", lines, csv, ["Characteristic", "Cohort", "Above 95th percentile"])

    # ---- eTable: the 6 encoders' sets
    cols = [coh] + [column(T, p) for p, _ in ENCODERS]
    heads = ["Cohort"] + [HEAD2.get(e, e) for _, e in ENCODERS]
    marks = {"Sex at birth": "b"}
    tab, csv = tabular(cols, heads, marks, "@{}>{\\raggedright\\arraybackslash}p{3.7cm}" + "r" * len(cols) + "@{}")
    cap = ("\\caption{\\textbf{eTable \\eTabEncoders. Characteristics of the Participants Above the 95th Percentile "
           "Under Each of the 6 Encoders.} Each encoder's column holds its own participants above the 95th percentile"
           "\\textsuperscript{a}, 4518 per encoder by construction. The sets overlap but are not identical. Values are "
           "No. (\\%) unless stated otherwise. gte-large-en-v1.5 is the prespecified encoder, and its set is the one the "
           "main text describes.}\\label{etab:encoders}")
    # the size goes INSIDE the float: a float resets the font to normal size
    lines = ["\\begin{table}[htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{2.5pt}", cap,
             "\\vspace{4pt}"] + tab + notes(
        [("a", FN_SET.format(enc="that sentence encoder")), ("b", FN_SEX),
         ("", ABBR_GED)]) + ["\\end{table}"]
    write("etable_encoders_jno", lines, csv, ["Characteristic"] + heads)

    # ---- eTable: the outcome subsets
    cols = [coh] + [column(T, p) for p, _ in SUBSETS]
    heads = ["Cohort"] + [f"\\shortstack[r]{{{SUB2[lab]}\\textsuperscript{{{m}}}}}" for (_, lab), m in zip(SUBSETS, "abcd")]
    marks = {"Sex at birth": "e"}
    tab, csv = tabular(cols, heads, marks, "@{}>{\\raggedright\\arraybackslash}p{5.0cm}" + "r" * len(cols) + "@{}")
    cap = ("\\caption{\\textbf{eTable \\eTabSubsets. Characteristics of the Participants Each Outcome Is Measured On.} "
           "Each outcome is measured on the part of the cohort that has the data it needs. Values are No. (\\%) "
           "unless stated otherwise.}\\label{etab:subsets}")
    lines = ["\\begin{table}[htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}", cap,
             "\\vspace{4pt}"] + tab + notes(
        [("a", "Answered the Health Care Access and Utilization survey. The barriers to care are measured on them."),
         ("b", "At least 1 visit in the linked electronic health record. The emergency department share is "
               "measured on the 61\\,558 of them who also have a diagnosis code (column c, less the 73 who have a "
               "diagnosis code but lack a visit record), because most of those without one had a single visit from the "
               "program's own systems (eMethods \\eMethMeasures{})."),
         ("c", "At least 1 diagnosis code in the linked electronic health record. The 8 recorded conditions are "
               "measured on them."),
         ("d", "At least 14 nights of main sleep recorded by a wearable device. Sleep is measured on them."),
         ("e", FN_SEX), ("", ABBR_GED)]) + ["\\end{table}"]
    write("etable_subsets_jno", lines, csv, ["Characteristic", "Cohort"] + [lab for _, lab in SUBSETS])

    print(f"wrote -> {OUT}: table1_jno, etable_encoders_jno, etable_subsets_jno (.tex and .csv)")
    print(f"  cohort {coh[('n', '')]:,} | gte set {gte[('n', '')]:,} | "
          + " | ".join(f"{e} {column(T, p)[('n', '')]:,}" for p, e in ENCODERS[1:]))


if __name__ == "__main__":
    main()
