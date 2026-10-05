#!/usr/bin/env python
"""etable_groupings_jno.py -- eTable 13: decile 10 of the distance under every grouping, the 6 sentence encoders and
the 6 other groupings of eMethods 5, adjusted for age and sex, with 95% CIs. The comparison of groupings is reported
in the Discussion and the supplement, with the exact methods in the supplement.

LOCAL. Reads downloaded aggregates only:
  outcome_deciles/outcome_deciles.csv    thresholds met, barriers, sleep, every grouping by decile
                                         (workbench/outcome_deciles.py, each grouping's own deciles, the sweeps'
                                         adjustment curves fitted once on the cohort)
  outcome_deciles/outcome_deciles_meta.json   each grouping's family, k and k rule, checked against the table below
The emergency department share is NOT in this table: its coded-care version (ed_check_deciles.csv) exists for the 6
encoders only, and the outcome_deciles version uses the broader visit denominator (eMethods 3).
OUTPUTS  ../outputs/tables/etable_groupings_jno.{tex,csv}

Run:  python etable_groupings_jno.py
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WB = os.path.join(HERE, "..", "outputs", "workbench", "R2025Q4R6")
OUT = os.path.join(HERE, "..", "outputs", "tables")
# (grouping, label, family label, k rule label) in the order of eMethods 5
GROUPINGS = [("gte-large-en-v1.5", "gte-large-en-v1.5 (prespecified)", "Sentence encoder", "Horn"),
             ("bge-m3", "bge-m3", "Sentence encoder", "Horn"),
             ("bge-large-en-v1.5", "bge-large-en-v1.5", "Sentence encoder", "Horn"),
             ("all-mpnet-base-v2", "all-mpnet-base-v2", "Sentence encoder", "Horn"),
             ("e5-large-v2", "e5-large-v2", "Sentence encoder", "Horn"),
             ("qwen3-embedding-0.6b", "Qwen3-Embedding-0.6B", "Sentence encoder", "Horn"),
             ("tfidf4", "Word counts (TF-IDF)", "Item text", "Horn"),
             ("tfidf25", "Word counts (TF-IDF)", "Item text", "Encoders' range"),
             ("pca", "Principal components", "Answers", "Horn"),
             ("factor", "Factor model", "Answers", "Horn"),
             ("mca90", "Multiple correspondence analysis", "Answers", "Horn"),
             ("mca", "Multiple correspondence analysis", "Answers", "Principal components' k")]
# (key, label, scale, decimals, cohort column)
MEASURES = [("mean_breadth", "Thresholds met, mean No.", 1, 2, "mean_breadth"),
            ("access", "Any barrier to care, \\%", 100, 1, "access_rate"),
            ("cost", "Cost barrier, \\%", 100, 1, "cost_rate"),
            ("avoid", "Delayed care, clinician differs, \\%", 100, 1, "avoid_rate"),
            ("sleep_mean", "Mean sleep per night, min", 1, 1, "sleep_mean"),
            ("sleep_var_median", "Night-to-night variation, min", 1, 1, "sleep_var_median")]


def fmt(v, dec):
    return "" if pd.isna(v) else f"{v:.{dec}f}"


def cell(v, lo, hi, dec):
    return f"\\shortstack{{{fmt(v, dec)}\\\\{{\\tiny ({fmt(lo, dec)}-{fmt(hi, dec)})}}}}"


def main():
    od = pd.read_csv(os.path.join(WB, "outcome_deciles", "outcome_deciles.csv"))
    meta = json.load(open(os.path.join(WB, "outcome_deciles", "outcome_deciles_meta.json")))
    k_of = {g["grouping"]: int(g["k"]) for g in meta["groupings_all"]}
    assert set(k_of) == {g for g, *_ in GROUPINGS}, "the groupings in the file are not the 12 of eMethods 5"
    fam_of = {g["grouping"]: g["family"] for g in meta["groupings_all"]}
    for g, _, fam, _ in GROUPINGS:
        assert (fam_of[g] == "item text") == (fam in ("Sentence encoder", "Item text")), f"{g}: family mismatch"
    coh = od[od.grouping == "cohort"].iloc[0]
    d10 = od[od.decile.astype(str) == "10"].set_index("grouping")
    assert (d10["n_set"] == 9035).all(), "decile 10 is not 9035 participants under every grouping"

    lines, rows = [], []
    coh_cells = [fmt(sc * coh[ccol], dec) for key, _, sc, dec, ccol in MEASURES]
    lines.append("Cohort & & & " + " & ".join(coh_cells) + " \\\\")
    last_fam = None
    for g, label, fam, krule in GROUPINGS:
        if fam != last_fam:
            header = {"Sentence encoder": "Item text, sentence encoders", "Item text": "Item text, word counts",
                      "Answers": "Estimated from the answers"}[fam]
            lines.append(f"\\midrule\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{header}}}}} \\\\")
            last_fam = fam
        r = d10.loc[g]
        cells = []
        for key, _, sc, dec, _ in MEASURES:
            v, lo, hi = sc * r[f"{key}_adj"], sc * r[f"{key}_adj_lo"], sc * r[f"{key}_adj_hi"]
            cells.append(cell(v, lo, hi, dec))
            rows.append({"grouping": g, "k": k_of[g], "measure": key, "value": v, "lo": lo, "hi": hi,
                         "cohort": sc * coh[MEASURES[[m[0] for m in MEASURES].index(key)][4]]})
        lines.append(f"\\hspace*{{1em}}{label} & {k_of[g]} & {krule} & " + " & ".join(cells) + " \\\\")
    head = ("Grouping & $k$ & $k$ set by & " + " & ".join(m[1] for m in MEASURES) + " \\\\")
    cap = ("\\caption{\\textbf{eTable \\eTabGroupings. Decile 10 of the Distance From the Cohort's Typical Scores Under "
           "Every Grouping.} Each grouping forms its own groups, its own distance, and its own deciles (eMethods \\eMethGroupings{}). "
           "Decile 10 is the most atypical tenth of the cohort under that grouping, 9035 "
           "participants in each, and the deciles of different groupings share participants. Values are adjusted for "
           "age and sex (eMethods \\eMethMeasures{}), with 95\\% CIs. The cohort row is the observed value, which the adjustment leaves "
           "unchanged. Each outcome is computed on the participants with the data it needs, as in eTable "
           "\\eTabDeciles{}. Horn indicates Horn's parallel analysis on that grouping's own matrix. The second word-count "
           "grouping uses 25 groups, within the encoders' range of 23 to 27, and the second multiple correspondence "
           "analysis the 25 of principal components on the same answers. TF-IDF indicates term frequency-inverse "
           "document frequency.}\\label{etab:groupings}")
    tex = ["% generated by scripts/etable_groupings_jno.py -- do not edit by hand",
           "\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.9}",
           "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{4.6cm}r>{\\raggedright\\arraybackslash}p{2.2cm}"
           "*{6}{>{\\centering\\arraybackslash}p{2.0cm}}@{}}",
           cap + " \\\\",
           "\\toprule", head, "\\midrule", "\\endfirsthead",
           "\\toprule", head, "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"] + lines + ["\\end{longtable}", "\\endgroup"]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "etable_groupings_jno.tex"), "w").write("\n".join(tex) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "etable_groupings_jno.csv"), index=False)
    gte = d10.loc["gte-large-en-v1.5"]
    print("gte decile 10 (checks against the Results):", round(gte["mean_breadth_adj"], 2),
          round(100 * gte["access_adj"], 1), round(100 * gte["cost_adj"], 1), round(100 * gte["avoid_adj"], 1),
          round(gte["sleep_mean_adj"], 1), round(gte["sleep_var_median_adj"], 1))
    print("wrote", os.path.join(OUT, "etable_groupings_jno.tex"), len(rows), "values")


if __name__ == "__main__":
    main()
