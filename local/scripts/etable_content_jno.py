#!/usr/bin/env python
"""etable_content_jno.py -- eTable 8 of the JAMA Network Open supplement: the groups contributing most to the distance
(Figure 3) under all 6 encoders.

LOCAL. Reads the downloaded direction_own/direction_top3_own{,_invisible}.csv (workbench/direction_own.py).
Per encoder and group, for 2 sets of participants: those above the encoder's own 95th percentile (4518) and those
among them with all 6 screening thresholds evaluable who met none (n_set of the _invisible file). For each set:
  No. (%)  participants with the group among the 3 largest parts of their squared distance (top 3 by signed share),
           percent of the set (each encoder's column sums to 300)
The printed table shows No. (%) only, because the paper does not use the mean |z|, mean |r| and mean share. The CSV
keeps every column of the download.
Groups numbered and named as in eTable 4 (etable_groups_jno.py: components.csv construct_profile, the
number of core items per construct). Rows where n_top was written "<=20" in the download keep the mark and carry no
averages (the download has none). gte-large-en-v1.5 = Figure 3 (fig_content_bars_jno.py reads the same files).
OUTPUTS  ../outputs/tables/etable_content_jno.{tex,csv}

Run:  python etable_content_jno.py
"""
import os

import pandas as pd

from construct_labels import display_listed   # display names in the printed table (CSV keeps the data labels)
from etable_deciles_jno import ENCODERS, fmt

HERE = os.path.dirname(os.path.abspath(__file__))
WB = os.path.join(HERE, "..", "outputs", "workbench", "R2025Q4R6", "direction_own")
OUT = os.path.join(HERE, "..", "outputs", "tables")
SETS = [("A", "direction_top3_own.csv"), ("B", "direction_top3_own_invisible.csv")]
STATS = ["n_top", "pct_top", "mean_abs_z", "mean_abs_r", "mean_share"]
esc = lambda s: str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def load(f, tag):
    d = pd.read_csv(os.path.join(WB, f))
    d = d[d.ranking == "share"]
    assert not d.duplicated(["encoder", "component"]).any(), f"{f}: duplicate rows"
    d = d[["encoder", "component", "n_set", "n_core_items", "semantic_groups"] + STATS]
    # every encoder's n_top partitions 3 slots per participant (the "<=20" rows included), checked where all are numeric
    for enc, g in d.groupby("encoder"):
        num = pd.to_numeric(g.n_top, errors="coerce")
        if num.notna().all():
            assert num.sum() == 3 * g.n_set.iloc[0], f"{f} {enc}: top-3 slots {num.sum()} != 3 x {g.n_set.iloc[0]}"
    return d.rename(columns={c: f"{c}_{tag}" for c in STATS + ["n_set"]})


def count(n, pct):
    if pd.isna(pct):
        return "$\\leq$20" if str(n).strip() == "<=20" else str(n)
    return f"{int(float(n))} ({pct:.1f})"


def avg(v):
    return "---" if pd.isna(v) else fmt(v, 2)


def main():
    A, B = (load(f, t) for t, f in SETS)
    M = A.merge(B.drop(columns=["n_core_items", "semantic_groups"]), on=["encoder", "component"], how="outer",
                validate="one_to_one")
    assert M.notna()[["n_set_A", "n_set_B", "semantic_groups"]].all().all(), "a group is missing from one set"
    M["constructs"] = M.semantic_groups.str.replace(":", " ", regex=False).str.replace("; ", ", ", regex=False)
    M["n_A"] = pd.to_numeric(M.n_top_A, errors="coerce")
    lines, rows = [], []
    for enc, enc_label in ENCODERS:
        g = M[M.encoder == enc].sort_values(["n_A", "component"], ascending=[False, True])
        assert len(g), enc
        nA, nB = int(g.n_set_A.iloc[0]), int(g.n_set_B.iloc[0])
        lines.append(f"\\midrule\\multicolumn{{4}}{{@{{}}l}}{{\\textbf{{{esc(enc_label)}}} \\quad {len(g)} groups, "
                     f"{nA} and {nB} participants}} \\\\")
        for r in g.itertuples():
            lines.append(f"G{r.component + 1} & {esc(display_listed(r.constructs))} & {count(r.n_top_A, r.pct_top_A)} & "
                         f"{count(r.n_top_B, r.pct_top_B)} \\\\")
            rows.append({"encoder": enc, "group": r.component, "constructs": r.constructs, "n_core_items": r.n_core_items,
                         **{c: getattr(r, c) for c in [f"{s}_{t}" for t in "AB" for s in ["n_set"] + STATS]}})
    cap = ("\\caption{\\textbf{eTable \\eTabContent. Top-3 Contributors to the Distance, Under Each of the 6 Encoders.} "
           "Each participant's squared distance divides exactly into 1 part per group, and each part depends on the "
           "participant's scores on all groups at once (eMethods \\eMethDistance{}). A group is a top-3 contributor for a "
           "participant when its part is among the participant's 3 largest. No. (\\%) is the number of participants for whom "
           "the group was a top-3 contributor, as a percentage of the participants above the 95th percentile (4518 under "
           "every encoder) or of those among them below every threshold, who could be evaluated on all 6 screening "
           "thresholds and did not meet any (the second number in each encoder's heading). Each column of percentages sums "
           # 2026-10-05 (step 4): 'documented construct(s)' -> 'construct(s)' (19 items carry the authors' grouping). Previous, VERBATIM:
           # "to 300 within an encoder. Groups are numbered and named as in eTable \\eTabGroups{}, by the documented "
           "to 300 within an encoder. Groups are numbered and named as in eTable \\eTabGroups{}, by the "
           "constructs of their core items with the number of core items per construct. Where 20 or fewer participants had "
           "a group as a top-3 contributor, the count is shown as $\\leq$20, following the All of Us Data and Statistics "
           "Dissemination Policy. gte-large-en-v1.5 is the prespecified encoder, and its percentages for the 10 most "
           "frequent groups of each set are plotted in Figure \\FigContent{}.}\\label{etab:content}")
    head1 = " & & Above the 95th percentile & Below every threshold \\\\"
    head2 = "Group & Constructs of the core items & No. (\\%) & No. (\\%) \\\\"
    tex = ["% generated by scripts/etable_content_jno.py -- do not edit by hand",
           "\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3pt}",
           "\\begin{longtable}{@{}r>{\\raggedright\\arraybackslash}p{9.0cm}rr@{}}",
           cap + " \\\\", "\\toprule", head1, head2, "\\endfirsthead", "\\toprule", head1, head2, "\\endhead",
           "\\bottomrule", "\\endfoot"] + lines + ["\\end{longtable}", "\\endgroup"]
    os.makedirs(OUT, exist_ok=True)   # a fresh copy has no outputs/tables yet
    open(os.path.join(OUT, "etable_content_jno.tex"), "w").write("\n".join(tex) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "etable_content_jno.csv"), index=False)
    g = M[M.encoder == ENCODERS[0][0]].sort_values("n_A", ascending=False).head(4)
    print(g[["component", "constructs", "n_top_A", "pct_top_A", "n_top_B", "pct_top_B"]].to_string(index=False))
    print("wrote", os.path.join(OUT, "etable_content_jno.tex"), f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
