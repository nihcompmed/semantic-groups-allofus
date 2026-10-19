#!/usr/bin/env python
"""appendix2_table1.py -- Appendix 2-table 1, every semantic group under each of the 6 encoders, one row per group,
named by the constructs of all its core items.

A group's core items are the fixed-point participation-ratio core down its column of the item-by-group matrix
(semantic_groups/step8_cores.py), so the core has no free threshold. A group whose core items carry opposite signs stays
1 row: groups are read by size only. Rows are ordered by the number of core items, largest first. The questionnaire of
a core item is the abbreviation in its instrument's name, and a stand-alone question is listed under the study it was
taken from (data/standalone_sources.csv). Construct labels are printed by their display names (common.DISPLAY); the
CSV keeps the data labels.

The heading of each encoder counts its groups on 1 construct, on 2, on 3 or more, the groups with core items from more
than 1 questionnaire, and the items not in any core.

Reads   outputs/basis/components.csv, data/items_151.csv, data/standalone_sources.csv
Writes  elife/tables/elife_app_groups.tex (longtable, caption inside); outputs/tables/groups.csv (one row per group)
Run     python appendix2_table1.py
"""
import os

import pandas as pd

from common import ENCODERS, ITEMS_CSV, OUTPUTS, SOURCES_CSV, TABLE_DATA, display_listed, esc, write_table

CAPTION = (r"\textbf{\AppTabGroups{}. The semantic groups under each of the six encoders.} One row per group. A "
           r"group's core items are the items with its largest weights, the fixed-point participation-ratio core of "
           r"its column (Appendix~\AppGroups{}), so the core does not depend on a chosen threshold. Each group is "
           r"named by the constructs of all its core items, with the number of core items per construct, largest "
           r"first. The groups were formed from the item text alone, before any participant's answers were used, so "
           r"an item's group membership does not use its questionnaire's name. Cores can overlap, an item can be in "
           r"more than one core, and some items are not in any core. The core items of every group and the text "
           r"encoded for each item are in Supplementary file~\SuppItems{}. Each stand-alone question is listed under the "
           r"study it was taken from, and the questionnaire and source abbreviations are spelled out in "
           r"Appendix~\AppGroups{}. gte-large-en-v1.5 is the prespecified encoder.")
HEAD = "Group & Core items & Constructs of the core items & Instruments of the core items"


def main():
    c = pd.read_csv(os.path.join(OUTPUTS, "basis", "components.csv"))
    it = pd.read_csv(ITEMS_CSV).set_index("item")
    abbr = it["instrument"].str.extract(r"\(([^()]*)\)\s*$")[0]      # the abbreviation in the instrument's name
    src = pd.read_csv(SOURCES_CSV).set_index("item")["source"]
    inst = abbr.where(it["instrument"] != "single item", src.reindex(it.index))
    assert inst.notna().all(), f"no abbreviation or source for {sorted(set(it.index[inst.isna()]))}"
    rows, head = [], []
    for enc in ENCODERS:
        d = c[c["encoder"] == enc].copy()
        d["instruments"] = [sorted({inst[i] for i in s.split()}) for s in d["core_items"]]
        d["n_instruments"] = d["instruments"].str.len()
        covered = set().union(*[set(s.split()) for s in d["core_items"]])
        head.append({"encoder": enc, "k": len(d), "one": int((d.n_constructs == 1).sum()),
                     "two": int((d.n_constructs == 2).sum()), "more": int((d.n_constructs >= 3).sum()),
                     "multi_instrument": int((d.n_instruments > 1).sum()), "no_core": 151 - len(covered)})
        for r in d.sort_values(["n_core_items", "component"], ascending=[False, True]).itertuples():
            rows.append({"encoder": enc, "group": int(r.component), "n_core_items": int(r.n_core_items),
                         "constructs": r.construct_profile.replace(":", " ").replace("; ", ", "),
                         "instruments": ", ".join(r.instruments)})
    R, H = pd.DataFrame(rows), pd.DataFrame(head)
    os.makedirs(TABLE_DATA, exist_ok=True)
    R.to_csv(os.path.join(TABLE_DATA, "groups.csv"), index=False)

    L = ["\\begingroup\\footnotesize\\setlength{\\tabcolsep}{4pt}",
         "\\begin{longtable}{@{}r r >{\\raggedright\\arraybackslash}p{0.44\\linewidth} "
         ">{\\raggedright\\arraybackslash}p{0.28\\linewidth}@{}}",
         f"\\caption{{{CAPTION}}}\\label{{apptab:groups}}\\\\",
         f"\\toprule {HEAD} \\\\ \\midrule",
         f"\\endfirsthead \\toprule {HEAD} \\\\ \\midrule \\endhead",
         "\\bottomrule \\endfoot"]
    for h in H.itertuples():
        L.append(f"\\multicolumn{{4}}{{@{{}}p{{0.97\\linewidth}}}}{{\\textbf{{{esc(h.encoder)}}} \\quad {h.k} groups, "
                 f"{h.one} on 1 construct, {h.two} on 2, {h.more} on 3 or more, {h.multi_instrument} with core items "
                 f"from more than 1 instrument, {h.no_core} items not in any core}} \\\\[2pt]")
        for r in R[R.encoder == h.encoder].itertuples():
            L.append(f"G{r.group + 1} & {r.n_core_items} & {esc(display_listed(r.constructs))} & {esc(r.instruments)} \\\\")
        L.append("\\addlinespace[4pt]")
    L += ["\\end{longtable}", "\\endgroup"]
    write_table("elife_app_groups", L, "appendix2_table1.py")
    print(H.to_string(index=False))


if __name__ == "__main__":
    main()
