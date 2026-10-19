#!/usr/bin/env python
"""appendix2_table2.py -- Appendix 2-table 2, the items that the semantic groups place with questions from another
questionnaire or with questions carrying another construct label, under all 6 encoders.

CROSSING. An item crosses a questionnaire under an encoder when the core of a group it belongs to holds items from a
different questionnaire, each stand-alone question counted under the study it was taken from (data/
standalone_sources.csv), and crosses a construct when that core holds items with a different construct label. A group
is the whole core of its column, as in Appendix 2-table 1. Both are counted over the 6 encoders: a crossing under 1
encoder is a property of that encoder, a crossing under all 6 a property of the questions. Only items that cross under
all 6 are listed.
  A  grouped with another questionnaire under all 6 encoders. Where the construct label is unchanged too, the same
     content is taken from 2 sources.
  B  grouped with another construct under all 6 encoders while staying inside the item's own questionnaire: the
     questionnaire's own subscales are regrouped by what the questions ask.
Partners are listed by count, ties alphabetically, so the order does not depend on Python's string hashing.

Reads   outputs/basis/components.csv, data/items_151.csv, data/standalone_sources.csv
Writes  elife/tables/elife_app_switch.tex (longtable, caption inside); outputs/tables/switch_items.csv (every item with
        its counts and partners; its wording only where the questionnaire's terms permit reprinting)
Run     python appendix2_table2.py
"""
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from common import ITEMS_CSV, OUTPUTS, SOURCES_CSV, TABLE_DATA, display, esc, reprintable, write_table

CAPTION = (r"\textbf{\AppTabSwitch{}. Items that the semantic groups place with questions from another "
           r"questionnaire or with questions carrying another construct label.} An item crosses a questionnaire "
           r"under an encoder when the core of a group it belongs to holds items from a different questionnaire, with "
           r"each stand-alone question counted under the study it was taken from, and "
           r"crosses a construct when that core holds items with a different construct label. Both are counted over "
           r"the six encoders, and only items that cross under all six are listed, so the list does not depend on "
           r"the choice of encoder. Part A lists the items grouped with another questionnaire. The eight of them that "
           r"never cross a construct label are the same content taken from two sources, suicidal thoughts and "
           r"behavior in the PHQ-9 and the SITBI, and tobacco use in the TUS-CPS and MVP and the PATH Study. "
           r"Part B lists the items grouped with another construct while staying inside their own "
           r"questionnaire, so that the groups formed from the question text combine items from different subscales "
           r"of the same questionnaire. Supplementary file~\SuppItems{} gives the counts for every item, including "
           r"those that never cross, the item wording where it is reprinted, and the construct labels. The "
           r"questionnaire abbreviations are spelled out in Appendix~\AppGroups{}.")
SOURCE = pd.read_csv(SOURCES_CSV).set_index("item")["source"]


def short(s):
    """The abbreviation in an instrument's name; a source study as it is."""
    s = str(s)
    if s in set(SOURCE.values):
        return s
    m = re.search(r"\(([^()]*)\)\s*$", s)
    assert m, f"no abbreviation in {s!r}"
    return m.group(1)


def switch_items():
    """One row per item: its questionnaire and construct, its wording where reprintable, and under how many encoders
    it crosses a questionnaire or a construct, with the partners and their counts."""
    T = pd.read_csv(ITEMS_CSV)
    C = pd.read_csv(os.path.join(OUTPUTS, "basis", "components.csv"))
    rows = []
    for enc, d in C.groupby("encoder"):
        holds = {}
        for r in d.itertuples():
            for it in r.core_items.split():
                holds.setdefault(it, []).append(f"C{int(r.component)}")
        for t in T.itertuples():
            rows.append({"encoder": enc, "item": t.item,
                         "instrument": SOURCE[t.item] if t.instrument == "single item" else t.instrument,
                         "construct": t.construct, "groups": " ".join(holds.get(t.item, []))})
    G = pd.DataFrame(rows)
    assert G.encoder.nunique() == 6 and len(G) == 6 * 151, "components.csv must hold the 6 encoders and items_151.csv 151 items"
    text = {r.item: (r.text_of_record if reprintable(r.instrument_code) else "") for r in T.itertuples()}
    ins, con = dict(zip(G.item, G.instrument)), dict(zip(G.item, G.construct))

    rec = defaultdict(lambda: {"have": 0, "ins": 0, "con": 0, "pi": Counter(), "pc": Counter()})
    for enc, d in G.groupby("encoder"):
        grp = defaultdict(list)
        for _, r in d.iterrows():
            for g in str(r.groups).split():
                if g and g != "nan":
                    grp[g].append(r["item"])
        for _, r in d.iterrows():
            gs = [g for g in str(r.groups).split() if g and g != "nan"]
            if not gs:
                continue
            R = rec[r["item"]]
            R["have"] += 1
            mates = {m for g in gs for m in grp[g] if m != r["item"]}
            oi = {ins[m] for m in mates if ins[m] != r.instrument}
            oc = {con[m] for m in mates if con[m] != r.construct}
            if oi:
                R["ins"] += 1
                R["pi"].update(oi)
            if oc:
                R["con"] += 1
                R["pc"].update(oc)
    listed = lambda c: "; ".join(f"{a} ({b})" for a, b in sorted(c.items(), key=lambda t: (-t[1], t[0])))
    return pd.DataFrame([{"item": k, "instrument": ins[k], "construct": con[k], "text": text.get(k, ""),
                          "encoders_placing": v["have"], "other_instrument": v["ins"], "other_construct": v["con"],
                          "partner_instruments": listed(v["pi"]), "partner_constructs": listed(v["pc"])}
                         for k, v in rec.items()])


def block(title, D):
    out = [f"\\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{{title}}}}} \\\\[2pt]"]
    for _, r in D.iterrows():
        # commas in the printed cell; switch_items.csv keeps "; " as its separator
        with_ = (", ".join(f"{display(m.group(1))} ({m.group(2)})"
                           for m in re.finditer(r"([^;]+?) \((\d+)\)", r.partner_constructs))
                 if r.partner_constructs else "the same construct, other instrument")
        # an item code may break after "_" (otherwise the longest overruns into the next column)
        item = esc(r["item"]).replace("\\_", "\\_\\allowbreak{}")
        out.append(f"{item} & {esc(short(r.instrument))} & {esc(display(r.construct))} & "
                   f"{r.other_instrument} & {r.other_construct} & {esc(with_)} \\\\")
    return out


def main():
    S = switch_items()
    os.makedirs(TABLE_DATA, exist_ok=True)
    S.to_csv(os.path.join(TABLE_DATA, "switch_items.csv"), index=False)
    n = 6
    A = S[S.other_instrument == n].sort_values(["other_construct", "construct", "item"])
    B = S[(S.other_construct == n) & (S.other_instrument < n)].sort_values(["construct", "item"])
    L = ["\\begingroup\\footnotesize\\setlength{\\tabcolsep}{4pt}",
         "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{0.18\\linewidth} "
         ">{\\raggedright\\arraybackslash}p{0.11\\linewidth} >{\\raggedright\\arraybackslash}p{0.17\\linewidth} c c "
         ">{\\raggedright\\arraybackslash}p{0.32\\linewidth}@{}}",
         f"\\caption{{{CAPTION}}}\\label{{apptab:switch}}\\\\",
         "\\toprule Item & Instrument & Construct & \\multicolumn{2}{c}{Crosses, No. of 6} & Grouped with \\\\",
         "\\cmidrule(lr){4-5} & & & Instr. & Constr. & \\\\ \\midrule",
         "\\endfirsthead \\toprule Item & Instrument & Construct & Instr. & Constr. & Grouped with \\\\ \\midrule \\endhead",
         "\\bottomrule \\endfoot"]
    L += block(f"A. Grouped with another instrument under all {n} encoders", A)
    L.append("\\addlinespace")
    L += block(f"B. Grouped with another construct under all {n} encoders, within the item's own instrument", B)
    L += ["\\end{longtable}", "\\endgroup"]
    write_table("elife_app_switch", L, "appendix2_table2.py")
    print(f"Part A {len(A)} items ({int((A.other_construct == 0).sum())} never crossing a construct), Part B {len(B)} items")


if __name__ == "__main__":
    main()
