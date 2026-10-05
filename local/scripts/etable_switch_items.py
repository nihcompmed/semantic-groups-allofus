#!/usr/bin/env python
"""eTable 12 -- the items the semantic grouping places with questions their instrument keeps apart,
and the items it separates from their own instrument's neighbors.

WHAT "SWITCHING" MEANS HERE, and why it is counted across encoders. An item switches when the core of the
group it belongs to holds items from another instrument, or items carrying another construct label. Both are
read per encoder and then counted over the 6, because a switch seen under 1 encoder is a property of that
encoder and a switch seen under all 6 is a property of the questions. Only unanimous switches are tabled.

TWO KINDS, and they are different claims.
  A. Grouped with another INSTRUMENT under all 6 encoders. Where the construct label is also unchanged, the
     same content is being asked in 2 questionnaires and scored apart in practice, which is the case the
     Introduction states. Where the construct label changes too, the instruments separate contents that the
     wording puts together.
  B. Grouped with another CONSTRUCT under all 6 encoders while staying inside the item's own instrument. The
     instrument's own subscales are regrouped by what the questions ask.

Reads  outputs/basis/components.csv       every group's core items, per encoder (build_basis_package.py)
       data/items_151.csv                  (item, instrument, construct, text_of_record)

A group is the WHOLE core of its column, as in eTable 4 (etable_groups_jno.py) and the main text. Part A has 27 items,
3 of which never cross a construct label, and Part B 7. The table reads components.csv, so it runs locally.
Writes outputs/tables/etable_switch_items.tex   longtable, caption inside
       outputs/tables/switch_items.csv          every item with its per-encoder counts and partners, with text
       outputs/tables/etable_switch_items.md    readable preview, not part of the build
Run:   python scripts/etable_switch_items.py
"""
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

import re as _re

from construct_labels import display, reprintable   # display names in the printed table (switch_items.csv keeps the data labels)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs", "tables")
os.makedirs(OUT_DIR, exist_ok=True)
esc = lambda s: (str(s).replace("\\", "").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
                 .replace("#", "\\#").replace("_", "\\_"))
# the abbreviation in the instrument's display name, as in eTable 4 (etable_groups_jno.py)
def short(s):
    s = str(s)
    if s == "single item":
        return "single question"
    m = re.search(r"\(([^()]*)\)\s*$", s)
    assert m, f"no abbreviation in {s!r}"
    return m.group(1)

# every item under every encoder, with the whole cores it belongs to (group = component index, as in components.csv)
_T = pd.read_csv(os.path.join(HERE, "data", "items_151.csv"))
_C = pd.read_csv(os.path.join(HERE, "outputs", "basis", "components.csv"))
_rows = []
for _enc, _d in _C.groupby("encoder"):
    _holds = {}
    for _r in _d.itertuples():
        for _it in _r.core_items.split():
            _holds.setdefault(_it, []).append(f"C{int(_r.component)}")
    for _t in _T.itertuples():
        _rows.append({"encoder": _enc, "item": _t.item, "instrument": _t.instrument, "construct": _t.construct,
                      "groups": " ".join(_holds.get(_t.item, []))})
G = pd.DataFrame(_rows)
assert G.encoder.nunique() == 6 and len(G) == 6 * 151, "components.csv must hold the 6 encoders and items_151.csv 151 items"
# the wording only where the owners' terms permit reprinting (construct_labels.REPRINT); otherwise left blank
TEXT = {r.item: (r.text_of_record if reprintable(r.instrument_code) else "")
        for r in pd.read_csv(os.path.join(HERE, "data", "items_151.csv")).itertuples()}
INS = dict(zip(G.item, G.instrument))
CON = dict(zip(G.item, G.construct))
N_ENC = G.encoder.nunique()

# partners are listed by count, ties alphabetically. most_common() would keep set-insertion order, which
# follows Python's per-run string hashing, so tied partners would come out in a different order on every run.
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
        oi = {INS[m] for m in mates if INS[m] != r.instrument}
        oc = {CON[m] for m in mates if CON[m] != r.construct}
        if oi:
            R["ins"] += 1
            R["pi"].update(oi)
        if oc:
            R["con"] += 1
            R["pc"].update(oc)

S = pd.DataFrame([{"item": k, "instrument": INS[k], "construct": CON[k], "text": TEXT.get(k, ""),
                   "encoders_placing": v["have"], "other_instrument": v["ins"], "other_construct": v["con"],
                   "partner_instruments": "; ".join(f"{a} ({b})" for a, b in sorted(v["pi"].items(), key=lambda t: (-t[1], t[0]))),
                   "partner_constructs": "; ".join(f"{a} ({b})" for a, b in sorted(v["pc"].items(), key=lambda t: (-t[1], t[0])))}
                  for k, v in rec.items()])
S.to_csv(os.path.join(OUT_DIR, "switch_items.csv"), index=False)

A = S[S.other_instrument == N_ENC].sort_values(["other_construct", "construct", "item"])
B = S[(S.other_construct == N_ENC) & (S.other_instrument < N_ENC)].sort_values(["construct", "item"])
n_pure = int((A.other_construct == 0).sum())

caption = (
    # title case as in the other eTables
    "\\textbf{eTable \\eTabSwitch. Items the Semantic Grouping Places With Questions From Another Instrument, or With "
    "Questions Carrying Another Construct Label.} An item is counted as crossing an instrument under an "
    "encoder when the core of a group it belongs to holds items from a different instrument, and as crossing "
    "a construct when that core holds items with a different construct label. Both are counted over the 6 "
    "encoders. Only items that cross under all 6 are listed, so the list does not depend on the choice of encoder. "
    "Part A lists the items grouped with another instrument. The "
    f"{n_pure} of them that never cross a construct label are the same content asked in 2 questionnaires and "
    "scored apart in practice. Part B lists the items grouped with another construct while staying inside "
    "their own instrument. For these items, the groups formed from the question text combine items from different "
    "subscales of the same instrument. Supplement~\\SuppItemText{} (switch\\_items.csv) gives the counts for every "
    "item, including those that never cross, and the item wording where it is reprinted (eTable \\eTabItems). "
    "Construct labels are the display names of eTable "
    "\\eTabItems, which also spells out the instrument abbreviations.")

L = ["% generated by scripts/etable_switch_items.py -- do not edit by hand",
     "\\begingroup\\footnotesize\\setlength{\\tabcolsep}{4pt}",
     # ragged right as in eTable 4 (no hyphenated "dis-order"); item column wide enough for the longest
     # part of an ID broken at "_" ("generalmentalhealth"); widths sum to 0.78 so the table fits the text width
     "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{0.18\\linewidth} >{\\raggedright\\arraybackslash}p{0.11\\linewidth} "
     ">{\\raggedright\\arraybackslash}p{0.17\\linewidth} c c >{\\raggedright\\arraybackslash}p{0.32\\linewidth}@{}}",
     f"\\caption{{{caption}}}\\label{{etab:switch}}\\\\",
     "\\toprule Item & Instrument & Construct & \\multicolumn{2}{c}{Crosses, No. of 6} & Grouped with \\\\",
     "\\cmidrule(lr){4-5} & & & Instr. & Constr. & \\\\ \\midrule",
     "\\endfirsthead \\toprule Item & Instrument & Construct & Instr. & Constr. & Grouped with \\\\ \\midrule \\endhead",
     "\\bottomrule \\endfoot"]


def block(title, D):
    out = [f"\\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{{title}}}}} \\\\[2pt]"]
    for _, r in D.iterrows():
        # commas in the printed cell (no semicolons in the paper); switch_items.csv keeps "; " as its separator
        with_ = (", ".join(f"{display(m.group(1))} ({m.group(2)})" for m in _re.finditer(r"([^;]+?) \((\d+)\)", r.partner_constructs))
                 if r.partner_constructs else "the same construct, other instrument")
        # item IDs may break after "_" (otherwise overallhealth_generalmentalhealth overruns into the instrument column)
        out.append(f"{esc(r['item']).replace(chr(92) + '_', chr(92) + '_' + chr(92) + 'allowbreak{}')} & {esc(short(r.instrument))} & {esc(display(r.construct))} & "
                   f"{r.other_instrument} & {r.other_construct} & {esc(with_)} \\\\")
    return out


L += block(f"A. Grouped with another instrument under all {N_ENC} encoders", A)
L.append("\\addlinespace")
L += block(f"B. Grouped with another construct under all {N_ENC} encoders, within the item's own instrument", B)
L += ["\\end{longtable}", "\\endgroup"]
open(os.path.join(OUT_DIR, "etable_switch_items.tex"), "w").write("\n".join(L) + "\n")

M = ["# Items that switch (preview)", "",
     f"Part A, grouped with another instrument under all {N_ENC} encoders: {len(A)} items "
     f"({n_pure} of them never crossing a construct label).",
     f"Part B, grouped with another construct under all {N_ENC} encoders within their own instrument: {len(B)} items.", ""]
for title, D in ((f"A. another instrument, {N_ENC}/{N_ENC}", A), (f"B. another construct, within the instrument, {N_ENC}/{N_ENC}", D2 := B)):
    M += [f"## {title}", "", "| item | instrument | construct | instr | constr | grouped with |", "|---|---|---|---|---|---|"]
    for _, r in D.iterrows():
        with_ = r.partner_constructs if r.partner_constructs else "*same construct, other instrument*"
        M.append(f"| {r['item']} | {short(r.instrument)} | {r.construct} | {r.other_instrument} | {r.other_construct} | {with_} |")
    M.append("")
open(os.path.join(OUT_DIR, "etable_switch_items.md"), "w").write("\n".join(M) + "\n")

print(f"Part A {len(A)} items ({n_pure} never crossing a construct), Part B {len(B)} items")
print(f"-> {OUT_DIR}/" + "{etable_switch_items.tex, switch_items.csv, etable_switch_items.md}")
