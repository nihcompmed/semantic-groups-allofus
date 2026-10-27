#!/usr/bin/env python
"""supplementary_file1.py -- Supplementary File 1: the 151 items, their coding, and the core items of every semantic
group and of every factor of the factor model.

SHEETS
  Items            one row per item: survey, questionnaire, construct (display name) and where the construct label comes
                   from, the wording as administered (the question text the Curated Data Repository, version 9, stores;
                   a label for 10 items), the stem removed before encoding, the text encoded (text_of_record, the text the
                   stored embeddings were computed from), the published wording, which of the 2 the encoded text was cut
                   from, the study each stand-alone question was taken from, and the item set aside in the models of
                   recorded conditions
  Answer coding    one row per answer option: the number it is coded as, the item's lowest and highest number, whether
                   the item is reverse-scored, and the value entering the group scores, X = 2 (raw - lo) / (hi - lo) - 1
                   clipped to [-1, 1], sign reversed where reverse-scored (the Workbench's scoring.load_battery)
  Group cores      one row per group and encoder: its core items and their constructs
  Switch items     every item: under how many of the 6 encoders it shares a core with items of another questionnaire
                   or with items of another construct label (appendix2_table2.py)
  Factor cores     the core items of the 25 factors under both rotations (appendix4_table1.py)
The wording is reprinted only where the questionnaire's terms allow it under the paper's CC BY license (common.
REPRINT_REPO: PHQ-9, GAD-7, the adverse childhood experiences module, the stand-alone questions); otherwise the text
columns are empty and the All of Us Survey Codebooks give the wording under the item code.

Reads   data/items_151.csv, data/encoding_map_all151.csv, data/standalone_sources.csv, outputs/basis/components.csv,
        the download factor_decile10
Writes  output/supplementary/Supplementary_file_1.xlsx (+ csv/)
Run     python supplementary_file1.py
"""
import os

import pandas as pd

from appendix2_table2 import switch_items
from appendix4_table1 import factor_rows
from common import DATA, ENCODERS, ITEMS_CSV, OUTPUTS, SOURCES_CSV, display, reprintable
from workbook import write

SOURCE = {"administered, CDR v9": "administered wording (Curated Data Repository, version 9)",
          "published question (CDR holds a label)": "published wording, All of Us Survey Codebooks (the repository stores a label only)"}
TEXT = ("text_administered", "text_embedded", "text_published", "stem_removed_before_embedding")
README = [
    ("Supplementary File 1", "The 151 questionnaire items, their coding, and the core items of every semantic group "
                             "and of every factor of the factor model."),
    ("Sheet Items", "Survey, questionnaire, construct (named in the item's documentation or, where marked, by us), "
                    "the wording as administered, the stem removed before encoding, the text encoded, and the "
                    "published wording. Wording is reprinted only for the PHQ-9, GAD-7, adverse childhood "
                    "experiences module, and the stand-alone questions; for the others the All of Us Survey "
                    "Codebooks give the wording under the item code. For each stand-alone question, the study "
                    "All of Us took it from, as its per-survey source document names it."),
    ("Sheet Answer coding", "Every answer option: the number it is coded as, the item's lowest and highest number, "
                            "whether the item is reverse-scored, and the value entering the group scores (rescaled "
                            "to -1 to 1, sign reversed for the 13 reverse-scored items)."),
    ("Sheet Group cores", "The core items of every group under each of the 6 encoders and their constructs."),
    ("condition_models", "The item set aside, with weight 0 in every group, for the models of recorded conditions."),
    ("Sheet Switch items", "For every item, under how many of the 6 encoders it shares a core with items of "
                           "another questionnaire or with items carrying another construct label "
                           "(Table S2 lists the items that do so under all 6)."),
    ("Sheet Factor cores", "The core items of the 25 factors of the factor model fitted to the answers, under "
                           "oblimin rotation (the factor numbers in the text) and varimax rotation, their "
                           "constructs and questionnaires (Table S4)."),
]


def items(d):
    assert len(d) == 151 and d["item"].is_unique
    assert set(d["text_source"]) <= set(SOURCE), f"unknown text source {set(d['text_source']) - set(SOURCE)}"
    shown = d["instrument_code"].map(lambda c: reprintable(c, repo=True))
    hide = lambda col: d[col].where(shown, "")
    ss = pd.read_csv(SOURCES_CSV).set_index("item")
    return pd.DataFrame({
        "item": d["item"],
        "survey": d["survey"],
        "instrument": d["instrument"],
        "construct": [display(c) for c in d["construct"]],
        "construct_source": d["construct_source"],
        "text_administered": hide("text_administered"),
        "stem_removed_before_embedding": d["stems_removed"].fillna("").where(shown, ""),
        "text_embedded": hide("text_of_record"),
        "text_published": hide("text_published"),
        "text_embedded_cut_from": d["text_source"].map(SOURCE),
        "wording_published_in": shown.map({True: "this sheet", False: "All of Us Survey Codebooks, under the item code"}),
        "source_of_standalone_question": [f"{ss.loc[i, 'source_full']} ({ss.loc[i, 'source']}; All of Us source document "
                                          f"{ss.loc[i, 'source_document']}, source {ss.loc[i, 'source_number']})"
                                          if i in ss.index else "" for i in d["item"]],
        "condition_models": ["weight 0 in every group (asks about a diagnosis, Appendix S5)" if i == "mhqukb_41" else ""
                             for i in d["item"]],
    })


def answer_coding(d):
    e = pd.read_csv(os.path.join(DATA, "encoding_map_all151.csv"))
    it = d.set_index("item")
    assert set(e["item"]) == set(d["item"]), "the encoding map and the item table cover different items"
    lo = e["item"].map(it["scale_min"]).astype(float)
    hi = e["item"].map(it["scale_max"]).astype(float)
    rev = e["item"].map(it["reverse_coded"]).astype(str).str.lower().eq("true")
    x = (2.0 * (e["numeric_value"].astype(float) - lo) / (hi - lo) - 1.0).clip(-1.0, 1.0)
    x = x.where(~rev, -x)
    coding = pd.DataFrame({"item": e["item"], "instrument": e["item"].map(it["instrument"]),
                           "answer_option": e["response"], "coded_value": e["numeric_value"],
                           "item_lowest": lo, "item_highest": hi, "reverse_scored": rev,
                           "value_in_group_scores": x.round(6) + 0.0})   # + 0.0 turns -0.0 into 0.0
    order = {k: n for n, k in enumerate(d["item"])}
    coding = coding.assign(_o=coding["item"].map(order)).sort_values(["_o", "coded_value", "answer_option"],
                                                                     kind="stable").drop(columns="_o")
    assert coding.groupby("item")["value_in_group_scores"].agg(["min", "max"]).abs().eq(1.0).all().all(), \
        "every item should run from -1 to 1"
    return coding


def group_cores(d):
    c = pd.read_csv(os.path.join(OUTPUTS, "basis", "components.csv"))
    assert set(c["encoder"]) == set(ENCODERS)
    known = set(d["item"])
    rows = []
    for enc in ENCODERS:
        for r in c[c["encoder"] == enc].sort_values("component").itertuples():
            core = r.core_items.split()
            assert set(core) <= known, f"{enc} group {r.component}: a core item is not among the 151"
            rows.append({"encoder": enc, "group": f"G{int(r.component) + 1}", "n_core_items": int(r.n_core_items),
                         "n_constructs": int(r.n_constructs), "constructs_of_core_items": r.construct_profile,
                         "core_items": "; ".join(core)})
    return pd.DataFrame(rows)


def main():
    d = pd.read_csv(ITEMS_CSV)
    code = d.set_index("item")["instrument_code"]
    sw = switch_items()
    sw["text"] = [t if reprintable(code[i], repo=True) else "" for i, t in zip(sw["item"], sw["text"])]
    fc, _ = factor_rows()
    fc = fc.assign(factor=["F" + str(f) for f in fc.factor],
                   constructs=[", ".join(f"{display(x.rsplit(' ', 1)[0])} {x.rsplit(' ', 1)[1]}" for x in c.split(", "))
                               for c in fc.constructs])
    write("Supplementary_file_1", README, [("Items", items(d)), ("Answer coding", answer_coding(d)),
                                           ("Group cores", group_cores(d)), ("Switch items", sw), ("Factor cores", fc)])


if __name__ == "__main__":
    main()
