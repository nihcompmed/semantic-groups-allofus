#!/usr/bin/env python
"""supp_items_jno.py -- Supplement 2 of the JAMA Network Open manuscript: the exact text embedded for each item, and
the core items of every semantic group under each encoder. The wording (administered, embedded, published) is given only
for the questionnaires whose owners' terms permit reprinting (construct_labels.REPRINT, eMethods 1); for every other
item the 3 text columns are empty and the All of Us Survey Codebooks give its wording under its item code.

  items_embedded_text.csv   one row per item: item, survey, instrument, construct, the ADMINISTERED wording (the
                            question text the Curated Data Repository, version 9, stores; a label for 10 items), the
                            stem removed before embedding (empty if none), the text embedded, the PUBLISHED wording
                            (the instrument's publication, or the All of Us Survey Codebooks for single questions and
                            the 10 label-only items), and which of the 2 the text embedded was cut from
                            (columns ordered so a row reads administered -> stem removed -> embedded, as eMethods 1
                            describes)
  item_answer_coding.csv    one row per answer option of every item: the number it is coded as, the item's lowest and
                            highest number, whether the item is reverse-scored, and the value that enters the group
                            scores Y = XP (rescaled to [-1, 1], sign reversed for the 13 reverse-scored items), exactly
                            as workbench/screen_battery.py load_battery computes it (eMethods 2, The answers). From
                            data/encoding_map_all151.csv and data/items_151.csv.
  semantic_group_cores.csv  one row per group and encoder: the group's core items (the items carrying most of its
                            weight) and the documented constructs of those items, as the main text and eTable 4
                            name the groups

The embedded text is `text_of_record` of data/items_151.csv, the column 0_embed.py reads by default. Re-embedding
text_of_record with gte-large-en-v1.5 on CPU reproduces every stored vector
(data/embeddings/admin151_items_gte-large-en-v1.5.npz) at cosine 1.000000 for all 151 items. The published wording
differs from it for 88 items (cosine 0.78 to below 0.9999).

JNO accepts large tables as an Excel file with a contents sheet; the 2 CSVs become its sheets at submission.

Reads   data/items_151.csv, data/encoding_map_all151.csv, outputs/basis/components.csv
Writes  outputs/supplementary_data_jno/items_embedded_text.csv, item_answer_coding.csv, semantic_group_cores.csv
Run     python scripts/supp_items_jno.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from construct_labels import reprintable   # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "outputs", "supplementary_data_jno")
ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5", "all-mpnet-base-v2", "e5-large-v2",
            "qwen3-embedding-0.6b"]                       # the prespecified encoder first
SOURCE = {"administered, CDR v9": "administered wording (Curated Data Repository, version 9)",
          "published question (CDR holds a label)": "published wording, All of Us Survey Codebooks (the repository stores a label only)"}


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(os.path.join(HERE, "data", "items_151.csv"))
    assert len(d) == 151 and d["item"].is_unique
    assert set(d["text_source"]) <= set(SOURCE), f"unknown text source {set(d['text_source']) - set(SOURCE)}"
    shown = d["instrument_code"].map(reprintable)
    hide = lambda col: d[col].where(shown, "")
    items = pd.DataFrame({
        "item": d["item"],
        "survey": d["survey"],
        "instrument": d["instrument"],
        "construct": d["construct"],
        "text_administered": hide("text_administered"),
        "stem_removed_before_embedding": d["stems_removed"].fillna(""),
        "text_embedded": hide("text_of_record"),
        "text_published": hide("text_published"),
        "text_embedded_cut_from": d["text_source"].map(SOURCE),
        "wording_published_in": shown.map({True: "this sheet",
                                           False: "All of Us Survey Codebooks, under the item code"}),
    })
    items.to_csv(os.path.join(OUT, "items_embedded_text.csv"), index=False)

    # the coding of every answer option and the value it enters the scores with (screen_battery.load_battery:
    # X = clip(2 (raw - lo) / (hi - lo) - 1, -1, 1), times -1 where reverse_coded)
    e = pd.read_csv(os.path.join(HERE, "data", "encoding_map_all151.csv"))
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
    coding.to_csv(os.path.join(OUT, "item_answer_coding.csv"), index=False)

    c = pd.read_csv(os.path.join(HERE, "outputs", "basis", "components.csv"))
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
    groups = pd.DataFrame(rows)
    groups.to_csv(os.path.join(OUT, "semantic_group_cores.csv"), index=False)

    k = groups.groupby("encoder", sort=False).size()
    print(f"wrote -> {OUT}: items_embedded_text.csv ({len(items)} items; stems removed from "
          f"{(items.stem_removed_before_embedding != '').sum()}; label-only {(d.text_source != 'administered, CDR v9').sum()}), "
          f"item_answer_coding.csv ({len(coding)} answer options, {coding.item.nunique()} items, "
          f"{int(coding.groupby('item').reverse_scored.first().sum())} reverse-scored), "
          f"semantic_group_cores.csv ({len(groups)} groups: " + ", ".join(f"{e} {n}" for e, n in k.items()) + ")")


if __name__ == "__main__":
    main()
