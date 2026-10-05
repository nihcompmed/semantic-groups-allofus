#!/usr/bin/env python
"""Build the single table of record for the 151 items: ../data/items_151.csv.

One row per item, assembled from the sources below and checked against them. Every script in the
pipeline reads this file for item order, concept ids, survey, instrument, construct, the text
that is embedded, and the response scale. Supplement 2 is drawn from it (supp_items_jno.py).

Columns
  item                  short id, the pipeline's key (e.g. phq9_2)
  concept_id            All of Us survey concept id (observation_source_concept_id)
  survey_code, survey   All of Us survey module (code as in the source files, and the full name)
  instrument_code       instrument label as used in construct_map_151.csv
  instrument            instrument name as printed in the manuscript
  construct             the item's one documented construct (30 in all)
  text_of_record        THE TEXT THAT IS EMBEDDED: administered v9 wording minus declared stems
  text_administered     the CDR v9 ds_survey question string (a label only for the 10 single items
                        from The Basics, Lifestyle and Overall Health)
  text_published        the instrument's published item wording
  text_source           which of the two the text of record was cut from
  stems_removed         the declared stems removed, pipe-separated
  scale_min, scale_max  the item's numeric range in codebook units (from the encoding map)
  n_levels              number of substantive response options
  reverse_coded         True where the keyed direction is the reverse of the codebook order
  encoding_rule         the encoding map's rule name for the item
  construct_source      the basis of the item's construct assignment (for eTable 1):
                        the instrument's published scale or subscale, the section of a questionnaire
                        that scores items one by one, or a grouping the authors made across
                        instruments or of single items, which is said so in the text
  assignment_kind       published scale | published section | single item | authors' grouping

Sources: ../data/sources/ core_items_151.csv, administered_text_v9.csv, items_text_of_record.csv,
construct_map_151.csv, scales_prism.csv, and ../data/encoding_map_all151.csv.
Run:  python build_items_table.py
"""
import csv
import os
from collections import Counter

import pandas as pd

from _common import PROJECT

D = os.path.join(PROJECT, "data")
OUT = os.path.join(D, "items_151.csv")

SURVEY_NAME = {"Basics": "The Basics", "Lifestyle": "Lifestyle", "OverallHealth": "Overall Health",
               "SDOH": "Social Determinants of Health", "bhp": "Behavioral Health and Personality",
               "ehhwb": "Emotional Health History and Well-Being"}
# instrument label in construct_map_151.csv -> name printed in the manuscript (UCLA-V3 is printed as ULS-8,
# MOS-SS as mMOS-SS, CIDI-5 as WMH-CIDI)
INSTRUMENT_NAME = {
    "BFI-2-XS": "Big Five Inventory-2 Extra-Short Form (BFI-2-XS)",
    "nds (adapted)": "Perceived Neighborhood Disorder Scale, adapted (NDS)",
    "MHQ": "UK Biobank Mental Health Questionnaire (MHQ)",
    "PSS-10": "Perceived Stress Scale-10 (PSS-10)",
    "ACE-BRFSS": "Adverse Childhood Experiences module, BRFSS (ACE)",
    "EDS": "Everyday Discrimination Scale (EDS)",
    "PHQ-9": "Patient Health Questionnaire-9 (PHQ-9)",
    "CIDI-5": "WHO World Mental Health CIDI (WMH-CIDI)",
    "UCLA-V3": "Short-Form UCLA Loneliness Scale (ULS-8)",
    "MOS-SS": "Modified MOS Social Support Survey (mMOS-SS)",
    "DMS": "Discrimination in Medical Settings scale (DMS)",
    "GAD-7": "Generalized Anxiety Disorder-7 (GAD-7)",
    "ASRS-v1.1": "Adult ADHD Self-Report Scale v1.1, Part A (ASRS)",
    "BMMRS-DSE": "Daily Spiritual Experiences, BMMRS (DSES)",
    "scns (adapted)": "Neighborhood Social Cohesion scale, adapted (SCNS)",
    "PROMIS-Global": "PROMIS Global Health (PROMIS)",
    "HVS": "Hunger Vital Sign (HVS)",
    "single item": "single item",
}

# The basis of each construct assignment (eTable 1). By instrument code where the instrument is one
# scale, by item where an instrument spans constructs or an item is joined to another instrument's
# construct. "authors' grouping" marks every assignment that is not read off a published scale or section.
SOURCE_BY_INSTRUMENT = {
    "nds (adapted)": "NDS total scale, as adapted by the program",
    "PSS-10": "PSS-10 total scale",
    "ACE-BRFSS": "BRFSS ACE module, scored as a count of exposure categories",
    "EDS": "EDS total scale",
    "UCLA-V3": "ULS-8 total scale",
    "MOS-SS": "mMOS-SS total scale",
    "DMS": "DMS total scale",
    "GAD-7": "GAD-7 total scale",
    "ASRS-v1.1": "ASRS v1.1 Part A screener (6 items)",
    "BMMRS-DSE": "BMMRS Daily Spiritual Experiences subscale",
    "scns (adapted)": "SCNS total scale, as adapted by the program",
    "PROMIS-Global": "PROMIS Global Health, Global Mental Health items",
    "HVS": "HVS 2-item screen",
}
SOURCE_BY_ITEM = {
    "phq9_9": "PHQ-9 item 9 (suicidal ideation), grouped with the suicide screen items rather than the PHQ-9 total (authors' grouping)",
    "mhqukb_5": "MHQ depression section (lifetime screen), joined to the PHQ-9 depression items (authors' grouping)",
    "mhqukb_6": "MHQ depression section (lifetime screen), joined to the PHQ-9 depression items (authors' grouping)",
    "mhqukb_43": "MHQ mania section (lifetime screen)",
    "mhqukb_44": "MHQ mania section (lifetime screen)",
    "mhqukb_49": "MHQ psychotic experiences section, joined to the WMH-CIDI psychosis screen (authors' grouping)",
    "mhqukb_34": "MHQ traumatic events section, partner and sexual violence items",
    "mhqukb_35": "MHQ traumatic events section, partner and sexual violence items",
    "mhqukb_36": "MHQ traumatic events section, partner and sexual violence items",
    "mhqukb_37": "MHQ traumatic events section, partner and sexual violence items",
    "mhqukb_41": "MHQ traumatic events section, life-threatening illness item, joined to the WMH-CIDI life-threatening event item (authors' grouping)",
    "mhqukb_57": "MHQ happiness and subjective well-being section",
    "mhqukb_58": "MHQ happiness and subjective well-being section",
    "cidi5_16": "WMH-CIDI screener items on panic attack, social phobia and agoraphobia, grouped as panic and phobic anxiety (authors' grouping)",
    "cidi5_27": "WMH-CIDI screener items on panic attack, social phobia and agoraphobia, grouped as panic and phobic anxiety (authors' grouping)",
    "cidi5_30": "WMH-CIDI screener items on panic attack, social phobia and agoraphobia, grouped as panic and phobic anxiety (authors' grouping)",
    "cidi5_21": "WMH-CIDI psychosis screen (3 items)",
    "cidi5_22": "WMH-CIDI psychosis screen (3 items)",
    "cidi5_23": "WMH-CIDI psychosis screen (3 items)",
    "cidi5_25": "WMH-CIDI screener items on skin picking and hair pulling, grouped as body-focused repetitive behavior (authors' grouping)",
    "cidi5_26": "WMH-CIDI screener items on skin picking and hair pulling, grouped as body-focused repetitive behavior (authors' grouping)",
    "cidi5_5": "WMH-CIDI trauma screener item on a life-threatening event, joined to the MHQ life-threatening illness item (authors' grouping)",
    "livingsituation_stablehouseconcern": "The Basics single item on housing concern",
    "ips_16": "SDOH survey single item on neighborhood safety, joined to the NDS items (authors' grouping)",
    "nhs_xx": "SDOH survey single item on religious attendance",
    "worryanxiety": "Emotional Health survey single item on worry and anxiety, joined to the GAD-7 items (authors' grouping)",
    "ss_1": "Emotional Health survey suicide screen (3 items)",
    "ss_2": "Emotional Health survey suicide screen (3 items)",
    "ss_3": "Emotional Health survey suicide screen (3 items)",
}
SUBSTANCE = "Lifestyle survey single item on alcohol or tobacco use, grouped as substance use (authors' grouping)"


def construct_source(item, code, construct):
    if item in SOURCE_BY_ITEM:
        return SOURCE_BY_ITEM[item]
    if code == "BFI-2-XS":
        return f"BFI-2-XS {construct.lower()} domain scale (3 items)"
    if code == "PHQ-9":
        return "PHQ-9 total scale (items 1 to 8)"
    if code == "single item":
        assert construct == "Substance use", f"unlisted single item {item}"
        return SUBSTANCE
    return SOURCE_BY_INSTRUMENT[code]


def assignment_kind(source, code):
    if "authors' grouping" in source:
        return "authors' grouping"
    if " section" in source:
        return "published section"
    if code == "single item":
        return "single item"
    return "published scale"


def main():
    S = os.path.join(D, "sources")
    core = pd.read_csv(os.path.join(S, "core_items_151.csv"))                    # item, prompt, survey, concept_id
    adm = pd.read_csv(os.path.join(S, "administered_text_v9.csv"))               # cid, item, survey, question
    tor = pd.read_csv(os.path.join(S, "items_text_of_record.csv"))               # item, concept_id, survey, source, administered, prompt, stems_removed
    cmap = pd.read_csv(os.path.join(S, "construct_map_151.csv"))                 # item, instrument, construct, domain, prompt
    emap = pd.read_csv(os.path.join(D, "encoding_map_all151.csv"))               # concept_id,item,survey,instrument,rule,response,numeric_value
    scales = pd.read_csv(os.path.join(S, "scales_prism.csv"))                    # item,min,max,sentinels,reverse

    assert len(core) == 151 and core.item.is_unique and core.concept_id.is_unique
    for name, df in [("administered", adm), ("text of record", tor), ("construct map", cmap), ("scales", scales)]:
        assert set(df.item) == set(core.item), f"{name}: item set differs from core_items_151.csv"
    assert set(emap.item) == set(core.item)
    # concept ids agree everywhere they appear
    assert (adm.set_index("item").cid.reindex(core.item).values == core.concept_id.values).all()
    assert (tor.set_index("item").concept_id.reindex(core.item).values == core.concept_id.values).all()
    assert (emap.groupby("item").concept_id.first().reindex(core.item).values == core.concept_id.values).all()

    rng = emap.groupby("item").agg(scale_min=("numeric_value", "min"), scale_max=("numeric_value", "max"),
                                   n_levels=("response", "nunique"), encoding_rule=("rule", "first"))
    sc = scales.set_index("item")
    assert (rng.scale_min.reindex(core.item).values == sc["min"].reindex(core.item).values).all()
    assert (rng.scale_max.reindex(core.item).values == sc["max"].reindex(core.item).values).all()

    t = tor.set_index("item"); a = adm.set_index("item"); c = cmap.set_index("item")
    rows = []
    for r in core.itertuples():
        rows.append({
            "item": r.item, "concept_id": int(r.concept_id),
            "survey_code": r.survey, "survey": SURVEY_NAME[r.survey],
            "instrument_code": c.loc[r.item, "instrument"], "instrument": INSTRUMENT_NAME[c.loc[r.item, "instrument"]],
            "construct": c.loc[r.item, "construct"],
            "text_of_record": t.loc[r.item, "prompt"],
            "text_administered": a.loc[r.item, "question"],
            "text_published": r.prompt,
            "text_source": t.loc[r.item, "source"],
            "stems_removed": t.loc[r.item, "stems_removed"] if isinstance(t.loc[r.item, "stems_removed"], str) else "",
            "scale_min": rng.loc[r.item, "scale_min"], "scale_max": rng.loc[r.item, "scale_max"],
            "n_levels": int(rng.loc[r.item, "n_levels"]),
            "reverse_coded": str(sc.loc[r.item, "reverse"]).lower() == "true",
            "encoding_rule": rng.loc[r.item, "encoding_rule"],
            "construct_source": construct_source(r.item, c.loc[r.item, "instrument"], c.loc[r.item, "construct"]),
        })
        rows[-1]["assignment_kind"] = assignment_kind(rows[-1]["construct_source"], c.loc[r.item, "instrument"])
    df = pd.DataFrame(rows)
    assert a.survey.reindex(core.item).map(lambda s: s).tolist() == df.survey.tolist(), "survey names disagree with the v9 strings"
    df.to_csv(OUT, index=False)

    print(f"{len(df)} items -> {OUT}")
    print("surveys:", dict(Counter(df.survey)))
    inst = df.groupby("instrument").size().sort_values(ascending=False)
    print(f"instruments: {len(inst) - 1} scales with {int(inst.drop('single item').sum())} items, "
          f"{int(inst.get('single item', 0))} single items")
    print(inst.to_string())
    print("constructs:", df.construct.nunique())
    print("texts that differ, of record vs published:", int((df.text_of_record != df.text_published).sum()))
    print("reverse-coded:", int(df.reverse_coded.sum()), "| items with a stem removed:", int((df.stems_removed != "").sum()))
    print("basis of assignment:", dict(Counter(df.assignment_kind)))


if __name__ == "__main__":
    main()
