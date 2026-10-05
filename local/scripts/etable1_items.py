#!/usr/bin/env python
"""eTable 1 (JAMA form) -- the 151 items, their instruments and constructs, and the basis of each construct
assignment. One row per item, grouped by survey and instrument, in the order of the table of record.

Reads  data/items_151.csv (columns item, survey, instrument, construct, text_of_record, construct_source, assignment_kind)
Writes outputs/tables/etable1_items.tex   (a longtable with its caption inside, \input by the supplement)
Run:   python scripts/etable1_items.py
"""
import os
from collections import Counter

import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from construct_labels import display, reprintable   # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = pd.read_csv(os.path.join(HERE, "data", "items_151.csv"))
OUT = os.path.join(HERE, "outputs", "tables", "etable1_items.tex")
SHORT = {"Social Determinants of Health": "SDOH", "Emotional Health History and Well-Being": "Emotional Health",
         "Behavioral Health and Personality": "Behavioral Health", "Overall Health": "Overall Health",
         "Lifestyle": "Lifestyle", "The Basics": "The Basics"}
ORDER = ["Emotional Health", "Behavioral Health", "SDOH", "Overall Health", "Lifestyle", "The Basics"]
esc = lambda s: (str(s).replace("\\", "").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
                 .replace("#", "\\#").replace("_", "\\_"))

T["survey_short"] = T.survey.map(SHORT)
# instruments in survey order, then by size; single items after the instruments of their survey
T["is_single"] = (T.instrument_code == "single item").astype(int)
first_survey = T.groupby("instrument_code").survey_short.first().map(ORDER.index)
size = T.groupby("instrument_code").size()
T["sort_inst"] = T.instrument_code.map(lambda c: (first_survey[c], -size[c]))
T = T.reset_index().rename(columns={"index": "order"})
T = T.sort_values(["is_single", "sort_inst", "order"])

kinds = Counter(T.assignment_kind)
n_auth = kinds.get("authors' grouping", 0)
# title in title case, as in the other eTables
caption = ("\\textbf{eTable \\eTabItems. The 151 Items, Their Instruments and Constructs, and the Basis of Each Construct Assignment.} "
           "Item text is the wording administered in the All of Us Research Program (Curated Data Repository version 9) with "
           "recall windows and response-format stems removed, the text that was embedded. It is shown for the "
           "questionnaires whose owners' terms permit reprinting (eMethods \\eMethGroups). Every other item is shown by "
           "its item code, under which the All of Us Survey Codebooks give its wording. Each item has 1 construct. "
           "The basis of assignment is the instrument's published scale or subscale, or the section of a questionnaire that "
           "scores items one by one, except where marked \\emph{authors' grouping}, which is an assignment the authors made "
           "from item content across instruments or across single items "
           f"({n_auth} of 151 items). "
           "Abbreviations: ACE, Adverse Childhood Experiences; ASRS, Adult ADHD Self-Report Scale; BFI-2-XS, Big Five "
           "Inventory-2 Extra-Short Form; BMMRS, Brief Multidimensional Measure of Religiousness and Spirituality; DMS, "
           "Discrimination in Medical Settings scale; EDS, Everyday Discrimination Scale; GAD-7, Generalized Anxiety "
           "Disorder-7; HVS, Hunger Vital Sign; MHQ, UK Biobank Mental Health Questionnaire; mMOS-SS, modified Medical "
           "Outcomes Study Social Support Survey; NDS, Perceived Neighborhood Disorder Scale; PHQ-9, Patient Health "
           "Questionnaire-9; PROMIS, Patient-Reported Outcomes Measurement Information System; PSS-10, Perceived Stress "
           "Scale-10; SCNS, Neighborhood Social Cohesion scale; SDOH, Social Determinants of Health survey; ULS-8, "
           "Short-Form UCLA Loneliness Scale; WMH-CIDI, World Health Organization World Mental Health Composite "
           "International Diagnostic Interview.")

L = ["\\begin{scriptsize}", "\\setlength{\\tabcolsep}{3pt}",
     "\\begin{longtable}{p{2.3cm}p{1.5cm}p{6.1cm}p{2.3cm}p{3.9cm}}",
     f"\\caption{{{caption}}}\\\\", "\\toprule",
     "Instrument & Survey & Item text & Construct & Basis of assignment \\\\", "\\midrule", "\\endfirsthead",
     "\\toprule", "Instrument & Survey & Item text & Construct & Basis of assignment \\\\", "\\midrule", "\\endhead",
     "\\bottomrule", "\\endfoot"]
last = None
for r in T.itertuples():
    inst = r.instrument if r.instrument != "single item" else "Single item"
    cell_inst = esc(inst) if inst != last else ""
    last = inst
    text = esc(r.text_of_record) if reprintable(r.instrument_code) else "\\texttt{" + esc(r.item) + "}"
    L.append(f"{cell_inst} & {esc(r.survey_short)} & {text} & {esc(display(r.construct))} & {esc(r.construct_source)} \\\\")
L += ["\\end{longtable}", "\\end{scriptsize}"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(L) + "\n")
print(f"{len(T)} rows -> {OUT}; basis of assignment: {dict(kinds)}")
