#!/usr/bin/env python
"""Build the text of record for the 151 items.

The text embedded is what the respondent read, the administered question string of CDR v9
(`ds_survey.question`), with a DECLARED set of leading stems removed. A stem is removed only when it
carries a recall window ("Over the last 2 weeks", "In the last month", "In your life") or a
response format ("how often have you been bothered by", "How much you agree or disagree that",
"How often do you", and the auxiliary inversion "are you / do you / does / do" that a format stem
leaves behind). A frame that carries content is kept: "During your first 18 years of life",
"Before age 18", "Did you ever", "Have you ever", "Were you ever in your life", and the
BFI-2-XS frame "I am someone who" and "In general,". Two trailing
response-format tails are removed ("Would you say", "Would you say that you..."). The rules are
applied at the start of the string, repeatedly, longest first, until none matches.

For the 10 single items from The Basics, Lifestyle and Overall Health the CDR string is only a
label ("Alcohol: Alcohol Participant"), so the published full question in core_items_151.csv is
used and the same stem rules are applied to it.

Reads  ../data/sources/administered_text_v9.csv    cid, item, survey, question   (CDR v9 survey metadata)
       ../data/sources/core_items_151.csv          item, prompt, survey, concept_id
Writes ../data/sources/items_text_of_record.csv    item, concept_id, survey, source, administered, prompt,
                                                   stems_removed  (prompt = the text of record)
Prints every item: administered -> text of record, so the result can be reviewed line by line.
Run:   python build_text_of_record.py
"""
import csv
import os
import re

from _common import PROJECT

ADMIN = os.path.join(PROJECT, "data", "sources", "administered_text_v9.csv")
ITEMS = os.path.join(PROJECT, "data", "sources", "core_items_151.csv")
OUT = os.path.join(PROJECT, "data", "sources", "items_text_of_record.csv")   # then build_items_table.py -> data/items_151.csv

LABEL_ONLY_SURVEYS = {"Basics", "Lifestyle", "OverallHealth"}   # CDR holds a label, not the question

# leading stems, longest and most specific first; applied repeatedly
LEADING = [
    r"over the last 2 weeks,?\s+how often have you been bothered by\s+",
    r"in the last month,?\s+how often have you\s+",
    r"in the past 7 days,?\s+how often have you been bothered by\s+",
    r"in your day-to-day life,?\s+how often\s+",
    r"how much you agree or disagree that\s+",
    r"how often (do you|does|are you)\s+",
    r"(over the last 2 weeks|in the last month|during the past 6 months|within the past 12 months|"
    r"in the past 7 days|in the past 6 months|in your entire life|in your life|"
    r"with this definition in mind),?\s+",
    r"(are you|do you|does|do)\s+",
]
# Not removed: "I am someone who" (the BFI-2-XS frame) and "In general,".
TRAILING = [r"\s*would you say( that you)?\s*\.{0,3}\s*$"]


def strip_stems(text):
    s = text.strip()
    removed = []
    changed = True
    while changed:
        changed = False
        for pat in LEADING:
            m = re.match(pat, s, flags=re.I)
            if m:
                removed.append(m.group(0).strip())
                s = s[m.end():].strip()
                changed = True
                break
    for pat in TRAILING:
        m = re.search(pat, s, flags=re.I)
        if m:
            removed.append(m.group(0).strip())
            s = s[:m.start()].strip()
    # sentence-initial capital, so the encoder sees ordinary text
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s, removed


def main():
    items = list(csv.DictReader(open(ITEMS)))
    admin = {r["item"]: r for r in csv.DictReader(open(ADMIN))}
    missing = [r["item"] for r in items if r["item"] not in admin]
    if missing:
        raise SystemExit(f"{len(missing)} items have no administered string: {missing[:5]}")
    rows = []
    for r in items:
        a = admin[r["item"]]["question"]
        if r["survey"] in LABEL_ONLY_SURVEYS:
            source, base = "published question (CDR holds a label)", r["prompt"]
        else:
            source, base = "administered, CDR v9", a
        text, removed = strip_stems(base)
        rows.append({"item": r["item"], "concept_id": r["concept_id"], "survey": r["survey"],
                     "source": source, "administered": a, "prompt": text,
                     "stems_removed": " | ".join(removed)})
        flag = "" if removed else "   (no stem)"
        print(f"{r['item']:28s} {text}{flag}")
        if removed:
            print(f"{'':28s}   removed: {' | '.join(removed)}")
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    n_strip = sum(1 for r in rows if r["stems_removed"])
    print(f"\n{len(rows)} items, {n_strip} with a stem removed, {len(rows) - n_strip} untouched -> {OUT}")


if __name__ == "__main__":
    main()
