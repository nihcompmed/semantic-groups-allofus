#!/usr/bin/env python
"""construct_labels.py -- the one place where a construct's internal label becomes the name a reader sees.

WHY. The 30 construct labels in `data/items_151.csv` are names for groups of items. Seven of them are also
the names of psychiatric diagnoses, and in a psychiatry journal a diagnosis attached to a person reads as a
clinical statement. The 4 items behind "Psychosis" are the World Mental Health CIDI screen's questions about
unusual experiences; the 2 behind "Mania" are 2 questions; "Adult ADHD" is a screener. A reader who meets
"512 participants led by psychosis" will read 512 people with a psychotic disorder, which the data do not
support. Three further labels are imprecise (the violence items are partner and sexual violence, the
self-rated items are the PROMIS Global MENTAL Health items) or are spelled against AMA style.

HOW. Nothing in the data changes. `construct` stays exactly as it is in `items_151.csv` and in every
Workbench aggregate, so every output already on disk stays valid and every join still works. This module maps
that label to the display name at the moment a table, a figure or the manuscript text shows it. The
manuscript prose uses the same names, so text and tables agree.

USE.  from construct_labels import display
      display("Psychosis")   -> "Psychotic experiences"

An unknown label raises, so a typo or a new construct is caught when the table is built rather than in proof.
"""

# internal label (data) -> display name (reader). Unchanged entries are listed so the map is the full record
# of the 30 constructs and a missing one is an error rather than a silent pass-through.
DISPLAY = {
    # --- renamed: symptom-screen item groups that share a name with a diagnosis ----------------------------
    "Psychosis": "Psychotic experiences",                       # 4 WMH-CIDI and MHQ screen items
    "Mania": "Manic experiences",                               # 2 MHQ items
    "Depression": "Depressive symptoms",                        # PHQ-9 items 1-8 and 2 MHQ items
    "Generalized anxiety": "Anxiety symptoms",                  # GAD-7 and 1 single item
    "Panic & phobia": "Panic and phobic symptoms",              # 3 WMH-CIDI screener items
    "Adult ADHD": "Adult ADHD symptoms",                        # ASRS v1.1 Part A screener
    "Suicidality & self-harm": "Suicidal ideation and self-harm",
    # --- renamed: precision and AMA style -----------------------------------------------------------------
    "Interpersonal & sexual violence": "Intimate partner or sexual violence",   # MHQ traumatic events section
    "Healthcare discrimination": "Health care discrimination",  # AMA spells health care as 2 words
    "Self-rated health": "Self-rated mental health",            # PROMIS Global Mental Health items
    # the documented name of the mMOS-SS construct; it asserts no direction.
    "Perceived social support": "Perceived social support",
    "Subjective wellbeing": "Subjective well-being",            # AMA style
    "Neighborhood disorder & safety": "Neighborhood disorder and safety",
    # --- unchanged ----------------------------------------------------------------------------------------
    "Body-focused repetitive behavior": "Body-focused repetitive behavior",
    "Childhood adversity": "Childhood adversity",
    "Daily spiritual experience": "Daily spiritual experience",
    "Everyday discrimination": "Everyday discrimination",
    "Food insecurity": "Food insecurity",
    "Housing instability": "Housing instability",
    "Life-threatening events": "Life-threatening events",
    "Loneliness": "Loneliness",
    "Neighborhood social cohesion": "Neighborhood social cohesion",
    "Perceived stress": "Perceived stress",
    "Religious participation": "Religious participation",
    "Substance use": "Substance use",
    "Agreeableness": "Agreeableness",
    "Conscientiousness": "Conscientiousness",
    "Extraversion": "Extraversion",
    "Negative emotionality": "Negative emotionality",
    "Open-mindedness": "Open-mindedness",
}

# Every printed construct name in the paper (text, Figures 1, 3, 4, eFigure 3, eTables 1, 4, 8, 11, 12) comes from
# DISPLAY. Data files (CSV, Supplement 2, Workbench downloads) keep the labels of items_151.csv.
def display_profile(profile, strict=True):
    """A construct profile as components.csv writes it ("Label:n; Label:n") with every label mapped to its display
    name, in the same format."""
    if not isinstance(profile, str) or not profile.strip():
        return profile
    parts = []
    for p in profile.split("; "):
        name, _, n = p.rpartition(":")
        parts.append(f"{display(name, strict=strict)}:{n}" if name else display(p, strict=strict))
    return "; ".join(parts)


def display_listed(text, strict=True):
    """A printed list "Label n, Label n" (how the eTables write a profile) with every label mapped."""
    out = []
    for p in str(text).split(", "):
        name, _, n = p.rpartition(" ")
        out.append(f"{display(name, strict=strict)} {n}" if name and n.isdigit() else display(p, strict=strict))
    return ", ".join(out)


# Rows that are not constructs but appear in the same column of some aggregates.
PASS_THROUGH = {"other", "Other", "Other (pooled)", "All", "Cohort", "Rest of cohort"}


def display(label, strict=True):
    """The name a reader sees for an internal construct label."""
    s = str(label).strip()
    if s in DISPLAY:
        return DISPLAY[s]
    if s in PASS_THROUGH or not strict:
        return s
    raise KeyError(f"construct label not in construct_labels.DISPLAY: {label!r}. "
                   f"Add it there rather than renaming it in a generator.")


def display_series(s, strict=True):
    """Map a pandas Series of internal labels to display names."""
    return s.map(lambda v: display(v, strict=strict))


# --- short forms, for annotating a figure where the display name will not fit -----------------------
# For a figure where 23 to 27 points are labeled on one panel. These are
# for PLOT LABELS ONLY. Prose and tables use display() so that text and tables agree, and a caption
# spells out any short form a reader could misread.
SHORT = {
    "Health care discrimination": "HC discrimination",
    "Everyday discrimination": "Everyday discrim.",
    "Neighborhood disorder and safety": "Nbhd disorder",
    "Neighborhood social cohesion": "Nbhd cohesion",
    "Psychotic experiences": "Psychotic exp.",
    "Manic experiences": "Manic exp.",
    "Depressive symptoms": "Depressive",
    "Anxiety symptoms": "Anxiety",
    "Panic and phobic symptoms": "Panic/phobia",
    "Adult ADHD symptoms": "ADHD",
    "Suicidal ideation and self-harm": "Suicidal ideation",
    "Intimate partner or sexual violence": "Partner/sexual viol.",
    "Body-focused repetitive behavior": "BFRB",
    "Childhood adversity": "Childhood adv.",
    "Daily spiritual experience": "Spiritual exp.",
    "Subjective well-being": "Well-being",
    "Self-rated mental health": "Self-rated MH",
    "Perceived stress": "Stress",
    "Life-threatening events": "Life-threat. events",
    "Religious participation": "Religious partic.",
    "Negative emotionality": "Neg. emotionality",
    "Open-mindedness": "Open-mind.",
    "Conscientiousness": "Conscient.",
    "Extraversion": "Extraversion",
    "Agreeableness": "Agreeableness",
}


_DISPLAY_NAMES = set(DISPLAY.values())


def short(label, strict=True):
    """A plot-label form of a construct name. Accepts an internal label or an already-displayed name,
    since some generated tables store display names."""
    s = str(label).strip()
    d = s if s in _DISPLAY_NAMES else display(s, strict=strict)
    return SHORT.get(d, d)


if __name__ == "__main__":
    changed = [(k, v) for k, v in DISPLAY.items() if k != v]
    print(f"{len(DISPLAY)} constructs, {len(changed)} renamed for display:\n")
    for k, v in changed:
        print(f"  {k:35s} -> {v}")


# The questionnaires whose item wording is reprinted (eTables 1 and 12, Supplement 2, Figure 1): their owners' terms
# permit it (eMethods 1). PHQ-9 and GAD-7 are in the public domain, the ACE module is a US government work, the mMOS-SS
# is free for noncommercial use, and the stand-alone questions are All of Us's own. Every other item is shown by its
# item code, under which the All of Us Survey Codebooks give its wording.
REPRINT = {"PHQ-9", "GAD-7", "ACE-BRFSS", "MOS-SS", "single item"}
# the public code repository is MIT licensed, which allows commercial use; RAND's terms for the mMOS-SS do not
REPRINT_REPO = REPRINT - {"MOS-SS"}


def reprintable(instrument_code, repo=False):
    return instrument_code in (REPRINT_REPO if repo else REPRINT)
