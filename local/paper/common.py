"""common.py -- the folders, names and labels every script in paper/ shares.

Folders
  local/outputs/workbench/R2025Q4R6/   the Researcher Workbench downloads (WB), one folder per download
  local/outputs/                       the semantic groups (basis/, cores/, k_selection/, random_starts/, xenc/, tables/)
  local/data/                          the item table, the answer coding, the item embeddings
  elife/figures, elife/tables, elife/supplementary   what these scripts write (beside local/)

Names. A construct keeps its internal label in every data file (items_151.csv, the downloads); DISPLAY maps it to the
name a reader sees, at the moment a figure or table prints it. Seven labels are also names of diagnoses ("Psychosis" is
4 screening questions about unusual experiences), so they are printed as symptom or experience names; three more are
renamed for precision or style. An unknown label raises, so a new construct is caught when the table is built.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.dirname(HERE)
ROOT = os.path.dirname(LOCAL)
DATA = os.path.join(LOCAL, "data")
OUTPUTS = os.path.join(LOCAL, "outputs")
WB = os.path.join(OUTPUTS, "workbench", "R2025Q4R6")
ITEMS_CSV = os.path.join(DATA, "items_151.csv")
SOURCES_CSV = os.path.join(DATA, "standalone_sources.csv")   # the study each stand-alone question was taken from
TABLE_DATA = os.path.join(OUTPUTS, "tables")                  # the appendix tables' values as CSV
FIGURES = os.path.join(ROOT, "elife", "figures")
TABLES = os.path.join(ROOT, "elife", "tables")
SUPPLEMENTARY = os.path.join(ROOT, "elife", "supplementary")

PRIMARY = "gte-large-en-v1.5"
ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5", "all-mpnet-base-v2", "e5-large-v2",
            "qwen3-embedding-0.6b"]

# the 8 recorded conditions: key (the downloads'), label (tables), short label (figures), group
CONDITIONS = [
    ("gad", "Generalized anxiety disorder", "GAD", "mental"),
    ("adhd", "Attention-deficit/hyperactivity disorder", "ADHD", "mental"),
    ("ptsd", "Posttraumatic stress disorder", "PTSD", "mental"),
    ("panic", "Panic disorder", "Panic disorder", "mental"),
    ("hyperchol", "Hypercholesterolemia", "Hypercholesterolemia", "physical"),
    ("asthma", "Asthma", "Asthma", "physical"),
    ("breast", "Breast cancer (women)", "Breast cancer", "physical"),
    ("prostate", "Prostate cancer (men)", "Prostate cancer", "physical"),
]
KEYS = [c[0] for c in CONDITIONS]
LABEL = {c[0]: c[1] for c in CONDITIONS}
SHORT_CONDITION = {c[0]: c[2] for c in CONDITIONS}
MENTAL = [c[0] for c in CONDITIONS if c[3] == "mental"]
PHYSICAL = [c[0] for c in CONDITIONS if c[3] == "physical"]
TESTED = {"hyperchol": "a cholesterol measurement or lipid panel", "breast": "a mammogram", "prostate": "a blood PSA test"}

DISPLAY = {
    "Psychosis": "Psychotic experiences",
    "Mania": "Manic experiences",
    "Depression": "Depressive symptoms",
    "Generalized anxiety": "Anxiety symptoms",
    "Panic & phobia": "Panic and phobic symptoms",
    "Adult ADHD": "Adult ADHD symptoms",
    "Suicidality & self-harm": "Suicidal ideation and self-harm",
    "Interpersonal & sexual violence": "Intimate partner or sexual violence",
    "Healthcare discrimination": "Health care discrimination",
    "Self-rated health": "Self-rated mental health",
    "Perceived social support": "Perceived social support",
    "Subjective wellbeing": "Subjective well-being",
    "Neighborhood disorder & safety": "Neighborhood disorder and safety",
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
PASS_THROUGH = {"other", "Other", "Other (pooled)", "All", "Cohort", "Rest of cohort"}


def display(label, strict=True):
    """The name a reader sees for an internal construct label."""
    s = str(label).strip()
    if s in DISPLAY:
        return DISPLAY[s]
    if s in PASS_THROUGH or not strict:
        return s
    raise KeyError(f"construct label not in common.DISPLAY: {label!r}")


def display_profile(profile, strict=True):
    """A construct profile ("Label:n; Label:n") with every label mapped to its display name, in the same format."""
    if not isinstance(profile, str) or not profile.strip():
        return profile
    parts = []
    for p in profile.split("; "):
        name, _, n = p.rpartition(":")
        parts.append(f"{display(name, strict=strict)}:{n}" if name else display(p, strict=strict))
    return "; ".join(parts)


def display_listed(text, strict=True):
    """A printed list "Label n, Label n" with every label mapped."""
    out = []
    for p in str(text).split(", "):
        name, _, n = p.rpartition(" ")
        out.append(f"{display(name, strict=strict)} {n}" if name and n.isdigit() else display(p, strict=strict))
    return ", ".join(out)


def display_series(s, strict=True):
    return s.map(lambda v: display(v, strict=strict))


# The questionnaires whose item wording may be reprinted: PHQ-9 and GAD-7 are in the public domain, the ACE module is a
# US government work, the mMOS-SS is free for noncommercial use, and the stand-alone questions are All of Us's own.
# Every other item is shown by its item code. The repository is MIT licensed, which allows commercial use, so it does
# not reprint the mMOS-SS.
REPRINT = {"PHQ-9", "GAD-7", "ACE-BRFSS", "MOS-SS", "single item"}
REPRINT_REPO = REPRINT - {"MOS-SS"}


def reprintable(instrument_code, repo=False):
    return instrument_code in (REPRINT_REPO if repo else REPRINT)


def esc(s):
    """Text for a LaTeX table cell."""
    return (str(s).replace("\\", "").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")
            .replace("#", "\\#").replace("_", "\\_"))


def write_table(name, lines, script):
    """Write elife/tables/<name>.tex, headed by the script that wrote it."""
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, f"{name}.tex")
    open(path, "w").write("\n".join([f"% generated by paper/{script} -- do not edit by hand"] + lines) + "\n")
    print("wrote", path)


def save(fig, name):
    """Write a figure to elife/figures/<name>.pdf and .png (300 dpi)."""
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"), dpi=300)
    print("wrote", os.path.join(FIGURES, f"{name}.png"))
