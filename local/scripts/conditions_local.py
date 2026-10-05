"""conditions_local.py -- the 8 recorded conditions of record for the LOCAL generators (figures and eTables), the mirror
of workbench/conditions_jno.py. Keys are the downloads' condition keys; labels are the supplement's names. Every local
generator that draws the conditions (eFigure 2, eTable 7, Figure 4, eFigure 3, eTables 9 to 11) takes the list, the
order and the labels from here, so the Workbench and the paper cannot drift apart.
"""
# key, label (supplement), short label (figures), group
CONDITIONS = [
    ("gad", "Generalized anxiety disorder", "GAD", "mental"),
    ("adhd", "Attention-deficit/hyperactivity disorder", "ADHD", "mental"),
    ("ptsd", "Posttraumatic stress disorder", "PTSD", "mental"),
    ("panic", "Panic disorder", "Panic disorder", "mental"),
    ("hyperchol", "Hypercholesterolemia", "High cholesterol", "physical"),
    ("asthma", "Asthma", "Asthma", "physical"),
    ("breast", "Breast cancer (women)", "Breast cancer", "physical"),
    ("prostate", "Prostate cancer (men)", "Prostate cancer", "physical"),
]
KEYS = [c[0] for c in CONDITIONS]
LABEL = {c[0]: c[1] for c in CONDITIONS}
SHORT = {c[0]: c[2] for c in CONDITIONS}
MENTAL = [c[0] for c in CONDITIONS if c[3] == "mental"]
PHYSICAL = [c[0] for c in CONDITIONS if c[3] == "physical"]
TESTED = {"hyperchol": "a cholesterol measurement or lipid panel", "breast": "a mammogram", "prostate": "a blood PSA test"}
