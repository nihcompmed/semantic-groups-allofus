#!/usr/bin/env python3
"""zip_cutoffs.py -- package cutoffs.py's summaries of the 6 screening thresholds for download (eTable 5: the 6
thresholds with the number evaluable and met).

cutoffs.py writes these into screen_out/ but no download zip. Nothing is recomputed here. The script checks that the
files hold the 6 thresholds (criterion 6 is suicidal ideation or attempt) and carry no person-level column, then zips
them.

INPUTS   screen_out/cutoff_summary.csv    per threshold: items, evaluable and met in the cohort (and in gte's flagged set)
         screen_out/cutoff_summary.json   the counts, for the record
         screen_out/breadth_table.csv     the number of thresholds met, 0 to 6
OUTPUT   screen_out/cutoffs_<CDR>.zip

Run:  python3 zip_cutoffs.py
"""
import json
import os
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sleep_device_deciles import pii_check                                 # noqa: E402  the person-level check

OUT_DIR = os.path.join(HERE, "screen_out")
CDR = os.environ.get("WORKSPACE_CDR", "")
FILES = ["cutoff_summary.csv", "cutoff_summary.json", "breadth_table.csv"]


def main():
    for f in FILES:
        assert os.path.exists(os.path.join(OUT_DIR, f)), f"screen_out/{f} is missing: run `python3 cutoffs.py` first"
    S = pd.read_csv(os.path.join(OUT_DIR, "cutoff_summary.csv"))
    J = json.load(open(os.path.join(OUT_DIR, "cutoff_summary.json")))
    names = set(S["criterion"])
    assert "Suicidal ideation or attempt" in names and int(J.get("n_criteria", 0)) == 6, \
        "cutoff_summary is not the six-threshold version (criterion 6 = ideation or attempt): rerun `python3 cutoffs.py`"
    for f in FILES:
        ok, why = pii_check(os.path.join(OUT_DIR, f))
        assert ok, f"{f}: {why}"
    print(S.to_string(index=False))
    zpath = os.path.join(OUT_DIR, f"cutoffs_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(os.path.join(OUT_DIR, f), arcname=f"cutoffs/{f}")
    print(f"\n[zip_cutoffs] download -> {zpath}")


if __name__ == "__main__":
    main()
