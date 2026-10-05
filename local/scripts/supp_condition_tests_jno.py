#!/usr/bin/env python
"""supp_condition_tests_jno.py -- Supplement 2 file condition_test_codes.csv: every code that marks a participant as
tested for hypercholesterolemia, breast cancer or prostate cancer, as found in the CDR's concept table. eMethods 3
(Recorded conditions) refers to this list.

LOCAL. Reads the download condition_tests_jno/condition_tests_concepts.csv (workbench/conditions_jno.py) and writes
../outputs/supplementary_data_jno/condition_test_codes.csv with the condition each test serves.

Run:  python supp_condition_tests_jno.py
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "outputs", "workbench", "R2025Q4R6", "condition_tests_jno", "condition_tests_concepts.csv")
OUT = os.path.join(HERE, "..", "outputs", "supplementary_data_jno", "condition_test_codes.csv")
SERVES = {"lipid": "Hypercholesterolemia (cholesterol measurement or lipid panel)",
          "breast": "Breast cancer, women (mammogram or breast-screening visit)",
          "prostate": "Prostate cancer, men (blood prostate-specific antigen test or prostate-screening visit)"}


def main():
    C = pd.read_csv(SRC, dtype=str)
    assert set(C["kind"]) == set(SERVES), f"unexpected test kinds: {sorted(set(C['kind']))}"
    C.insert(0, "condition_and_test", C["kind"].map(SERVES))
    C = C.rename(columns={"vocabulary_id": "vocabulary", "concept_code": "code", "concept_name": "name"})
    C = C[["condition_and_test", "vocabulary", "code", "name"]].sort_values(["condition_and_test", "vocabulary", "code"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    C.to_csv(OUT, index=False)
    print("wrote", OUT, len(C), "codes:", C.groupby("condition_and_test").size().to_dict())


if __name__ == "__main__":
    main()
