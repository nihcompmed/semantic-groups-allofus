#!/usr/bin/env python3
"""collect_conditions_jno.py -- the 16 step zips of the recorded conditions into ONE file for a single download:
~/conditions_jno_<CDR>.zip. It stops and names any zip that is missing. Nothing is recomputed; each zip was already
checked by its own script (no person-level column).

Run (after steps 14 and 18):  python3 collect_conditions_jno.py
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5", "all-mpnet-base-v2", "e5-large-v2", "qwen3-embedding-0.6b"]
STEPS = (["phecode_counts_jno", "condition_tests_jno", "conditions_deciles_jno"]
         + [f"conditions_assoc_jno_{e}" for e in ENCODERS]
         + [f"conditions_assoc_adjusted_jno_{e}" for e in ENCODERS]
         + ["breast_timing_jno"])


def main():
    cdr = os.environ.get("WORKSPACE_CDR", "") or "cdr"
    paths = [os.path.join(HERE, "screen_out", f"{s}_{cdr}.zip") for s in STEPS]
    missing = [os.path.basename(p) for p in paths if not os.path.exists(p)]
    assert not missing, f"missing: {missing}"
    out = os.path.expanduser(f"~/conditions_jno_{cdr}.zip")
    with zipfile.ZipFile(out, "w") as z:
        for p in paths:
            z.write(p, arcname=os.path.basename(p))
    print(f"[collect_conditions_jno] {len(paths)} step zips\n  download -> {out}")


if __name__ == "__main__":
    main()
