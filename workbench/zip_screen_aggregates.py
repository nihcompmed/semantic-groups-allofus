#!/usr/bin/env python3
"""zip_screen_aggregates.py -- zip the SINGLE-ENCODER screen aggregates for copying off the Workbench.

It puts in one zip the aggregates of these scripts, and of the others listed in PATTERNS:

    cutoffs.py           the six published criteria and breadth
    exemplar_encoder.py  the screen, breadth, group and construct layers on one encoder's flagged set
    ceiling.py           is the composition ranking a ranking of people or of rulers
    direction_own.py     |z| against |r| per group, on each encoder's OWN flagged set
    agreement_curves.py  how far the encoders agree on who is far out, as a function of the cut

★★ IT GLOBS BY PREFIX RATHER THAN LISTING FILES. The six encoders each write their own
`exemplar_<enc>_*`, and listing them by hand is how a file goes missing when a name changes.
Adding an encoder or an output needs no edit here.

★★ THE PER-PERSON GUARD IS THE POINT OF THE GLOB, NOT AN AFTERTHOUGHT. Person-level data never leave
the Workbench. Globbing will legitimately meet person-level files (`cutoff_flags.csv` is one), so a
file is EXCLUDED and REPORTED when it carries a person identifier -- a column named `id`,
`person_id`, `participant_id`, `research_id`, `subject_id` or `pid`, or an unnamed first column, which
is what a per-person frame written with its index looks like. There is NO ROW CAP: see the note by
ID_COLS. Nothing is skipped quietly -- every exclusion is printed with its reason, so a missing file is
visible rather than silent. The denylist is belt and braces; the column check catches these anyway.

Run:  python3 zip_screen_aggregates.py
      python3 zip_screen_aggregates.py --cohort k0     (the complete-case screen in screen_out_k0/)
"""
import glob
import json
import os
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COHORT = sys.argv[sys.argv.index("--cohort") + 1] if "--cohort" in sys.argv else "k5"
OUT_DIR = os.path.join(HERE, "screen_out" if COHORT == "k5" else "screen_out_k0")
CDR = os.environ.get("WORKSPACE_CDR", "").split(".")[-1] or "cdr"
# ★★ NO ROW CAP, DELIBERATELY. A row cap would be wrong twice over. It cannot catch what it is for --
# the cohort is ~90,351 and a flagged set ~4,518, so a per-person file on the flagged set passes any cap a
# large aggregate needs. And it refuses legitimate output: a cap the analysis quietly outgrows is silent
# data loss wearing the costume of a safety check. The guard is about WHAT THE FILE IS, not how big it is.
ID_COLS = {"id", "person_id", "participant_id", "research_id", "subject_id", "pid"}

# Globbed in order. A prefix is enough: every one of these scripts names its outputs by prefix.
PATTERNS = [
    "cutoff_summary.csv", "cutoff_summary.json", "breadth_table.csv",   # cutoffs.py (the six criteria)
    "exemplar_*_summary.json", "exemplar_*_breadth.csv",                # exemplar_encoder.py, any encoder
    "exemplar_*groups.csv", "exemplar_*constructs.csv",                 #   incl. the nocriterion_ subset
    "ceiling_*.csv", "ceiling_*.json",                                  # ceiling.py
    "direction_summary_*.csv", "direction_top3_*.csv",                  # direction_own.py (tag `own`)
    "agreement_*.csv", "agreement_*.json", "partial_flag_percentiles.csv",   # agreement_curves.py
    "run_config*.json",                                                 # screen_battery.py, for provenance
    # ---- the covariate, care, sleep, comparator and breadth aggregates
    "table1_covariates.csv", "covariates_answer_map.csv",               # covariates.py (eTables 2 and 3)
    "care_group_rates.csv", "care_group_rates.json",                    # care.py, flagged against the rest
    "care_gradient_covariates_*.csv",                                   # care_covariates.py --gradient
    "care_records_*.csv", "care_records_*.json",                        # care_records.py
    "sleep_by_model*.csv",                                              # sleep_by_model.py
    "comparator_flag_overlap*.csv", "comparator_breadth*.csv",          # comparators.py
    "comparator_leading_constructs*.csv", "comparator_summary*.json",
    "comparator_factors_*.csv", "comparator_units_scale*.csv",
    "matched_breadth*.csv", "breadth_sweep*.csv", "breadth_contrasts*.csv",
    "breadth_exclusive*.csv", "care_exclusive*.csv",
]
# explore_other_data.py writes its aggregates BESIDE THE SCRIPTS, not in screen_out/, next to its
# person-level caches (*_local.csv). These are looked for there, and a *_local.csv is never taken.
HERE_PATTERNS = ["fitbit_record_shape*.csv", "fitbit_shape_check*.csv", "fitbit_personal_by_decile_*.csv",
                 "sleep_*.csv",
                 "wear_study_by_decile*.csv", "wear_study_check*.csv"]
# ★ screen_summary.csv is NOT zipped. It is the consensus sweep across the 90th to 99th
# percentiles, which the paper does not use, and screen_battery.py writes it with no disclosure
# rule. agreement_curves.py carries the same agreement, under the policy.

# ★ THE LAST-LINE COUNT CHECK. Every script above applies the dissemination policy
# through disclosure.py. This refuses a file anyway if a participant-count column holds a number from
# 1 to 20, so a script that forgets the rule is caught here rather than after the download. A column
# counts as a participant count when it is named n, n_*, or intersection, except the structural ones
# below, which count items, groups or the size of a top-k set chosen by us.
COUNT_COL_EXEMPT = {"n_items", "n_core_items", "n_semantic_groups", "n_groups_on_direction",
                    "n_levels_used", "n_A", "n_B", "n_criteria", "n_ace_categories",
                    "n_criterion_items", "n_groups", "n_constructs",
                    # model and design sizes, not people: EM iterations, permutations, categories, units
                    "n_iter", "n_perm", "n_categories", "n_units_scale", "n_levels"}

# Known person-level files. The column check below would refuse these regardless; naming them keeps
# the printed reason exact rather than generic.
DENY = {"cutoff_flags.csv", "attribution_component_shares_outliers.csv",
        "attribution_component_shares_invisible.csv", "attribution_per_person_outliers.csv",
        "attribution_per_person_invisible.csv", "consensus_outliers.csv",
        "leading_group_per_person_outliers.csv", "leading_group_per_person_invisible.csv"}


def verdict(path):
    """(ok, reason). A file leaves only when it is demonstrably an aggregate."""
    name = os.path.basename(path)
    if name in DENY:
        return False, "person-level by name"
    if name.endswith(".json"):
        bad = []

        def walk(o, key=""):
            if isinstance(o, dict):
                for k2, v2 in o.items():
                    walk(v2, str(k2))
            elif isinstance(o, list):
                for v2 in o:
                    walk(v2, key)
            elif (key == "n" or key.startswith("n_")) and key not in COUNT_COL_EXEMPT \
                    and isinstance(o, (int, float)) and not isinstance(o, bool) and 0 < o <= 20:
                bad.append(key)
        try:
            walk(json.load(open(path)))
        except Exception as e:
            return False, f"unreadable ({type(e).__name__})"
        if bad:
            return False, f"key(s) {sorted(set(bad))} hold a count of 1 to 20 -- the policy was not applied"
        return True, ""
    try:
        d = pd.read_csv(path)
    except Exception as e:                                   # unreadable is not bundled
        return False, f"unreadable ({type(e).__name__})"
    hit = ID_COLS.intersection(str(c).strip().lower() for c in d.columns)
    if hit:
        return False, f"carries a person column ({', '.join(sorted(hit))})"
    # A per-person frame written with its index has a blank or `Unnamed: 0` first column.
    unnamed = [str(c) for c in d.columns if not str(c).strip() or str(c).startswith("Unnamed:")]
    if unnamed:
        return False, f"unnamed column {unnamed[0]!r} -- an index written out, possibly person ids"
    for c in d.columns:
        cs = str(c)
        if (cs == "n" or cs.startswith("n_") or cs == "intersection") and cs not in COUNT_COL_EXEMPT:
            v = pd.to_numeric(d[c], errors="coerce")
            bad = int(((v > 0) & (v <= 20)).sum())
            if bad:
                return False, f"column {cs!r} holds {bad} count(s) of 1 to 20 -- the policy was not applied"
    return True, ""


def main():
    if not os.path.isdir(OUT_DIR):
        raise SystemExit(f"{OUT_DIR} does not exist -- run the screen first")
    seen, entries, excluded = set(), [], []
    todo = [(OUT_DIR, pat) for pat in PATTERNS] + ([(HERE, pat) for pat in HERE_PATTERNS] if COHORT == "k5" else [])
    for base, pat in todo:
        for p in sorted(glob.glob(os.path.join(base, pat))):
            if p.endswith("_local.csv"):
                excluded.append((p, os.path.basename(p), "person-level cache (*_local.csv)"))
                continue
            if p in seen:
                continue
            seen.add(p)
            ok, why = verdict(p)
            (entries if ok else excluded).append((p, os.path.basename(p), why))

    if excluded:
        print("EXCLUDED, and why:")
        for _, name, why in excluded:
            print(f"  {name:52s} {why}")
    if not entries:
        raise SystemExit("nothing to bundle yet -- no aggregate matched the patterns")

    zpath = os.path.join(OUT_DIR, f"screen_aggregates_{COHORT}_{CDR}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p, name, _ in entries:
            z.write(p, arcname=name)

    print(f"\nBUNDLED {len(entries)} files, none with a row per person:")
    for _, name, _ in entries:
        print(f"  {name}")
    print(f"\n-> {zpath}")
    print(f"download it from the Jupyter file browser and unpack into "
          f"manuscript/outputs/workbench/{CDR}/")

    # A missing headline file is worth saying out loud, since the glob cannot tell "not written yet"
    # from "never going to be written".
    got = {n for _, n, _ in entries}
    for want, who in [("cutoff_summary.csv", "cutoffs.py"),
                      ("ceiling_summary.json", "ceiling.py"),
                      ("agreement_summary.json", "agreement_curves.py")]:
        if want not in got:
            print(f"  (note: {want} is absent -- {who} has not run in this directory)")
    if not any(n.startswith("exemplar_") for n in got):
        print("  (note: no exemplar_* files -- exemplar_encoder.py has not run in this directory)")


if __name__ == "__main__":
    main()
