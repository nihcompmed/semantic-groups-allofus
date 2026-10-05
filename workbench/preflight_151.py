#!/usr/bin/env python3
"""preflight_151.py -- check the 151-item folder before anything expensive runs.

WHY. The response build is a long BigQuery pull. Every mistake this script looks for is one that
would otherwise surface after that pull, or worse, halfway down the pipeline. It reads nothing from
the Repository and writes nothing; it only looks at the files beside it and at the environment.

THE ONE THAT MATTERS. Every script resolves the battery and the responses folder through wb_config.py,
which puts the build in responses_out_<stem of the item table> beside the scripts. Nothing is passed on
the command line and nothing is exported, so there is one folder by construction. What can still go
wrong is the input side: two batteries in one folder, an incomplete basis package, embeddings that do
not match the item table, or a script left behind by an extraction without overwrite. This checks those.
It also reports any script of the RETIRED list (scripts that are not part of this analysis) found beside
these, and fails when disclosure.py is missing or cutoffs.py does not have six criteria with ACE on 7 categories.

Run:  python3 preflight_151.py
"""
import glob
import importlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5", "all-mpnet-base-v2",
            "e5-large-v2", "qwen3-embedding-0.6b"]
# This folder holds the single-encoder scripts only. A script of the RETIRED list found beside these is
# reported, since an `unzip -o` over another folder leaves such scripts behind and they would still run
# against these outputs.
WB_CONFIG_USERS = ["screen_battery.py", "cutoffs.py", "exemplar_encoder.py", "semantic_groups.py",
                   "direction_own.py", "ceiling.py", "fig_directions.py",
                   "care.py", "covariates.py", "care_covariates.py", "care_records.py",
                   "explore_other_data.py", "sleep_by_model.py", "mca_basis.py", "comparators.py",
                   "matched_breadth.py", "breadth_sweep.py", "care_exclusive.py"]
RETIRED = ["attribution.py", "leading_group.py", "direction_summary.py", "outlier_profile.py",
           "construct_distribution.py", "severity_baseline.py", "care_burden_fixed.py",
           "comparator_profile.py", "care_groups.py", "joint_only.py"]

problems = []
notes = []


def fail(msg):
    problems.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg):
    print(f"  ok    {msg}")


def note(msg):
    notes.append(msg)
    print(f"  note  {msg}")


print("\n1. the battery beside the scripts")
items = sorted(glob.glob(os.path.join(HERE, "*items*.csv")))
emaps = sorted(glob.glob(os.path.join(HERE, "*encoding_map*.csv")))
if len(items) != 1:
    fail(f"{len(items)} item tables match *items*.csv ({', '.join(os.path.basename(p) for p in items) or 'none'}); "
         f"every script that discovers the battery will refuse. Keep one battery per folder.")
    items_csv = None
else:
    items_csv = items[0]
    ok(f"item table {os.path.basename(items_csv)}")
if len(emaps) != 1:
    fail(f"{len(emaps)} encoding maps match *encoding_map*.csv; build_responses_v9.py will refuse.")
else:
    ok(f"encoding map {os.path.basename(emaps[0])}")

n_items = None
try:
    import pandas as pd
except ImportError:                                   # reported in section 5; keep going so the rest still runs
    pd = None
    fail("pandas is not installed in this environment")
if items_csv and pd is not None:
    T = pd.read_csv(items_csv)
    n_items = len(T)
    missing_cols = [c for c in ("item", "construct") if c not in T.columns]
    if missing_cols:
        fail(f"{os.path.basename(items_csv)} has no column {missing_cols}")
    else:
        ok(f"{n_items} items, {T['construct'].nunique()} constructs")

print("\n2. the basis package")
basis = os.path.join(HERE, "basis")
emb_dir = os.path.join(basis, "item_embeddings")
if not os.path.isdir(basis):
    fail("no basis/ folder beside the scripts")
else:
    for e in ENCODERS:
        if not os.path.exists(os.path.join(basis, f"{e}.npz")):
            fail(f"basis/{e}.npz missing")
    for f in ("components.csv", "manifest.csv"):
        if not os.path.exists(os.path.join(basis, f)):
            fail(f"basis/{f} missing")
    prefixes = set()
    for p in glob.glob(os.path.join(emb_dir, "*.npz")):
        b = os.path.basename(p)[:-4]
        for e in ENCODERS:
            if b.endswith("_" + e):
                prefixes.add(b[: -len(e) - 1])
    if len(prefixes) != 1:
        fail(f"basis/item_embeddings/ holds {len(prefixes)} embedding prefixes ({', '.join(sorted(prefixes)) or 'none'}); "
             f"wb_config.py cannot choose. Keep one battery's embeddings per folder.")
    else:
        prefix = prefixes.pop()
        have = [e for e in ENCODERS if os.path.exists(os.path.join(emb_dir, f"{prefix}_{e}.npz"))]
        if len(have) != 6:
            fail(f"embeddings present for {len(have)} of 6 encoders under prefix {prefix}")
        else:
            ok(f"6 encoders under prefix {prefix}")
        if n_items:
            import numpy as np                        # present wherever pandas is
            for e in ENCODERS[:1] + ENCODERS[-1:]:
                z = np.load(os.path.join(emb_dir, f"{prefix}_{e}.npz"), allow_pickle=True)
                key = "E" if "E" in z else list(z.keys())[0]
                if z[key].shape[0] != n_items:
                    fail(f"{prefix}_{e}.npz has {z[key].shape[0]} rows, the item table has {n_items}")
            ok(f"embedding row count matches the item table ({n_items})")

print("\n3. the responses folder")
stem = os.path.basename(items_csv).replace(".csv", "") if items_csv else "?"
wb_dir = os.environ.get("RESPONSES_DIR") or os.path.join(HERE, f"responses_out_{stem}")
leftover = set()
for s in WB_CONFIG_USERS:
    p = os.path.join(HERE, s)
    if not os.path.exists(p):
        continue
    for m in re.finditer(r'"(/home/jupyter/[^"]*)"', open(p).read()):
        leftover.add((s, m.group(1)))
if leftover:
    fail("a script still carries an absolute responses path: "
         + "; ".join(f"{s} -> {d}" for s, d in sorted(leftover)))
else:
    ok("no script carries an absolute responses path")
print(f"        the build writes and everything reads  {wb_dir}")
if os.environ.get("RESPONSES_DIR"):
    note("RESPONSES_DIR is set in this shell and overrides the rule above; unset it unless you mean it")
if os.path.exists(os.path.join(wb_dir, "responses_k5.csv")):
    ok("responses_k5.csv is already there (the build has run)")
else:
    # A build may have run under another copy of these scripts with a different folder rule.
    # Saying "as expected before the build" would be wrong then, so look before saying it.
    strays = []
    for root in {HERE, os.path.dirname(HERE), "/home/jupyter", os.path.expanduser("~")}:
        for cand in sorted(glob.glob(os.path.join(root, "responses_out*"))):
            f = os.path.join(cand, "responses_k5.csv")
            if os.path.abspath(cand) != os.path.abspath(wb_dir) and os.path.exists(f):
                n = sum(1 for _ in open(f)) - 1
                strays.append((cand, n))
    if strays:
        fail("responses_k5.csv is NOT at the resolved path, but a build exists elsewhere:\n"
             + "\n".join(f"        {c}   {n:,} participants" for c, n in strays)
             + "\n        Point every script at the one the screen used:\n"
             + f"        export RESPONSES_DIR={strays[0][0]}")
    elif os.path.isdir(os.path.join(HERE, "screen_out")):
        fail("responses_k5.csv is nowhere, yet screen_out/ exists. The screen ran against a response "
             "file that is now missing, so nothing downstream can be trusted to match it.")
    else:
        note("responses_k5.csv is not there yet, as expected before the build")

print("\n4. no stale copies of the scripts")
stale = len(problems)
for s in WB_CONFIG_USERS:
    p = os.path.join(HERE, s)
    if not os.path.exists(p):
        fail(f"{s} missing")
    elif "wb_config" not in open(p).read():
        fail(f"{s} is an outdated copy (it does not import wb_config). Copy this folder again.")
if len(problems) == stale:
    ok(f"all {len(WB_CONFIG_USERS)} battery-agnostic scripts import wb_config")
if not os.path.exists(os.path.join(HERE, "disclosure.py")):
    fail("disclosure.py missing: every script whose output leaves the Workbench imports it to apply "
         "the All of Us dissemination policy")
left = [s for s in RETIRED if os.path.exists(os.path.join(HERE, s))]
if left:
    note(f"scripts not used for the paper are in this folder: {', '.join(left)}. "
         f"No step of the README reads their outputs.")
if os.path.exists(os.path.join(HERE, "cutoffs.py")) and "len(ACE_CATEGORIES) == 7" not in open(os.path.join(HERE, "cutoffs.py")).read():
    fail("cutoffs.py is not the six-criterion version with ACE on 7 categories")

print("\n5. packages")
for mod, why in [("numpy", ""), ("pandas", ""), ("sklearn", ""), ("scipy", ""),
                 ("statsmodels", ""), ("matplotlib", ""), ("google.cloud.bigquery", "")]:
    try:
        importlib.import_module(mod)
        ok(mod)
    except ImportError:
        fail(f"{mod} is not installed")
try:
    importlib.import_module("umap")
    ok("umap-learn")
except ImportError:
    note("umap-learn is not installed. No step of the README needs it (it served UMAP "
         "figures that are not in the paper)")

print("\n6. environment")
for var in ("WORKSPACE_CDR", "GOOGLE_PROJECT"):
    if os.environ.get(var):
        ok(f"{var}={os.environ[var]}")
    else:
        fail(f"{var} is not set; the Repository queries will fail")

print("\n" + "=" * 78)
if problems:
    print(f"{len(problems)} problem(s) to fix before running anything:")
    for p in problems:
        print(f"  - {p.splitlines()[0]}")
    sys.exit(1)
print("preflight clean." + (f" {len(notes)} note(s) above." if notes else ""))
