#!/usr/bin/env python3
"""wb_config.py -- the one place the Workbench scripts learn which battery they are working on.

There are no defaults that name a battery. Each setting is resolved as: command-line flag, then
environment variable, then discovery from what is actually beside the scripts; when discovery is
ambiguous the script stops and lists the candidates rather than guess.

    items table      --items <csv>           BUILD_ITEMS      the one *items*.csv beside the scripts
    basis package    --basis <dir>           BASIS_DIR        basis/ beside the scripts
    embedding prefix --emb-prefix <prefix>   EMB_PREFIX       the one prefix in basis/item_embeddings/
    responses folder --responses-dir <dir>   RESPONSES_DIR    responses_out_<stem of the items table>, BESIDE
                                                              THE SCRIPTS (the same rule build_responses_v9.py
                                                              uses for --out, so the build and everything that
                                                              reads it agree without being told)
    cohort           --cohort k5|k0                           k5 = cohort of record, k0 = complete cases
Everything reads:  from wb_config import CFG
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENCODERS = ["gte-large-en-v1.5", "bge-m3", "bge-large-en-v1.5", "all-mpnet-base-v2", "e5-large-v2", "qwen3-embedding-0.6b"]


def _arg(flag, env, default=None):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return os.environ.get(env) or default


def _only(hits, what, hint):
    hits = sorted(set(hits))
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"cannot tell which {what} to use: {len(hits)} candidates ({', '.join(hits) or 'none'}). {hint}")


class _Cfg:
    def __init__(self):
        self.here = HERE
        self.items_csv = _arg("--items", "BUILD_ITEMS") or _only(
            glob.glob(os.path.join(HERE, "*items*.csv")), "item table", "Pass --items <csv>.")
        self.stem = os.path.basename(self.items_csv).replace(".csv", "")
        self.basis_dir = _arg("--basis", "BASIS_DIR") or os.path.join(HERE, "basis")
        self.emb_dir = os.path.join(self.basis_dir, "item_embeddings")
        found = []
        for p in glob.glob(os.path.join(self.emb_dir, "*.npz")):
            b = os.path.basename(p)[:-4]
            for e in ENCODERS:
                if b.endswith("_" + e):
                    found.append(b[: -len(e) - 1])
        self.emb_prefix = _arg("--emb-prefix", "EMB_PREFIX") or _only(
            found, "embedding prefix in basis/item_embeddings/", "Pass --emb-prefix <prefix>.")
        self.cohort = _arg("--cohort", "COHORT", "k5")
        assert self.cohort in ("k5", "k0"), f"--cohort must be k5 (record) or k0 (complete cases), got {self.cohort}"
        self.responses_dir = _arg("--responses-dir", "RESPONSES_DIR") or os.path.join(HERE, f"responses_out_{self.stem}")
        self.responses_csv = os.path.join(self.responses_dir, f"responses_{self.cohort}.csv")
        self.weights_csv = os.path.join(self.responses_dir, f"weights_{self.cohort}.csv")
        self.mask_csv = os.path.join(self.responses_dir, "imputed_mask_k5.csv")
        self.out_dir = os.path.join(HERE, "screen_out" if self.cohort == "k5" else "screen_out_k0")
        import pandas as pd
        self.n_items = len(pd.read_csv(self.items_csv))

    def describe(self, script):
        print(f"[{script}] battery {self.items_csv} ({self.n_items} items) | embeddings {self.emb_prefix}_* | "
              f"responses {self.responses_csv} | cohort {self.cohort} | out {self.out_dir}", flush=True)
        if not os.path.exists(self.responses_csv):
            raise SystemExit(f"responses file not found: {self.responses_csv}\n  Pass --responses-dir <folder> "
                             f"(the --out folder build_responses_v9.py wrote to).")


CFG = _Cfg()
