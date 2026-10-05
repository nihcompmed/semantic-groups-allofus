#!/usr/bin/env python
"""build_tfidf_embeddings.py -- the item vectors of the word-count (TF-IDF) grouping, in one step.

The 2 files of record, data/embeddings/admin151_items_tfidf-subtlex-k4.npz and ..._k25.npz, come from
tfidf_ref.vectors(). They hold the SAME 151 x 586 vectors under 2 names, so each
carries its own k (4, Horn's k; 25, within the encoders' range) through the fitting chain (eMethods 5). This script
rebuilds them from the item text and the SUBTLEX-US file and compares them with the files of record.

WHAT IS BUILT (tfidf_ref.py holds the definition; nothing is redefined here)
  text        data/items_151.csv, column text_of_record (the text the encoders embed), in the file's item order
  vectors     tfidf_ref.vectors(): numerals as number words, lowercase words of 2 or more characters, function words
              kept, raw counts times idf(w) = ln((1 + 8388) / (1 + CDcount)) + 1 from SUBTLEX-US, unit-length rows
  npz keys    items (151), vectors (151 x 586, float64), vocabulary (586), meta (JSON string), as the files of record

SUBTLEX-US is not redistributed (README, "Inputs"). Download SUBTLEXus74286wordstextversion.txt (md5
ee7be8db21f26e1925ac0ea3ab598f32) and pass it with --subtlex, or keep it where tfidf_ref.py looks for it.

MODES
  (default)       build, compare with the files of record (items, vocabulary and vectors exactly), write nothing
  --out <dir>     also write the 2 files into <dir> (never into data/embeddings/ unless <dir> is that folder)

Run:  python build_tfidf_embeddings.py [--subtlex <file>] [--out <dir>]
Exit: 0 when the rebuilt vectors equal the files of record, 1 otherwise.
"""
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
# tfidf_ref.py sits beside this script (the second folder is a fallback for another layout)
for d in (HERE, os.path.abspath(os.path.join(PROJECT, "..", "..", "..", "examples", "allofus"))):
    if os.path.exists(os.path.join(d, "tfidf_ref.py")):
        sys.path.insert(0, d)
        break
import tfidf_ref   # noqa: E402

ITEMS_CSV = os.path.join(PROJECT, "data", "items_151.csv")
EMB_DIR = os.path.join(PROJECT, "data", "embeddings")
NAMES = ["admin151_items_tfidf-subtlex-k4.npz", "admin151_items_tfidf-subtlex-k25.npz"]
SUBTLEX_MD5 = "ee7be8db21f26e1925ac0ea3ab598f32"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if "--subtlex" in sys.argv:
        tfidf_ref.SUBTLEX = sys.argv[sys.argv.index("--subtlex") + 1]
    if not os.path.exists(tfidf_ref.SUBTLEX):
        sys.exit(f"SUBTLEX-US file not found at {tfidf_ref.SUBTLEX}; download it (README, 'Inputs') and pass --subtlex")
    got = md5(tfidf_ref.SUBTLEX)
    assert got == SUBTLEX_MD5, f"{tfidf_ref.SUBTLEX} has md5 {got}, not {SUBTLEX_MD5}: a different SUBTLEX-US file"

    if not os.path.exists(ITEMS_CSV):   # a fresh copy rebuilds it in step 2 of run_semantic_groups.sh (README)
        sys.exit(f"{ITEMS_CSV} not found: run build_text_of_record.py and build_items_table.py first (README, 'The semantic groups', steps 1 and 2)")
    items = pd.read_csv(ITEMS_CSV)
    assert len(items) == 151 and items["item"].is_unique, "items_151.csv is not the 151 unique items"
    X, voc, missing = tfidf_ref.vectors(items["text_of_record"].tolist())
    V = np.asarray(X.todense(), dtype=np.float64)
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-12), "a row is not unit length"
    meta = {"representation": "TF-IDF, IDF from SUBTLEX-US (examples/allofus/tfidf_ref.py)", "text": "text_of_record",
            "stop_words": "kept", "numerals": "number words", "absent_from_reference": list(missing),
            "vocabulary": int(len(voc)), "created": datetime.date.today().isoformat(),
            "written_by": "build_tfidf_embeddings.py"}
    print(f"[build_tfidf_embeddings] {V.shape[0]} items x {V.shape[1]} words | absent from SUBTLEX-US: "
          f"{', '.join(missing)}")

    ok = True
    for name in NAMES:
        path = os.path.join(EMB_DIR, name)
        if not os.path.exists(path):
            print(f"  {name}: no file of record at {path}, nothing to compare")
            ok = False
            continue
        r = np.load(path, allow_pickle=True)
        same_items = [str(x) for x in r["items"]] == items["item"].astype(str).tolist()
        same_voc = [str(x) for x in r["vocabulary"]] == [str(x) for x in voc]
        same_vec = r["vectors"].shape == V.shape and np.array_equal(r["vectors"], V)
        diff = float(np.abs(r["vectors"] - V).max()) if r["vectors"].shape == V.shape else float("nan")
        print(f"  {name}: items {'equal' if same_items else 'DIFFER'}, vocabulary {'equal' if same_voc else 'DIFFERS'}, "
              f"vectors {'identical' if same_vec else f'DIFFER (max |difference| {diff:.1e})'}")
        ok &= same_items and same_voc and same_vec

    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
        os.makedirs(out, exist_ok=True)
        for name in NAMES:
            np.savez(os.path.join(out, name), items=np.array(items["item"].astype(str).tolist(), dtype=object),
                     vectors=V, vocabulary=np.array([str(x) for x in voc], dtype=object), meta=json.dumps(meta))
        print(f"  wrote {len(NAMES)} files to {out}")
    print(f"\n{'the rebuilt vectors equal the files of record' if ok else 'the rebuilt vectors DIFFER from the files of record'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
