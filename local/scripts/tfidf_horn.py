#!/usr/bin/env python
"""tfidf_horn.py -- Horn's k for TF-IDF item vectors, and its stability when one item is left out.

Two versions, both stop words kept:
  battery   scikit-learn TfidfVectorizer defaults, IDF fitted on the 151 texts of record
  subtlex   IDF from SUBTLEX-US, the declaration in tfidf_ref.py (numerals as number words, words
            absent from the corpus at the rarest weight)

Horn's parallel analysis is ross_proj.select_k with the encoders' settings (1_select_k.py): items as
rows, unit-length rows, centered columns, each column shuffled independently, 200 permutations, the
95th percentile, seed 0, k = the first crossing.

THE STABILITY CHECK, declared before the run. For each of the 151 items, leave it out and repeat:
  battery   refit the vectorizer on the other 150 (vocabulary and IDF both move), then Horn's k
  subtlex   drop the item's row (every other row is unchanged by construction), then Horn's k
It reports the distribution of k over the 151 leave-outs, and for the battery version how far the
item-to-item cosines of the 150 move against the full fit (Pearson r over the off-diagonal pairs).

Also written, to read the result rather than only count it: for each version, the words carrying the
leading directions (the largest absolute loadings of the first 6 principal directions).

Writes  outputs/k_selection/tfidf/ and .../tfidf_subtlex/   (KResult, scree)
        outputs/k_selection/tfidf_loo.csv                   (one row per leave-out)
        outputs/k_selection/tfidf_summary.json
        outputs/k_selection/tfidf_top_words.csv
Run     CUDA_VISIBLE_DEVICES="" python scripts/tfidf_horn.py [--subtlex <file>]   (from the folder holding scripts/)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
# 2 layouts. Here this script sits in scripts/ beside tfidf_ref.py, with ross_proj/, data/ and outputs/ one level
# up. In the other layout (the else branch) it sits 2 levels below ross_proj/.
if os.path.isdir(os.path.join(HERE, "..", "ross_proj")) and os.path.isdir(os.path.join(HERE, "..", "data")):
    LOCAL = os.path.abspath(os.path.join(HERE, ".."))
    sys.path.insert(0, LOCAL)
    sys.path.insert(0, HERE)
    ITEMS = os.path.join(LOCAL, "data", "items_151.csv")
    OUT = os.path.join(LOCAL, "outputs", "k_selection")
else:
    ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "examples", "allofus"))
    ITEMS = os.path.join(ROOT, "allofus", "manuscript", "jama", "data", "items_151.csv")
    OUT = os.path.join(ROOT, "allofus", "manuscript", "jama", "outputs", "k_selection")
from ross_proj import select_k                                   # noqa: E402
from ross_proj.parallel_analysis import save_scree              # noqa: E402
from ross_proj.io import save_kresult                           # noqa: E402
import tfidf_ref                                                # noqa: E402
from tfidf_ref import subtlex_idf, vectors                      # noqa: E402
if "--subtlex" in sys.argv:                                     # the SUBTLEX-US file, when not in data/subtlexus/
    tfidf_ref.SUBTLEX = sys.argv[sys.argv.index("--subtlex") + 1]
PA = dict(n_perm=200, percentile=95.0, seed=0)


def battery_matrix(texts):
    vec = TfidfVectorizer()
    return vec.fit_transform(texts).toarray(), vec.get_feature_names_out()


def top_words(X, voc, n_dir=6, n_words=8):
    Xc = X - X.mean(0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    rows = []
    for j in range(n_dir):
        order = np.argsort(-np.abs(Vt[j]))[:n_words]
        rows.append({"direction": j + 1, "eigenvalue": float(s[j] ** 2),
                     "top_words": " ".join(f"{voc[i]}({Vt[j, i]:+.2f})" for i in order)})
    return rows


def offdiag_r(A, B):
    iu = np.triu_indices_from(A, k=1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1])


def main():
    T = pd.read_csv(ITEMS)
    items = T["item"].astype(str).tolist()
    texts = T["text_of_record"].astype(str).tolist()
    ref = subtlex_idf()

    full = {}
    Xb, vb = battery_matrix(texts)
    Xs_sp, vs, absent = vectors(texts, ref)
    Xs = Xs_sp.toarray()
    summary, tw = {}, []
    for name, X, voc in (("tfidf", Xb, vb), ("tfidf_subtlex", Xs, vs)):
        k = select_k(X, **PA)
        full[name] = k.k
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        save_kresult(k, d)
        save_scree(k, os.path.join(d, "scree.png"), title=f"{name}, stop words kept (k={k.k})")
        summary[name] = {"k": k.k, "k_count": k.k_count, "k_mean": k.k_mean, "d80": k.d80, "d90": k.d90,
                         "vocabulary": int(len(voc)),
                         "real_eigs_1_12": [round(float(v), 4) for v in k.real_eigs[:12]],
                         "null_p95_1_12": [round(float(v), 4) for v in k.null_percentile[:12]]}
        for r in top_words(X, voc):
            tw.append({"version": name, **r})
        print(f"{name:14s} vocabulary {len(voc)} | Horn's k {k.k} (count above null {k.k_count}, "
              f"vs null mean {k.k_mean}) | d80 {k.d80}", flush=True)
    summary["tfidf_subtlex"]["absent_from_reference"] = absent
    pd.DataFrame(tw).to_csv(os.path.join(OUT, "tfidf_top_words.csv"), index=False)

    # ---- leave one item out
    Cb_full = Xb @ Xb.T
    rows = []
    for i, it in enumerate(items):
        keep = [j for j in range(len(items)) if j != i]
        Xb_i, _ = battery_matrix([texts[j] for j in keep])
        kb = select_k(Xb_i, **PA).k
        r_cos = offdiag_r(Xb_i @ Xb_i.T, Cb_full[np.ix_(keep, keep)])
        Xs_i = Xs[keep]
        Xs_i = Xs_i[:, np.abs(Xs_i).sum(0) > 0]          # words only the left-out item used
        ks = select_k(Xs_i, **PA).k
        rows.append({"left_out": it, "k_battery": kb, "k_subtlex": ks, "cosine_r_battery": round(r_cos, 5)})
        if (i + 1) % 25 == 0:
            print(f"  leave-one-out {i + 1}/{len(items)}", flush=True)
    L = pd.DataFrame(rows)
    L.to_csv(os.path.join(OUT, "tfidf_loo.csv"), index=False)
    for name, col in (("tfidf", "k_battery"), ("tfidf_subtlex", "k_subtlex")):
        vc = L[col].value_counts().sort_index()
        summary[name]["loo_k_distribution"] = {int(a): int(b) for a, b in vc.items()}
        summary[name]["loo_share_equal_to_full_k"] = round(float((L[col] == full[name]).mean()), 4)
    summary["tfidf"]["loo_cosine_r_min_median"] = [round(float(L.cosine_r_battery.min()), 5),
                                                   round(float(L.cosine_r_battery.median()), 5)]
    summary["settings"] = {"horn": PA, "stop_words": "kept", "items": len(items)}
    json.dump(summary, open(os.path.join(OUT, "tfidf_summary.json"), "w"), indent=2)

    print("\nleave-one-out Horn's k:")
    for name, col in (("tfidf", "k_battery"), ("tfidf_subtlex", "k_subtlex")):
        print(f"  {name:14s} full k {full[name]} | " + ", ".join(
            f"k={a}: {b}" for a, b in summary[name]["loo_k_distribution"].items()))
    print(f"  battery cosines, r against the full fit: min {L.cosine_r_battery.min():.4f}, "
          f"median {L.cosine_r_battery.median():.4f}")
    print(f"\nwrote -> {OUT}: tfidf/, tfidf_subtlex/, tfidf_loo.csv, tfidf_summary.json, tfidf_top_words.csv")


if __name__ == "__main__":
    main()
