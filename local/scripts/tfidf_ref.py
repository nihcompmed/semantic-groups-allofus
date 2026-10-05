"""tfidf_ref.py -- TF-IDF item vectors with the IDF taken from SUBTLEX-US.

WHY. A TF-IDF fitted on the 151 items rests every word's weight on the handful of items that contain
it, and the weights move when items are added. With the IDF taken from an outside corpus, an item's
vector depends on its own words only, as an encoder's embedding does.

THE DECLARATION, made before the run:
  reference   SUBTLEX-US (Brysbaert and New 2009), data/subtlexus/. The documents are its 8,388 films
              (the Ghent page: CDcount is "the number of films in which the word appears"), and a word's
              document frequency is CDcount.
  idf         ln((1 + 8,388) / (1 + CDcount)) + 1, scikit-learn's smoothed formula on the outside
              corpus. "the" (CDcount 8,388) comes out at 1.0, the floor; "you" (8,381) at 1.0008.
  tokens      lowercase, scikit-learn's default token pattern (2+ word characters), which splits on
              apostrophes as the corpus does. Stop words kept.
  numerals    mapped to number words before tokenizing ("18" -> "eighteen", "100" -> "one hundred"),
              because the corpus holds no digits and a numeral would otherwise take the rarest weight.
              In the 151 this touches 13 items, with the numerals 1, 5, 6, 18, 20 and 100.
  missing     a word absent from the corpus gets CDcount 0, the rarest weight (IDF 10.03). In the 151:
              hookahs, mods, snus, vape, vaporizers.
  vectors     raw counts times the IDF, rows scaled to unit length.
  vocabulary  every word in the texts being represented. For a rewrite this includes words the 151 do
              not use; they scale both of its cosines equally, so no win or loss depends on them.
"""
import math
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
# the file is looked for in data/subtlexus/ beside this script or one level up (where this script sits in scripts/
# and the inputs in data/); a caller can set SUBTLEX itself.
_SUBTLEX_CANDIDATES = [os.path.join(HERE, "data", "subtlexus", "SUBTLEXus74286wordstextversion.txt"),
                      os.path.join(HERE, "..", "data", "subtlexus", "SUBTLEXus74286wordstextversion.txt")]
SUBTLEX = next((p for p in _SUBTLEX_CANDIDATES if os.path.exists(p)),
               _SUBTLEX_CANDIDATES[1] if os.path.isdir(os.path.join(HERE, "..", "data")) else _SUBTLEX_CANDIDATES[0])
N_DOCS = 8388

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_words(n):
    """0..999999 as English words, the way a number is said ("one hundred", "twenty one")."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    return number_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + number_words(n % 1000))


def numerals_to_words(text):
    return re.sub(r"\d+", lambda m: number_words(int(m.group(0))) if len(m.group(0)) <= 6 else m.group(0),
                  str(text))


def subtlex_idf():
    S = pd.read_csv(SUBTLEX, sep="\t", keep_default_na=False, na_values=[])
    df = dict(zip(S["Word"].str.lower(), S["CDcount"].astype(int)))
    assert max(df.values()) == N_DOCS, "CDcount is not on 8,388 films; wrong file?"
    return df


def vectors(texts, df_ref=None):
    """Unit-length TF-IDF rows for `texts` under the declaration above. Returns (matrix, vocabulary,
    the words absent from the corpus)."""
    df_ref = df_ref or subtlex_idf()
    t = [numerals_to_words(x) for x in texts]
    cv = CountVectorizer()                                   # lowercase, default token pattern
    C = cv.fit_transform(t).astype(float)
    voc = cv.get_feature_names_out()
    idf = np.array([math.log((1 + N_DOCS) / (1 + df_ref.get(w, 0))) + 1 for w in voc])
    X = C.multiply(idf).tocsr()
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    X = X.multiply(1.0 / np.maximum(norms, 1e-12)[:, None]).tocsr()
    missing = [w for w in voc if w not in df_ref]
    return X, voc, missing
