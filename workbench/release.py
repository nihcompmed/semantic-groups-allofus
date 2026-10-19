"""release.py -- what may leave the Workbench: the All of Us Data and Statistics Dissemination Policy, the checks run on
every file before it is zipped, and the zip itself.

THE POLICY. "No participant count of 1 to 20 can be published or distributed directly (a count of 0 is permitted)", and
no statistic may allow such a count to be derived from other reported cells, including through percentages or other
formulas. The range is 1 to 20 inclusive. The remedies used here are the ones the policy names:
  primary        a count of 1 to 20 is written MARK ("<=20"), and every statistic resting on those people is blanked
  complementary  where cells partition a known total, further cells are hidden, smallest first, until the hidden cells
                 sum to 0 or to more than 20; they are written SUPP ("suppressed"). A statistic that would solve back
                 to a suppressed count (a rate with its interval, a mean with its standard error) is blanked too.
  pairs          a count c of a known total T is hidden when c or T - c is 1 to 20
  pooling        adjacent categories are collapsed until every cell is 0 or more than 20
  precision      rates, shares and intervals are written to DECIMALS places, other statistics beside them to SIGNIF
                 significant figures, so that no count can be solved from the digits
"""
import importlib
import json
import os
import platform
import zipfile

import numpy as np
import pandas as pd

from settings import DOWNLOADS, RELEASE

MAX_SUPPRESSED = 20
MARK = "<=20"
SUPP = "suppressed"
DECIMALS = 3
SIGNIF = 4


# ------------------------------------------------------------------------------------------------- the policy
def small(n):
    """True for a participant count the policy forbids: 1 to 20 inclusive. 0 is permitted."""
    n = int(round(float(n)))
    return 0 < n <= MAX_SUPPRESSED


def pair_hidden(c, total):
    """A count c of a known total: hidden when c or its complement is 1 to 20."""
    return small(c) or small(float(total) - float(c))


def partition_mask(counts, avoid=()):
    """Cells to hide in a partition of a known total. Primary: every count of 1 to 20. Complementary: while the hidden
    cells sum to 1 to 20, hide the smallest remaining nonzero cell, taking cells in `avoid` only when nothing else is
    left. Returns (hide, primary) as boolean arrays."""
    c = np.asarray([int(round(float(x))) for x in counts], dtype=np.int64)
    primary = (c > 0) & (c <= MAX_SUPPRESSED)
    hide = primary.copy()
    avoid = set(int(i) for i in avoid)
    while True:
        s = int(c[hide].sum())
        if s == 0 or s > MAX_SUPPRESSED:
            break
        cand = [i for i in np.flatnonzero(~hide & (c > 0)) if i not in avoid]
        if not cand:
            cand = list(np.flatnonzero(~hide & (c > 0)))
        if not cand:
            break
        hide[min(cand, key=lambda i: c[i])] = True
    return hide, primary


def marked(counts, hide, primary):
    """Counts as written: the number, MARK where primary, SUPP where complementary."""
    return [MARK if p else SUPP if h else int(round(float(x))) for x, h, p in zip(counts, hide, primary)]


def one(n):
    """A standalone count: the number, or MARK."""
    return MARK if small(n) else int(round(float(n)))


def complete(counts, forced):
    """Cells of a partition of a public total to hide: the forced ones, every count of 1 to 20, then the smallest
    remaining nonzero cells until the hidden cells sum to 0 or to more than 20."""
    c = np.asarray(counts, dtype=np.int64)
    hide = np.array([small(x) for x in c], bool) | np.asarray(forced, bool)
    while True:
        s = int(c[hide].sum())
        if s == 0 or s > 20:
            return hide
        cand = np.flatnonzero(~hide & (c > 0))
        if len(cand) == 0:
            return hide
        hide[cand[np.argmin(c[cand])]] = True


def hide_cells(parts, forced):
    """complete() over several partitions of the same cells (cases, non-cases, totals) until nothing changes."""
    hide = np.asarray(forced, bool)
    while True:
        new = hide.copy()
        for counts in parts:
            new |= complete(counts, new)
        if (new == hide).all():
            return hide
        hide = new


def _num(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) \
        and np.isfinite(v)


def coarsen(df, cols, decimals=DECIMALS):
    """A copy of df with the listed columns rounded to `decimals` places (strings and NaN kept)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda v: round(float(v), decimals) if _num(v) else v)
    return out


def coarsen_sig(df, cols, sig=SIGNIF):
    """A copy of df with the listed columns rounded to `sig` significant figures (strings and NaN kept)."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda v: float(f"{float(v):.{sig}g}") if _num(v) else v)
    return out


# ------------------------------------------------------------------------------------------------- the checks
ID_COLS = {"id", "person_id", "participant_id", "research_id", "subject_id", "pid"}
# columns named like counts that count items, groups, model columns or iterations, not participants
COUNT_COL_EXEMPT = {"n_items", "n_core_items", "n_semantic_groups", "n_groups_on_direction", "n_levels_used", "n_A",
                    "n_B", "n_criteria", "n_ace_categories", "n_criterion_items", "n_groups", "n_constructs", "n_iter",
                    "n_perm", "n_categories", "n_units_scale", "n_levels"}


def no_person_columns(path):
    """(ok, reason): the file has no person identifier column and no written index."""
    if path.endswith(".json"):
        return True, ""
    d = pd.read_csv(path)
    hit = ID_COLS.intersection(str(c).strip().lower() for c in d.columns)
    if hit:
        return False, f"carries a person column ({', '.join(sorted(hit))})"
    unnamed = [str(c) for c in d.columns if not str(c).strip() or str(c).startswith("Unnamed:")]
    if unnamed:
        return False, f"unnamed column {unnamed[0]!r}: an index written out, possibly person ids"
    return True, ""


def aggregate_only(path):
    """(ok, reason): no_person_columns, and no participant-count column (n, n_*, intersection) holds a 1 to 20."""
    if path.endswith(".json"):
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
            return False, f"key(s) {sorted(set(bad))} hold a count of 1 to 20"
        return True, ""
    ok, why = no_person_columns(path)
    if not ok:
        return ok, why
    d = pd.read_csv(path)
    for c in d.columns:
        cs = str(c)
        if (cs == "n" or cs.startswith("n_") or cs == "intersection") and cs not in COUNT_COL_EXEMPT:
            v = pd.to_numeric(d[c], errors="coerce")
            bad = int(((v > 0) & (v <= 20)).sum())
            if bad:
                return False, f"column {cs!r} holds {bad} count(s) of 1 to 20"
    return True, ""


def write_download(name, files, check=no_person_columns):
    """Check every file of downloads/<name>/ and zip them into downloads/<name>_<release>.zip. Returns the zip path."""
    d = os.path.join(DOWNLOADS, name)
    for fn in files:
        ok, why = check(os.path.join(d, fn))
        assert ok, f"{name}/{fn} refused: {why}"
    zpath = os.path.join(DOWNLOADS, f"{name}_{RELEASE}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in files:
            z.write(os.path.join(d, fn), arcname=f"{name}/{fn}")
    print(f"  download -> {zpath}", flush=True)
    return zpath


def versions():
    """Python and the versions of the libraries the analysis uses (a library that is not installed is omitted)."""
    v = {"python": platform.python_version()}
    for name in ["statsmodels", "numpy", "pandas", "scipy", "sklearn", "google.cloud.bigquery"]:
        try:
            m = importlib.import_module(name)
            v["scikit-learn" if name == "sklearn" else name] = getattr(m, "__version__", "unknown")
        except ImportError:
            pass
    return v
