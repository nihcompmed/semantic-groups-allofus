#!/usr/bin/env python3
"""disclosure.py -- the All of Us Data and Statistics Dissemination Policy, in one place.

THE POLICY, verbatim from the program's policy page: "No participant count of 1 to 20 can be
published or distributed directly (a count of 0 is permitted)", and "No data or statistics can be
reported that allow a participant count of 1 to 20 to be derived from other reported cells or
information ... including the use of percentages or other mathematical formulas." The permitted
remedies are collapsing cells, coarsening, or cell suppression. The range is 1 to 20 INCLUSIVE, so
a count of exactly 20 is suppressed too.

WHAT EVERY SCRIPT WHOSE OUTPUT LEAVES THE WORKBENCH DOES WITH IT
  primary        a count of 1 to 20 is written MARK ("<=20"), and every statistic resting on those
                 people (a percent, an interval, a mean) is blanked
  complementary  where cells partition a total that is known from anywhere -- a breadth
                 distribution and its N, an item's levels and the cohort, top-3 slots and 3 x n --
                 one hidden cell is recovered by subtraction. More cells are hidden, smallest first,
                 until the hidden cells sum to 0 or to more than 20. They are written SUPP
                 ("suppressed"). ★ A statistic over a suppressed count that can be SOLVED BACK to it
                 is blanked too: a rate with its interval, a mean with its SE (n = (sd / se)^2), a
                 share of a subgroup each return the count exactly, and rounding does not help when
                 the hidden count is known to lie in a range of 20 (care_sweep: 3 decimals still fixed
                 it in 246 of 259 rows). An AUC with its DeLong interval (3 decimals) and model odds
                 ratios with Wald intervals may stay: their intervals depend on the spread of the
                 scores, which is written nowhere, so the count cannot be solved from them.
  pairs          a count c of a total T is hidden when c or T - c is 1 to 20 (an intersection and
                 its k, criteria met and evaluable)

  precision      rates, shares, AUCs and intervals are written to DECIMALS places, other statistics
                 beside them to SIGNIF significant figures (coarsen, coarsen_sig)

Rows are kept, so a reader sees that something was suppressed. ★ Printed output IS covered by the
policy too: the logs are pasted out of the Workbench.
"""
import numpy as np

MAX_SUPPRESSED = 20          # the policy's upper bound, inclusive
MARK = "<=20"                # the form the program's support page gives
SUPP = "suppressed"          # a complementary cell: more than 20 people, hidden to protect another


def small(n):
    """True for a participant count the policy forbids: 1 to 20 inclusive. 0 is permitted."""
    n = int(round(float(n)))
    return 0 < n <= MAX_SUPPRESSED


def pair_hidden(c, total):
    """A count c of a known total: hidden when c or its complement is 1 to 20."""
    return small(c) or small(float(total) - float(c))


def partition_mask(counts, avoid=()):
    """Cells to hide in a partition of a known total.

    Primary: every count of 1 to 20. Complementary: while the hidden cells sum to 1 to 20, hide the
    smallest remaining nonzero cell, taking cells in `avoid` only when nothing else is left (used to
    keep a cell another file reports, such as breadth 0, visible where the rule allows).
    Returns (hide, primary) as boolean arrays."""
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
    out = []
    for x, h, p in zip(counts, hide, primary):
        out.append(MARK if p else SUPP if h else int(round(float(x))))
    return out


def one(n):
    """A standalone count: the number, or MARK."""
    return MARK if small(n) else int(round(float(n)))


# ★ PRECISION. A statistic written to 16 digits can be solved back into the counts it was
# computed from: a share k/n is an exact fraction; a rate with its Wilson interval gives n and k by algebra;
# an AUC is U / (cases x non-cases); a mean of whole-number ages is a sum over n; two pseudo-R^2 and their
# likelihood-ratio statistic give the null log-likelihood, and with it n and k.
# Every rate, share, AUC and interval these scripts write is rounded to DECIMALS places; any other
# statistic written beside them to SIGNIF significant figures. Rounding is applied last, after every
# internal check, and printed values are formatted coarser still.
DECIMALS = 3
SIGNIF = 4


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
