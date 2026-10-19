"""models.py -- the regression models every comparison uses.

THE JOINT MODEL (joint). outcome ~ category (deciles, pooled bins or sets, as indicators) + age_z + age_z^2 (+ sex where
the population holds both sexes) (+ log(nights) for the night-to-night SD of sleep), with age standardized within the
population:
  thresholds met, the emergency department share, minutes of sleep   least squares, HC1 standard errors
  barriers to care, recorded conditions                              logistic
  night-to-night SD of sleep                                         median regression
A category's adjusted value is the model's prediction averaged over the whole population with every person's category
set to it (standardization), with a delta-method 95% interval from the coefficients' covariance. A contrast between 2
categories (joint_with_contrasts) is the difference of their 2 standardized predictions from the same fit, with its own
delta-method interval and P, so the tested difference is exactly the gap between the 2 values written beside it.

THE TREND (trend_fit). The same covariates with the decile number 1 to 10 as one slope: an odds ratio per decile for the
logistic models, a difference per decile otherwise, and the same over the 9 steps from decile 1 to decile 10.
"""
import warnings

import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from settings import Z

CONVERGED = {}                                  # tag -> whether the fit converged (checked by the steps)


def covs(age, sex, with_sex, extra=None):
    a = (age - age.mean()) / age.std()
    return np.column_stack([a, a ** 2] + ([sex] if with_sex else []) + ([extra] if extra is not None else []))


def design(cat, levels, C):
    """An intercept, 1 indicator per category but the first, then the covariates."""
    return np.column_stack([np.ones(len(cat))] + [(cat == lv).astype(float) for lv in levels[1:]] + [C])


def fit_model(kind, y, X, tag):
    if kind == "logit":
        f = sm.Logit(y, X).fit(disp=0, maxiter=200)
        CONVERGED[tag] = bool(f.mle_retvals.get("converged", False))
    elif kind == "ols":
        f = sm.OLS(y, X).fit(cov_type="HC1")
        CONVERGED[tag] = True
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f = sm.QuantReg(y, X).fit(q=0.5, max_iter=5000)
        CONVERGED[tag] = not any(issubclass(x.category, IterationLimitWarning) for x in w)
    return np.asarray(f.params, float), np.asarray(f.cov_params(), float)


def joint_with_contrasts(kind, y, cat, levels, C, tag, pairs=()):
    """{level: (value, lo, hi)}, the standardized prediction per category; and {label: (difference, lo, hi, P)} for
    each (label, a, b) in pairs, from the same fit."""
    beta, V = fit_model(kind, y, design(cat, levels, C), tag)
    pred = {}
    for lv in levels:
        Xl = design(np.full(len(cat), lv), levels, C)
        eta = Xl @ beta
        if kind == "logit":
            pr = 1.0 / (1.0 + np.exp(-eta))
            pred[lv] = (float(pr.mean()), ((pr * (1 - pr))[:, None] * Xl).mean(axis=0))
        else:
            pred[lv] = (float(eta.mean()), Xl.mean(axis=0))
    out = {}
    for lv, (val, g) in pred.items():
        se = float(np.sqrt(max(g @ V @ g, 0.0)))
        out[lv] = (val, val - Z * se, val + Z * se)
    con = {}
    for lab, a, b in pairs:
        d = pred[a][0] - pred[b][0]
        gd = pred[a][1] - pred[b][1]
        se = float(np.sqrt(max(gd @ V @ gd, 0.0)))
        con[lab] = (d, d - Z * se, d + Z * se, float(2 * norm.sf(abs(d) / se)) if se > 0 else np.nan)
    return out, con


def joint(kind, y, cat, levels, C, tag):
    """{level: (value, lo, hi)}: the standardized prediction per category with its delta-method interval."""
    return joint_with_contrasts(kind, y, cat, levels, C, tag)[0]


def check_intercept_only(kind, y, cat, levels, tag):
    """With no covariate, the standardized value of a category is its raw share or mean (a check of joint)."""
    est = joint(kind, y, cat, levels, np.empty((len(y), 0)), tag + " intercept-only")
    for lv in levels:
        raw = float(np.mean(y[cat == lv]))
        assert abs(est[lv][0] - raw) < 1e-6, f"{tag} {lv}: intercept-only {est[lv][0]} is not the raw value {raw}"


def contrast_two_sets(kind, y, ind, Cv):
    """A difference between 2 sets from a model of those 2 sets alone (indicator ind = 1 for the first): the
    standardized difference, its delta-method interval and P."""
    X = np.column_stack([np.ones(len(y)), ind, Cv])
    b, V = fit_model(kind, y, X, f"contrast {kind} {len(y)}")
    X1, X0 = X.copy(), X.copy()
    X1[:, 1], X0[:, 1] = 1.0, 0.0
    if kind == "logit":
        p1, p0 = 1 / (1 + np.exp(-(X1 @ b))), 1 / (1 + np.exp(-(X0 @ b)))
        d = float(p1.mean() - p0.mean())
        g = ((p1 * (1 - p1))[:, None] * X1).mean(0) - ((p0 * (1 - p0))[:, None] * X0).mean(0)
    else:
        d = float((X1 @ b).mean() - (X0 @ b).mean())
        g = X1.mean(0) - X0.mean(0)
    se = float(np.sqrt(max(g @ V @ g, 0.0)))
    return d, d - Z * se, d + Z * se, (float(2 * norm.sf(abs(d) / se)) if se > 0 else np.nan)


def trend_fit(kind, y, dec, C):
    """The decile number as one slope, with the covariates C."""
    X = sm.add_constant(np.column_stack([dec, C]), has_constant="add")
    if kind == "logit":
        f = sm.Logit(y, X).fit(disp=0, maxiter=200)
    elif kind == "ols":
        f = sm.OLS(y, X).fit(cov_type="HC1")
    else:
        f = sm.QuantReg(y, X).fit(q=0.5, max_iter=5000)
    b, se, p = float(f.params[1]), float(f.bse[1]), float(f.pvalues[1])
    if kind == "logit":
        return {"per_decile": np.exp(b), "lo": np.exp(b - Z * se), "hi": np.exp(b + Z * se), "p": p,
                "over_9": np.exp(9 * b), "over_9_lo": np.exp(9 * (b - Z * se)), "over_9_hi": np.exp(9 * (b + Z * se)),
                "scale": "odds ratio"}
    return {"per_decile": b, "lo": b - Z * se, "hi": b + Z * se, "p": p, "over_9": 9 * b, "over_9_lo": 9 * (b - Z * se),
            "over_9_hi": 9 * (b + Z * se), "scale": "difference"}


def logistic(y, X):
    """An unpenalized logistic regression with an intercept."""
    return sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=500)
