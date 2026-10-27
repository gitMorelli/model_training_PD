import numpy as np
import patsy
import statsmodels.api as sm

# ----------------------------------------------------------------------------
# 1. Model fitting
# ----------------------------------------------------------------------------
def fit_mixed(formula, df, reml=False, re_formula="~time"):
    """Linear mixed model with random intercept + random slope in time per subject.
    Use reml=False (ML) whenever fixed-effect structures are compared by LRT.

    Design matrices are built with patsy directly so that the spline knots are
    stored (result.design_info) and can be reused to predict on new time grids."""
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    Z = patsy.dmatrix(re_formula, df, return_type="dataframe")
    model = sm.MixedLM(y, X, groups=df["subject"], exog_re=Z)
    result = model.fit(reml=reml, method=["lbfgs", "powell"])
    result.design_info = X.design_info
    return result


def fit_full_and_null(df, time_term, adj=''):
    full = fit_mixed(f"y ~ {time_term} * C(group){adj}", df)  # class-specific trajectories
    null = fit_mixed(f"y ~ {time_term}{adj}", df)             # one shared trajectory
    return full, null

def _fit(formula, df, reml=False, matched_sets=False):
    """Mixed model with subject intercept + slope.
    matched_sets=False: subject random intercept and slope (correlated), as in fit_mixed.
    matched_sets=True:  adds a random intercept per matched set; subjects nested in sets.
                        (statsmodels cannot keep the subject intercept-slope correlation here.)"""
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    if matched_sets:
        vc = {"subj_int": "0 + C(subject)", "subj_slope": "0 + C(subject):time"}
        model = sm.MixedLM.from_formula(formula, df, groups="set", re_formula="1", vc_formula=vc)
    else:
        Z = patsy.dmatrix("~time", df, return_type="dataframe")
        model = sm.MixedLM(y, X, groups=df["subject"], exog_re=Z)
    result = model.fit(reml=reml, method=["lbfgs", "powell"])
    result.design_info = X.design_info
    return result