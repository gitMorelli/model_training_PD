import numpy as np
import patsy
import statsmodels.api as sm

# ----------------------------------------------------------------------------
# 1. Model fitting
# ----------------------------------------------------------------------------
# Optimisers tried in turn; the fit with the highest likelihood is kept. A single run of
# lbfgs (falling back to powell only on failure) sometimes stops at a clearly worse local
# optimum: seen as spikes in the change-point profile and as implausible LRTs between
# matched-set models. A separate powell run finds the better optimum in those cases.
# Set FIT_METHODS = (["lbfgs", "powell"],) to go back to a single run.
FIT_METHODS = (["lbfgs", "powell"], "powell")


def _best_fit(model, reml, methods=None):
    """Fit `model` with each optimiser in `methods`; keep the highest (RE)ML log-likelihood,
    preferring converged fits."""
    best = None
    for m in methods or FIT_METHODS:
        try:
            r = model.fit(reml=reml, method=m)
        except Exception:
            continue
        if not np.isfinite(r.llf):
            continue
        key = (bool(r.converged), r.llf)
        if best is None or key > best[0]:
            best = (key, r)
    if best is None:
        raise RuntimeError("MixedLM failed with every optimiser in FIT_METHODS")
    return best[1]


def fit_mixed(formula, df, reml=False, re_formula="~time"):
    """Linear mixed model with random intercept + random slope in time per subject.
    Use reml=False (ML) whenever fixed-effect structures are compared by LRT.

    Design matrices are built with patsy directly so that the spline knots are
    stored (result.design_info) and can be reused to predict on new time grids.

    The random-effect design uses time measured from its mean. With a correlated intercept
    and slope this is the same model (same likelihood and fixed effects), but measured from
    diagnosis the two are strongly correlated and the optimiser often stops early."""
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    Z = patsy.dmatrix(re_formula, df.assign(time=df["time"] - df["time"].mean()),
                      return_type="dataframe")
    model = sm.MixedLM(y, X, groups=df["subject"], exog_re=Z)
    result = _best_fit(model, reml)
    result.design_info = X.design_info
    return result


def fit_full_and_null(df, time_term, adj=''):
    full = fit_mixed(f"y ~ {time_term} * C(group){adj}", df)  # class-specific trajectories
    null = fit_mixed(f"y ~ {time_term}{adj}", df)             # one shared trajectory
    return full, null


RE_STRUCTURES = ("subject", "subject_uncorr", "set")


def _re_structure(matched_sets, re_structure=None):
    """matched_sets=True -> "set", False -> "subject", unless re_structure is given."""
    if re_structure is not None:
        if re_structure not in RE_STRUCTURES:
            raise ValueError(f"re_structure must be one of {RE_STRUCTURES}")
        return re_structure
    return "set" if matched_sets else "subject"


def _fit(formula, df, reml=False, matched_sets=False, re_structure=None, re_center="mean"):
    """Mixed model with a random intercept and a random slope in time per subject.

    re_structure (or matched_sets for the first and last):
      "subject"         subject intercept + slope, CORRELATED (as in fit_mixed)
      "subject_uncorr"  subject intercept + slope, UNCORRELATED - the same subject part as
                        "set", so "set" vs "subject_uncorr" differ ONLY by the set variance
      "set"             random intercept per matched set + subject intercept and slope
                        (uncorrelated: statsmodels cannot keep the correlation here)
    re_center  time origin of the random SLOPE: "mean" (mean time of `df`, default), a number,
               or None (time = 0, i.e. diagnosis). The fixed part always uses `time`, so the
               difference curves are unaffected. Irrelevant for "subject" (a correlated model
               fits equally well at any origin); for the uncorrelated structures it matters a
               lot: measured from diagnosis, intercept and slope are strongly correlated and
               dropping the correlation costs a lot of fit.
    The chosen structure and centre are stored on the result (re_structure, re_center) and
    reused by _heldout_loglik.
    """
    structure = _re_structure(matched_sets, re_structure)
    c = float(df["time"].mean()) if re_center == "mean" else float(re_center or 0.0)
    df = df.assign(time_re=df["time"] - c)
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    if structure == "set":
        vc = {"subj_int": "0 + C(subject)", "subj_slope": "0 + C(subject):time_re"}
        model = sm.MixedLM.from_formula(formula, df, groups="set", re_formula="1", vc_formula=vc)
    elif structure == "subject_uncorr":
        model = sm.MixedLM.from_formula(formula, df, groups="subject", re_formula="1",
                                        vc_formula={"subj_slope": "0 + time_re"})
    else:
        Z = patsy.dmatrix("~time_re", df, return_type="dataframe")
        model = sm.MixedLM(y, X, groups=df["subject"], exog_re=Z)
    result = _best_fit(model, reml)
    result.design_info = X.design_info
    result.re_structure, result.re_center = structure, c
    return result
