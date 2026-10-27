
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from .helpers import _adj_col, simulate_data, _slot_weights, _group_slot_weights, significant_intervals, _information_criteria, _interval_label
from .helpers import _in_interval, _estimable_slots, _heldout_loglik, adjustment_spec
from .fit import fit_mixed, fit_full_and_null, _fit
from .changepoint import changepoint_profile, cluster_bootstrap_tau
from .tests import lrt, lrt_re, permutation_lrt
from .plot import plot_results, _plot_comparison, _plot_range
from .difference import difference_curve, curves_by_value

warnings.filterwarnings("ignore")  # MixedLM is chatty about convergence/boundary warnings


preset_columns = {             # your column name -> name used internally
        "subject":"unique_id",
        "group": "true_label",
        "time": "case_dt",
        "y": "probability_1",
        "split": "slot",
        "set": "case_control_group",
        "n_quest": "n_quest"
    }

ADJ_LABEL = {None: "", "slot": "slot", "n_quest_add": "n_quest (additive)",
             "n_quest_int": "n_quest x class"}


def _resolve_adjust(adjust, model_split):
    """adjust="auto" keeps the old switch: "slot" if model_split else None."""
    if adjust == "auto":
        return "slot" if model_split else None
    return adjust


def _check_adjust_column(df, adjust):
    col = _adj_col(adjust)
    if col and col not in df.columns:
        raise KeyError(f'adjust={adjust!r} needs a "{col}" column: add it to `columns`'
                       + (' (e.g. "n_quest": "n_quest", built with groupby(subject).cumcount() + 1)'
                          if col == "n_quest" else ' (e.g. "split": "slot")'))


def _first_onset(curve):
    iv = significant_intervals(curve)
    return iv[0][0] if iv else np.nan


def run_analysis(raw, columns=preset_columns, logit=True, eps=1e-4, path=None, spline_df=4,
                 run_permutation_test=False, n_perm=200, run_bootstrap=False, n_boot=200, time_model="spline", model_split=True,
                 use_baseline=True, adjust="auto", n_form="log", n_values=None):
    '''logit = True            # model logit(probability_1) instead of the raw probability
    eps = 1e-4                # clip probabilities away from 0/1 before the logit
    adjust                    None, "slot", "n_quest_add" or "n_quest_int" (see helpers.adjustment_spec);
                              "auto" (default) = "slot" if model_split else None
    n_form                    form of the n_quest term: "log", "linear" or "cat"
    n_values                  n_quest values for the gap-at-fixed-n curves (n_quest_int only);
                              default: 10th/35th/65th/90th percentiles
    '''
    SPLINE_DF = spline_df  # flexibility of the time trend; 3-5 is typical, check sensitivity
    if time_model == "spline":
        TIME_TERM = f"cr(time, df={spline_df}, constraints='center')"
    elif time_model == "quadratic":
        TIME_TERM = "(center(time) + I(center(time)**2))"
    else:
        raise ValueError("time_model must be 'spline' or 'quadratic'")
    adjust = _resolve_adjust(adjust, model_split)
    ADJ, adj_col, _ = adjustment_spec(adjust, n_form)

    def prepare(raw):
        COLUMNS = {v: k for k, v in columns.items()}
        df = raw.rename(columns=COLUMNS)[list(COLUMNS.values())].dropna()
        df["time"] = df["time"].astype(float)
        if logit:
            p = df["y"].clip(eps, 1 - eps)
            df["y"] = np.log(p / (1 - p))
        assert df["group"].nunique() == 2, "expected exactly two classes in true_label"
        return df

    df = prepare(raw)
    _check_adjust_column(df, adjust)
    # df = simulate_data()              # <- uncomment to run on simulated data instead
    print(f"{df['subject'].nunique()} subjects, {len(df)} observations")
    print(f"Model: y ~ {TIME_TERM} * C(group){ADJ}\n")

    # Steps 1-2: models and global test
    full, null = fit_full_and_null(df, TIME_TERM, adj=ADJ)
    stat, dof, p = lrt(full, null)
    if adjust in ("slot", "n_quest_int"):    # the null keeps C(group) through the adjustment term
        what = f"does the gap change over time, given {ADJ_LABEL[adjust]}?"
    else:
        what = "class affects level or shape" + (f", adjusted for {ADJ_LABEL[adjust]}" if adjust else "")
    print(f"Global LRT ({what}): chi2={stat:.2f}, df={dof}, p={p:.3g}")

    # Step 3: difference curve (refit with REML for less biased variance components)
    full_reml = fit_mixed(f"y ~ {TIME_TERM} * C(group){ADJ}", df, reml=True)
    baseline_time = df["time"].quantile(0.05) if use_baseline else None
    curve, crit, groups = difference_curve(full_reml, df, model_split=adjust, baseline_time=baseline_time)
    print(f"Simultaneous critical value: {crit:.2f} (vs 1.96 pointwise)")
    intervals = significant_intervals(curve)
    if intervals:
        print("Simultaneous band excludes 0 on: "
              + ", ".join(f"[{a:.2f}, {b:.2f}]" for a, b in intervals))
        print(f"-> difference DETECTABLE from t ≈ {intervals[0][0]:.2f}")
    else:
        print("No time interval where the difference is significant.")

    # Step 4: change-point estimate of the onset
    tau_hat, tau_ci, profile = changepoint_profile(df, time_term=TIME_TERM+ADJ)
    print(f"\nChange-point onset: tau = {tau_hat:.2f}, "
          f"approx. 95% CI [{tau_ci[0]:.2f}, {tau_ci[1]:.2f}]")
    if adjust:
        print(f"  (conditional on {ADJ_LABEL[adjust]}: the hinge is fitted on top of the adjustment terms)")

    #step 5: does the class gap depend on the adjustment variable?
    by_value = None
    if adjust == "slot":
        '''# fit the additive model
        add = fit_mixed(f"y ~ {TIME_TERM} * C(group) + C(split)", df)
        het = fit_mixed(f"y ~ {TIME_TERM} * C(group) + C(split) * C(group)", df)
        stat, dof, p = lrt(het, add)   # does the class gap vary by slot beyond what time explains?
        print(f"\nInteraction test (does the class gap vary by slot beyond what time explains?): chi2={stat:.2f}, df={dof}, p={p:.3g}")'''
        # one extra gap slope per slot (reference slot excluded, so nothing is redundant)
        ref = df["split"].mode()[0]
        is_case = (df["group"] == sorted(df["group"].unique())[1]).astype(float)
        d, slope_cols = df.copy(), []
        for i, k in enumerate(sorted(df["split"].unique())):
            if k != ref:
                d[f"gslope_{i}"] = (df["split"] == k) * is_case * df["time"]
                slope_cols.append(f"gslope_{i}")
        het_slope = fit_mixed(f"y ~ {TIME_TERM} * C(group){ADJ} + " + " + ".join(slope_cols), d)
        stat, dof, p = lrt(het_slope, full)
        print(f"\nDoes the gap's time trend differ by slot? chi2={stat:.2f}, df={dof}, p={p:.3g}")
    elif adjust in ("n_quest_add", "n_quest_int"):
        add_terms, _, _ = adjustment_spec("n_quest_add", n_form)
        int_terms, _, _ = adjustment_spec("n_quest_int", n_form)
        f_add = full if adjust == "n_quest_add" else fit_mixed(f"y ~ {TIME_TERM} * C(group){add_terms}", df)
        f_int = full if adjust == "n_quest_int" else fit_mixed(f"y ~ {TIME_TERM} * C(group){int_terms}", df)
        stat, dof, p = lrt(f_int, f_add)
        print(f"\nDoes the separation depend on the number of questionnaires? "
              f"(n_quest x class vs additive n_quest) chi2={stat:.2f}, df={dof}, p={p:.3g}")
        if adjust == "n_quest_int":
            print("\nGap at a fixed number of questionnaires (only where that number is observed):")
            by_value = curves_by_value(full_reml, df, adjust, values=n_values)
            for v, cv in by_value.items():
                print(f"  n_quest = {v}: t in [{cv['time'].min():.2f}, {cv['time'].max():.2f}], "
                      f"significant from t ≈ {_first_onset(cv):.2f}")

    if path:
        print("\nSaved plot:", plot_results(df, full_reml, curve, groups, profile, path=path,
                                            model_split=adjust, by_value=by_value))

    # Step 5 (slow; uncomment for robustness checks)
    if run_permutation_test:
        obs, p_perm = permutation_lrt(df, TIME_TERM,n_perm=n_perm, adj=ADJ); print(f"Permutation p = {p_perm:.3g}")
    if run_bootstrap:
        ci_boot, _ = cluster_bootstrap_tau(df, TIME_TERM+ADJ,n_boot=n_boot); print(f"Bootstrap tau CI: {ci_boot}")



def compare_models(raw, columns=preset_columns, logit=True, eps=1e-4, path="model_comparison.png",
                    baseline_quantile=0.05, max_time=None, n_sim=10000, matched_sets=(False, True), nested_comparison=False,
                    models=None, n_form="log", re_center="mean"):
    """Fit several models on the SAME data and compare them.

    logit              model logit(probability) instead of the raw probability
    eps                clip probabilities away from 0/1 before the logit
    baseline_quantile  baseline time for the "growth of the gap" curves (quantile of time)
    max_time           keep only observations with time < max_time (0 = pre-diagnosis only);
                       None keeps the full time axis

    matched_sets       which random-effect structures to run: False = subject only (correlated
                       intercept and slope), True = subject nested in matched set (needs a "set"
                       column in `columns`) PLUS its fair baseline: the same subject part without
                       the set ("_uncorr" models; statsmodels cannot keep the intercept-slope
                       correlation in the matched model, so the baseline drops it too)
    re_center          time origin of the random slope: "mean" (default), a number, or None
                       (= diagnosis). With the mean, dropping the correlation costs little fit and
                       AIC is comparable across random-effect structures; from diagnosis the
                       uncorrelated models fit much worse for that reason alone.
    models             names from the spec list below to run (default: all whose columns exist)
    n_form             form of the n_quest term in the n_quest models: "log", "linear", "cat"
    nested_comparison  LRTs between nested pairs (e.g. n_quest x class vs additive n_quest)

    Each model gets two difference curves: the gap itself, and its growth since the baseline.
    Adjusted models' curves are averaged over slot / n_quest by P(value | t).
    """

    SPLINE_4 = f"cr(time, df=4, constraints='center')"
    SPLINE_8 = f"cr(time, df=8, constraints='center')"
    QUADRATIC = "(center(time) + I(center(time)**2))"
    specs = {                         # name -> (time term, adjustment, random-effect structure)
        "spline_4":                 (SPLINE_4, None, "subject"),
        "spline_4_w_slot":          (SPLINE_4, "slot", "subject"),
        "spline_4_w_nq_add":        (SPLINE_4, "n_quest_add", "subject"),
        "spline_4_w_nq_int":        (SPLINE_4, "n_quest_int", "subject"),
        "spline_4_uncorr":          (SPLINE_4, None, "subject_uncorr"),
        "spline_4_w_slot_uncorr":   (SPLINE_4, "slot", "subject_uncorr"),
        "spline_4_w_nq_add_uncorr": (SPLINE_4, "n_quest_add", "subject_uncorr"),
        "spline_4_w_nq_int_uncorr": (SPLINE_4, "n_quest_int", "subject_uncorr"),
        "spline_4_w_match":         (SPLINE_4, None, "set"),
        "spline_4_w_slot_match":    (SPLINE_4, "slot", "set"),
        "spline_4_w_nq_add_match":  (SPLINE_4, "n_quest_add", "set"),
        "spline_4_w_nq_int_match":  (SPLINE_4, "n_quest_int", "set"),
        "spline_8":                 (SPLINE_8, None, "subject"),
        "spline_8_uncorr":          (SPLINE_8, None, "subject_uncorr"),
        "spline_8_w_match":         (SPLINE_8, None, "set"),
        "quadratic_w_slot":         (QUADRATIC, "slot", "subject"),
        "quadratic":                (QUADRATIC, None, "subject"),
    }
    # same fixed effects, different random effects: (correlated subject, uncorrelated subject, set)
    re_triples = [("spline_4", "spline_4_uncorr", "spline_4_w_match"),
                  ("spline_4_w_slot", "spline_4_w_slot_uncorr", "spline_4_w_slot_match"),
                  ("spline_4_w_nq_add", "spline_4_w_nq_add_uncorr", "spline_4_w_nq_add_match"),
                  ("spline_4_w_nq_int", "spline_4_w_nq_int_uncorr", "spline_4_w_nq_int_match"),
                  ("spline_8", "spline_8_uncorr", "spline_8_w_match")]
    nested_pairs = [("spline_4_w_slot", "spline_4"),               # (bigger, smaller) model
                    ("spline_4_w_nq_add", "spline_4"),
                    ("spline_4_w_nq_int", "spline_4_w_nq_add"),
                    ("spline_4_w_slot_match", "spline_4_w_match"),
                    ("spline_4_w_nq_add_match", "spline_4_w_match"),
                    ("spline_4_w_nq_int_match", "spline_4_w_nq_add_match"),
                    ("quadratic_w_slot", "quadratic")]

    # --- data -----------------------------------------------------------------
    COLUMNS = {v: k for k, v in columns.items()}
    df = raw.rename(columns=COLUMNS)[list(COLUMNS.values())].dropna()
    df["time"] = df["time"].astype(float)
    if max_time is not None:
        df = df[df["time"] < max_time].copy()
    if logit:
        p = df["y"].clip(eps, 1 - eps)
        df["y"] = np.log(p / (1 - p))
    assert df["group"].nunique() == 2, "expected exactly two classes"
    if True in matched_sets:
        assert "set" in df.columns, 'add "set": "case_control_group" to `columns`'
        assert df.groupby("subject")["set"].nunique().max() == 1, "each subject must be in one set"
    print(f"{df['subject'].nunique()} subjects, {len(df)} observations"
          + (f" (time < {max_time})" if max_time is not None else "") + "\n")

    # --- which models can run on these columns --------------------------------
    if models is not None:
        unknown = set(models) - set(specs)
        assert not unknown, f"unknown models {unknown}; choose from {list(specs)}"
        specs = {k: specs[k] for k in models}
    wanted = {"subject": False in matched_sets, "subject_uncorr": True in matched_sets,
              "set": True in matched_sets}
    skipped = {k: _adj_col(a) for k, (_, a, re) in specs.items()
               if (_adj_col(a) and _adj_col(a) not in df.columns) or not wanted[re]}
    specs = {k: v for k, v in specs.items() if k not in skipped}
    if skipped:
        print("Skipped (column missing or matched-set structure not requested): " + ", ".join(skipped) + "\n")

    baseline_time = df["time"].quantile(baseline_quantile)
    ref_split = df["split"].mode()[0] if "split" in df.columns else None

    # --- fits -----------------------------------------------------------------
    rows, fits, curves, ml_fits = [], {}, {}, {}
    for base in specs:
        tt, adjust, re = specs[base]
        name = base + (" + set" if re == "set" else "")
        adj, _, _ = adjustment_spec(adjust, n_form)
        print(f"Fitting {name} ...")
        kw = dict(re_structure=re, re_center=re_center)
        full = _fit(f"y ~ {tt} * C(group){adj}", df, **kw)                     # ML: AIC/BIC/LRT
        no_trend = _fit(f"y ~ {tt} + C(group){adj}", df, **kw)                 # gap constant in time
        full_reml = _fit(f"y ~ {tt} * C(group){adj}", df, reml=True, **kw)

        k, aic, bic = _information_criteria(full)
        stat, dof, p = lrt(full, no_trend)
        gap, crit_gap, groups = difference_curve(full_reml, df, n_sim=n_sim, model_split=adjust)
        growth, crit_growth, _ = difference_curve(full_reml, df, n_sim=n_sim, model_split=adjust,
                                                  baseline_time=baseline_time)
        fits[name] = (full_reml, adjust)
        ml_fits[base] = full
        curves[name] = {"gap": gap, "growth": growth}
        rows.append(dict(model=name, set_var=float(full.cov_re.iloc[0, 0]) if re == "set" else np.nan,
                         n_params=k, loglik=full.llf, AIC=aic, BIC=bic,
                         trend_chi2=stat, trend_df=dof, trend_p=p,
                         onset_gap=_first_onset(gap), onset_growth=_first_onset(growth)))

    table = pd.DataFrame(rows).set_index("model")
    re_center_value = next(iter(ml_fits.values())).re_center if ml_fits else np.nan

    # --- random-effect tests (same fixed effects; reuses the ML fits) -----------------------
    re_tests = []
    for corr, unc, mat in re_triples:
        if mat in ml_fits and unc in ml_fits:
            stat, dof, p = lrt_re(ml_fits[mat], ml_fits[unc], boundary=True)
            re_tests.append(dict(comparison=f"{mat} vs {unc}", tests="matched-set variance",
                                 chi2=stat, df=dof, p=p))
        if corr in ml_fits and unc in ml_fits:
            stat, dof, p = lrt_re(ml_fits[corr], ml_fits[unc])
            re_tests.append(dict(comparison=f"{corr} vs {unc}", tests="intercept-slope correlation",
                                 chi2=stat, df=dof, p=p))
    re_tests = pd.DataFrame(re_tests, columns=["comparison", "tests", "chi2", "df", "p"]
                            ).set_index("comparison")
    table["dAIC"] = table["AIC"] - table["AIC"].min()
    table["dBIC"] = table["BIC"] - table["BIC"].min()

    nested = None
    if nested_comparison:
        # --- nested comparisons (same random effects, more fixed effects; reuses the ML fits) --
        nested = []
        for big, small in nested_pairs:
            if big in ml_fits and small in ml_fits:
                stat, dof, p = lrt(ml_fits[big], ml_fits[small])
                nested.append(dict(comparison=f"{big} vs {small}", chi2=stat, df=dof, p=p))
        nested = pd.DataFrame(nested, columns=["comparison", "chi2", "df", "p"]).set_index("comparison")

    # --- report -----------------------------------------------------------------
    pd.set_option("display.width", 160)
    print("\nInformation criteria (ML fits, lower = better; dAIC < 2 ~ tie)")
    print(table[["n_params", "loglik", "AIC", "dAIC", "BIC", "dBIC"]].round(1).to_string())
    print(f"(random slope measured from t = {re_center_value:.2f})")
    if True in matched_sets and table["set_var"].notna().any():
        print("\nMatched-set variance (ML; near 0 = the sets add nothing)")
        print(table["set_var"].dropna().round(4).to_string())
    if len(re_tests):
        print("\nRandom-effect tests (same fixed effects)")
        print("  matched-set variance: '+ set' vs '_uncorr' differ ONLY by the set variance;")
        print("                        p is boundary-corrected (half the chi2(1) p-value)")
        print("  intercept-slope correlation: what the matched models give up (depends on re_center)")
        print(re_tests.round(4).to_string())
    print("\nDoes the class gap change over time?  (LRT: full model vs gap constant in time)")
    print(table[["trend_chi2", "trend_df", "trend_p"]].round(4).to_string())
    if nested_comparison:
        print("\nNested comparisons (does the bigger model improve the fit?)")
        print("  ..._nq_int vs ..._nq_add: does the separation depend on the number of questionnaires?")
        print(nested.round(4).to_string())
        print("\nSpline vs quadratic, and with vs without set, are NOT nested: compare them with AIC/BIC only.")
    print(f"\nOnset (first time the simultaneous band excludes 0)\n"
          f"  onset_gap:    the gap itself (adjusted models: averaged over slot / n_quest by P(value | t))\n"
          f"  onset_growth: growth of the gap since baseline t = {baseline_time:.2f}")
    print(table[["onset_gap", "onset_growth"]].round(2).to_string())

    # --- figure -----------------------------------------------------------------
    if path:
        _plot_comparison(df, fits, curves, groups, table, baseline_time, ref_split, path)
        print("\nSaved plot:", path)
    return table, curves, nested, re_tests

def compare_range(raw, train_intervals, eval_interval, columns=preset_columns, time_model="spline",
                  spline_df=4, model_split=True, matched_sets=False, logit=True, eps=1e-4,
                  n_folds=5, seed=0, baseline_quantile=0.05, n_sim=10000, min_slot_obs=5,
                  path="range_comparison.png", adjust="auto", n_form="log", re_center="mean"):
    """Train ONE model on several time intervals and compare them on the same evaluation interval.

    train_intervals  list of (lo, hi): train on observations with lo <= time < hi (None = open end)
    eval_interval    (t_min, t_max): every version is scored on observations in this interval
    time_model       "spline" or "quadratic"
    adjust           None, "slot", "n_quest_add" or "n_quest_int"; "auto" = "slot" if model_split
    n_form           form of the n_quest term: "log", "linear" or "cat"
    matched_sets     True = subjects nested in matched sets (needs a "set" column)
    re_center        time origin of the random slope ("mean" of each training set, a number, or
                     None = diagnosis); the held-out likelihood uses the same origin

    Scoring is cross-validated: participants (or whole matched sets) are split into `n_folds`
    folds; each version is fitted on the other folds (restricted to its training interval) and
    scored on the held-out participants' observations inside eval_interval. Higher held-out
    log-likelihood = better description of new people's data in that interval.
    The figure shows each version (fitted on all participants) over the evaluation interval.
    """
    adjust = _resolve_adjust(adjust, model_split)
    ADJ, adj_col, categorical = adjustment_spec(adjust, n_form)
    # categorical adjustments (slot, C(n_quest)) need every level estimable in both classes
    restrict_levels = adj_col if categorical else None

    # --- data -----------------------------------------------------------------
    COLUMNS = {v: k for k, v in columns.items()}
    df = raw.rename(columns=COLUMNS)[list(COLUMNS.values())].dropna()
    df["time"] = df["time"].astype(float)
    if logit:
        p = df["y"].clip(eps, 1 - eps)
        df["y"] = np.log(p / (1 - p))
    assert df["group"].nunique() == 2, "expected exactly two classes"
    if matched_sets:
        assert "set" in df.columns, 'add "set": "case_control_group" to `columns`'
        assert df.groupby("subject")["set"].nunique().max() == 1, "each subject must be in one set"
    _check_adjust_column(df, adjust)

    # fixed spline boundaries over the whole time axis, so a model trained on one interval
    # can be evaluated on another (inner knots still come from the training data)
    lo_b, hi_b = df["time"].min() - 0.01, df["time"].max() + 0.01
    if time_model == "spline":
        TT = f"cr(time, df={spline_df}, constraints='center', lower_bound={lo_b:.6f}, upper_bound={hi_b:.6f})"
    elif time_model == "quadratic":
        TT = "(center(time) + I(center(time)**2))"
    else:
        raise ValueError("time_model must be 'spline' or 'quadratic'")
    formula = f"y ~ {TT} * C(group){ADJ}"
    labels = [_interval_label(iv) for iv in train_intervals]
    eval_label = _interval_label(eval_interval)

    def restrict(d, iv):
        d = d[_in_interval(d, iv)]
        if restrict_levels:
            ok = _estimable_slots(d, min_slot_obs, col=restrict_levels)
            return d[d[restrict_levels].isin(ok)], ok
        return d, None

    # --- folds: whole participants (or whole matched sets), stratified by class --------------
    unit = "set" if matched_sets else "subject"
    unit_class = df.groupby(unit)["group"].max()
    rng = np.random.default_rng(seed)
    fold_of = {}
    for _, units in unit_class.groupby(unit_class):
        for i, u in enumerate(rng.permutation(units.index.values)):
            fold_of[u] = i % n_folds
    fold = df[unit].map(fold_of).values

    # --- cross-validated scores ----------------------------------------------------------------
    scores, n_dropped = [], 0
    for k in range(n_folds):
        train_all = df[fold != k]
        test = df[(fold == k) & _in_interval(df, eval_interval)]
        parts = [restrict(train_all, iv) for iv in train_intervals]
        if restrict_levels:        # score every version on the SAME observations
            common = set.intersection(*[ok for _, ok in parts])
            n_dropped += int((~test[restrict_levels].isin(common)).sum())
            test = test[test[restrict_levels].isin(common)]
        for lab, (tr, _) in zip(labels, parts):
            res = _fit(formula, tr, matched_sets=matched_sets, re_center=re_center)
            ll, sse = _heldout_loglik(res, test, matched_sets)
            scores.append(dict(train=lab, fold=k, n_eval=len(test), ll=ll, sse=sse))
        print(f"fold {k + 1}/{n_folds} done")
    scores = pd.DataFrame(scores)

    per_fold = scores.pivot(index="fold", columns="train", values="ll")[labels]
    summary = scores.groupby("train", sort=False).agg(ll=("ll", "sum"), sse=("sse", "sum"),
                                                      n_eval=("n_eval", "sum"))
    summary["ll_per_obs"] = summary["ll"] / summary["n_eval"]
    summary["d_ll"] = summary["ll"] - summary["ll"].max()
    summary["rmse"] = np.sqrt(summary["sse"] / summary["n_eval"])
    summary["folds_best"] = per_fold.idxmax(axis=1).value_counts().reindex(labels, fill_value=0)
    best = summary["ll"].idxmax()
    summary["se_d_ll"] = [np.nan if lab == best else
                          np.std(per_fold[lab] - per_fold[best], ddof=1) * np.sqrt(n_folds)
                          for lab in summary.index]

    # --- fits on all participants, for the figure and the onsets -------------------------------
    fits = {}
    for lab, iv in zip(labels, train_intervals):
        tr, ok = restrict(df, iv)
        fits[lab] = (_fit(formula, tr, reml=True, matched_sets=matched_sets, re_center=re_center), ok, len(tr))
    ev = df[_in_interval(df, eval_interval)]
    if restrict_levels:
        ev = ev[ev[restrict_levels].isin(set.intersection(*[ok for _, ok, _ in fits.values()]))]
    ref_split = ev["split"].mode()[0] if adjust == "slot" else None
    baseline_time = ev["time"].quantile(baseline_quantile)
    curves = {}
    for lab, (res, _, n_tr) in fits.items():
        gap, _, groups = difference_curve(res, ev, n_sim=n_sim, model_split=adjust)
        growth, _, _ = difference_curve(res, ev, n_sim=n_sim, model_split=adjust,
                                        baseline_time=baseline_time)
        curves[lab] = {"gap": gap, "growth": growth}
        summary.loc[lab, "n_train"] = n_tr
        summary.loc[lab, "onset_gap"] = _first_onset(gap)
        summary.loc[lab, "onset_growth"] = _first_onset(growth)

    # --- report --------------------------------------------------------------------------------
    pd.set_option("display.width", 160)
    re_txt = "subject nested in matched set" if matched_sets else "subject"
    print(f"\nModel: {time_model}{' + ' + ADJ_LABEL[adjust] if adjust else ''}, random effects: {re_txt}")
    print(f"Evaluation interval {eval_label}: {summary['n_eval'].iloc[0]} held-out observations"
          f" over {n_folds} folds" + (f" ({n_dropped} dropped: {restrict_levels} levels not estimable "
                                       f"in every training interval)" if n_dropped else ""))
    print("\nHeld-out fit on the evaluation interval (higher ll = better; d_ll = difference to the best,"
          "\nse_d_ll = its cross-validation standard error; rmse of the population prediction)")
    cols_out = ["n_train", "ll", "ll_per_obs", "d_ll", "se_d_ll", "folds_best", "rmse"]
    print(summary[cols_out].round(3).to_string())
    print(f"\nOnsets within the evaluation interval (fits on all participants; growth baseline "
          f"t = {baseline_time:.2f})")
    print(summary[["onset_gap", "onset_growth"]].round(2).to_string())
    for lab, iv in zip(labels, train_intervals):
        lo, hi = iv
        if (lo is not None and (eval_interval[0] is None or eval_interval[0] < lo)) or \
           (hi is not None and (eval_interval[1] is None or eval_interval[1] > hi)):
            print(f"Note: {lab} does not cover the evaluation interval, so part of its "
                  f"evaluation is extrapolation.")

    if path:
        _plot_range(ev, fits, curves, groups, summary, train_intervals, labels, eval_interval,
                    baseline_time, ref_split, adjust, path)
        print("\nSaved plot:", path)
    return summary, scores, curves
