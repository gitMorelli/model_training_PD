
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from .helpers import _adj_col, simulate_data, _slot_weights, _group_slot_weights, significant_intervals, _information_criteria, _interval_label
from .helpers import _in_interval, _estimable_slots, _heldout_loglik
from .fit import fit_mixed, fit_full_and_null, _fit
from .changepoint import changepoint_profile, cluster_bootstrap_tau
from .tests import lrt, permutation_lrt
from .plot import plot_results, _plot_comparison
from .difference import difference_curve

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
def run_analysis(raw, columns=preset_columns, logit=True, eps=1e-4, path=None, spline_df=4,
                 run_permutation_test=False, n_perm=200, run_bootstrap=False, n_boot=200, time_model="spline", model_split=True,
                 use_baseline=True):
    '''logit = True            # model logit(probability_1) instead of the raw probability
    eps = 1e-4                # clip probabilities away from 0/1 before the logit
    '''
    SPLINE_DF = spline_df  # flexibility of the time trend; 3-5 is typical, check sensitivity
    if time_model == "spline":
        TIME_TERM = f"cr(time, df={spline_df}, constraints='center')"
    elif time_model == "quadratic":
        TIME_TERM = "(center(time) + I(center(time)**2))"
    else:
        raise ValueError("time_model must be 'spline' or 'quadratic'")
    if model_split:
        ADJ = " + C(split) * C(group)"
    else:
        ADJ = ""

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
    # df = simulate_data()              # <- uncomment to run on simulated data instead
    print(f"{df['subject'].nunique()} subjects, {len(df)} observations\n")

    # Steps 1-2: models and global test
    full, null = fit_full_and_null(df, TIME_TERM, adj=ADJ)
    stat, dof, p = lrt(full, null)
    what = "does the gap change over time, within slot?" if model_split else "class affects level or shape"
    print(f"Global LRT ({what}): chi2={stat:.2f}, df={dof}, p={p:.3g}")

    # Step 3: difference curve (refit with REML for less biased variance components)
    full_reml = fit_mixed(f"y ~ {TIME_TERM} * C(group){ADJ}", df, reml=True)
    baseline_time = df["time"].quantile(0.05) if use_baseline else None
    curve, crit, groups = difference_curve(full_reml, df, model_split=model_split, baseline_time=baseline_time)
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

    #step 5: interaction between split and group (optional; only if model_split=True)
    if model_split:
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

    if path:
        print("\nSaved plot:", plot_results(df, full_reml, curve, groups, profile, path=path, model_split=model_split))

    # Step 5 (slow; uncomment for robustness checks)
    if run_permutation_test:
        obs, p_perm = permutation_lrt(df, TIME_TERM,n_perm=n_perm, adj=ADJ); print(f"Permutation p = {p_perm:.3g}")
    if run_bootstrap:
        ci_boot, _ = cluster_bootstrap_tau(df, TIME_TERM+ADJ,n_boot=n_boot); print(f"Bootstrap tau CI: {ci_boot}")



def compare_models(raw, columns=preset_columns, logit=True, eps=1e-4, path="model_comparison.png",
                    baseline_quantile=0.05, max_time=None, n_sim=10000, matched_sets=(False, True), nested_comparison=False):
    """Fit several models on the SAME data and compare them.
 
    logit              model logit(probability) instead of the raw probability
    eps                clip probabilities away from 0/1 before the logit
    baseline_quantile  baseline time for the "growth of the gap" curves (quantile of time)
    max_time           keep only observations with time < max_time (0 = pre-diagnosis only);
                       None keeps the full time axis
 
    matched_sets       which random-effect structures to run: False = subject only,
                       True = subject nested in matched set (needs a "set" column in `columns`)
 
    Each model gets two difference curves: the gap itself, and its growth since the baseline.
    """

    SPLINE_4 = f"cr(time, df=4, constraints='center')"
    SPLINE_8 = f"cr(time, df=8, constraints='center')"
    QUADRATIC = "(center(time) + I(center(time)**2))"
    SLOT = " + C(split) * C(group)"
    specs = {                                   # name -> (time term, adjust for slot?, matched sets)
        "spline_4":           (SPLINE_4, False, False),
        "spline_4_w_slot":    (SPLINE_4, True, False),
        "spline_4_w_match": (SPLINE_4, False, True),
        "spline_4_w_slot_match":    (SPLINE_4, True, True),
        "spline_8":         (SPLINE_8, False, False),
        "spline_8_w_match":  (SPLINE_8, False, True),
        "quadratic_w_slot": (QUADRATIC, True, False),
        "quadratic":        (QUADRATIC, False, False),
    }
    nested_pairs = [("spline_w_slot", "spline"), ("quadratic_w_slot", "quadratic")]
 
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
 
    baseline_time = df["time"].quantile(baseline_quantile)
    ref_split = df["split"].mode()[0]
 
    # --- fits -----------------------------------------------------------------
    rows, fits, curves = [], {}, {}
    for base in specs:
        tt, slot, ms = specs[base]
        name = base + (" + set" if ms else "")
        adj = SLOT if slot else ""
        print(f"Fitting {name} ...")
        full = _fit(f"y ~ {tt} * C(group){adj}", df, matched_sets=ms)          # ML: AIC/BIC/LRT
        no_trend = _fit(f"y ~ {tt} + C(group){adj}", df, matched_sets=ms)      # gap constant in time
        full_reml = _fit(f"y ~ {tt} * C(group){adj}", df, reml=True, matched_sets=ms)
 
        k, aic, bic = _information_criteria(full)
        stat, dof, p = lrt(full, no_trend)
        gap, crit_gap, groups = difference_curve(full_reml, df, n_sim=n_sim, model_split=slot,
                                                 ref_split=ref_split)
        growth, crit_growth, _ = difference_curve(full_reml, df, n_sim=n_sim, model_split=slot,
                                                  ref_split=ref_split, baseline_time=baseline_time)
        iv_gap, iv_growth = significant_intervals(gap), significant_intervals(growth)
        fits[name] = (full_reml, slot)
        curves[name] = {"gap": gap, "growth": growth}
        rows.append(dict(model=name, set_var=float(full.cov_re.iloc[0, 0]) if ms else np.nan,
                         n_params=k, loglik=full.llf, AIC=aic, BIC=bic,
                         trend_chi2=stat, trend_df=dof, trend_p=p,
                         onset_gap=iv_gap[0][0] if iv_gap else np.nan,
                         onset_growth=iv_growth[0][0] if iv_growth else np.nan))
 
    table = pd.DataFrame(rows).set_index("model")
    table["dAIC"] = table["AIC"] - table["AIC"].min()
    table["dBIC"] = table["BIC"] - table["BIC"].min()

    if nested_comparison:
        # --- nested comparisons (same random effects, more fixed effects) ---------
        nested = []
        for ms in matched_sets:
            sfx = " + set" if ms else ""
            for big, small in nested_pairs:
                f_big = _fit(f"y ~ {specs[big][0]} * C(group){SLOT}", df, matched_sets=ms)
                f_small = _fit(f"y ~ {specs[small][0]} * C(group)", df, matched_sets=ms)
                stat, dof, p = lrt(f_big, f_small)
                nested.append(dict(comparison=f"{big}{sfx} vs {small}{sfx}", chi2=stat, df=dof, p=p))
        nested = pd.DataFrame(nested).set_index("comparison")
 
    # --- report -----------------------------------------------------------------
    pd.set_option("display.width", 160)
    print("\nInformation criteria (ML fits, lower = better; dAIC < 2 ~ tie)")
    print(table[["n_params", "loglik", "AIC", "dAIC", "BIC", "dBIC"]].round(1).to_string())
    if True in matched_sets:
        print("\nMatched-set variance (ML; near 0 = the sets add nothing)")
        print(table["set_var"].dropna().round(4).to_string())
    print("\nDoes the class gap change over time?  (LRT: full model vs gap constant in time)")
    print(table[["trend_chi2", "trend_df", "trend_p"]].round(4).to_string())
    if nested_comparison:
        print("\nNested comparisons (do the slot terms C(split) * C(group) improve the fit?)")
        print(nested.round(4).to_string())
        print("\nSpline vs quadratic, and with vs without set, are NOT nested: compare them with AIC/BIC only.")
    print(f"\nOnset (first time the simultaneous band excludes 0)\n"
          f"  onset_gap:    the gap itself (slot models: at reference slot {ref_split})\n"
          f"  onset_growth: growth of the gap since baseline t = {baseline_time:.2f}")
    print(table[["onset_gap", "onset_growth"]].round(2).to_string())
 
    # --- figure -----------------------------------------------------------------
    if path:
        _plot_comparison(df, fits, curves, groups, table, baseline_time, ref_split, path)
        print("\nSaved plot:", path)
    return table, curves

def compare_range(raw, train_intervals, eval_interval, columns=preset_columns, time_model="spline",
                  spline_df=4, model_split=True, matched_sets=False, logit=True, eps=1e-4,
                  n_folds=5, seed=0, baseline_quantile=0.05, n_sim=10000, min_slot_obs=5,
                  path="range_comparison.png"):
    """Train ONE model on several time intervals and compare them on the same evaluation interval.
 
    train_intervals  list of (lo, hi): train on observations with lo <= time < hi (None = open end)
    eval_interval    (t_min, t_max): every version is scored on observations in this interval
    time_model       "spline" or "quadratic";  model_split: add C(split) * C(group)
    matched_sets     True = subjects nested in matched sets (needs a "set" column)
 
    Scoring is cross-validated: participants (or whole matched sets) are split into `n_folds`
    folds; each version is fitted on the other folds (restricted to its training interval) and
    scored on the held-out participants' observations inside eval_interval. Higher held-out
    log-likelihood = better description of new people's data in that interval.
    The figure shows each version (fitted on all participants) over the evaluation interval.
    """
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
    if not model_split:
        df = df.drop(columns="split", errors="ignore")
 
    # fixed spline boundaries over the whole time axis, so a model trained on one interval
    # can be evaluated on another (inner knots still come from the training data)
    lo_b, hi_b = df["time"].min() - 0.01, df["time"].max() + 0.01
    if time_model == "spline":
        TT = f"cr(time, df={spline_df}, constraints='center', lower_bound={lo_b:.6f}, upper_bound={hi_b:.6f})"
    elif time_model == "quadratic":
        TT = "(center(time) + I(center(time)**2))"
    else:
        raise ValueError("time_model must be 'spline' or 'quadratic'")
    formula = f"y ~ {TT} * C(group)" + (" + C(split) * C(group)" if model_split else "")
    labels = [_interval_label(iv) for iv in train_intervals]
    eval_label = _interval_label(eval_interval)
 
    def restrict(d, iv):
        d = d[_in_interval(d, iv)]
        if model_split:
            ok = _estimable_slots(d, min_slot_obs)
            return d[d["split"].isin(ok)], ok
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
        if model_split:            # score every version on the SAME observations
            common = set.intersection(*[ok for _, ok in parts])
            n_dropped += int((~test["split"].isin(common)).sum())
            test = test[test["split"].isin(common)]
        for lab, (tr, _) in zip(labels, parts):
            res = _fit(formula, tr, matched_sets=matched_sets)
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
        fits[lab] = (_fit(formula, tr, reml=True, matched_sets=matched_sets), ok, len(tr))
    ev = df[_in_interval(df, eval_interval)]
    if model_split:
        ev = ev[ev["split"].isin(set.intersection(*[ok for _, ok, _ in fits.values()]))]
    ref_split = ev["split"].mode()[0] if model_split else None
    baseline_time = ev["time"].quantile(baseline_quantile)
    curves = {}
    for lab, (res, _, n_tr) in fits.items():
        gap, _, groups = difference_curve(res, ev, n_sim=n_sim, model_split=model_split, ref_split=ref_split)
        growth, _, _ = difference_curve(res, ev, n_sim=n_sim, model_split=model_split,
                                        ref_split=ref_split, baseline_time=baseline_time)
        curves[lab] = {"gap": gap, "growth": growth}
        iv_gap, iv_growth = significant_intervals(gap), significant_intervals(growth)
        summary.loc[lab, "n_train"] = n_tr
        summary.loc[lab, "onset_gap"] = iv_gap[0][0] if iv_gap else np.nan
        summary.loc[lab, "onset_growth"] = iv_growth[0][0] if iv_growth else np.nan
 
    # --- report --------------------------------------------------------------------------------
    pd.set_option("display.width", 160)
    re_txt = "subject nested in matched set" if matched_sets else "subject"
    print(f"\nModel: {time_model}{' + slot' if model_split else ''}, random effects: {re_txt}")
    print(f"Evaluation interval {eval_label}: {summary['n_eval'].iloc[0]} held-out observations"
          f" over {n_folds} folds" + (f" ({n_dropped} dropped: slots not estimable in every "
                                       f"training interval)" if n_dropped else ""))
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
                    baseline_time, ref_split, model_split, path)
        print("\nSaved plot:", path)
    return summary, scores, curves
 
