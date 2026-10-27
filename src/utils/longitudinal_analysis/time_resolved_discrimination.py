"""
Time-resolved discrimination
============================
From how long before diagnosis does the model's score separate cases from controls?

For each time window before diagnosis:
  1. keep ONE score per subject (default: the chronologically latest one inside the window,
     i.e. what the model knew at the end of that window)
  2. AUC              cases vs controls, all subjects in the window
     matched C        case-control pairs WITHIN the same matched set only (respects the matching:
                      age, sex, ... cancel out); = P(case score > matched control score)
     conditional OR   (optional) conditional logistic regression on logit(p), per SD
  3. 95% CIs by bootstrapping whole matched sets (or subjects, if there are no sets).
     Pointwise (percentile) and simultaneous over all windows (max-|t| over windows).
  4. Onset = the window farthest from diagnosis from which the lower CI bound stays above
     0.5 (or `min_effect`) in EVERY window closer to diagnosis ("persistent separation").

Notes
-----
* `pick="latest"` with sliding windows [t, t + width) is the "as-of" score: at t years before
  diagnosis, each subject's latest score, provided it is at most `width` old.
* The scores should be out-of-sample (out-of-fold, CV split by subject or matched set);
  in-sample scores inflate the separation, mostly far from diagnosis.
* Time units are those of the input time column. `time_sign="auto"` treats the time column as
  negative before diagnosis if most values are negative (e.g. -5 = 5 years before), otherwise
  as positive time before diagnosis. Internally `tbd` = time before diagnosis (>= 0 before).

Requires: numpy, pandas, scipy, matplotlib; statsmodels only for the conditional OR.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats

preset_columns = {             # internal name -> your column name
    "subject": "unique_id",
    "group": "true_label",
    "time": "case_dt",
    "y": "probability_1",
    "split": "slot",
    "set": "set",
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def prepare(raw, columns=preset_columns, time_sign="auto", eps=1e-6):
    """Rename columns and add: case (0/1), p (probability), lp (logit p), tbd (time before dx)."""
    rename = {v: k for k, v in columns.items() if v in raw.columns}
    df = raw.rename(columns=rename)[list(rename.values())].copy()
    df = df.dropna(subset=["subject", "group", "time", "y"]).reset_index(drop=True)
    groups = sorted(df["group"].unique())
    assert len(groups) == 2, "expected exactly two classes"
    df["case"] = (df["group"] == groups[1]).astype(int)

    df["time"] = df["time"].astype(float)
    if time_sign == "auto":
        time_sign = -1 if (df["time"] < 0).mean() > 0.5 else 1
    df["tbd"] = time_sign * df["time"]

    df["p"] = df["y"].astype(float)
    pc = df["p"].clip(eps, 1 - eps)
    df["lp"] = np.log(pc / (1 - pc))

    matched = "set" in df.columns
    if matched:
        df = df.dropna(subset=["set"]).reset_index(drop=True)
        assert df.groupby("subject")["set"].nunique().max() == 1, "each subject must be in one set"
    else:
        df["set"] = df["subject"]
    assert df.groupby("subject")["case"].nunique().max() == 1, "a subject changes class"
    return df, groups, time_sign, matched


def make_windows(tbd, width=None, step=None, t_min=0.0, t_max=None, n_windows=10):
    """Windows [lo, hi) in time-before-diagnosis units. Default: n_windows equal, non-overlapping
    windows from t_min to the 95th percentile of tbd. step < width gives sliding windows."""
    tbd = np.asarray(tbd)
    if t_max is None:
        t_max = np.quantile(tbd[tbd >= t_min], 0.95)
    if width is None:
        width = (t_max - t_min) / n_windows
    step = width if step is None else step
    starts = np.arange(t_min, max(t_max - width, t_min) + 1e-9, step)
    return [(float(a), float(a + width)) for a in starts]


def span_subjects(df, near, far):
    """Subjects observed both at tbd <= near and at tbd >= far (composition sensitivity analysis)."""
    g = df.groupby("subject")["tbd"]
    ok = (g.min() <= near) & (g.max() >= far)
    return set(ok.index[ok])


def _select(df, lo, hi, pick):
    """One row per subject among observations with lo <= tbd < hi."""
    w = df[(df["tbd"] >= lo) & (df["tbd"] < hi)]
    if w.empty:
        return w
    if pick in ("latest", "closest"):          # chronologically latest = smallest tbd
        return w.loc[w.groupby("subject")["tbd"].idxmin()]
    if pick in ("earliest", "farthest"):
        return w.loc[w.groupby("subject")["tbd"].idxmax()]
    if pick == "mean":
        return w.groupby("subject", as_index=False).agg(
            case=("case", "first"), set=("set", "first"),
            p=("p", "mean"), lp=("lp", "mean"), tbd=("tbd", "mean"))
    raise ValueError("pick must be 'latest', 'earliest' or 'mean'")


# ----------------------------------------------------------------------------
# Metrics (vectorised over bootstrap replicates)
# ----------------------------------------------------------------------------
def _weighted_auc(score, case, W):
    """AUC (ties = 1/2) for each row of weights W (R, n). Unit weights give the usual AUC."""
    order = np.argsort(score, kind="mergesort")
    s, c, Ws = score[order], case[order].astype(float), W[:, order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]          # groups of tied scores
    cw = np.add.reduceat(Ws * c, starts, axis=1)               # case weight per score value
    kw = np.add.reduceat(Ws * (1 - c), starts, axis=1)         # control weight per score value
    below = np.cumsum(kw, axis=1) - kw                         # control weight strictly below
    num = (cw * (below + 0.5 * kw)).sum(axis=1)
    den = cw.sum(axis=1) * kw.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _pair_stats(sel, win_obs, set_pos, S, pairing):
    """Per matched set: sum of case-control concordances (C) and number of pairs (P).

    pairing="subject": one score per subject (the window selection).
    pairing="slot":    a case and a control are compared only on questionnaires (slots) BOTH
                       completed in the window; the pair's concordance is the mean over those
                       slots. Removes differences in how many questionnaires the score uses.
    """
    if pairing == "subject":
        a = sel.loc[sel["case"] == 1, ["set", "subject", "p"]]
        b = sel.loc[sel["case"] == 0, ["set", "subject", "p"]]
        m = a.merge(b, on="set", suffixes=("_1", "_0"))
        m["c"] = (m["p_1"] > m["p_0"]) + 0.5 * (m["p_1"] == m["p_0"])
    elif pairing == "slot":
        a = win_obs.loc[win_obs["case"] == 1, ["set", "subject", "split", "p"]]
        b = win_obs.loc[win_obs["case"] == 0, ["set", "subject", "split", "p"]]
        m = a.merge(b, on=["set", "split"], suffixes=("_1", "_0"))
        m["c"] = (m["p_1"] > m["p_0"]) + 0.5 * (m["p_1"] == m["p_0"])
        m = m.groupby(["set", "subject_1", "subject_0"], as_index=False)["c"].mean()
    else:
        raise ValueError("pairing must be 'subject' or 'slot'")
    si = set_pos.reindex(m["set"]).values.astype(int)
    C = np.bincount(si, weights=m["c"].values.astype(float), minlength=S)
    P = np.bincount(si, minlength=S).astype(float)
    return C, P, len(m)


def _clogit(sel, mu, sd):
    """Conditional logistic regression of case on standardised logit(p), within matched sets."""
    from statsmodels.discrete.conditional_models import ConditionalLogit
    both = sel.groupby("set")["case"].transform("nunique") == 2
    d = sel[both]
    if d["case"].sum() < 5:
        return np.nan, np.nan
    x = ((d["lp"].values - mu) / sd)[:, None]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ConditionalLogit(d["case"].values, x, groups=d["set"].values).fit(disp=0)
        return float(r.params[0]), float(r.bse[0])
    except Exception:
        return np.nan, np.nan


def _bands(est, B, level, null=0.5):
    """Pointwise percentile CI, simultaneous (max-|t| over windows) CI, one-sided bootstrap p."""
    a = (1 - level) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pw_lo, pw_hi = np.nanquantile(B, a, axis=0), np.nanquantile(B, 1 - a, axis=0)
        se = np.nanstd(B, axis=0, ddof=1)
        ok = np.isfinite(est) & (se > 0)
        crit = np.nan
        if ok.any():
            z = np.nanmax(np.abs(B[:, ok] - est[ok]) / se[ok], axis=1)
            crit = float(np.nanquantile(z, level))
        p = (1 + np.nansum(B <= null, axis=0)) / (1 + np.sum(np.isfinite(B), axis=0))
    p = np.where(np.isfinite(est), p, np.nan)
    return dict(pw_lo=pw_lo, pw_hi=pw_hi, se=se, sim_lo=est - crit * se, sim_hi=est + crit * se,
                p_boot=p), crit


def onset(table, lower_col, threshold=0.5):
    """Farthest window from diagnosis such that it and every window closer to diagnosis have
    lower bound > threshold. Only windows entirely BEFORE diagnosis (lo >= 0) count, so windows
    after diagnosis can be shown for context. Returns that table row, or None."""
    t = table[table["lo"] >= 0].sort_values("center")   # closest to diagnosis first
    last = None
    for _, r in t.iterrows():
        if np.isfinite(r[lower_col]) and r[lower_col] > threshold:
            last = r
        else:
            break
    return last


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------
def time_resolved_discrimination(df, windows, matched=True, pick="latest", pairing="subject",
                                 n_boot=2000, level=0.95, min_n=10, clogit=True, seed=0):
    """Per-window AUC, matched concordance and conditional OR with bootstrap CIs.
    `df` comes from prepare(). Returns (table, info)."""
    sets = df["set"].unique()
    S = len(sets)
    set_pos = pd.Series(np.arange(S), index=sets)

    # bootstrap: resample whole matched sets (unmatched: subjects, stratified by class)
    rng = np.random.default_rng(seed)
    if matched:
        strata = [np.arange(S)]
    else:
        set_case = df.groupby("set")["case"].first().reindex(sets).values
        strata = [np.flatnonzero(set_case == k) for k in (0, 1)]
    M = np.zeros((n_boot, S))
    for st in strata:
        M[:, st] = rng.multinomial(len(st), np.full(len(st), 1 / len(st)), size=n_boot)

    mu, sd = df["lp"].mean(), df["lp"].std()       # common scale for the OR in every window
    nw = len(windows)
    B_auc, B_conc = np.full((n_boot, nw), np.nan), np.full((n_boot, nw), np.nan)
    rows = []
    for j, (lo, hi) in enumerate(windows):
        win_obs = df[(df["tbd"] >= lo) & (df["tbd"] < hi)]
        sel = _select(df, lo, hi, pick)
        n1 = int(sel["case"].sum()) if len(sel) else 0
        n0 = len(sel) - n1
        row = dict(lo=lo, hi=hi, center=(lo + hi) / 2, n_cases=n1, n_controls=n0,
                   n_pairs=0, n_sets=0, auc=np.nan, conc=np.nan, log_or=np.nan, se_log_or=np.nan)
        if n1 >= min_n and n0 >= min_n:
            p, c = sel["p"].values, sel["case"].values
            si = set_pos.reindex(sel["set"]).values.astype(int)
            row["auc"] = _weighted_auc(p, c, np.ones((1, len(p))))[0]
            B_auc[:, j] = _weighted_auc(p, c, M[:, si])
        if matched and len(sel):
            C, P, npairs = _pair_stats(sel, win_obs, set_pos, S, pairing)
            row["n_pairs"], row["n_sets"] = npairs, int((P > 0).sum())
            if row["n_sets"] >= min_n:
                row["conc"] = C.sum() / P.sum()
                with np.errstate(invalid="ignore", divide="ignore"):
                    B_conc[:, j] = (M @ C) / (M @ P)
            if clogit and row["n_sets"] >= min_n:
                row["log_or"], row["se_log_or"] = _clogit(sel, mu, sd)
        rows.append(row)

    table = pd.DataFrame(rows)
    crits = {}
    for name, B in [("auc", B_auc), ("conc", B_conc)]:
        bands, crits[name] = _bands(table[name].values, B, level)
        for k, v in bands.items():
            table[f"{name}_{k}"] = v
    z = stats.norm.ppf(0.5 + level / 2)
    table["or"] = np.exp(table["log_or"])
    table["or_lo"] = np.exp(table["log_or"] - z * table["se_log_or"])
    table["or_hi"] = np.exp(table["log_or"] + z * table["se_log_or"])
    info = dict(crit=crits, n_boot=n_boot, level=level, pick=pick, pairing=pairing,
                matched=matched, mu=mu, sd=sd)
    return table, info


def run_discrimination(raw, columns=preset_columns, windows=None, width=None, step=None,
                       t_min=0.0, t_max=None, n_windows=10, pick="latest", pairing="subject",
                       n_boot=2000, level=0.95, min_n=10, min_effect=0.6, clogit=True,
                       time_sign="auto", require_span=None, seed=0,
                       path="time_resolved_discrimination.png"):
    """End-to-end: prepare -> windows -> metrics -> onset report -> figure.

    windows       list of (lo, hi) in time-before-diagnosis units; or let width/step/t_max/n_windows
                  build them (step < width = sliding windows)
    t_min, t_max  window range in years BEFORE diagnosis; t_min < 0 adds windows after diagnosis
                  (shown for context, never used for the onset)
    pick          score kept per subject in a window: "latest" (chronologically), "earliest", "mean"
    pairing       matched C on one score per subject ("subject") or on common questionnaires ("slot")
    min_n         minimum cases AND controls (AUC) / informative matched sets (matched C) per window
    min_effect    second, "clinically relevant" onset threshold (lower bound > min_effect)
    require_span  (near, far): keep only subjects observed both within `near` and beyond `far`
                  before diagnosis - checks that the curve is not driven by who is observed when
    """
    df, groups, sign, matched = prepare(raw, columns, time_sign=time_sign)
    if pairing == "slot" and "split" not in df.columns:
        raise ValueError('pairing="slot" needs a "split" column in `columns`')
    df = df[df["tbd"] >= t_min] if windows is None else df
    if require_span is not None:
        keep = span_subjects(df, *require_span)
        df = df[df["subject"].isin(keep)]
    if windows is None:
        windows = make_windows(df["tbd"], width, step, t_min, t_max, n_windows)

    print(f"{df['subject'].nunique()} subjects ({df.groupby('case')['subject'].nunique().get(1, 0)} "
          f"cases), {df['set'].nunique() if matched else 'no'} matched sets, {len(df)} observations")
    print(f"time column read as {'negative' if sign < 0 else 'positive'} before diagnosis; "
          f"{len(windows)} windows of width {windows[0][1] - windows[0][0]:.3g}"
          + (f" (sliding, step {windows[1][0] - windows[0][0]:.3g})"
             if len(windows) > 1 and windows[1][0] < windows[0][1] else ""))
    print(f"score per subject and window: {pick}; matched pairs on: {pairing}; "
          f"bootstrap: {n_boot} resamples of {'matched sets' if matched else 'subjects'}\n")

    table, info = time_resolved_discrimination(df, windows, matched, pick, pairing, n_boot,
                                               level, min_n, clogit and matched, seed)
    info.update(time_sign=sign, groups=groups, min_effect=min_effect)

    show = ["lo", "hi", "n_cases", "n_controls", "auc", "auc_pw_lo", "auc_pw_hi"]
    if matched:
        show += ["n_sets", "n_pairs", "conc", "conc_pw_lo", "conc_pw_hi", "conc_p_boot"]
    if clogit and matched:
        show += ["or", "or_lo", "or_hi"]
    pd.set_option("display.width", 200)
    print(table[show].round(3).to_string(index=False))

    print("\nOnset = farthest window from diagnosis from which separation persists up to diagnosis")
    onsets = {}
    metrics = [("conc", "matched C")] * matched + [("auc", "AUC")]
    for m, label in metrics:
        print(f"  {label}: simultaneous band crit = {info['crit'][m]:.2f} (vs 1.96 pointwise)")
        for band in ("sim", "pw"):
            for thr in (0.5, min_effect):
                r = onset(table, f"{m}_{band}_lo", thr)
                onsets[(m, band, thr)] = r
                where = (f"window [{r['lo']:.2f}, {r['hi']:.2f}) before diagnosis"
                         if r is not None else "none")
                print(f"    {'simultaneous' if band == 'sim' else 'pointwise   '} lower bound > "
                      f"{thr:.2f}: {where}")
    info["onsets"] = onsets

    if path:
        plot_discrimination(table, info, path)
        print("\nSaved plot:", path)
    return table, info


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot_discrimination(table, info, path="time_resolved_discrimination.png"):
    """Top: matched C and AUC vs time (pointwise band, simultaneous band as thin lines).
    Middle (if available): conditional OR per SD. Bottom: subjects / informative sets per window."""
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#d9d8d3"
    col = {"conc": "#2a78d6", "auc": "#eb6834"}
    label = {"conc": "matched concordance", "auc": "AUC (all subjects)"}
    x = info["time_sign"] * table["center"].values
    has_or = np.isfinite(table["log_or"]).any()
    metrics = ["conc", "auc"] if info["matched"] else ["auc"]

    nrows = 3 if has_or else 2
    fig, ax = plt.subplots(nrows, 1, figsize=(8.5, 3.2 * nrows + 0.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2] + [1.2] * (nrows - 1)))

    a = ax[0]
    for m in metrics:
        a.fill_between(x, table[f"{m}_pw_lo"], table[f"{m}_pw_hi"], color=col[m], alpha=0.18, lw=0)
        a.plot(x, table[f"{m}_sim_lo"], color=col[m], lw=0.8, ls="--")
        a.plot(x, table[f"{m}_sim_hi"], color=col[m], lw=0.8, ls="--")
        a.plot(x, table[m], color=col[m], lw=2, marker="o", ms=5, mec="white", mew=1.2,
               label=label[m])
    a.axhline(0.5, color=muted, lw=1, ls=":")
    a.axhline(info["min_effect"], color=grid, lw=1)
    main = metrics[0]
    r = info["onsets"].get((main, "sim", 0.5))
    if r is not None:
        e1, e2 = sorted(info["time_sign"] * np.array([r["lo"], r["hi"]]))
        for a_ in ax:
            a_.axvspan(e1, e2, color=muted, alpha=0.10, lw=0, zorder=0)
        txt = (f"onset window: [{r['lo']:.3g}, {r['hi']:.3g}) before diagnosis\n"
               f"({label[main]}, simultaneous band > 0.5 from here on)")
    else:
        txt = f"no persistent separation ({label[main]}, simultaneous band)"
    a.text(0.01, 0.98, txt, transform=a.transAxes, va="top", fontsize=8, color=muted)
    a.set_ylabel("P(case score > control score)")
    a.set_title("Case vs control separation by time before diagnosis\n"
                "(shaded: pointwise 95% CI; dashed: simultaneous 95% band; dotted: chance)",
                fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="lower left")

    if has_or:
        a = ax[1]
        ok = np.isfinite(table["log_or"])
        a.errorbar(x[ok], table.loc[ok, "or"], yerr=[table.loc[ok, "or"] - table.loc[ok, "or_lo"],
                   table.loc[ok, "or_hi"] - table.loc[ok, "or"]], fmt="o", color=col["conc"],
                   ms=5, mec="white", mew=1.2, elinewidth=1.5, capsize=0)
        a.axhline(1, color=muted, lw=1, ls=":")
        a.set_yscale("log")
        a.set_ylabel("conditional OR\nper SD of logit(p)")

    a = ax[-1]
    a.plot(x, table["n_cases"], color=ink, lw=1.5, marker="o", ms=4, label="cases")
    a.plot(x, table["n_controls"], color=muted, lw=1.5, ls="--", marker="o", ms=4,
           label="controls")
    if info["matched"]:
        a.plot(x, table["n_sets"], color=col["conc"], lw=1.5, marker="o", ms=4,
               label="matched sets with a pair")
    a.set_ylabel("n per window")
    a.legend(frameon=False, fontsize=8)
    a.set_xlabel("time to diagnosis (window centre)")

    if (table["lo"] < 0).any():                    # windows after diagnosis present
        for a_ in ax:
            a_.axvline(0, color=ink, lw=1)
        ax[0].text(0, 0.02, " diagnosis", transform=ax[0].get_xaxis_transform(), fontsize=8,
                   color=ink)
    for a in ax:
        a.grid(alpha=0.6, color=grid, lw=0.6)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path