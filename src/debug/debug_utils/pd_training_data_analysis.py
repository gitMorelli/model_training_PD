from functools import cached_property
import pandas as pd
import numpy as np

####### Analyze PD data #########
N_PERIODS = 13
MODES = ("X", "text", "digit")
# pattern column -> name in long frame
PATTERN_COLS = {
    "rempli_pattern": "rempli",        # subject: filled / not filled
    "case_pattern": "case_rempli",     # matched case's filled value
    "grid_pattern": "grid",            # subject: data available / not
    "case_grid_pattern": "case_grid",  # matched case's availability
}
SUBJECT_CARRY = ["unique_id", "id", "group_id", "split", "case_control", "last_avail_q"]
 
def to_long(df, n_periods=N_PERIODS):
    """One row per (unique_id, period). Period k <- char k-1 of patterns,
    case_dt_dateq{k}, and q_{k}_num_{mode}. 'available' = period <= last_avail_q."""
    df = df.copy()
    ids = df["unique_id"].str.split("_", n=1, expand=True)
    df["id"], df["group_id"] = ids[0], ids[1]
 
    blocks = []
    for p in range(1, n_periods + 1):
        b = df[SUBJECT_CARRY].copy()
        b["period"] = p
        for src, dst in PATTERN_COLS.items():
            b[dst] = pd.to_numeric(df[src].str[p - 1], errors="coerce").astype("Int64")
        b["case_dt_dateq"] = df[f"case_dt_dateq{p}"]
        for m in MODES:
            col = pd.to_numeric(df[f"q_{p}_num_{m}"], errors="coerce")
            b[f"q_num_{m}"] = col.mask(col < 0)  # negatives are sentinel missing -> substituted with nans
        b["available"] = b["period"] <= b["last_avail_q"]
        blocks.append(b)
    return pd.concat(blocks, ignore_index=True)
 
# ---- registry -------------------------------------------------------------
PROPERTIES = {}
def prop(name=None):
    def deco(fn):
        PROPERTIES[name or fn.__name__] = fn
        return fn
    return deco
 
class Profiler:
    """Holds the wide df + a lazily computed, cached long frame.
    Subsets share the parent's long frame (filtered), so reshape runs once."""
    def __init__(self, df, _root_long=None, label="all"):
        self.df = df.reset_index(drop=True)
        self.label = label
        self._root_long = _root_long
 
    @cached_property
    def long(self):
        root = self._root_long if self._root_long is not None else to_long(self.df)
        ids = set(self.df["unique_id"])
        return root[root["unique_id"].isin(ids)].reset_index(drop=True)
 
    def subset(self, mask, label):
        return Profiler(self.df[mask], _root_long=self.long if self._root_long is None
                        else self._root_long, label=label)
 
def analyze(df_or_prof, by=None, properties=None):
    """Run properties over the whole frame, or broken down by column(s) in `by`."""
    p = df_or_prof if isinstance(df_or_prof, Profiler) else Profiler(df_or_prof)
    names = list(properties or PROPERTIES)
    run = lambda pr: {n: PROPERTIES[n](pr) for n in names}
    if not by:
        return {"all": run(p)}
    out = {}
    for key, idx in p.df.groupby(by, dropna=False).groups.items():
        label = "|".join(map(str, key)) if isinstance(key, tuple) else str(key)
        out[label] = run(p.subset(p.df.index.isin(idx), label))
    return out
 
# ---- example properties (you'll add more; these show both flavors) ---------
@prop()
def n_subjects(p):
    return {"n_rows": len(p.df), "n_unique_id": int(p.df["unique_id"].nunique())}
 
@prop()
def split_fraction(p):
    return p.df["split"].value_counts(normalize=True).round(4).to_dict()
 
@prop()
def case_control_balance(p):
    return p.df["case_control"].value_counts().sort_index().to_dict()
 
@prop()
def filled_periods_per_subject(p):  # long-based
    s = p.long.groupby("unique_id")["rempli"].sum()
    return s.describe().round(3).to_dict()

# ---- data-integrity: split leakage ----------------------------------------
def ids_in_multiple_splits(p, id_col="group_id", return_offenders=False):
    """Count ids that appear in more than one split.
    id_col:
      'group_id'  -> YY, the matched set (default; case+controls should share a split)
      'id'        -> XXXX, the subject id
      'unique_id' -> full XXXX_YY (flags a duplicated subject across splits)
      or any column name in p.df
    Derives id/group_id from unique_id, so it never triggers the long reshape."""
    df = p.df
    if id_col in ("group_id", "id"):
        parts = df["unique_id"].str.split("_", n=1, expand=True)
        key = (parts[1] if id_col == "group_id" else parts[0]).values
    else:
        key = df[id_col].values
 
    work = pd.DataFrame({"_key": key, "split": df["split"].values})
    splits_per_id = work.groupby("_key")["split"].nunique()
    multi = splits_per_id[splits_per_id > 1]
 
    out = {
        "id_col": id_col,
        "n_ids": int(splits_per_id.size),
        "n_multi_split": int(multi.size),
        "any_multi_split": bool(multi.size > 0),
    }
    if return_offenders:
        out["offenders"] = {
            str(k): sorted(map(str, g["split"].unique()))
            for k, g in work[work["_key"].isin(multi.index)].groupby("_key")
        }
    return out
 
@prop()
def group_split_leakage(p):
    """Matched-group (group_id) leakage across splits -- the case-control concern."""
    return ids_in_multiple_splits(p, id_col="group_id")

def test_ids_with_group_in_trainval(p, return_ids=False):
    ''' distinct unique_ids in test whose group_id also has a member in train or val. 
    It also returns n_test for context and any_leaking as a quick flag; pass return_ids=True for the offending id list'''
    df = p.df
    group_id = df["unique_id"].str.split("_", n=1, expand=True)[1]
    work = pd.DataFrame({"unique_id": df["unique_id"].values,
                         "group_id": group_id.values,
                         "split": df["split"].values})
    trainval_groups = set(work.loc[work["split"].isin(["train", "val"]), "group_id"])
    mask = (work["split"] == "test") & work["group_id"].isin(trainval_groups)
    leaking = work.loc[mask, "unique_id"].unique()
    out = {"n_test": int(work.loc[work["split"] == "test", "unique_id"].nunique()),
           "n_leaking": int(len(leaking)),
           "any_leaking": bool(len(leaking) > 0)}
    if return_ids:
        out["leaking_ids"] = sorted(map(str, leaking))
    return out

@prop()
def test_group_contamination(p):
    """Registry entry: test subjects whose matched group appears in train/val."""
    return test_ids_with_group_in_trainval(p)

# ---- matching ----------------
def check_matching(p, match_vars=("etudegp", "profq2", "lateralite"),
                   tol_vars=None, return_offenders=False):
    """Verify each matched group (group_id=YY) shares one value per matching var.
    match_vars: exact-match categoricals (all members must share one value -- which,
    since the case is in the group, means all equal the case).
    tol_vars: {var: tolerance} for numerics (e.g. relative_age); each member must lie
    within +/- tolerance of THE CASE's value in that group (case_control == 1).
    Reads only p.df, so no long reshape is triggered."""
    df = p.df
    gid = df["unique_id"].str.split("_", n=1, expand=True)[1]
    work = df.assign(_g=gid.values, _iscase=(df["case_control"] == 1).values)
    result = {"n_groups": int(work["_g"].nunique())}
    offenders = {}
    for v in match_vars:
        nuniq = work.groupby("_g")[v].nunique(dropna=False)
        bad = nuniq[nuniq > 1]
        result[v + "_n_mismatch"] = int(bad.size)
        if return_offenders and bad.size:
            offenders[v] = {str(g): sorted(map(str, work.loc[work._g == g, v].unique()))
                            for g in bad.index}
    if tol_vars:
        # max abs deviation of any member from the group's case value
        def max_dev_from_case(g, v):
            case_vals = g.loc[g["_iscase"], v]
            if case_vals.empty or pd.isna(case_vals.iloc[0]):
                return np.nan  # no usable case reference
            return (g[v] - case_vals.iloc[0]).abs().max()
        no_case = int((~work.groupby("_g")["_iscase"].any()).sum())
        result["n_groups_no_case_ref"] = no_case
        for v, tol in tol_vars.items():
            dev = work.groupby("_g").apply(lambda g: max_dev_from_case(g, v))
            bad = dev[dev > tol]
            result[v + "_n_out_of_tol"] = int(bad.size)
            if return_offenders and bad.size:
                offenders[v] = {str(g): round(float(dev[g]), 3) for g in bad.index}
    result["all_matched"] = bool(
        sum(result[k] for k in result if k.endswith(("_n_mismatch", "_n_out_of_tol"))) == 0)
    if return_offenders:
        result["offenders"] = offenders
    return result
 
@prop()
def matching_integrity(p):
    """Registry entry: exact-match check on the categorical matching variables."""
    return check_matching(p)

def group_size_distribution(p, return_groups_by_size=False):
    """Distribution of matched-group sizes. `size_counts` maps group_size -> number
    of groups with that many members (elements = distinct unique_id), sorted from
    largest size down. Reads only p.df, so no long reshape is triggered."""
    df = p.df
    gid = df["unique_id"].str.split("_", n=1, expand=True)[1]
    work = pd.DataFrame({"unique_id": df["unique_id"].values, "_g": gid.values})
    sizes = work.groupby("_g")["unique_id"].nunique()          # group_id -> size
    dist = sizes.value_counts().sort_index(ascending=False)    # size -> n_groups
    out = {
        "n_groups": int(sizes.size),
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "size_counts": {int(s): int(n) for s, n in dist.items()},
    }
    if return_groups_by_size:
        out["groups_by_size"] = {int(s): sorted(sizes[sizes == s].index)
                                 for s in dist.index}
    return out
 
@prop()
def group_sizes(p):
    """Registry entry: histogram of matched-group sizes."""
    return group_size_distribution(p)

def id_appearance_distribution(p, return_ids=False):
    """Within the given rows, how many times each id (XXXX) appears, summarized as
    `appearance_counts`: n_appearances -> how many ids appear that many times (ascending).
    Subset-agnostic: run via analyze(df, by=['split']) for the per-split breakdown.
    Reads only p.df, so no long reshape is triggered."""
    df = p.df
    xid = df["unique_id"].str.split("_", n=1, expand=True)[0]
    counts = pd.Series(xid.values).value_counts()        # id -> n_appearances
    dist = counts.value_counts().sort_index()            # n_appearances -> n_ids
    out = {
        "n_ids": int(counts.size),
        "n_rows": int(len(xid)),
        "appearance_counts": {int(k): int(v) for k, v in dist.items()},
    }
    if return_ids:  # only the reused ids (appear >1x) are worth listing
        out["ids_by_appearance"] = {int(k): sorted(counts[counts == k].index)
                                    for k in dist.index if k > 1}
    return out
 
@prop()
def id_appearances(p):
    """Registry entry: distribution of id (XXXX) repeat-counts within the rows given.
    Use analyze(df, by=['split']) to get it per split."""
    return id_appearance_distribution(p)

def rempli_vs_grid_mismatch(p, id_col="id", return_ids=False, strict=False):
    """Among distinct ids (XXXX by default), count those whose rempli_pattern differs
    from grid_pattern, and return the distribution of the number of differing positions
    (Hamming distance). `diff_position_counts`: n_diff_positions -> n_ids (ascending).
    Also verifies each id carries ONE (rempli, grid) across all its rows -- flagged via
    n_inconsistent / inconsistent_ids, or raised if strict=True -- and reports how many
    ids have both patterns present vs missing. Reads only p.df (no long reshape)."""
    df = p.df
    parts = df["unique_id"].str.split("_", n=1, expand=True)
    key = parts[0] if id_col == "id" else (parts[1] if id_col == "group_id" else df["unique_id"])
    raw = pd.DataFrame({"_key": key.values,
                        "rempli": df["rempli_pattern"].values,
                        "grid": df["grid_pattern"].values})
 
    # consistency: an id must carry one (rempli, grid) across all its rows
    # (dropna=False -> a present-in-one-row / missing-in-another also counts as varying)
    per_id = raw.groupby("_key")[["rempli", "grid"]].nunique(dropna=False)
    inconsistent = sorted(per_id[(per_id["rempli"] > 1) | (per_id["grid"] > 1)].index)
    if strict and inconsistent:
        raise AssertionError(
            f"{len(inconsistent)} id(s) have rempli/grid patterns varying across rows: "
            f"{inconsistent[:10]}{' ...' if len(inconsistent) > 10 else ''}")
 
    work = raw.drop_duplicates("_key")                       # one row per id
    present_mask = work["rempli"].notna() & work["grid"].notna()
    present = work[present_mask]
 
    diff_mask = present["rempli"] != present["grid"]
    differing = present[diff_mask]
 
    def hamming(a, b):
        return sum(ca != cb for ca, cb in zip(a, b)) + abs(len(a) - len(b))
 
    dists = (differing.apply(lambda r: hamming(r["rempli"], r["grid"]), axis=1)
             if len(differing) else pd.Series([], dtype=int))
    dist_counts = dists.value_counts().sort_index()
    out = {
        "n_ids": int(work["_key"].nunique()),
        "n_present": int(present_mask.sum()),
        "n_missing": int((~present_mask).sum()),
        "n_inconsistent": len(inconsistent),
        "n_differing": int(diff_mask.sum()),
        "n_identical": int(present_mask.sum() - diff_mask.sum()),
        "diff_position_counts": {int(k): int(v) for k, v in dist_counts.items()},
    }
    if return_ids:
        out["inconsistent_ids"] = list(map(str, inconsistent))
        out["ids_by_ndiff"] = {int(k): sorted(differing.loc[dists == k, "_key"])
                               for k in dist_counts.index}
    return out
 
@prop()
def rempli_grid_mismatch(p):
    """Registry entry: distribution of rempli_pattern vs grid_pattern differences per id."""
    return rempli_vs_grid_mismatch(p)

def pattern_position_flags(p, n_periods=N_PERIODS):
    """For the given rows (deduped by id=XXXX), per questionnaire position i:
      rempli0_by_period           : # unique ids with rempli_pattern[i] == '0' (not filled)
      grid0_and_rempli1_by_period : # unique ids with grid_pattern[i] == '0' AND
                                     rempli_pattern[i] == '1' (filled but marked unavailable)
    Subset-agnostic: run analyze(df, by=['case_control']) for the per-case_control split.
    Reads only p.df, so no long reshape is triggered."""
    df = p.df
    xid = df["unique_id"].str.split("_", n=1, expand=True)[0]
    work = (pd.DataFrame({"_id": xid.values,
                          "rempli": df["rempli_pattern"].values,
                          "grid": df["grid_pattern"].values})
            .drop_duplicates("_id"))
    rempli0, grid0_rempli1 = {}, {}
    for i in range(1, n_periods + 1):
        rbit = work["rempli"].str[i - 1]
        gbit = work["grid"].str[i - 1]
        rempli0[i] = int((rbit == "0").sum())
        grid0_rempli1[i] = int(((gbit == "0") & (rbit == "1")).sum())
    return {"n_ids": int(work["_id"].nunique()),
            "rempli0_by_period": rempli0,
            "grid0_and_rempli1_by_period": grid0_rempli1}
 
@prop()
def pattern_flags(p):
    """Registry entry: per-position counts of rempli=0 and (grid=0 & rempli=1) per id.
    Use analyze(df, by=['case_control']) to split by case/control."""
    return pattern_position_flags(p)

# ---- ordered text report --------------------------------------------------
def _fmt_value(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)
 
def _render_dict(d, indent, lines):
    """Recursively render a result dict with aligned keys; handles nesting."""
    pad = "  " * indent
    scalar_keys = [k for k, v in d.items() if not isinstance(v, dict)]
    width = max((len(str(k)) for k in scalar_keys), default=0)
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            _render_dict(v, indent + 1, lines)
        elif isinstance(v, (list, tuple)):
            lines.append(f"{pad}{str(k):<{width}} : {', '.join(map(_fmt_value, v))}")
        else:
            lines.append(f"{pad}{str(k):<{width}} : {_fmt_value(v)}")
 
def _is_analyze_shaped(results):
    """True if results looks like analyze() output: {group: {property: {metric: value}}}."""
    return (isinstance(results, dict) and len(results) > 0 and
            all(isinstance(v, dict) and len(v) > 0 and
                all(isinstance(mv, dict) for mv in v.values())
                for v in results.values()))
 
class Report:
    """Accumulate results from many analyze() calls, then render/write one ordered txt.
 
        r = Report("Cohort v1 profile")
        r.add("Split distribution", analyze(df, properties=["split_fraction"]))
        r.add("Fill counts by split", analyze(df, by=["split"], properties=["filled_periods_per_subject"]))
        r.write("profile_report.txt")
 
    add() accepts either analyze() output OR a bare result dict from a single
    function (e.g. ids_in_multiple_splits); bare dicts are wrapped automatically.
    Pass `label=` to give the bare block a name.
    """
    def __init__(self, title=None):
        self.title = title
        self.sections = []  # list of (section_title, results); order preserved
 
    def add(self, section_title, results, label=None):
        if not _is_analyze_shaped(results):
            results = {"all": {(label or ""): results}}
        self.sections.append((section_title, results))
        return self  # chainable
 
    def render(self):
        lines = []
        if self.title:
            lines += ["#" * 72, f"# {self.title}", "#" * 72, ""]
        for section_title, results in self.sections:
            lines += ["=" * 72, section_title.upper(), "=" * 72]
            groups = list(results.items())
            lone_all = len(groups) == 1 and groups[0][0] == "all"
            for gname, props in groups:
                if lone_all:
                    g_indent = 0
                else:
                    lines += ["", f"[{gname}]"]
                    g_indent = 1
                for pname, result in props.items():
                    if pname:
                        lines.append("  " * g_indent + pname)
                        _render_dict(result, g_indent + 1, lines)
                    else:  # bare block: no property header line
                        _render_dict(result, g_indent, lines)
            lines.append("")
        return "\n".join(lines)
 
    def write(self, path):
        with open(path, "w") as f:
            f.write(self.render())
        return path

# ----------- Figures --------------------------

# ---- q_num distribution figures -------------------------------------------
# Plotting deps are imported lazily so the core toolkit stays importable without them.
MODE_COLORS = {"X": "#4C72B0", "text": "#DD8452", "digit": "#55A868"}
 
def _qvals(df, i, mode):
    v = pd.to_numeric(df[f"q_{i}_num_{mode}"], errors="coerce").to_numpy()
    v = v[~np.isnan(v)]
    return v[v >= 0]  # negatives are sentinel missing values -> exclude
 
def _qcounts(df, i, mode):
    """Present / sentinel-negative / NaN breakdown for one (period, mode) cell."""
    raw = pd.to_numeric(df[f"q_{i}_num_{mode}"], errors="coerce")
    return {"n_rows": int(len(raw)),
            "present": int((raw >= 0).sum()),
            "negative": int((raw < 0).sum()),   # sentinel missing
            "nan": int(raw.isna().sum())}       # not collected / non-numeric
 
def q_num_missing_summary(p, n_periods=N_PERIODS, modes=MODES):
    """Summary of missing q_num values. Separates sentinel-negatives (collected but
    missing) from NaN (not collected). Returns per-mode totals and, for negatives,
    a per-period breakdown so you can see where sentinels concentrate."""
    df = p.df
    by_mode, neg_by_period = {}, {}
    tot_neg = tot_nan = tot_present = 0
    for m in modes:
        pres = neg = nan = 0
        per_period = {}
        for i in range(1, n_periods + 1):
            c = _qcounts(df, i, m)
            pres += c["present"]; neg += c["negative"]; nan += c["nan"]
            if c["negative"]:
                per_period[i] = c["negative"]
        by_mode[m] = {"present": pres, "negative": neg, "nan": nan}
        neg_by_period[m] = per_period
        tot_present += pres; tot_neg += neg; tot_nan += nan
    return {
        "n_present_total": tot_present,
        "n_negative_total": tot_neg,   # excluded as sentinel-missing
        "n_nan_total": tot_nan,
        "by_mode": by_mode,
        "negative_by_period": neg_by_period,
    }
 
@prop()
def q_num_missing(p):
    """Registry entry: missing (sentinel-negative / NaN) summary for q_num columns."""
    return q_num_missing_summary(p)
 
def plot_q_num_grid(p, path, n_periods=N_PERIODS, modes=MODES, bins=15,
                    density=True, stat="median", dpi=300):
    """Requested grid: rows = period i, columns = mode. Each subplot is on its OWN
    x/y scale (auto-fit to that cell's data) so individual panels are easy to inspect.
    Annotates n (present), miss (sentinel-negatives excluded), and the median."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = p.df
    fig, axes = plt.subplots(n_periods, len(modes),
                             figsize=(3.6 * len(modes), 1.55 * n_periods),
                             squeeze=False)  # independent scales -> no sharex/sharey
    for r, i in enumerate(range(1, n_periods + 1)):
        for c, m in enumerate(modes):
            ax = axes[r][c]
            v = _qvals(df, i, m)
            miss = _qcounts(df, i, m)["negative"]
            if v.size:
                nb = min(bins, max(3, np.unique(v).size))  # own range, sane bin count
                ax.hist(v, bins=nb, density=density, color=MODE_COLORS.get(m, "#888"),
                        alpha=0.85, edgecolor="white", linewidth=0.3)
                s = np.median(v) if stat == "median" else np.mean(v)
                ax.axvline(s, color="crimson", lw=1.2)
                ax.text(0.96, 0.92, f"n={v.size}  miss={miss}\n{stat[:3]}={s:.1f}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#333")
            else:
                ax.text(0.5, 0.5, f"no data\nmiss={miss}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="grey")
            if r == 0:
                ax.set_title(m, fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"q{i}", rotation=0, labelpad=16, fontsize=9,
                              va="center", fontweight="bold")
            ax.tick_params(labelsize=6, length=2)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.suptitle("Distribution of q_{i}_num_{mode}   (independent scales; red line = " + stat + ")",
                 fontsize=13, y=1.001)
    fig.tight_layout(h_pad=0.5, w_pad=0.6)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
 
def plot_q_num_ridgeline(p, path, n_periods=N_PERIODS, modes=MODES):
    """Comparison view: KDE ridgeline per mode, periods stacked top(q1)->bottom(q13),
    colored by period. Best for seeing how a mode's distribution shifts over time."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from scipy.stats import gaussian_kde
    df = p.df
    fig, axes = plt.subplots(1, len(modes), figsize=(4.2 * len(modes), 6.5), squeeze=False)
    cmap = cm.viridis
    for c, m in enumerate(modes):
        ax = axes[0][c]
        allv = np.concatenate([_qvals(df, i, m) for i in range(1, n_periods + 1)]
                              or [np.array([0.0])])
        if allv.size < 2:
            ax.text(0.5, 0.5, "no data", ha="center"); continue
        xs = np.linspace(allv.min(), allv.max(), 200)
        offset = 0.9
        for r, i in enumerate(range(1, n_periods + 1)):
            v = _qvals(df, i, m)
            base = (n_periods - r) * offset
            if v.size > 2 and np.ptp(v) > 0:
                try:
                    dens = gaussian_kde(v)(xs)
                    dens = dens / dens.max() * (offset * 1.6)
                    ax.fill_between(xs, base, base + dens, color=cmap(r / (n_periods - 1)),
                                    alpha=0.75, lw=0.8, edgecolor="white")
                except Exception:
                    pass
            ax.text(allv.min(), base, f"q{i}", va="bottom", ha="right", fontsize=7)
        ax.set_title(m, fontsize=11, fontweight="bold")
        ax.set_yticks([]); ax.set_xlabel("count")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("q_num distribution shift across periods (KDE ridgeline; top=q1 -> bottom=q13)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
 
def plot_q_num_summary(p, path, n_periods=N_PERIODS, modes=MODES, dpi=200):
    """Comparison view: compact heatmaps of median / mean / %zero / n / n miss over
    period x mode. The n panel exposes sample-size decay; n miss shows where
    sentinel-negative values concentrate."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = p.df
    labels = ["median", "mean", "% zero", "n", "n miss"]
    fig, axes = plt.subplots(1, len(labels), figsize=(3.0 * len(labels), 5.2), squeeze=False)
    for a, label in zip(axes[0], labels):
        M = np.full((n_periods, len(modes)), np.nan)
        for r, i in enumerate(range(1, n_periods + 1)):
            for c, m in enumerate(modes):
                v = _qvals(df, i, m)
                if label == "n miss":
                    M[r, c] = _qcounts(df, i, m)["negative"]
                elif v.size:
                    M[r, c] = {"median": np.median(v), "mean": np.mean(v),
                               "% zero": 100 * np.mean(v == 0), "n": v.size}[label]
        im = a.imshow(M, aspect="auto", cmap="magma")
        a.set_xticks(range(len(modes))); a.set_xticklabels(modes, fontsize=9)
        a.set_yticks(range(n_periods))
        a.set_yticklabels([f"q{i}" for i in range(1, n_periods + 1)], fontsize=7)
        a.set_title(label, fontsize=10, fontweight="bold")
        for r in range(n_periods):
            for c in range(len(modes)):
                if not np.isnan(M[r, c]):
                    a.text(c, r, f"{M[r, c]:.0f}" if label in ("n", "n miss") else f"{M[r, c]:.1f}",
                           ha="center", va="center", fontsize=6,
                           color="white" if im.norm(M[r, c]) < 0.6 else "black")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle("q_num summary by period x mode", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
 
def save_q_num_figures(p, outdir="."):
    """Save all three q_num figures and return {'paths': [...], 'missing': {...}}."""
    import os
    paths = [
        plot_q_num_grid(p, os.path.join(outdir, "q_num_hist_grid.png")),
        plot_q_num_ridgeline(p, os.path.join(outdir, "q_num_ridgeline.png")),
        plot_q_num_summary(p, os.path.join(outdir, "q_num_summary.png")),
    ]
    return {"paths": paths, "missing": q_num_missing_summary(p)}

# ---- case_dt_dateq distribution figures (cases only; no mode dimension) -----
def _dtvals(df, i):
    v = pd.to_numeric(df[f"case_dt_dateq{i}"], errors="coerce").to_numpy()
    return v[~np.isnan(v)]  # negatives are VALID here; only NaN is absent
 
def _dtcounts(df, i):
    raw = pd.to_numeric(df[f"case_dt_dateq{i}"], errors="coerce")
    return {"n_rows": int(len(raw)),
            "present": int(raw.notna().sum()),     # non-null (incl. negatives)
            "not_collected": int(raw.isna().sum()),  # NaN = questionnaire not collected
            "negative": int((raw < 0).sum())}       # valid values that are negative (info)
 
def _cases(p):
    """Rows for cases only (case_control == 1); controls duplicate the case values."""
    return p.df[p.df["case_control"] == 1]
 
def case_dt_missing_summary(p, n_periods=N_PERIODS):
    """Coverage summary for case_dt_dateq over cases only. Here negatives are VALID
    values, so the only absence is NaN (questionnaire not collected). Reports present
    vs not_collected per period, plus an informational count of negative values."""
    df = _cases(p)
    per = {}
    tot_present = tot_nc = tot_neg = 0
    for i in range(1, n_periods + 1):
        c = _dtcounts(df, i)
        per[i] = {"present": c["present"], "not_collected": c["not_collected"],
                  "negative": c["negative"]}
        tot_present += c["present"]; tot_nc += c["not_collected"]; tot_neg += c["negative"]
    return {"n_cases": int(len(df)),
            "n_present_total": tot_present,
            "n_not_collected_total": tot_nc,
            "n_negative_total": tot_neg,   # valid values that happen to be negative
            "by_period": per}
 
@prop()
def case_dt_missing(p):
    """Registry entry: coverage summary for case_dt_dateq (cases only; negatives valid)."""
    return case_dt_missing_summary(p)
 
def plot_case_dt_grid(p, path, n_periods=N_PERIODS, ncols=4, bins=15,
                      density=True, stat="median", dpi=300):
    """Grid of per-period case_dt_dateq histograms (cases only), each on its own
    x/y scale. Annotates n (present), miss (sentinel-negatives), and the median."""
    import math
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = _cases(p)
    nrows = math.ceil(n_periods / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.2 * nrows), squeeze=False)
    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        if idx >= n_periods:
            ax.axis("off"); continue
        i = idx + 1
        v = _dtvals(df, i)
        if v.size:
            nb = min(bins, max(3, np.unique(v).size))
            ax.hist(v, bins=nb, density=density, color="#6A51A3",
                    alpha=0.85, edgecolor="white", linewidth=0.3)
            s = np.median(v) if stat == "median" else np.mean(v)
            ax.axvline(s, color="crimson", lw=1.2)
            ax.text(0.96, 0.92, f"n={v.size}\n{stat[:3]}={s:.1f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#333")
        else:
            ax.text(0.5, 0.5, "not collected", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="grey")
        ax.set_title(f"case_dt_dateq{i}", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6, length=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Distribution of case_dt_dateq{i}  (case_control=1; independent scales; red = "
                 + stat + ")", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
 
def plot_case_dt_ridgeline(p, path, n_periods=N_PERIODS, dpi=200):
    """KDE ridgeline of case_dt_dateq across periods (cases only), top q1 -> bottom q13."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from scipy.stats import gaussian_kde
    df = _cases(p)
    fig, ax = plt.subplots(figsize=(7, 7))
    allv = np.concatenate([_dtvals(df, i) for i in range(1, n_periods + 1)] or [np.array([0.0])])
    if allv.size < 2:
        ax.text(0.5, 0.5, "no data", ha="center")
    else:
        xs = np.linspace(allv.min(), allv.max(), 200); offset = 0.9; cmap = cm.viridis
        for r, i in enumerate(range(1, n_periods + 1)):
            v = _dtvals(df, i); base = (n_periods - r) * offset
            if v.size > 2 and np.ptp(v) > 0:
                try:
                    dens = gaussian_kde(v)(xs); dens = dens / dens.max() * (offset * 1.6)
                    ax.fill_between(xs, base, base + dens, color=cmap(r / (n_periods - 1)),
                                    alpha=0.75, lw=0.8, edgecolor="white")
                except Exception:
                    pass
            ax.text(allv.min(), base, f"q{i}", va="bottom", ha="right", fontsize=8)
        ax.set_yticks([]); ax.set_xlabel("years before exit")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("case_dt_dateq across periods (case_control=1; top=q1 -> bottom=q13)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
 
def plot_case_dt_summary(p, path, n_periods=N_PERIODS, dpi=200):
    """Compact single heatmap: rows = period, columns = median/mean/%zero/n/n miss
    (cases only). Color is scaled per column so each stat's gradient is readable."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = _cases(p)
    labels = ["median", "mean", "min", "max", "n"]
    raw = np.full((n_periods, len(labels)), np.nan)
    for r, i in enumerate(range(1, n_periods + 1)):
        v = _dtvals(df, i)
        vals = {"median": np.median(v) if v.size else np.nan,
                "mean": np.mean(v) if v.size else np.nan,
                "min": np.min(v) if v.size else np.nan,
                "max": np.max(v) if v.size else np.nan,
                "n": v.size}
        for cc, l in enumerate(labels):
            raw[r, cc] = vals[l]
    norm = np.full_like(raw, 0.5)
    for cc in range(len(labels)):
        col = raw[:, cc]; lo, hi = np.nanmin(col), np.nanmax(col)
        if hi > lo:
            norm[:, cc] = (col - lo) / (hi - lo)
    fig, ax = plt.subplots(figsize=(1.15 * len(labels) + 2, 0.42 * n_periods + 1.2))
    im = ax.imshow(norm, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(n_periods)); ax.set_yticklabels([f"q{i}" for i in range(1, n_periods + 1)], fontsize=8)
    for r in range(n_periods):
        for cc, l in enumerate(labels):
            if not np.isnan(raw[r, cc]):
                ax.text(cc, r, f"{raw[r, cc]:.0f}" if l == "n" else f"{raw[r, cc]:.1f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if norm[r, cc] < 0.55 else "black")
    ax.set_title("case_dt_dateq summary by period (case_control=1; color scaled per column)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
 
def save_case_dt_figures(p, outdir="."):
    """Save all three case_dt figures and return {'paths': [...], 'missing': {...}}."""
    import os
    os.makedirs(outdir, exist_ok=True)
    paths = [
        plot_case_dt_grid(p, os.path.join(outdir, "case_dt_hist_grid.png")),
        plot_case_dt_ridgeline(p, os.path.join(outdir, "case_dt_ridgeline.png")),
        plot_case_dt_summary(p, os.path.join(outdir, "case_dt_summary.png")),
    ]
    return {"paths": paths, "missing": case_dt_missing_summary(p)}