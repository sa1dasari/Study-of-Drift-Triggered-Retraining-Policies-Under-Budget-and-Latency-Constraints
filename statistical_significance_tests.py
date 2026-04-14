"""
Statistical Significance Tests -- Paired comparisons across seeds for each
policy pair on experiment results.

Supports both result sets:
  - results_without_retrain  (no-partial-fit / no actual retraining)
  - results_with_retrain     (partial-fit / actual retraining applied)

For every pair of policies and every drift type, the script:
  1. Pairs observations by random_seed (and budget x latency when both
     policies have a budget/latency grid).
  2. Runs a paired t-test and a Wilcoxon signed-rank test.
  3. Computes Cohen's d (paired) and a 95% CI for the mean difference.
  4. Applies Holm-Bonferroni correction across all tests within a dataset.
  5. Flags both statistical significance (alpha=0.05, Holm-corrected) and
     practical significance (|Cohen's d| > 0.5 AND |delta mean| > 0.02).

Two levels of analysis are produced:
  - Grand-level  -- one row per (policy_A, policy_B, drift_type).
        For ALL policy pairs (including pairs of two retraining policies)
        the metric is first averaged over the budget x latency grid per seed
        before pairing.  This yields n = #seeds independent paired
        observations per comparison -- conservative but statistically clean.
        For pairs involving no_retrain the same averaging applies; since
        no_retrain has no grid the value is already one-per-seed.
  - Cell-level   -- one row per (policy_A, policy_B, drift_type, budget,
        total_latency), only for pairs of two retraining policies.
        Paired directly on random_seed within each cell.
        NOTE: cell-level observations are NOT independent across cells --
        the same seed appears once per budget x latency combination.
        Report these for exploratory detail, not as primary evidence.

Outputs are saved to <results_folder>/statistical_tests/.
A unified significance_overview.csv is also produced, concatenating the
grand-level tables across all datasets for easy inclusion in a paper.

Usage:
    python statistical_significance_tests.py                        # all 3 datasets, both result sets
    python statistical_significance_tests.py --dataset synthetic
    python statistical_significance_tests.py --dataset luflow
    python statistical_significance_tests.py --dataset lendingclub
    python statistical_significance_tests.py --seeds 3
    python statistical_significance_tests.py --metric overall_accuracy
    python statistical_significance_tests.py --results without_retrain
    python statistical_significance_tests.py --results with_retrain
    python statistical_significance_tests.py --results all           # default: both
"""

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# --- Project root ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Policy ordering & display names --------------------------------------
POLICY_ORDER = ["periodic", "error_threshold", "drift_triggered", "no_retrain"]
POLICY_DISPLAY = {
    "periodic":         "Periodic",
    "error_threshold":  "Error-Threshold",
    "drift_triggered":  "Drift-Triggered",
    "no_retrain":       "No-Retrain",
}
DRIFT_ORDER = ["abrupt", "gradual", "recurring"]

# --- Practical significance thresholds ------------------------------------
PRACTICAL_D_THRESH = 0.5     # medium Cohen's d
PRACTICAL_DIFF_THRESH = 0.02 # 2 percentage-point mean difference


# =====================================================================
#  DATA LOADING  (mirrors cross_policy_comparison.load_merged)
# =====================================================================

def _csv_search_patterns(dataset, seed_label, results_folder):
    """Return (dir, filename_list) for a given dataset and results folder."""
    if dataset == "synthetic":
        base = PROJECT_ROOT / results_folder / "synthetic" / "csv"
        patterns = [
            f"summary_results_periodic_retrain_{seed_label}.csv",
            f"summary_results_error_threshold_retrain_{seed_label}.csv",
            f"summary_results_drift_triggered_retrain_{seed_label}.csv",
            f"summary_results_no_retrain_{seed_label}.csv",
        ]
    elif dataset == "luflow":
        base = PROJECT_ROOT / results_folder / "luflow" / "csv"
        patterns = [
            f"luflow_summary_periodic_retrain_{seed_label}.csv",
            f"luflow_summary_error_threshold_retrain_{seed_label}.csv",
            f"luflow_summary_drift_triggered_retrain_{seed_label}.csv",
            f"luflow_summary_no_retrain_{seed_label}.csv",
        ]
    elif dataset == "lendingclub":
        base = PROJECT_ROOT / results_folder / "lendingclub" / "csv"
        patterns = [
            f"lendingclub_summary_periodic_retrain_{seed_label}.csv",
            f"lendingclub_summary_error_threshold_retrain_{seed_label}.csv",
            f"lendingclub_summary_drift_triggered_retrain_{seed_label}.csv",
            f"lendingclub_summary_no_retrain_{seed_label}.csv",
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return base, patterns


def load_merged(dataset, seed_label="10seed", results_folder="results_without_retrain"):
    """Load and merge all available policy CSVs for *dataset*."""
    base, patterns = _csv_search_patterns(dataset, seed_label, results_folder)
    frames = []
    for pat in patterns:
        fpath = base / pat
        if fpath.exists():
            frames.append(pd.read_csv(fpath))
        else:
            print(f"  [WARN] Not found (skipping): {fpath.name}")

    if not frames and seed_label != "3seed":
        print(f"  Falling back to 3seed CSVs for {dataset} ...")
        return load_merged(dataset, seed_label="3seed", results_folder=results_folder)

    if not frames:
        print(f"  [FAIL] No CSVs found for {dataset}. Skipping.")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged["total_latency"] = merged["retrain_latency"] + merged["deploy_latency"]
    return merged


# =====================================================================
#  STATISTICAL HELPERS
# =====================================================================

def cohens_d_paired(diffs):
    """Cohen's d for paired samples = mean(d) / std(d)."""
    m = np.mean(diffs)
    s = np.std(diffs, ddof=1)
    return m / s if s > 0 else np.inf * np.sign(m)


def ci_95(diffs):
    """95% confidence interval for the mean difference (paired)."""
    n = len(diffs)
    m = np.mean(diffs)
    se = np.std(diffs, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return m - t_crit * se, m + t_crit * se


def paired_tests(vals_a, vals_b):
    """Run paired t-test and Wilcoxon signed-rank test.

    Returns a dict with test statistics, p-values, effect size, etc.
    *vals_a* and *vals_b* must be aligned arrays (same seed order).
    """
    diffs = vals_a - vals_b
    n = len(diffs)
    mean_diff = np.mean(diffs)

    result = {
        "n_pairs": n,
        "mean_a": np.mean(vals_a),
        "mean_b": np.mean(vals_b),
        "mean_diff": mean_diff,
    }

    # 95% CI
    if n >= 2:
        lo, hi = ci_95(diffs)
        result["ci_lower"] = lo
        result["ci_upper"] = hi
        result["cohens_d"] = cohens_d_paired(diffs)
    else:
        result["ci_lower"] = np.nan
        result["ci_upper"] = np.nan
        result["cohens_d"] = np.nan

    # Paired t-test (requires n >= 2)
    if n >= 2 and np.std(diffs, ddof=1) > 0:
        t_stat, t_p = stats.ttest_rel(vals_a, vals_b)
        result["t_stat"] = t_stat
        result["t_pvalue"] = t_p
    else:
        result["t_stat"] = np.nan
        result["t_pvalue"] = np.nan

    # Wilcoxon signed-rank (requires n >= 5 for meaningful results;
    # scipy needs at least 1 non-zero difference)
    if n >= 5 and np.any(diffs != 0):
        try:
            w_stat, w_p = stats.wilcoxon(diffs)
            result["wilcoxon_stat"] = w_stat
            result["wilcoxon_pvalue"] = w_p
        except ValueError:
            result["wilcoxon_stat"] = np.nan
            result["wilcoxon_pvalue"] = np.nan
    else:
        result["wilcoxon_stat"] = np.nan
        result["wilcoxon_pvalue"] = np.nan
        if n < 5:
            result["wilcoxon_note"] = f"n={n} < 5, Wilcoxon unreliable"

    return result


def holm_bonferroni(pvalues):
    """Apply Holm-Bonferroni correction. Returns adjusted p-values."""
    n = len(pvalues)
    if n == 0:
        return np.array([])
    pv = np.asarray(pvalues, dtype=float)
    order = np.argsort(pv)
    adjusted = np.empty_like(pv)
    cummax = 0.0
    for rank, idx in enumerate(order):
        adj = pv[idx] * (n - rank)
        adj = max(adj, cummax)          # enforce monotonicity
        adj = min(adj, 1.0)
        adjusted[idx] = adj
        cummax = adj
    return adjusted


# =====================================================================
#  GRAND-LEVEL TESTS  (one row per policy_pair x drift_type)
# =====================================================================

def _seed_averaged_metric(df, policy, drift, metric):
    """Return a Series indexed by random_seed with the metric value.

    For retraining policies (which have a budget x latency grid), we first
    average the metric over all budget x latency conditions per seed.
    For no_retrain (single row per seed), we return the metric directly.
    """
    sub = df[(df["policy_type"] == policy) & (df["drift_type"] == drift)]
    return sub.groupby("random_seed")[metric].mean()


def grand_level_tests(df, metric):
    """Pairwise grand-level tests for all policy pairs x drift types.

    For EVERY pair (including two retraining policies), we use option (a):
    average the metric over budget x latency per seed first, then pair on
    seed.  This gives n = #seeds independent paired observations -- more
    conservative than pairing at the cell level, but statistically clean.
    """
    policies = [p for p in POLICY_ORDER if p in df["policy_type"].unique()]
    drifts = [d for d in DRIFT_ORDER if d in df["drift_type"].unique()]

    rows = []
    for pa, pb in itertools.combinations(policies, 2):
        for drift in drifts:
            sa = _seed_averaged_metric(df, pa, drift, metric)
            sb = _seed_averaged_metric(df, pb, drift, metric)

            # Align on shared seeds
            shared = sa.index.intersection(sb.index)
            if len(shared) < 2:
                continue

            vals_a = sa.loc[shared].values
            vals_b = sb.loc[shared].values

            res = paired_tests(vals_a, vals_b)
            res["policy_a"] = pa
            res["policy_b"] = pb
            res["drift_type"] = drift
            rows.append(res)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    # Holm-Bonferroni on t-test p-values
    t_pvals = result_df["t_pvalue"].values
    mask_valid = ~np.isnan(t_pvals)
    adj = np.full_like(t_pvals, np.nan)
    if mask_valid.any():
        adj[mask_valid] = holm_bonferroni(t_pvals[mask_valid])
    result_df["t_pvalue_adj"] = adj

    # Holm-Bonferroni on Wilcoxon p-values
    w_pvals = result_df["wilcoxon_pvalue"].values
    mask_valid_w = ~np.isnan(w_pvals)
    adj_w = np.full_like(w_pvals, np.nan)
    if mask_valid_w.any():
        adj_w[mask_valid_w] = holm_bonferroni(w_pvals[mask_valid_w])
    result_df["wilcoxon_pvalue_adj"] = adj_w

    # Statistical significance flags
    result_df["sig_t_005"] = result_df["t_pvalue_adj"] < 0.05
    result_df["sig_w_005"] = result_df["wilcoxon_pvalue_adj"] < 0.05

    # Practical significance flag: |d| > 0.5 AND |delta mean| > 0.02
    result_df["practical_sig"] = (
        (result_df["cohens_d"].abs() > PRACTICAL_D_THRESH) &
        (result_df["mean_diff"].abs() > PRACTICAL_DIFF_THRESH)
    )

    # Reorder columns for readability
    col_order = [
        "policy_a", "policy_b", "drift_type",
        "n_pairs", "mean_a", "mean_b", "mean_diff",
        "ci_lower", "ci_upper", "cohens_d",
        "t_stat", "t_pvalue", "t_pvalue_adj", "sig_t_005",
        "wilcoxon_stat", "wilcoxon_pvalue", "wilcoxon_pvalue_adj", "sig_w_005",
        "practical_sig",
    ]
    extra = [c for c in result_df.columns if c not in col_order]
    result_df = result_df[[c for c in col_order if c in result_df.columns] + extra]

    return result_df


# =====================================================================
#  CELL-LEVEL TESTS  (per budget x latency, retraining policies only)
#
#  NOTE: Cell-level observations are NOT independent across cells --
#  the same seed appears once per budget x latency combination within a
#  drift type.  These tests are useful for exploratory breakdowns but
#  should not be cited as primary evidence in a paper.
# =====================================================================

def cell_level_tests(df, metric):
    """Pairwise cell-level tests for each (policy_pair, drift, budget, latency).

    Observations within seeds are correlated across cells (the same seed
    appears in every budget x latency combination).  These results provide
    granular exploratory detail; the grand-level tests are the primary
    evidence for statistical significance.
    """
    retrain_policies = [p for p in POLICY_ORDER
                        if p in df["policy_type"].unique() and p != "no_retrain"]
    drifts = [d for d in DRIFT_ORDER if d in df["drift_type"].unique()]
    budgets = sorted(df[df["budget"] > 0]["budget"].unique())
    latencies = sorted(df[df["total_latency"] > 0]["total_latency"].unique())

    if len(retrain_policies) < 2:
        return pd.DataFrame()

    rows = []
    for pa, pb in itertools.combinations(retrain_policies, 2):
        for drift in drifts:
            for budget in budgets:
                for latency in latencies:
                    sub_a = df[
                        (df["policy_type"] == pa) &
                        (df["drift_type"] == drift) &
                        (df["budget"] == budget) &
                        (df["total_latency"] == latency)
                    ].set_index("random_seed")[metric]

                    sub_b = df[
                        (df["policy_type"] == pb) &
                        (df["drift_type"] == drift) &
                        (df["budget"] == budget) &
                        (df["total_latency"] == latency)
                    ].set_index("random_seed")[metric]

                    shared = sub_a.index.intersection(sub_b.index)
                    if len(shared) < 2:
                        continue

                    vals_a = sub_a.loc[shared].values
                    vals_b = sub_b.loc[shared].values

                    res = paired_tests(vals_a, vals_b)
                    res["policy_a"] = pa
                    res["policy_b"] = pb
                    res["drift_type"] = drift
                    res["budget"] = budget
                    res["total_latency"] = latency
                    rows.append(res)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    # Holm-Bonferroni correction
    t_pvals = result_df["t_pvalue"].values
    mask = ~np.isnan(t_pvals)
    adj = np.full_like(t_pvals, np.nan)
    if mask.any():
        adj[mask] = holm_bonferroni(t_pvals[mask])
    result_df["t_pvalue_adj"] = adj

    w_pvals = result_df["wilcoxon_pvalue"].values
    mask_w = ~np.isnan(w_pvals)
    adj_w = np.full_like(w_pvals, np.nan)
    if mask_w.any():
        adj_w[mask_w] = holm_bonferroni(w_pvals[mask_w])
    result_df["wilcoxon_pvalue_adj"] = adj_w

    result_df["sig_t_005"] = result_df["t_pvalue_adj"] < 0.05
    result_df["sig_w_005"] = result_df["wilcoxon_pvalue_adj"] < 0.05

    # Practical significance flag: |d| > 0.5 AND |delta mean| > 0.02
    result_df["practical_sig"] = (
        (result_df["cohens_d"].abs() > PRACTICAL_D_THRESH) &
        (result_df["mean_diff"].abs() > PRACTICAL_DIFF_THRESH)
    )

    col_order = [
        "policy_a", "policy_b", "drift_type", "budget", "total_latency",
        "n_pairs", "mean_a", "mean_b", "mean_diff",
        "ci_lower", "ci_upper", "cohens_d",
        "t_stat", "t_pvalue", "t_pvalue_adj", "sig_t_005",
        "wilcoxon_stat", "wilcoxon_pvalue", "wilcoxon_pvalue_adj", "sig_w_005",
        "practical_sig",
    ]
    extra = [c for c in result_df.columns if c not in col_order]
    result_df = result_df[[c for c in col_order if c in result_df.columns] + extra]

    return result_df


# =====================================================================
#  CONSOLE PRINTING
# =====================================================================

def _fmt_p(p, adj=None):
    """Format a p-value with significance stars."""
    if np.isnan(p):
        return "    -     "
    stars = ""
    ref = adj if (adj is not None and not np.isnan(adj)) else p
    if ref < 0.001:
        stars = " ***"
    elif ref < 0.01:
        stars = " ** "
    elif ref < 0.05:
        stars = " *  "
    return f"{p:.4f}{stars}"


def print_grand_summary(df_grand, dataset, metric):
    """Pretty-print the grand-level significance table."""
    print(f"\n{'=' * 100}")
    print(f"  GRAND-LEVEL PAIRED TESTS -- {dataset.upper()} -- metric: {metric}")
    print(f"  (Retraining policies averaged over budget x latency per seed before pairing)")
    print(f"  Practical significance: |d| > {PRACTICAL_D_THRESH} AND |delta| > {PRACTICAL_DIFF_THRESH}")
    print(f"{'=' * 100}")

    if df_grand.empty:
        print("  No comparisons available.\n")
        return

    header = (
        f"{'Policy A':<18} {'Policy B':<18} {'Drift':<12} "
        f"{'n':>3} {'d Mean':>8} {'95% CI':>18} {'d':>7} "
        f"{'t p-val':>11} {'W p-val':>11} {'Stat':>5} {'Prac':>5}"
    )
    print(header)
    print("-" * len(header))

    for _, r in df_grand.iterrows():
        ci_str = (f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
                  if not np.isnan(r["ci_lower"]) else "       -        ")
        stat_sig = ""
        if not np.isnan(r.get("t_pvalue_adj", np.nan)) and r["t_pvalue_adj"] < 0.05:
            stat_sig = " YES"
        elif not np.isnan(r.get("wilcoxon_pvalue_adj", np.nan)) and r["wilcoxon_pvalue_adj"] < 0.05:
            stat_sig = " YES"
        prac_sig = " YES" if r.get("practical_sig", False) else ""

        print(
            f"{POLICY_DISPLAY.get(r['policy_a'], r['policy_a']):<18} "
            f"{POLICY_DISPLAY.get(r['policy_b'], r['policy_b']):<18} "
            f"{r['drift_type']:<12} "
            f"{int(r['n_pairs']):>3} "
            f"{r['mean_diff']:>+8.4f} "
            f"{ci_str:>18} "
            f"{r['cohens_d']:>7.3f} "
            f"{_fmt_p(r['t_pvalue'], r.get('t_pvalue_adj')):>11} "
            f"{_fmt_p(r['wilcoxon_pvalue'], r.get('wilcoxon_pvalue_adj')):>11} "
            f"{stat_sig:>5}"
            f"{prac_sig:>5}"
        )

    n_sig_t = df_grand["sig_t_005"].sum() if "sig_t_005" in df_grand.columns else 0
    n_sig_w = df_grand["sig_w_005"].sum() if "sig_w_005" in df_grand.columns else 0
    n_prac = df_grand["practical_sig"].sum() if "practical_sig" in df_grand.columns else 0
    print(f"\n  Significant (Holm-corrected alpha=0.05): "
          f"t-test={n_sig_t}/{len(df_grand)}, Wilcoxon={n_sig_w}/{len(df_grand)}")
    print(f"  Practically significant (|d|>{PRACTICAL_D_THRESH} & "
          f"|delta|>{PRACTICAL_DIFF_THRESH}): {n_prac}/{len(df_grand)}")


def print_cell_summary(df_cell, dataset, metric):
    """Print a compact cell-level summary (count of significant comparisons)."""
    if df_cell.empty:
        return

    print(f"\n{'-' * 70}")
    print(f"  CELL-LEVEL TESTS SUMMARY -- {dataset.upper()} -- metric: {metric}")
    print(f"  (One test per policy-pair x drift x budget x latency cell)")
    print(f"  [WARN] Observations within seeds are correlated across cells;")
    print(f"    these results are exploratory, not primary evidence.")
    print(f"{'-' * 70}")

    total = len(df_cell)
    n_sig_t = df_cell["sig_t_005"].sum()
    n_sig_w = df_cell["sig_w_005"].sum()
    n_prac = df_cell["practical_sig"].sum()
    print(f"  Total cells tested : {total}")
    print(f"  Significant (t-test, Holm alpha=0.05)  : {n_sig_t} ({100 * n_sig_t / total:.1f}%)")
    print(f"  Significant (Wilcoxon, Holm alpha=0.05): {n_sig_w} ({100 * n_sig_w / total:.1f}%)")
    print(f"  Practically significant             : {n_prac} ({100 * n_prac / total:.1f}%)")
    print(f"  Both stat + practical (t-test)      : "
          f"{(df_cell['sig_t_005'] & df_cell['practical_sig']).sum()} "
          f"({100 * (df_cell['sig_t_005'] & df_cell['practical_sig']).sum() / total:.1f}%)")

    # Breakdown by policy pair
    for (pa, pb), grp in df_cell.groupby(["policy_a", "policy_b"]):
        n = len(grp)
        st = grp["sig_t_005"].sum()
        sw = grp["sig_w_005"].sum()
        sp = grp["practical_sig"].sum()
        mean_d = grp["cohens_d"].mean()
        print(
            f"    {POLICY_DISPLAY.get(pa, pa)} vs {POLICY_DISPLAY.get(pb, pb)}: "
            f"{st}/{n} sig (t), {sw}/{n} sig (W), {sp}/{n} prac, "
            f"mean d={mean_d:.3f}"
        )


# =====================================================================
#  ORCHESTRATOR
# =====================================================================

def run_significance_for_dataset(dataset, seed_label, metric, out_dir,
                                 results_folder="results_without_retrain"):
    """Run all paired significance tests for one dataset.

    Returns the grand-level DataFrame (with 'dataset' and 'results_set'
    columns added) for cross-dataset aggregation, or None if no data.
    """
    results_label = results_folder.replace("results_", "")   # e.g. "with_retrain"
    print(f"\n{'#' * 70}")
    print(f"  STATISTICAL SIGNIFICANCE TESTS -- {dataset.upper()} -- [{results_label}]")
    print(f"{'#' * 70}")

    df = load_merged(dataset, seed_label, results_folder=results_folder)
    if df.empty:
        return None

    n_policies = df["policy_type"].nunique()
    n_seeds = df["random_seed"].nunique()
    print(f"  Loaded {len(df)} rows -- {n_policies} policies, "
          f"{n_seeds} seeds, metric={metric}")

    ds_dir = out_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    # -- Grand-level tests --------------------------------------------------
    df_grand = grand_level_tests(df, metric)
    if not df_grand.empty:
        fpath = ds_dir / f"grand_significance_{dataset}.csv"
        df_grand.to_csv(fpath, index=False)
        print(f"  Saved: {fpath.name}")
        print_grand_summary(df_grand, dataset, metric)

    # -- Cell-level tests ---------------------------------------------------
    df_cell = cell_level_tests(df, metric)
    if not df_cell.empty:
        fpath = ds_dir / f"cell_significance_{dataset}.csv"
        df_cell.to_csv(fpath, index=False)
        print(f"  Saved: {fpath.name}")
        print_cell_summary(df_cell, dataset, metric)

    print()

    # Return grand-level results tagged with dataset + results_set for overview
    if not df_grand.empty:
        df_grand_out = df_grand.copy()
        df_grand_out.insert(0, "dataset", dataset)
        df_grand_out.insert(1, "results_set", results_label)
        return df_grand_out
    return None


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Paired statistical significance tests on experiment results."
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "luflow", "lendingclub", "all"],
        default="all",
        help="Which dataset to analyse (default: all).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        choices=[3, 10],
        default=10,
        help="Seed-set label: 3 or 10 (default: 10).",
    )
    parser.add_argument(
        "--metric",
        default="post_drift_accuracy",
        help="Metric column to test (default: post_drift_accuracy).",
    )
    parser.add_argument(
        "--results",
        choices=["without_retrain", "with_retrain", "all"],
        default="all",
        help="Which result set to analyse: without_retrain, with_retrain, "
             "or all (default: all).",
    )
    args = parser.parse_args()

    seed_label = f"{args.seeds}seed"
    datasets = (
        ["synthetic", "luflow", "lendingclub"]
        if args.dataset == "all"
        else [args.dataset]
    )

    # Determine which result folders to process
    RESULTS_FOLDERS = {
        "without_retrain": "results_without_retrain",
        "with_retrain":    "results_with_retrain",
    }
    if args.results == "all":
        results_folders = list(RESULTS_FOLDERS.values())
    else:
        results_folders = [RESULTS_FOLDERS[args.results]]

    grand_frames = []
    for results_folder in results_folders:
        out_dir = PROJECT_ROOT / results_folder / "statistical_tests"
        out_dir.mkdir(parents=True, exist_ok=True)

        folder_grand_frames = []
        for ds in datasets:
            df_grand = run_significance_for_dataset(
                ds, seed_label, args.metric, out_dir,
                results_folder=results_folder,
            )
            if df_grand is not None:
                folder_grand_frames.append(df_grand)
                grand_frames.append(df_grand)

        # -- Per-folder significance overview CSV ----------------------------
        if folder_grand_frames:
            overview = pd.concat(folder_grand_frames, ignore_index=True)
            overview_path = out_dir / "significance_overview.csv"
            overview.to_csv(overview_path, index=False)
            print(f"  Saved overview for {results_folder}: {overview_path.name}")
            print(f"  ({len(overview)} rows across "
                  f"{overview['dataset'].nunique()} dataset(s))")

    # -- Combined cross-folder overview (when running both) ------------------
    if len(results_folders) > 1 and grand_frames:
        combined = pd.concat(grand_frames, ignore_index=True)
        combined_dir = PROJECT_ROOT / "results_combined_statistical_tests"
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_path = combined_dir / "significance_overview_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n  Saved combined overview: {combined_path}")
        print(f"  ({len(combined)} rows across "
              f"{combined['results_set'].nunique()} result set(s), "
              f"{combined['dataset'].nunique()} dataset(s))")

    print(f"\n{'#' * 70}")
    print(f"  ALL SIGNIFICANCE TESTS COMPLETE")
    for rf in results_folders:
        print(f"  Output directory: {PROJECT_ROOT / rf / 'statistical_tests'}")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()

