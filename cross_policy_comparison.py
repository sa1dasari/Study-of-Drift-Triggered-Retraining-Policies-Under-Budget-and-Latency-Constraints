"""
Cross-Policy Comparison — Head-to-head analysis across all retraining
policies for synthetic, LUFlow, and LendingClub datasets.

Produces four key outputs per dataset:
  1. Table & heatmap  : Mean post-drift accuracy by policy × drift type
                        (averaged over seeds).  "Table 1 / Figure 1" of the paper.
  2. Budget-faceted   : Same breakdown faceted by budget level (K=5/10/20),
                        showing how budget changes the policy ranking.
  3. Budget efficiency : Accuracy gained per retrain used, by policy.
                        Highlights error-threshold's pre-drift budget waste.
  4. Latency sensitivity: Accuracy degradation as latency increases from
                        low → medium → high (→ extreme if available).
                        Shows periodic's interval-vs-latency collision.

Usage:
    python cross_policy_comparison.py                    # all 3 datasets
    python cross_policy_comparison.py --dataset synthetic
    python cross_policy_comparison.py --dataset luflow
    python cross_policy_comparison.py --dataset lendingclub
    python cross_policy_comparison.py --seeds 3          # use 3-seed CSVs
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Project root ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ─── Display names & consistent colours ──────────────────────────────────
POLICY_ORDER = ["periodic", "error_threshold", "drift_triggered", "no_retrain"]
POLICY_DISPLAY = {
    "periodic":         "Periodic",
    "error_threshold":  "Error-Threshold",
    "drift_triggered":  "Drift-Triggered",
    "no_retrain":       "No-Retrain",
}
POLICY_COLORS = {
    "periodic":         "#3498db",
    "error_threshold":  "#e67e22",
    "drift_triggered":  "#2ecc71",
    "no_retrain":       "#95a5a6",
}
POLICY_MARKERS = {
    "periodic":         "o",
    "error_threshold":  "s",
    "drift_triggered":  "D",
    "no_retrain":       "X",
}

DRIFT_ORDER = ["abrupt", "gradual", "recurring"]
DRIFT_COLORS = {
    "abrupt":    "#3498db",
    "gradual":   "#e74c3c",
    "recurring": "#9b59b6",
}

BUDGET_ORDER = [5, 10, 20]
BUDGET_COLORS = {5: "#2ecc71", 10: "#f39c12", 20: "#e74c3c"}

LATENCY_LABELS = {
    0:    "None (0)",
    3:    "Near-Zero (3)",
    11:   "Low (11)",
    105:  "Med (105)",
    520:  "High (520)",
    2050: "Extreme (2050)",
}


# =====================================================================
#  DATA LOADING — auto-discover & merge all policy CSVs for a dataset
# =====================================================================

def _csv_search_patterns(dataset, seed_label):
    """Return (dir, glob_patterns) for a given dataset."""
    if dataset == "synthetic":
        base = PROJECT_ROOT / "results_without_retrain" / "synthetic" / "csv"
        patterns = [
            f"summary_results_periodic_retrain_{seed_label}.csv",
            f"summary_results_error_threshold_retrain_{seed_label}.csv",
            f"summary_results_drift_triggered_retrain_{seed_label}.csv",
            f"summary_results_no_retrain_{seed_label}.csv",
        ]
    elif dataset == "luflow":
        base = PROJECT_ROOT / "results_without_retrain" / "luflow" / "csv"
        patterns = [
            f"luflow_summary_periodic_retrain_{seed_label}.csv",
            f"luflow_summary_error_threshold_retrain_{seed_label}.csv",
            f"luflow_summary_drift_triggered_retrain_{seed_label}.csv",
            f"luflow_summary_no_retrain_{seed_label}.csv",
        ]
    elif dataset == "lendingclub":
        base = PROJECT_ROOT / "results_without_retrain" / "lendingclub" / "csv"
        patterns = [
            f"lendingclub_summary_periodic_retrain_{seed_label}.csv",
            f"lendingclub_summary_error_threshold_retrain_{seed_label}.csv",
            f"lendingclub_summary_drift_triggered_retrain_{seed_label}.csv",
            f"lendingclub_summary_no_retrain_{seed_label}.csv",
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    return base, patterns


def load_merged(dataset, seed_label="10seed"):
    """Load and merge all available policy CSVs for *dataset*.

    Tries ``seed_label`` first; falls back to '3seed' if files are missing.
    Returns a single DataFrame with a ``total_latency`` column added.
    """
    base, patterns = _csv_search_patterns(dataset, seed_label)
    frames = []
    for pat in patterns:
        fpath = base / pat
        if fpath.exists():
            df = pd.read_csv(fpath)
            frames.append(df)
        else:
            print(f"  ⚠  Not found (skipping): {fpath.name}")

    # Fallback: try 3seed if nothing found with requested label
    if not frames and seed_label != "3seed":
        print(f"  Falling back to 3seed CSVs for {dataset} ...")
        return load_merged(dataset, seed_label="3seed")

    if not frames:
        print(f"  ✗  No CSVs found for {dataset}. Skipping.")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged["total_latency"] = merged["retrain_latency"] + merged["deploy_latency"]
    merged["latency_label"] = merged["total_latency"].map(
        lambda v: LATENCY_LABELS.get(v, f"{v}")
    )
    return merged


# =====================================================================
#  HELPER — consistent policy label on an axis
# =====================================================================

def _policy_label(p):
    return POLICY_DISPLAY.get(p, p)


def _active_policies(df):
    """Return policies present in df, in canonical order."""
    present = set(df["policy_type"].unique())
    return [p for p in POLICY_ORDER if p in present]


def _active_policies_retrain(df):
    """Active policies that actually retrain (excludes no_retrain)."""
    return [p for p in _active_policies(df) if p != "no_retrain"]


# =====================================================================
#  OUTPUT 1 — Table & Heatmap: post-drift accuracy by policy × drift
# =====================================================================

def table_postdrift_by_policy_drift(df, dataset, out_dir):
    """Produce Table 1: mean post-drift accuracy (policy × drift type).

    For policies with a budget/latency grid, we average over all
    budget × latency × seed combinations.  For no_retrain we average
    over seeds only.  This gives the "grand-mean" view.
    """
    policies = _active_policies(df)
    drifts = [d for d in DRIFT_ORDER if d in df["drift_type"].unique()]

    # --- Pivot table ---
    pivot = (
        df.groupby(["policy_type", "drift_type"])["post_drift_accuracy"]
        .agg(["mean", "std"])
        .unstack("drift_type")
    )
    # Reorder
    pivot = pivot.reindex(index=[p for p in policies if p in pivot.index],
                          columns=pd.MultiIndex.from_product([["mean", "std"], drifts]))

    # --- Print to console ---
    header = f"{'Policy':<22}" + "".join(f"{d.capitalize():>18}" for d in drifts)
    sep = "-" * len(header)
    print(f"\n{'=' * 70}")
    print(f" TABLE 1 — Mean Post-Drift Accuracy by Policy × Drift [{dataset.upper()}]")
    print(f"{'=' * 70}")
    print(header)
    print(sep)
    for p in policies:
        if p not in pivot.index:
            continue
        row = f"{_policy_label(p):<22}"
        for d in drifts:
            m = pivot.loc[p, ("mean", d)]
            s = pivot.loc[p, ("std", d)]
            row += f"  {m:.4f} ± {s:.4f}  "
        print(row)
    print(sep)

    # --- Save CSV table ---
    tbl = (
        df.groupby(["policy_type", "drift_type"])["post_drift_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    tbl.to_csv(out_dir / f"table1_postdrift_accuracy_{dataset}.csv", index=False)

    # --- Heatmap figure ---
    heat_data = (
        df.groupby(["policy_type", "drift_type"])["post_drift_accuracy"]
        .mean()
        .unstack("drift_type")
    )
    heat_data = heat_data.reindex(
        index=[p for p in policies if p in heat_data.index],
        columns=[d for d in drifts if d in heat_data.columns],
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(heat_data.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(heat_data.columns)))
    ax.set_yticks(np.arange(len(heat_data.index)))
    ax.set_xticklabels([d.capitalize() for d in heat_data.columns], fontsize=11)
    ax.set_yticklabels([_policy_label(p) for p in heat_data.index], fontsize=11)
    ax.set_xlabel("Drift Type", fontsize=12)
    ax.set_ylabel("Policy", fontsize=12)
    ax.set_title(
        f"Mean Post-Drift Accuracy — Policy × Drift Type\n[{dataset.upper()}]",
        fontsize=13, fontweight="bold",
    )

    # Annotate cells
    vmin, vmax = np.nanmin(heat_data.values), np.nanmax(heat_data.values)
    mid = (vmin + vmax) / 2
    for i in range(len(heat_data.index)):
        for j in range(len(heat_data.columns)):
            v = heat_data.values[i, j]
            if not np.isnan(v):
                clr = "white" if v < mid else "black"
                ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                        color=clr, fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Accuracy", shrink=0.85)
    plt.tight_layout()
    fpath = out_dir / f"fig1_postdrift_heatmap_{dataset}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath.name}")


# =====================================================================
#  OUTPUT 2 — Budget-faceted post-drift accuracy
# =====================================================================

def figure_budget_faceted(df, dataset, out_dir):
    """Bar chart of post-drift accuracy by policy, faceted by budget.

    No-retrain is drawn as a horizontal dashed line in each panel.
    """
    drifts = [d for d in DRIFT_ORDER if d in df["drift_type"].unique()]
    budgets = sorted(b for b in df["budget"].unique() if b > 0)
    policies_retrain = _active_policies_retrain(df)

    if not budgets or not policies_retrain:
        print(f"  ⚠  Skipping budget-faceted plot for {dataset} (no budget data)")
        return

    # Baseline reference (no_retrain) — one value per drift type
    baseline = (
        df[df["policy_type"] == "no_retrain"]
        .groupby("drift_type")["post_drift_accuracy"]
        .mean()
    )

    n_budgets = len(budgets)
    n_drifts = len(drifts)

    fig, axes = plt.subplots(1, n_budgets, figsize=(6 * n_budgets, 5), sharey=True)
    if n_budgets == 1:
        axes = [axes]

    for ax, budget in zip(axes, budgets):
        df_b = df[(df["budget"] == budget) & (df["policy_type"] != "no_retrain")]
        agg = df_b.groupby(["policy_type", "drift_type"])["post_drift_accuracy"].agg(["mean", "std"])

        x = np.arange(n_drifts)
        width = 0.8 / max(len(policies_retrain), 1)

        for i, p in enumerate(policies_retrain):
            means, stds = [], []
            for d in drifts:
                if (p, d) in agg.index:
                    means.append(agg.loc[(p, d), "mean"])
                    stds.append(agg.loc[(p, d), "std"])
                else:
                    means.append(np.nan)
                    stds.append(0)
            ax.bar(
                x + i * width - 0.4 + width / 2,
                means, width, yerr=stds,
                label=_policy_label(p),
                color=POLICY_COLORS.get(p, "gray"),
                alpha=0.85, capsize=3,
            )

        # Baseline reference lines
        for j, d in enumerate(drifts):
            if d in baseline.index:
                ax.hlines(
                    baseline[d], j - 0.45, j + 0.45,
                    colors=POLICY_COLORS["no_retrain"],
                    linestyles="dashed", linewidth=1.5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in drifts], fontsize=10)
        ax.set_title(f"Budget K = {budget}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Mean Post-Drift Accuracy", fontsize=11)
    # Single legend for the figure
    handles, labels = axes[0].get_legend_handles_labels()
    # Add baseline entry
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color=POLICY_COLORS["no_retrain"],
                          linestyle="--", linewidth=1.5))
    labels.append("No-Retrain (Baseline)")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"Post-Drift Accuracy by Policy × Drift Type — Faceted by Budget\n[{dataset.upper()}]",
        fontsize=13, fontweight="bold", y=1.08,
    )
    plt.tight_layout()
    fpath = out_dir / f"fig2_budget_faceted_{dataset}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath.name}")

    # --- Also save the underlying table ---
    tbl = (
        df[df["budget"] > 0]
        .groupby(["budget", "policy_type", "drift_type"])["post_drift_accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    tbl.to_csv(out_dir / f"table2_budget_faceted_{dataset}.csv", index=False)


# =====================================================================
#  OUTPUT 3 — Budget Efficiency: accuracy-per-retrain & waste analysis
# =====================================================================

def figure_budget_efficiency(df, dataset, out_dir):
    """Two-panel figure:
      Left  — Accuracy gained per retrain (post-drift acc − baseline) / retrains_after_drift.
      Right — Pre-drift budget waste: fraction of total retrains spent before drift.

    Both are grouped bars: one group per drift type, one bar per policy.
    """
    drifts = [d for d in DRIFT_ORDER if d in df["drift_type"].unique()]
    policies_retrain = _active_policies_retrain(df)

    if not policies_retrain:
        print(f"  ⚠  Skipping budget-efficiency plot for {dataset}")
        return

    # Baseline reference
    baseline = (
        df[df["policy_type"] == "no_retrain"]
        .groupby("drift_type")["post_drift_accuracy"]
        .mean()
    )

    # --- Aggregate active-policy data ---
    df_active = df[df["policy_type"] != "no_retrain"].copy()
    agg = (
        df_active
        .groupby(["policy_type", "drift_type"])
        .agg(
            post_acc=("post_drift_accuracy", "mean"),
            total_retrains=("total_retrains", "mean"),
            retrains_before=("retrains_before_drift", "mean"),
            retrains_after=("retrains_after_drift", "mean"),
        )
    )

    # Accuracy gain per retrain (use retrains_after_drift, not total)
    efficiency = {}
    waste = {}
    for p in policies_retrain:
        eff_row, waste_row = {}, {}
        for d in drifts:
            if (p, d) not in agg.index:
                eff_row[d] = np.nan
                waste_row[d] = np.nan
                continue
            row = agg.loc[(p, d)]
            bl = baseline.get(d, np.nan)
            gain = row["post_acc"] - bl
            r_after = row["retrains_after"]
            eff_row[d] = gain / r_after if r_after > 0 else 0.0
            total_r = row["total_retrains"]
            waste_row[d] = row["retrains_before"] / total_r if total_r > 0 else 0.0
        efficiency[p] = eff_row
        waste[p] = waste_row

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(drifts))
    width = 0.8 / max(len(policies_retrain), 1)

    # LEFT — accuracy gain per retrain after drift
    for i, p in enumerate(policies_retrain):
        vals = [efficiency[p].get(d, 0) for d in drifts]
        ax1.bar(
            x + i * width - 0.4 + width / 2,
            vals, width,
            label=_policy_label(p),
            color=POLICY_COLORS.get(p, "gray"),
            alpha=0.85,
        )
        for j, v in enumerate(vals):
            if not np.isnan(v):
                ax1.text(
                    x[j] + i * width - 0.4 + width / 2, v + 0.001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8,
                    rotation=45,
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels([d.capitalize() for d in drifts], fontsize=10)
    ax1.set_ylabel("Δ Post-Drift Acc / Retrain After Drift", fontsize=10)
    ax1.set_title("Accuracy Gain per Retrain Used", fontsize=12, fontweight="bold")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # RIGHT — pre-drift budget waste (fraction of retrains before drift)
    for i, p in enumerate(policies_retrain):
        vals = [waste[p].get(d, 0) for d in drifts]
        bars = ax2.bar(
            x + i * width - 0.4 + width / 2,
            vals, width,
            label=_policy_label(p),
            color=POLICY_COLORS.get(p, "gray"),
            alpha=0.85,
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v) and v > 0.01:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{v:.0%}", ha="center", va="bottom", fontsize=9,
                )

    ax2.set_xticks(x)
    ax2.set_xticklabels([d.capitalize() for d in drifts], fontsize=10)
    ax2.set_ylabel("Fraction of Retrains Before Drift", fontsize=10)
    ax2.set_title("Pre-Drift Budget Waste", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="_50% mark")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Budget Efficiency — Accuracy per Retrain & Pre-Drift Waste\n[{dataset.upper()}]",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fpath = out_dir / f"fig3_budget_efficiency_{dataset}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath.name}")

    # --- CSV ---
    rows = []
    for p in policies_retrain:
        for d in drifts:
            rows.append({
                "policy_type": p,
                "drift_type": d,
                "acc_gain_per_retrain": efficiency[p].get(d, np.nan),
                "pre_drift_waste_frac": waste[p].get(d, np.nan),
            })
    pd.DataFrame(rows).to_csv(
        out_dir / f"table3_budget_efficiency_{dataset}.csv", index=False,
    )


# =====================================================================
#  OUTPUT 4 — Latency Sensitivity Comparison
# =====================================================================

def figure_latency_sensitivity(df, dataset, out_dir):
    """Line plot: post-drift accuracy vs total latency for each policy.

    No-retrain is shown as a flat reference line.
    One subplot per drift type, all policies overlaid.
    """
    drifts = [d for d in DRIFT_ORDER if d in df["drift_type"].unique()]
    policies = _active_policies(df)

    # Identify latency levels present (excluding 0 for plotting)
    latencies_all = sorted(df["total_latency"].unique())
    latencies_active = [l for l in latencies_all if l > 0]
    if not latencies_active:
        print(f"  ⚠  Skipping latency sensitivity plot for {dataset} (no latency data)")
        return

    n_drifts = len(drifts)
    fig, axes = plt.subplots(1, n_drifts, figsize=(6 * n_drifts, 5), sharey=True)
    if n_drifts == 1:
        axes = [axes]

    for ax, drift in zip(axes, drifts):
        df_d = df[df["drift_type"] == drift]

        for p in policies:
            df_p = df_d[df_d["policy_type"] == p]

            if p == "no_retrain":
                # Flat baseline
                bl_mean = df_p["post_drift_accuracy"].mean()
                ax.axhline(
                    bl_mean, color=POLICY_COLORS[p],
                    linestyle="--", linewidth=1.5,
                    label=_policy_label(p) + f" ({bl_mean:.4f})",
                )
            else:
                # Group by total_latency
                grp = df_p.groupby("total_latency")["post_drift_accuracy"].agg(["mean", "std"])
                grp = grp.reindex(latencies_active)
                ax.errorbar(
                    grp.index, grp["mean"], yerr=grp["std"],
                    marker=POLICY_MARKERS.get(p, "o"),
                    color=POLICY_COLORS.get(p, "gray"),
                    linewidth=2, markersize=7, capsize=3,
                    label=_policy_label(p),
                )

        ax.set_xlabel("Total Latency (timesteps)", fontsize=10)
        ax.set_title(f"{drift.capitalize()} Drift", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)
        # Use log scale if latency range is wide
        if latencies_active[-1] / max(latencies_active[0], 1) > 20:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.get_major_formatter().set_scientific(False)

    axes[0].set_ylabel("Mean Post-Drift Accuracy", fontsize=11)
    fig.suptitle(
        f"Latency Sensitivity — Post-Drift Accuracy vs. Latency\n[{dataset.upper()}]",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fpath = out_dir / f"fig4_latency_sensitivity_{dataset}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath.name}")

    # --- Also produce a degradation table ---
    rows = []
    for p in _active_policies_retrain(df):
        for d in drifts:
            df_pd = df[(df["policy_type"] == p) & (df["drift_type"] == d)]
            grp = df_pd.groupby("total_latency")["post_drift_accuracy"].mean()
            low = grp.get(latencies_active[0], np.nan) if latencies_active else np.nan
            high = grp.get(latencies_active[-1], np.nan) if latencies_active else np.nan
            rows.append({
                "policy_type": p,
                "drift_type": d,
                "acc_low_latency": low,
                "acc_high_latency": high,
                "degradation": low - high if not (np.isnan(low) or np.isnan(high)) else np.nan,
            })
    pd.DataFrame(rows).to_csv(
        out_dir / f"table4_latency_sensitivity_{dataset}.csv", index=False,
    )


# =====================================================================
#  COMBINED CROSS-DATASET SUMMARY
# =====================================================================

def cross_dataset_summary(all_merged, out_dir):
    """Produce a single figure comparing policy rankings across datasets.

    One grouped bar per dataset, bars colored by policy, y = mean
    post-drift accuracy (averaged over all drift types, budgets, latencies,
    seeds).
    """
    datasets = list(all_merged.keys())
    if len(datasets) < 2:
        return  # nothing to compare

    policies = POLICY_ORDER  # canonical order

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets))
    n_policies = len(policies)
    width = 0.8 / n_policies

    for i, p in enumerate(policies):
        means = []
        for ds in datasets:
            df_ds = all_merged[ds]
            df_p = df_ds[df_ds["policy_type"] == p]
            means.append(df_p["post_drift_accuracy"].mean() if len(df_p) else np.nan)
        ax.bar(
            x + i * width - 0.4 + width / 2,
            means, width,
            label=_policy_label(p),
            color=POLICY_COLORS.get(p, "gray"),
            alpha=0.85,
        )
        for j, v in enumerate(means):
            if not np.isnan(v):
                ax.text(
                    x[j] + i * width - 0.4 + width / 2, v + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([ds.upper() for ds in datasets], fontsize=11)
    ax.set_ylabel("Mean Post-Drift Accuracy (all conditions)", fontsize=11)
    ax.set_title(
        "Cross-Dataset Policy Comparison — Grand Mean Post-Drift Accuracy",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fpath = out_dir / "fig_cross_dataset_summary.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath.name}")

    # --- CSV ---
    rows = []
    for ds in datasets:
        df_ds = all_merged[ds]
        for p in policies:
            df_p = df_ds[df_ds["policy_type"] == p]
            if len(df_p):
                rows.append({
                    "dataset": ds,
                    "policy_type": p,
                    "mean_post_drift_accuracy": df_p["post_drift_accuracy"].mean(),
                    "std_post_drift_accuracy": df_p["post_drift_accuracy"].std(),
                    "n_runs": len(df_p),
                })
    pd.DataFrame(rows).to_csv(
        out_dir / "table_cross_dataset_summary.csv", index=False,
    )


# =====================================================================
#  ORCHESTRATOR
# =====================================================================

def run_comparison_for_dataset(dataset, seed_label, out_dir):
    """Run all four comparison analyses for one dataset."""
    print(f"\n{'#' * 70}")
    print(f"  CROSS-POLICY COMPARISON — {dataset.upper()}")
    print(f"{'#' * 70}")

    df = load_merged(dataset, seed_label)
    if df.empty:
        return None

    print(f"  Loaded {len(df)} rows across policies: "
          f"{sorted(df['policy_type'].unique())}")
    print(f"  Drift types: {sorted(df['drift_type'].unique())}")
    budgets = sorted(df["budget"].unique())
    latencies = sorted(df["total_latency"].unique())
    print(f"  Budgets: {budgets}   Latencies: {latencies}")

    ds_dir = out_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    table_postdrift_by_policy_drift(df, dataset, ds_dir)
    figure_budget_faceted(df, dataset, ds_dir)
    figure_budget_efficiency(df, dataset, ds_dir)
    figure_latency_sensitivity(df, dataset, ds_dir)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Cross-policy head-to-head comparison for all datasets.",
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
        help="Seed-set label: 3 or 10 (default: 10). Falls back to 3 if CSVs not found.",
    )
    args = parser.parse_args()

    seed_label = f"{args.seeds}seed"
    datasets = (
        ["synthetic", "luflow", "lendingclub"]
        if args.dataset == "all"
        else [args.dataset]
    )

    out_dir = PROJECT_ROOT / "results_without_retrain" / "cross_policy_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_merged = {}
    for ds in datasets:
        df = run_comparison_for_dataset(ds, seed_label, out_dir)
        if df is not None and not df.empty:
            all_merged[ds] = df

    # Cross-dataset summary (only if ≥2 datasets available)
    if len(all_merged) >= 2:
        print(f"\n{'#' * 70}")
        print(f"  CROSS-DATASET SUMMARY")
        print(f"{'#' * 70}")
        cross_dataset_summary(all_merged, out_dir)

    print(f"\n{'#' * 70}")
    print(f"  ALL COMPARISONS COMPLETE")
    print(f"  Output directory: {out_dir}")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()

