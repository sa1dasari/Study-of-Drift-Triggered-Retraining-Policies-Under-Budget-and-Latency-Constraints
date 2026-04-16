"""
With vs Without partial_fit -- Side-by-side comparison visualizations.

Loads summary CSVs from both results_with_retrain/ and results_without_retrain/
for each dataset and produces:

  1. Side-by-side heatmaps:  Mean post-drift accuracy (policy x drift) for
     "with partial_fit" vs "without partial_fit", placed next to each other
     so a reader can compare at a glance.

  2. Grouped bar chart:  The accuracy gap (with − without) per policy x drift,
     making it immediately clear which policies benefit most from partial_fit.

  3. Cross-dataset summary:  A single multi-dataset figure showing the gap
     for every dataset side by side.

Usage:
    python compare_partial_fit.py                       # all 3 datasets
    python compare_partial_fit.py --dataset synthetic   # synthetic only
    python compare_partial_fit.py --dataset luflow
    python compare_partial_fit.py --dataset lendingclub
    python compare_partial_fit.py --seeds 3             # use 3-seed CSVs
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

# --- Project root --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Display names & colours (shared with cross_policy_comparison.py) ----
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

DRIFT_ORDER = ["abrupt", "gradual", "recurring"]

MODE_COLORS = {"with": "#2980b9", "without": "#c0392b"}
MODE_LABELS = {"with": "With partial_fit", "without": "Without partial_fit"}


# =====================================================================
#  DATA LOADING
# =====================================================================

def _csv_patterns(dataset, seed_label):
    """Return list of (filename,) for a given dataset."""
    if dataset == "synthetic":
        prefix = "summary_results"
    elif dataset == "luflow":
        prefix = "luflow_summary"
    elif dataset == "lendingclub":
        prefix = "lendingclub_summary"
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    patterns = [
        f"{prefix}_periodic_retrain_{seed_label}.csv",
        f"{prefix}_error_threshold_retrain_{seed_label}.csv",
        f"{prefix}_drift_triggered_retrain_{seed_label}.csv",
        f"{prefix}_no_retrain_{seed_label}.csv",
    ]
    return patterns


def _load_mode(dataset, mode, seed_label):
    """Load and merge all policy CSVs for a dataset+mode.

    mode: 'with' -> results_with_retrain/
          'without' -> results_without_retrain/
    """
    if mode == "with":
        base = PROJECT_ROOT / "results_with_retrain" / dataset / "csv"
    else:
        base = PROJECT_ROOT / "results_without_retrain" / dataset / "csv"

    patterns = _csv_patterns(dataset, seed_label)
    frames = []
    for pat in patterns:
        fpath = base / pat
        if fpath.exists():
            frames.append(pd.read_csv(fpath))
        else:
            pass  # silently skip missing files

    # Fallback to 3seed if nothing found
    if not frames and seed_label != "3seed":
        return _load_mode(dataset, mode, "3seed")

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged["total_latency"] = merged["retrain_latency"] + merged["deploy_latency"]
    return merged


def load_both_modes(dataset, seed_label="10seed"):
    """Return (df_with, df_without) for a dataset."""
    df_with = _load_mode(dataset, "with", seed_label)
    df_without = _load_mode(dataset, "without", seed_label)
    return df_with, df_without


# =====================================================================
#  HELPERS
# =====================================================================

def _policy_label(p):
    return POLICY_DISPLAY.get(p, p)


def _active_policies(df_w, df_wo):
    """Policies present in either dataframe, in canonical order."""
    present = set()
    if not df_w.empty:
        present |= set(df_w["policy_type"].unique())
    if not df_wo.empty:
        present |= set(df_wo["policy_type"].unique())
    return [p for p in POLICY_ORDER if p in present]


def _heatmap_data(df, policies, drifts):
    """Build a (policies x drifts) DataFrame of mean post-drift accuracy."""
    if df.empty:
        return pd.DataFrame(np.nan, index=policies, columns=drifts)
    pivot = (
        df.groupby(["policy_type", "drift_type"])["post_drift_accuracy"]
        .mean()
        .unstack("drift_type")
    )
    pivot = pivot.reindex(index=policies, columns=drifts)
    return pivot


# =====================================================================
#  FIGURE 1 -- Side-by-side heatmaps
# =====================================================================

def figure_side_by_side_heatmaps(df_w, df_wo, dataset, out_dir):
    """Two heatmaps side by side: with vs without partial_fit."""
    policies = _active_policies(df_w, df_wo)
    drifts = [d for d in DRIFT_ORDER
              if (not df_w.empty and d in df_w["drift_type"].unique()) or
                 (not df_wo.empty and d in df_wo["drift_type"].unique())]

    heat_w = _heatmap_data(df_w, policies, drifts)
    heat_wo = _heatmap_data(df_wo, policies, drifts)

    # Shared colour scale
    all_vals = np.concatenate([
        heat_w.values[~np.isnan(heat_w.values)],
        heat_wo.values[~np.isnan(heat_wo.values)],
    ])
    if len(all_vals) == 0:
        print(f"  [WARN] No data for heatmaps ({dataset}). Skipping.")
        return
    vmin, vmax = all_vals.min(), all_vals.max()
    mid = (vmin + vmax) / 2

    fig, (ax_w, ax_wo) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, heat, title in [
        (ax_w,  heat_w,  "With partial_fit"),
        (ax_wo, heat_wo, "Without partial_fit"),
    ]:
        im = ax.imshow(heat.values, cmap="RdYlGn", aspect="auto",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(drifts)))
        ax.set_xticklabels([d.capitalize() for d in drifts], fontsize=11)
        ax.set_xlabel("Drift Type", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")

        # Annotate cells
        for i in range(len(policies)):
            for j in range(len(drifts)):
                v = heat.values[i, j]
                if not np.isnan(v):
                    clr = "white" if v < mid else "black"
                    ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                            color=clr, fontsize=11, fontweight="bold")

    ax_w.set_yticks(np.arange(len(policies)))
    ax_w.set_yticklabels([_policy_label(p) for p in policies], fontsize=11)
    ax_w.set_ylabel("Policy", fontsize=11)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Mean Post-Drift Accuracy")

    fig.suptitle(
        f"Post-Drift Accuracy: With vs Without partial_fit  [{dataset.upper()}]",
        fontsize=14, fontweight="bold", y=1.02,
    )

    fpath = out_dir / f"fig_heatmap_with_vs_without_{dataset}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath.name}")


# =====================================================================
#  FIGURE 2 -- Grouped bar chart: accuracy gap (with − without)
# =====================================================================

def figure_accuracy_gap(df_w, df_wo, dataset, out_dir):
    """Grouped bar chart: delta = with − without, per policy x drift."""
    policies = _active_policies(df_w, df_wo)
    drifts = [d for d in DRIFT_ORDER
              if (not df_w.empty and d in df_w["drift_type"].unique()) or
                 (not df_wo.empty and d in df_wo["drift_type"].unique())]

    heat_w = _heatmap_data(df_w, policies, drifts)
    heat_wo = _heatmap_data(df_wo, policies, drifts)
    delta = heat_w - heat_wo  # positive means partial_fit helps

    n_policies = len(policies)
    n_drifts = len(drifts)
    x = np.arange(n_drifts)
    width = 0.8 / max(n_policies, 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, p in enumerate(policies):
        vals = [delta.loc[p, d] if p in delta.index and d in delta.columns
                else np.nan for d in drifts]
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            vals, width,
            label=_policy_label(p),
            color=POLICY_COLORS.get(p, "gray"),
            alpha=0.85,
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                va = "bottom" if v >= 0 else "top"
                offset = 0.002 if v >= 0 else -0.002
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{v:+.4f}", ha="center", va=va, fontsize=8,
                    fontweight="bold",
                )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in drifts], fontsize=11)
    ax.set_xlabel("Drift Type", fontsize=11)
    ax.set_ylabel("Δ Post-Drift Accuracy  (with − without partial_fit)", fontsize=11)
    ax.set_title(
        f"Accuracy Gap: With vs Without partial_fit  [{dataset.upper()}]",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fpath = out_dir / f"fig_accuracy_gap_with_vs_without_{dataset}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath.name}")

    # --- CSV table ---
    rows = []
    for p in policies:
        for d in drifts:
            w_val = heat_w.loc[p, d] if p in heat_w.index and d in heat_w.columns else np.nan
            wo_val = heat_wo.loc[p, d] if p in heat_wo.index and d in heat_wo.columns else np.nan
            rows.append({
                "dataset": dataset,
                "policy_type": p,
                "drift_type": d,
                "post_drift_acc_with_partial_fit": w_val,
                "post_drift_acc_without_partial_fit": wo_val,
                "delta": w_val - wo_val if not (np.isnan(w_val) or np.isnan(wo_val)) else np.nan,
            })
    pd.DataFrame(rows).to_csv(
        out_dir / f"table_accuracy_gap_{dataset}.csv", index=False,
    )


# =====================================================================
#  FIGURE 3 -- Cross-dataset summary (gap grouped by dataset)
# =====================================================================

def figure_cross_dataset_gap(all_data, out_dir):
    """One grouped bar per dataset, bars coloured by policy, y = Δ accuracy."""
    datasets = list(all_data.keys())
    if len(datasets) < 2:
        return

    policies = POLICY_ORDER

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(datasets))
    n_policies = len(policies)
    width = 0.8 / n_policies

    for i, p in enumerate(policies):
        vals = []
        for ds in datasets:
            df_w, df_wo = all_data[ds]
            if df_w.empty or df_wo.empty:
                vals.append(np.nan)
                continue
            mean_w = df_w[df_w["policy_type"] == p]["post_drift_accuracy"].mean()
            mean_wo = df_wo[df_wo["policy_type"] == p]["post_drift_accuracy"].mean()
            vals.append(mean_w - mean_wo)

        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            vals, width,
            label=_policy_label(p),
            color=POLICY_COLORS.get(p, "gray"),
            alpha=0.85,
        )
        for bar, v in zip(bars, vals):
            if v is not None and not np.isnan(v):
                va = "bottom" if v >= 0 else "top"
                offset = 0.002 if v >= 0 else -0.002
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{v:+.3f}", ha="center", va=va, fontsize=8,
                    fontweight="bold", rotation=45,
                )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ds.upper() for ds in datasets], fontsize=11)
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("Δ Mean Post-Drift Accuracy  (with − without partial_fit)", fontsize=11)
    ax.set_title(
        "Cross-Dataset: partial_fit Accuracy Gain by Policy",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fpath = out_dir / "fig_cross_dataset_partial_fit_gap.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath.name}")

    # --- CSV ---
    rows = []
    for ds in datasets:
        df_w, df_wo = all_data[ds]
        for p in policies:
            mean_w = df_w[df_w["policy_type"] == p]["post_drift_accuracy"].mean() if not df_w.empty else np.nan
            mean_wo = df_wo[df_wo["policy_type"] == p]["post_drift_accuracy"].mean() if not df_wo.empty else np.nan
            rows.append({
                "dataset": ds,
                "policy_type": p,
                "mean_post_drift_with": mean_w,
                "mean_post_drift_without": mean_wo,
                "delta": mean_w - mean_wo if not (np.isnan(mean_w) or np.isnan(mean_wo)) else np.nan,
            })
    pd.DataFrame(rows).to_csv(
        out_dir / "table_cross_dataset_partial_fit_gap.csv", index=False,
    )


# =====================================================================
#  ORCHESTRATOR
# =====================================================================

def run_for_dataset(dataset, seed_label, out_dir):
    """Produce all with-vs-without figures for one dataset."""
    print(f"\n{'#' * 70}")
    print(f"  WITH vs WITHOUT partial_fit -- {dataset.upper()}")
    print(f"{'#' * 70}")

    df_w, df_wo = load_both_modes(dataset, seed_label)

    if df_w.empty and df_wo.empty:
        print(f"  [FAIL] No CSVs found for {dataset}. Skipping.")
        return None

    for label, df in [("WITH", df_w), ("WITHOUT", df_wo)]:
        n = len(df)
        pols = sorted(df["policy_type"].unique()) if n else []
        print(f"  {label:>7} partial_fit: {n:>4} rows  policies={pols}")

    ds_dir = out_dir / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    figure_side_by_side_heatmaps(df_w, df_wo, dataset, ds_dir)
    figure_accuracy_gap(df_w, df_wo, dataset, ds_dir)

    return (df_w, df_wo)


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison: with vs without partial_fit.",
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

    out_dir = PROJECT_ROOT / "results_partial_fit_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for ds in datasets:
        result = run_for_dataset(ds, seed_label, out_dir)
        if result is not None:
            all_data[ds] = result

    # Cross-dataset summary
    if len(all_data) >= 2:
        print(f"\n{'#' * 70}")
        print(f"  CROSS-DATASET PARTIAL_FIT SUMMARY")
        print(f"{'#' * 70}")
        figure_cross_dataset_gap(all_data, out_dir)

    print(f"\n{'#' * 70}")
    print(f"  ALL DONE -- output directory: {out_dir}")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()

