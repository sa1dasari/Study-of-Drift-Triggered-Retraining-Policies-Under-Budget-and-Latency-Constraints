"""
Summary Results Visualization Script.

Provides both a reusable function (plot_summary_for_policy) and a
standalone entry point that generates plots from any policy's summary CSV.

Usage:
    python plot_summary.py                                  # all policies (incl. 10-seed)
    python plot_summary.py --policy periodic_10seed         # 10-seed periodic only
    python plot_summary.py --policy error_threshold_10seed  # 10-seed error-threshold only
    python plot_summary.py --policy drift_triggered_10seed  # 10-seed drift-triggered only
    python plot_summary.py --policy no_retrain_10seed       # 10-seed no-retrain baseline
"""

import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ── Known label mappings ────────────────────────────────────────────────
# Extend these dicts when new budget or latency levels are added.
_LATENCY_LABELS = {
    3:    'Near-Zero (3)',
    11:   'Low (11)',
    105:  'Med (105)',
    520:  'High (520)',
    2050: 'Extreme-High (2050)',
}

_BUDGET_LABELS = {
    5:  'Low (5)',
    10: 'Med (10)',
    20: 'High (20)',
}


def _lookup_label(value, known, kind='value'):
    """Return the human-readable label for *value*, or warn and fall back."""
    if value in known:
        return known[value]
    warnings.warn(
        f"Unmapped {kind} = {value}; no entry in label dict. "
        f"Add it to the corresponding mapping in plot_summary.py."
    )
    return f'{value}'


def plot_summary_for_policy(csv_path, output_path, policy_name):
    """Generate comprehensive 2×3 summary plots from a summary CSV.

    Args:
        csv_path (str):    Path to the summary CSV (one row per run).
        output_path (str): Path where the PNG will be saved.
        policy_name (str): Human-readable policy name for plot titles.
    """
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} experiment runs from {csv_path}")
    print(f"  Drift types : {sorted(df['drift_type'].unique())}")
    print(f"  Budgets     : {sorted(df['budget'].unique())}")
    print(f"  Latencies   : {sorted(df['retrain_latency'].unique())}")

    # ── Derived columns ─────────────────────────────────────────────────
    df['total_latency'] = df['retrain_latency'] + df['deploy_latency']
    df['latency_label'] = df['total_latency'].apply(
        lambda v: _lookup_label(v, _LATENCY_LABELS, kind='total_latency'))
    df['budget_label'] = df['budget'].apply(
        lambda v: _lookup_label(v, _BUDGET_LABELS, kind='budget'))

    # ── Dynamic ordering (sorted by numeric value) ──────────────────────
    latency_order = [
        _lookup_label(v, _LATENCY_LABELS, kind='total_latency')
        for v in sorted(df['total_latency'].unique())
    ]
    budget_order = [
        _lookup_label(v, _BUDGET_LABELS, kind='budget')
        for v in sorted(df['budget'].unique())
    ]

    # ── Figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'{policy_name} Policy: Summary Results Across All Configurations',
                 fontsize=14, fontweight='bold')


    colors_drift = {'abrupt': '#3498db', 'gradual': '#e74c3c', 'recurring': '#9b59b6'}
    markers = {'abrupt': 'o', 'gradual': 's', 'recurring': 'D'}
    colors_lat = ['#2ecc71', '#f39c12', '#e74c3c']
    colors_bud = ['#2ecc71', '#f39c12', '#e74c3c']

    # ── Plot 1: Accuracy vs Latency by Drift Type ──────────────────────
    ax1 = axes[0, 0]
    for drift_type in sorted(df['drift_type'].unique()):
        df_drift = df[df['drift_type'] == drift_type]
        grouped = df_drift.groupby('total_latency')['overall_accuracy'].mean()
        ax1.plot(grouped.index, grouped.values,
                 marker=markers.get(drift_type, 'o'),
                 color=colors_drift.get(drift_type, 'gray'),
                 linewidth=2, markersize=8,
                 label=f'{drift_type.capitalize()} Drift')
    ax1.set_xlabel('Total Latency (timesteps)', fontsize=10)
    ax1.set_ylabel('Overall Accuracy', fontsize=10)
    ax1.set_title('Accuracy vs Latency by Drift Type', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── Plot 2: Mean Accuracy by Drift Type ────────────────────────────
    ax2 = axes[0, 1]
    drift_means = df.groupby('drift_type')['overall_accuracy'].agg(['mean', 'std'])
    bar_colors = [colors_drift.get(dt, 'gray') for dt in drift_means.index]
    bars = ax2.bar(drift_means.index.str.capitalize(), drift_means['mean'],
                   yerr=drift_means['std'], color=bar_colors, alpha=0.8, capsize=5)
    for bar, (idx, row) in zip(bars, drift_means.iterrows()):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + row['std'] + 0.005,
                 f'{row["mean"]:.4f}', ha='center', va='bottom', fontsize=10)
    ax2.set_xlabel('Drift Type', fontsize=10)
    ax2.set_ylabel('Mean Overall Accuracy', fontsize=10)
    ax2.set_title('Mean Accuracy by Drift Type', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    # ── Plot 3: Heatmap – Accuracy across Budget × Latency ────────────
    ax3 = axes[0, 2]
    pivot = df.pivot_table(values='overall_accuracy', index='budget_label',
                           columns='latency_label', aggfunc='mean')
    pivot = pivot.reindex(index=[b for b in budget_order if b in pivot.index],
                          columns=[l for l in latency_order if l in pivot.columns])
    im = ax3.imshow(pivot.values, cmap='viridis', aspect='auto')
    ax3.set_xticks(np.arange(len(pivot.columns)))
    ax3.set_yticks(np.arange(len(pivot.index)))
    ax3.set_xticklabels(pivot.columns)
    ax3.set_yticklabels(pivot.index)
    ax3.set_xlabel('Latency', fontsize=10)
    ax3.set_ylabel('Budget', fontsize=10)
    ax3.set_title('Accuracy Heatmap: Budget × Latency', fontsize=11)
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Accuracy', fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                vmin, vmax = pivot.values[~np.isnan(pivot.values)].min(), pivot.values[~np.isnan(pivot.values)].max()
                mid = (vmin + vmax) / 2
                text_color = 'white' if val < mid else 'black'
                ax3.text(j, i, f'{val:.3f}', ha='center', va='center',
                         color=text_color, fontsize=9)

    # ── Plot 4: Budget Utilization by Budget & Latency ─────────────────
    ax4 = axes[1, 0]
    grouped = df.groupby(['budget_label', 'latency_label'])['budget_utilization'].mean().unstack()
    grouped = grouped.reindex(index=[b for b in budget_order if b in grouped.index],
                              columns=[l for l in latency_order if l in grouped.columns])
    x = np.arange(len(grouped.index))
    width = 0.25
    for i, (latency, color) in enumerate(zip(latency_order, colors_lat)):
        if latency in grouped.columns:
            vals = grouped[latency].values
            bars = ax4.bar(x + i*width, vals, width, label=latency, color=color, alpha=0.8)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                             f'{val:.0%}', ha='center', va='bottom', fontsize=8)
    ax4.set_xlabel('Budget', fontsize=10)
    ax4.set_ylabel('Budget Utilization', fontsize=10)
    ax4.set_title('Budget Utilization by Budget & Latency', fontsize=11)
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(grouped.index)
    ax4.legend(title='Latency', fontsize=8)
    ax4.set_ylim(0, 1.15)
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.grid(axis='y', alpha=0.3)

    # ── Plot 5: Retrains After Drift by Drift Type & Latency ──────────
    ax5 = axes[1, 1]
    grouped = df.groupby(['drift_type', 'latency_label'])['retrains_after_drift'].mean().unstack()
    grouped = grouped.reindex(columns=[l for l in latency_order if l in grouped.columns])
    x = np.arange(len(grouped.index))
    width = 0.25
    for i, (latency, color) in enumerate(zip(latency_order, colors_lat)):
        if latency in grouped.columns:
            vals = grouped[latency].values
            bars = ax5.bar(x + i*width, vals, width, label=latency, color=color, alpha=0.8)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                             f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    ax5.set_xlabel('Drift Type', fontsize=10)
    ax5.set_ylabel('Avg Retrains After Drift', fontsize=10)
    ax5.set_title('Retrains After Drift by Drift Type & Latency', fontsize=11)
    ax5.set_xticks(x + width)
    ax5.set_xticklabels([dt.capitalize() for dt in grouped.index])
    ax5.legend(title='Latency', fontsize=8)
    ax5.grid(axis='y', alpha=0.3)

    # ── Plot 6: Retrains After Drift by Drift Type & Budget ───────────
    ax6 = axes[1, 2]
    grouped = df.groupby(['drift_type', 'budget_label'])['retrains_after_drift'].mean().unstack()
    grouped = grouped.reindex(columns=[b for b in budget_order if b in grouped.columns])
    x = np.arange(len(grouped.index))
    width = 0.25
    for i, (budget, color) in enumerate(zip(budget_order, colors_bud)):
        if budget in grouped.columns:
            vals = grouped[budget].values
            bars = ax6.bar(x + i*width, vals, width, label=budget, color=color, alpha=0.8)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                             f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    ax6.set_xlabel('Drift Type', fontsize=10)
    ax6.set_ylabel('Avg Retrains After Drift', fontsize=10)
    ax6.set_title('Retrains After Drift by Drift Type & Budget', fontsize=11)
    ax6.set_xticks(x + width)
    ax6.set_xticklabels([dt.capitalize() for dt in grouped.index])
    ax6.legend(title='Budget', fontsize=8)
    ax6.grid(axis='y', alpha=0.3)

    # ── Save ────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graph saved: {output_path}")

    # ── Console summary stats ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"SUMMARY STATISTICS – {policy_name} Policy")
    print(f"{'=' * 60}")
    for drift_type in sorted(df['drift_type'].unique()):
        df_d = df[df['drift_type'] == drift_type]
        print(f"\n{drift_type.upper()} DRIFT:")
        print(f"  Overall Accuracy       : {df_d['overall_accuracy'].mean():.4f} ± {df_d['overall_accuracy'].std():.4f}")
        print(f"  Pre-Drift Accuracy     : {df_d['pre_drift_accuracy'].mean():.4f}")
        print(f"  Post-Drift Accuracy    : {df_d['post_drift_accuracy'].mean():.4f}")
        print(f"  Accuracy Drop          : {df_d['accuracy_drop'].mean():.4f}")
        print(f"  Avg Budget Utilization : {df_d['budget_utilization'].mean():.1%}")
        print(f"  Avg Retrains After Drift: {df_d['retrains_after_drift'].mean():.1f}")


def plot_summary_for_no_retrain(csv_path, output_path, policy_name):
    """Generate a 2×2 baseline-specific summary plot (no budget/latency axes).

    Args:
        csv_path (str):    Path to the no-retrain summary CSV (one row per run).
        output_path (str): Path where the PNG will be saved.
        policy_name (str): Human-readable policy name for plot titles.
    """
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} baseline runs from {csv_path}")
    print(f"  Drift types : {sorted(df['drift_type'].unique())}")
    print(f"  Seeds       : {len(df)} total rows (0 retrains expected)")

    colors_drift = {'abrupt': '#3498db', 'gradual': '#e74c3c', 'recurring': '#9b59b6'}
    drift_types = sorted(df['drift_type'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{policy_name}: Baseline Floor (partial_fit only, 0 retrains)',
                 fontsize=14, fontweight='bold')

    # ── Plot 1: Mean Overall Accuracy by Drift Type ─────────────────────
    ax1 = axes[0, 0]
    stats = df.groupby('drift_type')['overall_accuracy'].agg(['mean', 'std'])
    bar_colors = [colors_drift.get(dt, 'gray') for dt in stats.index]
    bars = ax1.bar(stats.index.str.capitalize(), stats['mean'],
                   yerr=stats['std'], color=bar_colors, alpha=0.85, capsize=6)
    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + row['std'] + 0.005,
                 f"{row['mean']:.4f}", ha='center', va='bottom', fontsize=10)
    ax1.set_ylabel('Overall Accuracy', fontsize=10)
    ax1.set_title('Mean Overall Accuracy by Drift Type', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # ── Plot 2: Pre- vs Post-Drift Accuracy ─────────────────────────────
    ax2 = axes[0, 1]
    pre = df.groupby('drift_type')['pre_drift_accuracy'].mean()
    post = df.groupby('drift_type')['post_drift_accuracy'].mean()
    x = np.arange(len(drift_types))
    width = 0.35
    bars_pre = ax2.bar(x - width / 2, [pre[dt] for dt in drift_types], width,
                       label='Pre-Drift', color='#2ecc71', alpha=0.85)
    bars_post = ax2.bar(x + width / 2, [post[dt] for dt in drift_types], width,
                        label='Post-Drift', color='#e74c3c', alpha=0.85)
    for bar in list(bars_pre) + list(bars_post):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{bar.get_height():.4f}", ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([dt.capitalize() for dt in drift_types])
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.set_title('Pre-Drift vs Post-Drift Accuracy', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # ── Plot 3: Accuracy Drop (post − pre) ──────────────────────────────
    ax3 = axes[1, 0]
    drop = df.groupby('drift_type')['accuracy_drop'].agg(['mean', 'std'])
    bar_colors_drop = [colors_drift.get(dt, 'gray') for dt in drop.index]
    bars = ax3.bar(drop.index.str.capitalize(), drop['mean'],
                   yerr=drop['std'], color=bar_colors_drop, alpha=0.85, capsize=6)
    for bar, (_, row) in zip(bars, drop.iterrows()):
        offset = row['std'] + 0.005 if row['mean'] >= 0 else -(row['std'] + 0.005)
        va = 'bottom' if row['mean'] >= 0 else 'top'
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + offset,
                 f"{row['mean']:.4f}", ha='center', va=va, fontsize=10)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Accuracy Drop (post − pre)', fontsize=10)
    ax3.set_title('Accuracy Drop by Drift Type', fontsize=11)
    ax3.grid(axis='y', alpha=0.3)

    # ── Plot 4: Box Plot – Overall Accuracy Distribution per Drift Type ─
    ax4 = axes[1, 1]
    box_data = [df[df['drift_type'] == dt]['overall_accuracy'].values for dt in drift_types]
    bp = ax4.boxplot(box_data, labels=[dt.capitalize() for dt in drift_types],
                     patch_artist=True, widths=0.5)
    for patch, dt in zip(bp['boxes'], drift_types):
        patch.set_facecolor(colors_drift.get(dt, 'gray'))
        patch.set_alpha(0.7)
    ax4.set_ylabel('Overall Accuracy', fontsize=10)
    ax4.set_title('Accuracy Distribution Across Seeds', fontsize=11)
    ax4.grid(axis='y', alpha=0.3)

    # ── Save ────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graph saved: {output_path}")

    # ── Console summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"BASELINE SUMMARY – {policy_name}")
    print(f"{'=' * 60}")
    for dt in drift_types:
        df_d = df[df['drift_type'] == dt]
        print(f"\n{dt.upper()} DRIFT:")
        print(f"  Overall Accuracy   : {df_d['overall_accuracy'].mean():.4f} ± {df_d['overall_accuracy'].std():.4f}")
        print(f"  Pre-Drift Accuracy : {df_d['pre_drift_accuracy'].mean():.4f}")
        print(f"  Post-Drift Accuracy: {df_d['post_drift_accuracy'].mean():.4f}")
        print(f"  Accuracy Drop      : {df_d['accuracy_drop'].mean():.4f}")
        print(f"  Total Retrains     : {df_d['total_retrains'].mean():.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# Legacy wrapper – keeps `python plot_summary.py` working for drift-triggered
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary():
    """Generate plots for the drift-triggered policy (backward-compatible)."""
    plot_summary_for_policy(
        csv_path='results/synthetic/csv/summary_results_drift_triggered_retrain_3seed.csv',
        output_path='results/synthetic/plots/summary_results_plot_drift_triggered_retrain_3seed.png',
        policy_name='Drift-Triggered (ADWIN)',
    )


if __name__ == "__main__":
    import sys

    policy_map = {
        'periodic': (
            'results/synthetic/csv/summary_results_periodic_retrain_3seed.csv',
            'results/synthetic/plots/summary_results_plot_periodic_retrain_3seed.png',
            'Periodic',
        ),
        'periodic_10seed': (
            'results/synthetic/csv/summary_results_periodic_retrain_10seed.csv',
            'results/synthetic/plots/summary_results_plot_periodic_retrain_10seed.png',
            'Periodic (10 seeds)',
        ),
        'error_threshold': (
            'results/synthetic/csv/summary_results_error_threshold_retrain_3seed.csv',
            'results/synthetic/plots/summary_results_plot_error_threshold_retrain_3seed.png',
            'Error-Threshold',
        ),
        'error_threshold_10seed': (
            'results/synthetic/csv/summary_results_error_threshold_retrain_10seed.csv',
            'results/synthetic/plots/summary_results_plot_error_threshold_retrain_10seed.png',
            'Error-Threshold (10 seeds)',
        ),
        'drift_triggered': (
            'results/synthetic/csv/summary_results_drift_triggered_retrain_3seed.csv',
            'results/synthetic/plots/summary_results_plot_drift_triggered_retrain_3seed.png',
            'Drift-Triggered (ADWIN)',
        ),
        'drift_triggered_10seed': (
            'results/synthetic/csv/summary_results_drift_triggered_retrain_10seed.csv',
            'results/synthetic/plots/summary_results_plot_drift_triggered_retrain_10seed.png',
            'Drift-Triggered (ADWIN, 10 seeds)',
        ),
        'no_retrain': (
            'results/synthetic/csv/summary_results_no_retrain_3seed.csv',
            'results/synthetic/plots/summary_results_plot_no_retrain_3seed.png',
            'No-Retrain Baseline',
        ),
        'no_retrain_10seed': (
            'results/synthetic/csv/summary_results_no_retrain_10seed.csv',
            'results/synthetic/plots/summary_results_plot_no_retrain_10seed.png',
            'No-Retrain Baseline (10 seeds)',
        ),
    }

    # Policies that use the no-retrain-specific plotter
    _NO_RETRAIN_KEYS = {'no_retrain', 'no_retrain_10seed'}

    if '--policy' in sys.argv:
        idx = sys.argv.index('--policy') + 1
        policies = [sys.argv[idx]] if idx < len(sys.argv) else list(policy_map.keys())
    else:
        policies = list(policy_map.keys())  # run ALL policies by default

    for policy in policies:
        if policy not in policy_map:
            print(f"Unknown policy '{policy}', skipping. Available: {list(policy_map.keys())}")
            continue
        csv_path, output_path, policy_name = policy_map[policy]
        print(f"\n{'=' * 60}")
        print(f"Generating plot for: {policy_name}")
        print(f"{'=' * 60}")
        try:
            if policy in _NO_RETRAIN_KEYS:
                plot_summary_for_no_retrain(csv_path, output_path, policy_name)
            else:
                plot_summary_for_policy(csv_path, output_path, policy_name)
        except FileNotFoundError:
            print(f"  ⚠ CSV not found: {csv_path} — skipping {policy_name}")
