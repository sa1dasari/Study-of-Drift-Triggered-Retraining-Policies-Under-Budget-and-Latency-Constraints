"""
Summary Results Visualization Script.

Run this script separately to generate comprehensive plots from summary_results_periodic_retrain.csv.
Usage: python plot_summary.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_summary():
    """Generate comprehensive plots from summary_results_periodic_retrain.csv."""

    # Load data
    df = pd.read_csv('results/summary_results_periodic_retrain.csv')

    print(f"Loaded {len(df)} experiment runs")
    print(f"Drift types: {df['drift_type'].unique()}")
    print(f"Budgets: {sorted(df['budget'].unique())}")
    print(f"Latencies: {sorted(df['retrain_latency'].unique())}")

    # Define categories
    def get_latency_label(row):
        total = row['retrain_latency'] + row['deploy_latency']
        if total <= 20:
            return 'Low (11)'
        elif total <= 150:
            return 'Med (105)'
        else:
            return 'High (520)'

    def get_budget_label(budget):
        if budget <= 5:
            return 'Low (5)'
        elif budget <= 10:
            return 'Med (10)'
        else:
            return 'High (20)'

    df['latency_label'] = df.apply(get_latency_label, axis=1)
    df['budget_label'] = df['budget'].apply(get_budget_label)
    df['total_latency'] = df['retrain_latency'] + df['deploy_latency']

    # Create figure with 2x3 subplots (5 plots)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Periodic Policy: Summary Results Across All Configurations', fontsize=14, fontweight='bold')

    latency_order = ['Low (11)', 'Med (105)', 'High (520)']
    budget_order = ['Low (5)', 'Med (10)', 'High (20)']

    # =========================================================================
    # Plot 1: Line plot - Accuracy vs Latency, separate lines per drift type
    # =========================================================================
    ax1 = axes[0, 0]

    colors_drift = {'abrupt': '#3498db', 'gradual': '#e74c3c'}
    markers = {'abrupt': 'o', 'gradual': 's'}

    for drift_type in df['drift_type'].unique():
        df_drift = df[df['drift_type'] == drift_type]
        grouped = df_drift.groupby('total_latency')['overall_accuracy'].mean()
        ax1.plot(grouped.index, grouped.values, marker=markers.get(drift_type, 'o'),
                color=colors_drift.get(drift_type, 'gray'), linewidth=2, markersize=8,
                label=f'{drift_type.capitalize()} Drift')

    ax1.set_xlabel('Total Latency (timesteps)', fontsize=10)
    ax1.set_ylabel('Overall Accuracy', fontsize=10)
    ax1.set_title('Accuracy vs Latency by Drift Type', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0.72, 0.80)

    # =========================================================================
    # Plot 2: Bar chart - Mean accuracy by drift type
    # =========================================================================
    ax2 = axes[0, 1]

    drift_means = df.groupby('drift_type')['overall_accuracy'].agg(['mean', 'std'])
    colors = ['#3498db', '#e74c3c']

    bars = ax2.bar(drift_means.index, drift_means['mean'], yerr=drift_means['std'],
                   color=colors, alpha=0.8, capsize=5)

    # Add value labels
    for bar, (idx, row) in zip(bars, drift_means.iterrows()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + row['std'] + 0.005,
                f'{row["mean"]:.4f}', ha='center', va='bottom', fontsize=10)

    ax2.set_xlabel('Drift Type', fontsize=10)
    ax2.set_ylabel('Mean Overall Accuracy', fontsize=10)
    ax2.set_title('Mean Accuracy by Drift Type', fontsize=11)
    ax2.set_ylim(0.70, 0.82)
    ax2.grid(axis='y', alpha=0.3)

    # =========================================================================
    # Plot 3: Heatmap - Accuracy across Budget × Latency
    # =========================================================================
    ax3 = axes[0, 2]

    # Pivot table for heatmap
    pivot = df.pivot_table(values='overall_accuracy', index='budget_label',
                           columns='latency_label', aggfunc='mean')
    pivot = pivot.reindex(index=[b for b in budget_order if b in pivot.index],
                          columns=[l for l in latency_order if l in pivot.columns])

    im = ax3.imshow(pivot.values, cmap='viridis', aspect='auto', vmin=0.73, vmax=0.78)

    # Set labels
    ax3.set_xticks(np.arange(len(pivot.columns)))
    ax3.set_yticks(np.arange(len(pivot.index)))
    ax3.set_xticklabels(pivot.columns)
    ax3.set_yticklabels(pivot.index)
    ax3.set_xlabel('Latency', fontsize=10)
    ax3.set_ylabel('Budget', fontsize=10)
    ax3.set_title('Accuracy Heatmap: Budget × Latency', fontsize=11)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Accuracy', fontsize=9)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                # Use white text for darker colors (lower values in viridis)
                text_color = 'white' if val < 0.76 else 'black'
                ax3.text(j, i, f'{val:.3f}', ha='center', va='center',
                        color=text_color, fontsize=9)

    # =========================================================================
    # Plot 4: Bar chart - Budget Utilization
    # =========================================================================
    ax4 = axes[1, 0]

    grouped = df.groupby(['budget_label', 'latency_label'])['budget_utilization'].mean().unstack()
    grouped = grouped.reindex(index=[b for b in budget_order if b in grouped.index],
                              columns=[l for l in latency_order if l in grouped.columns])

    x = np.arange(len(grouped.index))
    width = 0.25
    colors_lat = ['#2ecc71', '#f39c12', '#e74c3c']

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

    # =========================================================================
    # Plot 5: Bar chart - Errors during first drift window (by drift type)
    # =========================================================================
    ax5 = axes[1, 1]

    # Group by drift type and latency
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

    # =========================================================================
    # Plot 6: Bar chart - Error Count in Drift Window
    # =========================================================================
    ax6 = axes[1, 2]

    # Group by drift type and budget
    grouped = df.groupby(['drift_type', 'budget_label'])['retrains_after_drift'].mean().unstack()
    grouped = grouped.reindex(columns=[b for b in budget_order if b in grouped.columns])

    x = np.arange(len(grouped.index))
    width = 0.25
    colors_bud = ['#2ecc71', '#f39c12', '#e74c3c']

    for i, (budget, color) in enumerate(zip(budget_order, colors_bud)):
        if budget in grouped.columns:
            vals = grouped[budget].values
            bars = ax6.bar(x + i*width, vals, width, label=budget, color=color, alpha=0.8)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax6.set_xlabel('Drift Type', fontsize=10)
    ax6.set_ylabel('Avg Errors in Drift Window', fontsize=10)
    ax6.set_title('Error Count in Drift Window by Drift Type & Budget', fontsize=11)
    ax6.set_xticks(x + width)
    ax6.set_xticklabels([dt.capitalize() for dt in grouped.index])
    ax6.legend(title='Budget', fontsize=8)
    ax6.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/summary_results_plot_periodic_retrain.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nGraph saved: results/summary_results_plot_periodic_retrain.png")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for drift_type in ['abrupt', 'gradual']:
        print(f"\n{drift_type.upper()} DRIFT:")
        df_drift = df[df['drift_type'] == drift_type]

        print(f"  Overall Accuracy: {df_drift['overall_accuracy'].mean():.4f} ± {df_drift['overall_accuracy'].std():.4f}")
        print(f"  Pre-Drift Accuracy: {df_drift['pre_drift_accuracy'].mean():.4f}")
        print(f"  Post-Drift Accuracy: {df_drift['post_drift_accuracy'].mean():.4f}")
        print(f"  Accuracy Drop: {df_drift['accuracy_drop'].mean():.4f}")
        print(f"  Avg Budget Utilization: {df_drift['budget_utilization'].mean():.1%}")


if __name__ == "__main__":
    plot_summary()

