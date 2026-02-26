"""
Visualization module for plotting experiment results.

Generates summary plots showing drift timing, retrain events, and model accuracy.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json


def plot_results(seeds, policy, drift_point, drift_type, recurrence_period=1000, total_samples=10000, policy_type="periodic"):
    """
    Generate summary plots after all seeds have been run.

    Creates a 2-panel figure:
    1. Top: Drift & Retrain Timeline showing when drift occurs and retraining events
    2. Bottom: Rolling accuracy over time for all seeds

    Args:
        seeds: List of random seeds used
        policy: The policy object (for latency info)
        drift_point: Timestep where drift occurs
        drift_type: Type of drift (abrupt/gradual/recurring)
        recurrence_period: Period for recurring drift (timesteps between concept switches)
        total_samples: Total number of samples in experiment
        policy_type: Type of policy ("periodic" or "error_threshold")
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    total_latency = policy.retrain_latency + policy.deploy_latency

    # Calculate retrain times based on policy type
    if policy_type == "error_threshold":
        # For error threshold policy, read retrain times from the first seed's JSON results
        retrain_times = []
        try:
            with open(f'results/run_seed_{seeds[0]}.json', 'r') as f:
                run_data = json.load(f)
                retrain_times = run_data.get('retraining_metrics', {}).get('retrain_timestamps', [])
        except (FileNotFoundError, json.JSONDecodeError):
            retrain_times = []
    else:
        # Periodic policy: calculate retrain times from interval
        retrain_times = []
        t = 0
        while t < total_samples and len(retrain_times) < policy.budget:
            if t % policy.interval == 0:
                if not retrain_times or t >= retrain_times[-1] + total_latency:
                    retrain_times.append(t)
            t += 1

    # Plot 1: Drift and Retrain Timeline
    ax1 = axes[0]

    # Draw timeline base
    ax1.axhline(y=0.5, color='gray', linewidth=2, alpha=0.3)

    # Mark drift point/region based on drift type
    if drift_type == "abrupt":
        ax1.axvline(x=drift_point, color='red', linewidth=3, label='Abrupt Drift')
        ax1.fill_betweenx([0, 1], drift_point-50, drift_point+50, color='red', alpha=0.3)
    elif drift_type == "gradual":
        ax1.axvspan(drift_point, drift_point + 1000, color='red', alpha=0.2, label='Gradual Drift Region')
        ax1.axvline(x=drift_point, color='red', linewidth=2, linestyle='--')
        ax1.axvline(x=drift_point + 1000, color='red', linewidth=2, linestyle='--')
    elif drift_type == "recurring":
        # Mark recurring drift regions - concept alternates every recurrence_period
        ax1.axvline(x=drift_point, color='red', linewidth=2, linestyle='--', label='Drift Start')
        period_num = 0
        t = drift_point
        while t < total_samples:
            period_end = min(t + recurrence_period, total_samples)
            if period_num % 2 == 0:
                # Drifted state (new distribution)
                ax1.axvspan(t, period_end, color='red', alpha=0.2,
                           label='Drifted Distribution' if period_num == 0 else '')
            else:
                # Original state (return to initial distribution)
                ax1.axvspan(t, period_end, color='blue', alpha=0.2,
                           label='Original Distribution' if period_num == 1 else '')
            t += recurrence_period
            period_num += 1

    # Mark retrain events and latency windows
    for i, rt in enumerate(retrain_times):
        # Retrain trigger point
        ax1.scatter(rt, 0.5, color='green', s=100, zorder=5,
                   label='Retrain Trigger' if i == 0 else '')

        # Latency window (retrain + deploy)
        ax1.fill_betweenx([0.3, 0.7], rt, rt + policy.retrain_latency,
                         color='orange', alpha=0.5, label='Retrain Period' if i == 0 else '')
        ax1.fill_betweenx([0.3, 0.7], rt + policy.retrain_latency, rt + total_latency,
                         color='blue', alpha=0.3, label='Deploy Period' if i == 0 else '')

    ax1.set_xlim(0, total_samples)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Timestep', fontsize=11)
    ax1.set_title(f'Drift & Retrain Timeline ({drift_type.capitalize()} Drift)', fontsize=12)
    ax1.set_yticks([])
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='x', alpha=0.3)

    # Add annotations
    if drift_type == "recurring":
        ax1.text(drift_point, 0.85, f'Drift Start\n(t={drift_point})', ha='center', fontsize=9, color='red')
        ax1.text(5000, 0.1, f'Retrains: {len(retrain_times)} | Latency: {total_latency} | Period: {recurrence_period}',
                 ha='center', fontsize=9, style='italic')
    else:
        ax1.text(drift_point, 0.85, f'Drift\n(t={drift_point})', ha='center', fontsize=9, color='red')
        ax1.text(5000, 0.1, f'Retrains: {len(retrain_times)} | Latency: {total_latency} timesteps/retrain',
                 ha='center', fontsize=9, style='italic')

    # Plot 2: Rolling accuracy over time
    ax2 = axes[1]
    for seed, color in zip(seeds, colors):
        df = pd.read_csv(f'results/per_sample_metrics_seed_{seed}.csv')
        df['accuracy'] = pd.to_numeric(df['accuracy'])
        df['rolling_accuracy'] = df['accuracy'].rolling(window=200, min_periods=1).mean()
        ax2.plot(df['timestamp'], df['rolling_accuracy'], color=color,
                linewidth=1.2, label=f'Seed {seed}', alpha=0.8)

    # Mark drift regions based on drift type
    if drift_type == "recurring":
        ax2.axvline(x=drift_point, color='red', linestyle='--', linewidth=2, label='Drift Start')
        # Show recurring concept switches
        period_num = 0
        t = drift_point
        while t < total_samples:
            period_end = min(t + recurrence_period, total_samples)
            if period_num % 2 == 0:
                ax2.axvspan(t, period_end, color='red', alpha=0.1)
            else:
                ax2.axvspan(t, period_end, color='blue', alpha=0.1)
            t += recurrence_period
            period_num += 1
    else:
        ax2.axvline(x=drift_point, color='red', linestyle='--', linewidth=2, label='Drift Point')
        if drift_type == "gradual":
            ax2.axvspan(drift_point, drift_point + 1000, color='red', alpha=0.1)

    # Mark retrain events
    for i, rt in enumerate(retrain_times):
        ax2.axvline(x=rt, color='green', linestyle=':', alpha=0.5,
                   label='Retrain' if i == 0 else '')
        ax2.axvspan(rt, rt + total_latency, alpha=0.1, color='orange')

    ax2.set_xlabel('Timestep', fontsize=11)
    ax2.set_ylabel('Rolling Accuracy (window=200)', fontsize=11)
    ax2.set_title('Model Accuracy Over Time', fontsize=12)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.set_xlim(0, total_samples)
    ax2.set_ylim(0.5, 1.0)
    ax2.grid(alpha=0.3)

    # Add config info
    if policy_type == "error_threshold":
        config_text = f'Policy: Error Threshold | Budget={policy.budget} | Threshold={policy.error_threshold} | Window={policy.window_size} | Latency={total_latency}'
    else:
        config_text = f'Policy: Periodic | Budget={policy.budget} | Interval={policy.interval} | Latency={total_latency}'
    fig.text(0.5, 0.02, config_text, ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('results/experiment_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nGraph saved: results/experiment_results.png")

