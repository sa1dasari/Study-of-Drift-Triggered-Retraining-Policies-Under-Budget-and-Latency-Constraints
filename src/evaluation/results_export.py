"""
Results export utilities for saving experiment metrics to file formats.

Provides functions to export MetricsTracker data to CSV and JSON formats
for downstream analysis, visualization, and comparison across experiments.
"""

import json
import csv
from pathlib import Path


def export_to_json(metrics, policy, config, filepath):
    """
    Export experiment results to JSON format.

    Args:
        metrics (MetricsTracker): Metrics from the experiment
        policy (RetrainPolicy): Policy used in the experiment
        config (dict): Configuration parameters (drift_type, budget, latency, etc.)
        filepath (str): Path to save JSON file
    """
    summary = metrics.get_summary()

    export_data = {
        "configuration": {
            "drift_type": config.get("drift_type", "unknown"),
            "drift_point": metrics.drift_point,
            "policy_type": config.get("policy_type", "unknown"),
            "policy_interval": config.get("policy_interval", None),
            "budget": metrics.total_budget,
            "retrain_latency": policy.retrain_latency,
            "deploy_latency": policy.deploy_latency,
            "total_samples": metrics.sample_count,
            "random_seed": config.get("random_seed", None),
        },
        "performance_metrics": {
            "overall_accuracy": summary.get("overall_accuracy"),
            "pre_drift_accuracy": summary.get("pre_drift_accuracy"),
            "post_drift_accuracy": summary.get("post_drift_accuracy"),
            "accuracy_drop": summary.get("accuracy_drop"),
        },
        "retraining_metrics": {
            "total_retrains": summary.get("total_retrains"),
            "budget_used": summary.get("budget_used"),
            "budget_total": summary.get("budget_total"),
            "budget_utilization": summary.get("budget_utilization"),
            "retrains_before_drift": summary.get("retrains_before_drift"),
            "retrains_after_drift": summary.get("retrains_after_drift"),
            "retrain_timestamps": metrics.retrain_times,
            "retrain_latency_windows": metrics.retrain_latency_windows,
        },
        "data_metrics": {
            "predictions_made": summary.get("predictions_made"),
            "total_samples": summary.get("total_samples"),
        },
    }

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"Results exported to JSON: {filepath}")


def export_to_csv(metrics, policy, config, filepath):
    """
    Export per-sample metrics to CSV format for detailed analysis.

    Args:
        metrics (MetricsTracker): Metrics from the experiment
        policy (RetrainPolicy): Policy used in the experiment
        config (dict): Configuration parameters
        filepath (str): Path to save CSV file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "accuracy",
                "error",
                "in_latency_window",
                "retrain_active",
                "deploy_active",
            ]
        )
        writer.writeheader()

        for t, acc, err in zip(metrics.timestamps, metrics.accuracies, metrics.errors):
            # Check if timestamp falls in any latency window
            in_latency = False
            retrain_active = False
            deploy_active = False

            for start, end in metrics.retrain_latency_windows:
                if start <= t < end:
                    in_latency = True
                    # Retrain phase: [start, start + retrain_latency)
                    if t < start + policy.retrain_latency:
                        retrain_active = True
                    else:
                        deploy_active = True
                    break

            writer.writerow({
                "timestamp": t,
                "accuracy": f"{acc:.4f}",
                "error": f"{err:.4f}",
                "in_latency_window": in_latency,
                "retrain_active": retrain_active,
                "deploy_active": deploy_active,
            })

    print(f"Per-sample metrics exported to CSV: {filepath}")


def export_summary_to_csv(metrics, policy, config, filepath):
    """
    Export high-level summary statistics to CSV (one row per experiment).

    Useful for comparing results across multiple experimental runs.

    Args:
        metrics (MetricsTracker): Metrics from the experiment
        policy (RetrainPolicy): Policy used in the experiment
        config (dict): Configuration parameters
        filepath (str): Path to save CSV file
    """
    summary = metrics.get_summary()

    summary_data = {
        "drift_type": config.get("drift_type", "unknown"),
        "drift_point": metrics.drift_point,
        "policy_type": config.get("policy_type", "unknown"),
        "policy_interval": config.get("policy_interval", None),
        "budget": metrics.total_budget,
        "retrain_latency": policy.retrain_latency,
        "deploy_latency": policy.deploy_latency,
        "overall_accuracy": summary.get("overall_accuracy"),
        "pre_drift_accuracy": summary.get("pre_drift_accuracy"),
        "post_drift_accuracy": summary.get("post_drift_accuracy"),
        "accuracy_drop": summary.get("accuracy_drop"),
        "total_retrains": summary.get("total_retrains"),
        "budget_utilization": summary.get("budget_utilization"),
        "retrains_before_drift": summary.get("retrains_before_drift"),
        "retrains_after_drift": summary.get("retrains_after_drift"),
        "random_seed": config.get("random_seed", None),
    }

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists to append or create
    file_exists = Path(filepath).exists()

    with open(filepath, "a" if file_exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(summary_data)

    print(f"Summary statistics exported to CSV: {filepath}")

