"""
Main entry point for the Real-World Fraud Detection retraining-policy experiment.

Runs the same full-factorial sweep as main.py but on real CIS Fraud Detection
data instead of synthetic data.  Drift is constructed from temporal segments:

    abrupt    – stream 50k rows as-is (purely organic)
    gradual   – blend two consecutive 50k time pools across a transition window
    recurring – alternate between two 50k time pools after drift point

Seeds are window offsets (0, 10000, 20000) — each slides the data window to
a different temporal region, providing variance across runs.

Usage:
    python main_fraud_detection.py                            # all 4 policies (252 runs)
    python main_fraud_detection.py --policy periodic          # periodic only
    python main_fraud_detection.py --policy drift_triggered   # drift-triggered only
    python main_fraud_detection.py --policy error_threshold   # error-threshold only
    python main_fraud_detection.py --policy no_retrain        # baseline only
    python main_fraud_detection.py --n_samples 100000         # larger stream
"""

import argparse
from pathlib import Path
import time

from src.data.fraud_data_loader import FraudDataLoader
from src.data.real_drift_generator import build_drift_stream
from src.models.base_model import StreamingModel
from src.policies.periodic import PeriodicPolicy
from src.policies.error_threshold_policy import ErrorThresholdPolicy
from src.policies.drift_triggered_policy import DriftTriggeredPolicy
from src.policies.never_retrain_policy import NeverRetrainPolicy
from src.evaluation.metrics import MetricsTracker
from src.runner.experiment_runner import ExperimentRunner
from src.evaluation.results_export import export_to_json, export_to_csv, export_summary_to_csv

# Seed sets (window offsets into the time-sorted data)
WINDOW_OFFSETS = [0, 10_000, 20_000]

# Experiment grid
DRIFT_TYPES = ["abrupt", "gradual", "recurring"]
BUDGETS = [5, 10, 20]
LATENCY_CONFIGS = [
    (10, 1),      # Low latency   (total = 11)
    (100, 5),     # Medium latency (total = 105)
    (500, 20),    # High latency  (total = 520)
]

N_SAMPLES_DEFAULT = 50_000

# Per-policy fixed parameters
POLICY_PARAMS = {
    "periodic": {},
    "error_threshold": {"error_threshold": 0.27, "window_size": 200},
    "drift_triggered": {"delta": 0.002, "window_size": 500, "min_samples": 100},
    "no_retrain": {},
}

POLICY_DISPLAY = {
    "periodic":         "Periodic",
    "error_threshold":  "Error-Threshold",
    "drift_triggered":  "Drift-Triggered (ADWIN)",
    "no_retrain":       "No-Retrain (Baseline)",
}

# ── Data directory ──────────────────────────────────────────────────────
DATA_DIR = Path("src/data/CIS Fraud Detection")


def _build_policy(policy_type, budget, retrain_latency, deploy_latency, n_samples):
    """Instantiate the requested policy with the correct parameters."""
    if policy_type == "periodic":
        interval = n_samples // budget
        return PeriodicPolicy(
            interval=interval,
            budget=budget,
            retrain_latency=retrain_latency,
            deploy_latency=deploy_latency,
        )
    elif policy_type == "error_threshold":
        p = POLICY_PARAMS["error_threshold"]
        return ErrorThresholdPolicy(
            error_threshold=p["error_threshold"],
            window_size=p["window_size"],
            budget=budget,
            retrain_latency=retrain_latency,
            deploy_latency=deploy_latency,
        )
    elif policy_type == "drift_triggered":
        p = POLICY_PARAMS["drift_triggered"]
        return DriftTriggeredPolicy(
            delta=p["delta"],
            window_size=p["window_size"],
            min_samples=p["min_samples"],
            budget=budget,
            retrain_latency=retrain_latency,
            deploy_latency=deploy_latency,
        )
    elif policy_type == "no_retrain":
        return NeverRetrainPolicy()
    else:
        raise ValueError(f"Unknown policy_type: {policy_type!r}")


def _build_config(policy_type, drift_type, budget, seed, n_samples, drift_point):
    """Build the config dict passed to export helpers."""
    config = {
        "drift_type": drift_type,
        "drift_point": drift_point,
        "recurrence_period": n_samples // 10,
        "policy_type": policy_type,
        "budget": budget,
        "random_seed": seed,
        "n_samples": n_samples,
        "dataset": "CIS_Fraud_Detection",
    }
    if policy_type == "periodic":
        config["policy_interval"] = n_samples // budget
    elif policy_type == "error_threshold":
        config.update(POLICY_PARAMS["error_threshold"])
    elif policy_type == "drift_triggered":
        config.update(POLICY_PARAMS["drift_triggered"])
    return config


def _get_stream(loader, drift_type, n_samples, offset, drift_point,
                recurrence_period, seed):
    """
    Build the (X, y) stream for one run.

    For abrupt:    single pool of n_samples rows (organic drift).
    For gradual/recurring: two consecutive pools of n_samples rows each,
                           combined via build_drift_stream().
    """
    X_pre, y_pre = loader.get_pool(start_offset=offset, n_samples=n_samples)

    if drift_type == "abrupt":
        return build_drift_stream(
            X_pre, y_pre, X_pre, y_pre,  # post-pool ignored for abrupt
            drift_type=drift_type,
            drift_point=drift_point,
            recurrence_period=recurrence_period,
            seed=seed,
        )

    # Gradual / Recurring: need a second pool from the next time window
    X_post, y_post = loader.get_pool(
        start_offset=offset + n_samples,
        n_samples=n_samples,
    )
    return build_drift_stream(
        X_pre, y_pre, X_post, y_post,
        drift_type=drift_type,
        drift_point=drift_point,
        recurrence_period=recurrence_period,
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────
# Sweep runners
# ─────────────────────────────────────────────────────────────────────────

def run_policy_sweep(policy_type, loader, n_samples):
    """Execute the full-factorial sweep for a single policy."""
    if policy_type == "no_retrain":
        return _run_no_retrain_sweep(loader, n_samples)

    drift_point = n_samples // 2
    recurrence_period = n_samples // 10
    seed_label = f"{len(WINDOW_OFFSETS)}seed"

    n_drifts = len(DRIFT_TYPES)
    n_budgets = len(BUDGETS)
    n_latencies = len(LATENCY_CONFIGS)
    total_runs = n_drifts * n_budgets * n_latencies * len(WINDOW_OFFSETS)

    # Output paths
    results_dir = Path(
        f"results/fraud_detection/{policy_type}_{seed_label}"
        f"_{n_drifts}drift_{n_budgets}budget_{n_latencies}latency"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = (
        f"results/fraud_detection/"
        f"summary_results_{policy_type}_retrain_{seed_label}.csv"
    )
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_csv).unlink(missing_ok=True)

    display = POLICY_DISPLAY[policy_type]
    run_count = 0
    start_time = time.time()

    print(f"\n{'=' * 70}")
    print(f"{display} POLICY – Real-Data Sweep ({total_runs} runs)")
    print(f"{'=' * 70}")
    print(f"  Dataset          : CIS Fraud Detection")
    print(f"  Stream length    : {n_samples:,}")
    print(f"  Drift point      : {drift_point:,} (midpoint)")
    print(f"  Recurrence period: {recurrence_period:,}")
    print(f"  Drift types      : {DRIFT_TYPES}")
    print(f"  Budgets          : {BUDGETS}")
    print(f"  Latency configs  : {LATENCY_CONFIGS}")
    print(f"  Window offsets   : {WINDOW_OFFSETS}")
    if policy_type == "periodic":
        print(f"  Intervals        : {[n_samples // b for b in BUDGETS]}")
    else:
        for k, v in POLICY_PARAMS[policy_type].items():
            print(f"  {k:<18}: {v}")
    print(f"{'=' * 70}\n")

    for drift_type in DRIFT_TYPES:
        for budget in BUDGETS:
            for retrain_latency, deploy_latency in LATENCY_CONFIGS:
                for offset in WINDOW_OFFSETS:
                    run_count += 1

                    # 1. Build stream
                    X, y = _get_stream(
                        loader, drift_type, n_samples, offset,
                        drift_point, recurrence_period, seed=offset,
                    )

                    # 2. Build components
                    model = StreamingModel()
                    policy = _build_policy(
                        policy_type, budget,
                        retrain_latency, deploy_latency, n_samples,
                    )
                    metrics = MetricsTracker()
                    metrics.set_drift_point(drift_point)
                    metrics.set_budget(budget)

                    # 3. Run experiment
                    runner = ExperimentRunner(model, policy, metrics)
                    runner.run(X, y)

                    # 4. Progress
                    summary = metrics.get_summary()
                    elapsed = time.time() - start_time
                    eta = (elapsed / run_count) * (total_runs - run_count)

                    print(
                        f"[{run_count:>3}/{total_runs}] "
                        f"drift={drift_type:<10} budget={budget:<3} "
                        f"latency=({retrain_latency}+{deploy_latency})  "
                        f"offset={offset:<6} | "
                        f"acc={summary['overall_accuracy']:.4f}  "
                        f"retrains={summary['total_retrains']:>2} | "
                        f"ETA {eta / 60:.1f} min"
                    )

                    # 5. Export per-run results
                    run_tag = (
                        f"{drift_type}_b{budget}"
                        f"_l{retrain_latency}+{deploy_latency}"
                        f"_off{offset}"
                    )
                    config = _build_config(
                        policy_type, drift_type, budget,
                        seed=offset, n_samples=n_samples,
                        drift_point=drift_point,
                    )

                    export_to_json(
                        metrics, policy, config,
                        str(results_dir / f"run_{run_tag}.json"),
                    )
                    export_to_csv(
                        metrics, policy, config,
                        str(results_dir / f"per_sample_{run_tag}.csv"),
                    )
                    export_summary_to_csv(metrics, policy, config, summary_csv)

    elapsed = time.time() - start_time
    print(f"\n{display}: {total_runs} runs completed in {elapsed / 60:.1f} minutes")
    print(f"Summary CSV → {summary_csv}")

    # Generate summary dashboard
    print(f"Generating {display} summary plot...")
    from plot_summary import plot_summary_for_policy
    plot_summary_for_policy(
        csv_path=summary_csv,
        output_path=(
            f"results/fraud_detection/"
            f"summary_results_plot_{policy_type}_retrain_{seed_label}.png"
        ),
        policy_name=f"{display} (Fraud Detection)",
    )

    return summary_csv


def _run_no_retrain_sweep(loader, n_samples):
    """Execute the no-retrain baseline: 3 drift types × 3 window offsets."""
    policy_type = "no_retrain"
    drift_point = n_samples // 2
    recurrence_period = n_samples // 10
    seed_label = f"{len(WINDOW_OFFSETS)}seed"
    total_runs = len(DRIFT_TYPES) * len(WINDOW_OFFSETS)

    # Output paths
    results_dir = Path(
        f"results/fraud_detection/{policy_type}_{seed_label}_{len(DRIFT_TYPES)}drift"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = (
        f"results/fraud_detection/"
        f"summary_results_{policy_type}_{seed_label}.csv"
    )
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_csv).unlink(missing_ok=True)

    display = POLICY_DISPLAY[policy_type]
    run_count = 0
    start_time = time.time()

    print(f"\n{'=' * 70}")
    print(f"{display} POLICY – Real-Data Baseline Sweep ({total_runs} runs)")
    print(f"{'=' * 70}")
    print(f"  Dataset          : CIS Fraud Detection")
    print(f"  Stream length    : {n_samples:,}")
    print(f"  Drift types      : {DRIFT_TYPES}")
    print(f"  Budget           : N/A (always 0)")
    print(f"  Latency          : N/A (always 0)")
    print(f"  Window offsets   : {WINDOW_OFFSETS}")
    print(f"{'=' * 70}\n")

    for drift_type in DRIFT_TYPES:
        for offset in WINDOW_OFFSETS:
            run_count += 1

            # 1. Build stream
            X, y = _get_stream(
                loader, drift_type, n_samples, offset,
                drift_point, recurrence_period, seed=offset,
            )

            # 2. Build components
            model = StreamingModel()
            policy = _build_policy(
                policy_type, budget=0,
                retrain_latency=0, deploy_latency=0,
                n_samples=n_samples,
            )
            metrics = MetricsTracker()
            metrics.set_drift_point(drift_point)
            metrics.set_budget(0)

            # 3. Run experiment
            runner = ExperimentRunner(model, policy, metrics)
            runner.run(X, y)

            # 4. Progress
            summary = metrics.get_summary()
            elapsed = time.time() - start_time
            eta = (elapsed / run_count) * (total_runs - run_count)

            print(
                f"[{run_count:>3}/{total_runs}] "
                f"drift={drift_type:<10} offset={offset:<6} | "
                f"acc={summary['overall_accuracy']:.4f}  "
                f"retrains={summary['total_retrains']:>2} | "
                f"ETA {eta / 60:.1f} min"
            )

            # 5. Export per-run results
            run_tag = f"{drift_type}_off{offset}"
            config = _build_config(
                policy_type, drift_type, budget=0,
                seed=offset, n_samples=n_samples,
                drift_point=drift_point,
            )

            export_to_json(
                metrics, policy, config,
                str(results_dir / f"run_{run_tag}.json"),
            )
            export_to_csv(
                metrics, policy, config,
                str(results_dir / f"per_sample_{run_tag}.csv"),
            )
            export_summary_to_csv(metrics, policy, config, summary_csv)

    elapsed = time.time() - start_time
    print(f"\n{display}: {total_runs} runs completed in {elapsed / 60:.1f} minutes")
    print(f"Summary CSV → {summary_csv}")

    # Generate baseline summary plot
    print(f"Generating {display} summary plot...")
    from plot_summary import plot_summary_for_no_retrain
    plot_summary_for_no_retrain(
        csv_path=summary_csv,
        output_path=(
            f"results/fraud_detection/"
            f"summary_results_plot_{policy_type}_{seed_label}.png"
        ),
        policy_name=f"{display} (Fraud Detection)",
    )

    return summary_csv


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run retraining-policy experiments on CIS Fraud Detection data."
    )
    parser.add_argument(
        "--policy",
        choices=["periodic", "error_threshold", "drift_triggered",
                 "no_retrain", "all"],
        default="all",
        help="Which policy to run (default: all).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=N_SAMPLES_DEFAULT,
        help=(
            f"Number of rows per stream (default: {N_SAMPLES_DEFAULT:,}). "
            "Gradual/recurring use an additional pool of the same size."
        ),
    )
    args = parser.parse_args()

    n_samples = args.n_samples

    policies = (
        list(POLICY_PARAMS.keys()) if args.policy == "all"
        else [args.policy]
    )

    # Pre-load the dataset once (cached in the loader)
    loader = FraudDataLoader(DATA_DIR)

    # Validate that the dataset is large enough for the largest window
    max_offset = max(WINDOW_OFFSETS)
    # Gradual/recurring need 2 pools, so worst case = offset + 2 * n_samples
    required_rows = max_offset + 2 * n_samples
    if loader.total_rows < required_rows:
        print(
            f"WARNING: Dataset has {loader.total_rows:,} rows but the "
            f"experiment requires up to {required_rows:,} rows "
            f"(offset={max_offset} + 2×{n_samples:,}). "
            f"Reduce --n_samples or remove larger window offsets."
        )
        return

    # Compute total runs
    total_runs = 0
    for p in policies:
        if p == "no_retrain":
            total_runs += len(DRIFT_TYPES) * len(WINDOW_OFFSETS)
        else:
            total_runs += (
                len(DRIFT_TYPES) * len(BUDGETS)
                * len(LATENCY_CONFIGS) * len(WINDOW_OFFSETS)
            )

    total_start = time.time()

    print(f"{'#' * 70}")
    print(f"  FRAUD DETECTION EXPERIMENT — {len(policies)} policy(ies), "
          f"{len(WINDOW_OFFSETS)} window offsets, "
          f"{total_runs} total runs")
    print(f"  Stream length: {n_samples:,} rows")
    print(f"{'#' * 70}")

    for policy_type in policies:
        run_policy_sweep(policy_type, loader, n_samples)

    total_elapsed = time.time() - total_start
    print(f"\n{'#' * 70}")
    print(f"  ALL DONE — {len(policies)} policy(ies) completed "
          f"in {total_elapsed / 60:.1f} minutes")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()

