"""
Main entry point for the Drift-Triggered Retraining Policy experiment.

Runs full-factorial sweeps for one or all retraining policies using
two **extreme** latency levels (Near-Zero=3, Extreme-High=2050):
  Periodic / Error-Threshold / Drift-Triggered:
      3 drift types × 3 budgets × 2 latency levels × N seeds
  No-Retrain (baseline):
      3 drift types × N seeds  (30 runs @10 seeds) — no budget/latency grid

Usage:
    python main.py                            # all 4 policies (840 runs with 10 seeds)
    python main.py --policy periodic          # periodic only          (270 runs)
    python main.py --policy error_threshold   # error-threshold only   (270 runs)
    python main.py --policy drift_triggered   # drift-triggered only   (270 runs)
    python main.py --policy no_retrain        # baseline only          ( 30 runs)
    python main.py --seeds 3                  # use 3-seed set instead of 10
"""

import argparse
from pathlib import Path
import time

from src.data.drift_generator import DriftGenerator
from src.models.base_model import StreamingModel
from src.policies.periodic import PeriodicPolicy
from src.policies.error_threshold_policy import ErrorThresholdPolicy
from src.policies.drift_triggered_policy import DriftTriggeredPolicy
from src.policies.never_retrain_policy import NeverRetrainPolicy
from src.evaluation.metrics import MetricsTracker
from src.runner.experiment_runner import ExperimentRunner
from src.evaluation.results_export import export_to_json, export_to_csv, export_summary_to_csv

# ── Seed sets ───────────────────────────────────────────────────────────
SEEDS_3 = [42, 123, 456]
SEEDS_10 = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# ── Shared experiment grid ──────────────────────────────────────────────
DRIFT_TYPES = ["abrupt", "gradual", "recurring"]
BUDGETS = [5, 10, 20]
LATENCY_CONFIGS = [
    (2, 1),       # Near-Zero latency    (total = 3)    — isolates pure policy behavior
    (2000, 50),   # Extreme-High latency (total = 2050) — forces minimal post-drift retrains
]
DRIFT_POINT = 5000
N_SAMPLES = 10000
RECURRENCE_PERIOD = 1000

# ── Per-policy fixed parameters ─────────────────────────────────────────
POLICY_PARAMS = {
    "periodic": {},                                          # interval derived from budget
    "error_threshold": {"error_threshold": 0.27, "window_size": 200},
    "drift_triggered": {"delta": 0.002, "window_size": 500, "min_samples": 100},
    "no_retrain": {},                                        # baseline — no retraining
}

POLICY_DISPLAY = {
    "periodic":         "Periodic",
    "error_threshold":  "Error-Threshold",
    "drift_triggered":  "Drift-Triggered (ADWIN)",
    "no_retrain":       "No-Retrain (Baseline)",
}


def _build_policy(policy_type, budget, retrain_latency, deploy_latency):
    """Instantiate the requested policy with the correct parameters."""
    if policy_type == "periodic":
        interval = N_SAMPLES // budget          # 5→2000, 10→1000, 20→500
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


def _build_config(policy_type, drift_type, budget, seed):
    """Build the config dict passed to export helpers."""
    config = {
        "drift_type": drift_type,
        "drift_point": DRIFT_POINT,
        "recurrence_period": RECURRENCE_PERIOD,
        "policy_type": policy_type,
        "budget": budget,
        "random_seed": seed,
    }
    if policy_type == "periodic":
        config["policy_interval"] = N_SAMPLES // budget
    elif policy_type == "error_threshold":
        config.update(POLICY_PARAMS["error_threshold"])
    elif policy_type == "drift_triggered":
        config.update(POLICY_PARAMS["drift_triggered"])
    return config


def run_policy_sweep(policy_type, seeds):
    """Execute the full-factorial sweep for a single policy."""
    # Delegate to the specialized no-retrain sweep when appropriate
    if policy_type == "no_retrain":
        return _run_no_retrain_sweep(seeds)

    seed_label = f"{len(seeds)}seed"
    n_drifts = len(DRIFT_TYPES)
    n_budgets = len(BUDGETS)
    n_latencies = len(LATENCY_CONFIGS)
    total_runs = n_drifts * n_budgets * n_latencies * len(seeds)

    # Output paths
    run_label = f"ExtremeLatency_{seed_label}"
    results_dir = Path(f"results/{policy_type}_{run_label}")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = f"results/summary_results_{policy_type}_retrain_ExtremeLatency_{seed_label}.csv"
    Path(summary_csv).unlink(missing_ok=True)

    display = POLICY_DISPLAY[policy_type]
    run_count = 0
    start_time = time.time()

    print(f"\n{'=' * 70}")
    print(f"{display} POLICY – Full Experiment Sweep ({total_runs} runs)")
    print(f"{'=' * 70}")
    print(f"  Drift types      : {DRIFT_TYPES}")
    print(f"  Budgets          : {BUDGETS}")
    print(f"  Latency configs  : {LATENCY_CONFIGS}")
    print(f"  Seeds            : {len(seeds)} seeds")
    if policy_type == "periodic":
        print(f"  Intervals        : {[N_SAMPLES // b for b in BUDGETS]}")
    else:
        for k, v in POLICY_PARAMS[policy_type].items():
            print(f"  {k:<18}: {v}")
    print(f"{'=' * 70}\n")

    for drift_type in DRIFT_TYPES:
        for budget in BUDGETS:
            for retrain_latency, deploy_latency in LATENCY_CONFIGS:
                for seed in seeds:
                    run_count += 1

                    # 1. Generate data
                    generator = DriftGenerator(
                        drift_type=drift_type,
                        drift_point=DRIFT_POINT,
                        recurrence_period=RECURRENCE_PERIOD,
                        seed=seed,
                    )
                    X, y = generator.generate(N_SAMPLES)

                    # 2. Build components
                    model = StreamingModel()
                    policy = _build_policy(policy_type, budget, retrain_latency, deploy_latency)
                    metrics = MetricsTracker()
                    metrics.set_drift_point(DRIFT_POINT)
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
                        f"latency=({retrain_latency}+{deploy_latency})  seed={seed:<5} | "
                        f"acc={summary['overall_accuracy']:.4f}  "
                        f"retrains={summary['total_retrains']:>2} | "
                        f"ETA {eta / 60:.1f} min"
                    )

                    # 5. Export per-run results
                    run_tag = (
                        f"{drift_type}_b{budget}"
                        f"_l{retrain_latency}+{deploy_latency}"
                        f"_s{seed}"
                    )
                    config = _build_config(policy_type, drift_type, budget, seed)

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
        output_path=f"results/summary_results_plot_{policy_type}_retrain_ExtremeLatency_{seed_label}.png",
        policy_name=display,
    )

    return summary_csv


def _run_no_retrain_sweep(seeds):
    """Execute the no-retrain baseline sweep: 3 drift types × N seeds.

    Budget and latency are always 0 — there is no budget/latency grid.
    The model is only updated via partial_fit (incremental learning).
    """
    policy_type = "no_retrain"
    seed_label = f"{len(seeds)}seed"
    total_runs = len(DRIFT_TYPES) * len(seeds)

    # Output paths
    run_label = f"ExtremeLatency_{seed_label}"
    results_dir = Path(f"results/{policy_type}_{run_label}")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = f"results/summary_results_{policy_type}_ExtremeLatency_{seed_label}.csv"
    Path(summary_csv).unlink(missing_ok=True)

    display = POLICY_DISPLAY[policy_type]
    run_count = 0
    start_time = time.time()

    print(f"\n{'=' * 70}")
    print(f"{display} POLICY – Baseline Sweep ({total_runs} runs)")
    print(f"{'=' * 70}")
    print(f"  Drift types      : {DRIFT_TYPES}")
    print(f"  Budget           : N/A (always 0)")
    print(f"  Latency          : N/A (always 0)")
    print(f"  Seeds            : {len(seeds)} seeds")
    print(f"  Stream length    : {N_SAMPLES}")
    print(f"{'=' * 70}\n")

    for drift_type in DRIFT_TYPES:
        for seed in seeds:
            run_count += 1

            # 1. Generate data
            generator = DriftGenerator(
                drift_type=drift_type,
                drift_point=DRIFT_POINT,
                recurrence_period=RECURRENCE_PERIOD,
                seed=seed,
            )
            X, y = generator.generate(N_SAMPLES)

            # 2. Build components (budget=0, latency=0)
            model = StreamingModel()
            policy = _build_policy(policy_type, budget=0, retrain_latency=0, deploy_latency=0)
            metrics = MetricsTracker()
            metrics.set_drift_point(DRIFT_POINT)
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
                f"drift={drift_type:<10} seed={seed:<5} | "
                f"acc={summary['overall_accuracy']:.4f}  "
                f"retrains={summary['total_retrains']:>2} | "
                f"ETA {eta / 60:.1f} min"
            )

            # 5. Export per-run results
            run_tag = f"{drift_type}_s{seed}"
            config = _build_config(policy_type, drift_type, budget=0, seed=seed)

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
        output_path=f"results/summary_results_plot_{policy_type}_ExtremeLatency_{seed_label}.png",
        policy_name=display,
    )

    return summary_csv


def main():
    parser = argparse.ArgumentParser(
        description="Run retraining-policy experiments (full-factorial sweep)."
    )
    parser.add_argument(
        "--policy",
        choices=["periodic", "error_threshold", "drift_triggered", "no_retrain", "all"],
        default="all",
        help="Which policy to run (default: all).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        choices=[3, 10],
        default=10,
        help="Number of seeds: 3 (Phase 1) or 10 (Phase 2, default).",
    )
    args = parser.parse_args()

    seeds = SEEDS_3 if args.seeds == 3 else SEEDS_10
    policies = (
        list(POLICY_PARAMS.keys()) if args.policy == "all"
        else [args.policy]
    )

    # Compute total runs accounting for no_retrain's simpler grid
    total_runs = 0
    for p in policies:
        if p == "no_retrain":
            total_runs += len(DRIFT_TYPES) * len(seeds)
        else:
            total_runs += len(DRIFT_TYPES) * len(BUDGETS) * len(LATENCY_CONFIGS) * len(seeds)

    total_start = time.time()

    print(f"{'#' * 70}")
    print(f"  EXPERIMENT SWEEP — {len(policies)} policy(ies), "
          f"{len(seeds)} seeds, "
          f"{total_runs} total runs")
    print(f"{'#' * 70}")

    for policy_type in policies:
        run_policy_sweep(policy_type, seeds)

    total_elapsed = time.time() - total_start
    print(f"\n{'#' * 70}")
    print(f"  ALL DONE — {len(policies)} policy(ies) completed in {total_elapsed / 60:.1f} minutes")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
