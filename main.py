"""
 Main entry point for the Periodic Retraining Policy experiment.

Runs all combinations for the Periodic retraining policy:
  3 drift types × 3 budgets × 3 latency levels × 10 seeds = 270 runs

Results are saved to:
  - results/periodic_<N>seed_<D>drift_<B>budget_<L>latency/summary_results_periodic_retrain_10seed.csv
  - results/periodic_<N>seed_<D>drift_<B>budget_<L>latency/summary_results_plot_periodic_retrain_10seed.png
"""

from pathlib import Path
import time

from src.data.drift_generator import DriftGenerator
from src.models.base_model import StreamingModel
from src.policies.periodic import PeriodicPolicy
from src.evaluation.metrics import MetricsTracker
from src.runner.experiment_runner import ExperimentRunner
from src.evaluation.results_export import export_to_json, export_to_csv, export_summary_to_csv


def main():
    seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
    drift_types = ["abrupt", "gradual", "recurring"]
    budgets = [5, 10, 20]                       # Low / Med / High budget
    latency_configs = [
        (10, 1),     # Low latency   (total = 11)
        (100, 5),    # Medium latency (total = 105)
        (500, 20),   # High latency  (total = 520)
    ]

    # Periodic-policy fixed parameters
    interval = 500
    policy_type = "periodic"

    # Data generation parameters
    drift_point = 5000
    n_samples = 10000
    recurrence_period = 1000

    #  Prepare output
    run_label = f"{len(seeds)}seed_{len(drift_types)}drift_{len(budgets)}budget_{len(latency_configs)}latency"
    results_dir = Path(f"results/periodic_{run_label}")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = str(results_dir / "summary_results_periodic_retrain_10seed.csv")
    Path(summary_csv).unlink(missing_ok=True)          # fresh file

    total_runs = len(drift_types) * len(budgets) * len(latency_configs) * len(seeds)
    run_count = 0
    start_time = time.time()

    print(f"{'=' * 70}")
    print(f"PERIODIC RETRAIN POLICY – Full Experiment Sweep ({total_runs} runs)")
    print(f"{'=' * 70}")
    print(f"  Drift types      : {drift_types}")
    print(f"  Budgets           : {budgets}")
    print(f"  Latency configs   : {latency_configs}")
    print(f"  Seeds             : {len(seeds)} seeds")
    print(f"  Interval          : {interval}")
    print(f"{'=' * 70}\n")

    for drift_type in drift_types:
        for budget in budgets:
            for retrain_latency, deploy_latency in latency_configs:
                for seed in seeds:
                    run_count += 1

                    # 1. Generate data
                    generator = DriftGenerator(
                        drift_type=drift_type,
                        drift_point=drift_point,
                        recurrence_period=recurrence_period,
                        seed=seed,
                    )
                    X, y = generator.generate(n_samples)

                    # 2. Build components
                    model = StreamingModel()
                    policy = PeriodicPolicy(
                        interval=interval,
                        budget=budget,
                        retrain_latency=retrain_latency,
                        deploy_latency=deploy_latency,
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
                        f"latency=({retrain_latency}+{deploy_latency})  seed={seed:<5} | "
                        f"acc={summary['overall_accuracy']:.4f}  "
                        f"retrains={summary['total_retrains']:>2} | "
                        f"ETA {eta / 60:.1f} min"
                    )

                    # 5. Export per-run results (disambiguated filenames)
                    run_tag = (
                        f"{drift_type}_b{budget}"
                        f"_l{retrain_latency}+{deploy_latency}"
                        f"_s{seed}"
                    )
                    config = {
                        "drift_type": drift_type,
                        "drift_point": drift_point,
                        "recurrence_period": recurrence_period,
                        "policy_type": policy_type,
                        "interval": interval,
                        "budget": budget,
                        "random_seed": seed,
                    }

                    export_to_json(
                        metrics, policy, config,
                        str(results_dir / f"run_{run_tag}.json"),
                    )
                    export_to_csv(
                        metrics, policy, config,
                        str(results_dir / f"per_sample_{run_tag}.csv"),
                    )
                    export_summary_to_csv(
                        metrics, policy, config,
                        summary_csv,
                    )

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"All {total_runs} runs completed in {elapsed / 60:.1f} minutes")
    print(f"Summary CSV → {summary_csv}")

    # Generate summary plot
    print("\nGenerating summary plot...")
    from plot_summary import plot_summary_for_policy

    plot_summary_for_policy(
        csv_path=summary_csv,
        output_path=str(results_dir / "summary_results_plot_periodic_retrain_10seed.png"),
        policy_name="Periodic",
    )
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
