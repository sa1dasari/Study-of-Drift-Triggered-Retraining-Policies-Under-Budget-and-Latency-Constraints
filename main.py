"""
Main entry point for the Drift-Triggered Retraining Policies experiment.

This script orchestrates an end-to-end machine learning pipeline that handles
concept drift in streaming data. It demonstrates how different retraining policies
can adapt a model to changing data distributions under budget and latency constraints.
"""

from src.data.drift_generator import DriftGenerator
from src.models.base_model import StreamingModel
from src.policies.error_threshold import ErrorThresholdPolicy
from src.evaluation.metrics import MetricsTracker
from src.runner.experiment_runner import ExperimentRunner
from src.evaluation.results_export import export_to_json, export_to_csv, export_summary_to_csv
from src.evaluation.plot_results import plot_results

def main():
    """
    Main execution function that sets up and runs the entire experiment pipeline.

    Flow:
    1. Generate synthetic streaming data with concept drift
    2. Create a streaming model and retraining policy
    3. Run the experiment using the ExperimentRunner
    4. Report final model accuracy
    """
    # Step 1: Generate synthetic data with abrupt drift
    # To ensure reproducibility, we run multiple seeds to see how the policy performs under different random conditions.
    seeds = [42, 123, 456]
    drift_type = "abrupt"

    for seed in seeds:
        generator = DriftGenerator(
            drift_type="abrupt",
            drift_point=5000,
            recurrence_period=1000,  # Concept switches every 1000 timesteps after drift_point
            seed=seed
        )
        X, y = generator.generate(10000)

        # Step 2: Initialize components
        # - StreamingModel: Uses SGDClassifier for online learning
        # - ErrorThresholdPolicy: Retrain when recent error rate exceeds threshold
        #   error_threshold=0.27: retrain when >27% error rate in recent window
        #   window_size=200: evaluate error rate over last 200 predictions
        #   budget=5: allows up to 5 retrains during the experiment
        # - Low latency: retrain_latency=10, deploy_latency=1
        # - MetricsTracker: Records prediction accuracy/errors over time
        model = StreamingModel()
        policy = ErrorThresholdPolicy(error_threshold=0.27, window_size=200, budget=5, retrain_latency=10, deploy_latency=1)
        metrics = MetricsTracker()

        # Set metadata in metrics for post-analysis
        metrics.set_drift_point(5000)  # Drift starts at t=5000
        metrics.set_budget(policy.budget)

        # Step 3: Run the experiment
        # The runner processes each sample sequentially, updating metrics,
        # checking the retraining policy, and adapting the model
        runner = ExperimentRunner(model, policy, metrics)
        runner.run(X, y)

        # Step 4: Generate and report comprehensive results
        summary = metrics.get_summary()

        print("EXPERIMENT RESULTS")
        print(f"\nConfiguration:")
        print(f"  Drift Type: {drift_type} (starting at t={metrics.drift_point})")
        print(f"  Recurrence Period: {generator.recurrence_period} timesteps")
        print(f"  Policy: ErrorThreshold (threshold={policy.error_threshold}, window={policy.window_size})")
        print(f"  Budget: {policy.budget} retrains")
        print(f"  Seed: {seed}")
        print(f"  Latency: retrain={policy.retrain_latency}s, deploy={policy.deploy_latency}s")
        print(f"  Total Samples: {len(X)}")

        print(f"\nPerformance Metrics:")
        print(f"  Overall Accuracy: {summary['overall_accuracy']:.4f}")
        pre_acc = summary.get('pre_drift_accuracy')
        if pre_acc is not None:
            print(f"  Pre-Drift Accuracy: {pre_acc:.4f} (t=[0, {metrics.drift_point}))")
        post_acc = summary.get('post_drift_accuracy')
        if post_acc is not None:
            print(f"  Post-Drift Accuracy: {post_acc:.4f} (t=[{metrics.drift_point}, {len(X)}))")
        if pre_acc is not None and post_acc is not None:
            print(f"  Accuracy Drop (post - pre): {summary['accuracy_drop']:.4f}")
        print(f"  Cumulative Error Rate: {summary['cumulative_error_rate']:.4f}")
        print(f"  Cumulative Error: {summary['cumulative_error']:.0f}")
        print(f"  Max Degradation: {summary['max_degradation']:.4f}")
        print(f"  Average Degradation: {summary['average_degradation']:.4f}")
        print(f"  Latency Cost: {summary['latency_cost']} timesteps")
        if summary.get('errors_during_latency') is not None:
            print(f"  Errors During Latency: {summary['errors_during_latency']:.0f}")
        if summary.get('error_in_drift_window') is not None:
            print(f"  Errors in Drift Window (t=[{metrics.drift_point}, {metrics.drift_point + 1000})): {int(summary['error_in_drift_window'])}")

        print(f"\nRetraining Events:")
        print(f"  Total Retrains: {summary['total_retrains']}")
        print(f"  Budget Used: {summary.get('budget_used')} / {summary.get('budget_total')}")
        if summary.get('budget_utilization') is not None:
            print(f"  Budget Utilization: {summary['budget_utilization']:.1%}")
        print(f"  Retrains Before Drift: {summary.get('retrains_before_drift')}")
        print(f"  Retrains After Drift: {summary.get('retrains_after_drift')}")

        print(f"\nRetrain Timing:")
        if metrics.retrain_times:
            print(f"  Retrain Timesteps: {metrics.retrain_times}")
            print(f"  Latency Windows (retrain + deploy):")
            for start, end in metrics.retrain_latency_windows:
                print(f"    [{start}, {end})")
        else:
            print(f"  No retrains occurred")

        print(f"\nData Points:")
        print(f"  Predictions Made: {summary['predictions_made']}")
        print(f"  Total Samples: {summary['total_samples']}")
        print("=" * 70)

        # Step 5: Export results to file formats for analysis
        config = {
            "drift_type": drift_type,
            "drift_point": 5000,
            "recurrence_period": generator.recurrence_period,
            "policy_type": "error_threshold",
            "error_threshold": policy.error_threshold,
            "window_size": policy.window_size,
            "budget": policy.budget,
            "random_seed": seed,
        }

        print("\nExporting results...")
        export_to_json(metrics, policy, config, f"results/run_seed_{seed}.json")
        export_to_csv(metrics, policy, config, f"results/per_sample_metrics_seed_{seed}.csv")
        export_summary_to_csv(metrics, policy, config, "results/summary_results.csv")
        print(f"Results exported for seed {seed}!")

    # Generate plots after all seeds complete
    print("\n" + "=" * 70)
    print("Generating visualization...")
    plot_results(seeds, policy, drift_point=5000, drift_type=drift_type)
    print("=" * 70)

if __name__ == "__main__":
    main()
