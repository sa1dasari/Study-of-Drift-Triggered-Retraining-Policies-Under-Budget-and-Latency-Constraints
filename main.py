"""
Main entry point for the Drift-Triggered Retraining Policies experiment.

This script orchestrates an end-to-end machine learning pipeline that handles
concept drift in streaming data. It demonstrates how different retraining policies
can adapt a model to changing data distributions under budget and latency constraints.
"""

from src.data.drift_generator import DriftGenerator
from src.models.base_model import StreamingModel
from src.policies.periodic import PeriodicPolicy
from src.evaluation.metrics import MetricsTracker
from src.runner.experiment_runner import ExperimentRunner

def main():
    """
    Main execution function that sets up and runs the entire experiment pipeline.

    Flow:
    1. Generate synthetic streaming data with concept drift
    2. Create a streaming model and retraining policy
    3. Run the experiment using the ExperimentRunner
    4. Report final model accuracy
    """
    # Step 1: Generate synthetic data with abrupt drift at t=5000
    # (default parameters: 10 features, drift_point=5000, seed=42)
    generator = DriftGenerator(drift_type="abrupt")
    X, y = generator.generate(10000)

    # Step 2: Initialize components
    # - StreamingModel: Uses SGDClassifier for online learning
    # - PeriodicPolicy: Retrain every 500 samples, max 10 retrains allowed
    # - MetricsTracker: Records prediction accuracy/errors over time
    model = StreamingModel()
    policy = PeriodicPolicy(interval=500, budget=10, latency=0)
    metrics = MetricsTracker()

    # Step 3: Run the experiment
    # The runner processes each sample sequentially, updating metrics,
    # checking the retraining policy, and adapting the model
    runner = ExperimentRunner(model, policy, metrics)
    runner.run(X, y)

    # Step 4: Report final results
    # Calculate average accuracy as 1 - average_error
    print(f"Total samples processed: {len(X)}")
    print(f"Metrics recorded: {len(metrics.errors)}")
    print(f"Retrains executed: {policy.budget - policy.remaining_budget}")

    if len(metrics.errors) > 0:
        final_accuracy = 1 - sum(metrics.errors) / len(metrics.errors)
        print("Final accuracy:", final_accuracy)
    else:
        print("No metrics recorded - model may not have been initialized properly")

if __name__ == "__main__":
    main()
