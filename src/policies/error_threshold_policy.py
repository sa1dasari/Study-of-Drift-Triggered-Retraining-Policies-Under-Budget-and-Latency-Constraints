"""
Error-threshold retraining policy for performance-based model adaptation.

This module implements a retraining strategy that triggers retraining when the
model's recent error rate exceeds a configurable threshold.
"""

import numpy as np

from src.policies.base_policy import RetrainPolicy


class ErrorThresholdPolicy(RetrainPolicy):
    """
    Triggers retraining when the recent error rate exceeds a threshold.

    This policy monitors a sliding window of recent prediction errors and
    computes the average error rate over that window. If the error rate
    exceeds the configured threshold, a retrain is triggered.

    Attributes:
        error_threshold (float): Error rate threshold in [0, 1] above which
            retraining is triggered (e.g., 0.4 means retrain when >40% errors).
        window_size (int): Number of recent samples over which to compute
            the rolling error rate.
        remaining_budget (int): Inherited from RetrainPolicy, counts down retrains.
    """

    def __init__(self, error_threshold, window_size, budget, retrain_latency, deploy_latency):
        """
        Initialize error-threshold retraining policy.

        Args:
            error_threshold (float): Error rate threshold (0.0–1.0)
            window_size (int): Number of recent predictions to consider when
                computing the rolling error rate
            budget (int): Maximum number of retrains allowed.
            retrain_latency (int): Timesteps to complete retraining.
            deploy_latency (int): Timesteps to wait after retraining before deployment.
        """
        super().__init__(budget, retrain_latency, deploy_latency)
        self.error_threshold = error_threshold
        self.window_size = window_size

    def should_retrain(self, t, metrics):
        """
        Decide whether to retrain based on recent error rate.

        Retraining is triggered when all the following are true:
        1. Not currently in a latency period (retrain/deploy not in progress)
        2. Budget allows (remaining_budget > 0)
        3. Enough samples have been observed (at least window_size predictions)
        4. The average error rate over the last window_size predictions
           exceeds the error_threshold

        Args:
            t (int): Current timestep.
            metrics (MetricsTracker): Metrics object with per-sample error history.

        Returns:
            bool: True if the recent error rate exceeds the threshold and
                  budget/latency constraints allow retraining.
        """
        # Check if already in latency period: if so, no new retrain
        if self.is_in_latency_period(t):
            return False

        # Check budget constraint: if no budget left, never retrain
        if self.remaining_budget <= 0:
            return False

        # Need enough error history to fill the window
        if len(metrics.errors) < self.window_size:
            return False

        # Compute rolling error rate over the most recent window_size samples
        recent_errors = metrics.errors[-self.window_size:]
        error_rate = np.mean(recent_errors)

        # Trigger retrain if error rate exceeds the threshold
        return error_rate > self.error_threshold

