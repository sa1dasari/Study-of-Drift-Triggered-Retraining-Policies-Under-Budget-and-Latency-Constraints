"""
Never-retrain (baseline) policy -- partial_fit only, no full retraining.

This policy serves as the **floor baseline** for all experiments.
The model is updated incrementally via partial_fit on every sample but is
never retrained from scratch. Budget and latency are both fixed at 0.

Comparing other policies against this baseline makes every accuracy number
interpretable: any improvement over NeverRetrain is attributable to the
retraining strategy.
"""

from src.policies.base_policy import RetrainPolicy


class NeverRetrainPolicy(RetrainPolicy):
    """
    Baseline policy that never triggers retraining.

    The model relies entirely on incremental learning (partial_fit).
    Budget = 0, retrain_latency = 0, deploy_latency = 0.

    This provides a lower-bound accuracy that other retraining policies
    should exceed if they are adding value.
    """

    def __init__(self):
        """Initialize with zero budget and zero latency."""
        super().__init__(budget=0, retrain_latency=0, deploy_latency=0)

    def should_retrain(self, t, metrics):
        """Never retrain -- always returns False.

        Args:
            t (int): Current timestep.
            metrics (MetricsTracker): Historical accuracy/error metrics.

        Returns:
            bool: Always False.
        """
        return False

