"""
Periodic retraining policy for regular model adaptation.

This module implements a simple retraining strategy that triggers retraining
at fixed intervals, allowing the model to periodically adapt to recent data
regardless of whether drift has been detected.
"""

from src.policies.base_policy import RetrainPolicy

class PeriodicPolicy(RetrainPolicy):
    """
    Triggers retraining at regular intervals.

    This is a straightforward baseline policy: retrain every 'interval' timesteps.
    It's simple but effective for moderate levels of drift, though it may
    retrain unnecessarily if drift hasn't occurred, or insufficiently often
    if drift is rapid.

    The policy respects the budget constraint - once remaining_budget reaches 0,
    it stops triggering retrains.

    Attributes:
        interval (int): Number of timesteps between retrains
        remaining_budget (int): Inherited from RetrainPolicy, counts down retrains
    """

    def __init__(self, interval, budget, latency):

        # Initialize periodic retraining policy.
        super().__init__(budget, latency)
        self.interval = interval

    def should_retrain(self, t, metrics):
        """
        Decide whether to retrain based on periodic schedule.

        Retrains are triggered when:
        1. Current timestep t is a multiple of the interval
        2. Budget allows (remaining_budget > 0)

        Args:
            t (int): Current timestep
            metrics (MetricsTracker): Metrics object (not used in periodic policy)

        Returns:
            bool: True if timestep is multiple of interval AND budget remains
        """
        # First check budget constraint: if no budget left, never retrain
        if self.remaining_budget <= 0:
            return False

        # Retrain if timestep is a multiple of interval
        return t % self.interval == 0
