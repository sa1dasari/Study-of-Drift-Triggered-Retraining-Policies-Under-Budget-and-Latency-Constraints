"""
Base policy module defining the interface for retraining decision strategies.

A retraining policy determines WHEN the model should be retrained to adapt to
concept drift. Different policies implement different strategies while respecting
budget and latency constraints.
"""


class RetrainPolicy:
    """
    Abstract base class for retraining policies.

    A retraining policy makes decisions about when to trigger model retraining
    in a streaming scenario. This allows the system to adapt to concept drift
    while respecting computational budget (max number of retrains allowed).

    Attributes:
        budget (int): Maximum number of times the model can be retrained
        latency (int): Maximum delay allowed before retraining (not yet implemented)
        remaining_budget (int): Retraining budget left (decremented each retrain)
    """

    def __init__(self, budget, latency):

        # Initialize the policy with constraints.
        self.budget = budget
        self.latency = latency
        self.remaining_budget = budget

    def should_retrain(self, t, metrics):
        """
        Determine whether to retrain the model at timestep t.

        This is the key decision-making method. Subclasses implement specific
        strategies (e.g., periodic, drift-detection-based, performance-based).

        Args:
            t (int): Current timestep
            metrics (MetricsTracker): Historical accuracy/error metrics

        Returns:
            bool: True if retraining should occur, False otherwise

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError

    def on_retrain(self):
        """
        Called after a retrain decision is executed.

        Updates the remaining budget by decrementing it by 1.
        This ensures the policy respects the budget constraint.
        """
        self.remaining_budget -= 1
