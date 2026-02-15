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
    while respecting computational budget (max number of retrains allowed) and
    latency constraints.

    The latency model works as follows:
    1. At timestep t, if should_retrain() returns True, a retrain is triggered
    2. The model is retrained offline for retrain_latency timesteps (t to t+retrain_latency-1)
    3. After retraining completes, there's a deploy_latency delay (t+retrain_latency to t+retrain_latency+deploy_latency-1)
    4. The new model becomes active at timestep t+retrain_latency+deploy_latency
    5. During both periods, should_retrain() returns False to prevent overlapping retrains

    Attributes:
        budget (int): Maximum number of times the model can be retrained
        retrain_latency (int): Number of timesteps to complete retraining (offline training time)
        deploy_latency (int): Delay after retraining before new model is deployed
        remaining_budget (int): Retraining budget left (decremented each retrain)
        last_retrain_time (int): Timestep when last retrain was triggered (-inf initially)
    """

    def __init__(self, budget, retrain_latency, deploy_latency):
        """
        Initialize the policy with budget and latency constraints.

        Args:
            budget (int): Maximum number of retrains allowed
            retrain_latency (int): Timesteps needed to complete retraining (e.g., 10, 100, 500)
            deploy_latency (int): Timesteps to wait after retraining before deployment (e.g., 1, 5, 20)
        """
        self.budget = budget
        self.retrain_latency = retrain_latency
        self.deploy_latency = deploy_latency
        self.remaining_budget = budget
        self.last_retrain_time = float('-inf')  # No retrain has occurred yet

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

    def is_in_latency_period(self, t):
        """
        Check if a retrain is currently in progress or being deployed.

        Returns True if the current timestep t falls within either:
        - Retrain window: [last_retrain_time, last_retrain_time + retrain_latency)
        - Deploy window: [last_retrain_time + retrain_latency, last_retrain_time + retrain_latency + deploy_latency)

        During this period, should_retrain() must return False to prevent overlapping retrains.

        Args:
            t (int): Current timestep

        Returns:
            bool: True if t is within retrain or deploy latency windows
        """
        if self.last_retrain_time == float('-inf'):
            return False  # No retrain has occurred yet, not in latency period

        # Retrain period ends at: last_retrain_time + retrain_latency
        # Deploy period ends at: last_retrain_time + retrain_latency + deploy_latency
        total_latency = self.retrain_latency + self.deploy_latency
        return t < self.last_retrain_time + total_latency

    def on_retrain(self, t):
        """
        Called after a retrain decision is executed.

        Updates the remaining budget by decrementing it by 1, and records
        the timestep at which retraining was triggered. This enables the
        is_in_latency_period() check to prevent overlapping retrains.

        Args:
            t (int): Current timestep when retrain was triggered
        """
        self.remaining_budget -= 1
        self.last_retrain_time = t
