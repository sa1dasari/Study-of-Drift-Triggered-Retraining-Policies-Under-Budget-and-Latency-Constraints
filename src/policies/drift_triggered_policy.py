"""
Drift-triggered retraining policy using ADWIN-based concept drift detection.

This module implements a retraining strategy that uses the ADWIN (ADaptive WINdowing)
algorithm to detect statistically significant changes in the prediction error
distribution. When drift is detected, a retrain is triggered.

Unlike the error-threshold policy (which reacts to high error rates) or the periodic
policy (which retrains on a fixed schedule), this policy specifically looks for
*changes* in the error distribution -- the hallmark of concept drift.
"""

import numpy as np

from src.policies.base_policy import RetrainPolicy


class DriftTriggeredPolicy(RetrainPolicy):
    """
    Triggers retraining when concept drift is detected via ADWIN.

    ADWIN (ADaptive WINdowing) detects distributional changes by maintaining a
    sliding window of recent prediction errors and checking whether any two
    contiguous sub-windows have statistically significantly different means,
    using the Hoeffding bound as a significance test.

    When a significant change is detected, drift is signalled and retraining
    is triggered (subject to budget and latency constraints).

    Attributes:
        delta (float): Confidence parameter for the Hoeffding bound.
            Lower values make the detector *less* sensitive (fewer false alarms).
            Typical range: 0.0001 -- 0.05.
        window_size (int): Maximum number of recent errors to consider
            for drift detection.  Older errors are discarded.
        min_samples (int): Minimum number of observed errors before drift
            detection activates.  Avoids spurious detections during warm-up.
        remaining_budget (int): Inherited from RetrainPolicy, counts down retrains.
    """

    def __init__(self, delta=0.002, window_size=500, min_samples=100,
                 budget=20, retrain_latency=500, deploy_latency=20):
        """
        Initialize drift-triggered retraining policy.

        Args:
            delta (float): Confidence parameter for ADWIN Hoeffding bound.
                Smaller -> less sensitive (fewer false positives, may miss subtle drift).
                Larger  -> more sensitive (catches subtle drift, more false alarms).
            window_size (int): Max number of recent errors kept for detection.
            min_samples (int): Minimum errors observed before detection begins.
            budget (int): Maximum number of retrains allowed.
            retrain_latency (int): Timesteps to complete retraining.
            deploy_latency (int): Timesteps to wait after retraining before deployment.
        """
        super().__init__(budget, retrain_latency, deploy_latency)
        self.delta = delta
        self.window_size = window_size
        self.min_samples = min_samples

    def _adwin_detect(self, errors):
        """
        Run ADWIN-style drift detection on a sequence of error values.

        Scans every valid split point in the window.  For each split, the
        left and right sub-window means are compared.  If the absolute
        difference exceeds the Hoeffding bound epsilon, drift is detected.

        The Hoeffding bound used is:
            epsilon = sqrt( ln(4/delta) / (2*m) )
        where m = 1 / (1/n_left + 1/n_right)  (harmonic-style sample size).

        Args:
            errors (list | np.ndarray): Recent error values (0 or 1 each).

        Returns:
            bool: True if a statistically significant change is found.
        """
        n = len(errors)
        # Need enough data in both sub-windows
        min_split = max(30, n // 10)
        if n < 2 * min_split:
            return False

        # Pre-compute cumulative sums for O(n) mean lookups
        cum_sum = np.cumsum(errors)
        total_sum = cum_sum[-1]
        ln_term = np.log(4.0 / self.delta)

        for i in range(min_split, n - min_split + 1):
            n_left = i
            n_right = n - i

            sum_left = cum_sum[i - 1]
            sum_right = total_sum - sum_left

            mean_left = sum_left / n_left
            mean_right = sum_right / n_right

            # Harmonic-style effective sample size
            m = 1.0 / (1.0 / n_left + 1.0 / n_right)
            epsilon = np.sqrt(ln_term / (2.0 * m))

            if abs(mean_left - mean_right) >= epsilon:
                return True

        return False

    def should_retrain(self, t, metrics):
        """
        Decide whether to retrain based on ADWIN drift detection.

        Retraining is triggered when all the following are true:
        1. Not currently in a latency period (retrain/deploy not in progress)
        2. Budget allows (remaining_budget > 0)
        3. Enough samples have been observed (at least min_samples predictions)
        4. ADWIN detects a statistically significant change in the recent
           error distribution

        Args:
            t (int): Current timestep.
            metrics (MetricsTracker): Metrics object with per-sample error history.

        Returns:
            bool: True if concept drift is detected and budget/latency
                  constraints allow retraining.
        """
        # Check if already in latency period: if so, no new retrain
        if self.is_in_latency_period(t):
            return False

        # Check budget constraint: if no budget left, never retrain
        if self.remaining_budget <= 0:
            return False

        # Need enough error history before detection kicks in
        if len(metrics.errors) < self.min_samples:
            return False

        # Extract the detection window (most recent window_size errors)
        recent_errors = metrics.errors[-self.window_size:]

        # Run ADWIN drift detection
        return self._adwin_detect(recent_errors)

