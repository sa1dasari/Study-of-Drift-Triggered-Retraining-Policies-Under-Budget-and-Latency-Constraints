"""
Metrics tracking module for recording model performance during streaming.

This module tracks prediction accuracy, errors, retraining events, and latency
impacts as the model processes streaming data. Enables comprehensive analysis of
how different retraining policies (periodic, error-threshold, drift-triggered)
adapt to concept drift under budget and latency constraints.
"""

import numpy as np

class MetricsTracker:
    """
    Comprehensive metrics tracker for streaming ML experiments.

    Captures per-sample performance metrics, retraining event logs, and statistics
    for analyzing model behavior under concept drift and latency constraints.

    Tracked Metrics:
    1. Per-sample performance: accuracy/errors at each timestep
    2. Retraining events: when retrains occur and latency periods
    3. Drift impact: accuracy before/after drift point
    4. Aggregate stats: overall accuracy, accuracy in regions of interest
    5. Budget usage: retrains executed vs. budget available

    Attributes:
        accuracies (list): Per-sample accuracies (1 if correct, 0 if wrong)
        errors (list): Per-sample errors (complement of accuracy)
        retrain_times (list): Timesteps when retrains were triggered
        retrain_latency_windows (list): (start_t, end_t) tuples for latency periods
        sample_count (int): Number of samples processed (including t=0 without prediction)
        drift_point (int): Timestep where concept drift occurs (set externally if known)
    """

    def __init__(self):
        """Initialize empty metrics tracking lists and counters."""
        # Per-sample metrics
        self.accuracies = []
        self.errors = []
        self.timestamps = []  # Track which timestep each metric corresponds to

        # Retraining event tracking
        self.retrain_times = []  # Timesteps when retrain was triggered
        self.retrain_latency_windows = []  # (start_t, end_t) tuples for latency periods

        # Metadata (set by experiment runner or externally)
        self.sample_count = 0
        self.drift_point = None  # Set externally if known (e.g., 5000 for drift start)
        self.total_budget = None  # Set externally (e.g., 5, 10, 20)

    def update(self, y_true, y_pred, t=None):
        """
        Record prediction accuracy for a single sample (or batch).

        Compares true labels against predictions and computes accuracy as the
        proportion of correct predictions. Both accuracies and errors lists
        are updated, along with the timestep.

        Args:
            y_true (np.ndarray): True labels, shape (n_samples,) with values in {0, 1}
            y_pred (np.ndarray): Predicted labels, shape (n_samples,) with values in {0, 1}
            t (int, optional): Timestep/sample index (auto-incremented if not provided)
        """
        # Compute accuracy: fraction of correct predictions
        acc = np.mean(y_true == y_pred)

        # Record accuracy, error, and timestamp
        self.accuracies.append(acc)
        self.errors.append(1 - acc)

        if t is None:
            t = len(self.accuracies) - 1
        self.timestamps.append(t)
        self.sample_count = t + 1

    def record_retrain(self, t, retrain_latency, deploy_latency):
        """
        Record that a retraining event occurred at timestep t.

        Tracks when retrains happen and the latency periods during which
        the model is unavailable (retraining) or being deployed.

        Args:
            t (int): Timestep when retrain was triggered
            retrain_latency (int): Time to complete retraining (offline training time)
            deploy_latency (int): Delay after retraining before deployment
        """
        self.retrain_times.append(t)

        # Latency window: [t, t + retrain_latency + deploy_latency)
        latency_end = t + retrain_latency + deploy_latency
        self.retrain_latency_windows.append((t, latency_end))

    def set_drift_point(self, drift_point):
        """
        Set the known drift point for post-analysis segmentation.

        Used to compute pre-drift and post-drift accuracy separately.

        Args:
            drift_point (int): Timestep at which concept drift occurs
        """
        self.drift_point = drift_point

    def set_budget(self, budget):
        """
        Set the retraining budget for reference.

        Args:
            budget (int): Maximum number of retrains allowed (e.g., 5, 10, 20)
        """
        self.total_budget = budget

    def get_accuracy_before_drift(self):
        """
        Compute average accuracy before the drift point.

        Returns:
            float: Mean accuracy for t < drift_point, or None if drift_point not set
        """
        if self.drift_point is None:
            return None

        pre_drift_acc = [
            acc for acc, t in zip(self.accuracies, self.timestamps)
            if t < self.drift_point
        ]
        return np.mean(pre_drift_acc) if pre_drift_acc else None

    def get_accuracy_after_drift(self):
        """
        Compute average accuracy after the drift point.

        Returns:
            float: Mean accuracy for t >= drift_point, or None if drift_point not set
        """
        if self.drift_point is None:
            return None

        post_drift_acc = [
            acc for acc, t in zip(self.accuracies, self.timestamps)
            if t >= self.drift_point
        ]
        return np.mean(post_drift_acc) if post_drift_acc else None

    def get_overall_accuracy(self):
        """
        Compute overall average accuracy across all predictions.

        Returns:
            float: Mean accuracy, or 0 if no predictions recorded
        """
        if not self.accuracies:
            return 0.0
        return np.mean(self.accuracies)

    def get_accuracy_window(self, start_t, end_t):
        """
        Compute average accuracy in a timestep window [start_t, end_t).

        Useful for analyzing performance in different regions (e.g., around drift).

        Args:
            start_t (int): Start timestep (inclusive)
            end_t (int): End timestep (exclusive)

        Returns:
            float: Mean accuracy in the window, or None if no samples in window
        """
        windowed_acc = [
            acc for acc, t in zip(self.accuracies, self.timestamps)
            if start_t <= t < end_t
        ]
        return np.mean(windowed_acc) if windowed_acc else None

    def get_retrains_before_drift(self):
        """
        Count how many retrains occurred before the drift point.

        Returns:
            int: Number of retrains with t < drift_point, or -1 if drift_point not set
        """
        if self.drift_point is None:
            return -1
        return sum(1 for t in self.retrain_times if t < self.drift_point)

    def get_retrains_after_drift(self):
        """
        Count how many retrains occurred at or after the drift point.

        Returns:
            int: Number of retrains with t >= drift_point, or -1 if drift_point not set
        """
        if self.drift_point is None:
            return -1
        return sum(1 for t in self.retrain_times if t >= self.drift_point)

    def get_retrain_count(self):
        """
        Get total number of retrains executed.

        Returns:
            int: Length of retrain_times list
        """
        return len(self.retrain_times)

    def get_cumulative_error(self):
        """
        Compute cumulative error (sum of all errors).

        Returns:
            float: Sum of all per-sample errors
        """
        return np.sum(self.errors) if self.errors else 0.0

    def get_max_degradation(self):
        """
        Compute max degradation (worst single-sample error).

        Returns:
            float: Maximum error value across all samples
        """
        return np.max(self.errors) if self.errors else 0.0

    def get_average_degradation(self):
        """
        Compute average degradation (mean of all errors).

        This represents the typical error rate across all predictions,
        providing a measure of overall model degradation.

        Returns:
            float: Mean error value across all samples
        """
        return np.mean(self.errors) if self.errors else 0.0

    def get_max_degradation_in_window(self, start_t, end_t):
        """
        Compute max degradation within a specific time window.

        Args:
            start_t (int): Start timestep (inclusive)
            end_t (int): End timestep (exclusive)

        Returns:
            float: Maximum error in the window, or 0 if no samples
        """
        errors_in_window = [e for e, t in zip(self.errors, self.timestamps) if start_t <= t < end_t]
        return max(errors_in_window) if errors_in_window else 0.0

    def get_error_count_in_window(self, start_t, end_t):
        """
        Count total errors within a specific time window.

        Args:
            start_t (int): Start timestep (inclusive)
            end_t (int): End timestep (exclusive)

        Returns:
            float: Sum of errors in the window (count of wrong predictions)
        """
        errors_in_window = [e for e, t in zip(self.errors, self.timestamps) if start_t <= t < end_t]
        return sum(errors_in_window) if errors_in_window else 0.0

    def get_latency_cost(self):
        """
        Compute total latency cost as the sum of all latency window durations.

        Latency cost represents the total number of timesteps during which
        the model was unavailable due to retraining or deployment delays.
        Higher latency cost means more time spent in degraded operation.

        Returns:
            int: Total timesteps spent in latency windows across all retrains
        """
        total_latency = 0
        for start_t, end_t in self.retrain_latency_windows:
            total_latency += (end_t - start_t)
        return total_latency

    def get_errors_during_latency(self):
        """
        Compute cumulative errors during latency windows.

        This measures performance degradation specifically during retraining
        and deployment periods when the model may be using stale weights.

        Returns:
            float: Sum of errors during all latency windows
        """
        latency_errors = 0.0
        for acc, t in zip(self.errors, self.timestamps):
            for start_t, end_t in self.retrain_latency_windows:
                if start_t <= t < end_t:
                    latency_errors += acc
                    break
        return latency_errors

    def get_summary(self):
        """
        Generate a summary dictionary of key metrics.

        Returns:
            dict: Summary with keys:
                - overall_accuracy
                - pre_drift_accuracy (if drift_point set)
                - post_drift_accuracy (if drift_point set)
                - accuracy_drop (post - pre, if both set)
                - total_retrains
                - retrains_before_drift (if drift_point set)
                - retrains_after_drift (if drift_point set)
                - total_samples
                - predictions_made (excludes t=0 warm-up)
        """
        summary = {
            'overall_accuracy': self.get_overall_accuracy(),
            'total_retrains': self.get_retrain_count(),
            'total_samples': self.sample_count,
            'predictions_made': len(self.accuracies),
        }

        if self.drift_point is not None:
            pre_acc = self.get_accuracy_before_drift()
            post_acc = self.get_accuracy_after_drift()
            summary['pre_drift_accuracy'] = pre_acc
            summary['post_drift_accuracy'] = post_acc
            if pre_acc is not None and post_acc is not None:
                summary['accuracy_drop'] = post_acc - pre_acc
            summary['retrains_before_drift'] = self.get_retrains_before_drift()
            summary['retrains_after_drift'] = self.get_retrains_after_drift()

        if self.total_budget is not None:
            summary['budget_used'] = self.get_retrain_count()
            summary['budget_total'] = self.total_budget
            summary['budget_utilization'] = self.get_retrain_count() / self.total_budget

        summary['cumulative_error_rate'] = self.get_cumulative_error() / self.sample_count if self.sample_count > 0 else 0.0
        summary['cumulative_error'] = self.get_cumulative_error()
        summary['max_degradation'] = self.get_max_degradation()
        summary['average_degradation'] = self.get_average_degradation()
        summary['latency_cost'] = self.get_latency_cost()
        summary['errors_during_latency'] = self.get_errors_during_latency()
        summary['error_in_drift_window'] = self.get_error_count_in_window(self.drift_point, self.drift_point + 1000) if self.drift_point is not None else None

        return summary
