"""
Metrics tracking module for recording model performance during streaming.

This module tracks prediction accuracy, errors, F1 score, AUC, retraining
events, and latency impacts as the model processes streaming data. Enables
comprehensive analysis of how different retraining policies adapt to concept
drift under budget and latency constraints.

F1 and AUC are essential on imbalanced datasets (e.g. CIS Fraud Detection,
3.5% fraud) where accuracy is dominated by the majority class and cannot
distinguish between policies.
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

class MetricsTracker:
    """
    Comprehensive metrics tracker for streaming ML experiments.

    Captures per-sample performance metrics, retraining event logs, and statistics
    for analyzing model behavior under concept drift and latency constraints.

    Tracked Metrics:
    1. Per-sample performance: accuracy/errors at each timestep
    2. Per-sample labels, predictions, and probabilities (for F1 / AUC)
    3. Retraining events: when retrains occur and latency periods
    4. Drift impact: accuracy, F1, AUC before/after drift point
    5. Aggregate stats: overall accuracy, F1, AUC
    6. Budget usage: retrains executed vs. budget available

    Attributes:
        accuracies (list): Per-sample accuracies (1 if correct, 0 if wrong)
        errors (list): Per-sample errors (complement of accuracy)
        y_trues (list): Per-sample true labels (int 0/1)
        y_preds (list): Per-sample predicted labels (int 0/1)
        y_probs (list): Per-sample predicted probability of class 1
        retrain_times (list): Timesteps when retrains were triggered
        retrain_latency_windows (list): (start_t, end_t) tuples for latency periods
        sample_count (int): Number of samples processed
        drift_point (int): Timestep where concept drift occurs (set externally)
    """

    def __init__(self):
        """Initialize empty metrics tracking lists and counters."""
        # Per-sample metrics
        self.accuracies = []
        self.errors = []
        self.timestamps = []  # Track which timestep each metric corresponds to

        # Raw per-sample labels / predictions / probabilities for F1 & AUC
        self.y_trues = []
        self.y_preds = []
        self.y_probs = []   # P(class=1); empty if probabilities unavailable

        # Retraining event tracking
        self.retrain_times = []
        self.retrain_latency_windows = []

        # Metadata (set by experiment runner or externally)
        self.sample_count = 0
        self.drift_point = None
        self.total_budget = None

    def update(self, y_true, y_pred, t=None, y_prob=None):
        """
        Record prediction results for a single sample (or batch).

        Args:
            y_true (np.ndarray): True labels, shape (n_samples,) with values in {0, 1}
            y_pred (np.ndarray): Predicted labels, shape (n_samples,) with values in {0, 1}
            t (int, optional): Timestep/sample index (auto-incremented if not provided)
            y_prob (np.ndarray, optional): Predicted probability of class 1,
                shape (n_samples,).  Required for AUC computation.
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

        # Store raw values (scalar extraction for single-sample calls)
        self.y_trues.append(int(y_true[0]))
        self.y_preds.append(int(y_pred[0]))
        if y_prob is not None:
            self.y_probs.append(float(y_prob[0]))

    def record_retrain(self, t, retrain_latency, deploy_latency):
        """
        Record that a retraining event occurred at timestep t.

        Args:
            t (int): Timestep when retrain was triggered
            retrain_latency (int): Time to complete retraining
            deploy_latency (int): Delay after retraining before deployment
        """
        self.retrain_times.append(t)
        latency_end = t + retrain_latency + deploy_latency
        self.retrain_latency_windows.append((t, latency_end))

    def set_drift_point(self, drift_point):
        """Set the known drift point for post-analysis segmentation."""
        self.drift_point = drift_point

    def set_budget(self, budget):
        """Set the retraining budget for reference."""
        self.total_budget = budget

    # ── Accuracy helpers ────────────────────────────────────────────────

    def get_accuracy_before_drift(self):
        if self.drift_point is None:
            return None
        pre = [acc for acc, t in zip(self.accuracies, self.timestamps)
               if t < self.drift_point]
        return np.mean(pre) if pre else None

    def get_accuracy_after_drift(self):
        if self.drift_point is None:
            return None
        post = [acc for acc, t in zip(self.accuracies, self.timestamps)
                if t >= self.drift_point]
        return np.mean(post) if post else None

    def get_overall_accuracy(self):
        if not self.accuracies:
            return 0.0
        return np.mean(self.accuracies)

    def get_accuracy_window(self, start_t, end_t):
        windowed = [acc for acc, t in zip(self.accuracies, self.timestamps)
                    if start_t <= t < end_t]
        return np.mean(windowed) if windowed else None

    # ── F1 helpers ──────────────────────────────────────────────────────

    def _f1_for_mask(self, mask):
        """Compute F1(class=1) for samples selected by *mask* (bool list)."""
        yt = [self.y_trues[i] for i, m in enumerate(mask) if m]
        yp = [self.y_preds[i] for i, m in enumerate(mask) if m]
        if not yt or len(set(yt)) < 1:
            return None
        return float(f1_score(yt, yp, zero_division=0.0))

    def get_overall_f1(self):
        if not self.y_trues:
            return 0.0
        return float(f1_score(self.y_trues, self.y_preds, zero_division=0.0))

    def get_f1_before_drift(self):
        if self.drift_point is None or not self.y_trues:
            return None
        mask = [t < self.drift_point for t in self.timestamps]
        return self._f1_for_mask(mask)

    def get_f1_after_drift(self):
        if self.drift_point is None or not self.y_trues:
            return None
        mask = [t >= self.drift_point for t in self.timestamps]
        return self._f1_for_mask(mask)

    # ── AUC helpers ─────────────────────────────────────────────────────

    def _auc_for_mask(self, mask):
        """Compute ROC-AUC for samples selected by *mask*.

        Returns None when probabilities are unavailable or only one class
        is present in the selected window (AUC is undefined).
        """
        if not self.y_probs:
            return None
        yt = [self.y_trues[i] for i, m in enumerate(mask) if m]
        yp = [self.y_probs[i] for i, m in enumerate(mask) if m]
        if len(set(yt)) < 2:
            return None  # AUC undefined with single class
        return float(roc_auc_score(yt, yp))

    def get_overall_auc(self):
        if not self.y_probs or len(set(self.y_trues)) < 2:
            return None
        return float(roc_auc_score(self.y_trues, self.y_probs))

    def get_auc_before_drift(self):
        if self.drift_point is None or not self.y_probs:
            return None
        mask = [t < self.drift_point for t in self.timestamps]
        return self._auc_for_mask(mask)

    def get_auc_after_drift(self):
        if self.drift_point is None or not self.y_probs:
            return None
        mask = [t >= self.drift_point for t in self.timestamps]
        return self._auc_for_mask(mask)

    # ── Retrain counters ────────────────────────────────────────────────

    def get_retrains_before_drift(self):
        if self.drift_point is None:
            return -1
        return sum(1 for t in self.retrain_times if t < self.drift_point)

    def get_retrains_after_drift(self):
        if self.drift_point is None:
            return -1
        return sum(1 for t in self.retrain_times if t >= self.drift_point)

    def get_retrain_count(self):
        return len(self.retrain_times)

    # ── Error / degradation helpers ─────────────────────────────────────

    def get_cumulative_error(self):
        return np.sum(self.errors) if self.errors else 0.0

    def get_max_degradation(self):
        return np.max(self.errors) if self.errors else 0.0

    def get_average_degradation(self):
        return np.mean(self.errors) if self.errors else 0.0

    def get_max_degradation_in_window(self, start_t, end_t):
        errs = [e for e, t in zip(self.errors, self.timestamps)
                if start_t <= t < end_t]
        return max(errs) if errs else 0.0

    def get_error_count_in_window(self, start_t, end_t):
        errs = [e for e, t in zip(self.errors, self.timestamps)
                if start_t <= t < end_t]
        return sum(errs) if errs else 0.0

    def get_latency_cost(self):
        total = 0
        for s, e in self.retrain_latency_windows:
            total += (e - s)
        return total

    def get_errors_during_latency(self):
        lat_err = 0.0
        for acc, t in zip(self.errors, self.timestamps):
            for s, e in self.retrain_latency_windows:
                if s <= t < e:
                    lat_err += acc
                    break
        return lat_err

    # ── Summary ─────────────────────────────────────────────────────────

    def get_summary(self):
        """
        Generate a summary dictionary of key metrics.

        Returns:
            dict: Summary including accuracy, F1, AUC (overall and pre/post
                  drift), retrain counts, budget utilisation, error stats.
        """
        summary = {
            'overall_accuracy': self.get_overall_accuracy(),
            'overall_f1': self.get_overall_f1(),
            'overall_auc': self.get_overall_auc(),
            'total_retrains': self.get_retrain_count(),
            'total_samples': self.sample_count,
            'predictions_made': len(self.accuracies),
        }

        if self.drift_point is not None:
            pre_acc  = self.get_accuracy_before_drift()
            post_acc = self.get_accuracy_after_drift()
            summary['pre_drift_accuracy']  = pre_acc
            summary['post_drift_accuracy'] = post_acc
            if pre_acc is not None and post_acc is not None:
                summary['accuracy_drop'] = post_acc - pre_acc

            pre_f1  = self.get_f1_before_drift()
            post_f1 = self.get_f1_after_drift()
            summary['pre_drift_f1']  = pre_f1
            summary['post_drift_f1'] = post_f1
            if pre_f1 is not None and post_f1 is not None:
                summary['f1_drop'] = post_f1 - pre_f1

            pre_auc  = self.get_auc_before_drift()
            post_auc = self.get_auc_after_drift()
            summary['pre_drift_auc']  = pre_auc
            summary['post_drift_auc'] = post_auc
            if pre_auc is not None and post_auc is not None:
                summary['auc_drop'] = post_auc - pre_auc

            summary['retrains_before_drift'] = self.get_retrains_before_drift()
            summary['retrains_after_drift']  = self.get_retrains_after_drift()

        if self.total_budget is not None:
            summary['budget_used'] = self.get_retrain_count()
            summary['budget_total'] = self.total_budget
            summary['budget_utilization'] = (
                self.get_retrain_count() / self.total_budget
                if self.total_budget > 0 else 0.0
            )

        summary['cumulative_error_rate'] = (
            self.get_cumulative_error() / self.sample_count
            if self.sample_count > 0 else 0.0
        )
        summary['cumulative_error'] = self.get_cumulative_error()
        summary['max_degradation'] = self.get_max_degradation()
        summary['average_degradation'] = self.get_average_degradation()
        summary['latency_cost'] = self.get_latency_cost()
        summary['errors_during_latency'] = self.get_errors_during_latency()
        summary['error_in_drift_window'] = (
            self.get_error_count_in_window(self.drift_point, self.drift_point + 1000)
            if self.drift_point is not None else None
        )

        return summary
