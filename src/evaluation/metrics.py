"""
Metrics tracking module for recording model performance during streaming.

This module tracks prediction accuracy and errors as the model processes
streaming data, enabling post-analysis of performance over time.
"""

import numpy as np

class MetricsTracker:
    """
    Tracks model accuracy and prediction errors over time.

    For each sample prediction, records whether it was correct or incorrect.
    Maintains running lists of accuracies and errors that can be queried
    for performance analysis, especially to detect concept drift or
    evaluate retraining effectiveness.

    Attributes:
        accuracies (list): List of per-sample accuracies (1 if correct, 0 if wrong)
        errors (list): List of per-sample errors (1 if wrong, 0 if correct)
                      These are complementary: errors[i] = 1 - accuracies[i]
    """

    def __init__(self):
        # Initialize empty metrics lists.
        self.accuracies = []
        self.errors = []

    def update(self, y_true, y_pred):
        """
        Record prediction accuracy for a single sample (or batch).

        Compares true labels against predictions and computes accuracy as the
        proportion of correct predictions. Both lists are updated accordingly.

        Args:
            y_true (np.ndarray): True labels, shape (n_samples,) with values in {0, 1}
            y_pred (np.ndarray): Predicted labels, shape (n_samples,) with values in {0, 1}
        """
        # Compute accuracy: fraction of correct predictions
        # np.mean([True, False, True]) = 0.667 (2 out of 3 correct)
        acc = np.mean(y_true == y_pred)

        # Record accuracy and corresponding error
        self.accuracies.append(acc)
        self.errors.append(1 - acc)  # Error is complement of accuracy
