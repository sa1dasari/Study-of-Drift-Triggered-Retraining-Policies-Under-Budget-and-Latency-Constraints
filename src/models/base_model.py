"""
Streaming model module for online learning on concept drift data.

This module implements a model that can learn incrementally from streaming data
and can be retrained from scratch when needed.

Uses fixed class weights derived from the global fraud rate to handle class
imbalance.  This is critical for both partial_fit (single-sample updates) and
retrain (small-buffer training):

  * partial_fit with a single sample can't compute balanced weights (only one
    class present), so without fixed weights every sample gets weight 1.0 and
    the minority class is under-represented.
  * retrain on a small buffer may have very few fraud cases; buffer-local
    balanced weights are noisy and a fresh model trained on ~175 fraud
    examples learns a poor fraud representation.

Fixed weights ensure every fraud sample receives weight ≈ 1/(2·fraud_rate)
regardless of how many fraud cases happen to be in the current window.
"""

import numpy as np
from sklearn.linear_model import SGDClassifier


# Default global fraud rate for the CIS Fraud Detection dataset.
# Used when no explicit rate is provided.
_DEFAULT_FRAUD_RATE = 0.035


class StreamingModel:
    """
    A wrapper around scikit-learn's SGDClassifier for streaming scenarios.

    SGDClassifier supports online learning through partial_fit(), allowing the model
    to be updated with new data without forgetting previous patterns.

    Fixed class weights are computed once from the known global fraud rate and
    applied as sample_weight in every partial_fit / retrain call.  This avoids
    the pitfall of buffer-local balanced weights which are either unavailable
    (single-sample partial_fit) or noisy (small retrain buffer).

    Args:
        fraud_rate (float): Known proportion of the positive (fraud) class
            in the full dataset.  Defaults to 0.035 (CIS Fraud Detection).

    Attributes:
        model (SGDClassifier): The underlying classifier with log loss
        is_initialized (bool): Whether the model has been initialized with data
        class_weights (dict): {0: w_neg, 1: w_pos} fixed weights
    """

    def __init__(self, fraud_rate=_DEFAULT_FRAUD_RATE):
        """
        Initialize the streaming model.

        Args:
            fraud_rate (float): Known positive-class proportion.  Used to
                compute fixed per-class sample weights.
        """
        self.model = SGDClassifier(loss="log_loss")
        self.is_initialized = False

        # Fixed class weights: w_c = 1 / (n_classes · p_c)
        # For binary: w_0 = 1/(2·(1-fraud_rate)), w_1 = 1/(2·fraud_rate)
        # This mirrors sklearn's "balanced" formula but uses the global rate.
        self.class_weights = {
            0: 1.0 / (2.0 * (1.0 - fraud_rate)),
            1: 1.0 / (2.0 * fraud_rate),
        }

    # ── helpers ──────────────────────────────────────────────────────────

    def _sample_weights(self, y):
        """
        Compute per-sample weights from the fixed class weights.

        Every fraud sample gets weight ≈ 14.3 (for 3.5% fraud rate) and
        every legitimate sample gets weight ≈ 0.518, regardless of how
        many of each are in the current batch.

        Args:
            y (np.ndarray): Label array, shape (n_samples,).

        Returns:
            np.ndarray: Per-sample weights, shape (n_samples,).
        """
        return np.array([self.class_weights[int(label)] for label in y])

    # ── public API ──────────────────────────────────────────────────────

    def predict(self, X):
        """
        Make predictions on new samples.

        Args:
            X (np.ndarray): Input features, shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted labels {0, 1}, shape (n_samples,)
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Return probability estimates for new samples.

        Args:
            X (np.ndarray): Input features, shape (n_samples, n_features)

        Returns:
            np.ndarray: Probability of the positive class (class 1),
                        shape (n_samples,).
        """
        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def partial_fit(self, X, y):
        """
        Incrementally train the model on new samples (streaming update).

        Fixed class weights are applied so that a single fraud sample
        receives ~14× the weight of a single legitimate sample, matching
        the global class imbalance.

        Args:
            X (np.ndarray): Input features, shape (n_samples, n_features)
            y (np.ndarray): Binary labels {0, 1}, shape (n_samples,)
        """
        sw = self._sample_weights(y)
        if not self.is_initialized:
            self.model.partial_fit(X, y, classes=[0, 1], sample_weight=sw)
            self.is_initialized = True
        else:
            self.model.partial_fit(X, y, sample_weight=sw)

    def retrain(self, X, y, n_epochs=5):
        """
        Completely retrain the model from scratch on a data buffer.

        Uses the same fixed class weights as partial_fit so the fresh model
        starts with the correct fraud/legitimate weighting regardless of
        how many fraud cases happen to fall in the retrain buffer.

        Multiple epochs are run because a single pass over a small buffer
        (~5 k samples, 428 features) is insufficient for a fresh
        SGDClassifier to converge.

        Args:
            X (np.ndarray): Input features, shape (n_samples, n_features)
            y (np.ndarray): Binary labels {0, 1}, shape (n_samples,)
            n_epochs (int): Number of passes over the buffer (default 5).
        """
        self.model = SGDClassifier(loss="log_loss")
        sw = self._sample_weights(y)
        self.model.partial_fit(X, y, classes=[0, 1], sample_weight=sw)
        for _ in range(n_epochs - 1):
            self.model.partial_fit(X, y, sample_weight=sw)
        self.is_initialized = True
