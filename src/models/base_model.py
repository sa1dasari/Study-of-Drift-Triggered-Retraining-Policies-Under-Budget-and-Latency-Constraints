"""
Streaming model module for online learning on concept drift data.

This module implements a model that can learn incrementally from streaming data
and can be retrained from scratch when needed.
"""

from sklearn.linear_model import SGDClassifier

class StreamingModel:
    """
    A wrapper around scikit-learn's SGDClassifier for streaming scenarios.

    SGDClassifier supports online learning through partial_fit(), allowing the model
    to be updated with new data without forgetting previous patterns (controlled by
    the learning algorithm's internal mechanisms).

    The model tracks initialization state to handle the first call to partial_fit
    differently (must provide class labels).

    Attributes:
        model (SGDClassifier): The underlying classifier with log loss (logistic regression)
        is_initialized (bool): Whether the model has been initialized with data
    """

    def __init__(self):
        """
        Initialize the streaming model.

        Uses log loss (logistic regression objective) which is appropriate for
        binary classification with concept drift.
        """
        # SGDClassifier with log_loss implements logistic regression
        # loss="log_loss" minimizes log(1 + exp(-y * decision_function))
        self.model = SGDClassifier(loss="log_loss")
        self.is_initialized = False

    def predict(self, X):

        # Make predictions on new samples.
        return self.model.predict(X)

    def partial_fit(self, X, y):
        """
        Incrementally train the model on new samples (streaming update).

        This is the core method for online learning. Each call updates the model
        weights to incorporate the new data. The model learns the relationship
        between X and y without completely forgetting previous patterns.

        The first call must include classes=[0, 1] to inform scikit-learn about
        all possible class labels. Subsequent calls can omit this.

        Args:
            X (np.ndarray): Input features, shape (n_samples, n_features)
            y (np.ndarray): Binary labels {0, 1}, shape (n_samples,)
        """
        if not self.is_initialized:
            # First time: must tell partial_fit about all possible classes
            self.model.partial_fit(X, y, classes=[0, 1])
            self.is_initialized = True
        else:
            # Subsequent calls: just update with new data
            self.model.partial_fit(X, y)

    def retrain(self, X, y):
        """
        Completely retrain the model from scratch.

        This creates a fresh model and trains it on all provided data.
        Used when the retraining policy decides that drift has occurred
        and the model needs to adapt to the new data distribution.

        This is more expensive than partial_fit but allows the model to
        completely reset and learn from a clean slate.

        Args:
            X (np.ndarray): Input features, shape (n_samples, n_features)
            y (np.ndarray): Binary labels {0, 1}, shape (n_samples,)
        """
        # Create a fresh model (discarding previous weights)
        self.model = SGDClassifier(loss="log_loss")

        # Train on all provided data from scratch
        # Use partial_fit with classes specified to handle cases where window
        # may only contain one class (e.g., all 0s before first retrain)
        self.model.partial_fit(X, y, classes=[0, 1])

        # Mark as initialized since partial_fit initializes the model
        self.is_initialized = True
