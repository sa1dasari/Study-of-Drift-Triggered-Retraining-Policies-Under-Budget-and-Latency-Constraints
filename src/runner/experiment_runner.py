"""
Experiment runner module orchestrating the streaming learning pipeline.

This module implements the core event loop for processing streaming data,
coordinating between the model, policy, and metrics tracking.
"""

import numpy as np

class ExperimentRunner:
    """
    Orchestrates the streaming learning experiment.

    This class manages the main loop that processes streaming data samples:
    1. Makes predictions on new samples
    2. Evaluates prediction accuracy
    3. Collects samples in a window
    4. Checks if retraining is needed via the policy
    5. Performs retraining if triggered
    6. Updates the model incrementally

    Attributes:
        model (StreamingModel): The ML model (makes predictions, partial_fit, retrain)
        policy (RetrainPolicy): Retraining policy (decides when to retrain)
        metrics (MetricsTracker): Tracks accuracy and errors over time
    """

    def __init__(self, model, policy, metrics):
        """
        Initialize the experiment runner with core components.

        Args:
            model (StreamingModel): Initialized StreamingModel instance
            policy (RetrainPolicy): Initialized RetrainPolicy instance
            metrics (MetricsTracker): Initialized MetricsTracker instance
        """
        self.model = model
        self.policy = policy
        self.metrics = metrics

    def run(self, X, y):
        """
        Execute the streaming learning pipeline.

        For each sample at timestep t:
        1. Extract single sample x_t and label y_t
        2. Make prediction (if model is initialized)
        3. Record prediction accuracy/error
        4. Add sample to sliding window buffer
        5. Check if policy says to retrain
        6. If retraining needed: reset and retrain on window data
        7. Always: do incremental update (partial_fit)

        The window accumulates all data since last retraining and is cleared
        after each retrain, so the model retrains on the most recent samples
        collected since the last retrain event.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features)
            y (np.ndarray): Labels array, shape (n_samples,)
        """
        # Initialize sliding window to accumulate data between retrains
        # These are reset whenever retraining occurs
        window_X, window_y = [], []

        # Initialize model with first sample so predictions can be made from t=0
        # This ensures we get predictions for all samples, not n-1
        if len(X) > 0:
            x_init = X[0].reshape(1, -1)
            y_init = np.array([y[0]])
            self.model.partial_fit(x_init, y_init)

        # Process each sample sequentially (streaming)
        for t in range(len(X)):
            # Step 1: Extract single sample and reshape for model compatibility
            # reshape(1, -1) converts 1D array to 2D with shape (1, n_features)
            x_t = X[t].reshape(1, -1)
            y_t = np.array([y[t]])

            # Step 2 & 3: Make prediction and record accuracy
            # Model is initialized before loop, so predictions are made for all samples
            if self.model.is_initialized:
                # Predict on current sample
                y_pred = self.model.predict(x_t)
                # Update metrics with true label, prediction, and timestamp
                self.metrics.update(y_t, y_pred, t=t)

            # Step 4: Add current sample to sliding window
            # window_X grows until next retrain, window_y tracks corresponding labels
            window_X.append(x_t[0])  # x_t[0] extracts the 1D array from (1, n_features)
            window_y.append(y_t[0])  # y_t[0] extracts the scalar label

            # Step 5 & 6: Check retraining policy and retrain if needed
            # Policy.should_retrain() checks timestamp and budget constraints
            if self.policy.should_retrain(t, self.metrics):
                # Retrain on accumulated window data
                # This resets model weights and trains from scratch on window
                self.model.retrain(np.array(window_X), np.array(window_y))

                # Update policy's remaining budget counter and record retrain time
                self.policy.on_retrain(t)

                # Record retrain event in metrics for post-analysis
                self.metrics.record_retrain(t, self.policy.retrain_latency, self.policy.deploy_latency)

                # Reset window for next cycle
                # Next retrain will be on the new samples collected from now on
                window_X, window_y = [], []

            # Step 7: Incremental update regardless of retraining
            # This updates model weights based on the single new sample
            # Even after a full retrain, we still do partial_fit on the new sample
            self.model.partial_fit(x_t, y_t)
