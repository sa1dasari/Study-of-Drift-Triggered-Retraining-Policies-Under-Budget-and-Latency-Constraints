"""
Experiment runner module -- static model variant (NO incremental learning).

Identical to ExperimentRunner except the per-sample ``partial_fit`` call
is removed.  The model is initialized once on the first sample and then
**only** updated when the policy triggers an explicit retrain.  Between
retrains the model is completely frozen.

This isolates the effect of the retraining *policy* from any benefit
provided by continuous incremental learning.
"""

import numpy as np


class ExperimentRunnerNoPartialFit:
    """
    Orchestrates a streaming experiment *without* incremental learning.

    Processing loop:
    1. Predict on the new sample
    2. Record accuracy / error
    3. Accumulate sample in the sliding window
    4. Ask the policy whether to retrain
    5. If yes -> retrain on the window, reset window
    6. (NO partial_fit -- model is static between retrains)

    Attributes:
        model (StreamingModel): The ML model
        policy (RetrainPolicy): Retraining policy
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
        Execute the streaming pipeline WITHOUT incremental learning.

        For each sample at timestep t:
        1. Extract single sample x_t and label y_t
        2. Make prediction (if model is initialized)
        3. Record prediction accuracy/error
        4. Add sample to sliding window buffer
        5. Check if policy says to retrain
        6. If retraining needed: reset and retrain on window data
        7. (Skipped) NO partial_fit -- model stays frozen between retrains

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features)
            y (np.ndarray): Labels array, shape (n_samples,)
        """
        # Initialize sliding window to accumulate data between retrains
        window_X, window_y = [], []

        # Initialize model with first sample so predictions can be made from t=0
        if len(X) > 0:
            x_init = X[0].reshape(1, -1)
            y_init = np.array([y[0]])
            self.model.partial_fit(x_init, y_init)

        # Process each sample sequentially (streaming)
        for t in range(len(X)):
            # Step 1: Extract single sample
            x_t = X[t].reshape(1, -1)
            y_t = np.array([y[t]])

            # Step 2 & 3: Predict and record accuracy
            if self.model.is_initialized:
                y_pred = self.model.predict(x_t)
                self.metrics.update(y_t, y_pred, t=t)

            # Step 4: Accumulate in sliding window
            window_X.append(x_t[0])
            window_y.append(y_t[0])

            # Step 5 & 6: Check retraining policy and retrain if needed
            if self.policy.should_retrain(t, self.metrics):
                self.model.retrain(np.array(window_X), np.array(window_y))
                self.policy.on_retrain(t)
                self.metrics.record_retrain(t, self.policy.retrain_latency, self.policy.deploy_latency)
                window_X, window_y = [], []

            # Step 7: SKIPPED -- no partial_fit, model is frozen between retrains

