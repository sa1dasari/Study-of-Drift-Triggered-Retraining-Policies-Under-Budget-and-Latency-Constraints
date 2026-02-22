"""
Data generation module for creating synthetic streaming data with concept drift.

Concept drift occurs when the statistical properties of the target variable change
over time.
"""

import numpy as np

class DriftGenerator:
    """
    Generates synthetic binary classification data with concept drift.

    The data is generated using a logistic regression model where:
    - Features (X) are sampled from a standard normal distribution
    - Labels (y) are sampled from a Bernoulli distribution with probability
      determined by the logistic function: P(y=1|x) = 1 / (1 + exp(-x*w))
    - Concept drift is introduced by switching weight vectors (w) at a drift point

    Attributes:
        n_features (int): Number of input features
        drift_type (str): Type of drift - "abrupt", "gradual", or "recurring"
        drift_point (int): Timestep at which drift is introduced
        recurrence_period (int): Period for recurring drift (timesteps between concept switches)
        rng (np.random.Generator): Random number generator with fixed seed
        weights_1 (np.ndarray): Initial weight vector before drift
        weights_2 (np.ndarray): New weight vector after drift
    """

    def __init__(
            self,
            n_features=10,
            drift_type="gradual",
            drift_point=5000,
            recurrence_period=2000,
            seed=42
    ):
        self.n_features = n_features
        self.drift_type = drift_type
        self.drift_point = drift_point
        self.recurrence_period = recurrence_period
        self.rng = np.random.default_rng(seed)

        # Initial concept: random weight vector before drift
        self.weights_1 = self.rng.normal(size=n_features)
        # Drifted concept: random weight vector after drift
        self.weights_2 = self.rng.normal(size=n_features)

    def _get_weights(self, t):
        """
        Get the weight vector at timestep t based on drift type.

        Args:
            t (int): Current timestep

        Returns:
            np.ndarray: Weight vector for the current timestep
        """
        if self.drift_type == "abrupt":
            # Sudden switch from weights_1 to weights_2 at drift_point
            return self.weights_1 if t < self.drift_point else self.weights_2

        elif self.drift_type == "gradual":
            # Smooth transition from weights_1 to weights_2 over 1000 timesteps
            # alpha goes from 0 to 1, linearly interpolating between the two weight vectors
            alpha = min(1, max(0, (t - self.drift_point) / 1000))
            return (1 - alpha) * self.weights_1 + alpha * self.weights_2

        elif self.drift_type == "recurring":
            # Periodic switching between concepts after drift_point
            # The concept alternates between weights_1 and weights_2 every recurrence_period timesteps
            if t < self.drift_point:
                return self.weights_1
            else:
                # Calculate which period we're in after drift starts
                periods_elapsed = (t - self.drift_point) // self.recurrence_period
                # Even periods use weights_2, odd periods return to weights_1
                if periods_elapsed % 2 == 0:
                    return self.weights_2
                else:
                    return self.weights_1

        else:
            # Default: no drift, always use initial weights
            return self.weights_1

    def generate(self, n_samples=10000):
        """
        Generate synthetic data with concept drift.

        For each timestep t:
        1. Get the appropriate weight vector for this timestep
        2. Calculate logit = X[t] @ w (linear combination)
        3. Transform to probability using logistic function
        4. Sample binary label from Bernoulli distribution

        Args:
            n_samples (int): Number of samples to generate (default: 10000)

        Returns:
            tuple: (X, y) where
                X (np.ndarray): Shape (n_samples, n_features) - feature matrix
                y (np.ndarray): Shape (n_samples,) - binary labels {0, 1}
        """
        # Generate all features at once: shape (n_samples, n_features)
        X = self.rng.normal(size=(n_samples, self.n_features))
        y = []

        for t in range(n_samples):
            w = self._get_weights(t)

            # Linear model: logit = X[t] · w
            logit = X[t] @ w

            # Apply logistic transformation to get probability
            prob = 1 / (1 + np.exp(-logit))

            # Sample label from Bernoulli distribution with this probability
            label = self.rng.binomial(1, prob)
            y.append(label)

        return X, np.array(y)
