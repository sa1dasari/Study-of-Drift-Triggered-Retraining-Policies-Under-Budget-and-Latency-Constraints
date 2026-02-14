## Three retraining policies that are being compared

1. **Periodic Retraining**: Retrain the model at fixed time intervals (e.g., every 24 hours) regardless of observed performance or drift.
2. **Error-Threshold Retraining**: Retrain the model when the error rate on a validation set exceeds a predefined threshold (e.g., 10%).
3. **Drift-Triggered Retraining**: Retrain the model when a concept drift is detected using a drift detection method (e.g., ADWIN, DDM).