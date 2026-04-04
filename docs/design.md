# System Design

## Overview

This document describes the architecture of the streaming ML simulator used to compare retraining policies under concept drift, budget constraints, and deployment latency. Experiments are conducted on both **synthetic data** (controlled weight-vector drift) and a **real-world dataset** (LUFlow network intrusion detection), sharing the same model, policy, runner, and evaluation components.

---

## Architecture

```
main.py        (CLI entry point – synthetic data full-factorial sweep)
luflow_main.py (CLI entry point – LUFlow real-world data full-factorial sweep)
│
├── src/data/drift_generator.py      – Synthetic data with concept drift
│       └── DriftGenerator            Logistic-regression data with weight-vector switching
│
├── src/data/LUFlow_Network_Intrusion/
│       └── datasets/                 28 day-CSVs (downloaded separately; ~1.5 GB+)
│
├── src/models/base_model.py         – Online learning model
│       └── StreamingModel            SGDClassifier wrapper (partial_fit + retrain)
│
├── src/policies/                    – Retraining decision strategies
│       ├── base_policy.py            RetrainPolicy (abstract; budget + latency logic)
│       ├── periodic.py               PeriodicPolicy (fixed-interval retraining)
│       ├── error_threshold_policy.py ErrorThresholdPolicy (rolling error rate trigger)
│       ├── drift_triggered_policy.py DriftTriggeredPolicy (ADWIN-based drift detection)
│       └── never_retrain_policy.py   NeverRetrainPolicy (baseline — partial_fit only)
│
├── src/runner/experiment_runner.py  – Streaming event loop
│       └── ExperimentRunner          Processes samples sequentially
│
├── src/evaluation/
│       ├── metrics.py                MetricsTracker (per-sample accuracy, retrain logs)
│       ├── results_export.py         CSV / JSON export utilities
│       └── plot_results.py           Per-run timeline + rolling-accuracy plots
│
├── luflow_fitness_check.py          – LUFlow dataset suitability gate checks
│
└── plot_summary.py                  – Cross-run 2×3 summary dashboard
```

---

## Component Details

### Data Generation (`DriftGenerator`)

| Aspect | Detail |
|---|---|
| **Features** | 10 i.i.d. standard-normal features per sample |
| **Label model** | Bernoulli with probability `σ(X · w)` (logistic function) |
| **Concept switch** | Two random weight vectors `w₁`, `w₂` drawn once per seed |
| **Drift injection** | Weight vector changes at `drift_point = 5000` according to drift type |

### Real-World Data — LUFlow Network Intrusion Detection

| Aspect | Detail |
|---|---|
| **Dataset** | LUFlow (Lancaster University), 28 day-CSVs, ~21 M rows |
| **Features** | 11 flow-level features (avg_ipt, bytes_in/out, ports, entropy, etc.) |
| **Labels** | Binary: benign vs malicious |
| **Pool configs** | 3 seed configurations select pre-/post-drift pools by filtering days on malicious-class percentage |
| **Stream** | 50,000 samples per run, drift injected at t = 25,000 |
| **Drift injection** | Abrupt (hard switch), Gradual (5,000-step blend), Recurring (alternates every 5,000 steps) |
| **Standardisation** | `StandardScaler` applied after stream assembly |

The LUFlow experiments use the same `ExperimentRunner`, policies, and metrics as the synthetic runs. Data loading and stream construction are handled by `luflow_main.py`. A three-gate fitness check (`luflow_fitness_check.py`) validated the dataset before the full sweep.

### Streaming Model (`StreamingModel`)

- **Algorithm**: `sklearn.linear_model.SGDClassifier(loss="log_loss")` — stochastic gradient descent logistic regression.
- **Online update**: `partial_fit(x_t, y_t)` is called on every incoming sample for incremental learning.
- **Full retrain**: `retrain(window_X, window_y)` creates a fresh `SGDClassifier` and trains from scratch on the accumulated window. This happens only when the active policy triggers a retrain.

### Policy Framework (`RetrainPolicy` → subclasses)

All policies inherit from `RetrainPolicy`, which enforces:

1. **Budget constraint**: `remaining_budget` is decremented on each retrain; `should_retrain()` returns `False` when exhausted.
2. **Latency guard**: A retrain at timestep `t` blocks further retrains for `retrain_latency + deploy_latency` timesteps. During this window the model operates on stale weights.

### Experiment Runner (`ExperimentRunner`)

The streaming event loop for each timestep `t`:

```
for t in 0 … N-1:
    predict(x_t)  →  record accuracy  →  buffer x_t
    if policy.should_retrain(t, metrics):
        model.retrain(buffer)
        policy.on_retrain(t)
        metrics.record_retrain(t, ...)
        clear buffer
    model.partial_fit(x_t, y_t)   # always do incremental update
```

### Metrics (`MetricsTracker`)

| Metric | Definition |
|---|---|
| **Overall Accuracy** | Mean correct predictions across all timesteps (10,000 synthetic; 50,000 LUFlow) |
| **Pre-Drift Accuracy** | Mean accuracy for `t ∈ [0, drift_point)` |
| **Post-Drift Accuracy** | Mean accuracy for `t ∈ [drift_point, N)` |
| **Accuracy Drop** | `post_drift − pre_drift` (negative = degradation) |
| **Budget Utilization** | `retrains_used / budget_total` |
| **Retrains Before/After Drift** | Count of retrains in each half of the stream |

### Export Pipeline

Each run produces:
- **JSON** — full configuration + structured results
- **Per-sample CSV** — per-timestep accuracy, error, latency flags
- **Summary CSV row** — one row appended to the policy's summary file

---

## Reproducibility

- **Fixed seeds**: `DriftGenerator` uses `np.random.default_rng(seed)` for full determinism. LUFlow stream construction uses a fixed RNG seed for the gradual-drift blending.
- **Identical stream**: All policies see the same data sequence for a given `(drift_type, seed)` pair (synthetic) or `(drift_type, pool_config)` pair (LUFlow), enabling fair comparison.
- **CLI-driven**: `python main.py --policy <name> --seeds <N>` reproduces any synthetic experiment subset. `python luflow_main.py --policy <name>` reproduces any LUFlow experiment subset.
