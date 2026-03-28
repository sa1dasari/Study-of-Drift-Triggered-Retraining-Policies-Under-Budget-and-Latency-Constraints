# System Design: When to Retrain Simulator

## Overview

This document describes the architecture and design of the streaming ML simulator used to empirically compare retraining policies under concept drift, budget constraints, and deployment latency.

## Purpose

Enable reproducible comparison of three retraining policies (periodic, error-threshold, drift-triggered) across different concept drift scenarios, budget levels, and latency configurations.
The simulator processes 10,000-sample synthetic streams, applying a streaming SGD-based logistic regression model that adapts via online learning and policy-governed full retrains.

---

## Architecture

```
main.py  (entry point – configures & launches experiment runs)
│
├── src/data/drift_generator.py      – Synthetic data with concept drift
│       └── DriftGenerator            Logistic-regression data with weight-vector switching
│
├── src/models/base_model.py         – Online learning model
│       └── StreamingModel            SGDClassifier wrapper (partial_fit + retrain)
│
├── src/policies/                    – Retraining decision strategies
│       ├── base_policy.py            RetrainPolicy (abstract; budget + latency logic)
│       ├── periodic.py               PeriodicPolicy (fixed-interval retraining)
│       ├── error_threshold_policy.py ErrorThresholdPolicy (rolling error rate trigger)
│       └── drift_triggered_policy.py DriftTriggeredPolicy (ADWIN-based drift detection)
│
├── src/runner/experiment_runner.py  – Streaming event loop
│       └── ExperimentRunner          Processes samples sequentially; predict → evaluate → retrain? → partial_fit
│
├── src/evaluation/
│       ├── metrics.py                MetricsTracker (per-sample accuracy, retrain logs, drift segmentation)
│       ├── results_export.py         CSV / JSON export utilities
│       └── plot_results.py           Per-run timeline + rolling-accuracy plots
│
└── plot_summary.py                  – Cross-run 2×3 summary dashboard (all configs for a policy)
```

## Component Details

### Data Generation (`DriftGenerator`)

| Aspect | Detail |
|---|---|
| **Features** | 10 i.i.d. standard-normal features per sample |
| **Label model** | Bernoulli with probability `σ(X · w)` (logistic function) |
| **Concept switch** | Two random weight vectors `w₁`, `w₂` drawn once per seed |
| **Drift injection** | Weight vector changes at `drift_point = 5000` according to drift type |

### Streaming Model (`StreamingModel`)

- **Algorithm**: `sklearn.linear_model.SGDClassifier(loss="log_loss")` — stochastic gradient descent logistic regression.
- **Online update**: `partial_fit(x_t, y_t)` is called on every incoming sample for incremental learning.
- **Full retrain**: `retrain(window_X, window_y)` creates a fresh `SGDClassifier` and trains from scratch on the accumulated window. This happens only when the active policy triggers a retrain.

### Policy Framework (`RetrainPolicy` → subclasses)

All policies inherit from `RetrainPolicy`, which enforces:

1. **Budget constraint**: `remaining_budget` is decremented on each retrain; `should_retrain()` returns `False` when exhausted.
2. **Latency guard**: A retrain at timestep `t` blocks further retrains for `retrain_latency + deploy_latency` timesteps. During this window the model continues operating on stale weights.

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

### Metrics & Evaluation (`MetricsTracker`)

Key metrics tracked per experiment run:

| Metric | Definition |
|---|---|
| **Overall Accuracy** | Mean of per-sample correct predictions across all 10,000 timesteps |
| **Pre-Drift Accuracy** | Mean accuracy for `t ∈ [0, 5000)` |
| **Post-Drift Accuracy** | Mean accuracy for `t ∈ [5000, 10000)` |
| **Accuracy Drop** | `post_drift − pre_drift` (negative = degradation) |
| **Cumulative Error** | Total count of incorrect predictions |
| **Max Degradation** | Worst single-sample error |
| **Latency Cost** | Total timesteps spent in retrain + deploy windows |
| **Errors During Latency** | Errors accumulated while model is stale during retrain/deploy |
| **Budget Utilization** | `retrains_used / budget_total` |
| **Error in Drift Window** | Errors in the first 1,000 timesteps after `drift_point` |

### Export Pipeline

Each seed produces:
- `results/run_seed_{seed}.json` — full configuration + structured results
- `results/per_sample_metrics_seed_{seed}.csv` — per-timestep accuracy, error, latency flags
- `results/summary_results_{policy}_retrain_{N}seed.csv` — one summary row appended per run

Two sets of summary CSVs exist, corresponding to the two experiment phases:
- **Phase 1 (3-seed):** `summary_results_{policy}_retrain_3seed.csv` — 81 rows per policy (243 total)
- **Phase 2 (10-seed):** `summary_results_{policy}_retrain_10seed.csv` — 270 rows per policy (810 total)

---

## Reproducibility

- **Fixed seeds**: Phase 1 uses seeds `[42, 123, 456]` (3 seeds). Phase 2 uses seeds `[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]` (10 seeds) for stronger variance estimation.
- **Deterministic data**: `DriftGenerator` uses `np.random.default_rng(seed)` for full reproducibility.
- **Identical stream**: All three policies see the same data sequence for a given `(drift_type, seed)` pair, enabling fair comparison.

## Git Branching Strategy

The experiments were executed across multiple Git branches to organize results:

### Phase 1 — Per-Configuration Branches (243 runs, 3 seeds)

Each individual configuration was developed and tested in its own feature branch:
- **Branch naming:** `exp/<drift>-<policy>-<Budget>budget-<Latency>Latency`
- **Example:** `exp/gradual-drift-triggered-Medbudget-HighLatency`

Cumulative results per policy were merged into dedicated develop branches:
- `develop_3seed_periodic_retrain`
- `develop_3seed_error_threshold_retrain`
- `develop_3seed_drift_triggered_retrain`

### Phase 2 — Per-Policy Batch Branches (810 runs, 10 seeds)

The 810 runs were executed in 3 batches (270 runs per batch), one batch per policy:
- `develop-10Seed-periodic-retrain-tests`
- `develop-10Seed-error-threshold-retrain-tests`
- `develop-10Seed-drift-triggered-retrain-tests`

### Merged Results

The **`main`** and **`develop`** branches contain all results from both phases (1,053 total runs).

