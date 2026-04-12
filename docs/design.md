# System Design

## Overview

This document describes the architecture of the streaming ML simulator used to compare retraining policies under concept drift, budget constraints, and deployment latency. Experiments are conducted on **synthetic data** (controlled weight-vector drift) and two **real-world datasets** — the **LUFlow network intrusion detection dataset** and the **LendingClub loan default dataset** — sharing the same model, policy, runner, and evaluation components.

Two experiment modes are supported:
- **With partial_fit (incremental learning):** The model receives a `partial_fit` update on every incoming sample, providing continuous incremental learning between retrains. This is the original experiment mode.
- **Without partial_fit (static model):** The model is frozen between explicit retrains. This isolates the effect of the retraining *policy* from any benefit provided by continuous incremental learning.

---

## Architecture

```
experiments/                            (CLI entry points for all experiment runs)
├── main.py                             (Synthetic data — with partial_fit)
├── main_no_partial_fit.py              (Synthetic data — NO partial_fit, static model)
├── luflow_main.py                      (LUFlow real-world data — with partial_fit)
├── luflow_main_no_partial_fit.py       (LUFlow real-world data — NO partial_fit)
├── lendingclub_main.py                 (LendingClub real-world data — with partial_fit)
└── lendingclub_main_no_partial_fit.py  (LendingClub real-world data — NO partial_fit)
│
├── src/data/synthetic_data_drift_generator.py      – Synthetic data with concept drift
│       └── DriftGenerator                          Logistic-regression data with weight-vector switching
│
├── src/data/LUFlow_Network_Intrusion/
│       └── datasets/                 28 day-CSVs (downloaded separately; ~1.5 GB+)
│
├── src/data/LendingClub_Loan_Data/
│       ├── lendingclub_loader.py     Loader & preprocessor for LendingClub CSV
│       └── datasets/                 accepted_2007_to_2018Q4.csv (downloaded separately; ~1.6 GB)
│
├── src/models/base_model.py         – Online learning model
│       └── StreamingModel            SGDClassifier wrapper (partial_fit + retrain)
│
├── src/policies/                    – Retraining decision strategies
│       ├── base_policy.py            RetrainPolicy (abstract; budget + latency logic)
│       ├── periodic.py               PeriodicPolicy (fixed-interval retraining)
│       ├── error_threshold_policy.py ErrorThresholdPolicy (rolling error rate trigger)
│       ├── drift_triggered_policy.py DriftTriggeredPolicy (ADWIN-based drift detection)
│       └── never_retrain_policy.py   NeverRetrainPolicy (baseline — no retraining)
│
├── src/runner/
│       ├── experiment_runner.py              – Streaming event loop (with partial_fit)
│       │       └── ExperimentRunner           Processes samples with incremental learning
│       └── experiment_runner_no_partial_fit.py – Streaming event loop (NO partial_fit)
│               └── ExperimentRunnerNoPartialFit  Static model between retrains
│
├── src/evaluation/
│       ├── metrics.py                MetricsTracker (per-sample accuracy, retrain logs)
│       ├── results_export.py         CSV / JSON export utilities
│       └── plot_results.py           Per-run timeline + rolling-accuracy plots
│
├── cross_policy_comparison.py       – Cross-policy head-to-head comparison (reads from results_without_retrain/)
├── luflow_fitness_check.py          – LUFlow dataset suitability gate checks
├── lendingclub_fitness_check.py     – LendingClub dataset suitability gate checks
├── plot_summary.py                  – Cross-run 2×3 summary dashboard
│
├── results_with_retrain/            – Results from experiments WITH partial_fit (incremental learning)
│       ├── synthetic/               csv/, plots/, per_run/
│       ├── luflow/                  csv/, plots/, per_run/
│       ├── lendingclub/             csv/, plots/, per_run/
│       └── cross_policy_comparison/ Per-dataset & cross-dataset comparison outputs
│
└── results_without_retrain/         – Results from experiments WITHOUT partial_fit (static model)
        ├── synthetic/               csv/, plots/, per_run/
        ├── luflow/                  csv/, plots/, per_run/
        ├── lendingclub/             csv/, plots/, per_run/
        └── cross_policy_comparison/ Per-dataset & cross-dataset comparison outputs
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

### Real-World Data — LendingClub Loan Default

| Aspect | Detail |
|---|---|
| **Dataset** | LendingClub (Kaggle), accepted loans 2007–2018, ~1.35 M rows after filtering |
| **Features** | 16 origination-time features (loan_amnt, int_rate, FICO, DTI, etc.) → 34 after one-hot encoding |
| **Labels** | Binary: Fully Paid (0) vs Charged Off (1) |
| **Seed configs** | 3 year-pair configurations select pre-/post-drift pools from different calendar years |
| **Stream** | 50,000 samples per run, drift injected at t = 25,000 |
| **Drift source** | Real-world feature-space drift from LendingClub underwriting policy changes (2012–2016) |
| **Drift injection** | Abrupt (hard switch), Gradual (5,000-step blend), Recurring (alternates every 5,000 steps) |
| **Standardisation** | `StandardScaler` applied after stream assembly |

The LendingClub experiments use the same `ExperimentRunner`, policies, and metrics as the synthetic and LUFlow runs. Data loading and stream construction are handled by `lendingclub_main.py` via `lendingclub_loader.py`. A three-gate fitness check (`lendingclub_fitness_check.py`) validated the dataset before the full sweep.

### Streaming Model (`StreamingModel`)

- **Algorithm**: `sklearn.linear_model.SGDClassifier(loss="log_loss")` — stochastic gradient descent logistic regression.
- **Online update**: `partial_fit(x_t, y_t)` is called on every incoming sample for incremental learning (with-partial-fit mode only).
- **Full retrain**: `retrain(window_X, window_y)` creates a fresh `SGDClassifier` and trains from scratch on the accumulated window. This happens only when the active policy triggers a retrain.

### Policy Framework (`RetrainPolicy` → subclasses)

All policies inherit from `RetrainPolicy`, which enforces:

1. **Budget constraint**: `remaining_budget` is decremented on each retrain; `should_retrain()` returns `False` when exhausted.
2. **Latency guard**: A retrain at timestep `t` blocks further retrains for `retrain_latency + deploy_latency` timesteps. During this window the model operates on stale weights.

### Experiment Runners

Two runner variants share the same interface but differ in incremental learning:

#### `ExperimentRunner` (with partial_fit)

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

#### `ExperimentRunnerNoPartialFit` (static model)

```
for t in 0 … N-1:
    predict(x_t)  →  record accuracy  →  buffer x_t
    if policy.should_retrain(t, metrics):
        model.retrain(buffer)
        policy.on_retrain(t)
        metrics.record_retrain(t, ...)
        clear buffer
    # NO partial_fit — model is frozen between retrains
```

The no-partial-fit variant isolates the pure effect of the retraining policy from any benefit provided by continuous incremental learning. Results from both modes are stored in separate directories (`results_with_retrain/` and `results_without_retrain/`) so they never overwrite each other.

### Metrics (`MetricsTracker`)

| Metric | Definition |
|---|---|
| **Overall Accuracy** | Mean correct predictions across all timesteps (10,000 synthetic; 50,000 LUFlow/LendingClub) |
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

## Results Directory Structure

Results are organized into two top-level directories based on the experiment mode:

```
results_with_retrain/          ← Experiments WITH partial_fit (incremental learning)
├── synthetic/
│   ├── csv/                   Summary CSVs (one row per run)
│   ├── plots/                 Dashboard PNGs (one per policy × seed-label)
│   └── per_run/               Per-run JSONs and per-sample CSVs
├── luflow/
│   ├── csv/
│   ├── plots/
│   └── per_run/
├── lendingclub/
│   ├── csv/
│   ├── plots/
│   └── per_run/
└── cross_policy_comparison/
    ├── synthetic/             Per-dataset comparison tables & figures
    ├── luflow/
    ├── lendingclub/
    ├── fig_cross_dataset_summary.png
    └── table_cross_dataset_summary.csv

results_without_retrain/       ← Experiments WITHOUT partial_fit (static model)
├── synthetic/
│   ├── csv/
│   ├── plots/
│   └── per_run/
├── luflow/
│   ├── csv/
│   ├── plots/
│   └── per_run/
├── lendingclub/
│   ├── csv/
│   ├── plots/
│   └── per_run/
└── cross_policy_comparison/
    ├── synthetic/
    ├── luflow/
    ├── lendingclub/
    ├── fig_cross_dataset_summary.png
    └── table_cross_dataset_summary.csv
```

---

## Reproducibility

- **Fixed seeds**: `DriftGenerator` uses `np.random.default_rng(seed)` for full determinism. LUFlow and LendingClub stream construction uses a fixed RNG seed for the gradual-drift blending.
- **Identical stream**: All policies see the same data sequence for a given `(drift_type, seed)` pair (synthetic) or `(drift_type, pool_config)` pair (LUFlow/LendingClub), enabling fair comparison.
- **CLI-driven**: All experiment scripts live in the `experiments/` directory.
  - `python experiments/main.py --policy <name> --seeds <N>` reproduces any synthetic experiment (with partial_fit).
  - `python experiments/main_no_partial_fit.py --policy <name> --seeds <N>` reproduces any synthetic experiment (without partial_fit).
  - `python experiments/luflow_main.py --policy <name>` reproduces any LUFlow experiment (with partial_fit).
  - `python experiments/luflow_main_no_partial_fit.py --policy <name>` reproduces any LUFlow experiment (without partial_fit).
  - `python experiments/lendingclub_main.py --policy <name>` reproduces any LendingClub experiment (with partial_fit).
  - `python experiments/lendingclub_main_no_partial_fit.py --policy <name>` reproduces any LendingClub experiment (without partial_fit).
