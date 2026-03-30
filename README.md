# Study of Drift-Triggered Retraining Policies Under Budget and Latency Constraints

## Overview

This repository contains a reproducible empirical systems study comparing three model retraining policies (periodic, error-threshold, drift-triggered) for streaming ML systems under concept drift, budget constraints, and deployment latency.

## Core Research Question

> *How do different model-refresh policies (periodic, error-threshold, drift-triggered) trade off accuracy, cost, and latency under concept drift?*

## Repository Structure

```
├── main.py                  # CLI entry point – full-factorial sweep (--policy, --seeds)
├── plot_summary.py          # Generates 2×3 summary dashboard from summary CSVs
├── docs/
│   ├── setup_and_run_guide.md   # ★ Start here — setup, running, reproducing experiments
│   ├── design.md            # System architecture and component details
│   ├── drift_guide.md       # Concept drift types and simulation mechanics
│   ├── experiment_scope.md  # Full factorial design and configuration matrix
│   ├── policies_guide.md    # Policy algorithms, parameters, and trade-offs
│   ├── research_log.md      # Week-by-week experiment log with observations
│   ├── results_interpretation_guide.md  # How to read CSVs and dashboard plots
│   └── requirements.txt     # Python dependencies
├── src/
│   ├── data/
│   │   └── drift_generator.py          # Synthetic stream with weight-vector drift
│   ├── models/
│   │   └── base_model.py               # SGDClassifier wrapper (partial_fit + retrain)
│   ├── policies/
│   │   ├── base_policy.py              # Abstract policy (budget + latency guard)
│   │   ├── periodic.py                 # Fixed-interval retraining
│   │   ├── error_threshold_policy.py   # Rolling error-rate trigger
│   │   ├── drift_triggered_policy.py   # ADWIN-based drift detection
│   │   └── never_retrain_policy.py     # No-retrain baseline (partial_fit only)
│   ├── runner/
│   │   └── experiment_runner.py        # Streaming event loop
│   └── evaluation/
│       ├── metrics.py                  # MetricsTracker (per-sample + aggregate)
│       ├── results_export.py           # CSV / JSON export utilities
│       └── plot_results.py             # Per-run timeline + rolling-accuracy plots
└── results/
    ├── summary_results_periodic_retrain_3seed.csv
    ├── summary_results_periodic_retrain_10seed.csv
    ├── summary_results_error_threshold_retrain_3seed.csv
    ├── summary_results_error_threshold_retrain_10seed.csv
    ├── summary_results_drift_triggered_retrain_3seed.csv
    ├── summary_results_drift_triggered_retrain_10seed.csv
    ├── summary_results_no_retrain_3seed.csv
    ├── summary_results_no_retrain_10seed.csv
    ├── summary_results_plot_periodic_retrain_3seed.png
    ├── summary_results_plot_periodic_retrain_10seed.png
    ├── summary_results_plot_error_threshold_retrain_3seed.png
    ├── summary_results_plot_error_threshold_retrain_10seed.png
    ├── summary_results_plot_drift_triggered_retrain_3seed.png
    ├── summary_results_plot_drift_triggered_retrain_10seed.png
    ├── summary_results_plot_no_retrain_3seed.png
    └── summary_results_plot_no_retrain_10seed.png
```
---

## Experiment Design

### Factors & Levels

| Factor | Levels | Values |
|---|---|---|
| Drift type | 3 | Abrupt, Gradual, Recurring |
| Policy | 4 | Periodic, Error-Threshold, Drift-Triggered (ADWIN), **No-Retrain (baseline)** |
| Budget (K) | 3 | 5 (low), 10 (medium), 20 (high) — *N/A for No-Retrain* |
| Latency | 3 | Low (11 steps), Medium (105 steps), High (520 steps) — *N/A for No-Retrain* |
| Seeds (Phase 1) | 3 | 42, 123, 456 |
| Seeds (Phase 2) | 10 | 42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021 |

### Phase 1 — Initial Runs (243 experiments, 3 seeds)

**Total: 3 drift types × 3 policies × 3 budgets × 3 latencies × 3 seeds = 243 runs**

Each individual configuration was developed and tested in its own feature branch:
- Branch naming: `exp/<drift>-<policy>-<Budget>budget-<Latency>Latency`
- Example: `exp/gradual-drift-triggered-Medbudget-HighLatency`

Cumulative results for each policy (81 runs per policy) were aggregated in dedicated develop branches:
- `develop_3seed_periodic_retrain`
- `develop_3seed_error_threshold_retrain`
- `develop_3seed_drift_triggered_retrain`

### Phase 2 — Extended Runs (810 experiments, 10 seeds)

**Total: 3 drift types × 3 policies × 3 budgets × 3 latencies × 10 seeds = 810 runs**

These runs were executed in 3 batches of 270 runs each (one batch per policy), rather than per-configuration branches:
- `develop-10Seed-periodic-retrain-tests` (270 runs)
- `develop-10Seed-error-threshold-retrain-tests` (270 runs)
- `develop-10Seed-drift-triggered-retrain-tests` (270 runs)

### No-Retrain Baseline (39 experiments, 3 + 10 seeds)

**Total: 3 drift types × 3 seeds = 9 runs (Phase 1) + 3 drift types × 10 seeds = 30 runs (Phase 2) = 39 runs** — no budget/latency grid (always 0).

The model relies solely on incremental `partial_fit` with zero full retrains. This provides the **accuracy floor** against which all other policies are compared.

- **Branch:** `develop_NoRetrain_NoBudget_NoLatency`

**10-seed results:**

| Drift Type | Overall Accuracy (mean ± std) | Pre-Drift | Post-Drift | Accuracy Drop |
|---|---|---|---|---|
| Abrupt | 0.7777 ± 0.0321 | 0.7774 | 0.7779 | +0.0005 |
| Gradual | 0.7753 ± 0.0310 | 0.7774 | 0.7731 | −0.0044 |
| Recurring | 0.7736 ± 0.0289 | 0.7774 | 0.7697 | −0.0077 |

### Combined Results

The **`main`** and **`develop`** branches contain the merged results from all phases: **1,092 total experiment runs** (243 + 810 + 39). All result CSVs and dashboard plots are in the `results/` folder.

### Shared Parameters

| Parameter | Value |
|---|---|
| Features | 10 (i.i.d. standard-normal) |
| Stream length | 10,000 samples |
| Drift point | t = 5,000 |
| Label model | Bernoulli(σ(X · w)) |
| Learner | SGDClassifier (log_loss) with per-sample partial_fit |

## Documentation

| Document | Description |
|---|---|
| [setup_and_run_guide.md](docs/setup_and_run_guide.md) | **Start here** — environment setup, running experiments, full-factorial reproduction |
| [design.md](docs/design.md) | System architecture, component details, event loop, metrics |
| [drift_guide.md](docs/drift_guide.md) | Drift types, mechanics, parameters, impact on results |
| [experiment_scope.md](docs/experiment_scope.md) | Full configuration matrix, output artifacts, CSV schema |
| [policies_guide.md](docs/policies_guide.md) | Policy algorithms, calibration, strengths/weaknesses, comparison |
| [research_log.md](docs/research_log.md) | Week-by-week experiment log with quantitative observations |
| [results_interpretation_guide.md](docs/results_interpretation_guide.md) | How to read the summary CSV files and dashboard plots |
