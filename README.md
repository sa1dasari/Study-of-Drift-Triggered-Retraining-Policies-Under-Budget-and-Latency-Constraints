# Study of Drift-Triggered Retraining Policies Under Budget and Latency Constraints

## Overview

This repository contains a reproducible empirical systems study comparing three model retraining policies (periodic, error-threshold, drift-triggered) for streaming ML systems under concept drift, budget constraints, and deployment latency.

## Core Research Question

> *How do different model-refresh policies (periodic, error-threshold, drift-triggered) trade off accuracy, cost, and latency under concept drift?*

## Repository Structure

```
├── main.py                  # Entry point – configures and launches experiment runs
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
│   │   └── drift_triggered_policy.py   # ADWIN-based drift detection
│   ├── runner/
│   │   └── experiment_runner.py        # Streaming event loop
│   └── evaluation/
│       ├── metrics.py                  # MetricsTracker (per-sample + aggregate)
│       ├── results_export.py           # CSV / JSON export utilities
│       └── plot_results.py             # Per-run timeline + rolling-accuracy plots
└── results/
    ├── summary_results_periodic_retrain.csv
    ├── summary_results_error_threshold_retrain.csv
    ├── summary_results_drift_triggered_retrain.csv
    ├── summary_results_plot_periodic_retrain.png
    ├── summary_results_plot_error_threshold_retrain.png
    └── summary_results_plot_drift_triggered_retrain.png
```
---

## Experiment Design

### Factors & Levels

| Factor | Levels | Values |
|---|---|---|
| Drift type | 3 | Abrupt, Gradual, Recurring |
| Policy | 3 | Periodic, Error-Threshold, Drift-Triggered (ADWIN) |
| Budget (K) | 3 | 5 (low), 10 (medium), 20 (high) |
| Latency | 3 | Low (11 steps), Medium (105 steps), High (520 steps) |
| Seeds | 3 | 42, 123, 456 |

**Total: 3 × 3 × 3 × 3 = 27 configs per policy × 3 seeds = 81 runs per policy × 3 policies = 243 runs**

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
