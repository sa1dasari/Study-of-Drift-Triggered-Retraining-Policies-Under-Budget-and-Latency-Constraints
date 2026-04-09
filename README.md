# Study of Drift-Triggered Retraining Policies Under Budget and Latency Constraints

## Overview

A reproducible empirical study comparing three model retraining policies — periodic, error-threshold, and drift-triggered (ADWIN) — for streaming ML systems under concept drift, budget constraints, and deployment latency. A no-retrain baseline provides the accuracy floor.

Experiments are conducted on **synthetic data** (controlled weight-vector drift, 1,833 runs) and two **real-world datasets** — the **LUFlow Network Intrusion Detection dataset** (Lancaster University, 252 runs) and the **LendingClub Loan Default dataset** (Kaggle, 252 runs) — for a combined total of **2,337 experiment runs**.

## Core Research Question

> *How do different model-refresh policies trade off accuracy, cost, and latency under concept drift?*

---

## Experiment Design

### Synthetic Data

| Factor | Levels |
|---|---|
| Drift type | Abrupt, Gradual, Recurring |
| Policy | Periodic, Error-Threshold, Drift-Triggered (ADWIN), No-Retrain (baseline) |
| Budget (K) | 5, 10, 20 |
| Latency (Phase 1/2) | Low (11), Medium (105), High (520) |
| Latency (Phase 3) | Near-Zero (3), Extreme-High (2050) |
| Seeds | 3 (Phase 1) or 10 (Phase 2/3) |

| Phase | Runs | Description |
|---|---|---|
| Phase 1 | 243 | 3 policies × 3 drifts × 3 budgets × 3 latencies × 3 seeds |
| Phase 2 | 810 | Same grid × 10 seeds |
| Baseline | 39 | No-retrain × 3 drifts × (3 + 10) seeds |
| Phase 3 | 741 | 2 extreme latencies × 3 drifts × 3 budgets × (3 + 10) seeds + baseline |
| **Synthetic Total** | **1,833** | |

| Parameter | Value |
|---|---|
| Features | 10 (i.i.d. standard-normal) |
| Stream length | 10,000 samples |
| Drift point | t = 5,000 |
| Learner | SGDClassifier (log_loss) with per-sample partial_fit |

### Real-World Data — LUFlow Network Intrusion Detection

| Factor | Levels |
|---|---|
| Drift type | Abrupt, Gradual, Recurring |
| Policy | Periodic, Error-Threshold, Drift-Triggered (ADWIN), No-Retrain (baseline) |
| Budget (K) | 5, 10, 20 |
| Latency | Low (11), Medium (105), High (520) |
| Pool configs | 3 (class-balance shift, feature drift, extreme shift) |

| Phase | Runs | Description |
|---|---|---|
| Phase 4 | 252 | 3 pools × 3 drifts × 3 budgets × 3 latencies × 3 policies + 9 baseline |

| Parameter | Value |
|---|---|
| Features | 11 flow-level (avg_ipt, bytes_in/out, ports, entropy, etc.) |
| Stream length | 50,000 samples |
| Drift point | t = 25,000 |
| Dataset | LUFlow (Lancaster University) — 28 day-CSVs, ~21 M rows |

### Real-World Data — LendingClub Loan Default

| Factor | Levels |
|---|---|
| Drift type | Abrupt, Gradual, Recurring |
| Policy | Periodic, Error-Threshold, Drift-Triggered (ADWIN), No-Retrain (baseline) |
| Budget (K) | 5, 10, 20 |
| Latency | Low (11), Medium (105), High (520) |
| Seed configs | 3 (year-pair configurations) |

| Phase | Runs | Description |
|---|---|---|
| Phase 5 | 252 | 3 seeds × 3 drifts × 3 budgets × 3 latencies × 3 policies + 9 baseline |

| Parameter | Value |
|---|---|
| Features | 16 origination-time (loan_amnt, int_rate, FICO, DTI, etc.) → 34 after one-hot encoding |
| Stream length | 50,000 samples |
| Drift point | t = 25,000 |
| Dataset | LendingClub (Kaggle) — accepted loans 2007–2018, ~1.35 M rows after filtering |
| Drift source | Real-world feature-space drift from underwriting policy changes (2012–2016) |

### Grand Total: **2,337 experiment runs**

---

## Repository Structure

```
├── main.py                  # CLI entry point — synthetic experiments (--policy, --seeds)
├── luflow_main.py           # CLI entry point — LUFlow real-world experiments (--policy)
├── lendingclub_main.py      # CLI entry point — LendingClub real-world experiments (--policy)
├── luflow_fitness_check.py  # LUFlow dataset suitability gate checks
├── lendingclub_fitness_check.py # LendingClub dataset suitability gate checks
├── plot_summary.py          # Dashboard PNG generator
├── docs/                    # All documentation
├── src/
│   ├── data/                # DriftGenerator + LUFlow & LendingClub dataset loaders
│   ├── models/              # StreamingModel (SGDClassifier wrapper)
│   ├── policies/            # Periodic, ErrorThreshold, DriftTriggered, NeverRetrain
│   ├── runner/              # ExperimentRunner (streaming event loop)
│   └── evaluation/          # MetricsTracker, CSV/JSON export, plots
└── results/                 # Organised output directory
    ├── synthetic/           #   Synthetic experiment results
    │   ├── csv/             #     Summary CSVs
    │   ├── plots/           #     Dashboard PNGs
    │   └── per_run/         #     Per-run JSONs & per-sample CSVs
    ├── luflow/              #   LUFlow real-world experiment results
    │   ├── csv/             #     Summary CSVs
    │   ├── plots/           #     Dashboard PNGs
    │   └── per_run/         #     Per-run JSONs & per-sample CSVs
    └── lendingclub/         #   LendingClub real-world experiment results
        ├── csv/             #     Summary CSVs
        ├── plots/           #     Dashboard PNGs
        └── per_run/         #     Per-run JSONs & per-sample CSVs
```

---

## Quick Start

```bash
python -m venv .venv && .venv\Scripts\Activate.ps1   # Windows
pip install -r docs/requirements.txt
python main.py                                        # synthetic experiments (all policies, 10 seeds)
python luflow_main.py                                 # LUFlow experiments (all policies, 252 runs)
python lendingclub_main.py                            # LendingClub experiments (all policies, 252 runs)
```

See [setup_and_run_guide.md](docs/setup_and_run_guide.md) for full instructions.

---

## Git Branching Strategy

| Phase | Branch(es) | Runs |
|---|---|---|
| Phase 1 (3 seeds) | `exp/<drift>-<policy>-<Budget>budget-<Latency>Latency` → `develop_3seed_<policy>_retrain` | 243 |
| Phase 2 (10 seeds) | `develop-10Seed-<policy>-retrain-tests` | 810 |
| Baseline | `develop_NoRetrain_NoBudget_NoLatency` | 39 |
| Phase 3 — Extreme Latency | `develop_ExtremeLatencyLevels` | 741 |
| Phase 4 — LUFlow Dataset | `develop_LUFlow_Dataset` | 252 |
| Phase 5 — LendingClub Dataset | `develop_LendingClub_Data` | 252 |

All **summary CSVs and dashboard PNGs** are merged into the **`main`** branch. Per-run artifacts (JSON results, per-sample CSVs) remain in their respective experiment branches only. See [experiment_scope.md](docs/experiment_scope.md) for full details.

---

## Documentation

| Document | Purpose                                                                      |
|---|------------------------------------------------------------------------------|
| [setup_and_run_guide.md](docs/setup_and_run_guide.md) | Setup, running experiments                                                   |
| [design.md](docs/design.md) | System architecture and component details                                    |
| [drift_guide.md](docs/drift_guide.md) | Concept drift types and simulation mechanics                                 |
| [experiment_scope.md](docs/experiment_scope.md) | Full experiment scope — all 3 phases, run counts, branches, output artifacts |
| [policies_guide.md](docs/policies_guide.md) | Policy algorithms, parameters, and trade-offs                                |
| [research_log.md](docs/research_log.md) | Week-by-week experiment log                                                  |
| [results_interpretation_guide.md](docs/results_interpretation_guide.md) | How to read the CSV and PNG files                                            |
