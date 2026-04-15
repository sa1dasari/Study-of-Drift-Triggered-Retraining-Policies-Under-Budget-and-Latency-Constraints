# Study of Drift-Triggered Retraining Policies Under Budget and Latency Constraints

## Overview

A reproducible empirical study comparing three model retraining policies — periodic, error-threshold, and drift-triggered (ADWIN) — for streaming ML systems under concept drift, budget constraints, and deployment latency. A no-retrain baseline provides the accuracy floor.

Experiments are conducted in two modes:
- **With partial_fit (incremental learning):** The model receives `partial_fit` on every sample — 2,337 runs across synthetic, LUFlow, and LendingClub datasets.
- **Without partial_fit (static model):** The model is frozen between explicit retrains, isolating the pure policy effect — 1,596 runs across the same datasets.

**Grand total: 3,933 experiment runs.**

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

### Grand Total: **3,933 experiment runs** (2,337 with partial_fit + 1,596 without partial_fit)

---

## Repository Structure

```
├── experiments/                     # CLI entry points for all experiment runs
│   ├── main.py                      #   Synthetic experiments (with partial_fit)
│   ├── main_no_partial_fit.py       #   Synthetic experiments (NO partial_fit)
│   ├── luflow_main.py               #   LUFlow experiments (with partial_fit)
│   ├── luflow_main_no_partial_fit.py#   LUFlow experiments (NO partial_fit)
│   ├── lendingclub_main.py          #   LendingClub experiments (with partial_fit)
│   └── lendingclub_main_no_partial_fit.py # LendingClub experiments (NO partial_fit)
├── cross_policy_comparison.py       # Cross-policy head-to-head comparison (--dataset, --seeds)
├── statistical_significance_tests.py # Paired statistical significance tests (t-test, Wilcoxon, Holm-Bonferroni)
├── luflow_fitness_check.py          # LUFlow dataset suitability gate checks
├── lendingclub_fitness_check.py     # LendingClub dataset suitability gate checks
├── plot_summary.py                  # Per-policy dashboard PNG generator
├── docs/                            # All documentation
├── src/
│   ├── data/                        # DriftGenerator + LUFlow & LendingClub dataset loaders
│   ├── models/                      # StreamingModel (SGDClassifier wrapper)
│   ├── policies/                    # Periodic, ErrorThreshold, DriftTriggered, NeverRetrain
│   ├── runner/
│   │   ├── experiment_runner.py     #   Streaming event loop (with partial_fit)
│   │   └── experiment_runner_no_partial_fit.py # Streaming event loop (NO partial_fit)
│   └── evaluation/                  # MetricsTracker, CSV/JSON export, plots
├── results_with_retrain/            # Results from experiments WITH partial_fit
│   ├── synthetic/                   #   csv/, plots/, per_run/
│   ├── luflow/                      #   csv/, plots/, per_run/
│   ├── lendingclub/                 #   csv/, plots/, per_run/
│   ├── cross_policy_comparison/     #   Head-to-head cross-policy outputs
│   └── statistical_tests/           #   Paired significance test CSVs
├── results_combined_statistical_tests/ # Combined significance overview across both modes
└── results_without_retrain/         # Results from experiments WITHOUT partial_fit
    ├── synthetic/                   #   csv/, plots/, per_run/
    ├── luflow/                      #   csv/, plots/, per_run/
    ├── lendingclub/                 #   csv/, plots/, per_run/
    ├── cross_policy_comparison/     #   Head-to-head cross-policy outputs
    └── statistical_tests/           #   Paired significance test CSVs
```

---

## Quick Start

```bash
python -m venv .venv && .venv\Scripts\Activate.ps1   # Windows
pip install -r docs/requirements.txt

# With partial_fit (incremental learning)
python experiments/main.py                                        # synthetic experiments
python experiments/luflow_main.py                                 # LUFlow experiments
python experiments/lendingclub_main.py                            # LendingClub experiments

# Without partial_fit (static model)
python experiments/main_no_partial_fit.py                         # synthetic experiments
python experiments/luflow_main_no_partial_fit.py                  # LUFlow experiments
python experiments/lendingclub_main_no_partial_fit.py             # LendingClub experiments

# Cross-policy comparison (reads from results_without_retrain/ by default)
python cross_policy_comparison.py                                 # all datasets
```

See [setup_and_run_guide.md](docs/setup_and_run_guide.md) for full instructions.

---

## Cross-Policy Comparison

`cross_policy_comparison.py` merges all per-policy summary CSVs for a given dataset into a single DataFrame and produces four head-to-head comparison outputs per dataset, plus a cross-dataset summary when multiple datasets are available.

```bash
python cross_policy_comparison.py                      # all 3 datasets (auto-detects seed label)
python cross_policy_comparison.py --dataset synthetic   # synthetic only
python cross_policy_comparison.py --dataset luflow      # LUFlow only
python cross_policy_comparison.py --dataset lendingclub # LendingClub only
python cross_policy_comparison.py --seeds 3             # force 3-seed CSVs
```

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
| Phase 7 — Without partial_fit | `develop_no_partial_fit` | 1,596 |

All **summary CSVs and dashboard PNGs** are merged into the **`main`** branch. Per-run artifacts (JSON results, per-sample CSVs) remain in their respective experiment branches only. See [experiment_scope.md](docs/experiment_scope.md) for full details.

---

## Documentation

| Document | Purpose                                                                      |
|---|------------------------------------------------------------------------------|
| [setup_and_run_guide.md](docs/setup_and_run_guide.md) | Setup, running experiments                                                   |
| [design.md](docs/design.md) | System architecture and component details                                    |
| [drift_guide.md](docs/drift_guide.md) | Concept drift types and simulation mechanics                                 |
| [experiment_scope.md](docs/experiment_scope.md) | Full experiment scope — all phases, run counts, branches, output artifacts |
| [policies_guide.md](docs/policies_guide.md) | Policy algorithms, parameters, and trade-offs                                |
| [research_log.md](docs/research_log.md) | Week-by-week experiment log                                                  |
| [results_interpretation_guide.md](docs/results_interpretation_guide.md) | How to read the per-policy CSV and PNG files                                 |
| [cross_policy_comparison_guide.md](docs/cross_policy_comparison_guide.md) | How to read the cross-policy comparison outputs                              |
| [statistical_significance_guide.md](docs/statistical_significance_guide.md) | How to read the paired significance test outputs                             |
