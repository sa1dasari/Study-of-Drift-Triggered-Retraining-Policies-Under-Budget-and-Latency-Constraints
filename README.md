# Study of Drift-Triggered Retraining Policies Under Budget and Latency Constraints

## Overview

A reproducible empirical study comparing three model retraining policies — periodic, error-threshold, and drift-triggered (ADWIN) — for streaming ML systems under concept drift, budget constraints, and deployment latency. A no-retrain baseline provides the accuracy floor.

## Core Research Question

> *How do different model-refresh policies trade off accuracy, cost, and latency under concept drift?*

---

## Experiment Design

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
| **Total** | **1,833** | |

| Parameter | Value |
|---|---|
| Features | 10 (i.i.d. standard-normal) |
| Stream length | 10,000 samples |
| Drift point | t = 5,000 |
| Learner | SGDClassifier (log_loss) with per-sample partial_fit |

---

## Repository Structure

```
├── main.py                  # CLI entry point (--policy, --seeds)
├── plot_summary.py          # Dashboard PNG generator
├── docs/                    # All documentation
├── src/
│   ├── data/                # DriftGenerator
│   ├── models/              # StreamingModel (SGDClassifier wrapper)
│   ├── policies/            # Periodic, ErrorThreshold, DriftTriggered, NeverRetrain
│   ├── runner/              # ExperimentRunner (streaming event loop)
│   └── evaluation/          # MetricsTracker, CSV/JSON export, plots
└── results/                 # Summary CSVs and dashboard PNGs
```

---

## Quick Start

```bash
python -m venv .venv && .venv\Scripts\Activate.ps1   # Windows
pip install -r docs/requirements.txt
python main.py                                        # runs all policies, 10 seeds
```

See [setup_and_run_guide.md](docs/setup_and_run_guide.md) for full instructions.

---

## Documentation

| Document | Purpose |
|---|---|
| [setup_and_run_guide.md](docs/setup_and_run_guide.md) | **Start here** — setup, running experiments |
| [design.md](docs/design.md) | System architecture and component details |
| [drift_guide.md](docs/drift_guide.md) | Concept drift types and simulation mechanics |
| [experiment_scope.md](docs/experiment_scope.md) | Full experiment scope — all 3 phases, run counts, branches, output artifacts |
| [policies_guide.md](docs/policies_guide.md) | Policy algorithms, parameters, and trade-offs |
| [research_log.md](docs/research_log.md) | Week-by-week experiment log |
| [results_interpretation_guide.md](docs/results_interpretation_guide.md) | How to read the CSV and PNG files |
