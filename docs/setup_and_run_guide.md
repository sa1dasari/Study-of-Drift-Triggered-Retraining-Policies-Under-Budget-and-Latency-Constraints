# Setup & Run Guide

A step-by-step guide for anyone who has freshly cloned this repository and wants to reproduce the experiments or run new configurations.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone & Install](#2-clone--install)
3. [Project Layout at a Glance](#3-project-layout-at-a-glance)
4. [Running Experiments](#4-running-experiments)
5. [Experiment Configuration Details](#5-experiment-configuration-details)
6. [Full-Factorial Run (Reproducing All 1,053 Experiments)](#6-full-factorial-run-reproducing-all-1053-experiments)
7. [Generating Summary Dashboard Plots](#7-generating-summary-dashboard-plots)
8. [Understanding the Output Files](#8-understanding-the-output-files)

---

## 1. Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| **Python** | 3.10+ | Tested on 3.13.2. Any 3.10+ should work. |
| **pip** | latest | Comes with Python. Run `python -m pip install --upgrade pip` to update. |
| **Git** | any | Only needed to clone the repo. |
| **OS** | Windows / macOS / Linux | All scripts are cross-platform. Path separators are handled by `pathlib`. |

No GPU is required — all experiments use scikit-learn's `SGDClassifier` which runs on CPU.

---

## 2. Clone & Install

### Windows (PowerShell)

```powershell
# Clone the repository
git clone https://github.com/<your-username>/Study-of-Drift-Triggered-Retraining-Policies-Under-Budget-and-Latency-Constraints.git
cd Study-of-Drift-Triggered-Retraining-Policies-Under-Budget-and-Latency-Constraints

# Create a virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\Activate.ps1
# If you get an execution-policy error, run this first:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# Install dependencies
pip install -r docs/requirements.txt
```

### macOS / Linux (Bash)

```bash
git clone https://github.com/<your-username>/Study-of-Drift-Triggered-Retraining-Policies-Under-Budget-and-Latency-Constraints.git
cd Study-of-Drift-Triggered-Retraining-Policies-Under-Budget-and-Latency-Constraints

python3 -m venv .venv
source .venv/bin/activate

pip install -r docs/requirements.txt
```

### Verify installation

```bash
python -c "import numpy, pandas, matplotlib, sklearn; print('All dependencies OK')"
```

---

## 3. Project Layout at a Glance

```
├── main.py                  ← CLI entry point: --policy and --seeds flags drive the sweep
├── plot_summary.py          ← Generates dashboard PNGs (called automatically by main.py)
├── docs/
│   └── requirements.txt     ← Python dependencies
├── src/
│   ├── data/drift_generator.py             ← Synthetic data stream with concept drift
│   ├── models/base_model.py                ← SGDClassifier wrapper
│   ├── policies/
│   │   ├── periodic.py                     ← PeriodicPolicy
│   │   ├── error_threshold_policy.py       ← ErrorThresholdPolicy
│   │   ├── drift_triggered_policy.py       ← DriftTriggeredPolicy (ADWIN)
│   │   └── never_retrain_policy.py         ← NeverRetrainPolicy (baseline)
│   ├── runner/experiment_runner.py         ← Streaming event loop
│   └── evaluation/
│       ├── metrics.py                      ← MetricsTracker
│       ├── results_export.py               ← CSV / JSON exporters
│       └── plot_results.py                 ← Per-run timeline plots
└── results/                 ← All outputs land here (CSVs, PNGs, JSONs)
```

---

## 4. Running Experiments

`main.py` is the single entry point for all experiment runs. It accepts two CLI flags and automatically sweeps the full factorial grid (3 drift types × 3 budgets × 3 latencies × N seeds) for the selected policy(ies).

### CLI Reference

```
python main.py [--policy POLICY] [--seeds N]
```

| Flag | Values | Default | Description |
|---|---|---|---|
| `--policy` | `periodic`, `error_threshold`, `drift_triggered`, `no_retrain`, `all` | `all` | Which retraining policy to sweep. `all` runs all four sequentially. |
| `--seeds` | `3`, `10` | `10` | Seed set size. `3` = Phase 1 seeds `[42, 123, 456]`. `10` = Phase 2 seeds `[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]`. |

### Examples

```bash
# Run ALL 4 policies with 10 seeds (840 runs — full reproduction)
python main.py

# Run only the periodic policy with 10 seeds (270 runs)
python main.py --policy periodic

# Run only drift-triggered (ADWIN) policy with 3 seeds (81 runs — Phase 1)
python main.py --policy drift_triggered --seeds 3

# Run only the no-retrain baseline with 10 seeds (30 runs)
python main.py --policy no_retrain --seeds 10

# Run all 4 policies with 3 seeds (252 runs — Phase 1 + baseline)
python main.py --policy all --seeds 3
```

### What happens when you run it

For each selected policy, `main.py`:

1. Deletes the old summary CSV for that policy (clean start — no duplicate rows).
2. Iterates over every `(drift_type, budget, latency, seed)` combination (or just `(drift_type, seed)` for no-retrain).
3. For each run: generates data → builds model + policy → streams 10,000 samples → exports JSON, per-sample CSV, and appends one row to the summary CSV.
4. Prints live progress with accuracy, retrain count, and ETA.
5. After all runs for a policy complete, calls `plot_summary.py` to generate the dashboard PNG (2×3 for active policies, 2×2 for no-retrain baseline).

### Commands to run specific policies or seed sets:

|---|---|---|
| `python main.py --policy <policy-name> --seeds <3 or 10>`
| `python main.py --policy <policy-name>` 
| `python main.py`

> **Important:** Always run `main.py` from the project root directory. The script uses relative imports (`from src.…`) and writes to `results/`, both of which require the working directory to be the project root.

---

## 5. Experiment Configuration Details

All experiment parameters are defined as constants at the top of `main.py`. **No manual editing is required** to run the standard experiment — the CLI flags select which subset to execute. The constants are documented here for reference.

### Drift Types

Controlled internally by the `DRIFT_TYPES` list — all three are always included in every sweep:

| Drift type | What it does |
|---|---|
| `"abrupt"` | Instant concept switch at t = 5,000 |
| `"gradual"` | Linear transition from old → new concept over t ∈ [5000, 6000] |
| `"recurring"` | Alternates between concepts every 1,000 steps after t = 5,000 |

### Retraining Policies

Selected via the `--policy` CLI flag. Each policy's fixed parameters are defined in `POLICY_PARAMS`:

| Policy | Key Parameters | Notes |
|---|---|---|
| `periodic` | `interval = 10000 // budget` | Interval is derived automatically from the budget level |
| `error_threshold` | `error_threshold=0.27`, `window_size=200` | Calibrated in Week 4 |
| `drift_triggered` | `delta=0.002`, `window_size=500`, `min_samples=100` | ADWIN; calibrated in Week 5 |
| `no_retrain` | None | Baseline — budget = 0, latency = 0, no full retrains |

### Budget & Latency Levels

Controlled internally by `BUDGETS` and `LATENCY_CONFIGS` — all combinations are swept automatically:

| Level | Budget (K) | Retrain Latency | Deploy Latency | Total Latency |
|---|---|---|---|---|
| Low | 5 | 10 | 1 | 11 |
| Medium | 10 | 100 | 5 | 105 |
| High | 20 | 500 | 20 | 520 |

### Seed Sets

Selected via the `--seeds` CLI flag:

| Flag value | Seeds | Phase | Runs per active policy | No-Retrain baseline runs |
|---|---|---|---|---|
| `3` | `[42, 123, 456]` | Phase 1 | 81 | 9 |
| `10` | `[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]` | Phase 2 | 270 | 30 |

### Summary CSV Naming Convention

`main.py` automatically names output files using the pattern:

```
results/summary_results_{policy}_retrain_{N}seed.csv
```

| Policy | Phase 1 (3-seed) CSV | Phase 2 (10-seed) CSV |
|---|---|---|
| Periodic | `summary_results_periodic_retrain_3seed.csv` | `summary_results_periodic_retrain_10seed.csv` |
| Error-Threshold | `summary_results_error_threshold_retrain_3seed.csv` | `summary_results_error_threshold_retrain_10seed.csv` |
| Drift-Triggered | `summary_results_drift_triggered_retrain_3seed.csv` | `summary_results_drift_triggered_retrain_10seed.csv` |
| No-Retrain | `summary_results_no_retrain_3seed.csv` | `summary_results_no_retrain_10seed.csv` |

> **Note:** `main.py` deletes the target summary CSV before starting each policy sweep, ensuring a clean file with a proper header. You do not need to manually delete CSVs.

---

## 6. Full-Factorial Run (Reproducing All 1,092 Experiments)

To reproduce the complete study, run both experiment phases plus the baseline using `main.py`:

```bash
# Phase 1 — 252 runs (3 seeds × 3 drifts × 3 budgets × 3 latencies × 3 policies + 9 baseline)
python main.py --seeds 3

# Phase 2 — 840 runs (10 seeds × 3 drifts × 3 budgets × 3 latencies × 3 policies + 30 baseline)
python main.py --seeds 10

# Or both phases in sequence:
python main.py --seeds 3 && python main.py --seeds 10
```

Or reproduce a single policy at a time:

```bash
python main.py --policy periodic --seeds 3         # 81 runs
python main.py --policy error_threshold --seeds 3   # 81 runs
python main.py --policy drift_triggered --seeds 3   # 81 runs
python main.py --policy no_retrain --seeds 3         # 9 runs (baseline)
python main.py --policy no_retrain --seeds 10        # 30 runs (baseline)
python main.py --policy periodic --seeds 10         # 270 runs
python main.py --policy error_threshold --seeds 10  # 270 runs
python main.py --policy drift_triggered --seeds 10  # 270 runs
```

**Total: 243 + 810 + 9 + 30 = 1,092 experiment runs.**

Each command automatically:
- Deletes the previous summary CSV for the selected policy(ies) (clean start — no manual cleanup needed).
- Sweeps the full `DRIFT_TYPES × BUDGETS × LATENCY_CONFIGS × seeds` grid (or `DRIFT_TYPES × seeds` for no-retrain).
- Exports per-run JSON, per-sample CSV, and appends to the summary CSV.
- Generates the dashboard PNG after all runs for each policy complete.

### Git Branching Strategy Used in This Study

#### Phase 1 — Per-Configuration Branches (243 runs, 3 seeds)

Each individual configuration was developed and tested in its own feature branch:
- **Branch naming:** `exp/<drift>-<policy>-<Budget>budget-<Latency>Latency`
- **Examples:**
  - `exp/gradual-drift-triggered-Medbudget-HighLatency`
  - `exp/abrupt-periodic-Lowbudget-LowLatency`
  - `exp/recurring-error-threshold-Highbudget-MedLatency`

Cumulative results per policy (81 runs) were merged into dedicated develop branches:
- `develop_3seed_periodic_retrain`
- `develop_3seed_error_threshold_retrain`
- `develop_3seed_drift_triggered_retrain`

#### Phase 2 — Per-Policy Batch Branches (810 runs, 10 seeds)

The 810 runs were executed in 3 batches (270 runs per batch), one batch per policy:
- `develop-10Seed-periodic-retrain-tests` (270 runs)
- `develop-10Seed-error-threshold-retrain-tests` (270 runs)
- `develop-10Seed-drift-triggered-retrain-tests` (270 runs)

#### No-Retrain Baseline (39 runs, 3 + 10 seeds)

The baseline sweep (3 drift types × N seeds, no budget/latency grid) was executed in:
- `develop_NoRetrain_NoBudget_NoLatency` (9 runs with 3 seeds + 30 runs with 10 seeds)

#### Merged Results

The **`main`** and **`develop`** branches contain all results from all phases (1,092 total runs). All CSVs and dashboard plots are in the `results/` folder.

### Why the original experiments were NOT run all at once

If you look at the [research log](research_log.md) you will notice the experiments were executed incrementally across six weeks, not as a single batch. This was deliberate:

1. **Policies were developed sequentially.** The periodic policy was implemented and run first (Weeks 2–3), the error-threshold policy second (Week 4), and the ADWIN policy last (Weeks 5–6). You cannot run all three policies in one batch when two of them do not exist yet.

2. **Parameter calibration depended on earlier results.** The error-threshold value of `0.27` was selected in Week 4 after sweeping multiple candidates (`0.20`, `0.25`, `0.27`, `0.30`, `0.35`) against the periodic-policy baseline and observing which threshold minimized pre-drift false alarms. Similarly, the ADWIN confidence parameter `δ = 0.002` was chosen in Week 5 by sweeping `{0.05, 0.01, 0.005, 0.002, 0.001}` and verifying zero false positives pre-drift. Running all 243 experiments upfront would have required guessing these values, likely producing inferior or invalid results.

3. **Analysis between batches guided the next batch.** Observations from the periodic runs (e.g., the latency–budget interaction at high latency, per-seed accuracy variance) directly informed what to look for in the error-threshold and ADWIN runs. This iterative cycle of *run → analyze → adjust → run next* is a core part of the experimental methodology.

4. **Incremental runs aided debugging and validation.** Running one policy at a time made it possible to inspect console output, verify that metrics were recorded correctly, and catch bugs before committing to 81 more runs.

> **For reproduction purposes**, however, you now have the final calibrated parameters baked into `main.py`. You can safely run all 1,092 experiments (or just the 810 Phase 2 runs + 30 baseline runs for the strongest variance estimates) with a single CLI command.

---

## 7. Generating Summary Dashboard Plots

`main.py` automatically generates a dashboard PNG for each policy after its sweep completes. You do **not** need to run `plot_summary.py` separately in the normal workflow.

### Automatic generation (via main.py)

When `main.py` finishes all runs for a policy, it calls the appropriate plotting function internally:
- **Active policies** (periodic, error-threshold, drift-triggered): `plot_summary_for_policy()` → 2×3 dashboard
- **No-retrain baseline**: `plot_summary_for_no_retrain()` → 2×2 baseline dashboard

The output PNG is saved alongside the summary CSV:

```
results/summary_results_plot_{policy}_retrain_{N}seed.png    # active policies
results/summary_results_plot_no_retrain_{N}seed.png          # baseline
```

### Manual / standalone generation

If you want to regenerate a dashboard without re-running experiments (e.g., after tweaking plot styles), call `plot_summary.py` directly:

```bash
python plot_summary.py                                  # all policies (10-seed by default)
python plot_summary.py --policy periodic_10seed         # 10-seed periodic only
python plot_summary.py --policy drift_triggered_3seed   # 3-seed drift-triggered only
python plot_summary.py --policy no_retrain_10seed       # 10-seed no-retrain baseline only
```

Output files:

| Policy | Phase 1 (3-seed) Dashboard | Phase 2 (10-seed) Dashboard |
|---|---|---|
| Periodic | `summary_results_plot_periodic_retrain_3seed.png` | `summary_results_plot_periodic_retrain_10seed.png` |
| Error-Threshold | `summary_results_plot_error_threshold_retrain_3seed.png` | `summary_results_plot_error_threshold_retrain_10seed.png` |
| Drift-Triggered | `summary_results_plot_drift_triggered_retrain_3seed.png` | `summary_results_plot_drift_triggered_retrain_10seed.png` |
| No-Retrain | `summary_results_plot_no_retrain_3seed.png` | `summary_results_plot_no_retrain_10seed.png` |

---

## 8. Understanding the Output Files

After running experiments, you will find these files in `results/`:

| File | Created by | Description |
|---|---|---|
| `run_{run_tag}.json` | `main.py` | Full config + structured metrics per run. Stored in policy-specific subdirectories. |
| `per_sample_{run_tag}.csv` | `main.py` | Per-timestep accuracy, error, and latency flags. Stored in policy-specific subdirectories. |
| `summary_results_{policy}_retrain_3seed.csv` | `main.py` | Phase 1: one row per run; 81 rows per active policy (243 total). |
| `summary_results_{policy}_retrain_10seed.csv` | `main.py` | Phase 2: one row per run; 270 rows per active policy (810 total). |
| `summary_results_no_retrain_3seed.csv` | `main.py` | No-retrain baseline: one row per run; 9 rows (3 drift types × 3 seeds). |
| `summary_results_no_retrain_10seed.csv` | `main.py` | No-retrain baseline: one row per run; 30 rows (3 drift types × 10 seeds). |
| `summary_results_plot_{policy}_retrain_3seed.png` | `plot_summary.py` | 2×3 dashboard for Phase 1 (3-seed) runs of that policy. |
| `summary_results_plot_{policy}_retrain_10seed.png` | `plot_summary.py` | 2×3 dashboard for Phase 2 (10-seed) runs of that policy. |
| `summary_results_plot_no_retrain_3seed.png` | `plot_summary.py` | 2×2 baseline dashboard for no-retrain policy (3 seeds). |
| `summary_results_plot_no_retrain_10seed.png` | `plot_summary.py` | 2×2 baseline dashboard for no-retrain policy (10 seeds). |

> **Tip:** For detailed interpretation of the CSV columns and the dashboard panels, see [results_interpretation_guide.md](results_interpretation_guide.md).