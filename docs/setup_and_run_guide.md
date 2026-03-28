# Setup & Run Guide

A step-by-step guide for anyone who has freshly cloned this repository and wants to reproduce the experiments or run new configurations.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone & Install](#2-clone--install)
3. [Project Layout at a Glance](#3-project-layout-at-a-glance)
4. [Running a Single Experiment](#4-running-a-single-experiment)
5. [Configuring an Experiment](#5-configuring-an-experiment)
   - [Step A — Choose a Drift Type](#step-a--choose-a-drift-type)
   - [Step B — Choose a Retraining Policy](#step-b--choose-a-retraining-policy)
   - [Step C — Choose Budget & Latency](#step-c--choose-budget--latency)
   - [Step D — Choose Seeds](#step-d--choose-seeds)
   - [Step E — Set the Summary CSV Filename](#step-e--set-the-summary-csv-filename)
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
├── main.py                  ← You edit this to configure and launch experiments
├── plot_summary.py          ← Run after experiments to generate dashboard PNGs
├── docs/
│   └── requirements.txt     ← Python dependencies
├── src/
│   ├── data/drift_generator.py             ← Synthetic data stream with concept drift
│   ├── models/base_model.py                ← SGDClassifier wrapper
│   ├── policies/
│   │   ├── periodic.py                     ← PeriodicPolicy
│   │   ├── error_threshold_policy.py       ← ErrorThresholdPolicy
│   │   └── drift_triggered_policy.py       ← DriftTriggeredPolicy (ADWIN)
│   ├── runner/experiment_runner.py         ← Streaming event loop
│   └── evaluation/
│       ├── metrics.py                      ← MetricsTracker
│       ├── results_export.py               ← CSV / JSON exporters
│       └── plot_results.py                 ← Per-run timeline plots
└── results/                 ← All outputs land here (CSVs, PNGs, JSONs)
```

---

## 4. Running a Single Experiment

From the project root directory (where `main.py` is located):

```bash
python main.py
```

This will:
1. Generate synthetic data for each seed.
2. Train and evaluate the streaming model using the configured policy.
3. Print detailed results to the console.
4. Export results to `results/` (JSON, per-sample CSV, summary CSV).
5. Generate a per-run plot in `results/experiment_results.png`.

**Runtime:** A single seed takes ~5–15 seconds. The default `main.py` runs 3 seeds, so expect ~15–45 seconds total.

> **Important:** Always run `main.py` from the project root directory. The script uses relative imports (`from src.…`) and writes to `results/`, both of which require the working directory to be the project root.

---

## 5. Configuring an Experiment

All experiment configuration is done by editing `main.py`. There is no config file — the parameters are set directly in Python code. Here is how to change each factor:

### Step A — Choose a Drift Type

Near the top of the `main()` function, find and set the `drift_type` variable:

```python
drift_type = "recurring"   # Options: "abrupt", "gradual", "recurring"
```

Also update the `DriftGenerator` constructor to match:

```python
generator = DriftGenerator(
    drift_type="recurring",    # ← Must match the drift_type variable above
    drift_point=5000,
    recurrence_period=1000,    # Only relevant for "recurring"
    seed=seed
)
```

| Drift type | What it does |
|---|---|
| `"abrupt"` | Instant concept switch at t = 5,000 |
| `"gradual"` | Linear transition from old → new concept over t ∈ [5000, 6000] |
| `"recurring"` | Alternates between concepts every 1,000 steps after t = 5,000 |

### Step B — Choose a Retraining Policy

The current `main.py` uses the drift-triggered (ADWIN) policy. To switch policies, change the **import** and the **policy construction**.

#### Option 1: Periodic Policy

```python
# Change the import
from src.policies.periodic import PeriodicPolicy

# Change the policy construction
policy = PeriodicPolicy(
    interval=1000,          # Retrain every 1,000 timesteps
    budget=10,              # Max 10 retrains allowed
    retrain_latency=100,    # 100 timesteps to retrain offline
    deploy_latency=5        # 5 timesteps to deploy
)
```

Also update the `policy_type` string in the config dict and the `plot_results` call:

```python
config = {
    ...
    "policy_type": "periodic",
    "policy_interval": policy.interval,
    ...
}
export_summary_to_csv(metrics, policy, config, "results/summary_results_periodic_retrain_3seed.csv")
plot_results(seeds, policy, drift_point=5000, drift_type=drift_type, policy_type="periodic")

```

#### Option 2: Error-Threshold Policy

```python
# Change the import
from src.policies.error_threshold_policy import ErrorThresholdPolicy

# Change the policy construction
policy = ErrorThresholdPolicy(
    error_threshold=0.27,   # Retrain when rolling error rate > 27%
    window_size=200,        # Compute error rate over last 200 predictions
    budget=10,              # Max 10 retrains allowed
    retrain_latency=100,    # 100 timesteps to retrain offline
    deploy_latency=5        # 5 timesteps to deploy
)
```

Update the config dict:

```python
config = {
    ...
    "policy_type": "error_threshold",
    "error_threshold": policy.error_threshold,
    "window_size": policy.window_size,
    ...
}
export_summary_to_csv(metrics, policy, config, "results/summary_results_error_threshold_retrain_3seed.csv")
plot_results(seeds, policy, drift_point=5000, drift_type=drift_type, policy_type="error_threshold")
```

#### Option 3: Drift-Triggered (ADWIN) Policy (default)

```python
from src.policies.drift_triggered_policy import DriftTriggeredPolicy

policy = DriftTriggeredPolicy(
    delta=0.002,            # ADWIN confidence parameter (lower = less sensitive)
    window_size=500,        # Max recent errors for drift detection
    min_samples=300,        # Wait 300 samples before detection activates
    budget=10,              # Max 10 retrains allowed
    retrain_latency=100,    # 100 timesteps to retrain offline
    deploy_latency=5        # 5 timesteps to deploy
)
```

### Step C — Choose Budget & Latency

These are set directly in the policy constructor (see Step B). The study uses these standard levels:

| Level | Budget (K) | Retrain Latency | Deploy Latency | Total Latency |
|---|---|---|---|---|
| Low | 5 | 10 | 1 | 11 |
| Medium | 10 | 100 | 5 | 105 |
| High | 20 | 500 | 20 | 520 |

> **Note for Periodic Policy:** When changing budget, also update the `interval` to match: `interval = 10000 / budget`. So K=5 → interval=2000, K=10 → interval=1000, K=20 → interval=500.

### Step D — Choose Seeds

The seeds list controls how many runs are performed and with which random initializations:

```python
seeds = [42, 123, 456]     # Phase 1: 3 seeds for initial exploration
```

For the extended 10-seed Phase 2 runs:

```python
seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]  # Phase 2: 10 seeds
```

You can use a single seed for a quick test:

```python
seeds = [42]                # Fast: 1 run only
```

### Step E — Set the Summary CSV Filename

Each policy should write to its own summary CSV so results don't mix. The filename should include the seed count to distinguish between experiment phases. Make sure the `export_summary_to_csv` call uses the correct filename:

| Policy | Phase 1 (3-seed) CSV | Phase 2 (10-seed) CSV |
|---|---|---|
| Periodic | `results/summary_results_periodic_retrain_3seed.csv` | `results/summary_results_periodic_retrain_10seed.csv` |
| Error-Threshold | `results/summary_results_error_threshold_retrain_3seed.csv` | `results/summary_results_error_threshold_retrain_10seed.csv` |
| Drift-Triggered | `results/summary_results_drift_triggered_retrain_3seed.csv` | `results/summary_results_drift_triggered_retrain_10seed.csv` |

> **Important:** The summary CSV is opened in **append mode**. Each run adds one row. If you re-run the same configuration, duplicate rows will be appended. **Delete the CSV first** if you want a clean start:
>
> ```powershell
> Remove-Item results/summary_results_periodic_retrain_3seed.csv   # Windows
> # rm results/summary_results_periodic_retrain_3seed.csv          # macOS/Linux
> ```

---

## 6. Full-Factorial Run (Reproducing All 1,053 Experiments)

To reproduce the complete study, you need to run both experiment phases:
- **Phase 1:** 243 runs (3 seeds × 3 drifts × 3 budgets × 3 latencies × 3 policies)
- **Phase 2:** 810 runs (10 seeds × 3 drifts × 3 budgets × 3 latencies × 3 policies)
- **Total:** 1,053 experiment runs

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

#### Merged Results

The **`main`** and **`develop`** branches contain all results from both phases (1,053 total runs). All CSVs and dashboard plots are in the `results/` folder.

### Why the original experiments were NOT run all at once

If you look at the [research log](research_log.md) you will notice the experiments were executed incrementally across six weeks, not as a single batch. This was deliberate:

1. **Policies were developed sequentially.** The periodic policy was implemented and run first (Weeks 2–3), the error-threshold policy second (Week 4), and the ADWIN policy last (Weeks 5–6). You cannot run all three policies in one batch when two of them do not exist yet.

2. **Parameter calibration depended on earlier results.** The error-threshold value of `0.27` was selected in Week 4 after sweeping multiple candidates (`0.20`, `0.25`, `0.27`, `0.30`, `0.35`) against the periodic-policy baseline and observing which threshold minimized pre-drift false alarms. Similarly, the ADWIN confidence parameter `δ = 0.002` was chosen in Week 5 by sweeping `{0.05, 0.01, 0.005, 0.002, 0.001}` and verifying zero false positives pre-drift. Running all 243 experiments upfront would have required guessing these values, likely producing inferior or invalid results.

3. **Analysis between batches guided the next batch.** Observations from the periodic runs (e.g., the latency–budget interaction at high latency, per-seed accuracy variance) directly informed what to look for in the error-threshold and ADWIN runs. This iterative cycle of *run → analyze → adjust → run next* is a core part of the experimental methodology.

4. **Incremental runs aided debugging and validation.** Running one policy at a time made it possible to inspect console output, verify that metrics were recorded correctly, and catch bugs before committing to 81 more runs.

> **For reproduction purposes**, however, you now have the final calibrated parameters for all three policies. You can safely run all 1,053 experiments (or just the 810 Phase 2 runs for the strongest variance estimates) using the wrapper scripts provided below.

### 6.1 Clear existing results

```powershell
# Windows PowerShell
Remove-Item results/summary_results_periodic_retrain_3seed.csv -ErrorAction SilentlyContinue
Remove-Item results/summary_results_error_threshold_retrain_3seed.csv -ErrorAction SilentlyContinue
Remove-Item results/summary_results_drift_triggered_retrain_3seed.csv -ErrorAction SilentlyContinue
Remove-Item results/summary_results_periodic_retrain_10seed.csv -ErrorAction SilentlyContinue
Remove-Item results/summary_results_error_threshold_retrain_10seed.csv -ErrorAction SilentlyContinue
Remove-Item results/summary_results_drift_triggered_retrain_10seed.csv -ErrorAction SilentlyContinue
```

```bash
# macOS / Linux
rm -f results/summary_results_*_retrain_*seed.csv
```

### 6.2 Run all configurations

For each policy, you need to iterate over 3 drift types × 3 budget/latency levels and run `main.py` each time. In practice this means editing `main.py` **27 times per policy** (or writing a wrapper script).

Below is the full configuration matrix. For each row, set the matching values in `main.py` and run it:

**Periodic Policy — 27 invocations (× 3 seeds each = 81 rows)**

| # | drift_type | interval | budget | retrain_latency | deploy_latency |
|---|---|---|---|---|---|
| 1 | abrupt | 2000 | 5 | 10 | 1 |
| 2 | abrupt | 2000 | 5 | 100 | 5 |
| 3 | abrupt | 2000 | 5 | 500 | 20 |
| 4 | abrupt | 1000 | 10 | 10 | 1 |
| 5 | abrupt | 1000 | 10 | 100 | 5 |
| 6 | abrupt | 1000 | 10 | 500 | 20 |
| 7 | abrupt | 500 | 20 | 10 | 1 |
| 8 | abrupt | 500 | 20 | 100 | 5 |
| 9 | abrupt | 500 | 20 | 500 | 20 |
| 10–18 | gradual | *(same 9 budget/latency combos)* | | | |
| 19–27 | recurring | *(same 9 budget/latency combos)* | | | |

**Error-Threshold Policy — 27 invocations**

Same drift × budget × latency grid. Fixed parameters: `error_threshold=0.27`, `window_size=200`.

**Drift-Triggered (ADWIN) Policy — 27 invocations**

Same drift × budget × latency grid. Fixed parameters: `delta=0.002`, `window_size=500`, `min_samples=300`.

### 6.3 Alternatively: Use below automation wrapper

Instead of editing `main.py` many times, you can use the below small wrapper script. Save this as `run_all_experiments.py` in the project root and run it. It will loop through all configurations for all three policies, running each one sequentially and appending results to the appropriate summary CSV.

> **Note:** The script below reproduces the Phase 1 (3-seed, 243 runs) configuration. To reproduce Phase 2 (10-seed, 810 runs), change the `SEEDS` list to include all 10 seeds and update the CSV filenames to use `_10seed` suffixes.

```python
"""
run_all_experiments.py — Runs the full experiment matrix.

Phase 1 (3 seeds):  243 runs  → *_3seed.csv
Phase 2 (10 seeds): 810 runs  → *_10seed.csv

To switch between phases, change SEEDS and SEED_LABEL below.
"""
import os
from pathlib import Path
from src.data.drift_generator import DriftGenerator
from src.models.base_model import StreamingModel
from src.policies.periodic import PeriodicPolicy
from src.policies.error_threshold_policy import ErrorThresholdPolicy
from src.policies.drift_triggered_policy import DriftTriggeredPolicy
from src.evaluation.metrics import MetricsTracker
from src.runner.experiment_runner import ExperimentRunner
from src.evaluation.results_export import export_summary_to_csv

# ── Phase 1 (3 seeds, 243 runs) ──
SEEDS = [42, 123, 456]
SEED_LABEL = "3seed"

# ── Uncomment below for Phase 2 (10 seeds, 810 runs) ──
# SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
# SEED_LABEL = "10seed"

DRIFT_TYPES = ["abrupt", "gradual", "recurring"]
BUDGET_LATENCY = [
    # (budget, retrain_latency, deploy_latency)
    (5,  10,  1),
    (5,  100, 5),
    (5,  500, 20),
    (10, 10,  1),
    (10, 100, 5),
    (10, 500, 20),
    (20, 10,  1),
    (20, 100, 5),
    (20, 500, 20),
]

def interval_for_budget(budget):
    return 10000 // budget   # 5→2000, 10→1000, 20→500

def run_one(policy, policy_type, config, drift_type, seed, budget, r_lat, d_lat):
    gen = DriftGenerator(drift_type=drift_type, drift_point=5000, recurrence_period=1000, seed=seed)
    X, y = gen.generate(10000)
    model = StreamingModel()
    metrics = MetricsTracker()
    metrics.set_drift_point(5000)
    metrics.set_budget(budget)
    runner = ExperimentRunner(model, policy, metrics)
    runner.run(X, y)
    csv_file = f"results/summary_results_{policy_type}_retrain_{SEED_LABEL}.csv"
    export_summary_to_csv(metrics, policy, config, csv_file)

def main():
    Path("results").mkdir(exist_ok=True)

    # Delete old summary CSVs for a clean start
    for pt in ["periodic", "error_threshold", "drift_triggered"]:
        f = Path(f"results/summary_results_{pt}_retrain_{SEED_LABEL}.csv")
        if f.exists():
            f.unlink()

    total = len(DRIFT_TYPES) * len(BUDGET_LATENCY) * len(SEEDS) * 3  # 3 policies
    run_num = 0

    for drift_type in DRIFT_TYPES:
        for budget, r_lat, d_lat in BUDGET_LATENCY:
            for seed in SEEDS:
                # --- Periodic ---
                run_num += 1
                interval = interval_for_budget(budget)
                p = PeriodicPolicy(interval=interval, budget=budget,
                                   retrain_latency=r_lat, deploy_latency=d_lat)
                cfg = {"drift_type": drift_type, "policy_type": "periodic",
                       "policy_interval": interval, "budget": budget, "random_seed": seed}
                print(f"[{run_num}/{total}] periodic | {drift_type} | K={budget} | lat={r_lat}+{d_lat} | seed={seed}")
                run_one(p, "periodic", cfg, drift_type, seed, budget, r_lat, d_lat)

                # --- Error-Threshold ---
                run_num += 1
                p = ErrorThresholdPolicy(error_threshold=0.27, window_size=200,
                                         budget=budget, retrain_latency=r_lat, deploy_latency=d_lat)
                cfg = {"drift_type": drift_type, "policy_type": "error_threshold",
                       "error_threshold": 0.27, "window_size": 200, "budget": budget, "random_seed": seed}
                print(f"[{run_num}/{total}] error_threshold | {drift_type} | K={budget} | lat={r_lat}+{d_lat} | seed={seed}")
                run_one(p, "error_threshold", cfg, drift_type, seed, budget, r_lat, d_lat)

                # --- Drift-Triggered (ADWIN) ---
                run_num += 1
                p = DriftTriggeredPolicy(delta=0.002, window_size=500, min_samples=300,
                                         budget=budget, retrain_latency=r_lat, deploy_latency=d_lat)
                cfg = {"drift_type": drift_type, "policy_type": "drift_triggered",
                       "window_size": 500, "budget": budget, "random_seed": seed}
                print(f"[{run_num}/{total}] drift_triggered | {drift_type} | K={budget} | lat={r_lat}+{d_lat} | seed={seed}")
                run_one(p, "drift_triggered", cfg, drift_type, seed, budget, r_lat, d_lat)

    print(f"\nDone! {run_num} experiments completed.")
    print("Summary CSVs written to results/")

if __name__ == "__main__":
    main()
```

Run it:

```bash
python run_all_experiments.py
```

### 6.4 Generate all dashboard plots

After all runs are complete:

```bash
python plot_summary.py
```

> **Note:** `plot_summary.py` provides a `plot_summary_for_policy()` function that accepts a CSV path and output path. You can call it for each policy and seed phase. See [Section 7](#7-generating-summary-dashboard-plots) for details.

---

## 7. Generating Summary Dashboard Plots

### Per-run timeline plot

A timeline + rolling-accuracy plot is automatically generated at the end of each `main.py` run and saved to:

```
results/experiment_results.png
```

This file is overwritten on each run. It shows drift regions, retrain events, and accuracy over time for the most recently executed configuration.

### Cross-run summary dashboard

The `plot_summary.py` script reads one of the summary CSVs and generates a 2×3 panel dashboard.

```bash
python plot_summary.py
```

By default it reads the drift-triggered 10-seed CSV. To generate dashboards for other policies or phases, update the CSV path and output path:

```python
# Phase 1 (3-seed) examples:
df = pd.read_csv('results/summary_results_periodic_retrain_3seed.csv')
df = pd.read_csv('results/summary_results_error_threshold_retrain_3seed.csv')
df = pd.read_csv('results/summary_results_drift_triggered_retrain_3seed.csv')

# Phase 2 (10-seed) examples:
df = pd.read_csv('results/summary_results_periodic_retrain_10seed.csv')
df = pd.read_csv('results/summary_results_error_threshold_retrain_10seed.csv')
df = pd.read_csv('results/summary_results_drift_triggered_retrain_10seed.csv')
```

Also update the figure title and the output filename to match.

Output files:

| Policy | Phase 1 (3-seed) Dashboard | Phase 2 (10-seed) Dashboard |
|---|---|---|
| Periodic | `results/summary_results_plot_periodic_retrain_3seed.png` | `results/summary_results_plot_periodic_retrain_10seed.png` |
| Error-Threshold | `results/summary_results_plot_error_threshold_retrain_3seed.png` | `results/summary_results_plot_error_threshold_retrain_10seed.png` |
| Drift-Triggered | `results/summary_results_plot_drift_triggered_retrain_3seed.png` | `results/summary_results_plot_drift_triggered_retrain_10seed.png` |

---

## 8. Understanding the Output Files

After running experiments, you will find these files in `results/`:

| File | Created by | Description |
|---|---|---|
| `run_{run_tag}.json` | `main.py` | Full config + structured metrics per run. Stored in policy-specific subdirectories. |
| `per_sample_{run_tag}.csv` | `main.py` | Per-timestep accuracy, error, and latency flags. Stored in policy-specific subdirectories. |
| `summary_results_{policy}_retrain_3seed.csv` | `main.py` | Phase 1: one row per run; 81 rows per policy (243 total). |
| `summary_results_{policy}_retrain_10seed.csv` | `main.py` | Phase 2: one row per run; 270 rows per policy (810 total). |
| `summary_results_plot_{policy}_retrain_3seed.png` | `plot_summary.py` | 2×3 dashboard for Phase 1 (3-seed) runs of that policy. |
| `summary_results_plot_{policy}_retrain_10seed.png` | `plot_summary.py` | 2×3 dashboard for Phase 2 (10-seed) runs of that policy. |

> **Tip:** For detailed interpretation of the CSV columns and the dashboard panels, see [results_interpretation_guide.md](results_interpretation_guide.md).