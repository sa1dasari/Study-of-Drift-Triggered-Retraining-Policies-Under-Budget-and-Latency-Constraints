# Setup & Run Guide

A step-by-step guide for anyone who has freshly cloned this repository and wants to run the experiments.

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

## 3. Project Layout

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

### CLI Reference

```
python main.py [--policy POLICY] [--seeds N]
```

| Flag | Values | Default | Description |
|---|---|---|---|
| `--policy` | `periodic`, `error_threshold`, `drift_triggered`, `no_retrain`, `all` | `all` | Which retraining policy to sweep. `all` runs all four sequentially. |
| `--seeds` | `3`, `10` | `10` | Seed set size. `3` = `[42, 123, 456]`. `10` = `[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]`. |

### Examples

```bash
python main.py                                    # all 4 policies, 10 seeds
python main.py --policy periodic                  # periodic only, 10 seeds
python main.py --policy drift_triggered --seeds 3 # drift-triggered only, 3 seeds
python main.py --policy no_retrain --seeds 10     # no-retrain baseline only
```

### What happens when you run it

For each selected policy, `main.py`:

1. Deletes the old summary CSV for that policy (clean start).
2. Iterates over every `(drift_type, budget, latency, seed)` combination (or just `(drift_type, seed)` for no-retrain).
3. For each run: generates data → builds model + policy → streams 10,000 samples → exports JSON, per-sample CSV, and appends one row to the summary CSV.
4. Prints live progress with accuracy, retrain count, and ETA.
5. Generates a dashboard PNG after all runs complete.

> **Important:** Always run `main.py` from the project root directory.

---

## 5. Experiment Configuration

All parameters are defined as constants at the top of `main.py`. The CLI flags select which subset to execute.

### Drift Types

| Drift type | What it does |
|---|---|
| `"abrupt"` | Instant concept switch at t = 5,000 |
| `"gradual"` | Linear transition over t ∈ [5000, 6000] |
| `"recurring"` | Alternates between concepts every 1,000 steps after t = 5,000 |

### Policies

| Policy | Key Parameters |
|---|---|
| `periodic` | `interval = 10000 // budget` |
| `error_threshold` | `error_threshold=0.27`, `window_size=200` |
| `drift_triggered` | `delta=0.002`, `window_size=500`, `min_samples=100` |
| `no_retrain` | Baseline — budget = 0, latency = 0 |

### Budget Levels

| Level | K (max retrains) |
|---|---|
| Low | 5 |
| Medium | 10 |
| High | 20 |

### Latency Levels

The latency levels in `LATENCY_CONFIGS` determine which experiment phase is run. See [experiment_scope.md](experiment_scope.md) for full details on each phase.

> **Note:** `main.py` deletes the target summary CSV before starting each sweep, ensuring a clean file. You do not need to manually delete CSVs.

---

## 6. Output Files

After running experiments, you will find these files in `results/`:

| File pattern | Description |
|---|---|
| `summary_results_{policy}_retrain_{tag}.csv` | One summary row per run for that policy |
| `summary_results_plot_{policy}_retrain_{tag}.png` | 2×3 dashboard PNG for that policy |
| `summary_results_no_retrain_{tag}.csv` | Baseline summary (no budget/latency grid) |
| `summary_results_plot_no_retrain_{tag}.png` | 2×2 baseline dashboard |

Where `{tag}` is `3seed`, `10seed`, `ExtremeLatency_3seed`, or `ExtremeLatency_10seed`.

> **Tip:** For detailed interpretation of the CSV columns and dashboard panels, see [results_interpretation_guide.md](results_interpretation_guide.md). For the full list of experiment phases and run counts, see [experiment_scope.md](experiment_scope.md).
