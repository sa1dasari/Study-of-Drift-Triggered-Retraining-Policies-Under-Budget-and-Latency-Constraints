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
├── main.py                  ← CLI entry point: synthetic experiments (--policy and --seeds)
├── luflow_main.py           ← CLI entry point: LUFlow real-world experiments (--policy)
├── lendingclub_main.py      ← CLI entry point: LendingClub real-world experiments (--policy)
├── luflow_fitness_check.py  ← LUFlow dataset suitability gate checks
├── lendingclub_fitness_check.py ← LendingClub dataset suitability gate checks
├── plot_summary.py          ← Generates dashboard PNGs (called automatically by main.py / luflow_main.py / lendingclub_main.py)
├── docs/
│   └── requirements.txt     ← Python dependencies
├── src/
│   ├── data/drift_generator.py             ← Synthetic data stream with concept drift
│   ├── data/LUFlow_Network_Intrusion/
│   │   └── datasets/                        ← 28 day-CSVs (downloaded separately)
│   ├── data/LendingClub_Loan_Data/
│   │   ├── lendingclub_loader.py            ← LendingClub CSV loader & preprocessor
│   │   └── datasets/                        ← accepted_2007_to_2018Q4.csv (downloaded separately)
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
└── results/                 ← All outputs land here
    ├── synthetic/
    │   ├── csv/             ← Summary CSVs (synthetic experiments)
    │   ├── plots/           ← Dashboard PNGs (synthetic experiments)
    │   └── per_run/         ← Per-run JSONs and per-sample CSVs
    ├── luflow/
    │   ├── csv/             ← Summary CSVs (LUFlow experiments)
    │   ├── plots/           ← Dashboard PNGs (LUFlow experiments)
    │   └── per_run/         ← Per-run JSONs and per-sample CSVs
    └── lendingclub/
        ├── csv/             ← Summary CSVs (LendingClub experiments)
        ├── plots/           ← Dashboard PNGs (LendingClub experiments)
        └── per_run/         ← Per-run JSONs and per-sample CSVs
```

---

## 4. Running Experiments

### Synthetic Experiments

#### CLI Reference

```
python main.py [--policy POLICY] [--seeds N]
```

| Flag | Values | Default | Description |
|---|---|---|---|
| `--policy` | `periodic`, `error_threshold`, `drift_triggered`, `no_retrain`, `all` | `all` | Which retraining policy to sweep. `all` runs all four sequentially. |
| `--seeds` | `3`, `10` | `10` | Seed set size. `3` = `[42, 123, 456]`. `10` = `[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]`. |

#### Examples

```bash
python main.py                                    # all 4 policies, 10 seeds
python main.py --policy periodic                  # periodic only, 10 seeds
python main.py --policy drift_triggered --seeds 3 # drift-triggered only, 3 seeds
python main.py --policy no_retrain --seeds 10     # no-retrain baseline only
```

### LUFlow Real-World Experiments

Before running, download the 28 LUFlow day-CSVs into `src/data/LUFlow_Network_Intrusion/datasets/`. See [src/data/LUFlow_Network_Intrusion/README.md](../src/data/LUFlow_Network_Intrusion/README.md) for download instructions.

#### CLI Reference

```
python luflow_main.py [--policy POLICY]
```

| Flag | Values | Default | Description |
|---|---|---|---|
| `--policy` | `periodic`, `error_threshold`, `drift_triggered`, `no_retrain`, `all` | `all` | Which retraining policy to sweep. |

#### Examples

```bash
python luflow_main.py                             # all 4 policies (252 runs)
python luflow_main.py --policy drift_triggered    # drift-triggered only (81 runs)
python luflow_main.py --policy no_retrain         # baseline only (9 runs)
```

### LendingClub Real-World Experiments

Before running, download the LendingClub accepted-loans CSV into `src/data/LendingClub_Loan_Data/datasets/`. See [src/data/LendingClub_Loan_Data/README.md](../src/data/LendingClub_Loan_Data/README.md) for download instructions.

#### CLI Reference

```
python lendingclub_main.py [--policy POLICY]
```

| Flag | Values | Default | Description |
|---|---|---|---|
| `--policy` | `periodic`, `error_threshold`, `drift_triggered`, `no_retrain`, `all` | `all` | Which retraining policy to sweep. |

#### Examples

```bash
python lendingclub_main.py                             # all 4 policies (252 runs)
python lendingclub_main.py --policy periodic           # periodic only (81 runs)
python lendingclub_main.py --policy drift_triggered    # drift-triggered only (81 runs)
python lendingclub_main.py --policy no_retrain         # baseline only (9 runs)
```

### What happens when you run it

For each selected policy, `main.py` (synthetic), `luflow_main.py` (LUFlow), or `lendingclub_main.py` (LendingClub):

1. Deletes the old summary CSV for that policy (clean start).
2. Iterates over every `(drift_type, budget, latency, seed)` combination (or just `(drift_type, seed)` for no-retrain).
3. For each run: generates data → builds model + policy → streams samples → exports JSON, per-sample CSV, and appends one row to the summary CSV.
4. Prints live progress with accuracy, retrain count, and ETA.
5. Generates a dashboard PNG after all runs complete.

> **Important:** Always run scripts from the project root directory. For LUFlow experiments, the dataset CSVs must be downloaded first. For LendingClub experiments, the accepted-loans CSV must be downloaded first (see instructions above).

---

## 5. Experiment Configuration

All parameters are defined as constants at the top of `main.py` (synthetic), `luflow_main.py` (LUFlow), and `lendingclub_main.py` (LendingClub). The CLI flags select which subset to execute.

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

> **Real-world datasets (LUFlow & LendingClub):** Policy parameters are re-calibrated for 50,000-sample streams — `periodic` uses interval = 50,000 / K; `error_threshold` uses threshold = 0.20; `drift_triggered` uses δ = 0.005 with min_samples = 100. See `luflow_main.py` and `lendingclub_main.py` for exact values.

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

### Synthetic experiments (`results/synthetic/`)

| File pattern | Location | Description |
|---|---|---|
| `summary_results_{policy}_retrain_{tag}.csv` | `results/synthetic/csv/` | One summary row per run for that policy |
| `summary_results_plot_{policy}_retrain_{tag}.png` | `results/synthetic/plots/` | 2×3 dashboard PNG for that policy |
| `summary_results_no_retrain_{tag}.csv` | `results/synthetic/csv/` | Baseline summary (no budget/latency grid) |
| `summary_results_plot_no_retrain_{tag}.png` | `results/synthetic/plots/` | 2×2 baseline dashboard |

### LUFlow experiments (`results/luflow/`)

| File pattern | Location | Description |
|---|---|---|
| `luflow_summary_{policy}_retrain_{tag}.csv` | `results/luflow/csv/` | One summary row per run for that policy |
| `luflow_summary_plot_{policy}_retrain_{tag}.png` | `results/luflow/plots/` | 2×3 dashboard PNG for that policy |

### LendingClub experiments (`results/lendingclub/`)

| File pattern | Location | Description |
|---|---|---|
| `lendingclub_summary_{policy}_retrain_{tag}.csv` | `results/lendingclub/csv/` | One summary row per run for that policy |
| `lendingclub_summary_plot_{policy}_retrain_{tag}.png` | `results/lendingclub/plots/` | 2×3 dashboard PNG for that policy |

Where `{tag}` is `3seed`, `10seed`, `ExtremeLatency_3seed`, or `ExtremeLatency_10seed`.

> **Note:** Only summary CSVs and dashboard PNGs are present on the `develop` and `main` branches. Per-run artifacts (JSON results, per-sample CSVs in `per_run/`) remain in their respective experiment branches only due to their large size.

> **Tip:** For detailed interpretation of the CSV columns and dashboard panels, see [results_interpretation_guide.md](results_interpretation_guide.md). For the full list of experiment phases and run counts, see [experiment_scope.md](experiment_scope.md).
