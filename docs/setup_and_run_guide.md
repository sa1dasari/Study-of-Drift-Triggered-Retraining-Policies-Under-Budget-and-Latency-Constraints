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
├── experiments/                     ← CLI entry points for all experiment runs
│   ├── main.py                      ← Synthetic experiments (with partial_fit)
│   ├── main_no_partial_fit.py       ← Synthetic experiments (NO partial_fit, static model)
│   ├── luflow_main.py               ← LUFlow experiments (with partial_fit)
│   ├── luflow_main_no_partial_fit.py← LUFlow experiments (NO partial_fit)
│   ├── lendingclub_main.py          ← LendingClub experiments (with partial_fit)
│   └── lendingclub_main_no_partial_fit.py ← LendingClub experiments (NO partial_fit)
├── cross_policy_comparison.py       ← Cross-policy head-to-head comparison
├── luflow_fitness_check.py          ← LUFlow dataset suitability gate checks
├── lendingclub_fitness_check.py     ← LendingClub dataset suitability gate checks
├── plot_summary.py                  ← Generates dashboard PNGs (called automatically by experiment scripts)
├── docs/
│   └── requirements.txt             ← Python dependencies
├── src/
│   ├── data/synthetic_data_drift_generator.py ← Synthetic data stream with concept drift
│   ├── data/LUFlow_Network_Intrusion/
│   │   └── datasets/                ← 28 day-CSVs (downloaded separately)
│   ├── data/LendingClub_Loan_Data/
│   │   ├── lendingclub_loader.py    ← LendingClub CSV loader & preprocessor
│   │   └── datasets/                ← accepted_2007_to_2018Q4.csv (downloaded separately)
│   ├── models/base_model.py         ← SGDClassifier wrapper
│   ├── policies/
│   │   ├── periodic.py              ← PeriodicPolicy
│   │   ├── error_threshold_policy.py← ErrorThresholdPolicy
│   │   ├── drift_triggered_policy.py← DriftTriggeredPolicy (ADWIN)
│   │   └── never_retrain_policy.py  ← NeverRetrainPolicy (baseline)
│   ├── runner/
│   │   ├── experiment_runner.py     ← Streaming event loop (with partial_fit)
│   │   └── experiment_runner_no_partial_fit.py ← Streaming event loop (NO partial_fit)
│   └── evaluation/
│       ├── metrics.py               ← MetricsTracker
│       ├── results_export.py        ← CSV / JSON exporters
│       └── plot_results.py          ← Per-run timeline plots
├── results_with_retrain/            ← Results from experiments WITH partial_fit
│   ├── synthetic/
│   │   ├── csv/                     ← Summary CSVs
│   │   ├── plots/                   ← Dashboard PNGs
│   │   └── per_run/                 ← Per-run JSONs and per-sample CSVs
│   ├── luflow/
│   │   ├── csv/
│   │   ├── plots/
│   │   └── per_run/
│   ├── lendingclub/
│   │   ├── csv/
│   │   ├── plots/
│   │   └── per_run/
│   └── cross_policy_comparison/     ← Head-to-head cross-policy outputs
└── results_without_retrain/         ← Results from experiments WITHOUT partial_fit
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
    └── cross_policy_comparison/     ← Head-to-head cross-policy outputs
```

---

## 4. Running Experiments

Two experiment modes are available:
- **With partial_fit** — the model receives incremental updates on every sample. Use the standard entry points (`main.py`, `luflow_main.py`, `lendingclub_main.py`). Results go to `results_with_retrain/`.
- **Without partial_fit** — the model is frozen between explicit retrains. Use the `_no_partial_fit` entry points. Results go to `results_without_retrain/`.

### Synthetic Experiments

#### With partial_fit

```
python experiments/main.py [--policy POLICY] [--seeds N]
```

| Flag | Values | Default | Description |
|---|---|---|---|
| `--policy` | `periodic`, `error_threshold`, `drift_triggered`, `no_retrain`, `all` | `all` | Which retraining policy to sweep. `all` runs all four sequentially. |
| `--seeds` | `3`, `10` | `10` | Seed set size. `3` = `[42, 123, 456]`. `10` = `[42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]`. |

#### Without partial_fit (static model)

```
python experiments/main_no_partial_fit.py [--policy POLICY] [--seeds N]
```

Same flags as above. Results are written to `results_without_retrain/synthetic/`.

#### Examples

```bash
# With partial_fit
python experiments/main.py                                    # all 4 policies, 10 seeds
python experiments/main.py --policy periodic                  # periodic only, 10 seeds
python experiments/main.py --policy drift_triggered --seeds 3 # drift-triggered only, 3 seeds

# Without partial_fit (static model)
python experiments/main_no_partial_fit.py                                    # all 4 policies, 10 seeds
python experiments/main_no_partial_fit.py --policy periodic                  # periodic only, 10 seeds
python experiments/main_no_partial_fit.py --policy drift_triggered --seeds 3 # drift-triggered only, 3 seeds
```

### LUFlow Real-World Experiments

Before running, download the 28 LUFlow day-CSVs into `src/data/LUFlow_Network_Intrusion/datasets/`. See [src/data/LUFlow_Network_Intrusion/README.md](../src/data/LUFlow_Network_Intrusion/README.md) for download instructions.

#### With partial_fit

```
python experiments/luflow_main.py [--policy POLICY]
```

#### Without partial_fit (static model)

```
python experiments/luflow_main_no_partial_fit.py [--policy POLICY]
```

| Flag | Values | Default | Description |
|---|---|---|---|
| `--policy` | `periodic`, `error_threshold`, `drift_triggered`, `no_retrain`, `all` | `all` | Which retraining policy to sweep. |

#### Examples

```bash
# With partial_fit
python experiments/luflow_main.py                             # all 4 policies (252 runs)
python experiments/luflow_main.py --policy drift_triggered    # drift-triggered only (81 runs)

# Without partial_fit
python experiments/luflow_main_no_partial_fit.py              # all 4 policies (252 runs)
python experiments/luflow_main_no_partial_fit.py --policy no_retrain # baseline only (9 runs)
```

### LendingClub Real-World Experiments

Before running, download the LendingClub accepted-loans CSV into `src/data/LendingClub_Loan_Data/datasets/`. See [src/data/LendingClub_Loan_Data/README.md](../src/data/LendingClub_Loan_Data/README.md) for download instructions.

#### With partial_fit

```
python experiments/lendingclub_main.py [--policy POLICY]
```

#### Without partial_fit (static model)

```
python experiments/lendingclub_main_no_partial_fit.py [--policy POLICY]
```

| Flag | Values | Default | Description |
|---|---|---|---|
| `--policy` | `periodic`, `error_threshold`, `drift_triggered`, `no_retrain`, `all` | `all` | Which retraining policy to sweep. |

#### Examples

```bash
# With partial_fit
python experiments/lendingclub_main.py                             # all 4 policies (252 runs)
python experiments/lendingclub_main.py --policy periodic           # periodic only (81 runs)

# Without partial_fit
python experiments/lendingclub_main_no_partial_fit.py              # all 4 policies (252 runs)
python experiments/lendingclub_main_no_partial_fit.py --policy periodic # periodic only (81 runs)
```

### What happens when you run it

For each selected policy, the experiment script:

1. Deletes the old summary CSV for that policy (clean start).
2. Iterates over every `(drift_type, budget, latency, seed)` combination (or just `(drift_type, seed)` for no-retrain).
3. For each run: generates data → builds model + policy → streams samples → exports JSON, per-sample CSV, and appends one row to the summary CSV.
4. Prints live progress with accuracy, retrain count, and ETA.
5. Generates a dashboard PNG after all runs complete.

The `_no_partial_fit` variants follow the same flow but skip the per-sample `partial_fit` call, so the model is static between explicit retrains.

> **Important:** Always run experiment scripts from the `experiments/` directory or the project root. For LUFlow experiments, the dataset CSVs must be downloaded first. For LendingClub experiments, the accepted-loans CSV must be downloaded first (see instructions above).

---

## 5. Output Files

After running experiments, results are organized into two top-level directories:
- **`results_with_retrain/`** — outputs from experiments **with** partial_fit (incremental learning).
- **`results_without_retrain/`** — outputs from experiments **without** partial_fit (static model).

Both directories share the same internal structure.

### Synthetic experiments

| File pattern | Location | Description |
|---|---|---|
| `summary_results_{policy}_retrain_{tag}.csv` | `{results_dir}/synthetic/csv/` | One summary row per run for that policy |
| `summary_results_plot_{policy}_retrain_{tag}.png` | `{results_dir}/synthetic/plots/` | 2×3 dashboard PNG for that policy |
| `summary_results_no_retrain_{tag}.csv` | `{results_dir}/synthetic/csv/` | Baseline summary (no budget/latency grid) |
| `summary_results_plot_no_retrain_{tag}.png` | `{results_dir}/synthetic/plots/` | 2×2 baseline dashboard |

### LUFlow experiments

| File pattern | Location | Description |
|---|---|---|
| `luflow_summary_{policy}_retrain_{tag}.csv` | `{results_dir}/luflow/csv/` | One summary row per run for that policy |
| `luflow_summary_plot_{policy}_retrain_{tag}.png` | `{results_dir}/luflow/plots/` | 2×3 dashboard PNG for that policy |

### LendingClub experiments

| File pattern | Location | Description |
|---|---|---|
| `lendingclub_summary_{policy}_retrain_{tag}.csv` | `{results_dir}/lendingclub/csv/` | One summary row per run for that policy |
| `lendingclub_summary_plot_{policy}_retrain_{tag}.png` | `{results_dir}/lendingclub/plots/` | 2×3 dashboard PNG for that policy |

Where `{results_dir}` is `results_with_retrain` or `results_without_retrain`, and `{tag}` is `3seed`, `10seed`, `ExtremeLatency_3seed`, or `ExtremeLatency_10seed`.

> **Note:** Only summary CSVs and dashboard PNGs are present on the `develop` and `main` branches. Per-run artifacts (JSON results, per-sample CSVs in `per_run/`) remain in their respective experiment branches only due to their large size.

> **Tip:** For detailed interpretation of the CSV columns and dashboard panels, see [results_interpretation_guide.md](results_interpretation_guide.md). For the full list of experiment phases and run counts, see [experiment_scope.md](experiment_scope.md).
