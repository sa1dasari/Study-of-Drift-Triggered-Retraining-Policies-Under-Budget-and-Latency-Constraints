# Experiment Scope & Configuration Matrix

## Overview

This document specifies every factor varied across experiment runs, the exact parameter values used, and the total number of configurations executed.
The study follows a **full-factorial design** — every combination of drift type, policy, budget, and latency is run with multiple random seeds.

Experiments were conducted in two modes:
- **With partial_fit (incremental learning):** The model receives `partial_fit` on every sample. Results stored in `results_with_retrain/`.
- **Without partial_fit (static model):** The model is frozen between explicit retrains. Results stored in `results_without_retrain/`.

### With partial_fit — Phases 1–6

Experiments were conducted in **five phases** plus **baseline runs**, followed by a **cross-policy comparison** phase:
- **Phase 1 (3 seeds):** 243 runs — initial exploration with per-configuration Git branches.
- **Phase 2 (10 seeds):** 810 runs — extended evaluation with per-policy batch Git branches.
- **No-Retrain Baseline (3 seeds):** 9 runs — accuracy floor with Phase 1 seeds.
- **No-Retrain Baseline (10 seeds):** 30 runs — accuracy floor with Phase 2 seeds.
- **Phase 3 — Extreme Latency (3 seeds):** 171 runs — 2 extreme latency levels (Near-Zero=3, Extreme-High=2050).
- **Phase 3 — Extreme Latency (10 seeds):** 570 runs — same 2 extreme levels with full 10-seed set.
- **Phase 4 — LUFlow Real-World Dataset:** 252 runs — 3 pool configs × 3 drift types × 3 budgets × 3 latencies × 3 policies + 9 baseline runs.
- **Phase 5 — LendingClub Real-World Dataset:** 252 runs — 3 year-pair configs × 3 drift types × 3 budgets × 3 latencies × 3 policies + 9 baseline runs.
- **Phase 6 — Cross-Policy Comparison:** Merges all per-policy summary CSVs and produces head-to-head comparison tables and figures across all three datasets. No new experiment runs — this phase analyses the 2,337 runs from Phases 1–5.
- **Combined total (with partial_fit): 2,337 experiment runs.**

### Without partial_fit — Phase 7

The same factorial grid was re-run using `ExperimentRunnerNoPartialFit`, which removes the per-sample `partial_fit` call. The model is frozen between explicit retrains, isolating the pure effect of the retraining policy.

- **Phase 7a — Synthetic (3 seeds):** 252 runs — 3 drift types × 3 budgets × 3 latencies × 3 seeds × 3 active policies + 9 no-retrain baseline.
- **Phase 7b — Synthetic (10 seeds):** 840 runs — same grid × 10 seeds + 30 baseline.
- **Phase 7c — LUFlow (3 seeds):** 252 runs — 3 pool configs × 3 drift types × 3 budgets × 3 latencies × 3 policies + 9 baseline.
- **Phase 7d — LendingClub (3 seeds):** 252 runs — 3 year-pair configs × 3 drift types × 3 budgets × 3 latencies × 3 policies + 9 baseline.
- **Phase 7e — Cross-Policy Comparison (without partial_fit):** Analysis only — merges summary CSVs from Phases 7a–7d.
- **Combined total (without partial_fit): 1,596 experiment runs.**

### Grand total across both modes: **3,933 experiment runs.**

---

## Factor 1 — Drift Types (3)

| Drift Type | Drift Point | Transition Window | Recurrence Period | Description |
|---|---|---|---|---|
| **Abrupt** | t = 5,000 | Instantaneous | N/A | Weight vector switches from `w₁` to `w₂` in a single step |
| **Gradual** | t = 5,000 | 1,000 steps (t ∈ [5000, 6000]) | N/A | Linear interpolation `(1−α)·w₁ + α·w₂` over 1,000 timesteps |
| **Recurring** | t = 5,000 | Instantaneous per switch | 1,000 steps | Concept alternates between `w₂` and `w₁` every 1,000 steps after drift point |

All drift types share the same pre-drift concept (`w₁`) for `t ∈ [0, 5000)`, making pre-drift accuracy a useful sanity check.

---

## Factor 2 — Retraining Policies (3)

| Policy | Key Parameters | Trigger Mechanism |
|---|---|---|
| **Periodic** | `interval` ∈ {500, 1000, 2000} (derived from budget) | Retrain every `interval` timesteps on a fixed schedule |
| **Error-Threshold** | `error_threshold = 0.27`, `window_size = 200` | Retrain when rolling error rate over last 200 predictions exceeds 27 % |
| **Drift-Triggered (ADWIN)** | `delta = 0.002`, `window_size = 500`, `min_samples = 300` | Retrain when ADWIN detects a statistically significant shift in error distribution |
| **No-Retrain (Baseline)** | None — budget = 0, latency = 0 | Never retrains; in with-partial-fit mode, model adapts only via `partial_fit`; in without-partial-fit mode, model is frozen after initial training |

> The **No-Retrain baseline** has no budget or latency grid — it is a single-configuration policy run across 3 drift types × N seeds. It provides the accuracy floor that makes all other policies' numbers interpretable.

## Factor 3 — Budget Levels (3)

| Level | K (max retrains) | Motivation |
|---|---|---|
| **Low** | 5 | Simulates a very cost-constrained environment; only 5 full retrains allowed across 10,000 samples |
| **Medium** | 10 | A moderate budget that balances cost and adaptability |
| **High** | 20 | A generous budget permitting frequent model refreshes |

---

## Factor 4 — Latency Levels (3 original + 2 extreme)

### Original Levels (Phases 1 & 2)

| Level | Retrain Latency (steps) | Deploy Latency (steps) | Total Latency (steps) | Interpretation |
|---|---|---|---|---|
| **Low** | 10 | 1 | 11 | Near real-time — fast retraining and near-instant deployment |
| **Medium** | 100 | 5 | 105 | Moderate — e.g., nightly batch retrain with brief staging |
| **High** | 500 | 20 | 520 | Heavy — large model retrain with substantial deployment pipeline overhead |

### Extreme Levels (Phase 3)

| Level | Retrain Latency (steps) | Deploy Latency (steps) | Total Latency (steps) | Interpretation |
|---|---|---|---|---|
| **Near-Zero** | 2 | 1 | 3 | Removes latency as a factor entirely — isolates pure policy behaviour |
| **Extreme-High** | 2,000 | 50 | 2,050 | Forces even K=20 periodic to execute almost no retrains post-drift |

During each latency window the model continues to serve predictions on stale weights, and no new retrain can be initiated.

---

## Factor 5 — Random Seeds

### Phase 1 — 3 Seeds

| Seed | Purpose |
|---|---|
| 42 | Primary reproducibility seed |
| 123 | Alternate seed for variance estimation |
| 456 | Third seed for variance estimation |

### Phase 2 — 10 Seeds

| Seed | Purpose |
|---|---|
| 42 | Primary reproducibility seed |
| 123 | Variance estimation |
| 456 | Variance estimation |
| 789 | Extended variance estimation |
| 1011 | Extended variance estimation |
| 1213 | Extended variance estimation |
| 1415 | Extended variance estimation |
| 1617 | Extended variance estimation |
| 1819 | Extended variance estimation |
| 2021 | Extended variance estimation |

Each seed generates a unique pair of weight vectors (`w₁`, `w₂`) and a unique feature matrix, so results vary across seeds even for the same drift type. Phase 1 (3 seeds) provided an initial view of seed sensitivity. Phase 2 (10 seeds) provides stronger variance estimates and reduces the impact of outlier seeds (e.g., the ADWIN 0-detection issue seen with seeds 42 and 123 under abrupt drift).

---

## Full Configuration Matrix

### Phase 1 — 3 Seeds (243 runs)

```
3 drift types  ×  3 budget levels  ×  3 latency levels  =  27 unique configs per policy
27 configs  ×  3 seeds  =  81 experiment runs per policy
81 runs  ×  3 policies  =  243 total experiment runs
```

Each configuration was run in its own Git feature branch:
- **Branch naming:** `exp/<drift>-<policy>-<Budget>budget-<Latency>Latency`
- **Example:** `exp/gradual-drift-triggered-Medbudget-HighLatency`

Cumulative results per policy were aggregated in dedicated develop branches:
- `develop_3seed_periodic_retrain`
- `develop_3seed_error_threshold_retrain`
- `develop_3seed_drift_triggered_retrain`

### Phase 2 — 10 Seeds (810 runs)

```
3 drift types  ×  3 budget levels  ×  3 latency levels  =  27 unique configs per policy
27 configs  ×  10 seeds  =  270 experiment runs per policy
270 runs  ×  3 policies  =  810 total experiment runs
```

The 810 runs were executed in **3 batches of 270 runs** (one batch per policy), rather than per-configuration branches:
- `develop-10Seed-periodic-retrain-tests` (270 runs)
- `develop-10Seed-error-threshold-retrain-tests` (270 runs)
- `develop-10Seed-drift-triggered-retrain-tests` (270 runs)

### No-Retrain Baseline — 3 Seeds (9 runs)

```
3 drift types  ×  3 seeds  =  9 experiment runs
Budget  = 0  (always)
Latency = 0  (always)
```

### No-Retrain Baseline — 10 Seeds (30 runs)

```
3 drift types  ×  10 seeds  =  30 experiment runs
Budget  = 0  (always)
Latency = 0  (always)
```

The no-retrain baseline has **no budget or latency grid** — the model is never retrained from scratch. In with-partial-fit mode, it uses only `partial_fit` (incremental learning). In without-partial-fit mode, the model is completely frozen after initial training. This provides the **accuracy floor** that makes all other policies' accuracy numbers interpretable.

- **Branch:** `develop_NoRetrain_NoBudget_NoLatency`

### Phase 3 — Extreme Latency, 3 Seeds (171 runs)

```
3 drift types  ×  3 budget levels  ×  2 extreme latency levels  =  18 unique configs per policy
18 configs  ×  3 seeds  =  54 experiment runs per active policy
54 runs  ×  3 active policies  =  162 active runs  +  9 baseline runs  =  171 total
```

### Phase 3 — Extreme Latency, 10 Seeds (570 runs)

```
3 drift types  ×  3 budget levels  ×  2 extreme latency levels  =  18 unique configs per policy
18 configs  ×  10 seeds  =  180 experiment runs per active policy
180 runs  ×  3 active policies  =  540 active runs  +  30 baseline runs  =  570 total
```

### Phase 4 — LUFlow Real-World Dataset (252 runs)

```
3 pool configs  ×  3 drift types  ×  3 budget levels  ×  3 latency levels  =  81 unique configs per policy
81 configs  ×  3 active policies  =  243 active runs  +  9 baseline runs  =  252 total
```

The LUFlow experiment uses real-world network intrusion detection data from Lancaster University (28 day-CSVs, ~21 M rows). Instead of random seeds, three **pool-pair configurations** define pre-/post-drift data by filtering days on their malicious-class percentage. Streams are 50,000 samples with drift at t = 25,000.

| Config | Pre-drift pool | Post-drift pool | Shift type |
|--------|---------------|----------------|------------|
| Pool 1 | Jan 2021 low-mal days (≤ 5 %) | Feb 2021 high-mal days (≥ 15 %) | Class-balance shift |
| Pool 2 | Jan 2021 high-mal days (≥ 15 %) | Feb 2021 high-mal days (≥ 15 %) | Feature drift (similar balance) |
| Pool 3 | Jan 2021 low-mal days (≤ 5 %) | Feb 2021 extreme-mal days (≥ 40 %) | Extreme class-balance shift |

Policy parameters were re-calibrated on LUFlow data:
- **Periodic:** interval = 50,000 / K
- **Error-Threshold:** threshold = 0.20, window_size = 200
- **Drift-Triggered (ADWIN):** δ = 0.005, window_size = 500, min_samples = 100

- **Branch:** `develop_LUFlow_Dataset`

### Phase 5 — LendingClub Real-World Dataset (252 runs)

```
3 year-pair configs  ×  3 drift types  ×  3 budget levels  ×  3 latency levels  =  81 unique configs per policy
81 configs  ×  3 active policies  =  243 active runs  +  9 baseline runs  =  252 total
```

The LendingClub experiment uses real-world loan default data from Kaggle (accepted loans 2007–2018, ~1.35 M rows after filtering). Instead of random seeds, three **year-pair configurations** define pre-/post-drift data by selecting different calendar-year cohorts. The drift arises from real-world feature-space drift caused by LendingClub's changing underwriting policy between 2012 and 2016. Streams are 50,000 samples with drift at t = 25,000.

| Config | Pre-drift year | Post-drift year | Gap | Shift description |
|--------|---------------|----------------|-----|-------------------|
| Seed 1 | 2013 | 2016 | 3 years | Maximum policy shift |
| Seed 2 | 2014 | 2016 | 2 years | Moderate drift |
| Seed 3 | 2013 | 2015 | 2 years | Different cohort pair |

Policy parameters were re-calibrated on LendingClub data:
- **Periodic:** interval = 50,000 / K
- **Error-Threshold:** threshold = 0.20, window_size = 200
- **Drift-Triggered (ADWIN):** δ = 0.005, window_size = 500, min_samples = 100

- **Branch:** `develop_LendingClub_Dataset`

### Combined Summary (With partial_fit)

| Dimension | Values | Count |
|---|---|---|
| Drift types | abrupt, gradual, recurring | 3 |
| Policies | periodic, error_threshold, drift_triggered, **no_retrain** | 4 |
| Budgets (K) | 5, 10, 20 *(N/A for no_retrain)* | 3 |
| Latency levels (Phase 1 & 2) | Low (11), Medium (105), High (520) *(N/A for no_retrain)* | 3 |
| Latency levels (Phase 3) | Near-Zero (3), Extreme-High (2050) *(N/A for no_retrain)* | 2 |
| Seeds (Phase 1 / Phase 3-3seed) | 42, 123, 456 | 3 |
| Seeds (Phase 2 / Phase 3-10seed) | 42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021 | 10 |
| LUFlow pool configs (Phase 4) | Pool 1, Pool 2, Pool 3 | 3 |
| LendingClub year-pair configs (Phase 5) | Seed 1 (2013→2016), Seed 2 (2014→2016), Seed 3 (2013→2015) | 3 |
| **Total Phase 1 runs** | | **243** |
| **Total Phase 2 runs** | | **810** |
| **Total No-Retrain Baseline runs** | | **39** (9 + 30) |
| **Total Phase 3 runs (Extreme Latency)** | | **741** (171 + 570) |
| **Total Phase 4 runs (LUFlow)** | | **252** (243 + 9) |
| **Total Phase 5 runs (LendingClub)** | | **252** (243 + 9) |
| **Phase 6 (Cross-Policy Comparison)** | | **0** (analysis only) |
| **Grand total (with partial_fit)** | | **2,337** |

### Phase 7 — Without partial_fit (Static Model)

All experiment grids below use `ExperimentRunnerNoPartialFit`. The model is frozen between explicit retrains. Entry points live in `experiments/`:
- `experiments/main_no_partial_fit.py` (synthetic)
- `experiments/luflow_main_no_partial_fit.py` (LUFlow)
- `experiments/lendingclub_main_no_partial_fit.py` (LendingClub)

Results are written to `results_without_retrain/` (never overwriting `results_with_retrain/`).

#### Phase 7a — Synthetic, 3 Seeds (252 runs)

```
3 drift types  ×  3 budget levels  ×  3 latency levels  =  27 unique configs per policy
27 configs  ×  3 seeds  =  81 experiment runs per active policy
81 runs  ×  3 active policies  =  243 active runs  +  9 baseline runs  =  252 total
```

#### Phase 7b — Synthetic, 10 Seeds (840 runs)

```
3 drift types  ×  3 budget levels  ×  3 latency levels  =  27 unique configs per policy
27 configs  ×  10 seeds  =  270 experiment runs per active policy
270 runs  ×  3 active policies  =  810 active runs  +  30 baseline runs  =  840 total
```

#### Phase 7c — LUFlow, 3 Seeds (252 runs)

```
3 pool configs  ×  3 drift types  ×  3 budget levels  ×  3 latency levels  =  81 unique configs per policy
81 configs  ×  3 active policies  =  243 active runs  +  9 baseline runs  =  252 total
```

#### Phase 7d — LendingClub, 3 Seeds (252 runs)

```
3 year-pair configs  ×  3 drift types  ×  3 budget levels  ×  3 latency levels  =  81 unique configs per policy
81 configs  ×  3 active policies  =  243 active runs  +  9 baseline runs  =  252 total
```

#### Phase 7e — Cross-Policy Comparison (Analysis Only)

No new experiments. `cross_policy_comparison.py` reads summary CSVs from `results_without_retrain/` and produces head-to-head comparison outputs to `results_without_retrain/cross_policy_comparison/`.

#### Combined Summary (Without partial_fit)

| Phase | Runs | Description |
|---|---|---|
| Phase 7a — Synthetic (3 seeds) | 252 | 3 policies × 3 drifts × 3 budgets × 3 latencies × 3 seeds + 9 baseline |
| Phase 7b — Synthetic (10 seeds) | 840 | 3 policies × 3 drifts × 3 budgets × 3 latencies × 10 seeds + 30 baseline |
| Phase 7c — LUFlow (3 seeds) | 252 | 3 policies × 3 drifts × 3 budgets × 3 latencies × 3 pool configs + 9 baseline |
| Phase 7d — LendingClub (3 seeds) | 252 | 3 policies × 3 drifts × 3 budgets × 3 latencies × 3 year-pair configs + 9 baseline |
| Phase 7e — Cross-Policy Comparison | 0 | Analysis only |
| **Grand total (without partial_fit)** | **1,596** | |

The **`main`** branch contains all merged results from both experiment modes.

---

## Output Artifacts per Policy

> **Note:** Results from with-partial-fit experiments are stored in `results_with_retrain/`. Results from without-partial-fit experiments are stored in `results_without_retrain/`. The sub-directory structure is identical.

### With partial_fit — Phases 1–5

#### Phase 1 (3-seed) — 81 runs per policy

| Artifact | Path | Description |
|---|---|---|
| JSON result | `results_with_retrain/synthetic/per_run/<policy_dir>/run_<run_tag>.json` | Full config + structured metrics per run |
| Per-sample CSV | `results_with_retrain/synthetic/per_run/<policy_dir>/per_sample_<run_tag>.csv` | Per-timestep accuracy, error, latency flags |
| Summary CSV row | `results_with_retrain/synthetic/csv/summary_results_{policy}_retrain_3seed.csv` | One row appended per run; 81 rows total per policy |

#### Phase 2 (10-seed) — 270 runs per policy

| Artifact | Path | Description |
|---|---|---|
| JSON result | `results_with_retrain/synthetic/per_run/<policy_dir>/run_<run_tag>.json` | Full config + structured metrics per run |
| Per-sample CSV | `results_with_retrain/synthetic/per_run/<policy_dir>/per_sample_<run_tag>.csv` | Per-timestep accuracy, error, latency flags |
| Summary CSV row | `results_with_retrain/synthetic/csv/summary_results_{policy}_retrain_10seed.csv` | One row appended per run; 270 rows total per policy |

#### No-Retrain Baseline — 3-seed (9 runs) and 10-seed (30 runs)

| Artifact | Path | Description |
|---|---|---|
| JSON result | `results_with_retrain/synthetic/per_run/no_retrain_{N}seed_3drift/run_<drift>_s<seed>.json` | Full config + structured metrics per run |
| Per-sample CSV | `results_with_retrain/synthetic/per_run/no_retrain_{N}seed_3drift/per_sample_<drift>_s<seed>.csv` | Per-timestep accuracy and error (no latency flags) |
| Summary CSV (3-seed) | `results_with_retrain/synthetic/csv/summary_results_no_retrain_3seed.csv` | One row per run; 9 rows total |
| Summary CSV (10-seed) | `results_with_retrain/synthetic/csv/summary_results_no_retrain_10seed.csv` | One row per run; 30 rows total |

#### Phase 3 — Extreme Latency (3-seed: 54 runs per policy, 10-seed: 180 runs per policy)

| Artifact | Path | Description |
|---|---|---|
| JSON result | `results_with_retrain/synthetic/per_run/<policy>_ExtremeLatency_{N}seed/run_<run_tag>.json` | Full config + structured metrics per run |
| Per-sample CSV | `results_with_retrain/synthetic/per_run/<policy>_ExtremeLatency_{N}seed/per_sample_<run_tag>.csv` | Per-timestep accuracy, error, latency flags |
| Summary CSV (3-seed) | `results_with_retrain/synthetic/csv/summary_results_{policy}_retrain_ExtremeLatency_3seed.csv` | One row per run; 54 rows per active policy |
| Summary CSV (10-seed) | `results_with_retrain/synthetic/csv/summary_results_{policy}_retrain_ExtremeLatency_10seed.csv` | One row per run; 180 rows per active policy |
| Baseline CSV (3-seed) | `results_with_retrain/synthetic/csv/summary_results_no_retrain_ExtremeLatency_3seed.csv` | One row per run; 9 rows |
| Baseline CSV (10-seed) | `results_with_retrain/synthetic/csv/summary_results_no_retrain_ExtremeLatency_10seed.csv` | One row per run; 30 rows |

#### Visualisation Artifacts (With partial_fit)

| File | Description |
|---|---|
| `results_with_retrain/synthetic/plots/summary_results_plot_periodic_retrain_3seed.png` | 2 × 3 dashboard for periodic policy (Phase 1, 3 seeds) |
| `results_with_retrain/synthetic/plots/summary_results_plot_periodic_retrain_10seed.png` | 2 × 3 dashboard for periodic policy (Phase 2, 10 seeds) |
| `results_with_retrain/synthetic/plots/summary_results_plot_error_threshold_retrain_3seed.png` | 2 × 3 dashboard for error-threshold policy (Phase 1, 3 seeds) |
| `results_with_retrain/synthetic/plots/summary_results_plot_error_threshold_retrain_10seed.png` | 2 × 3 dashboard for error-threshold policy (Phase 2, 10 seeds) |
| `results_with_retrain/synthetic/plots/summary_results_plot_drift_triggered_retrain_3seed.png` | 2 × 3 dashboard for drift-triggered (ADWIN) policy (Phase 1, 3 seeds) |
| `results_with_retrain/synthetic/plots/summary_results_plot_drift_triggered_retrain_10seed.png` | 2 × 3 dashboard for drift-triggered (ADWIN) policy (Phase 2, 10 seeds) |
| `results_with_retrain/synthetic/plots/summary_results_plot_no_retrain_3seed.png` | 2 × 2 baseline dashboard for no-retrain policy (3 seeds) |
| `results_with_retrain/synthetic/plots/summary_results_plot_no_retrain_10seed.png` | 2 × 2 baseline dashboard for no-retrain policy (10 seeds) |
| `results_with_retrain/synthetic/plots/summary_results_plot_periodic_retrain_ExtremeLatency_3seed.png` | 2 × 3 dashboard for periodic policy (Phase 3, 3 seeds, extreme latency) |
| `results_with_retrain/synthetic/plots/summary_results_plot_periodic_retrain_ExtremeLatency_10seed.png` | 2 × 3 dashboard for periodic policy (Phase 3, 10 seeds, extreme latency) |
| `results_with_retrain/synthetic/plots/summary_results_plot_error_threshold_retrain_ExtremeLatency_3seed.png` | 2 × 3 dashboard for error-threshold policy (Phase 3, 3 seeds, extreme latency) |
| `results_with_retrain/synthetic/plots/summary_results_plot_error_threshold_retrain_ExtremeLatency_10seed.png` | 2 × 3 dashboard for error-threshold policy (Phase 3, 10 seeds, extreme latency) |
| `results_with_retrain/synthetic/plots/summary_results_plot_drift_triggered_retrain_ExtremeLatency_3seed.png` | 2 × 3 dashboard for drift-triggered (ADWIN) policy (Phase 3, 3 seeds, extreme latency) |
| `results_with_retrain/synthetic/plots/summary_results_plot_drift_triggered_retrain_ExtremeLatency_10seed.png` | 2 × 3 dashboard for drift-triggered (ADWIN) policy (Phase 3, 10 seeds, extreme latency) |
| `results_with_retrain/synthetic/plots/summary_results_plot_no_retrain_ExtremeLatency_3seed.png` | 2 × 2 baseline dashboard for no-retrain policy (Phase 3, 3 seeds, extreme latency) |
| `results_with_retrain/synthetic/plots/summary_results_plot_no_retrain_ExtremeLatency_10seed.png` | 2 × 2 baseline dashboard for no-retrain policy (Phase 3, 10 seeds, extreme latency) |

#### Phase 4 — LUFlow Output Artifacts (252 runs)

| Artifact | Path | Description |
|---|---|---|
| JSON result | `results_with_retrain/luflow/per_run/luflow_{policy}_3seed/run_<run_tag>.json` | Full config + structured metrics per run |
| Per-sample CSV | `results_with_retrain/luflow/per_run/luflow_{policy}_3seed/per_sample_<run_tag>.csv` | Per-timestep accuracy, error, latency flags |
| Summary CSV | `results_with_retrain/luflow/csv/luflow_summary_{policy}_retrain_3seed.csv` | One row per run; 81 rows per active policy |
| Baseline CSV | `results_with_retrain/luflow/csv/luflow_summary_no_retrain_3seed.csv` | One row per run; 9 rows |

#### LUFlow Visualisation Artifacts (With partial_fit)

| File | Description |
|---|---|
| `results_with_retrain/luflow/plots/luflow_summary_plot_periodic_retrain_3seed.png` | 2 × 3 dashboard for periodic policy (LUFlow) |
| `results_with_retrain/luflow/plots/luflow_summary_plot_error_threshold_retrain_3seed.png` | 2 × 3 dashboard for error-threshold policy (LUFlow) |
| `results_with_retrain/luflow/plots/luflow_summary_plot_drift_triggered_retrain_3seed.png` | 2 × 3 dashboard for drift-triggered (ADWIN) policy (LUFlow) |
| `results_with_retrain/luflow/plots/luflow_summary_plot_no_retrain_3seed.png` | 2 × 2 baseline dashboard for no-retrain policy (LUFlow) |

#### Phase 5 — LendingClub Output Artifacts (252 runs)

| Artifact | Path | Description |
|---|---|---|
| JSON result | `results_with_retrain/lendingclub/per_run/lendingclub_{policy}_3seed/run_<run_tag>.json` | Full config + structured metrics per run |
| Per-sample CSV | `results_with_retrain/lendingclub/per_run/lendingclub_{policy}_3seed/per_sample_<run_tag>.csv` | Per-timestep accuracy, error, latency flags |
| Summary CSV | `results_with_retrain/lendingclub/csv/lendingclub_summary_{policy}_retrain_3seed.csv` | One row per run; 81 rows per active policy |
| Baseline CSV | `results_with_retrain/lendingclub/csv/lendingclub_summary_no_retrain_3seed.csv` | One row per run; 9 rows |

#### LendingClub Visualisation Artifacts (With partial_fit)

| File | Description |
|---|---|
| `results_with_retrain/lendingclub/plots/lendingclub_summary_plot_periodic_retrain_3seed.png` | 2 × 3 dashboard for periodic policy (LendingClub) |
| `results_with_retrain/lendingclub/plots/lendingclub_summary_plot_error_threshold_retrain_3seed.png` | 2 × 3 dashboard for error-threshold policy (LendingClub) |
| `results_with_retrain/lendingclub/plots/lendingclub_summary_plot_drift_triggered_retrain_3seed.png` | 2 × 3 dashboard for drift-triggered (ADWIN) policy (LendingClub) |
| `results_with_retrain/lendingclub/plots/lendingclub_summary_plot_no_retrain_3seed.png` | 2 × 2 baseline dashboard for no-retrain policy (LendingClub) |

#### Phase 6 — Cross-Policy Comparison (With partial_fit, Analysis Only)

Phase 6 runs no new experiments. It merges all per-policy summary CSVs from Phases 1–5 and produces head-to-head comparison outputs across all four policies for each dataset, plus a cross-dataset summary.

All per-dataset outputs are saved to `results_with_retrain/cross_policy_comparison/{dataset}/`.

### Without partial_fit — Phase 7

#### Synthetic Output Artifacts

| Artifact | Path | Description |
|---|---|---|
| JSON result | `results_without_retrain/synthetic/per_run/<policy_dir>/run_<run_tag>.json` | Full config + structured metrics per run |
| Per-sample CSV | `results_without_retrain/synthetic/per_run/<policy_dir>/per_sample_<run_tag>.csv` | Per-timestep accuracy, error, latency flags |
| Summary CSV (3-seed) | `results_without_retrain/synthetic/csv/summary_results_{policy}_retrain_3seed.csv` | One row per run |
| Summary CSV (10-seed) | `results_without_retrain/synthetic/csv/summary_results_{policy}_retrain_10seed.csv` | One row per run |
| Baseline CSV (3-seed) | `results_without_retrain/synthetic/csv/summary_results_no_retrain_3seed.csv` | One row per run |
| Baseline CSV (10-seed) | `results_without_retrain/synthetic/csv/summary_results_no_retrain_10seed.csv` | One row per run |

#### Synthetic Visualisation Artifacts (Without partial_fit)

| File | Description |
|---|---|
| `results_without_retrain/synthetic/plots/summary_results_plot_periodic_retrain_3seed.png` | Dashboard for periodic policy (3 seeds, no partial_fit) |
| `results_without_retrain/synthetic/plots/summary_results_plot_periodic_retrain_10seed.png` | Dashboard for periodic policy (10 seeds, no partial_fit) |
| `results_without_retrain/synthetic/plots/summary_results_plot_error_threshold_retrain_3seed.png` | Dashboard for error-threshold policy (3 seeds, no partial_fit) |
| `results_without_retrain/synthetic/plots/summary_results_plot_error_threshold_retrain_10seed.png` | Dashboard for error-threshold policy (10 seeds, no partial_fit) |
| `results_without_retrain/synthetic/plots/summary_results_plot_drift_triggered_retrain_3seed.png` | Dashboard for drift-triggered policy (3 seeds, no partial_fit) |
| `results_without_retrain/synthetic/plots/summary_results_plot_drift_triggered_retrain_10seed.png` | Dashboard for drift-triggered policy (10 seeds, no partial_fit) |
| `results_without_retrain/synthetic/plots/summary_results_plot_no_retrain_3seed.png` | Baseline dashboard (3 seeds, no partial_fit) |
| `results_without_retrain/synthetic/plots/summary_results_plot_no_retrain_10seed.png` | Baseline dashboard (10 seeds, no partial_fit) |

#### LUFlow Output Artifacts (Without partial_fit)

| Artifact | Path | Description |
|---|---|---|
| Summary CSV | `results_without_retrain/luflow/csv/luflow_summary_{policy}_retrain_3seed.csv` | One row per run |
| Baseline CSV | `results_without_retrain/luflow/csv/luflow_summary_no_retrain_3seed.csv` | One row per run |

#### LUFlow Visualisation Artifacts (Without partial_fit)

| File | Description |
|---|---|
| `results_without_retrain/luflow/plots/luflow_summary_plot_periodic_retrain_3seed.png` | Dashboard for periodic policy (LUFlow, no partial_fit) |
| `results_without_retrain/luflow/plots/luflow_summary_plot_error_threshold_retrain_3seed.png` | Dashboard for error-threshold policy (LUFlow, no partial_fit) |
| `results_without_retrain/luflow/plots/luflow_summary_plot_drift_triggered_retrain_3seed.png` | Dashboard for drift-triggered policy (LUFlow, no partial_fit) |
| `results_without_retrain/luflow/plots/luflow_summary_plot_no_retrain_3seed.png` | Baseline dashboard (LUFlow, no partial_fit) |

#### LendingClub Output Artifacts (Without partial_fit)

| Artifact | Path | Description |
|---|---|---|
| Summary CSV | `results_without_retrain/lendingclub/csv/lendingclub_summary_{policy}_retrain_3seed.csv` | One row per run |
| Baseline CSV | `results_without_retrain/lendingclub/csv/lendingclub_summary_no_retrain_3seed.csv` | One row per run |

#### LendingClub Visualisation Artifacts (Without partial_fit)

| File | Description |
|---|---|
| `results_without_retrain/lendingclub/plots/lendingclub_summary_plot_periodic_retrain_3seed.png` | Dashboard for periodic policy (LendingClub, no partial_fit) |
| `results_without_retrain/lendingclub/plots/lendingclub_summary_plot_error_threshold_retrain_3seed.png` | Dashboard for error-threshold policy (LendingClub, no partial_fit) |
| `results_without_retrain/lendingclub/plots/lendingclub_summary_plot_drift_triggered_retrain_3seed.png` | Dashboard for drift-triggered policy (LendingClub, no partial_fit) |
| `results_without_retrain/lendingclub/plots/lendingclub_summary_plot_no_retrain_3seed.png` | Baseline dashboard (LendingClub, no partial_fit) |

#### Phase 7e — Cross-Policy Comparison (Without partial_fit, Analysis Only)

**Script:** `cross_policy_comparison.py` (reads from `results_without_retrain/`)

```bash
python cross_policy_comparison.py                      # all 3 datasets
python cross_policy_comparison.py --dataset synthetic   # synthetic only
python cross_policy_comparison.py --dataset luflow      # LUFlow only
python cross_policy_comparison.py --dataset lendingclub # LendingClub only
python cross_policy_comparison.py --seeds 3             # force 3-seed CSVs
```

Per-dataset and cross-dataset outputs are saved to `results_without_retrain/cross_policy_comparison/`.

#### Per-Dataset Output Artifacts

For each dataset (`synthetic`, `luflow`, `lendingclub`), four comparison outputs are produced:

| # | Output | Table CSV | Figure PNG | Description |
|---|--------|-----------|------------|-------------|
| 1 | Post-Drift Accuracy | `table1_postdrift_accuracy_{dataset}.csv` | `fig1_postdrift_heatmap_{dataset}.png` | Mean post-drift accuracy by policy × drift type (averaged over seeds, budgets, latencies) |
| 2 | Budget-Faceted | `table2_budget_faceted_{dataset}.csv` | `fig2_budget_faceted_{dataset}.png` | Same breakdown faceted by K=5/10/20; no-retrain baseline as dashed reference |
| 3 | Budget Efficiency | `table3_budget_efficiency_{dataset}.csv` | `fig3_budget_efficiency_{dataset}.png` | Accuracy gain per retrain after drift (left panel) + pre-drift budget waste fraction (right panel) |
| 4 | Latency Sensitivity | `table4_latency_sensitivity_{dataset}.csv` | `fig4_latency_sensitivity_{dataset}.png` | Post-drift accuracy vs total latency per policy, one subplot per drift type |

#### Cross-Dataset Summary Artifacts

When run on ≥ 2 datasets, a cross-dataset summary is also generated:

| Output | Path | Description |
|--------|------|-------------|
| Grand-mean bar chart | `results_without_retrain/cross_policy_comparison/fig_cross_dataset_summary.png` | Policy ranking across all datasets — validates whether findings generalise |
| Summary table | `results_without_retrain/cross_policy_comparison/table_cross_dataset_summary.csv` | Mean ± std post-drift accuracy per policy per dataset, with run counts |

> See [cross_policy_comparison_guide.md](cross_policy_comparison_guide.md) for detailed interpretation of each output.

### Visualisation Dashboard Layout

Each retraining-policy dashboard contains six panels:
1. **Line plot** — Accuracy vs. total latency, grouped by drift type
2. **Bar chart** — Mean accuracy (± std) by drift type
3. **Heatmap** — Accuracy across Budget × Latency
4. **Grouped bar** — Budget utilization by budget level and latency
5. **Grouped bar** — Average retrains after drift by drift type and latency
6. **Grouped bar** — Average retrains after drift by drift type and budget

The no-retrain baseline dashboard contains four panels:
1. **Bar chart** — Mean overall accuracy by drift type (± std)
2. **Grouped bar** — Pre-drift vs post-drift accuracy by drift type
3. **Bar chart** — Accuracy drop by drift type (± std)
4. **Box plot** — Accuracy distribution across seeds per drift type

---

## Git Branching Strategy

### Phase 1 — Per-Configuration Branches (243 runs, 3 seeds)

Each individual configuration was developed and tested in its own feature branch:
- **Naming:** `exp/<drift>-<policy>-<Budget>budget-<Latency>Latency`
- **Example:** `exp/gradual-drift-triggered-Medbudget-HighLatency`

Cumulative results per policy were merged into dedicated develop branches:
- `develop_3seed_periodic_retrain`
- `develop_3seed_error_threshold_retrain`
- `develop_3seed_drift_triggered_retrain`

### Phase 2 — Per-Policy Batch Branches (810 runs, 10 seeds)

- `develop-10Seed-periodic-retrain-tests` (270 runs)
- `develop-10Seed-error-threshold-retrain-tests` (270 runs)
- `develop-10Seed-drift-triggered-retrain-tests` (270 runs)

### Phase 3 — Extreme Latency (741 runs, 3 + 10 seeds)

- `develop_ExtremeLatencyLevels` (171 runs with 3 seeds + 570 runs with 10 seeds)

### Phase 4 — LUFlow Real-World Dataset (252 runs)

- `develop_LUFlow_Dataset` (243 active runs + 9 baseline runs)

### Phase 5 — LendingClub Real-World Dataset (252 runs)

- `develop_LendingClub_Dataset` (243 active runs + 9 baseline runs)

### Phase 6 — Cross-Policy Comparison (with partial_fit, analysis only)

- No dedicated experiment branch — `cross_policy_comparison.py` reads existing summary CSVs from `results_with_retrain/` and writes output to `results_with_retrain/cross_policy_comparison/`. Committed directly on `main`.

### Phase 7 — Without partial_fit (1,596 runs)

- `develop_no_partial_fit` — All without-partial-fit experiments (synthetic 3-seed, 10-seed; LUFlow 3-seed; LendingClub 3-seed) plus cross-policy comparison. Results in `results_without_retrain/`.

### No-Retrain Baseline (39 runs, with partial_fit)

- `develop_NoRetrain_NoBudget_NoLatency` (9 runs with 3 seeds + 30 runs with 10 seeds)

### Merged Results

The **`main`** branch contains **summary CSVs, dashboard PNGs, and cross-policy comparison outputs** from all phases across both experiment modes (with and without partial_fit). Per-run artifacts (JSON results, per-sample CSVs) are too large to merge and remain in their respective experiment branches only.
