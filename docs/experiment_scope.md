# Experiment Scope & Configuration Matrix

## Overview

This document specifies every factor varied across experiment runs, the exact parameter values used, and the total number of configurations executed.
The study follows a **full-factorial design** — every combination of drift type, policy, budget, and latency is run with multiple random seeds.

Experiments were conducted in **three phases** plus **baseline runs**:
- **Phase 1 (3 seeds):** 243 runs — initial exploration with per-configuration Git branches.
- **Phase 2 (10 seeds):** 810 runs — extended evaluation with per-policy batch Git branches.
- **No-Retrain Baseline (3 seeds):** 9 runs — accuracy floor with Phase 1 seeds.
- **No-Retrain Baseline (10 seeds):** 30 runs — accuracy floor with Phase 2 seeds.
- **Phase 3 — Extreme Latency (3 seeds):** 171 runs — 2 extreme latency levels (Near-Zero=3, Extreme-High=2050).
- **Phase 3 — Extreme Latency (10 seeds):** 570 runs — same 2 extreme levels with full 10-seed set.
- **Combined total: 1,833 experiment runs.**

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
| **No-Retrain (Baseline)** | None — budget = 0, latency = 0 | Never retrains; model adapts only via `partial_fit` (incremental learning) |

> The **No-Retrain baseline** has no budget or latency grid — it is a single-configuration policy run across 3 drift types × N seeds. It provides the accuracy floor that makes all other policies' numbers interpretable.

### Periodic Interval Selection

The periodic interval is chosen so that the maximum possible retrains exactly equals the budget:

| Budget (K) | Interval | Calculation |
|---|---|---|
| 5 | 2,000 | 10,000 / 5 = 2,000 |
| 10 | 1,000 | 10,000 / 10 = 1,000 |
| 20 | 500 | 10,000 / 20 = 500 |

---

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

The no-retrain baseline has **no budget or latency grid** — the model uses only `partial_fit` (incremental learning) and is never retrained from scratch. This provides the **accuracy floor** that makes all other policies' accuracy numbers interpretable.

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

### Combined Summary

| Dimension | Values | Count |
|---|---|---|
| Drift types | abrupt, gradual, recurring | 3 |
| Policies | periodic, error_threshold, drift_triggered, **no_retrain** | 4 |
| Budgets (K) | 5, 10, 20 *(N/A for no_retrain)* | 3 |
| Latency levels (Phase 1 & 2) | Low (11), Medium (105), High (520) *(N/A for no_retrain)* | 3 |
| Latency levels (Phase 3) | Near-Zero (3), Extreme-High (2050) *(N/A for no_retrain)* | 2 |
| Seeds (Phase 1 / Phase 3-3seed) | 42, 123, 456 | 3 |
| Seeds (Phase 2 / Phase 3-10seed) | 42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021 | 10 |
| **Total Phase 1 runs** | | **243** |
| **Total Phase 2 runs** | | **810** |
| **Total No-Retrain Baseline runs** | | **39** (9 + 30) |
| **Total Phase 3 runs (Extreme Latency)** | | **741** (171 + 570) |
| **Grand total experiment runs** | | **1,833** |

The **`main`** and **`develop`** branches contain all merged results from all phases.

---

## Output Artifacts per Policy

### Phase 1 (3-seed) — 81 runs per policy

| Artifact | Path | Description |
|---|---|---|
| JSON result | `results/<policy_dir>/run_<run_tag>.json` | Full config + structured metrics per run |
| Per-sample CSV | `results/<policy_dir>/per_sample_<run_tag>.csv` | Per-timestep accuracy, error, latency flags |
| Summary CSV row | `results/summary_results_{policy}_retrain_3seed.csv` | One row appended per run; 81 rows total per policy |

### Phase 2 (10-seed) — 270 runs per policy

| Artifact | Path | Description |
|---|---|---|
| JSON result | `results/<policy_dir>/run_<run_tag>.json` | Full config + structured metrics per run |
| Per-sample CSV | `results/<policy_dir>/per_sample_<run_tag>.csv` | Per-timestep accuracy, error, latency flags |
| Summary CSV row | `results/summary_results_{policy}_retrain_10seed.csv` | One row appended per run; 270 rows total per policy |

### No-Retrain Baseline — 3-seed (9 runs) and 10-seed (30 runs)

| Artifact | Path | Description |
|---|---|---|
| JSON result | `results/no_retrain_{N}seed_3drift/run_<drift>_s<seed>.json` | Full config + structured metrics per run |
| Per-sample CSV | `results/no_retrain_{N}seed_3drift/per_sample_<drift>_s<seed>.csv` | Per-timestep accuracy and error (no latency flags) |
| Summary CSV (3-seed) | `results/summary_results_no_retrain_3seed.csv` | One row per run; 9 rows total |
| Summary CSV (10-seed) | `results/summary_results_no_retrain_10seed.csv` | One row per run; 30 rows total |

### Phase 3 — Extreme Latency (3-seed: 54 runs per policy, 10-seed: 180 runs per policy)

| Artifact | Path | Description |
|---|---|---|
| JSON result | `results/<policy>_ExtremeLatency_{N}seed/run_<run_tag>.json` | Full config + structured metrics per run |
| Per-sample CSV | `results/<policy>_ExtremeLatency_{N}seed/per_sample_<run_tag>.csv` | Per-timestep accuracy, error, latency flags |
| Summary CSV (3-seed) | `results/summary_results_{policy}_retrain_ExtremeLatency_3seed.csv` | One row per run; 54 rows per active policy |
| Summary CSV (10-seed) | `results/summary_results_{policy}_retrain_ExtremeLatency_10seed.csv` | One row per run; 180 rows per active policy |
| Baseline CSV (3-seed) | `results/summary_results_no_retrain_ExtremeLatency_3seed.csv` | One row per run; 9 rows |
| Baseline CSV (10-seed) | `results/summary_results_no_retrain_ExtremeLatency_10seed.csv` | One row per run; 30 rows |

### Visualisation Artifacts

| File | Description |
|---|---|
| `results/summary_results_plot_periodic_retrain_3seed.png` | 2 × 3 dashboard for periodic policy (Phase 1, 3 seeds) |
| `results/summary_results_plot_periodic_retrain_10seed.png` | 2 × 3 dashboard for periodic policy (Phase 2, 10 seeds) |
| `results/summary_results_plot_error_threshold_retrain_3seed.png` | 2 × 3 dashboard for error-threshold policy (Phase 1, 3 seeds) |
| `results/summary_results_plot_error_threshold_retrain_10seed.png` | 2 × 3 dashboard for error-threshold policy (Phase 2, 10 seeds) |
| `results/summary_results_plot_drift_triggered_retrain_3seed.png` | 2 × 3 dashboard for drift-triggered (ADWIN) policy (Phase 1, 3 seeds) |
| `results/summary_results_plot_drift_triggered_retrain_10seed.png` | 2 × 3 dashboard for drift-triggered (ADWIN) policy (Phase 2, 10 seeds) |
| `results/summary_results_plot_no_retrain_3seed.png` | 2 × 2 baseline dashboard for no-retrain policy (3 seeds) |
| `results/summary_results_plot_no_retrain_10seed.png` | 2 × 2 baseline dashboard for no-retrain policy (10 seeds) |
| `results/summary_results_plot_periodic_retrain_ExtremeLatency_3seed.png` | 2 × 3 dashboard for periodic policy (Phase 3, 3 seeds, extreme latency) |
| `results/summary_results_plot_periodic_retrain_ExtremeLatency_10seed.png` | 2 × 3 dashboard for periodic policy (Phase 3, 10 seeds, extreme latency) |
| `results/summary_results_plot_error_threshold_retrain_ExtremeLatency_3seed.png` | 2 × 3 dashboard for error-threshold policy (Phase 3, 3 seeds, extreme latency) |
| `results/summary_results_plot_error_threshold_retrain_ExtremeLatency_10seed.png` | 2 × 3 dashboard for error-threshold policy (Phase 3, 10 seeds, extreme latency) |
| `results/summary_results_plot_drift_triggered_retrain_ExtremeLatency_3seed.png` | 2 × 3 dashboard for drift-triggered (ADWIN) policy (Phase 3, 3 seeds, extreme latency) |
| `results/summary_results_plot_drift_triggered_retrain_ExtremeLatency_10seed.png` | 2 × 3 dashboard for drift-triggered (ADWIN) policy (Phase 3, 10 seeds, extreme latency) |
| `results/summary_results_plot_no_retrain_ExtremeLatency_3seed.png` | 2 × 2 baseline dashboard for no-retrain policy (Phase 3, 3 seeds, extreme latency) |
| `results/summary_results_plot_no_retrain_ExtremeLatency_10seed.png` | 2 × 2 baseline dashboard for no-retrain policy (Phase 3, 10 seeds, extreme latency) |

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

### No-Retrain Baseline (39 runs)

- `develop_NoRetrain_NoBudget_NoLatency` (9 runs with 3 seeds + 30 runs with 10 seeds)

### CIS Fraud Detection Investigation (not merged)

- `develop_CIS-Fraud_detection` — Calibration, drift diagnosis, and streaming sanity checks on the IEEE-CIS Fraud Detection dataset. Discarded after sanity checks showed inconsistent post-drift degradation across temporal offsets (see [research_log.md](research_log.md), Week 8).

### Merged Results

The **`main`** and **`develop`** branches contain all results from all phases (1,833 total runs).
