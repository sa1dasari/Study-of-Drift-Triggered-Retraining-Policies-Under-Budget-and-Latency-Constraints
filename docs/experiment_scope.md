# Experiment Scope & Configuration Matrix

## Overview

This document specifies every factor varied across experiment runs, the exact parameter values used, and the total number of configurations executed.
The study follows a **full-factorial design** — every combination of drift type, policy, budget, and latency is run with multiple random seeds.

Experiments were conducted in **two phases**:
- **Phase 1 (3 seeds):** 243 runs — initial exploration with per-configuration Git branches.
- **Phase 2 (10 seeds):** 810 runs — extended evaluation with per-policy batch Git branches.
- **Combined total: 1,053 experiment runs.**

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

## Factor 4 — Latency Levels (3)

| Level | Retrain Latency (steps) | Deploy Latency (steps) | Total Latency (steps) | Interpretation |
|---|---|---|---|---|
| **Low** | 10 | 1 | 11 | Near real-time — fast retraining and near-instant deployment |
| **Medium** | 100 | 5 | 105 | Moderate — e.g., nightly batch retrain with brief staging |
| **High** | 500 | 20 | 520 | Heavy — large model retrain with substantial deployment pipeline overhead |

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

### Combined Summary

| Dimension | Values | Count |
|---|---|---|
| Drift types | abrupt, gradual, recurring | 3 |
| Policies | periodic, error_threshold, drift_triggered | 3 |
| Budgets (K) | 5, 10, 20 | 3 |
| Latency levels | Low (11), Medium (105), High (520) | 3 |
| Seeds (Phase 1) | 42, 123, 456 | 3 |
| Seeds (Phase 2) | 42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021 | 10 |
| **Total unique configurations** | | **27 per policy / 81 across all** |
| **Total Phase 1 runs** | | **243** |
| **Total Phase 2 runs** | | **810** |
| **Grand total experiment runs** | | **1,053** |

The **`main`** and **`develop`** branches contain all merged results from both phases.

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

### Visualisation Artifacts

| File | Description |
|---|---|
| `results/summary_results_plot_periodic_retrain_3seed.png` | 2 × 3 dashboard for periodic policy (Phase 1, 3 seeds) |
| `results/summary_results_plot_periodic_retrain_10seed.png` | 2 × 3 dashboard for periodic policy (Phase 2, 10 seeds) |
| `results/summary_results_plot_error_threshold_retrain_3seed.png` | 2 × 3 dashboard for error-threshold policy (Phase 1, 3 seeds) |
| `results/summary_results_plot_error_threshold_retrain_10seed.png` | 2 × 3 dashboard for error-threshold policy (Phase 2, 10 seeds) |
| `results/summary_results_plot_drift_triggered_retrain_3seed.png` | 2 × 3 dashboard for drift-triggered (ADWIN) policy (Phase 1, 3 seeds) |
| `results/summary_results_plot_drift_triggered_retrain_10seed.png` | 2 × 3 dashboard for drift-triggered (ADWIN) policy (Phase 2, 10 seeds) |

Each dashboard contains six panels:
1. **Line plot** — Accuracy vs. total latency, grouped by drift type
2. **Bar chart** — Mean accuracy (± std) by drift type
3. **Heatmap** — Accuracy across Budget × Latency
4. **Grouped bar** — Budget utilization by budget level and latency
5. **Grouped bar** — Average retrains after drift by drift type and latency
6. **Grouped bar** — Average retrains after drift by drift type and budget