# Results Interpretation Guide

This document explains how to read and interpret the CSV and PNG files in the results directories.

Results are organized into two top-level directories based on the experiment mode:
- **`results_with_retrain/`** — Experiments **with** partial_fit (incremental learning).
- **`results_without_retrain/`** — Experiments **without** partial_fit (static model between retrains).

Both directories share the same internal structure:
- **`{results_dir}/synthetic/`** — Synthetic data experiments (`csv/` for summary CSVs, `plots/` for dashboard PNGs, `per_run/` for per-run details).
- **`{results_dir}/luflow/`** — LUFlow real-world data experiments (same sub-structure).
- **`{results_dir}/lendingclub/`** — LendingClub real-world data experiments (same sub-structure).
- **`{results_dir}/cross_policy_comparison/`** — Cross-policy head-to-head comparison outputs (tables & figures). See [cross_policy_comparison_guide.md](cross_policy_comparison_guide.md) for a detailed interpretation of each output.

---

## 1. Summary CSV Files — Column Reference

The CSV structure is the same across both experiment modes and all datasets. Key differences between datasets:
- **Synthetic:** `random_seed` identifies the data generation seed; stream length = 10,000; drift point = 5,000.
- **LUFlow:** `random_seed` is the pool-config ID (1, 2, or 3); stream length = 50,000; drift point = 25,000. Additional columns `dataset` and `pool_config` identify the LUFlow experiment.
- **LendingClub:** `random_seed` is the year-pair seed ID (1, 2, or 3); stream length = 50,000; drift point = 25,000. Additional columns `dataset` and `pool_config` identify the LendingClub experiment (pool_config contains the year-pair label, e.g. "2013->2016 (3-yr gap, max policy shift)").

### Configuration Columns

These columns describe the experimental setup for each run. Use them as **filter/group keys**.

| Column | Type | Example Values | Description |
|---|---|---|---|
| `drift_type` | string | `abrupt`, `gradual`, `recurring` | Type of concept drift injected |
| `drift_point` | int | `5000` | Timestep at which drift begins |
| `policy_type` | string | `periodic`, `error_threshold`, `drift_triggered`, `no_retrain` | Retraining policy used |
| `error_threshold` | float | `0.27` | Rolling error-rate trigger (error-threshold policy only) |
| `window_size` | int | `200` or `500` | Sliding window size (error-threshold: 200, ADWIN: 500) |
| `budget` | int | `5`, `10`, `20` | Maximum full retrains allowed |
| `retrain_latency` | int | `2`, `10`, `100`, `500`, `2000` | Timesteps to complete offline retraining |
| `deploy_latency` | int | `1`, `5`, `20`, `50` | Timesteps to deploy the retrained model |
| `random_seed` | int | `42`–`2021` | Random seed controlling data generation |

> **Tip:** Total latency per retrain = `retrain_latency + deploy_latency`. The study uses five total-latency levels across phases: **3**, **11**, **105**, **520**, and **2,050**.

### Performance Columns

| Column | Type | Range | Description |
|---|---|---|---|
| `overall_accuracy` | float | [0, 1] | Mean accuracy across all timesteps (10,000 synthetic; 50,000 LUFlow/LendingClub) — the primary metric |
| `pre_drift_accuracy` | float | [0, 1] | Mean accuracy before drift point — stable period |
| `post_drift_accuracy` | float | [0, 1] | Mean accuracy after drift point — measures adaptation to drift |
| `accuracy_drop` | float | typically [−0.05, +0.06] | `post_drift − pre_drift`. Negative = degradation after drift |

> **How to interpret `accuracy_drop`:**
> - `−0.04` → model lost 4 pp of accuracy after drift.
> - `≈ 0.00` → policy maintained accuracy through drift.
> - Positive → post-drift concept was inherently easier for this seed (not necessarily a good policy).

### Retraining Columns

| Column | Type | Range | Description |
|---|---|---|---|
| `total_retrains` | int | [0, budget] | Full retrains actually executed |
| `budget_utilization` | float | [0, 1] | `total_retrains / budget` — fraction of budget consumed |
| `retrains_before_drift` | int | [0, budget] | Retrains before the drift point — ideally low (budget wasted on stable concept) |
| `retrains_after_drift` | int | [0, budget] | Retrains at or after the drift point — ideally high (retraining where it matters) |

> `total_retrains = retrains_before_drift + retrains_after_drift`

---

## 2. Filtering & Grouping the CSV

| Question | Filter / Group by |
|---|---|
| How does accuracy vary across drift types? | Group by `drift_type`, average `overall_accuracy` |
| Does more budget help? | Group by `budget`, average `post_drift_accuracy` |
| How does latency affect budget use? | Group by `retrain_latency`, average `budget_utilization` |
| Is the policy wasting budget before drift? | Compare `retrains_before_drift` vs `retrains_after_drift` |
| How stable are results across seeds? | Group by `random_seed`, check std of `overall_accuracy` |

---

## 3. Dashboard Plots — Panel-by-Panel Guide

Each active-policy dashboard is a **2×3 PNG** with six panels:

```
┌─────────────────────┬──────────────────────┬────────────────────────┐
│  Panel 1 (top-left) │ Panel 2 (top-center) │  Panel 3 (top-right)  │
│  Line Plot          │ Bar Chart            │  Heatmap              │
│  Accuracy vs        │ Mean Accuracy        │  Accuracy across      │
│  Latency            │ by Drift Type        │  Budget × Latency     │
├─────────────────────┼──────────────────────┼────────────────────────┤
│ Panel 4 (bot-left)  │ Panel 5 (bot-center) │  Panel 6 (bot-right)  │
│ Grouped Bar         │ Grouped Bar          │  Grouped Bar          │
│ Budget Utilization  │ Retrains After Drift │  Retrains After Drift │
│ by Budget & Latency │ by Drift & Latency   │  by Drift & Budget    │
└─────────────────────┴──────────────────────┴────────────────────────┘
```

The **no-retrain baseline** dashboard is a **2×2 PNG** with four panels:
1. Mean overall accuracy by drift type (± std)
2. Pre-drift vs post-drift accuracy by drift type
3. Accuracy drop by drift type (± std)
4. Box plot — accuracy distribution across seeds per drift type

---

### Panel 1 — Accuracy vs Latency (Line Plot)

- **X-axis:** Total latency (retrain + deploy).
- **Y-axis:** Overall accuracy (mean across budgets and seeds).
- **Lines:** One per drift type.
- **Upward slope** → accuracy *increases* with latency (fewer retrains = fewer stale-model windows).
- **Downward slope** → stale-weight cost outweighs.
- **Flat** → latency has little impact.

### Panel 2 — Mean Accuracy by Drift Type (Bar Chart)

- **Bars:** One per drift type, with ± 1 std error bars.
- Tallest bar = easiest drift type for this policy.
- Large error bars = high sensitivity to config/seed.

### Panel 3 — Accuracy Heatmap (Budget × Latency)

- **Rows:** Budget (5, 10, 20). **Columns:** Latency levels.
- **Cell value:** Mean accuracy across all drifts and seeds.
- Uniform color → policy is robust to budget/latency. Gradient → one factor dominates.

### Panel 4 — Budget Utilization (Grouped Bar)

- **X-axis:** Budget level. **Bars:** One per latency level.
- Bar at 1.0 = 100% budget consumed. Below 1.0 = leftover budget.
- Comparing bars within a group shows latency's effect on budget use.

### Panel 5 — Retrains After Drift by Drift & Latency (Grouped Bar)

- **Y-axis:** Average post-drift retrains.
- Higher bars = more retrains devoted to the post-drift period.
- Lower bars under high latency = latency blocking post-drift retrains.

### Panel 6 — Retrains After Drift by Drift & Budget (Grouped Bar)

- **Y-axis:** Average post-drift retrains.
- Staircase pattern (K=20 > K=10 > K=5) = budget is the bottleneck.
- Flat bars = policy doesn't trigger more retrains even with more budget.

---

## 4. Cross-Policy Comparison Outputs

The `cross_policy_comparison.py` script merges all per-policy summary CSVs and produces head-to-head comparison tables and figures across all four policies. Outputs are saved to `{results_dir}/cross_policy_comparison/{dataset}/` (where `{results_dir}` is `results_with_retrain` or `results_without_retrain` depending on the mode).

For a full panel-by-panel interpretation guide, see **[cross_policy_comparison_guide.md](cross_policy_comparison_guide.md)**.

