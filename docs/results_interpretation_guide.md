# Results Interpretation Guide

This document explains how to read and interpret every output artifact in the `results/` folder — the three summary CSV files and the three 2×3 dashboard PNG plots (one per retraining policy).

---

## Table of Contents

1. [Results Overview](#1-results-overview)
2. [Summary CSV Files — Column-by-Column Reference](#2-summary-csv-files--column-by-column-reference)
   - [Configuration Columns](#21-configuration-columns)
   - [Performance Columns](#22-performance-columns)
   - [Retraining Columns](#23-retraining-columns)
   - [Differences Between the Three CSVs](#24-differences-between-the-three-csvs)
3. [How to Read the CSV Data](#3-how-to-read-the-csv-data)
   - [Filtering and Grouping](#31-filtering-and-grouping)
   - [Worked Example: Comparing Policies Under Abrupt Drift](#32-worked-example)
4. [Dashboard Plots — Panel-by-Panel Guide](#4-dashboard-plots--panel-by-panel-guide)
   - [Panel 1 — Accuracy vs Latency by Drift Type (Line Plot)](#41-panel-1--accuracy-vs-latency-by-drift-type)
   - [Panel 2 — Mean Accuracy by Drift Type (Bar Chart)](#42-panel-2--mean-accuracy-by-drift-type)
   - [Panel 3 — Accuracy Heatmap: Budget × Latency](#43-panel-3--accuracy-heatmap-budget--latency)
   - [Panel 4 — Budget Utilization by Budget & Latency (Grouped Bar)](#44-panel-4--budget-utilization-by-budget--latency)
   - [Panel 5 — Retrains After Drift by Drift Type & Latency (Grouped Bar)](#45-panel-5--retrains-after-drift-by-drift-type--latency)
   - [Panel 6 — Retrains After Drift by Drift Type & Budget (Grouped Bar)](#46-panel-6--retrains-after-drift-by-drift-type--budget)
5. [Interpreting Each Policy's Dashboard](#5-interpreting-each-policys-dashboard)
   - [Periodic Policy](#51-periodic-policy)
   - [Error-Threshold Policy](#52-error-threshold-policy)
   - [Drift-Triggered (ADWIN) Policy](#53-drift-triggered-adwin-policy)
6. [Cross-Policy Comparison Tips](#6-cross-policy-comparison-tips)
7. [Common Patterns and What They Mean](#7-common-patterns-and-what-they-mean)

---

## 1. Results Overview

The `results/` folder contains six key artifacts:

| File | Type | Contents |
|---|---|---|
| `summary_results_periodic_retrain.csv` | CSV | 81 rows — one per (drift × budget × latency × seed) run for the **periodic** policy |
| `summary_results_error_threshold_retrain.csv` | CSV | 81 rows — same matrix for the **error-threshold** policy |
| `summary_results_drift_triggered_retrain.csv` | CSV | 81 rows — same matrix for the **drift-triggered (ADWIN)** policy |
| `summary_results_plot_periodic_retrain.png` | PNG | 2×3 dashboard summarising all 81 periodic runs |
| `summary_results_plot_error_threshold_retrain.png` | PNG | 2×3 dashboard summarising all 81 error-threshold runs |
| `summary_results_plot_drift_triggered_retrain.png` | PNG | 2×3 dashboard summarising all 81 ADWIN runs |

Each CSV has **81 data rows** = 3 drift types × 3 budget levels × 3 latency levels × 3 random seeds.

---

## 2. Summary CSV Files — Column-by-Column Reference

### 2.1 Configuration Columns

These columns describe the experimental setup for each run. They are your **filter/group keys**.

| Column | Type | Values | Description |
|---|---|---|---|
| `drift_type` | string | `abrupt`, `gradual`, `recurring` | Type of concept drift injected into the data stream |
| `drift_point` | int | Always `5000` | Timestep at which drift begins |
| `policy_type` | string | `periodic`, `error_threshold`, `drift_triggered` | Which retraining policy was used |
| `policy_interval` | int | `500`, `1000`, `2000` | *Periodic only* — timesteps between scheduled retrains. Empty for other policies. |
| `error_threshold` | float | `0.27` | *Error-threshold & ADWIN only* — rolling error-rate trigger threshold (for error-threshold) or empty (for ADWIN). |
| `window_size` | int | `200` or `500` | Sliding window size for error-rate calculation (error-threshold: 200) or ADWIN detection (drift-triggered: 500). Empty for periodic. |
| `budget` | int | `5`, `10`, `20` | Maximum number of full retrains allowed during the 10,000-sample stream |
| `retrain_latency` | int | `10`, `100`, `500` | Timesteps needed to complete offline retraining |
| `deploy_latency` | int | `1`, `5`, `20` | Timesteps needed to deploy the retrained model |
| `random_seed` | int | `42`, `123`, `456` | Random seed controlling data generation (weight vectors + features) |

> **Tip:** The *total latency* per retrain event is `retrain_latency + deploy_latency`. The three levels used in this study are **11** (Low), **105** (Medium), and **520** (High).

### 2.2 Performance Columns

These columns measure how well the model performed.

| Column | Type | Range | Description |
|---|---|---|---|
| `overall_accuracy` | float | [0, 1] | Mean prediction accuracy across **all** 10,000 timesteps. This is the single most important performance metric. |
| `pre_drift_accuracy` | float | [0, 1] | Mean accuracy for timesteps `t ∈ [0, 5000)` — the stable period before drift. Acts as a baseline; should be similar across policies for the same seed since no drift has occurred yet. |
| `post_drift_accuracy` | float | [0, 1] | Mean accuracy for timesteps `t ∈ [5000, 10000)` — the period after drift begins. The policy's ability to adapt to drift is measured here. |
| `accuracy_drop` | float | typically [−0.05, +0.06] | Computed as `post_drift_accuracy − pre_drift_accuracy`. **Negative** values mean performance **degraded** after drift. **Positive** values mean the model actually performed better post-drift (this can happen when the post-drift concept is easier for the learner). |

> **How to interpret `accuracy_drop`:**
> - A value of `−0.04` means the model lost 4 percentage points of accuracy after drift started.
> - A value near `0.00` means the policy successfully maintained accuracy through drift (or drift had no net effect).
> - A positive value does **not** necessarily mean the policy is good — it may simply mean the post-drift concept was inherently easier for this seed.

### 2.3 Retraining Columns

These columns describe how the retraining budget was spent.

| Column | Type | Range | Description |
|---|---|---|---|
| `total_retrains` | int | [0, budget] | Number of full retrains actually executed during the run |
| `budget_utilization` | float | [0, 1] | `total_retrains / budget` — fraction of the available budget consumed. A value of `1.0` (100%) means all allowed retrains were used. |
| `retrains_before_drift` | int | [0, budget] | Retrains that occurred at `t < 5000`. Ideally low — retraining before drift is a **waste of budget** since the concept is still stable. |
| `retrains_after_drift` | int | [0, budget] | Retrains that occurred at `t ≥ 5000`. Ideally high — this is when retraining is actually needed. |

> **Key relationship:** `total_retrains = retrains_before_drift + retrains_after_drift`
>
> **Budget efficiency** can be assessed by comparing `retrains_after_drift / total_retrains`. A policy that spends most of its budget post-drift is more efficient than one that wastes retrains pre-drift.

---

### 2.4 Differences Between the Three CSVs

The three CSV files share the same column schema with minor variations in which policy-specific columns are populated:

| Column | Periodic CSV | Error-Threshold CSV | Drift-Triggered CSV |
|---|---|---|---|
| `policy_interval` | ✅ Filled (`500`/`1000`/`2000`) | — Empty | — Empty |
| `error_threshold` | — Empty | ✅ Filled (`0.27`) | — Empty |
| `window_size` | — Empty | ✅ Filled (`200`) | ✅ Filled (`500`) |

---

## 3. How to Read the CSV Data

### 3.1 Filtering and Grouping

The 81 rows in each CSV are a full-factorial design. To extract meaningful comparisons, filter or group by one or more dimensions:

| Question you want to answer | Filter / Group by |
|---|---|
| "How does accuracy vary across drift types?" | Group by `drift_type`, average `overall_accuracy` |
| "Does more budget help?" | Group by `budget`, average `overall_accuracy` or `post_drift_accuracy` |
| "How does latency affect budget use?" | Group by `retrain_latency`, average `budget_utilization` |
| "Is the policy wasting budget before drift?" | Filter by `drift_type`, compare `retrains_before_drift` vs `retrains_after_drift` |
| "How stable are results across seeds?" | Group by `random_seed`, check standard deviation of `overall_accuracy` |

### 3.2 Worked Example

**Question:** Under abrupt drift with low budget (K=5) and low latency (11 steps), how does each policy perform?

Filter each CSV with: `drift_type == "abrupt"` AND `budget == 5` AND `retrain_latency == 10`

This gives 3 rows (one per seed). Average the `overall_accuracy` column for each policy:

| Policy | Seed 42 | Seed 123 | Seed 456 | Mean |
|---|---|---|---|---|
| Periodic | 0.7500 | 0.7376 | 0.8056 | **0.764** |
| Error-Threshold | 0.7530 | 0.7366 | 0.8022 | **0.764** |
| Drift-Triggered | 0.7529 | 0.7395 | 0.8034 | **0.765** |

Then compare `retrains_before_drift` to see which policy wasted budget:

| Policy | Seed 42 | Seed 123 | Seed 456 |
|---|---|---|---|
| Periodic | 3 | 3 | 3 |
| Error-Threshold | 5 | 5 | 0 |
| Drift-Triggered | 0 | 0 | 0 |

**Interpretation:** The periodic policy always fires 3 retrains before drift (deterministic schedule). The error-threshold policy exhausts all 5 retrains before drift for seeds 42 and 123 (noise triggers). ADWIN fires 0 retrains before drift for all seeds (conservative detection preserves budget).

---

## 4. Dashboard Plots — Panel-by-Panel Guide

Each policy produces a 2×3 dashboard PNG with six panels. The layout is:

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

---

### 4.1 Panel 1 — Accuracy vs Latency by Drift Type

**Chart type:** Line plot with markers

**X-axis:** Total latency in timesteps (11, 105, 520) — the sum of `retrain_latency + deploy_latency`.

**Y-axis:** Overall accuracy (mean across all budgets and seeds for that drift × latency combination).

**Lines:** One per drift type (Abrupt = blue circle, Gradual = red square, Recurring = purple diamond).

**How to read it:**
- Each point on a line represents the **average accuracy** across 9 runs (3 budgets × 3 seeds) for a particular drift type at a particular latency level.
- An **upward slope** from left to right means accuracy *increases* with higher latency — this is counterintuitive and usually indicates that the policy retrains *less* at high latency (fewer disruptions from stale-model windows).
- A **downward slope** means accuracy degrades with higher latency — the cost of operating on stale weights during long latency windows outweighs any other effects.
- **Flat lines** mean latency has little impact on accuracy for that drift type.

**What to look for:**
- Compare the vertical spread between lines: a large gap means drift type matters more than latency.
- Compare the slope of lines: steep slopes mean latency is a dominant factor.
- For ADWIN: note that abrupt drift may show an upward trend because higher latency *blocks more retrains*, and for seeds where ADWIN fires, fewer but better-timed retrains can actually improve accuracy slightly.

---

### 4.2 Panel 2 — Mean Accuracy by Drift Type

**Chart type:** Bar chart with error bars (mean ± 1 standard deviation)

**X-axis:** Drift type (Abrupt, Gradual, Recurring).

**Y-axis:** Mean overall accuracy across all 27 runs (3 budgets × 3 latencies × 3 seeds) for that drift type.

**Error bars:** Standard deviation — shows how much accuracy varies across the 27 configurations.

**How to read it:**
- Taller bars indicate better average performance against that drift type.
- **Large error bars** indicate that accuracy varies significantly across different budget/latency/seed combinations — the policy's performance is sensitive to configuration.
- **Small error bars** indicate consistent performance regardless of configuration.
- The numeric label above each bar gives the exact mean accuracy value.

**What to look for:**
- Which drift type is "hardest" for this policy (shortest bar)?
- Are the bars roughly equal (policy is drift-agnostic) or very different (policy is drift-sensitive)?
- Large error bars on a specific drift type suggest that budget/latency choice matters a lot for that scenario.

---

### 4.3 Panel 3 — Accuracy Heatmap: Budget × Latency

**Chart type:** 2D heatmap (colour-coded matrix)

**Rows:** Budget level — Low (5), Med (10), High (20), from top to bottom.

**Columns:** Latency level — Low (11), Med (105), High (520), from left to right.

**Cell value:** Mean overall accuracy averaged across all 3 drift types and 3 seeds (9 runs per cell).

**Colour scale:** Viridis colour-map — darker (purple) = lower accuracy, brighter (yellow/green) = higher accuracy. A colour-bar on the right shows the scale.

**How to read it:**
- Each cell is labelled with its exact accuracy value (3 decimal places).
- Look for colour gradients **across rows** (budget effect) and **across columns** (latency effect).
- A **uniform heatmap** (all cells similar colour/value) means neither budget nor latency significantly affects overall accuracy — the policy is robust.
- A **gradient from left to right** means latency is the dominant factor.
- A **gradient from top to bottom** means budget is the dominant factor.

**What to look for:**
- Is there a "sweet spot" cell that clearly outperforms others?
- Does increasing budget (moving down) consistently help?
- Does increasing latency (moving right) consistently hurt?
- For ADWIN: the heatmap may appear very flat (values ≈ 0.764–0.765) because ADWIN fires so few retrains that budget and latency barely matter for most seeds.

---

### 4.4 Panel 4 — Budget Utilization by Budget & Latency

**Chart type:** Grouped bar chart

**X-axis:** Budget level — Low (5), Med (10), High (20).

**Bars within each group:** One per latency level (Green = Low, Orange = Medium, Red = High).

**Y-axis:** Mean budget utilisation (0.0 to 1.0+). A dashed grey line at `1.0` marks 100% utilisation.

**Bar label:** Percentage value (e.g., "100%", "51%").

**How to read it:**
- A bar at `1.0` (100%) means the policy used every available retrain — the budget was fully consumed.
- A bar below `1.0` means the policy had leftover budget — either it didn't need to retrain that many times, or latency windows blocked retrain triggers.
- Comparing bars within a group shows **latency's effect on budget use**: if the red (High latency) bar is much shorter than green (Low latency), then high latency is preventing the policy from using its full budget.

**What to look for:**
- **Periodic policy:** Expect 100% utilisation at low/medium latency. At high latency with K=20, expect ~50% because the 520-step latency window overlaps with the 500-step interval, blocking every other retrain.
- **Error-threshold policy:** Expect ~75–100% utilisation across most configs. High latency with K=20 may show lower utilisation.
- **ADWIN policy:** Expect very low utilisation (3%–33%) because ADWIN only fires when it detects a distributional shift, which doesn't happen for every seed.

---

### 4.5 Panel 5 — Retrains After Drift by Drift Type & Latency

**Chart type:** Grouped bar chart

**X-axis:** Drift type (Abrupt, Gradual, Recurring).

**Bars within each group:** One per latency level (Green = Low, Orange = Medium, Red = High).

**Y-axis:** Average number of retrains that occurred at or after `t = 5000` (the drift point).

**How to read it:**
- Higher bars mean more retrains were devoted to the post-drift period — this is where retraining is valuable.
- Lower bars under high latency (red) compared to low latency (green) indicate that latency is preventing the policy from retraining as often after drift.
- A bar at `0.0` means the policy never retrained after drift for that combination — extremely problematic since the model has no mechanism to adapt beyond `partial_fit`.

**What to look for:**
- **Periodic policy:** Bars should be roughly proportional to budget since periodic retrains are evenly distributed. Expect roughly half the total retrains to occur after drift (since drift is at the midpoint).
- **Error-threshold policy:** Most retrains should occur after drift (when error rate spikes). If bars are low, the budget was wasted pre-drift.
- **ADWIN policy:** Bars for abrupt drift may be surprisingly low (because ADWIN doesn't detect drift for 2 out of 3 seeds). Recurring drift may show the highest bars because repeated concept switches give ADWIN multiple detection opportunities.

---

### 4.6 Panel 6 — Retrains After Drift by Drift Type & Budget

**Chart type:** Grouped bar chart

**X-axis:** Drift type (Abrupt, Gradual, Recurring).

**Bars within each group:** One per budget level (Green = Low K=5, Orange = Medium K=10, Red = High K=20).

**Y-axis:** Average number of retrains that occurred at or after `t = 5000`.

**How to read it:**
- Higher budget (red bars) should generally allow more post-drift retrains — but only if the policy actually triggers them.
- If all three bars are similar height for a drift type, then budget is not the bottleneck — the policy simply doesn't trigger more retrains even when budget is available.

**What to look for:**
- **Periodic policy:** Clear staircase pattern — K=20 allows many more post-drift retrains than K=5.
- **Error-threshold policy:** Similar staircase, but with seed variance.
- **ADWIN policy:** If bars for gradual drift are all `0.0`, ADWIN cannot detect gradual drift at the calibrated δ = 0.002. If bars for recurring drift are the tallest, ADWIN benefits from repeated concept switches.

> **Note:** Some dashboard variants may show **"Error Count in Drift Window by Drift Type & Budget"** in Panel 6 instead. This shows the average number of errors in the first 1,000 timesteps after `drift_point` (t ∈ [5000, 6000)). Higher bars mean more errors occurred in the critical drift-onset window — indicating slower policy response.

---

## 5. Interpreting Each Policy's Dashboard

### 5.1 Periodic Policy

**Key characteristics visible in the dashboard:**

| Panel | What you should see | Why |
|---|---|---|
| Panel 1 (Accuracy vs Latency) | Nearly flat lines across all drift types | Periodic retraining is drift-agnostic — accuracy is insensitive to latency because the schedule fires at fixed intervals regardless |
| Panel 2 (Mean Accuracy) | Similar bar heights for all drift types (~0.76), moderate error bars | Periodic policy treats all drift types equally (no detection mechanism) |
| Panel 3 (Heatmap) | Uniform colour (all cells ≈ 0.762–0.764) | Neither budget nor latency significantly impacts overall accuracy for this policy |
| Panel 4 (Budget Util.) | 100% at low/medium latency; drops to ~51% at high latency with K=20 | High latency (520 steps) overlaps the 500-step interval, blocking every other retrain |
| Panel 5 (Retrains After Drift × Latency) | Consistent bars, reduced at high latency for recurring drift | High latency blocks more post-drift retrains |
| Panel 6 (Retrains After Drift × Budget) | Clear staircase: K=20 > K=10 > K=5 | More budget → more retrains scheduled → more land post-drift |

### 5.2 Error-Threshold Policy

**Key characteristics visible in the dashboard:**

| Panel | What you should see | Why |
|---|---|---|
| Panel 1 (Accuracy vs Latency) | Slightly higher accuracy at medium latency than low | Moderate latency can reduce retrain frequency, which paradoxically helps if pre-drift noise triggers are reduced |
| Panel 2 (Mean Accuracy) | Similar to periodic (~0.76) with comparable error bars | Performance is in the same range because the threshold (0.27) was calibrated to be comparable |
| Panel 3 (Heatmap) | Slight gradient — medium latency cells may be marginally brighter | Medium latency slightly reduces unnecessary pre-drift retrains |
| Panel 4 (Budget Util.) | High utilisation (75–100%) at low/medium latency; lower at high latency with K=20 | Error threshold is frequently exceeded, consuming budget; high latency blocks some triggers |
| Panel 5 (Retrains After Drift × Latency) | Higher bars than ADWIN, but lower than periodic for recurring drift with high latency | Error rate rises post-drift, triggering retrains; high latency limits the number that can fire |
| Panel 6 (Retrains After Drift × Budget) | Staircase pattern, but some bars may plateau | Once the error rate drops below threshold (model adapts), additional budget isn't used |

### 5.3 Drift-Triggered (ADWIN) Policy

**Key characteristics visible in the dashboard:**

| Panel | What you should see | Why |
|---|---|---|
| Panel 1 (Accuracy vs Latency) | Abrupt drift line slopes **upward** (higher accuracy at higher latency) | At high latency, fewer but better-timed retrains may slightly help for the one seed that detects drift; for the two seeds that don't detect, latency is irrelevant |
| Panel 2 (Mean Accuracy) | Similar bar heights (~0.763–0.766), large error bars | High seed sensitivity: seed 456 achieves ~0.80 while seeds 42/123 achieve ~0.74–0.75 |
| Panel 3 (Heatmap) | Very flat / uniform (~0.764–0.765) | ADWIN fires so few retrains for most seeds that budget/latency has negligible impact |
| Panel 4 (Budget Util.) | **Very low** (3–33%) | ADWIN with δ=0.002 is highly conservative; for 2 out of 3 seeds it detects 0 drift under abrupt/gradual |
| Panel 5 (Retrains After Drift × Latency) | Recurring drift bars highest; gradual drift bars at or near 0 | ADWIN detects repeated concept switches in recurring drift but misses the slow shift in gradual drift entirely |
| Panel 6 (Retrains After Drift × Budget) | Gradual drift: all bars at 0. Recurring drift: staircase visible | ADWIN cannot detect gradual drift at δ=0.002. For recurring drift, more budget allows more responses to repeated switches |

---

## 6. Cross-Policy Comparison Tips

When comparing dashboards side-by-side across the three policies:

### Accuracy (Panel 2)
- All three policies achieve similar mean accuracy (~0.76). This is because the model's per-sample `partial_fit` provides baseline adaptation even without full retrains.
- The policies differ more in **how** they use their budget than in raw accuracy.

### Budget Efficiency (Panel 4)
- **Periodic:** Predictable, deterministic utilisation. Good if you want guaranteed budget consumption.
- **Error-threshold:** High utilisation but potentially wasteful — may fire retrains pre-drift when noise pushes the error rate above threshold.
- **ADWIN:** Very low utilisation — preserves budget but may *under-spend*, leaving adaptation on the table.

### Post-Drift Responsiveness (Panel 5)
- Compare the **height of green bars** (low latency) across the three policy dashboards for the same drift type.
- Periodic and error-threshold generally deploy more retrains post-drift than ADWIN.
- ADWIN excels specifically under **recurring drift** where repeated distributional changes give the Hoeffding bound test more signal to work with.

### The Latency Tax (Panel 4 + Panel 5)
- Compare the gap between green (low latency) and red (high latency) bars in Panel 4 across policies.
- The periodic policy suffers the most from the "latency tax" at K=20 because its 500-step interval collides with the 520-step latency window, effectively halving its effective budget.

---

## 7. Common Patterns and What They Mean

| Pattern | Where to see it | Interpretation |
|---|---|---|
| **Pre-drift accuracy varies by seed (0.71–0.82)** | `pre_drift_accuracy` column in CSV | The random weight vectors create easier/harder classification problems per seed. This is expected — compare policies within the same seed for fairness. |
| **Accuracy drop is positive for some seeds** | `accuracy_drop` column | The post-drift concept (w₂) is inherently easier than the pre-drift concept (w₁) for that seed. Not a policy effect. |
| **0 total retrains (ADWIN, certain seeds)** | `total_retrains` = 0 in drift-triggered CSV | ADWIN did not detect drift for that seed/config. The model relied entirely on `partial_fit` incremental updates. |
| **budget_utilization > 0 but retrains_after_drift = 0** | Error-threshold CSV, seeds 42/123 | The error threshold was exceeded by pre-drift noise, consuming the entire budget before drift occurred. |
| **budget_utilization = 0.5 at high latency + K=20** | Periodic CSV, `retrain_latency=500`, `budget=20` | The 520-step latency window exceeds the 500-step interval, blocking every alternate retrain. Only ~10 of 20 fire. |
| **Recurring drift has the most retrains after drift** | Panel 5 across all policy dashboards | Recurring drift generates 5 concept switches in the post-drift period, giving every policy repeated opportunities/triggers to retrain. |
| **Gradual drift + ADWIN = 0 retrains** | Panel 5/6 of ADWIN dashboard | Gradual drift changes the error distribution so slowly that the Hoeffding bound is never exceeded within the 500-sample window at δ = 0.002. |
| **Flat heatmap (Panel 3)** | ADWIN dashboard | Since most runs have 0 retrains, budget and latency have no effect — the model is just doing `partial_fit` in every case. |
| **Nearly identical accuracy across all 3 policies** | Panel 2 across all dashboards | The SGD online learner's `partial_fit` provides a strong baseline; full retrains offer marginal improvement on top of incremental learning. |

