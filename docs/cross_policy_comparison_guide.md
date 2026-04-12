# Cross-Policy Comparison — Results Interpretation Guide

This document explains how to read and interpret the outputs of `cross_policy_comparison.py`, which merges all per-policy summary CSVs for a given dataset and produces head-to-head comparison tables and figures.

For general CSV column definitions and per-policy dashboard panel guides, see [results_interpretation_guide.md](results_interpretation_guide.md).

---

## Overview

The cross-policy comparison produces **four outputs per dataset** (synthetic, LUFlow, LendingClub) plus a **cross-dataset summary** when multiple datasets are available.

| # | Output | What it answers |
|---|--------|-----------------|
| 1 | Post-Drift Accuracy Table & Heatmap | Which policy achieves the highest post-drift accuracy, and does the ranking change across drift types? |
| 2 | Budget-Faceted Comparison | Does the policy ranking change when budget is low (K=5) vs high (K=20)? |
| 3 | Budget Efficiency | Which policy gets the most accuracy improvement per retrain used? Which wastes budget pre-drift? |
| 4 | Latency Sensitivity | Which policy degrades most as deployment latency increases? |
| 5 | Cross-Dataset Summary | Do the policy rankings hold across synthetic, LUFlow, and LendingClub data? |

All outputs are saved to `{results_dir}/cross_policy_comparison/{dataset}/` (where `{results_dir}` is `results_with_retrain` or `results_without_retrain` depending on which mode's CSVs are being analyzed).

---

## Output 1 — Post-Drift Accuracy Table & Heatmap

**Files:** `table1_postdrift_accuracy_{dataset}.csv`, `fig1_postdrift_heatmap_{dataset}.png`

### What it shows

A **policy × drift type** matrix of mean post-drift accuracy, averaged over all seeds, budgets, and latency levels. For the no-retrain baseline (which has no budget/latency grid), the average is over seeds only. This is the "grand-mean" view — the simplest possible policy comparison.

### How to read the heatmap

- **Rows** = policies (Periodic, Error-Threshold, Drift-Triggered, No-Retrain).
- **Columns** = drift types (Abrupt, Gradual, Recurring).
- **Cell colour** = RdYlGn colormap (red = low accuracy, green = high accuracy).
- **Cell text** = exact mean post-drift accuracy to 4 decimal places.

### What to look for

| Pattern | Interpretation |
|---------|---------------|
| One row consistently greener than others | That policy is the overall winner |
| All rows similar colour | Policies perform comparably — retraining may not help much |
| One column redder than others | That drift type is harder for all policies |
| No-retrain row ≈ other rows | Retraining policies are not adding value over incremental learning alone |

### CSV columns

| Column | Description |
|--------|-------------|
| `policy_type` | Policy identifier |
| `drift_type` | Drift type |
| `mean` | Mean post-drift accuracy |
| `std` | Standard deviation across all runs (seeds × budgets × latencies) |
| `count` | Number of runs averaged |

---

## Output 2 — Budget-Faceted Comparison

**Files:** `table2_budget_faceted_{dataset}.csv`, `fig2_budget_faceted_{dataset}.png`

### What it shows

Post-drift accuracy by policy × drift type, shown in **separate panels for each budget level** (K=5, K=10, K=20). A horizontal dashed grey line in each panel marks the no-retrain baseline per drift type.

### How to read the figure

- **Panels** (left to right) = Budget K=5, K=10, K=20.
- **X-axis** = drift types (Abrupt, Gradual, Recurring).
- **Bars** = one per active policy, coloured by policy.
- **Dashed line** = no-retrain baseline for that drift type.
- **Error bars** = ± 1 standard deviation across seeds and latencies.

### What to look for

| Pattern | Interpretation |
|---------|---------------|
| Bars above dashed line | Policy outperforms baseline (retraining helps) |
| Bars below dashed line | Policy underperforms baseline (retraining may be hurting — e.g., latency cost outweighs adaptation) |
| Ranking changes across panels | Budget level changes which policy is best — budget-sensitive result |
| All bars converge at K=20 | With enough budget, all policies behave similarly |
| Error-threshold bars drop at K=5 | Budget waste is most damaging when budget is scarce |

### CSV columns

| Column | Description |
|--------|-------------|
| `budget` | Budget level (5, 10, 20) |
| `policy_type` | Policy identifier |
| `drift_type` | Drift type |
| `mean` | Mean post-drift accuracy for this budget × policy × drift |
| `std` | Standard deviation |
| `count` | Number of runs |

---

## Output 3 — Budget Efficiency

**Files:** `table3_budget_efficiency_{dataset}.csv`, `fig3_budget_efficiency_{dataset}.png`

### What it shows

A **two-panel figure** analysing how efficiently each policy uses its retrain budget:

- **Left panel — Accuracy Gain per Retrain Used:** `(post_drift_accuracy − baseline_accuracy) / retrains_after_drift`. Higher is better — each retrain buys more accuracy improvement.
- **Right panel — Pre-Drift Budget Waste:** `retrains_before_drift / total_retrains`. Lower is better — the policy should save its budget for after drift occurs.

### How to read the left panel (efficiency)

- **X-axis** = drift types.
- **Bars** = one per policy.
- **Y-axis** = Δ accuracy per retrain after drift.
- **Positive bars** = policy improves over baseline per retrain used.
- **Negative bars** = policy is worse than baseline per retrain (retraining is actively harmful, likely due to latency cost).
- **Zero line** = no improvement over baseline.
- Taller bars = more efficient use of each retrain.

### How to read the right panel (waste)

- **Y-axis** = fraction of total retrains that fired before drift.
- **Bar at 0.5** = half the budget was spent pre-drift (periodic policy behaviour — it retrains on a schedule regardless of drift).
- **Bar near 1.0** = nearly all retrains fired pre-drift (error-threshold's noise-trigger problem).
- **Bar near 0.0** = almost all retrains fired post-drift (ideal — drift-triggered policy should achieve this).
- **Dashed line at 50%** = reference mark.

### Key story: error-threshold budget waste

The error-threshold policy is known to trigger on pre-drift noise for certain seeds (especially on synthetic data with threshold = 0.27). When `retrains_before_drift / total_retrains ≈ 1.0`, it means the entire budget was exhausted before drift arrived, leaving zero retrains for the actual concept change. This is the central budget-waste narrative.

### CSV columns

| Column | Description |
|--------|-------------|
| `policy_type` | Policy identifier |
| `drift_type` | Drift type |
| `acc_gain_per_retrain` | (mean post-drift accuracy − baseline) / mean retrains after drift |
| `pre_drift_waste_frac` | mean retrains before drift / mean total retrains |

---

## Output 4 — Latency Sensitivity

**Files:** `table4_latency_sensitivity_{dataset}.csv`, `fig4_latency_sensitivity_{dataset}.png`

### What it shows

**Line plots** of post-drift accuracy vs total latency for each policy, with **one subplot per drift type**. The no-retrain baseline appears as a flat horizontal dashed line (it has no latency).

### How to read the figure

- **Subplots** (left to right) = drift types (Abrupt, Gradual, Recurring).
- **X-axis** = total latency in timesteps (log-scaled if range is wide). Levels: 11 (Low), 105 (Medium), 520 (High), and optionally 3 (Near-Zero) or 2050 (Extreme).
- **Lines** = one per active policy, with markers and error bars (± 1 std).
- **Dashed line** = no-retrain baseline (flat, since it has no latency).

### What to look for

| Pattern | Interpretation |
|---------|---------------|
| Downward-sloping line | Policy degrades with increasing latency — stale-weight cost dominates |
| Flat line | Policy is robust to latency changes |
| Upward-sloping line | Counter-intuitive: higher latency → fewer completed retrains → fewer stale-model windows → net accuracy gain |
| Line crossing the baseline | At some latency level, the policy becomes worse than not retraining at all |
| Periodic line drops sharply at high latency | **Interval-vs-latency collision**: when total latency (520) exceeds the periodic interval (500 for K=20), the policy can only execute ~half its budget, and retrains overlap |

### Key story: periodic policy collision

The periodic policy with K=20 has an interval of 500 timesteps. At high latency (total = 520), the latency window from one retrain overlaps the next scheduled retrain, effectively halving the usable budget. This shows up as a sharp accuracy drop at the 520 latency mark.

### CSV columns

| Column | Description |
|--------|-------------|
| `policy_type` | Policy identifier |
| `drift_type` | Drift type |
| `acc_low_latency` | Mean post-drift accuracy at the lowest latency level |
| `acc_high_latency` | Mean post-drift accuracy at the highest latency level |
| `degradation` | `acc_low_latency − acc_high_latency` — positive means accuracy dropped with higher latency |

---

## Output 5 — Cross-Dataset Summary

**Files:** `fig_cross_dataset_summary.png`, `table_cross_dataset_summary.csv`

### What it shows

A single **grouped bar chart** with one group per dataset (Synthetic, LUFlow, LendingClub), and one bar per policy within each group. The y-axis is grand-mean post-drift accuracy across all conditions (all drift types, budgets, latencies, seeds).

### How to read it

- **X-axis** = datasets.
- **Bars** = one per policy, colored consistently.
- **Bar labels** = accuracy value to 3 decimal places.

### What to look for

| Pattern | Interpretation |
|---------|---------------|
| Same colour bar is tallest across all datasets | That policy is consistently the best — finding generalises |
| Rankings flip between datasets | Policy performance is data-dependent — findings do not generalise |
| Bars within a group are close together | All policies perform similarly on this dataset — drift may not be severe enough to differentiate |
| No-retrain bar ≈ other bars | Retraining does not help much on this dataset |

### CSV columns

| Column | Description |
|--------|-------------|
| `dataset` | Dataset name (synthetic, luflow, lendingclub) |
| `policy_type` | Policy identifier |
| `mean_post_drift_accuracy` | Grand-mean post-drift accuracy across all conditions |
| `std_post_drift_accuracy` | Standard deviation across all runs |
| `n_runs` | Number of runs contributing to the mean |

---

## Quick Reference — File Inventory

For each dataset `{ds}` ∈ {`synthetic`, `luflow`, `lendingclub`}:

```
{results_dir}/cross_policy_comparison/
├── {ds}/
│   ├── table1_postdrift_accuracy_{ds}.csv
│   ├── fig1_postdrift_heatmap_{ds}.png
│   ├── table2_budget_faceted_{ds}.csv
│   ├── fig2_budget_faceted_{ds}.png
│   ├── table3_budget_efficiency_{ds}.csv
│   ├── fig3_budget_efficiency_{ds}.png
│   ├── table4_latency_sensitivity_{ds}.csv
│   └── fig4_latency_sensitivity_{ds}.png
├── fig_cross_dataset_summary.png
└── table_cross_dataset_summary.csv
```

Where `{results_dir}` is `results_with_retrain` or `results_without_retrain`.

**Total:** 8 files per dataset × 3 datasets + 2 cross-dataset files = **26 output files** (per experiment mode).

