# Statistical Significance Tests — Results Interpretation Guide

This document explains how to read and interpret the outputs of `statistical_significance_tests.py`, which performs paired statistical significance tests across all policy pairs and drift types on the experiment results.

For general CSV column definitions and per-policy dashboard panel guides, see [results_interpretation_guide.md](results_interpretation_guide.md). For cross-policy comparison outputs, see [cross_policy_comparison_guide.md](cross_policy_comparison_guide.md).

---

## Overview

The statistical significance tests complement the descriptive cross-policy comparison by answering: *Are the observed performance differences between policies statistically significant, or could they be due to random seed variation?*

The script supports both result sets (`results_with_retrain` and `results_without_retrain`) and produces outputs at two levels of granularity.

| Level | Scope | Independence | Use as evidence |
|-------|-------|-------------|-----------------|
| **Grand-level** | One row per (policy_A, policy_B, drift_type) | Independent — n = #seeds observations | **Primary evidence** for significance |
| **Cell-level** | One row per (policy_A, policy_B, drift_type, budget, latency) | Correlated — same seed appears in every cell | **Exploratory detail** only |

---

## Methodology

### Pairing Strategy

Observations are paired by `random_seed`. For all policy pairs (including pairs of two retraining policies), the metric is **first averaged over the budget × latency grid per seed**, then paired. This yields `n = #seeds` independent paired observations per comparison — conservative but statistically clean.

For pairs involving the no-retrain baseline (which has no budget/latency grid), the metric value is already one-per-seed, so averaging is a no-op.

### Statistical Tests

| Test | Type | Requirements | Purpose |
|------|------|-------------|---------|
| **Paired t-test** | Parametric | n ≥ 2, non-zero variance | Tests whether the mean of paired differences is significantly different from zero |
| **Wilcoxon signed-rank** | Non-parametric | n ≥ 5, at least one non-zero difference | Distribution-free alternative to the paired t-test |

### Effect Size & Confidence Interval

| Measure | Formula | Interpretation |
|---------|---------|---------------|
| **Cohen's d (paired)** | `mean(diffs) / std(diffs, ddof=1)` | Small: 0.2, Medium: 0.5, Large: 0.8 |
| **95% CI** | `mean ± t_crit × SE` (t-distribution, df = n−1) | If CI excludes zero, the difference is significant at α = 0.05 |

### Multiple Testing Correction

**Holm-Bonferroni correction** is applied to all p-values (t-test and Wilcoxon separately) within each dataset. This controls the family-wise error rate at α = 0.05 while being less conservative than plain Bonferroni.

### Significance Flags

| Flag | Criterion | Meaning |
|------|----------|---------|
| `sig_t_005` | Holm-corrected t-test p < 0.05 | Statistically significant (parametric) |
| `sig_w_005` | Holm-corrected Wilcoxon p < 0.05 | Statistically significant (non-parametric) |
| `practical_sig` | \|Cohen's d\| > 0.5 **AND** \|mean_diff\| > 0.02 | Practically significant (medium effect + 2 pp difference) |

A comparison is considered robust when **both** statistical and practical significance are flagged.

---

## Grand-Level Significance Table — Column Reference

**File:** `{results_dir}/statistical_tests/{dataset}/grand_significance_{dataset}.csv`

### Identification Columns

| Column | Description |
|--------|-------------|
| `policy_a` | First policy in the pair |
| `policy_b` | Second policy in the pair |
| `drift_type` | Drift type (abrupt, gradual, recurring) |

### Sample & Difference Columns

| Column | Description |
|--------|-------------|
| `n_pairs` | Number of paired observations (= number of shared seeds) |
| `mean_a` | Mean metric for policy A (averaged over budget × latency grid, then over seeds) |
| `mean_b` | Mean metric for policy B |
| `mean_diff` | `mean_a − mean_b` — positive means A outperforms B |
| `ci_lower` | Lower bound of the 95% CI for the mean difference |
| `ci_upper` | Upper bound of the 95% CI for the mean difference |
| `cohens_d` | Cohen's d effect size (paired) |

### Paired t-test Columns

| Column | Description |
|--------|-------------|
| `t_stat` | Paired t-test statistic |
| `t_pvalue` | Raw (uncorrected) p-value |
| `t_pvalue_adj` | Holm-Bonferroni corrected p-value |
| `sig_t_005` | `True` if `t_pvalue_adj` < 0.05 |

### Wilcoxon Signed-Rank Columns

| Column | Description |
|--------|-------------|
| `wilcoxon_stat` | Wilcoxon test statistic |
| `wilcoxon_pvalue` | Raw (uncorrected) p-value |
| `wilcoxon_pvalue_adj` | Holm-Bonferroni corrected p-value |
| `sig_w_005` | `True` if `wilcoxon_pvalue_adj` < 0.05 |

### Practical Significance

| Column | Description |
|--------|-------------|
| `practical_sig` | `True` if \|Cohen's d\| > 0.5 AND \|mean_diff\| > 0.02 |

---

## Cell-Level Significance Table — Column Reference

**File:** `{results_dir}/statistical_tests/{dataset}/cell_significance_{dataset}.csv`

Same columns as the grand-level table, plus:

| Column | Description |
|--------|-------------|
| `budget` | Budget level (5, 10, 20) |
| `total_latency` | Total latency (retrain + deploy) |

> **Warning:** Cell-level observations are NOT independent across cells. The same seed appears once per budget × latency combination within a drift type. These results are useful for exploratory breakdowns (e.g., "does significance hold at K=5 but not K=20?") but should **not** be cited as primary evidence in a paper.

---

## Significance Overview Files

### Per-Result-Set Overview

**File:** `{results_dir}/statistical_tests/significance_overview.csv`

Concatenates all grand-level rows across datasets for a single result set, with an additional `dataset` column.

### Combined Cross-Mode Overview

**File:** `results_combined_statistical_tests/significance_overview_combined.csv`

Concatenates grand-level rows across **both** result sets and all datasets, with `dataset` and `results_set` columns. This is the single file to consult when comparing significance patterns across the with-partial-fit and without-partial-fit experiments.

---

## How to Interpret the Results

### Reading the Console Output

When running the script, a formatted table is printed for each dataset:

```
  Policy A           Policy B           Drift         n   d Mean          95% CI        d    t p-val     W p-val     Stat  Prac
  Periodic           Error-Threshold    abrupt        10  +0.0123  [+0.0045, +0.0201]   0.812  0.0034 *    0.0051 *    YES  YES
```

- **Stars** (`*`, `**`, `***`) indicate raw p-value significance levels (0.05, 0.01, 0.001).
- **Stat = YES** means the comparison is statistically significant after Holm-Bonferroni correction.
- **Prac = YES** means the comparison is practically significant (\|d\| > 0.5 and \|Δ\| > 0.02).

### Decision Framework

| Stat Sig? | Prac Sig? | Interpretation |
|-----------|-----------|---------------|
| YES | YES | **Strong evidence** — the difference is real and meaningful |
| YES | NO | **Statistically detectable** but the effect is too small to matter in practice |
| NO | YES | **Large observed difference** but insufficient evidence to rule out chance (often n is too small) |
| NO | NO | **No evidence** of a meaningful difference between the two policies |

### Common Patterns

| Pattern | Meaning |
|---------|---------|
| All comparisons involving no-retrain are significant | Retraining policies provide a genuine advantage over the baseline |
| Periodic vs drift-triggered is not significant | These two policies perform comparably on this dataset/drift type |
| Statistical significance without practical significance | The difference exists but is < 2 percentage points — unlikely to matter in deployment |
| Cell-level shows significance at K=5 but not K=20 | The policy advantage is budget-dependent |

---

## Quick Reference — File Inventory

For each result set `{results_dir}` ∈ {`results_with_retrain`, `results_without_retrain`}:

```
{results_dir}/statistical_tests/
├── synthetic/
│   ├── grand_significance_synthetic.csv
│   └── cell_significance_synthetic.csv
├── luflow/
│   ├── grand_significance_luflow.csv
│   └── cell_significance_luflow.csv
├── lendingclub/
│   ├── grand_significance_lendingclub.csv
│   └── cell_significance_lendingclub.csv
└── significance_overview.csv

results_combined_statistical_tests/
└── significance_overview_combined.csv
```

**Total:** 2 files per dataset × 3 datasets + 1 overview = **7 files per result set**, plus 1 combined cross-mode overview = **15 output files** across both result sets.

