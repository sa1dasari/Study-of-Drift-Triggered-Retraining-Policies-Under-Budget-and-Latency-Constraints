# Concept Drift Simulation Guide

## Overview

Concept drift occurs when the statistical relationship between input features and the target variable changes over time.
In this simulator, drift is modeled by switching the weight vector `w` that defines the logistic data-generating process `P(y=1|x) = σ(x · w)`.
Two weight vectors (`w₁` = pre-drift, `w₂` = post-drift) are drawn randomly per seed, and the drift type determines how and when the switch occurs.

---

## Three Types of Concept Drift Simulated

### 1. Sudden / Abrupt Drift

**Mechanism:** The weight vector switches instantaneously from `w₁` to `w₂` at `drift_point = 5000`.

```
t < 5000  →  w = w₁   (original concept)
t ≥ 5000  →  w = w₂   (new concept — immediate switch)
```

**Characteristics:**
- Produces a sharp accuracy cliff at `t = 5000`.
- The model must detect and adapt to the sudden change quickly.
- Ideal for testing a policy's reaction speed to a clear distributional break.

### 2. Gradual Drift

**Mechanism:** The weight vector transitions smoothly from `w₁` to `w₂` over a 1,000-timestep window (`t ∈ [5000, 6000]`) via linear interpolation.

```
t < 5000           →  w = w₁
5000 ≤ t ≤ 6000    →  w = (1 − α)·w₁ + α·w₂,   where α = (t − 5000) / 1000
t > 6000           →  w = w₂
```

**Characteristics:**
- Accuracy degrades slowly rather than dropping in one step.
- Harder to detect than abrupt drift because the signal is diluted over time.
- Tests whether a policy can identify subtle, creeping performance loss before it becomes severe.

### 3. Recurring / Cyclical Drift

**Mechanism:** After `drift_point = 5000`, the active concept alternates between `w₂` and `w₁` every `recurrence_period` timesteps (default 1,000).

```
t < 5000                               →  w = w₁
5000 ≤ t < 6000  (period 0, even)      →  w = w₂   (drifted)
6000 ≤ t < 7000  (period 1, odd)       →  w = w₁   (original returns)
7000 ≤ t < 8000  (period 2, even)      →  w = w₂   (drifted again)
…and so on
```

**Characteristics:**
- The concept keeps switching back and forth, creating repeated accuracy drops and recoveries.
- Challenges the retraining budget because each concept switch may trigger a new retrain.
- Tests a policy's ability to operate under limited budget when drift events are frequent.
- Particularly stresses the budget constraint: with 5 concept switches in 5,000 post-drift timesteps, a budget of K = 5 leaves only one retrain per switch.

---

## Data Generation Parameters (Shared Across All Experiments)

| Parameter | Value | Description |
|---|---|---|
| `n_features` | 10 | Number of input features (i.i.d. standard-normal) |
| `n_samples` | 10,000 | Total stream length |
| `drift_point` | 5,000 | Timestep where drift begins |
| `recurrence_period` | 1,000 | Concept switch period for recurring drift |
| `seeds` (Phase 1) | [42, 123, 456] | 3 random seeds for initial exploration |
| `seeds` (Phase 2) | [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021] | 10 random seeds for extended variance estimation |
| Label model | `Bernoulli(σ(X · w))` | Logistic probability from linear model |
