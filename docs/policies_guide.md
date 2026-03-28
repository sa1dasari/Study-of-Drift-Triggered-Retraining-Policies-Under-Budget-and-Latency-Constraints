# Retraining Policies Guide

## Overview

Three retraining policies are compared to answer the core research question: *How do different model-refresh strategies trade off accuracy, cost, and latency under concept drift?*
Every policy inherits from `RetrainPolicy`, which enforces a shared **budget constraint** (maximum `K` retrains per stream) and a **latency guard** (no new retrain may start while a prior retrain + deploy window is still active).

---

## 1. Periodic Retraining (`PeriodicPolicy`)

**Strategy:** Retrain the model at fixed timestep intervals, regardless of observed performance or drift.

### Trigger Condition

```python
should_retrain = (t % interval == 0) and (remaining_budget > 0) and not in_latency_period
```

### Parameters Used in Experiments

| Budget (K) | Interval | Rationale |
|---|---|---|
| 5 | 2,000 | `10,000 / 5 = 2,000` — evenly spaces 5 retrains across the stream |
| 10 | 1,000 | `10,000 / 10 = 1,000` — evenly spaces 10 retrains |
| 20 | 500 | `10,000 / 20 = 500` — evenly spaces 20 retrains |

### Strengths & Weaknesses

| Aspect | Detail |
|---|---|
| ✅ Simplicity | No tuning required beyond choosing the interval; fully predictable schedule |
| ✅ Deterministic budget use | Every retrain fires on schedule until budget or latency blocks it |
| ⚠️ Drift-unaware | Retrains fire even when the concept is stable (wasted budget) |
| ⚠️ May miss drift | If drift occurs between two retrain points, the model degrades until the next scheduled retrain |
| 🔑 Key finding | Under high latency (520 timesteps), even a budget of 20 may only execute ~10 retrains because the latency window consumes the entire interval between scheduled retrains |

---

## 2. Error-Threshold Retraining (`ErrorThresholdPolicy`)

**Strategy:** Retrain the model when the rolling error rate over a recent window exceeds a calibrated threshold.

### Trigger Condition

```python
recent_errors = metrics.errors[-window_size:]
error_rate = mean(recent_errors)
should_retrain = (error_rate > threshold) and (remaining_budget > 0) and not in_latency_period
```

### Parameters Used in Experiments

| Parameter | Value | Calibration Notes |
|---|---|---|
| `error_threshold` | 0.27 | Calibrated in Week 4 — chosen so the policy fires shortly after drift begins, but not during normal noise fluctuations |
| `window_size` | 200 | Sliding window of the most recent 200 predictions |

### Strengths & Weaknesses

| Aspect | Detail |
|---|---|
| ✅ Performance-aware | Only retrains when the model is actually performing poorly |
| ✅ Budget-efficient | Does not waste retrains during stable periods |
| ⚠️ Threshold sensitivity | A threshold too low causes false alarms pre-drift; too high delays reaction post-drift |
| ⚠️ Gradual drift blind spot | Gradual drift may never push the rolling error rate above 0.27 in one window, delaying detection |
| 🔑 Key finding | Under some seeds (e.g., seed 42, abrupt drift), all 5 budget retrains fire *before* drift at t = 5000, leaving zero retrains for the actual drift event — the threshold was exceeded by normal noise |

---

## 3. Drift-Triggered Retraining (`DriftTriggeredPolicy`)

**Strategy:** Retrain the model when ADWIN (ADaptive WINdowing) detects a statistically significant change in the error distribution.

### Trigger Condition

```python
recent_errors = metrics.errors[-window_size:]
drift_detected = adwin_detect(recent_errors)  # Hoeffding-bound split test
should_retrain = drift_detected and (remaining_budget > 0) and not in_latency_period
```

### ADWIN Detection Logic

ADWIN scans every valid split point in the error window. For each split into left (n₁ samples) and right (n₂ samples) sub-windows:

```
ε = sqrt( ln(4/δ) / (2·m) )    where m = 1 / (1/n₁ + 1/n₂)
if |mean_left − mean_right| ≥ ε  →  drift detected
```

The minimum split size is `max(30, n // 10)` to avoid noise-driven false detections on tiny sub-windows.

### Parameters Used in Experiments

| Parameter | Value | Calibration Notes |
|---|---|---|
| `delta` (δ) | 0.002 | Calibrated in Week 5 — low δ reduces false positives; higher sensitivity was too noisy pre-drift |
| `window_size` | 500 | Maximum recent errors considered for detection |
| `min_samples` | 300 | Detection does not activate until 300 predictions have been made (avoids warm-up false alarms) |

### Strengths & Weaknesses

| Aspect | Detail |
|---|---|
| ✅ Change-aware | Detects *distributional shifts* rather than just high error rates, targeting the root cause of drift |
| ✅ Budget-conservative | With δ = 0.002 and min_samples = 300, the policy rarely fires pre-drift, reserving budget for real drift events |
| ⚠️ Seed sensitivity | For seeds where the baseline error pattern is noisy (e.g., seed 42, seed 123 under abrupt drift), ADWIN may never trigger, resulting in 0 retrains and 0 % budget utilization |
| ⚠️ Recurring drift challenge | With 5 concept switches and high latency, the detector may trigger on each switch but run out of budget before the stream ends |
| 🔑 Key finding | Under seeds 42 and 123 with abrupt drift, ADWIN did not detect drift at all (0 retrains across all budgets/latencies), while seed 456 fully utilised the budget — demonstrating high variance across seeds. The Phase 2 expansion to 10 seeds provides a broader picture of this seed sensitivity. |
