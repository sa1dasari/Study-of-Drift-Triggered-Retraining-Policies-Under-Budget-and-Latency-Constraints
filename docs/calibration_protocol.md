# Calibration Protocol for Policy Hyperparameters

## Purpose

Before running the full-factorial experiment, we run a small calibration sweep to
find optimal hyperparameter values for the two adaptive retraining policies:

| Policy | Hyperparameters Calibrated |
|---|---|
| **Error-Threshold** | `error_threshold`, `window_size` |
| **Drift-Triggered (ADWIN)** | `delta`, `window_size`, `min_samples` |

Without calibration, poorly chosen values lead to either:
- **Too many false alarms** (retraining pre-drift wastes budget on noise), or
- **No detection at all** (policy never fires, equivalent to no-retrain baseline)

---

## Methodology

### Fixed Conditions

All calibration runs use a single, controlled configuration to isolate the effect
of each hyperparameter:

| Parameter | Value | Rationale |
|---|---|---|
| Budget (K) | 10 | Mid-range; enough budget to observe multiple retrains |
| Latency | retrain=10, deploy=1 (Low) | Minimal latency so it doesn't mask policy behaviour |
| Drift type | Abrupt | Cleanest signal for measuring false alarms vs. detection delay |
| Seed | 42 (synthetic) / offset=0 (fraud) | Single reproducible run per configuration |
| n_samples | 10,000 (synthetic) / 50,000 (fraud) | Standard stream length |
| drift_point | n_samples / 2 | Midpoint; equal pre/post-drift observation windows |

### Sweep Grids

**Error-Threshold Policy** (15 combinations per dataset):

| Parameter | Values Swept |
|---|---|
| `error_threshold` | 0.20, 0.25, 0.30, 0.35, 0.40 |
| `window_size` | 100, 200, 500 |

**Drift-Triggered (ADWIN) Policy** (54 combinations per dataset):

| Parameter | Values Swept |
|---|---|
| `delta` | 0.05, 0.01, 0.005, 0.002, 0.001, 0.0005 |
| `window_size` | 300, 500, 1000 |
| `min_samples` | 50, 100, 200 |

### Selection Criteria

For each configuration, two key metrics are measured:

1. **False alarms** = number of retrains triggered *before* the drift point (t < drift_point).
   These are wasted budget — the model retrains on stable data for no benefit.

2. **Detection delay** = timesteps between drift_point and the *first* post-drift retrain.
   Shorter is better — the policy should react quickly after real drift occurs.

The best configuration is selected by:
1. **Filter**: false_alarms <= 1 (zero or one false alarm)
2. **Sort**: shortest detection_delay
3. **Tie-break**: highest post_drift_accuracy

---

## Drift Diagnosis: Does Meaningful Drift Exist?

Before interpreting calibration results, we first verified that genuine
distributional drift exists at the chosen split point (t = n_samples / 2).

**Method**: Train SGDClassifier on 80% of pre-drift half, evaluate on:
- (a) Held-out 20% of pre-drift → same-distribution baseline
- (b) Full post-drift half → cross-distribution test

| Offset | Acc (pre) | Acc (post) | Acc Drop | F1 (pre) | F1 (post) | F1 Drop | KS sig features |
|---|---|---|---|---|---|---|---|
| 0 | 0.7944 | 0.7743 | +0.0201 | 0.1288 | 0.0990 | +0.0298 | 190 / 428 (44%) |
| 10,000 | 0.8120 | 0.7964 | +0.0156 | 0.1408 | 0.1206 | +0.0202 | 168 / 428 (39%) |
| 20,000 | 0.5646 | 0.5108 | +0.0538 | 0.1023 | 0.0733 | +0.0289 | 108 / 428 (25%) |
| **Average** | | | **+0.0298** | | | **+0.0263** | **36.3%** |

**Verdict: STRONG DRIFT.** Meaningful distributional drift exists at the midpoint
split across all three window offsets. The model trained on pre-drift data
performs significantly worse on post-drift data (+2.98% accuracy drop, +2.63%
F1 drop on average). Over one-third of features show statistically significant
distributional shifts (KS test, p < 0.001). The fraud rate itself also shifts
slightly (−0.35 percentage points post-drift).

This confirms that ADWIN's failure to detect drift is a **property of the
detector, not the data** — genuine drift exists but ADWIN cannot separate it
from the noise inherent in a ~5% error stream.

### Non-Uniform Drift Across Temporal Windows

The three offsets reveal that drift is **not uniform** across the dataset:

- **Offset=0** has the strongest *feature-level* shift (190/428 = 44% of features
  significant at p<0.001) but a moderate accuracy drop (+0.0201).
- **Offset=20,000** has a weaker feature-level shift (108/428 = 25%) but the
  *largest* accuracy drop (+0.0538).

This decoupling — more features shifting doesn't necessarily mean more accuracy
degradation — is characteristic of real-world financial data where fraud patterns
evolve in complex, non-stationary ways. Some temporal windows exhibit broad
feature-level distributional shift that doesn't strongly affect the classification
boundary, while others show concentrated shifts in decision-relevant features that
degrade model performance more severely.

**Note:** These static train/test accuracy drops (measured without streaming
`partial_fit`) did not translate to the streaming pipeline. See the sanity-check
section below for the streaming results that led to disregarding this dataset.

*(Full diagnosis output: `results/calibration/drift_diagnosis.txt`)*
*(Reproducible via: `python diagnose_drift.py`)*

## How to Reproduce

Run the calibration script from the project root:

```bash
# Calibrate on synthetic data only
python calibrate.py --dataset synthetic

# Calibrate on fraud detection data only (requires actual CSV files)
python calibrate.py --dataset fraud

# Calibrate on both datasets
python calibrate.py --dataset both
```

### Output Files

| File | Description |
|---|---|
| `results/calibration/calibration_error_threshold_fraud.csv` | All 15 error-threshold sweep results (fraud) |
| `results/calibration/calibration_drift_triggered_fraud.csv` | All 54 ADWIN sweep results (fraud) |
| `results/calibration/calibration_summary.txt` | Best configs per dataset |

### Total Cost

| Dataset | Error-Threshold Runs | ADWIN Runs | Total | Runtime |
|---|---|---|---|---|
| Fraud Detection (50k samples) | 15 | 54 | 69 | ~40-60 minutes (est.) |

This is a small cost (<1% of the 1,833 full experiment runs) for much more
defensible and reproducible parameter choices.

---

## Final Calibrated Values Used in Experiments

### CIS Fraud Detection (`main_fraud_detection.py`)

```python
# Fraud-specific calibration (calibrate.py --dataset fraud)
POLICY_PARAMS = {
    "periodic": {},
    "error_threshold": {"error_threshold": 0.30, "window_size": 100},
    "drift_triggered": {"delta": 0.0005, "window_size": 1000, "min_samples": 200},
    "no_retrain": {},
}
```

---

## Real-Data Sanity Check: Why CIS Fraud Detection Was Disregarded

### Motivation

After calibration confirmed feature-level drift and selected policy parameters, we
ran three-configuration sanity checks (no-retrain, periodic K=10 low latency,
periodic K=5 high latency) at three temporal offsets to verify that the drift
signal produces measurable **performance degradation** — the prerequisite for a
meaningful policy comparison.

### Setup

- Stream: 50,000 rows, drift point at t=25,000
- Drift type: abrupt (organic temporal drift, no construction)
- Metric: F1 on the fraud class (accuracy is uninformative at 97% majority class)
- Three offsets tested: 0, 20,000, and 40,000

### Results

| Offset | No-Retrain F1 Drop | K=10 Low-Lat F1 Drop | K=5 High-Lat F1 Drop | Viable? |
|---|---|---|---|---|
| 0 | +0.0008 (flat) | −0.0212 | −0.0052 | ✗ No degradation in baseline |
| 20,000 | −0.0027 | −0.0045 | −0.0040 | ~ Mild, weak policy ordering |
| 40,000 | +0.0288 (improvement) | +0.0288 | +0.0249 | ✗ Wrong direction |

Accuracy improved post-drift at all three offsets (+0.02 to +0.04), confirming it
is dominated by the majority class and cannot distinguish policies.

### Conclusion: Dataset Disregarded

Only 1 of 3 offsets (20,000) showed any F1 degradation, and the signal was mild
(−0.0027 for no-retrain). The other two offsets showed flat or improving F1
post-drift. A full-factorial experiment requires at least 3 consistent seed
offsets where drift degrades performance; this dataset cannot provide them.

**Root cause:** Feature-level distributional shift (confirmed by KS tests at
190/428 features, p<0.001) does not translate to consistent task-level
degradation. The fraud prediction task sometimes gets *easier* after the drift
point, depending on the temporal window. This is characteristic of real-world
financial data where distributional shift and task difficulty are decoupled.

The calibration, diagnosis, and sanity-check infrastructure (`calibrate.py`,
`diagnose_drift.py`, `main_fraud_detection.py`) are retained in the repository
as documentation of the investigation.

