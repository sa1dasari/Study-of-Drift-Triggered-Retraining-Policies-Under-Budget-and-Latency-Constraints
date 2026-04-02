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

**Primary condition: offset=0.** For the main experiment, offset=0 is the cleanest
baseline: it has the strongest feature-level drift signal, sensible accuracy drops
across all metrics, and corresponds to the natural start-of-dataset midpoint split.
The other offsets (10,000 and 20,000) provide cross-validation of drift presence
but offset=0 should be treated as the primary analysis condition.

*(Full diagnosis output: `results/calibration/drift_diagnosis.txt`)*
*(Reproducible via: `python diagnose_drift.py`)*

---

## Results: CIS Fraud Detection Dataset

**Dataset properties**: 590,540 rows × 428 features, fraud rate 3.50% (20,663 frauds).

### Error-Threshold Sweep (15 runs)

| error_threshold | window_size | False Alarms | Detection Delay | Total Retrains | Post-Drift Acc |
|---|---|---|---|---|---|
| 0.20 | 100 | 10 | 25000 (none) | 10 | 0.9506 |
| 0.20 | 200 | 5 | 12524 | 10 | 0.9503 |
| 0.20 | 500 | 0 | 25000 (none) | 0 | 0.9513 |
| 0.25 | 100 | 2 | 6694 | 10 | 0.9502 |
| 0.25 | 200 | 0 | 25000 (none) | 0 | 0.9513 |
| 0.25 | 500 | 0 | 25000 (none) | 0 | 0.9513 |
| **0.30** | **100** | **0** | **6700** | **7** | **0.9495** |
| 0.30 | 200 | 0 | 25000 (none) | 0 | 0.9513 |
| 0.30 | 500 | 0 | 25000 (none) | 0 | 0.9513 |
| 0.35 | 100 | 0 | 6714 | 1 | 0.9515 |
| 0.35 | 200 | 0 | 25000 (none) | 0 | 0.9513 |
| 0.35 | 500 | 0 | 25000 (none) | 0 | 0.9513 |
| 0.40 | 100 | 0 | 25000 (none) | 0 | 0.9513 |
| 0.40 | 200 | 0 | 25000 (none) | 0 | 0.9513 |
| 0.40 | 500 | 0 | 25000 (none) | 0 | 0.9513 |

**Key observations:**
- The fraud dataset has a ~5% baseline error rate (vs. ~27% for synthetic), so only
  the most sensitive thresholds (≤0.20, ws≤200) produce false alarms.
- Only `ws=100` configurations ever detect drift at all — `ws≥200` smooths too much
  over the low ~5% error rate, making fluctuations invisible.
- `thresh=0.30, ws=100` is the best available option: 0 false alarms, detects drift
  at t≈31,700 (delay=6,700), and uses 7 of 10 budget slots for post-drift adaptation.

**Selected: `error_threshold=0.30, window_size=100`**
- False alarms: 0 | Detection delay: 6700 | Retrains: 7 | Post-drift acc: 0.9495

**Limitations of this selection:**
- **Slow detection.** A delay of 6,700 timesteps means 26.8% of the post-drift window
  (6,700 / 25,000) passes before the first retrain fires. The model operates on stale
  weights for over a quarter of the post-drift stream.
- **Small window sensitivity.** A `window_size=100` rolling window on a dataset with
  ~3.5% fraud rate means the error estimate is based on very few fraud observations
  per window (~3–4 frauds per 100 samples). This makes the policy sensitive to
  *clusters of fraud events* rather than genuine distributional shift — a random
  cluster of 6–7 frauds in a 100-sample window would spike the error rate above 0.30
  and trigger retraining, even absent real drift.

### ADWIN (Drift-Triggered) Sweep (54 runs)

Top configurations (sorted by fewest false alarms):

| delta | window_size | min_samples | False Alarms | Detection Delay | Total Retrains | Post-Drift Acc |
|---|---|---|---|---|---|---|
| **0.0005** | **1000** | **200** | **3** | **6700** | **10** | **0.9508** |
| 0.01 | 1000 | 50 | 4 | 2262 | 10 | 0.9498 |
| 0.0005 | 500 | 200 | 8 | 6674 | 10 | 0.9499 |
| 0.0005 | 500 | 100 | 9 | 6674 | 10 | 0.9507 |
| 0.0005 | 500 | 50 | 9 | 6674 | 10 | 0.9513 |
| All other 49 configs | | | 10 | 25000 (none) | 10 | ~0.950 |

**Selected: `delta=0.0005, window_size=1000, min_samples=200`** (least-bad option)
- False alarms: 3 | Detection delay: 6700 | Retrains: 10 | Post-drift acc: 0.9508

### Critical Analysis: ADWIN's Fundamental Failure on Fraud Data

The ADWIN results reveal a **systemic failure**, not a calibration problem:

- **51 out of 54 configurations** (94.4%) hit FA=10 with delay=25,000 — meaning ADWIN
  exhausted the entire retraining budget pre-drift and never detected the actual drift
  event. This is not sensitivity that can be tuned away.
- **The 3 exceptions** (delta=0.0005, ws=500, rows 49–51 in the CSV) achieved FA=8 or
  FA=9: 8–9 false alarms before drift, leaving only 1–2 retrains for actual adaptation.
- **The "recommended" configuration** (delta=0.0005, ws=1000, ms=200, FA=3) is the
  **least bad option, not a good one**. Three false alarms pre-drift with budget K=10
  means 30% of the retraining budget is wasted before drift starts.

**Root cause**: ADWIN's Hoeffding-bound change detector operates on a binary error
stream (correct/incorrect per sample). With a ~5% baseline error rate, the error stream
is ~95% zeros and ~5% ones. Even tiny random fluctuations in this sparse stream
produce statistically "significant" changes under the Hoeffding bound, triggering
false alarms. The signal-to-noise ratio is inherently poor for ADWIN on imbalanced
classification tasks with low overall error rates.

The drift diagnosis (above) confirms that
meaningful distributional drift *does* exist at the split point — the problem is
specifically that ADWIN cannot distinguish real drift from noise on this type of
real-world data. This is exactly the kind of practical finding that belongs in a paper
studying when and how to retrain: **ADWIN's sensitivity to class imbalance and error
stream noise makes it unsuitable for low-error-rate fraud detection data, even when
genuine drift is present.**

---

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

