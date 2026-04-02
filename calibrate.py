"""
Practical Calibration Protocol for Error-Threshold and Drift-Triggered Policies.

Runs a small hyperparameter sweep BEFORE the full experiment to find optimal
values of (error_threshold, window_size) and (delta, window_size, min_samples)
for each dataset.

Sweep design (per dataset):
  ┌────────────────────────┬────────────────────────────────────────────────┐
  │ Error-Threshold Policy │ error_threshold × window_size                 │
  │                        │ [0.20, 0.25, 0.30, 0.35, 0.40] × [100,200,500]│
  │                        │ = 15 runs per dataset                         │
  ├────────────────────────┼────────────────────────────────────────────────┤
  │ Drift-Triggered (ADWIN)│ delta × window_size × min_samples             │
  │                        │ [0.05,0.01,0.005,0.002,0.001,0.0005]          │
  │                        │  × [300,500,1000] × [50,100,200]              │
  │                        │ = 54 runs per dataset                         │
  └────────────────────────┴────────────────────────────────────────────────┘

Fixed for all runs:
    budget       = 10    (K=10)
    latency      = low   (retrain=10, deploy=1)
    seed         = 42    (synthetic) / offset=0 (fraud)
    drift_type   = abrupt
    n_samples    = 10,000 (synthetic) / 50,000 (fraud)
    drift_point  = n_samples // 2

Criteria for selecting the best configuration:
    1. Zero or one false alarm pre-drift  (retrains before drift_point)
    2. Shortest detection delay post-drift (first retrain after drift_point)
    3. Tie-break: higher post-drift accuracy

Usage:
    python calibrate.py                     # both datasets
    python calibrate.py --dataset synthetic # synthetic only
    python calibrate.py --dataset fraud     # fraud detection only

Outputs:
    results/calibration/calibration_error_threshold_synthetic.csv
    results/calibration/calibration_drift_triggered_synthetic.csv
    results/calibration/calibration_error_threshold_fraud.csv
    results/calibration/calibration_drift_triggered_fraud.csv
    results/calibration/calibration_summary.txt
    docs/calibration_protocol.md  (updated with final chosen values)
"""

import argparse
import csv
import time
from pathlib import Path


from src.data.drift_generator import DriftGenerator
from src.data.fraud_data_loader import FraudDataLoader
from src.data.real_drift_generator import build_drift_stream
from src.models.base_model import StreamingModel
from src.policies.error_threshold_policy import ErrorThresholdPolicy
from src.policies.drift_triggered_policy import DriftTriggeredPolicy
from src.evaluation.metrics import MetricsTracker
from src.runner.experiment_runner import ExperimentRunner


# ─── Calibration grids ──────────────────────────────────────────────────

ERROR_THRESHOLD_GRID = {
    "error_threshold": [0.20, 0.25, 0.30, 0.35, 0.40],
    "window_size":     [100, 200, 500],
}

ADWIN_GRID = {
    "delta":       [0.05, 0.01, 0.005, 0.002, 0.001, 0.0005],
    "window_size": [300, 500, 1000],
    "min_samples": [50, 100, 200],
}

# ─── Fixed calibration conditions ───────────────────────────────────────

BUDGET            = 10
RETRAIN_LATENCY   = 10
DEPLOY_LATENCY    = 1
DRIFT_TYPE        = "abrupt"

# Synthetic dataset params
SYN_N_SAMPLES     = 10_000
SYN_DRIFT_POINT   = 5_000
SYN_SEED          = 42

# Fraud dataset params
FRAUD_N_SAMPLES   = 50_000
FRAUD_DRIFT_POINT = 25_000
FRAUD_OFFSET      = 0          # window offset (analogous to seed)
FRAUD_DATA_DIR    = Path("src/data/CIS Fraud Detection")

OUTPUT_DIR        = Path("results/calibration_for_CISFraudDetectionDataset")


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════

def _generate_synthetic_stream():
    """Generate the synthetic abrupt-drift stream (once)."""
    gen = DriftGenerator(
        n_features=10,
        drift_type=DRIFT_TYPE,
        drift_point=SYN_DRIFT_POINT,
        recurrence_period=SYN_N_SAMPLES // 10,
        seed=SYN_SEED,
    )
    return gen.generate(n_samples=SYN_N_SAMPLES)


def _generate_fraud_stream(loader):
    """Generate the fraud detection abrupt-drift stream (once)."""
    X_pre, y_pre = loader.get_pool(start_offset=FRAUD_OFFSET,
                                    n_samples=FRAUD_N_SAMPLES)
    return build_drift_stream(
        X_pre, y_pre, X_pre, y_pre,      # post ignored for abrupt
        drift_type=DRIFT_TYPE,
        drift_point=FRAUD_DRIFT_POINT,
        recurrence_period=FRAUD_N_SAMPLES // 10,
        seed=FRAUD_OFFSET,
    )


def _run_single(X, y, policy, drift_point):
    """Run one experiment and return calibration metrics."""
    model   = StreamingModel()
    metrics = MetricsTracker()
    metrics.set_drift_point(drift_point)
    metrics.set_budget(BUDGET)

    runner = ExperimentRunner(model, policy, metrics)
    runner.run(X, y)

    summary = metrics.get_summary()

    # ── Calibration-specific metrics ────────────────────────────────
    false_alarms = metrics.get_retrains_before_drift()

    # Detection delay: timesteps between drift_point and first post-drift retrain
    post_drift_retrains = sorted(
        t for t in metrics.retrain_times if t >= drift_point
    )
    if post_drift_retrains:
        detection_delay = post_drift_retrains[0] - drift_point
    else:
        detection_delay = len(X) - drift_point   # never detected = max delay

    return {
        "false_alarms":       false_alarms,
        "detection_delay":    detection_delay,
        "total_retrains":     summary["total_retrains"],
        "retrains_pre_drift": false_alarms,
        "retrains_post_drift":summary.get("retrains_after_drift", 0),
        "overall_accuracy":   round(summary["overall_accuracy"], 5),
        "pre_drift_accuracy": round(summary.get("pre_drift_accuracy", 0) or 0, 5),
        "post_drift_accuracy":round(summary.get("post_drift_accuracy", 0) or 0, 5),
        "accuracy_drop":      round(summary.get("accuracy_drop", 0) or 0, 5),
        "retrain_times":      metrics.retrain_times,
    }


def _write_csv(rows, filepath, fieldnames):
    """Write a list of dicts to CSV."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> Saved {filepath}")


def _print_table(rows, cols, title):
    """Pretty-print a table of dicts to the console."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")

    # Column widths
    widths = {}
    for c in cols:
        widths[c] = max(len(c), max(len(str(r.get(c, ""))) for r in rows))

    # Header
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(f"  {header}")
    print(f"  {'-+-'.join('-' * widths[c] for c in cols)}")

    # Rows
    for r in rows:
        line = " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols)
        print(f"  {line}")


def _select_best(rows, max_false_alarms=1):
    """
    Select the best configuration:
      1. false_alarms <= max_false_alarms
      2. shortest detection_delay
      3. tie-break: highest post_drift_accuracy
    """
    eligible = [r for r in rows if r["false_alarms"] <= max_false_alarms]
    if not eligible:
        # Relax: pick the one with fewest false alarms
        min_fa = min(r["false_alarms"] for r in rows)
        eligible = [r for r in rows if r["false_alarms"] == min_fa]

    eligible.sort(key=lambda r: (r["detection_delay"], -r["post_drift_accuracy"]))
    return eligible[0]


# ═════════════════════════════════════════════════════════════════════════
# Sweep: Error-Threshold Policy
# ═════════════════════════════════════════════════════════════════════════

def sweep_error_threshold(X, y, drift_point, dataset_name):
    """Sweep error_threshold × window_size and return result rows."""
    thresholds  = ERROR_THRESHOLD_GRID["error_threshold"]
    window_sizes = ERROR_THRESHOLD_GRID["window_size"]
    total = len(thresholds) * len(window_sizes)
    rows = []
    count = 0
    start = time.time()

    print(f"\n{'-' * 70}")
    print(f"  Error-Threshold Calibration -- {dataset_name}  ({total} runs)")
    print(f"{'-' * 70}")
    print(f"  error_threshold values : {thresholds}")
    print(f"  window_size values     : {window_sizes}")
    print(f"  budget={BUDGET}  latency=({RETRAIN_LATENCY}+{DEPLOY_LATENCY})"
          f"  drift_point={drift_point}  n_samples={len(X)}")
    print()

    for thresh in thresholds:
        for ws in window_sizes:
            count += 1
            policy = ErrorThresholdPolicy(
                error_threshold=thresh,
                window_size=ws,
                budget=BUDGET,
                retrain_latency=RETRAIN_LATENCY,
                deploy_latency=DEPLOY_LATENCY,
            )
            result = _run_single(X, y, policy, drift_point)

            row = {
                "dataset":            dataset_name,
                "error_threshold":    thresh,
                "window_size":        ws,
                **{k: v for k, v in result.items() if k != "retrain_times"},
                "retrain_times":      str(result["retrain_times"]),
            }
            rows.append(row)

            elapsed = time.time() - start
            eta = (elapsed / count) * (total - count)
            print(
                f"  [{count:>2}/{total}]  thresh={thresh:.2f}  ws={ws:<4}  "
                f"FA={result['false_alarms']}  delay={result['detection_delay']:<6}  "
                f"retrains={result['total_retrains']:>2}  "
                f"post_acc={result['post_drift_accuracy']:.4f}  "
                f"ETA {eta:.0f}s"
            )

    return rows


# ═════════════════════════════════════════════════════════════════════════
# Sweep: Drift-Triggered (ADWIN) Policy
# ═════════════════════════════════════════════════════════════════════════

def sweep_adwin(X, y, drift_point, dataset_name):
    """Sweep delta × window_size × min_samples and return result rows."""
    deltas       = ADWIN_GRID["delta"]
    window_sizes = ADWIN_GRID["window_size"]
    min_samples  = ADWIN_GRID["min_samples"]
    total = len(deltas) * len(window_sizes) * len(min_samples)
    rows = []
    count = 0
    start = time.time()

    print(f"\n{'-' * 70}")
    print(f"  ADWIN (Drift-Triggered) Calibration -- {dataset_name}  ({total} runs)")
    print(f"{'-' * 70}")
    print(f"  delta values       : {deltas}")
    print(f"  window_size values : {window_sizes}")
    print(f"  min_samples values : {min_samples}")
    print(f"  budget={BUDGET}  latency=({RETRAIN_LATENCY}+{DEPLOY_LATENCY})"
          f"  drift_point={drift_point}  n_samples={len(X)}")
    print()

    for delta in deltas:
        for ws in window_sizes:
            for ms in min_samples:
                count += 1
                policy = DriftTriggeredPolicy(
                    delta=delta,
                    window_size=ws,
                    min_samples=ms,
                    budget=BUDGET,
                    retrain_latency=RETRAIN_LATENCY,
                    deploy_latency=DEPLOY_LATENCY,
                )
                result = _run_single(X, y, policy, drift_point)

                row = {
                    "dataset":            dataset_name,
                    "delta":              delta,
                    "window_size":        ws,
                    "min_samples":        ms,
                    **{k: v for k, v in result.items() if k != "retrain_times"},
                    "retrain_times":      str(result["retrain_times"]),
                }
                rows.append(row)

                elapsed = time.time() - start
                eta = (elapsed / count) * (total - count)
                print(
                    f"  [{count:>2}/{total}]  d={delta:<8}  ws={ws:<5}  "
                    f"ms={ms:<4}  "
                    f"FA={result['false_alarms']}  delay={result['detection_delay']:<6}  "
                    f"retrains={result['total_retrains']:>2}  "
                    f"post_acc={result['post_drift_accuracy']:.4f}  "
                    f"ETA {eta:.0f}s"
                )

    return rows


# ═════════════════════════════════════════════════════════════════════════
# Analysis & reporting
# ═════════════════════════════════════════════════════════════════════════

def analyse_and_report(et_rows, adwin_rows, dataset_name, summary_lines):
    """Analyse sweep results, print tables, and select best configs."""

    # ── Error-Threshold ─────────────────────────────────────────────
    et_cols = [
        "error_threshold", "window_size",
        "false_alarms", "detection_delay", "total_retrains",
        "overall_accuracy", "post_drift_accuracy",
    ]
    _print_table(et_rows, et_cols,
                 f"Error-Threshold Results -- {dataset_name}")

    best_et = _select_best(et_rows)
    print(f"\n  * RECOMMENDED (Error-Threshold, {dataset_name}):")
    print(f"    error_threshold = {best_et['error_threshold']}")
    print(f"    window_size     = {best_et['window_size']}")
    print(f"    false_alarms    = {best_et['false_alarms']}, "
          f"detection_delay = {best_et['detection_delay']}, "
          f"post_drift_acc  = {best_et['post_drift_accuracy']}")

    # ── ADWIN ───────────────────────────────────────────────────────
    adwin_cols = [
        "delta", "window_size", "min_samples",
        "false_alarms", "detection_delay", "total_retrains",
        "overall_accuracy", "post_drift_accuracy",
    ]
    _print_table(adwin_rows, adwin_cols,
                 f"ADWIN (Drift-Triggered) Results -- {dataset_name}")

    best_adwin = _select_best(adwin_rows)
    print(f"\n  * RECOMMENDED (ADWIN, {dataset_name}):")
    print(f"    delta       = {best_adwin['delta']}")
    print(f"    window_size = {best_adwin['window_size']}")
    print(f"    min_samples = {best_adwin['min_samples']}")
    print(f"    false_alarms    = {best_adwin['false_alarms']}, "
          f"detection_delay = {best_adwin['detection_delay']}, "
          f"post_drift_acc  = {best_adwin['post_drift_accuracy']}")

    # ── Append to summary text ──────────────────────────────────────
    summary_lines.append(f"\n{'=' * 80}")
    summary_lines.append(f"  {dataset_name} -- Calibration Results")
    summary_lines.append(f"{'=' * 80}")

    summary_lines.append(f"\n  Error-Threshold Policy -- Best Configuration:")
    summary_lines.append(f"    error_threshold = {best_et['error_threshold']}")
    summary_lines.append(f"    window_size     = {best_et['window_size']}")
    summary_lines.append(f"    false_alarms    = {best_et['false_alarms']}")
    summary_lines.append(f"    detection_delay = {best_et['detection_delay']}")
    summary_lines.append(f"    post_drift_acc  = {best_et['post_drift_accuracy']}")
    summary_lines.append(f"    total_retrains  = {best_et['total_retrains']}")

    summary_lines.append(f"\n  Drift-Triggered (ADWIN) Policy -- Best Configuration:")
    summary_lines.append(f"    delta           = {best_adwin['delta']}")
    summary_lines.append(f"    window_size     = {best_adwin['window_size']}")
    summary_lines.append(f"    min_samples     = {best_adwin['min_samples']}")
    summary_lines.append(f"    false_alarms    = {best_adwin['false_alarms']}")
    summary_lines.append(f"    detection_delay = {best_adwin['detection_delay']}")
    summary_lines.append(f"    post_drift_acc  = {best_adwin['post_drift_accuracy']}")
    summary_lines.append(f"    total_retrains  = {best_adwin['total_retrains']}")

    return best_et, best_adwin


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Calibration sweep for Error-Threshold and ADWIN policies."
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "fraud", "both"],
        default="both",
        help="Which dataset to calibrate on (default: both).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "=" * 80,
        "  Calibration Protocol -- Summary of Results",
        "=" * 80,
        f"  Date       : (auto-generated)",
        f"  Budget     : {BUDGET}",
        f"  Latency    : retrain={RETRAIN_LATENCY}, deploy={DEPLOY_LATENCY}",
        f"  Drift type : {DRIFT_TYPE}",
        "",
        "  Selection criteria:",
        "    1. <= 1 false alarm pre-drift",
        "    2. Shortest detection delay post-drift",
        "    3. Tie-break: highest post-drift accuracy",
    ]

    best_configs = {}   # {dataset: {policy: best_row}}
    grand_start = time.time()

    # ─── Synthetic Dataset ──────────────────────────────────────────
    if args.dataset in ("synthetic", "both"):
        print(f"\n{'#' * 70}")
        print(f"  CALIBRATING ON SYNTHETIC DATASET")
        print(f"  n_samples={SYN_N_SAMPLES}, drift_point={SYN_DRIFT_POINT}, seed={SYN_SEED}")
        print(f"{'#' * 70}")

        X_syn, y_syn = _generate_synthetic_stream()

        et_rows_syn    = sweep_error_threshold(X_syn, y_syn, SYN_DRIFT_POINT, "Synthetic")
        adwin_rows_syn = sweep_adwin(X_syn, y_syn, SYN_DRIFT_POINT, "Synthetic")

        # Save CSVs
        et_fields_syn = [k for k in et_rows_syn[0].keys()]
        adwin_fields_syn = [k for k in adwin_rows_syn[0].keys()]
        _write_csv(et_rows_syn, OUTPUT_DIR / "calibration_error_threshold_synthetic.csv", et_fields_syn)
        _write_csv(adwin_rows_syn, OUTPUT_DIR / "calibration_drift_triggered_synthetic.csv", adwin_fields_syn)

        best_et_syn, best_adwin_syn = analyse_and_report(
            et_rows_syn, adwin_rows_syn, "Synthetic", summary_lines
        )
        best_configs["synthetic"] = {
            "error_threshold": best_et_syn,
            "drift_triggered": best_adwin_syn,
        }

    # ─── Fraud Detection Dataset ────────────────────────────────────
    if args.dataset in ("fraud", "both"):
        print(f"\n{'#' * 70}")
        print(f"  CALIBRATING ON CIS FRAUD DETECTION DATASET")
        print(f"  n_samples={FRAUD_N_SAMPLES}, drift_point={FRAUD_DRIFT_POINT}, offset={FRAUD_OFFSET}")
        print(f"{'#' * 70}")

        loader = FraudDataLoader(FRAUD_DATA_DIR)
        X_fraud, y_fraud = _generate_fraud_stream(loader)

        et_rows_fraud    = sweep_error_threshold(X_fraud, y_fraud, FRAUD_DRIFT_POINT, "Fraud Detection")
        adwin_rows_fraud = sweep_adwin(X_fraud, y_fraud, FRAUD_DRIFT_POINT, "Fraud Detection")

        # Save CSVs
        et_fields_fraud = [k for k in et_rows_fraud[0].keys()]
        adwin_fields_fraud = [k for k in adwin_rows_fraud[0].keys()]
        _write_csv(et_rows_fraud, OUTPUT_DIR / "calibration_error_threshold_fraud.csv", et_fields_fraud)
        _write_csv(adwin_rows_fraud, OUTPUT_DIR / "calibration_drift_triggered_fraud.csv", adwin_fields_fraud)

        best_et_fraud, best_adwin_fraud = analyse_and_report(
            et_rows_fraud, adwin_rows_fraud, "Fraud Detection", summary_lines
        )
        best_configs["fraud"] = {
            "error_threshold": best_et_fraud,
            "drift_triggered": best_adwin_fraud,
        }

    # ─── Write summary text ─────────────────────────────────────────
    summary_lines.append(f"\n{'=' * 80}")
    summary_lines.append(f"  FINAL RECOMMENDED VALUES FOR main.py / main_fraud_detection.py")
    summary_lines.append(f"{'=' * 80}")

    if "synthetic" in best_configs:
        bc = best_configs["synthetic"]
        summary_lines.append(f"\n  Synthetic Dataset (main.py):")
        summary_lines.append(f'    POLICY_PARAMS["error_threshold"] = '
                             f'{{"error_threshold": {bc["error_threshold"]["error_threshold"]}, '
                             f'"window_size": {bc["error_threshold"]["window_size"]}}}')
        summary_lines.append(f'    POLICY_PARAMS["drift_triggered"] = '
                             f'{{"delta": {bc["drift_triggered"]["delta"]}, '
                             f'"window_size": {bc["drift_triggered"]["window_size"]}, '
                             f'"min_samples": {bc["drift_triggered"]["min_samples"]}}}')

    if "fraud" in best_configs:
        bc = best_configs["fraud"]
        summary_lines.append(f"\n  Fraud Detection Dataset (main_fraud_detection.py):")
        summary_lines.append(f'    POLICY_PARAMS["error_threshold"] = '
                             f'{{"error_threshold": {bc["error_threshold"]["error_threshold"]}, '
                             f'"window_size": {bc["error_threshold"]["window_size"]}}}')
        summary_lines.append(f'    POLICY_PARAMS["drift_triggered"] = '
                             f'{{"delta": {bc["drift_triggered"]["delta"]}, '
                             f'"window_size": {bc["drift_triggered"]["window_size"]}, '
                             f'"min_samples": {bc["drift_triggered"]["min_samples"]}}}')

    summary_text = "\n".join(summary_lines)
    summary_path = OUTPUT_DIR / "calibration_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\n  Summary written to {summary_path}")

    elapsed = time.time() - grand_start
    print(f"\n{'#' * 70}")
    print(f"  CALIBRATION COMPLETE -- {elapsed / 60:.1f} minutes")
    print(f"{'#' * 70}")
    print(summary_text)

    return best_configs


if __name__ == "__main__":
    main()

