"""
Main entry point for the Real-World Fraud Detection retraining-policy experiment.

Runs the same full-factorial sweep as main.py but on real CIS Fraud Detection
data instead of synthetic data.  Drift is constructed from temporal segments:

    abrupt    – stream 50k rows as-is (purely organic drift at midpoint)
    gradual   – blend two 75k temporal pools across a 5k transition window
    recurring – alternate between two 75k temporal pools after drift point
                every 5,000 rows (five concept switches)

Seeds are window offsets (0, 50000, 100000) — each slides the data window to
a different temporal region, providing variance across runs.

Full matrix per active policy:
    3 drift types × 3 budgets × 3 latencies × 3 seeds = 81 runs
    no_retrain baseline: 3 drift types × 3 seeds = 9 runs

Usage:
    python main_fraud_detection.py                            # all 4 policies
    python main_fraud_detection.py --policy periodic          # periodic only
    python main_fraud_detection.py --policy drift_triggered   # drift-triggered only
    python main_fraud_detection.py --policy error_threshold   # error-threshold only
    python main_fraud_detection.py --policy no_retrain        # baseline only
    python main_fraud_detection.py --sanity                   # single sanity-check run
    python main_fraud_detection.py --n_samples 100000         # larger stream
"""

import argparse
from pathlib import Path
import time

from src.data.fraud_data_loader import FraudDataLoader
from src.data.real_drift_generator import build_drift_stream
from src.models.base_model import StreamingModel
from src.policies.periodic import PeriodicPolicy
from src.policies.error_threshold_policy import ErrorThresholdPolicy
from src.policies.drift_triggered_policy import DriftTriggeredPolicy
from src.policies.never_retrain_policy import NeverRetrainPolicy
from src.evaluation.metrics import MetricsTracker
from src.runner.experiment_runner import ExperimentRunner
from src.evaluation.results_export import export_to_json, export_to_csv, export_summary_to_csv

# Seed sets (window offsets into the time-sorted data)
WINDOW_OFFSETS = [0, 50_000, 100_000]

# Experiment grid
DRIFT_TYPES = ["abrupt", "gradual", "recurring"]
BUDGETS = [5, 10, 20]
LATENCY_CONFIGS = [
    (10, 1),      # Low latency   (total = 11)
    (100, 5),     # Medium latency (total = 105)
    (500, 20),    # High latency  (total = 520)
]

N_SAMPLES_DEFAULT = 50_000

# Pool size for gradual/recurring (pre and post pools drawn from separate
# temporal regions).  Each pool is 75 k rows; combined with the offset the
# required data window is offset + 2 × POOL_SIZE rows.
POOL_SIZE = 75_000

# Fixed recurrence period (also used as the gradual transition window width).
# 5,000 rows → five concept switches in the 25 k post-drift window, matching
# the synthetic experiment design with recurrence_period = n_samples // 10.
RECURRENCE_PERIOD = 5_000

# Per-policy fixed parameters
# NOTE: Values calibrated on the CIS Fraud Detection dataset (590,540 rows, 428 features, 3.5% fraud rate).
#       See docs/calibration_protocol.md for full sweep results and rationale.
POLICY_PARAMS = {
    "periodic": {},
    "error_threshold": {"error_threshold": 0.30, "window_size": 100},
    "drift_triggered": {"delta": 0.0005, "window_size": 1000, "min_samples": 200},
    "no_retrain": {},
}

POLICY_DISPLAY = {
    "periodic":         "Periodic",
    "error_threshold":  "Error-Threshold",
    "drift_triggered":  "Drift-Triggered (ADWIN)",
    "no_retrain":       "No-Retrain (Baseline)",
}

# ── Data directory ──────────────────────────────────────────────────────
DATA_DIR = Path("src/data/CIS Fraud Detection")


def _build_policy(policy_type, budget, retrain_latency, deploy_latency, n_samples):
    """Instantiate the requested policy with the correct parameters."""
    if policy_type == "periodic":
        interval = n_samples // budget
        return PeriodicPolicy(
            interval=interval,
            budget=budget,
            retrain_latency=retrain_latency,
            deploy_latency=deploy_latency,
        )
    elif policy_type == "error_threshold":
        p = POLICY_PARAMS["error_threshold"]
        return ErrorThresholdPolicy(
            error_threshold=p["error_threshold"],
            window_size=p["window_size"],
            budget=budget,
            retrain_latency=retrain_latency,
            deploy_latency=deploy_latency,
        )
    elif policy_type == "drift_triggered":
        p = POLICY_PARAMS["drift_triggered"]
        return DriftTriggeredPolicy(
            delta=p["delta"],
            window_size=p["window_size"],
            min_samples=p["min_samples"],
            budget=budget,
            retrain_latency=retrain_latency,
            deploy_latency=deploy_latency,
        )
    elif policy_type == "no_retrain":
        return NeverRetrainPolicy()
    else:
        raise ValueError(f"Unknown policy_type: {policy_type!r}")


def _build_config(policy_type, drift_type, budget, seed, n_samples, drift_point):
    """Build the config dict passed to export helpers."""
    config = {
        "drift_type": drift_type,
        "drift_point": drift_point,
        "recurrence_period": RECURRENCE_PERIOD,
        "policy_type": policy_type,
        "budget": budget,
        "random_seed": seed,
        "n_samples": n_samples,
        "dataset": "CIS_Fraud_Detection",
    }
    if policy_type == "periodic":
        config["policy_interval"] = n_samples // budget
    elif policy_type == "error_threshold":
        config.update(POLICY_PARAMS["error_threshold"])
    elif policy_type == "drift_triggered":
        config.update(POLICY_PARAMS["drift_triggered"])
    return config


def _get_stream(loader, drift_type, n_samples, offset, drift_point,
                recurrence_period, seed):
    """
    Build the (X, y) stream for one run.

    For abrupt:    single pool of n_samples rows (organic drift).
    For gradual:   pre-pool [offset, offset+POOL_SIZE), post-pool
                   [offset+POOL_SIZE, offset+2*POOL_SIZE). Stream of
                   n_samples rows with a transition window of
                   recurrence_period rows.
    For recurring: same two pools; after drift_point alternate every
                   recurrence_period rows between post-pool and pre-pool.
    """
    if drift_type == "abrupt":
        # Single contiguous window — organic drift only
        X_pre, y_pre = loader.get_pool(start_offset=offset, n_samples=n_samples)
        return build_drift_stream(
            X_pre, y_pre, X_pre, y_pre,  # post-pool ignored for abrupt
            drift_type=drift_type,
            drift_point=drift_point,
            recurrence_period=recurrence_period,
            n_samples=n_samples,
            seed=seed,
        )

    # Gradual / Recurring: two temporally-separated 75 k pools
    X_pre, y_pre = loader.get_pool(
        start_offset=offset, n_samples=POOL_SIZE,
    )
    X_post, y_post = loader.get_pool(
        start_offset=offset + POOL_SIZE, n_samples=POOL_SIZE,
    )
    return build_drift_stream(
        X_pre, y_pre, X_post, y_post,
        drift_type=drift_type,
        drift_point=drift_point,
        recurrence_period=recurrence_period,
        n_samples=n_samples,
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────
# Sweep runners
# ─────────────────────────────────────────────────────────────────────────

def run_policy_sweep(policy_type, loader, n_samples):
    """Execute the full-factorial sweep for a single policy."""
    if policy_type == "no_retrain":
        return _run_no_retrain_sweep(loader, n_samples)

    drift_point = n_samples // 2
    recurrence_period = RECURRENCE_PERIOD
    seed_label = f"{len(WINDOW_OFFSETS)}seed"

    n_drifts = len(DRIFT_TYPES)
    n_budgets = len(BUDGETS)
    n_latencies = len(LATENCY_CONFIGS)
    total_runs = n_drifts * n_budgets * n_latencies * len(WINDOW_OFFSETS)

    # Output paths
    results_dir = Path(
        f"results/fraud_detection/{policy_type}_{seed_label}"
        f"_{n_drifts}drift_{n_budgets}budget_{n_latencies}latency"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = (
        f"results/fraud_detection/"
        f"summary_results_{policy_type}_retrain_{seed_label}.csv"
    )
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_csv).unlink(missing_ok=True)

    display = POLICY_DISPLAY[policy_type]
    run_count = 0
    start_time = time.time()

    print(f"\n{'=' * 70}")
    print(f"{display} POLICY – Real-Data Sweep ({total_runs} runs)")
    print(f"{'=' * 70}")
    print(f"  Dataset          : CIS Fraud Detection")
    print(f"  Stream length    : {n_samples:,}")
    print(f"  Drift point      : {drift_point:,} (midpoint)")
    print(f"  Recurrence period: {recurrence_period:,}")
    print(f"  Drift types      : {DRIFT_TYPES}")
    print(f"  Budgets          : {BUDGETS}")
    print(f"  Latency configs  : {LATENCY_CONFIGS}")
    print(f"  Window offsets   : {WINDOW_OFFSETS}")
    if policy_type == "periodic":
        print(f"  Intervals        : {[n_samples // b for b in BUDGETS]}")
    else:
        for k, v in POLICY_PARAMS[policy_type].items():
            print(f"  {k:<18}: {v}")
    print(f"{'=' * 70}\n")

    for drift_type in DRIFT_TYPES:
        for budget in BUDGETS:
            for retrain_latency, deploy_latency in LATENCY_CONFIGS:
                for offset in WINDOW_OFFSETS:
                    run_count += 1

                    # 1. Build stream
                    X, y = _get_stream(
                        loader, drift_type, n_samples, offset,
                        drift_point, recurrence_period, seed=offset,
                    )

                    # 2. Build components
                    model = StreamingModel()
                    policy = _build_policy(
                        policy_type, budget,
                        retrain_latency, deploy_latency, n_samples,
                    )
                    metrics = MetricsTracker()
                    metrics.set_drift_point(drift_point)
                    metrics.set_budget(budget)

                    # 3. Run experiment
                    runner = ExperimentRunner(model, policy, metrics)
                    runner.run(X, y)

                    # 4. Progress
                    summary = metrics.get_summary()
                    elapsed = time.time() - start_time
                    eta = (elapsed / run_count) * (total_runs - run_count)

                    print(
                        f"[{run_count:>3}/{total_runs}] "
                        f"drift={drift_type:<10} budget={budget:<3} "
                        f"latency=({retrain_latency}+{deploy_latency})  "
                        f"offset={offset:<6} | "
                        f"acc={summary['overall_accuracy']:.4f}  "
                        f"retrains={summary['total_retrains']:>2} | "
                        f"ETA {eta / 60:.1f} min"
                    )

                    # 5. Export per-run results
                    run_tag = (
                        f"{drift_type}_b{budget}"
                        f"_l{retrain_latency}+{deploy_latency}"
                        f"_off{offset}"
                    )
                    config = _build_config(
                        policy_type, drift_type, budget,
                        seed=offset, n_samples=n_samples,
                        drift_point=drift_point,
                    )

                    export_to_json(
                        metrics, policy, config,
                        str(results_dir / f"run_{run_tag}.json"),
                    )
                    export_to_csv(
                        metrics, policy, config,
                        str(results_dir / f"per_sample_{run_tag}.csv"),
                    )
                    export_summary_to_csv(metrics, policy, config, summary_csv)

    elapsed = time.time() - start_time
    print(f"\n{display}: {total_runs} runs completed in {elapsed / 60:.1f} minutes")
    print(f"Summary CSV → {summary_csv}")

    # Generate summary dashboard
    print(f"Generating {display} summary plot...")
    from plot_summary import plot_summary_for_policy
    plot_summary_for_policy(
        csv_path=summary_csv,
        output_path=(
            f"results/fraud_detection/"
            f"summary_results_plot_{policy_type}_retrain_{seed_label}.png"
        ),
        policy_name=f"{display} (Fraud Detection)",
    )

    return summary_csv


def _run_no_retrain_sweep(loader, n_samples):
    """Execute the no-retrain baseline: 3 drift types × 3 window offsets."""
    policy_type = "no_retrain"
    drift_point = n_samples // 2
    recurrence_period = RECURRENCE_PERIOD
    seed_label = f"{len(WINDOW_OFFSETS)}seed"
    total_runs = len(DRIFT_TYPES) * len(WINDOW_OFFSETS)

    # Output paths
    results_dir = Path(
        f"results/fraud_detection/{policy_type}_{seed_label}_{len(DRIFT_TYPES)}drift"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = (
        f"results/fraud_detection/"
        f"summary_results_{policy_type}_{seed_label}.csv"
    )
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_csv).unlink(missing_ok=True)

    display = POLICY_DISPLAY[policy_type]
    run_count = 0
    start_time = time.time()

    print(f"\n{'=' * 70}")
    print(f"{display} POLICY – Real-Data Baseline Sweep ({total_runs} runs)")
    print(f"{'=' * 70}")
    print(f"  Dataset          : CIS Fraud Detection")
    print(f"  Stream length    : {n_samples:,}")
    print(f"  Drift types      : {DRIFT_TYPES}")
    print(f"  Budget           : N/A (always 0)")
    print(f"  Latency          : N/A (always 0)")
    print(f"  Window offsets   : {WINDOW_OFFSETS}")
    print(f"{'=' * 70}\n")

    for drift_type in DRIFT_TYPES:
        for offset in WINDOW_OFFSETS:
            run_count += 1

            # 1. Build stream
            X, y = _get_stream(
                loader, drift_type, n_samples, offset,
                drift_point, recurrence_period, seed=offset,
            )

            # 2. Build components
            model = StreamingModel()
            policy = _build_policy(
                policy_type, budget=0,
                retrain_latency=0, deploy_latency=0,
                n_samples=n_samples,
            )
            metrics = MetricsTracker()
            metrics.set_drift_point(drift_point)
            metrics.set_budget(0)

            # 3. Run experiment
            runner = ExperimentRunner(model, policy, metrics)
            runner.run(X, y)

            # 4. Progress
            summary = metrics.get_summary()
            elapsed = time.time() - start_time
            eta = (elapsed / run_count) * (total_runs - run_count)

            print(
                f"[{run_count:>3}/{total_runs}] "
                f"drift={drift_type:<10} offset={offset:<6} | "
                f"acc={summary['overall_accuracy']:.4f}  "
                f"retrains={summary['total_retrains']:>2} | "
                f"ETA {eta / 60:.1f} min"
            )

            # 5. Export per-run results
            run_tag = f"{drift_type}_off{offset}"
            config = _build_config(
                policy_type, drift_type, budget=0,
                seed=offset, n_samples=n_samples,
                drift_point=drift_point,
            )

            export_to_json(
                metrics, policy, config,
                str(results_dir / f"run_{run_tag}.json"),
            )
            export_to_csv(
                metrics, policy, config,
                str(results_dir / f"per_sample_{run_tag}.csv"),
            )
            export_summary_to_csv(metrics, policy, config, summary_csv)

    elapsed = time.time() - start_time
    print(f"\n{display}: {total_runs} runs completed in {elapsed / 60:.1f} minutes")
    print(f"Summary CSV → {summary_csv}")

    # Generate baseline summary plot
    print(f"Generating {display} summary plot...")
    from plot_summary import plot_summary_for_no_retrain
    plot_summary_for_no_retrain(
        csv_path=summary_csv,
        output_path=(
            f"results/fraud_detection/"
            f"summary_results_plot_{policy_type}_{seed_label}.png"
        ),
        policy_name=f"{display} (Fraud Detection)",
    )

    return summary_csv


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def _run_one_sanity(loader, n_samples, label, policy_type, budget,
                    retrain_latency, deploy_latency):
    """
    Run a single sanity-check configuration and return the summary dict.

    Returns:
        (summary_dict, fraud_rate)
    """
    drift_point = n_samples // 2
    recurrence_period = RECURRENCE_PERIOD
    offset = 40_000

    # 1. Build stream (abrupt, offset=40000)
    X, y = _get_stream(
        loader, "abrupt", n_samples, offset,
        drift_point, recurrence_period, seed=offset,
    )

    # 2. Build components
    model = StreamingModel()
    policy = _build_policy(policy_type, budget, retrain_latency,
                           deploy_latency, n_samples)
    metrics = MetricsTracker()
    metrics.set_drift_point(drift_point)
    metrics.set_budget(budget)

    # 3. Run
    runner = ExperimentRunner(model, policy, metrics)
    runner.run(X, y)

    # 4. Results
    summary = metrics.get_summary()
    fraud_rate = float(y.mean())

    print(f"\n  {'─' * 56}")
    print(f"  {label}")
    print(f"  {'─' * 56}")
    if policy_type == "no_retrain":
        print(f"    Policy           : No-Retrain (partial_fit only)")
    else:
        interval = n_samples // budget
        print(f"    Policy           : Periodic K={budget}, "
              f"interval={interval:,}")
        print(f"    Latency          : ({retrain_latency}+{deploy_latency}) "
              f"= {retrain_latency + deploy_latency} total")
    print(f"    Overall accuracy : {summary['overall_accuracy']:.4f}")
    print(f"    Pre-drift acc    : {summary['pre_drift_accuracy']:.4f}")
    print(f"    Post-drift acc   : {summary['post_drift_accuracy']:.4f}")
    print(f"    Accuracy drop    : {summary['accuracy_drop']:+.4f}  "
          f"({'degradation' if summary['accuracy_drop'] < 0 else 'improvement'})")
    print(f"    Overall F1       : {summary['overall_f1']:.4f}")
    print(f"    Pre-drift F1     : {summary.get('pre_drift_f1', 0):.4f}")
    print(f"    Post-drift F1    : {summary.get('post_drift_f1', 0):.4f}")
    if summary.get('f1_drop') is not None:
        print(f"    F1 drop          : {summary['f1_drop']:+.4f}  "
              f"({'degradation' if summary['f1_drop'] < 0 else 'improvement'})")
    auc_val = summary.get('overall_auc')
    print(f"    Overall AUC      : {auc_val:.4f}" if auc_val is not None else
          f"    Overall AUC      : N/A")
    pre_auc = summary.get('pre_drift_auc')
    post_auc = summary.get('post_drift_auc')
    print(f"    Pre-drift AUC    : {pre_auc:.4f}" if pre_auc is not None else
          f"    Pre-drift AUC    : N/A")
    print(f"    Post-drift AUC   : {post_auc:.4f}" if post_auc is not None else
          f"    Post-drift AUC   : N/A")
    if summary.get('auc_drop') is not None:
        print(f"    AUC drop         : {summary['auc_drop']:+.4f}  "
              f"({'degradation' if summary['auc_drop'] < 0 else 'improvement'})")
    print(f"    Total retrains   : {summary['total_retrains']}")
    if summary.get('retrains_before_drift') is not None:
        print(f"    Retrains pre/post: "
              f"{summary['retrains_before_drift']}/{summary['retrains_after_drift']}")
    if summary.get('budget_utilization') is not None and budget > 0:
        print(f"    Budget util      : {summary['budget_utilization']:.0%}")
    print(f"    Fraud rate       : {fraud_rate:.4f}")

    return summary, fraud_rate


def run_sanity_check(loader, n_samples):
    """
    Run three sanity-check configurations before the full sweep.

    1. No-retrain baseline   — reveals the raw drift signal floor.
    2. Periodic K=10, low latency  — aggressive adaptation (the original check).
    3. Periodic K=5, high latency  — constrained; should show larger drop if
       drift is real but absorbed by the aggressive policy.

    All three use abrupt drift, offset=40000, same stream.
    """
    drift_point = n_samples // 2

    print(f"\n{'=' * 70}")
    print(f"  SANITY CHECK — Abrupt Drift, Offset=40000, Three Configurations")
    print(f"{'=' * 70}")
    print(f"  Stream length    : {n_samples:,}")
    print(f"  Drift point      : {drift_point:,}")
    print(f"{'=' * 70}")

    # ── Config A: no-retrain baseline ────────────────────────────────────
    sum_nr, fraud_rate = _run_one_sanity(
        loader, n_samples,
        label="A) No-Retrain Baseline (partial_fit only, 0 retrains)",
        policy_type="no_retrain", budget=0,
        retrain_latency=0, deploy_latency=0,
    )

    # ── Config B: periodic K=10, low latency (original sanity) ───────────
    sum_pk10, _ = _run_one_sanity(
        loader, n_samples,
        label="B) Periodic K=10, Low Latency (10+1)",
        policy_type="periodic", budget=10,
        retrain_latency=10, deploy_latency=1,
    )

    # ── Config C: periodic K=5, high latency ─────────────────────────────
    sum_pk5, _ = _run_one_sanity(
        loader, n_samples,
        label="C) Periodic K=5, High Latency (500+20)",
        policy_type="periodic", budget=5,
        retrain_latency=500, deploy_latency=20,
    )

    # ── Comparative interpretation ───────────────────────────────────────
    majority_acc = max(fraud_rate, 1 - fraud_rate)
    drop_nr  = sum_nr['accuracy_drop']
    drop_k10 = sum_pk10['accuracy_drop']
    drop_k5  = sum_pk5['accuracy_drop']
    acc_nr   = sum_nr['overall_accuracy']

    f1_drop_nr  = sum_nr.get('f1_drop', 0) or 0
    f1_drop_k10 = sum_pk10.get('f1_drop', 0) or 0
    f1_drop_k5  = sum_pk5.get('f1_drop', 0) or 0

    auc_drop_nr  = sum_nr.get('auc_drop')
    auc_drop_k10 = sum_pk10.get('auc_drop')
    auc_drop_k5  = sum_pk5.get('auc_drop')

    print(f"\n{'=' * 70}")
    print(f"  COMPARATIVE INTERPRETATION")
    print(f"{'=' * 70}")
    print(f"  Majority-class baseline : {majority_acc:.4f}")
    print(f"  No-retrain accuracy     : {acc_nr:.4f}")
    print(f"  No-retrain F1           : {sum_nr['overall_f1']:.4f}")
    auc_nr = sum_nr.get('overall_auc')
    print(f"  No-retrain AUC          : {auc_nr:.4f}" if auc_nr else
          f"  No-retrain AUC          : N/A")
    print()
    print(f"  Accuracy drops (post-pre):")
    print(f"    A) No-retrain         : {drop_nr:+.4f}")
    print(f"    B) Periodic K=10 low  : {drop_k10:+.4f}")
    print(f"    C) Periodic K=5 high  : {drop_k5:+.4f}")
    print()
    print(f"  F1 drops (post-pre):   << key metric for imbalanced data >>")
    print(f"    A) No-retrain         : {f1_drop_nr:+.4f}")
    print(f"    B) Periodic K=10 low  : {f1_drop_k10:+.4f}")
    print(f"    C) Periodic K=5 high  : {f1_drop_k5:+.4f}")
    print()
    print(f"  AUC drops (post-pre):")
    for label, val in [("A) No-retrain", auc_drop_nr),
                       ("B) Periodic K=10 low", auc_drop_k10),
                       ("C) Periodic K=5 high", auc_drop_k5)]:
        if val is not None:
            print(f"    {label:<22}: {val:+.4f}")
        else:
            print(f"    {label:<22}: N/A")
    print()

    # Check: majority-class prediction?
    if acc_nr >= majority_acc - 0.005:
        print("  X  PROBLEM: No-retrain accuracy is at majority-class level.")
        print("     Balanced sample weights may not be working.\n")
        return

    # ── Primary interpretation: F1 (key metric for imbalanced data) ──────
    # Accuracy is dominated by the majority class on this dataset (97% legit)
    # and cannot distinguish policies. F1 on the fraud class is the signal.
    print("  ── F1-based interpretation (primary for imbalanced data) ──")
    print()

    if f1_drop_nr < -0.001 or f1_drop_k5 < -0.001:
        # At least one config shows F1 degradation post-drift
        if f1_drop_k5 < f1_drop_nr - 0.002:
            print(f"  OK F1 degrades post-drift and constrained policy is worse:")
            print(f"     No-retrain F1 drop    : {f1_drop_nr:+.4f}")
            print(f"     K=10 low-lat F1 drop  : {f1_drop_k10:+.4f}")
            print(f"     K=5 high-lat F1 drop  : {f1_drop_k5:+.4f}")
            print(f"     Constrained policy (K=5) shows {abs(f1_drop_k5)/max(abs(f1_drop_nr),1e-6):.1f}× more")
            print(f"     F1 degradation than baseline — budget/latency effects are visible.")
        else:
            print(f"  ~  F1 degrades post-drift but policy ordering is weak:")
            print(f"     No-retrain F1 drop    : {f1_drop_nr:+.4f}")
            print(f"     K=10 low-lat F1 drop  : {f1_drop_k10:+.4f}")
            print(f"     K=5 high-lat F1 drop  : {f1_drop_k5:+.4f}")

        # Also check AUC gradient
        if auc_drop_nr is not None and auc_drop_k5 is not None:
            if auc_drop_k5 < auc_drop_nr - 0.005:
                print(f"     AUC confirms: constrained policy loses {auc_drop_nr - auc_drop_k5:+.4f}")
                print(f"     more AUC than baseline post-drift.")

        print()
        print("     Accuracy still improves post-drift because it is dominated by the")
        print("     majority class (97% legit). This is expected — F1 and AUC are the")
        print("     informative metrics on imbalanced data.")
        print()
        print("     Pipeline is working correctly. Safe to launch full sweep.")
        print("     Gradual/recurring conditions (pool separation) will amplify the signal.\n")
    elif f1_drop_nr > 0.005:
        print(f"  !  F1 improves post-drift ({f1_drop_nr:+.4f}).")
        print("     Drift at this window makes fraud prediction easier, not harder.")
        print("     Consider trying a different window offset.\n")
    else:
        print(f"  ~  F1 is nearly flat post-drift ({f1_drop_nr:+.4f}).")
        print("     Organic abrupt drift is subtle at this window. Constructed")
        print("     gradual/recurring streams (75k pool separation) will produce")
        print("     stronger signals. Safe to launch full sweep — abrupt will be")
        print("     the weakest condition.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run retraining-policy experiments on CIS Fraud Detection data."
    )
    parser.add_argument(
        "--policy",
        choices=["periodic", "error_threshold", "drift_triggered",
                 "no_retrain", "all"],
        default="all",
        help="Which policy to run (default: all).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=N_SAMPLES_DEFAULT,
        help=(
            f"Number of rows per stream (default: {N_SAMPLES_DEFAULT:,}). "
            "Gradual/recurring use an additional pool of the same size."
        ),
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help=(
            "Run a single sanity-check (periodic, K=10, low latency, "
            "abrupt drift, offset=0) and print detailed metrics. "
            "Use this before launching the full sweep."
        ),
    )
    args = parser.parse_args()

    n_samples = args.n_samples

    # Pre-load the dataset once (cached in the loader)
    loader = FraudDataLoader(DATA_DIR)

    # Validate that the dataset is large enough for the largest window
    max_offset = max(WINDOW_OFFSETS)
    # Gradual/recurring need 2 pools of POOL_SIZE each
    required_rows = max_offset + 2 * POOL_SIZE
    if loader.total_rows < required_rows:
        print(
            f"WARNING: Dataset has {loader.total_rows:,} rows but the "
            f"experiment requires up to {required_rows:,} rows "
            f"(offset={max_offset} + 2×{POOL_SIZE:,}). "
            f"Reduce --n_samples or remove larger window offsets."
        )
        return

    # ── Sanity check mode ────────────────────────────────────────────────
    if args.sanity:
        run_sanity_check(loader, n_samples)
        return

    # ── Full sweep mode ──────────────────────────────────────────────────
    policies = (
        list(POLICY_PARAMS.keys()) if args.policy == "all"
        else [args.policy]
    )

    # Compute total runs
    total_runs = 0
    for p in policies:
        if p == "no_retrain":
            total_runs += len(DRIFT_TYPES) * len(WINDOW_OFFSETS)
        else:
            total_runs += (
                len(DRIFT_TYPES) * len(BUDGETS)
                * len(LATENCY_CONFIGS) * len(WINDOW_OFFSETS)
            )

    total_start = time.time()

    print(f"{'#' * 70}")
    print(f"  FRAUD DETECTION EXPERIMENT — {len(policies)} policy(ies), "
          f"{len(WINDOW_OFFSETS)} window offsets, "
          f"{total_runs} total runs")
    print(f"  Stream length: {n_samples:,} rows")
    print(f"{'#' * 70}")

    for policy_type in policies:
        run_policy_sweep(policy_type, loader, n_samples)

    total_elapsed = time.time() - total_start
    print(f"\n{'#' * 70}")
    print(f"  ALL DONE — {len(policies)} policy(ies) completed "
          f"in {total_elapsed / 60:.1f} minutes")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()

