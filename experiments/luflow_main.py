"""
LUFlow Experiment Runner.

Uses real-world LUFlow intrusion-detection data with three pool-pair
configurations ("seeds") and three drift injection methods (abrupt,
gradual, recurring) to evaluate retraining policies under budget and
latency constraints.

Design (mirrors the synthetic study):
    3 seeds x 3 drift types x 3 budgets x 3 latency levels x 3 policies
    = 243 active runs  +  9 baseline runs  =  252 total

Seeds (pool-pair configurations):
    seed1: Jan-2021 low-mal  ->  Feb-2021 high-mal   (class-balance shift)
    seed2: Jan-2021 high-mal ->  Feb-2021 high-mal   (feature drift, similar balance)
    seed3: Jan-2021 low-mal  ->  Feb-2021 extreme-mal (class-balance shift, extreme attacks)

Drift types (applied to the pre/post pools selected by each seed):
    abrupt:    hard switch at t = 25,000
    gradual:   linear blend over 5,000 steps [25,000 ... 30,000)
    recurring: concept alternates every 5,000 steps after t = 25,000

Locked policy parameters (from calibration on LUFlow abrupt condition):
    Periodic:        interval = 50,000 / K  (K=5->10,000; K=10->5,000; K=20->2,500)
    Error-Threshold: threshold = 0.20, window_size = 200
    Drift-Triggered: delta = 0.005, window_size = 500, min_samples = 100

Stream: 50,000 samples per run, drift point at t = 25,000.

Usage:
    python luflow_main.py                            # all 4 policies (252 runs)
    python luflow_main.py --policy periodic          # periodic only  (81 runs)
    python luflow_main.py --policy error_threshold   # error-threshold only (81 runs)
    python luflow_main.py --policy drift_triggered   # drift-triggered only (81 runs)
    python luflow_main.py --policy no_retrain        # baseline only  (9 runs)
"""

import sys
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -- Ensure project root is on sys.path ----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.base_model import StreamingModel
from src.policies.periodic import PeriodicPolicy
from src.policies.error_threshold_policy import ErrorThresholdPolicy
from src.policies.drift_triggered_policy import DriftTriggeredPolicy
from src.policies.never_retrain_policy import NeverRetrainPolicy
from src.evaluation.metrics import MetricsTracker
from src.runner.experiment_runner import ExperimentRunner
from src.evaluation.results_export import (
    export_to_json, export_to_csv, export_summary_to_csv,
)

warnings.filterwarnings("ignore")

# =====================================================================
#  CONSTANTS
# =====================================================================

DATA_DIR = PROJECT_ROOT / "src" / "data" / "LUFlow_Network_Intrusion" / "datasets"

FEATURE_COLS = [
    "avg_ipt", "bytes_in", "bytes_out", "dest_port",
    "entropy", "num_pkts_out", "num_pkts_in", "proto",
    "src_port", "total_entropy", "duration",
]
POSITIVE_LABEL = "malicious"
NEGATIVE_LABEL = "benign"
MAX_ROWS_PER_DAY = 50_000

# -- Stream parameters ----------------------------------------------------
N_SAMPLES = 50_000
DRIFT_POINT = 25_000
GRADUAL_WINDOW = 5_000       # transition window for gradual drift
RECURRENCE_PERIOD = 5_000    # alternation period for recurring drift

# -- Experiment grid ------------------------------------------------------
DRIFT_TYPES = ["abrupt", "gradual", "recurring"]
BUDGETS = [5, 10, 20]
LATENCY_CONFIGS = [
    (10, 1),      # Low latency   (total = 11)
    (100, 5),     # Medium latency (total = 105)
    (500, 20),    # High latency  (total = 520)
]

# -- Pool-pair configurations ("seeds") -----------------------------------
#   Each entry defines which LUFlow days form the pre-drift and post-drift
#   data pools.  Month prefixes filter by filename; mal_min / mal_max
#   constrain the % malicious of included days.
POOL_CONFIGS = [
    {
        "id": 1,
        "label": "Jan-low -> Feb-high",
        "pre":  {"months": ["2021.01.01", "2021.01.02", "2021.01.03"], "mal_max": 5.0},
        "post": {"months": ["2021.02"], "mal_min": 15.0},
    },
    {
        "id": 2,
        "label": "Jan-high -> Feb-high",
        "pre":  {"months": ["2021.01"], "mal_min": 15.0},
        "post": {"months": ["2021.02"], "mal_min": 15.0},
    },
    {
        "id": 3,
        "label": "Jan-low -> Feb-extreme",
        "pre":  {"months": ["2021.01.06", "2021.01.07"], "mal_max": 5.0},
        "post": {"months": ["2021.02"], "mal_min": 40.0},
    },
]

# -- Locked policy parameters (calibrated on LUFlow abrupt condition) -----
POLICY_PARAMS = {
    "periodic": {},                                          # interval = N_SAMPLES / K
    "error_threshold": {"error_threshold": 0.20, "window_size": 200},
    "drift_triggered": {"delta": 0.005, "window_size": 500, "min_samples": 100},
    "no_retrain": {},
}

POLICY_DISPLAY = {
    "periodic":         "Periodic",
    "error_threshold":  "Error-Threshold",
    "drift_triggered":  "Drift-Triggered (ADWIN)",
    "no_retrain":       "No-Retrain (Baseline)",
}


# =====================================================================
#  DATA LOADING
# =====================================================================

def _scan_days():
    """Scan LUFlow day-CSVs and return per-day metadata.

    Returns:
        list of (stem, total_rows, binary_rows, malicious_rows, mal_pct)
    """
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"  ERROR: No CSV files found in {DATA_DIR}")
        sys.exit(1)

    per_day = []
    for f in csv_files:
        df = pd.read_csv(f, usecols=["label"])
        counts = df["label"].value_counts()
        n_bin = counts.get(POSITIVE_LABEL, 0) + counts.get(NEGATIVE_LABEL, 0)
        n_mal = counts.get(POSITIVE_LABEL, 0)
        pct = 100 * n_mal / n_bin if n_bin > 0 else 0
        per_day.append((f.stem, len(df), n_bin, n_mal, pct))
    return per_day


def _load_pool(per_day, months, mal_min=None, mal_max=None):
    """Load and concatenate binary-labelled rows from matching day files.

    Args:
        per_day: Output of _scan_days().
        months:  List of filename prefixes (e.g. ["2021.01"]).
        mal_min: Minimum malicious % to include a day (None = no lower bound).
        mal_max: Maximum malicious % to include a day (None = no upper bound).

    Returns:
        X (ndarray): Feature matrix  (N, 11).
        y (ndarray): Binary labels   (N,).
    """
    stems = []
    for stem, _, n_bin, _, pct in per_day:
        if not any(stem.startswith(m) for m in months):
            continue
        if n_bin < 100:
            continue
        if mal_min is not None and pct < mal_min:
            continue
        if mal_max is not None and pct > mal_max:
            continue
        stems.append(stem)

    if not stems:
        raise ValueError(
            f"No days match filter: months={months}, "
            f"mal_min={mal_min}, mal_max={mal_max}"
        )

    frames = []
    for stem in stems:
        fpath = DATA_DIR / (stem + ".csv")
        df = pd.read_csv(fpath, nrows=MAX_ROWS_PER_DAY)
        if "time_start" in df.columns:
            df = df.sort_values("time_start").reset_index(drop=True)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[
        combined["label"].isin([POSITIVE_LABEL, NEGATIVE_LABEL])
    ].copy()
    combined["target"] = (combined["label"] == POSITIVE_LABEL).astype(int)

    X = combined[FEATURE_COLS].fillna(0).values.astype(np.float64)
    y = combined["target"].values
    return X, y


def _load_pools_for_config(per_day, cfg):
    """Load pre- and post-drift pools for a single POOL_CONFIG entry."""
    pre_kw = {k: v for k, v in cfg["pre"].items() if k != "months"}
    post_kw = {k: v for k, v in cfg["post"].items() if k != "months"}
    pre_X, pre_y = _load_pool(per_day, cfg["pre"]["months"], **pre_kw)
    post_X, post_y = _load_pool(per_day, cfg["post"]["months"], **post_kw)
    return pre_X, pre_y, post_X, post_y


# =====================================================================
#  STREAM CONSTRUCTION
# =====================================================================

def build_stream(pre_X, pre_y, post_X, post_y, drift_type):
    """
    Construct a 50,000-sample stream with the requested drift injection.

    Indexing into each pool wraps with modular arithmetic so the stream
    is always exactly N_SAMPLES long, even if a pool is smaller.

    Drift injection methods (mirroring the synthetic DriftGenerator):

      abrupt:    [0, dp)  = pre-pool
                 [dp, N)  = post-pool

      gradual:   [0, dp)           = pre-pool
                 [dp, dp+GW)       = blend (probability of post increases
                                     linearly from 0 -> 1 over GW steps)
                 [dp+GW, N)        = post-pool

      recurring: [0, dp)           = pre-pool
                 after dp, alternate post-pool / pre-pool every
                 RECURRENCE_PERIOD steps (even periods = drifted concept,
                 odd periods = original concept)

    Returns:
        X  (ndarray): StandardScaler-transformed features (N_SAMPLES, 11).
        y  (ndarray): Binary labels (N_SAMPLES,).
        dp (int):     DRIFT_POINT (25,000).
    """
    n = N_SAMPLES
    dp = DRIFT_POINT
    pre_n, post_n = len(pre_X), len(post_X)

    if drift_type == "abrupt":
        # -- Hard switch at dp ----------------------------------------
        pre_idx = np.arange(dp) % pre_n
        post_idx = np.arange(n - dp) % post_n
        X = np.vstack([pre_X[pre_idx], post_X[post_idx]])
        y = np.concatenate([pre_y[pre_idx], post_y[post_idx]])

    elif drift_type == "gradual":
        # -- Linear blend over GRADUAL_WINDOW steps -------------------
        rng = np.random.default_rng(42)
        gw = GRADUAL_WINDOW
        post_only = n - dp - gw

        # Pre-drift
        pre_idx = np.arange(dp) % pre_n
        pre_cursor = dp          # next unused position in pre-pool

        # Transition window
        trans_X, trans_y = [], []
        post_cursor = 0
        for t in range(gw):
            alpha = t / gw       # 0 -> 1 linearly
            if rng.random() < alpha:
                idx = post_cursor % post_n
                trans_X.append(post_X[idx])
                trans_y.append(post_y[idx])
                post_cursor += 1
            else:
                idx = pre_cursor % pre_n
                trans_X.append(pre_X[idx])
                trans_y.append(pre_y[idx])
                pre_cursor += 1

        # Post-transition
        post_rem_idx = np.arange(post_cursor, post_cursor + post_only) % post_n

        X = np.vstack([pre_X[pre_idx], np.array(trans_X), post_X[post_rem_idx]])
        y = np.concatenate([pre_y[pre_idx], np.array(trans_y), post_y[post_rem_idx]])

    elif drift_type == "recurring":
        # -- Alternating chunks after dp ------------------------------
        pre_idx = np.arange(dp) % pre_n
        pre_cursor = dp
        post_cursor = 0

        chunks_X = [pre_X[pre_idx]]
        chunks_y = [pre_y[pre_idx]]

        remaining = n - dp
        period = 0
        while remaining > 0:
            chunk = min(RECURRENCE_PERIOD, remaining)
            if period % 2 == 0:
                # Drifted concept (post-pool)
                idx = np.arange(post_cursor, post_cursor + chunk) % post_n
                chunks_X.append(post_X[idx])
                chunks_y.append(post_y[idx])
                post_cursor += chunk
            else:
                # Original concept (pre-pool)
                idx = np.arange(pre_cursor, pre_cursor + chunk) % pre_n
                chunks_X.append(pre_X[idx])
                chunks_y.append(pre_y[idx])
                pre_cursor += chunk
            remaining -= chunk
            period += 1

        X = np.vstack(chunks_X)
        y = np.concatenate(chunks_y)

    else:
        raise ValueError(f"Unknown drift_type: {drift_type!r}")

    # -- Standardise features across the whole stream -----------------
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, dp


# =====================================================================
#  POLICY & CONFIG BUILDERS
# =====================================================================

def _build_policy(policy_type, budget, retrain_latency, deploy_latency):
    """Instantiate the requested policy with locked parameters."""
    if policy_type == "periodic":
        interval = N_SAMPLES // budget       # K=5->10000, K=10->5000, K=20->2500
        return PeriodicPolicy(
            interval=interval, budget=budget,
            retrain_latency=retrain_latency, deploy_latency=deploy_latency,
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


def _build_config(policy_type, drift_type, budget, seed_id, pool_label):
    """Build the config dict consumed by export helpers."""
    config = {
        "drift_type": drift_type,
        "drift_point": DRIFT_POINT,
        "recurrence_period": RECURRENCE_PERIOD,
        "policy_type": policy_type,
        "budget": budget,
        "random_seed": seed_id,
        "dataset": "LUFlow",
        "pool_config": pool_label,
        "n_samples": N_SAMPLES,
    }
    if policy_type == "periodic":
        config["policy_interval"] = N_SAMPLES // budget if budget > 0 else 0
    elif policy_type == "error_threshold":
        config.update(POLICY_PARAMS["error_threshold"])
    elif policy_type == "drift_triggered":
        config.update(POLICY_PARAMS["drift_triggered"])
    return config


# =====================================================================
#  SWEEP RUNNERS
# =====================================================================

def run_policy_sweep(policy_type, per_day, pool_cache):
    """Execute the full-factorial sweep for one policy on LUFlow data."""

    if policy_type == "no_retrain":
        return _run_no_retrain_sweep(per_day, pool_cache)

    seed_label = "3seed"
    n_pools = len(POOL_CONFIGS)
    n_drifts = len(DRIFT_TYPES)
    n_budgets = len(BUDGETS)
    n_latencies = len(LATENCY_CONFIGS)
    total_runs = n_pools * n_drifts * n_budgets * n_latencies

    # Output paths
    results_dir = PROJECT_ROOT / f"results_with_retrain/luflow/per_run/luflow_{policy_type}_{seed_label}"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = str(PROJECT_ROOT / f"results_with_retrain/luflow/csv/luflow_summary_{policy_type}_retrain_{seed_label}.csv")
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_csv).unlink(missing_ok=True)

    display = POLICY_DISPLAY[policy_type]
    run_count = 0
    start_time = time.time()

    print(f"\n{'=' * 72}")
    print(f"LUFlow -- {display} POLICY  ({total_runs} runs)")
    print(f"{'=' * 72}")
    print(f"  Dataset       : LUFlow  ({n_pools} pool configs x {n_drifts} drift types)")
    print(f"  Stream        : {N_SAMPLES:,} samples, drift at t = {DRIFT_POINT:,}")
    print(f"  Budgets       : {BUDGETS}")
    print(f"  Latency levels: {LATENCY_CONFIGS}")
    if policy_type == "periodic":
        print(f"  Intervals     : {[N_SAMPLES // b for b in BUDGETS]}")
    else:
        for k, v in POLICY_PARAMS[policy_type].items():
            print(f"  {k:<16}: {v}")
    print(f"{'=' * 72}\n")

    for cfg in POOL_CONFIGS:
        sid = cfg["id"]
        slabel = cfg["label"]
        pre_X, pre_y, post_X, post_y = pool_cache[sid]

        for drift_type in DRIFT_TYPES:
            # Build stream once per (seed, drift_type) -- reused across
            # the 9 budget x latency combos that share this stream.
            X, y, dp = build_stream(pre_X, pre_y, post_X, post_y, drift_type)

            for budget in BUDGETS:
                for retrain_latency, deploy_latency in LATENCY_CONFIGS:
                    run_count += 1

                    # Build components
                    model = StreamingModel()
                    policy = _build_policy(
                        policy_type, budget, retrain_latency, deploy_latency,
                    )
                    metrics = MetricsTracker()
                    metrics.set_drift_point(dp)
                    metrics.set_budget(budget)

                    # Run
                    runner = ExperimentRunner(model, policy, metrics)
                    runner.run(X, y)

                    # Progress
                    summary = metrics.get_summary()
                    elapsed = time.time() - start_time
                    eta = (elapsed / run_count) * (total_runs - run_count)

                    print(
                        f"[{run_count:>3}/{total_runs}] "
                        f"pool={sid} drift={drift_type:<10} "
                        f"budget={budget:<3} "
                        f"latency=({retrain_latency}+{deploy_latency}) | "
                        f"acc={summary['overall_accuracy']:.4f}  "
                        f"retrains={summary['total_retrains']:>2} | "
                        f"ETA {eta / 60:.1f} min"
                    )

                    # Export per-run results
                    run_tag = (
                        f"{drift_type}_s{sid}"
                        f"_b{budget}"
                        f"_l{retrain_latency}+{deploy_latency}"
                    )
                    config = _build_config(
                        policy_type, drift_type, budget, sid, slabel,
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
    print(f"\nLUFlow {display}: {total_runs} runs completed in "
          f"{elapsed / 60:.1f} minutes")
    print(f"Summary CSV -> {summary_csv}")

    # -- Generate summary dashboard -----------------------------------
    print(f"Generating LUFlow {display} summary plot ...")
    from plot_summary import plot_summary_for_policy
    plot_output = str(PROJECT_ROOT / f"results_with_retrain/luflow/plots/luflow_summary_plot_{policy_type}_retrain_{seed_label}.png")
    Path(plot_output).parent.mkdir(parents=True, exist_ok=True)
    plot_summary_for_policy(
        csv_path=summary_csv,
        output_path=plot_output,
        policy_name=f"LUFlow {display}",
    )

    return summary_csv


def _run_no_retrain_sweep(per_day, pool_cache):
    """No-retrain baseline: 3 seeds x 3 drift types = 9 runs."""

    policy_type = "no_retrain"
    seed_label = "3seed"
    total_runs = len(POOL_CONFIGS) * len(DRIFT_TYPES)

    results_dir = PROJECT_ROOT / f"results_with_retrain/luflow/per_run/luflow_{policy_type}_{seed_label}"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = str(PROJECT_ROOT / f"results_with_retrain/luflow/csv/luflow_summary_{policy_type}_{seed_label}.csv")
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_csv).unlink(missing_ok=True)

    display = POLICY_DISPLAY[policy_type]
    run_count = 0
    start_time = time.time()

    print(f"\n{'=' * 72}")
    print(f"LUFlow -- {display}  Baseline Sweep ({total_runs} runs)")
    print(f"{'=' * 72}")
    print(f"  Stream   : {N_SAMPLES:,} samples, drift at t = {DRIFT_POINT:,}")
    print(f"  Budget   : N/A (always 0)")
    print(f"  Latency  : N/A (always 0)")
    print(f"{'=' * 72}\n")

    for cfg in POOL_CONFIGS:
        sid = cfg["id"]
        slabel = cfg["label"]
        pre_X, pre_y, post_X, post_y = pool_cache[sid]

        for drift_type in DRIFT_TYPES:
            run_count += 1

            X, y, dp = build_stream(pre_X, pre_y, post_X, post_y, drift_type)

            model = StreamingModel()
            policy = NeverRetrainPolicy()
            metrics = MetricsTracker()
            metrics.set_drift_point(dp)
            metrics.set_budget(0)

            ExperimentRunner(model, policy, metrics).run(X, y)

            summary = metrics.get_summary()
            elapsed = time.time() - start_time
            eta = (elapsed / run_count) * (total_runs - run_count)

            print(
                f"[{run_count:>3}/{total_runs}] "
                f"pool={sid} drift={drift_type:<10} | "
                f"acc={summary['overall_accuracy']:.4f}  "
                f"retrains={summary['total_retrains']:>2} | "
                f"ETA {eta / 60:.1f} min"
            )

            run_tag = f"{drift_type}_s{sid}"
            config = _build_config(
                policy_type, drift_type, budget=0,
                seed_id=sid, pool_label=slabel,
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
    print(f"\nLUFlow {display}: {total_runs} runs completed in "
          f"{elapsed / 60:.1f} minutes")
    print(f"Summary CSV -> {summary_csv}")

    # -- Generate baseline summary plot --------------------------------
    print(f"Generating LUFlow {display} summary plot ...")
    from plot_summary import plot_summary_for_no_retrain
    plot_output = str(PROJECT_ROOT / f"results_with_retrain/luflow/plots/luflow_summary_plot_{policy_type}_{seed_label}.png")
    Path(plot_output).parent.mkdir(parents=True, exist_ok=True)
    plot_summary_for_no_retrain(
        csv_path=summary_csv,
        output_path=plot_output,
        policy_name=f"LUFlow {display}",
    )

    return summary_csv


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run LUFlow retraining-policy experiments "
                    "(full-factorial sweep, 252 runs).",
    )
    parser.add_argument(
        "--policy",
        choices=["periodic", "error_threshold", "drift_triggered",
                 "no_retrain", "all"],
        default="all",
        help="Which policy to run (default: all).",
    )
    args = parser.parse_args()

    policies = (
        list(POLICY_PARAMS.keys()) if args.policy == "all"
        else [args.policy]
    )

    # -- Banner --------------------------------------------------------
    print(f"{'#' * 72}")
    print(f"  LUFlow EXPERIMENT RUNNER")
    print(f"  Data directory : {DATA_DIR}")
    print(f"  Stream         : {N_SAMPLES:,} samples, drift at t = {DRIFT_POINT:,}")
    print(f"  Drift types    : {DRIFT_TYPES}")
    print(f"  Budgets        : {BUDGETS}")
    print(f"  Latencies      : {[rl + dl for rl, dl in LATENCY_CONFIGS]}")
    print(f"  Pool configs   : {len(POOL_CONFIGS)} seeds")
    print(f"{'#' * 72}")

    # -- Scan data -----------------------------------------------------
    per_day = _scan_days()
    print(f"\n  Scanned {len(per_day)} day-CSVs:")
    for stem, n, nb, nm, pct in per_day:
        tag = "LOW " if pct <= 5 else ("HIGH" if pct >= 15 else "MID ")
        print(f"    {stem}  binary={nb:>7,}  mal={nm:>6,} ({pct:5.1f}%)  [{tag}]")

    # -- Pre-load all pools --------------------------------------------
    print(f"\n  Loading pool data ...")
    pool_cache = {}
    for cfg in POOL_CONFIGS:
        sid = cfg["id"]
        pre_X, pre_y, post_X, post_y = _load_pools_for_config(per_day, cfg)
        pool_cache[sid] = (pre_X, pre_y, post_X, post_y)
        print(f"    Seed {sid} ({cfg['label']}):  "
              f"pre = {len(pre_X):>6,} samples ({100*pre_y.mean():.1f}% mal)   "
              f"post = {len(post_X):>6,} samples ({100*post_y.mean():.1f}% mal)")

    # -- Locked parameters ---------------------------------------------
    print(f"\n  Locked policy parameters (from calibration):")
    print(f"    Periodic:        interval = {N_SAMPLES} / K")
    print(f"    Error-Threshold: {POLICY_PARAMS['error_threshold']}")
    print(f"    Drift-Triggered: {POLICY_PARAMS['drift_triggered']}")

    # -- Compute total runs --------------------------------------------
    total_runs = 0
    for p in policies:
        if p == "no_retrain":
            total_runs += len(POOL_CONFIGS) * len(DRIFT_TYPES)
        else:
            total_runs += (len(POOL_CONFIGS) * len(DRIFT_TYPES)
                           * len(BUDGETS) * len(LATENCY_CONFIGS))

    total_start = time.time()

    print(f"\n{'#' * 72}")
    print(f"  LUFlow FULL SWEEP -- {len(policies)} policy(ies), "
          f"{total_runs} total runs")
    print(f"{'#' * 72}")

    # -- Run sweeps ----------------------------------------------------
    for policy_type in policies:
        run_policy_sweep(policy_type, per_day, pool_cache)

    total_elapsed = time.time() - total_start
    print(f"\n{'#' * 72}")
    print(f"  ALL DONE -- {len(policies)} policy(ies) completed in "
          f"{total_elapsed / 60:.1f} minutes")
    print(f"{'#' * 72}")


if __name__ == "__main__":
    main()

