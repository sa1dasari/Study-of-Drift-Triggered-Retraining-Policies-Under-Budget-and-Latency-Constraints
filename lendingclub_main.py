"""
LendingClub Experiment Runner.

Uses real-world LendingClub loan data with three year-pair configurations
("seeds") and three drift injection methods (abrupt, gradual, recurring)
to evaluate retraining policies under budget and latency constraints.

The drift here is *feature-space drift*, not label-frequency drift.
Between 2012 and 2016 LendingClub changed underwriting policy: average
FICO dropped from 716 to 703, average interest rates rose from 10.8% to
13%, and the correlation between FICO and loan grade fell from 80% to 35%.
A model trained on one cohort sees a genuinely different joint distribution
when tested on a later cohort.

Design (mirrors the synthetic and LUFlow studies):
    3 seeds x 3 drift types x 3 budgets x 3 latency levels x 3 policies
    = 243 active runs  +  9 baseline runs  =  252 total

Seeds (year-pair configurations):
    Seed 1: PRE = 2013 -> POST = 2016  (3-year gap, maximum policy shift)
    Seed 2: PRE = 2014 -> POST = 2016  (2-year gap, moderate drift)
    Seed 3: PRE = 2013 -> POST = 2015  (2-year gap, different cohort pair)

Drift types (applied to the pre/post pools selected by each seed):
    abrupt:    hard switch at t = 25,000
    gradual:   linear blend over 5,000 steps [25,000 ... 30,000)
    recurring: concept alternates every 5,000 steps after t = 25,000

Locked policy parameters (to be calibrated on LendingClub abrupt):
    Periodic:        interval = 50,000 / K  (K=5->10,000; K=10->5,000; K=20->2,500)
    Error-Threshold: threshold = 0.20, window_size = 200
    Drift-Triggered: delta = 0.005, window_size = 500, min_samples = 100

Stream: 50,000 samples per run, drift point at t = 25,000.

**Sampling**: Each seed uses ``rng.choice`` (via ``get_year_cohort``'s
``random_state`` parameter) so that subsampled pools are *shuffled*,
not sequential.  Each seed gets a distinct random_state derived from
the seed id, guaranteeing different 25 K subsets from the same year.

Usage:
    python lendingclub_main.py                              # all 4 policies (252 runs)
    python lendingclub_main.py --policy periodic            # periodic only  (81 runs)
    python lendingclub_main.py --policy error_threshold     # error-threshold only
    python lendingclub_main.py --policy drift_triggered     # drift-triggered only
    python lendingclub_main.py --policy no_retrain          # baseline only  (9 runs)
"""

import sys
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

# -- Ensure project root is on sys.path ----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.LendingClub_Loan_Data.lendingclub_loader import (
    load_lendingclub, get_year_cohort,
)
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
#   Each entry maps to a pre-year / post-year pair.  ``random_state`` varies
#   per seed so that ``get_year_cohort`` draws a *different* shuffled subset
#   from the same year when two seeds share a year (e.g. seeds 1 & 3 both
#   use pre_year=2013, but get non-overlapping 25 K samples).
POOL_CONFIGS = [
    {
        "id": 1,
        "label": "2013->2016 (3-yr gap, max policy shift)",
        "pre_year": 2013, "post_year": 2016,
        "random_state": 100,
    },
    {
        "id": 2,
        "label": "2014->2016 (2-yr gap, moderate drift)",
        "pre_year": 2014, "post_year": 2016,
        "random_state": 200,
    },
    {
        "id": 3,
        "label": "2013->2015 (2-yr gap, different cohort)",
        "pre_year": 2013, "post_year": 2015,
        "random_state": 300,
    },
]

# -- Locked policy parameters ---------------------------------------------
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

# -- Max samples per pool half (25 K pre + 25 K post) --------------------
POOL_HALF = N_SAMPLES // 2   # 25,000


# =====================================================================
#  DATA LOADING  (shuffled via rng.choice)
# =====================================================================

def _load_pools_for_config(df, cfg):
    """Load pre- and post-drift pools with seed-specific shuffled sampling.

    Uses ``get_year_cohort(..., random_state=...)`` which internally calls
    ``rng.choice`` so each seed draws a *different* random subset.
    """
    rs = cfg["random_state"]
    pre_X, pre_y = get_year_cohort(
        df, cfg["pre_year"], max_samples=POOL_HALF, random_state=rs,
    )
    post_X, post_y = get_year_cohort(
        df, cfg["post_year"], max_samples=POOL_HALF, random_state=rs + 1,
    )
    return pre_X, pre_y, post_X, post_y


# =====================================================================
#  STREAM CONSTRUCTION
# =====================================================================

def build_stream(pre_X, pre_y, post_X, post_y, drift_type):
    """
    Construct a 50,000-sample stream with the requested drift injection.

    Pools are already shuffled (via rng.choice in the loader), so modular
    indexing here merely wraps if a pool is shorter than the half-size.

    Drift injection methods (mirroring the synthetic DriftGenerator):

      abrupt:    [0, dp)  = pre-pool
                 [dp, N)  = post-pool

      gradual:   [0, dp)           = pre-pool
                 [dp, dp+GW)       = blend (probability of post increases
                                     linearly from 0 -> 1 over GW steps)
                 [dp+GW, N)        = post-pool

      recurring: [0, dp)           = pre-pool
                 after dp, alternate post-pool / pre-pool every
                 RECURRENCE_PERIOD steps

    Returns:
        X  (ndarray): StandardScaler-transformed features (N_SAMPLES, D).
        y  (ndarray): Binary labels (N_SAMPLES,).
        dp (int):     DRIFT_POINT (25,000).
    """
    n = N_SAMPLES
    dp = DRIFT_POINT
    pre_n, post_n = len(pre_X), len(post_X)

    if drift_type == "abrupt":
        pre_idx  = np.arange(dp) % pre_n
        post_idx = np.arange(n - dp) % post_n
        X = np.vstack([pre_X[pre_idx], post_X[post_idx]])
        y = np.concatenate([pre_y[pre_idx], post_y[post_idx]])

    elif drift_type == "gradual":
        rng = np.random.default_rng(42)
        gw = GRADUAL_WINDOW
        post_only = n - dp - gw

        pre_idx = np.arange(dp) % pre_n
        pre_cursor = dp

        trans_X, trans_y = [], []
        post_cursor = 0
        for t in range(gw):
            alpha = t / gw
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

        post_rem_idx = np.arange(post_cursor, post_cursor + post_only) % post_n
        X = np.vstack([pre_X[pre_idx], np.array(trans_X), post_X[post_rem_idx]])
        y = np.concatenate([pre_y[pre_idx], np.array(trans_y), post_y[post_rem_idx]])

    elif drift_type == "recurring":
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
                idx = np.arange(post_cursor, post_cursor + chunk) % post_n
                chunks_X.append(post_X[idx])
                chunks_y.append(post_y[idx])
                post_cursor += chunk
            else:
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

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, dp


# =====================================================================
#  POLICY & CONFIG BUILDERS
# =====================================================================

def _build_policy(policy_type, budget, retrain_latency, deploy_latency):
    """Instantiate the requested policy with locked parameters."""
    if policy_type == "periodic":
        interval = N_SAMPLES // budget
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
        "dataset": "LendingClub",
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

def run_policy_sweep(policy_type, df, pool_cache):
    """Execute the full-factorial sweep for one policy on LendingClub data."""

    if policy_type == "no_retrain":
        return _run_no_retrain_sweep(df, pool_cache)

    seed_label = "3seed"
    n_pools    = len(POOL_CONFIGS)
    n_drifts   = len(DRIFT_TYPES)
    n_budgets  = len(BUDGETS)
    n_latencies = len(LATENCY_CONFIGS)
    total_runs = n_pools * n_drifts * n_budgets * n_latencies

    results_dir = Path(f"results/lendingclub/per_run/lendingclub_{policy_type}_{seed_label}")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = f"results/lendingclub/csv/lendingclub_summary_{policy_type}_retrain_{seed_label}.csv"
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_csv).unlink(missing_ok=True)

    display = POLICY_DISPLAY[policy_type]
    run_count = 0
    start_time = time.time()

    print(f"\n{'=' * 72}")
    print(f"LendingClub -- {display} POLICY  ({total_runs} runs)")
    print(f"{'=' * 72}")
    print(f"  Dataset       : LendingClub  ({n_pools} pool configs x {n_drifts} drift types)")
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
        sid    = cfg["id"]
        slabel = cfg["label"]
        pre_X, pre_y, post_X, post_y = pool_cache[sid]

        for drift_type in DRIFT_TYPES:
            X, y, dp = build_stream(pre_X, pre_y, post_X, post_y, drift_type)

            for budget in BUDGETS:
                for retrain_latency, deploy_latency in LATENCY_CONFIGS:
                    run_count += 1

                    model   = StreamingModel()
                    policy  = _build_policy(policy_type, budget, retrain_latency, deploy_latency)
                    metrics = MetricsTracker()
                    metrics.set_drift_point(dp)
                    metrics.set_budget(budget)

                    ExperimentRunner(model, policy, metrics).run(X, y)

                    summary = metrics.get_summary()
                    elapsed = time.time() - start_time
                    eta = (elapsed / run_count) * (total_runs - run_count)

                    print(
                        f"[{run_count:>3}/{total_runs}] "
                        f"seed={sid} drift={drift_type:<10} "
                        f"budget={budget:<3} "
                        f"latency=({retrain_latency}+{deploy_latency}) | "
                        f"acc={summary['overall_accuracy']:.4f}  "
                        f"retrains={summary['total_retrains']:>2} | "
                        f"ETA {eta / 60:.1f} min"
                    )

                    run_tag = (
                        f"{drift_type}_s{sid}"
                        f"_b{budget}"
                        f"_l{retrain_latency}+{deploy_latency}"
                    )
                    config = _build_config(policy_type, drift_type, budget, sid, slabel)

                    export_to_json(metrics, policy, config,
                                   str(results_dir / f"run_{run_tag}.json"))
                    export_to_csv(metrics, policy, config,
                                  str(results_dir / f"per_sample_{run_tag}.csv"))
                    export_summary_to_csv(metrics, policy, config, summary_csv)

    elapsed = time.time() - start_time
    print(f"\nLendingClub {display}: {total_runs} runs completed in "
          f"{elapsed / 60:.1f} minutes")
    print(f"Summary CSV -> {summary_csv}")

    print(f"Generating LendingClub {display} summary plot ...")
    from plot_summary import plot_summary_for_policy
    plot_summary_for_policy(
        csv_path=summary_csv,
        output_path=(f"results/lendingclub/plots/lendingclub_summary_plot_"
                     f"{policy_type}_retrain_{seed_label}.png"),
        policy_name=f"LendingClub {display}",
    )

    return summary_csv


def _run_no_retrain_sweep(df, pool_cache):
    """No-retrain baseline: 3 seeds x 3 drift types = 9 runs."""

    policy_type = "no_retrain"
    seed_label  = "3seed"
    total_runs  = len(POOL_CONFIGS) * len(DRIFT_TYPES)

    results_dir = Path(f"results/lendingclub/per_run/lendingclub_{policy_type}_{seed_label}")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = f"results/lendingclub/csv/lendingclub_summary_{policy_type}_{seed_label}.csv"
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_csv).unlink(missing_ok=True)

    display = POLICY_DISPLAY[policy_type]
    run_count  = 0
    start_time = time.time()

    print(f"\n{'=' * 72}")
    print(f"LendingClub -- {display}  Baseline Sweep ({total_runs} runs)")
    print(f"{'=' * 72}")
    print(f"  Stream   : {N_SAMPLES:,} samples, drift at t = {DRIFT_POINT:,}")
    print(f"  Budget   : N/A (always 0)")
    print(f"  Latency  : N/A (always 0)")
    print(f"{'=' * 72}\n")

    for cfg in POOL_CONFIGS:
        sid    = cfg["id"]
        slabel = cfg["label"]
        pre_X, pre_y, post_X, post_y = pool_cache[sid]

        for drift_type in DRIFT_TYPES:
            run_count += 1

            X, y, dp = build_stream(pre_X, pre_y, post_X, post_y, drift_type)

            model   = StreamingModel()
            policy  = NeverRetrainPolicy()
            metrics = MetricsTracker()
            metrics.set_drift_point(dp)
            metrics.set_budget(0)

            ExperimentRunner(model, policy, metrics).run(X, y)

            summary = metrics.get_summary()
            elapsed = time.time() - start_time
            eta = (elapsed / run_count) * (total_runs - run_count)

            print(
                f"[{run_count:>3}/{total_runs}] "
                f"seed={sid} drift={drift_type:<10} | "
                f"acc={summary['overall_accuracy']:.4f}  "
                f"retrains={summary['total_retrains']:>2} | "
                f"ETA {eta / 60:.1f} min"
            )

            run_tag = f"{drift_type}_s{sid}"
            config  = _build_config(policy_type, drift_type, budget=0,
                                    seed_id=sid, pool_label=slabel)

            export_to_json(metrics, policy, config,
                           str(results_dir / f"run_{run_tag}.json"))
            export_to_csv(metrics, policy, config,
                          str(results_dir / f"per_sample_{run_tag}.csv"))
            export_summary_to_csv(metrics, policy, config, summary_csv)

    elapsed = time.time() - start_time
    print(f"\nLendingClub {display}: {total_runs} runs completed in "
          f"{elapsed / 60:.1f} minutes")
    print(f"Summary CSV -> {summary_csv}")

    print(f"Generating LendingClub {display} summary plot ...")
    from plot_summary import plot_summary_for_no_retrain
    plot_summary_for_no_retrain(
        csv_path=summary_csv,
        output_path=f"results/lendingclub/plots/lendingclub_summary_plot_{policy_type}_{seed_label}.png",
        policy_name=f"LendingClub {display}",
    )

    return summary_csv


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run LendingClub retraining-policy experiments "
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
    print(f"  LendingClub EXPERIMENT RUNNER")
    print(f"  Stream         : {N_SAMPLES:,} samples, drift at t = {DRIFT_POINT:,}")
    print(f"  Drift types    : {DRIFT_TYPES}")
    print(f"  Budgets        : {BUDGETS}")
    print(f"  Latencies      : {[rl + dl for rl, dl in LATENCY_CONFIGS]}")
    print(f"  Pool configs   : {len(POOL_CONFIGS)} seeds")
    print(f"  Sampling       : rng.choice (shuffled, per-seed random_state)")
    print(f"{'#' * 72}")

    # -- Load data ------------------------------------------------------
    df = load_lendingclub()

    # -- Pre-load all pools (shuffled) ---------------------------------
    print(f"\n  Loading pool data (shuffled via rng.choice) ...")
    pool_cache = {}
    for cfg in POOL_CONFIGS:
        sid = cfg["id"]
        pre_X, pre_y, post_X, post_y = _load_pools_for_config(df, cfg)
        pool_cache[sid] = (pre_X, pre_y, post_X, post_y)
        print(f"    Seed {sid} ({cfg['label']}):  "
              f"pre = {len(pre_X):>6,} samples ({100*pre_y.mean():.1f}% def)   "
              f"post = {len(post_X):>6,} samples ({100*post_y.mean():.1f}% def)")

    # -- Locked parameters ---------------------------------------------
    print(f"\n  Locked policy parameters:")
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
    print(f"  LendingClub FULL SWEEP -- {len(policies)} policy(ies), "
          f"{total_runs} total runs")
    print(f"{'#' * 72}")

    # -- Run sweeps ----------------------------------------------------
    for policy_type in policies:
        run_policy_sweep(policy_type, df, pool_cache)

    total_elapsed = time.time() - total_start
    print(f"\n{'#' * 72}")
    print(f"  ALL DONE -- {len(policies)} policy(ies) completed in "
          f"{total_elapsed / 60:.1f} minutes")
    print(f"{'#' * 72}")


if __name__ == "__main__":
    main()


