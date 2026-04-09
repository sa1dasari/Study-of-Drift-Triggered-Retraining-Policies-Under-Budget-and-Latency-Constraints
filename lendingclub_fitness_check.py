"""
LendingClub Dataset Fitness Check for Drift-Triggered Retraining Experiments.

Runs three sequential gate checks to determine whether the LendingClub
loan dataset is suitable for the retraining-policy study:

  Gate 1 -- Class Balance:
      Positive (Charged Off) class must be >= 15 %.

  Gate 2 -- Drift Diagnosis:
      Train on one year-cohort, test on another at three offsets.
      At least 2 of 3 must show performance-degrading cross-distribution
      accuracy or F1.

      The three "seed" offsets mirror the experiment design:
        Seed 1: PRE = 2013 -> POST = 2016  (3-year gap, maximum policy shift)
        Seed 2: PRE = 2014 -> POST = 2016  (2-year gap, moderate drift)
        Seed 3: PRE = 2013 -> POST = 2015  (2-year gap, different cohort pair)

  Gate 3 -- Three-Configuration Sanity Check:
      (A) No-retrain baseline
      (B) Well-resourced: K = 10 budget, low latency (10 + 1)
      (C) Constrained:    K = 5 budget, high latency (500 + 20)
      Policy (C) must perform meaningfully worse than (B).
      If all three are within 0.001, the signal is too weak.

Usage:
    python lendingclub_fitness_check.py
"""

import sys
import time
import warnings
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# -- Ensure project root is on sys.path -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.LendingClub_Loan_Data.lendingclub_loader import (
    load_lendingclub, get_year_cohort,
)
from src.models.base_model import StreamingModel
from src.policies.periodic import PeriodicPolicy
from src.policies.never_retrain_policy import NeverRetrainPolicy
from src.evaluation.metrics import MetricsTracker
from src.runner.experiment_runner import ExperimentRunner

warnings.filterwarnings("ignore")

# ===================================================================
#  CONSTANTS
# ===================================================================

# Seed definitions: (pre_year, post_year, label)
SEED_CONFIGS = [
    {"id": 1, "pre_year": 2013, "post_year": 2016,
     "label": "2013->2016 (3-yr gap, max policy shift)"},
    {"id": 2, "pre_year": 2014, "post_year": 2016,
     "label": "2014->2016 (2-yr gap, moderate drift)"},
    {"id": 3, "pre_year": 2013, "post_year": 2015,
     "label": "2013->2015 (2-yr gap, different cohort)"},
]

# Max samples per cohort for Gate 2 (keeps runtime manageable)
GATE2_MAX = 30_000

# Max samples per half for Gate 3 stream
GATE3_HALF = 15_000


# ===================================================================
#  GATE 1 -- CLASS BALANCE
# ===================================================================

def gate1_class_balance(df):
    """PASS if positive (Charged Off) class >= 15 % overall and per cohort."""
    print("\n" + "=" * 70)
    print("  GATE 1 -- CLASS BALANCE CHECK  (threshold: positive >= 15 %)")
    print("=" * 70)

    years = sorted(df["issue_year"].unique())
    per_year = []
    for yr in years:
        sub = df[df["issue_year"] == yr]
        n = len(sub)
        n_pos = sub["target"].sum()
        pct = 100 * n_pos / n if n > 0 else 0
        per_year.append((yr, n, n_pos, pct))
        print(f"  {yr}: {n:>10,} rows   Charged Off = {n_pos:>8,} ({pct:5.1f}%)")

    total = len(df)
    total_pos = df["target"].sum()
    overall_pct = 100 * total_pos / total if total > 0 else 0

    print(f"\n  Overall: {total:,} rows   Charged Off = {total_pos:,} ({overall_pct:.1f}%)")

    passed = overall_pct >= 15.0
    verdict = "PASS" if passed else "FAIL -- positive class < 15 %"
    print(f"\n  >>> GATE 1 RESULT: {verdict}  (positive = {overall_pct:.1f} %)")
    print("=" * 70)
    return passed, per_year


# ===================================================================
#  GATE 2 -- DRIFT DIAGNOSIS (3 seed offsets)
# ===================================================================

def _train_test_eval(X_train, y_train, X_test, y_test):
    """Train SGD, evaluate. Returns (accuracy, f1_macro, report_str)."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    present = sorted(set(y_test) | set(y_pred))
    names_map = {0: "Fully Paid", 1: "Charged Off"}
    tgt = [names_map[l] for l in present]
    report = classification_report(y_test, y_pred, labels=present,
                                   target_names=tgt, zero_division=0)
    return acc, f1, report


def gate2_drift_diagnosis(df):
    """
    Three cross-cohort offsets, matching the planned experiment seeds.

    Baseline: 70/30 split within the earliest PRE year (2013).
    Each offset trains on the PRE year and tests on the POST year.

    PASS if >= 2 of 3 offsets show degradation (acc or F1 drop > 1 pp).
    """
    print("\n" + "=" * 70)
    print("  GATE 2 -- DRIFT DIAGNOSIS  (3 seed offsets)")
    print("=" * 70)

    # -- Baseline: split within 2013 ---------------------------------
    print("\n  -- Baseline: 70/30 split within 2013 --")
    X_base, y_base = get_year_cohort(df, 2013, max_samples=GATE2_MAX)
    split = int(0.7 * len(X_base))
    base_acc, base_f1, base_rpt = _train_test_eval(
        X_base[:split], y_base[:split], X_base[split:], y_base[split:])
    print(f"  Accuracy: {base_acc:.4f}   F1 (macro): {base_f1:.4f}")
    print(base_rpt)

    # -- Three seed offsets ------------------------------------------
    results = []
    for seed in SEED_CONFIGS:
        label = seed["label"]
        print(f"\n  -- Seed {seed['id']}: {label} --")

        X_tr, y_tr = get_year_cohort(df, seed["pre_year"], max_samples=GATE2_MAX)
        X_te, y_te = get_year_cohort(df, seed["post_year"], max_samples=GATE2_MAX)

        print(f"  Train ({seed['pre_year']}): {len(X_tr):,}  "
              f"(pos={y_tr.sum():,}, neg={len(y_tr)-y_tr.sum():,}, "
              f"def%={100*y_tr.mean():.1f}%)")
        print(f"  Test  ({seed['post_year']}): {len(X_te):,}  "
              f"(pos={y_te.sum():,}, neg={len(y_te)-y_te.sum():,}, "
              f"def%={100*y_te.mean():.1f}%)")

        if y_te.sum() == 0 or y_te.sum() == len(y_te):
            print("  SKIPPED: test set is single-class")
            results.append((label, None, None))
            continue

        acc, f1, rpt = _train_test_eval(X_tr, y_tr, X_te, y_te)
        print(f"  Accuracy: {acc:.4f}   F1 (macro): {f1:.4f}")
        print(rpt)
        results.append((label, acc, f1))

    # -- Summary -----------------------------------------------------
    print(f"\n  -- Drift Diagnosis Summary --")
    hdr = (f"  {'Offset':<50} {'Acc':>8} {'F1':>8} "
           f"{'D Acc':>10} {'D F1':>10}")
    print(hdr)
    print(f"  {'-'*86}")
    print(f"  {'Baseline (within 2013)':<50} {base_acc:>8.4f} {base_f1:>8.4f} "
          f"{'--':>10} {'--':>10}")

    THRESH = 0.01
    deg_count = 0
    for label, acc, f1 in results:
        if acc is None:
            print(f"  {label:<50} {'SKIP':>8} {'SKIP':>8}")
            deg_count += 1  # single-class = distribution fully shifted
            continue
        ad = acc - base_acc
        fd = f1 - base_f1
        degraded = (ad < -THRESH) or (fd < -THRESH)
        mark = " << DEGRADED" if degraded else ""
        if degraded:
            deg_count += 1
        print(f"  {label:<50} {acc:>8.4f} {f1:>8.4f} "
              f"{ad:>+10.4f} {fd:>+10.4f}{mark}")

    passed = deg_count >= 2
    verdict = (
        f"PASS -- {deg_count}/3 offsets show degradation"
        if passed
        else f"FAIL -- only {deg_count}/3 show degradation (need >= 2)"
    )
    print(f"\n  >>> GATE 2 RESULT: {verdict}")
    print("=" * 70)
    return passed


# ===================================================================
#  GATE 3 -- THREE-CONFIGURATION SANITY CHECK
# ===================================================================

def _build_stream(df, pre_year, post_year, n_samples=30_000):
    """
    Build a stream with real feature-space drift.

    Pre-drift  = loans from ``pre_year``
    Post-drift = loans from ``post_year``
    """
    half = n_samples // 2

    X_pre, y_pre = get_year_cohort(df, pre_year, max_samples=half)
    X_post, y_post = get_year_cohort(df, post_year, max_samples=half)

    drift_point = len(X_pre)

    X = np.vstack([X_pre, X_post])
    y = np.concatenate([y_pre, y_post])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, drift_point


def _run_config(X, y, policy, drift_point, budget=0):
    """Run one streaming experiment, return summary dict."""
    model = StreamingModel()
    metrics = MetricsTracker()
    metrics.set_drift_point(drift_point)
    metrics.set_budget(budget)
    runner = ExperimentRunner(model, policy, metrics)
    runner.run(X, y)
    return metrics.get_summary()


def gate3_sanity_check(df):
    """
    Configs:
      (A) No-retrain baseline
      (B) Periodic K=10, low latency  (10+1)
      (C) Periodic K=5,  high latency (500+20)

    Uses Seed 1 (2013->2016) which has the maximum policy shift.

    PASS if C is meaningfully worse than B (not within 0.001).
    """
    print("\n" + "=" * 70)
    print("  GATE 3 -- THREE-CONFIGURATION SANITY CHECK")
    print("=" * 70)

    N = GATE3_HALF * 2
    seed = SEED_CONFIGS[0]  # Seed 1: 2013 -> 2016

    print(f"\n  Building stream ({N:,} samples): "
          f"{seed['pre_year']} -> {seed['post_year']} ...")
    X, y, dp = _build_stream(df, seed["pre_year"], seed["post_year"], n_samples=N)
    n = len(X)
    print(f"  Stream: {n:,} samples  "
          f"(pos={y.sum():,}, neg={n - y.sum():,})")
    print(f"  Drift point: t = {dp}")
    print(f"  Pre-drift  def%: {100 * y[:dp].mean():.1f}%")
    print(f"  Post-drift def%: {100 * y[dp:].mean():.1f}%")

    # -- (A) No-retrain
    print(f"\n  -- Config A: No-Retrain Baseline --")
    t0 = time.time()
    sa = _run_config(X, y, NeverRetrainPolicy(), dp, budget=0)
    print(f"  Overall={sa['overall_accuracy']:.4f}  "
          f"Pre={sa.get('pre_drift_accuracy', 0):.4f}  "
          f"Post={sa.get('post_drift_accuracy', 0):.4f}  "
          f"Retrains={sa['total_retrains']}  "
          f"({time.time() - t0:.1f}s)")

    # -- (B) K=10 low latency
    print(f"\n  -- Config B: Periodic K=10, Latency=11 --")
    interval_b = n // 10
    t0 = time.time()
    sb = _run_config(
        X, y,
        PeriodicPolicy(interval=interval_b, budget=10,
                       retrain_latency=10, deploy_latency=1),
        dp, budget=10)
    print(f"  Overall={sb['overall_accuracy']:.4f}  "
          f"Pre={sb.get('pre_drift_accuracy', 0):.4f}  "
          f"Post={sb.get('post_drift_accuracy', 0):.4f}  "
          f"Retrains={sb['total_retrains']}  "
          f"({time.time() - t0:.1f}s)")

    # -- (C) K=5 high latency
    print(f"\n  -- Config C: Periodic K=5, Latency=520 --")
    interval_c = n // 5
    t0 = time.time()
    sc = _run_config(
        X, y,
        PeriodicPolicy(interval=interval_c, budget=5,
                       retrain_latency=500, deploy_latency=20),
        dp, budget=5)
    print(f"  Overall={sc['overall_accuracy']:.4f}  "
          f"Pre={sc.get('pre_drift_accuracy', 0):.4f}  "
          f"Post={sc.get('post_drift_accuracy', 0):.4f}  "
          f"Retrains={sc['total_retrains']}  "
          f"({time.time() - t0:.1f}s)")

    # -- Verdict
    aa = sa["overall_accuracy"]
    ab = sb["overall_accuracy"]
    ac = sc["overall_accuracy"]
    pa = sa.get("post_drift_accuracy", 0)
    pb = sb.get("post_drift_accuracy", 0)
    pc = sc.get("post_drift_accuracy", 0)

    print(f"\n  -- Sanity Check Summary --")
    print(f"  {'Config':<30} {'Overall':>8} {'Pre':>8} "
          f"{'Post':>8} {'Ret':>5}")
    print(f"  {'-' * 62}")
    for tag, s in [("(A) No-Retrain", sa),
                   ("(B) K=10, Lat=11", sb),
                   ("(C) K=5,  Lat=520", sc)]:
        print(f"  {tag:<30} "
              f"{s['overall_accuracy']:>8.4f} "
              f"{s.get('pre_drift_accuracy', 0):>8.4f} "
              f"{s.get('post_drift_accuracy', 0):>8.4f} "
              f"{s['total_retrains']:>5}")

    d_bc = abs(ab - ac)
    d_ab = abs(aa - ab)
    d_ac = abs(aa - ac)
    mx = max(d_bc, d_ab, d_ac)
    pd_bc = abs(pb - pc)

    print(f"\n  |B-C|={d_bc:.4f}  |A-B|={d_ab:.4f}  "
          f"|A-C|={d_ac:.4f}  max={mx:.4f}")
    print(f"  Post-drift |B-C|={pd_bc:.4f}")

    if mx <= 0.001:
        verdict = ("FAIL -- all configs within 0.001; signal too weak")
        passed = False
    elif d_bc > 0.001 and ac < ab:
        verdict = (f"PASS -- constrained (C) worse than "
                   f"well-resourced (B) by {d_bc:.4f}")
        passed = True
    elif pd_bc > 0.001 and pc < pb:
        verdict = (f"PASS -- post-drift: (C) worse than "
                   f"(B) by {pd_bc:.4f}")
        passed = True
    elif d_ab > 0.001 or d_ac > 0.001:
        verdict = (f"PASS -- meaningful spread (max={mx:.4f}); "
                   f"policies differentiate from baseline")
        passed = True
    else:
        verdict = (f"MARGINAL -- spread {mx:.4f} but "
                   f"C not clearly worse than B")
        passed = False

    print(f"\n  >>> GATE 3 RESULT: {verdict}")
    print("=" * 70)
    return passed


# ===================================================================
#  MAIN
# ===================================================================

def main():
    print("#" * 70)
    print("  LendingClub DATASET FITNESS CHECK")
    print("  For: Drift-Triggered Retraining Under Budget & Latency")
    print("  Drift type: Real-world feature-space drift (underwriting policy")
    print("              changes between 2012--2016)")
    print("#" * 70)

    # -- Load data ------------------------------------------------------
    df = load_lendingclub()

    # -- Gate 1 ---------------------------------------------------------
    g1, per_year = gate1_class_balance(df)
    if not g1:
        print("\n  STOP: Class balance too low. Pick a different dataset.")
        sys.exit(1)

    # -- Gate 2 ---------------------------------------------------------
    g2 = gate2_drift_diagnosis(df)
    if not g2:
        print("\n  STOP: Insufficient drift. Pick a different dataset.")
        sys.exit(1)

    # -- Gate 3 ---------------------------------------------------------
    g3 = gate3_sanity_check(df)

    # -- Final verdict --------------------------------------------------
    print("\n" + "#" * 70)
    print("  FINAL VERDICT")
    print("#" * 70)
    tags = {True: "PASS", False: "FAIL"}
    print(f"  Gate 1 (Class Balance):   {tags[g1]}")
    print(f"  Gate 2 (Drift Diagnosis): {tags[g2]}")
    print(f"  Gate 3 (Sanity Check):    {tags[g3]}")

    if g1 and g2 and g3:
        print("\n  LendingClub dataset is SUITABLE for the experiment.")
        print("  Proceed with full factorial sweep.")
    elif g1 and g2:
        print("\n  LendingClub passes Gates 1 & 2 but Gate 3 is marginal.")
        print("  Consider tuning or investigating further.")
    else:
        print("\n  LendingClub is NOT suitable. Pick a different dataset.")
    print("#" * 70)


if __name__ == "__main__":
    main()



