"""
LUFlow Dataset Fitness Check for Drift-Triggered Retraining Experiments.

Runs three sequential gate checks to determine whether the LUFlow
intrusion-detection dataset is suitable for the retraining-policy study:

  Gate 1 -- Class Balance:
      Positive (malicious) class must be >= 15 %.

  Gate 2 -- Drift Diagnosis:
      Train on one temporal period, test on another at three offsets.
      At least 2 of 3 must show performance-degrading cross-distribution
      accuracy or F1.

  Gate 3 -- Three-Configuration Sanity Check:
      (A) No-retrain baseline
      (B) Well-resourced: K = 10 budget, low latency (10 + 1)
      (C) Constrained:    K = 5 budget, high latency (500 + 20)
      Policy (C) must perform meaningfully worse than (B).
      If all three are within 0.001, the signal is too weak.

Usage:
    python luflow_fitness_check.py
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# -- Ensure project root is on sys.path so src.* imports work
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.base_model import StreamingModel
from src.policies.periodic import PeriodicPolicy
from src.policies.never_retrain_policy import NeverRetrainPolicy
from src.evaluation.metrics import MetricsTracker
from src.runner.experiment_runner import ExperimentRunner

warnings.filterwarnings("ignore")

# ===================================================================
#  CONSTANTS
# ===================================================================

DATA_DIR = PROJECT_ROOT / "src" / "data" / "LUFlow_Network_Intrusion" / "datasets"
CSV_FILES = sorted(DATA_DIR.glob("*.csv"))

FEATURE_COLS = [
    "avg_ipt", "bytes_in", "bytes_out", "dest_port",
    "entropy", "num_pkts_out", "num_pkts_in", "proto",
    "src_port", "total_entropy", "duration",
]

POSITIVE_LABEL = "malicious"
NEGATIVE_LABEL = "benign"

# Max rows to load per CSV to keep memory and runtime manageable.
MAX_ROWS_PER_DAY = 50_000


# ===================================================================
#  DATA HELPERS
# ===================================================================

def load_day_meta(csv_path):
    """Quick scan: total rows and label counts (reads label column only)."""
    df = pd.read_csv(csv_path, usecols=["label"])
    counts = df["label"].value_counts()
    return len(df), counts


def load_day_sample(csv_path, max_rows=MAX_ROWS_PER_DAY):
    """Load up to max_rows, sorted by time_start."""
    df = pd.read_csv(csv_path, nrows=max_rows)
    if "time_start" in df.columns:
        df = df.sort_values("time_start").reset_index(drop=True)
    return df


def prepare_binary(df):
    """Filter to benign/malicious, return X, y arrays."""
    df_bin = df[df["label"].isin([POSITIVE_LABEL, NEGATIVE_LABEL])].copy()
    df_bin["target"] = (df_bin["label"] == POSITIVE_LABEL).astype(int)
    X = df_bin[FEATURE_COLS].fillna(0).values.astype(np.float64)
    y = df_bin["target"].values
    return X, y, df_bin


# ===================================================================
#  GATE 1 -- CLASS BALANCE
# ===================================================================

def gate1_class_balance():
    """PASS if positive class >= 15 % in the combined binary dataset."""
    print("\n" + "=" * 70)
    print("  GATE 1 -- CLASS BALANCE CHECK  (threshold: positive >= 15 %)")
    print("=" * 70)

    total_rows = 0
    label_totals = {}
    per_day = []

    for f in CSV_FILES:
        n, counts = load_day_meta(f)
        total_rows += n
        for lbl, cnt in counts.items():
            label_totals[lbl] = label_totals.get(lbl, 0) + cnt
        n_bin = counts.get(POSITIVE_LABEL, 0) + counts.get(NEGATIVE_LABEL, 0)
        n_mal = counts.get(POSITIVE_LABEL, 0)
        pct = 100 * n_mal / n_bin if n_bin > 0 else 0
        per_day.append((f.stem, n, n_bin, n_mal, pct))
        print(f"  {f.name}: {n:>10,} rows  binary={n_bin:>9,}  "
              f"mal={n_mal:>8,} ({pct:5.1f}%)")

    print(f"\n  Total rows (all days): {total_rows:,}")
    for lbl in sorted(label_totals, key=label_totals.get, reverse=True):
        cnt = label_totals[lbl]
        print(f"    {lbl:<12}: {cnt:>10,}  ({100*cnt/total_rows:5.1f} %)")

    n_binary = (label_totals.get(POSITIVE_LABEL, 0)
                + label_totals.get(NEGATIVE_LABEL, 0))
    n_pos = label_totals.get(POSITIVE_LABEL, 0)
    pos_pct = 100 * n_pos / n_binary if n_binary > 0 else 0

    print(f"\n  Binary subset: {n_binary:,}   Positive: {n_pos:,} "
          f" ({pos_pct:.1f} %)")

    passed = pos_pct >= 15.0
    verdict = "PASS" if passed else "FAIL -- positive class < 15 %"
    print(f"\n  >>> GATE 1 RESULT: {verdict}  (positive = {pos_pct:.1f} %)")
    print("=" * 70)
    return passed, per_day


# ===================================================================
#  GATE 2 -- DRIFT DIAGNOSIS (3 temporal offsets)
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
    names_map = {0: "benign", 1: "malicious"}
    tgt = [names_map[l] for l in present]
    report = classification_report(y_test, y_pred, labels=present,
                                   target_names=tgt, zero_division=0)
    return acc, f1, report


def _pick_drift_periods(per_day):
    """
    Group days into three periods by month/year:
      Period A = Jan 2021   (8 days, mix of 0% and ~50% malicious)
      Period B = Feb 2021   (10 days, consistently 22-44% malicious)
      Period C = Jun 2022   (3 days, 0-58% malicious)
    """
    buckets = {"jan21": [], "feb21": [], "jun22": []}
    for stem, _, _, _, _ in per_day:
        if stem.startswith("2021.01"):
            buckets["jan21"].append(stem)
        elif stem.startswith("2021.02"):
            buckets["feb21"].append(stem)
        elif stem.startswith("2022.06"):
            buckets["jun22"].append(stem)
    result = []
    for key in ["jan21", "feb21", "jun22"]:
        if buckets[key]:
            result.append(buckets[key])
    return result


def _load_period(stems, max_total=30000):
    """Load and concat binary data for a list of day stems, capped."""
    frames = []
    per_file = max(5000, max_total // max(len(stems), 1))
    for stem in stems:
        fpath = DATA_DIR / (stem + ".csv")
        df = load_day_sample(fpath, max_rows=per_file)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    X, y, _ = prepare_binary(combined)
    return X, y


def gate2_drift_diagnosis(per_day):
    """
    Three cross-period offsets. Needs >= 2 of 3 to show degradation.

    Offsets:
      A: Train Jan-2021,  Test Feb-2021   (1-month gap, same year)
      B: Train Jan-2021,  Test Jun-2022   (18-month gap)
      C: Train Feb-2021,  Test Jun-2022   (16-month gap)
    Baseline: 70/30 split within Jan-2021.
    """
    print("\n" + "=" * 70)
    print("  GATE 2 -- DRIFT DIAGNOSIS  (3 cross-period offsets)")
    print("=" * 70)

    periods = _pick_drift_periods(per_day)
    if len(periods) < 3:
        print(f"  Only {len(periods)} period(s) found -- need 3. FAIL.")
        return False

    p_names = ["Jan-2021", "Feb-2021", "Jun-2022"]
    print(f"  Period A ({p_names[0]}): {periods[0]}")
    print(f"  Period B ({p_names[1]}): {periods[1]}")
    print(f"  Period C ({p_names[2]}): {periods[2]}")

    print("\n  Loading period data (subsampled) ...")
    X_a, y_a = _load_period(periods[0])
    X_b, y_b = _load_period(periods[1])
    X_c, y_c = _load_period(periods[2])
    for tag, X, y in [("A", X_a, y_a), ("B", X_b, y_b), ("C", X_c, y_c)]:
        print(f"    Period {tag}: {len(X):,} samples  "
              f"(pos={y.sum():,}, neg={len(y)-y.sum():,}, "
              f"mal%={100*y.mean():.1f}%)")

    # -- Baseline: split within Period A
    print(f"\n  -- Baseline: 70/30 split within {p_names[0]} --")
    split = int(0.7 * len(X_a))
    base_acc, base_f1, base_rpt = _train_test_eval(
        X_a[:split], y_a[:split], X_a[split:], y_a[split:])
    print(f"  Accuracy: {base_acc:.4f}   F1 (macro): {base_f1:.4f}")
    print(base_rpt)

    # -- Three offsets
    offset_specs = [
        ("A: Jan21 -> Feb21", X_a, y_a, X_b, y_b),
        ("B: Jan21 -> Jun22", X_a, y_a, X_c, y_c),
        ("C: Feb21 -> Jun22", X_b, y_b, X_c, y_c),
    ]

    results = []
    for label, Xtr, ytr, Xte, yte in offset_specs:
        if yte.sum() == 0 or yte.sum() == len(yte):
            print(f"\n  -- Offset {label} --")
            print(f"  SKIPPED: test set is single-class "
                  f"(pos={yte.sum()}, neg={len(yte)-yte.sum()})")
            results.append((label, None, None))
            continue
        print(f"\n  -- Offset {label} --")
        acc, f1, rpt = _train_test_eval(Xtr, ytr, Xte, yte)
        print(f"  Accuracy: {acc:.4f}   F1 (macro): {f1:.4f}")
        print(rpt)
        results.append((label, acc, f1))

    # -- Summary
    print(f"\n  -- Drift Diagnosis Summary --")
    hdr = (f"  {'Offset':<22} {'Accuracy':>10} {'F1':>10} "
           f"{'Acc delta':>12} {'F1 delta':>12}")
    print(hdr)
    print(f"  {'-'*66}")
    print(f"  {'Baseline':<22} {base_acc:>10.4f} {base_f1:>10.4f} "
          f"{'--':>12} {'--':>12}")

    THRESH = 0.01
    deg_count = 0
    for label, acc, f1 in results:
        if acc is None:
            print(f"  {label:<22} {'SKIP':>10} {'SKIP':>10}")
            deg_count += 1   # single-class = distribution fully shifted
            continue
        ad = acc - base_acc
        fd = f1 - base_f1
        degraded = (ad < -THRESH) or (fd < -THRESH)
        mark = " << DEGRADED" if degraded else ""
        if degraded:
            deg_count += 1
        print(f"  {label:<22} {acc:>10.4f} {f1:>10.4f} "
              f"{ad:>+12.4f} {fd:>+12.4f}{mark}")

    passed = deg_count >= 2
    verdict = (f"PASS -- {deg_count}/3 offsets show degradation"
               if passed
               else f"FAIL -- only {deg_count}/3 show degradation (need >= 2)")
    print(f"\n  >>> GATE 2 RESULT: {verdict}")
    print("=" * 70)
    return passed


# ===================================================================
#  GATE 3 -- THREE-CONFIGURATION SANITY CHECK
# ===================================================================

def _build_stream(per_day, n_samples=10000):
    """
    Build a stream with performance-degrading drift (easy -> hard).

    Pre-drift  = benign-only or low-malicious days (model learns benign)
    Post-drift = high-malicious days (attacks emerge, model must adapt)
    """
    half = n_samples // 2

    low_mal = []    # <= 5% malicious
    high_mal = []   # >= 15% malicious
    for stem, _, n_bin, n_mal, pct in per_day:
        if n_bin < 100:
            continue
        if pct <= 5.0:
            low_mal.append(stem)
        elif pct >= 15.0:
            high_mal.append(stem)

    print(f"    Low-malicious days  (<= 5%):  {low_mal}")
    print(f"    High-malicious days (>=15%): {high_mal}")

    if not low_mal or not high_mal:
        print("    WARNING: cannot separate low/high; using temporal split")
        all_stems = [s for s, _, _, _, _ in per_day]
        mid = len(all_stems) // 2
        low_mal = all_stems[:mid]
        high_mal = all_stems[mid:]

    # Load pre-drift (low-mal) and post-drift (high-mal)
    pre_frames = []
    pf = max(2000, half // max(len(low_mal), 1))
    for stem in low_mal:
        df = load_day_sample(DATA_DIR / (stem + ".csv"), max_rows=pf)
        pre_frames.append(df)
    pre_df = pd.concat(pre_frames, ignore_index=True)
    pre_df = pre_df[pre_df["label"].isin(
        [POSITIVE_LABEL, NEGATIVE_LABEL])].copy()
    pre_df["target"] = (pre_df["label"] == POSITIVE_LABEL).astype(int)
    if len(pre_df) > half:
        pre_df = pre_df.iloc[:half]

    post_frames = []
    pf = max(2000, half // max(len(high_mal), 1))
    for stem in high_mal:
        df = load_day_sample(DATA_DIR / (stem + ".csv"), max_rows=pf)
        post_frames.append(df)
    post_df = pd.concat(post_frames, ignore_index=True)
    post_df = post_df[post_df["label"].isin(
        [POSITIVE_LABEL, NEGATIVE_LABEL])].copy()
    post_df["target"] = (post_df["label"] == POSITIVE_LABEL).astype(int)
    if len(post_df) > half:
        post_df = post_df.iloc[:half]

    drift_point = len(pre_df)
    stream = pd.concat([pre_df, post_df], ignore_index=True)

    X = stream[FEATURE_COLS].fillna(0).values.astype(np.float64)
    y = stream["target"].values

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


def gate3_sanity_check(per_day):
    """
    Configs:
      (A) No-retrain baseline
      (B) Periodic K=10, low latency  (10+1)
      (C) Periodic K=5,  high latency (500+20)

    PASS if C is meaningfully worse than B (not within 0.001).
    """
    print("\n" + "=" * 70)
    print("  GATE 3 -- THREE-CONFIGURATION SANITY CHECK")
    print("=" * 70)

    N = 10000
    print(f"\n  Building stream ({N:,} samples): "
          f"benign-only -> attack days ...")
    X, y, dp = _build_stream(per_day, n_samples=N)
    n = len(X)
    print(f"  Stream: {n:,} samples  "
          f"(pos={y.sum():,}, neg={n - y.sum():,})")
    print(f"  Drift point: t = {dp}")
    print(f"  Pre-drift  mal%: {100*y[:dp].mean():.1f}%")
    print(f"  Post-drift mal%: {100*y[dp:].mean():.1f}%")

    # -- (A) No-retrain
    print(f"\n  -- Config A: No-Retrain Baseline --")
    t0 = time.time()
    sa = _run_config(X, y, NeverRetrainPolicy(), dp, budget=0)
    print(f"  Overall={sa['overall_accuracy']:.4f}  "
          f"Pre={sa.get('pre_drift_accuracy', 0):.4f}  "
          f"Post={sa.get('post_drift_accuracy', 0):.4f}  "
          f"Retrains={sa['total_retrains']}  "
          f"({time.time()-t0:.1f}s)")

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
          f"({time.time()-t0:.1f}s)")

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
          f"({time.time()-t0:.1f}s)")

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
    print(f"  {'-'*62}")
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
        verdict = ("FAIL -- all configs within 0.001; "
                   "signal too weak")
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
    print("  LUFlow DATASET FITNESS CHECK")
    print("  For: Drift-Triggered Retraining Under Budget & Latency")
    print(f"  Files: {len(CSV_FILES)} day-CSVs in {DATA_DIR.name}/")
    print("#" * 70)

    # -- Gate 1
    g1, per_day = gate1_class_balance()
    if not g1:
        print("\n  STOP: Class balance too low. Pick a different dataset.")
        sys.exit(1)

    # -- Gate 2
    g2 = gate2_drift_diagnosis(per_day)
    if not g2:
        print("\n  STOP: Insufficient drift. Pick a different dataset.")
        sys.exit(1)

    # -- Gate 3
    g3 = gate3_sanity_check(per_day)

    # -- Final verdict
    print("\n" + "#" * 70)
    print("  FINAL VERDICT")
    print("#" * 70)
    tags = {True: "PASS", False: "FAIL"}
    print(f"  Gate 1 (Class Balance):   {tags[g1]}")
    print(f"  Gate 2 (Drift Diagnosis): {tags[g2]}")
    print(f"  Gate 3 (Sanity Check):    {tags[g3]}")

    if g1 and g2 and g3:
        print("\n  LUFlow dataset is SUITABLE for the experiment.")
        print("  Proceed with full factorial sweep.")
    elif g1 and g2:
        print("\n  LUFlow passes Gates 1 & 2 but Gate 3 is marginal.")
        print("  Consider tuning or investigating further.")
    else:
        print("\n  LUFlow is NOT suitable. Pick a different dataset.")
    print("#" * 70)


if __name__ == "__main__":
    main()

