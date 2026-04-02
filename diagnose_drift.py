"""
Drift Diagnosis for CIS Fraud Detection Dataset.

Before running calibration or experiments, this script checks whether
meaningful distributional drift actually exists at the chosen split point.

Methodology:
    1. Load the first 50,000 time-ordered rows (the calibration/experiment stream).
    2. Split at the midpoint (t=25,000) into pre-drift and post-drift halves.
    3. Train a classifier on 80% of the pre-drift half.
    4. Evaluate on:
       (a) Held-out 20% of pre-drift half  → "same-distribution" baseline
       (b) The full post-drift half          → "cross-distribution" test
    5. If accuracy on (b) is significantly lower than (a), genuine drift exists.
       If they are similar, the two temporal halves are not meaningfully different.

Additional diagnostics:
    - Class balance (fraud rate) in each half
    - Feature-level KS tests for distributional shift
    - Multiple window offsets (0, 10k, 20k) to check consistency

Usage:
    python diagnose_drift.py
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy import stats

from src.data.fraud_data_loader import FraudDataLoader


DATA_DIR = Path("src/data/CIS Fraud Detection")
N_SAMPLES = 50_000
DRIFT_POINT = 25_000
OFFSETS = [0, 10_000, 20_000]
OUTPUT_PATH = Path("results/calibration_for_CISFraudDetectionDataset/drift_diagnosis.txt")


def diagnose_single_window(X, y, offset_label):
    """Run drift diagnosis on one 50k window."""
    drift_point = DRIFT_POINT
    lines = []

    # Split into pre and post
    X_pre, y_pre = X[:drift_point], y[:drift_point]
    X_post, y_post = X[drift_point:], y[drift_point:]

    # ── Class balance ────────────────────────────────────────────────
    fraud_pre = y_pre.mean()
    fraud_post = y_post.mean()
    lines.append(f"  Fraud rate (pre-drift) : {fraud_pre:.4f}  ({y_pre.sum():,} / {len(y_pre):,})")
    lines.append(f"  Fraud rate (post-drift): {fraud_post:.4f}  ({y_post.sum():,} / {len(y_post):,})")
    lines.append(f"  Fraud rate shift       : {fraud_post - fraud_pre:+.4f}  "
                 f"({'higher' if fraud_post > fraud_pre else 'lower'} post-drift)")
    lines.append("")

    # ── Train/test split within pre-drift ────────────────────────────
    n_pre = len(X_pre)
    split_idx = int(n_pre * 0.8)
    X_train, y_train = X_pre[:split_idx], y_pre[:split_idx]
    X_val, y_val = X_pre[split_idx:], y_pre[split_idx:]

    # Train SGDClassifier (same model type as experiments)
    clf = SGDClassifier(loss="log_loss", random_state=42, max_iter=1000, tol=1e-3)
    clf.fit(X_train, y_train)

    # ── Evaluate: same-distribution (pre-drift held-out) ────────────
    y_pred_val = clf.predict(X_val)
    y_prob_val = clf.decision_function(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    try:
        auc_val = roc_auc_score(y_val, y_prob_val)
    except ValueError:
        auc_val = float('nan')

    lines.append(f"  Same-distribution (pre-drift held-out, n={len(X_val):,}):")
    lines.append(f"    Accuracy : {acc_val:.4f}")
    lines.append(f"    F1       : {f1_val:.4f}")
    lines.append(f"    AUC      : {auc_val:.4f}")
    lines.append("")

    # ── Evaluate: cross-distribution (post-drift) ───────────────────
    y_pred_post = clf.predict(X_post)
    y_prob_post = clf.decision_function(X_post)
    acc_post = accuracy_score(y_post, y_pred_post)
    f1_post = f1_score(y_post, y_pred_post, zero_division=0)
    try:
        auc_post = roc_auc_score(y_post, y_prob_post)
    except ValueError:
        auc_post = float('nan')

    lines.append(f"  Cross-distribution (post-drift, n={len(X_post):,}):")
    lines.append(f"    Accuracy : {acc_post:.4f}")
    lines.append(f"    F1       : {f1_post:.4f}")
    lines.append(f"    AUC      : {auc_post:.4f}")
    lines.append("")

    # ── Performance delta ───────────────────────────────────────────
    acc_drop = acc_val - acc_post
    f1_drop = f1_val - f1_post
    auc_drop = auc_val - auc_post if not (np.isnan(auc_val) or np.isnan(auc_post)) else float('nan')

    lines.append(f"  Performance delta (pre - post):")
    lines.append(f"    Accuracy drop : {acc_drop:+.4f}  "
                 f"({'DRIFT SIGNAL' if abs(acc_drop) > 0.02 else 'WEAK/NO SIGNAL'})")
    lines.append(f"    F1 drop       : {f1_drop:+.4f}  "
                 f"({'DRIFT SIGNAL' if abs(f1_drop) > 0.02 else 'WEAK/NO SIGNAL'})")
    lines.append(f"    AUC drop      : {auc_drop:+.4f}  "
                 f"({'DRIFT SIGNAL' if abs(auc_drop) > 0.02 else 'WEAK/NO SIGNAL'})")
    lines.append("")

    # ── Feature-level KS tests (top shifts) ─────────────────────────
    n_features = X_pre.shape[1]
    ks_results = []
    for i in range(n_features):
        stat, pval = stats.ks_2samp(X_pre[:, i], X_post[:, i])
        ks_results.append((i, stat, pval))

    # Sort by KS statistic (descending)
    ks_results.sort(key=lambda r: r[1], reverse=True)

    n_sig_001 = sum(1 for _, _, p in ks_results if p < 0.001)
    n_sig_01 = sum(1 for _, _, p in ks_results if p < 0.01)
    n_sig_05 = sum(1 for _, _, p in ks_results if p < 0.05)

    lines.append(f"  Feature-level KS tests ({n_features} features):")
    lines.append(f"    Significant at p < 0.001 : {n_sig_001} / {n_features}  ({n_sig_001/n_features*100:.1f}%)")
    lines.append(f"    Significant at p < 0.01  : {n_sig_01} / {n_features}  ({n_sig_01/n_features*100:.1f}%)")
    lines.append(f"    Significant at p < 0.05  : {n_sig_05} / {n_features}  ({n_sig_05/n_features*100:.1f}%)")
    lines.append(f"    Top 10 most-shifted features:")
    for feat_idx, ks_stat, ks_pval in ks_results[:10]:
        lines.append(f"      Feature {feat_idx:>3}: KS={ks_stat:.4f}  p={ks_pval:.2e}")
    lines.append("")

    return lines, {
        "offset": offset_label,
        "fraud_pre": fraud_pre, "fraud_post": fraud_post,
        "acc_val": acc_val, "acc_post": acc_post, "acc_drop": acc_drop,
        "f1_val": f1_val, "f1_post": f1_post, "f1_drop": f1_drop,
        "auc_val": auc_val, "auc_post": auc_post, "auc_drop": auc_drop,
        "n_sig_001": n_sig_001, "n_features": n_features,
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_lines = []
    all_lines.append("=" * 80)
    all_lines.append("  DRIFT DIAGNOSIS — CIS Fraud Detection Dataset")
    all_lines.append("=" * 80)
    all_lines.append(f"  Stream length : {N_SAMPLES:,}")
    all_lines.append(f"  Drift point   : {DRIFT_POINT:,} (midpoint)")
    all_lines.append(f"  Window offsets: {OFFSETS}")
    all_lines.append(f"  Model         : SGDClassifier(loss='log_loss')")
    all_lines.append("")
    all_lines.append("  Question: Does meaningful distributional drift exist at the")
    all_lines.append("  chosen split point (t = n_samples/2)?")
    all_lines.append("")
    all_lines.append("  Method: Train on 80% of pre-drift, evaluate on:")
    all_lines.append("    (a) Held-out 20% of pre-drift  → same-distribution baseline")
    all_lines.append("    (b) Full post-drift half        → cross-distribution test")
    all_lines.append("  If (b) is significantly worse than (a), drift exists.")
    all_lines.append("")

    loader = FraudDataLoader(DATA_DIR)

    summaries = []
    for offset in OFFSETS:
        all_lines.append("-" * 80)
        all_lines.append(f"  Window offset = {offset:,}")
        all_lines.append("-" * 80)

        X, y = loader.get_pool(start_offset=offset, n_samples=N_SAMPLES)
        lines, summary = diagnose_single_window(X, y, offset)
        all_lines.extend(lines)
        summaries.append(summary)

    # ── Overall verdict ──────────────────────────────────────────────
    all_lines.append("=" * 80)
    all_lines.append("  OVERALL VERDICT")
    all_lines.append("=" * 80)
    all_lines.append("")

    all_lines.append("  Summary across all window offsets:")
    all_lines.append(f"  {'Offset':<10} {'Acc(pre)':<10} {'Acc(post)':<10} {'Acc Drop':<10} "
                     f"{'F1(pre)':<10} {'F1(post)':<10} {'F1 Drop':<10} "
                     f"{'KS sig':<10}")
    all_lines.append(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} "
                     f"{'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for s in summaries:
        all_lines.append(
            f"  {s['offset']:<10} {s['acc_val']:<10.4f} {s['acc_post']:<10.4f} {s['acc_drop']:<+10.4f} "
            f"{s['f1_val']:<10.4f} {s['f1_post']:<10.4f} {s['f1_drop']:<+10.4f} "
            f"{s['n_sig_001']}/{s['n_features']}"
        )
    all_lines.append("")

    # Determine drift strength
    avg_acc_drop = np.mean([s["acc_drop"] for s in summaries])
    avg_f1_drop = np.mean([s["f1_drop"] for s in summaries])
    avg_fraud_shift = np.mean([s["fraud_post"] - s["fraud_pre"] for s in summaries])
    avg_ks_frac = np.mean([s["n_sig_001"] / s["n_features"] for s in summaries])

    all_lines.append(f"  Average accuracy drop : {avg_acc_drop:+.4f}")
    all_lines.append(f"  Average F1 drop       : {avg_f1_drop:+.4f}")
    all_lines.append(f"  Average fraud rate shift: {avg_fraud_shift:+.4f}")
    all_lines.append(f"  Average KS-sig fraction : {avg_ks_frac:.1%} of features shifted at p<0.001")
    all_lines.append("")

    if abs(avg_acc_drop) > 0.02 or abs(avg_f1_drop) > 0.05:
        verdict = "STRONG DRIFT"
        detail = ("Meaningful distributional drift exists at the midpoint split. "
                  "The model trained on pre-drift data performs significantly worse "
                  "on post-drift data, confirming genuine temporal shift.")
    elif abs(avg_acc_drop) > 0.005 or abs(avg_f1_drop) > 0.02 or avg_ks_frac > 0.3:
        verdict = "WEAK/MODERATE DRIFT"
        detail = ("Some distributional drift exists but the effect on model performance "
                  "is modest. Feature distributions shift but the classification boundary "
                  "is only mildly affected. This may explain why ADWIN struggles to "
                  "distinguish drift from noise.")
    else:
        verdict = "NO MEANINGFUL DRIFT"
        detail = ("The pre-drift and post-drift halves are not meaningfully different "
                  "from the model's perspective. The chosen split point does not "
                  "produce a detectable concept drift. Consider using semi-synthetic "
                  "drift injection even for the abrupt condition.")

    all_lines.append(f"  VERDICT: {verdict}")
    all_lines.append(f"  {detail}")
    all_lines.append("")

    # ── Implications ─────────────────────────────────────────────────
    all_lines.append("=" * 80)
    all_lines.append("  IMPLICATIONS FOR EXPERIMENT DESIGN")
    all_lines.append("=" * 80)
    all_lines.append("")
    all_lines.append("  1. If STRONG DRIFT: calibration results are valid; ADWIN's failure")
    all_lines.append("     is a legitimate finding about its unsuitability for imbalanced data.")
    all_lines.append("")
    all_lines.append("  2. If WEAK DRIFT: the organic temporal shift may be too subtle for")
    all_lines.append("     statistical detectors. Consider using semi-synthetic drift injection")
    all_lines.append("     (build_drift_stream with two distinct temporal pools) even for the")
    all_lines.append("     abrupt condition, to guarantee a detectable distributional shift.")
    all_lines.append("")
    all_lines.append("  3. If NO DRIFT: the abrupt condition on this dataset is not")
    all_lines.append("     meaningful. Only gradual/recurring (which use two distinct temporal")
    all_lines.append("     pools) will produce useful results.")
    all_lines.append("")

    output = "\n".join(all_lines)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(output)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

