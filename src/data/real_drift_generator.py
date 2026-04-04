"""
Real-data drift stream constructor.

Builds a single streaming array (X, y) from two temporal pools of real
CIS Fraud Detection data, applying one of three drift strategies:

    abrupt    – stream the pre-drift pool as-is (purely organic)
    gradual   – blend pre → post pool across a transition window
    recurring – alternate between pre- and post-pools after drift point

Labels and features are never synthesized or permuted — only the temporal
ordering of which pool segment gets streamed is controlled.

The pools may be larger than the desired output stream (e.g. 75 k pools
producing a 50 k stream).  Pass ``n_samples`` to control the output length;
when omitted it defaults to ``len(X_pre)`` for backward compatibility.
"""

import numpy as np


def build_drift_stream(
    X_pre, y_pre,
    X_post, y_post,
    drift_type,
    drift_point,
    recurrence_period,
    n_samples=None,
    seed=42,
):
    """
    Construct a single streaming dataset with the requested drift type.

    Args:
        X_pre (np.ndarray):  Feature matrix for the pre-drift temporal pool.
        y_pre (np.ndarray):  Labels for the pre-drift pool.
        X_post (np.ndarray): Feature matrix for the post-drift temporal pool.
        y_post (np.ndarray): Labels for the post-drift pool.
        drift_type (str):    One of "abrupt", "gradual", "recurring".
        drift_point (int):   Row index where drift begins in the output stream.
        recurrence_period (int): Rows per segment for recurring alternation,
                                 and width of the gradual transition window.
        n_samples (int|None): Length of the output stream.  Defaults to
                              ``len(X_pre)`` when *None*.
        seed (int):          RNG seed for probabilistic sampling in gradual mode.

    Returns:
        (X_stream, y_stream): numpy arrays with shape (n_samples, n_features)
                              and (n_samples,) respectively.
    """
    if n_samples is None:
        n_samples = len(X_pre)

    if drift_type == "abrupt":
        return _build_abrupt(X_pre, y_pre, n_samples)
    elif drift_type == "gradual":
        return _build_gradual(
            X_pre, y_pre, X_post, y_post,
            drift_point, recurrence_period, n_samples, seed,
        )
    elif drift_type == "recurring":
        return _build_recurring(
            X_pre, y_pre, X_post, y_post,
            drift_point, recurrence_period, n_samples,
        )
    else:
        raise ValueError(f"Unknown drift_type: {drift_type!r}")


# Abrupt: purely organic
def _build_abrupt(X_pre, y_pre, n_samples):
    """
    Return the first *n_samples* rows of the pre-drift pool verbatim.

    No manipulation — whatever organic distributional shift exists in the
    time-ordered real data IS the drift signal.
    """
    return X_pre[:n_samples].copy(), y_pre[:n_samples].copy()


# Gradual: probabilistic blending

def _build_gradual(X_pre, y_pre, X_post, y_post,
                   drift_point, transition_window, n_samples, seed):
    """
    Build a stream that transitions from pre-pool to post-pool gradually.

    Layout (for n_samples=50k, drift_point=25k, transition_window=5k):
        [0,      25,000)  → rows drawn from pre-pool
        [25,000, 30,000)  → each row drawn from post-pool with probability
                            α = (t − drift_point) / transition_window,
                            otherwise from pre-pool.  α ramps 0 → 1.
        [30,000, 50,000)  → rows drawn from post-pool

    Rows are consumed sequentially from each pool (no replacement).
    The pools may be larger than *n_samples*; only the needed rows are used.
    """
    n_features = X_pre.shape[1]
    rng = np.random.default_rng(seed)

    X_stream = np.empty((n_samples, n_features), dtype=X_pre.dtype)
    y_stream = np.empty(n_samples, dtype=y_pre.dtype)

    pre_idx = 0
    post_idx = 0

    for t in range(n_samples):
        if t < drift_point:
            # Pure pre-drift
            X_stream[t] = X_pre[pre_idx]
            y_stream[t] = y_pre[pre_idx]
            pre_idx += 1

        elif t < drift_point + transition_window:
            # Transition zone: increasing probability of post-pool
            alpha = (t - drift_point) / transition_window
            if rng.random() < alpha:
                X_stream[t] = X_post[post_idx]
                y_stream[t] = y_post[post_idx]
                post_idx += 1
            else:
                X_stream[t] = X_pre[pre_idx]
                y_stream[t] = y_pre[pre_idx]
                pre_idx += 1

        else:
            # Pure post-drift
            X_stream[t] = X_post[post_idx]
            y_stream[t] = y_post[post_idx]
            post_idx += 1

    return X_stream, y_stream


# Recurring: alternating segments

def _build_recurring(X_pre, y_pre, X_post, y_post,
                     drift_point, recurrence_period, n_samples):
    """
    Build a stream that alternates between pools after the drift point.

    Layout (for n_samples=50k, drift_point=25k, recurrence_period=5k):
        [0,  25k)  → pre-pool
        [25k, 30k) → post-pool     (period 0, even → post)
        [30k, 35k) → pre-pool      (period 1, odd  → pre)
        [35k, 40k) → post-pool     (period 2, even → post)
        [40k, 45k) → pre-pool      (period 3, odd  → pre)
        [45k, 50k) → post-pool     (period 4, even → post)

    Rows are consumed sequentially from each pool (no replacement).
    The pools may be larger than *n_samples*; only the needed rows are used.
    """
    n_features = X_pre.shape[1]

    X_stream = np.empty((n_samples, n_features), dtype=X_pre.dtype)
    y_stream = np.empty(n_samples, dtype=y_pre.dtype)

    pre_idx = 0
    post_idx = 0

    for t in range(n_samples):
        if t < drift_point:
            # Pre-drift: always from pre-pool
            X_stream[t] = X_pre[pre_idx]
            y_stream[t] = y_pre[pre_idx]
            pre_idx += 1
        else:
            # After drift: alternate every recurrence_period rows
            period = (t - drift_point) // recurrence_period
            if period % 2 == 0:
                # Even period → post-pool
                X_stream[t] = X_post[post_idx]
                y_stream[t] = y_post[post_idx]
                post_idx += 1
            else:
                # Odd period → pre-pool
                X_stream[t] = X_pre[pre_idx]
                y_stream[t] = y_pre[pre_idx]
                pre_idx += 1

    return X_stream, y_stream

