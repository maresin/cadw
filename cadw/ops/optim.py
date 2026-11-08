"""Streaming optimization primitives (v0.9.89).

CADW-native streaming versions of classical training loops: state is
O(d * K) bytes (parameters + accumulators), independent of n.

Contrast with closed-form solutions (e.g. ridge via XTX = O(d^2)),
which force an arbitrary memory-driven cutoff on feature dimension d.
For wide-feature models (ROCKET with n_kernels=10000 → d=20000 →
XTX=3.2 GB), closed-form is a base-case violation
(METHODOLOGY_ALGORITHM_SELECTION.md §1.1).

See docs/ml_tools/METHODOLOGY_ALGORITHM_SELECTION.md §5 (planned
pattern 4: streaming SGD).
- v0.9.156: added grid_search + random_search (L14, P1).
  Consumers: GridSearch, RandomizedSearch (O_eda).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.numeric import sigmoid




def sgd_streaming(
    batch_stream,
    n_features: int,
    n_outputs: int,
    *,
    loss: str = "ridge",
    lam: float = 1.0,
    lr: float = 0.01,
    epochs: int = 30,
    beta: float = 0.9,
    eps: float = 1e-8,
    penalty_mask: Optional[np.ndarray] = None,
    use_rmsprop: bool = True,
    mg: Optional[MemoryGuard] = None,
    name: str = "sgd",
) -> np.ndarray:
    """Streaming SGD + RMSprop. Supports ridge and logistic loss.

    Returns W (n_outputs, d).

    Parameters
    ----------
    batch_stream : callable() -> iterator of (X_b, Y_b)
        X_b: (b, d) float64; Y_b: (b, K) float64 (one-hot or ±1).
        Called once for warm start, then `epochs` times for gradient
        steps. Each call must produce a fresh iterator.
    n_features : int
        Feature dimension d (user-chosen hyperparameter of the caller,
        NOT derived from n). SGD state is O(d), independent of d².
    n_outputs : int
        1 for binary (±1 target), K for one-vs-rest multiclass.

    Loss
    ----
    loss="ridge" (default):
        0.5 * mean((X W^T - Y)^2) + 0.5 * lam * ||W||^2
    loss="logistic":
        mean(-[Y * log(sigmoid(X W^T))
              + (1-Y) * log(1 - sigmoid(X W^T))])
        + 0.5 * lam * ||W||^2
        (Y_b expected in {0, 1}; for ±1 targets, caller transforms
        to 0/1 before passing to batch_stream.)

    Update (gradient-only difference; optimizer identical)
    ------
    z    = X_b @ W^T                           (b, K)
    pred = z (ridge) | sigmoid(z) (logistic)
    grad = (pred - Y_b).T @ X_b / b + lam * W  (K, d)
    h    = beta * h + (1-beta) * grad**2       (K, d)  [RMSprop]
    W   -= lr * grad / (sqrt(h) + eps)

    Warm start
    ----------
    ridge: W_0 = mean(Y^T X) over stream (one gradient step at W=0).
    logistic: W_0 = 0 (standard; mean(Y^T X) would saturate sigmoid
      for ±1 targets and 0/1 targets where class imbalance is high).

    Caller responsibility for lam
    -----------------------------
    To match a closed-form objective `sum_loss + α·||W||²`:
      ridge:    lam = α / n
      logistic: lam = α / n
    (Derivation: f/n where f = -sum_loss + α/2·||W||² gives
    -mean_loss + α/(2n)·||W||². SGD's lam/2·||W||² matches when
    lam = α/n in both cases. Earlier note "2·α/n" was incorrect
    — the factor-2 in logistic's objective is already folded into
    α/2 penalty scale; linear Ridge's alpha is on sum_sq not
    sum_sq/2, so both reduce to lam = α/n.)

    penalty_mask : np.ndarray, optional
        Shape (n_outputs, n_features) or (1, n_features). Multiplicative
        mask on the L2 penalty: grad += lam * W * penalty_mask.
        Use to exempt intercept (mask[:, 0] = 0). Default None → all
        parameters penalized.

    use_rmsprop : bool, default True
        True: RMSprop (h accumulator, per-param adaptive lr). Good for
        ridge (quadratic, well-scaled gradients) and default rocket.
        False: plain GD W -= lr*grad. Required for strongly-regularized
        logistic (RMSprop oscillates at amplitude ~lr when penalty
        dominates gradient; plain GD contracts geometrically at rate
        1 - lr*lam). See _fit_sgd_logistic.

    State
    -----
    W (K, d) + h (K, d) = 2 * K * d * 8 bytes.
    For d=20000, K=1: 320 KB. Independent of n and n_series.

    Determinism
    -----------
    Batch order in batch_stream determines SGD order. Same batch_stream
    → bit-identical result across runs (no random seed involved).
    """
    if loss not in ("ridge", "logistic"):
        raise ValueError(
            f"{name}: loss must be 'ridge' or 'logistic'; got {loss!r}")

    d = n_features
    W = np.zeros((n_outputs, d), dtype=np.float64)
    h = np.zeros((n_outputs, d), dtype=np.float64)
    if mg is not None:
        # Classifier state: W + h + small numpy overhead.
        mg.check_state(2 * d * n_outputs * 8 + d * 8,
                       name=name + ".classifier.state")

    if loss == "ridge":
        # Warm start: W_0 ≈ (1/n) X^T Y (one gradient step at W=0).
        n_seen = 0
        for X_b, Y_b in batch_stream():
            if mg is not None:
                mg.check_state(len(X_b) * d * 8 + n_outputs * d * 8,
                               name=name + ".warm")
            W += Y_b.T @ X_b
            n_seen += X_b.shape[0]
        if n_seen == 0:
            raise ValueError(f"{name}: empty feature stream")
        W /= float(n_seen)
    else:  # logistic — no warm start (Y^T X would saturate sigmoid
           # for ±1 targets or imbalanced 0/1 targets)
        n_seen = 0
        for X_b, _ in batch_stream():
            n_seen += X_b.shape[0]
        if n_seen == 0:
            raise ValueError(f"{name}: empty feature stream")

    for _epoch in range(epochs):
        for X_b, Y_b in batch_stream():
            if mg is not None:
                # Transients: pred (b,K), grad (K,d), h update (K,d).
                # Estimate 3 * (K*d + b*K) * 8 bytes.
                mg.check_state(3 * (n_outputs * d + len(X_b) * n_outputs) * 8,
                               name=name + ".batch")
            z = X_b @ W.T                              # (b, K)
            if loss == "ridge":
                pred = z
            else:  # logistic
                pred = sigmoid(z)
            grad = (pred - Y_b).T @ X_b / max(len(X_b), 1)
            if penalty_mask is not None:
                grad = grad + lam * W * penalty_mask
            else:
                grad = grad + lam * W
            if use_rmsprop:
                h = beta * h + (1.0 - beta) * grad * grad
                W -= lr * grad / (np.sqrt(h) + eps)
            else:
                # Plain GD. Strongly-regularized logistic: RMSprop
                # oscillates at amplitude ~lr (adaptive denominator
                # can't resolve small grads). Plain GD contracts
                # geometrically (rate 1 - lr*lam).
                W -= lr * grad
    return W


# ============================================================================
# Hyperparameter search (v0.9.156, P1 L14)
# ============================================================================

def grid_search(
    param_grid: dict,
    scorer,
    *,
    maximize: bool = True,
    memory_guard=None,
) -> dict:
    """Exhaustive grid search over param combinations.

    Args:
      param_grid: dict[param_name] -> list of candidate values.
      scorer: callable(params_dict) -> float. Called once per combination.
      maximize: True -> pick max(score); False -> min(score).
      memory_guard: optional; checked before storing results (n_combos *
                    ~200 B). Very large grids should be chunked by caller.

    Returns:
      dict with keys:
        best_params: dict, best combination found
        best_score:  float
        results:     list[dict] of {params: dict, score: float}
                     (ordered lexicographically by param names)

    Raises:
      ValueError: empty grid, empty value list, or memory limit exceeded.

    Complexity: n_combos = prod(len(v) for v in param_grid.values()).
    Score computed n_combos times, O(1) state per call.
    """
    if not param_grid:
        raise ValueError("grid_search: param_grid must be non-empty")
    param_names = sorted(param_grid.keys())
    for name in param_names:
        vals = param_grid[name]
        if not hasattr(vals, "__iter__"):
            raise ValueError(f"grid_search: {name} values must be iterable")
        if not list(vals):
            raise ValueError(f"grid_search: {name} has empty value list")

    # Cartesian product
    import itertools
    value_lists = [list(param_grid[n]) for n in param_names]
    total = 1
    for vl in value_lists:
        total *= len(vl)

    if memory_guard is not None:
        memory_guard.check_state(total * 200 + 256, name="grid_search.results")

    results = []
    best_params = None
    best_score = None
    for combo in itertools.product(*value_lists):
        params = dict(zip(param_names, combo))
        score = float(scorer(params))
        results.append({"params": params, "score": score})
        if best_score is None or (
                (maximize and score > best_score)
                or (not maximize and score < best_score)):
            best_score = score
            best_params = params

    return {
        "best_params": best_params,
        "best_score": best_score,
        "results": results,
    }


def random_search(
    param_dist: dict,
    scorer,
    *,
    n_iter: int = 10,
    maximize: bool = True,
    seed: int = 0,
    memory_guard=None,
) -> dict:
    """Random search over param distributions.

    Args:
      param_dist: dict[param_name] -> distribution spec, one of:
        - tuple (a, b):  uniform sample. int if both a, b are ints
                        (rng.integers(a, b+1) inclusive); float otherwise
                        (rng.uniform(a, b)).
        - list:          categorical sample via rng.choice.
        - callable:      rng -> value (advanced custom).
      scorer: callable(params_dict) -> float.
      n_iter: number of samples.
      maximize: True -> pick max(score); False -> min(score).
      seed: RNG seed (reproducibility).
      memory_guard: optional; checked before storing results.

    Returns:
      Same dict schema as grid_search.

    Raises:
      ValueError: empty param_dist, invalid spec, n_iter < 1.
    """
    import numpy as np
    if not param_dist:
        raise ValueError("random_search: param_dist must be non-empty")
    if n_iter < 1:
        raise ValueError(f"random_search: n_iter must be >= 1; got {n_iter}")
    if memory_guard is not None:
        memory_guard.check_state(n_iter * 200 + 256, name="random_search.results")

    rng = np.random.default_rng(seed)
    param_names = sorted(param_dist.keys())
    results = []
    best_params = None
    best_score = None

    for _ in range(n_iter):
        params = {}
        for name in param_names:
            spec = param_dist[name]
            if callable(spec):
                params[name] = spec(rng)
            elif isinstance(spec, tuple) and len(spec) == 2:
                a, b = spec
                if isinstance(a, int) and isinstance(b, int):
                    params[name] = int(rng.integers(a, b + 1))
                else:
                    params[name] = float(rng.uniform(float(a), float(b)))
            elif isinstance(spec, (list, np.ndarray)):
                options = list(spec)
                if not options:
                    raise ValueError(
                        f"random_search: {name} has empty value list")
                idx = int(rng.integers(0, len(options)))
                params[name] = options[idx]
            else:
                raise ValueError(
                    f"random_search: {name} spec must be tuple(a,b), "
                    f"list, or callable; got {type(spec).__name__}")

        score = float(scorer(params))
        results.append({"params": params, "score": score})
        if best_score is None or (
                (maximize and score > best_score)
                or (not maximize and score < best_score)):
            best_score = score
            best_params = params

    return {
        "best_params": best_params,
        "best_score": best_score,
        "results": results,
    }
