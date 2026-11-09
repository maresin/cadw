# PARAMS_DESIGN — unified parameter surface for ML/TS models

**Version:** 1.0 (2026-11-10, v0.9.175)
**Status:** normative design (implemented v0.9.176+).
**Companions:** DEC-067 (work front), DEC-068 (unified params), DEC-069 (df.apply).
**Hierarchy:** Tier 2 (methodological). Complements AI_CONSTITUTION §4.0 (R1-R7).

---

## §1. Principle

Every ML/TS `fit_*` accepts **5 standard parameters** (subset per model) plus
model-specific kwargs. The 5 are orthogonal: each addresses a different OOC axis.

| Param | Axis | Purpose | Values |
|---|---|---|---|
| `window` | n (algorithm state) | Bound history | `"auto"` \| `int` \| `None` |
| `solver` | d (state = O(d²)) | Algorithm variant | `"closed"` \| `"sgd"` \| `"auto"` |
| `output` | n (returned array) | RAM vs disk | `"memory"` \| `"auto"` \| `"disk"` |
| `params` | hyperparams | Grid search | `"manual"` \| `"auto"` |
| `strict` | control | Refuse vs auto-adjust | `False` \| `True` |

Plus `memory_guard: MemoryGuard | None` (mandatory in signatures, default `None` =
use `df._memory_guard`).

**Not all params on all models.** Card declares `params_supported: [window, params]`
etc. Extra params on the function → `TypeError` early.

---

## §2. `window` — history bound

**Applies to:** ETS/AutoETS, Theta/AutoTheta/OTMTheta, SARIMA/ARIMA, VAR,
NaiveForecaster, TemporalHierarchy, TransformedTarget, EnsembleForecaster,
GBDTForecaster (as `lag_window`), pelt (as `max_window`), rolling/expanding/ewm
(in df).

**Semantics:**

| Value | Behavior | Warning if `W_eff < n` |
|---|---|---|
| `"auto"` | `W_eff = min(n, W_max)` where `W_max = mg.get_window_size(unit_bytes=8, target_fraction=0.25)` | Yes (`ApproximationWarning`) |
| `int W` | `W_eff = min(W, n, W_max)` | Yes, if `W_eff < W` (adjusted) or `W_eff < n` (truncated) |
| `None` | `W_eff = min(n, W_max)` (unbounded requested, auto-adjusted) | Yes, if `W_eff < n` |

**`strict=True`:** if adjustment needed (`W_eff < requested` or `W_eff < n`) →
raise `MemoryLimitExceeded`.

**Insufficient (`W_max < min_history(model)`):** raise `MemoryLimitExceeded`
insufficient_no_alt tier. `min_history` per model — e.g., ETS requires `2s+1`
obs, SARIMA requires `p+d+q+(P+D+Q)s + 10`.

**Prediction semantics:**
- `predict(steps)` — state extrapolation, same for all W.
- `predict_in_sample(df)` — **mandatory df** (breaking):
  - `W_eff >= n`: exact `[0, n)`, `approximation=None`.
  - `W_eff < n`: exact `[n-W_eff, n)`, **NaN** `[0, n-W_eff)`,
    `approximation=[window_truncated, ...]`.

**State after refactor:** `O(seasonal_period)` — `y_`/`fitted_` **removed** from
state. Only `last_k` buffer kept (for `predict(steps)`).

---

## §3. `solver` — algorithm variant for state O(d²)

**Applies to:** fit_linear, fit_logistic, fit_pca, fit_lasso (elasticnet), fit_coxph.

| Value | State | Speed | Accuracy |
|---|---|---|---|
| `"closed"` | `O(d²)` (XTX) | fast | exact (MLE) |
| `"sgd"` | `O(d)` (coef only) | n_sweeps slower | convergence (DEC-033 §4.2) |
| `"auto"` (default) | auto-switch | auto | auto |

**`"auto"` resolution:** if `d²·8 <= budget/2` → closed; else sgd. Emit
`ApproximationInfo(type="solver_switch")` if switched.

**`strict=True`:** `"auto"` with switch → refuse. Explicit `"closed"` with tight
budget → refuse. Explicit `"sgd"` never refuses (state minimal).

---

## §4. `output` — n-sized result on disk

**Applies to:** anomaly (ZMS/CUSUM existing; 10 more v0.9.192+), GBDT family (fit
transient on disk, v0.9.187+), GBDTForecaster lag matrix (v0.9.190), pivot/sample
(df, v0.9.191).

| Value | Behavior |
|---|---|
| `"memory"` | return `np.ndarray` in RAM; raise if `n*bytes > budget*0.25` (was `MemoryLimitExceeded`; **now auto-upgrade** if `strict=False`) |
| `"auto"` (default) | `SpillArray`/`SpillBuffer` with `spill_ratio=0.5` (RAM half, disk half) |
| `"disk"` | `SpillArray` with `spill_ratio=0.0` (immediate spill) |

Return type: `np.ndarray` for memory, `AutoArrayResult` for auto/disk. `to_numpy()`
on `AutoArrayResult` raises `MemoryLimitExceeded`.

---

## §5. `params` — hyperparameter search

**Applies to:** Theta/AutoTheta (merged), ETS/AutoETS (merged), SARIMA (auto p,d,q),
ARIMA (auto), GBDTForecaster (auto `lags`).

| Value | Behavior | Time |
|---|---|---|
| `"manual"` (default) | use user-supplied hyperparams | baseline |
| `"auto"` | grid search / AICc / CV-optimal | 5-20× baseline |

**Output:** best `params` stored on result, `params_used: dict` attribute. Not in
`approximation` (no accuracy loss — `auto` finds better hyperparams).

---

## §6. `strict` — refuse on any adjustment

Default `False`:
- Auto-switch through `auto_switch` registry (6 levels, DEC-067).
- Auto-adjust `window` to `W_max`.
- Auto-upgrade `output` to `"auto"` if memory doesn't fit.
- Emit `ApproximationWarning` when real approximation occurs.

`strict=True`:
- Any needed adjustment → `MemoryLimitExceeded` (message: "requested X, would need
  to adjust to Y to fit budget Z; strict=True refuses").
- No warnings (refuse is the signal).
- User handles trade-off explicitly.

**Use cases for `strict=True`:** CI validation (must be exact or fail), reproducible
pipelines (no silent change), benchmarking (`params` must match config).

---

## §7. Auto-switch registry (YAML `auto_switch` field)

```yaml
auto_switch:
  - from: {solver: closed}
    to: {solver: sgd}
    condition: "budget < min_ram(solver=closed)"
    accuracy_impact: "convergence — DEC-033 §4.2"
  - from: {window: null}
    to: {window: auto}
    condition: "budget < n * f64"
    accuracy_impact: "negligible (W >> model_order)"
  - from: {mode: batch_tree}
    to: {mode: round_by_round_spill}
    condition: "budget < n_estimators * 2^max_depth * 200"
    accuracy_impact: "none (exact)"
  - from: {K: K_user, mode: full}
    to: {K: K_user, K_block: auto, mode: blocked}
    condition: "budget < (n_users + n_items) * K_user * 8"
    accuracy_impact: "none (exact), ~K/K_block slower"
  - from: {components_mode: dense}
    to: {components_mode: sparse_csr}
    condition: "budget < K * V * 8 AND nnz < K * V"
    accuracy_impact: "none (exact) if naturally sparse"
  - from: {mode: batch}
    to: {mode: online}
    condition: "budget < n * 8"
    accuracy_impact: "RLS != MLE (adaptive) — see DEC-067 L5"
  - from: {horizon_mode: direct}
    to: {horizon_mode: recursive}
    condition: "budget < horizon * state_per_model"
    accuracy_impact: "algorithm switch, often better for long H"
```

**Resolution order** (v0.9.176 `resolve_params()`):
1. Try `user_params`. If `budget >= min_ram(user_params)`, return (no approximation).
2. For each candidate in `auto_switch` matching `from`, compute `min_ram(to)`.
3. Pick candidate with smallest `min_ram(to)` that fits `budget`.
4. If no candidate fits → `MemoryLimitExceeded` (insufficient_no_alt).

---

## §8. `result.approximation` — schema

```python
@dataclass
class ApproximationInfo:
    type: str                          # see list below
    axis: str                          # "n" | "d" | "K" | "V"
    requested: Any | None              # user's value
    effective: Any                     # what was applied
    reason: str                        # "budget_limit" | "n_smaller_than_requested" | "explicit"
    lost_range: tuple[int, int] | None # for window_truncated
    accuracy_impact: str               # "none" | "negligible" | "estimated: ~3%" | "documented DEC-033 §4.2"

@dataclass
class ModelResult:  # mixin
    approximation: list[ApproximationInfo] | None = None
```

`type` values: `window_adjusted`, `window_truncated`, `solver_switch`,
`streaming_switch`, `subsample`, `output_spilled`, `round_by_round`,
`blocked`, `sparse`, `recursive`.

0 entries = exact (or no approximation).
1 entry = single adjustment.
2 entries = cascaded (e.g., window_adjusted + window_truncated).

---

## §9. Warning classes

```python
class ApproximationWarning(UserWarning):
    """Fires when algorithm auto-adjusted due to budget/data."""
    # Attached info via warnings.warn(ApproximationWarning(msg))

class SlowOperationWarning(UserWarning):
    """Fires when estimated time > 60s (AP22)."""
    # Existing (v0.9.103+)

class ApplySafetyWarning(UserWarning):
    """df.apply(func) detected dangerous pattern (DEC-069)."""
```

Suppression: user can `warnings.filterwarnings("ignore", category=ApproximationWarning)`
or pass `suppress_approximation_warnings=True` on fit (still attaches
`ApproximationInfo` to result).

---

## §10. Migration from existing API

**Backward-compatible:**
- `fit_theta(df, y, seasonal_period=7)` — unchanged. `window="auto"` default → exact
  if budget >= n*8.
- `fit_linear(df, X, y)` — unchanged. `solver="auto"` default → closed if fits.
- `fit_cusum(df, x)` — unchanged. `output="auto"` (already).

**Breaking:**
- `predict_in_sample()` → `predict_in_sample(df)` (mandatory df).
  Migration: `fit(..., save_y=True)` retains legacy behavior.

**New:**
- `strict=True` — opt-in refuse (no existing code relies on it).
- `result.approximation` — new field (default `None`).
- `ApproximationWarning` — new class (existing code unaffected).

---

## §11. Non-goals

- **No `mode="batch"|"streaming"` for models where window covers both.** For
  MLE-forecasters, `window=n` is equivalent to batch; no separate mode needed.
- **No `mode` param for GBDT/linear/logistic.** They have `solver` / iteration
  variants (L1/L2), no "batch vs stream".
- **`mode` reserved for models with explicit streaming alternative** (SARIMA has
  `fit_online_arima` as separate card, not `mode` param).

---

## §12. Implementation roadmap

| Version | Scope |
|---|---|
| v0.9.176 | `MemoryGuard.get_window_size()`, `ApproximationInfo`, `ApproximationWarning`, `resolve_params()`, `result.approximation` field |
| v0.9.177-178 | `min_ram_formula` migration (90+ cards) |
| v0.9.182-186 | TS window refactor (drop y_/fitted_, `predict_in_sample(df)`) |
| v0.9.187-190 | GBDT round-by-round SpillArray (L2) |
| v0.9.192-208 | Anomaly SpillArray (output="auto" wiring) + streamers |

---

**End of PARAMS_DESIGN v1.0.**
