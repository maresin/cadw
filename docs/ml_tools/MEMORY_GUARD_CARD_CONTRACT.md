# MemoryGuard Card Contract

**Version:** 1.0 (v0.9.234, DEC-082)
**Status:** NORMATIVE for new cards; existing cards audited in P2b

Card author / model implementer contract for `ram_formula`.

---

## §1. Formula variables (what `ram_formula` may reference)

### Auto-injected from DataChars (universal)
- `n` — rows (`dc.n_rows`)
- `d` — features (`len(dc.schema)`)
- `row_size` — bytes/row (`dc.row_size_bytes`)
- `batch` — batch size (free variable for `solve`)
- `k`, `K`, `k_est` — cardinality (`dc.k_est`, if present)

### Auto-injected constants (`docs/ml_tools/memory_constants.yaml`)
- `f64`, `f32`, `i8`, `i16`, `i32`, `i64`, `bool` — type sizes
- `dict_entry_bytes`, `py_str_overhead_bytes`, `list_ptr_bytes`

### Builtins (eval)
- `max`, `min`, `abs` — safe arithmetic

### Hyperparams — MUST be in `dc.extras` (or `**extra` to `plan_op`)
Examples: `n_estimators`, `max_depth`, `n_classes`, `C`, `n_arms`,
`n_trees`, `subsample_size`, `num_kernels`, `n_centers`,
`branching_factor`, `n_factors`, `n_epochs`, `hidden_size`,
`latent_size`, `n_memory`, `max_samples`, `n_layers`, `r`, `c`,
`n_biclusters`, `k_sv`, `n_neighbors`, `n_clusters`, `n_components`,
`window_size`, `n_cats_cols`, `total_categories`, `n_oversamples`,
`w`, `V`, `avg_word_len`, `max_query_size`, `seasonal_period`,
`period`, `p`, `q`, `s`, `W_eff`, `levels`, `n_transformers`,
`n_members`, `n_horizons`

### Data-dependent params — MUST be in `dc.extras` (from pre-scan)
Examples: `n_events`, `U`, `I`, `n_users`, `n_items`, `n_nodes`, `S`,
`n_samples`, `n_content`

## §2. Reserved names (do NOT use as hyperparams/data params)
`n`, `d`, `row_size`, `batch`, `k`, `K`, `k_est`,
`f64`, `f32`, `i8`, `i16`, `i32`, `i64`, `bool`,
`dict_entry_bytes`, `py_str_overhead_bytes`, `list_ptr_bytes`,
`max`, `min`, `abs`.

**Known collision (P2b):** SARIMA/VAR use `d` for differencing order.
`d` is reserved for feature count. Fix: rename in formula to `d_order`
(P2b audit).

## §3. Formula constraints (checked by YamlFormulaModel)

1. **Evaluable**: all names resolvable in `env = dc + constants +
   dc.extras + batch`.
2. **Arithmetic only**: numeric literals, `+ - * / ** %`, `max/min/abs`.
3. **Linear in batch** (for `solve` → BatchParams):
   slope `Δestimate/Δbatch` must be constant (within 1% at samples 1, 100,
   1000, 10000). Non-linear (e.g. `batch*batch`) → `solve` returns None.
4. **State-only** (no `batch` term): `solve` returns None (honest).
5. **Invariant**: `estimate(solve(budget)) <= budget * safety` (asserted
   inside solve).

## §4. Code contract (fit_X writers)

If your `fit_X` computes a formula-based batch size:

```python
mg = memory_guard or df._memory_guard
dc = DataChars(
    schema=df._dataset.schema,
    n_rows=df.count(),
    row_size_bytes=mg.estimate_row_size_from_schema(df._dataset.schema),
    k_est=<from sample>,
    extras={
        # hyperparams — from step config
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        # data params — from first pass (if needed)
        "n_events": first_pass_count(df),
    },
)
params = mg.plan_op("fit_X", dc)   # or pass **extra to plan_op
if params is None:
    # fallback heuristic + LegacyFormulaWarning
    batch_size = mg.get_batch_rows(expansion_factor=ef)
else:
    batch_size = params.batch_size
    mg.check_state(params.state_bytes, name="fit_X.state")
```

## §5. Card schema (P2b will formalize)

Optional field to declare params explicitly (machine-checkable):

```yaml
formula_params:
  hyperparams: [n_estimators, max_depth]
  data_scan: [n_events]   # from first pass
```

R-lint test (P2b): every name in `ram_formula` (except reserved) appears
in `formula_params`.

## §6. Audit roadmap (P2b, v0.9.235+)

- 95 cards: (parseable + params declared + fit_X provides them +
  linearity honest)
- Rename collisions (`d` in SARIMA/VAR → `d_order`)
- Add transient terms to state-only formulas where honest (GBDT,
  Agglomerative — n-sized arrays)
- Verification matrix: n × d × hyperparams × data_params
- Formula accuracy per STABILITY_PROTOCOL §4.1.1 (component-wise
  acceptable when RSS unmeasurable)

## §7. P1/P2a acknowledged limits (v0.9.232/233)

- **P1 `Pipeline.plan`:** wire A (peak preflight). GBDT `ram_formula`
  state-only → `plan` gives state floor, transient not modelled. Spec
  `test_plan_covers_transient` xfail (P2b fix).
- **P1/P2a `estimate`:** for formulas with hyperparams, `estimate` at
  batch=1 works only if `dc.extras` populated. GBDT/wire falls back to
  `step.estimate_state_bytes` (defensive, honest).
