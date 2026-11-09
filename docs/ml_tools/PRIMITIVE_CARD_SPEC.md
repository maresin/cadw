# PRIMITIVE_CARD_SPEC — schema for cadw/ops/* function cards

**Version:** 1.0 (2026-11-10, v0.9.175)
**Status:** normative (schema dataclass implemented v0.9.176).
**Companions:** DEC-067 (work front), PRIMITIVES_BACKLOG (source data),
PRIMITIVES_AND_PRINCIPLES.
**Hierarchy:** Tier 4 (reference). Complements AI_CONSTITUTION §4.0 (R1-R7).

---

## §1. Why separate from ModelCard

Primitives in `cadw/ops/*` are **not** `fit_*` models:
- No `fit/predict` lifecycle.
- No `bounded_state` in the model sense (leaf-layer operates on caller arrays).
- No `verified` vs `draft` in the sense of "fit MLE accuracy" — verification is
  "numerical correctness + guard behaviour".

PrimitiveCard schema captures this orthogonality.

---

## §2. Schema

```python
@dataclass
class PrimitiveCard:
    name: str                            # e.g., "knn_query"
    cadw_function: str                   # "cadw.ops.linalg.knn_query"
    module: str                          # "cadw.ops.linalg"
    layer: str                           # "batch" | "leaf" | "intrinsic"
    status: str                          # "verified" | "draft" | "planned"

    # RAM contract
    ram_formula: str | None              # peak, for verified
    min_ram_formula: str | None          # floor, for verified
    formula_evidence: dict[str, str]     # term -> "file:line -- desc"
    bounded_state: bool | None           # true only if state n-independent

    # Leaf-layer specifics
    streaming_variant: str | None        # e.g., "knn_query_chunks"
    intrinsic_bound: dict | None         # {axis: "n", bound: 1e4, reason: "O(n^2) factor"}

    # Intrinsic specifics
    refuse_above: int | None             # n threshold for hard refuse
    alternative: str | None              # "use apriori for n>1e5"

    # Dependencies / consumers
    consumers: list[str]                 # [card names]
    depends_on: list[str]                # [dotted paths]
    primitive_reuse: list[str]           # [reduce, hash_shuffle, sort, spill_array, chunked_strided]

    # Measurement
    measurement: dict | None             # {cells_passed, matrix, date, ...}
    notes: str | None
```

---

## §3. Layer classification

| Layer | State n-dependence | Caller responsibility | Example |
|---|---|---|---|
| **batch** | O(1) or O(model) given fixed batch | None (primitive is OOC) | `reduce`, `hash_shuffle`, `arm_stats`, `nmf_mu` |
| **leaf** | O(n) or O(n·k) output | Caller batches / chunks | `knn_query`, `pairwise_kernel`, `eigh_topk` |
| **intrinsic** | O(n²) or O(n·d) unavoidable | Caller controls n ≤ bound | `chol_solve` (n≤1e4), `eclat` (n≤1e5) |

**Leaf-layer streaming variant** (optional): if caller-friendly chunked API exists,
declare in `streaming_variant` field.

---

## §4. Example cards

### §4.1. Batch layer (Reduce pattern)

```yaml
- name: arm_stats
  cadw_function: cadw.ops.stats.arm_stats
  module: cadw.ops.stats
  layer: batch
  status: verified
  ram_formula: "k_arms * f64 + 3 * batch * row_size + 512"
  min_ram_formula: "k_arms * f64 + 3 * row_size + 512"
  formula_evidence:
    state: "stats.py:75 -- ArmStatsOp state = (k, 2) + (k,) + (k,)"
    batch: "stats.py:XXX -- check_batch(extra_factor=3.0)"
  bounded_state: true
  primitive_reuse: [reduce]
  consumers: [EpsilonGreedy, UCB1, ThompsonSampling, LinUCB]
  measurement: {cells_passed: "9/9", matrix: {...}}
```

### §4.2. Leaf layer with streaming variant

```yaml
- name: knn_query
  cadw_function: cadw.ops.linalg.knn_query
  module: cadw.ops.linalg
  layer: leaf
  status: verified
  ram_formula: "n_q * k * (f64 + i64) + chunk_n * d * f64 + 3 * batch * row_size"
  min_ram_formula: "n_q * k * (f64 + i64) + 3 * row_size + 512"
  bounded_state: false  # output O(n_q * k) in RAM (leaf)
  streaming_variant: knn_query_chunks  # planned v0.9.209+
  intrinsic_bound: null  # not intrinsic; n_q bounded by chunk_n
  primitive_reuse: [chunked_strided]
  consumers: [LOF, DBSCAN, HDBSCAN, SMOTE, ML-kNN, ItemKNN, KNNImputer]
  notes: "OOC in n_X (chunked); O(n_q * k) output. Use *_chunks for full OOC."
```

### §4.3. Intrinsic (unavoidable O(n²))

```yaml
- name: chol_solve
  cadw_function: cadw.ops.linalg.chol_solve
  module: cadw.ops.linalg
  layer: intrinsic
  status: verified
  ram_formula: "n * n * f64 + 2 * n * f64 + 512"
  min_ram_formula: "n * n * f64 + 2 * n * f64 + 512"
  bounded_state: false
  streaming_variant: null
  intrinsic_bound:
    axis: n
    bound: 10000
    reason: "Cholesky factor O(n^2) = 800 MB at n=1e4; unusable above"
  refuse_above: 10000
  alternative: "Use preconditioned CG or sparse Cholesky for n > 1e4"
  primitive_reuse: []
  consumers: [GaussianProcessRegressor, GaussianProcessForecaster, GP-EI, SMAC, BayesianOpt_GP]
```

---

## §5. Tests

- `test_primitive_cards_schema`: all cards validate against `PrimitiveCard`.
- `test_primitive_card_layer`: `layer` in {batch, leaf, intrinsic}; consistency
  with `bounded_state` (batch => bounded_state true unless declared override).
- `test_primitive_card_formula_evidence`: each term in `ram_formula` has
  `formula_evidence` entry with `file:line`.
- `test_primitive_card_intrinsic_bound`: `intrinsic_bound` set <=> `layer=intrinsic`.

---

## §6. File

All cards in `docs/ml_tools/PRIMITIVES.yaml` (single file, ~50-70 cards for
cadw/ops/* functions). Managed by hand + verified by schema test.

**Not** in ml_tools/A_*.yaml (those are ModelCards). **Not** in
DF_operations.yaml (those are DataFrameOpCards).

---

## §7. Non-goals

- **No `fit/predict` lifecycle.** Primitives are callable functions, not models.
- **No `status: verified` in the sense of "MLE fit verified".** Verification = "numerics
  match reference + guard behaviour correct".
- **No `depends_on` for primitives.** Primitives are leaf-level; dependencies are
  the primitives themselves.

---

**End of PRIMITIVE_CARD_SPEC v1.0.**
