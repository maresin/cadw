# DF_OPERATIONS_SPEC — schema for CadwDataFrame method cards

**Version:** 1.0 (2026-11-10, v0.9.175)
**Status:** normative (schema dataclass implemented v0.9.176).
**Companions:** DEC-067, DEC-069 (df.apply 3-variant), DF_operations.yaml.
**Hierarchy:** Tier 4 (reference). Complements AI_CONSTITUTION §4.0 (R1-R7).

---

## §1. Scope

CadwDataFrame methods that allocate or transform n-sized data. Not all methods
have cards — only those with potential RAM impact.

**Example methods (subset):**
- `cumsum`, `cumprod`, `cummax`, `cummin` — cumulative (already OOC, P19)
- `rolling`, `expanding`, `ewm` — windowed (already OOC)
- `diff`, `shift`, `pct_change` — shift-like (already OOC)
- `ffill`, `fillna`, `dropna` — fill/filter
- `pivot`, `melt` — reshape
- `sample` — sampling
- `apply` — arbitrary user function (3 variants, DEC-069)
- `to_pandas`, `to_arrow_table` — explicit escape hatches

**Not covered:** `groupby`, `join`, `sort_values`, `drop_duplicates`, `quantile`,
`value_counts` — these go through primitives (HashShuffle / Sort / Reduce), see
PRIMITIVE_CARD_SPEC for the underlying mechanism.

---

## §2. Schema

```python
@dataclass
class DataFrameOpCard:
    name: str                            # "cumsum"
    cadw_method: str                     # "CadwDataFrame.cumsum"
    category: str                        # "cumulative" | "rolling" | "shift" |
                                         # "fill" | "reshape" | "sample" | "apply" |
                                         # "export"
    status: str                          # "verified" | "draft" | "escape_hatch"
    ram_formula: str | None
    min_ram_formula: str | None
    bounded_state: bool | None
    memory_guard_site: str | None        # e.g., "cumsum.output", or "none"
    variants: dict | None                # for apply: {default, checked, safe}
    escape_hatch: bool = False           # True for to_pandas/apply default
    depends_on: list[str]                # primitive paths
    measurement: dict | None
    notes: str | None
```

---

## §3. Categories and their behavior

| Category | Examples | OOC status (v0.9.174) |
|---|---|---|
| **cumulative** | cumsum, cumprod, cummax, cummin | already OOC (external_sort + SpillDir, P19). Guard: `check_batch` on batch input; output on disk incrementally. |
| **rolling** | rolling, expanding, ewm | already OOC (batch-iter -> `Dataset.from_batches`). Guard: `check_batch(extra_factor=3.5)` on input. |
| **shift** | diff, shift, pct_change, ffill | already OOC (Arrow tail buffer). Guard: `check_batch(extra_factor=3.5)`. |
| **fill** | fillna, dropna | Requires probe (P25/P26): fillna = n·row_size output; dropna = filter (n variable). |
| **reshape** | pivot, melt | Requires probe (P27): pivot may need O(n·K) RAM. Auto-spill or guard. |
| **sample** | sample(frac, n) | `frac` -> O(n·frac) result; `n` -> O(n) reservoir (bounded by n_samples). Guard on output. |
| **apply** | df.apply(func) | 3 variants (DEC-069). Default: escape_hatch + AST warning. |
| **export** | to_pandas, to_arrow_table | Explicit escape hatches. Documented user contract. |

---

## §4. Example cards

### §4.1. Already OOC (cumulative)

```yaml
- name: cumsum
  cadw_method: "CadwDataFrame.cumsum"
  category: cumulative
  status: verified
  ram_formula: "3.5 * batch * row_size + 512"
  min_ram_formula: "3.5 * row_size + 512"
  bounded_state: true
  memory_guard_site: "cumsum.sorted_input"
  depends_on: [cadw.ops.sort.external_sort]
  measurement: {cells_passed: "9/9", matrix: {...}}
  notes: "Output written to SpillDir incrementally; RAM O(batch). Eager-to-disk via Dataset.from_batches."
```

### §4.2. Escape hatch (apply)

```yaml
- name: apply
  cadw_method: "CadwDataFrame.apply"
  category: apply
  status: escape_hatch
  ram_formula: null
  min_ram_formula: null
  bounded_state: false
  memory_guard_site: null
  variants:
    default:
      bounded_state: false
      escape_hatch: true
      ast_warning: "ApplySafetyWarning on {append, add, extend, Global, np.concatenate}"
      ooc_guarantee: "none"
    checked:
      bounded_state: true
      runtime_check: "tracemalloc + RSS sampler every 1000 rows; raise if growth > 2x baseline"
      overhead: "3-7%"
      ooc_guarantee: "for detectable accumulation"
    safe:
      bounded_state: true
      signature: "func(batch: pa.RecordBatch, mg: MemoryGuard) -> pa.RecordBatch"
      ooc_guarantee: "user-responsible (mg.check_state contract)"
  notes: "3-variant design per DEC-069."
```

### §4.3. Export (explicit escape)

```yaml
- name: to_pandas
  cadw_method: "CadwDataFrame.to_pandas"
  category: export
  status: escape_hatch
  bounded_state: false
  escape_hatch: true
  notes: "Explicit user contract: loads full dataset into pandas DataFrame (RAM O(n*row_size)). Documented. No guard."
```

### §4.4. Requires probe (fillna, pivot, sample)

```yaml
- name: pivot
  cadw_method: "CadwDataFrame.pivot"
  category: reshape
  status: draft
  draft_reason: "OOC status requires probe (P27). Possibly O(n*K) or SpillBuffer path."
  ram_formula: null
  min_ram_formula: null
  bounded_state: null
  memory_guard_site: null
  notes: "Probe P27 (v0.9.175) determines if current impl is OOC; then design auto-spill or guard."
```

---

## §5. Tests

- `test_df_op_cards_schema`: validate against `DataFrameOpCard`.
- `test_df_op_categories`: `category` in known set.
- `test_df_op_escape_hatch`: `status: escape_hatch` implies `escape_hatch: true`.
- `test_df_op_verified_has_formula`: `status: verified` implies `ram_formula` present.

---

## §6. File

All cards in `docs/ml_tools/DF_operations.yaml` (single file, ~11 cards for
listed methods). Managed by hand + verified by schema test.

---

## §7. Non-goals

- **No coverage of every method.** Only methods that allocate or transform n-sized
  data. `df.shape`, `df.columns`, `df.head(10)` — no card (O(1) or tiny).
- **No `verified` for `apply`.** Escape hatch, not verifiable.
- **No verification of user-provided func.** User responsible (or use
  `apply_checked`/`apply_safe`).

---

**End of DF_OPERATIONS_SPEC v1.0.**
