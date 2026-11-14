# MemoryGuard Architecture

**Version:** 1.2 (approved for P1, v0.9.232)
**Status:** ACTIVE
**Validated in:** /tmp/memory_guard_proto_v2.py (11/11 equivalence)
**Companions:** AI_CONSTITUTION §2, DECISIONS §9.2, DEC-044, DEC-081

---

## §1. Goal

Один механизм RAM-estimate + params-sizing для **всех** операций:
- dataframe steps (dropna, astype, fillna, sort, groupby, join)
- ML fits (fit_coxph, fit_als, fit_gbdt, ...)
- pipeline chain
- standalone (`df.fit_X()`)

Текущее состояние (probe v0.9.231): **3 параллельных пути**
(`OpMemoryModel` / YAML formula / heuristic `ef`), не связаны.
Один budget → до 4× разницы в batch_size.

## §2. Design

**Один класс, N экземпляров.** Formula — единственный source of truth per op.

```python
class YamlFormulaModel(OpMemoryModel):
    def __init__(self, op_name, card): ...
    def estimate(self, params, dc) -> int
    def solve(self, available, dc, *, safety) -> Params | None
```

**Card** (из YAML):
```yaml
ram_formula: "24 * d + batch * row_size"
bounded_state_dims: {d: 1}
n_scalability: constant | linear | bounded_by_window
composition: {...}                    # OPTIONAL (P2+)
```

**Semantics (validated):**
| Condition | Result |
|---|---|
| `state > budget` | `MemoryLimitExceeded` |
| `batch < 1` | `None` |
| unknown op | `None` |
| formula batch-independent | `None` |
| invariant `estimate <= budget*safety` | asserted |

**API:** `plan_op(op, dc, **extra)` + `plan_chain(ops, dc)`.
Legacy (`get_batch_rows`, `solve_batch`, `plan`, `plan_pipeline`) — unchanged in P1.

## §3. Composition — 5 types (opt-in, default flat)

**Default `flat`**: `ram_formula` as-is. Backward-compat, 123 existing cards без изменений.

Opt-in (P2b, per-model audit):
```yaml
composition:
  mode: flat | sum | max | sequential | iteration
  components:
    - primitive: external_sort
      mode: max
      n_calls: 1
  remaining_formula: "..."
```

| mode | Formula |
|---|---|
| `flat` | `ram_formula` as-is |
| `sum` | `Σ(primitive_peak × n_calls) + remaining` (conservative) |
| `max` | `max(primitive_peak) + remaining` (tight, sequential) |
| `sequential` | explicit chain, state carries |
| `iteration` | per-iter `max`, not sum |

## §4. Scope (inventory v0.9.231)

| Phase | Scope | Sessions |
|---|---|---|
| **P1** | `YamlFormulaModel` + auto-register 123 verified. Cards не меняются. | 1-2 |
| **P2a** | Wire A: `Pipeline.plan` → `plan_chain(peak)`. Preflight honest. | 0.5 |
| **P2b** | Audit 123 cards: code + formula. Composition где нужно. | 3-4 |
| **P3** | Streaming (spec готов: `test_pipeline_streaming_spec.py`). | по spec |

## §5. Formulas — flat, state via bounded_state_dims

`ram_formula` — плоское выражение. State vs transient не разделяется в формуле;
`bounded_state_dims` — annotation, какие оси state.

`primitive_reuse` — информативное поле (AI-навигация). Не обрабатывается
рантаймом. Не делим, не мигрируем.

## §6. Что не трогаем

- **P1** не трогает: `get_batch_rows`, `solve_batch`, `plan`, existing
  `plan_pipeline` (v0.9.119), `Pipeline.plan`, `SortMemoryModel`, 123 YAML cards.
- **P2a** не трогает batch sizes, не передаёт params в fit_X.
- **Wire B** (params propagation в fit_X) — не планируется, только по user pain.
- `primitive_reuse` миграция — не делается.

## §7. Tests

- **P1:** prototype-equivalence (4 narrow cards), invariant assert, edge cases
  (`test_yaml_formula_model.py`: 12 tests, mirror prototype v2)
- **P2a:** `test_plan_covers_transient` (spec xfail → pass)
- **P2b:** per-card audit (composition vs flat ratio [0.7, 1.4])

## §8. Anti-scope

- ❌ Individual memory classes per operation (v1 prototype — deprecated)
- ❌ Composite transformer с собственными estimate/solve
- ❌ Auto-inference formula из measurement (AP23: formula from code only)
- ❌ Массовая миграция карточек в "flat=correct" assertion (P1 = "как сейчас",
  не "мы проверили")
- ❌ Расширение API больше 2 методов

## §9. Approval

- [x] §1-§2 (design) — approved
- [x] §3 (composition 5 types, default flat) — approved
- [x] §4 (P1/P2a/P2b/P3) — approved
- [x] §5-§6 (что не трогаем) — approved
- [x] §7-§8 (tests, anti-scope) — approved
