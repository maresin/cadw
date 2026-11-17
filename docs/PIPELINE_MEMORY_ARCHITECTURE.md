# Pipeline Memory Architecture (v0.9.240)

**Status:** design doc for v0.9.240 (P-A/P-B/P-C done in v0.9.239; A1/A2/A4/A5 in v0.9.240-A; P-Y in v0.9.240-B2; feature_effect deferred to v0.9.241).

**Audience:** authors of step cards / models; anyone debugging `PipelineMemoryError`.

**Scope.** Как Pipeline считает память (`plan`) и как исполняет (`fit`/`predict`/`fit_transform`); какие инварианты держим; какие limitations осознанны. **Не** документируем: `MemoryGuard` API, `YamlFormulaModel` internals, формулы примитивов (см. `docs/ml_tools/PRIMITIVES.yaml`).

---

## §0. Инварианты (TL;DR)

1. **`Pipeline(memory_guard=mg)` — authority.** На время `step.fit`/`transform`/`predict` `df._memory_guard` переопределяется на `mg` (try/finally restore).
2. **`plan()` — preflight по O(1)-метаданным.** Без скана данных. Оценивает `peak_bytes` (max по stages) и `state_bytes` (sum по steps).
3. **`plan()` не эквивалентен execution'у.** Execution идёт `stage_0 → stage_1 → …`, с реальными `n_rows`/`schema` **после** каждого barrier. `plan` использует **входной** `dc` и **не** обновляет его после барьеров (кроме известных моделей вроде PCA — deferred, §3.3).
4. **Multi-phase модель** внутри — flat formula `max(phase_1_peak, phase_2_peak)` (upper bound). Execution сама решает, где материализовать phase boundaries (§4).
5. **`PipelineMemoryError` содержит `step_idx`, `stage_idx`, `describe`** — actionable, какой именно stage и kind (narrow / barrier / total) не влезает.

---

## §1. Guard authority (P-C)

### §1.1. Eager path (`fit` / `predict`)

`Pipeline.fit` и `Pipeline.predict` в loop'е по steps:

```python
_orig_mg = getattr(df_in, "_memory_guard", None)
try:
    if _orig_mg is not None and hasattr(df_in, "_memory_guard"):
        df_in._memory_guard = self._memory_guard
    step.fit(df_in, ...) / step.transform(df_in) / step.predict(df_in)
finally:
    if _orig_mg is not None and hasattr(df_in, "_memory_guard"):
        df_in._memory_guard = _orig_mg
```

Это покрывает eager path (v0.9.239). Доказательство: `test_pipeline_guard_authority` — `Pipeline(mg=10MB)`, `df(guard=500MB)` → `derive_batch_size` log `available=10485760`.

### §1.2. Streaming path (`fit_transform` → `_execute_stream`)

`_execute_stream` создаёт `_LazyView(factory, schema, df)`, который наследует `_memory_guard = template_df._memory_guard` (см. `_LazyView.__init__`). Чтобы override не потерялся, `fit_transform` переопределяет `df._memory_guard` **до** `_execute_stream` и восстанавливает **после** (v0.9.240-A1):

```python
_orig_mg = getattr(df, "_memory_guard", None)
try:
    if _orig_mg is not None and hasattr(df, "_memory_guard"):
        df._memory_guard = self._memory_guard
    result = self._execute_stream(df, target=target)
finally:
    if _orig_mg is not None and hasattr(df, "_memory_guard"):
        df._memory_guard = _orig_mg
```

Это покрывает: `fit_iter`/`transform_batch` (narrow fused chain) **и** `fit` на `_LazyView` (barrier + predictor). Доказательство: `test_streaming_guard_authority` — `available=10485760` в streaming path + `_execute_stream: completed (streaming)` (не eager fallback).

### §1.3. Known gap

Custom step, читающий `mg` из глобального контекста (не из `df`/`template_df`) — вне нашей власти. Документируем как ответственность автора step: **читайте `mg` из `df._memory_guard`**.

Direct `df.fit_X()` (вне Pipeline) — **не** трогаем; out-of-pipeline users сохраняют `df`-контекст.

---

## §2. Segmented plan (P-A + P-B)

### §2.1. Stages

`_split_stages()` бьёт `self._steps` на `(narrow: list, barrier: step|None, barrier_name)` по `is_barrier` / `_is_predictor`. Последний stage может быть без barrier (transform-only tail).

### §2.2. Batch per stage

- **eager mode** (`plan(df, mode="eager")`) — `narrow_peak = sum(peaks)` (narrow materialize individually; matches `Pipeline.fit`).
- **streaming mode** (`plan(df, mode="streaming")`) — `narrow_peak = max(peaks)` (narrow fused per-batch; matches `_execute_stream`).

### §2.3. Peak per step (`_step_peak_estimate`)

Resolution order:
1. `step.cadw_function` + `get_registered_model(cf)`:
   - **`peak_source=formula_batch`** (P-A): `params = solve(budget, dc, safety=0.7, **step._kwargs)`; `effective = min(n_rows, params.batch_size)`; `return _eval(dc, batch=effective, **step._kwargs)`.
   - **`peak_source=formula_state`**: state-only formula (no batch term) — `_eval(dc, batch=1, **step._kwargs)`.
2. **`peak_source=fallback`**: `step.estimate_state_bytes(dc)` — no formula / `KeyError` (missing hyperparams).

`peak_source` пишется в `_log.debug` (caplog). `test_plan_real_peak` **ассертит** `peak_source=formula_batch` — защита от fallback-masking (тест не должен проходить «случайно»).

### §2.4. Total peak / state

```
stage_total_k = cum_state_prior_k + stage_peak_k
total_peak    = max_k(stage_total_k)      # sequential execution
total_state   = sum(state_k)              # states held for predict
```

`step_plans[i].estimate_bytes = peak_estimates[i]` (per step, для справки). `PipelinePlan.peak_bytes = total_peak`. `Pipeline.total_state_bytes()` до fit'а = `sum(_plan_state_estimates)` (A2).

### §2.5. Multi-stage error

`PipelineMemoryError` (v0.9.239) несёт `stage_idx` + `kind ∈ {narrow, barrier, total}` + `describe` с `peak`/`state`/`budget`. `test_plan_segmented` ассертит `narrow_peak=` / `barrier_peak=` / `stage_total=` в логе.

---

## §3. `dc` (DataChars) evolution между stages

### §3.1. Сейчас (v0.9.240)

`dc = _dc_from_df(df)` — один раз в `plan()`. Поля: `schema`, `n_rows = df.count()`, `row_size_bytes = mg.estimate_row_size_from_schema(schema)`, `k_est=None` (если нет sample), `extras={}`.

`dc` **не** обновляется между stages. Для `[Standardize, PCAProjector, LinearRegressor]` plan для stage 1 использует **исходный** `dc` (d=4: x1,x2,x3,y), а не post-PCA (d=3: pc0,pc1,y). Разница в `_eval` мала при малых `d` (32256 vs 32144), но для OHE с `K=1000` будет значительной.

### §3.2. Что меняется в execution (факты)

| Операция | `n_rows` | `d` | `cardinality` | `row_size` |
|---|---|---|---|---|
| **OHE** | = | `d-1+K` | +K | = |
| **Target/frequency encoding** | = | = (replace) | = | = |
| **Feature hashing** | = | `d+n_hash` | = | = |
| **Polynomial** | = | `O(d^k)` | = | = |
| **Binning** | = | = | = | = |
| **Standardize** | = | = | = | = |
| **dropna** | ↓ | = | = | = |
| **sample** | ↓ | = | = | = |
| **groupby.agg** | ↓ до `n_groups` | зависит | ↓ | = |
| **join** | ↑/↓ | `+new` | = | = |

### §3.3. Plan's limitation

Plan **не** обновляет `dc` после barrier для неизвестных типов. Для известных: PCA — `d_new = n_components` (deferred, не реализовано); LinearRegressor — sink, не нужен.

**Решение (v0.9.241, feature_effect):** в card шага — `feature_effect: {d_delta, cardinality_delta, n_rows_effect, row_size_effect}`. Planner применяет к `dc` после barrier. Без этого — OHE/hash/dropna дадут неверный `d`/`n_rows` в plan (execution корректен — execution видит реальный `out_df`).

### §3.4. `row_size` conservative

Для строк — 32 B (без скана). Осознанно: 1 scan = O(N); отложено до feature_effect или opt-in sample. Числовые — точные (`bit_width // 8`).

---

## §4. Multi-phase models (P-Y)

### §4.1. Что такое multi-phase

Модель, которая **внутри** материализует intermediate:
- **GBDT**: binning (codes) → boosting rounds (grad/hess/tree).
- **LDA**: vocab build → EM (`n·K` doc-topic).
- **CoxPH**: sort by time → Newton iterations.

Phase 2 зависит от результата phase 1 (`n_bins`, `V`, `n_events` известны только после phase 1).

### §4.2. Plan не может итеративно перепланировать

Phase 1 требует **реального** fit (минуты для GBDT). Plan — preflight, **до** execution. Поэтому card formula = `max(phase_1_peak, phase_2_peak)`.

### §4.3. Это upper bound

Честно **только** если автор формулы честно записал max по коду. **P2b audit** — ответственность автора card.

`MultiPhaseModel` (synthetic, v0.9.240-B2) тестирует **принцип**: `plan_peak == max(phase_1, phase_2)` (не sum, не 367MB), execution не OOM, лог обеих фаз. Real GBDT/LDA — **P2b**.

### §4.4. Что НЕ делаем

- **Expand модели в pipeline stages** (`[gbdt_bin, gbdt_round_1, …]`) — invasive, ломает `Model` API, CompositeStep не подходит (predictor обязан быть sink).
- **Phase-by-phase plan** — невозможно без execution.

---

## §5. Composition from primitives (P2b, deferred)

### §5.1. Зачем

Сейчас card модели — **одна** формула (flat). Для multi-phase модели автор **вручную** пишет `max(phase_1, phase_2)`, дублируя знания из primitive'ов. Composition автоматизирует: `model.depends_on = [prim_1, prim_2, ...]`, `ram_formula = compose(...)`.

### §5.2. `PRIMITIVES.yaml` (существующий seed)

`docs/ml_tools/PRIMITIVES.yaml` — 5 seed cards: `name`, `cadw_function`, `module`, `layer` (batch/leaf/intrinsic), `ram_formula`, `min_ram_formula`, `formula_evidence`, `consumers`. Loader — `cadw.core.cards.load_primitive_cards`.

### §5.3. Composition modes

- `max` — независимые фазы (typical multi-phase).
- `sum` — накапливающиеся state (sequential).

### §5.4. R-lint

Формула модели должна ссылаться на primitives (или разрешённые символы). Автор карточки отвечает за честность max — lint помогает, но не заменяет audit.

### §5.5. Статус

**Не реализовано** в v0.9.240. Placeholder для v0.9.241+. Модели регистрируются в Python (`register_model(instance)`) — не YAML. См. `cadw/core/models_*.py`.

---

## §6. Limitations (явный список v0.9.240)

| # | Limitation | Влияние | План |
|---|---|---|---|
| L1 | `dc` не обновляется после barrier (без feature_effect) | OHE/hash/dropna — неверный `d`/`n_rows` в plan (execution корректен) | v0.9.241 (feature_effect) |
| L2 | multi-phase flat formula = upper bound | over-estimate при нечестном max | P2b audit |
| L3 | `row_size` conservative для строк (32 B) | under-estimate при длинных строках | feature_effect.row_size_effect или opt-in sample |
| L4 | Streaming guard: custom step, читающий mg из глобального контекста | вне нашей власти | документировано (§1.3) |
| L5 | `_LazyView.count()` = template count | при `FrameMethod("dropna")` сломает `derive_batch_size` | A3 → feature_effect.n_rows_effect (v0.9.241) |
| L6 | `Pipeline.total_state_bytes()` до `plan()` = 0 | информационно | документировано (A2) |
| L7 | ~~`ColumnTransformer` + narrow steps broken on `df[cols]`~~ **FIXED v0.9.241**: `_mvp/Standardize.transform` read `df._dataset.schema` instead of `df._columns` view; `df[cols]` is correct by design (shared dataset). Fix: build output schema from `df._columns` + `_dataset.schema.field(c)`. | resolved | done |
| L8 | composite formula from primitives | не реализовано | P2b |
| L9 | MultiPhaseModel: estimate_state_bytes returns max(phase_1, phase_2) for plan peak (P-Y), but actual state_bytes() = sum → plan's cum_state for downstream stages under-estimated | synthetic MultiPhaseModel collapses peak and state; real models separate via cadw_function (peak) + estimate_state_bytes (persistent state) | P2b (real composite) |

---

## §7. Tests (v0.9.240)

| Test | Что проверяет |
|---|---|
| `test_pipeline_guard_authority` | P-C eager: `available=10485760` в `derive_batch_size` |
| `test_streaming_guard_authority` | P-C streaming (A1): то же + `_execute_stream: completed` |
| `test_plan_real_peak` | P-A + A2: `plan_peak < 10MB`, `peak_source=formula_batch`, `total_state_b=592` |
| `test_plan_segmented` | P-B: `>=2 stage logs`, все поля `narrow_peak`/`barrier_peak`/`stage_total` |
| `test_execute_stream_preflight` | `plan[streaming]` лог до `_execute_stream` |
| `test_pca_serialization_roundtrip` | PCA state survives `to_json`/`from_json` |
| `test_batch_selection_splits_multi_batch` | `n=200k, budget=10MB` → `effective_batch=76457`, 3 batches (slow) |
| `test_composite_step_transform_chain` | A4: CompositeStep == flat chain (rtol=1e-12) |
| `test_composite_step_barrier_flag` | A4: `is_barrier` True iff any child barrier |
| `test_smoke_a5.py::*` | A5: KFold split, ColumnTransformer validation, register_step_type |
| `test_multi_phase_model` | P-Y (B2): `plan_peak = max(phase_1, phase_2)`, execution OK |
