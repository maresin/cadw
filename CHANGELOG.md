## [0.9.0] — 2026-10-20

### Добавлено (v0.9.0c — Jupyter integration)

- **`CadwDataFrame._repr_html_`** — sample 20 rows + schema + shape
  (upper bound) + filter note. Bounded: `.to_pandas().to_html()` на
  20-row `head()` только, не full df. AST guard whitelist с rationale.
- **`CadwDataFrame.plot` accessor** — matplotlib default через
  `df.plot.hist/line/box/scatter/hist2d`; explicit backends:
  `df.plot.plotly.*`, `df.plot.seaborn.*`. Все методы делегируют
  `cadw.viz.*` adapters (aggregate-first discipline preserved).
- **`Pipeline._repr_html_`** — HTML-таблица steps с per-step
  `state_bytes`, fitted marker, total state.
- **`cadw/plot_accessor.py`** — новый модуль, thin wrapper over
  `cadw.viz.matplotlib_adapter/plotly_adapter/seaborn_adapter`.

### Ограничения (v0.9.0c, §11.5)

- `df.info()` / `df.dtypes` / `df.describe()` — v0.10.
- `Pipeline` DAG визуализация — v0.9.1 (после DAG support в Pipeline).
- `df.plot` — только явные методы (`df.plot.hist(...)`), не
  `df.plot(kind='hist', ...)` (pandas-style). Method-style selected
  as primary; kind-based — backlog, если запрос.

### Тесты (v0.9.0c)

- +16 fast: 8 `test_repr_html.py`, 8 `test_plot_accessor.py`.
- AST guard whitelist: `_repr_html_` (bounded 20-row to_pandas).

- fast: **~1278** (+16). slow: **9** (unchanged).

### Добавлено (v0.9.0b — Pipeline skeleton)

Главный архитектурный шаг v0.9.0: chain of Transformers с plan-first
feasibility check. Превращает CADW из «набора primitives» в
out-of-core framework.

- **`cadw/pipeline.py`** — новый модуль:
  - **`Pipeline`** — sequence of Transformers. Model (if present)
    всегда LAST.
    - `add_step(name, step)` — chainable.
    - `plan(df) -> PipelinePlan` — pre-fit feasibility: sum(state_bytes)
      ≤ state_limit. Raises `PipelineMemoryError` if not feasible
      **до** любого materialize. Пользователь может вызвать явно для
      early check.
    - `fit(df, target=None)` — fit всех steps последовательно, plan-first.
    - `transform(df)` — применяет pure-transform steps; **останавливается
      перед Predictor** (если он есть).
    - `predict(df)` — fit chain + last Predictor. Требует последний step
      быть Predictor.
    - `fit_transform(df, target=None)` — fit + return features (before model).
    - `total_state_bytes()` — sum fitted state, интроспекция.
  - **`Transformer` / `Predictor` Protocol** (typing only) — контракт:
    `fit(df, target=None)`, `transform(df)`, `estimate_state_bytes(dc)`,
    `state_bytes()`; Predictor adds `predict(df)`.

- **Wrappers (MVP, для тестов)**:
  - **`CastStep(mapping, safe=True)`** — wrap `df.astype()`. Stateless.
  - **`DropnaStep(how, subset)`** — wrap `df.dropna()`. Stateless.
  - **`GBDTStep(X_cols, y_col, **kwargs)`** — wrap `fit_gbdt_regressor`.
    `estimate_state_bytes` = n_estimators × 2^(max_depth+1) × 48 + 1024
    (conservative per-node).

### Design decisions (locked-in, v0.9.0)

- **Sequence-first.** DAG (via `inputs=[...]`) — v0.9.1.
- **Model LAST.** `transform()` stops before Predictor; `predict()`
  requires it.
- **Plan-first fit.** `Pipeline.fit` calls `plan(df)` before any work;
  `PipelineMemoryError` if not feasible.
- **2-pass fit/transform.** State frozen after fit; transform re-applies
  pure steps. No fused execution (state-after-fit стабилен — экономии нет).
- **In-memory state.** Bounded (K × #features). Serialize to JSON — v0.9.1.
- **Per-batch transform guard — per-step.** Each step uses `MemoryGuard`
  internally; Pipeline не reimplement'ит.

### Ограничения (v0.9.0b, §11.5)

- `Pipeline.plan` — fit feasibility only (`total_state ≤ state_limit`).
  `spill_boundaries` always empty (deferred v0.9.1). Batch propagation
  per step handled by step's own MemoryGuard (not re-planned).
- `Pipeline.to_json` / `from_json` — v0.9.1.
- `ColumnTransformer` — v0.9.1.
- `TimeSeriesFeatures` composite — v0.9.2.
- `Pipeline` не валидирует, что Predictor is last при `add_step`
  (проверяется при `fit`). Predictor not-last на `fit` → RuntimeError.

### Тесты (v0.9.0b)

- +15 fast (`test_pipeline.py`): construction, wrappers, compose,
  fit/transform/predict contract, plan (fits, does_not_fit, with GBDT),
  state_bytes tracking, small GBDT fit+predict.
- +1 slow (`test_pipeline_e2e.py`): 20K rows, timestamp → dt features
  → categorical encoding → GBDT; R² > 0.3.

- fast: **~1257** (+15). slow: **9** (+1).

### Добавлено (v0.9.0a — accessors + type ops)

Первый шаг v0.9.0 «out-of-core framework»: подготовка данных
(datetime features, string cleaning, type casting, null handling).
Arrow-native, bounded per-batch, zero pandas.

**Accessors:**
- **`df['ts'].dt.{year,month,day,hour,minute,second,weekday,day_of_year,quarter}`** —
  Arrow `pc.year/month/day/hour/minute/second/day_of_week/day_of_year/quarter`.
  Возвращает CadwDataFrame с `{col}_{feature}` (original columns preserved).
- **`df['s'].str.{len,lower,upper}`** — `pc.utf8_length/utf8_lower/utf8_upper`.
  Возвращает CadwDataFrame с `{col}_{op}`.
- **`df['s'].str.contains(pat)`** — возвращает **MaskFilter** (для `df[mask]`).
  na=False (pandas default): null не матчится.
- **`df['col'].isnull()` / `.notnull()`** — MaskFilter.

**Facade:**
- **`df.astype({col: dtype})`** — cast per column. `dtype` = `pa.DataType`
  или str-alias (`"int64"`, `"float32"`). `safe=True` (default) → raise на
  overflow; `safe=False` → wrap (Arrow `pc.cast`).
- **`df.dropna(how='any'|'all', subset=None)`** — **lazy**: композиция
  `notnull()` фильтров. `how='any'`: drop if ANY null. `how='all'`: drop
  if ALL null.
- **`df.drop(cols)`** — selection minus. Lazy.
- **`df.rename(mapping)`** — materialized (schema change). `{old: new}`.

**Внутренний helper:**
- `CadwDataFrame._apply_column_transform(column, feature, fn, out_type,
  accessor=...)` — generic per-batch Arrow transform + append `{col}_{feature}`.
- Новая папка `cadw/accessors/` с `_dt.py`, `_str.py`.

**AST guards:** `tests/test_arrow_call_memory_guard.py` — whitelisted
`_apply_column_transform`, `astype`, `dropna`, `drop`, `rename`.

### Исправлено (v0.9.0a-fix1 — first-run bugs)

- **`PredicateFilter.__init__` не знал про новые ops (8 test failures).**
  v0.9.0a расширил `mask_for_batch` (is_null/not_null/contains), но
  `__init__` валидировал по старому `_OPS` → `ValueError: unsupported op`.
  Fix: `_OPS_NO_SCALAR = frozenset({"is_null", "not_null", "contains"})`,
  `__init__` проверяет `_OPS ∪ _OPS_NO_SCALAR`. Class: partial update
  within one class (handler extended, validation not).
- **`test_dt_null_timestamp_propagates` — NaT не конвертируется через
  `ts.tolist()` (1 failure).** `pa.table({"ts": [Timestamp, NaT, ...]})`
  → `NaTType does not support utcoffset`. Fix: `pa.array(ts)` — Arrow-native,
  NaT → null. Class: §13.4/134-like (input conversion not Arrow-safe).
- **CHANGELOG `[Unreleased]` отсутствовал после v0.8.11 tag** —
  silent skip (v0.9.0a entry не записан). Fix: prepend `[Unreleased]`
  before first `## [` if missing. Class: §13.4/140 recurrence (2nd hit
  of same trap — v0.8.11a-fix1 was first).

### Тесты (v0.9.0a)

- fast: **~1228** (+30 accessors + type ops). slow: **8**.

## [0.8.11] — 2026-10-20

### Исправлено (safety — P-D1 close)

- **P-D1 (S2, live): Arrow-native `drop_duplicates` rewrite.** pandas
  chain (`to_pandas → sort_values → drop_duplicates → drop → from_pandas`)
  в `_local_dedup` заменён на Path C: `dictionary_encode(subset)` +
  `group_by(subset).aggregate(min/max row_id)` + `is_in` filter.
  Удалён `import pandas` из `cadw/ops/dedup.py`.

  **Bench (prototype `scripts/proto_dedup_arrow.py`):**
  | n | pandas | Path B (arrow) | **Path C (dict)** |
  |---|---|---|---|
  | 100k | 9.3 MB | 10.2 MB | **5.9 MB** |
  | 300k | 29.3 MB | 39.0 MB | **3.6 MB** |
  | 1M   | 87.7 MB | 47.2 MB | **10.9 MB** |

  avg peak(C)/peak(A) = **0.29**. Correctness A==B==C across keep ∈
  {first, last, False} × selectivity ∈ {10%, 50%, 90% dupes}.

  **Stopgap removed:** `check_state(5 × table.nbytes)` → `check_state(1.5 ×
  table.nbytes)` (honest Arrow-native bound, proto-backed). Effective
  partition limit back from `avail × 0.1` to `avail × 0.33`.

- **Dead pandas helpers removed из `cadw/ops/join.py`.** `import pandas
  as pd`, `_drop_null_keys` (superseded by `_drop_null_keys_arrow` в
  v0.8.5.3) — удалены. Zero pandas in ops except `shuffle.py::_pandas_hash_column`
  (documented perf exception, §9.2.20).

### Добавлено

- **`tests/test_no_pandas_in_ops.py`** — AST guard: `import pandas` в
  `cadw/ops/*` разрешён только в `shuffle.py` (whitelist + rationale).
  Также `test_dedup_uses_arrow_native` (Path C marker).
- **`tests/test_dedup_arrow_native.py`** — 16 behavior tests: first/last/
  False, nulls в subset/value, multi-subset, int key, empty, K=N, K=1,
  pandas-equivalence, oversize → raise, dedup→sort chain, filtered dedup.

### Ограничения (v0.8.11a, §11.5)

- `drop_duplicates` порядок вывода: partition order, без
  `sort_values("row_id")` внутри partition (был в pandas-версии).
  Set-equivalence сохранена; для детерминированного порядка — `.sort_values(...)` после.
- `drop_duplicates` recursion при skew — не реализован (как в v0.3.0).
  Oversize partition → MemoryLimitExceeded. Backlog.

### Тесты

- fast: **~1190** (+16 behavior, +5 AST guard). slow: **8**.

### v0.8.11b — strict join conditional + schema-order audit

#### Изменено

- **`merge()` strict join enforcement.** `budget >= 160 MB` →
  `MemoryLimitExceeded` on whole-operation RSS over-budget (was: warn).
  Threshold: `4 × _JOIN_FIXED_OVERHEAD_BYTES = 160 MB`, plus secondary
  `state_limit >= 40 MB` (protects against reservation drop).
  Below threshold → warn only (backward compat with `mem_mb=20` tests;
  fixed 40 MB overhead dominates small budgets — strict check would fire
  trivially even for tiny joins). See §9.2.6.

- **`MemoryGuard.state_limit_bytes`** — public property. Callers need
  `state_limit` without raising `MemoryLimitExceeded` (used by merge's
  strict-mode decision).

- **`cadw/ops/join.py::_JOIN_STRICT_BUDGET_THRESHOLD_BYTES`** = `4 ×
  _JOIN_FIXED_OVERHEAD_BYTES` = 160 MB. Module constant, documented.

#### Добавлено (schema-order audit — §13.4/145 backlog)

- **`tests/test_schema_order_audit.py`** — 4 AST guards for the audit
  findings:
  - `join._safe_materialize_partition`: explicit `schema=out_schema`.
  - `sort._write_run`: `ArrowIPCBatchWriter(table.schema)`.
  - `groupby._read_partition_safe`: no `.select()` on batches.
  - `dedup._local_dedup`: `select_schema` (v0.8.11a-fix3).
  No additional bugs found across 4 audited call sites.

- **`tests/test_join_strict.py`** — 4 tests: threshold constant, public
  property, strict raise at budget 200 MB, strict pass small join.

#### Ограничения (v0.8.11b, §11.5)

- **Join strict enforcement requires `budget >= 160 MB`.** Below — warn
  only, per-batch guards still enforce per-batch memory. For guaranteed
  strict enforcement, set `MemoryGuard(max_memory_mb=200)` or higher.
  Rationale: 40 MB fixed overhead is non-trivial below 160 MB budget;
  raise would fire trivially for tiny joins.

#### Тесты (v0.8.11b)

- fast: **~1198** (+8), slow: **8**.

### Тесты (итого v0.8.11)

- fast: **~1198**, slow: **8**.


## [0.8.10] — 2026-10-20

### Добавлено (safety follow-up, P-A3 addendum)

- **Whole-operation RSS warning on `merge()`.** `CadwDataFrame.merge`
  вычисляет `predicted_output = n_left × avg_matches × row_bytes_out`
  из planner'а и логирует WARNING, если
  `40 MB + 5 × predicted_output > state_limit`.
  `MemoryLimitExceeded` не бросается — enforcement остаётся per-batch в
  `_process_partition_pair`, warning — диагностический сигнал.

  **Почему warning, а не raise:** per-batch проверка выполняется 8–16
  раз (по числу пар partition'ов), поэтому добавление 40 MB offset
  per-batch over-predicts в 8–16×. Добавление offset на уровне
  `merge()` требует поднятия `mem_mb=20` → `100` в ~10 тестовых файлах
  (~30 тестов падают без этого). Это отдельная работа v0.8.11, не
  «5 минут». Warning даёт сигнал сегодня без test churn.

- **`cadw/ops/join.py::_JOIN_FIXED_OVERHEAD_BYTES = 40 * 1024 * 1024`**
  — module const, измеренный floor для `Table.join` RSS beyond logical
  output. Across 3 runs of pk join @ N=1M: 114.7 / 142.8 / 156.8 MB
  RSS delta для 24 MB output. Offset покрывает Arrow pool init (~16 MB)
  + hash-table build + non-Arrow temporaries (~24 MB). Invariant of
  output size — added once, not per-batch.

### Ограничения (v0.8.10-pre, §11.5 addendum)

- Join guarantee остаётся **best-effort на RSS**, не strict. Для
  бюджетирования: `estimated_rss = 40 MB + 5 × predicted_output`; если
  превышает `state_limit = budget × 0.5`, actual peak может превысить
  budget независимо от per-batch guards. Strict enforcement — v0.8.11
  с ревизией тестовых бюджетов.
- Run-to-run variance 30–40% для одинаковой нагрузки (pool retention,
  jemalloc page release timing). Формулы target'ят worst-case.

### Добавлено (per-group feature)

- **`df.groupby('sensor').shift(k, on='ts', columns=[...])`** — per-group
  shift. `on=None` → within-group order = row_id. Group cols excluded
  from default `columns` set.
- **`df.groupby('sensor').diff(k, on='ts', columns=[...])`** — per-group
  first difference.
- **`df.groupby('sensor').rolling(window, agg, on='ts', columns=[...])`** —
  per-group rolling. APPENDS `{col}_rolling_{agg}`; min_periods applies
  from group start.

  **Algorithm (prototype-validated, 11 cases A==B):**
  composite sort `[*group_cols, *on_cols]` → single deque per col,
  reset on group change → sort back by `__orig_rid`. **2 sort passes,
  0 shuffle.** Peak RAM = batch + O(window × #cols), independent of N
  and G (group count).

  **Semantics:** null in group col → own group (SQL, not pandas
  `dropna=True`). Group columns are NOT shifted by default (they're
  the key). This differs from pandas `groupby().shift(1)` which shifts
  everything including group key.

  **Reference:** `tests/_helpers/per_group_reference.py` — pandas
  ground truth (§13.4/128, /129, /139).

### Тесты

- fast: **~1130 passed** (+3 behavior warnings: tight/loose budget). slow: **8**.

## [0.8.9] — 2026-10-20

По результатам Pingis library-memory audit (TASK2, run 2026-10-20):
6 accept, 0 reject, 2 needs-research, 5 confirmed live violations.

### Исправлено (safety — Arrow memory discipline)

- **P-A1 (S2, live)** — `join._safe_materialize_partition` и
  `dedup._local_dedup` теперь используют generator form
  `Table.from_batches(gen, schema=...)`. Measured v0.8.8: list = 0.93–1.03×
  file (retained batches), gen = 0.0× (zero-copy from Arrow buffers).
- **P-A3 (S2, live)** — `Table.join` pre-check: `check_state(5 × row_bytes_out
  × actual_matches)` **до** join'а. `row_bytes_out` = сумма bit_width//8 по
  `_empty_result_schema` + null bitmap (не 256 B/row heuristic). Measured
  v0.8.8: RSS delta = 6.85× output on narrow int64 @ 1M; 5× conservative.
- **P-D1 (S2, live)** — `dedup._local_dedup` pandas chain (to_pandas +
  sort_values + drop_duplicates + drop + from_pandas) проверяется
  `check_state(5 × table.nbytes)` **до** `to_pandas`. Measured v0.8.8:
  3.95–8.29× table.nbytes. Effective partition limit shrinks to `avail × 0.1`.
- **P-G3 (S2, live)** — `_groupby_with_skew` вызывает
  `check_partition_buffers(num_partitions, min(buf, 64KB))` перед
  открытием P writers (mirror `shuffle.py:273`).
- **P-G5 (S2, blind spot)** — sample cache bounded by `min(10k rows,
  0.3×avail)` bytes. Wide schemas (>=10 KB/row) hit byte cap before row cap.
- **P-SH2 (S2)** — `_hash_column` uint64 → `pc.cast(arr, int64, safe=False)`
  bitcast → pandas C-hash (19× faster than Python fallback). Bitcast =
  two's-complement reinterpretation, bijection over uint64 ↔ int64 →
  same value → same hash (partition-disjoint preserved). Nulls propagate
  through `pc.cast` unchanged. (v0.8.9-fix2: prior XOR-2^63-then-cast
  variant raised `ArrowInvalid` because values ≥ 2^63 overflow
  `pc.cast(safe=True)` — every uint64 value, since XOR with sign bit
  pushes all into [2^63, 2^64). Fix: skip XOR, cast directly with
  `safe=False`. REPL snippet documented in `_hash_column` docstring.)
- **P-A4 (S4)** — удалён dead `if False` block в `_exact_actual_matches_arrow`.

### Добавлено

- **§9.2.6–9.2.19 в DECISIONS.md** — канонические формулы пика для 14
  Arrow/pandas calls, derived из **RSS bench** (не из `pa_delta`).
  Каждая формула сопровождается guard recommendation и residual от
  fitted regression. Пример: `f_join = 5 × output_nbytes` (RSS-based,
  было бы 3.5× из pa_delta).
- **§3.11** — новый обязательный принцип: **Arrow call memory discipline.**
  Любой Arrow/pandas call, аллоцирующий > O(batch), требует:
  (1) формула пика в §9.2.N; (2) явный check_state до вызова;
  (3) AST guard `test_arrow_call_memory_guard.py`.
- **§13.4/138** — «Доверие Arrow = доверие LightGBM». Same class as
  §13.4/25 (LightGBM adapter) и §13.4/26 (partial_fit whitelist).
  MemoryGuard проверяет RSS, но не видит allocation **внутри** Arrow-функции.
  Отсюда и `pa_delta` vs RSS разница в 2× на больших joins.
- **`tests/test_arrow_call_memory_guard.py`** — AST guard: 14 high-risk
  attributes (`join`, `group_by`, `sort_by`, `to_pandas`, `from_pandas`,
  `from_batches` list-form, `sort_indices`, `quantile`, `rank`,
  `value_counts`, `tdigest`, `index_in`, `is_in`, `dictionary_encode`).
  Каждый должен иметь preceding guard (check_state/check_batch/stat)
  в том же function scope. Whitelist для helpers (`_safe_materialize_partition`,
  `_local_dedup`, `_groupby_with_skew`, `hash_shuffle`).
- **`tests/test_pingis_safety.py`** — 14 behavior tests:
  `_estimate_row_bytes` (4), uint64 hash (3), groupby skew+budget (1),
  groupby sample byte-cap (1), dedup pandas chain raise (1),
  AST guards (2), round-trip sanity (3).

### Изменено

- `cadw/ops/join.py` — новая module-level `_estimate_row_bytes(schema)`.
- `cadw/ops/shuffle.py` — uint64 в `_hash_column` не идёт через Python loop.
- `dedup._local_dedup` — effective partition size ~5× меньше на тех же
  бюджетах (honest memory bound).
- AST guard при проверке `Table.from_batches` различает `ListComp` (bad)
  vs `GeneratorExp` (good).

### Ограничения (v0.8.9, §11.5)

- **P-Q2 `pc.quantile` whole column** — sort-based, ~1× column working
  copy. **NEEDS-BATCHED**, не используется в `cadw/`. Backlog §11.6.
- **P-R1 `pc.rank` whole column** — ~2× output + sort buffer.
  **NEEDS-BATCHED**. Per-batch use в `rank.py:94` already safe.
- **P-G2/P-G5 `pc.value_counts` при K≈N** — O(N) hash. **NEEDS-BATCHED**.
- **P-SH1 string shuffle** — needs research (dictionary growth не
  детерминирован, partition-disjoint invariant at risk).
- **P-A3 wide-schema benchmark** — measured on int64 only; wide string
  schemas deferred to v0.8.10 (proposed: N=500k, 5 string cols).
- **P-G3 skew workload benchmark** — confirmed by code review, not
  benchmarked. Deferred to v0.8.10 (N=1M, one key at 60%, P=16).

### Тесты

- fast: **~1124 passed, 1 skipped**. slow: **8 passed**.
- +14 в `tests/test_pingis_safety.py`, +2 AST в
  `tests/test_arrow_call_memory_guard.py` (guard + sanity on generator
  vs list-comp).

## [0.8.8] — 2026-10-19

### Добавлено (housekeeping, не функциональность)
- **`scripts/collect_project.py`** — переписан с `--mode=smart|full|tiny`.
  - `smart` (default): full для `cadw/`, `tests/`, `README`, `pyproject`,
    latest-3 `patch_*.py`, top-5 CHANGELOG, DECISIONS (§13.4 truncated to
    last 10 episodes). Stub: old patches, archive, backups, `uv.lock`,
    old CHANGELOG entries. **~2 MB → ~500 KB** (75% reduction).
  - `full`: старое поведение (all files).
  - `tiny`: README + tree + top-3 CHANGELOG (~50 KB).
- **`scripts/patch_v088_housekeeping.py`** — one-shot cleanup + audit:
  - Phase A: rewrite `collect_project.py`.
  - Phase B: move `scripts/patch_*.{sh,py}` (older than keep=3) →
    `scripts/archive/v0.8.7/`. `git mv` (сохраняет history).
  - Phase C: move `.backups/*` → `/tmp/cadw_backups_<ts>/` (вне repo).
  - Phase D: append [0.8.8] + §13.4/136 + §11.6.
  - Phase E: audit (layers, primitives, §13.4/§11.5/§11.6 counts).

### Framework status (audit подтвердил)
- 7 layers (Storage / Dataset / Access / Primitives / Operations /
  DataFrame / Export) — все на месте.
- 3 primitives (Reduce / HashShuffle / Sort) — stable API.
- Все `ops/*.py` (кроме primitives) импортируют `from cadw.ops.{reduce,
  shuffle, sort}` — нет reinvention.
- Все §13.4 episodes closed через CHANGELOG-записи.
- §11.5 — 18 known limitations + backlog (documented, not bugs).

### Рабочая папка
- `scripts/` → только `collect_project.py`, `_lib.sh`, latest-3 patches,
  `archive/v0.8.7/` (n-исторических), `bench_*`, `diagnose_*`.
- `.backups/` → очищено (`/tmp/cadw_backups_*`).
- `.gitignore` — без изменений (archive tracked).

### Тесты
- fast: **~1096 passed**. slow: **8 passed**.
- CHANGELOG guard: 3 passed (top == tag, descending, exists).

# Changelog

Все значимые изменения проекта CADW. Формат: [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/).

<!--
  Порядок: **descending by version** (v0.8.5 сверху). Восстановлен в
  v0.8.5 из-за сломанной цепочки v0.1.5+ (§13.4/111): каждый
  `patch_changelog()` вставлял перед anchor `[0.X.Y-1]`, а при
  отсутствии anchor `str.replace` был silent no-op. Как побочный
  эффект, anchors дрейфовали — top становился out-of-order. Пропущенные
  версии (v0.1.5, v0.1.6, v0.2.2, v0.3.5, v0.4.2, v0.5.0, v0.5.2-v0.5.4,
  v0.6.x, v0.7.0-v0.7.7, v0.7.10) — в `git log --oneline --grep='^v'`.
  AST guard `test_changelog_has_latest_tag` (top == tag on master) +
  `test_changelog_is_descending` (v0.8.5) фиксируют инвариант.
-->

## [0.8.7] — 2026-10-19

### Добавлено
- **`CadwDataFrame.resample(freq, agg, on=..., columns=None)`** —
  bucket aggregation по временной/целочисленной колонке.
  - `freq: int | str` — секунды (`5`) или строка (`'5s'`, `'1min'`/
    `'1m'`, `'1h'`, `'1D'`).
  - `agg ∈ {sum, mean, min, max, count}`.
  - `on: str` — timestamp / date / integer column. Bucket =
    `(t_sec // freq) * freq`.
  - Output: `[on, f'{col}_{agg}'...]`. Bucket column dtype = `on` dtype
    (timestamp → timestamp, int → int, date → date).
  - Peak RAM: batch + O(#cols) state + 1 output batch (BATCH=1000
    buckets). **Independent of N и bucket size** (running accumulator,
    не `list(values)`).

### Валидировано prototype-first (§13.4/123)
- `scripts/proto_resample.py`: A (pandas groupby ts//freq) == B
  (external_sort + bucket transition) == C (manual). 5 buckets × sum:
  `[(0,4),(5,13),(10,23),(15,32),(20,22)]`.

### Renamed: v0.8.3 → v0.8.7 (§13.4/132)
- Планировалось как v0.8.3 (roadmap). После v0.8.4/v0.8.5/v0.8.6 уже
  tagged, `test_changelog_is_descending` требует top = highest →
  v0.8.7 (не v0.8.3).

### Semantics (MVP, §11.5)
- **Empty bucket** (gap in ts) → skip (no NaN fill; для fill —
  `.reindex` после).
- **Null in `on`** → skip row (cannot bucket).
- **All-null agg bucket** → skip row (all columns None).
- **Null in agg col** → skip value (matches pandas skipna).
- **Nulls_all**: bucket с только null значениями (все cols) → skip row.
- **Partition**: bucket spanning batches — accumulator живёт через
  batch boundary (running state, O(1) RAM).

### Ограничения (v0.8.7, §11.5)
- `on` — single column. Multi-column (e.g. `on=['ts', 'sensor']` для
  per-group buckets) — не поддерживается (composite sort ≠ per-group;
  для per-group — `groupby(...)`).
- `freq` — только `int` секунды или строка `'Ns'|'Nmin'|'Nm'|'Nh'|'ND'`.
  Compound (`'5min30s'`), `'W'`, `'M'`, `'Y'`, sub-second (`'500ms'`) —
  не поддерживаются.
- `uint64` on — reject (overflow risk при bucket arithmetic; cast в
  int64 required).
- `date` on — granularity = days (freq should be ≥ `'1D'`; `'5s'` на
  date даст O(#days) buckets с 0 rows → skip → фактически пустой
  output для одного дня).
- All-null bucket + `count` — skip (не 0; для zero-fill — post-process
  через `left_merge`).
- `nan` timestamp bucket on far future (>year 2262 для ns) — overflow
  (int64). Documented, edge case.

### Тесты
- +28 в `test_resample.py`: sum/mean/min/max/count (prototype data),
  multibatch bucket, empty bucket skip, nulls, all-null bucket, null
  in on, timestamp('s'/'ms'), date32, validation (missing on,
  non-time-type, uint64, on-in-columns, invalid agg, non-numeric,
  empty columns), freq parser (9 param + 5 edge), preserves original,
  filtered, 2 AST guards.
- fast: **~1092 passed**. slow: **8 passed**.

## [0.8.6] — 2026-10-19

### Добавлено
- **Multi-column `on`** для `shift` / `diff` / `rolling`:
  `on=["ts", "sensor"]` (Sequence[str]).
  - **Лексикографический composite sort**, не per-group. Сортировка по
    `[on_cols..., row_id]`, применение shift/rolling per row, sort-back
    по `__orig_rid` (как в v0.8.2 / v0.8.4).
  - Per-group — `groupby('sensor').shift(1, on='ts')` (v0.8.5a+).
- `CadwDataFrame._normalize_on(on, fn_name)` — helper: `str | Sequence[str]`
  → `list[str]` + validation.
- `_shift_on(..., on_cols: list[str])` / `_rolling_on(..., on_cols:
  list[str])` — принимают pre-normalized list; `external_sort` получает
  `key_columns=on_cols`.

### Backward compat
- `on="ts"` — как раньше (single-element normalization).
- `on=["ts"]` — идентично `on="ts"`.
- `on=[]` — `ValueError: {fn}: on must be non-empty`.

### Ограничения (v0.8.6, §11.5)
- Multi-column `on` — **composite sort**, не per-group. Для per-group
  windows — `groupby(...).shift/rolling(..., on=...)` (v0.8.5a).
- `_iter_shifted` / `_iter_zipped` не дедуплицируются — дублирование
  остаётся (refactor в backlog).
- Stability with duplicate `on` values — tie-break by row_id (heap),
  как в v0.8.2/0.8.4.

### Тесты
- +5 в `test_shift_on_col.py`: multi-key shift/diff, single-element-list
  equivalence, empty-list ValueError, stable tie-break.
- +3 в `test_rolling_on_col.py`: multi-key sum, min_periods=1,
  empty-list ValueError.
- AST guards (v0.8.2/0.8.4) — unaffected (first sort uses `on_cols`
  variable; second still literal `["__orig_rid"]`).
- fast: **~1064 passed**. slow: **8 passed**.

## [0.8.5] — 2026-10-19

### Исправлено
- **CHANGELOG отсортирован descending** (§13.4/111 + §13.4/130). Исторически
  `patch_changelog()` вставлял перед anchor `[0.X.Y-1]`. Anchor дрейфовал,
  top CHANGELOG перестал быть «latest». `test_changelog_has_latest_tag`
  (master + tag) корректно упал: `top [0.7.4c] != tag v0.8.4`. Fix:
  одноразовый sort всех `## [X]` блоков by (major, minor, patch, suffix)
  descending. Новый guard `test_changelog_is_descending` (в
  `test_changelog.py`) ловит регресс. v0.8.4 tag **не** ретагнут —
  тест не запускается на старых tags (`--exact-match` на HEAD).
- **`row_id` reserved guard** в `external_sort` и `hash_shuffle`
  (§13.4/126). User column named `row_id` коллизит с internal identity
  column (appended by external_sort/hash_shuffle) → previously:
  `KeyError: Field "row_id" exists 2 times in schema` (через
  `ArrowIPCBatchReader`, в `sort_values`); `Multiple matches for
  FieldRef.Name(row_id)` (в `pc.sort_indices`). Теперь: `ValueError:
  external_sort: column name 'row_id' is reserved for CADW internal
  identity column. Rename your column before calling.` Guard выполняется
  **до** materialize batches.

### Добавлено
- **`tests/_helpers/rolling_reference.py`** — `expected_rolling(vals,
  window, agg, min_periods)` — deque ground truth (§13.4/128). Matches
  pandas: positional window, nulls consume slot, min_periods applies to
  non-null count.
- 4 rolling-теста (`test_rolling_min/max_with_nulls` в
  `test_rolling.py`; `test_rolling_on_min/max_with_nulls` в
  `test_rolling_on_col.py`) → `expected_rolling(...)`. Устраняет корень
  §13.4/128 (manual arithmetic).
- 8 новых тестов в `test_row_id_reserved.py`: external_sort
  (reject key/value + valid works), hash_shuffle (reject key/value +
  valid works), E2E (sort_values/groupby на user `row_id`).
- `test_changelog.py::test_changelog_is_descending` — strict descending
  invariant.

### Ограничения (v0.8.5, §11.5)
- `row_id` — reserved. User column → rename. `__orig_rid` тоже reserved
  (internal sortback). Рекомендуется префикс `my_`.
- Guard fires on arguments (`key_columns` / `value_columns`). Internal
  callers (`_shift_on`, `_rolling_on`, `rank`, `groupby`) либо передают
  `__orig_rid`, либо не содержат `row_id` в value list — не затронуты.

### Тесты
- +8 в `test_row_id_reserved.py`.
- +1 в `test_changelog.py` (descending guard).
- 4 rolling-теста → helper (behavior unchanged).
- fast: **~1056 passed**. slow: **8 passed**.

## [0.8.4] — 2026-10-19

### Добавлено
- **`CadwDataFrame.rolling(window, agg, on=None)`** — time-aware rolling.
  `on=None` — positional (v0.6.2c, unchanged); `on=str` — sort-aware:
  1. external_sort by `[on]` (row_id tie-break from heap).
  2. `_RollState.push` per row (reuse v0.6.2c state — same min/max
     monotone deque, same null-skip, same min_periods).
  3. external_sort by `__orig_rid` (rename `row_id` → `__orig_rid`,
     §13.4/122).
  4. Zip original + rolling cols **APPENDED** (`{col}_rolling_{agg}` —
     rolling adds new columns, doesn't replace).
  - Peak RAM = batch + O(window × #cols). Independent of N.

### Валидировано prototype-first (§13.4/123)
- `scripts/proto_rolling_on.py` — A (sort_values+rolling+sort_values)
  == B (external_sort + deque + external_sort) == C (pure Python manual).
- Same 5-row case as prototype; 1 итерация патча (v0.8.2: было 5 fix).

### Исправлено (v0.8.4-fix — S1, latent bug с v0.6.2c)
- **`_RollState.push(None)` для agg='min'/'max'** — `while _dq and
  _dq[-1][1] >= None` → `TypeError: '>=' not supported between int and
  NoneType`. Bug существовал с v0.6.2c (позиционный rolling), не
  покрыт тестами (`test_rolling_with_nulls` использует sum). Fix:
  guard `if not is_null` вокруг monotone-deque maintenance. Window
  slot всё равно потребляется (positional semantics, как pandas).
  Добавлены тесты: `test_rolling_min_with_nulls`,
  `test_rolling_max_with_nulls`, `test_rolling_min_all_nulls`
  (positional) + `test_rolling_on_min_with_nulls`,
  `test_rolling_on_max_with_nulls` (on=col).
- **v0.8.4 tests (2, not code)**: `test_rolling_on_with_nulls` и
  `test_rolling_on_all_columns` — неверные ожидания. Actual code
  корректен (nulls потребляют window slot, positional; matches pandas
  skipna=True). Fixed expected values + comments.

### Обнаружен latent bug (§13.4/126)
- User column named `row_id` collides with internal `row_id`
  (external_sort/hash_shuffle/merge) → `KeyError: Field "row_id" exists
  2 times in schema` (в `sort_values`). Documented §11.5 as reserved
  column name; fail-fast guard — v0.9.

### Ограничения (v0.8.4, §11.5)
- `on` — single column. Multi-column `on=['ts', 'sensor']` — v0.8.4a.
- Per-group rolling (`groupby('sensor').rolling(...)`) — v0.8.5+.
- `row_id` — reserved column name (internal).

### Тесты
- +17 в `test_rolling_on_col.py`: sum/mean/min/max/count, min_periods,
  nulls, multibatch, preserved, all columns, sort-after, missing col,
  non-numeric, no-on backward compat, +2 AST guards (row_id rename,
  append-not-replace).
- fast: **~1043 passed**. slow: **8 passed**.

## [0.8.2] — 2026-10-19

### Добавлено
- **`CadwDataFrame.shift(k, fill_value=None, columns=None, on=None)`** —
  time-aware shift. `on=None` — positional (v0.6.2b, unchanged);
  `on=str` — sort-aware:
  1. external_sort by `[on, row_id]` (stable tie-break).
  2. Stateful shift (deque per column, O(k)).
  3. external_sort by `row_id` → original order.
  4. Zip with original.
  - Peak RAM = batch + O(k × #cols). Independent of N.
  - 2 external_sorts (vs manual `df.sort_values(x).shift(x).sort_values(x)`).
  - Multibatch consistent (chunk size не влияет).

- **`CadwDataFrame.diff(k, columns=None, on=None)`** — time-aware diff.
  `on=None` — positional (v0.6.2b); `on=str` — sort-aware via `_shift_on`
  + `pc.subtract`. Reuses shift's 2 sorts, no extra sort.

### Ограничения (v0.8.2, §11.5)
- `on` — single column. Multi-column `on=['ts', 'sensor']` — v0.8.2a.
- `on` с duplicates — stable sort (tie-break by row_id asc). Documented.
- `on` должен быть sortable (monotonic не требуется — обычный sort).
- Per-group shift (`groupby('sensor').shift(1, on='ts')`) — v0.8.3+.

### Тесты
- +17 в `test_shift_on_col.py`: shift on basic/k2/fill/string/all,
  preserved, nulls, multibatch, missing col; diff on basic/k2/preserved,
  non-numeric raises, non-monotonic stable, nulls; chain sort; backward
  compat (no-on unchanged).
- fast: **~1014 passed**. slow: **8 passed**.

### Цена
- Shift(1, on='ts') on 10^6 rows: ~2-3s (2× external_sort).
- Same as user-manual `df.sort_values('ts').shift(1).sort_values('ts')`,
  but API one-liner.

## [0.8.1] — 2026-10-19

### Изменено — performance
- **`compute_hashes`: pandas C-hash для numeric, Python для string.**
  - Numeric/boolean/timestamp/date → `pd.util.hash_pandas_object`
    (C-level SipHash). Bench: int64 39.2→2.0 ms (**19.6×**),
    float64 39.2→1.0 ms (**39.2×**).
  - String/large_string → Python `repr().encode()` + xxhash. Причина:
    pandas `hash_pandas_object` для object dtype — per-element Cython,
    50 ms/100k, **22% медленнее** Python (§13.4/107). Без регрессии.
  - struct/list/dict/uint64 → Python loop (fallback).
  - Seed через `h_np ^= np.uint64(seed)` **до** avalanche.
- **Avalanche (splitmix64) — всегда**, не только при seed. Причина:
  pandas `hash_pandas_object` для object dtype `hash << 1` (reserves 0
  для NA), все string hashes чётные → `h & 3` использует только
  партиции `{0, 2}` → recursion split 2-way вместо 4-way →
  `MemoryLimitExceeded` при depth=5 (§13.4/107). Fix: 3 умножения +
  3 XOR-shift, ~0.3 ms/100k.

### Критично для корректности
- **`types_mapper=Int64Dtype`** в `arr.to_pandas()` (numeric): без него
  `int64` с nulls → `float64` (NaN), без nulls → `int64` → разные hash
  для одного значения в разных батчах → нарушение partition-disjoint.
- Тесты: `test_int64_consistent_hash_across_null_presence`,
  `test_avalanche_spreads_low_bits`,
  `test_avalanche_seed_spreads_low_bits`.
- `test_shuffle.py::test_stable_hash_across_batches` — убран лишний
  `assert a_parts != b_parts` (partition-disjoint ≠ different partitions;
  коллизии разных ключей в одной партиции — норма).

### Тесты
- +20 в `test_hash_v081.py`: cross-batch consistency (int32/int64/string
  с/без nulls), deterministic, seed, nulls, dispatch (string→python,
  numeric→pandas), partition-disjoint (int+string), avalanche (low-bit
  spread + seed + similar strings), groupby recursion end-to-end.

### Ожидание
- int64 shuffle 1M: 0.61s → **~0.22s** (2.76×).
- 500 GB / 50 B / int64 keys: ~100 min → **~40 min**.

## [0.8.0] — 2026-10-19

### Добавлено — time-series
- `CadwDataFrame.expanding(agg, min_periods=1, columns=None)` —
  кумулятивное окно. Aggs: `sum`/`mean`/`min`/`max`/`count`. O(1) state
  (scalar accumulators; min/max — текущий min/max, без deque). Nulls
  skip; null row → null output. Output: `{col}_expanding_{agg}`.
- `CadwDataFrame.ewm(alpha/span/halflife/com, adjust=True, min_periods=0,
  columns=None)` — экспоненциально-взвешенное среднее. Ровно один из
  4 параметров. `adjust=True`: `(s_num, s_den)`, `y = s_num/s_den`.
  `adjust=False`: `(ewm,)`, `y = (1-α)·y_prev + α·x`. Nulls skip
  (`ignore_na=True` семантика). Output: `{col}_ewm`.

### Добавлено — PCA single-pass
- `CovarianceMeanOp(CovarianceOp)` — тот же state, `finalize` →
  `{cov, mean, n}`. `fit_pca` → **2× I/O reduction**.
- `CovarianceOp` (bare matrix) не меняется — backward compat.

### Ограничения (v0.8.0, §11.5)
- `expanding`/`ewm` не поддерживают `on=col`.
- `ewm(ignore_na=False)`, `ewm(bias=True)` — не реализованы.

### Тесты
- +26 в `test_expanding_ewm.py`, +3 (CovarianceMeanOp + fit_pca regression).

## [0.7.9] — 2026-10-19

### Исправлено в v0.7.9-fix
- **`_AssignLeavesOp.update` 4-unpack** — после введения 5-tuple spec (`..., gain`) старый `j, thr, left, right = spec` падал `ValueError: too many values to unpack`. Fix: `spec[:4]` (forward-compatible slice). Тот же fix применён к `_assign_leaf_from_array` в изначальном v0.7.9.
- **Multiclass `_fit_boosting_multiclass` 4-tuple** — `s.replace(old, new, 1)` дважды не сработало (второй replace не нашёл pattern из-за отличий в форматировании). Fix: replace **all occurrences** без count → multiclass тоже 5-tuple.
- **+AST guard** `test_all_spec_unpack_uses_slice` — все `= spec` должны быть `= spec[:4]` / `= spec[:5]`. Ловит класс «4-unpack + 5-tuple» багов.

### Добавлено
- **`compute_feature_importance(model, importance_type='gain'|'split',
  top_k=None)`** — extraction helper in `cadw.ml.gbdt`.
  - `'gain'` (default): sum of split gains per feature — XGBoost-style,
    better reflects feature usefulness (big gain на 1 split > 10 малых).
  - `'split'`: split count (v0.7.4a semantics, backward compat).
  - Multiclass: gain summed across all (tree, class).
  - Returns dict sorted by importance ascending (for horizontal barh).
  - Handles both binary (list[tree]) and multiclass (list[list[tree]]).

- **Tree structure extended:** `spec = (feat, thr, left, right, gain)`
  (5-tuple) instead of 4-tuple. `gain` stored at index 4.
  `_assign_leaf_from_array` tolerates both 4- and 5-tuple (`spec[:4]`)
  — backward compat for pre-v0.7.9 pickled models.

- **`feature_importance(..., importance_type='gain')`** in matplotlib and
  plotly adapters — use the new helper. Split-count still available via
  `importance_type='split'`.

### Изменено
- **`feature_importance` default** — был `split count`, стал `gain`.
  Breaking change для downstream code, который relied on integer counts
  — но семантически правильнее; документировано.

### Ограничения (v0.7.9, §11.5)
- **Permutation importance** — не реализована (требует out-of-core eval
  на holdout; v0.7.10+).
- **SHAP** — не реализован (слишком тяжёлый для out-of-core; вне scope).
- Legacy 4-tuple spec (pre-v0.7.9 pickled): `gain` returns 0 for those
  trees (silently, gain not available). `split` works.
- `gain` — raw sum (not normalized); разные tree scale → разные magnitudes.
  Для normalized importance — `gain / gain.sum()` в user code.

### Тесты
- +13 в `test_feature_importance_gain.py`: helper (invalid type, non-GBDT,
  sorted), gain semantics (important feature, split differs), top_k,
  zero model, multiclass aggregation, adapters (matplotlib/plotly, both
  importance_type), legacy 4-tuple fallback, spec is 5-tuple.
- fast: **~995 passed**. slow: **8 passed**.

## [0.7.8] — 2026-10-19

### Исправлено в v0.7.8-fix3
- **Dispatch preserves backward-compat error messages.**
  `fit_gbdt_classifier(n_classes=None)` считает distinct **перед**
  `_validate_multiclass_y`. 2 distinct (или non-integer) → binary
  path (`_validate_binary_y` даёт исторический `"binary (0/1)"` /
  `"both classes"`); >2 distinct → multiclass path (`"0-indexed"`).
  Без этого `{2, 3}` и `{1}` давали `"0-indexed"` вместо ожидаемых
  сообщений — ломалось `test_gbdt_classification.py`.
- **+AST export guard** (`tests/test_api_availability.py`):
  `test_cadw_all_exports_importable`,
  `test_cadw_ml_all_exports_importable`,
  `test_cadw_ast_all_matches_imports`. Ловит класс багов
  «__all__ обновлён, import забыт» (§13.4/115).

### Добавлено
- **`fit_gbdt_multiclass`** — multiclass GBDT (softmax loss), per-class
  loop, reuse существующих ops (`_AllNodeHistOp`, `_AssignLeavesOp`,
  `_LeafValuesOp` — 0 новых ops).
  - K деревьев на round (по одному на класс); `trees_` = `list[n_estimators]`
    of `list[K]`.
  - Grad: `p[i,c] - onehot(y[i], c)`; Hess: `p[i,c]*(1-p[i,c])`
    (XGBoost diagonal approximation).
  - `base_prediction_ = zeros(K)` (деревья находят сдвиг).
  - `predict_proba → (n, K)`, `predict → (n,) int64`.
- **`GBDTMulticlassClassifier`** — model class; sklearn-подобные поля:
  `trees_`, `n_classes_`, `feature_names_`, `n_features_in_`,
  `predict_proba`, `predict`, `to_dict`.
- **`fit_gbdt_classifier(..., n_classes=None)`** — auto-dispatch:
  - `n_classes` inferred из `y` (`{0..K-1}`).
  - `K == 2` → `GBDTClassifier` (binary, unchanged).
  - `K > 2` → `GBDTMulticlassClassifier`.
  - `K < 2` → ValueError.
- **`CadwDataFrame.fit_gbdt_multiclass`** — фасад.

### Ограничения (v0.7.8, §11.5)
- y должен быть `{0, 1, ..., K-1}` (0-indexed). 1-indexed `{1..K}` —
  ValueError (сдвиньте y на -1).
- `K × binary` fit cost (per-class loop). K=3 → ~3× медленнее binary.
  Для K > 5, n > 1M — v0.7.8a добавит `_MulticlassAllNodeHistOp`
  (1 pass для всех K, ~K× speedup).
- One-vs-rest не поддерживается (только softmax).
- Multiclass quantile/huber — не реализованы.

### Тесты
- +22 в `test_gbdt_multiclass.py`: `_softmax` (sum, stability, dim),
  fit (basic, accuracy, n_estimators improves, deterministic), predict
  (proba shape, int labels, df columns, prob_df columns, nulls),
  dispatch (binary, multiclass, single-class error), y validation
  (1-indexed, missing class, K=2 rejected by fit_multiclass),
  memory guard, convenience, to_dict, binary regression.
- fast: **~979 passed**, slow: **8 passed**.

### Время
- K=3, n=1000, depth=3, 30 rounds: ~15-20s (CI). Локально — быстрее.

## [0.7.4c] — 2026-10-19

### Добавлено
- **`matplotlib_adapter.hexbin_plot(df, x, y, gridsize=30, n=10_000,
  random=True, ...)`** — 2D hexagonal binning (matplotlib native).
  - Sample-based (`sample(n)` + `plt.hexbin`), **не** aggregate-first:
    hexbin aggregation на гексагональной сетке требует custom binning,
    matplotlib делает это сам через `plt.hexbin`.
  - `log_scale=True` (default): LogNorm для colorbar (плотность scatter).
  - **Plotly/Seaborn:** нет native hexbin — используйте `hist2d_plot`
    (прямоугольная сетка). Documented.

- **`_aggregate.sample(df, n, *, random=False)`** — новый параметр.
  - `random=False` (default, fast): `df.head(n)` — bounded I/O,
    но **может быть смещён** на time-sorted данных (только префикс).
  - `random=True`: uniform random sample через
    `df.iloc[np.sort(rng.choice(cnt, n, replace=False))]`. Full O(N)
    scan, O(n) memory. Deterministic (seed=42).
  - `scatter` / `_sample_pair`: `random=False` по умолчанию
    (backward compat v0.7.4a).

### Ограничения (v0.7.4c, §11.5)
- `hexbin_plot` — только matplotlib. Plotly/Seaborn — нет hexbin
  (документировано; use hist2d).
- `random=True`: 2× O(N) scan (count + iloc to max position). Для 10^7
  строк ~секунды. `random=False` — fast preview.
- hexbin sample-based, не aggregate-first — n=10_000 bounded, shape
  approximate.

### Тесты
- +4 в `test_viz_aggregate.py`: sample(random=True) uniformity,
  small-df returns-self, deterministic, head default unchanged.
- +5 в `test_viz_matplotlib.py`: hexbin_plot writes PNG, random sample
  spreads, missing column raises, invalid gridsize raises, nulls skipped.
- fast: **~1026 passed**. slow: **8 passed**.

## [0.5.2a] — 2026-10-19

По результатам Pingis audit: P5 (S2) — sort merge Python-list buffer.

### Исправлено
- **P5: `_merge_keys`** — заменить Python-list buffer (`buffer[c].append(
  as_py())`) на `pa.Array.Builder` (`Int64Builder`/`Float64Builder`/
  `StringBuilder`/etc.). Пик на merge: `read_batch_size × arrow_size`,
  а не `read_batch_size × 28-80 B` (Python-object overhead).
  - `_make_builder(dtype)` — выбор builder по Arrow-типу.
  - `_append_to_builders` — append одного значения (с `.as_py()` O(1)).
  - `_flush_builders` — finish → `RecordBatch`.
  - Dictionary/list/struct → `NotImplementedError` (honest; не покрыто).

### Добавлено
- `scripts/bench_p5_sort.py` — RSS-сэмплер benchmark для sort
  (сравнение с Pingis baseline +31.9 MB).
- `tests/test_sort.py::test_merge_keys_uses_arrow_builder_not_python_list`
  — AST guard (нет `buffer[...].append`).
- `tests/test_sort.py::test_merge_keys_uses_arrow_builder` — presence check.

### Ожидаемый эффект
- B3a (1M rows, 30 MB budget): Δpeak ≤ 30 MB (baseline: +31.9 MB).

### Ограничения (v0.5.2a, §11.5)
- Dictionary/list/struct columns: `NotImplementedError` в sort merge.
- `.as_py()` per value — CPU overhead остаётся; P5 — memory-fix, не
  speed-fix. CPU-оптимизация (batch-wise `pc.sort_indices`) — v0.5.4.

## [0.5.1] — 2026-10-19

По результатам мультиагентного анализа Pingis (`_FOR_PINGIS/REPORT.md`):
P2 (S1) + P8 (S2).

### Исправлено
- **P2 (S1, streaming join):**
  - `_process_partition_pair` — теперь generator (`yield` вместо
    `out_batches.extend(...)`).
  - `_left_preserve_with_nulls_iter` — streaming-версия.
  - `_join_core` — общий setup (validate, peek, shuffle) → `(schema, iter)`.
  - `hash_join_streaming(...) -> (schema, Path)` — **NEW**: пишет
    выходные батчи напрямую в `ArrowIPCBatchWriter`. Peak RAM =
    `max(build_partition) + probe_batch + writer_buffer`, **не зависит
    от размера output**.
  - `CadwDataFrame.merge` использует `hash_join_streaming`; промежуточный
    `.arrow` удаляется после `Dataset.from_batches`.
  - `hash_join(...)` (старый API) сохраняет семантику — собирает
    streaming-output в `pa.Table` (backward compat).

- **P8 (S2, state_bytes API):**
  - `ReduceOp.estimate_state_nbytes() -> int | None` — метод ABC
    (default None → fallback на `MemoryGuard.estimate_state_size`).
  - Built-in ops override: `CountOp`/`SumOp`/`MinMaxOp`/`MeanOp` → 16;
    `StdOp` → 24; `CovarianceOp` → `8+8d+8d²`; `HistogramOp` → `8bins+16`.
  - `reduce()` вызывает `op.estimate_state_nbytes()` **до** `op.init()`.
  - User `ReduceOp` can override — публичный контракт для ML-слоя.

### Добавлено
- `cadw.ops.hash_join_streaming` — streaming join API.
- `tests/test_reduce_state_bytes.py` — 8 тестов (built-in + custom +
  fallback + oversized-state-rejects-before-init).
- `tests/test_join_streaming.py` — 10 тестов (generator, disk write,
  incremental batches, backward compat, merge uses streaming, cleanup,
  empty input).
- `.pingis/TASK.md` — periodic analysis task template.
- `.gitignore`: exception для `.pingis/TASK.md`.

### Изменено
- `test_join_oom_guard.py::test_oversize_partition_raises_with_max_depth_zero`
  — `next(gen)` вместо `gen(...)` (generator).

### Ожидаемый эффект
- Join B1a (100k × 10k под 80 MB) → Δpeak с ~140 MB до ≤ 80 MB (цель).
- ML custom `ReduceOp` (XTXOp, KMeansStepOp, HistOp) не получают
  warning про 1 MiB; правильный state-размер.

### Ограничения (v0.5.1, §11.5)
- P1 (join `to_pandas`/`pd.merge`) — отдельный патч (v0.5.2).
- P3/P4/P5/P6/P7/P10/P11/P12/P14 — в roadmap после v0.5.2.
- `hash_join` (не-streaming) всё ещё может OOM при большом output —
  используйте `hash_join_streaming` или `CadwDataFrame.merge`.

## [0.4.1] — 2026-10-19

### Добавлено
- `CadwDataFrame.loc` — open-endpoint slices на monotonic index:
  - `loc[:hi]` → от начала до `hi` (inclusive).
  - `loc[lo:]` → от `lo` до конца.
  - `loc[:]` → все строки (lazy view).
  - descending monotonic тоже работает (`loc['b':'d']` → d,c,b).
- `CadwDataFrame.loc[label]` — duplicate labels → `CadwDataFrame`
  (pandas: Series→DataFrame). Unique → `dict` (без изменений).
- `CadwDataFrame.loc[[labels]]` — все вхождения для каждого label
  (ранее first-wins).
- `MaterializedIndex.is_monotonic_increasing()`,
  `is_monotonic_decreasing()`, `is_monotonic()` — O(N), nulls skipped.
- `MaterializedIndex.has_duplicates()` — O(N), set-based.
- `MaterializedIndex.get_positions_all(label)` — все позиции по `==`.

### Изменено
- `loc[slice]` на **non-monotonic** index → `NotImplementedError`
  (pandas 2.x тоже).
- `loc[slice]` на **duplicate** index → `NotImplementedError`
  (pandas: cannot slice non-unique).
- `loc[slice]` с `step != 1` → `NotImplementedError`.
- `.at[label, col]` на duplicate label → `KeyError` (pandas: undefined;
  используйте `loc[[label]]`).

### Ограничения (v0.4.1, §11.5)
- **MultiIndex** (`set_index(['a','b'])`) — v0.4.2.
- **`loc[slice]` step** — v0.4.2.
- **Non-monotonic + slice** — `NotImplementedError` (pandas-стиль).
- **`.at` на duplicate** — `KeyError`.

### Тесты
- 22 новых в `test_loc.py`: open-endpoints (start/stop/both на
  asc/desc), monotonic API, `get_positions_all`, `has_duplicates`,
  duplicate-label `loc` (dict vs CadwDataFrame), `loc[[duplicates]]`,
  slice на duplicate → raise, `.at` на duplicate → raise.
- Обновлён 1 тест: `test_loc_slice_none_endpoint_raises` → 8 новых.

## [0.4.0d] — 2026-10-19

### Добавлено
- `_iter_filtered_batches_with_positions()` — yield `(batch, positions)`,
  positions = numpy int64 в unfiltered dataset. Композиция фильтров:
  MaskFilter → `positions[mask]`; SliceFilter → slice по local counter
  с сохранением raw positions.
- `_apply_filter_with_positions(filter, it)` — helper для filter chain.

### Изменено (filter-compat)
- `df.index` при фильтрах → `MaterializedIndex(_index_values[positions])`
  (pandas-совместимо). Раньше — `NotImplementedError`.
- `set_index(col)` при фильтрах → работает; материализует **полный**
  col (O(N) scan), `_index_values` = full → любой комбинации filter/set
  через positions.
- `reset_index(drop=True)` при фильтрах → materialize filtered view +
  RangeIndex(0, filtered_n).
- `reset_index(drop=False)` при фильтрах → колонка `index` =
  `_index_values[positions]`.
- `iloc[list]` на filtered view — **сохраняет** parent index через
  raw positions (раньше reset). Более pandas-совместимо.
- `loc` на filtered view — работает (через filtered.index).

### Закрыто (из v0.4.0a-c, §11.5)
- ~~`df.index`/`set_index`/`reset_index` с активными фильтрами — raise~~
- ~~`iloc[list]` на filtered view не сохраняет parent index~~
- ~~`loc` на filtered view — NotImplementedError~~

### Тесты
- 15 новых в `test_filter_compat.py`: filtered.index (range/mat/
  slice/mask_then_slice/empty), set_index+filter, filter+set_index,
  reset_index drop=True/False при фильтрах, iloc[list] preserves
  parent (range/materialized), loc на filtered, no-filter unchanged.
- Обновлены 4 теста: `test_index_with_filter_*`, `test_set_index_with_
  filter_*`, `test_reset_index_with_filter_*`, `test_iloc_list_on_
  filtered_*`, `test_loc_on_filtered_*` — теперь проверяют
  сохранение/работу, не raise.

## [0.4.0c] — 2026-10-19

### Добавлено
- `CadwDataFrame.loc` — label-based indexer (pandas-style).
  - `loc[label]` → `dict` (KeyError если label нет).
  - `loc[lo:hi]` → `CadwDataFrame` — inclusive labels, reverse → empty.
  - `loc[[l1, l2, ...]]` → `CadwDataFrame`.
  - Требует `MaterializedIndex`; `RangeIndex` → `TypeError`.
- `CadwDataFrame.at[label, col]` — scalar access by label + column.
- `CadwDataFrame.iat[pos, col]` — scalar access by position + column.
- `MaterializedIndex.get_indexer(labels)` → `list[int]` (O(N) scan, dict;
  unhashable fallback linear).
- `.at`/`.iat` — **read-only** (`__setitem__` → `NotImplementedError`).

### Ограничения (v0.4.0c, §11.5)
- **`loc` требует `MaterializedIndex`** (после `set_index`).
  `RangeIndex` → `TypeError`; используйте `iloc`.
- **`loc[lo:hi]`** — только оба endpoint обязательны; `loc[:hi]` /
  `loc[lo:]` / `loc[::step]` → `NotImplementedError`. v0.4.1.
- **`loc[bool_array]`** — `NotImplementedError` (use `df[mask]`).
- **`loc` на filtered view** — `NotImplementedError`. v0.4.0d.
- **duplicate labels** — первое вхождение; v0.4.1 — все.
- **`.at`/`.iat` set** — `NotImplementedError` (immutable).

### Тесты
- 30 тестов в `test_loc.py`: get_indexer (scalar/list/dup/first-wins/str),
  loc on RangeIndex raises, loc scalar/slice(inclusive/reverse/missing),
  loc list (basic/missing/dup/empty), loc bool_array raise, .at
  scalar/missing/set-raise, .iat scalar/out-of-range/set-raise,
  filter limit, loc preserves index.

## [0.4.0b] — 2026-10-19

### Добавлено
- `CadwDataFrame.iloc` — positional indexer (pandas-style).
  - `iloc[int]` → `dict[str, Any]` — одна строка.
  - `iloc[slice]` → `CadwDataFrame` — lazy, через `SliceFilter`.
  - `iloc[list[int]]` → `CadwDataFrame` — materialize K=len(list) rows.
  - Negative indices работают (int/list — resolve через count();
    slice — через `key.indices(n)`).
- `cadw/indexers.py` — модуль для indexers (`_ILocIndexer`). Отдельно
  от facade (`dataframe.py`), чтобы не ломать AST guard
  `test_facade_has_no_as_py`.

### Ограничения (v0.4.0b, §11.5)
- **`iloc[bool_array]`** — `NotImplementedError` (use `df[mask]`).
- **`iloc[list]` на filtered view** → default `RangeIndex` (не сохраняет
  parent index); v0.4.0c — filter-compat.
- **`iloc[list]` на unfiltered** — preserves parent index:
  RangeIndex → MaterializedIndex([start+p, ...]);
  MaterializedIndex → subset values (dtype preserved).
- **`iloc[list]`** — O(N) I/O scan (нужно дойти до последней позиции);
  O(K) память результата.

### Тесты
- 21 тест в `test_iloc.py`: iloc[int] (mid/last/negative/out-of-range),
  iloc[slice] (basic/step/negative/filtered), iloc[list]
  (basic/duplicates/negative/empty/out-of-range/filtered), index
  preservation (range/materialized/filtered-reset), errors
  (bool_array/unsupported type), AST guard sanity.

## [0.4.0a] — 2026-10-19

### Добавлено
- `CadwIndex` (ABC) + `RangeIndex(start, stop, step)` (lazy, O(1))
  + `MaterializedIndex(values)` (Arrow Array).
- `CadwDataFrame.index` property — RangeIndex (default) или
  MaterializedIndex (после set_index).
- `CadwDataFrame.set_index(col, drop=True, inplace=False)`:
  материализует колонку в память (Arrow Array).
- `CadwDataFrame.reset_index(drop=False, inplace=False)`:
  `drop=True` — отбрасывает index; `drop=False` — выносит в колонку
  `index` (или `level_0` при коллизии).
- Экспорт: `cadw.CadwIndex`, `cadw.RangeIndex`,
  `cadw.MaterializedIndex`.
- `cadw.dataset.index` — модуль.

### Ограничения (v0.4.0a, §11.5)
- **с активными фильтрами**: `df.index`, `set_index`, `reset_index` —
  raise `NotImplementedError` (RangeIndex+filter = O(N) scan;
  синхронизация — v0.4.0b).
- **`set_index`** материализует колонку (O(N) RAM). Для больших N —
  future (streaming loc — v0.4.1).
- **Index не сохраняется в Parquet** (`write_parquet` → reload =
  RangeIndex). Pandas-совместимое поведение.

### Тесты
- 20 тестов в `test_cadw_index.py`: RangeIndex/MaterializedIndex API,
  set_index drop=True/False/inplace, reset_index drop=True/False,
  коллизия «index» → «level_0», chained, filter-limit raise,
  persistence.

## [0.3.4] — 2026-10-19

### Добавлено
- **Combiner** для `groupby`: partial aggregation до shuffle.
  - `CadwDataFrame.groupby(...).agg(..., combiner=True)` — opt-in.
  - `groupby_aggregate(..., combiner=True)` — runtime.
  - Поддерживаемые агрегаты: `sum`, `count`, `min`, `max`, `mean`.
  - Для `mean`: partial (sum, count), финально `sum/count`.
  - Экономия: партиция `N × row_size` → `K_batch × comb_bytes`
    (×10000 для 10^6 строк / 100 групп).

### Как работает
1. `_combine_batches(batch_iter, by, agg_cols, agg_funcs)` — generator:
   `pandas.groupby(by).agg(partial_specs)` per batch. State O(batch_size).
2. `hash_shuffle` получает `value_columns = __comb_*`.
3. `_local_groupby(..., partial_merge=True)` вызывает `_merge_partials`:
   `table.group_by(by).aggregate(merge_specs)` + переименование + dtype.
4. Результат идентичен `combiner=False`.

### Ограничения (v0.3.4, §11.5)
- **std/var**: не поддерживаются в combiner (требуют Welford partial).
  Warning + fallback на обычный путь. v0.3.5.
- **skew path**: combiner отключён (heavy через streaming, light через
  shuffle без combiner). Warning. v0.3.5 — combiner для light.
- **auto**: default `combiner=False` (opt-in). Auto по `k_est` — v0.3.5.

### Тесты
- 12 тестов в `test_combiner.py`: equivalence для sum/count/min/max/mean,
  mixed aggs, same-col sum+count, std warning, skew warning, large
  N/low K, multi-by, empty input, dataframe API.

## [0.3.3] — 2026-10-19

### Добавлено
- `CadwDataFrame.quantile(column, q, interpolation='linear')`:
  scalar q → `float`; list q → `list[float]`.
  - `q ∈ [0, 1]`.
  - `interpolation='linear'` (pandas default; v0.3.3 only).
  - `skipna=True` (nulls не участвуют).
  - Sort: всегда ascending (квантиль инвариантен к порядку).
- `cadw.ops.quantile.quantile(batch_iter, column, q, mg, spill_dir,
  interpolation='linear')` — runtime.
- Экспорт: `cadw.quantile`, `cadw.ops.quantile`.

### Алгоритм
- Pass 1: external_sort по `column` (ascending) + side-effect count
  `n_valid` (один проход по входам, без двойного I/O).
- Positions: `idx_i = q_i*(n_valid-1)`; `lo_i = floor`, `hi_i = ceil`,
  `frac_i = idx_i - lo_i`.
- Pass 2: streaming scan sorted, dict `{pos → value}` для
  unique(`lo_i`, `hi_i`). Early exit при `pos > max(targets)`.
- Interpolation 'linear': `v_lo + frac*(v_hi - v_lo)`.

### Ограничения (v0.3.3, §11.5)
- **interpolation**: `'lower'/'higher'/'nearest'/'midpoint'` — v0.3.4.
- **skipna=False**: не поддерживается; nulls всегда пропускаются.
- **non-numeric columns**: `linear` требует numeric; для строк —
  `NotImplementedError` (см. v0.3.4 `lower`/`higher`).

### Тесты
- 20 тестов в `test_quantile.py`: median/min/max, matches numpy для
  5 q, scalar vs list return, n=1, nulls (skip, all-null), after filter,
  pandas-equivalence (с/без nulls), errors (missing/invalid q/invalid
  interp/empty list).

## [0.3.2] — 2026-10-19

### Добавлено
- `CadwDataFrame.rank` теперь поддерживает `method='max'|'last'|'average'`
  через **двухпроходный streaming** (O(1) RAM):
  - Pass 1: scan sorted_by → `(first_rank, group_count)` per tie-group
    на диск. State O(1) + per-batch accumulators (O(batch_size)).
  - Pass 2: scan sorted_by + advance по group file (sequential).
    `pos_in_group = global_pos - first_rank + 1`;
    `last_rank = first_rank + group_count - 1`;
    `max → last_rank`; `last → first_rank + count - pos`;
    `average → (first + last) / 2`.
- `VALID_METHODS = ('first', 'min', 'dense', 'max', 'last', 'average')`.
- `_VALID_STREAMABLE` / `_VALID_BUFFERABLE` — разделение методов.
- `_rank_dtype(method, pct)`: int64 для int-методов без pct; float64
  для average или pct.

### Изменено
- `compute_ranks_to_disk(sorted_path: Path, ...)` — принимает **Path**
  вместо `Iterable[RecordBatch]`, потому что bufferable-методы требуют
  двух проходов. Единственный caller — `CadwDataFrame.rank` — обновлён.

### Ограничения (v0.3.2, §11.5)
- **null values**: участвуют как значения (null == null); pandas
  `rank(skipna=True)` не поддерживается.
- **cost**: `max`/`last`/`average` = 2 passes by sorted_by + 1 pass
  by row_id (re-sort). Против 1 pass у streamable.

### Тесты
- 15 новых в `test_rank.py`: max/last/average на asc и desc, pct для
  average, dtype-проверки, after filter, pandas-equivalence для всех
  трёх, all-same для max/last/average.

## [0.3.1] — 2026-10-19

### Добавлено
- `CadwDataFrame.rank(by, method='first'|'min'|'dense', ascending=True,
  pct=False, output_col=None, batch_size=10_000)`:
  - external_sort по `by` → потоковое вычисление rank → external_sort
    rank-пар по `row_id` → zip с исходным в original order.
  - `first`: 1-based позиция в sorted order (ties по row_id asc).
  - `min`:   минимум ранга в tie-группе.
  - `dense`: 1, 2, 3, ... для уникальных значений.
  - `pct=True`: rank / total_n.
- `cadw.ops.rank.compute_ranks_to_disk` — primitive: sorted stream → file
  (row_id, rank).
- Экспорт: `cadw.compute_ranks_to_disk`, `cadw.VALID_METHODS`.

### Принцип
- Streamable методы — O(1) state. Bufferable (`max`/`average`/`last`) —
  O(max tie group size) = O(N) для pathological данных → **не в MVP**,
  требуют двухпроходного streaming.
- Результат в original order (две сортировки: по `by` + по `row_id`).

### Ограничения (v0.3.1, §11.5)
- **method**: `first`/`min`/`dense` только. `max`/`average`/`last` — v0.3.2.
- **null values**: участвуют как значения (null == null); pandas
  `rank(skipna=True)` не поддерживается.
- **cost**: 2× external_sort + 1× count-scan. Для 10^9 строк — ~3 полных
  прохода I/O.

### Тесты
- 19 тестов в `test_rank.py`: все три метода на asc/desc, pct, edge
  cases (all-unique / all-same / ties), after filter, custom output_col,
  errors (invalid method / missing column / collision), 4 × pandas-
  equivalence, chained с groupby.

## [0.3.0] — 2026-10-19

### Добавлено (первая операция v0.3)
- `CadwDataFrame.drop_duplicates(subset=None, keep='first', num_partitions=None)`:
  HashShuffle по subset + локальный `pandas.drop_duplicates` per partition.
  - `subset=None` → dedup по всем колонкам.
  - `keep ∈ {'first', 'last', False}` — все три pandas-режима.
  - `num_partitions=None` → авто из sample (k_est на subset).
- `cadw.ops.dedup.drop_duplicates(...)` — runtime, возвращает `pa.Table`.
- Экспорт через `cadw.drop_duplicates` и `cadw.ops.drop_duplicates`.

### Принцип
- Корректность: партиции `HashShuffle` не пересекаются по subset keys →
  локальный dedup = глобальный, concat без merge.
- `row_id` сохраняется в shuffle; сортировка по row_id перед dedup даёт
  корректный keep='first'/'last' (min/max row_id per key).
- `keep=False` (pandas semantics) локально = глобально.

### Ограничения (v0.3.0, §11.5)
- **Output order**: partition-concatenation, не pandas-original order.
  Для стабильного порядка — `.sort_values(...)` после.
- **Skew**: при oversized partition → `MemoryLimitExceeded` без recursion.
  Recursion в v0.3.1 требует save-row_id-at-resplit или streaming dedup.

### Тесты
- 15 тестов в `test_drop_duplicates.py`: все keep-режимы, subset=None,
  multi-column, edge cases (empty, missing, invalid), skew reject,
  pandas-equivalence, after filter, chained with sort.

## [0.2.5] — 2026-10-19
### Исправлено (§13.4 эпизод 47)
- `hash_join`: cartesian check заменён на **точную** оценку совпадений
  `actual_matches = sum over common keys(build_count × probe_count)` через
  `groupby.size()`. O(build + probe), дёшево. Устраняет ложные reject'ы
  для PK-FK (cartesian 12.6M при реальном output 2519).
- Удалены: `max_cartesian_rows` (все слои), `c_cartesian`.
- 2 AST guard'а: `test_actual_matches_checked_before_pd_merge`,
  `test_no_cartesian_check_in_join`.



### Добавлено (финал v0.2)
- `tests/test_e2e_join.py` (3 slow):
  - `test_e2e_join_100k_x_10k`: 100k FK × 10k PK (string keys), 20 MB
    budget, inner join; pandas-equivalence; RSS-дельта < 200 MB;
    elapsed < 60 s.
  - `test_e2e_join_skew_reject`: 100k × 100k один ключ →
    `MemoryLimitExceeded` до OOM; RSS-дельта < 500 MB.
  - `test_e2e_join_left_preserve`: 100k × 5k, half unmatched → 100k
    preserved; null-mask совпадает с pandas.

### Закрыто (v0.2)
- `CadwDataFrame.merge(inner/left)` — пользовательская функция.
- `hash_join(how, max_output_rows, max_cartesian_rows)` — runtime.
- `JoinMemoryModel` + `propagate_chars("join")` — planner integration.
- OOM safety: size-check до materialize, schema-correct empty IPC,
  recursive partition split, per-partition unique_build check.
- `use_planner=True` в `merge`, sampling helpers в `cadw/core/sampling.py`.

### Тесты v0.2
- fast: 330 passed, 1 skipped.
- slow: 8 passed (5 был + 3 E2E join).
- Всего: **338 passed, 1 skipped**.

## [0.2.4] — 2026-10-19

### Исправлено (§13.4 эпизод 42)
- Модель `JoinMemoryModel` возвращает два лимита: `max_output_rows`
  (expected, из `output_budget / row_size_out`) и
  `max_cartesian_rows` (`c_cartesian × max_output_rows`, default 10×).
  Runtime `hash_join` разделяет проверки:
  - `expected_probe_output = len(probe) × avg_matches_per_probe >
    max_output_rows` → raise (per-partition skew);
  - `len(build) × len(probe) > max_cartesian_rows` → raise (extreme cartesian).
  Ранее `len(build) × len(probe) > max_output_rows` ошибочно сравнивал
  cartesian с expected → `plan_pipeline` мог сказать `"fits"`, а `merge` —
  упасть.

### Добавлено
- `JoinMemoryModel.c_cartesian` (default 10.0) — множитель для worst-case.
- `JoinParams.max_cartesian_rows`.
- `hash_join(..., max_cartesian_rows=None)`; default `10 × max_output_rows`.
- `CadwDataFrame.merge(..., num_partitions=None, use_planner=True)`:
  - `use_planner=True` (default): sample first batch каждой стороны,
    `mg.plan("join", ...)`, параметры пробрасываются в `hash_join`.
  - `use_planner=False`: старые дефолты (`num_partitions=8`,
    `max_output_rows = 0.3 × available / 256`).
  - `num_partitions` явно → override модели, но `max_output_rows` /
    `max_cartesian_rows` всё равно из модели.
- Per-partition `unique_build_keys` (pandas `.nunique`/`drop_duplicates`
  на build) — уточнённый `avg_matches_per_probe`, недоступный глобальной
  модели.

### Тесты
- 8 новых в `test_merge_planner.py`: planner default, `use_planner=False`,
  explicit `num_partitions`, early-fail план, cartesian-skew reject,
  left-planner, empty-side inner/left.

## [0.2.3] — 2026-10-19

### Добавлено
- `JoinMemoryModel` (§9.2.5): `f_join = build_partition + probe_batch +
  output_batch × (1 + c_merge)`.
  - `solve()`: `P` = max от shuffle обеих сторон; `build_side` по `how`
    (right для left join; меньший state для inner);
    `max_output_rows` из `output_budget / (row_size_out × (1+c_merge))`
    вместо ad-hoc `0.3 × available / 256`;
    `avg_matches` = upper bound `N_L × N_R / max(K_L, K_R)` (floor 1).
  - `JoinParams`: `build_side`, `how`, `num_partitions`,
    `probe_batch_size`, `build_partition_bytes`, `max_output_rows`,
    `right_chars`, `probe_row_size`, `output_row_size`, `avg_matches`.
- `propagate_chars("join")`: `N' = N_left × avg_matches`,
  `row_size' = output_row_size`, `k_est' = N'` (upper bound),
  `size_of_key_entry' = row_size' + 80`, `order = "random"`.
  Требует `JoinParams` в `params` (из `get_model("join").solve()`).
- `plan_pipeline` теперь планирует цепочки с join
  (`filter → join → groupby`, `join → join`).

### Открыто (11-B-2c-3)
- Модель считает **expected** output, runtime — **cartesian worst case**.
  `plan_pipeline` может предсказать `"fits"`, а `merge` — упасть.
  Fix: разделить `max_output_rows` (expected) и `max_cartesian_rows`
  (worst case, ~10× larger). §13.4 эпизод 42.

## [0.2.1] — 2026-10-19

### Добавлено
- `CadwDataFrame.merge(..., how='left')`: left join, все строки self
  сохраняются; right-колонки -> NULL при отсутствии совпадения.
- `cadw.ops.join.hash_join(how='inner'|'left')` — общий runtime;
  `inner_join`, `left_join` — обёртки.
- SQL-семантика NULL:
  - inner: `_drop_null_keys` на обе стороны;
  - left: `_drop_null_keys` только на build (right); probe (left)
    сохраняется (NULL в left key -> NULL right cols).
- `_empty_result_schema` — nullable=True для всех полей (left join с
  unmatched: NaN в int/string -> null, cast safe=False).

### Ограничения v0.2.2
- `how='right'`, `how='outer'`, `left_on`/`right_on`, `on=index` — v0.3.
- `JoinMemoryModel` + `propagate_chars("join")` + E2E — 11-B-2b/2c.

## [0.2.0] — 2026-10-19

### Добавлено
- `CadwDataFrame.merge(right, on, how='inner', suffixes=('_x','_y'),
  num_partitions=8)`: Grace hash inner join.
- `cadw.ops.join.inner_join`: HashShuffle обеих сторон (один seed) →
  партиции с одинаковыми ключами → для каждой пары: build = меньшая
  партиция (`check_state`), probe батчами (`check_batch`),
  `pd.merge(how='inner', suffixes=...)`.
- SQL-семантика NULL: `_drop_null_keys` до `pd.merge` (pandas.merge
  по умолчанию матчит NaN==NaN — не то).
- Защита от cartesian-skew: `len(build) * len(probe) > max_output_rows`
  → `MemoryLimitExceeded` **до** `pd.merge`.

### Ограничения v0.2.1
- `how='inner'` только; `left`/`right`/`outer` — v0.2.2.
- `on` по имени в обеих сторонах; `left_on`/`right_on`, `on=index` —
  v0.2.2.
- `JoinMemoryModel` + `propagate_chars("join")` + E2E — 11-B-2.

## [0.1.7] — 2026-10-19

### Исправлено (must-fix §11.4)
- `_local_groupby`: dict check до `group_by`. `_read_partition_safe`
  проверял только `file_size`, но `group_by` строит hash-таблицу
  `K_i × size_of_key_entry` — для узких строк dict больше файла, OOM
  внутри pyarrow. `_estimate_distinct_keys` (product of nunique) ×
  `size_of_key_entry` проверяется до `group_by`; overflow -> in-memory
  recursion через `hash_shuffle`; `depth >= max_depth` ->
  `MemoryLimitExceeded` с file/dict/safety.

### Добавлено
- `_estimate_distinct_keys(table, by_columns) -> int` — upper bound без
  построения hash-таблицы (`pc.count_distinct` per column).
- AST guard `test_local_groupby_dict_check_before_group_by`.
- Behavior tests: `test_dict_overflow_triggers_split`,
  `test_dict_fits_no_recursion`, `test_max_depth_raises_on_dict_overflow`.

## [0.1.4] — 2026-10-18

### Исправлено (must-fix §11.4, класс «materialize-before-check»)
- `_local_groupby` материализовал всю партицию (`list(ArrowIPCBatchReader)`)
  до size-check. 02b1 добавил проверку **после** materialize — это не
  закрыло баг. Fix: `partition_path.stat().st_size` до чтения + helper
  `_read_partition_safe` (единственное место, где разрешён `list(reader)`).
- `reduce` вызывал `op.init()` до `estimate_state_size`/`check_state`.
  `CovarianceOp(d=10^5)` → `np.zeros((10^5,10^5))` = 80 GB → OOM **внутри**
  init. Fix: `estimate_state_size` + `check_state` **до** `op.init()`.

### Добавлено
- §3.10 «Check before materialize» — новый обязательный принцип.
- `tests/test_materialize_guard.py` — AST guards:
  - `test_no_list_of_whole_reader_outside_helper`
  - `test_reduce_estimate_state_before_init`
- `tests/test_oom_resistance.py` (slow):
  - 10^6 групп, 10 MB budget;
  - 90/10 skew, 10^6 строк, 10 MB;
  - `CovarianceOp(d=5000)` под 10 MB → `MemoryLimitExceeded` до `init`.

### Уроки (DECISIONS §13.4)
- Эпизод 27: «закрытие must-fix» требует теста на **порядок** операций,
  а не только на наличие вызова. 02b1 проверял, что `hash_shuffle`
  вызывается, но не проверял, что `list(reader)` **не** вызывается до.
- Эпизод 28: аудит перед новыми функциями находит унаследованные баги
  (`reduce.init()` — не был замечен при 02a).

## [0.1.3] — 2026-10-18

### Удалено
- `CadwDataFrame.partial_fit` — адаптер к `sklearn.partial_fit`. Причина:
  whitelist (`_SAFE_PARTIAL_FIT`) — политика на доверии; любая модель вне
  whitelist или модель с выросшим state OOM-ит без предупреждения от CADW.
  См. DECISIONS.md §3.9, §13.4 эпизод 26.
- `_SAFE_PARTIAL_FIT` whitelist.
- `tests/test_ml_integration.py` (11 + 1 slow).
- `tests/test_sklearn_extras.py` (4 + 1 slow).
- sklearn-секция из `tests/test_api_availability.py` (7 assertions).
- `scikit-learn` из `[dependency-groups]` (dev).
- ML-секции из README, заменены на «ML и внешние библиотеки» с policy.

### Добавлено в DECISIONS.md
- §3.9: «Никаких ML-адаптеров» (не только `fit`, но и `partial_fit`).
- §11.5: расширенная «Политика ML» (3 класса: экспорт / адаптеры запрещены
  / собственные реализации v0.3).
- §12.5.1: план v0.3 — GBDT, Linear, k-means, PCA, NB на примитивах (~8–11
  сессий на весь ML-слой).
- §13.4 эпизод 26: whitelist `partial_fit` — политика на доверии, не
  гарантия.

### Единственный ML-интерфейс CADW
- `to_batches(format='arrow'|'pandas'|'numpy'|'torch')` — экспорт батчей.
  Ответственность за OOM в сторонней модели — на пользователе.

## [0.1.2] — 2026-10-18

### Удалено
- `CadwDataFrame.to_lightgbm` — адаптер к LightGBM. Причина: `lgb.train`
  требует O(N)-state (все градиенты в RAM), OOM происходит внутри
  библиотеки, `MemoryGuard` не спасает. Противоречит §3.9 «no adapters to
  O(N)-algorithms». См. DECISIONS.md §13.4 эпизод 25.
- `tests/test_lightgbm.py` (11 + 1 slow тестов).
- `lightgbm` из `[dependency-groups]` (dev).
- LightGBM-секции из README, DECISIONS §4.1/§11.5/§12.5/§13.3.

### Изменено
- `CadwDataFrame.partial_fit`: добавлен whitelist `_SAFE_PARTIAL_FIT`
  (SGD, StandardScaler, MiniBatchKMeans, IncrementalPCA, NB, MLP).
  Модели вне whitelist → `RuntimeWarning` с указанием на §11.5.
  Docstring обновлён: контракт безопасности (модель должна иметь
  O(1)-state w.r.t. N, это ответственность модели, не CADW).
- `tests/test_api_availability.py`: убрана LightGBM-секция.

### Добавлено в DECISIONS.md
- §3.9: «Никаких адаптеров к O(N)-алгоритмам».
- §11.5: блок «Политика ML» (3 класса: incremental / batch-only /
  own implementation).
- §11.4: обнаружена регрессия `_local_groupby` (materialize до
  `check_state`) — вынесена как must-fix.
- §13.4 эпизод 25: почему адаптер к LightGBM — ошибка.

### Планы
- v0.3: свой GBDT (level-wise, аналог `sklearn.HistGradientBoosting`) на
  примитивах: `HistogramOp` (binning) + `HashShuffle` (узел,фича,бин) +
  `Reduce` (sum_grad/sum_hess/count).

## [0.1.0] — 2026-10-18

Первый production-ready каркас: три примитива out-of-core с безусловной
гарантией памяти, pandas-подобный фасад, ML-интеграция (sklearn + LightGBM),
E2E на 200k строк / 20 MB budget.

### Добавлено — storage (уровни 0–1)
- `ParquetBatchReader/Writer`, `ArrowIPCBatchReader/Writer`.
- `Dataset` (03a): per-column Arrow IPC, синхронные чанки, `add_column`
  (валидирует boundaries, атомарная запись `.tmp.arrow` → `rename`),
  `iter_batches(batch_size=None|N, columns=...)` с re-chunk, persistence в
  `dataset_meta.json` (schema через `Schema.serialize()`).

### Добавлено — примитивы (уровень 3)
- `Reduce` (02a): `ReduceOp`/`reduce`, `CountOp`, `SumOp`, `MinMaxOp`,
  `MeanOp`, `StdOp` (Welford), `CovarianceOp` (d×d), `HistogramOp`
  (binning LightGBM). Инвариант «не зависит от `batch_size`», «не пишет
  на диск».
- `HashShuffle` (02b): `compute_hashes` (`xxhash.xxh64` + `repr(v).encode`),
  `_mod_p` (NumPy `uint64` для wrap-around), `row_id` через `order_np`
  (а не позицию в отсортированном батче), `estimate_num_partitions`,
  skew-изоляция (`_detect_heavy_keys` >5%, `_groupby_with_skew`).
- `Sort` (02b): `external_sort`, `compute_fan_in`, `_Key` (null_rank
  + per-column asc/desc), k-way merge, многоуровневый merge, `row_id`
  в каждом run'е для стабильности.

### Добавлено — операции (уровень 4)
- `groupby_aggregate` (02a/02b): `HashShuffle + LocalGroupBy + Concat`,
  авто-P через sample, рекурсивное партиционирование (`MAX_DEPTH=5`),
  skew-изоляция (тяжёлые потоково, лёгкие shuffle), single-pass.
  Корректен при `K > памяти` (`test_groupby_many_groups`: 1000 групп,
  200 KB budget) и при экстремальном skew (90/10, one-hot).

### Добавлено — доступ (уровень 2)
- `RowFilter` (03c): `PredicateFilter`, `SliceFilter`, `AndFilter`,
  `OrFilter`, `NotFilter`. `& | ~` для `MaskFilter`; slice+mask — через
  chaining.

### Добавлено — фасад (уровень 5)
- `CadwDataFrame` (03b/03c): `from_parquet`, `from_dataset`,
  `columns`, `shape` (upper bound), `count()` (точное), `head(n)`,
  `to_batches(batch_size, format='arrow'|'pandas'|'numpy'|'torch')`,
  `__getitem__(slice|list[str]|str|RowFilter)`, `groupby().agg()`,
  `unique()`, `sort_values()`, `write_parquet()`,
  `partial_fit(model, X_cols, y_col=None, **kwargs)`,
  `to_lightgbm(X_cols, y_col, max_rows=N, categorical_cols=None, **params)`.
- Ленивость: `df[df['val'] > 15][1:5]` не материализует данные до
  `to_batches`/`write_parquet`/`groupby`/`sort_values`/`head`.
- AST guard-тесты: `test_facade_has_no_as_py`, `test_facade_has_no_row_loop_ast`.

### Добавлено — MemoryGuard (уровень 2)
- Байтовый контракт: `available_bytes`, `get_batch_rows`,
  `check_batch(extra_factor)`, `check_state`, `check_partition_buffers`,
  `recommended_partition_buffer_bytes`, `estimate_state_size`,
  `estimate_unique_keys`, `reserve/release`, `log_step`.

### Добавлено — ML (уровень 6)
- sklearn: `SGDClassifier` (`classes=[0,1]` на первом вызове),
  `StandardScaler`, `MiniBatchKMeans`, `IncrementalPCA` (без `n_features`
  в `partial_fit` — выводится из `X.shape[1]`).
- LightGBM: `to_lightgbm(max_rows=N)` — честный downsampling
  (LightGBM требует весь датасет в RAM).
- E2E: sklearn pipeline (50k строк, 10 MB budget), LightGBM pipeline
  (50k строк, downsampling до 20k).

### Добавлено — E2E и guards
- E2E (04): filter → groupby → sort → parquet на 200k строк, 20 MB budget,
  pandas-equivalence.
- Architecture guards: `test_no_legacy_package`, `test_legacy_tests_removed`,
  `test_main_code_does_not_reference_legacy`.

### Удалено
- Legacy `cadw/_legacy/` и `tests/_legacy/` (05): `SpillManager`,
  `UniqueCollector`, `GroupByAggregator` — заменены примитивами и фасадом.
- `cadw/dataframe.py` (старая заготовка) — переписан с нуля.

### Ограничения (документированы в DECISIONS.md §11.5)
- `to_lightgbm` не out-of-core: LightGBM требует весь датасет в RAM.
- `PCA` (batch-only) не имеет `partial_fit`; для batch-only — `vstack`.
- `SGDClassifier.partial_fit` требует `classes=[...]` на первом вызове.
- Категориальные признаки для LightGBM — int-encoded, не строки.
- `shape` на отфильтрованном `CadwDataFrame` — upper bound; точное — `.count()`.
- `nulls_first` в `Sort` — глобальный флаг, не per-column.
- `df[-5:]` (отрицательный slice) — не поддержано.
- `compute_hashes` — Python-цикл (векторизация отложена).
- `CadwIndex`/`loc`/`iloc`, `join`/`merge`, `drop_duplicates`, `rank`,
  `rolling` — не реализованы (v0.2).

### Тесты
- **213 fast + 5 slow = 218 passed, 1 skipped** (torch опционален).
- Покрытие: MemoryGuard (25), Reduce (33), HashShuffle (13), Sort (10),
  groupby + must-fix (10 + 28), Dataset (13), CadwDataFrame (17),
  RowFilter (25), ML (sklearn + LightGBM, 27), E2E (5 slow), guards (3).

[0.1.0]: https://example.invalid/cadw/compare/v0.0.0...v0.1.0
