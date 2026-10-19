# Changelog

Все значимые изменения проекта CADW. Формат: [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/).

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
