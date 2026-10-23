# 10. Dataset, RowFilter, CadwDataFrame (реализованы)

> v0.9.28 B.2 split — original DECISIONS.md lines 994-1050.


### 10.1. `Dataset` (03a)

Per-column Arrow IPC в `base_dir/columns/<col>.arrow`; `dataset_meta.json`
со schema (base64 `Schema.serialize()`), `chunk_boundaries`, `total_rows`.
Все колонки имеют **одинаковые** boundaries. Методы: `from_parquet`,
`from_batches`, `add_column` (валидирует boundaries, атомарная запись через
`.tmp.arrow` → `rename`), `iter_batches(batch_size=None|N, columns=...)`
(re-chunk через `pa.concat_arrays` / `.slice`), свойства `columns`, `schema`,
`total_rows`, `chunk_boundaries`.

Инвариант: `i`-й batch всех колонок — одни и те же глобальные строки.

### 10.2. `RowFilter` (03c)

| Класс | Что |
|---|---|
| `PredicateFilter(col, op, value)` | `pc.greater/less/equal/...` per batch |
| `SliceFilter(start, stop, step)` | stateful, O(1) состояния (счётчик pos) |
| `AndFilter` / `OrFilter` / `NotFilter` | `pc.and_` / `pc.or_` / `pc.invert` per batch |

`&`, `|`, `~` определены для `MaskFilter` (predicate/and/or/not). Комбинация
`Slice` с mask — через chaining (`df[10:50][mask]`), не через `&` (slice
меняет позиции, маска — нет; объединение сломало бы семантику).

### 10.3. `CadwDataFrame` (03b/03c)

`(Dataset, MemoryGuard, spill_dir, columns, row_filters)`. Методы:

| Метод | Реализация |
|---|---|
| `from_parquet(path, base_dir, chunk_size)` | `Dataset.from_parquet` + `cls(...)` |
| `columns`, `shape` | `shape` — верхняя оценка (`total_rows`, `len(columns)`); точное — `.count()` |
| `count()` | сканирует отфильтрованный поток |
| `head(n)` | итерирует отфильтрованно, `slice` |
| `to_batches(batch_size, format='arrow'|'pandas'|'numpy'|'torch')` | `_iter_arrow_batches` + `_convert_batch` |
| `__getitem__(slice)` | `SliceFilter` |
| `__getitem__(list[str])` | selection, фильтры сохранены |
| `__getitem__(str)` | `_ColumnProxy` для `df['a'] > x` |
| `__getitem__(RowFilter)` | `_with_filter` (chaining) |
| `groupby(by).agg(cols, funcs)` | `groupby_aggregate(self._iter_filtered_batches(), ...)` |
| `unique(col)` | `groupby_aggregate([col],[col],['count'])` → select |
| `sort_values(by, ascending)` | `external_sort(...)` → новый `Dataset` |
| `write_parquet(path)` | `ParquetBatchWriter` поверх `_iter_filtered_batches` |
| `partial_fit(model, X_cols, y_col=None, **kwargs)` | per-batch `np.column_stack` + `model.partial_fit` |
| `to_lightgbm(X_cols, y_col, max_rows=N, categorical_cols=None, **params)` | `head(N)` → `lgb.Dataset(X, y)` |

**Честность фасада** проверяется AST-guard-тестами:
`test_facade_has_no_as_py`, `test_facade_has_no_row_loop_ast`. Если в
`dataframe.py` появляется построчная обработка — тесты падают.

**Воспроизводить `BlockManager` не нужно** (скрытые копии — причина OOM).
Воспроизведён интерфейс на модели «независимые `ChunkedArray` + `row_id`».

---
