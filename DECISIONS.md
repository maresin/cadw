# DECISIONS.md — Архитектурные решения CADW

> Версия: 2026-10-18 (v0.1.0)
> Назначение: единый документ, восстанавливающий контекст проекта с нуля.
> Читатель, не знакомый с CADW, должен понять цель, устройство, алгоритмы,
> обоснования и **границы применимости**.

---

## 0. Метаданные и статус

**Как читать.** §1–§4 — цель, глоссарий, принципы, архитектура. §5–§9 —
спецификации трёх примитивов и `MemoryGuard`. §10 — реализованные
`Dataset`/`CadwDataFrame`/`RowFilter`. §11 — что работает (11.1), что
удалено (11.2), что осталось (11.3), закрытые must-fix (11.4), **явные
ограничения** (11.5), техдолг (11.6). §12 — дорожная карта. §13 — история
решений и уроки. §14 — резюме.

**Статус v0.1.0.** Каркас production-ready: три примитива с безусловной
гарантией памяти, фасад `CadwDataFrame` с pandas-подобным API, ML-интеграция
(sklearn + LightGBM), E2E. Тесты: **213 fast + 5 slow = 218 passed, 1 skipped**
(torch опционален). Дальнейшие функции (join, drop_duplicates, rank,
rolling) — композиции готовых примитивов, не переделка ядра.

---

## 1. Что такое CADW и для кого

**CADW** — локальный out-of-core фреймворк для табличных данных, не
помещающихся в RAM, на слабых машинах (ноутбук, десктоп без GPU,
без распределённого кластера).

**Типичный пользователь:** data scientist с 50–500 ГБ Parquet, 8–16 ГБ RAM,
желающий запустить `groupby`, `sort_values`, `partial_fit`, LightGBM
без покупки кластера.

**Что CADW не делает:**
- не распределяет вычисления (это Dask/Spark);
- не ускоряет через GPU (это RAPIDS);
- не заменяет pandas для данных, помещающихся в RAM.

**Что CADW делает:** гарантирует, что операция завершится в заданном
бюджете памяти за счёт сброса промежуточных результатов на диск и
потоковой обработки батчами. Гарантия **безусловна** для `Reduce`,
`HashShuffle`, `Sort`, `groupby` (§11.4 закрыт). Для операций,
делегированных внешним библиотекам (LightGBM `lgb.train`), гарантия
ограничена их собственными требованиями (см. §11.5).

---

## 2. Глоссарий

| Термин | Значение |
|---|---|
| **Батч (batch)** | Порция строк, находящаяся в RAM. Размер контролируется `MemoryGuard`. Батч всегда помещается в память. |
| **Чанк (chunk)** | Физический блок строк на диске (Row Group Parquet, `RecordBatch` Arrow IPC). Может быть больше батча. |
| **Партиция (partition)** | Файл, куда `HashShuffle` сложил строки с одинаковым хэшем ключа. Не пересекаются по ключам. |
| **Run** | Отсортированный файл на диске (этап 1 внешней сортировки). |
| **`row_id`** | Глобальная позиция строки в исходном датасете (0..N-1). Сохраняется во всех промежуточных структурах. |
| **Примитив** | `Reduce`, `HashShuffle`, `Sort`. |
| **Моноид** | Ассоциативная операция слияния + нейтральный элемент. Если агрегат — моноид, считается за один проход. |
| **`RowFilter`** | Ленивый фильтр строк: predicate / slice / and / or / not. **Реализован** (03c). |
| **`Dataset`** | Хранилище: per-column Arrow IPC, синхронные чанки, `add_column`, `iter_batches`. **Реализован** (03a). |
| **`CadwDataFrame`** | Пользовательский фасад (pandas-подобный). **Реализован** (03b/03c). |
| **`CadwIndex`** | Слой идентичности строк (`RangeIndex` / `MaterializedIndex` / `DeleteMask`). **Не реализован** (§11.3); текущая идентичность — `row_id` в shuffle/sort + позиционный порядок в `Dataset`. |

---

## 3. Ключевые принципы (обязательные)

1. **Запрет накопления в памяти.** В любой момент — не более одного батча плюс контролируемые буферы.
2. **Неизменяемость.** Файлы не перезаписываются. Новая версия колонки — новый файл.
3. **Apache Arrow** — единый формат в памяти и для кэша. Источник — Parquet.
4. **Pandas-подобный API.**
5. **`MemoryGuard` в байтах.**
6. **Ленивость.** Срезы/фильтры — ленивые; физическая перестройка — по явному запросу.
7. **Три примитива.** Тяжёлое сводится к `Reduce`, `HashShuffle`, `Sort`. Остальное — композиции.
8. **Честный фасад.** `CadwDataFrame` не реализует алгоритмы; guard-тесты
   (`test_facade_has_no_as_py`, `test_facade_has_no_row_loop_ast`) ловят
   регресс на уровне AST.
9. **Никаких ML-адаптеров.** CADW не подключает сторонние ML-модели
   через адаптеры, ни через `fit`, ни через `partial_fit`.
   - `fit`-модели (`lgb.train`, `PCA.fit`, `XGBRegressor.fit`,
     `HistGradientBoosting.fit`) требуют весь датасет в RAM —
     OOM происходит **внутри** библиотеки, `MemoryGuard` не спасает.
   - `partial_fit`-модели (`SGDClassifier`, `StandardScaler`,
     `MiniBatchKMeans`, `IncrementalPCA`) **могут** быть O(1)-state,
     но whitelist — политика на доверии: любая модель вне whitelist
     или модель, чей state вырос, OOM-ит без предупреждения от CADW.
     Ответственность за state модели не должна перекладываться на
     CADW.
   - **Единственный ML-интерфейс CADW — `to_batches(format=...)`.**
     Это экспорт: CADW выдаёт батчи, пользователь сам решает. Если
     его модель OOM-ит — это его модель, CADW не создаёт иллюзию.
   - ML-функции, которые действительно нужны в out-of-core, пишутся
     на примитивах (v0.3: GBDT, Linear, k-means, PCA, NB).

---

## 4. Архитектура: уровни и модули

### 4.1. Уровни

```
┌─────────────────────────────────────────────────────────────┐
│ 6. Экспорт: to_batches(format='arrow'|'pandas'|'numpy'|'torch') │
├─────────────────────────────────────────────────────────────┤
│ 5. CadwDataFrame: pandas-подобный API + ленивый __getitem__  │
├─────────────────────────────────────────────────────────────┤
│ 4. Операции: groupby, unique, sort_values                    │
├─────────────────────────────────────────────────────────────┤
│ 3. Примитивы: Reduce | HashShuffle | Sort                    │
├─────────────────────────────────────────────────────────────┤
│ 2. Доступ: RowFilter (predicate/slice/and/or/not)            │
├─────────────────────────────────────────────────────────────┤
│ 1. Dataset: per-column Arrow IPC, синхронные чанки           │
├─────────────────────────────────────────────────────────────┤
│ 0. Хранение: Parquet (источник), Arrow IPC (кэш, партиции)   │
└─────────────────────────────────────────────────────────────┘
```

### 4.2. Фактические модули

```
cadw/
├── io/
│   ├── batch_reader.py      # ParquetBatchReader, ArrowIPCBatchReader
│   └── batch_writer.py      # ParquetBatchWriter, ArrowIPCBatchWriter
├── core/
│   ├── memory_guard.py      # бюджет, check_batch/check_state/check_partition_buffers
│   └── logging.py           # get_logger, configure
├── dataset/
│   ├── dataset.py           # Dataset: per-column IPC, sync chunks, iter_batches
│   └── row_filter.py        # PredicateFilter, SliceFilter, AndFilter, OrFilter, NotFilter
├── ops/
│   ├── reduce.py            # ReduceOp, reduce(), CountOp..HistogramOp
│   ├── shuffle.py           # compute_hashes, _mod_p, hash_shuffle, estimate_num_partitions
│   ├── sort.py              # external_sort, _merge_keys, _Key, compute_fan_in
│   └── groupby.py           # groupby_aggregate (HashShuffle + LocalGroupBy + Concat)
├── dataframe.py             # CadwDataFrame (фасад)
└── __init__.py
```

Legacy (`_legacy/`) удалён в патче 05; guard в `tests/test_architecture.py`
ловит регресс.

### 4.3. Сквозной пример: `df[df['val'] > 15].groupby('cat').agg('val','mean')`

Данные (8 строк, `cat ∈ {A,B}`, `val = 0..7`):

1. `df['val'] > 15` → `PredicateFilter('val', 'gt', 15)` — **не читает данные**.
2. `df[...]` → новый `CadwDataFrame` с `_row_filters=[pred]`.
3. `.groupby('cat').agg('val','mean')`:
   - `_iter_filtered_batches()` → батч за батчем, `b.filter(mask)` per batch.
   - `groupby_aggregate(...)`: sample → K_est → P → `hash_shuffle` →
     `_local_groupby` per partition → `pa.concat_tables`.
4. Результат: новый `Dataset`, обёрнутый в `CadwDataFrame`.
5. Пик RAM: один батч + состояние агрегации на партицию (O(K/P)).

---

## 5. Примитив 1: `Reduce`

### 5.1. Назначение и граница

Свёртка потока батчей в один аккумулятор. Состояние **не зависит от N**.

| Подходит | Не подходит |
|---|---|
| `count/sum/min/max/mean` | `median`, точные `quantile` |
| `var/std` (Welford) | `nunique`, `mode` |
| `cov/corr` (d×d) | `distinct` |
| `histogram` (binning LightGBM) | |
| `partial_fit`-состояние | |

### 5.2. Контракт

```python
class ReduceOp:
    def init(self) -> State
    def update(self, state, batch) -> State
    def merge(self, a, b) -> State
    def finalize(self, state) -> Any
    def state_nbytes(self, state) -> int

def reduce(batch_iter, op, memory_guard, *, log=True) -> Any
```

**Правило:** `update`/`merge` — только векторизованные (`pa.compute`, `np`).
Построчные `as_py()` запрещены.

**Инвариант:** результат не зависит от `batch_size` (проверяется параметризованными тестами).

### 5.3. Числовой пример (mean, batch_size=2)

`x = [10,20,30,40,50]`. State `(sum, count)`:

| Шаг | В RAM | state |
|---|---|---|
| init | — | `(0, 0)` |
| batch #1 | 2 числа | `(30, 2)` |
| batch #2 | 2 числа | `(100, 4)` |
| batch #3 | 1 число | `(150, 5)` |
| finalize | — | `30.0` |

**Reduce — единственный примитив без записи на диск.** Если реализация
пишет — архитектурная ошибка (тест `test_reduce_does_not_spill`).

### 5.4. Реализации

| Операция | Состояние | Замечание |
|---|---|---|
| `CountOp` | `(n_valid, n_null)` | skipna |
| `SumOp` | `(sum_f64, n_valid)` | промоушен float64; int64 overflow не обрабатывается (§11.5) |
| `MinMaxOp` | `(scalar, valid)` | `pc.min_max` |
| `MeanOp` | `(sum, n)` | — |
| `StdOp` | `(n, mean, M2)` | Welford (Chan et al.) |
| `CovarianceOp` | `(n, sum_x[d], sum_xx[d,d])` | O(d²); при большом d — §11.5 |
| `HistogramOp` | `counts[B]` | требует min/max (двухпроходный) |

Welford: на `1e9 + [1,2,3]` даёт `std=1.0`; наивная формула — мусор.
Числовая устойчивость — часть контракта.

---

## 6. Примитив 2: `HashShuffle`

### 6.1. Инвариант

Строки раскладываются по P файлам так, что все строки с одинаковым ключом —
в одном файле. Партиции **не пересекаются по множеству ключей**
(хэш-коллизии разных ключей — норма; downstream различает по значению).

Применяется для: `groupby`, `join`, `unique`, `drop_duplicates`,
`set operations`, `pivot`, `target encoding`, `stratified sampling`.

### 6.2. `row_id` — обязательное поле

Каждая строка несёт `row_id` (глобальная позиция в исходном датасете).
Критическая деталь: при `pc.sort_indices` `row_id` строки на позиции `i` —
это `order_np[i] + offset`, а не `i + offset`.

### 6.3. Детерминированный хэш

- `xxhash.xxh64(_stable_bytes(v), seed).intdigest()`.
- `_stable_bytes(v) = repr(v).encode('utf-8')` — различает типы.
- Multi-key: boost `hash_combine` — `h ^= ch + 0x9E3779B97F4A7C15 + (h<<6) + (h>>2)`.
- Арифметика — NumPy `uint64` (Arrow `pc.add`/`pc.shift_left` возбуждают
  `ArrowInvalid` на overflow).
- NULL хэшируется детерминированно, отдельная группа.

### 6.4. Выбор P

1. **Sample:** `K_est = K_sample * N / s`; `P = ceil(K_est * safety / gpp)`,
   `safety=1.5`, `gpp = available*0.6 / bytes_per_key`.
2. **Рекурсия** при переполнении партиции (02b1): `_local_groupby` при
   `sum(b.nbytes) > available*0.5` вызывает `hash_shuffle(seed=depth+1)`,
   `MAX_DEPTH=5`; при превышении — `MemoryLimitExceeded` с указанием на skew.
3. **Skew-изоляция** (02b2): `_detect_heavy_keys` по sample (>5%) →
   тяжёлые агрегируются потоково (`_accumulate_heavy`), лёгкие — обычный
   shuffle + `_local_groupby`; конкатенация.

### 6.5. Числовой пример (P=4)

При `h(A)%4=0, h(B)%4=1, h(C)%4=2, h(D)%4=3`:

| partition | строки |
|---|---|
| 0 | (A,10,0), (A,11,4) |
| 1 | (B,20,1), (B,21,5) |
| 2 | (C,30,2), (C,31,6) |
| 3 | (D,40,3), (D,41,7) |

`A` только в 0, `B` только в 1, … — агрегация независима, конкатенация.

### 6.6. Тесты (13/13)

Инвариант «for each key, all rows in exactly one partition»;
`row_id` восстанавливает исходные данные; P>N; P=1; пустой вход → `ValueError`;
стабильность между батчами; seed меняет хэш; `estimate_num_partitions`;
`check_partition_buffers` срабатывает.

---

## 7. Примитив 3: `Sort`

### 7.1. Назначение

Внешняя сортировка: `sort_values`, `rank`, точные `quantile`/`median`,
окна с `ORDER BY`, `head(n)`/`tail(n)`.

### 7.2. Стабильность

Каждая строка в run'е несёт `row_id`. Tie-break при равных ключах — `row_id`.

### 7.3. Алгоритм

**Этап 1.** Батчи до `available_bytes * 0.5`; `pc.sort_indices(...)`;
run в Arrow IPC (все нужные колонки + `row_id`). Run содержит
**объединение** `value_columns` и `key_columns` без дублей — иначе
`Multiple matches for FieldRef.Name(val)`.

**Этап 2.** k-way merge. `k = compute_fan_in(available, read_batch_size,
row_size)` (02a). Heap по `_Key` (null_rank + per-column asc/desc), tie-break
`row_id`. Многоуровневый merge при `run'ов > k`.

### 7.4. Числовой пример

`val = [40,10,30,20,50,5,15,25]`, batch=2. 4 run'а → merge:
`5,10,15,20,25,30,40,50`; `row_id`: `5,1,6,3,7,2,0,4`.

### 7.5. Тесты (10/10)

Asc/desc; стабильность; multi-key; nulls_first/last; многоуровневый merge;
уже отсортировано; пустой вход → `ValueError`; `row_id` уникален.

---

## 8. Почему три примитива, а не N

| Операция | Композиция |
|---|---|
| `mean/sum/std/cov/histogram` | `Reduce` |
| `groupby().agg()` | `HashShuffle` + LocalGroupBy + `Concat` (реализовано) |
| `unique`, `drop_duplicates` | `HashShuffle` + first + `Concat` (unique реализован) |
| `sort_values` | `Sort` (реализовано) |
| `merge`/`join` | `HashShuffle` обеих сторон + merge в партиции + `Concat` |
| `pivot`/`crosstab` | `HashShuffle` + агрегация + `Concat` |
| `rank`, точные `median`/`quantile` | `Sort` + локальный ранг |
| окна `ORDER BY` | `HashShuffle` (PARTITION BY) + `Sort` + локальное окно |
| `rolling`/`resample` | `HashShuffle` + `Sort` + halo-буфер |

**Не сводится:** cross join (O(m×n)); произвольный `apply` над группой,
требующий всю группу (материализация); экстремальный skew (изоляция
тяжёлых ключей — реализована для sum/count/min/max/mean).

---

## 9. `MemoryGuard`

### 9.1. Размер батча

```
batch_rows = floor(available_bytes * safety / (row_bytes_input * expansion_factor))
```
`safety=0.7`; `expansion_factor` — по плану (`filter/project ≤ 1.0`,
`with_column = 1 + bytes_new/bytes_row`, неизвестная — `2.0`).

### 9.2. Интерфейс

| Метод | Назначение |
|---|---|
| `available_bytes()` | текущий бюджет |
| `get_batch_rows(row_bytes, ef)` | верхняя граница |
| `min_batch_rows()` | нижняя (≥1000) |
| `check_batch(batch, name, extra_factor=1.0)` | `batch.nbytes * ef ≤ available` |
| `check_state(state_bytes)` | `state ≤ available * 0.5` |
| `estimate_state_size(op, d, bins)` | до запуска |
| `estimate_unique_keys(sample, total)` | выбор P |
| `check_partition_buffers(P, buffer)` | `P*buffer ≤ available * 0.3` |
| `recommended_partition_buffer_bytes(P)` | `buffer_limit // P` |
| `log_step(...)` | подробное логирование |

**Инвариант:** если `check_*` падают — операция **не продолжается**.

### 9.3. Параметры по умолчанию

`safety=0.7`, `state_safety=0.5`, `buffer_safety=0.3`, `min_batch_rows=1000`,
`default_expansion_factor=2.0`, `bytes_per_key=80`.

---

## 10. Dataset, RowFilter, CadwDataFrame (реализованы)

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

## 11. Степень реализации

### 11.1. Что работает (проверено тестами)

| Модуль | Статус | Тестов |
|---|---|---|
| `ParquetBatchReader`/`Writer`, `ArrowIPCBatchReader`/`Writer` | ✅ | legacy |
| `MemoryGuard` (байтовый контракт) | ✅ | 25 |
| `Reduce` (`count/sum/min/max/mean/std/cov/histogram`) | ✅ | 33 |
| `HashShuffle` (`xxhash`, `row_id`, `_mod_p`) | ✅ | 13 |
| `Sort` (k-way, stable, multi-level) | ✅ | 10 |
| `groupby_aggregate` (авто-P, рекурсия, skew) | ✅ | 10 + 12 + 5 + 11 (must-fix) |
| `Dataset` (per-column IPC, sync chunks) | ✅ | 13 |
| `CadwDataFrame` (фасад) | ✅ | 17 |
| `RowFilter` + ленивый `__getitem__` | ✅ | 14 + 11 |
| E2E (04, filter→groupby→sort→parquet, 20 MB) | ✅ | 2 (slow) |
| ML: `to_batches(format)` + `partial_fit` | ✅ | 11 + 1 (slow) |
| LightGBM `to_lightgbm(max_rows)` | ✅ | 10 + 1 (slow) |
| Экспорт батчей: `to_batches(format=...)` | ✅ | (в CadwDataFrame) |
| ~~`partial_fit`~~ | ❌ удалён (09: whitelist — политика на доверии, §3.9) | 0 |
| ~~LightGBM `to_lightgbm`~~ | ❌ удалён (08: O(N), §3.9) | 0 |
| Architecture guards (legacy, AST) | ✅ | 3 |
| API availability guard (07, 18 assertions) | ✅ | 18 |
| **Итого** | | **213 fast + 5 slow, 1 skipped** |

**Ключевые результаты:**
- `groupby` при `K > памяти`: `test_groupby_many_groups` (1000 групп, 200 KB бюджет, 16 партиций).
- Skew-изоляция: `test_groupby_90_10_skew`, `test_groupby_one_hot_skew` при бюджете 200 KB.
- E2E 200k строк / 20 MB: `test_e2e_filter_groupby_sort` + pandas-equivalence.
- ML E2E: sklearn pipeline (scaler→sgd), LightGBM pipeline (50k строк / 10 MB budget с `max_rows`).

### 11.2. Legacy удалён (патч 05)

`cadw/_legacy/`, `tests/_legacy/` удалены. Замены:

| Было | Замена |
|---|---|
| `GroupByAggregator` (построчный цикл) | `groupby_aggregate` |
| `UniqueCollector` (терял первый батч) | `df.unique()` |
| `SpillManager` (`merge` перезаписывал) | `hash_shuffle` (партиции не пересекаются — merge не нужен) |

Guard: `tests/test_architecture.py::test_main_code_does_not_reference_legacy`,
`test_no_legacy_package`, `test_legacy_tests_removed`.

### 11.3. Что ещё отсутствует

- `CadwIndex` (`set_index`, `reset_index`, `loc`, `iloc`) — сейчас идентичность
  строк обеспечивается `row_id` в shuffle/sort и позиционным порядком в
  `Dataset`. `loc/iloc` — кандидат в v0.2.
- `join`/`merge`, `drop_duplicates`, `pivot`, `rank`, `rolling`/`resample` —
  композиции готовых примитивов, не написаны.
- `Plan` (ленивый план) для `estimate_expansion_factor` в `MemoryGuard` —
  сейчас `expansion_factor` передаётся вручную или берётся дефолт `2.0`.
- Параллелизм чтения колонок — не реализован.
- `nulls_first` per-column в `Sort` — глобальный флаг.
- CLI-настройка `configure()` логирования.
- Автокодирование категорий (`df.encode_categorical`) — пользователь кодирует сам.
- `scripts/check_api.py` (pre-flight guard для API PyArrow/sklearn/LightGBM).

### 11.4. Must-fix §11.4 — ✅ закрыто (02a–02b2)

Все 5 must-fix закрыты. Гарантия out-of-core для `Reduce`, `HashShuffle`,
`Sort`, `groupby` — **безусловна** при условиях §11.5.

| # | Пункт | Патч | Итог |
|---|---|---|---|
| 1 | `_local_groupby` без `check_batch` + рекурсия | 02b1 | `sum(b.nbytes) > available*0.5` → `hash_shuffle(seed=depth+1)`, `MAX_DEPTH=5`; sample-буфер тоже под бюджетом |
| 2 | `num_partitions` жёстко 8 | 02a | sample → `estimate_num_partitions` → `max(8, P_est)` |
| 3 | skew (один ключ 90%) | 02b2 | `_detect_heavy_keys` (>5% sample) → `_groupby_with_skew`: тяжёлые потоково, лёгкие shuffle; single-pass |
| 4 | `merge_fan_in` эвристика `*128` | 02a | `compute_fan_in(available, read_batch_size, row_size)` |
| 5 | `compute_hashes` overhead `to_pylist` | 02a | `check_batch(..., extra_factor=3.0)` |

### 11.5. Осознанные ограничения (документированы)

Принцип памяти не нарушается, но поведение отличается от «идеального»
или ограничено API внешней библиотеки.

| Ограничение | Где | Обход / статус |
|---|---|---|
| `sum` промоутит в `float64`; int64 overflow не обрабатывается | `SumOp` | для точных int-сумм — своя реализация с `int128`/Decimal |
| `cov` состояние O(d²); при d=10⁴ — 800 MB | `CovarianceOp` | блочный Reduce по столбцам / PCA до cov |
| `corr`, `t-digest`, `nunique`, `median`, `quantile` (точные) — не реализованы | `ReduceOp` | `median`/`quantile` — через `Sort`; `nunique` — через `HashShuffle + count` |
| `nulls_first` — глобальный флаг, не per-column | `Sort` | per-column — расширение `_Key` |
| Пустой вход → `ValueError` | `hash_shuffle`, `external_sort`, `groupby_aggregate` | schema явно — не реализовано |
| Агрегации ограничены `{mean, sum, count, min, max}` | `groupby_aggregate` | `std`/`var` — через `Reduce` (sum, count, M2) — не подключено |
| Хэш-коллизии разных ключей — норма | `HashShuffle` | downstream различает по значению |
| `compute_hashes` — Python-цикл | `HashShuffle` | векторизация в §13.5 |
| `shape` на отфильтрованном `CadwDataFrame` — upper bound | `CadwDataFrame` | точное — `.count()` (сканирует) |
| `df['a'] in [...]` через `==` возвращает `PredicateFilter`, не bool | `_ColumnProxy` | `df['a'].isin([...])` — кандидат v0.2; как в pandas |
| `df[-5:]` (отрицательный slice) — не поддержано | `SliceFilter` | `step > 0`, `start >= 0`; для отрицательных — `count()` + пересчёт |

**Политика ML (после патча 09).** CADW **не является ML-фреймворком**.
Он предоставляет примитивы (Reduce, HashShuffle, Sort) и фасад
(CadwDataFrame). ML-функциональность делится на три класса:

1. **Экспорт батчей** (`to_batches(format='arrow'|'pandas'|'numpy'|'torch')`):
   поддерживается. Ответственность за OOM в сторонней модели — на
   пользователе. CADW не создаёт иллюзию, что «обучил на больших данных».

2. **Адаптеры к ML-моделям** (`fit`, `partial_fit`, `lgb.train`,
   `PCA.fit`, `SGDClassifier.partial_fit`): **не поддерживаются** (§3.9).
   Причины: `fit` — O(N)-state, OOM внутри библиотеки; `partial_fit` —
   whitelist на доверии, любая модель вне whitelist или модель с выросшим
   state OOM-ит без предупреждения. Разделение ответственности: state
   модели — её ответственность, не CADW.

3. **Собственные ML-реализации на примитивах** (v0.3+):
   - GBDT (level-wise, аналог `HistGradientBoosting`) —
     classification/regression/ranking/time series;
   - Linear (OLS, Ridge, Lasso, Logistic) — `CovarianceOp` + numpy.linalg;
   - k-means — `HashShuffle` (по кластеру) + `Reduce` (центроиды);
   - PCA — `CovarianceOp` (уже есть) + `eigh`;
   - Naive Bayes — `HashShuffle` (класс,фича) + `Reduce` (counts).

   Итого ~8–11 сессий на весь ML-слой. Каждая функция — композиция
   примитивов, не адаптер, гарантия out-of-core сохраняется.

### 11.6. Технический долг

| Долг | Влияние |
|---|---|
| ~~`FutureWarning: null_placement deprecated`~~ | ~~шум в тестах~~ → **закрыто** (патч 07: per-key `sort_keys`) |
| Combiner (частичная агрегация до записи партиций) отсутствует | объём партиций больше, чем мог бы быть |
| Параллелизм чтения колонок отсутствует | медленнее на SSD |
| `_Key` работает через `as_py()` | медленно для сложных типов |
| `configure()` логирования не вызывается | логи по умолчанию не настроены |
| ~~`scripts/check_api.py` отсутствует~~ | ~~24 эпизода API-контрактов~~ → **закрыто** (патч 07: `tests/test_api_availability.py`, 18 assertions) |

---

## 12. Дорожная карта

### 12.1–12.5. Закрыто в v0.1.0

- §12.1 must-fix — закрыто (02a–02b2).
- §12.2 фасад — закрыто: `Dataset` (03a), `CadwDataFrame` (03b),
  `RowFilter` + ленивый `__getitem__` (03c). `CadwIndex`/`loc`/`iloc` —
  перенесены в v0.2.
- §12.3 legacy — закрыто (05).
- §12.4 доводка — частично: E2E (04) ✅; `nulls_first` per-column, CLI
  `configure()`, параллелизм — отложены.
- §12.5 ML — закрыто (06a `partial_fit`, 06b LightGBM, 06c sklearn extras).

### 12.6. v0.2 (следующее)

1. `join`/`merge` через `HashShuffle` обеих сторон.
2. `drop_duplicates`, `rank`, `quantile` — композиции.
3. `CadwIndex` + `loc`/`iloc` + `set_index`.
4. `scripts/check_api.py` — pre-flight guard (hasattr/signature).
5. `encode_categorical` через `Reduce` + map.

### 12.7. v0.3+

- `rolling`/`resample` через `HashShuffle + Sort` + halo.
- Параллельное чтение колонок.
- Векторизация `compute_hashes`.
- `nulls_first` per-column.
- Combiner в `hash_shuffle`.

---

## 13. История обсуждения и уроки

### 13.1. Почему отказались от предыдущей реализации

Прежний код (`merged_cadw8.py`) содержал `VirtualOperationManager` с
`np.arange(total_rows)` (материализация всех индексов), заглушки
`FilterOperation`/`DropNaOperation`/`DropDuplicatesOperation` (возвращали все
индексы), `chunked_filter.py` (физические копии), `pd.DataFrame` внутри
чанков (overhead). Вывод: «виртуальные индексы в памяти» несовместимы с
принципом «не накапливать».

### 13.2. Антипаттерны аналогичных проектов

| Проект | Антипаттерн | Урок для CADW |
|---|---|---|
| **Dask** | `split_out=1` по умолчанию → OOM; партиции — `pd.DataFrame` | явный расчёт размера батча и P |
| **Spark** | JVM-overhead, GC-паузы | Python + Arrow — предсказуемый контроль |
| **pandas (BlockManager)** | скрытые копии, непубличный API, консолидация | Arrow уже готова |
| **Polars** | до 97% фрагментации при мелких чанках | контролировать `chunk_size` |
| **Lance/Vortex** | сложные page-based форматы | soft-delete + compaction, но на Arrow IPC + Parquet |

### 13.3. Принятые решения и отвергнутые альтернативы

| Решение | Альтернатива | Почему отвергли |
|---|---|---|
| Arrow IPC для кэша | pickle / Lance | медленно/сложно |
| Parquet для источника | CSV | медленно, нет типов, нет row groups |
| `dict[str, ChunkedArray]` в `CadwDataFrame` | единый `pa.Table` | навязывает синхронизацию |
| `HashShuffle + локальная агрегация` | `dict + spill + merge` | merge требует, чтобы K влезало |
| Три примитива | алгоритм на операцию | дублирование |
| `xxhash.xxh64(repr(v).encode())` | `pc.hash` | `pc.hash` отсутствует в PyArrow 24–26 |
| NumPy `uint64` для hash_combine | `pc.add`/`pc.shift_left` | Arrow `ArrowInvalid` на overflow |
| `_unique_columns(value, key)` | передавать как есть | `Multiple matches` |
| Никаких ML-адаптеров (§3.9) | `partial_fit` whitelist | whitelist — политика на доверии; любая модель вне whitelist → OOM без предупреждения |
| Свой GBDT на примитивах (v0.3) | Адаптер к LightGBM / XGBoost / HGB | адаптер O(N) → OOM в сторонней библиотеке; `MemoryGuard` не спасает |
| Свой GBDT на примитивах (v0.3) | Адаптер к LightGBM / XGBoost / HGB | адаптер O(N) → OOM в сторонней библиотеке; `MemoryGuard` не спасает |

### 13.4. Уроки из отладки (по классам)

Все эпизоды сгруппированы по классу. Число в скобках — сколько раз встретилось.

**A. API-контракты (10 эпизодов).**
- `pa.array(..., type_=...)` → `type` (PyArrow).
- `arr.length` → `len(arr)`.
- `pc.hash` отсутствует в PyArrow 24–26 → `xxhash`.
- `xxhash.xxh64(str)` → `TypeError` → `.encode()`.
- `pa.ipc.serialize_schema` отсутствует → `Schema.serialize()`.
- `RecordBatch.group_by` отсутствует → `Table.from_batches(...).group_by`.
- `lgb.Dataset.num_data` — метод, не атрибут → `num_data()`.
- `SGDClassifier.partial_fit` требует `classes=` на первом вызове.
- `PCA` не имеет `partial_fit` → `IncrementalPCA`.
- `IncrementalPCA(n_components=2, random_state=0)` → нет `random_state`; `n_features` в `partial_fit` — нет.

**B. Семантические баги в тестах (4).**
- `test_reserve_release` ожидал исключение при `200 + 300 ≤ 1000`.
- `test_partitions_disjoint_by_key` — «≤1 ключ в партиции» (хэш-коллизии ожидаемы).
- `test_facade_has_no_as_py` — текстовый поиск ловил docstring (фикс: AST).
- `test_to_lightgbm_categorical` — «cat в X, но cat не в X».

**C. Реальные баги в коде (5).**
- `row_id` присваивался по позиции в отсортированном батче, а не `order_np[start]`.
- `Multiple matches for FieldRef.Name(val)` — колонка в value и key (`Sort`, `groupby`).
- `itertools.tee` в sample + `chain(sample, rest)` дублировал sample (count удваивался, mean маскировал).
- `_accumulate_heavy` использовал `RecordBatch.group_by`.
- `mean + явный sum` использовали один ключ `val_sum`.

**D. Архитектурные (5).**
- Sample-буфер в `groupby_aggregate` не проверялся на `available*0.5`.
- `skip_skew` терял `sample_batches` (фикс: `skip_skew=True`, chain сохраняет).
- `test_facade_has_no_as_py` — текстовый vs AST.
- `git rm` не удаляет `__pycache__` — физическое состояние vs guard-тест.
- §11.5 «документированное ограничение» было неверным (PCA `n_features`) — самокритика.

**Обобщение.** ~40% эпизодов — API-контракты внешних библиотек. Меры:
- `scripts/check_api.py` (кандидат v0.2) — `hasattr`/`inspect.signature` в CI.
- Guard-тесты на архитектуру — AST, а не текст; проверка физического состояния, а не только импортов.
- «Документированное ограничение» проверяется тестом; ложные ограничения хуже отсутствующих.

**Эпизод 25 (архитектурный, важный).** `to_lightgbm(max_rows=N)` был
реализован как «честный downsampling» — с явным `max_rows` и документированным
ограничением. Но это **противоречит цели проекта**: пользователь думает, что
обучил модель на 500 ГБ, а фактически — на 1 млн строк. `lgb.train` — O(N)-state
(все градиенты в RAM), `lgb.Sequence` не помогает (LightGBM всё равно строит
гистограммы по всему датасету), форк LightGBM — вне нашей задачи.
**Урок:** адаптер к O(N)-алгоритму — иллюзия out-of-core. `MemoryGuard` не
может предотвратить OOM, который происходит **внутри** сторонней библиотеки,
после возврата управления в её код. Поэтому принцип §3.9: no adapters to
O(N)-algorithms. ML-функции, требующие весь датасет, должны быть
реализованы на примитивах (v0.3: свой GBDT) или не реализованы вовсе.
`partial_fit` — исключение, потому что это **истинно инкрементальный**
паттерн с O(1)-state (whitelist).

**Эпизод 26 (архитектурный, ключевой).** `partial_fit`-адаптер был
следующим шагом после `to_lightgbm`. Аргумент «whitelist безопасных
моделей» (SGD, StandardScaler, MiniBatchKMeans, IncrementalPCA) выглядел
разумным: эти модели имеют O(1)-state. Но whitelist — **политика на
доверии**, а не гарантия. Модель в whitelist может изменить state в новой
версии; новая модель может быть добавлена в whitelist по имени без
проверки. Более того, `partial_fit` создаёт **иллюзию**: пользователь
думает, что CADW «обучил на 500 ГБ», а фактически CADW выдал батчи в
sklearn, который мог OOM-нуть внутри. **Урок:** единственный
безопасный ML-интерфейс — **экспорт батчей** (`to_batches`).
Ответственность за OOM в сторонней модели — на пользователе, а не на
CADW. Всё, что требует out-of-core ML, пишется на примитивах (v0.3). Это
укрепляет §3.9: не только `fit`-адаптеры, но и `partial_fit`-адаптеры
запрещены. `partial_fit` может вернуться только как **собственная
реализация** на примитивах (например, `SGD` на `Reduce`), а не как
обёртка над sklearn.

### 13.5. Открытые вопросы

1. `unique` через `Reduce` при `K ≤ памяти` (порог калибруется бенчмарком).
2. Векторизация `compute_hashes`: `pc.cast` в binary + xxhash по буферу или `np.void` view.
3. Формат спилла для агрегатов: struct vs два массива (`sum`, `count`).
4. Порог compaction: фиксированный процент или адаптивный.
5. Параллелизм чтения: детектировать SSD/HDD.
6. `to_batches(format='torch'/'numpy')`: в итераторе или в операции.
7. Skew-стратегия: изоляция vs рекурсия с новым seed.
8. `shape` на отфильтрованном `CadwDataFrame`: оставить upper bound или считать точно (сканирует).

---

## 14. Резюме для восстановления контекста

- **Работает:** `MemoryGuard` (25), `Reduce` (33), `HashShuffle` (13),
  `Sort` (10), `groupby_aggregate` + must-fix (10 + 28), `Dataset` (13),
  `CadwDataFrame` (17), `RowFilter` (25), ML (sklearn + LightGBM, 27),
  E2E (5 slow), guards (3).
- **Итого тестов:** **213 fast + 5 slow = 218 passed, 1 skipped** (torch опционален).
- **Главный результат:** три примитива с безусловной гарантией памяти;
  `groupby` корректен при `K > памяти` и при экстремальном skew;
  E2E на 200k строк / 20 MB budget и ML pipeline (sklearn, LightGBM).
- **Гарантия полная:** §11.4 закрыт.
- **Честные ограничения:** §11.5 (включая LightGBM downsampling и API sklearn).
- **Отсутствует:** `CadwIndex`/`loc`/`iloc`, `join`/`merge`,
  `drop_duplicates`, `rank`, `rolling`, `Plan`, параллелизм,
  `scripts/check_api.py`, `encode_categorical`.
- **Следующее по дорожной карте:** §12.6 (v0.2) — `join`/`merge`,
  `drop_duplicates`, `CadwIndex`, `check_api.py`, `encode_categorical`.

**v0.1.0 зафиксирован: каркас production-ready. Дальнейшие функции —
композиции готовых примитивов, не переделка ядра.**

**v0.1.2 (патч 08):** `to_lightgbm` удалён (§3.9).

**v0.1.3 (патч 09):** `partial_fit` удалён полностью (§3.9). ML-слой
CADW = `to_batches(format=...)` только. Собственные ML-алгоритмы —
v0.3 (GBDT, Linear, k-means, PCA, NB) на примитивах.