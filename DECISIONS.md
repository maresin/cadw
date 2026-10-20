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
10. **Check before materialize.** Любая операция, читающая весь
    источник в память (`list(reader)`, `Table.from_batches(all)`,
    `to_pylist()` на полном датасете), обязана предварительно
    вызвать `check_state(estimated)` или
    `partition_path.stat().st_size`-проверку. Исключение —
    materialize, ограниченный `batch_size` (один батч, уже
    проверен `check_batch`). Нарушение — must-fix. Enforced
    AST guard'ом `test_no_list_of_whole_reader_outside_helper`.
    Также: `op.init()` может аллоцировать большой state
    (`CovarianceOp(d²)` = 80 GB при d=10^5); `reduce` вызывает
    `estimate_state_size` + `check_state` **до** `op.init()`.
11. **Arrow call memory discipline.** Любой вызов Arrow/pandas/numpy,
    аллоцирующий более O(batch), требует:
    (a) документированной формулы пика в §9.2.N;
    (b) явного `check_state(estimated)` / `check_batch` / `path.stat()`
        **до** вызова, в том же function scope;
    (c) AST guard (`test_arrow_call_memory_guard.py`) — список
        high-risk attrs, каждый проверяется на preceding guard.
    Whitelist: helpers с собственным guard'ом (`_safe_materialize_partition`,
    `_local_dedup`, `_groupby_with_skew`, `hash_shuffle`).
    Нарушение — must-fix. Trust = explicit bound, не «Arrow же оптимизирован».
    См. §13.4/138.
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

## 9. `MemoryGuard`: планировщик памяти

### 9.0. Задача

MemoryGuard решает **обратную задачу**: по `available_bytes` (RAM-бюджет) и
`data_chars` (schema, `N`, `K_est`, `row_size`, `avg_key_size`, `order`)
**выбрать параметры** операции (`batch_size`, `P`, `max_dict_keys`,
`k_fan_in`, `recurse_depth`, размеры буферов), такие что:

    f_op(params, data_chars) <= available_bytes * safety,   safety = 0.7

где `f_op` — **модель памяти** операции (пик RAM). Если решения нет ни при
каких params — `MemoryLimitExceeded` с обоснованием: какая часть `f_op`
превышает лимит, при каких params достигнут минимум, что можно изменить
(увеличить available, уменьшить `K`, pre-aggregate).

**MemoryGuard не задаёт `limit` от данных.** `limit` — вход (свободная RAM).
Данные — тоже вход. Выход — план (params + predicted peak + outcome).

**Никаких литералов в размерах.** Размеры значений — из `pyarrow.types`
(`bit_width`, `avg_len` из sample). Калибровочные коэффициенты (`C_op`,
`C_buffer`, `C_hash`) — из `Calibration` (§9.5), не хардкод. Если литерал
появился в модели — это баг спеки.

#### 9.0.1. Два уровня рекурсии: predict vs execute

Рекурсия в CADW существует на двух уровнях, их нельзя смешивать.

**Planner (predict).** `GroupByMemoryModel.solve` вычисляет `recurse_depth`
по формуле:

    recurse_depth = ceil(log(K_est / P / max_dict_keys) / log(P_recurse))

Это **арифметика на числах**: данные не читаются, материализации нет.
Результат — **upper bound**: максимальное число уровней, которое может
понадобиться.

**Runtime (execute).** `_local_groupby` обходит дерево партиций. На каждом
узле проверяет:

- (a) `file_size <= available * 0.5` — помещается ли файл партиции;
- (b) `estimated_K_i * size_of_key <= available * 0.5` — помещается ли
  hash-таблица `group_by` (must-fix, §11.4).

Если (a) и (b) — `group_by` (terminal), узел материализуется **один раз**
(≤ safety), родитель освобождается после split. Если хоть одно не
выполнено — `hash_shuffle(seed=depth+1)` и рекурсия на подпартициях. В
RAM одновременно — **один узел + его split-буферы** (`P_recurse *
writer_buffer`), не всё дерево.

**Инвариант.** `actual_max_depth <= plan.recurse_depth`. Runtime **не
выходит за** `recurse_depth`. Если на глубине `recurse_depth` узел всё
ещё не влезает — `PlannerError` (внутренний), сигнал «модель сломана», а
не повод продолжать рекурсию молча. Planner предсказал feasibility; если
prediction не сбылся, чинить нужно модель (калибровку `C_op`), а не
обходить её.

##### 9.0.2. Planner upper bound vs runtime per-node

`recurse_depth` — **upper bound**, а не «ровно столько split'ов». Runtime
решает per-node:

- **Лёгкие партиции** (`K_i` мал, `file_size` мал) → `group_by` **без
  split**, даже если `recurse_depth > 0`.
- **Тяжёлые партиции** (`K_i` или `file_size` превышают safety) → split.

Причины per-node (а не uniform split всех до `recurse_depth`):

1. **Неравномерность `hash_shuffle`**: `K_est / P` — среднее, `K_i` —
   фактическое. На skewed/кластеризованных данных `K_i` может быть в
   разы больше или меньше среднего.
2. **Экономия IO**: uniform split даёт пустые подпартиции для лёгких
   корней (лишний `hash_shuffle` + writer flushes).
3. **Defense**: check на каждом узле (a)+(b) ловит ошибку planner'а
   (`C_op` вне `[0.5, 2.0]`), но не маскирует её — `PlannerError`, не
   молчаливый обход.

Per-node — прежде всего **оптимизация** (меньше IO на неравномерных
данных). Защита — побочный эффект: проверки всё равно делаются.

### 9.1. Типы и размеры

    type_size(dtype, sample=None) -> int
    row_size(schema, sample_batch=None) -> int

| dtype | откуда размер |
|---|---|
| `bool` | 1 B (Arrow Array: 1 B/значение при маскировании) |
| `int*`, `uint*`, `float*` | `bit_width // 8` |
| `timestamp`, `date*`, `time*` | `bit_width // 8` |
| `string`, `binary` | `avg_len` из sample + offset (int32: 4 B). Без sample — `ValueError`. |
| `dictionary` | `type_size(value) * nunique_est + type_size(index)` — v0.2 |
| `list`, `struct` | рекурсивно `type_size(children)` + offset — v0.2 |

`avg_len` — **измерение** на sample (`mean(len(x))`), не константа.
Если sample пуст — `ValueError` (нельзя планировать без оценки).

`DataChars` — вход планировщика:

| Поле | Тип | Назначение |
|---|---|---|
| `schema` | `pa.Schema` | для `row_size` |
| `n_rows` | `int` | `N` |
| `row_size_bytes` | `int` | из `row_size` |
| `k_est` | `int \| None` | оценка K (sample), для groupby/shuffle |
| `avg_key_size_bytes` | `int \| None` | средний размер `repr(key).encode()` для `compute_hashes` |
| `size_of_key_entry_bytes` | `int \| None` | tuple key + agg state, для dict groupby |
| `order` | `"random" \| "sorted" \| "clustered"` | подсказка для sample-strategy |
| `extras` | `dict` | расширение (без ломания API) |

### 9.2. Модели операций

Базовый интерфейс:

    class OpMemoryModel:
        op_name: str
        def estimate(params, data_chars) -> int
        def solve(available_bytes, data_chars, *, safety=0.7, op_context=None) -> Params | None
        def describe() -> str

Реестр: `_MODELS: dict[str, OpMemoryModel]`, `register_model(model)`,
`get_model(op_name)`, `list_models()`. Идемпотентная перезапись с warning.

**Контракт `solve`**: возвращает `Params`, такой что `estimate(params) <=
available * safety`, или `None` если такого params не существует.
`MemoryGuard.plan` при `None` поднимает `MemoryLimitExceeded` с
`describe()`.

**Контракт `estimate`**: пик RAM (байт), **верхняя** оценка, не средняя.

#### 9.2.1. `ReduceMemoryModel` — ✅ реализовано (11-A-1)

    f_reduce = batch_size * row_size + 2 * state_bytes

`2 * state_bytes` — старый и новый state на время `update` (merge).

`solve`:
    usable = available * safety
    available_for_batch = usable - 2 * state_bytes
    если available_for_batch < row_size -> None (state не влезает)
    batch_size = available_for_batch // row_size

`op_context = {"state_bytes": int}` — обязателен. `state_bytes` считает
вызывающий через `op.state_nbytes(op.init())` или через
`MemoryGuard.estimate_state_size`. Модель не знает про конкретные `ReduceOp`.

**Размеры state (не литералы, а вычисление из dtype):**

| Op | state_bytes |
|---|---|
| `CountOp` | `2 * sizeof(int64) = 16` |
| `SumOp` | `2 * sizeof(int64) = 16` |
| `MeanOp` | `2 * sizeof(int64) = 16` |
| `MinMaxOp` | `sizeof(dtype) + sizeof(bool)` |
| `StdOp` | `3 * sizeof(float64) = 24` (Welford) |
| `CovarianceOp(d)` | `sizeof(int64) + d * sizeof(float64) + d² * sizeof(float64)` |
| `HistogramOp(bins)` | `bins * sizeof(int64) + 2 * sizeof(float64)` |
| `t-digest` (v0.3) | `size(ε) * sizeof(centroid)`, `size = O(1/ε)` |

**Пример.** `CovarianceOp(d=100)`: state = 80 808 B. `available = 10 MB`,
`safety = 0.7` → usable 7 MB. `batch_size = (7 340 032 - 161 616) / 16 ≈
448 000`. Для `d=10^5`: state ≈ 80 GB > 7 MB → `None` →
`MemoryLimitExceeded("state 80 GB > available 7 MB")`.

#### 9.2.2. `HashShuffleMemoryModel` — 11-A-2

`Params = {batch_size, P, writer_buffer_bytes}`.

    f_shuffle = batch_size * row_size
              + P * writer_buffer_bytes
              + C_hash * batch_size

`C_hash` — overhead `compute_hashes` (`to_pylist`): `repr(v).encode()` +
Python int. Из `Calibration` (v0.2), начально — из
`DataChars.avg_key_size_bytes` + overhead Python int (`sys.getsizeof(int)`,
тоже калибровка).

`solve`:
- `P` — из `K_est` и целевого `max_keys_per_partition =
  available * 0.5 / size_of_key`. `P = max(2, ceil(K_est * safety /
  max_keys_per_partition))`, округление до степени двойки (для `_mod_p`
  fast path).
- `writer_buffer_bytes = available * 0.1 / P`, ограничено снизу
  `C_writer_min` и сверху `C_writer_max` (из калибровки).
- `batch_size = (available * safety - P * writer_buffer) / (row_size +
  C_hash)`.

Если `P * C_writer_min > available * 0.4` — `None` (слишком много
партиций для бюджета).

#### 9.2.3. `SortMemoryModel` — 11-A-2

`Params = {run_buffer_bytes, read_buffer_bytes, k, read_batch_size}`.

    f_sort_phase1 = run_buffer_bytes
    f_sort_phase2 = k * read_buffer_bytes
                  + k * sizeof(heap_entry)
                  + read_batch_size * row_size

`solve`:
- `run_buffer_bytes = available * 0.5` (фаза 1, накопление run)
- `read_batch_target = from_calibration` (начально `max(1000, ...)`)
- `k_target = max(2, available * 0.3 / (read_batch_target * row_size))`
- `k = k_target` (число run'ов, открываемых одновременно)
- `read_buffer_bytes = available * 0.3 / k`
- `read_batch_size = read_buffer_bytes // row_size`

Если `k < 2` — `None` (merge невозможен).

#### 9.2.4. `GroupByMemoryModel` — 11-A-2

Составная: `HashShuffle` (9.2.2) + `_local_groupby` + `concat`.

`Params = {shuffle_params, max_dict_keys, recurse_depth, P_recurse}`.

    f_groupby = max(f_shuffle, f_local_groupby)
    f_local_groupby = partition_bytes + dict_bytes + C_group_by_temp

- `partition_bytes = (K_est / P) * row_size`
- `dict_bytes = (K_est / P) * size_of_key_entry`
  — **must-fix (§11.4)**: сейчас `_local_groupby` проверяет только
  `table.nbytes`, не `dict_bytes`. Для узких строк dict больше файла,
  pyarrow OOM-ит внутри `group_by` после успешного `check_state`.
- `C_group_by_temp` — temp в `pyarrow.group_by` (калибровка; начально
  `2 * output_bytes`).

`solve`:
- `P` из `HashShuffleMemoryModel.solve`
- `max_dict_keys = available * 0.5 / size_of_key_entry`
- если `K_est / P > max_dict_keys`:
  - `recurse_depth = ceil(log((K_est/P) / max_dict_keys) / log(P_recurse))`,
    `P_recurse = 4` (калибровка)
  - если `recurse_depth > MAX_DEPTH (5)` → `None` (K не сводится)
- `P_recurse` — из `Params`, может быть переопределён planner'ом для
  оптимизации IO.

### 9.2.6. `Table.join` — internal peak (from RSS bench)

Measured v0.8.8 (N ∈ {100k, 500k, 1M}, narrow int64 schema):
peak_delta = **34.7 / 72.5 / 114.7 MB** for 2.4 / 12 / 24 MB output
(fit: `5 × output_nbytes + 40 MB`, residual < 3%).

Note: `pa_delta` (Arrow allocation) = 3.4 × output; RSS = 6.85 × output
for narrow schema. RSS is what MemoryGuard checks. Use `5 × output_nbytes`
(conservative for narrow, adequate for wide where baseline 40 MB is
relatively smaller).

**Guard**: `check_state(5 × row_bytes_out × actual_matches)` **before** call.
`row_bytes_out` = `_estimate_row_bytes(schema)` (bit_width//8 + null bitmap).

**Verdict**: SAFE with guard. Currently guarded — this formula is the
canonical reference for future code.

**v0.8.10-pre (variance analysis).** Измерено в 3 раздельных прогонах
`pk join @ N=1M` (int64 key + int64 payload, 24 MB output):
- 114.7 MB (audit first run)
- 142.8 MB (rerun, same binary)
- 156.8 MB (P-A3 live-violation bench, same binary)

Разброс 42 MB на идентичном коде. Источник: Arrow memory pool retention
policy + jemalloc page release timing. Формула `5 × output + 40 MB`
target'ит upper end; при бюджетировании на медиану (`5 × output + 25 MB`)
— на свой риск.

**v0.8.11b (strict enforcement policy).** `merge()` now enforces
`40 MB + 5 × predicted_output <= state_limit` **strictly** (raise) when
budget is large enough. Threshold:

    strict_mode = (budget >= 4 × _JOIN_FIXED_OVERHEAD_BYTES = 160 MB)
                  AND (state_limit >= _JOIN_FIXED_OVERHEAD_BYTES = 40 MB)

Rationale:
- **Budget < 160 MB:** 40 MB fixed overhead is > 25% of budget (> 50%
  of state_limit). Strict check would fire trivially for tiny joins,
  breaking backward compat with `mem_mb=20` tests and making CADW
  unusable on constrained machines. **Warn only**; per-batch guards
  in `_process_partition_pair` still enforce per-batch memory.
  Honest limitation documented (§11.5).
- **Budget >= 160 MB:** overhead ≤ 25% of budget (≤ 50% of state_limit),
  leaving room for meaningful output estimate. **Strict: raise.** Real
  guarantee.
- **Secondary check `state_limit >= overhead`:** protects against
  reservation drop. If `available_bytes` shrinks such that
  `state_limit < 40 MB`, warn (not raise) — raise would fire trivially
  regardless of output size.

**Enforcement note (§11.5 addendum).** v0.8.9 per-batch pre-check —
**per-batch enforcement only**. Whole-operation warning добавлен в
v0.8.10-pre (log WARNING при `40 MB + 5 × predicted_output > state_limit`);
strict enforcement (raise) отложен в v0.8.11 с ревизией тестовых
бюджетов (20 → 100 MB в ~10 файлах).

### 9.2.7. `Table.sort_by`

peak_delta @ 100k/1M/3M = 8 / 28 / 74 MB (input 1.6/16/48 MB).
Fit: `1.5 × input_nbytes + 6 MB`, residual 2.8%.

**Guard**: `check_state(1.5 × table.nbytes)` before call. UNGUARDED at
`viz/_aggregate.py:153` — candidate fix in v0.8.10.

### 9.2.8. `Table.group_by(...).aggregate(...)`

peak_delta @ K=100: 16.3 MB flat; K=10k: 20.5 MB flat; K=N: grows with
output (24.5/32.4/58.4 MB @ 30k/100k/300k).
Fit: `20 MB + 20 × K × key_size + 8 × output_nbytes` (K-bounded, N-independent).

**Guard**: `check_state(20 MB + 20×K×key + 8×output)` before call.
`X × K` term dominates for K > ~10k.

### 9.2.9. `Table.from_batches(list | generator, schema)`

**Generator** (v0.8.9 P-A1 fix): `peak_delta = 0` (zero-copy — measured
`pa_delta = 0.0` across N=100k..3M).
**List**: `peak_delta ≈ file_size` (retained batches; measured
0.93–1.03× file).

**Guard**: gen form — none needed. List form —
`check_state(Σ batch.nbytes)` before or `path.stat().st_size`.

**Verdict**: SAFE. Prefer generator; list-only when source is a Python
list already (or fall back to `check_state`).

### 9.2.10. `pc.quantile(arr, q)` — single column

peak_delta @ 100k/1M/3M = 5.3/11.3/27.3 MB; pa_delta = 0 / 7.6 / 22.9 MB
(≈1× input). Sort-based working copy.

**Guard**: `check_state(1.2 × N × element_size + 5 MB)` before call.
For whole-partition quantile use per-batch + merge, or §9.2.11.
**NEEDS-BATCHED** for columns > 0.3 × avail.

### 9.2.11. `pc.tdigest(arr, delta=...)` — approximate quantile

**`pc.hash_tdigest` does NOT exist in pyarrow 26.0.0** (AttributeError
verified by bench). Working alternative: `pc.tdigest(delta=...)`.

State = **3.5 MB constant** across N=100k..3M. O(1/δ) as claimed.

**Guard**: none (fixed 4 MB reservation). Multiple concurrent tdigests:
reserve `4 MB × count`.

**Verdict**: SAFE (O(1) state).

### 9.2.12. `pc.sort_indices(table | array)`

peak_delta @ 100k/1M/3M = 5.3/14.5/37 MB (table); pa_delta = exactly 8×N
(output = `int64[N]`; no internal buffer).

**Guard**: `check_state(4 MB + 11 × N)` before call. Existing
`buf ≤ 0.2 × avail` in `sort.py` is conservative (safe).

### 9.2.13. `pc.value_counts(arr)`

peak_delta K=100: 5.3 MB flat; K=10k: 9.3 MB flat; K=N: 13.3/15.3/31.3
@ 30k/100k/300k. Fit: `max(5 MB, 8 MB + 90 × K)` for bounded K; O(K)
when K ≈ N.

**Guard**: `check_state(10 MB + 90 × K)` for bounded K; **NEEDS-BATCHED**
for K ≈ N.

### 9.2.14. `pc.index_in` / `pc.is_in`

`index_in` peak_delta @ 100k/1M/3M = 5.7/11.7/17.6 MB → `5×N + 6 MB`
(residual 17.8%, dominated by value-set hash build).
`is_in` = `6 MB` flat (bool output = N bytes; allocated inside Arrow
pool, released → no RSS growth).

**Guard**: `check_state(N × 1 + 2 × K × key_size)` before call.

### 9.2.15. `pc.dictionary_encode(arr)`

peak_delta K=100/1000: 1.3 MB flat; K=N: 7.3/9.3/19.2 @ 30k/100k/300k.
Fit: `max(3 MB, 60 × K)` for bounded K; O(N) when K≈N.

**Guard**: `check_state(max(3 MB, 60 × K) + 4 × N)` before call.
**NEEDS-BATCHED** for K≈N (dictionary degenerates to full column).

### 9.2.16. `pa.concat_arrays(parts)`

peak_delta = output_nbytes exactly (no fixed overhead; residual 175%
against near-zero slope is trivial — fit is `0 × N + 0`).

**Guard**: `check_state(output_nbytes)` before call.

### 9.2.17. `pc.binary_join_element_wise(a, b, sep)`

peak_delta @ 1M = 14.5 MB ≈ output 12.8 MB. Fit: `output_nbytes`.

**Guard**: `check_state(output_nbytes)` before call.

### 9.2.18. `Table.to_pandas()` — string and int

**pandas 3.0.2 defaults to `StringDtype(na_value=nan)`, NOT `object`.**
Measured ratio = `df.memory_usage(deep=True) / table.nbytes`:
- string: **1.138–1.143×** (not the prior-audit claimed 3–5×)
- int: **1.0×** (zero-copy under pandas 3 arrow-backed columns)
- 3–5× figure only reachable with forced `.astype(object)`, which CADW
  does not do.

**Guard**: `check_state(1.5 × table.nbytes)` for `to_pandas` alone.
If surrounding chain includes sort/group/drop (see P-D1 below), use 5×.

**Verdict**: SAFE for `to_pandas` alone; the surrounding chain is
the danger. See §11.6 P-D1 status.

### 9.2.19. `Table.from_pandas(df, schema, preserve_index=False)`

peak_delta ≈ 1.9–3.9 MB total, roughly flat across N; `pa_delta = 0.12 MB`
for 30 MB df (near zero-copy — pandas 3 arrow-backed columns already
backed by Arrow).

**Guard**: `check_state(df.memory_usage(deep=True).sum())` before call.

### 9.3. Pipeline planner

    plan_pipeline(ops: list[Op], data_chars, available) -> PipelinePlan

`PipelinePlan`:
- `steps: list[StepPlan]` — по одной на op: `{op_name, params, estimate_bytes}`
- `peak_bytes: int`
- `spill_boundaries: list[int]` — индексы между steps, где вставлен spill
- `estimated_io: int` — грубая оценка (байт)
- `predicted_outcome: "fits" | "fits_with_spill" | "does_not_fit"`

**Алгоритм:**

1. **Пропагация `DataChars`** через цепочку. `output_chars(op, input_chars,
   params)`:
   - `filter`: `N' <= N`, `k_est'` из sample, `row_size' = row_size`
   - `groupby`: `N' = K`, `k_est' = K`, `row_size'` от агрегата
   - `sort`: `N' = N`, `order = "sorted"`
   - `with_column`: `row_size' = row_size + type_size(new_col)`

2. **Peak по цепочке** = `max(estimate(step_i))`. Если `estimate(step_i) >
   available * safety`, но `estimate(step_{i-1})` можно **сбросить** до
   `estimate(step_i)` — вставить spill-boundary между `i-1` и `i`.

3. **Spill-boundary** = `write_parquet(temp) + read_from_temp` — композиция,
   не новая операция. `peak` после spill = `max(estimate(write),
   estimate(read), estimate(step_i))`.

   **Статус v0.1.x: reserved for v0.2.** `solve` по инварианту
   возвращает params с `estimate ≤ available × safety`,
   поэтому `spill_boundaries` всегда пуст;
   `predicted_outcome ∈ {"fits", "does_not_fit"}`. Если после
   `solve` `estimate > budget` — нарушение инварианта модели
   (`RuntimeError`), а не повод для spill. Реальный spill
   (chunked execution) — v0.2.
4. **Если и после spill** `estimate > available * safety` для какого-то
   `step_i` при **всех** допустимых params → `predicted_outcome =
   "does_not_fit"` + `PipelineMemoryError` с указанием step, estimate,
   available, `describe()`.

5. **`estimated_io`** — сумма по steps: `N_i * row_size_i / throughput`
   (грубая; используется для выбора между closed-form и grid search).

**Тесты pipeline проверяют prediction vs actual:**
- `fits` → операция успешна, `peak_rss <= available * 1.5`;
- `fits_with_spill` → успешна, spill-граница фактически сработала;
- `does_not_fit` → `MemoryLimitExceeded` **до** чтения данных.

### 9.4. Оптимизация

**Closed form** — для `Reduce`, `HashShuffle`, `Sort` (§9.2.1–9.2.3). Одна
формула, O(1).

**Grid search** — если closed form не существует или multi-objective:
- `GroupBy` (композиция) — поиск по `(P, max_dict_keys, recurse_depth)`
  с ограничением `estimate <= available * safety`, минимизация
  `estimated_io` (число проходов × P).
- `Join` (v0.2) — аналогично + выбор build-side.
- `GBDT` (v0.3) — `(num_leaves, num_bins, batch_size)`, grid по
  `histograms_bytes = num_leaves * num_features * num_bins * 2 *
  sizeof(float64)`.

**Multi-op joint optimization** (v0.3+): если соседние stages делят `P`
или `batch_size`, planner решает совместно. В v0.2 — каждый op независимо.

**Контракт grid search:** O(log(range) * #params) шагов; каждый шаг —
`estimate(params)`, не прогон. Если `estimate` дорогой — кэш.

### 9.5. Калибровка

    class Calibration:
        def measure(op, sample, params) -> MeasuredPeak
        def update_coefficients(op_name, measured, estimated)
        def get(op_name) -> Coefficients | None

**Процедура:**

1. Sample `min(10_000, N)` строк, `data_chars = data_chars_from_sample(...)`.
2. Для каждой операции — пробный прогон на sample с `params =
   solve(available, sample_chars)`. Замер `peak_rss` через
   `psutil.Process().memory_info().rss` до/после (или `tracemalloc` для
   allocator-level).
3. `C_op = measured / estimated`. Если `C_op ∈ [0.5, 2.0]` — модель
   верна, коэффициент применяется в сессии. Если вне — warning + попытка
   скорректировать `C_hash`/`C_group_by_temp`; если не удаётся — следующее
   `plan` с этим op поднимает `MemoryLimitExceeded` (модель ненадёжна для
   данной схемы).
4. **Кэш:** `cache_dir/calibration.json`, ключ `(op_name,
   pyarrow_version, schema_hash)`. Инвалидация при смене версии или схемы.
   `schema_hash = sha256(schema.serialize())`.
5. **Когда обязательна:** при первом запуске на новой схеме/pyarrow.
   `MemoryGuard.plan` проверяет `get(op) is None` → `measure`.

**Тест (11-A-4):** `C_op ∈ [0.5, 2.0]` для sample каждой из 4 операций.
Если вне — fail с `C_op`, `measured`, `estimated`.

### 9.6. Гарантии и failure modes

**Инвариант:** перед каждой операцией — `assert estimate(params) <=
available * safety`. Нарушение — bug в planner: `RuntimeError` в
debug-режиме, `MemoryLimitExceeded` в production.

**Runtime monitor:** каждые N батчей — `psutil.Process().memory_info().rss`.
- `rss > available * 1.2` → `logger.warning`
- `rss > available * 1.8` → abort с `MemoryLimitExceeded("runtime RSS
  drift: ...")`

Не заменяет планирование, но ловит дрейф из-за GC (не мгновенный),
pyarrow temp buffers (не в модели), внешних библиотек (не должны, но).

**Failure modes, документированные:**

| Условие | Сигнал |
|---|---|
| `state_bytes > available * 0.5` (`CovarianceOp(d)`) | `MemoryLimitExceeded` до `op.init()`. `describe()` даёт `d`, `state_bytes`, `available` |
| `K_est / P / P_recurse^d > max_dict_keys` при `d = MAX_DEPTH` | `MemoryLimitExceeded` с указанием на skew. Рекомендация: увеличить available, уменьшить K (pre-aggregate), увеличить P_max |
| `P * C_writer_min > available * 0.4` (`HashShuffle`) | `MemoryLimitExceeded`. Рекомендация: уменьшить P (но тогда dict больше — противоречие, значит бюджет недостаточен) |
| `k < 2` (`Sort`) | `MemoryLimitExceeded`. Рекомендация: увеличить available |
| Runtime `depth >= recurse_depth` и узел не влезает | `PlannerError` (внутренний). Модель сломана; нужна калибровка |

### 9.7. Расширение

Чтобы добавить операцию:

1. `NewOpMemoryModel(OpMemoryModel)` с `estimate`, `solve`, `describe`.
2. `register_model(NewOpMemoryModel())` — в `cadw/core/models_<op>.py`.
3. Коэффициенты в `Calibration` (по умолчанию `C_op = 1.0`).
4. Пропагация `output_chars` для `NewOpMemoryModel` — в `planner.py`.
5. Тесты:
   - `estimate(solve(x)) <= x * safety` (unit);
   - прогон на sample → `C_op ∈ [0.5, 2.0]` (calibration);
   - pipeline test с `NewOpMemoryModel` в цепочке (integration).

**Список операций v0.2 (приоритет):**
- `join` (build-side выбор, sort-merge как fallback), модель
  `f_join = build_partition + probe_batch + output`;
- `drop_duplicates` — `HashShuffle + first`, модель как `GroupBy`;
- `rank`/`quantile` — `Sort` + локальный ранг, модель как `Sort`.

**v0.3:** GBDT (`f_gbdt = histograms_bytes + gradients_buffer +
tree_structure`), `num_leaves * num_features * num_bins * 2 *
sizeof(float64)` для гистограмм; градиенты — `N * 2 * sizeof(float64)` на
диске (потоково при tree building).

### 9.8. Статус реализации

| Подраздел | Компонент | Статус |
|---|---|---|
| §9.0 | задача, обратная MemoryGuard | ✅ |
| §9.0.1 | два уровня рекурсии | ✅ (спека) |
| §9.0.2 | per-node vs uniform | ✅ (спека) |
| §9.1 | `type_size`, `row_size`, `DataChars` | ✅ 11-A-1 |
| §9.2 | `OpMemoryModel`, реестр, `Params` | ✅ 11-A-1 |
| §9.2.1 | `ReduceMemoryModel` | ✅ 11-A-1 |
| §9.2.2 | `HashShuffleMemoryModel` | ⏳ 11-A-2 |
| §9.2.3 | `SortMemoryModel` | ⏳ 11-A-2 |
| §9.2.4 | `GroupByMemoryModel` | ⏳ 11-A-2 |
| §9.3 | `PipelinePlan`, `plan_pipeline` | ✅ 11-A-3 (spill — v0.2) |
| §9.4 | оптимизация (closed + grid) | ⏳ 11-A-3 |
| §9.5 | `Calibration` | ⏳ 11-A-4 |
| §9.6 | runtime monitor, failure modes | ⏳ 11-A-4 |
| §9.7 | `register_model` для новых op | ✅ (API готов) |

**Тесты 11-A-1:** 16 в `tests/test_models.py`. Полный прогон: 216 fast + 5 slow = 221 passed, 1 skipped.

**Must-fix §11.4, связанные с §9:**
- (b) `_local_groupby` dict check до `group_by` — 11-A-4;
- `Plan` pipeline для цепочек — 11-A-3.

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
| Materialize-before-check fix (10) | ✅ | (см. test_materialize_guard) |
| dict check в `_local_groupby` (11-A-4a, must-fix §11.4) | ✅ | 5 |
| `hash_join` inner/left (v0.2.2: HashShuffle обеих сторон + pd.merge) | ✅ | 15 |
| `JoinMemoryModel` (v0.2.3: §9.2.5) | ✅ | 17 |
| `propagate_chars("join")` (v0.2.3: planner chain с join) | ✅ | 9 |
| `CadwDataFrame.merge` + `JoinMemoryModel` (v0.2.4) | ✅ | 8 |
| E2E join (v0.2.5: 100k×10k, skew, left) | ✅ | 3 slow |
| `drop_duplicates` (v0.3.0: HashShuffle + local pandas) | ✅ | 19 |
| `rank` (v0.3.1/3.2: Sort + stream; first/min/dense/max/last/average) | ✅ | 34 |
| `quantile` (v0.3.3: Sort + indexed lookup; scalar q, linear) | ✅ | 20 |
| Combiner для `groupby` (v0.3.4: partial agg до shuffle) | ✅ | 12 |
| `fit_lasso`/`fit_elasticnet` (v0.7.7: coord descent) | ✅ | 22 |
| `fit_gbdt_multiclass` (v0.7.8: softmax, per-class) | ✅ | 22 |
| `compute_feature_importance` (v0.7.9: gain/split) | ✅ | 15 |
| `shift`/`diff` on=col (v0.8.2: sort-aware) | ✅ | 17 |
| `rolling(on=col)` (v0.8.4: sort-aware, prototype-validated) | ✅ | 17 |
| `expanding` + `ewm` (v0.8.0: cumulative/EWMA) | ✅ | 26 |
| P-hash: pandas C-hash (v0.8.1) | ✅ | 17 |
| `CovarianceMeanOp` — PCA single-pass (v0.8.0) | ✅ | 3 |
| `CadwIndex`/`set_index`/`reset_index` (v0.4.0a) | ✅ | 26 |
| `iloc` (v0.4.0b: positional indexer) | ✅ | 22 |
| `loc`/`.at`/`.iat` (v0.4.0c: label-based) | ✅ | 31 |
| filter-compat index/set/reset (v0.4.0d) | ✅ | 16 |
| `loc` open-endpoints + duplicate labels (v0.4.1) | ✅ | 22 |
| **P2 + P8 RAM fixes (v0.5.1, from Pingis)** | ✅ | 15 |
| **P5 sort buffer (v0.5.2a, from Pingis)** | ✅ | 2 |
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

**v0.9.1b (DAG fan-in MVP):** `_merge_fanout` требует **одинаковое
row count** во всех branches (positional alignment). Dropna /
row-selection в одной ветви → RuntimeError. Row-selection fan-in
(join по `__orig_rid` с пересечением) — v0.9.1c backlog. Predictor =
sink; нельзя быть input. Forward references запрещены (DAG ациклический
by construction; cycle detection — defensive).

**v0.9.1c (Pipeline save/load):** `to_json` / `from_json` — JSON for
everything, no pickle (safety). GBDT trees ~30 KB per model (100 trees
× depth 3). Large models (1000+ trees) — Arrow IPC sidecar backlog
(v0.9.2). `LambdaTransform` non-serializable (callable). Custom steps
must implement `to_dict`/`from_dict`; `from_json` registry по умолчанию
поддерживает только `CastStep`/`DropnaStep`/`GBDTStep` (`register_step_type`
backlog v0.9.2). Version mismatch — warn; `strict_version=True` для fail.

**v0.9.1d (ColumnTransformer):** column-level fanout только.
- Predictor НЕ может быть inside (model — только final Pipeline step).
- Transformers must preserve row count (dropna внутри → RuntimeError).
  Row filtering — до ColumnTransformer.
- `remainder='drop'|'passthrough'`; sklearn-style `remainder=transformer`
  не поддерживается (list explicitly в transformers).
- Nested state inline in pipeline.json (nested GBDT ~30 KB; extraction
  to separate file — backlog v0.9.2).
- `Pipeline.from_json` uses `_STEP_REGISTRY`; custom steps must be
  registered via `register_step_type(name, cls)`.

**v0.9.1e (apply(fn)):** per-group user callable.
- Memory: per-group bounded (`2 × group_nbytes ≤ 0.3 × available`
  BEFORE `to_pandas`). Oversized → `MemoryLimitExceeded` before `fn()`.
- **fn's own alloc — user responsibility** (§3.9 extended).
- Three modes: scalar, pd.Series, pd.DataFrame. Uniform across groups
  (mixed → RuntimeError).
- Null in group key → own group (dropna=False).
- Output sorted by group key.
- **`transform(fn)` NOT included** — v0.9.2.
- **`fn` returning unsupported type** (list, dict) → TypeError.
- **fn exceptions propagate** — no wrapping.

**v0.9.1f (pivot_table / melt / crosstab):**
- `melt`: streaming, `value_vars` must share one dtype (mixed →
  TypeError, cast first). Output rows = N × len(value_vars).
- `pivot_table`: `k_pairs × row_size × 2 > 0.3 × avail` →
  `MemoryLimitExceeded` **before** groupby (sample-based estimate).
  Missing combos → null. Multi-index supported. Null in `columns`
  dropped (dropna=True).
- `crosstab`: filters nulls in index/columns, delegates to
  `pivot_table(aggfunc='count')`.
- **Sparse pivot**: fail-fast. Wide with N/A filled → user manual
  `fillna(0)` after.
- **melt heterogeneous** → TypeError; use `.astype(...)` first.
- **pivot column names** = `str(value)`. Int → "0", "1", ....

**Sample-extrapolation K_columns overestimate (v0.9.1f):**
`pivot_table` estimates K_index, K_columns via sample
(`MemoryGuard.estimate_unique_keys`), extrapolating linearly to N.
For **low-cardinality periodic data** (e.g., `col = ["c0", ..., "c19"] * 5000`),
sample sees 20 unique in 5000 rows → extrapolates to 400 over 100K
rows, but real unique is still 20. This is a **conservative upper bound**
(safety-first for fail-fast) — not a bug. If user knows `columns`
cardinality is small, budget can be increased or they can `groupby`
long format. Diagnostic message includes `K_columns ~N` — user sees
estimate.

**v0.9.2a (metrics):** `CadwDataFrame.metrics()` — regression
(r2/mse/rmse/mae) + binary log_loss, Reduce-based, O(1) state.
- Requires `y_pred` (regression) / `y_proba` (log_loss). Combined OK.
- Empty → NaN for all.
- Nulls in y_true/y_pred/y_proba → skip per row.
- R² matches sklearn: ss_tot == 0 → 1.0 if perfect else 0.0.
- **Classification metrics** (accuracy/precision/recall/f1) — v0.9.2b.
  Require confusion matrix (K×K state).
- **`roc_auc`** — v0.9.2c (needs sort + cumulative TPR/FPR).
- **Multiclass log_loss** (proba matrix) — v0.9.2b.
- **Per-group metrics** (`groupby(g).metrics()`) — v0.9.2d.
- **y_true numeric** for regression — string labels → `encode_categorical`.

**v0.9.2b (classification metrics):** `accuracy`, `precision`,
`recall`, `f1` (binary + multiclass).
- State: `tp[K] + fp[K] + fn[K]` int64 = `24×K` bytes.
- `average` ∈ `binary|macro|micro|weighted`; auto binary (K=2) / macro
  (K>2).
- `classes` explicit for K>2; K=2 default when None.
- zero_division=0 (sklearn default).
- Mixed regression + classification on same `y_pred` → ValueError.
- Classification path uses `.to_pylist()` per batch; `reduce()` passes
  `check_batch(extra_factor=8.0)` (honest peak).
- Fail-fast: `K × 24 + 64 > state_limit` → `MemoryLimitExceeded`.
- **`roc_auc`** — v0.9.2c (sort-based).
- **Multiclass log_loss** (proba matrix) — v0.9.2c.
- **Confusion matrix** — separate method if requested, not scalar.
- `average='binary'` requires K=2 and `pos_label` in `classes`.

**v0.9.2c (roc_auc + multiclass log_loss):**
- `roc_auc` — multipass (Sort-based, not Reduce). Binary (y_proba scalar)
  or macro (y_proba list K, classes K>2).
- Per-threshold grouping (ties) — matches sklearn.
- Peak RAM = batch + Sort disk. Independent of N.
- NaN if all-pos or all-neg.
- Multiclass log_loss — y_proba list of K, y_true encoded via classes.
- Conflict: roc_auc + regression on same y_true → ValueError.
- **`roc_curve`** arrays (TPR/FPR) — не возвращаются; separate if asked.
- `roc_auc` macro — unweighted (`average='macro'`); weighted — backlog.
- **Binary y_true validation**: y_proba scalar + pos_label → any y_true
  values treated as positive (pos_label) / negative (rest). String
  binary → use multiclass form.

**v0.9.3a (KFold keyed PRNG):**
- `KFold(shuffle=True)` default = keyed splitmix64 on row_id. Fold
  sizes ~binomial (mean n/k, per-fold sigma ~= sqrt(n/k)); not
  exact-balanced. For exact-balanced use `materialize=True`
  (opt-in, O(N) RAM, guarded via `check_state(20*N)`).
- `random_state=None` -> seed=42 (deterministic reproducibility).
- `KFold.split()` calls `df.count()` - one full scan, O(N) time,
  bounded memory.
- Each `split(df)` yields `n_splits` pairs; iterating all folds
  does `n_splits * N` row touches (once per fold per side). O(1)
  memory per iteration.

**v0.9.3c (dedup string keys):**
- `_local_dedup` guard: type-based multiplier.
  - String subset: `12 * table.nbytes` (11.26x measured at K=N,
    +7% safety).
  - Numeric subset: `2.5 * table.nbytes` (1.97x measured at K=N,
    +27% safety).
- **Over-conservative for K<<N strings** (measured ~0x, checked
  against 12x). Budget may need bump for string dedup even when
  K small. Workaround: `encode_categorical` before dedup, or
  v0.9.4 sample-based K_est estimator.
- `pc.dictionary_encode` on all-unique strings copies every value
  (not just dictionary) + Arrow BufferBuilder geometric growth
  (~2x) + int32 indices + hash table. Peak 11x table, not 0.4x.

### 11.5. Осознанные ограничения (документированы)

**v0.8.11b:** join strict enforcement requires `budget >= 160 MB`
(§9.2.6). Below threshold — warn only, per-batch guards still enforce
per-batch memory. For guaranteed strict enforcement, set
`MemoryGuard(max_memory_mb=200)` or higher.

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
| ~~P2: out_batches accumulation (join)~~ | ~~`_process_partition_pair`~~ | **закрыто v0.5.1**: streaming via `hash_join_streaming` |
| ~~P8: 1 MiB hardcode в `estimate_state_size`~~ | ~~`reduce()`~~ | **закрыто v0.5.1**: `ReduceOp.estimate_state_nbytes()` |
| `drop_duplicates` returns rows in partition-concatenation order (v0.8.11a: no intra-partition sort by row_id) | `drop_duplicates` | set-equivalence preserved; `.sort_values(...)` после для детерминированного порядка. §11.5, §13.4/48 |
| ~~P5: Python-list buffer в sort merge~~ | ~~`_merge_keys`~~ | **закрыто v0.5.2a**: `pa.Array.Builder`. Dictionary/list/struct — `NotImplementedError` (honest). |
| `drop_duplicates` без recursion при skew | `drop_duplicates` (v0.3.0) | oversize partition → `MemoryLimitExceeded`; v0.3.1 — recursion + streaming dedup |
| `rank(method='last')` — CADW extension (pandas не имеет 'last') | `rank` (v0.3.2) | Семантика: reverse of 'first' в tie-группе; identity `last(i) = group_first + group_last - first(i)`. Тестируется без pandas-эталона. |
| `groupby_aggregate` combiner не поддерживает std (v0.3.4) | `groupby_aggregate` | warning + fallback на обычный путь. v0.3.5 — Welford partial. |
| `loc[slice]` на duplicate index — `NotImplementedError` (v0.4.1) | `_LocIndexer` | pandas: cannot slice non-unique; v0.4.2 — MultiIndex semantics. |
| `loc[slice]` step != 1 — `NotImplementedError` (v0.4.1) | `_LocIndexer` | v0.4.2. |
| `.at` на duplicate label — `KeyError` (v0.4.1) | `_AtIndexer` | pandas: undefined для duplicates; используйте `loc[[label]]`. |
| `loc` требует `MaterializedIndex` (после `set_index`) | `_LocIndexer` | `RangeIndex` → `TypeError`; используйте `iloc`. |
| `iloc[bool_array]` не поддерживается (v0.4.0b) | `_ILocIndexer` | raise `NotImplementedError`; используйте `df[mask]` (RowFilter). |
| `set_index` материализует колонку (O(N) RAM) | `CadwDataFrame` | Для больших N — future: `RangeIndex` fast path + streaming loc. v0.4.1. |
| Index не сохраняется в Parquet (write_parquet → reload = RangeIndex) | `write_parquet` | Parquet не имеет index-метаданных; документировано. §13.4/55. |
| `fit_lasso` — warm-start / path | `cadw/ml/linear.py` (v0.7.7) | alpha-путь переобучается с нуля; v0.7.8+ |
| `fit_lasso` — `alpha` raw (не sklearn-style /n) | `cadw/ml/linear.py` (v0.7.7) | Совместимо с v0.7.0a Ridge alpha; документировано |
| `feature_importance` default = 'gain' (v0.7.9) | viz adapters | Breaking change: было 'split'. Downstream relying on integer counts — `importance_type='split'` explicit |
| permutation importance / SHAP — не реализовано | `cadw.ml.gbdt` | v0.7.10+ / вне scope |
| `resample` empty bucket → skip (v0.8.7) | `resample` | NaN-fill нет; для gap-fill — `.reindex` после. §13.4/132 |
| `resample` uint64 on → reject (v0.8.7) | `resample` | overflow risk; cast to int64 |
| `resample` date64 through parquet → date32 (v0.8.7) | `resample` | Parquet has no ms-date type; date64 read back as date32. Same day-granularity semantics; use `freq >= '1D'`. §13.4/135 |
| `resample` freq — int seconds / '5s'|'1min'|'1h'|'1D' only (v0.8.7) | `parse_freq` | compound/sub-second/calendar — нет |
| ~~`on` multi-column — composite sort, not per-group~~ | ~~`_shift_on` / `_rolling_on`~~ | **закрыто v0.8.10**: `groupby(...).shift/rolling(on=[...])` |
| `rolling` on=col — single column (v0.8.4) | `_rolling_on` | Multi-column on — v0.8.4a |
| ~~`groupby(...).rolling(...)` — не поддержано~~ | ~~`CadwDataFrame`~~ | **закрыто v0.8.10**: composite-sort, single deque reset-on-group-change |
| `shift`/`diff` on=col — single column (v0.8.2) | `_shift_on` | Multi-column on — v0.8.2a |
| ~~`groupby(...).shift(1, on='ts')` — не поддержано~~ | ~~`CadwDataFrame`~~ | **закрыто v0.8.10** |
| `resample(freq)` — не поддержано | `CadwDataFrame` | timestamp parsing + bucket — v0.8.2 |
| `ewm(ignore_na=False)` — не поддержано (только ignore_na=True semantics) | `ewm` (v0.8.0) | pandas default `ignore_na=False`; наша семантика: nulls skip state, output null для null row |
| `_detect_heavy_keys` absolute min = 100 (v0.3.4) | skew detection | Без абсолютного минимума tiny data (5 строк) ложно флагалась как skew. §13.4/52b. |
| `groupby_aggregate` combiner не работает в skew-пути (v0.3.4) | `groupby_aggregate` | skew → warning + combiner disabled; heavy через streaming, light через shuffle без combiner. v0.3.5 — combiner для light. |
| `quantile` без list q? — НЕТ, list q поддерживается. Без других interpolations ('lower'/'higher'/'nearest'/'midpoint') | `quantile` (v0.3.3) | v0.3.4 — добавление других interpolations (тривиально: другие формулы на v_lo/v_hi). List q уже работает. §13.4/51 |
| `rank` nulls участвуют как значения (pandas skipna не поддерживается) | `rank` (v0.3.1) | v0.3.2: skip nulls (аналог pandas `rank(na_option='keep')`) |

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



| Долг | Влияние |
|---|---|
| ~~`FutureWarning: null_placement deprecated`~~ | ~~шум в тестах~~ → **закрыто** (патч 07: per-key `sort_keys`) |
| Combiner (частичная агрегация до записи партиций) отсутствует | объём партиций больше, чем мог бы быть |
| Параллелизм чтения колонок отсутствует | медленнее на SSD |
| `_Key` работает через `as_py()` | медленно для сложных типов |
| `configure()` логирования не вызывается | логи по умолчанию не настроены |
| ~~`scripts/check_api.py` отсутствует~~ | ~~24 эпизода API-контрактов~~ → **закрыто** (патч 07: `tests/test_api_availability.py`, 18 assertions) |


**Pingis library-memory audit (2026-10-20) — findings closed in v0.8.9:**

- ~~P-A1: list-form `Table.from_batches` in `_safe_materialize_partition`
  and `_local_dedup`~~ — **closed v0.8.9**: generator form (measured
  gen = 0.0× file vs list = 0.93–1.03×).
- ~~P-A3: `Table.join` internal peak = 3.4–6.85× output~~ —
  **closed v0.8.9**: pre-check `5 × row_bytes_out × actual_matches`
  before call, real schema-based row size.
- ~~P-D1: dedup pandas chain unchecked~~ — **closed v0.8.11a**:
  Arrow-native rewrite (Path C: dictionary_encode + group_by + is_in),
  pandas removed. Peak 0.29× of prior pandas chain (proto-validated).
- ~~P-G3: `_groupby_with_skew` opens writers without
  check_partition_buffers~~ — **closed v0.8.9**.
- ~~P-G5: sample cache byte-cap~~ — **closed v0.8.9**.
- ~~P-SH2: uint64 in Python hash fallback~~ — **closed v0.8.9**:
  XOR 2^63 bitcast → int64 → pandas C-hash.

**NEEDS-BATCHED (deferred to v0.8.10+ with prototype):**

- P-Q2 `pc.quantile` whole column (sort-based, ~1× input) — per-batch
  + merge, or `pc.tdigest` (§9.2.11).
- P-R1 `pc.rank` whole column (2× output + sort buffer) — per-batch only.
- P-G2/P-G5 `pc.value_counts` with K≈N — batch it, or `pc.dictionary_encode`.
- P-SH1 `pc.dictionary_encode` with K≈N — same. String shuffle still
  needs research (partition-disjoint at risk).

**NEEDS-RESEARCH (deferred):**

- P-A3 wide-schema benchmark — measured on int64 only; wide string
  deferred to v0.8.10 (N=500k, 5 string cols).
- P-G3 skew workload benchmark — code-review confirmed; workload
  benchmark deferred.


---

### 11.6. Технический долг

(Header restored v0.9.1a-fix1. Original §11.6 header was removed
during v0.8.11/v0.9.0 patches; content below was previously at
end-of-file from v0.9.0b-fix1.)

**v0.9.0b (Pipeline v1):**
- `Pipeline.plan(df)` — **fit feasibility only** (`sum(step.state_bytes)
  ≤ state_limit`). `spill_boundaries` always empty (deferred v0.9.1).
  Per-batch transform guard — **per-step** (each step's `MemoryGuard`
  handles its own batch).
- `Pipeline.transform(df)` **stops before Predictor** (model is last).
  `Pipeline.predict(df)` requires last step to be Predictor.
- `Pipeline.fit(df, target=None)` — target required if pipeline contains
  Predictor. `plan` called first → `PipelineMemoryError` if not feasible
  (honest fail-fast, §9.3).
- **Predictor contract:** `fit + predict` (transform optional; Predictor
  never goes through `.transform()`).
- **State in-memory** (bounded by K × #features per step). `to_json` /
  `from_json` — v0.9.1.
- **`ColumnTransformer`** — v0.9.1 (fanout by columns).
- **DAG** (via `inputs=[...]`) — v0.9.1.
- **`TimeSeriesFeatures`** composite — v0.9.2.

**v0.9.3 backlog (from v0.9.2e code review):**

- **Silent hash fallback** (`cadw/ops/shuffle.py::_pandas_hash_column`):
  `try: ... except Exception: return None` → silent Python hash fallback
  (~30× slower, no warning). Test `test_pandas_path_used_for_int64`
  spies correctly on happy path, but future pandas API changes →
  invisible perf regression. Fix: `logging.warning` on fallback; optional
  `strict=True` to raise. ~10 min.
- **`pa.scalar(pos_label, type=y.type)` arrow-type mismatch**
  (`cadw/ops/metrics.py::_compute_roc_auc_binary`): if user passes
  `pos_label="yes"` with int64 y → `ArrowInvalid`. Rare, but cryptic.
  Fix: upfront `_validate_pos_label_type(y.type, pos_label)` → clear
  error. ~15 min.
- **`_exact_actual_matches_arrow` per-probe-batch recompute**
  (`cadw/ops/join.py`): `build_table.value_counts` computed per probe
  batch (~B× redundant). Fix: cache per partition-pair. Expected
  speedup ~B× on build-side value_counts (real for PK-FK joins). ~1 h.
- **`_coordinate_descent` O(d²) per iter** (`cadw/ml/linear.py`):
  **retracted** — residual maintenance is also O(d²) (BLAS dot vs axpy;
  no asymptotic win). Real win is warm-start over alpha-path
  (feature, not optimization). Defer to Lasso-path feature.


**§11.6 addendum (v0.9.2f observation):**

- **`test_arrow_call_memory_guard.py` whitelist by function NAME,
  not qualname.** `"fit"`, `"transform"`, `"predict"` whitelisted
  globally → `RobustScaler.fit` (calls `df.quantile`) not caught by
  AST guard. Fix: `_collect_violations` walk ClassDef for
  `ClassName.method` qualname; whitelist with qualname keys.
  ~30 min, v0.9.3.
- **RobustScaler.fit — O(d) external_sorts.** For d=100 × n=1M,
  ~17 min fit. For d > 100 prefer StandardScaler/MinMaxScaler.
  Documented in class docstring + §11.5.


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

**Эпизод 27 (must-fix, класс).** 02b1 «закрыл» must-fix п.1
(`_local_groupby` без check) через `sum(b.nbytes for b in batches)`. Но
проверка была **после** `list(ArrowIPCBatchReader(...))` — то есть после
OOM. Тест `test_local_groupby_recursion_called` проверял, что
`hash_shuffle` вызывается, но **не** проверял, что `list(reader)` не
вызывается до него. **Урок:** «закрытие must-fix» требует теста, который
падает при **неправильном порядке** операций, а не только при
отсутствии проверки. AST guard `test_no_list_of_whole_reader_outside_helper`
+ `test_reduce_estimate_state_before_init` фиксируют **порядок**
(`estimate_state_size` до `init`, `stat` до `list`), а не только наличие
вызовов.

**Эпизод 28 (обнаружено аудитом).** `reduce` вызывает `op.init()`
**до** `estimate_state_size`/`check_state`. Для `CovarianceOp(d=10^5)`
`op.init()` = `np.zeros((10^5, 10^5))` = 80 GB — OOM **внутри** init,
до первого `check_state`. Аудит нашёл этот случай через grep по
`materialize`-паттернам (`np.zeros`, `Table.from_batches`, `list(reader)`)
+ анализ `op.init()`. **Урок:** аудит перед новыми функциями (join, GBDT)
находит унаследованные баги. Фикс: `estimate_state_size` перед init
(+ AST guard `test_reduce_estimate_state_before_init`).

**Эпизод 35 (must-fix, dict check).** До 11-A-4a `_local_groupby`
проверял только `file_size <= safety`. Но `group_by` строит hash-таблицу
`K_i × size_of_key_entry`; для узких строк dict больше файла: 10^6
уникальных ключей при `row_size=16` дают file 16 MB, dict 80 MB. При
budget 50 MB: file проходит (16 ≤ 25), dict нет (80 > 25) — OOM **внутри**
pyarrow после `check_state`, который проверял только file. Это третья
форма класса «materialize-before-check»: file check есть, но не тот
объект, который materialize-ится. **Fix (11-A-4a):** `_estimate_distinct_keys`
(product of nunique, upper bound, без построения hash-таблицы) ×
`size_of_key_entry` проверяется до `group_by`; при overflow — in-memory
recursion (не перечитываем с диска: `table.to_batches()`); при
`depth >= max_depth` — `MemoryLimitExceeded`. **Урок:** `check_state`
должен проверять **тот** объект, который реально строится следующей
операцией. File size и dict size — разные объекты; проверять нужно оба.

**Эпизод 36 (первый join, SQL null semantics).** При первом
`inner_join` через `pd.merge` обнаружилось: **pandas.merge по умолчанию
матчит NaN==NaN**, что отличается от SQL (NULL не совпадает). Для
CADW выбрана SQL-семантика: `_drop_null_keys` фильтрует строки с NULL в
on-колонках до `pd.merge`. Без фильтра `null` в key дал бы cartesian
NaN×NaN — скрытый OOM. **Урок:** `pd.merge` — не «SQL join»; на key
колонках NULL — отдельная семантика, её нужно реализовывать явно.
Также: `inner_join` проверяет `len(build) * len(probe) > max_output_rows`
**до** `pd.merge` — защита от cartesian-skew, который pyarrow/pandas
молча материализуют. max_output_rows = 0.3 * available / 256 B
(консервативно); в v0.2.2 — из `JoinMemoryModel.solve`.

**Эпизод 37 (left join и nullable-типы).** Для `how='left'` build = right
всегда (left — preserved), probe (left) не проходит `_drop_null_keys`:
NULL-ключ в left сохраняется с NULL в right-колонках (SQL: NULL не
совпадает, но left preserved). При этом `pd.merge(how='left')` даёт NaN
в right-колонках для unmatched, и `pa.Table.from_pandas` без schema
конвертирует int64 NaN -> float64 — несовместимо с `empty_schema`.
**Fix:** `_empty_result_schema` — `nullable=True` для всех полей;
`out_table.cast(empty_schema, safe=False)` (NaN в nullable-int
конвертируется в null). Fallback — `pa.Table.from_pandas(merged,
schema=empty_schema)`. **Урок:** для left/outer join **все** output
колонки должны быть nullable, иначе materialize unmatched падает на
типах. Right/outer (v0.3) — та же проблема, тот же фикс.

**Эпизод 42 (planner vs runtime: expected vs cartesian).** Модель
`JoinMemoryModel.estimate` считает пик по **ожидаемому** выходу:
`probe_batch × avg_matches × row_size_out × (1+c_merge)`. Runtime `hash_join`
проверяет `len(build_df) × len(probe_df) > max_output_rows` — это
**cartesian worst case** (все ключи совпадают). Расхождение: при
`build=250`, `probe=2187`, `avg_matches=10` модель предсказывает output
~21 870 (влезает), runtime проверяет 250 × 2187 = 546 750 (не влезает).
**Симптом:** `plan_pipeline` говорит `"fits"`, а `merge` поднимает
`MemoryLimitExceeded`. **Причины:** модель оптимистичнее (правильно для
типичных данных), runtime консервативнее (правильно для защиты от
skew). **Статус (закрыто 11-B-2c-3):** разделены `max_output_rows` (expected,
из `output_budget / row_size_out`) и `max_cartesian_rows` (`c_cartesian
× max_output_rows`, default 10×). Runtime проверяет:
- `expected_probe_output = len(probe) × avg_matches_per_probe >
  max_output_rows` → raise (per-partition skew);
- `len(build) × len(probe) > max_cartesian_rows` → raise (extreme skew);
где `avg_matches_per_probe = len(build) / unique_build_keys` — оценка
по **фактической** build-партиции, а не глобальная. Это позволяет
ловить partition-level skew, который глобальная модель не видит.
`CadwDataFrame.merge(use_planner=True)` пробрасывает `num_partitions`,
`max_output_rows`, `max_cartesian_rows` из `mg.plan("join", ...)`.
**Урок:** модель предсказывает **ожидаемый** пик, runtime защищает
от **worst case**. Это два разных числа, их нельзя смешивать в одном
параметре. Плюс — runtime может compute more precise estimate per
partition (unique_build_keys), чем глобальная модель.

**Эпизод 43 (sample-based plan в merge).** `CadwDataFrame.merge` теперь
сэмплирует первый батч с обеих сторон (`_peek_first`), вычисляет `DataChars`
через `_data_chars_for_join` (`k_est` из `MemoryGuard.estimate_unique_keys`
на on-колонках sample, `avg_key_size` из repr().encode() первой строки),
и вызывает `mg.plan("join", left_chars, op_context={"right_chars": ...})`.
Плюсы: пользователь получает раннюю ошибку `MemoryLimitExceeded`, если
план не сходится, а не OOM в середине merge. Минусы: (a) sample — это
первый батч, может быть нерепрезентативным; (b) `k_est` экстраполируется
наивно (`unique_in_sample × n_total / sample_rows`), при сильно скошенном
sample — over-/under-estimate; (c) `_iter_filtered_batches` теперь
вызывается дважды (sample + real), но sample = первый батч, цена
пренебрежима. **Урок:** planner в операции — баланс между точностью
(нужен sample) и стоимостью (лишний проход). Первый батч — разумный
компромисс: не читает всё, даёт порядок величины. Если пользователь
знает cardinality лучше, `use_planner=False` + явный `num_partitions`.

**Эпизод 44 (str.replace silent no-op).** Патч 11-B-2c-3 содержал
две замены: (2d) вставить `unique_build_keys`/`avg_matches_per_probe`
перед `out_batches`, и (2e) заменить cartesian-check на два новых.
Шаг 2e сработал, шаг 2d — **молча нет**: шаблон `old` включал комментарий
`# how='left': пустой build — продолжаем (left preserved).`, которого в
файле после 11-B-2b-fix2 не было. `str.replace(old, new)` при отсутствии
`old` возвращает исходную строку без ошибки. Итог: код ссылается на
`avg_matches_per_probe`, которая никогда не вычисляется. **Fix:** guard
`assert old in s` перед каждым `s.replace(old, new, 1)`, или (лучше)
AST guard на порядок define→use (`test_process_partition_pair_defines_
avg_matches_before_use`). **Урок:** `str.replace` в патч-скриптах —
это DSL без проверки синтаксиса. Любой silent no-op допустим только
если он явно логируется; в нашем случае молчание скрыло баг от
`test_join.py` (там `avg_matches_per_probe` не использовался в
существующих тестах, только в новых из `test_merge_planner.py`).
**Более сильный фикс:** AST guard `define before use` — ловит **класс**
багов, а не конкретный случай.

**Эпизод 45 (facade AST guard scope).** `merge(use_planner=True)`
требует sample first batch → `DataChars` → `MemoryGuard.plan("join", ...)`.
Помощник `_data_chars_for_join` использует `sample.column(c)[0].as_py()` —
O(1) (одна строка на колонку), не O(N). Но был положен в
`cadw/dataframe.py` — AST guard `test_facade_has_no_as_py` проверяет
**весь модуль**, не только класс `CadwDataFrame`, и упал (2 теста:
`test_facade_has_no_as_py` + `test_facade_has_no_as_py_ast`).
**Fix:** вынести `peek_first_batch`/`data_chars_for_join` в новый
`cadw/core/sampling.py` с документированным «as_py здесь O(1)». Facade
(`dataframe.py`) остаётся без `as_py`. Guard-скоуп «весь модуль» —
намеренно строгий: помогает держать `dataframe.py` без любых row-level
примитивов, даже O(1). **Альтернатива** (сузить guard до класса) —
отвергнута: ослабила бы инвариант, который уже ловил регрессии (§13.4
эпизод 4b: `test_facade_has_no_as_py` был текстовым vs AST). **Урок:**
если хочется добавить `as_py` в facade «по-мелочи» — сигнал, что место
кода не в facade. AST guard фиксирует **слой**, а не только намерение.

**Эпизод 46 (E2E join, планировщик предсказал).** 100k FK × 10k PK
(string keys, 20 MB budget): planner видит k_est=10k/10k (sample даёт
реалистичную cardinality), `JoinMemoryModel.solve` выбирает P из
`HashShuffleMemoryModel.solve` обеих сторон, `max_output_rows` из
`output_budget / (row_size_out × (1+c_merge))`. Результат: 100k строк
(каждый left матчит ровно один right PK), RSS-дельта < 200 MB (при
бюджете 20 MB — запас на Arrow/pandas overhead), время < 60 s. Skew-тест
(100k × 100k один ключ) ловится **per-partition** `expected_probe_output
= len(probe) × avg_matches_per_probe` — 10k × 100k = 10^9 >> `max_output_rows`.
RSS-дельта до raise < 500 MB (фактически ~50 MB: build_df 100k × ~24 B +
probe_batch 10k × ~24 B, `pd.merge` не вызывается). Left join (100k ×
5k, half unmatched): 100k preserved, right nulls совпадают с pandas по
null-mask. **Урок:** E2E с string keys на реальном бюджете — единственный
тест, доказывающий, что planner+shuffle+merge укладываются в RAM на
данных, где есть и hash-partitioning, и `pd.merge` temp. 20 MB budget +
100k строк — минимальный масштаб, где все три компонента встречаются
без «тестовых поблажек». Если бы planner не работал, `expected_probe_output`
на одной партиции (10k × 100k = 10^9) уже упал бы на первом batche.

**Эпизод 47 (cartesian — не OOM-граница).** Runtime проверял
`len(build) × len(probe) > max_cartesian_rows` с `max_cartesian_rows =
10 × max_output_rows`. Для PK-FK join 100k × 10k (8 партиций):
cartesian = 5005 × 2519 = 12.6M при **реальном output 2519 строк**.
`max_cartesian_rows = 407k` → ложный `MemoryLimitExceeded`. **Причина:**
`pd.merge` материализует только совпадения, не cartesian. **Fix:** точная
оценка `actual_matches = sum over common keys(build_count × probe_count)`
через `build_df.groupby(on).size()` (один раз) × `probe_df.groupby(on)
.size()` (per batch) × `merge`. O(build + probe) на уникальных ключах —
дёшево. **Урок:** верхняя граница должна быть **звуковой** (≥ output);
cartesian over-reject'ит, exact match — нет.

**Эпизод 47b (regex в патч-скриптах).** Шаг 1d 11-B-2d-fix использовал
`s.replace(old, new)` с многострочным шаблоном. Шаблон не совпал (символы
`×`/`≈`/отступы), `str.replace` молча вернул исходную строку. Файл
остался в промежуточном состоянии: определение удалено (шаг 1c), check'и
ссылаются на несуществующее имя → NameError. **Fix:** regex по маркерам
+ явная проверка «что заменяем». **Урок (усиливает эпизод 44):** для
больших блоков — regex по маркерам, не точный шаблон.

**Эпизод 47c (strict check на docstring).** Шаг 3 проверял `if
"max_cartesian_rows" in s` и падал на **docstring** `merge`. Проверка
включила комментарии/докстринги — over-strict. **Урок:** «нет упоминаний»
— два контракта: «нет в коде» (важно, AST) и «нет в тексте» (обычно
избыточно). Default — AST.

**Эпизод 47d (`raise` до `write_text` теряет прогресс).** fix2 сделал
замены в памяти, но проверка docstring упала **до** `p.write_text(s)` —
файл не сохранён. fix3 увидел те же 3 refs, поправил docstring (в
памяти), снова упал на той же проверке — тоже не сохранил. Итог: две
сессии «делали» одно и то же, файл не менялся. **Fix:** `write_text`
(сохранить) **→** проверить **→** вывести warning, но не raise (или
raise **после** write). **Урок:** патч-скрипт с `try/finally` вокруг
всех замен + `write_text` в `finally`. Или простая дисциплина:
сохраняем, потом проверяем. Потеря прогресса хуже, чем неполный warning.

**Эпизод 48 (drop_duplicates через HashShuffle).** Первая операция v0.3 —
`drop_duplicates` — реализована как композиция `HashShuffle` + локальный
`pandas.drop_duplicates`. Ключевое наблюдение: **invariant HashShuffle**
(«все строки с одинаковым subset key — в одной партиции») делает
локальный dedup эквивалентным глобальному. Не нужен merge, не нужен
финальный `drop_duplicates` — concat даёт корректный результат.

Три режима `keep` реализуются через сортировку partition по `row_id`
(сохраняется в shuffle) и стандартный `df.drop_duplicates`:
- `'first'`: sort row_id asc + keep='first' = min row_id per key;
- `'last'`: sort row_id asc + keep='last' = max row_id per key;
- `False`: pandas drops all rows with duplicate keys — локально = глобально.

**Ordering (документированное ограничение):** результат в порядке
partition-concatenation, **не** в pandas-original order. Для keep='first'
SURVIVING rows правильные (min row_id), но их позиции в выводе — по
партициям. §11.5. Fix в v0.3.1: финальный `external_sort` по row_id, если
нужен pandas-order.

**Skew (документированное ограничение):** при oversized partition (одна
partition > available*0.5) — `MemoryLimitExceeded` **до** materialize.
Recursion требует сохранения ГЛОБАЛЬНОГО row_id при resplit —
`hash_shuffle` сейчас **переприсваивает** row_id (позиция в новом
потоке). Для dedup это ломает keep='first'/'last'. v0.3.1: либо сохранять
row_id в resplit (через `value_columns=["row_id"]` + rename), либо
streaming dedup (dict `key → (min_row_id, payload)` в RAM). Для MVP — raise.

**Урок:** композиция из готового примитива (`HashShuffle`) для новой
операции — дешёвая. Сложность — не в алгоритме (он тривиален), а в
корректности деталей: (a) порядок row_id при keep='first'/'last';
(b) reassignment row_id при recursion; (c) манифест output-order в
документации.

**Эпизод 49 (rank: streamable vs bufferable).** `rank` требует
глобального порядка (sort) и tie-breaking. Разделены методы по
streamability:
- `first`/`min`/`dense` — O(1) state, стримятся в один проход;
- `max`/`average`/`last` — требуют знать конец tie-группы → буферизация
  группы, O(N) при all-equal (патология: 10^9 одинаковых значений →
  10^9 рангов в RAM). Пандас не парится (in-memory), но для out-of-core
  это OOM.

MVP v0.3.1 = streamable trio. v0.3.2 (закрыто): `max`/`average`/`last`
реализованы двухпроходным streaming:
  Pass 1: scan sorted_by → `(first_rank, group_count)` per tie-group
         на диск. State: prev_val/group_start/global_pos + per-batch
         accumulators (O(batch_size)).
  Pass 2: scan sorted_by + advance по group file (sequential).
         pos_in_group = global_pos - first_rank + 1.
         last_rank = first_rank + group_count - 1.
         max → last_rank; last → first_rank + count - pos;
         average → (first + last) / 2.
Итого: O(1) RAM независимо от max tie group size. Плата — два
прохода по sorted_by (I/O) + промежуточный group file
(`N_groups × 16 B`, обычно << N × row_size). API `compute_ranks_to_disk`
принимает Path (не iterable) — файл открывается 1–2 раза.
**Урок:** буферизуемое состояние можно эмулировать O(1) RAM
**двухпроходным** streaming'ом; цена — 2× I/O и промежуточный файл.
Это общий паттерн: `median`/`quantile`/`mode` per-group — та же схема
(счётчики → файл → join). Выбор 'однопроходный streamable' vs
'двухпроходный bufferable' — по принципу: streamable только если
реально O(1)-state; иначе — не притворяться.

**Порядок результата:** rank — свойство исходной строки, поэтому
возвращаем в original order. Реализуется двумя сортировками:
1. sort by `by` (нужно для streamable rank);
2. sort ranks by `row_id` (нужно для original order).
Второй sort — маленький (row_id + rank = 16 B/строка), дешевле первого.
Zip с original — позиционный, batch-by-batch с одним `batch_size`.

**Null values:** участвуют как значения (`null == null` → same group).
Pandas `rank(skipna=True)` → NaN ranks для null строк. Наш MVP этого не
делает — документировано в §11.5. v0.3.2: null-детект per row + emit NaN.

**Урок:** «работает для типичных случаев» — не то же, что «звуковая
верхняя граница». Streamable методы — гарантированно O(1) state.
Bufferable методы — O(max tie group size), где max — параметр данных,
не бюджета; для pathological данных = O(N). Разделение по методам
делает ограничение явным.

**Эпизод 51 (quantile: только что материализовали — что дальше).**
`quantile` — композиция `external_sort` + streamed indexed lookup по
позициям. Ключевая идея: **позиции квантилей зависят только от n_valid
и q**; сканируя sorted поток, мы собираем значения на фиксированных
позициях `{lo_i, hi_i}`. Early exit при `pos > max(targets)` — если все
q маленькие (`q=0.1`), не читаем 90% файла.

**Порядок операций:** sort *до* count (не наоборот) — потому что sort
это один поток, в котором мы параллельно считаем `n_valid` (side-effect
счётчика в генераторе). Иначе нужно два прохода по входным batches
(дорого — данные читаются с диска). Это оптимизация: сортируем +
считаем в одном проходе; затем второй проход — по sorted (I/O на диск
уже сделан).

**Nulls:** `skipna=True` — nulls не участвуют в n и позициях. `arr.
to_pylist()` даёт `None` — пропускаем, `pos` инкрементится только на
valid. Это делает `n_valid` = число non-null, и `idx = q*(n_valid-1)`
корректно покрывает valid-only позиции.

**Interpolation:** в v0.3.3 только `'linear'` (pandas default). Другие
('lower'/'higher'/'nearest'/'midpoint') — те же v_lo/v_hi, другие
формулы. Отложены из-за объёма тестов: каждый вариант — 5+ тестов;
2×20 строк vs 6×20 строк — разница ощутима, но не в функциональности.
v0.3.4: 4 интерполяции + их комбинации с list q.

**Урок:** «работает для scalar q + linear» — это 80% случаев. Оставшиеся
20% (list q, other interp) — расширение, не переделка. МVP: только
базовый путь, документировать остальное как v0.3.4. Для out-of-core
это критично: каждая новая формула — новый раунд тестов на память,
на интерполяцию на границах (q=0, q=1), на nulls.

**Sort-or-ascending-only:** quantile не имеет `ascending` — квантили
инвариантны к порядку сортировки (это про значения, не про позиции в
исходном датасете). `q=0` = min, `q=1` = max; при desc-sort v[0] = max
— пришлось бы инвертировать `idx → n_valid-1-idx`. Проще всегда asc.

**Эпизод 52 (Combiner для groupby).** Паттерн MapReduce: Combiner —
частичная агрегация на стороне map (до shuffle) — уменьшает объём I/O
и RAM в local phase. В CADW: `_combine_batches` перед `hash_shuffle`;
`_local_groupby` с `partial_merge=True` мержит частичные агрегаты.

**Ключевое свойство:** partial aggregates ассоциативны. Для sum/count —
прямой partial (sum of sums, count of counts). Для min/max — direct.
Для mean — sum + count partial, финально sum/count. Партиция после
combiner: `N × row_size` → `K_batch × comb_bytes`. Пример: 10^6 строк,
100 групп → 10 MB → 1 KB (×10000).

**Ограничения v0.3.4:**
1. **std/var** — требуют Welford partial (n, mean, M2) с merge по
   формуле Chan. Реализовано в `StdOp.merge`, но не в combiner.
   v0.3.5: combiner для std через Welford partial.
2. **skew path** — heavy keys уже streamed (`_accumulate_heavy`), combiner
   не нужен. Light part теоретически можно комбинировать, но смешивание
   двух путей усложняет код. v0.3.4: warning + combiner off in skew path;
   v0.3.5 — combiner только для light part.

**API:** `combiner=False` (default) — opt-in через `combiner=True`.
Причины: (a) для высокой кардинальности (`K ≈ N`) combiner не сжимает,
но добавляет overhead pandas.groupby per batch; (b) default-off
сохраняет текущее поведение — никаких сюрпризов; (c) пользователь,
знающий `k_est`, включает явно.

**Почему не auto по `k_est`:** auto требует sample до shuffle — уже
делаем (для skew detection). Но `k_est` из sample — оценка, не факт.
Auto может случайно включить combiner на высокой кардинальности (sample
недооценил) → overhead без выигрыша. Opt-in — предсказуемо.

**Урок:** Combiner — оптимизация, не новая функциональность. API `agg(...,
combiner=True)` — opt-in. Метрика успеха — не «быстрее» (тяжело
измерить в unit-тестах), а «I/O reduced»: партиции после combiner
содержат `K_batch` строк вместо `N`. Проверяется тестом
`test_combiner_large_n_low_k` — но фактическое уменьшение file size —
в логах, не assert. Планировщик (`GroupByMemoryModel`) не трогаем —
с combiner фактический `f_groupby` ниже предсказанного, это безопасно
(safety > 0.7).

**Следующее (v0.3.5):**
- Combiner для std (Welford partial).
- Combiner для light part в skew-пути.
- Auto-combiner по `k_est < N / threshold` (с явным `combiner=None` для auto).

**Эпизод 52b (skew detection без абсолютного минимума = combiner
не тестируется).** Первая версия `test_combiner.py` прошла 12/13, но с
9 предупреждениями «combiner отключён в skew-пути». Причина:
`_detect_heavy_keys` флагил как heavy любой ключ с `ratio > 0.05` — без
абсолютного минимума. На данных 5 строк / 3 ключа (40/40/20%) — все
heavy; на 1000 строк / 10 групп (10% каждый) — все heavy. Combiner
**никогда не активировался**, тесты проходили через fallback (тот же
результат → ложные зелёные).

**Fix:** `_HEAVY_ABSOLUTE_MIN = 100`. Ключ считается heavy только если
`count >= 100 AND ratio > 0.05`. Существующие skew-тесты (`test_mustfix_skew
.py`) не сломались: 90/10 на n=1000 (A=900, B=100) — оба >= 100; 100/1000
uniform — каждый 1 < 100. Combiner-тесты переписаны на n=1000, k=50
(20 rows/group = 2% < 5%) — combiner активен.

**Дополнительно:** `test_combiner_std_warns_and_falls_back` был
некорректным — `std` не входит в `_AGG_FUNCS` и отклоняется **до**
combiner-логики. Заменён на `test_combiner_std_rejected_by_groupby`
(ожидает `ValueError`).

**Новый guard:** `test_combiner_path_actually_used` — monkeypatch
`_combine_batches`, проверяет что путь реально исполняется. Это
template для будущих оптимизаций: **«зелёные тесты не значит —
упражняется новая функциональность»**. Если fallback даёт тот же
результат, тест на эквивалентность проходит без вызова нового кода.

**Урок:** любой `if new_feature_enabled` требует теста, который
проверяет, что `new_feature` **вызывается** (spy/monkeypatch). Иначе
все «equivalence»-тесты могут проходить через старую ветку.

**Эпизод 53 (ChunkedArray vs Array в RecordBatch).** Первая версия
`_combine_batches` использовала `pa.RecordBatch.from_arrays(arrays, ...)`,
где `arrays` собирались из `grouped.column(c)`. `grouped` — `pa.Table`,
`.column()` возвращает **`ChunkedArray`** (может быть несколько chunks
после `group_by().aggregate()`). `pa.RecordBatch.from_arrays` требует
**`Array`** → `TypeError: Cannot convert ChunkedArray to Array`. 11 из 14
тестов упали.

**Fix:** `grouped.select(orig_names).rename_columns(new_names)` даёт
`pa.Table`, `.to_batches()` даёт `RecordBatch` (со всеми chunk'ами,
склеенными автоматически). Один batch на входной batch.

**Правило:** в out-of-core коде мы работаем с `RecordBatch` (батчи
фиксированного размера), но промежуточные pandas-подобные операции
(`group_by().aggregate()`) естественно возвращают `Table`. Склеивать
через `combine_chunks()` или `to_batches()` — не через ручной
`pa.array(...)`. Это тот же класс, что эпизод 44/47b: `.column()`
возвращает не то, что ожидалось; `from_arrays` строже, чем `from_pandas`.
**Эпизод 54 (дубликат в hash_shuffle: key ∩ value ≠ ∅).** `df.unique(col)`
вызывает `groupby_aggregate([col], [col], ["count"])`. В `groupby_aggregate`
`value_columns = [c for c in agg_columns if c not in seen]` → пусто (все
agg в by); fallback `if not value_columns: value_columns = list(by_columns)`
→ `[col]`. Затем `hash_shuffle(key=[col], value=[col], ...)`: `fields =
[key, key, row_id]` — Arrow допускает duplicate field names в schema
(no validation), но downstream `Table.group_by(["col"])` падает с
`Multiple matches for FieldRef.Name(col)`.

**Почему баг существовал, но не всплывал:** `test_unique` был написан
ещё в v0.1.0 и проходил — вероятно, из-за особенностей Arrow 24–25 (или
порядка тестов, где уникальные partition-файлы не триггерили Acero).
Только на PyArrow 26 Acero стал строже валидировать duplicate input
columns. Это класс «latent bug, проявляется при обновлении зависимости».

**Fix:** dedupe `key_columns + value_columns` в `hash_shuffle` (и
`_shuffle_write`) через `list(dict.fromkeys(...))`. Контракт не меняется
— duplicate value column семантически избыточен (value уже присутствует
как key). Функция `_dedup_columns` вынесена в `shuffle.py`.

**Regression test:** `test_groupby_same_col_as_by` — прямой тест
`df.groupby('cat').agg('cat', 'count')`, agg == by. Плюс `test_unique`
уже есть (и теперь зелёный).

**Урок:** API Arrow `Table.group_by()` может быть строже, чем ожидается:
он валидирует duplicate columns в input. `hash_shuffle` — primitive,
который должен быть защищён от неаккуратных caller'ов (caller может
передать `value_columns=["cat"]` когда `"cat"` уже в `key_columns`).
Defensive dedup — обязательный шаг для reusable primitive. Не полагаться
на то, что caller dedup'нет.

**Следующее (v0.3.5):** тест на `hash_shuffle` с explicit duplicate
key/value — должен не падать, dedupe молча. Плюс regression `test_unique`
уже есть (был всегда).

**Эпизод 55 (CadwIndex — MVP без filter-compat).** Первый кусок
`CadwIndex` — абстрактный класс + два наследника:
- `RangeIndex(start, stop, step)` — lazy, O(1) память. `__len__`,
  `__getitem__` (int/slice), `__iter__`, `to_arrow`. Эмулирует pandas.
- `MaterializedIndex(values: pa.Array)` — Arrow Array с типом колонки.
  Для `set_index`; `.equals()` через Arrow `.equals()`.

`CadwDataFrame` хранит `_index_kind` (`"range"` | `"materialized"`),
`_index_start`/`_index_stop` (для range), `_index_values` (для
materialized). Default — `RangeIndex(0, total_rows)`.

**Ключевое ограничение MVP:** index не синхронизируется с активными
фильтрами. `df[mask].index` — `NotImplementedError` (RangeIndex+filter
требует O(N) scan: считать row_id, прошедшие маску). `set_index` и
`reset_index` тоже raise с фильтрами.

**Почему не «просто scan»:** `_iter_filtered_batches` теряет row_id
(отдаёт только `_columns`), поэтому scan нужно делать отдельно. В
v0.4.0b (`iloc`) появится `_iter_filtered_batches_with_positions()`
yield'ит `(batch, positions)` — тогда index через filter = O(N) scan
позиций (не O(N) data). Это приемлемо для `.index` access.

**Почему не index-as-Dataset-column:** можно было добавить `__index__`
в Dataset и позволить filter его фильтровать. Проблемы: (a) колонка
входит в `_columns`, нужен hidden-флаг; (b) при `sort_values` index
пересчитывается (a не сохраняется); (c) `Dataset` immutable — удаление
колонки при `reset_index` = копирование. Итого — сложнее, чем хранить
index в CadwDataFrame.

**`reset_index(drop=True)` для RangeIndex:** no-op (pandas: то же).
`reset_index(drop=False)` для RangeIndex — raise `ValueError` (pandas:
«cannot insert level_0, already have default index»). Это
pandas-совместимое поведение.

**`set_index(inplace=True)`:** mutate `_columns`, `_index_*`. Возвращает
self. `reset_index(inplace=True)` — mutate или создать новый Dataset.
Это единственные «inplace»-операции в CadwDataFrame (facade остальной
immutable).

**Урок:** index — это **отдельный объект идентичности**, не колонка.
Pandas делает его частью BlockManager (O(N) даже для RangeIndex),
Arrow не имеет встроенного index. Хранить в CadwDataFrame с явным
`_index_kind` (дискриминатор) — правильно для ленивой семантики.
MVP без filter-compat — осознанное решение: v0.4.0b (iloc) даст
правильный механизм, и тогда filter+index станет доступным без
«временных» API.

**Следующее (v0.4.0b — iloc):**
- `_iter_filtered_batches_with_positions()` — yield `(batch, positions)`.
- `iloc[i]` — scalar tuple.
- `iloc[lo:hi]` — slice (lazy).
- `iloc[[i,j,k]]` — позиции.
- `filtered.index` — RangeIndex subset через positions (O(N) scan).
- `set_index`/`reset_index` с фильтрами — использовать filtered positions.

**v0.4.0c (loc):** label-based через MaterializedIndex.get_indexer.

**Эпизод 57 (iloc — indexer в отдельном модуле).** `_ILocIndexer` живёт
в `cadw/indexers.py`, не в `cadw/dataframe.py`. Причина: `iloc[int]`
использует `.as_py()` для возврата dict {col: value}. AST guard
`test_facade_has_no_as_py` проверяет **весь** `dataframe.py` и упал бы.
`.as_py()` в `iloc` — O(K), K=1 или len(list); это не row-loop (не O(N)),
но guard не различает. Отдельный модуль — правильно.

**`iloc[slice]` = `df[slice]`.** Слайс-семантика для позиций в filtered
view **идентична** SliceFilter: `df[5:10]` на raw = global positions 5..9;
`df[mask][5:10]` = filtered positions 5..9. Значит `_ILocIndexer._slice`
просто делегирует `parent[slice]`. Нулевой код, полная lazy-семантика.

**`iloc[int]`** — stream-forward-count. Итерируем `_iter_filtered_batches`,
считаем строки; на позиции `i` возвращаем dict. O(i) RAM, O(i) I/O —
пользователь просил конкретную строку.

**`iloc[list]`** — stream-scan, collect target positions. O(K) memory,
O(N) I/O (нужно дойти до последней target). Результат materialize как
новый `Dataset`. Index — preserve parent (MaterializedIndex) если no-filter;
reset на filtered (v0.4.0c — filter-compat).

**Negative indices** — работают. Для slice — через `key.indices(n)`.
Для int/list — resolve через `_view_size()` (count() с фильтрами —
O(N) scan; total_rows без — O(1)). «Работает всегда, цена зависит».

**`iloc[bool_array]`** — raise `NotImplementedError`, use `df[mask]`.
Причина: у нас уже есть `MaskFilter`, `iloc[bool]` семантически =
`df[mask]`, только с другим API. Не дублировать.

**Урок:** indexers — отдельный layer, как `_GroupBy`. Facade остаётся
без data-примитивов. Каждый следующий indexer (`loc`, `.at`, `.iat`)
— сюда же. `dataframe.py` — только property, возвращающее indexer.

**Следующее (v0.4.0c — loc):**
- `MaterializedIndex.get_indexer(labels)` → positions (O(N) scan, dict).
- `iloc[list]` на filtered view — preserve parent index через
  `_iter_filtered_batches_with_positions` (для `filtered.index` тоже).
- `.at[label, col]` / `.iat[pos, col]` — scalar access.
- Duplicate labels — v0.4.1.

**Эпизод 58 (loc — label-based, требует MaterializedIndex).** `_LocIndexer`
— pandas-совместимый label indexer. Работает только на `MaterializedIndex`
(после `set_index`). `RangeIndex` → `TypeError` с подсказкой «use iloc».
Это намеренное ограничение: `RangeIndex.get_indexer(label)` — это
`label - start` при `label` int; для произвольных labels нужна
материализация, что делает range бессмысленным.

**Реализация через `MaterializedIndex.get_indexer(labels)`.** O(N) scan
`values.to_pylist()`, dict `label → first_pos`. Возвращает `list[int]`,
где `-1` = not found. Используется в `loc[label]`, `loc[[...]]`,
`loc[lo:hi]`. Для hashable labels — dict (быстро); для unhashable —
fallback linear scan (O(N×K), редко).

**`loc[lo:hi]` inclusive.** Pandas semantics: `['b':'d']` включает
и `'b'`, и `'d'`. Позиции: `get_indexer(lo), get_indexer(hi)`;
`range(lo_pos, hi_pos+1)`. Если `lo_pos > hi_pos` (например, `'d':'b'`)
— пустой результат, не ошибка. Если `lo`/`hi` not found — `KeyError`
(pandas: `KeyError` для missing endpoint).

**Open endpoints:** `loc[:hi]` / `loc[lo:]` — `NotImplementedError` в
v0.4.0c (требует направления сортировки — при asc индексе это
`[first : hi_pos]`, при desc — `[hi_pos : last]`; нужно проверять
is_monotonic). v0.4.1.

**`.at`/`.iat` — read-only.** `__setitem__` — `NotImplementedError`
(CadwDataFrame immutable; mutation через `write_parquet + reload`).
Это консистентно с общей immutable-моделью (кроме `set_index(inplace=True)`
и `reset_index(inplace=True)` — единственные inplace-операции).

**`loc` на filtered view** — `NotImplementedError` (как `df.index`).
v0.4.0d: filter-compat для `loc` через `_iter_filtered_batches_with_positions`.

**Дубликаты в индексе.** `get_indexer` возвращает первое вхождение
(MVP). v0.4.1: `get_indexer_all` — все вхождения, `loc[label]` вернёт
`CadwDataFrame` (несколько строк) вместо `dict`. Это отдельный API
контракт; в MVP — единственное вхождение.

**Урок:** indexers (`iloc`, `loc`, `.at`, `.iat`) — 4 разных API с
разной семантикой:
- positional vs label;
- array-like vs scalar;
- preserved vs reset index.

Все живут в `cadw/indexers.py`; facade `dataframe.py` — только property
возвращающие indexer. Это позволяет AST guard `test_facade_has_no_as_py`
проходить (`.as_py()` в indexers), и не «раздувать» `dataframe.py`.

**Следующее (v0.4.0d — filter-compat):**
- `_iter_filtered_batches_with_positions()` → `(batch, positions)`.
- `filtered.index` — RangeIndex subset или MaterializedIndex subset.
- `filtered.loc[label]` — работает.
- `filtered.set_index` / `reset_index` — работают.

**v0.4.1:** loc open-endpoints, loc with duplicate labels (все вхождения),
MultiIndex.

**Эпизод 59 (filter-compat — positions-механизм).** v0.4.0a-c
бросали `NotImplementedError` для `filtered.index` / `set_index` /
`reset_index` / `loc` при активных фильтрах. v0.4.0d закрывает это.

**Ключевая идея — `_iter_filtered_batches_with_positions`.** Yield
`(filtered_batch, positions_in_raw_dataset)`, где positions — numpy
int64 массив, 0-based позиции в **unfiltered** dataset (после последнего
set_index/reset_index). Реализация — композиция фильтров:

  Init: (raw_batch, arange(offset, offset+n)).
  Для каждого filter f в chain:
    - MaskFilter: mask = f.mask_for_batch(b); yield b.filter(mask),
      positions[mask];
    - SliceFilter: local_pos counter (filtered view);
      mask = (local_pos..local_pos+n satisfies slice); yield
      b.filter(mask), positions[mask].
  Затем b.select(_columns).

**Position semantics:**
- `MaskFilter` сохраняет raw positions (прошли маску → их raw pos).
- `SliceFilter` берёт позиции по **локальному counter** (позиция в
  filtered view), но raw pos — из накопленного positions.

**`_index_values` — always full dataset (unfiltered).** Filter не трогает
`_index_values`; `_with_filter` копирует его без изменений (уже так).
`filtered.index` = `_index_values[positions]` для MaterializedIndex, или
`positions + _index_start` для RangeIndex.

**Pandas-совместимо:** `df[df.v>10].index` — subset исходного; не
reset_index. `df[df.v>10].set_index("k")` — index = filtered k values
(через `_index_values[positions]` — потому что `_index_values` = полный
k, positions = filtered).

**`set_index(col)` при фильтрах** — материализует col **всего** dataset
(O(N) scan, не только filtered). Плата за корректность: `_index_values`
всегда full → любая комбинация filter/set_index/filter работает через
positions. Для больших N это цена; в v0.4.1 — streaming set_index с
RangeIndex fast path для уникальных col.

**`reset_index` при фильтрах:**
- `drop=True`: materialize filtered view как новый Dataset
  + RangeIndex(0, filtered_n). O(filtered_n) I/O — стандартный
  materialize.
- `drop=False`: добавляет колонку `index` = `_index_values[positions]`
  (для materialized) или `positions + start` (для range).

**`loc` на filtered view** — работает, потому что `_LocIndexer._materialized`
теперь возвращает `filtered.index` (MaterializedIndex). `get_indexer`
поверх filtered index.

**`iloc[list]` на filtered view** — preserves parent index через raw
positions, а не reset. Изменение с v0.4.0b.

**Урок:** позиции — универсальный механизм совместимости filter/index.
Один scan строит `(batch, positions)`; всё остальное — subset через
positions. Pandas делает то же (BlockManager + index → take), но
in-memory; мы делаем streaming с O(1) state.

**Следующее (v0.4.1):**
- loc open-endpoints (`loc[:hi]`, `loc[lo:]`) — требуют is_monotonic.
- Duplicate labels in index — `loc[label]` возвращает все вхождения.
- MultiIndex (`MultiIndex([col1, col2])`).
- Streaming set_index (RangeIndex fast path).

**Эпизод 60 (substring check маскирует отсутствие def).** Патч v0.4.0d
добавлял helper `_apply_filter_with_positions` условно:
`if "_apply_filter_with_positions" not in s: s = s + helper`. К моменту
проверки **call site** helper'а уже был в `s` (из шага 2 — метода
`_iter_filtered_batches_with_positions`). Проверка по подстроке увидела
call, решила «уже есть» — def не добавлен. 12 из 15 тестов упали на
`NameError: name '_apply_filter_with_positions' is not defined`.

**Fix:** проверка по `"def _apply_filter_with_positions("` (не просто имя).
Плюс AST guard `test_helper_is_defined_not_just_called`.

**Урок (усиливает эпизод 44/47b):** `if name not in src` — плохая
проверка существования функции. Различайте **call site** и **definition**:
- `f"def {name}(" in src` — определение;
- `f"{name}(" in src` — может быть call.

Тот же класс, что `str.replace` silent no-op (эпизод 44), но через
**false negative** — не «не заметили ошибку замены», а «не заметили
отсутствия».

**Правило:** для условной вставки **функции** — проверять по `def `.
Для условной вставки **константы** — по `NAME = `. Для условной вставки
**метода класса** — AST walk по `FunctionDef` внутри `ClassDef`. Grep
по имени — только для «вызов присутствует / отсутствует», не для
«определение присутствует».

**Более широкое правило:** любой `if X not in src` в патч-скрипте
нужно сопроводить комментарием, **что именно** `X` означает —
call, definition, import. Если X встречается в нескольких формах —
проверка должна быть по форме, не по имени.

**Эпизод 61 (loc complete: open endpoints + duplicates).** v0.4.1 закрывает
последние дыры в `loc`:

1. **Open endpoints.** `loc[:hi]`, `loc[lo:]`, `loc[:]` — работают на
   monotonic index (asc или desc). Pandas использует `slice_locs`, который
   понимает направление; мы делаем проще:
   - `loc[:hi]` → `positions[0 : get_indexer(hi) + 1]` (для любого
     монотонного направления).
   - `loc[lo:]` → `positions[get_indexer(lo) : end]`.
   - `loc[lo:hi]` explicit — для **increasing** `pl <= ph` → `positions[pl:ph+1]`,
     иначе empty; для **decreasing** `ph <= pl` → `positions[ph:pl+1]`,
     иначе empty. Reverse ranges → empty (pandas semantics).
   - Non-monotonic → `NotImplementedError` (pandas 2.x тоже).

2. **Duplicate labels.** `loc[label]`:
   - unique → dict (1 строка);
   - duplicate → `CadwDataFrame` (все вхождения).
   `loc[[labels]]` — все вхождения для каждого label, конкатенированные в
   original order.
   `loc[lo:hi]` на duplicate → `NotImplementedError` (pandas: cannot slice
   non-unique). Step — тоже.
   `.at[label, col]` на duplicate → `KeyError` (pandas: undefined).

**`MaterializedIndex.is_monotonic_*`** — O(N) scan, пропускает nulls.
`has_duplicates()` — set-based, O(N). `get_positions_all(label)` — O(N),
`==` по значениям (не hash).

**Pandas-семантика reverse range:** `df.loc['d':'b']` на ascending index
→ empty (не ошибка). Это уже было в v0.4.0c, теперь тестируется явно.
На descending `df.loc['b':'d']` → `['d','c','b']` (по позиции в index).
Это **общий** принцип: `loc[lo:hi]` на монотонном индексе возвращает
позиции между `pl` и `ph`, упорядоченные по index direction.

**Duplicate label → type change (`dict` → `CadwDataFrame`).** Pandas
делает то же (Series → DataFrame). Это может ломать code, который
делает `df.loc[label]['col']` — для unique это OK, для duplicate
`['col']` вернёт `_ColumnProxy` (не скаляр). Правильно — `.at[label,
col]` для scalar (unique) или `df.loc[label]` для «может быть
несколько». Pandas имеет тот же gotcha (`.loc[a]['b']` deprecated).

**Не в v0.4.1: MultiIndex.** `set_index(['a','b'])`, tuple labels,
`loc[(x, y)]` — отдельная подсистема. v0.4.2. Также: `loc` на
non-monotonic с явными lo/hi без duplicate — pandas 1.x предупреждал,
2.x требует monotonic; мы явно raise.

**Урок:** pandas API эволюционировал от «loc flexible» к «loc only if
monotonic + unique-for-slice». Мы следуем этой строгости — honest
`NotImplementedError` вместо «странных» результатов на non-monotonic.

**Следующее (v0.4.2 — MultiIndex):**
- `MaterializedMultiIndex(values: pa.ChunkedArray × k)` или
  `pa.StructArray`.
- `set_index(['a','b'])` — compose 2 columns в multi-key.
- `loc[(x, y)]` — tuple labels.
- `loc[('a','b'):('c','d')]` — tuple slice, требует лексикографической
  монотонности.
- `reset_index()` — «explode» multi-index в k колонок.

**Следующее (v0.4.3): rolling/resample** — Sort + halo buffer.

**Эпизод 63 (Pingis: P2 + P8 fixes).** Внешний мультиагентный
аудит (`_FOR_PINGIS/`) нашёл 15 проблем RAM (P1–P15). Две самые
показательные:
- **P2**: `_process_partition_pair` возвращал `list[RecordBatch]` —
  весь output 1:M join в RAM. Бенчмарк B1a: 100k×10k под 80 MB →
  ΔRSS +140 MB = **2.5× бюджета**.
- **P8**: `MemoryGuard.estimate_state_size` для custom `ReduceOp`
  возвращал 1 MiB (хардкод). ML-прототипы (XTXOp, KMeansStepOp,
  HistOp) получали warning; для state > 1 MiB — silently passed,
  OOM после. Блокер ML-слоя.

**v0.5.1 fixes:**

- **P2 (streaming join):**
  - `_process_partition_pair` → generator (yield вместо extend).
  - `_left_preserve_with_nulls` → `_left_preserve_with_nulls_iter`
    (streaming wrapper оставлен для совместимости).
  - `_join_core` — общий setup, возвращает `(schema, iterator)`.
  - `hash_join` — backward compat (собирает в Table).
  - `hash_join_streaming` — NEW: пишет прямо в `ArrowIPCBatchWriter`,
    возвращает `(schema, Path)`.
  - `CadwDataFrame.merge` — использует `hash_join_streaming`, оборачивает
    через `ArrowIPCBatchReader` + `Dataset.from_batches`, удаляет
    промежуточный файл через `path.unlink(missing_ok=True)`.
  - Peak: `max(build_partition) + probe_batch + writer_buffer` — **не
    зависит от размера output**.

- **P8 (state_bytes API):**
  - `ReduceOp.estimate_state_nbytes() -> int | None` — метод ABC.
    Default None (fallback на `MemoryGuard.estimate_state_size`).
  - Built-in ops override: CountOp/SumOp/MinMaxOp/MeanOp → 16;
    StdOp → 24; CovarianceOp → 8+8d+8d²; HistogramOp → 8bins+16.
  - `reduce()` вызывает `op.estimate_state_nbytes()` **до** `op.init()`;
    при None — fallback на `estimate_state_size` (старое поведение).
  - Пользовательские ops могут override — это публичный контракт для ML.

**Урок:** мультиагентный анализ дал **две независимые находки, которые
мы сами пропустили**: (a) out_batches accumulation (2.5× OOM,
бенчмарк — единственный способ увидеть); (b) хардкод 1 MiB в
`estimate_state_size` блокировал ML. Обе — из класса «аудит горячих
путей» — то, что мы не делали систематически. `_pingis/TASK.md` фиксирует
этот процесс для будущих прогонов (после новых тегов, Arrow-версий,
новых ops).

**Второстепенно:** бенчмарк (RSS-сэмплер) — единственный способ
подтвердить OOM, потому что `estimate` из модели ≠ реальный пик
(эпизоды 42, 47). Мы должны включить RSS-сэмплер в собственные E2E-тесты
(сейчас только `psutil` в e2e-join, но не в unit).

**Roadmap после v0.5.1:** P1 (join без pandas), P3/P10 (groupby
materialize), P5 (sort buffer), P6/P11 (to_pylist), P7 (модели с
Python-overhead), P4 (heavy accum). Порядок — по стоимости/эффекту
(сначала локальные, потом архитектурные).

**Эпизод 65 (P5: Arrow Builder вместо Python-list в sort merge).**
`_merge_keys` (k-way merge в `external_sort`) копил выход в Python-list:
`buffer[c].append(b.column(c)[pos].as_py())`, затем
`pa.array(buffer[c], ...)`. Пик: `read_batch_size × 28-80 B` (Python
int/float/string object) + `read_batch_size × arrow_size` (при `pa.array`)
= **×3-30 от Arrow**. Пингис B3a: +31.9 MB на 30 MB бюджете (1.06×
over). Fix: `pa.Array.Builder` (`Int64Builder`/`Float64Builder`/
`StringBuilder`/etc.) — Arrow-managed буферы, пик = `read_batch_size ×
arrow_size`. `_make_builder(dtype)` выбирает builder по типу;
`_append_to_builders` пишет по одному скаляру через `.as_py()` (O(1)),
`_flush_builders` → `RecordBatch`. After flush — `builder.finish()`
автоматически resets.

**Ограничение:** для dictionary/list/struct — `_make_builder` возвращает
None; `_merge_keys` raise `NotImplementedError`. Это honest fallback:
sort dictionary/list в v0.5.x не поддерживает; для этого нужны custom
builders, отдельная задача.

**Проверка:** 2 AST guard'а в `test_sort.py` — нет `buffer[...].append`
в `_merge_keys`; присутствует `_append_to_builders`.

**Ожидание:** B3a-new = Δpeak ≤ 30 MB (было +31.9). Даже небольшой
over-budget (10-20%) означает fix неполный — тогда следующий шаг —
уменьшить `read_batch_size` в `SortMemoryModel` для узких строк, или
k-way merge batch-wise через `pc.sort_indices`.

**Урок (усиливает эпизод 63):** бенчмарк — единственный способ поймать
P5. AST guard + unit-тесты этого не ловят (результат корректен). То же,
что P2: fix был «код корректен, но пик памяти не тот». Правило: для
memory-fix — бенчмарк с RSS-сэмплером в CI. `scripts/bench_p5_sort.py`
— первая ласточка; следующие (v0.5.2b, v0.5.3) получат свои benchmarks.

**Эпизод 99 (B2: `_NumpyBackedBuilder` инверсия bitmap).** Первый
настоящий баг корректности за долгое время. `_NumpyBackedBuilder.finish()`
делал `np.packbits(mask)` где `mask[i]=True = null`. Arrow-спека:
validity bitmap, **`1=valid, 0=null`**. `packbits(True)=1` попадал в
null-позицию → null↔valid инвертированы. Диагностика (v0.7.6.1): 
`packbits([F,F,T]) → [None,None,30]` (2 nulls) вместо `[10,20,None]` (1).
Fix: `packbits(~mask)`.

**Почему latent:** в PyArrow 26 есть native `Int64Builder`/`Float64Builder`
— `_make_builder` выбирает native (fast path), numpy fallback не
использовался в тестах. Тесты `test_sort_*_nulls_builder_path` шли через
**native** builder, инверсия не срабатывала. Это второй эпизод после
§13.4/52b («зелёные тесты ≠ упражняется новая функциональность»).

**Fix (v0.7.6.1):** `packbits(~mask)` + 3 direct regression tests на
`_NumpyBackedBuilder` (**не через `external_sort`** — чтобы не маскировать
native fallback'ом). Плюс AST guard в sort.py не нужен: `packbits` есть
только в одном месте (`_NumpyBackedBuilder.finish`).

**Урок:** custom Arrow builder/encoder — тестировать **directly** (не через
pipeline) + проверять **Arrow spec invariants**: `arr.null_count` и
`arr.to_pylist()` с известными позициями. Pipeline-тест проходит мимо
fallback'а, если native API доступен для типа. Правило: «для custom encoder'а
— direct test; pipeline test — отдельно». Также: `_NumpyBackedBuilder` для
`date64`/`timestamp` — сейчас latent; после мутации PyArrow может стать
real. Фикс превентивный.

**Эпизод 100 (time-series: expanding проще, чем rolling; ewm через
формулу pandas).** `expanding` — тривиальная композиция: `_RollState` с
deque заменяется scalar `_ExpandingState`: `sum`/`mean`/`count` — running
accumulator; `min`/`max` — monotone scalar. Причина: cumulative min is
monotone increasing by construction (каждая новая строка может только
уменьшить min), поэтому monotone deque из rolling не нужен. O(1) state,
O(1) per row.

**ewm — две формулы, легко запоминаются:**
- `adjust=True` (pandas default): `s_num = (1-α)·s_num + x`, `s_den =
  (1-α)·s_den + 1`, `y = s_num/s_den`. Init `s_num = x_0, s_den = 1`.
  Это `Σ (1-α)^i·x_{t-i} / Σ (1-α)^i`, реализованное рекурсивно за O(1).
- `adjust=False`: `y = (1-α)·y_prev + α·x`, `y_0 = x_0`. Один аккумулятор.

**Параметризация α:** `alpha ∈ (0,1]`; `span ≥ 1 → α=2/(span+1)`;
`halflife > 0 → α = 1-exp(-ln2/halflife)`; `com ≥ 0 → α = 1/(1+com)`. Всё
из pandas, 4 lines.

**Nulls:** наша семантика = `ignore_na=True` в pandas: null-строка не
обновляет state; output для неё null. `ignore_na=False` (pandas default)
учитывает null в decay — сложнее, не реализовано (документировано §11.5).

**PCA single-pass — почему подкласс, а не модификация `CovarianceOp`.**
`CovarianceOp.finalize` возвращает `d×d` matrix (bare `pa.Array`); это
публичный контракт (`test_reduce.py::test_covariance_2d`). Ломать его ради
`mean` + `n` — over-head. Вместо этого — `CovarianceMeanOp(CovarianceOp)`
с override `finalize` → `{cov, mean, n}`. Полиморфизм без API-break.
`fit_pca` переключился на новый op — 2× I/O reduction без изменения
`CovarianceOp` поведения.

**Урок (для v0.8.1/v0.8.2):** для time-series «cumulative» проще, чем
«sliding»; для windowed — можно комбинировать: `expanding = rolling(window=∞)`.
Для `resample` — не строить «новую примитивную операцию»; это `Sort` по
timestamp + `Reduce` по bucket'ам + halo-буфер на границе bucket'а.
**Композиция > новый примитив**, пока не доказано обратное.

**Эпизод 103 (`pc.list_functions()` ≠ `pc.hash_one`; `types_mapper`
для cross-batch consistency).** Два урока из v0.8.1.

**(1) `pc.list_functions()` содержит `hash_*` — но это Acero aggregate
kernels, не per-row hash.** `hash_count_distinct`, `hash_sum`,
`hash_first` — внутренние реализации `Table.group_by().aggregate()`,
экспонированные для удобства. **Не подходят** для `HashShuffle` (нужен
per-row uint64 для `_mod_p`). `pc.hash_one` (per-row) **отсутствует в
PyArrow 26** (PR #45001 open с 2023). Diagnostic должен проверять
**конкретные имена** (`hash_one`, `hash_64`), а не факт наличия
подстроки `hash` — иначе ложноположительно.

**(2) `arr.to_pandas()` даёт inconsistent dtype для integer в зависимости
от null presence.** Без nulls → `int64`; с nulls → `float64` (NaN).
Pandas `hash_pandas_object` hashes int64 и float64 **разными путями**
(SipHash разных kernel) → hash(1) в batch без nulls ≠ hash(1.0) в batch
с nulls. Для out-of-core **partition-disjoint** это критично: одно
значение в разных батчах попадает в разные партиции → wrong results
**молча** (нет crash, нет wrong count, только wrong groupby/join/dedup).

**Fix:** `types_mapper=Int64Dtype` в `to_pandas()` — форсирует
nullable Int64 независимо от null pattern. Then hash(1) consistent
across batches.

**Урок (усиливает §13.4/52b, 99):** для любой конвертации, влияющей на
partitioning/hashing, **cross-batch consistency — обязательный тест**.
Не только "same value → same hash **within a batch**" (тривиально), но
"same value → same hash **across batches with different null patterns**".
Правило: если операция недетерминирована на null-паттерне, тест на
неё — обязателен.

**Дополнительно:** `pandas.util.hash_pandas_object` в pandas 3.x
изменил API — `hash_key` для string требует `str`, не `int` (numeric
всё ещё `int`). Мы не передаём `hash_key` явно (default `None`) —
работает и для numeric, и для string. Seed применяется через XOR —
универсально.

**Ожидаемый эффект:** hash_shuffle 1M int64: 0.61s → ~0.25s (2.5×).
На 500 GB: 100 min → 40 min. String — зависит от pandas 3.x string-hash
стабильности (v0.8.1-fix при необходимости).

**Следующее (v0.7.8):** multiclass GBDT (softmax, k деревьев/round).
Откладывать hash fix было бы ошибкой — 2.5× на всё shuffle-based.

**Эпизод 107 (pandas hash для strings медленнее Python; pandas `<<1`
для NA; avalanche).** Два независимых наблюдения из v0.8.1.

**(1) `pd.util.hash_pandas_object` string → Cython per-element, медленнее.**
v0.8.1 bench:
    int64    39.2 → 2.02 ms  (19.4× speedup)
    float64  39.2 → 0.99 ms  (39.6× speedup)
    string   39.2 → 50.44 ms (0.78× — 22% SLOWER)

`hash_pandas_object` на object dtype вызывает `hash_object_array` —
Cython-цикл per-element (~500 ns/string) + categorical detection.
Python `repr(v).encode()` + `xxhash.xxh64` — ~390 ns/string. Pandas
не имеет fast path для strings в 3.x. **Fix (v0.8.1-fix3): strings →
`_python_hash_column`.** Numeric → pandas.

**(2) pandas `<<1` для object dtype; avalanche ОБЯЗАТЕЛЕН.**
`hash_pandas_object` для object/string dtype сдвигает hash << 1
(reserves `0` для NA) → **все string hashes чётные** (`h & 1 == 0`).
Для P=4 (`h & 3`) используются только партиции {0, 2}; для P=2 —
только {0}. Recursion split'ит 2-way вместо 4-way → на depth=5 партиция
1/32 от исходной (не 1/1024) → **`MemoryLimitExceeded`** в
`test_groupby_aggregate_recursion_end_to_end`.

**Fix:** splitmix64 avalanche (3 умножения + 3 XOR-shift, numpy) —
применяется всегда, seed XOR до avalanche, ~0.3 ms/100k.
`(h ^ (h>>30)) * C1; (h ^ (h>>27)) * C2; h ^ (h>>31)` — стандартный
финализатор Java SplittableRandom.

**Урок (усиливает §13.4/52b, 99, 106):** hash-функция требует
**low-bit distribution test**, не только `different value → different
hash`. `test_different_value_different_hash` проходил, но `h & 3` давал
2-way split. Правильный тест: `np.bincount(h & (P-1), minlength=P)` для
P ∈ {2,4,8,16} — каждая ~N/P. **Третий случай за сессию**, когда
«зелёные тесты скрывали структурную проблему».

**Правило hash-tests: минимум 3** — determinism, distinct-values,
**low-bit distribution**.

**Эпизод 113 (Multiclass GBDT: reuse ops через per-class loop).**
После v0.7.5 (binary logloss) следующий логичный шаг — multiclass
softmax. Два подхода:

**(A) `_MulticlassAllNodeHistOp`** — один Reduce для всех K классов:
    State: `g_hist (S, K, F, B)`, batch columns `__g_0..__g_{K-1}`,
    `__h_0..`, `__leaf_0..`. K× меньше pass'ов, но ~600 строк
    специализированного кода. Плюс: `_AssignLeavesOp` нужен V2 (K
    деревьев одновременно); `_LeafValuesOp` — V2 (K× nodes);
    `cur_leaf` — матрица (n, K). Итого: 5 новых ops + дубли.

**(B) Per-class loop** — K итераций, каждая с существующими
    `_AllNodeHistOp`, `_AssignLeavesOp`, `_LeafValuesOp`:
    - Pred (n, K) — обновляется per-column.
    - Grad/hess (n, K) — считаются ОДИН раз per round (softmax),
      потом per-class slices подаются как `__g`, `__h`.
    - K× медленнее binary, но **0 новых ops**, 0 новых багов в hist/
      leaf/assign.

**Выбор — B.** Причины:
1. **Correctness > speed** для MVP: existing ops уже покрыты 33+ тестами
   (`test_reduce.py`, `test_gbdt_regression.py`, `test_gbdt_classification.
   py`). Новые ops потребуют полного параллельного test-set.
2. **Bug surface:** мой первый черновик (A) содержал 2 бага в
   pattern-matching (`__leaf_0` vs `__leaf`, V1/V2 дубли). При B —
   просто нет места для таких ошибок: pipeline идентичен binary, только
   K раз.
3. **Cost:** K=3 → 3× медленнее. Для CI (n=1000, depth=3, 30 rounds):
   ~15s. Для production (n=10^6, K=5, 100 rounds): ~15-30 min — но
   это **уже out-of-core**, GBDT всё равно bottleneck, v0.7.8a
   оптимизирует.

**Урок:** для новых фич — **сначала работающая простота (reuse), потом
оптимизация (specialize)**. Специализированный `_MulticlassAllNodeHistOp`
имеет смысл **после** того, как per-class версия прошла все тесты
и доказала корректность. Иначе — debugging multiclass hist + debugging
specialized op + debugging multiple-model dispatch = три бага
одновременно.

**v0.7.8a backlog:** `_MulticlassAllNodeHistOp` — один pass на hist,
один на assign (K trees per row), один на leaf-values. Ожидаемый
speedup ~K×. Нужны V2 ops (или универсализация через `class_idx` axis).
Отдельная сессия.

**Семантическое решение:** `y` **строго** `{0, 1, ..., K-1}`.
1-indexed (`{1..K}`) — ValueError с указанием «сдвиньте y на -1».
Pandas/scikit-learn принимают любой набор labels, но у нас — целочисленный
target (не string), и 1-indexed — самый частый источник багов
(misclass label 1..K vs 0..K-1). Строгая валидация дешевле, чем
debugging 3-level trees с неправильным base prediction.

**Следующее:** v0.8.2 (`shift/diff on=col`), v0.8.3 (`resample(freq)`),
v0.7.9 (`feature_importance(gain)`). ML-слой закрыт (GBDT reg+bin+multi,
Linear/Ridge/Lasso/ENet, Logistic, KMeans, PCA, NB×3).

**Эпизод 116 (feature importance: split count vs gain; backward
compat через tuple slicing).** v0.7.4a `feature_importance` считал split
count — сколько раз feature использован в split'ах. Стандарт XGBoost/
LightGBM — **total gain** (Σ gain_i). Разница:
- 10 слабых splits на feature A (gain=0.01 each) → split=10, gain=0.1.
- 1 сильный split на feature B (gain=1.0) → split=1, gain=1.0.
- Split-count: A > B (WRONG). Gain: B > A (CORRECT).

Fix v0.7.9:
1. **5-tuple spec** — `(feat, thr, left, right, gain)`. Store gain at
   build time (`_fit_boosting`, `_fit_boosting_multiclass`).
2. **Backward compat:** `_assign_leaf_from_array` использует `spec[:4]`
   — толерантно к 4-tuple (pre-v0.7.9 pickled models) и 5-tuple.
3. **`compute_feature_importance`** — helper в `cadw.ml.gbdt`, handles
   binary (`list[tree]`) и multiclass (`list[list[tree]]`) через `_iter_trees`.
4. **Adapters** — `feature_importance(model, importance_type='gain'|'split')`.
   Default='gain' (breaking, но семантически правильно).

**Урок (backward compat для pickled models):** tuple slicing `spec[:4]` —
это **forward-compatible unlock** tuple'ов. Добавление N полей — не
breaking, если читать через slice. То же для dict (`spec.get("gain", 0)`),
но tuple быстрее. Правило: **если структура эволюционирует, читай через
`s[:N]` (или dict.get with default), не через `a, b, c, d, e = spec`
(exact unpack падает на missing field).**

**Семантика:** `gain` — raw sum (не normalized). 1 tree × 100 gain vs
1 tree × 10 gain — magnitudes не comparable, но ranking comparable
(что нам и нужно для importance). Для normalized — `gain / gain.sum()`
в user code. Не делаем normalized default (пользователь может хотеть
и raw, и normalized).

**Ограничение:** 4-tuple legacy → gain=0 для старых trees. Split works.
Документировано. Не auto-convert (это pickled models, которые пользователь
уже загрузил — его responsibility).

**Future (v0.7.10):** permutation importance — out-of-core eval на holdout.
Не реализуем сейчас: требует `sort_values` + `Reduce` на каждой
перестановке feature, K×N passes. Для 100 features × 10^6 rows — дорого.
В backlog, приоритет — ниже `resample`.

**Эпизод 119 (shift(on=col) — 2 sorts, not dict).** v0.6.2b `shift`
был positional — работал на физическом порядке строк. Для time-series
это бессмысленно: пользователь часто имеет unsorted parquet.

**Два подхода:**

**(A) dict `{row_id → (col, k) → value}`** — один sort, accumulate в
dict, потом zip. Но: **O(N) RAM** — нарушает memory contract. Для 10^7
строк × d cols × 8 B = гигабайты.

**(B) 2 external_sort** — sort by `[on, row_id]`, shift (deque O(k)),
sort by `row_id` → original order. **O(k × #cols) RAM**, независимо от N.
Цена — **2 полных I/O прохода** (sort). Для 10^6 строк ~2-3s.

**Выбор — B.** Правильный по memory contract. Пользователь мог бы сделать
`df.sort_values('ts').shift(1).sort_values('row_id')` сам — тоже 2 sort.
`on=col` — convenience API, не speedup.

**Row_id tie-break:** при duplicate `on` values — stable sort (row_id asc).
Это **pandas semantics** для sort_values (stable по умолчанию). Документировано.

**Implementing: re-chunking на zip.** `_iter_arrow_batches(batch_size=10k)`
(original) + `ArrowIPCBatchReader(sorted_by_rowid)` (может иметь другие
boundaries от external_sort). Re-chunk через accum + `Table.slice` —
каждый batch выровнен. Гарантия: original и shifted имеют identical
`num_rows` в каждом zip-batch. Иначе `RuntimeError`.

**Урок:** для position-sensitive операций с sort-aware API — **не
обходить memory contract** через dict. Даже если "просто dict" работает
для 10^6, для 10^8 = OOM. 2 сорта — стандартный паттерн (используется
в Dask, Spark sortWithinPartitions). Цена I/O приемлема; выгода —
гарантия O(1) extra RAM.

**Не в v0.8.2 (backlog):**
- Multi-column `on=['ts', 'sensor']` — тривиально (key_columns list), но
  требует валидации. v0.8.2a.
- Per-group shift: `groupby('sensor').shift(1, on='ts')` — нужен
  `HashShuffle` по group + sort по ts внутри group. v0.8.3+.
- `shift(-k)` (lead) — направление: sort by `[on, row_id]` ascending,
  shift backward = `shift(+k)` на reversed. То же 2 сорта + reverse.
  v0.8.2a.

**Следующее (v0.8.3):** `resample(freq)` — Sort by timestamp + bucket +
halo buffer на границе bucket'а. Плюс `freq` parser (`'5min'` → 5*60s).
~1 сессия, сложнее.

**Эпизод 126 (prototype-first: 5 fix → 1; + latent row_id bug).**
`rolling(on=col)` — третья composite API после `_shift_on` (v0.8.2) и
`_diff_on`. v0.8.2 прошёл **5 fix-итераций** (anchor, import, duplicate
row_id, replace vs append, ChunkedArray). Чтобы не повторить —
**prototype-first** (§13.4/123): `scripts/proto_rolling_on.py` (10 минут,
pure algorithm без touching `cadw/`), 3-path compare
(A=sort_values+rolling+sort_values, B=external_sort+deque+external_sort,
C=manual Python). Все три совпали → переносим код в `_rolling_on`.

**Результат:** **1 итерация** (v0.8.4, 17/17 passed с первого раза).
Экономия: 4 fix-итерации × ~20 минут = **80 минут**.

**Прототип выявил latent bug:** user column `row_id` →
`external_sort` appends свой `row_id` → duplicate → `KeyError: Field
"row_id" exists 2 times in schema` (в `ArrowIPCBatchReader` через
`sort_values`). Prototype `DATA = {"ts", "v", "row_id"}` — это
**reserved name**, переименовал в `seq_id`. Тесты `test_shift_on_col.py`
не всплыли, потому что их колонки `ts, v, cat` — без `row_id`.

**Fix (v0.9):** guard в `external_sort` + `hash_shuffle`:
`if "row_id" in all_columns: raise ValueError("row_id is reserved; rename user column")`.
Fail-fast лучше, чем cryptic `KeyError` в `sort_values`. Плюс
`Dataset` meta — `__row_id__` prefix для internal (v0.9+).

**Урок (prototype-first — обобщение):**
- Любая composite API (>50 строк, 2+ sorts) → **prototype** в `scripts/`,
  **до** вставки в `cadw/`.
- Prototype: 3-path compare (reference через существующий API / proposed
  algorithm / manual ground truth). Расхождение хотя бы в одной паре —
  debug **до** патча.
- Prototype **не модифицирует** `cadw/`. Если найден latent bug в
  существующем API — фиксировать в §11.5/backlog, не блокировать.
- **Метрика успеха:** 1 итерация патча vs 5. При prototype-first
  — достижимо (v0.8.4 показал 17/17 с первого раза).

**Ограничение:** prototype не ловит import/anchor/ChunkedArray — те же
runtime-классы. Только **алгоритм** валидируется. Для import/anchor
— AST guards (уже есть); для ChunkedArray — AST guard `combine_chunks`
(добавлен в `test_rolling_on_col.py::test_rolling_on_iter_zipped_appends_not_replaces`).

**Следующее (v0.8.3 `resample(freq)`):** prototype-first, обязательно.
Timestamp → bucket + halo buffer на границе. `freq` parser ('5min' →
300s, '1h' → 3600s, '1D' → 86400s). Prototype `scripts/proto_resample.py`
— 30 строк, 1 agg (sum), 2 bucket'а, проверка границ.

**Эпизод 127 (S1 latent bug: min/max + null в rolling).** Найдено
при аудите v0.8.4 — `_RollState.push(None)` с `agg='min'/'max'` и
непустым `_dq` (monotone deque) падает `TypeError: '>=' not supported
between 'int' and 'NoneType'`. Bug существует с v0.6.2c (позиционный
rolling), **не покрыт тестами**: `test_rolling_with_nulls` использует
`agg="sum"` только, а `test_rolling_min_w3`/`test_rolling_max_w3` — без
nulls. Значит `df.rolling(3, 'min')` на данных с null падал.

**Fix (v0.8.4-fix):** guard `if not is_null:` вокруг `while _dq and
_dq[-1][1] >= v` (и `<=` для max). Nulls skip monotone-deque
maintenance; window slot **всё равно потребляется** (позиционная
семантика, matches pandas skipna=True: `pd.Series([3,None,1]).
rolling(2,min_periods=1).min()` → `[3,3,1]`).

**Урок (усиливает §13.4/52b, 99, 106, 52b):** "зелёные тесты ≠
упражняется функциональность". `test_rolling_with_nulls` (sum) проходил,
но `min`/`max` ветка `_dq` не тестировалась с null. Класс "fallback
скрывает bug": sum-ветка не касается `_dq`, min/max — касается, но
никто не проверял комбинацию.

**Правило (новый AST/coverage guard для будущего):** для каждой
ветки `if self._agg == X` в stateful op — тест с nulls. В `_RollState`
ветки: sum/mean, count, min, max. Сейчас покрыты: sum+null (есть),
count+null (есть), min+null (добавлено), max+null (добавлено). Все 4 — 
хорошо. Для будущих composite (resample, groupby-shift) — тот же
принцип.

**Backlog:** добавить AST guard `test_rollstate_all_aggs_with_nulls` —
walk по `_RollState.push` и assert каждый `_agg ==` имеет тест с
nulls (не покрытие, а явная проверка). 20 минут, отложить.

**Общий аудит v0.8.4 (см. ниже):**
- 2 test-expectation errors (fixed).
- 1 S1 latent bug (fixed).
- Прочее: multi-key join cartesian fallback (S3), product-of-nunique
  overestimate (S3), `_iter_zipped` duplication (S3), `row_id` reserved
  (S3). Documented; см. §11.5/§11.6.

**Эпизод 128 (test expectations ≠ prototype).** v0.8.4-fix добавил
тесты `test_rolling_min/max_with_nulls` (позиционные) и `..._on_...`
(+on=col) — 5 тестов, 4 прошли, 1 упал. Причина: **ошибка в
ожидаемом значении** для max, не в коде.

Дано: `v = [3, None, 1, None, 5]`, window=2, min_periods=1, positional.
Walk:
    pos0: [3]         → max=3
    pos1: [3, None]   → max=3
    pos2: [None, 1]   → max=1   (3 evicted — 3 в окне pos1, не pos2)
    pos3: [1, None]   → max=1
    pos4: [None, 5]   → max=5
Правильно: `[3, 3, 1, 1, 5]`. Я написал `[3, 3, 3, 1, 5]` — мысленно
"оставил" 3 как максимум на pos2.

**Почему prototype это не поймал:** prototype проверил алгоритм
(manual == external_sort+deque == sort_values+rolling), но **не проверял
test expectations**. Test writer сгенерировал ожидания вручную
(мысленная арифметика) — тот же класс ошибки, что "off-by-one" в
ручных тестах.

**Урок:** test expectations для composite ops (rolling, quantile,
rank tie-breaking, shift с nulls) **нельзя** писать мысленной
арифметикой. Три варианта:
1. **pandas-equivalence:** `pd.Series(v).rolling(...).max().tolist()`
   в самом тесте (медленно, но надёжно).
2. **Explicit trace в docstring** — пошагово по позициям, как выше.
   Пишешь → видишь ошибку в reasoning до запуска.
3. **Prototype ground-truth** — `manual_rolling_on` уже содержит
   правильную логику (deque на sorted); использовать **его** output
   для test expectation, не пересчитывать вручную.

**Применено:** docstring в `test_rolling_max_with_nulls` и
`test_rolling_on_max_with_nulls` теперь содержит полный пошаговый trace
(после fix2). `test_rolling_min_with_nulls` случайно прошёл — на этом
датасете min и max совпадают (в окне 2 позиции, после pos 0 только
один non-null → min == max).

**Правило:** для тестов на rolling/shift/quantile+nulls — trace в
docstring обязателен. Если trace длинный (>5 позиций) — используй
`pandas.Series` или manual-функцию из prototype.

**Связанный эпизод (§13.4/52b, 99, 106):** "зелёные тесты ≠
упражняется функциональность". Здесь — "prototype passed ≠ test
expectations correct". Два независимых failure mode в test-слое.
Prototype-first (§13.4/123) фиксирует **алгоритм**, но не **тест**.

**Backlog (v0.9):** `tests/_helpers/rolling_reference.py` — функция
`expected_rolling(vals, window, agg, min_periods)` на deque, которая
используется в тестах как ground truth. Разные тесты — одна реализация
(не 4 копии) + нет мысленной арифметики.

**Эпизод 129 (meta-guard: `rolling_reference` helper).** §13.4/128
зафиксировал: test-expectation для rolling+nulls — источник ошибок
(2 итерации v0.8.4-fix). Корень — **ручная арифметика** в expected,
не в коде.

**Fix (v0.8.5):** `tests/_helpers/rolling_reference.py` —
`expected_rolling(vals, window, agg, min_periods)`. Deque-based, matches
pandas: positional window, nulls consume slot (не contribute),
`min_periods` applies to **non-null count** (не window size):
`pd.Series([1, None, 3]).rolling(2, min_periods=2).sum()` → `[NaN, NaN, NaN]`.

Применено в 4 тестах (min/max + nulls, positional + on=col).
Expected **derived**, не authored: property of algorithm, не of arithmetic.

**Урок (meta):** для rolling/shift/quantile/rank — hard-coded expected =
потенциальный баг. Три уровня:
1. Reference helper (`expected_rolling`).
2. pandas-equivalence (`pd.Series(vals).rolling(...).agg().tolist()`).
3. Per-position trace в docstring (если helper не подходит).

**Эпизод 130 (CHANGELOG descending — anchor drift).** `test_changelog_has_latest_tag`
правильно упал на master+v0.8.4: `top [0.7.4c] != tag v0.8.4`. Root:
`patch_changelog()` вставлял перед anchor (`## [0.8.2]` для `[0.8.4]`,
`## [0.8.4]` для `[0.8.5]`, …). Со временем `[0.7.4c]` (patch_v074c)
вставлен ещё раньше **выше** `[0.8.2]`, а `[0.8.4]` — перед `[0.8.2]`
→ `[0.8.4]` ниже `[0.7.4c]`.

**Fix (v0.8.5):** одноразовый sort всех `## [X]` блоков descending by
(major, minor, patch, suffix). Новый guard `test_changelog_is_descending`
(check sorted) — ловит **причину**, не только симптом.

**v0.8.4 tag не ретагнут** — тест `--exact-match` запускается только
на HEAD; старые tags не проверяются. v0.8.4 tag остаётся
out-of-order (historical artifact); v0.8.5 commit — fix.

**Урок (усиливает §13.4/111):** «anchor-based insert» в
CHANGELOG/build-files — source of drift. Правильно: **insert-at-top**
(искать первый `## [` regex, вставлять перед ним) или **build** из
separate files. Anchor `## [X.Y.Z-1]` предполагает, что `[X.Y.Z-1]`
существует и стоит на правильном месте — оба предположения нарушаются
по мере изменений.

**Правило:** для append-style docs (CHANGELOG, RELEASE_NOTES) — top
insertion by regex `^## [` first-match, или sort на read. Anchor-based
— только для стабильных anchors (e.g. `[tool.pytest.ini_options]` в
pyproject.toml, который никогда не двигается).

**Следующее:** v0.8.4a (multi-column `on`) → v0.8.3 `resample(freq)`
(prototype-first).

**Эпизод 131 (multi-column `on`: composite sort, not per-group).**
v0.8.6 добавляет `on=["ts", "sensor"]` для shift/diff/rolling. Технически
это малая правка — `external_sort` уже принимает `key_columns:
Sequence[str]` и корректно обрабатывает multi-key (per-column `_Key` +
row_id tie-break в heap). Единственное — `_shift_on`/`_rolling_on`
передавали `key_columns=[on]` (single literal); заменили на `on_cols`
(pre-normalized list).

**Семантика (задокументирована, важно не путать):**
- `on=["ts", "sensor"]` — **лексикографический composite sort**.
  Все строки сортируются по (ts, sensor, row_id), затем shift/rolling
  per row, sort-back.
- **НЕ per-group.** Для независимых окон по каждому sensor:
  `groupby('sensor').shift(1, on='ts')` (v0.8.5a+, не реализовано).

Пример: ts=[2,1,1,2], sensor=[a,a,b,b], v=[40,10,20,30]. Sorted v:
[10 (1,a), 20 (1,b), 40 (2,a), 30 (2,b)]. shift(1): [None, 10, 20, 40].
Back to orig: [20, None, 10, 40]. Если бы пользователь ожидал
per-sensor, он получил бы [None, 10, None, 20] — **не наш случай**.

**`_normalize_on` helper** — `str | Sequence[str]` → `list[str]` +
validation (`non-empty`, each in `self._columns`). Вызывается в dispatch
(`shift`/`rolling`) и в `_diff_on` (до numeric-type check на columns,
чтобы `on=[]` давал `ValueError` не `TypeError`).

**AST guards v0.8.2/0.8.4 unaffected:** second `external_sort` всё ещё
`key_columns=["__orig_rid"]` (literal), первый — `key_columns=on_cols`
(variable, `keys=None` в AST). Тест `test_shift_on_second_sort_renames_
row_id` смотрит только `ext_calls[-1]` — passes.

**Урок (малые фичи):** composite sort для multi-key — уже работал в
`external_sort`, единственная ошибка была в caller'е (`[on]` вместо
`on_cols`). Это **не** повод для prototype-first (§13.4/123) — нет
нового алгоритма. Но важен **честный docstring**: "composite sort, not
per-group" — иначе пользователь неправильно поймёт семантику.

**Backlog:** per-group shift/rolling (v0.8.5a+) — **другая** фича:
`HashShuffle` по group, `external_sort` по `on` внутри group, application,
concat. Требует prototype-first (как v0.8.2/v0.8.4).

**Следующее:** v0.8.3 `resample(freq)` — prototype-first, prototype
`scripts/proto_resample.py` (30 строк, 1 agg, bucket boundaries).

**Эпизод 132 (resample: prototype-first + renumber).**
`resample(freq)` — bucket aggregation, planned as v0.8.3. Prototype-first
(§13.4/123): `scripts/proto_resample.py` (3-path compare:
manual == pandas groupby(bucket) == external_sort + bucket transition).
5 buckets × sum: `[(0,4),(5,13),(10,23),(15,32),(20,22)]`. Все три
совпали → 1 итерация патча (28/28 passed first try). При v0.8.2 (без
prototype) было 5 fix-итераций.

**Rename v0.8.3 → v0.8.7.** CHANGELOG descending guard (v0.8.5,
§13.4/130) требует top = highest version. После v0.8.4/v0.8.5/v0.8.6
уже tagged, `[0.8.3]` над `[0.8.6]` → `test_changelog_is_descending`
fails. Также `test_changelog_has_latest_tag` (top == tag on master)
fails, если `[0.8.7]` положить не сверху, а tag — v0.8.7. Circular →
renamed. Roadmap-нумерация (§12.7) может дрейфовать; фактический
порядок = теги.

**Semantics decision (empty bucket):** pandas `resample()` даёт NaN для
всех aggs включая `count` (для gap-бакетов, без rows). Мы **skip**
все empty buckets — потому что output = «compact summary», NaN-rows
бесполезны для типичных post-processing (join, agg). Для fill —
`.reindex` после (pandas-стиль). Documented §11.5.

**Semantics decision (all-null bucket):** bucket с rows, но все values
null в agg-колонке → `count=0`, `sum/mean/min/max=NaN` в pandas.
Наш MVP: **skip** row (все колонки None → skip). Причина: consistency
с empty bucket; `count=0` бесполезен если `sum=NaN`. Documented.

**Semantics decision (null in `on`):** null timestamp → skip row
(cannot bucket). pandas: `groupby(ts.dt.floor('5s'))` — null → отдельный
NaN-bucket → `groupby(dropna=True)` (default) drop'ает. Совпадает.

**Урок:** prototype-first работает (28/28 с 1-й итерации). Renumber —
честный trade-off: документировать в CHANGELOG + §13.4, не «молча
исправить». Future tags — только последовательные (v0.8.8, v0.8.9, …),
не backport'ить roadmap-номера.

**Time-series API закрыт (v0.8.7):**
- `shift/diff` — positional + sort-aware single/multi.
- `rolling` — positional + sort-aware single/multi.
- `expanding/ewm` — cumulative/EWMA.
- `resample` — bucket agg.
- `row_id` reserved guard (v0.8.5).

**Следующее:** v0.8.5a per-group rolling/shift (~1.5 сессии,
prototype-first) — последний большой remaining в time-series.
Non-time-series backlog: `_iter_zipped` refactor (30 мин), reference
helpers для shift/diff/quantile (1.5 ч), multi-key join estimate
(45 мин).

**Эпизод 133 (pyarrow 26: date32 → int64 cast not implemented).**
`_to_unix_seconds` для `pa.types.is_date32` делал `pc.cast(arr,
pa.int64())` — **ArrowNotImplementedError: Unsupported cast from
date32[day] to int64**. `date32 = int32 days-since-epoch` — по
представлению это int32, но pyarrow не реализовал `cast(date32, int64)`
(только `date32 → int32` через same bit-width reinterpretation).

**Fix:** route через int32: `date32 → int32 → int64 → *86400 → int64
seconds`. Similarly для output (`int64 → int32 → date32`). date64
(int64 ms) — direct cast работает.

**Урок (API contract):** cast matrix pyarrow **неполный**. То, что
`date32` физически int32, не гарантирует `cast(date32, int64)`. При
cross-type cast — сначала `cast` в same-bit-width (int32/int64),
потом scale. **Правило:** для `date*` типов — через int32/date32 или
int64/date64 (matching bit-width), не jump через оба уровня.

**Эпизод 134 (`null`-typed column в test data).** `test_resample_all_
null_bucket_skipped` использовал `{"v": [None, None, None]}` — pyarrow
вывел `null` type (не `float64`). `resample` корректно отвергает
`null` как non-numeric (honest: null type has no aggregation semantics).
Test authoring: для "all-null float64" явно `pa.array([None]*3,
type=pa.float64())`.

Добавлен `test_resample_null_typed_agg_col_raises` — верифицирует
честное отклонение `null`-typed columns. Documented §11.5.

**Урок (test-authoring):** при `pa.table({...})` из Python lists —
pyarrow **infers** тип. `[None, None]` → `null`, `[None, 1.0]` →
`float64`, `[1, 2]` → `int64`. Если нужен конкретный тип — explicit
`pa.array(..., type=...)`. Тот же класс, что v0.8.4-fix2 (§13.4/128 —
"expected value derived, not authored"), но в **data setup**, не в
expected.

**Связанные:** §13.4/99 (numpy fallback hidden by native), §13.4/107
(pandas low-bit), §13.4/128 (test expectations). Все — "hidden by
green tests": (99) numpy fallback не exercised; (107) low-bit test
missing; (128) expected values wrong; (134) input type different from
intended.

**Следующее:** v0.8.8 per-group (prototype-first).

**Эпизод 135 (date64 через parquet → date32: silent downcast).**
`test_resample_date64` использовал `pa.array([datetime(2020,1,1,0,0,2),
...], type=pa.date64())` — ожидал 2 buckets по 5s. Failure: 1 bucket.
Output schema — `ts: date32[day]`, не `date64`. Два независимых
проблемы:

**Проблема 1 (test data, semantics).** `date64` — day-granularity тип
семантически. `datetime(2020,1,1,0,0,2)` → pyarrow **truncates** до
midnight (2020-01-01). Date64 = int64 ms, но **всегда** midnight-aligned.
Sub-day bucket (`freq='5s'`) на date64 бессмыслен: 5s-бакет ≈
86400*N-бакет ≈ bucket-per-day.

**Проблема 2 (parquet roundtrip).** `pq.write_table(date64)` →
`pq.read_table` → `date32[day]`. Parquet имеет **только Date logical
type** = int32 days; ms-date не представлен. `make_df` (from_parquet)
видит on_type = date32. Так и произошло.

**Fix (v0.8.7-fix2):**
1. Убрать `test_resample_date64` (parquet roundtrip невозможен).
2. Unit-тесты `_to_unix_seconds(date64)` / `_bucket_to_arrow(...,
   date64)` / `_to_unix_seconds(date32)` — direct on helpers.
3. `test_resample_date64_day_freq_through_parquet_limitation` —
   документирует, что real usage date64→date32 через parquet, работает
   корректно для `freq >= '1D'`.
4. §11.5: "date64 through parquet → date32; use freq >= '1D'".

**Урок (аналогичен §13.4/134 «input type different from intended»):**
`pa.date64` — **legacy/compat type** в pyarrow. Parquet не поддерживает
(нет logical type для ms-date). Arrow IPC сохраняет, но
`write_parquet` → `read_parquet` = date32. Правило: **не тестировать
date64 через parquet**; для date64-specific — unit-level helpers или
in-memory Dataset (без parquet).

**Ещё урок (semantics):** date64 = day, не sub-second. Если нужна
sub-second precision — только timestamp, не date. `resample(on=date_col)`
с `freq='5s'` — баг пользователя (не CADW): все → 1 bucket/день.

**Связанные:** §13.4/133 (pyarrow cast matrix incomplete: no
date32→int64), §13.4/134 (null vs float64 inference). Класс «hidden by
green tests»: (133) date32→int64 nonexistent — caught; (134) null→
float64 mismatch — caught; (135) date64→date32 parquet — caught (и
выявил test-authoring error, не code).

**Следующее:** v0.8.8 per-group (prototype-first).

**Эпизод 137 (rank.py: post-processor pattern).** Аудит v0.8.8
проверяет, что каждый `cadw/ops/*.py` (кроме primitives) импортирует
primitive (`reduce`/`shuffle`/`sort`) — invariant «high-level ops =
compositions of primitives». `rank.py` — исключение:
`compute_ranks_to_disk(sorted_path: Path, ...)` принимает УЖЕ
отсортированный IPC-path; caller (`CadwDataFrame.rank`) делает
`external_sort` до вызова.

**Pattern:** `rank.py` — **post-processor** of pre-sorted stream, не
composite over unsorted input. Legitimate architectural exception:
- Sort (expensive, 2 passes I/O) — в caller (facade, координатор).
- Ranking (cheap, 1 pass I/O) — в `rank.py` (post-processor primitive).

Split обоснован: `CadwDataFrame.rank` делает `external_sort` по `by`,
затем `external_sort` по `row_id` (sort-back). Между ними —
`compute_ranks_to_disk`. Вынести `external_sort` внутрь `rank.py` =
дублировать sort-back, нарушит "post-processor не сортирует" инвариант.

**Аудит fix:** whitelist `POST_PROCESSORS = {"rank.py": ...}` в
`patch_v088_housekeeping.py::audit` + docstring note в `rank.py`.
Для сравнения `quantile.py` **импортирует** `external_sort` — потому
что делает sort внутри (2-pass: sort + scan). Разница: `rank.py`
вызывается **после** sort, `quantile.py` сортирует сам.

**Урок:** composition invariant «ops = primitives» — не «каждый файл
импортирует primitive», а «op выполняется через primitives, где бы они
ни вызывались (caller или local)». `rank.py` — legitimate exception с
явной архитектурной причиной. Whitelist + docstring лучше, чем
force-import в post-processor (последнее = dead import).

**Связанные:** §13.4/133 (pyarrow cast matrix — code vs tests),
§13.4/134 (input type inference), §13.4/135 (parquet limitation),
§13.4/136 (housekeeping). Класс «hidden by green tests» /
«assumptions not verified»: (133/134/135) test-authoring, (137)
audit-assumption too strict.

**Эпизод 136 (housekeeping v0.8.8: smart snapshot, cleanup, audit).**
Проект вырос до ~2 MB в single-file snapshot; CHANGELOG 1000+ lines,
DECISIONS 2300+, patch scripts 120× (~1.5 MB — bulk). Snapshot занимал
~30% context budget, что ограничивает дальнейшие сессии.

**Fix (v0.8.8):** `collect_project.py` с `--mode=smart|full|tiny`.
Smart: full для `cadw/`, `tests/`, README, pyproject, latest-3 patches,
top-5 CHANGELOG, DECISIONS (§13.4 → last 10 episodes). Stub: old patches
(1 line + `[STUB: N B]`), archive, backups, uv.lock. **~2 MB → ~500 KB.**

**Cleanup:** patch scripts (older than 3) → `scripts/archive/v0.8.7/`
(`git mv` — history preserved, excluded from smart snapshot). `.backups/`
→ `/tmp/cadw_backups_<ts>/` (outside repo).

**Audit (подтвердил):**
- §13.4 episodes: 135, все closed (via CHANGELOG entries).
- §11.5: 18 open limitations (feature/refactor backlog, not bugs).
- §11.6 tech debt: 5 items (mostly perf refactors).
- 7 layers (per §4.1): all files present.
- 3 primitives: stable API (ReduceOp, compute_hashes, external_sort).
- `ops/*.py` (non-primitives) all `import from cadw.ops.{reduce,
  shuffle, sort}` — composition invariant preserved (no reinvention).

**Урок (meta, экономия context budget):** snapshot для chat — не
production artifact, а **in-memory view для AI-сессии**. Bulk (patch
scripts, history) должен быть stub'нут; только current-state (source,
tests, current limitations) — в полном виде. Правило: **default smart**
(минимальное для review+new-feature); full — только для отладки
исторических регрессий; tiny — для quick check.

**Связанные:** §13.4/111 (CHANGELOG chain), §13.4/130 (descending).
Все three — из класса "snapshot / log hygiene". Правило: любые
append-only files (CHANGELOG, DECISIONS, patch scripts) → archive-политика
+ smart viewer.

**Backlog §11.6 — обновлено v0.8.8:**
- + `scripts/*.py` без `sys.path.insert` (AST guard, §13.4/125) —
  раз 3-й за сессию (`bench_p5_sort.py`, `bench_hash_v081.py`,
  `proto_rolling_on.py`), раз fix при каждом — правило "первая
  не-comment строка после `from __future__`".
- + `_iter_zipped` duplication (`_shift_on` + `_rolling_on`) —
  ~100 строк, кандидат на `_zip_stream_by_row_id(orig, rolled,
  replace: bool)`.
- + reference helpers для `shift`/`diff`/`expanding`/`ewm`/`quantile`
  (§13.4/129) — тесты всё ещё с hard-coded expected; латентный риск
  §13.4/128-класса.

**Следующее (после v0.8.8):** v0.8.9 per-group (`groupby('sensor').
shift(1, on='ts')`) — prototype-first + type matrix (int64/timestamp/
date32 + nulls). Последний большой remaining в time-series.

**Эпизод 138 (доверие Arrow = доверие LightGBM).** Pingis
library-memory audit (TASK2, run 2026-10-20) обнаружил, что CADW
вызывал Arrow-функции (`Table.join`, `Table.group_by`, `to_pandas`)
без явных проверок пика памяти до вызова. MemoryGuard контролировал
`batch.nbytes` на входе, но не видел allocation **внутри**
Arrow-функции.

Это **тот же класс ошибки**, что §13.4/25 (LightGBM adapter) и
§13.4/26 (partial_fit whitelist): «библиотека оптимизирована, значит
безопасна». В случае Arrow — 5 подтверждённых live violations (P-A1,
P-A3, P-D1, P-G3, P-G5) и 2 blind spots (P-Q2 full column, P-R1
whole column для pc.rank).

**Ключевое:** audit использовал `pa_delta` (pyarrow.total_allocated_bytes)
для формул, но MemoryGuard проверяет **RSS** через psutil. Разница:
- `Table.join` narrow int64 @ 1M: `pa_delta = 82 MB` (3.4× output),
  RSS delta = **157 MB** (6.85× output). Factor 2× из-за Arrow memory
  pool retained capacity + non-Arrow temporaries.
- Все 14 формул §9.2.6–9.2.19 derived из **RSS**, не из `pa_delta`.

**Fix (v0.8.9):**
- §3.11 — принцип Arrow call memory discipline (AST guard).
- §9.2.6–9.2.19 — канонические RSS-based формулы.
- 6 fixes в коде (P-A1/P-A3/P-D1/P-G3/P-G5/P-SH2).
- AST guard `test_arrow_call_memory_guard.py` — 14 high-risk attrs,
  каждый требует preceding guard.

**Урок:** любой «Arrow-оптимизирован» в комментарии — сигнал пересчитать
через bench. Формула из документации библиотеки ≠ реальный RSS. Правило:
**для любого call'а, аллоцирующего > O(batch), явный `check_state` +
формула в §9.2.N**. Тот же rigor, что для своих ops (§9.2.1–9.2.5).

**Связанные:** §13.4/25, §13.4/26 (тот же класс «trust в библиотеку»);
§13.4/133–135 (тот же класс «hidden by green tests», но для type matrix);
§3.10 (check before materialize — расширение на library calls).

**v0.8.9-fix2 addendum (§13.4/139).** Три последовательных раунда правок
(v0.8.9 → v0.8.9-fix → v0.8.9-fix2) — каждый содержал свежий баг,
внесённый самим же фиксом:

- **v0.8.9**: AST guard путал `Table.from_batches` (материализует) с
  `Dataset.from_batches` (стримингует); широкий `ListComp` guard ловил
  `[f for f in schema]` (безопасно, K-bounded).
- **v0.8.9-fix**: `pc.cast(XOR(arr, 2^63) → int64)` — `safe=True` raises
  `ArrowInvalid` на всех значениях (XOR с sign bit переносит **всё**
  в `[2^63, 2^64)`); `pa.field(...)` по умолчанию `nullable=True` —
  test expected values захардкожены как non-nullable.
- **v0.8.9-fix2**: `pc.cast(arr, int64, safe=False)` = bitcast
  (two's-complement reinterpretation), bijection, nulls propagate;
  заменил XOR-вариант. Добавил REPL-сниппет в docstring `_hash_column`.

**Класс ошибки — тот же, что §13.4/128** («test expectations authored, not
derived») и §13.4/138 (audit formulas assumed, not measured). Здесь
«**fix semantics assumed, not verified**». Правило: **если fix добавляет
или меняет вызов библиотечной функции — REPL-сниппет в docstring
обязателен**, показывающий edge case: `safe=True` vs `safe=False`,
null propagation, boundary values (≥ `INT64_MAX`, empty, all-null).
Это защищает от ровно того класса, что случился трижды подряд.

**§13.4/140 (pre-release versioning vs guard).** Продолжение кластера
139. `patch_v0810_pre_join_offset.py` вставил `## [0.8.10-pre] —
2026-10-20` в CHANGELOG. Уронил `test_changelog_is_descending::_key`
(regex `^(\d+)\.(\d+)\.(\d+)([a-z]?)$` не матчит `-pre`) и
`test_changelog_has_latest_tag` (top `[0.8.10-pre]` != tag `v0.8.9`,
потому что fast упал до `git commit` — HEAD остался на теге).

**Fix (v0.8.10-pre-fix):** `[Unreleased]` (Keep a Changelog convention).
`^## \[(\d[^\]]*)\]` требует цифру → `[Unreleased]` невидим для
обоих guard'ов. At tag-time: `[Unreleased]` → `[0.8.10]` в одном
коммите, оба guard'а снова активны.

**Правило:** pre-release/rolling entries — **только под `[Unreleased]`**,
никогда `[X.Y.Z-pre]` / `[X.Y.Z-rcN]`. Patch-скрипт, трогающий
CHANGELOG, в собственном `main()` **обязан** прогнать `test_changelog_*`
до `git commit`. Класс ошибки — тот же, что §13.4/139
(«assumed semantics, not verified»); здесь — «assumed guard
compatibility».

**Эпизод 141 (per-group shift/diff/rolling — prototype-first).**
v0.8.10 добавляет `df.groupby('g').shift/diff/rolling(on='ts')`.
Алгоритм: composite sort `[*group_cols, *on_cols]`, single deque per
column, reset on group change, sort back by `__orig_rid`. 2 sort passes,
0 shuffle. Peak RAM = batch + O(window × #cols), independent of N и G.

**Prototype-first (§13.4/123):** `scripts/proto_per_group.py` — 11
кейсов, A (pandas) vs B (composite-sort deque). Все PASS **до** того,
как был написан патч-код. Сравнение с историей: v0.8.2 (shift on=col)
— 5 итераций без prototype; v0.8.4 (rolling on=col) — 1 итерация с
prototype; v0.8.10 (per-group) — 0 fix-итераций.

**Semantics:** null in group → own group (SQL, not pandas dropna=True).
Group columns NOT shifted by default (unlike pandas). Documented в
`_per_group_shift` docstring + §11.5.

**Reference:** `tests/_helpers/per_group_reference.py` — pandas ground
truth, same pattern as `rolling_reference.py` (§13.4/129). Test
expectations **derived**, not authored — §13.4/128, /129, /139.

**Closes §11.5 entries:** `groupby().shift(on=)` (was "not supported"),
`groupby().rolling(on=)` (was "v0.8.5+"), multi-column `on` per-group
(was "composite sort only, not per-group").

**Backlog:** per-group vectorization (§11.6) — current per-row Python
loop (for group-change detection), O(N) but constant factor higher than
vectorized. Optimize via segment detection (pc.not_equal between adjacent
group values) — future release.

**Эпизод 142 (P-D1: prototype-first vs stopgap).** v0.8.9 закрыл
P-D1 «stopgap»-ом: `check_state(5 × table.nbytes)` **до** `to_pandas`.
Это honest bound (5× покрывает pandas object/StringDtype overhead), но
**не rewrite** — pandas оставался в hot path, effective partition limit
падал в 5× (`avail × 0.1`).

v0.8.11a — prototype-first по §13.4/123: `scripts/proto_dedup_arrow.py`
сравнил 3 пути (pandas / arrow group_by / arrow dictionary_encode) на
correctness × selectivity × 3 размера. Path C (dictionary_encode +
group_by + is_in) выиграл с 0.29× peak vs pandas; **8× reduction на 1M
rows** (10.9 MB vs 87.7 MB). Correctness A==B==C на всех keep × selectivity.

**Adoption без fix-iterations.** Сравнить: v0.8.9 stopgap → 1 round
(warn-only); v0.8.11a rewrite → 1 round (prototype → patch → green).
Prototype стоит ~30 минут; он **экономит** fix-итерации и **даёт
honest numbers** для §9.2.N формул.

**Урок (расширение §13.4/141):** stopgap — не permanent fix. Каждый
stopgap должен иметь «revisit criteria» (в §11.6: «Arrow-native rewrite
when budget allows»). v0.8.9 → v0.8.11a gap = одна major-версия,
приемлемо. Если бы gap был >2 версий — stopgap становится долгом.

**Связанные:** §13.4/141 (prototype-first), §13.4/138 (audit формулы
из RSS), §11.6 (backlog discipline).

**Эпизод 143 (test substring ≠ token; CHANGELOG skip false positive).**
Пятый подряд раунд кластера §13.4/139-142: fix (или его тест) вносит
свежий баг.

**Баг 1 (test):** `test_dedup_removes_stopgap_5x` проверял
`assert "5 * table.nbytes" not in src`. Правильный новый код содержит
`check_state(1.5 * table.nbytes, ...)` — и `"1.5 * table.nbytes"`
содержит `"5 * table.nbytes"` как **подстроку**. Python `in` не уважает
границы токенов, тест падает на правильном коде.

**Баг 2 (CHANGELOG):** skip-условие
`if "Arrow-native dedup" in s or "P-D1" in s and "closed" in s: skip`
матчит v0.8.9 entry (там есть `P-D1` и `closed` в разных контекстах).
Результат: v0.8.11a CHANGELOG entry молча пропущен. Плюс: после v0.8.10
tag `[Unreleased]` header не существует (squash переименовал в
`[0.8.10]`) — anchor-path не сработал бы в любом случае.

**Fix (v0.8.11a-fix1):** (a) test проверяет **уникальный маркер** —
`"dedup.pandas_chain"` (имя старого guard'а) вместо `5 * table.nbytes`;
(b) CHANGELOG — проверить специфичный маркер
`### Исправлено (safety — P-D1 close)`, создать `[Unreleased]` header
если отсутствует, затем append.

**Правило (расширение 139+140):** любой `assert "X" in/not in src`
должен пройти проверку:
1. **X — уникальный маркер** (имя функции/guard'а, тег комментария),
   не code fragment, который может быть подстрокой правильного кода.
2. **Guard-тесты прогоняются в том же patch-скрипте до `git commit`**
   (см. §13.4/140). Не после.

Если X — code fragment (арифметика, вызов), использовать
**word-boundary regex** `r'\b' + re.escape(X) + r'\b'` или специфичное
имя. `1.5 * table.nbytes` vs `5 * table.nbytes` — учебный пример: 1
символ (`.`), и подстрока ломает тест.

**Кластер (5 раундов):**
| раунд | источник | фикс |
|---|---|---|
| v0.8.9 | AST guard broad (Table vs Dataset) | whitelist + `_is_table_from_batches` |
| v0.8.9-fix | pc.cast overflow / pa.field nullable | `safe=False` + expected +1 byte |
| v0.8.9-fix2 | stale test (uint64) | rename + assertion update |
| v0.8.10-pre | `[0.8.10-pre]` vs `[Unreleased]` | rename to `[Unreleased]` |
| **v0.8.11a** | **test substring / CHANGELOG skip** | **specific markers** |

**Связанные:** §13.4/128 (expectations authored), /129 (reference helpers),
/139 (fix semantics = REPL), /140 (pre-release), /141 (prototype-first),
/142 (stopgap→rewrite). Класс «assumed compatibility, not verified».

**Эпизод 144 (cross-file test staleness on guard rename).** Шестой
раунд кластера §13.4/139-143.

**Что случилось:** v0.8.9 добавил
`tests/test_pingis_safety.py::test_dedup_pandas_chain_precheck_raises`,
проверяющий `match="pandas_chain"` (guard 5×T stopgap). v0.8.11a
(P-D1 close) заменил guard на `arrow_native` (1.5×T), но test
в **другом файле** не был обновлён — patch-скрипт v0.8.11a правил
`test_dedup_arrow_native.py` и `test_no_pandas_in_ops.py`, а
`test_pingis_safety.py` (историческая куча) был вне его поля зрения.

**Как поймали:** fast run в `patch_v0811a_fix1.py` (§13.4/140
дисциплина: guard-tests до commit). Один wasted round.

**Fix (v0.8.11a-fix2):** переписать test напрямую на `_local_dedup`
с `budget = 2.5 × table.nbytes` (детерминированно: file ≈ T,
size passes, 1.5×T fires). Не через `drop_duplicates` — там
неконтролируемый hash_shuffle, размер партиции зависит от key-size
estimate.

**Правило (расширение 143):** patch, который renaming/removing guard
name, **обязан** `grep <old_name> tests/` и обновить **все** ссылки —
не только в своих новых файлах. Grep — часть pre-check, не после.

**Long-term (v0.9.0 backlog):** guard names → module constants
(`DEDUP_GUARD_NAME = "dedup.arrow_native"`, `JOIN_RSS_WARNING_NAME =
"join.whole_operation_rss"`), tests import from module. Renaming = одна
строка в source; tests follow automatically. Заодно — единый источник
правды для документации (§9.2.N ссылается на те же имена).

**Кластер (6 раундов):**
| раунд | источник | фикс |
|---|---|---|
| v0.8.9 | AST guard broad (Table vs Dataset) | whitelist + `_is_table_from_batches` |
| v0.8.9-fix | pc.cast overflow / pa.field nullable | `safe=False` + expected +1 byte |
| v0.8.9-fix2 | stale test (uint64) | rename + assertion update |
| v0.8.10-pre | `[0.8.10-pre]` vs `[Unreleased]` | rename to `[Unreleased]` |
| v0.8.11a | test substring (5* in 1.5*) / CHANGELOG skip | specific markers |
| **v0.8.11a-fix1** | **stale test in другой файл** (pandas_chain) | **grep before rename** |

**Связанные:** §13.4/128, /129, /139, /140, /141, /142, /143.

**Эпизод 145 (discovery, not creation — test found latent bug).**
Кластер §13.4/139-144 = 6 раундов «fix вносит новый баг». Round 7
(v0.8.11a-fix2) — **другой класс**: fix-тест **обнаружил** старую
latent проблему.

**Что нашлось:** `_local_dedup` (v0.8.11a, и v0.8.9 до неё) делал:
`Table.from_batches((batch.select([out_fields..., row_id]) for ...),
schema=reader.schema)`. `.select()` переупорядочивает в
`[out_fields..., row_id]`. `reader.schema` — file order. В production
`hash_shuffle` всегда кладёт row_id последним → порядки совпадают →
баг невидим. fix2-тест написал partition с `row_id` ПЕРВЫМ →
`ArrowInvalid: Schema at index 0 was different`.

**Различие классов:**

| раунд | что | кто нашёл |
|---|---|---|
| 1-6 (§13.4/139-144) | fix-код внёс баг | reviewer / CI |
| **7 (§13.4/145)** | **pre-existing latent bug** | **fix2-тест** |

**Fix (v0.8.11a-fix3):** explicit `select_schema = pa.schema(out_fields
+ [reader.schema.field("row_id")])`. Robust в любом file column order.
Новый `test_dedup_schema_order_agnostic` пишет row_id first — coverage
robustness, не workaround.

**Урок (positive):** тест-дисциплина (§13.4/140 — guard-tests before
commit) работает как задумано. fix2-тест, написанный со смещённым
допущением, **нашёл реальную хрупкость**, которая в production
никогда не проявлялась. 1 wasted round — приемлемая цена.

**Правило:** production код не должен зависеть от **неявных инвариантов
layout входных данных**. Если инвариант нужен — сделать explicit
(`select_schema`, schema cast) или документировать + assert.

**Backlog §11.6 (v0.9.0+):** audit всех `Table.from_batches(gen, schema=...)`
в ops на аналогичные неявные предположения о column order. Кандидаты:
`join._safe_materialize_partition` (явный `out_schema` — OK),
`sort._write_run` (schema после take — OK), `groupby._read_partition_safe`
— проверить в v0.8.11b.


**Update v0.8.11b (audit completed).** Verified 4 additional call sites
of `Table.from_batches(gen, schema=...)` in `cadw/ops/`:
- `join._safe_materialize_partition`: explicit `schema=out_schema`,
  order matches `.select(keep_cols)`. OK.
- `sort._write_run`: `ArrowIPCBatchWriter(table.schema)` — `table.schema`
  preserved through `pc.take(order)`. OK.
- `sort._merge_keys`: explicit `schema=schema` in `RecordBatch.from_arrays`.
  OK.
- `groupby._read_partition_safe`: `_iter_batches` yields reader batches
  as-is (no `.select(...)`), `schema=_schema` matches reader order
  exactly. OK.
- `dedup._local_dedup`: fixed v0.8.11a-fix3.

**No additional schema-order bugs found.** 4 AST guards in
`tests/test_schema_order_audit.py` lock the invariants.

**Связанные:** §13.4/128, /129, /139-144.

**Эпизод 146 (v0.9.0a — three first-run bugs, three classes).**
Не кластер §13.4/139-144 (fix→bug) и не §13.4/145 (discovery через
fix-тест). Это **первый прогон нового кода** — три бага трёх разных
классов:

**1. Partial update within one class (8 test failures).**
`PredicateFilter._OPS` validation в `__init__` осталась
`{gt, ge, lt, le, eq, ne}`. `patch_v090a` расширил `mask_for_batch`
(is_null/not_null/contains), но **не добавил их в validation** →
`ValueError: unsupported op 'is_null'`. Восемь тестов упали (isnull,
notnull, dropna_any/all/subset, contains×2, dropna_then_astype).

**2. Input conversion not Arrow-safe (1 test failure).**
`test_dt_null_timestamp_propagates` использовал `ts.tolist()` —
это даёт `[Timestamp, NaT, Timestamp]`. `pa.table({"ts": [...]})`
вызывает `NaT.utcoffset()` → `NaTType does not support utcoffset`.
Fix: `pa.array(ts)` — Arrow-native, NaT → null.
Класс — §13.4/134 («input type different from intended»), не bug кода.

**3. `[Unreleased]`-after-tag trap, 2nd hit (silent skip).**
После v0.8.11 tag секции `[Unreleased]` в CHANGELOG нет (squash
переименовал). `patch_v090a` код:
```python
if "## [Unreleased]\n" not in s:
    print("WARNING: ... — skip")
    return
```
Silent skip. Это **ровно та же** ловушка, что в `patch_v0811a-fix1`.
Первый раз (v0.8.11a-fix1) — 1 ч работы на fix; второй раз (v0.9.0a) —
опять fix. **Правило §13.4/144** («grep before rename») технически
выполнено, но **assumption "если anchor отсутствует — skip"**
не был заменён на «если anchor отсутствует — создать». Это **error
of omission** — отсутствие действия вместо неверного действия.

**Правило (расширение 140+143+146):** patch-скрипт, appending to
CHANGELOG, **обязан** вызвать `_ensure_unreleased()` — prepend
`## [Unreleased]` перед первым `## [` если его нет. Никаких
skip-ов при отсутствии anchor — fallback create.

**Long-term (§11.6 backlog, v0.9.1+):**
- `cadw/patch_utils.py` — общие утилиты для patch-скриптов:
  - `ensure_unreleased_header(text) -> text`
  - `append_unreleased_entry(text, entry) -> text`
  - `assert_guard_green(cmd) -> None` (§13.4/140)
- `PredicateFilter` — единый `_VALID_OPS = frozenset(_OPS) | _OPS_NO_SCALAR`,
  класс-attribute; `__init__` проверяет только `_VALID_OPS`. Никогда
  не может быть out-of-sync с handler'ом, если handler тоже
  использует `_VALID_OPS` для dispatch.

**Связанные:** §13.4/128, /129, /139-145. Кластер 139-145 — 6 раундов
fix→bug; здесь — 3 first-run bugs (новый код, не fix). Класс
принципиально другой, но **итог тот же**: 9 wasted tests, 1 round fix.

**Эпизод 147 (partial contract — Predictor ≠ Transformer).**
v0.9.0b: `add_step` валидировал `fit + transform`. `GBDTStep` —
Predictor (fit + predict, без transform). 6 tests fail.

**Класс — partial contract** (как §13.4/146 bug 1 «partial update within
one class», но здесь — partial contract между двумя классами):
- `Transformer` Protocol декларирует `fit + transform`.
- `Predictor` Protocol — `fit + transform + predict` (в v0.9.0b коде).
- Но `GBDTStep` реализует `fit + predict` (без transform).
- `Pipeline.transform` **останавливается перед** Predictor → Predictor
  никогда не вызывает `.transform()`.
- `add_step` проверяет `transform` → Predictor'ы (GBDTStep) не проходят
  validation.

**Fix (v0.9.0b-fix1):** `add_step` validation —
`has_fit and (has_transform or has_predict)`. Протокол `Transformer`
остаётся `fit + transform` (для pure-transform steps); `Predictor` —
`fit + predict` (для model steps); `Pipeline` принимает оба, различает
по `_is_predictor(step) = hasattr(step, "predict")`.

**Proactive fix:** `add_step` также ловит «Predictor not last» в обоих
направлениях:
- new step is Predictor + существующие contain Predictor → RuntimeError.
- existing last is Predictor + new step non-Predictor → RuntimeError.

Раньше это ловилось только на `fit` (`if is_pred and i != len(self._steps) - 1`),
что поздно. Теперь — fail-fast при `add_step`.

**Урок:** Protocol `Transformer` в v0.9.0b коде декларирует `fit +
transform + predict` для `Predictor` (строгое подмножество полного
набора). Реально `predict`-only steps **не нуждаются** в `transform`.
Contract должен быть **minimal**: `Predictor = fit + predict`, а
`Pipeline.transform` проверяет `_is_predictor` и пропускает последний
шаг. Декларируй больше, чем нужно → validation ломается на классах,
которые не реализуют ненужное.

**Правило:** для каждого Protocol/ABC — спросить «какой минимум
методов реально вызывается?». Всё, что «для симметрии» — либо убрать,
либо сделать optional. `Transformer` (pure transform) vs `Predictor`
(model) — не подмножество, а два разных минимальных контракта.

**Backlog (v0.9.1):** отделить `Transformer` и `Predictor` Protocols
строго (не `Predictor ⊃ Transformer`). `Predictor` Protocol: `fit,
predict, estimate_state_bytes, state_bytes` — без `transform`.
`Pipeline` типизирует `steps: list[Transformer | Predictor]`.

**Связанные:** §13.4/128, /129, /139-146.

**Эпизод 148 (design intent vs validation — DAG multiple sources).**
v0.9.1b: `add_step` validation запретил `inputs=[]` после первого шага:
```python
if len(inputs) == 0 and self._steps:
    raise ValueError("explicit source only valid for first step")
```
Я перепутал **«explicit source»** (избыточно для первого шага — там
`inputs=None` уже даёт source) с **«multiple sources»** (главная цель
DAG — fanout от raw). Мои же тесты используют `a inputs=[]`,
`b inputs=[]` — оба source. Validation убил фичу в самом патче,
который эту фичу добавляет. 6 tests fail.

**Различие семантик:**
- Sequence (v0.9.0): «step N+1 идёт после step N» — один linear source.
- DAG (v0.9.1b): «любой step может читать raw независимо» — **multiple
  sources by design**. `inputs=[]` на любом шаге = source, игнорируя prev.

**Fix (v0.9.1b-fix1):** убрать проверку. `inputs=[]` валиден на любом
шаге. `inputs=None` (default) = chain to prev (backward compat).

**Сопутствующий bug 2 (validation order):**
`add_step("post", inputs=["model"])` где `model` = Predictor. Мой
порядок: Predictor-after check → RuntimeError «cannot add step after
Predictor»; inputs validation не достигается. Но `inputs=["model"]` —
это семантически **более специфичная** ошибка («зависимость от
Predictor»). Fix: reorder — inputs validation до Predictor-after.

**Сопутствующий bug 3 (latent code smell):**
`_merge_fanout._iter` имел `if c in seen and c not in set(b.schema.names[:0])`.
`b.schema.names[:0]` = пустой list → `set()` → условие сводится к `c in
seen`. Работает случайно. Заменено на `if c in base_cols_set: continue`.

**Урок (расширение 146-147):** validation в новом API должен пройти
через **все репрезентативные use cases** *до коммита*. В v0.9.1b я
написал тесты для: sequence, fan-in, fanout, diamond, mismatch,
unknown input, Predictor-as-input, explicit source validation, topo
order (chain, fanout), transform-stops-at-model, predict, repr DAG.
Первый прогон — 6 fails. Правило: **«написать 1 тест на каждую
семантическую ветку + прогон до коммита»** — но этого не хватило,
потому что **тесты и validation были от одной модели** (я писал и
то, и другое). Нужна **независимая проверка** — либо prototype (A/B
против pandas groupby — не применимо к DAG), либо **явное документирование
design intent** до написания validation.

**Backlog (§11.6, v0.9.1c+):** перед `add_step`-style API — сначала
docstring «## API contract» с явными примерами, **затем** validation.
Не наоборот. Для DAG это спасло бы: «inputs=[] → source, может быть
несколько» — одной строки в docstring хватило бы.

**Связанные:** §13.4/146 (3 first-run bugs), §13.4/147 (partial
contract).

**Эпизод 149 (§13.4/144 REPEAT — round 2, same-file stale test).**
v0.9.1b-fix1 удалил validation `inputs=[] → error after first step`
(правильно: multiple sources — DAG feature). Но `test_add_step_explicit_
source_only_first` (написан в том же `patch_v091b_dag.py`, том же файле
тестов) продолжал **утверждать** старое поведение → 1 fail.

**Почему round 2:**
- v0.8.11a-fix2 (§13.4/144): cross-file stale test (guard rename).
  Я установил правило: «grep `<old_behavior> tests/` before commit».
- v0.9.1b-fix1: удалил validation, не grep'нул. Второй раз та же
  ошибка.

**Специфика this round:** patch_v091b_dag.py сам создал и код (buggy
validation), и тест (`test_add_step_explicit_source_only_first`), который
эту buggy validation проверял. **Одна mental model** — тест слепо
воспроизводит то, что код говорит, вместо того чтобы проверять
намерение (design intent). Тест «прошёл» в v0.9.1b → баг закоммичен.

**Fix (v0.9.1b-fix2):** заменить тест на `test_add_step_multiple_sources_
allowed` — проверяет правильное поведение. **Derived from docstring
API contract**, не из implementation.

**Механический grep в этом фиксе:** проверить другие тесты на отголоски
старого сообщения об ошибке (`explicit source only`, `explicit source.*first`).

**Правило (усиление §13.4/144):** перед коммитом patch, который
**удаляет или меняет** behavior (validation, error message, guard name,
return contract) — обязательный grep по `tests/`:
```python
# Пример:
r = subprocess.run(["grep", "-rn", old_error_pattern, "tests/"], ...)
if r.stdout.strip(): log_warning()
```
Это **не** автоматический fail — но **явный warning** перед `git add`.
Если warning сработал — разобраться до коммита.

**Дополнительно (§13.4/148 связано):** тесты, написанные **из той же
mental model**, что и код, не ловят ошибок этой mental model. Правильно:
писать тесты **из docstring API contract** (что должно быть), не из
implementation (что есть). Но это требует независимого автора (человек/
LLM после паузы/ревью) — в одной сессии я сам себя не проверю.

**Backlog (§11.6, v0.9.2):** AST guard `test_removed_behavior_stale.py`
— парсит `scripts/patch_*.py` на предмет удалённых error messages /
guard names; grep'ает `tests/` на совпадения; warning (не fail) в
pytest output.

**Связанные:** §13.4/144 (cross-file stale test), §13.4/146 (partial
update — validation not updated with handler), §13.4/148 (design
intent vs validation).

**Эпизод 150 (error message clarity for ambiguous inputs).**
`Pipeline.from_json(path)` принимает file или directory. Для несуществующего
path `"nope"` старая логика: `if path.is_dir(): ... else: meta_file =
path` → treated as file → error `"/tmp/.../nope not found"`. Test
ожидал `match="pipeline.json"`. 1 fail.

**Root cause:** ambiguous input handling молча выбирает интерпретацию
(«file») без подсказки, что искали. Для пользователя `from_json("mypipe")`
где `mypipe` — directory, сообщение не помогает — он видит, что
`"mypipe"` не найден, но не понимает, что внутри ищется `pipeline.json`.

**Fix (code-side):** для несуществующего path — **assume directory**,
показать `path/pipeline.json` в error. Явно указать: «path can be a
pipeline directory or the pipeline.json file inside it».

**Урок:** для API с **неоднозначным input** (file-or-directory, scalar-or-
list, str-or-Path) — не «resolve silent», а либо **явный выбор с clear
error**, либо **best-guess + helpful message**. `Path.is_dir()` на
несуществующем path = False → easy to misinterpret; `Path.exists()`
проверять явно.

**Backlog (§11.6):** audit similar ambiguities:
- `CadwDataFrame.from_parquet(path, base_dir)` — path must be file
  (base_dir separate). OK, но неясно что если path directory.
- `Dataset.from_parquet(parquet_path, base_dir)` — same.
- `Pipeline.from_json(path)` — fixed here.
- `write_parquet(path)` — path is file (out), ambiguous if directory
  passed → check в v0.9.2.

**Связанные:** §13.4/146 (first-run bugs — errors not helpful),
§13.4/148 (design intent vs validation — user mental model matters).

**Эпизод 151 (registry key vs class name — namespace mismatch).**
v0.9.1d: `register_step_type("MyCustom", _CustomStep)` регистрирует под
ключом `"MyCustom"`. Но `to_json` пишет `type = type(step).__name__ =
"_CustomStep"` — **class name**, не registry key. `from_json` ищет
`"_CustomStep"` в registry → не находит. 3 tests fail.

**Тот же design flaw, что был в v0.9.1c (`_STEP_REGISTRY` hardcoded
dict) — но другая грань.** Там я устранил hardcoded registry, но не
устранил **namespace mismatch** между тем, что регистрируется, и тем,
что пишется в JSON.

**Fix (v0.9.1d-fix1):**
- `register_step_type` sets `cls.__serialization_name__ = name`
  (attribute-based namespace binding).
- `to_json` prefers `getattr(type(step), "__serialization_name__",
  type(step).__name__)` — backward compat для старых pipeline.json.
- Built-in types pre-registered с `__serialization_name__` = canonical
  name (CastStep, DropnaStep, GBDTStep, ColumnTransformer).
- `ColumnTransformer` регистрируется **после** class definition
  (pre-registration block runs before class exists — NameError).

**Сопутствующие fixes (в том же прогоне):**
- test regex `"does not implement to_dict"` → `"LambdaTransform cannot"`
  (LambdaTransform **имеет** to_dict — он raises с honest message;
  test был wrong, code correct).
- `SyntaxWarning: invalid escape sequence '\+'` — non-raw regex string,
  `match="must have fit \+ transform"` → raw `match=r"..."`.

**Урок (расширение §13.4/144, /149):** если API использует **lookup key**
(registry, dict, enum) — **key at write и key at read должны быть одной
природы**. В v0.9.1d: write by class name (implicit), read by registry
key (explicit). Fix: явная `__serialization_name__` как единая точка.

**Метаправило:** для **lookup-based API** (registry, plugin system,
event bus) — обязательное `assert write_key == read_key` в тесте на
round-trip. Здесь это round-trip `to_json → from_json` — but only for
built-ins (registry key = class name). Custom step would catch it. Что
не хватило: **тест с custom registered name ≠ class name**.

**Backlog (§11.6):** audit similar lookups:
- `_STEP_REGISTRY` (fixed here).
- `_MODELS` в `memory_guard.py` — `register_model(model.op_name)`:
  registry key = `model.op_name`, no round-trip. Меньше риск, но
  `get_model(name)` uses registry directly — OK.
- `_VARIANTS` в `naive_bayes.py` — enum-like, no write side.

**Связанные:** §13.4/144 (write/read key mismatch, round 2 → 144's rule
needed), §13.4/150 (error message clarity).

**Эпизод 152 (third recurrence — manual discipline ≠ tooling).**
v0.9.1d-fix1 добавил `_STEP_REGISTRY["ColumnTransformer"] = ...`
(правильно — был баг 3 в v0.9.1d). Test `test_list_step_types_default`
(написан в v0.9.1d, ожидает 3 built-in types) — stale, реально 4 → 1 fail.

**Тот же паттерн, ТРЕТИЙ раз:**
- v0.8.11a-fix2 (§13.4/144): fix изменил guard name; stale test в другом файле.
- v0.9.1b-fix2 (§13.4/149): fix удалил validation; stale test в том же файле.
- **v0.9.1d-fix1 (§13.4/152)**: fix добавил registration; stale test ожидает старое число.

**Правило «grep before commit» установлено дважды (144, 149). Забыто каждый раз.**

**Причина:** «правило» — это текст в DECISIONS. Работает только если помнить. 3 из 3 раз — не вспомнил. **Manual discipline ≠ tooling.**

**Fix (this round — TOOLING, не текст):**
- `cadw.patch_utils.grep_and_warn(patterns, dirs, label=...)` —
  `grep -rFn` по указанным папкам, печатает hits, **не** raise.
  Для ручной инспекции перед `git add`.
- `cadw.patch_utils.assert_no_stale_behavior(patterns, dirs, label,
  allow=...)` — как grep_and_warn, но **raise SystemExit(1)** если
  hits (кроме allow-list). Для **dogfooding** в patch-скриптах:
  до `git add`, вызвать с patterns = old behavior strings, dirs =
  ["tests/"]. Любой hit = тест на старое поведение — фиксить.
- **fix2 dogfoods:** `assert_no_stale_behavior` вызывается в самом
  patch-скрипте до commit. Так следующий патч **не сможет** забыть —
  script упадёт при stale.

**Правило (окончательное):** patch, который меняет/удаляет/добавляет
**behavior** (validation, error message, guard name, return contract,
list of built-ins, enum values) → **обязан** вызвать
`patch_utils.assert_no_stale_behavior(...)` перед commit. Не «grep-
warning» — hard fail. Именно потому что warning игнорируется 3× подряд.

**Backlog (§11.6):** в patch-scripts **template** (когда он появится)
добавить `assert_no_stale_behavior` как обязательный шаг в `main()`.
Сейчас каждый патч пишется с нуля — до template, каждый раз явный
вызов, надежда на память. Скоро: `cadw/scaffold_patch.py` —
generate skeleton с pre-filled guard + stale checks.

**Связанные:** §13.4/144 (write/read key), /149 (stale test round 2),
/151 (registry namespace).

**Эпизод 153 (tooling caught its own bug — recursion in discipline).**
v0.9.1d-fix2 добавил `patch_utils.grep_and_warn` /
`assert_no_stale_behavior` — mechanical tooling для §13.4/152. Первый же
dogfood-вызов упал с `NameError: Path not defined` — `grep_and_warn`
использует `Path(d).exists()`, но `from pathlib import Path` не был
добавлен в импорты `patch_utils.py`.

**Ирония:** инструмент против stale behaviors содержал **того же класса
баг**, что и все предыдущие раунды (§13.4/146-152): partial update —
написал новый метод, забыл про import для его работы. Но **dogfood caught
it BEFORE commit**, не после. Не потому что код был правильнее, а потому
что **проверка была немедленной**, не отложенной.

**Класс ошибки:** тот же (§13.4/146-152), но **механизм обнаружения
другой** — встроенная проверка в том же прогоне, а не CI / следующая
сессия. Это ровно то, что должно быть для всех патчей: guard-runs в
`main()` до commit, а не «потом».

**Fix (v0.9.1d-fix3):** добавить `from pathlib import Path` в
`patch_utils.py`. 4 строки.

**Урок:** mechanical tooling **работает**, если его **вызывать сразу**.
`assert_no_stale_behavior` в `main()` (dogfood) поймал missing-import
до коммита. Без dogfood этот баг ушёл бы в коммит и всплыл бы на
следующем использовании `grep_and_warn` (следующий патч).

**Метаправило:** для любого patch-скрипта с новыми tooling/helpers —
**dogfood в самом скрипте** до `git add`. Не «протестируем отдельно» —
**вызвать в main()** в том же прогоне. Иначе tooling сам становится
источником багов.

**Связанные:** §13.4/144-152 (7 раундов «fix creates bug»), §13.4/146
(partial update), §13.4/152 (tooling required — manual discipline fails).

**Эпизод 154 (call-site vs definition substring — §13.4/53 repeat).**
v0.9.1e: `add_helpers()` проверял `if "_detect_mode" in s: skip`.
`_detect_mode` **есть** в файле — как **вызов** в `_per_group_apply`
(`_detect_mode(result)`), не как **def**. Проверка не отличила →
helpers никогда не добавлены → 14 tests fail с `NameError`.

**Тот же класс, что §13.4/53** (из docs — `if "name" not in s` видел
call site и не добавлял def). Забыто через 140 эпизодов.

**Fix (v0.9.1e-fix1):** проверять `if "def NAME(" in s`. `def ` prefix
отличает определение от вызова.

**Правило (§13.4/154):** при проверке наличия **функции/класса** в
исходнике — проверять `"def NAME("` или `"class NAME("`, никогда
голое `"NAME"`. Аналогично для полей класса — `"self.NAME ="`, не
`".NAME"`.

**Backlog (§11.6):** audit всех patch scripts на `if "X" in s` where
X — function/class/attribute name. Regex check: `if f"def {name}(" in s`.
Сейчас ~30+ патчей с этим паттерном; все исторические, не трогаем.
Новые — по правилу.

**Связанные:** §13.4/53 (original discovery), §13.4/146 (first-run
bugs — that round also included a "call site matched" issue),
§13.4/153 (tooling caught its own bug — but this one not caught).

**Эпизод 155 (chicken-egg guard + lazy-import AST over-strictness).**
v0.9.1e-fix1 прошёл apply-тесты (19 fast + 1 slow E2E), но упал на
**двух guard-тестах**:

**Bug 1 (chicken-egg):** `_flush_group` делает `pa.Table.from_batches(
group_buf)` **до** `check_state(est)`. Нужен `gt.nbytes` для est — а
`from_batches` и есть момент аллокации. AST guard (§3.11) правильно
флагает: high-risk Arrow call без preceding check.

Fix: **pre-estimate из `sum(b.nbytes for b in group_buf) * 2`** до
`from_batches` (pyarrow 26 from_batches — zero-copy per §9.2.9, значит
`sum(batch.nbytes) ≈ gt.nbytes`). `check_state(est_pre)` → `from_batches`
→ defense-in-depth `est_actual = gt.nbytes * 2`.

**Bug 2 (too-strict AST):** `test_no_pandas_in_dataframe_facade`
walks **весь AST** — ловит lazy `import pandas as pd` **внутри
`_emit_group`** (legitimate: нужен для `pd.Series`/`pd.DataFrame` при
обработке user fn output). Module-level import pandas в dataframe.py —
запрещён (тянет pandas при `import cadw`); function-level — допустим.

Fix: `_uses_pandas_import_at_module_level(tree)` — только `ast.Module.body`.
Strict `_uses_pandas_import` остаётся для `cadw/ops/*` (там даже lazy
не хотим, кроме whitelisted shuffle.py).

**Оба бага — фиксы не кода, а:**
- Bug 1 — не «код неверен», а «порядок операций в коде». `from_batches`
  нельзя сдвинуть после `check_state` без pre-estimate. Pre-estimate из
  `batch.nbytes` — стандартный паттерн (как `stat().st_size`).
- Bug 2 — не «код неверен», а «guard слишком широк». Lazy import vs
  eager import — разные вещи, AST должен различать.

**Урок (расширение §13.4/154):** AST guard должен быть **precise**:
- `def NAME(` vs `NAME` (— §13.4/154).
- `ast.Module.body` vs `ast.walk` (module-level vs any).
- `attribute == X` vs `substring X`.

Too-broad guard блокирует legitimate use cases; too-narrow — пропускает
баги. Fix always **narrow** to specific semantic (def, module-level,
attr name), never `walk` or substring.

**Backlog (§11.6):** audit guard-тестов на over-strictness:
- `test_no_pandas_in_ops` (strict for ops — OK; strict for facade — fixed).
- `test_no_unguarded_high_risk_arrow_calls` (func-level check — OK).
- `test_facade_has_no_as_py` (AST Call — OK).
- `test_materialize_guard` (substring `list(Reader)` — potential false
  positive if docstring contains pattern; но AST-based, OK).

**Связанные:** §13.4/154 (def-prefix), §13.4/152 (tooling), §13.4/151
(registry namespace), §3.11 (Arrow call memory discipline).

**Эпизод 156 (round 4 stale test — blanket-walk guard fragility).**
v0.9.1e-fix2 прошёл все guard-тесты, упал один **fast test**:
`test_repr_html.py::test_no_pandas_import_in_dataframe` — walks **весь
AST** и запрещает `import pandas` anywhere в `dataframe.py`. Legitimate
lazy import в `_emit_group` (fix2, needed to construct pd.Series/DataFrame)
— флаг.

**Тот же паттерн, 4-й раз:** §13.4/144 → /149 → /152 → **/156**.
Manual discipline + `assert_no_stale_behavior` (в fix2 искал **только**
`list_step_types` pattern) — не поймали.

**Корень (расширение §13.4/155):** file-wide AST guard'ы («no X
anywhere») — хрупки. Ломаются как только legit lazy X появляется.
`test_no_pandas_import_in_dataframe` и
`test_no_pandas_in_ops.py::test_no_pandas_in_dataframe_facade` — **два
теста в разных файлах** проверяют одно и то же; fix2 обновил один,
забыл другой. Дублирование guard'ов — source of drift.

**Fix (v0.9.1e-fix3):** `ast.Module.body` вместо `ast.walk` в
`test_no_pandas_import_in_dataframe`. Consistent с fix2 в
`test_no_pandas_in_ops.py`.

**Правило (§13.4/156, расширение 154, 155):**
- **AST guard'ы должны быть precise** — `tree.body` (module-level) для
  «import policy», `ast.walk` только для «anywhere at all is forbidden»
  (где это реально запрещено).
- **File-wide blanket guards** («no X in file Y») почти всегда
  хрупкие — уточняйте semantic: module-level / specific function /
  specific call pattern.
- **Не дублировать guard-тесты в разных файлах** — один canonical
  test, `import` helper'ы (например
  `_uses_pandas_import_at_module_level` из test_no_pandas_in_ops.py)
  для consistency.

**Backlog (§11.6):**
- Audit file-wide AST guards (`test_repr_html`, `test_no_pandas_in_ops`,
  `test_arrow_call_memory_guard`, `test_facade_has_no_as_py`) —
  убедиться, что каждый использует **precise** semantic.
- Consolidate duplicate guards (2 test_no_pandas_* — one canonical).
- Extend `assert_no_stale_behavior` patterns в каждом patch script:
  не только specific pattern, но и «все guards, что касаются файла
  X» — broader grep. `grep_and_warn(["import pandas"], ["tests/"])` в
  patch_v091e_fix2 поймал бы оба.

**Метаправило:** при **любом** изменении AST-свойства файла
(`import` policy, `from __future__` presence, module-level constants,
public API names) — `grep_and_warn` **по всем** тестам, не только по
одному известному pattern. «Какие тесты могут сломаться» — не «какой
я знаю», а «какие упоминают X».

**Связанные:** §13.4/144, /149, /152 (stale test repeats), /154
(def-prefix), /155 (module-level vs walk).

**Эпизод 157 (hasattr on pyarrow types misleading; count
values=index).** v0.9.1f: 9 tests fail — 6 на `bit_width`, 3 на
`crosstab values=index`.

**Bug 1 (bit_width):** `if hasattr(t, "bit_width"): t.bit_width` —
`hasattr` returns True для всех типов (атрибут **есть** в dir()),
но accessing `.bit_width` raises `ValueError: Non-fixed width type`
на string/binary/list/struct. Проверка не отличает «fixed-width» от
«variable-length».

Fix: explicit `pa.types.is_integer/is_floating/is_timestamp/is_date/
is_time/is_boolean` → `.bit_width`; string/binary → 50 B default;
остальное → 8 B fallback.

**Bug 2 (crosstab):** `crosstab` delegates to
`pivot_table(values=index[0], aggfunc='count')`. `pivot_table` validation
отклоняет `values in index_cols` без исключения для `count`. Но для
`count` values **не используется** (только non-null counting).

Fix: allow `values in index_cols` **только для `aggfunc='count'`**.
Для `mean/sum/min/max` — strict check остаётся (values должны быть
distinct, иначе результат — тождественный).

**Урок (§13.4/157, расширение 154-156):** `hasattr(t, "attr")` на
**pyarrow types** — MISLEADING для feature detection. Многие атрибуты
существуют на всех типах, но raise при wrong-type access:
- `bit_width` → raises на variable-length.
- `num_fields` → raises на non-struct.
- `value_type` → raises на non-list.
- `field(i)` → raises на non-struct.

Правило: для pyarrow type introspection — **только** `pa.types.is_*(t)`
или `try: t.attr except ValueError`. `hasattr` — никогда.

**Backlog (§11.6):** audit `hasattr` на pyarrow types в `cadw/`:
- `cadw/dataframe.py` — fixed (pivot_table).
- `cadw/core/models.py::type_size` — uses `pa.types.is_*` — OK.
- `cadw/core/memory_guard.py::estimate_row_size_from_schema` —
  uses `pa.types.is_*` — OK.
- `cadw/ops/resample.py::_TS_UNIT_FACTOR[t.unit]` — unit access only
  after `is_timestamp(t)` — OK.
- `cadw/ml/gbdt.py` — no type introspection in hot path.

**Связанные:** §13.4/154 (def-prefix), /155 (module-level AST),
/156 (blanket-walk).

**Эпизод 158 (E2E test ignores inherent complexity).** v0.9.1f E2E
test asserted pivot of 100K **unique** rows × 20 cols fits **50 MB**.
Real K_pairs = 2M; even perfect cardinality estimate → 2M × ~108 B ×
2 (pandas) = **432 MB**. State_limit = 15 MB. Fail-fast fires → test
claims failure on an impossible workload.

**Fail-fast — правильное поведение.** Pivot in pandas requires
**materializing** K_pairs cells in RAM. `pandas.DataFrame.pivot` не
out-of-core. Любая попытка «умного» pivot либо OOM, либо возвращает
данные с silent loss. Fail-fast — honest contract: «это не влезает,
уменьши размер или используй groupby (long format)».

**Fix (v0.9.1f-fix2):**
1. E2E success: n=10_000 × 20 cols, budget=200 MB → K_pairs=200K,
   est 43 MB < 60 MB state_limit. Fits.
2. E2E fail-fast: n=100_000 × 20 cols, budget=50 MB →
   `MemoryLimitExceeded` с `match="K_pairs"`. Contract verified.

**Урок (расширение §13.4/128, /134):** E2E test должен **уважать
неотъемлемую сложность** алгоритма. Если op имеет inherent O(K)
материализацию — test **не должен** пытаться впихнуть большой K в
маленький budget. Правильно:
- Success path — с размером, который реально влезает.
- Fail-fast path — verify, что правильная ошибка бросается (не OOM).
- Hand-written expected values (как в v0.9.1f — `rows["r0"]["c0"]
  is not None`) — проверяют контракт, не математику.

**Связанные issue (§13.4/157 secondary):** sample extrapolation даёт
`k_cols_est = 400` vs real 20 (для periodic data). Это **консервативная
upper bound** — правильная для fail-fast. Документировать в §11.5:
`pivot_table` K_columns может быть переоценено для low-cardinality
columns; если пользователь знает, что cardinality мала — можно
увеличить budget или явно ограничить.

**Backlog (§11.6):** audit E2E tests на inherent complexity:
- `test_pivot_100k` — fixed.
- `test_e2e_join_*` — already realistic (PK-FK, small right).
- `test_e2e_groupby_*` — budget realistic.
- `test_pipeline_save_load_e2e` — 5K rows, GBDT small. OK.
- `test_pipeline_e2e_time_series` — 20K rows, small GBDT. OK.

**Связанные:** §13.4/128 (test expectations authored), /134 (input
type not Arrow-safe), /157 (hasattr pyarrow types).

**Эпизод 159 (pivot pre-check = dense memory, not long-format).**
v0.9.1f-fix2: E2E success test 10K rows × 20 cols в 200 MB budget
падал с `MemoryLimitExceeded: K_pairs ~1000000 × row_size 108 × 2 =
216 MB > 0.3 × avail 60 MB`.

**Ошибка формулы.** `pivot_table` использует `groupby([*index,
columns])` (long format) + `pandas.pivot` (dense). Pre-check считал
long-format memory (`k_pairs × row_size`) вместо **dense pivot**
(`K_index × K_columns × value_size`).

- Long format: `K_pairs × 108 B` (row + col + value).
- Dense pivot: `K_index × K_columns × 8 B` (только value, все cells
  включая null).

Для 10K unique × 20 cols:
- Long: `200000 × 108 × 2 = 43 MB` (с точным K_columns=20).
- Extrapolated: `1M × 108 × 2 = 216 MB` (K_columns_est=100, 5× над).
- Dense: `10000 × 100 × 8 × 2 = 16 MB` (extrapolated) / 3.2 MB (точный).

**Dense — правильная метрика для `pandas.pivot`.** Long-format
groupby **уже** streamed на диск (groupby_aggregate → pa.Table, materialized
но bounded). Проблема — `pivot_table` materializes dense.

**Плюс:** custom `0.3 × avail` был inconsistent с
`MemoryGuard.state_limit_bytes` (0.5). Fix: use
`state_limit_bytes` — единая политика.

**Плюс:** отсутствовал pre-check для `grouped_table.to_pandas()`
(× 3 pandas overhead). Fix: `check_state(grouped_table.nbytes * 3)`
before to_pandas.

**Урок (§13.4/159):** для ops с **промежуточной materialization**
(long → dense) — pre-check должен оценивать **максимальный** из
промежуточных буферов, не первый. Pivot: long (bounded by groupby)
→ pandas × 3 → dense. Pre-check = max(long, pandas×3, dense) × safety.

Формула «K_pairs × row_size» была буквально **не той метрикой** — K_pairs
describe long row count, row_size — long row width; их произведение =
long table nbytes, но pivot dense ≠ long nbytes.

**Правило:** для каждого intermediate buffer — своя формула. Не
«универсальный `n_rows × row_width`». Pivot dense:
`K_index × K_columns × value_width`.

**Backlog (§11.6):** audit pre-checks в ops с multi-step materialization:
- `pivot_table` — fixed.
- `apply(fn)` — group materialization (per-group bounded). OK.
- `merge` — build + output. `JoinMemoryModel` покрывает.
- `dedup` — 1.5× table. OK.
- `groupby_aggregate` — output = K rows. `GroupByMemoryModel` покрывает.

**Связанные:** §13.4/158 (E2E respects inherent complexity), §13.4/157
(hasattr pyarrow types).

**Эпизод 160 (stale test regex — round 5, §13.4/144).**
fix3 заменил pre-check формулу `K_pairs × row_size` на dense `K_index
× K_columns × value_size`. Тест `test_pivot_100k_rows_fails_fast`
(fix2) имел `match="K_pairs"` — подстрока нового сообщения отсутствует.
1 E2E fails.

**Тот же паттерн, ПЯТЫЙ раз:**
- §13.4/144 — guard rename, other file.
- §13.4/149 — validation removal, same file.
- §13.4/152 — registration add, same file.
- §13.4/156 — blanket AST, other file.
- **§13.4/160** — formula change, same file.

**Причина each time:** patch script changes behavior; test asserts old
behavior; manual grep before commit **не сделан**.

**Tooling §13.4/152** (`assert_no_stale_behavior`) существует и
работает. **Но fix3 его не вызвал** — не применил собственный инструмент.
Это regression of discipline, not tooling failure.

**Fix (v0.9.1f-fix4):**
- Test regex `K_pairs` → `pivot_table` (broader, stable across formulas).
- Docstring `pivot_table` — dense formula (fix3 изменил код, docstring
  остался старым).
- Docstrings E2E tests — dense formula.
- **§13.4/160 — правило усилено:** каждый patch, изменяющий **error
  message** (even formula-only), обязан `grep_and_warn(["<old_substring>"],
  ["tests/"])` **перед** `git add`. Не «forgot», не «test passed
  locally» — **всегда**.

**Правило (усиление 149, 152):**
> Patch scripts **always** invoke `patch_utils.assert_no_stale_behavior`
> for:
> - Any removed/renamed validation error message.
> - Any changed formula in error message.
> - Any removed guard name.
> - Any changed return contract.
>
> Manual discipline failed 5x. Tooling exists. **Use it.**

**Backlog (§11.6):** `scaffold_patch.py` должен **enforce**
`assert_no_stale_behavior` call как mandatory step. Пока scaffold нет,
каждый patch — ручное решение (которое 5 раз подряд не принято).

**Метаправило:** когда **у тебя есть инструмент, который поймал бы
баг** — использовать его. `assert_no_stale_behavior` в patch_utils
ловит stale patterns механически. Пять раз подряд **не использовать
готовый механизм** — это не «забыл», это **систематическая ошибка
процесса**. Fix процесса — не «вспомнить», а **enforce** (scaffold).

**Связанные:** §13.4/144, /149, /152, /156 (stale test repeats),
/157 (hasattr pyarrow), /158 (E2E complexity), /159 (dense pre-check).

**Эпизод 161 (round 6 stale regex + AST substring, layered failure).**
v0.9.1f-fix4 обновил regex `K_pairs` → `pivot_table` в
`test_pivot_table_e2e.py`, но **забыл** `test_pivot_table.py::test_pivot_
too_large_raises` (тот же `match="K_pairs"`). Плюс `test_pivot_table_has_
check_state_before_to_pandas` использует `src.index()` — после fix4
docstring упоминает `to_pandas` **раньше** кода → test на substring, не
на AST node.

**Layered failure (три независимых источника):**

1. **Manual discipline 6-й раз** (§13.4/144 → /149 → /152 → /156 →
   /160 → /161): forgot to grep `test_pivot_table.py`.
2. **Dogfood grep found it — но non-fatal.** fix4 вызвал
   `grep_and_warn` (warning, не exit) вместо `assert_no_stale_behavior`
   (fatal). Tooling **сработал**, но wrong mode → script продолжился.
3. **AST test на substring** вместо AST node — после docstring change
   test false-fails на правильном коде.

**Fix (v0.9.1f-fix5):**
- Stale regex `K_pairs` → `pivot_table` в test_pivot_table.py.
- AST test — `ast.parse(inspect.getsource(...))`, найти `Call(func.attr=
  "check_state")` и `Call(func.attr="to_pandas")`, сравнить `lineno`.
  Docstring не парсится как Call.
- **Dogfood в этом fix — `assert_no_stale_behavior`** (fatal), не
  `grep_and_warn`. Must pass BEFORE commit.

**Правило (§13.4/161, три части):**
1. **When a fix changes behavior used by multiple tests** — `grep -rFn`
   **все** test files, не только те, что менял. Список изменённых файлов
   ≠ список affected files.
2. **Use `assert_no_stale_behavior` (fatal), not `grep_and_warn`
   (warning).** Warning-mode = "showed up, I'll fix later" = 0 iterations
   saved. Fatal-mode = "script stops, must fix now" = discipline enforced.
3. **AST tests must `ast.parse` — never `src.index()`.** `src.index`
   matches docstrings, comments, and string literals before code. Class
   of §13.4/143 (test substring vs token), /154 (def-prefix), /156
   (blanket-walk). Use AST for structural checks.

**Мета (усиление 160):** when tooling catches a bug **but is
non-fatal** — the discipline still fails (script continues, bug
propagates to pytest). **Fatal-by-default**: for any grep of patterns
known to be stale-after-behavior-change, use `assert_no_stale_behavior`.
Reserve `grep_and_warn` for exploratory diagnostics, never for patch
gates.

**Backlog (§11.6):** `scaffold_patch.py` — enforce `assert_no_stale_
behavior` (fatal) as mandatory step. Without scaffold, this decision
repeats 6+ times.

**Связанные:** §13.4/144, /149, /152, /156, /160 (stale repeats),
/143 (substring vs token), /154 (def-prefix), /156 (blanket-walk).

**Эпизод 162 (redundant pre-check masks per-metric validation).**
v0.9.2a: `metrics()` raises generic `"at least one of y_pred / y_proba
required"` **before** `MetricsOp.__init__` успевает дать specific
`"y_pred_col required for ['r2']"` / `"y_proba_col required for
['log_loss']"`. 2 test failures (test expects specific, code gives
generic).

**Диагноз:** layered validation — outer layer defends against case,
inner layer validates more precisely. Generic check **fires first**,
specific message never shown.

**Fix:** убрать generic check. `MetricsOp` — единственный источник
правды для per-metric validation. UX: «you asked for r2 but passed no
y_pred» лучше чем «at least one required».

**Плюс:** `test_metrics_no_pred_or_proba` дублирует
`test_metrics_missing_y_pred_for_r2` (тот же сценарий `metrics(y_true=
"y", metrics=["r2"])` без y_pred). Один test ожидал generic, другой
specific — **несогласованные ожидания** на одно поведение. Delete
redundant.

**Правило (§13.4/162, расширение 148):** validation должна быть на
**одном уровне**, не distributed между caller и callee. Outer defends
только против того, что inner не может (например, `metrics` validates
column existence, потому что `MetricsOp` не знает про Dataset);
inner validates то, что знает лучше (specific metric requirements).
Не дублировать.

**Backlog (§11.6):** audit layered validations:
- `metrics()` — fixed.
- `Pipeline.fit` → `plan` → `step.estimate_state_bytes` — outer validates
  target presence; inner validates state. OK.
- `pivot_table` → `groupby_aggregate` — outer validates K_pairs; inner
  validates partition budget. OK (different levels).
- `apply(fn)` → `_per_group_apply` — outer validates fn callable; inner
  validates group size. OK.

**Связанные:** §13.4/148 (design intent vs validation), /156 (blanket
guard), /157 (hasattr pyarrow), /161 (AST substring).

**Эпизод 163 (aggregate formulas need domain validation).**
v0.9.2a: E2E test uses continuous `y` (rng.normal) для `log_loss`.
Формула `-(y·log(p) + (1-y)·log(1-p))` предполагает `y ∈ {0, 1}`.
С `y = 2.0`: `-(2·log(p) + (1-2)·log(1-p))` → **negative** output
(assert `log_loss > 0` fails: got -0.203).

**Две независимые проблемы:**

1. **Production validation gap:** `MetricsOp` не валидирует `y ∈
   {0, 1}` для log_loss. Формула молча даёт мусор. Пользователь
   получает число, не подозревая, что оно неправильное.

2. **Test-authoring semantic mixing:** E2E test использовал **одну
   `y` (continuous)** для regression (r2/rmse/mae, где continuous OK)
   и для log_loss (где binary required). Один столбец — два разных
   семантических контракта. Test «passed» на regression и дал garbage
   на log_loss.

**Fix (v0.9.2a-fix2):**
- `MetricsOp.update` log_loss branch: `bad_mask = (y_m != 0) & (y_m
  != 1)` → raise `ValueError` с первыми 5 invalid values + счётчик +
  рекомендация (`encode_categorical` для multiclass).
- E2E: split на `test_metrics_100k_regression` (continuous y,
  r2/rmse/mae) + `test_metrics_100k_log_loss_binary` (binary y_bin =
  y > 0, proba = sigmoid(y)). Две независимые семантики.
- New fast test: `test_metrics_log_loss_rejects_continuous_y` —
  `y=[0, 1, 2]` → ValueError.

**Правило (§13.4/163, расширение 162):** aggregate formula имеет
**domain contract** (`log_loss`: y ∈ {0,1}, p ∈ (0,1); `r2`: y, y_pred
— continuous; `accuracy`: labels). Contract должен валидироваться
**at Op boundary** — где формула применяется, не у caller. Иначе
silent garbage.

**Метаправило (§13.4/163, расширение 128, /134, /158):** test,
смешивающий семантики (continuous + binary в одном столбце),
не может поймать validation bugs — потому что test сам не осознаёт
границу. Правильно: **один test = одна семантика**. Если нужно
проверить два режима — два test'а.

**Backlog (§11.6):** audit **всех** aggregate-формул на domain
validation:
- `log_loss` — fixed (y ∈ {0,1}).
- `r2/mse/rmse/mae` — no domain (любые finite float).
- `accuracy/f1/...` (v0.9.2b) — labels; validation в K×K confusion
  matrix (class set must match).
- `roc_auc` (v0.9.2c) — binary y; same log_loss validation.
- `groupby().agg` `mean/sum/...` — numeric required (existing
  validation).
- `std/var` — numeric, n >= 1 (existing).
- `PCA` CovarianceOp — numeric, d >= 1 (existing).

**Связанные:** §13.4/128 (test expectations authored), /134 (input
type not Arrow-safe), /158 (E2E respects complexity), /162 (redundant
check masks specific).

**Эпизод 164 (pos_label default ambiguity + test-authoring FP
miss + np.where division warnings).** v0.9.2b: 3 test fails, 4
RuntimeWarnings.

**Bug 1 (test-authoring, §13.4/128-класс):**
`test_metrics_precision_binary_pos_label_0` — ручной расчёт:
y=[1,1,1,0,0], yp=[1,0,1,1,0]. Comment said «class 0: FP=0» — **wrong**.
Row 1 (y=1, yp=0) — это **FP для class 0** (модель предсказала 0, а
истина 1). Реально TP=1, FP=1, FN=1 → precision=0.5 (не 1.0).
**Code правильный**, test ожидал неверное. Класс — §13.4/128
(expectations authored, not derived).

**Bug 2 (missing import):** `MemoryLimitExceeded` не импортирован в
test_metrics.py.

**Bug 3 (design, §13.4/164):** `average='binary'` (auto для K=2)
требует `pos_label ∈ classes`. Default `pos_label=1` (int) не работает
для `classes=['cat','dog']`. Sklearn also fails (default int on string
y → error). Но API может быть **дружелюбнее**: `pos_label=None` →
`classes[-1]` — для `[0,1]` даёт `1` (= sklearn classes[1]), для
strings — last class. Explicit `pos_label` — validate.

**Bug 4 (warnings, latent):** `np.where((tp+fp)>0, tp/(tp+fp), 0)` —
`tp/(tp+fp)` evaluates for all elements, включая zeros → RuntimeWarning.
Fix: `np.divide(tp, tp+fp, out=pp, where=(tp+fp)>0)` — vectorized,
no warning. Same для pr, pf1.

**Урок (§13.4/164, усиление 128, 163):**
- **Manual arithmetic в test expectations** — 3-й случай в сессии
  (128 rolling, 163 log_loss semantic, 164 precision FP).
  Правило (уже было): reference helper или trace. Здесь применимо:
  `expected_precision(y_true, y_pred, classes, pos_label)` — helper.
- **pos_label default как int** — design assumption, ломается на
  strings. Optional=None + resolve pattern — consistent с sklearn
  (classes[1]), но friendlier.
- **np.where для zero-division** — anti-pattern. Всегда
  `np.divide(..., where=...)`.

**Связанные:** §13.4/128, /129, /163.

**Эпизод 165 (gate by metric needs, not column presence).**
v0.9.2b: `MetricsOp.update` gated regression block by column presence
`if self.y_pred_col is not None:`. Для `metrics=['accuracy'],
y_pred='yp', y_true='cat'` (string labels): regression block runs →
`_to_f64('cat')` → `ValueError: could not convert string to float`.
1 test fails.

**Правильный gate:** `_needs_regression` / `_needs_proba` /
`_needs_classification` — установлены в `__init__` из requested
metrics. v092b добавил `_needs_classification` gate корректно, но
регрессию/логи оставил на старом `y_pred_col is not None`.

**Правило (§13.4/165, расширение 148, 162):** gate в `update` /
`finalize` — **by metric needs** (`_needs_*`), не by input presence.
Presence говорит «пользователь передал col», needs говорит «этот
блок действительно нужен». Первое — input, второе — semantic.

**Backlog (§11.6):** log_loss с string y по-прежнему cryptic error
(`could not convert 'cat'`). Улучшение: `MetricsOp._to_f64_numeric(
arr, "log_loss requires numeric binary y")` — context-specific
message. v0.9.2c.

**Связанные:** §13.4/148 (design vs validation), /162 (redundant check
masks specific), /163 (domain validation), /164 (pos_label ambiguity).

**Эпизод 166 (numpy 2.4 trapz, roc_auc ties, prototype as test-
authoring layer).** v0.9.2c prototype: 2 fix iterations before PASS.

**Layer 1 (external API contract):** `np.trapz` → `np.trapezoid`
(numpy 2.0 deprecation, removed in 2.4). pyproject pins
`numpy>=2.4.4`. Prototype on old API → AttributeError.

**Layer 2 (algorithm — ties):** `roc_auc` requires **per-threshold
grouping**, not per-row. All tied scores → one ROC point (sklearn
behavior). My per-row iteration gave wrong AUC for tied scores:
- Case 4 (all same score 0.5): 0.25 vs correct 0.5
- Case 7 (partial ties 0.5,0.5,0.7,0.7): 0.25 vs correct 0.5

**Layer 3 (test-authoring in prototype):** expected values in prototype
were manual arithmetic. Case 3 expected 5/6=0.833, correct 3/4=0.75
(different sklearn example). Case 2 expected 0.75 (label "random"),
correct 0.0 (fully inverted). **4th time in session** (§13.4/128, /163,
/164, /166).

**Fix (v0.9.2c):**
- `np.trapz` → `np.trapezoid`.
- **Per-threshold grouping** in streaming: `cur_score` sentinel,
  accumulate `cur_tp/cur_fp` per group, flush on score change.
- `tests/_helpers/metrics_reference.py::expected_roc_auc` — reference
  implementation (validated by prototype PASS + sklearn equivalence).
  Tests import it; no manual arithmetic in test bodies.

**Урок (§13.4/166, расширение 128, 129, 163, 164, 165):**
- **Prototype is also code, also requires API contract validation**
  (`np.trapz` in pyproject numpy>=2.4 → would fail on CI). Backlog:
  extend `test_api_availability.py` to numpy (`np.trapezoid`,
  `np.divide(where=)`, `np.int64`).
- **Prototype with expected values = test-authoring.** 4th time manual
  arithmetic failed. Rule: prototype expected — **only** from reference
  library (sklearn) OR **no expected** (A==B consistency only).
  Expected-tests — separate, with `tests/_helpers/metrics_reference.py`.
- **Sort-based algorithms need tie-heavy cases in prototype**
  (all-same-value, partial ties). Random continuous proba misses ties
  because ties are rare — but 100K E2E with quantized proba might
  have them. Prototype case 4 (all 0.5) + case 7 (ties 0.5/0.5/0.7/0.7)
  caught the bug in 5 minutes; without prototype — v092c would ship
  with tie-bug (missed on random tests, caught only in specific
  production data).

**Backlog (§11.6):**
- Extend `test_api_availability.py`: `np.trapezoid` (numpy 2.4+),
  `np.divide(where=)` (numpy 1.13+), `np.int64`, `np.float64`,
  `np.asarray`, `np.clip`, `np.isnan`, `np.log`, `np.exp` — что
  используем в metrics/pca/linear/gbdt.
- Prototype template: `expected = reference_fn(...)` (sklearn/pandas)
  or `expected = None` (consistency only). Manual arithmetic — fail.

**Связанные:** §13.4/128, /129, /163, /164, /165.

**Эпизод 167 (partial migration: dual field left active; 5th
test-authoring).** v0.9.2c: 14 test failures, 4 root causes — 2 code
bugs (same class), 1 test-authoring (§13.4/166 protocol not applied),
1 message-regex mismatch.

**Bug 1 — partial migration of `y_proba_col` → `y_proba_cols`:**
v0.9.2c added `y_proba_cols: list[str]` (unified binary + multiclass).
Kept `y_proba_col: str | None` (legacy, backward compat). In
`__init__`, validation checked **legacy** `y_proba_col`:
```python
if self._needs_proba and y_proba_col is None:  # WRONG
```
Callers passing `y_proba_cols=[...]` but not `y_proba_col` get
`y_proba_col=None` → false "y_proba_col required". 6 tests fail.
Fix: `self.y_proba_cols is None`.

**Bug 2 — facade generic column loop not updated after type change:**
v0.9.2a `y_proba: str | None` (scalar). v0.9.2c `y_proba:
str | Sequence[str] | None` (can be list). Facade column-existence
loop still iterates `(y_proba, "y_proba")` and checks
`if y_proba not in self._columns`. For `y_proba=['p0','p1','p2']` →
`['p0','p1','p2'] not in ['y','p0','p1','p2']` = True → KeyError.
Specific per-column check already exists below — remove y_proba
from generic loop.

**Bug 3 — 5th test-authoring mistake in session (§13.4/166):**
`test_metrics_roc_auc_with_accuracy` expected 0.875 manually.
Correct AUC = 1.0 (all 4 pos/neg pairs correctly ordered).
The test-authoring chain: 128 (rolling) → 163 (log_loss semantic)
→ 164 (precision FP) → 166 (roc_auc prototype) → **167 (roc_auc mixed
test)**. Same anti-pattern. Fix: import `expected_roc_auc` helper.
Test was written BEFORE helper was imported — should've
derived expected from helper at authoring time.

**Bug 4 — message-regex mismatch:** message "computed by
CadwDataFrame.metrics via external_sort" vs test regex
`"roc_auc.*separately"`. Fix message: add "separately" — clearer
intent, matches test expectation.

**Правило (§13.4/167, расширение 149, 152, 160, 161):**
Когда поле X **deprecated** в пользу X_new:
- (a) **Все** validation checks — на X_new. Deprecated X — **только**
  для input compat (set from X_new[0] если X_new not None). Никогда
  не validation target.
- (b) **Все** generic loops / column-existence checks — обновить
  после type changes (str → str | list[str]).
- (c) **Test** на mixed-types (scalar + list) — обязательны при
  расширении типа. Facade y_proba (scalar) tests v0.9.2a **не**
  ловили list case.

**Связанные:** §13.4/149 (validation add/rename), /152 (stale
behavior), /160 (stale regex round 5), /161 (AST substring),
/166 (test-authoring).

**Эпизод 168 (classification setup gate omitted multiclass
log_loss; same partial-migration class as /167).** v0.9.2c-fix1: 2
tests fail — `multiclass_log_loss` and `multiclass_log_loss_known`.
Both: `ValueError: log_loss (multiclass) y_true has labels not in
classes []: [0, 1, 2]`.

**Bug:** multiclass log_loss в `update()` мапит y_true → proba column
index через `self._cls_to_idx`. Этот dict заполняется в
classification setup block, gated на `if self._needs_classification:`.
Pure `metrics=["log_loss"]` → `_needs_classification = False` →
`_cls_to_idx = {}` → все y_true labels уходят в `missing_yt`.

**Тот же partial-migration класс, что §13.4/167:** новое feature
(multiclass log_loss) добавлено, но gate классического блока не
расширен.

**Fix:**
- Gate: `_needs_classification OR (_needs_proba AND
  _needs_multiclass_proba)`. Multiclass log_loss использует
  `_cls_to_idx` — значит, нужна та же инициализация.
- `batch_memory_factor` → 8.0 для multiclass proba (`.to_pylist()`
  на K+1 columns; §3.11 honest peak).
- Defense: `_needs_regression AND _needs_multiclass_proba` → ValueError.
  Domain conflict: regression y_true continuous; multiclass log_loss
  y_true = class index. Mutually exclusive.
- Tests: (a) pure multiclass log_loss без classification metrics;
  (b) regression + multiclass conflict; (c) K=2 multiclass form;
  (d) batch_memory_factor contract.

**Правило (§13.4/168, расширение 167):** если feature F использует
internal state S, а S инициализируется под gate G — расширить G при
добавлении F, **даже если семантически F не входит в группу G.**
«Multiclass log_loss — не classification metric» истина по имени, но
**ложна по shared state**. Naming ≠ state-sharing.

**Backlog (§11.6):** audit всех internal-state gates после
v0.9.2 series:
- `_cls_to_idx` — classification + multiclass log_loss (fixed).
- `_classes` — same (fixed).
- `_k` — same (fixed; K-only state; also used by classification
  fail-fast `K×24+64 > state_limit`).
- `tp/fp/fn` — classification only (correct; log_loss O(1) state).
- `sum_log_loss/n_proba` — proba only (correct; binary + multiclass
  share).

**Связанные:** §13.4/167 (partial migration: y_proba_cols unified
check), /148 (design vs validation), /162 (redundant check masks
specific), /165 (gate by needs not presence).

**Эпизод 169 (nullable=False violates Dataset.from_batches
convention; TSS+GBDT data design).** v0.9.2e: 10 test failures,
2 root causes.

**Bug 1 (9 failures, `ArrowInvalid: different schema`):**
`_attach_column` built `out_schema` with
`pa.field("__fold", type, nullable=False)`. `Dataset.from_batches`
(core) reconstructs per-column singleton via
`pa.RecordBatch.from_arrays([col], names=[field_name])` — **always**
creates nullable=True field. Writer target created with
`pa.schema([field])` where field nullable=False → schema mismatch
→ ArrowInvalid.

This is a **CADW convention**: all output fields must be nullable=True
(matching `from_arrays` default). Existing ops (shift/diff/rank/
expanding/resample) all use nullable=True. My `_attach_column`
violated convention by explicit nullable=False on `__fold`.

Fix: force nullable=True everywhere. Yield batches with explicit
`pa.RecordBatch.from_arrays(arrays, schema=out_schema)` — matches
writer target.

**Bug 2 (1 failure, r2 = -23.9):**
`test_cv_score_timeseries` used linear trend data
(`y = 2*ts`, `x = y`). TSS split fold 0: train ts=[0,50), test
ts=[50,100). Test x values 100..198 OUTSIDE training bin range
(0..98). GBDT without lag features **does not extrapolate** — all
test values clip to last bin (pred ≈ 95 constant). SS_res >> SS_tot
→ negative r2.

**Not a CADW bug** — inherent tree-model limitation, documented
§11.5, motivation for TimeSeriesPipeline (lag features — backlog).
**Bug in test**: unsuitable data for TSS+GBDT. Test design must
respect inherent model limitation.

Fix: i.i.d. x ~ N(0,1), y = 2x + noise. TSS splits by ts order but
x/y distributions align train vs test. r2 well above 0.3.

**Правило (§13.4/169, расширение 128, 129, 158, 166):**
- **Output schema nullability = nullable=True** для всех полей,
  produced через `Dataset.from_batches`. Core rebuilds singleton
  via `from_arrays(names=...)` (always nullable=True); writer target
  must match. Non-nullable output fields → ArrowInvalid. Latent
  backlog: align `Dataset.from_batches` to handle nullable=False by
  using `from_arrays(arrays, schema=pa.schema([field]))` for singleton
  (core change).
- **Test data must respect inherent model limitations.** TSS + GBDT
  (без lag features) cannot extrapolate. Test with trend data
  ensures failure, не от bug'а в коде. Real use case — TimeSeriesPipeline
  с lags. Test i.i.d. data, or bounded (periodic) features.

**Связанные:** §13.4/128 (test expectations authored), /129 (rolling
reference), /158 (E2E respects inherent complexity), /166 (prototype
test-authoring).

**Эпизод 170 (scalers — no new lessons; AST whitelist
observation).** v0.9.2f added StandardScaler / MinMaxScaler /
MaxAbsScaler / RobustScaler. **No new §13.4 lessons** — pattern
identical to CastStep (v0.9.0b) + `_NumericStatsOp` (single-pass
Welford+min/max, custom ReduceOp).

**Why no prototype:** all building blocks already tested —
`Reduce` (54 tests in `test_reduce.py`), `df.quantile` (20 tests in
`test_quantile.py`), `Transformer` Protocol + Pipeline (v0.9.0–0.9.1),
`_attach_column` nullability (§13.4/169). Scalers are **composition**,
not new algorithm. §13.4/123 prototype-first applies to **new
algorithms** (sort, roc_auc, CV fold math) — not to compositions of
tested primitives.

**Observation (§11.6 backlog):** AST guard whitelist in
`test_arrow_call_memory_guard.py` is **by function name**
(`node.name in WHITELISTED_FUNCS`), not by qualname.
- `"fit"`, `"transform"`, `"predict"` in whitelist → **ALL** methods
  named `fit`/`transform`/`predict` in `cadw/` are whitelisted
  (Pipeline, GBDTStep, scalers, ...).
- Rationale string per key is misleading (e.g., `"fit": "delegates
  to step.fit"` — true for Pipeline, false for RobustScaler which
  calls `df.quantile` directly).
- Effect: `RobustScaler.fit` calls `df.quantile` (high-risk)
  without preceding `check_state`/`check_batch` — not flagged.
  Not a bug (quantile has its own guards), but the guard cannot
  distinguish safe from unsafe calls in any future `*_fit` method.

**Proper fix (v0.9.3):** `_collect_violations` walks ClassDef to
build `ClassName.method_name` qualname; whitelist by qualname
(`"Pipeline.fit": "..."`, `"RobustScaler.fit": "..."`). ~30 min.

**Связанные:** §13.4/123 (prototype applies to new algorithms, not
compositions), §13.4/164 (whitelist discipline), §13.4/169
(nullability).

**Эпизод 171 (test-authoring #7; deterministic-vs-statistical
test pattern).** v0.9.2f-fix1: 1 failure
`test_standard_scaler_unseen_data`. Two bugs in same test:

**Bug A (manual arithmetic, §13.4/128-class):** comment said
«Test mean close to train mean (10.0/2.0 = 5.0)», assertion
`abs(mean - 5.0) < 0.5`. But standardization is
`(x - mean_train) / std_train` → output mean = **0**, not `mean/std = 5`.
Actual result `mean ≈ -0.036` (correct). Manual arithmetic wrong.

**Bug B (test doesn't test what name claims):** `x_train` and
`x_test` both `rng.normal(loc=10.0, scale=2.0)`. Frozen-stats transform
and refit-on-test both give output mean ≈ 0 (trivially, since
distribution matches). Test called `unseen_data` but couldn't
distinguish frozen-vs-refit. Property under test ("which stats were
used?") **requires different distributions** to distinguish.

**Fix:**
- `tests/_helpers/scaler_reference.py` — reference transforms for
  all 4 scalers. Test bodies use reference (never manual arithmetic).
- Deterministic data: `x_train = [8, 12]` → mean=10, std=2 exactly.
  `x_test = [12]` → frozen: `(12-10)/2 = 1.0`, refit: `(12-12)/1 = 0.0`.
  Frozen gives 1.0, refit gives 0.0. Distinguishable. Exact assertion.

**Правило (§13.4/171, расширение 128, 163, 164, 166, 167):**

1. **For EXACT-property tests, use deterministic data.** The property
   «does transform use frozen stats?» is exact (which stats were used).
   RNG washes out the signal. Deterministic data + reference helper
   gives exact assertion. RNG + statistical tolerance only for
   «does the model converge/learn?» — not «which stats were used?».

2. **Test invariants, not derivable values.** «Output mean ≈ 0» is
   derivable from StandardScaler invariant (`(x-mean)/std` of the
   training set has mean 0). «Output mean ≈ 5.0 because mean/std = 5»
   is not derivable — it's manual arithmetic that happened to be wrong.

3. **Reference helper for every scaler transform.** Once one test
   derives value manually, every test needs same reference. Codify
   in `tests/_helpers/scaler_reference.py` (tested separately via
   scalers self-tests).

4. **7th test-authoring error in session** (128 → 163 → 164 → 166
   → 167 → 169 → 171). Pattern: manual arithmetic + RNG where
   exact-property test needed. The `tests/_helpers/*_reference.py`
   discipline (started §13.4/164 with `metrics_reference.py`,
   extended §13.4/169 with `rolling_reference.py`, now §13.4/171
   with `scaler_reference.py`) is the systematic fix: every test
   that can use a reference helper — must.

**Backlog (§11.6):**
- All scaler tests (once they exist) → import from `scaler_reference`.
- When GBDT ranking/quantile-loss tests are written → reference helper
  per loss.
- When TimeSeriesFeatures (composite) tests are written →
  `ts_features_reference.py`.

**Связанные:** §13.4/128 (rolling nulls), /163 (log_loss semantic),
/164 (precision FP; metrics_reference.py created), /166 (roc_auc
prototype), /167 (dual proba field), /169 (TSS+GBDT test data).

**Эпизод 172 (closed-list stale-bait on registry growth).**
v0.9.2f-fix2: 1 failure — `test_list_step_types_default` asserted
closed 4-item list, registry grew to 8 (added 4 scalers via
`_register_scalers()` auto-called at `cadw/scalers.py` import).

**Not a bug** — auto-registration intentional: `cadw/__init__.py`
imports `from .scalers import ...` → `_register_scalers()` runs →
`Pipeline.from_json("pipeline_with_scaler")` works without explicit
`register_step_type` in user code.

**Test intent** was "verify no test-scoped registry pollution" (guard
against leaks from test_registry_* fixtures), not "only 4 forever".

**Two competing fixes considered:**
- (a) Update expected list to 8. Churn per built-in add; but churn is
  VISIBLE — patch review of expected-vs-actual catches typos in step
  name (e.g., `"StandartScaler"` → test fails, not silent).
- (b) Assert superset-of-N + no-unknown-types:
  `assert set(list_step_types()) >= {4 core}` +
  `assert set(list_step_types()) <= KNOWN_TYPES`. Requires
  KNOWN_TYPES set that also goes stale. Not simpler.

**Chosen: (a)** — update expected list in each patch that adds a
built-in. Convention: when adding a built-in step to `_STEP_REGISTRY`,
update `test_list_step_types_default` in same commit.

**Правило (§13.4/172, расширение 152, 161):** closed lists ARE
stale-baits. Two valid patterns:
- **(a) Explicit update per add** (chosen here): visible churn in
  patch, review catches typos in new names.
- **(b) Property-based test** (open-set assert): doesn't work well
  for "exactly these N" because the assertion is the SPECIFICATION.
- **(c) No test** — loses the "no pollution" guard.

For small stable registries (built-in steps: 8), pattern (a) is
simplest. For large expanding registries (ML models: 10+), pattern
(b) would be better ("every registered type has from_dict", "no
Test* leaked"). Different scales, different patterns.

**Backlog (§11.6):** audit closed lists across tests:
- `test_register_step_type.py::test_list_step_types_default` — fixed
  (8 items).
- `test_api_availability.py::test_pyarrow_version_and_imports` —
  asserts `pc.X` exists (open set, safe).
- `test_api_availability.py::test_cadw_all_exports_importable` —
  iterates `__all__` (adaptive, safe).
- `test_no_pandas_in_ops.py::ALLOWED_PANDAS_MODULES` — closed set,
  1 item (shuffle.py). Adding new pandas exception → update needed.
  Same pattern as this episode.

**Связанные:** §13.4/152 (stale behavior), /161 (AST substring),
/167 (partial migration — stale check), /171 (test-authoring).

**Эпизод 173 (KFold O(N) fold_id без MemoryGuard; docstring lie).**
v0.9.3a: P-CV1 S1 из v0.9.2 audit.

**Bug.** `KFold.split` в v0.9.2e материализовал O(N) fold-id массив:

- `shuffle=False`: `np.arange(n) % k` + `astype(int32)` - 12·N байт
  (int32 fold_id + int64 arange temp).
- `shuffle=True`:  + `rng.permutation(n)` - 20·N байт.

**Ноль MemoryGuard involvement.** Для N >> RAM - guaranteed OOM,
не `MemoryLimitExceeded`. Нарушение ядра гарантии CADW.

**Docstring claim.** Pre-v0.9.3a: "O(N) int8 fold_id array".
Actual code: int32 + int64 temporaries. 3-4x understatement.

**Почему не поймали в v0.9.2e.** Тесты KFold: n=100-1000 (12 KB
array - незаметно). Slow E2E: n=100k (1.2 MB на 50 MB budget -
тоже OK). Никто не проверил N=1e8 с budget=16 MB - core user
scenario. **Small-N hides O(N) bugs.**

**Fix (v0.9.3a):**

- `shuffle=False` (default): `fold_id = row_id % k` per batch.
  O(batch) state. 1:1 с прежним поведением.
- `shuffle=True` (default new): keyed splitmix64(row_id, seed) % k.
  Deterministic per row_id. O(batch) state. Fold sizes ~binomial
  (mean n/k, per-fold sigma ~= sqrt(n/k)). Split-quality.
- `materialize=True` (opt-in): exact-balanced, O(N) RAM,
  `check_state(20*N)` guarded.
- `_attach_column_with_fn(fn: (offset, n) -> pa.Array)`: streaming
  helper, O(batch) state.

**Правило (13.4/173, расширение 158, 166, 172):**

- **Docstring memory claims must match actual code.** Pre-commit
  audit: grep все "O(N)"/"O(K)" утверждения в docstrings, сверить
  с фактическими alloc.
- **Small-N тесты скрывают O(N) bugs.** Каждый публичный op с O(N)
  state должен иметь:
  (a) AST guard "no np.arange(N)/permutation(N) в default path";
  (b) functional test с tight budget + большим N;
  (c) либо задокументированное O(N) state в 11.5 + fail-fast.

**Связанные:** 13.4/158, 166, 172, **13.4/174**.

**Эпизод 174 (patch-script embedded escape trap - 3 rounds).**
v0.9.3a: 6 fix iterations before tests green - all from patch-script
syntax bugs, not CADW code bugs.

**Round 1 (v0.9.3a original).** CV_KEYED_TESTS triple-quoted
template contained `"_attach_column(\n"` - Python parsed `\n` as
escape -> 1 newline char -> written file had unterminated string
literal -> SyntaxError on py_compile.

**Round 2 (v0.9.3a-fix1).** Same class, different char: CHANGELOG
entry had `r'''...'''` inside `'''...'''` outer -> `'''` terminated
outer early -> parser error "Statements must be separated by
newlines or semicolons".

**Round 3 (v0.9.3a-fix2).** Docstring itself:
  "Fix2 uses `"""` for ALL outer triple-quoted strings"
The `"""` inside `"""..."""` terminated the docstring.

**Round 4 (v0.9.3a-fix4).** Pattern match failure: fix4 looked for
"ASCII `->`" but file had unicode `->` (U+2192) in docstring. `old
not in s` -> abort. Classic test-authoring (13.4/166): pattern
written from memory, not from file.

**Round 5 (v0.9.3a-fix6).** 2 stale tests in test_cv.py still
expected exact sizes [20,20,20,20,20] under shuffle=True -> failed
[21,16,20,19,24] because v0.9.3a made default binomial. **13.4/160
repeat:** behavior change should grep ALL tests before commit.

**Rules (13.4/174):**

1. **Patch scripts embedding Python code in template strings:**
   - NEVER triple-quote for content. Build from `list[str]` +
     `"\n".join()`.
   - If any content line has backslashes, keep it single-line
     `'...'` or `"..."` (no `\n`, `\t`, `\x..` in template).
   - If content has `'''` or `"""` - escape or reword.
   - **Best: `pathlib.Path.write_text()` from separate template
     file** (v0.9.4 backlog: `patch_utils.write_content()`).

2. **Pattern matching in `.replace()`:**
   - NEVER include docstrings, comments, or unicode in the pattern.
   - Match only signatures (def, class) + structural lines.
   - If block contains docstring - line-based: `startswith` +
     `endswith`, edit `lines[i] = new`.
   - If not found: print actual context (nearby `def`s), не голый
     "not found".

3. **Self-compile check before run:**
   - `uv run python -c "import py_compile; py_compile.compile(
     'scripts/patch_X.py', doraise=True)"`
   - Tooling fix (13.4/174 backlog): `patch_utils.assert_compiles(
     path)` invoked by scaffold.

4. **Split code and docs patches:**
   - code+test commit: `v0.9.3a`.
   - docs commit: `v0.9.3a-docs` (this).
   - Reason: abort in docs template does NOT lose code commit.
     Sister to 13.4/140 (CHANGELOG discipline).

**Мета.** Tooling exists (`patch_utils.assert_no_stale_behavior`),
но я его не вызвал в v0.9.3a. **Use it.**

**Связанные:** 13.4/140, 144, 149, 152, 156, 160, 161, 166, 173.

**Эпизод 175 (memory formula validity scope).** v0.9.3c: P-D2
из v0.9.2 audit.

**Bug.** `_local_dedup` guard `check_state(1.5 * table.nbytes)`
был выведен из prototype v0.8.11a: int64 keys, K<<N, measured
peak 0.23x table. Экстраполяция на **string keys, K=N** дала
7.5x underestimate (real 11.26x). Consequence: partition passes
1.5x check, but real peak 11x -> OOM вместо MemoryLimitExceeded.

**Правило (13.4/175, расширение 128, 158, 173):**

- **Memory formula scope must be measured, not extrapolated.**
  Prototype on int K<<N не доказывает что formula holds для
  string K=N. Kernel cost depends on: dtype (fixed-width vs
  variable-length), cardinality (K vs N), and internal BufferBuilder
  geometric growth (2x peak).
- **Bench matrix for each guard:**
  - dtype: int / float / string (at least)
  - cardinality: K<<N / K=N/2 / K~N
  - table size: small / medium / large
  Формула должна быть conservative на **worst cell** матрицы,
  не average.
- **`pc.dictionary_encode` is not O(table.nbytes).** For variable-
  length types it copies every value + builds offsets + indices +
  hash table. Peak = table.nbytes + K*(avg_len+8)*2 + 
  4*N (indices) + hash table + BufferBuilder 2x. Simplify: 12x
  table.nbytes for string subset (measured, bench-backed).

**Backlog (11.6):**
- Audit all `check_state` formulas — grep `* table.nbytes`,
  `* grouped_table.nbytes`, etc. Each should have bench matrix
  source in docstring, not single-cell extrapolation.
- Sample-based K_est tightening for dedup: for K<<N strings,
  could use 2.5x instead of 12x via `count_distinct(sample)`
  extrapolation. v0.9.4.

**Связанные:** 13.4/128, 158, 173, **P-D1 v0.8.11a** (guarded
int path only; strings edge case deferred to v0.9.2 audit).

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

**v0.1.4 (патч 10):** must-fix «materialize-before-check»: `_local_groupby`
(stat до чтения) и `reduce` (estimate_state_size до init()). Добавлены
AST guards + OOM E2E. §3.10 — новый принцип.
