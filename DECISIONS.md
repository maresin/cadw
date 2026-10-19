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
| ~~P2: out_batches accumulation (join)~~ | ~~`_process_partition_pair`~~ | **закрыто v0.5.1**: streaming via `hash_join_streaming` |
| ~~P8: 1 MiB hardcode в `estimate_state_size`~~ | ~~`reduce()`~~ | **закрыто v0.5.1**: `ReduceOp.estimate_state_nbytes()` |
| `drop_duplicates` returns rows in partition-concatenation order | `drop_duplicates` (v0.3.0) | pandas-order: `.sort_values(...)` после. §13.4/48 |
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
| `on` multi-column — composite sort, not per-group (v0.8.6) | `_shift_on` / `_rolling_on` | `on=["ts","sensor"]` = lexicographic; per-group — `groupby(...)` (v0.8.5a+) |
| `rolling` on=col — single column (v0.8.4) | `_rolling_on` | Multi-column on — v0.8.4a |
| `groupby(...).rolling(...)` — не поддержано | `CadwDataFrame` | Per-group rolling — v0.8.5+ |
| `shift`/`diff` on=col — single column (v0.8.2) | `_shift_on` | Multi-column on — v0.8.2a |
| `groupby(...).shift(1, on='ts')` — не поддержано | `CadwDataFrame` | Per-group shift — v0.8.3+ |
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
- ~~P-D1: dedup pandas chain unchecked~~ — **closed v0.8.9**:
  `check_state(5 × table.nbytes)` before `to_pandas`. Effective partition
  limit shrinks to `avail × 0.1` (honest RSS bound; Arrow-native rewrite
  deferred to v0.8.10).
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