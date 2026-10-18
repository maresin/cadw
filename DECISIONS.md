# DECISIONS.md — Архитектурные решения CADW

> Версия: 2026-10-18
> Назначение: единый документ, восстанавливающий контекст проекта с нуля.
> Читатель, не знакомый с CADW, должен понять цель, устройство, алгоритмы и обоснования.
> Спецификации примитивов включены полностью. Раздел §11.4 явно перечисляет известные нарушения принципа памяти (must-fix) — это не «приятные улучшения», а условия, при которых гарантия out-of-core пока не выполнена.

---

## 0. Метаданные документа

**Как читать.** Документ самодостаточен. Если вы не знакомы с CADW, начните с §1–§4 (цель, глоссарий, принципы, архитектура). Затем §5–§9 — алгоритмы. §11 — что реально работает и где ещё есть нарушения принципа. §12 — что делать дальше. §13 — почему приняты те или иные решения и какие грабли пройдены.

**Что означает «Что осталось за кадром» в отчётах о шагах.** Это не «приятные улучшения». Каждый такой пункт классифицируется в §11.4–§11.6 в одну из трёх категорий:
- **§11.4 — нарушение принципа памяти (must-fix).** При определённых данных или настройках гарантия «в RAM не более одного батча» нарушается. До исправления гарантия условна.
- **§11.5 — осознанное ограничение (документировано).** Принцип не нарушается, но функциональность неполная или поведение отличается от pandas.
- **§11.6 — технический долг.** Принцип не нарушается, но замедляет/шумит/усложняет миграцию.

Этот раздел пересматривается на каждом шаге: если пункт из §11.6 переходит в §11.4 (например, эвристика оказалась опасной), это фиксируется как регрессия архитектуры, а не как баг.

---

## 1. Что такое CADW и для кого

**CADW** — локальный out-of-core фреймворк для обработки табличных данных, не помещающихся в оперативную память, на слабых машинах (ноутбук, десктоп без GPU, без распределённого кластера).

**Типичный пользователь:** data scientist, у которого 50–500 ГБ Parquet-файлов, 8–16 ГБ RAM и желание запустить `groupby`, `join`, `sort_values`, обучить LightGBM или сделать feature engineering, не покупая кластер.

**Что CADW не делает:**
- не распределяет вычисления по узлам (это Dask/Spark);
- не ускоряет отдельные операции за счёт GPU (это RAPIDS);
- не заменяет pandas для данных, помещающихся в RAM.

**Что CADW делает:** гарантирует, что операция завершится в заданном бюджете памяти за счёт сброса промежуточных результатов на диск и потоковой обработки батчами. **Гарантия условна там, где §11.4 ещё не закрыт** — это явно указано для каждого случая.

---

## 2. Глоссарий

| Термин | Значение |
|---|---|
| **Батч (batch)** | Порция строк, находящаяся в RAM в данный момент. Размер контролируется `MemoryGuard`. Батч всегда помещается в память. |
| **Чанк (chunk)** | Физический блок строк на диске (Row Group в Parquet, `RecordBatch` в Arrow IPC). Может быть больше батча. |
| **Партиция (partition)** | Файл на диске, куда `HashShuffle` сложил строки с одинаковым хэшем ключа. Партиции не пересекаются по ключам. |
| **Run** | Отсортированный файл на диске, созданный на первом этапе внешней сортировки. |
| **`row_id`** | Глобальный идентификатор строки: позиция в исходном датасете (0..N-1). Сохраняется во всех промежуточных структурах. |
| **Примитив** | Один из трёх фундаментальных алгоритмов: `Reduce`, `HashShuffle`, `Sort`. |
| **Локальная операция** | Функция, применяемая к одному батчу или к одной партиции. Векторизована. |
| **Моноид** | Ассоциативная операция слияния + нейтральный элемент. Если агрегат — моноид, его можно считать за один проход. |
| **`RowFilter`** | Ленивый фильтр строк (срезы, маски, удаление). Не реализован (см. §10, §11.3). |
| **`CadwIndex`** | Слой идентичности строк. Не реализован (см. §10, §11.3). |
| **`Dataset`** | Хранилище: файлы колонок, схема, индекс, маска удалённых. Не реализован (см. §10). |
| **`CadwDataFrame`** | Пользовательский фасад. Не реализован (см. §10, §11.3). |

---

## 3. Ключевые принципы (обязательные)

1. **Запрет на накопление данных в памяти.** В любой момент — не более одного батча плюс контролируемые буферы. Нарушение = баг. Известные нарушения перечислены в §11.4.
2. **Неизменяемость.** Файлы не перезаписываются. Новая версия колонки — новый файл.
3. **Apache Arrow** — единый формат в памяти и для кэша (IPC). Исходные данные — Parquet.
4. **Pandas-подобный API.** Пользователь не переучивается.
5. **`MemoryGuard` в байтах.** Размер батча и лимиты буферов вычисляются.
6. **Ленивость.** Срезы, фильтры, удаление строк — ленивые. Физическая перестройка — по явному запросу.
7. **Три примитива.** Тяжёлое сводится к `Reduce`, `HashShuffle`, `Sort`. Остальное — композиции.

---

## 4. Архитектура: уровни и модули

### 4.1. Уровни

```
┌─────────────────────────────────────────────────────────────┐
│ 6. ML / визуализация: конвертеры, partial_fit, батчи в torch │
├─────────────────────────────────────────────────────────────┤
│ 5. CadwDataFrame: pandas-подобный API                        │
├─────────────────────────────────────────────────────────────┤
│ 4. Операции: groupby, join, unique, sort_values, ...         │
├─────────────────────────────────────────────────────────────┤
│ 3. Примитивы: Reduce | HashShuffle | Sort                    │
├─────────────────────────────────────────────────────────────┤
│ 2. Доступ: BatchIterator + RowFilter                         │
├─────────────────────────────────────────────────────────────┤
│ 1. Dataset: файлы колонок, схема, CadwIndex, маска удалённых │
├─────────────────────────────────────────────────────────────┤
│ 0. Хранение: Parquet (источник), Arrow IPC (кэш, партиции)   │
└─────────────────────────────────────────────────────────────┘
```

### 4.2. Модули

```
cadw/
├── io/
│   ├── batch_reader.py      # ParquetBatchReader, ArrowIPCBatchReader
│   └── batch_writer.py      # ParquetBatchWriter, ArrowIPCBatchWriter
├── core/
│   ├── memory_guard.py      # бюджет, размеры батча/состояния, check_*
│   ├── logging.py           # get_logger, configure
│   └── spill_manager.py     # LEGACY (удаляется, см. §11.2)
├── ops/
│   ├── reduce.py            # ReduceOp, reduce(), CountOp..HistogramOp
│   ├── shuffle.py           # compute_hashes, _mod_p, hash_shuffle
│   ├── sort.py              # external_sort, _merge_keys, _Key
│   └── groupby.py           # groupby_aggregate
├── operations/
│   ├── unique_collector.py  # LEGACY
│   └── groupby_aggregator.py# LEGACY
├── dataframe.py             # ЗАГОТОВКА (см. §10)
└── __init__.py
```

### 4.3. Сквозной пример: `df.groupby('cat')['val'].mean()`

Данные (4 строки):

| cat | val |
|---|---|
| A | 10 |
| B | 20 |
| A | 30 |
| B | 40 |

**Ожидание:** A → 20.0, B → 40.0.

1. **`HashShuffle`** по `cat` с P партициями. Для P=2: `A` → partition_0, `B` → partition_1 (при данных seed/P). В каждой партиции — строки одного ключа, плюс `row_id`.
2. **`_local_groupby`** по каждой партиции: `pa.Table.group_by('cat').aggregate([('val','mean')])` → 1 строка на партицию.
3. **`pa.concat_tables`** — партиции не пересекаются по ключам, merge не нужен.
4. Пик RAM: один батч на фазе 1, один батч партиции + словарь на фазу 2. **При условии**, что партиция влезает в память (см. §11.4).

---

## 5. Примитив 1: `Reduce`

### 5.1. Назначение и граница

**Reduce** — свёртка потока батчей в один аккумулятор. Состояние **не зависит от N**. Если состояние растёт с N — это GroupBy или Sort.

| Подходит | Не подходит |
|---|---|
| `count`, `sum`, `min`, `max` | `median`, точные `quantile` |
| `mean` | `nunique`, `mode` |
| `var`, `std` (Welford) | `distinct` |
| `cov`, `corr` (d×d) | |
| `histogram` (binning для LightGBM) | |
| состояние SGD / `partial_fit` | |

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

**Архитектурное правило:** `update` и `merge` — только векторизованные операции (`pa.compute`, `np`). Построчные `as_py()` запрещены.

**Инвариант:** результат не зависит от `batch_size` с точностью до плавающей точки.

### 5.3. Числовой пример (mean, batch_size=2)

Данные: `x = [10, 20, 30, 40, 50]`.

| Шаг | В RAM | state `(sum, count)` |
|---|---|---|
| init | — | `(0.0, 0)` |
| batch #1 `[10,20]` | 2 числа | `(30.0, 2)` |
| batch #2 `[30,40]` | 2 числа | `(100.0, 4)` |
| batch #3 `[50]` | 1 число | `(150.0, 5)` |
| finalize | — | `150/5 = 30.0` |

Спилла нет. **Reduce — единственный примитив без записи на диск.** Если реализация пишет — это архитектурная ошибка.

### 5.4. Реализации и замечания

| Операция | Состояние | Замечание |
|---|---|---|
| `CountOp` | `(n_valid, n_null)` | skipna по умолчанию |
| `SumOp` | `(sum_f64, n_valid)` | промоушен до float64, **int64 overflow не обрабатывается** (§11.5) |
| `MinMaxOp` | `(scalar, valid)` | `pc.min_max` |
| `MeanOp` | `(sum, n)` | — |
| `StdOp` | `(n, mean, M2)` | Welford; **не** `sum(x²)/n − mean²` |
| `CovarianceOp` | `(n, sum_x[d], sum_xx[d,d])` | состояние O(d²); при большом d — §11.5 |
| `HistogramOp` | `counts[B]` | требует `min_val`/`max_val` (двухпроходный) |

**Welford (Chan et al.):** при `1e9 + [1,2,3]` даёт `std=1.0`; наивная формула — мусор. Числовая устойчивость — часть контракта.

### 5.5. Пограничные и узкие места

- 0 строк → `finalize` даёт NaN/0 по конвенции.
- 1 строка → `std` sample = NaN; population = 0.
- Все null → skipna → `mean=NaN`.
- `int64` overflow в `sum` → **не обрабатывается** (float64 теряет точность за 2^53). §11.5.
- `to_pylist()` на батче — O(N) аллокаций; использовать `to_numpy()` для чисел.
- `finalize` при `count=0` — явно делить на ноль.
- Маленький batch — накладные расходы Python доминируют; `min_batch_rows ≥ 1000`.

### 5.6. Тесты (33/33)

Инвариант batch_size для `mean/sum/std/min_max/count`; пограничные (0/1/null/последний partial); `1e9+[1,2,3]` для std; «Reduce не пишет на диск» (monkeypatch `tempfile`); cov 2D; histogram; `check_state` для cov(d=10) при 1 KB.

---

## 6. Примитив 2: `HashShuffle`

### 6.1. Назначение и инвариант

Раскладывает строки по P файлам так, что **все строки с одинаковым ключом — в одном файле**. Партиции **не пересекаются по множеству ключей** (хэш-коллизии между разными ключами — норма; downstream-агрегация их различает).

Применяется для: `groupby`, `join`, `unique`, `drop_duplicates`, `set operations`, `pivot`, `target encoding`, `stratified sampling`.

### 6.2. `row_id` — обязательное поле

**Каждая строка в каждой партиции несёт `row_id`** — глобальную позицию в исходном датасете. Зачем: восстановить идентичность после shuffle/sort; корректный `join`; `drop_duplicates(keep='first')`; `groupby + first/last`.

**Критическая деталь:** при переупорядочивании через `pc.sort_indices` `row_id` строки на позиции `i` в отсортированном батче — это `order_np[i] + offset`, а не `i + offset`. Ошибка здесь была поймана тестом `test_concat_equals_original`.

### 6.3. Детерминированный хэш

`hash()` в Python зависит от `PYTHONHASHSEED` — нестабилен между процессами. Для join/groupby это фатально.

**Реализация:**
- `xxhash.xxh64(_stable_bytes(v), seed=seed).intdigest()`.
- `_stable_bytes(v) = repr(v).encode('utf-8')` — различает типы (`repr('1')` ≠ `repr(1)`).
- Для нескольких ключей — `boost::hash_combine` (64-битная версия):
  `h = h ^ (ch + 0x9E3779B97F4A7C15 + (h << 6) + (h >> 2))`.
- Арифметика — **через NumPy `uint64`**, а не `pc.add`/`pc.shift_left`: Arrow-функции над `uint64` возбуждают `ArrowInvalid` при переполнении, тогда как wrap-around — часть контракта хэша.
- `NULL` в ключе хэшируется детерминированно (`repr(None)`), образуя отдельную группу (SQL-семантика).

### 6.4. Выбор P

K (число групп) заранее неизвестно. Стратегия:
1. **Sample:** прочитать `min(10_000, N)`, `K_est = K_sample * N / s`, `P = ceil(K_est * safety / groups_per_partition)`, `safety=1.5`, `groups_per_partition = available_bytes * 0.6 / bytes_per_key`.
2. **Рекурсия** при переполнении партиции — **не реализована** (§11.4).
3. **Комбинация** — цель.

**Инвариант:** каждый файл партиции ≤ `available_bytes * 0.7` — **проверяется только через `check_partition_buffers` (буферы записи), не через фактический размер партиции.** §11.4.

### 6.5. Алгоритм

```python
for batch in batch_iter:
    memory_guard.check_batch(batch)
    hashes = compute_hashes(batch, key_columns, seed)       # uint64
    parts = _mod_p(hashes, num_partitions)                  # см. ниже
    order = pc.sort_indices(parts)                          # одна перестановка
    sorted_batch = batch.select(key + value).take(order)
    sorted_parts = parts.take(order)
    order_np = order.to_numpy(zero_copy_only=False)
    for (start, end, p) in boundaries(sorted_parts):
        piece = sorted_batch.slice(start, end - start)
        row_ids = pa.array(order_np[start:end] + row_id_offset, pa.uint64())
        writers[p].write_batch(piece.append_column("row_id", row_ids))
    row_id_offset += batch.num_rows
```

`_mod_p`: для `P = 2^k` → `pc.bit_wise_and(hashes, P-1)`; иначе → `np.uint64 % np.uint64` (float64 теряет точность на 64-битных хэшах).

### 6.6. Числовой пример (8 строк, P=4)

| row_id | cat | val |
|---|---|---|
| 0 | A | 10 |
| 1 | B | 20 |
| 2 | C | 30 |
| 3 | D | 40 |
| 4 | A | 11 |
| 5 | B | 21 |
| 6 | C | 31 |
| 7 | D | 41 |

При `h(A)%4=0, h(B)%4=1, h(C)%4=2, h(D)%4=3`:

| partition | строки |
|---|---|
| 0 | (A,10,0), (A,11,4) |
| 1 | (B,20,1), (B,21,5) |
| 2 | (C,30,2), (C,31,6) |
| 3 | (D,40,3), (D,41,7) |

**Инвариант:** `A` только в 0, `B` только в 1, и т.д. Агрегация независима, результаты конкатенируются.

### 6.7. Пограничные, узкие места, skew

- 0 строк → **`ValueError`**: без schema нельзя создать writers (§11.5).
- P = 1 → один файл.
- P > N → часть партиций пуста.
- NULL в ключе → отдельная группа.
- **Skew** (один ключ > 10% N): партиция не влезает при любом P. Решение — изоляция тяжёлых ключей (потоковая агрегация без хранения) — **не реализовано** (§11.4).
- **Combiner** (частичная агрегация до записи) — §11.6.
- `compute_hashes` использует `to_pylist()` — временный Python-список; memory overhead не покрыт `check_batch` (§11.4, near-violation).

### 6.8. Тесты (13/13)

Партиции не пересекаются по ключам (правильный инвариант — «for each key, all rows in exactly one partition», не «partition has ≤ 1 key»); `row_id` сохраняется и восстанавливает исходные данные; P>N; P=1; пустой вход → ValueError; стабильность хэша между батчами; разные seed → разные хэши; `estimate_num_partitions`; `check_partition_buffers` срабатывает.

---

## 7. Примитив 3: `Sort`

### 7.1. Назначение

Внешняя сортировка. Нужна для `sort_values`, `rank`, точных `quantile`/`median`, оконных функций с `ORDER BY`, `head(n)`/`tail(n)` по группе.

### 7.2. Стабильность

Каждая строка в run'е содержит `row_id`. Tie-break при равных ключах — по `row_id`. Сортировка стабильна и воспроизводима.

### 7.3. Алгоритм (два этапа)

**Этап 1 — генерация run'ов.** Читаем батчи до `available_bytes * 0.5`. Сортируем в памяти: `pc.sort_indices(table, sort_keys=[(col, asc/desc), ...], null_placement=...)`. Пишем run в Arrow IPC (все нужные колонки + `row_id`).

**Важно:** run содержит **объединение** `value_columns` и `key_columns` без дублей (`_unique_columns`). Если пользователь передал `val` и как value, и как key — в схеме run'а `val` один раз. Иначе `pc.sort_indices` падает с `Multiple matches for FieldRef.Name(val)`.

**Этап 2 — k-way merge.** Открываем k run'ов. Heap по `_Key` (класс с `null_rank ∈ {0,1,2}` для `nulls_first`/`nulls_last`, per-column `asc/desc`), tie-break — `row_id`. Извлекаем строку, пишем в выходной `BatchWriter` батчами по `read_batch_size`. При исчерпании блока run'а — подгружаем следующий. Если run'ов > k — многоуровневый merge.

### 7.4. Числовой пример (8 строк, batch_size=2)

| row_id | val |
|---|---|
| 0 | 40 |
| 1 | 10 |
| 2 | 30 |
| 3 | 20 |
| 4 | 50 |
| 5 | 5 |
| 6 | 15 |
| 7 | 25 |

**4 run'а:** `(10,1),(40,0)`; `(20,3),(30,2)`; `(5,5),(50,4)`; `(15,6),(25,7)`.

**k=4 merge:** heap `10,20,5,15` → извлекаем `5` (run_2), подгружаем `50`; `10` (run_0) → `40`; `15` (run_3) → `25`; `20` (run_1) → `30`; `25`; `30`; `40`; `50`.

**Итог:** `5,10,15,20,25,30,40,50`; `row_id`: `5,1,6,3,7,2,0,4`.

**Пик RAM:** `k * buffer_size` + heap + output buffer.

### 7.5. Пограничные, узкие места, ограничения

- 0/1 строка → тривиально.
- Уже отсортировано → run'ы генерируются, merge быстрый.
- Все равны → tie-break по `row_id`.
- NULL → `nulls_first`/`nulls_last` (сейчас глобальный флаг, не per-column — §11.5).
- `k < 2` → ошибка конфигурации.
- **`merge_fan_in` эвристика:** `k = available // (read_batch_size * 128)`. Не выведена из реального размера строки; при больших `read_batch_size` может держать слишком много батчей → **near-violation** (§11.4).
- **FutureWarning** `null_placement` в `SortOptions` deprecated с 25.0.0 — §11.6.

### 7.6. Тесты (10/10)

Asc/desc; стабильность по `row_id`; multi-key; nulls_first/last; многоуровневый merge (100 строк, batch=1, fan_in=4); уже отсортировано; пустой вход → ValueError; `row_id` уникален.

---

## 8. Почему три примитива, а не N

### 8.1. Композиции

| Операция | Композиция |
|---|---|
| `mean/sum/std/cov/histogram` | `Reduce` |
| `StandardScaler`, `partial_fit` | `Reduce` (+ map) |
| `t-digest` (приближённые квантили) | `Reduce` со скетчем |
| `groupby().agg()` | `HashShuffle` + локальная агрегация + `Concat` (**реализовано**) |
| `merge` / `join` | `HashShuffle` обеих сторон + merge в партиции + `Concat` |
| `unique`, `drop_duplicates` | `HashShuffle` + `first` + `Concat` |
| `pivot`, `crosstab` | `HashShuffle` + агрегация + `Concat` |
| `sort_values` | `Sort` (**реализовано**) |
| `rank`, точные `median`/`quantile` | `Sort` + локальный ранг |
| окна с `ORDER BY` | `HashShuffle` (PARTITION BY) + `Sort` + локальное окно |
| `resample`, `rolling` | `HashShuffle` + `Sort` + локальное окно с halo-буфером |

### 8.2. Что не сводится

- **Cross join** — O(m×n) строк; материализация неизбежна. Не наш случай.
- **Произвольный `apply` над группой**, требующий всю группу: после `HashShuffle` группа может влезть (тогда ок) или нет (тогда `Sort`/потоковый алгоритм под задачу).
- **Экстремальный skew** — изоляция тяжёлых ключей (§11.4).

---

## 9. `MemoryGuard`: бюджет и проверки

### 9.1. Как выбирается размер батча

```
batch_rows = floor( available_bytes * safety / (row_bytes_input * expansion_factor) )
```
- `row_bytes_input` — по схеме файла или по первому батчу.
- `expansion_factor` — по плану: `filter/project ≤ 1.0`, `with_column = 1 + bytes_new/bytes_row`, для неизвестной операции — `2.0` по умолчанию.
- `safety ≈ 0.7`.

**Пока план не реализован** (`RowFilter`, `Plan` — §10), `expansion_factor` передаётся вызывающим вручную или берётся дефолт `2.0`.

### 9.2. Обязательный интерфейс

| Метод | Назначение |
|---|---|
| `available_bytes() -> int` | текущий бюджет |
| `get_batch_rows(row_bytes, expansion_factor) -> int` | верхняя граница |
| `min_batch_rows() -> int` | нижняя граница (≥1000) |
| `check_batch(batch)` | `batch.nbytes ≤ available`; иначе `MemoryLimitExceeded` |
| `check_state(state_bytes)` | `state ≤ available * 0.5` |
| `estimate_state_size(op, d, bins)` | до запуска (cov, histogram) |
| `estimate_unique_keys(sample_batch, total_rows)` | для выбора P |
| `check_partition_buffers(P, buffer)` | `P * buffer ≤ available * 0.3` |
| `recommended_partition_buffer_bytes(P)` | `buffer_limit // P` |
| `reserve/release(nbytes)` | для параллельных операций (позже) |
| `log_step(...)` | подробное логирование |

**Инвариант:** если `check_batch`/`check_state` падают — операция **не продолжается**. Молчаливое превышение — баг.

### 9.3. Параметры по умолчанию

`safety=0.7`, `state_safety=0.5`, `buffer_safety=0.3`, `min_batch_rows=1000`, `default_expansion_factor=2.0`, `bytes_per_key=80`, `groups_per_partition = available * 0.6 / 80`.

---

## 10. Dataset, CadwIndex, CadwDataFrame (проект, не реализовано)

### 10.1. `Dataset` — хранилище

`Dataset` — набор файлов колонок (Arrow IPC или Parquet), схема, `CadwIndex`, маска удалённых, метаданные чанков.

| Компонент | Назначение |
|---|---|
| `_columns: dict[str, ColumnMetadata]` | файлы колонок, `chunk_size`, `num_chunks`, dtype |
| `_schema: pa.Schema` | имена и типы |
| `_index: CadwIndex` | идентичность строк |
| `_deleted_mask` | битовая маска удалённых (ленивое удаление) |
| `_row_filter` | ленивый фильтр (срезы, маски) |

**Инвариант:** все колонки имеют одинаковый `chunk_size`. При добавлении колонки с другим чанкованием — датасет перестраивается.

### 10.2. `CadwIndex` — идентичность строк

| Реализация | Когда |
|---|---|
| `RangeIndex(start, stop, step)` | по умолчанию, без материализации |
| `MaterializedIndex(pa.Array)` | после `set_index` или sort |
| `DeleteMask(bitmap)` | ленивое удаление |

### 10.3. `CadwDataFrame` — фасад

`CadwDataFrame = (Dataset, RowFilter, column_selection)`. API — pandas-подобный. Методы либо строят ленивый план, либо делегируют примитивам.

**Пример `df['c'] = df['a'] + df['b']`:**
- **pandas (BlockManager):** `a`, `b` — в одном 2D-блоке; `a+b` — по столбцам; присваивание `c` может потребовать консолидации (скрытая копия).
- **CADW (проект):** `a`, `b` — независимые `ChunkedArray`; `a+b` — поэлементно; `c` — новый `ChunkedArray`. Никакой консолидации.

**Воспроизводить `BlockManager` не нужно** (скрытые копии — причина OOM). Воспроизводим интерфейс (`loc`, `iloc`, `join`, `sort`, `groupby`) на модели «независимые `ChunkedArray` + явный `CadwIndex`».

---

## 11. Степень реализации

### 11.1. Что работает (проверено тестами)

| Модуль | Статус | Тестов |
|---|---|---|
| `ParquetBatchReader`, `ArrowIPCBatchReader` | ✅ | legacy |
| `ParquetBatchWriter`, `ArrowIPCBatchWriter` | ✅ | legacy |
| `MemoryGuard` (байтовый контракт) | ✅ | 25 |
| `Reduce` (`count/sum/min/max/mean/std/cov/histogram`) | ✅ | 33 |
| `HashShuffle` (`xxhash`, `row_id`, `_mod_p`) | ✅ | 13 |
| `Sort` (k-way merge, стабильность, multi-level) | ✅ | 10 |
| `groupby_aggregate` (`HashShuffle + LocalGroupBy + Concat`) | ✅ | 10 |
| must-fix §11.4 (02a, 02b1, 02b2) | ✅ | 28 |
| `Dataset` (03a, per-column Arrow IPC, sync batches) | ✅ | 13 |
| `CadwDataFrame` (03b, минимальный фасад) | ✅ | 17 |

**Ключевой результат:** `groupby_aggregate` корректен при `K > памяти` — `test_groupby_many_groups` (1000 групп, 200 KB бюджет, 16 партиций → ~62 группы на партицию). Это **условная** гарантия: при `num_partitions=8` и K=10^6 та же операция упадёт (§11.4).

### 11.2. Legacy (заменяется, не удалено)

| Модуль | Почему legacy | План |
|---|---|---|
| `GroupByAggregator` | построчный Python-цикл | удалить после миграции фасада |
| `UniqueCollector` | теряет первый батч; `SpillManager.merge` с багом | заменить на `HashShuffle + first` |
| `SpillManager` | `merge()` перезаписывает вместо слияния | удалить |

Старые тесты (`test_unique.py`, `test_groupby.py`) проходят, но проверяют legacy-путь.

### 11.3. Что отсутствует

- `CadwDataFrame` (фасад), `Dataset`, `CadwIndex`, `RowFilter` — §10.
- Ленивые срезы, `__getitem__` для колонок/фильтров.
- `unique`/`drop_duplicates` через новые примитивы.
- `join`, `merge`.
- `Plan` (ленивый план) — нужен для `estimate_expansion_factor` в `MemoryGuard`.
- Параллелизм чтения колонок.
- E2E тест «Parquet → groupby → Parquet».
- CLI-настройка логирования (`configure()` не вызывается).

### 11.4. Нарушения принципа памяти (закрыто, патчи 02a–02b2)

**Статус: закрыто полностью.** Все 5 пунктов исправлены в патчах 02a (п.2, 4, 5), 02b1 (п.1) и 02b2 (п.3). Гарантия out-of-core для `Reduce`, `HashShuffle`, `Sort`, `groupby_aggregate` — **безусловна** при условиях §11.5.

| # | Где | Что происходит | Риск | План |
|---|---|---|---|---|
| 1 | `_local_groupby` в `groupby.py` | `pa.Table.from_batches(list(reader))` материализует **всю партицию** без `MemoryGuard.check_batch`. | Если партиция > бюджета → OOM. `test_groupby_many_groups` проходит только потому, что P=16 при K=1000. | Добавить `check_batch` на аккумулируемый размер; при переполнении — рекурсивное партиционирование партиции. |
| 2 | `groupby_aggregate` | `num_partitions=8` по умолчанию, **не выводится** из `MemoryGuard` / `K_est`. | При K=10^6 и бюджете 100 MB каждая партиция ~125k групп → OOM. | Авто-выбор P: `estimate_unique_keys` (sample) + `estimate_num_partitions`. |
| 3 | `hash_shuffle` | Skew-рекурсия **не реализована**. | Один ключ с 90% строк → его партиция не влезает при любом P → OOM в `_local_groupby`. | Изоляция тяжёлых ключей: sample перед shuffle, для ключей > 10% N — потоковая агрегация без хранения. |
| 4 | `Sort` | `merge_fan_in` эвристика: `k = available // (read_batch_size * 128)`. Не выведена из реального размера строки. | При больших `read_batch_size` и «широких» строках `k * batch` может превысить бюджет. | `k = available / (read_batch_size * row_bytes_estimate)`, с проверкой через `check_partition_buffers`. |
| 5 | `compute_hashes` | `to_pylist()` создаёт Python-список на батч; memory overhead (Python int ~28 B) не покрыт `check_batch`. | Пик RAM ≈ `batch.nbytes + num_rows * ~30 B`. Для int-колонок — x3–x5. Bounded by batch, но константа существенна. | Векторизовать: `pc.cast` в binary + xxhash по буферу, или `np.void` view; либо явно закладывать коэффициент в `check_batch` перед вызовом. |

**Обновление (патч 02b1):** пункт **1** (`_local_groupby` без `check_batch` + нет рекурсии) — **закрыт**: партиция оценивается по `sum(b.nbytes)`, при переполнении — рекурсивный `hash_shuffle` с новым seed, при `depth >= MAX_DEPTH` — `MemoryLimitExceeded` с указанием на skew. Дополнительно: `groupby_aggregate` проверяет размер sample-буфера (`sample_bytes > available*0.5` → исключение). **Остаётся пункт 3 (skew):** при экстремальном skew (один ключ > 10% N) рекурсия не помогает, нужен skew-детектор + изоляция тяжёлых ключей (патч 02b2).

**Обновление (патч 02a):** пункты **2** (авто-`num_partitions`), **4** (`merge_fan_in` из `row_bytes`), **5** (`check_batch(extra_factor)` для `compute_hashes`) — **закрыты**. Остаются пункты 1 и 3 (см. патч 02b).

**Итоговый статус (после 02b2-fix3):**
- `Reduce` — полная (нет записи на диск, состояние O(1) по N).
- `HashShuffle` — полная (детерминированный xxhash, skew-изоляция, учёт overhead).
- `Sort` — полная (k из `compute_fan_in`, консервативно).
- `groupby_aggregate` — полная (авто-P, рекурсия, skew-изоляция).

Тестов: **119 passed** (было 91 на baseline, 25 memory_guard, 33 reduce, 13 shuffle, 10 sort, 10 groupby_v2, 12 must-fix-easy, 5 must-fix-local-groupby, 11 must-fix-skew, + legacy в `_legacy/`, исключён из основного прогона).

### 11.5. Осознанные ограничения (документированы, не нарушения)

Принцип памяти не нарушается, но поведение отличается от «идеального».

| Ограничение | Где | Обход |
|---|---|---|
| `sum` промоутит в `float64`; int64 overflow не обрабатывается | `SumOp` | для точных int-сумм — своя реализация с `int128`/Decimal |
| `cov` состояние O(d²); при d=10^4 — 800 MB | `CovarianceOp` | блочный Reduce по столбцам, PCA до cov |
| `corr`, `t-digest`, `nunique`, `median`, `quantile` (точные) — не реализованы | `ReduceOp` | `median`/`quantile` — через `Sort`; `nunique` — через `HashShuffle + count` |
| `nulls_first` — глобальный флаг, не per-column | `Sort` | per-column — расширение `_Key` |
| Пустой вход → `ValueError` | `hash_shuffle`, `external_sort`, `groupby_aggregate` | передать schema явно — не реализовано |
| Агрегации ограничены `{mean, sum, count, min, max}` | `groupby_aggregate` | `std`/`var` — через `Reduce`-совместимую пару (sum, count, M2) — не реализовано |
| Хэш-коллизии между разными ключами — норма | `HashShuffle` | downstream различает по значению ключа |
| `compute_hashes` — Python-цикл (медленно на больших батчах) | `HashShuffle` | векторизация (см. §11.4 п.5) |

### 11.6. Технический долг (не нарушает принцип, но замедляет/шумит)

| Долг | Где | Влияние |
|---|---|---|
| `FutureWarning: null_placement deprecated` | `Sort` | шум в тестах; заменить на per-sort-key |
| `estimate_num_partitions` не вызывается автоматически | `groupby_aggregate` | см. §11.4 п.2 |
| Combiner (частичная агрегация до записи партиций) отсутствует | `HashShuffle` | объём партиций больше, чем мог бы быть |
| Параллелизм чтения колонок отсутствует | `BatchReader` | медленнее на SSD |
| Legacy `GroupByAggregator`, `UniqueCollector`, `SpillManager` | `operations/`, `core/` | дублирование кода |
| `configure()` логирования не вызывается | CLI | логи по умолчанию не настроены |
| `_Key` работает через `as_py()` — медленно для сложных типов | `Sort` | сравнение строк/дат дороже, чем чисел |

---

## 12. Дорожная карта

### 12.1. Приоритет 0 — закрыть §11.4 (must-fix) — ✅ ЗАКРЫТО (02a–02b2)

1. `_local_groupby`: `check_batch` на аккумулируемый размер партиции; рекурсия при переполнении.
2. `groupby_aggregate`: авто-выбор `num_partitions` через `MemoryGuard.estimate_unique_keys` + `estimate_num_partitions`.
3. `hash_shuffle`: skew-детектор (sample) + изоляция тяжёлых ключей (потоковая агрегация для sum/count/min/max/mean).
4. `Sort`: `merge_fan_in` из `row_bytes_estimate`, проверка через `check_partition_buffers`.
5. `compute_hashes`: векторизация без `to_pylist` (или явный коэффициент в `check_batch`).

### 12.2. Приоритет 1 — фасад

6. `CadwDataFrame` (минимальный): держит `Dataset`, методы `groupby`, `unique`, `to_batches`, `write_parquet`. Внутри — примитивы.
7. `Dataset` (базовый): файлы колонок, метаданные, `iter_batches`.
8. `RowFilter`: `Slice`, `BooleanMask`, `And/Or`.
9. `CadwIndex`: `RangeIndex` + материализация.
10. Ленивые `__getitem__`.

### 12.3. Приоритет 2 — миграция legacy

11. `GroupByAggregator` → `groupby_aggregate`.
12. `UniqueCollector` → `HashShuffle + first` (или `Reduce` со `SetOp` при `K ≤ памяти` — см. §13.5).
13. Удалить `SpillManager` и legacy-тесты после зелёной миграции.

### 12.4. Приоритет 3 — доводка

14. E2E тест (Parquet → groupby → Parquet, бюджет 10–50 MB).
15. Параллельное чтение колонок.
16. CLI-настройка логирования.
17. `nulls_first` per-column в `Sort`.

### 12.5. Приоритет 4 — ML и визуализация

18. `to_batches(format='numpy'/'torch')`.
19. `partial_fit`-интеграция (SGD, MiniBatchKMeans).
20. LightGBM binning через `HistogramOp` + `Dataset`-адаптер.

---

## 13. История обсуждения и антипаттерны

### 13.1. Почему отказались от предыдущей реализации

Прежний код (`merged_cadw8.py`) содержал `VirtualOperationManager` с `np.arange(total_rows)` (материализация всех индексов), заглушки `FilterOperation`/`DropNaOperation`/`DropDuplicatesOperation` (возвращали все индексы), `chunked_filter.py` (физические копии), `pd.DataFrame` внутри чанков (overhead). Вывод: «виртуальные индексы в памяти» несовместимы с принципом «не накапливать».

### 13.2. Антипаттерны аналогичных проектов

| Проект | Антипаттерн | Урок для CADW |
|---|---|---|
| **Dask** | `split_out=1` по умолчанию → OOM; партиции — `pd.DataFrame` | Размер батча и число партиций вычислять явно |
| **Spark** | JVM-overhead, GC-паузы | Python + Arrow дают предсказуемый контроль |
| **pandas (BlockManager)** | Скрытые копии, непубличный API, консолидация | Не изобретать блочную структуру; Arrow готова |
| **Polars** | Собственный `ChunkedArray`; до 97% фрагментации | Контролировать `chunk_size` |
| **Lance/Vortex** | Сложные page-based форматы | Взять идею soft-delete + compaction, остаться на Arrow IPC + Parquet |

### 13.3. Принятые решения и отвергнутые альтернативы

| Решение | Альтернатива | Почему отвергли |
|---|---|---|
| Arrow IPC для кэша | pickle | медленно, небезопасно, не Arrow-native |
| Arrow IPC для кэша | Lance | сложность, зависимость, ошибки |
| Parquet для источника | CSV | медленно, нет типов, нет row groups |
| Батчи как `RecordBatch` | `pd.DataFrame` | overhead, риск OOM |
| `dict[str, ChunkedArray]` | единый `pa.Table` | Table навязывает синхронизацию |
| `HashShuffle + локальная агрегация` | `dict + spill + merge` | merge требует, чтобы K влезало; целевой случай — K > памяти |
| Три примитива | алгоритм на операцию | дублирование, несогласованность |
| `xxhash.xxh64(repr(v).encode())` | `pc.hash` | `pc.hash` отсутствует в PyArrow 24–26 |
| NumPy `uint64` для `hash_combine` | `pc.add`/`pc.shift_left` | Arrow возбуждает `ArrowInvalid` на overflow |
| `_unique_columns(value, key)` | передавать как есть | дубликат колонки → `Multiple matches for FieldRef.Name` |

### 13.4. Уроки из отладки (9 эпизодов)

Падения классифицированы по причинам. Каждый эпизод — повод для чек-листа.

1. **Тест, а не код:** `test_reserve_release` ожидал исключение, которого по логике быть не должно (`_reserved=200`, `reserve(300)`, budget=1000 → OK). **Симптом:** тест «логически неверный». **Урок:** перед интерпретацией падения проверить арифметику теста.
2. **API-контракт:** `pa.array(..., type_=...)` → в PyArrow параметр `type`. **Симптом:** все тесты падают на одной строке хелпера. **Урок:** если ошибка `TypeError`/`AttributeError` на одной библиотечной функции и все тесты доходят до неё — это контракт API, а не логика.
3. **API-контракт:** `arr.length` → `len(arr)`. **Симптом:** `AttributeError` на `.length`. **Урок:** тот же.
4. **Версия:** `pc.hash` отсутствует в PyArrow 24/26. **Симптом:** `AttributeError: module 'pyarrow.compute' has no attribute 'hash'`. **Урок:** проверять `hasattr` перед использованием; фиксировать версии в `pyproject.toml`.
5. **Контракт `xxhash`:** `xxh64(str)` → `TypeError: Strings must be encoded`. **Урок:** `xxhash` принимает `bytes`; `repr(v).encode('utf-8')` — стабильная сериализация с типами.
6. **Семантика `row_id`:** `row_id` присваивался по позиции в отсортированном батче (`start + offset`), а не по `order_np[start] + offset`. **Симптом:** `test_concat_equals_original` падал, `test_row_id_preserved` проходил (проверял только множество `{0..7}`). **Урок:** `sort_indices` переупорядочивает; исходная позиция строки на месте `i` — это `order_np[i]`. Тест, проверяющий только множество, слабее теста, проверяющего восстановление данных.
7. **Ложный инвариант:** `test_partitions_disjoint_by_key` утверждал «≤1 ключ в партиции». Хэш-коллизии ожидаемы. **Урок:** проверять реальный инвариант («for each key, all rows in exactly one partition»), а не усиленный.
8. **Дубликат схемы:** `Multiple matches for FieldRef.Name(val)` в `Sort` и `groupby_aggregate` — одна и та же колонка передана как value и key. **Урок:** `_unique_columns` перед построением схемы; проверять инвариант «имена колонок уникальны» на входе примитива.
9. **Overflow `uint64`:** `pc.add`/`pc.shift_left` возбуждают `ArrowInvalid`. **Урок:** для хэш-арифметики — NumPy `uint64` (wrap-around по определению), не Arrow Compute.

**Обобщение:** баги концентрируются на **границах контрактов** (API-версии, типы, семантика), а не внутри модулей. При появлении композиции (`groupby` использует `HashShuffle`) баги смещаются к стыкам. Каждая композиция нуждается в собственном наборе тестов. Следующие — `test_unique_v2`, `test_join_v2`.

### 13.5. Открытые вопросы

1. **Два пути к `unique`:** `HashShuffle + first` (корректен при `K > памяти`) vs `Reduce` со `SetOp` (быстрее при `K ≤ памяти`). Выбор — по `MemoryGuard.estimate_unique_keys` до запуска: если `K_est ≤ 0.5 * max_dict_bytes / bytes_per_key`, идти через `Reduce`; иначе — через `HashShuffle`. Порог калибруется бенчмарком.
2. **Векторизация `compute_hashes`:** `pc.cast` в binary + xxhash по буферу, или `np.void` view — бенчмарк.
3. **Формат спилла для агрегатов:** один struct или два массива (`sum`, `count`) — второй дешевле.
4. **Порог compaction:** фиксированный процент или адаптивный.
5. **Параллелизм чтения:** включать на SSD, выключать на HDD — как детектировать.
6. **`to_batches(format='torch'/'numpy')`:** где конвертировать — в итераторе или в операции.
7. **Skew-стратегия:** изоляция тяжёлых ключей потоково vs рекурсия с другим seed — что лучше для `A=90%, B=10%`.

---

## 14. Резюме для восстановления контекста

- **Работает (тесты зелёные):** `MemoryGuard` (25), `Reduce` (33), `HashShuffle` (13), `Sort` (10), `groupby_aggregate` (10) — всего 91 тест по новым примитивам, плюс legacy.
- **Главный результат:** `groupby` при `K > памяти` — `test_groupby_many_groups` (1000 групп, 200 KB бюджет).
- **Гарантия полная:** §11.4 закрыт в 02a–02b2. `Reduce`, `HashShuffle`, `Sort`, `groupby` работают при любом K и любом распределении ключей в пределах бюджета, заданного `MemoryGuard`.
- **Отсутствует:** `CadwDataFrame`, `Dataset`, `CadwIndex`, `RowFilter`, `Plan`, `unique`/`join` через новые примитивы, параллелизм, E2E.
- **Технический долг:** §11.6 (legacy-модули, FutureWarning, combiner, параллелизм).
- **Следующее действие по дорожной карте:** §12.2 — `CadwDataFrame` как фасад над `Dataset`; §12.3 — миграция legacy на новые примитивы.

