# 9. `MemoryGuard`: планировщик памяти

> v0.9.28 B.2 split — original DECISIONS.md lines 361-993.


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


### 9.2.20. GBDT top-level peak (multiclass) — §13.4/241

**Persistent per-op state** (multiclass, K classes):

    est_n_state = n·8·(1 + 4K)

Components: `y_all` (n·8), `pred` (n·K·8), `grad_all` (n·K·8),
`hess_all` (n·K·8), `round_cur_leaves` (K·n·8). Prior formula
n·K·8·3 missed `y_all` + one `round_cur_leaves` slot → 33-50%
under-report (AUDIT_ML §1.1 measured 1.37-1.44× at K=3-10; N=10M
K=10 → 880 MB miss). Fixed in v0.9.22.

**Per-batch hist peak** (still separate `check_state` in
`_AllNodeHistOp.estimate_state_nbytes`): `S × F × B × 32 B` for
current level S nodes × F features × B bins. For F=1000, B=255,
S=63 (d=5): ~514 MB — larger than base state at small N. Audited
exact (AUDIT_ML §3.1 retracted): `estimate == state_nbytes(init())`
for S∈{3,7,16,31,63} × F∈{2,100,1000} × B∈{16,255}.

**Full peak ≈** `est_n_state + max_hist + 2 × batch_X`. Not
aggregated into one formula: per-op `check_state` calls cover each
phase. For small N (hist dominates), ratio vs `n·32` blows up to
7-10×; safe because hist is separately checked.


### 9.2.21. Pipeline.plan() = sum(state_bytes), not peak — §13.4/247

`plan(df)` checks `sum(step.state_bytes) ≤ state_limit` and returns
`predicted_outcome ∈ {fits, does_not_fit}`. **State-only**: does
**not** model per-step peak (transform buffer + batch). Honest by
design: transform is per-batch streamed, each step's MemoryGuard
handles its own batch. `plan` = feasibility of fitted state, not
runtime peak. If a transform step internally needs large temp
buffers (e.g. `RobustScaler.fit` O(d) sorts), its own
`check_state` catches at runtime — plan may say `fits` and
transform may raise. Documented as intentional (§13.4/247).


### 9.2.22. KFold(materialize=True) budget ≈ 2 × check_state — §13.4/248

`KFold(n_splits, materialize=True)` calls
`check_state(20 * n)` — bounds int32 fold_id (4n) + int64
`np.arange` (8n) + int64 `rng.permutation` (8n) numpy transients.
**But** the resulting Arrow `augmented` dataset (~n × K_cols × 8 B)
is outside that bound. Measured N=100k: RSS 4.33 MB vs 20n=2 MB
(ratio 2.17×). User must budget ~2× check_state for total KFold
memory. Default (streaming) mode is O(batch), no such caveat.

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
