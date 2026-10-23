# CADW: аудит ошибок, RAM-ограничений и неэффективных алгоритмов

Дата: 2026-10-22. Версия cadw: 0.9.14 (README шапка говорит 0.8.8 / 0.9.24 — рассинхрон).
Окружение: RAM 1.9 GB, pyarrow, Python 3.12.

## 0. Тесты
`uv run pytest tests -q -m "not slow"` -> **2261 passed, 4 skipped, 0 failed** (131 s).
Открытых падений нет. Далее — эффективность, потолки RAM и недокументированные риски.
Все числа измерены на этой машине; «оценка» помечено отдельно.

## 1. Дефекты (не падающие тесты, но реальные)

| # | Место | Суть | Серьёзность |
|---|---|---|---|
| B1 | ops/sort.py:374-389 `_RunStream.refill` | Квадратичный рост буфера: while-цикл конкатенирует по одному куску `pa.concat_tables([buffer, chunk])` — каждая итерация копирует весь буфер. | Высокая |
| B2 | dataframe.py:1914-1929 `encode_categorical` | Pass1 копит Python-set всех уникальных значений по колонке поверх всех батчей. Пик O(K) Python-объектов, MemoryGuard не видит. | Высокая при high-cardinality |
| B3 | ops/metrics.py:639-641 ROC-AUC pass2 | Построчный Python-цикл zip(probas, ys) по всему потоку. O(batch) RAM, но CPU линейно-питоновский. | Средняя (CPU) |
| B4 | core/memory_guard.py:696-700 | Фолбэк `len(set(col.to_pylist()))` без extra_factor — Python-объекты не учтены. Путь редкий, молчаливый. | Низкая |
| B5 | ops/sort.py:314-322 `_append_to_builders` | Живой per-row as_py()+builder путь; в merge не вызывается (Arrow-native), но экспортирован — риск возврата. | Низкая (мёртвый код) |
| B6 | README / __init__ | README: v0.8.8 / разделы 0.9.24; `__version__="0.9.14"`. Рассинхрон в трёх местах. | Низкая (доки) |
| B7 | ops/shuffle.py:150-226 | `_string_hash_column`: dictionary_encode + Python-цикл по словарю (O(K) на батч). | Низкая/Средняя |
| B8 | dataframe.py:2598,4450,4571 | `src_vals = b.column(c).to_pylist()` без `check_batch(extra_factor)` (см. §2.3). | Средняя (RAM) |

## 2. RAM-ограничения — с числами

### 2.1 B1 — квадратичный merge-буфер (главный RAM-риск)

Измерено (2000 кусков по 1 строке): повторный concat = **0.078 s**, один `concat_tables(list)` = **0.0003 s** -> ~250x и растёт суперлинейно.

Сколько копирований при реальных rbs (row_bytes=640, батч чтения 1000 строк):

| rbs (строк) | concat-итераций M | скопировано строк | скопировано объёма |
|---|---|---|---|
| 50 000 | 50 | 1.28e6 | **0.8 GB** |
| 100 000 | 100 | 5.05e6 | **3.2 GB** |
| 200 000 | 200 | 2.01e7 | **12.9 GB** |
| 1 000 000 | 1000 | 5.01e8 | **320 GB** (оценка) |

Пик RSS = ~2x rbs x row_bytes (старый+новый буфер живут на время concat), формально в норме. Но суммарный скопированный объём до 320 GB -> давление на page-cache на слабой машине -> своп/OOM-killer при неудачном раскладе. Это нарушение духа ограничения RAM (I/O и time), а не пикового RSS.

Проверенное решение:
```python
def refill(self, memory_guard):
    parts = [self.buffer] if len(self.buffer) else []
    have = len(self.buffer)
    while have < self.rbs:
        r = self._next_batch(memory_guard)
        if r is None:
            self.exhausted = True; break
        b, pos = r
        take = min(self.rbs - have, b.num_rows - pos)
        if take <= 0: continue
        parts.append(pa.Table.from_batches([b.slice(pos, take)]))
        have += take
        self.cur_pos += take
        if self.cur_pos >= b.num_rows:
            self.cur_batch = None; self.cur_pos = 0
    self.buffer = pa.concat_tables(parts) if len(parts) > 1 else (parts[0] if parts else self.buffer)
```
Замер: concat_tables(list) на 1000 кусков = 0.0001 s (линейно). Экономия на rbs=200k: 12.9 GB -> 0.13 GB скопированного.

### 2.2 B2 — encode_categorical держит Python-set всех уникальных

Копит `uniques: set` поверх всех батчей. Python-str в set ~50-60 B + слот, Arrow utf8 ~13 B/строка -> ~4.7x (измерено для to_pylist). На 10M уникальных строк: Arrow ~130 MB -> Python-set **600+ MB**, MemoryGuard не видит (считает batch.nbytes ~130 KB на батч по 10k).

Решение: K_est по сэмплу (`core/_k_est.py:estimate_k_sample_arrow` уже есть) + `check_state(K_est * (value_bytes + 50))` до Pass1; либо Arrow dictionary IPC + join (Pass1 O(1) RAM).

### 2.3 Недоучёт to_pylist() — измеренные коэффициенты (1M строк)

| Тип | Arrow nbytes | ΔRSS | factor |
|---|---|---|---|
| int64 | 8.0 MB | 30.3 MB | **3.78x** |
| float64 | 8.0 MB | 29.4 MB | **3.67x** |
| utf8 (8 симв.) | 13.0 MB | 61.6 MB | **4.74x** |

Контракт check_batch(extra_factor=…) есть (memory_guard.py:292-322), но **18 call-site его не передают**. Actionable таблица (file:line, что материализуется, рекомендуемый factor):

| file:line | материализуется | factor |
|---|---|---|
| dataframe.py:2598 | src_vals (shift) | 4.0/3.8 |
| dataframe.py:4450 | src_vals (shift on) | 4.0 |
| dataframe.py:4571 | src_vals (diff on) | 4.0 |
| dataframe.py:2193 | col_vals (rolling/diff) | 4.0 |
| dataframe.py:3060 | for rec in b.to_pylist() | **8-12** |
| dataframe.py:3321/3549/3891 | slice(start, group_len) | 4.0 |
| dataframe.py:1922 | values.to_pylist() (enc_cat) | 4.0 |
| ops/shuffle.py:161/177/223 | arr.to_pylist() fallback | 4.0 |
| ops/rank.py:197-198/304-305 | vals/row_ids/firsts/counts | 3.8 |
| ops/metrics.py:285-286/340-341 | yt/proba | 4.0 |
| ops/metrics.py:640-641 | ROC-AUC stream (check_batch есть, factor нет) | 3.8 |
| ts/features.py:131-133/268-270 | vals/grps | 4.0 |
| ts/stats.py:255-257/417-420 | vals/xs/ys | 4.0 |
| ts/models.py:66-68 | vals/gvals | 4.0 |
| ts/thf.py:143 / ts/backtest.py:191 | group_col | 4.0 |
| ops/pivot.py:141/208-210 | idx/col/value | 4.0 |
| ops/resample.py:275-276 | ts/col (check_batch есть, factor нет) | 3.8 |
| ops/groupby.py:444-464/526/581 | dictionary/joined | 4.0 |

OK (factor=1.0, K=O(1)): quantile.py:112 (K=#targets), sort.py:527-528 (K<=8 runs), viz/*_adapter.py (агрегат).

### 2.4 Задокументированные потолки, красные для 1.9 GB

| Операция | Когда ломается | TODO |
|---|---|---|
| CovarianceOp | state O(d^2); d=1e4 -> 800 MB | A3 блочный Reduce |
| MaterializedIndex (set_index) | O(N) RAM | A7 RangeIndex fast-path |
| RobustScaler.fit | O(d) external_sorts; d=100,n=1M -> ~17 min | A9 один sort с col_id |
| PCA | cov O(d^2) + eigh O(d^3) | A4 randomized SVD |
| fit_lasso path | переобучение с нуля | A6 warm-start |
| groupby skew | oversize partition -> MemoryLimitExceeded | A2 рекурсия |
| pc.quantile/rank whole column | 1-2x input | A1 t-digest/per-batch |

## 3. Неэффективные алгоритмы (с проверенными заменами)

| # | Алгоритм | Где | Измерено | Замена |
|---|---|---|---|---|
| E1 | Квадратичный concat_tables в refill | ops/sort.py:388 | 2000 кусков: 0.078 s vs 0.0003 s | список + один concat |
| E2 | encode_categorical Python-set поверх батчей | dataframe.py:1914 | 4.7x RAM (utf8) | K_est по сэмплу + check_state; или Arrow dictionary IPC |
| E3 | ROC-AUC pass2 построчный Python | ops/metrics.py:639 | 500k: 0.073 s Python vs 0.168 s Arrow group_by (+72 MB, K=N) | не group_by; streaming O(1) state на границе score; либо оставить (не узкое место) |
| E4 | _string_hash_column Python-цикл по словарю | ops/shuffle.py:177 | O(K) Python на батч | dictionary_encode один раз на партицию; хэш словаря один раз |
| E5 | _exact_actual_matches_arrow пересчёт value_counts на probe-батч | ops/join.py | B x redundant | кэш per partition-pair (уже в DECISIONS §11.6) |
| E6 | RobustScaler.fit - O(d) отдельных external_sort | cadw/scalers.py | ~17 min при d=100 | один внешний sort с колонкой col_id, streaming квантили по (col_id,q) |
| E7 | compute_hashes для строк через to_pylist | ops/shuffle.py:161 | 4x RAM | fast-path dict_encode есть; фолбэк с extra_factor=4 |
| E8 | _append_to_builders per-row as_py() | ops/sort.py:314 | мёртвый код, экспортирован | удалить или пометить _ |
| E9 | estimate_unique_keys multi-col set(to_pylist()) | core/memory_guard.py:700 | на сэмпле, без factor | count_distinct fast-path уже есть; фолбэк с factor=4 |
| E10 | to_numpy строк через np.concatenate(object) | dataframe.py:231-238 | object array + Python-строки | документировать; для строк to_arrow/to_batches |

## 4. Функции и алгоритмы для добавления во фреймворк

Приоритет: A=RAM/корректность, B=производительность, C=новые возможности.

| # | Что | Зачем | Приоритет |
|---|---|---|---|
| A1 | approx_quantile (t-digest / KLL) на Reduce | точные quantile/median требуют Sort O(N log N) I/O; t-digest O(1) state, merge-able | A |
| A2 | Рекурсивный skew-путь groupby/drop_duplicates | сейчас oversize partition -> MemoryLimitExceeded | A |
| A3 | Блочный CovarianceOp (chunked d x d) | cov O(d^2) ломается на d=1e4 | A |
| A4 | PCA через randomized SVD на cov | eigh O(d^3) + cov O(d^2) | A |
| A5 | check_python_objects(n_objs, obj_bytes) в MemoryGuard + контракт extra_factor по типу | Python-объекты невидимы | A |
| B1 | sort_merge refill линейный (§2.1) | 250x time, до 320 GB копий | A |
| A7 | RangeIndex fast-path для set_index + streaming loc | O(N) RAM сейчас | A |
| A8 | nunique на HashShuffle + count | точный nunique без set | B |
| A9 | RobustScaler одним sort с col_id | 17 min -> минуты | B |
| A6 | fit_lasso warm-start / alpha-path | переобучение с нуля | B |
| A10 | str/category колонки - словарь на диск (Arrow dictionary IPC) | снимает O(K) Python set | B |
| B2 | Кэш value_counts per partition-pair в join | B x | B |
| B3 | Векторизованный хэш строк через pc.binary_join + один xxh64 по словарю | O(K) Python -> O(K) C++ | B |
| B4 | Pipeline.plan учитывает пики per-step, а не sum(state) | DECISIONS §9.2.21 | B |
| B5 | KFold(materialize=True) budget = 2x check_state -> 1x (streaming folds) | DECISIONS §9.2.22 | B |
| C1 | rolling/ewm/shift векторно, без to_pylist() per batch | dataframe.py:2193,3321,3549 | B |
| C2 | isin (кандидат v0.2) | паритет с pandas | C |
| C3 | reindex/gap-fill для resample | NaN-fill | C |
| C4 | nulls_first per-column | расширение _Key | C |
| C5 | swap/astype без материализации | паритет | C |
| C6 | describe расширить (skew/kurt/quantiles) | tsfresh-подмножество уже в ts | C |
| C7 | to_batches(format='polars') / 'arrow_c_stream' | интероп | C |
| C8 | fit_lasso alpha raw vs /n унифицировать с sklearn | совместимость | C |

Отдельно: **B5 в DECISIONS §11.6 — test_arrow_call_memory_guard.py whitelist по имени функции, не qualname** -> RobustScaler.fit не ловится. Исправить: обход ClassDef и whitelist по ClassName.method.

## 5. Итоговые выводы по Части 1

1. Падений нет (2261 тест зелёный). Есть 8 дефектов эффективности/RAM, из них 2 высокой серьёзности (B1, B2).
2. Главный RAM-риск — не пиковый RSS, а суммарный скопированный объём в external_sort.refill: до 320 GB копирования при rbs=1M (оценка), до 12.9 GB измерено при 200k. Лечится в 15 строк.
3. MemoryGuard слеп к Python-объектам: 18 call-site без extra_factor, измеренные коэффициенты 3.7-4.7x (до ~12x для to_pylist() всего батча в dict).
4. encode_categorical — единственный реально неограниченный O(K) Python-накопитель среди горячих путей.
5. Предложен список A1-A10, B1-B5, C1-C8 — каждое с обоснованием и ссылкой на существующие примитивы.
