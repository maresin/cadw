# Streaming Output Architecture

**Version:** 1.0 (2026-10-29, v0.9.90-S1)
**Status:** normative
**Companion:** METHODOLOGY_ALGORITHM_SELECTION.md §1.3, §5
**Related:** `cadw/core/spill.py`, PRIMITIVES_AND_PRINCIPLES.md §1

---

## §1. Принцип: spill-on-demand

CADW гарантирует **O(batch) RAM при любом n** — но только если результат
операции попадает в `AutoResult` / `AutoArrayResult`. Функции, возвращающие
`pa.Table` или `np.ndarray` целиком, нарушают §1.1 независимо от того,
насколько хорош их внутренний state.

**Spill-on-demand, не always-disk:** аккумулируй в RAM, спилль при
исчерпании бюджета. Не «всегда диск» (медленно для маленьких данных),
не «всегда RAM» (OOM для больших). Порог = `guard.available_bytes() ×
spill_ratio` (default `spill_ratio = 0.5`, совпадает с `state_safety`).

**Один раз спилил → до конца.** Никакого un-spill. После первого
`write_batch`, превысившего порог, все последующие батчи идут на диск
напрямую (буфер не ре-аккумулируется).

---

## §2. API контракт

### §2.1. SpillBuffer + AutoResult (для `pa.RecordBatch` / `pa.Table`)

```python
with SpillBuffer(guard=mg, schema=schema, name="groupby",
                 spill_ratio=0.5) as buf:
    for batch in stream:
        buf.write_batch(batch)
    result: AutoResult = buf.finalize()

# result — либо InMemoryResult, либо OnDiskResult
for batch in result:                   # re-iterable
    process(batch)
len(result)                            # O(1) для in-memory, O(1) для disk (IPC footer)
result.schema                          # pa.Schema
result.is_spilled                      # bool
result.to_table()                      # in-memory только; для spilled — raise
```

### §2.2. SpillArray + AutoArrayResult (для `np.ndarray`)

```python
with SpillArray(guard=mg, width=K, dtype=np.float64,
                name="predict_proba") as buf:
    for batch in df._iter_filtered_batches():
        buf.write_array(compute_batch(batch))   # (b, K) numpy
    result: AutoArrayResult = buf.finalize()

for arr in result:                     # yield (b, K) arrays
    process(arr)
result.to_numpy()                      # in-memory только; для spilled — raise
len(result)                            # total rows
```

### §2.3. Escape hatch: `to_table()` / `to_numpy()` **бросают на spilled**

Это не баг и не «неудобство». `MemoryLimitExceeded` с сообщением,
указывающим на §1.3. Пользователь должен использовать `for batch in
result`. Если он хочет «всё-таки получить numpy» — `np.concatenate(
list(result))`, осознанно беря риск OOM в свои руки.

### §2.4. Параметр `output=` в функциях-продуцентах

```python
def groupby_aggregate(..., *, output="memory") -> "pa.Table | AutoResult":
```

| `output` | Поведение | Use case |
|---|---|---|
| `"memory"` (**default в S2**, back-compat) | Старое: `pa.Table`/`np.ndarray`, raise при OOM | существующий код |
| `"auto"` (**default в S4+**) | `AutoResult`/`AutoArrayResult` — in-memory если влезло, disk если нет | новый код |
| `"disk"` | Всегда на диск | явный контроль |

Default `"memory"` в S2 минимально инвазивен. Переключение default'а на
`"auto"` — отдельный breaking-коммит после того, как все продуценты
мигрированы (v0.9.91+).

---

## §3. Матрица миграции

Функции, чей результат **O(n) или O(K)** — обязаны вернуть `AutoResult`
или `AutoArrayResult`. Функции с O(1) state (reduce-метрики) — не
трогаем: они уже корректны.

### S2 — ops-слой (текущий Sprint)

| Функция | Сейчас | После | Спринт |
|---|---|---|---|
| `groupby_aggregate` | `pa.Table` | `AutoResult` (output=) | S2 |
| `drop_duplicates` | `pa.Table` | `AutoResult` (output=) | S2 |
| `pivot_to_disk` | `(schema, Path)` | `AutoResult` (output=) | S2 |
| `topk_to_disk` | `pa.Table` | `AutoResult` (output=) | S2 |

### S3 — G2 (guard-after-alloc) в ML `_collect`

Не возврат, а **внутренний** паттерн. `_collect`-функции получают
`SpillBuffer`/`SpillArray` для accumulators + `check_state` **до**
первого `append`. См. §5.

### S4 — ML predict/transform/score (output=)

| Функция | width | Спринт |
|---|---|---|
| `iforest.score_samples` | 1 | S4 |
| `eif.fit_eif` (return scores) | 1 | S4 |
| `chains.predict_proba` | K | S4 |
| `ordinal.predict_proba` | K | S4 |
| `tfidf.transform` (sparse) | V | S4 |
| `randsvd.transform` | k | S4 |
| `linear.predict` (df path) | 1 | S4 |
| `logistic.predict_proba` (df) | 1 | S4 |
| `gbdt.predict_proba` (df) | 1 | S4 |
| `kmeans.predict` (df) | 1 | S4 |

### S5 — TS layer (chunked `_extract_series`)

`_extract_series` в `cadw/ts/models.py` — корень O(n) в TS.
Заменить Python-list на `np.empty` + прямое заполнение (peak n×8 B, не
n×36 B). Плюс `guard.check_state(n × 8)` до аллокации.

`ts/var.py::fit`, `ts/models.py::GBDTForecaster._build_lag_matrix`,
`ts/auto_theta.py::_MultiplicativeWrapper.fit` — добавить guard.

### S6 — refuse-at-guard (8 модулей)

`autoencoder`, `binseg`, `hampel`, `zms`, `stray`, `kmeans_ad`, `eif`,
`tspq` — перевести с «raise» на streaming через SpillDir (сложнее —
требуется chunked алгоритм). Или пометить `documented tier1_core_exception`.

---

## §4. Обнаруженные нарушения (аудит 2026-10-29)

Полный список — в 30-пунктном аудите. Краткая сводка по классам:

| Класс | Кол-во | Закрывается |
|---|---|---|
| `_collect` + vstack без SpillDir | 8 | S3 (SpillArray для accumulators) |
| G2 (guard-after-alloc, `_collect`-паттерн) | 11 | S3 (перестановка guard) |
| Ops output O(K) в RAM | 4 | S2 (AutoResult) |
| ML predict/transform O(n) output | 8 | S4 (AutoArrayResult) |
| TS unguarded O(n) (`var`, `GBDTForecaster`, `auto_theta`) | 3 | S5 (guard) |
| refuse-at-guard | 8 | S6 (spill или documented) |

`_fallback_unique_count`, `als` u_set/i_set, `gbdt` seen/distinct —
**не в матрице миграции** (специфика случая, см. отдельный спринт
после S6).

---

## §5. Правила для будущих функций

**P1. Результат-правило.** Если функция возвращает O(n) или O(K)
результат — она **обязана** вернуть `AutoResult`/`AutoArrayResult`,
не `pa.Table`/`np.ndarray`. Единственное исключение — функции с
`output="memory"` в back-compat периоде.

**P2. Guard-before-alloc.** Любая аллокация O(n) (или O(K), или
O(n×d)) внутри функции должна быть предварена `guard.check_state` /
`guard.check_batch` в том же function scope. G2.

**P3. Lifecycle.** `AutoResult` / `AutoArrayResult` имеют:
- `.close()` — idempotent;
- `__del__` — вызывает `.close()`;
- `__enter__` / `__exit__` — для `with`.

OnDisk-результат **переносит владение tempdir** из `SpillBuffer`.
SpillBuffer.__exit__ **не удаляет** tmpdir, переданный в OnDiskResult.

**P4. Escape hatch.** `to_table()`/`to_numpy()` на spilled результатах
**бросают `MemoryLimitExceeded`**. Не пытаться «вернуть пустой» или
«спилить обратно в RAM».

**P5. `spill_ratio`.** Параметр `spill_ratio` передаётся в
`SpillBuffer`/`SpillArray`, дефолт `0.5` (= `state_safety`). Не
хардкодить.

**P6. Тесты.** Каждая функция с `AutoResult`-выходом покрыта тремя
тестами:
- small (in-memory result, `to_table()` работает);
- large (spilled result, `to_table()` бросает, `for batch in result` работает);
- empty (нулевой output, `len(result) == 0`).

**P7. `guard=None`** интерпретируется как «без ограничений» (threshold
= 2^62). Для тестов и малых данных. Production всегда передаёт
MemoryGuard.

**P8. Не пытаться переизобрести.** Если нужен новый продуцент
(новая `groupby`-подобная операция) — используй `SpillBuffer`, не
пиши свой аккумулятор. Единственный writer — `ArrowIPCBatchWriter`
через `SpillBuffer._spill()`.

---

## §6. Roadmap спринтов

| Sprint | Документ | LOC (est) | Session | Блокирует |
|---|---|---|---|---|
| **S0** | этот документ | ~350 | **v0.9.90** | — |
| **S1** | `SpillBuffer`/`SpillArray` + тесты | ~450 | **v0.9.90** | S2–S6 |
| S2 | 4 ops → `output=` | ~200 | v0.9.91 | — |
| S3 | 11 G2 + `_collect`-refactor | ~150 | v0.9.91 | — |
| S4 | ML predict/transform → `AutoArrayResult` | ~300 | v0.9.92 | S2 (S2 не блокирует S4, но связно) |
| S5 | TS guards | ~150 | v0.9.92 | — |
| S6 | refuse → spill | ~300 | v0.9.93+ | S1 |

**Текущее состояние:** S0+S1 в v0.9.90. S2–S6 разбиты на сессии.
Каждый спринт — свой набор файлов (см. сводку в CHANGELOG после S1).