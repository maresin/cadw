# CADW Session Brief — v0.9.92: Finish Streaming-Output Migration

**Версия:** 1.0 (2026-10-29, после v0.9.91.1)
**Цель:** закрыть streaming-output миграцию (flip default `output="auto"`),
BREAKING change, тег `v0.9.92`. Одна сессия, ~2–2.5 часа.
**Не делать:** новый функционал, bit-identical refactors (см. §6).

---

## §1. Что читать в начале сессии (обязательно, порядок важен)

1. **`docs/ml_tools/STREAMING_OUTPUT_ARCHITECTURE.md`** — нормативный
   документ. Вся методология OOC (§1–§6) здесь. §5 P1–P11 — правила.
   §6.1 — что уже решили **не** делать и почему.

2. **`docs/ml_tools/METHODOLOGY_ALGORITHM_SELECTION.md`** — базис §1.1
   (base cases, not exceptions), §3 (state taxonomy: state_class,
   state_type, bounded_state, n_scalability), §5 (SpillArray),
   §7 (YAML mandatory fields), §8 (G1–G4 AST gates).

3. **`docs/ml_tools/PRIMITIVES_AND_PRINCIPLES.md`** — cheat sheet.
   §1 (6 postulates), §2 (Reduce/HashShuffle/Sort/Batch I/O API),
   §3 (MemoryGuard API), §5 (3 spill patterns), §6 (anti-patterns).

4. **`CHANGELOG.md`** — прочитать записи v0.9.90, v0.9.91, v0.9.91.1
   (в конце). Иначе не понять, что уже сделано.

5. **`.pingis/plan_v0.9.87_refactor.md`** — старый whitelist, сейчас
   полностью закрыт. Читать для контекста «как было», не для работы.

---

## §2. Код для образца (reference patterns)

Читать **перед** написанием нового кода в v0.9.92. Каждый файл —
образец для конкретной задачи:

| Задача | Файл | Функция / класс |
|---|---|---|
| **SpillBuffer API** | `cadw/core/spill.py` | `SpillBuffer`, `AutoResult`, `InMemoryResult`, `OnDiskResult` |
| **SpillArray API (numpy)** | `cadw/core/spill.py` | `SpillArray`, `AutoArrayResult` |
| **`output=` в ops** | `cadw/ops/groupby.py` | `groupby_aggregate` (три ветки: memory / auto / disk) |
| **`output=` в ops (topk)** | `cadw/ops/topk.py` | `topk_to_disk` (incremental merge + `output=`) |
| **`output=` в ML (numpy)** | `cadw/ml/iforest.py` | `score_samples` (SpillArray) |
| **`output=` в ML (pa.Table)** | `cadw/ml/gmm.py` | `score_samples`, `predict` (SpillBuffer) |
| **Two-path (duck vs real guard)** | `cadw/ml/kmeans_ad.py` | `fit_kmeans_ad` (`has_check_state` branch) |
| **Chunked strided (sliding window)** | `cadw/ml/hampel.py` | `fit_hampel` (chunk-loop для median/MAD) |
| **Chunked multi-pass (mean → cov → score)** | `cadw/ml/tspq.py` | `fit_t2q` (3 passes) |
| **Chunked SGD (bit-identical)** | `cadw/ml/autoencoder.py` | `fit_autoencoder` (`rng.permutation(n_w)` int64, on-demand fetch) |
| **Reservoir + chunked init** | `cadw/ml/eif.py` | `fit_eif` (pre-sample indices, gather unique, manual build) |
| **TS np.empty pattern** | `cadw/ts/models.py` | `_extract_series` (no-filter path) |
| **`sgd_streaming` (linear only)** | `cadw/ops/optim.py` | `sgd_streaming` — **НЕ для MLP, НЕ для EM** |

Все эти файлы — свежие, чистые образцы после S0–S6 миграции.
Копируй структуру один-в-один, не изобретай.

---

## §3. Единый подход к «всё стало OOC» (методология, distilled)

### §3.1. Три уровня гарантии (по приоритету)

Любая функция, обрабатывающая данные, должна находиться в одном из
трёх состояний:

1. **Per-batch streaming (лучший).** Возврат — `CadwDataFrame` через
   `Dataset.from_batches` (функция-генератор по батчам). Пик RAM =
   O(batch). Примеры: `linear.predict`, `logistic.predict_proba`,
   `gbdt.predict_proba`, `kmeans.predict`, `df.write_parquet`,
   `df.groupby().agg()` (facade).
   → **Это уже OOC. Не трогать.**

2. **`output=` (средний).** Возврат — `AutoResult` / `AutoArrayResult`
   (при `output="auto"` default в v0.9.92). Внутри — `SpillBuffer` /
   `SpillArray`: аккумулирует в RAM, спилль при исчерпании
   `guard.available_bytes() × 0.5`. Пик RAM = min(output_size,
   state_limit × 0.5). Примеры: `groupby_aggregate`, `drop_duplicates`,
   `topk_to_disk`, `iforest.score_samples`, `chains.predict_proba`,
   `ordinal.predict`, `tfidf.transform_dense`, `randsvd.transform`,
   `gmm.score_samples`, `gmm.predict`.
   → **Миграция завершена в v0.9.90–v0.9.91. Не трогать.**

3. **`refuse_at_guard` (документированный exception).** Функция
   аллоцирует O(n) или O(n²), intrinsic для алгоритма. Guard
   корректно бросает `MemoryLimitExceeded` вместо OOM. Должно быть
   помечено в YAML `bounded_state: false`, `n_scalability:
   refuse_at_guard`, `tier1_core_exception: <reason>`. Примеры:
   `binseg` (prefix sums), `stray` (brute KNN при n > 20k), TS fits
   (`_extract_series`, `stl_decompose`, `VARModel.fit` — для
   single-series TS, n ≤ 1e6 типично).
   → **Осознанное решение. Не «фиксить» без явного запроса.**

### §3.2. Two-path pattern (обязателен для новых функций)

Если функция может стать OOC, но есть существующий код/тесты на
старое поведение:

```python
has_check_state = (
    guard is not None
    and hasattr(guard, "check_state")
    and callable(guard.check_state)
)

if not has_check_state:
    # Duck-typed guard (max_memory_mb only) или guard=None:
    # Старый путь (bit-identical к pre-миграции).
    # Tight budget → MemoryLimitExceeded.
    ...

else:
    # Real MemoryGuard (has check_state):
    # Chunked / SpillBuffer / SpillArray путь.
    # MemoryLimitExceeded только при невозможности (state > state_limit).
    ...
```

**Зачем:** duck-typed guard'ы используются в старых тестах
(`class _FakeMB: max_memory_mb = 0.001`). Two-path сохраняет их
зелёными + даёт OOC для реальных пользователей.

### §3.3. `output=` параметр (для ops и ML)

**Сигнатура:**

```python
def my_function(..., *, output: str = "memory" | "auto" | "disk"):
    if output not in ("memory", "auto", "disk"):
        raise ValueError(
            f"my_function: output must be 'memory', 'auto' or 'disk'; "
            f"got {output!r}")
    ...
```

**Три ветки:**

- `"memory"` — старое поведение: `pa.Table` / `np.ndarray` целиком.
  Back-compat в v0.9.90–v0.9.91. **Default до v0.9.92.**
- `"auto"` — `SpillBuffer(guard, schema, spill_ratio=0.5)` /
  `SpillArray(guard, width, dtype, spill_ratio=0.5)`. Спилль при
  исчерпании `guard.available_bytes() × 0.5`. Возврат —
  `AutoResult` / `AutoArrayResult`.
- `"disk"` — форсированный спилль: `spill_ratio=0.0` → threshold 0,
  первый непустой батч уходит на диск. Возврат — `OnDiskResult` /
  `OnDiskArrayResult`. `.to_table()` / `.to_numpy()` бросает
  `MemoryLimitExceeded` (§1.3).

**Default в v0.9.92 — `"auto"` во всех 4 ops и 9 ML функциях**
(это и есть задача сессии).

### §3.4. Facade responsibility — всегда unwrap к CadwDataFrame

Facade (`df.groupby().agg()`, `df.drop_duplicates()`,
`df.nlargest()`, `df.pivot_table()`) **всегда возвращает
`CadwDataFrame`**, независимо от `output=`:

```python
result = groupby_aggregate(..., output="auto")   # AutoResult
return CadwDataFrame.from_dataset(
    Dataset.from_batches(
        result.schema, iter(result), self._new_dataset_dir()))
```

`iter(AutoResult)` работает для обоих классов (InMemoryResult,
OnDiskResult), итерирует по `pa.RecordBatch`. Это **единственный
правильный способ** — прямой `.to_table()` нарушает §1.3
(no escape hatches).

### §3.5. Что НЕ делать (anti-patterns)

| Anti-pattern | Правильно |
|---|---|
| `np.zeros((n, ...))` без guard | `check_state(n*...*8)` до alloc |
| `Xs.append(...)` в batch-loop, потом `check_state` | `check_state` **до** append (G2) |
| `np.vstack(list_of_batches)` | `SpillBuffer` / `SpillArray` |
| `sliding_window_view(...).copy()` | strided + chunk-loop (`hampel` pattern) |
| `len(set(col.to_pylist()))` | `HashShuffle` + `distinct_count` или `check_state` |
| Возврат `pa.Table` из O(n) операции | `AutoResult` (facade unwrap'ает) |
| Возврат `np.ndarray` из O(n) операции | `AutoArrayResult` (facade unwrap'ает) |
| `to_pandas()` / `to_numpy()` внутри fit | chunk-loop по батчам |
| `rng.choice` **внутри** цикла переопределяется | сохранить rng state последовательность (для bit-identical) |
| `sgd_streaming` для MLP/EM | dedicated loop (`autoencoder` pattern) |

### §3.6. Bit-identical vs not-bit-identical — decision tree

При рефакторинге функции задай вопрос:

```
Был ли старый путь детерминистичен? (нет rng или rng только для init)
│
├─ ДА (нет rng или rng.choice в конце): 
│  Chunked реализация СТАНЕТ bit-identical, если:
│    - использовать ту же математику (sum → cumsum)
│    - сохранить порядок суммирования per chunk
│  → Приложи verification (diff < 1e-12).
│  → Пример: tspq (mean/cov), hampel, autoencoder.
│
└─ НЕТ (rng.choice/per-tree, per-iter, random shuffle):
   Bit-identical требует:
     - либо полный массив в RAM (S6b-full: 40 GB I/O)
     - либо on-demand fetch (n_est × O(n*w) I/O)
   → Если k×I/O за ноль функциональной пользы → WON'T-FIX.
   → Задокументируй в §6.1 STREAMING_OUTPUT_ARCHITECTURE.md.
   → Пример: eif (per-tree rng.choice), kmeans_ad (kmeans++ init).
```

**В v0.9.92 это не касается** — мы не рефакторим алгоритмы,
только flip default. §3.6 — для будущих сессий.

### §3.7. Документирование `refuse_at_guard`

Если функция не может стать OOC без смены алгоритма:

1. YAML card: `bounded_state: false`, `n_scalability: refuse_at_guard`,
   `tier1_core_exception: "<конкретная причина>"`.
2. Docstring: явно указать порог (`n ≤ state_limit / 24`, `n ≤ 20k`,
   `single-series TS, n ≤ 1e6`).
3. Тест: `MemoryLimitExceeded` при n > порог (не OOM).
4. CHANGELOG: `refuse_at_guard (documented)` — как S6e.

**Не пытайся сплить prefix sums** (binseg) или brute KNN (stray) —
это уже другие алгоритмы.

---

## §4. Current state (v0.9.91.1, 2026-10-29)

**Теги:**
- `v0.9.90` — streaming-output API (S0–S5b)
- `v0.9.91` — S6a–e (refuse-at-guard → chunked / documented)
- `v0.9.91.1` — P11 + §6 status consolidation (docs)

**Тесты:** 3981 passed, 8 skipped, 25 deselected (fast suite, `-m "not slow"`).
Baseline до миграции был 3946; +35 от новых тестов.

**Что уже OOC (по §3.1):**
- 4 ops: `groupby_aggregate`, `drop_duplicates`, `topk_to_disk` (с
  `output=`, default `"memory"`); `pivot_to_disk` (уже disk-oriented,
  `output="auto"` внутренне).
- 9 ML: `iforest.score_samples`, `chains.predict_proba`,
  `chains.predict`, `ordinal.predict_proba`, `ordinal.predict`,
  `tfidf.transform_dense`, `randsvd.transform`, `gmm.score_samples`,
  `gmm.predict` (все с `output=`, default `"memory"`).
- 8 chunked: `hampel`, `zms`, `eif` (not bit-identical), `tspq`,
  `kmeans_ad` (two-path), `autoencoder` (bit-identical), `dwt_mlead`
  (pre-guard), `sub_pca`/`s2g`/`kshape`/`patch_memory_ad` (S-серия
  v0.9.87). `binseg`/`stray` — refuse_at_guard.

**Что НЕ OOC (осознанно):**
- TS fits (`_extract_series`, `stl_decompose`, `VARModel.fit`) —
  `tier1_core_exception`, документировано.
- `eif.fit_eif` принимает `np.ndarray` (user contract).
- `tfidf.transform` (sparse) — переменная длина per-row, отдельный
  backlog.

**Незакрытые backlog (см. §6):**
- S6b-full (eif bit-identical): 40 GB I/O, won't-fix.
- S6c-full (kmeans_ad bit-identical): 3.2 GB I/O, won't-fix.
- S6d-full (autoencoder beyond n_w=state_limit/16): 2–3 сессии, defer.
- Wide-feature OOC (logistic/coxph/linear при d > 10⁴): `sgd_streaming`
  есть, миграция не сделана. Отдельная сессия.
- Sparse `tfidf.transform`: нужен `SpillBuffer` с CSR-encoding.

---

## §5. Task v0.9.92 — flip default `output="auto"` (BREAKING)

### §5.1. Scope (все — одна сессия, ~2–2.5 часа)

1. **Flip default** `"memory"` → `"auto"` в 13 местах:
   - `cadw/ops/groupby.py::groupby_aggregate`
   - `cadw/ops/dedup.py::drop_duplicates`
   - `cadw/ops/topk.py::topk_to_disk`
   - `cadw/ml/iforest.py::score_samples`
   - `cadw/ml/chains.py::predict_proba`, `predict`
   - `cadw/ml/ordinal.py::predict_proba`, `predict`
   - `cadw/ml/tfidf.py::transform_dense`
   - `cadw/ml/randsvd.py::transform`
   - `cadw/ml/gmm.py::score_samples`, `predict`

2. **Обновить facade** (`cadw/dataframe/_core.py`), чтобы unwrap'ать
   `AutoResult` / `AutoArrayResult` в `CadwDataFrame`:
   - `_groupby_impl`
   - `drop_duplicates`
   - `nlargest`, `nsmallest`
   - `pivot_table` (внутренне уже auto — проверить)
   - `crosstab` (через pivot_table)
   - `value_counts` (через groupby)

   Facade public API не меняется — всегда `CadwDataFrame`.
   Внутри: `Dataset.from_batches(result.schema, iter(result), ...)`.

3. **Обновить тесты** (ожидают `pa.Table` / `np.ndarray` по умолчанию):
   - `tests/test_groupby_v2.py`, `test_drop_duplicates.py`,
     `test_topk_arrow_native.py`, `test_streaming_output_ops.py`
   - `tests/test_streaming_output_ml.py`
   - `tests/test_oom_resistance.py` (проверить
     `test_groupby_1m_unique_10mb` — `.num_rows` → `len(result)`)
   - любые другие, использующие результат напрямую:
     `isinstance(result, pa.Table)` → `isinstance(result, AutoResult)`
     `result.num_rows` → `len(result)`
     `result.to_pylist()` → `[r for b in result for r in b.to_pylist()]`

   **Не** менять тесты, которые передают явно `output="memory"`
   (там back-compat семантика).

4. **CHANGELOG** — BREAKING entry с явным перечнем:
   - 13 функций: default `output="memory"` → `"auto"`.
   - Facade: без изменений (`CadwDataFrame`).
   - Direct ops callers: получают `AutoResult` (`for b in result`)
     или `AutoArrayResult` (`for a in result`).
   - Escape hatch: `.to_table()` / `.to_numpy()` на spilled бросает
     `MemoryLimitExceeded` (§1.3).
   - Migration recipe: если нужен старый тип — `output="memory"`.

5. **Tag `v0.9.92`** (annotated). Полный `pytest -q -m "not slow"`
   перед тегом.

### §5.2. Acceptance criteria

- [ ] 13 функций: `grep 'output: str = "auto"'` в `cadw/ops/` +
      `cadw/ml/` находит все 13.
- [ ] Facade возвращает `CadwDataFrame` для `df.groupby().agg()`,
      `df.drop_duplicates()`, `df.nlargest()`, `df.nsmallest()`,
      `df.pivot_table()` — checked поведением.
- [ ] `pytest -q -m "not slow"` = **3981+ passed, 8 skipped**.
      Счётчик может вырасти на +N (тесты на `AutoResult` могут быть
      добавлены, чтобы подтвердить default).
- [ ] CHANGELOG entry с баннером **BREAKING** в заголовке.
- [ ] Tag `v0.9.92`.

### §5.3. Estimated time

- Flip defaults (13 строк): **10 мин**
- Facade update: **1 ч** (4 функции × 15 мин)
- Test updates: **30 мин** (~15 assertions)
- CHANGELOG + docs: **15 мин**
- Full fast suite: **10 мин**
- Итого: **~2.5 ч, 1 сессия**

### §5.4. Risk

**Что может сломаться:**
- Facade не unwrap'ает AutoResult → `df.groupby().agg()` возвращает
  `AutoResult` вместо `CadwDataFrame`. Test: `isinstance(result, CadwDataFrame)`.
- `iter(AutoResult)` на закрытом (после `close()`) бросает `RuntimeError`.
  Facade должен вызывать `Dataset.from_batches` внутри, пока result
  открыт.
- `OnDiskResult._owned_tmpdir` удаляется в `__del__` / `close()`. Если
  Facade создаёт `Dataset.from_batches` (eager, но iteration lazy)
  и результат становится мусором — tmpdir может удалиться **до**
  чтения. **Решение:** `Dataset.from_batches` уже делает eager
  consumption при создании? Нет — использует generator. Нужно
  в Facade: **скопировать result в Dataset до выхода из scope
  result** (либо `Dataset.from_batches` будет тянуть из `iter(result)`
  при следующем `to_batches`, а result уже закрыт).

  **Правильный паттерн:**
  ```python
  def _groupby_impl(self, ...):
      result = groupby_aggregate(..., output="auto")
      # Materialize into Dataset eagerly (consumes iter(result))
      return CadwDataFrame(
          Dataset.from_batches(result.schema, iter(result), tmpdir),
          ...)
  ```
  Проверить, что `Dataset.from_batches` consumed eagerly, или передать
  factory: `lambda: iter(result)`. Это может быть единственный
  реальный риск. Посмотреть `cadw/dataset/dataset.py::from_batches`
  (не в бандле — прислать отдельно).

---

## §6. Backlog — НЕ решать в v0.9.92

Если увидишь эти проблемы — **закрыты или решены won't-fix**:

| Проблема | Статус | Где rationale |
|---|---|---|
| eif bit-identical chunked | **Won't-fix** (40 GB I/O) | STREAMING §6.1 S6b-full |
| kmeans_ad bit-identical init | **Won't-fix** (3.2 GB I/O) | STREAMING §6.1 S6c-full |
| autoencoder OOC за `n_w=state_limit/16` | Defer (2-3 сессии, no urgent) | STREAMING §6.1 S6d-full |
| TS O(n) fits (`_extract_series`, `stl_decompose`, `VARModel.fit`) | `tier1_core_exception` | METHODOLOGY §13 |
| `eif.fit_eif` принимает `np.ndarray` | User contract, не нарушение | `eif.py` docstring |
| `tfidf.transform` sparse (не `transform_dense`) | Отдельный backlog (CSR SpillBuffer) | §6.2 |
| Wide-feature OOC (logistic/coxph/linear при d>10⁴) | Отдельная сессия (sgd_streaming есть) | §6.2, P11 |
| Default `output="auto"` в prod | v0.9.92 (эта сессия) | §5 |

**Не изобретай заново:**
- `SpillBuffer` / `AutoResult` / `SpillArray` / `AutoArrayResult` —
  уже есть в `cadw/core/spill.py`.
- `sgd_streaming` — уже есть в `cadw/ops/optim.py` (для linear heads
  только, P11).
- `_resolve_chunk_nw_*` — уже есть в каждом из 8 chunked модулей
  (hampel, zms, eif, tspq, kmeans_ad, autoencoder, dwt_mlead,
  sub_pca/s2g/kshape/patch_memory_ad).
- `Two-path pattern` — образец `cadw/ml/kmeans_ad.py`.

---

## §7. Session discipline (как работать)

### §7.1. Patch workflow

**Не использовать `apply_or_fail` на многострочных блоках** с
декоративными комментариями (`──`, `══`, unicode dash). Почему:
unicode variance между файлами — реальный риск ANCHOR MISS.

**Правильный паттерн:**

```python
# 1. Read file
src = Path('cadw/...').read_text()

# 2. Content-based anchor (unique line from function signature/body)
ANCHOR = '    def method(self, x):\n        return 5'

# 3. Verify anchor exists
if ANCHOR not in src:
    print("ERROR: anchor not found")
    sys.exit(1)

# 4. Replace
src = src.replace(ANCHOR, NEW_BLOCK, 1)
Path('cadw/...').write_text(src)
```

**Или `apply_or_fail` только для одной-двух строк** (`old='foo'`,
`new='bar'`), которые точно уникальны.

### §7.2. Test before commit

После **каждого** patch:

```bash
uv run python -c "from cadw.X import Y; print('OK')"   # импорт
uv run pytest -q tests/test_X.py 2>&1 | tail -3         # модуль
uv run pytest -q -m "not slow" 2>&1 | tail -3           # полный
```

Только потом `git commit`.

### §7.3. Commit message discipline

- Заголовок: `vX.Y.Z-<sprint>: <краткое>`
- Тело: что изменилось, почему, fix iterations если были.
- **CHANGELOG entry — до commit**, не «в commit message сказал
  CHANGELOG». `git diff --stat CHANGELOG.md` перед `git add`.

### §7.4. Урок из v0.9.91 (не повторять)

**§13.4-class**: `git add <file>` на **неизменённый** файл — silent
no-op. В commit-скриптах S6a–S6e был `git add CHANGELOG.md`, но
CHANGELOG edit script **не запускался**. `git add` молча пропустил.
Fix: backfill в v0.9.91.1.

**Правило:** если commit-скрипт добавляет файл — сначала **убедиться**
через `git diff --stat <file>` или `git status`, что файл изменён.
Иначе create иллюзию попадания в коммит.

---

## §8. Appendix: state taxonomy (быстрая справка)

| state_class | state_type | bounded_state | n_scalability | Пример |
|---|---|---|---|---|
| O(batch) | streaming | true | constant | linear.predict, df.write_parquet |
| O(batch) + O(output) | streaming + spill | true | constant | groupby_aggregate (output="auto") |
| O(d²) | aggregation | true | constant | PCA, logistic (closed-form) |
| O(n) | windowed | true | linear | hampel (chunked), sub_pca |
| O(n) | windowed | false | refuse_at_guard | eif (pre-S6b), kmeans_ad (reservoir) |
| O(n²) | cross_row | false | refuse_at_guard | stray (brute KNN) |
| O(n) | — | false | refuse_at_guard | binseg (prefix sums), VARModel.fit |

**Правило**: `bounded_state: false` + `tier1_core` → нужен
`tier1_core_exception: <reason>` в YAML, иначе G4 gate fail.

**При добавлении нового алгоритма** — заполни таблицу в YAML card
(§7 METHODOLOGY) **до** реализации. Если state_class = O(n) и
state_type ≠ streaming, нужен явный способ спилля (chunked / SpillDir)
или documented exception.

---

## §9. Если что-то пошло не так

**Тест падает непонятно** → `uv run pytest -q <test> -vv --tb=long`
(полный traceback), потом `git diff` — что фактически в файле.

**ANCHOR MISS** → прочитать текущее состояние файла (`Path.read_text()`,
первые 50 строк после найденного ключевого слова). Мой anchor устарел.

**Bit-identical расхождение** → пересчитать, что изменилось: rng state?
порядок суммирования? floating-point associativity (mean/cov)?

**Facade возвращает AutoResult вместо CadwDataFrame** → забыл
`Dataset.from_batches(result.schema, iter(result), ...)`.

**Spill не происходит при маленьких данных** → это норма. `auto`
default = in-memory если влезает в `state_limit × 0.5`. Для
форсированного спилля — `output="disk"`.

---

**КОНЕЦ BRIEF**
