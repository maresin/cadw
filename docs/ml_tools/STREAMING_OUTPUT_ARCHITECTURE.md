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

### §1.5. Producer API — input side (v0.9.113)

**Rule:** every public `fit_*` / `predict_*` / `transform_*` in
`cadw/ml/`, `cadw/ts/` takes a **`CadwDataFrame` first positional arg**
and iterates via `df._iter_filtered_batches(batch_size=...)`.

**Private helpers** (`_foo_batch(arr, ...)`, `_fit_X_series(arr, ...)`) may
take `np.ndarray` — that's the internal building block. Public producer
is the df-based wrapper.

**Legacy producers** (`fit_zms`, `fit_cusum`, `fit_kshape`, `fit_eif`,
`fit_kmeans_ad`, `fit_bocpd`, plus `fit_hampel`, `fit_sub_pca`, `fit_s2g`,
`fit_dwt_mlead`, `fit_patch_memory_ad`, `fit_autoencoder`) currently take
`series: np.ndarray` — a v0.9.40 artefact that undermines OOC (input already
O(n) in RAM). Migration tracked in ROADMAP §Legacy producer inventory.

**Signal rules (machine-checkable):**

| Signature | Verdict |
|---|---|
| `fit_*(df, col, ...)` | canonical |
| `fit_*(series: np.ndarray, ...)` | legacy (migration pending) |
| `predict_*(model, df, *, output="auto")` | canonical |
| `predict_*(model, arr: np.ndarray)` | legacy |

Dual dispatch (`df_or_series` with type sniff) is **transitional**: it
preserves backward compatibility while the caller migrates to df path.
v0.9.113 `fit_zms` is the pilot.


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

**P11. `sgd_streaming` is for linear heads only.**

`sgd_streaming` (`cadw/ops/optim.py`) fits a single `W: (K, d)` on
`(X_b, Y_b)` pairs with ridge loss + RMSprop. It is the correct
primitive for **linear** models that OOC their features to disk:
ROCKET (current use), and future logistic / coxph / linear-ridge
when they need wide-feature OOC (currently closed-form XTX, which
is O(d²) — fine for d ≤ 10⁴, wrong for wider).

It is NOT for:
- MLP (autoencoder, patch_memory_ad): 4+ weight matrices, ReLU
  backprop, momentum. Need dedicated chunked training loop
  (S6d pattern: `rng.permutation(n_w)` as int64 in RAM,
  on-demand window fetch per mini-batch).
- Clustering (kmeans_ad): EM (assign + accumulate + update),
  not gradient descent.
- Non-parametric (iforest, eif, matrix_profile): no gradient.

For non-linear models the reusable parts are the batch_stream
contract (`callable() -> iterator of (X_b, Y_b)`) and
`mg.check_state` discipline — NOT the function itself.


## §6. Roadmap спринтов — status (v0.9.91)

**Closed in v0.9.90 (streaming-output API):**
- S0 — этот документ
- S1 — `SpillBuffer` / `AutoResult` / `SpillArray` / `AutoArrayResult`
- S2 — 4 ops → `output=`
- S2-fix — `spill_ratio=0.0`, `output=` validation
- S3 — 11 G2 (guard-after-alloc)
- S4 — 9 ML predict/transform/score → `output=`
- S5 — TS root `_extract_series` + `var` / `_build_lag_matrix` /
  `_MultiplicativeWrapper`
- S5b — TS base path (thf / features / stats)

**Closed in v0.9.91 (S6: refuse-at-guard → chunked / documented):**
- S6a — `hampel` + `zms`: strided+chunked. **Bit-identical.**
- S6b-eif — `eif`: chunked fit (`n_est*psi*d` hyperparameter-bound).
  **NOT bit-identical** (rng state divergence from per-tree
  `rng.choice`; see S6b-full below).
- S6b-tspq — `tspq`: chunked mean/cov/scores. **Bit-identical.**
- S6c-kmeans_ad — `kmeans_ad`: two paths (duck-typed guard → old
  `_prepare_windows` bit-identical; real `MemoryGuard` → chunked
  reservoir + Lloyd + score). NOT bit-identical (reservoir init
  vs exact `rng.choice(n_w, size=k)`; see S6c-full below).
- S6d-autoencoder — `autoencoder`: chunked SGD via
  `rng.permutation(n_w)` as `(n_w,)` int64 (8·n_w bytes = 50× smaller
  than `(n_w, d)` float64). **Bit-identical.**
- S6e — `binseg` + `stray`: `refuse_at_guard` (documented).
  O(n) prefix sums (binseg) and O(n²) brute KNN (stray) are intrinsic.

### §6.1 Won't-fix (reclassified, with rationale)

**S6b-full (eif bit-identical).** One-pass on-demand fetch
(`n_est × O(n·w)` I/O) reproduces the pre-S6b rng sequence.
Cost: `n_est × n_w × d × 8` extra reads per fit. For
`n_est=100, n_w=10⁶, d=50`: 40 GB I/O, several minutes.
Benefit: zero (functional result already meets recall threshold,
test_fit_eif_detects_segments >= 0.5 passes). Rejected as
cost >> benefit.

**S6c-full (kmeans_ad bit-identical).** k-means++ init requires
`k-1` chunked passes over `n_w` (each pass computes D² distance
to current centroid set, then `rng.choice(n_w, p=probs)`). For
`k=8, n_w=10⁶, d=50`: 3.2 GB extra I/O, ~100 seconds per fit.
Current chunked reservoir init is **deterministic** (same
`random_state` → same centroids) and tests pass at recall
threshold. Rejected as cost >> benefit.

**S6d-full (autoencoder bit-identical beyond `n_w = state_limit/16`).**
Requires disk-permutation (`SpillDir` for `rng.permutation(n_w)`
as Arrow IPC) + disk-batch-reader. 2-3 sessions, no urgent use
case (`n_w = state_limit/16` is typically ≥ 10⁶ at 100 MB budget).
Deferred to v0.9.92+ if a user reports needing it.

### §6.2 Backlog (v0.9.92+)

**Default `output="auto"` migration.** Currently `output="memory"`
is default everywhere (back-compat). After all producers migrated
(this version), flip default to `"auto"`. BREAKING (return type
changes from `pa.Table`/`np.ndarray` to `AutoResult`/`AutoArrayResult`).
Requires:
- ~40 tests updated (`isinstance(x, pa.Table)` → `isinstance(x, AutoResult)`)
- CHANGELOG entry with **BREAKING** banner
- full fast+slow test suite run (~15 min)
- independent v0.9.92 session

See also STREAMING_OUTPUT_ARCHITECTURE.md §3 (migration matrix).
