# CADW

Out-of-core фреймворк для табличных данных, не помещающихся в RAM, на слабых машинах.


## Текущий статус (v0.8.8)

**7 слоёв:** Storage (Parquet/IPC) → Dataset → Access (RowFilter,
MemoryGuard, Planner) → Primitives (Reduce, HashShuffle, Sort) →
Operations (groupby, join, dedup, rank, quantile, resample) →
DataFrame (CadwDataFrame, iloc/loc/at/iat) → Export (to_batches, ML, viz).

**Time-series:** `shift/diff/rolling` (positional + sort-aware single/
multi-key `on`), `expanding`, `ewm`, `resample(freq)`.

**ML (собственные реализации на primitives):** Linear/Ridge/Lasso/ENet,
Logistic, k-means, PCA, GBDT (regression / binary / multiclass),
Gaussian/Multinomial/Bernoulli NB.

**Viz (aggregate-first):** hist, hist2d, box, line, sample, hexbin;
adapters для matplotlib, plotly, seaborn.

**Индексация:** `CadwIndex` (RangeIndex / MaterializedIndex /
MaterializedMultiIndex), `iloc` / `loc` / `at` / `iat`, filter-compat.

**Тесты:** ~1096 fast + 8 slow (E2E + OOM). См. §11.1 (DECISIONS.md).

**Известные limitations** — DECISIONS.md §11.5 (18 items, documented).
**Backlog** — §11.6 (refactor/perf). Нет открытых багов (§13.4 closed).

**Snapshot:** `uv run python scripts/collect_project.py` (~500 KB, smart).

## Принципы

- Не более одного батча данных в RAM; остальное — на диск (Arrow IPC).
- Apache Arrow — единый формат в памяти и для кэша.
- Три примитива: `Reduce`, `HashShuffle`, `Sort`. Всё тяжёлое — их композиция.

## Статус

Реализованы и покрыты тестами: `MemoryGuard`, `Reduce`, `HashShuffle`,
`Sort`, `groupby_aggregate`.

`groupby` работает при числе групп, превышающем память
(`tests/test_groupby_v2.py::test_groupby_many_groups`).

**Гарантия out-of-core условна** — см. `DECISIONS.md` §11.4 (5 must-fix).

Реализовано: `CadwDataFrame`, `Dataset`, `RowFilter`, ленивые `__getitem__`, `unique`, `sort_values`, `groupby`, `to_batches(format=...)`, E2E на 200k строк / 20 MB budget. ML — вне scope CADW. См. `DECISIONS.md` §11.1.

## Установка

```bash
uv sync
```

## Пример

```python
from cadw import groupby_aggregate, MemoryGuard
from cadw.io.batch_reader import ParquetBatchReader

reader = ParquetBatchReader("data.parquet", ["cat", "val"], batch_size=10000)
guard = MemoryGuard(max_memory_mb=200)
result = groupby_aggregate(reader, ["cat"], ["val"], ["mean"], guard, "./spill")
print(result)
```

## ML и внешние библиотеки

CADW **не является ML-фреймворком**. Он даёт `to_batches(format=...)` для
экспорта батчей в вашу модель. Если ваша модель OOM-ит — это её состояние,
а не CADW. Гарантия out-of-core действует только на код CADW, не на
сторонние библиотеки.

```python
for batch in df.to_batches(format="numpy", batch_size=10_000):
    # batch: dict[str, np.ndarray]
    process_batch(batch)
```

**Не поддерживается:**
- адаптеры к `lgb.train`, `xgb.fit`, `PCA.fit`, `HistGradientBoosting.fit`
  и любым `fit`-методам без `partial_fit` — они OOM-ят внутри библиотеки,
  `MemoryGuard` не спасает;
- `partial_fit`-адаптеры — whitelist невозможен без доверия к модели.

**Планируется** (v0.3+): собственные алгоритмы на примитивах:
GBDT (level-wise, аналог `HistGradientBoosting`), Linear, k-means, PCA,
Naive Bayes. См. `DECISIONS.md` §12.5.

