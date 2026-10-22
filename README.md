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

## Time series (v0.9.24)

Comprehensive time-series toolkit built on the same out-of-core
primitives. Import from `cadw.ts` or use the thin `df.ts.*` accessor.

**Feature engineering** (`cadw.ts.features`): lag/lead, diff (regular +
seasonal), calendar (year/month/day/weekday/hour/...), cyclical sin/cos,
Fourier terms, holiday/weekend flags, and a `make_features` pipeline.
All streaming, O(batch) state, MemoryGuard-checked, group-aware (per-entity
lags, no cross-entity leakage).

**Window statistics** (`cadw.ts.stats`): tsfresh-subset — mean/std/sum/
min/max/median/skew/kurt, fixed quantiles q10..q90, abs_energy/energy/cid,
crossings, slope, entropy, count_above_mean, autocorr_lagK, plus
`rolling_crosscorr` between two series. One shared deque per group;
explicit RAM formula per statistic.

**Forecasters** (`cadw.ts.models`): `NaiveForecaster` (last/seasonal/
drift), `SARIMAForecaster` (CSS/Hannan-Rissanen, bounded auto-AIC grid,
seasonal orders), `GBDTForecaster` (default mode='direct' with deferred
per-horizon training; 'recursive' and 'multi_output' also available).
Common `ForecasterBase` API: fit / predict / predict_in_sample.

**Backtesting** (`cadw.ts.backtest`): expanding and sliding-window CV,
multi-series, metrics MAE/RMSE/MAPE/sMAPE/MASE (m=seasonal_period),
optional pinball for quantile forecasters. Forecasters cloned per fold.

**Temporal Hierarchical Forecasting** (`cadw.ts.thf`):
`TemporalHierarchy` aggregates the base series to K temporal levels, fits
a forecaster per level, and reconciles with bottom-up, top-down-proportional,
optimal-OLS, or proper MinT-WLS (diagonal residual-variance weights,
1e-8 floor, OLS fallback on insufficient residuals). Multi-series supported.

### Examples

```python
import cadw
from cadw.ts import (
    lag_features, rolling_stats, make_features,
    NaiveForecaster, GBDTForecaster, backtest_forecaster,
    TemporalHierarchy,
)

# df has columns y, ts (timestamp), maybe g (group)
# Features
f = lag_features(df, "y", lags=(1, 7, 14), group_col="g", time_col="ts")
# OR full pipeline (do not chain both):
# f2 = make_features(df, target_col="y", ts_col="ts", group_col="g")

# Window stats
s = rolling_stats(df, "y", windows=(7, 14),
                  stats=("mean", "std", "slope", "autocorr_lag1"))

# Forecasters
m = NaiveForecaster(strategy="seasonal", period=7).fit(df, "y")
print(m.predict(7))

g = GBDTForecaster(lags=(1, 2, 3)).fit(df, "y")
print(g.predict(7))    # deferred: 7 horizon models trained on demand

# Backtesting
summary, folds = backtest_forecaster(
    df, NaiveForecaster("last"),
    target_col="y", time_col="ts",
    horizon=7, n_folds=5, strategy="expanding",
    metrics=("mae", "rmse", "mase"),
)

# Temporal hierarchy with MinT-WLS reconciliation
th = TemporalHierarchy(
    levels=[1, 7, 30],
    forecaster_factory=lambda: NaiveForecaster("last"),
    reconciliation="optimal_wls",
).fit(df, "y", time_col="ts")
print(th.predict(30))

# Or via the accessor:
f = df.ts.lag("y", lags=(1, 7), group="g", time_col="ts")
f = f.ts.rolling_features("y", windows=(7,), stats=("mean", "std"))
model = df.ts.fit_forecaster("gbdt", "y", time_col="ts")
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

