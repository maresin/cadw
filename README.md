# CADW

Out-of-core фреймворк для табличных данных, не помещающихся в RAM, на слабых машинах.

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

Реализовано: `CadwDataFrame`, `Dataset`, `RowFilter`, ленивые `__getitem__`, `unique`, `sort_values`, `groupby`, `partial_fit` (sklearn, потоково), E2E на 200k строк / 20 MB budget. См. `DECISIONS.md` §11.1.

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


## ML-интеграция

### sklearn: partial_fit (потоковое обучение)

**Политика ML.** CADW поддерживает только модели с **O(1)-state** относительно числа строк (истинно инкрементальные). Модели, которым нужен весь датасет в RAM (`PCA.fit`, `lgb.train`, любой `fit` без `partial_fit`), **не поддерживаются** — они OOM-ят независимо от `MemoryGuard`. См. `DECISIONS.md` §3.9 и §11.5.

```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA

guard = MemoryGuard(max_memory_mb=100)
df = CadwDataFrame.from_parquet("data.parquet", "cache/", memory_guard=guard)

# Модели без y
scaler = StandardScaler()
df.partial_fit(scaler, X_cols=["x1", "x2"], y_col=None, batch_size=5000)

kmeans = MiniBatchKMeans(n_clusters=5)
df.partial_fit(kmeans, X_cols=["x1", "x2"], y_col=None, batch_size=5000)

ipca = IncrementalPCA(n_components=2)
df.partial_fit(ipca, X_cols=["x1", "x2"], y_col=None, batch_size=5000)

# Модель с y (классификация)
model = SGDClassifier()
for _ in range(3):
    df.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=5000,
                   classes=[0, 1])
```

Батчи конвертируются в numpy на лету, память ограничена `MemoryGuard`.

**Ограничения API sklearn** (не CADW):
- `SGDClassifier.partial_fit` требует `classes=[0, 1]` на первом вызове.
- `IncrementalPCA.partial_fit` требует `n_features=d` на первом вызове.
- `PCA` (не `Incremental`) не имеет `partial_fit`: для batch-only
  используйте `to_batches(format='numpy')` → `np.vstack` → `PCA.fit`
  (это материализует весь датасет, не out-of-core).

### Экспорт батчей

```python
for batch in df.to_batches(format="numpy", batch_size=10000):
    # batch: dict[str, np.ndarray]
    ...
for batch in df.to_batches(format="torch"):
    # batch: dict[str, torch.Tensor]
    ...
```

