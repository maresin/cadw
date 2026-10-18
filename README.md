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

Отсутствует: `CadwDataFrame`, `Dataset`, `RowFilter`, `unique`/`join` через
новые примитивы (см. `DECISIONS.md` §11.3, §12).

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

## Документация

- `DECISIONS.md` — архитектура, алгоритмы, статус, must-fix, уроки.

## Legacy

`cadw/_legacy/`, `tests/_legacy/` — старые реализации (`SpillManager`,
`UniqueCollector`, `GroupByAggregator`). Не импортируются по умолчанию,
исключены из `pytest`. См. `DECISIONS.md` §11.2.
