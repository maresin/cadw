import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_reader import ParquetBatchReader
from cadw._legacy.groupby_aggregator import GroupByAggregator

import pytest

pytestmark = pytest.mark.skip(reason="legacy: SpillManager.merge incompatible with Arrow IPC")

def test_groupby_mean():
    # Создаём тестовые данные
    data = {
        'cat': ['A', 'B', 'A', 'B', 'A', 'B'],
        'val': [1, 2, 3, 4, 5, 6]
    }
    table = pa.table(data)
    with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp:
        pq.write_table(table, tmp.name)
        guard = MemoryGuard(10)  # 10 MB
        reader = ParquetBatchReader(tmp.name, ['cat', 'val'], guard.get_read_batch_size())
        agg = GroupByAggregator(reader, ['cat'], ['val'], guard, agg_func='mean')
        result = agg.collect()
        # Ожидаем: A: (1+3+5)/3 = 3.0, B: (2+4+6)/3 = 4.0
        assert result[('A',)]['val'] == 3.0
        assert result[('B',)]['val'] == 4.0
        print("Тест groupby mean пройден!")

def test_groupby_sum():
    data = {
        'cat': ['A', 'B', 'A', 'B', 'A', 'B'],
        'val': [1, 2, 3, 4, 5, 6]
    }
    table = pa.table(data)
    with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp:
        pq.write_table(table, tmp.name)
        guard = MemoryGuard(10)
        reader = ParquetBatchReader(tmp.name, ['cat', 'val'], guard.get_read_batch_size())
        agg = GroupByAggregator(reader, ['cat'], ['val'], guard, agg_func='sum')
        result = agg.collect()
        assert result[('A',)]['val'] == 9   # 1+3+5
        assert result[('B',)]['val'] == 12  # 2+4+6
        print("Тест groupby sum пройден!")

if __name__ == '__main__':
    test_groupby_mean()
    test_groupby_sum()