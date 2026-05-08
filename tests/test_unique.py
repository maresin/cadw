import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_reader import ParquetBatchReader
from cadw.operations.unique_collector import UniqueCollector

def test_unique_with_spill():
    # Создаём данные: 200 000 строк, 50 000 уникальных
    np.random.seed(42)
    unique_vals = list(range(50000))
    data = np.array(unique_vals * 4)[:200000]
    table = pa.table({'col': data})
    
    with tempfile.NamedTemporaryFile(suffix='.parquet') as f:
        pq.write_table(table, f.name)
        guard = MemoryGuard(10)  # 10 MB – заставит сбросить на диск
        reader = ParquetBatchReader(f.name, ['col'], guard.get_read_batch_size())
        collector = UniqueCollector(reader, 'col', guard)
        uniq = collector.collect()
        assert len(uniq) == 50000
        print("Тест unique пройден!")

if __name__ == '__main__':
    test_unique_with_spill()