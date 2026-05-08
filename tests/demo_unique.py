# tests/demo_unique.py
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import tempfile
import os
from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_reader import ParquetBatchReader
from cadw.operations.unique_collector import UniqueCollector

def generate_test_data(parquet_path, num_rows=200_000, num_unique=50_000):
    np.random.seed(42)
    unique_vals = list(range(num_unique))
    data = np.array(unique_vals * (num_rows // num_unique))[:num_rows]
    table = pa.table({'col': data})
    pq.write_table(table, parquet_path)
    print(f"Сгенерирован {parquet_path} с {num_rows} строк, {num_unique} уникальных.")

def main():
    # Создаём тестовый файл
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, 'test.parquet')
        generate_test_data(input_path, num_rows=200_000, num_unique=50_000)

        # Ограничение памяти 10 MB – вынуждаем спилл
        guard = MemoryGuard(10)
        reader = ParquetBatchReader(input_path, ['col'], guard.get_read_batch_size())
        collector = UniqueCollector(reader, 'col', guard)
        uniq = collector.collect()

        print(f"Найдено уникальных значений: {len(uniq)}")
        assert len(uniq) == 50_000
        print("Тест пройден!")

if __name__ == '__main__':
    main()