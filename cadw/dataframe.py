import pyarrow as pa
from typing import List, Any, Dict, Union
from cadw.io.batch_reader import ParquetBatchReader, ArrowIPCBatchReader
from cadw.io.batch_writer import ParquetBatchWriter, ArrowIPCBatchWriter
from cadw.core.memory_guard import MemoryGuard
from cadw.operations.unique_collector import UniqueCollector
from cadw.operations.groupby_aggregator import GroupByAggregator

class CadwDataFrame:
    def __init__(self, source: str, format: str = 'parquet', memory_limit_mb: int = 100):
        self.memory_guard = MemoryGuard(memory_limit_mb)
        if format == 'parquet':
            self.reader = ParquetBatchReader(source, [], self.memory_guard.get_read_batch_size())
            # Временно читаем все колонки, но для начала нужно определить схему
            # Пока просто сохраним путь и формат; при реальных операциях будем создавать reader с нужными колонками
            self._source = source
            self._format = format
        else:
            raise NotImplementedError("Only Parquet source supported initially")

    def _get_reader(self, columns: List[str]) -> ParquetBatchReader:
        return ParquetBatchReader(self._source, columns, self.memory_guard.get_read_batch_size())

    def unique(self, column: str):
        reader = self._get_reader([column])
        collector = UniqueCollector(reader, column, self.memory_guard)
        return collector.collect()

    def groupby(self, by: Union[str, List[str]], agg_columns: List[str], agg_func: str = 'mean'):
        if isinstance(by, str):
            by = [by]
        # Читаем все необходимые колонки: by + agg_columns
        cols = by + agg_columns
        reader = self._get_reader(cols)
        aggregator = GroupByAggregator(reader, by, agg_columns, self.memory_guard, agg_func)
        return aggregator.collect()