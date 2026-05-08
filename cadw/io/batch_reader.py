# cadw/io/batch_reader.py
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.ipc as ipc
from abc import ABC, abstractmethod
from typing import List, Optional, Iterator

class BatchReader(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[pa.RecordBatch]:
        pass

    @abstractmethod
    def __next__(self) -> pa.RecordBatch:
        pass

    @property
    @abstractmethod
    def schema(self) -> pa.Schema:
        pass

    @property
    @abstractmethod
    def total_rows(self) -> int:
        pass

    @property
    @abstractmethod
    def columns(self) -> Optional[List[str]]:
        """Список запрошенных колонок (может быть None для всех)."""
        pass

class ParquetBatchReader(BatchReader):
    def __init__(self, path: str, columns: Optional[List[str]], batch_size: int):
        self.path = path
        self._columns = columns
        self.batch_size = batch_size
        self._pf = pq.ParquetFile(path)
        # Если columns не заданы, читаем все колонки
        cols = columns if columns is not None else self._pf.schema_arrow.names
        self._iter = self._pf.iter_batches(batch_size=batch_size, columns=cols)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    @property
    def schema(self) -> pa.Schema:
        # Используем schema_arrow, у которой есть метод field
        full = self._pf.schema_arrow
        if self._columns is None:
            return full
        # Возвращаем схему только для запрошенных колонок
        return pa.schema([full.field(c) for c in self._columns if c in full.names])

    @property
    def total_rows(self) -> int:
        return self._pf.metadata.num_rows

    @property
    def columns(self) -> Optional[List[str]]:
        return self._columns

class ArrowIPCBatchReader(BatchReader):
    def __init__(self, path: str, columns: Optional[List[str]], batch_size: Optional[int] = None):
        self.path = path
        self._columns = columns
        self._reader = ipc.open_file(path)
        self._num_batches = self._reader.num_record_batches
        self._current = 0
        # batch_size в IPC игнорируется, так как RecordBatch уже предопределены
        # Если нужна разбивка, придётся дополнительно обрабатывать, но для простоты оставим

    def __iter__(self):
        return self

    def __next__(self):
        if self._current >= self._num_batches:
            raise StopIteration
        batch = self._reader.get_batch(self._current)
        if self._columns is not None:
            # Выбираем только указанные колонки
            cols = [batch.column(c) for c in self._columns]
            batch = pa.RecordBatch.from_arrays(cols, names=self._columns)
        self._current += 1
        return batch

    @property
    def schema(self) -> pa.Schema:
        full = self._reader.schema
        if self._columns is None:
            return full
        return pa.schema([full.field(c) for c in self._columns if c in full.names])

    @property
    def total_rows(self) -> int:
        if self._num_batches == 0:
            return 0
        first = self._reader.get_batch(0)
        return first.num_rows * self._num_batches

    @property
    def columns(self) -> Optional[List[str]]:
        return self._columns