# cadw/io/batch_writer.py
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.ipc as ipc
from abc import ABC, abstractmethod

class BatchWriter(ABC):
    @abstractmethod
    def write_batch(self, batch: pa.RecordBatch):
        pass

    @abstractmethod
    def close(self):
        pass

class ParquetBatchWriter(BatchWriter):
    def __init__(self, path: str, schema: pa.Schema, compression='snappy', row_group_size=None):
        self.writer = pq.ParquetWriter(path, schema, compression=compression)
        self.row_group_size = row_group_size

    def write_batch(self, batch: pa.RecordBatch):
        self.writer.write_batch(batch, row_group_size=self.row_group_size)

    def close(self):
        self.writer.close()

class ArrowIPCBatchWriter(BatchWriter):
    def __init__(self, path: str, schema: pa.Schema):
        self.sink = pa.OSFile(path, 'wb')
        self.writer = ipc.new_file(self.sink, schema)

    def write_batch(self, batch: pa.RecordBatch):
        self.writer.write_batch(batch)

    def close(self):
        self.writer.close()
        self.sink.close()