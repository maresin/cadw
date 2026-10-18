from cadw.io.batch_reader import BatchReader
from cadw._legacy.spill_manager import SpillManager
from typing import Set, Any

class UniqueCollector:
    def __init__(self, reader: BatchReader, column: str, memory_guard):
        self.reader = reader
        self.column = column
        self.mem = memory_guard
        self.spiller = SpillManager(memory_guard)

    def collect(self) -> Set[Any]:
        try:
            first_batch = next(self.reader)
            # Передаём sample_batch вместо schema
            self.mem.estimate_row_size(sample_batch=first_batch)
        except StopIteration:
            return set()

        for batch in self.reader:
            arr = batch.column(self.column)
            for v in arr.to_pylist():
                self.spiller.add(v, None)

        merged = self.spiller.merge()
        self.spiller.cleanup()
        return set(merged.keys())