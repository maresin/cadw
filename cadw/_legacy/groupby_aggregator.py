from cadw.io.batch_reader import BatchReader
from cadw._legacy.spill_manager import SpillManager
from typing import Dict, Any, List, Tuple
from collections import defaultdict

class GroupByAggregator:
    def __init__(self, reader: BatchReader, by_columns: List[str], agg_columns: List[str], 
                 memory_guard, agg_func: str = 'mean'):
        self.reader = reader
        self.by_columns = by_columns
        self.agg_columns = agg_columns
        self.mem = memory_guard
        self.agg_func = agg_func
        self.spillers = {col: SpillManager(memory_guard) for col in agg_columns}

    def _make_key(self, batch, row_idx) -> tuple:
        return tuple(batch.column(col)[row_idx].as_py() for col in self.by_columns)

    def _get_value(self, batch, col, row_idx):
        return batch.column(col)[row_idx].as_py()

    def collect(self) -> Dict[Any, Dict[str, Any]]:
        for batch in self.reader:
            num_rows = batch.num_rows
            for row_idx in range(num_rows):
                key = self._make_key(batch, row_idx)
                for col in self.agg_columns:
                    val = self._get_value(batch, col, row_idx)
                    spill = self.spillers[col]
                    if key not in spill.current_buffer:
                        if self.agg_func == 'sum':
                            spill.current_buffer[key] = 0.0
                        elif self.agg_func == 'mean':
                            spill.current_buffer[key] = (0.0, 0)
                        elif self.agg_func == 'count':
                            spill.current_buffer[key] = 0
                    if self.agg_func == 'sum':
                        spill.current_buffer[key] += val
                    elif self.agg_func == 'mean':
                        s, c = spill.current_buffer[key]
                        spill.current_buffer[key] = (s + val, c + 1)
                    elif self.agg_func == 'count':
                        spill.current_buffer[key] += 1
                    if len(spill.current_buffer) >= spill._max_size:
                        spill._spill()

        # Объединение результатов
        result = defaultdict(dict)
        for col, spill in self.spillers.items():
            merged = spill.merge()
            for key, agg in merged.items():
                if self.agg_func == 'sum':
                    result[key][col] = agg
                elif self.agg_func == 'count':
                    result[key][col] = agg
                elif self.agg_func == 'mean':
                    s, c = agg
                    result[key][col] = s / c if c > 0 else 0.0
            spill.cleanup()
        return dict(result)