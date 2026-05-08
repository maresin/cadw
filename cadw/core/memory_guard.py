import pyarrow as pa

class MemoryGuard:
    def __init__(self, max_memory_mb: int):
        self.max_memory_mb = max_memory_mb
        self.row_size_bytes = 80   # значение по умолчанию (консервативная оценка)

    def estimate_row_size(self, schema=None, sample_batch=None):
        if sample_batch is not None:
            self.row_size_bytes = sample_batch.nbytes / sample_batch.num_rows
        elif schema is not None:
            total = 0
            for field in schema:
                if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
                    total += 8
                elif pa.types.is_string(field.type):
                    total += 50
                else:
                    total += 80
            self.row_size_bytes = max(1, total)
        else:
            self.row_size_bytes = 80
        return self.row_size_bytes

    def get_read_batch_size(self) -> int:
        max_bytes = self.max_memory_mb * 0.10 * 1024 * 1024
        return max(1, int(max_bytes / self.row_size_bytes))

    def get_write_batch_size(self) -> int:
        max_bytes = self.max_memory_mb * 0.20 * 1024 * 1024
        return max(1, int(max_bytes / self.row_size_bytes))

    def get_max_dict_elements(self) -> int:
        max_bytes = self.max_memory_mb * 0.60 * 1024 * 1024
        return max(1, int(max_bytes / 80))