import pyarrow as pa
import pyarrow.ipc as ipc
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Iterator

class SpillManager:
    def __init__(self, memory_guard, spill_dir=None):
        self.memory_guard = memory_guard
        self.spill_dir = Path(spill_dir) if spill_dir else Path(tempfile.mkdtemp())
        self.current_buffer: Dict[Any, Any] = {}
        self.spilled_files: List[Path] = []
        self._max_size = memory_guard.get_max_dict_elements()

    def add(self, key: Any, value: Any = None):
        self.current_buffer[key] = value
        if len(self.current_buffer) >= self._max_size:
            self._spill()

    def _spill(self):
        # Сериализуем current_buffer в Arrow Table и записываем в IPC файл
        keys = list(self.current_buffer.keys())
        values = list(self.current_buffer.values())
        # Создаём таблицу с двумя колонками: 'key' и 'value'
        key_array = pa.array(keys)
        # values могут быть разными типами (для группировки – кортежи, для unique – None)
        # Преобразуем value в массив (если None, то пустой массив)
        if values and all(v is None for v in values):
            value_array = pa.array([0] * len(values))  # фиктивный массив, не будем использовать
        else:
            value_array = pa.array(values)
        table = pa.Table.from_arrays([key_array, value_array], names=['key', 'value'])
        # Записываем в файл
        fd, path = tempfile.mkstemp(suffix='.arrow', dir=self.spill_dir)
        with open(fd, 'wb') as f:
            writer = ipc.new_file(f, table.schema)
            writer.write_table(table)
            writer.close()
        self.spilled_files.append(Path(path))
        self.current_buffer.clear()

    def merge(self) -> Dict[Any, Any]:
        """Объединяет все спиллы в один словарь (для unique достаточно ключей)."""
        # Сначала загружаем все спилл-файлы
        for spill_path in self.spilled_files:
            with open(spill_path, 'rb') as f:
                reader = ipc.open_file(f)
                table = reader.read_all()
                keys = table.column('key').to_pylist()
                vals = table.column('value').to_pylist()
                for k, v in zip(keys, vals):
                    self.current_buffer[k] = v
            spill_path.unlink()
        self.spilled_files.clear()
        return self.current_buffer

    def cleanup(self):
        for p in self.spilled_files:
            if p.exists():
                p.unlink()
        self.spilled_files.clear()