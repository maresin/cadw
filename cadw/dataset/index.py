"""
CadwIndex: идентичность строк в CadwDataFrame (v0.4.0a).

Два класса:
  - RangeIndex(start, stop, step) — lazy, O(1) память. Аналог pandas
    RangeIndex. Не материализуется.
  - MaterializedIndex(values) — Arrow Array/ChunkedArray. Для index
    из колонки (set_index).

Ограничение v0.4.0a: `df.index` и `set_index`/`reset_index` работают
только без активных фильтров. С фильтрами:
  - `df.index` — raise NotImplementedError (RangeIndex+filter = O(N) scan).
  - `set_index`/`reset_index` — raise NotImplementedError.
Синхронизация index с фильтрами — v0.4.0b (iloc).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import pyarrow as pa


class CadwIndex(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, key): ...

    @abstractmethod
    def __iter__(self) -> Iterator: ...

    @abstractmethod
    def to_arrow(self) -> pa.Array: ...


class RangeIndex(CadwIndex):
    """Lazy index: start, start+step, ..., < stop."""

    def __init__(self, start: int, stop: int, step: int = 1) -> None:
        if step <= 0:
            raise ValueError(f"RangeIndex: step must be > 0; got {step}")
        if start < 0:
            raise ValueError(f"RangeIndex: start must be >= 0; got {start}")
        if stop < start:
            raise ValueError(
                f"RangeIndex: stop ({stop}) must be >= start ({start})"
            )
        self.start = int(start)
        self.stop = int(stop)
        self.step = int(step)

    def __len__(self) -> int:
        return (self.stop - self.start) // self.step

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if not (0 <= key < len(self)):
                raise IndexError(f"RangeIndex: index out of range: {key}")
            return self.start + key * self.step
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            new_start = self.start + start * self.step
            new_stop = self.start + stop * self.step
            new_step = self.step * step if step else self.step
            return RangeIndex(new_start, new_stop, new_step)
        raise TypeError(f"RangeIndex: unsupported key {type(key).__name__}")

    def __iter__(self) -> Iterator[int]:
        for i in range(self.start, self.stop, self.step):
            yield i

    def to_arrow(self) -> pa.Array:
        return pa.array(list(self), type=pa.int64())

    def equals(self, other: "CadwIndex") -> bool:
        if not isinstance(other, RangeIndex):
            return False
        return (
            self.start == other.start
            and self.stop == other.stop
            and self.step == other.step
        )

    def __repr__(self) -> str:
        return f"RangeIndex(start={self.start}, stop={self.stop}, step={self.step})"


class MaterializedIndex(CadwIndex):
    """Materialized index: Arrow Array/ChunkedArray (сохраняет исходный dtype)."""

    def __init__(self, values: pa.Array | pa.ChunkedArray) -> None:
        if isinstance(values, pa.ChunkedArray):
            # combine_chunks для детерминированного доступа; O(N) память,
            # но index обычно уже материализован caller'ом.
            if values.num_chunks > 1:
                values = values.combine_chunks()
            values = values.chunk(0) if values.num_chunks == 1 else values
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key].as_py()
        if isinstance(key, slice):
            start, stop, _step = key.indices(len(self))
            return MaterializedIndex(self._values.slice(start, stop - start))
        raise TypeError(
            f"MaterializedIndex: unsupported key {type(key).__name__}"
        )

    def __iter__(self) -> Iterator:
        yield from self._values.to_pylist()

    def to_arrow(self) -> pa.Array:
        return self._values

    @property
    def dtype(self) -> pa.DataType:
        return self._values.type

    def equals(self, other: "CadwIndex") -> bool:
        if not isinstance(other, MaterializedIndex):
            return False
        if len(self) != len(other):
            return False
        if self._values.type != other._values.type:
            return False
        return self._values.equals(other._values)

    def get_indexer(self, labels) -> list[int]:
        """Map labels -> positions. O(N) scan, dict lookup.

        - labels: scalar -> [position] (или [] если not found).
        - labels: list -> list[int] (по одному position на label; -1 для
          not found).
        - duplicate labels in index: first occurrence wins (v0.4.0c MVP);
          v0.4.1 — all occurrences.
        """
        single = not isinstance(labels, (list, tuple))
        targets = [labels] if single else list(labels)

        # Build label -> first position map. O(N) scan, O(distinct) memory.
        # Naive dict lookup must be hashable; for non-hashable fallback —
        # O(N × K) linear scan (documented; rare).
        try:
            m: dict = {}
            for i, v in enumerate(self._values.to_pylist()):
                if v not in m:
                    m[v] = i
            positions = [m.get(t, -1) for t in targets]
        except TypeError:
            # Unhashable labels — fallback linear scan.
            pos_list = self._values.to_pylist()
            positions = []
            for t in targets:
                found = -1
                for i, v in enumerate(pos_list):
                    if v == t:
                        found = i
                        break
                positions.append(found)
        return positions

    def __repr__(self) -> str:
        n = len(self)
        preview = self._values.slice(0, min(5, n)).to_pylist()
        suffix = "..." if n > 5 else ""
        return f"MaterializedIndex({preview}{suffix}, n={n})"
