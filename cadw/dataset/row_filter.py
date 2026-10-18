"""
RowFilter — ленивые фильтры строк.

Контракт:
  iter_filtered(batch_iter, schema) -> Iterator[RecordBatch]:
    применяет фильтр к потоку батчей, не материализуя.

  columns_required() -> set[str]:
    колонки, нужные фильтру (для selection перед итерацией).

Два вида:
  - MaskFilter: per-batch булев массив (Predicate, And, Or, Not).
  - SliceFilter: сквозной позиционный срез (не mask, нужна stateful итерация).

And/Or определены только для MaskFilter: комбинировать Slice с mask
нужно через chaining (`df[10:50][mask]`), а не через `&`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


class RowFilter(ABC):
    @abstractmethod
    def iter_filtered(self, batch_iter, schema) -> Iterator[pa.RecordBatch]: ...

    @abstractmethod
    def columns_required(self) -> set[str]: ...

    def __and__(self, other: "RowFilter") -> "AndFilter":
        return AndFilter(self, other)

    def __or__(self, other: "RowFilter") -> "OrFilter":
        return OrFilter(self, other)

    def __invert__(self) -> "NotFilter":
        return NotFilter(self)


class MaskFilter(RowFilter):
    @abstractmethod
    def mask_for_batch(self, batch: pa.RecordBatch) -> pa.Array: ...

    def iter_filtered(self, batch_iter, schema):
        for b in batch_iter:
            mask = self.mask_for_batch(b)
            if pc.any(mask).as_py():
                yield b.filter(mask)


class PredicateFilter(MaskFilter):
    _OPS = {
        "gt": pc.greater,
        "ge": pc.greater_equal,
        "lt": pc.less,
        "le": pc.less_equal,
        "eq": pc.equal,
        "ne": pc.not_equal,
    }

    def __init__(self, column: str, op: str, value) -> None:
        if op not in self._OPS:
            raise ValueError(f"unsupported op {op!r}; expected one of {list(self._OPS)}")
        self.column = column
        self.op = op
        self.value = value

    def mask_for_batch(self, batch):
        arr = batch.column(self.column)
        scalar = pa.scalar(self.value, type=arr.type)
        return self._OPS[self.op](arr, scalar)

    def columns_required(self):
        return {self.column}

    def __repr__(self) -> str:
        return f"PredicateFilter({self.column!r} {self.op} {self.value!r})"


class AndFilter(MaskFilter):
    def __init__(self, left, right) -> None:
        if not isinstance(left, MaskFilter):
            raise TypeError(f"AndFilter left: MaskFilter required, got {type(left).__name__}")
        if not isinstance(right, MaskFilter):
            raise TypeError(f"AndFilter right: MaskFilter required, got {type(right).__name__}")
        self.left, self.right = left, right

    def mask_for_batch(self, batch):
        return pc.and_(self.left.mask_for_batch(batch), self.right.mask_for_batch(batch))

    def columns_required(self):
        return self.left.columns_required() | self.right.columns_required()

    def __repr__(self) -> str:
        return f"AndFilter({self.left!r}, {self.right!r})"


class OrFilter(MaskFilter):
    def __init__(self, left, right) -> None:
        if not isinstance(left, MaskFilter):
            raise TypeError(f"OrFilter left: MaskFilter required, got {type(left).__name__}")
        if not isinstance(right, MaskFilter):
            raise TypeError(f"OrFilter right: MaskFilter required, got {type(right).__name__}")
        self.left, self.right = left, right

    def mask_for_batch(self, batch):
        return pc.or_(self.left.mask_for_batch(batch), self.right.mask_for_batch(batch))

    def columns_required(self):
        return self.left.columns_required() | self.right.columns_required()

    def __repr__(self) -> str:
        return f"OrFilter({self.left!r}, {self.right!r})"


class NotFilter(MaskFilter):
    def __init__(self, inner) -> None:
        if not isinstance(inner, MaskFilter):
            raise TypeError(f"NotFilter: MaskFilter required, got {type(inner).__name__}")
        self.inner = inner

    def mask_for_batch(self, batch):
        return pc.invert(self.inner.mask_for_batch(batch))

    def columns_required(self):
        return self.inner.columns_required()

    def __repr__(self) -> str:
        return f"NotFilter({self.inner!r})"


class SliceFilter(RowFilter):
    def __init__(self, start, stop, step) -> None:
        if step is None:
            step = 1
        if step <= 0:
            raise ValueError("step must be positive")
        self.start = start if start is not None else 0
        self.stop = stop
        self.step = step

    def iter_filtered(self, batch_iter, schema):
        pos = 0
        for b in batch_iter:
            n = b.num_rows
            positions = np.arange(pos, pos + n, dtype=np.int64)
            mask = positions >= self.start
            if self.stop is not None:
                mask &= positions < self.stop
            if self.step > 1:
                mask &= ((positions - self.start) % self.step == 0)
            if mask.any():
                yield b.filter(pa.array(mask, type=pa.bool_()))
            pos += n
            if self.stop is not None and pos >= self.stop:
                return

    def columns_required(self):
        return set()

    def __repr__(self) -> str:
        return f"SliceFilter({self.start}, {self.stop}, {self.step})"
