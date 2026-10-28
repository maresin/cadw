"""Shared streaming series iterator for D4 TS classification models.

Extracted from cadw/ml/minirocket.py::_iter_series (v0.9.89-D4 refactor).
One series at a time; memory O(longest series length). Input must be
grouped/sorted by id_col (dataset contract for exported batches).

Guard contract:
  - per batch:      mg.check_batch(b, extra_factor=3.0)
  - per series buf: mg.check_state(sum(x.size for x in buf) * 8)
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard


def iter_series(df, value_col: str, id_col: Optional[str],
                mg: MemoryGuard, name: str
                ) -> Iterator[Tuple[object, np.ndarray]]:
    """Yield (series_id, np.ndarray[float64]) one series at a time.

    For id_col=None: yields (0, all_values_concatenated) exactly once.
    For id_col given: requires sorted/grouped input; raises ValueError
    if a series id appears again after being yielded.
    """
    if id_col is None:
        vals = []
        n = 0
        for b in df._iter_filtered_batches():
            mg.check_batch(b, name=name, extra_factor=3.0)
            vals.append(b.column(value_col).cast(pa.float64(), safe=False)
                        .to_numpy(zero_copy_only=False))
            n += b.num_rows
            mg.check_state(n * 8, name=name + ".state")
        if n == 0:
            raise ValueError("iter_series: empty input")
        yield 0, np.concatenate(vals)
        return

    cur_id = None
    buf = []
    seen = set()
    for b in df._iter_filtered_batches():
        mg.check_batch(b, name=name, extra_factor=3.0)
        ids = b.column(id_col).to_pylist()
        vals = b.column(value_col).cast(pa.float64(), safe=False)
        arr = vals.to_numpy(zero_copy_only=False)
        i, n = 0, b.num_rows
        while i < n:
            j = i
            while j < n and ids[j] == ids[i]:
                j += 1
            sid = ids[i]
            if cur_id is None:
                cur_id = sid
            if sid != cur_id:
                if sid in seen:
                    raise ValueError(
                        "iter_series: input must be grouped/sorted by "
                        f"id_col; saw {sid!r} again after yielding it")
                if buf:
                    seen.add(cur_id)
                    yield cur_id, np.concatenate(buf)
                cur_id = sid
                buf = []
            buf.append(arr[i:j])
            mg.check_state(sum(x.size for x in buf) * 8,
                           name=name + ".state")
            i = j
    if cur_id is None:
        raise ValueError("iter_series: empty input")
    if buf:
        yield cur_id, np.concatenate(buf)
