"""K-sample estimation via Arrow-native dictionary_encode."""
from __future__ import annotations

from typing import Sequence

import pyarrow as pa
import pyarrow.compute as pc


def estimate_k_sample_arrow(
    sample_batch: pa.RecordBatch,
    subset_cols: Sequence[str],
) -> int:
    """Estimate unique-key count in `sample_batch` on `subset_cols`.

    Arrow-native. No pandas. Used by groupby and dedup for K_est
    before HashShuffle (P selection).

    Single column: `pc.dictionary_encode(...).dictionary.length`
    (exact). Multi-column: binary-join (cast to string) +
    dictionary_encode (safe upper bound — string concat collisions
    not possible with "\x01" separator).

    Empty batch → 0.
    """
    cols = list(subset_cols)
    if not cols:
        raise ValueError("estimate_k_sample_arrow: subset_cols empty")
    if sample_batch.num_rows == 0:
        return 0
    if len(cols) == 1:
        arr = sample_batch.column(cols[0])
        enc = arr.dictionary_encode()
        if enc.dictionary is not None:
            return len(enc.dictionary)
        # Fallback: count_distinct (some dict-enc kernels missing)
        return int(pc.count_distinct(arr).as_py())
    # Multi-column: binary_join_element_wise with "\x01" separator.
    if not hasattr(pc, "binary_join_element_wise"):
        # Fallback: product upper bound. Used only on Arrow < X.
        product = 1
        for c in cols:
            product *= int(pc.count_distinct(sample_batch.column(c)).as_py())
            if product >= sample_batch.num_rows:
                return sample_batch.num_rows
        return product
    str_cols = [
        sample_batch.column(c)
        if pa.types.is_string(sample_batch.column(c).type)
        else pc.cast(sample_batch.column(c), pa.string())
        for c in cols
    ]
    joined = str_cols[0]
    for c in str_cols[1:]:
        joined = pc.binary_join_element_wise(joined, c, "\x01")
    enc = joined.dictionary_encode()
    if enc.dictionary is not None:
        return len(enc.dictionary)
    return int(pc.count_distinct(joined).as_py())


__all__ = ["estimate_k_sample_arrow"]

