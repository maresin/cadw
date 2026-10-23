"""v0.9.34 (B7): _string_hash_column native dictionary-encode path.

Already-fixed-at-baseline proof: v0.9.18 (AUDIT §1.3) replaced
per-element to_pylist+xxhash with dictionary_encode + hash K uniques
+ vectorized map-back. 11.7x faster, -13.2 MB.
"""
import inspect

import pyarrow as pa

from cadw.ops.shuffle import _string_hash_column


def test_uses_dictionary_encode():
    src = inspect.getsource(_string_hash_column)
    assert "dictionary_encode" in src
    assert "uniq_hashes" in src
    # The old per-element path must only exist as the fallback branch.
    # In the fast path there is no `for i, v in enumerate(arr.to_pylist())`.


def test_correctness_cross_row():
    arr = pa.array(["a", "b", "c", "a", "b", None])
    h = _string_hash_column(arr)
    assert h[0] == h[3]
    assert h[1] == h[4]
    assert h[0] != h[1]
    # nulls all equal
    assert h[5] == _string_hash_column(pa.array([None, None]))[0]
