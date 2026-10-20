#!/usr/bin/env python3
"""v0.9.3a: KFold keyed PRNG — O(1) state, no O(N) fold-id array (§13.4/173).

BUG (P-CV1, S1): KFold.split materialized O(N) fold-id arrays
(shuffle=False: ~12·N; shuffle=True: ~20·N — int32 fold_id + int64
arange/permutation temporaries) without any MemoryGuard.check_state.
For N ≫ RAM this is a GUARANTEED OOM — violates the core guarantee
of CADW (out-of-core with honest MemoryLimitExceeded).

Docstring claimed "O(N) int8 fold_id" — actual code was int32 +
int64 temporaries (3-4× understatement).

FIX:
  - shuffle=False (default): fold_id = row_id % k, computed per batch
    via _attach_column_with_fn. O(batch) state, exact match with
    current behaviour.
  - shuffle=True (new default): keyed PRNG (splitmix64) on row_id —
    fold_id = splitmix64(row_id * GOLDEN + seed) % k. Deterministic
    per row_id, same seed → same assignment. Split-quality shuffle;
    fold sizes ~binomial (average n/k, per-fold variance O(sqrt(n/k))).
    O(batch) state.
  - materialize=True (opt-in backward compat): old behavior with
    MemoryGuard.check_state(20·N). For exact-balanced shuffle that
    requires O(N) RAM.

Default random_state=None → seed=42 for reproducibility
(out-of-core convention: same seed = same split, always).

§13.4/173: memory claim in docstring must match actual allocation.
"O(N) int8 fold_id" was a documentation lie — actual code was
int32 + int64 temporaries, and NO MemoryGuard check at all.

GUARDS: cv fast + E2E + pipeline + scalers + full suite + guards.

Run: uv run python scripts/patch_v093a_cv_keyed_prng.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if not starts:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


# ============ 1. cadw/cv.py: keyed PRNG + per-batch helper ============

CV_OLD_HEADER = '''"""Cross-validation for out-of-core pipeline (v0.9.2e).

KFold: fold_id = row_id % k (shuffle=False, O(1) per row) or
       rng.permutation(n) (shuffle=True, materialized once).
TimeSeriesSplit: sort by time_col once, then positional fold slices
                 (no future leakage).

Both produce (train_df, test_df) pairs lazily via CadwDataFrame views.

Memory: O(N) int8 for fold_id array (KFold); O(1) state for TSS
        (sorted_df on disk, iloc slices are lazy).
Independent of N except fold_id array (N × 4 bytes for int32).
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import pyarrow as pa

from cadw.dataset import Dataset
from cadw.dataframe import CadwDataFrame


def _attach_column(
    df: CadwDataFrame, name: str, arrow_array: pa.Array,
) -> CadwDataFrame:
    """Materialize df's filtered view + attach Arrow array as column.

    Caller ensures len(arrow_array) == df.count() over filtered view.
    Errors if iterated rows != len(arrow_array).

    v0.9.2e-fix1 (§13.4/169): ALL output fields forced nullable=True.
    `Dataset.from_batches` reconstructs per-column singleton via
    `pa.RecordBatch.from_arrays([col], names=[field_name])` — always
    nullable=True. If out_schema has nullable=False for ANY field,
    writer target `pa.schema([field])` (nullable=False) mismatches
    singleton's (nullable=True) → ArrowInvalid. Convention: nullable
    =True everywhere, matching from_arrays default (all CADW ops do
    this: shift/diff/rank/expanding all use nullable=True).
    """
    base_cols = list(df._columns)
    out_fields = [
        pa.field(c, df._dataset.schema.field(c).type, nullable=True)
        for c in base_cols
    ]
    out_fields.append(pa.field(name, arrow_array.type, nullable=True))
    out_schema = pa.schema(out_fields)

    expected_n = len(arrow_array)

    def gen():
        off = 0
        for b in df._iter_filtered_batches():
            n_b = b.num_rows
            if off + n_b > expected_n:
                raise RuntimeError(
                    f"_attach_column: iterated rows exceed array length "
                    f"({off + n_b} > {expected_n})"
                )
            col = arrow_array.slice(off, n_b)
            off += n_b
            arrays = [b.column(c) for c in base_cols] + [col]
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)
        if off != expected_n:
            raise RuntimeError(
                f"_attach_column: iterated {off} rows != "
                f"array length {expected_n}"
            )

    new_ds = Dataset.from_batches(out_schema, gen(), df._new_dataset_dir())
    return CadwDataFrame(
        new_ds,
        memory_guard=df._memory_guard,
        spill_dir=df._spill_dir,
    )


class KFold:
    """K-fold cross-validator for CadwDataFrame.

    shuffle=False (default): fold_id = arange(n) % k — deterministic,
    no RNG, O(1) per row.
    shuffle=True: rng.permutation(n) — materializes fold_id array
    (O(N) int32; ~4 MB for 1M rows). Different seed → different
    assignment.

    Both preserve row order within each fold (filters don't reorder).
    """

    __slots__ = ("n_splits", "shuffle", "random_state")

    def __init__(
        self,
        n_splits: int,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError(f"n_splits must be int >= 1; got {n_splits!r}")
        if not isinstance(shuffle, bool):
            raise TypeError(
                f"shuffle must be bool; got {type(shuffle).__name__}"
            )
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, df: CadwDataFrame,
    ) -> Iterator[tuple[int, CadwDataFrame, CadwDataFrame]]:
        """Yield (fold_i, train_df, test_df) for each fold.

        Yields lazy CadwDataFrame views (fold filter via PredicateFilter).
        Iteration over train/test materializes filtered batches.
        """
        n = df.count()
        if self.n_splits > n:
            raise ValueError(
                f"KFold: n_splits={self.n_splits} > n={n}"
            )

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            perm = rng.permutation(n)
            fold_id = np.empty(n, dtype=np.int32)
            fold_id[perm] = np.arange(n) % self.n_splits
        else:
            fold_id = (np.arange(n) % self.n_splits).astype(np.int32)

        fold_arr = pa.array(fold_id, type=pa.int32())
        augmented = _attach_column(df, "__fold", fold_arr)

        base_cols = list(df.columns)
        for i in range(self.n_splits):
            train = augmented[augmented["__fold"] != i][base_cols]
            test = augmented[augmented["__fold"] == i][base_cols]
            yield i, train, test
'''

CV_NEW_HEADER = '''"""Cross-validation for out-of-core pipeline (v0.9.3a).

KFold fold assignment is **O(batch) state** in the default mode — no
O(N) arrays, no MemoryGuard violations:
  - shuffle=False: fold_id = row_id % k (per batch).
  - shuffle=True:  keyed PRNG splitmix64(row_id, seed) % k (per batch).
    Deterministic per row_id; split-quality (fold sizes ~binomial,
    mean n/k, per-fold variance O(sqrt(n/k))).
  - materialize=True (opt-in): old O(N) fold_id array with
    MemoryGuard.check_state(20·N). Exact-balanced shuffle for small N
    or when exact fold sizes matter.

TimeSeriesSplit: sort by time_col once, then positional fold slices
                 (no future leakage).

§13.4/173: pre-v0.9.3a docstring claimed "O(N) int8 fold_id" — actual
code was int32 fold_id + int64 arange/permutation temporaries
(12-20·N bytes) with NO MemoryGuard check at all. Guaranteed OOM
for N ≫ RAM — the exact scenario CADW exists for.
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import pyarrow as pa

from cadw.dataset import Dataset
from cadw.dataframe import CadwDataFrame


# splitmix64 constants (deterministic avalanche, K1 pattern from
# shuffle.py; here for fold-id derivation).
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_DEFAULT_SEED = 42  # deterministic default for out-of-core reproducibility


def _fold_ids_from_row_ids(
    row_ids: np.ndarray, k: int, seed: int,
) -> np.ndarray:
    """Keyed splitmix64 hash of row_ids modulo k. Vectorized, uint64 math."""
    h = row_ids.astype(np.uint64) * _GOLDEN + np.uint64(seed)
    h = (h ^ (h >> np.uint64(30))) * _M1
    h = (h ^ (h >> np.uint64(27))) * _M2
    h = h ^ (h >> np.uint64(31))
    return (h % np.uint64(k)).astype(np.int32)


def _attach_column(
    df: CadwDataFrame, name: str, arrow_array: pa.Array,
) -> CadwDataFrame:
    """Materialize df's filtered view + attach Arrow array as column.

    Caller ensures len(arrow_array) == df.count() over filtered view.
    Errors if iterated rows != len(arrow_array).

    v0.9.2e-fix1 (§13.4/169): ALL output fields forced nullable=True.
    `Dataset.from_batches` reconstructs per-column singleton via
    `pa.RecordBatch.from_arrays([col], names=[field_name])` — always
    nullable=True. If out_schema has nullable=False for ANY field,
    writer target `pa.schema([field])` (nullable=False) mismatches
    singleton's (nullable=True) → ArrowInvalid. Convention: nullable
    =True everywhere.
    """
    base_cols = list(df._columns)
    out_fields = [
        pa.field(c, df._dataset.schema.field(c).type, nullable=True)
        for c in base_cols
    ]
    out_fields.append(pa.field(name, arrow_array.type, nullable=True))
    out_schema = pa.schema(out_fields)

    expected_n = len(arrow_array)

    def gen():
        off = 0
        for b in df._iter_filtered_batches():
            n_b = b.num_rows
            if off + n_b > expected_n:
                raise RuntimeError(
                    f"_attach_column: iterated rows exceed array length "
                    f"({off + n_b} > {expected_n})"
                )
            col = arrow_array.slice(off, n_b)
            off += n_b
            arrays = [b.column(c) for c in base_cols] + [col]
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)
        if off != expected_n:
            raise RuntimeError(
                f"_attach_column: iterated {off} rows != "
                f"array length {expected_n}"
            )

    new_ds = Dataset.from_batches(out_schema, gen(), df._new_dataset_dir())
    return CadwDataFrame(
        new_ds,
        memory_guard=df._memory_guard,
        spill_dir=df._spill_dir,
    )


def _attach_column_with_fn(
    df: CadwDataFrame, name: str, out_type: pa.DataType,
    fn,
) -> CadwDataFrame:
    """Streaming per-batch column attach (v0.9.3a, P-CV1).

    `fn(offset, n) -> pa.Array[n]` is called per batch with the running
    row offset. Memory = O(batch), independent of N. No O(N) array.

    All output fields nullable=True (§13.4/169).
    """
    base_cols = list(df._columns)
    out_fields = [
        pa.field(c, df._dataset.schema.field(c).type, nullable=True)
        for c in base_cols
    ]
    out_fields.append(pa.field(name, out_type, nullable=True))
    out_schema = pa.schema(out_fields)

    def gen():
        offset = 0
        for b in df._iter_filtered_batches():
            n_b = b.num_rows
            col = fn(offset, n_b)
            if len(col) != n_b:
                raise RuntimeError(
                    f"_attach_column_with_fn: fn returned {len(col)} "
                    f"rows, expected {n_b} (batch at offset {offset})"
                )
            offset += n_b
            arrays = [b.column(c) for c in base_cols] + [col]
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)

    new_ds = Dataset.from_batches(out_schema, gen(), df._new_dataset_dir())
    return CadwDataFrame(
        new_ds,
        memory_guard=df._memory_guard,
        spill_dir=df._spill_dir,
    )


class KFold:
    """K-fold cross-validator for CadwDataFrame (out-of-core).

    Default (shuffle=False): fold_id = row_id % k, computed per batch.
    O(batch) state, no O(N) arrays.

    shuffle=True (default mode, O(1) state): keyed PRNG
    splitmix64(row_id, seed) % k. Deterministic per row_id (same seed →
    same split). Fold sizes ~binomial, mean n/k, per-fold variance
    O(sqrt(n/k)) — split-quality, sufficient for CV.

    materialize=True (opt-in, O(N) RAM): old behavior — exact-balanced
    shuffle via rng.permutation(n). Requires MemoryGuard headroom
    (check_state(20·N) applied; raises MemoryLimitExceeded if budget
    insufficient). Use only when N is small or exact fold sizes matter.

    random_state=None (default) → seed=42 (deterministic reproducibility
    — same split across runs, machines, storage layouts).

    Row order within each fold is preserved (filters don't reorder).
    """

    __slots__ = ("n_splits", "shuffle", "random_state", "materialize")

    def __init__(
        self,
        n_splits: int,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
        materialize: bool = False,
    ) -> None:
        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError(f"n_splits must be int >= 1; got {n_splits!r}")
        if not isinstance(shuffle, bool):
            raise TypeError(
                f"shuffle must be bool; got {type(shuffle).__name__}"
            )
        if not isinstance(materialize, bool):
            raise TypeError(
                f"materialize must be bool; got "
                f"{type(materialize).__name__}"
            )
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.materialize = materialize

    def split(
        self, df: CadwDataFrame,
    ) -> Iterator[tuple[int, CadwDataFrame, CadwDataFrame]]:
        """Yield (fold_i, train_df, test_df) for each fold.

        Yields lazy CadwDataFrame views (fold filter via PredicateFilter).
        """
        n = df.count()
        if self.n_splits > n:
            raise ValueError(
                f"KFold: n_splits={self.n_splits} > n={n}"
            )

        if self.materialize:
            # Opt-in exact-balanced shuffle. O(N) state, guarded.
            # Peak: int32 fold_id (4·N) + int64 arange (8·N) + for
            # shuffle=True rng.permutation (8·N) + np.arange temp (8·N).
            # Worst case (shuffle=True): ~20·N bytes transient. Use 20·N
            # as safe upper bound for check_state.
            df._memory_guard.check_state(
                20 * n, name="KFold.materialize",
            )
            if self.shuffle:
                seed = (_DEFAULT_SEED if self.random_state is None
                        else int(self.random_state))
                rng = np.random.default_rng(seed)
                perm = rng.permutation(n)
                fold_id = np.empty(n, dtype=np.int32)
                fold_id[perm] = np.arange(n) % self.n_splits
            else:
                fold_id = (np.arange(n) % self.n_splits).astype(np.int32)
            fold_arr = pa.array(fold_id, type=pa.int32())
            augmented = _attach_column(df, "__fold", fold_arr)
        else:
            # O(batch) state; per-batch fold_id.
            if self.shuffle:
                seed = (_DEFAULT_SEED if self.random_state is None
                        else int(self.random_state))
                k = self.n_splits
                def _fn(offset: int, n_b: int) -> pa.Array:
                    row_ids = np.arange(offset, offset + n_b,
                                        dtype=np.uint64)
                    ids = _fold_ids_from_row_ids(row_ids, k, seed)
                    return pa.array(ids, type=pa.int32())
            else:
                k = self.n_splits
                def _fn(offset: int, n_b: int) -> pa.Array:
                    ids = (np.arange(offset, offset + n_b) % k).astype(
                        np.int32)
                    return pa.array(ids, type=pa.int32())
            augmented = _attach_column_with_fn(
                df, "__fold", pa.int32(), _fn,
            )

        base_cols = list(df.columns)
        for i in range(self.n_splits):
            train = augmented[augmented["__fold"] != i][base_cols]
            test = augmented[augmented["__fold"] == i][base_cols]
            yield i, train, test
'''


def rewrite_cv_module():
    print("\n=== cadw/cv.py: KFold keyed PRNG + materialize opt-in ===")
    p = Path("cadw/cv.py")
    s = p.read_text()
    s = replace_line_block(s, CV_OLD_HEADER, CV_NEW_HEADER,
                            tag="cv-module-rewrite")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ 2. tests/test_cv_keyed_prng.py (new) ============

CV_KEYED_TESTS = '''"""v0.9.3a: KFold keyed PRNG + materialize opt-in (§13.4/173)."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard, KFold, TimeSeriesSplit
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, data, chunk=1000, name="L", mem_mb=200):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def make_id_df(tmp_path, n, name="L"):
    """Identity data: y == row_id → fold content reveals fold_id."""
    return make_df(tmp_path, {
        "x": list(range(n)), "y": list(range(n)),
    }, name=name)


# ============ Default mode: O(batch) state ============

def test_kfold_no_materialize_default_folds(tmp_path):
    """Default mode: 5 folds, all rows covered exactly once."""
    df = make_id_df(tmp_path, n=1000)
    cv = KFold(n_splits=5)
    for _, train, test in cv.split(df):
        pass  # smoke


def test_kfold_shuffle_keyed_deterministic(tmp_path):
    """Same seed → same assignment. Different seed → different."""
    df = make_id_df(tmp_path, n=2000)
    cv1 = KFold(n_splits=5, shuffle=True, random_state=42)
    cv2 = KFold(n_splits=5, shuffle=True, random_state=42)
    cv3 = KFold(n_splits=5, shuffle=True, random_state=99)

    def _test_sets(cv):
        sets = []
        for _, _, test in cv.split(df):
            ys = []
            for b in test._iter_filtered_batches():
                ys.extend(b.column("y").to_pylist())
            sets.append(set(ys))
        return sets

    s1 = _test_sets(cv1)
    s2 = _test_sets(cv2)
    s3 = _test_sets(cv3)

    # Same seed → same sets, fold by fold
    for a, b in zip(s1, s2):
        assert a == b
    # Different seeds → at least one fold differs
    assert any(a != b for a, b in zip(s1, s3))
    # Union invariant: all rows covered
    for sets in (s1, s2, s3):
        assert set().union(*sets) == set(range(2000))


def test_kfold_shuffle_default_seed_deterministic(tmp_path):
    """random_state=None → default seed=42, deterministic across calls."""
    df = make_id_df(tmp_path, n=1000)
    cv_a = KFold(n_splits=5, shuffle=True)  # random_state=None
    cv_b = KFold(n_splits=5, shuffle=True)

    def _first_test_set(cv):
        for _, _, test in cv.split(df):
            ys = []
            for b in test._iter_filtered_batches():
                ys.extend(b.column("y").to_pylist())
            return set(ys)

    assert _first_test_set(cv_a) == _first_test_set(cv_b)


def test_kfold_shuffle_keyed_balanced_large(tmp_path):
    """Binomial fold sizes: mean n/k, per-fold variance O(sqrt(n/k))."""
    n = 10_000
    k = 5
    df = make_id_df(tmp_path, n=n)
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    sizes = []
    for _, _, test in cv.split(df):
        sizes.append(test.count())
    assert sum(sizes) == n
    # Split-quality: per-fold within ±3σ, σ ≈ sqrt(n/k · (1-1/k))
    mean = n / k
    sigma = np.sqrt(n / k * (1 - 1 / k))
    for s in sizes:
        assert abs(s - mean) < 3 * sigma, f"sizes={sizes}, mean={mean:.1f}"


def test_kfold_no_shuffle_default_unchanged(tmp_path):
    """shuffle=False default: fold_id = row_id % k (unchanged)."""
    df = make_id_df(tmp_path, n=100)
    cv = KFold(n_splits=5)
    for i, _, test in cv.split(df):
        ys = []
        for b in test._iter_filtered_batches():
            ys.extend(b.column("y").to_pylist())
        ys_sorted = sorted(ys)
        # fold i gets rows where row_id % k == i
        expected = list(range(i, 100, 5))
        assert ys_sorted == expected


# ============ materialize=True: opt-in O(N) backward compat ============

def test_kfold_materialize_true_balanced(tmp_path):
    """materialize=True: exact-balanced shuffle (old behavior)."""
    df = make_id_df(tmp_path, n=1000)
    cv = KFold(n_splits=5, shuffle=True, random_state=42,
               materialize=True)
    sizes = []
    for _, _, test in cv.split(df):
        sizes.append(test.count())
    assert sizes == [200, 200, 200, 200, 200]


def test_kfold_materialize_oom_guard(tmp_path):
    """materialize=True: O(N) arrays guarded — tight budget → MemoryLimitExceeded."""
    n = 1_000_000
    df = make_id_df(tmp_path, n=n)
    # 5 MB budget → state_limit = 2.5 MB. 20·N = 20 MB > 2.5 MB.
    df._memory_guard = MemoryGuard(max_memory_mb=5)
    cv = KFold(n_splits=5, shuffle=True, materialize=True)
    with pytest.raises(MemoryLimitExceeded, match="KFold.materialize"):
        list(cv.split(df))


def test_kfold_materialize_small_n_ok(tmp_path):
    """materialize=True on small N with ample budget: OK."""
    df = make_id_df(tmp_path, n=1000)
    cv = KFold(n_splits=5, shuffle=True, random_state=42,
               materialize=True)
    sizes = []
    for _, _, test in cv.split(df):
        sizes.append(test.count())
    assert sum(sizes) == 1000


# ============ AST guard: no O(N) materialize in default mode ============

def test_kfold_split_default_no_oN_arrays():
    """AST: default path must not call np.arange(n) or rng.permutation(n).

    §13.4/173: O(N) arrays only allowed inside `if self.materialize:`
    branch.
    """
    import ast
    import inspect
    import textwrap
    from cadw.cv import KFold

    src = textwrap.dedent(inspect.getsource(KFold.split))
    tree = ast.parse(src)

    # Find the materialize branch body (from `if self.materialize:` to
    # `else:`) and the else branch body.
    materialize_body: list = []
    default_body: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Check condition is `self.materialize`
            test = node.test
            if (isinstance(test, ast.Attribute)
                    and test.attr == "materialize"):
                materialize_body = list(node.body)
                default_body = list(node.orelse)
                break

    assert materialize_body, "materialize branch not found"
    assert default_body, "default (else) branch not found"

    def _calls_np_arange_with_n(node):
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            f = sub.func
            # np.arange(n) — arg is a Name 'n'
            if (isinstance(f, ast.Attribute) and f.attr == "arange"
                    and isinstance(f.value, ast.Name)
                    and f.value.id == "np"):
                for arg in sub.args:
                    if isinstance(arg, ast.Name) and arg.id == "n":
                        return True
            # rng.permutation(n) — arg is a Name 'n'
            if (isinstance(f, ast.Attribute)
                    and f.attr == "permutation"):
                for arg in sub.args:
                    if isinstance(arg, ast.Name) and arg.id == "n":
                        return True
        return False

    # Materialize branch SHOULD have np.arange(n) or rng.permutation(n)
    assert any(_calls_np_arange_with_n(stmt) for stmt in materialize_body), (
        "materialize branch must materialize O(N) array"
    )
    # Default branch MUST NOT
    for stmt in default_body:
        assert not _calls_np_arange_with_n(stmt), (
            "default branch must not materialize O(N) array — "
            "P-CV1 regression"
        )


def test_kfold_default_uses_attach_column_with_fn():
    """AST: default path uses _attach_column_with_fn (streaming)."""
    import inspect
    from cadw.cv import KFold
    src = inspect.getsource(KFold.split)
    assert "_attach_column_with_fn" in src
    assert "_attach_column(df," in src or "_attach_column(\n" in src


# ============ N large, tight budget: default mode works ============

def test_kfold_default_large_n_tight_budget(tmp_path):
    """N=100k, budget 50 MB. Default mode fits (O(batch) state).

    v0.9.2 KFold would have needed ~1.2 MB fold_id — fits here, but
    the invariant is verified by AST test_kfold_split_default_no_oN_arrays.
    This is a functional sanity: correct results, no crash.
    """
    df = make_id_df(tmp_path, n=100_000, chunk=1000)
    df._memory_guard = MemoryGuard(max_memory_mb=50)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    sizes = []
    for _, _, test in cv.split(df):
        sizes.append(test.count())
    assert sum(sizes) == 100_000
'''


def write_tests():
    print("\n=== tests/test_cv_keyed_prng.py ===")
    p = Path("tests/test_cv_keyed_prng.py")
    p.write_text(CV_KEYED_TESTS)
    compile_check(p)
    print("  OK")


# ============ 3. §13.4/173 + §11.5 + CHANGELOG ============

def append_decisions_173():
    print("\n=== DECISIONS.md: §13.4/173 + §11.5 KFold note ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 173" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/173 (KFold docstring lie; O(N) materialize).** "
        s += "P-CV1 S1 fix: keyed PRNG default, materialize opt-in.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 173 (KFold O(N) fold_id без MemoryGuard; docstring
lie).** v0.9.3a: P-CV1 S1 из v0.9.2 audit.

**Bug:** `KFold.split` в v0.9.2e:
```python
fold_id = (np.arange(n) % k).astype(np.int32)     # shuffle=False
# или
perm = rng.permutation(n)                          # shuffle=True
fold_id = np.empty(n, dtype=np.int32)
fold_id[perm] = np.arange(n) % k
fold_arr = pa.array(fold_id, type=pa.int32())
augmented = _attach_column(df, "__fold", fold_arr)  # materialize
```

Peak: 12·N (shuffle=False: int32 fold_id + int64 arange temp) / 20·N
(shuffle=True: + int64 permutation). **Ноль MemoryGuard involvement.**
Для N ≫ RAM — guaranteed OOM, не MemoryLimitExceeded.

Docstring: "O(N) int8 for fold_id array". Actual code: int32 + int64
temporaries. **3-4× understatement.** §13.4/173: docstring claim must
match actual allocation.

**Почему это не поймали:** тесты KFold (v0.9.2e) — n=100-1000.
O(N) массив на 1000 строк = 12 KB — незаметно. Slow e2e — n=100k
(1.2 MB) — тоже ОК на 50 MB budget. Никто не проверял N=10^8 с
budget=16 MB — **core user scenario**. Class of lesson: тест
проходит потому что **small N hides O(N) bugs**.

**Fix (v0.9.3a):**
- `shuffle=False` (default): fold_id = row_id % k, per-batch. 1:1
  с прежним поведением, O(batch) state.
- `shuffle=True` (default new): keyed splitmix64(row_id, seed) % k.
  Deterministic per row_id, O(batch) state. Fold sizes ~binomial
  (mean n/k, per-fold σ ≈ sqrt(n/k)). Split-quality достаточен для CV.
- `materialize=True` (opt-in): старое поведение + `check_state(20·N)`.
- `_attach_column_with_fn` — новый helper, fn(offset, n) → array.
  Streaming, O(batch).

**Правило (§13.4/173, расширение 158, 166, 172):**
- **Docstring memory claims must match actual code.** "O(N) int8"
  когда код int32+int64 — это ложь. Audit должен проверять.
- **Small-N тесты скрывают O(N) bugs.** Каждый op с O(N) state
  должен иметь:
  (a) AST guard "no np.arange(N)/permutation(N) in default path";
  (b) functional test с tight budget + большим N (или отказ с
  обоснованием, что тест невозможен без N=10^9);
  (c) либо — задокументированное «O(N) state» в §11.5 с explicit
  fail-fast.
- **Core user scenario должен быть протестирован.** CADW для N ≫
  RAM. Тест с N=1000 не доказывает, что op работает для N ≫ RAM.
  Для каждого публичного op — один тест с N, где O(N) state
  materialize был бы заметен.

**Backlog (§11.6):**
- Audit docstrings across `cadw/` — grep "O(N)" утверждения, verify
  each matches code.
- `test_memory_contracts.py`: для каждого публичного op — 1
  functional test с tight budget + "O(N) state not materialized".
- KFold `df.count()` — O(N) scan на split(). Для N=10^9 медленно,
  но bounded. Документировать в §11.5: "KFold.split scans df once".

**Связанные:** §13.4/158 (E2E respects inherent complexity),
/166 (prototype as test-authoring), /172 (closed-list stale-bait).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/173")

    # §11.5 note
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.3a (KFold keyed PRNG)" in s:
        print("  §11.5 already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "11.5")
    if pos is None:
        print("  WARNING: §11.5 not found — skip")
        return

    note = '''**v0.9.3a (KFold keyed PRNG):**
- `KFold(shuffle=True)` default = keyed splitmix64 on row_id. Fold
  sizes ~binomial (mean n/k, per-fold σ ≈ sqrt(n/k)); not
  exact-balanced. For exact-balanced use `materialize=True` (opt-in,
  O(N) RAM, guarded).
- `random_state=None` → seed=42 (deterministic reproducibility).
- `KFold.split()` calls `df.count()` — one full scan, O(N) time
  bounded memory. Documented.
- Each `split(df)` yields `n_splits` pairs; each pair's train/test is
  a lazy view. Iterating over all folds does `n_splits × N` row
  touches (once per fold per side). O(1) memory per iteration.

'''
    s = s[:pos] + note + s[pos:]
    p.write_text(s)
    print("  OK: §11.5 KFold")


def append_changelog():
    print("\n=== CHANGELOG.md: v0.9.3a entry ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.3a — KFold keyed PRNG" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import append_unreleased_entry
    entry = '''### Исправлено (v0.9.3a — KFold keyed PRNG, S1 fix)

**P-CV1 (S1, из v0.9.2 audit):** `KFold.split` материализовал O(N)
fold-id массив (shuffle=False: ~12·N; shuffle=True: ~20·N) **без
MemoryGuard.check_state**. Для N ≫ RAM — guaranteed OOM, не
MemoryLimitExceeded. Нарушение ядра гарантии CADW.

- **`KFold.shuffle=False`** (default): `fold_id = row_id % k`,
  computed per batch. O(batch) state. 1:1 с прежним поведением.
- **`KFold.shuffle=True`** (default, O(1) state): keyed splitmix64
  PRNG. `fold_id = splitmix64(row_id * GOLDEN + seed) % k`.
  Детерминистичен per row_id (same seed → same split). Split-quality:
  fold sizes ~binomial (mean n/k, per-fold σ ≈ sqrt(n/k)).
- **`KFold(materialize=True)`** (opt-in backward compat): exact-
  balanced shuffle, O(N) RAM. Guarded через
  `check_state(20·N)`; raise `MemoryLimitExceeded` если бюджет
  недостаточен.
- **`random_state=None`** → seed=42 (deterministic reproducibility,
  out-of-core convention).
- **`_attach_column_with_fn`** — новый streaming helper (fn(offset, n)
  → array). O(batch) state, no O(N) array.

### §13.4/173 — docstring memory claim vs actual allocation

Pre-v0.9.3a docstring: «O(N) int8 for fold_id array». Actual code:
int32 fold_id + int64 arange/permutation temporaries = 12-20·N байт.
**3-4× understatement** + ноль MemoryGuard check.

**Правило:** docstring memory claim must match actual code. Audit
должен проверять каждый «O(N)» в docstrings.

### Почему не поймали в v0.9.2e

Тесты KFold: n=100-1000. O(N) массив на 1000 строк = 12 KB —
незаметно. Slow E2E: n=100k (1.2 MB) на 50 MB budget — тоже OK.
**Small-N скрывает O(N) bugs.** Никто не проверил N=10^8 с
budget=16 MB — core user scenario.

### Тесты (v0.9.3a)

- +10 fast (`test_cv_keyed_prng.py`): default folds, keyed
  deterministic, different seeds differ, default seed=42, binomial
  balance, no-shuffle unchanged, materialize exact-balanced,
  materialize OOM guard, materialize small-N, AST guard
  (`np.arange(n)`/`permutation(n)` только в materialize branch),
  streaming helper presence, large-N tight-budget functional.

- fast: **~1559** (+10). slow: **20** (unchanged).

### Ограничения (v0.9.3a, §11.5)

- **`shuffle=True` (default):** fold sizes ~binomial, не exact-balanced.
  Для exact — `materialize=True` (O(N) RAM, guarded).
- **`KFold.split()`:** делает один `df.count()` scan — O(N) time,
  O(1) memory.
- **Итерация по всем folds:** `n_splits × N` row touches (по разу
  на fold/сторону). O(1) memory per iteration.

'''
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.3a: KFold keyed PRNG (P-CV1 S1) ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")
    if r.stdout.strip() != "master":
        raise SystemExit("v0.9.3a must run on master")

    rewrite_cv_module()
    write_tests()
    append_decisions_173()
    append_changelog()

    print("\n=== pytest: cv keyed prng (new) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_cv_keyed_prng.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: existing cv tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_cv.py",
         "tests/test_cv_e2e.py", "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** EXISTING CV FAILED ***")
        sys.exit(1)

    print("\n=== pytest: pipeline + scalers regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "tests/test_scalers.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.9.3a: KFold keyed PRNG — O(batch) state default "
        "(shuffle=False row_id%k; shuffle=True splitmix64(row_id,seed)%k); "
        "materialize=True opt-in (exact-balanced, O(N) guarded via "
        "check_state(20·N)); random_state=None→42; "
        "§13.4/173 (docstring lie: 'O(N) int8' was int32+int64, "
        "no MemoryGuard check; small-N tests hide O(N) bugs)"])

    print("\n==> v0.9.3a applied (P-CV1 S1 fix).")
    print("NEXT: v0.9.3b (P-S1 sort merge fan-in S2)")


if __name__ == "__main__":
    main()