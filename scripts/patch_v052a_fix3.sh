#!/usr/bin/env bash
# patch_v052a_fix3: _make_builder — расширить _try sources + manual fallback
# для fixed-width; диагностика pa.*__file__ и lazy getattr.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v052a-sort-buffer"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v052a-fix3-${TS}"
git diff > ".backups/backup-pre-v052a-fix3-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: v0.5.2a-fix3 (Float64Builder missing in bench context)" || true
require_clean_tree || exit 1

# ============ DIAGNOSTIC ============
echo "=== pyarrow builder diagnostic (deep) ==="
uv run python << 'PYEOF'
import sys
import pyarrow as pa
import pyarrow.compute as pc
try:
    import pyarrow.lib as plib
except ImportError:
    plib = None

# 1. cadw source
import cadw
import cadw.ops.sort as sort_mod
print(f"cadw: {cadw.__file__}")
print(f"sort: {sort_mod.__file__}")

# 2. dir + getattr for Float64Builder
print(f"\npa top dir builders: {[n for n in dir(pa) if 'Builder' in n]}")
if plib:
    print(f"pa.lib dir builders: {[n for n in dir(plib) if 'Builder' in n]}")
print(f"pc dir builders: {[n for n in dir(pc) if 'Builder' in n]}")

print(f"\nhasattr(pa, 'Float64Builder'): {hasattr(pa, 'Float64Builder')}")
print(f"getattr(pa, 'Float64Builder', None): {getattr(pa, 'Float64Builder', None)}")
print(f"hasattr(pa, 'Int64Builder'): {hasattr(pa, 'Int64Builder')}")
print(f"getattr(pa, 'Int64Builder', None): {getattr(pa, 'Int64Builder', None)}")
print(f"hasattr(plib, 'Float64Builder'): {hasattr(plib, 'Float64Builder') if plib else 'N/A'}")
print(f"hasattr(plib, 'StringBuilder'): {hasattr(plib, 'StringBuilder') if plib else 'N/A'}")

# 3. types check
print(f"\npa.types.is_float64(pa.float64()): {pa.types.is_float64(pa.float64())}")
print(f"pa.types.is_float64(pa.float32()): {pa.types.is_float64(pa.float32())}")

# 4. try import from _compute
try:
    import pyarrow._compute as pcomp
    print(f"_compute dir builders: {[n for n in dir(pcomp) if 'Builder' in n]}")
except ImportError:
    print("_compute: no import")

# 5. what does dir(pa) actually show
print(f"\nfull dir(pa) count: {len(dir(pa))}")
print(f"sample names: {dir(pa)[:20]}")
PYEOF
echo ""

# ============ FIX: manual fallback for fixed-width ============
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/sort.py")
s = p.read_text()

# Add manual builder class for fixed-width + extend _try sources
old_try = '''def _make_builder(dtype: pa.DataType):
    """Create pa.Array.Builder for given type (v0.5.2a-fix2).

    pyarrow 26 compat: `StringBuilder` may be at `pa.lib`, renamed to
    `Utf8Builder`, or absent. Fallback chain:
      1. `StringBuilder` / `Utf8Builder` / `LargeStringBuilder` in pa / pa.lib
      2. `BinaryBuilder` (binary → cast to string at flush)

    Returns builder (or None if no builder available — caller raises).
    `_flush_builders` auto-casts if builder output type != target type
    (e.g. binary → string).
    """
    try:
        import pyarrow.lib as _plib
    except ImportError:
        _plib = None

    def _try(*names):
        for name in names:
            for mod in (pa, _plib):
                if mod is None:
                    continue
                cls = getattr(mod, name, None)
                if cls is not None:
                    return cls()
        return None'''

new_try = '''class _NumpyBackedBuilder:
    """v0.5.2a-fix3: minimal fixed-width builder backed by np.ndarray.

    Для числовых типов (int/uint/float/timestamp/date), PyArrow 26 может
    не экспортировать *Builder классы. Этот fallback использует
    numpy-массив фиксированного размера — peak = capacity × arrow_size
    (не ×28-80 на Python-object). Совместим с интерфейсом `_flush_builders`:
    методы append/append_null/finish.

    Buffer grows geometric ×2 — amortized O(n) append.
    """

    __slots__ = ("_dtype", "_arrow_type", "_buf", "_nulls", "_n", "_capacity")

    def __init__(self, arrow_type: pa.DataType, np_dtype):
        self._dtype = np_dtype
        self._arrow_type = arrow_type
        self._capacity = 64
        self._buf = np.empty(self._capacity, dtype=np_dtype)
        self._nulls = np.ones(self._capacity, dtype=bool)  # True = null
        self._n = 0

    def _grow(self):
        newcap = self._capacity * 2
        new_buf = np.empty(newcap, dtype=self._dtype)
        new_buf[:self._n] = self._buf[:self._n]
        new_nulls = np.ones(newcap, dtype=bool)
        new_nulls[:self._n] = self._nulls[:self._n]
        self._buf = new_buf
        self._nulls = new_nulls
        self._capacity = newcap

    def append(self, value):
        if self._n == self._capacity:
            self._grow()
        self._buf[self._n] = value
        self._nulls[self._n] = False
        self._n += 1

    def append_null(self):
        if self._n == self._capacity:
            self._grow()
        self._buf[self._n] = 0  # placeholder
        self._nulls[self._n] = True
        self._n += 1

    def finish(self) -> pa.Array:
        values = self._buf[:self._n].copy()
        mask = self._nulls[:self._n]
        if mask.any():
            arr = pa.Array.from_buffers(
                self._arrow_type, self._n,
                [pa.py_buffer(mask.astype(np.uint8)),
                 pa.py_buffer(values)],
            )
        else:
            arr = pa.Array.from_buffers(
                self._arrow_type, self._n,
                [None, pa.py_buffer(values)],
            )
        # Reset for reuse
        self._capacity = 64
        self._buf = np.empty(self._capacity, dtype=self._dtype)
        self._nulls = np.ones(self._capacity, dtype=bool)
        self._n = 0
        return arr


def _make_builder(dtype: pa.DataType):
    """Create pa.Array.Builder (v0.5.2a-fix3).

    Fallback chain:
      1. Native pyarrow *Builder (pa / pa.lib / pa.compute).
      2. `_NumpyBackedBuilder` for fixed-width numeric (int/uint/float/
         timestamp/date).
      3. BinaryBuilder for string (cast at flush).

    Returns builder or None.
    """
    import numpy as np
    try:
        import pyarrow.lib as _plib
    except ImportError:
        _plib = None
    try:
        import pyarrow.compute as _pcomp
    except ImportError:
        _pcomp = None

    _mods = [pa, _plib, _pcomp]

    def _try(*names):
        for name in names:
            for mod in _mods:
                if mod is None:
                    continue
                cls = getattr(mod, name, None)
                if cls is not None:
                    try:
                        return cls()
                    except Exception:
                        continue
        return None'''

if old_try not in s:
    raise SystemExit("_make_builder head not found")
s = s.replace(old_try, new_try, 1)

# Replace the individual builder branches with numpy fallback
old_body = '''    if pa.types.is_int64(dtype):
        return _try("Int64Builder")
    if pa.types.is_int32(dtype):
        return _try("Int32Builder")
    if pa.types.is_uint64(dtype):
        return _try("UInt64Builder")
    if pa.types.is_uint32(dtype):
        return _try("UInt32Builder")
    if pa.types.is_float64(dtype):
        return _try("Float64Builder")
    if pa.types.is_float32(dtype):
        return _try("Float32Builder")
    if pa.types.is_boolean(dtype):
        return _try("BooleanBuilder")

    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        # Try dedicated string builders first
        b = _try("StringBuilder", "Utf8Builder",
                 "LargeStringBuilder", "LargeUtf8Builder")
        if b is not None:
            return b
        # Fallback: binary builder (cast at flush)
        return _try("BinaryBuilder", "LargeBinaryBuilder")

    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return _try("BinaryBuilder", "LargeBinaryBuilder")

    if pa.types.is_timestamp(dtype):
        for mod in (pa, _plib):
            if mod is None:
                continue
            cls = getattr(mod, "TimestampBuilder", None)
            if cls is not None:
                return cls(dtype.unit)
        return None
    if pa.types.is_date32(dtype):
        return _try("Date32Builder")
    if pa.types.is_date64(dtype):
        return _try("Date64Builder")
    return None'''

new_body = '''    # Numeric — native builder if available; else numpy-backed.
    if pa.types.is_int64(dtype):
        return _try("Int64Builder") or _NumpyBackedBuilder(dtype, np.int64)
    if pa.types.is_int32(dtype):
        return _try("Int32Builder") or _NumpyBackedBuilder(dtype, np.int32)
    if pa.types.is_uint64(dtype):
        return _try("UInt64Builder") or _NumpyBackedBuilder(dtype, np.uint64)
    if pa.types.is_uint32(dtype):
        return _try("UInt32Builder") or _NumpyBackedBuilder(dtype, np.uint32)
    if pa.types.is_float64(dtype):
        return _try("Float64Builder") or _NumpyBackedBuilder(dtype, np.float64)
    if pa.types.is_float32(dtype):
        return _try("Float32Builder") or _NumpyBackedBuilder(dtype, np.float32)
    if pa.types.is_boolean(dtype):
        # Boolean: native or binary fallback
        return _try("BooleanBuilder")

    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        b = _try("StringBuilder", "Utf8Builder",
                 "LargeStringBuilder", "LargeUtf8Builder")
        if b is not None:
            return b
        return _try("BinaryBuilder", "LargeBinaryBuilder")

    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return _try("BinaryBuilder", "LargeBinaryBuilder")

    if pa.types.is_timestamp(dtype):
        for mod in _mods:
            if mod is None:
                continue
            cls = getattr(mod, "TimestampBuilder", None)
            if cls is not None:
                return cls(dtype.unit)
        # Fallback: int64 numpy
        return _NumpyBackedBuilder(dtype, np.int64)
    if pa.types.is_date32(dtype):
        return _try("Date32Builder") or _NumpyBackedBuilder(dtype, np.int32)
    if pa.types.is_date64(dtype):
        return _try("Date64Builder") or _NumpyBackedBuilder(dtype, np.int64)
    return None'''

if old_body not in s:
    raise SystemExit("_make_builder body not found")
s = s.replace(old_body, new_body, 1)
print("sort.py: _make_builder — numpy fallback for fixed-width + extended _try mods")

# _flush_builders: handle _NumpyBackedBuilder's finish() which returns array directly
old_flush = '''def _flush_builders(builders: dict, schema: pa.Schema) -> pa.RecordBatch:
    """Finish builders → RecordBatch (v0.5.2a-fix2).

    Auto-casts if builder output type differs from schema target
    (e.g. BinaryBuilder used for string → cast to string).
    """
    arrays = []
    for f in schema:
        b = builders[f.name]
        if b is None:
            raise RuntimeError(
                f"_flush_builders: no builder for {f.name} ({f.type}); "
                f"bug in _make_builder (v0.5.2a-fix2)"
            )
        arr = b.finish()
        if arr.type != f.type:
            arr = arr.cast(f.type)
        arrays.append(arr)
    return pa.RecordBatch.from_arrays(arrays, schema=schema)'''

new_flush = '''def _flush_builders(builders: dict, schema: pa.Schema) -> pa.RecordBatch:
    """Finish builders → RecordBatch (v0.5.2a-fix3).

    Auto-casts if builder output type differs (e.g. binary → string,
    timestamp with different unit).
    """
    arrays = []
    for f in schema:
        b = builders[f.name]
        if b is None:
            raise RuntimeError(
                f"_flush_builders: no builder for {f.name} ({f.type}); "
                f"bug in _make_builder (v0.5.2a-fix3)"
            )
        arr = b.finish()
        if arr.type != f.type:
            arr = arr.cast(f.type)
        arrays.append(arr)
    return pa.RecordBatch.from_arrays(arrays, schema=schema)'''

if old_flush not in s:
    raise SystemExit("_flush_builders not found")
s = s.replace(old_flush, new_flush, 1)
print("sort.py: _flush_builders — fix3 docstring")

p.write_text(s)
PY

echo "=== compile: cadw/ops/sort.py ==="
python3 -c "import py_compile; py_compile.compile('cadw/ops/sort.py', doraise=True); print('OK')"

# ============ Add unit test 1M float64 (mirrors bench) ============
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_sort.py")
s = p.read_text()

if "test_sort_1m_float64_bench_repr" in s:
    print("test_sort.py: 1M test already present")
else:
    extra = '''

def test_sort_1m_float64_bench_repr(tmp_path):
    """Reproduces bench_p5_sort scenario in unit context: 1M rows,
    float64 + string, budget 30 MB. If bench fails here too, it's
    _make_builder issue; if it passes, bench is env issue.

    Not @slow — this is a small correctness test with 1M rows via
    chunked iteration; total ~2-3 s.
    """
    import numpy as np
    n = 1_000_000
    s_arr = pa.array([f"{(i*7919)%n:07d}" for i in range(n)], type=pa.string())
    v_arr = pa.array(np.arange(n, dtype=np.float64), type=pa.float64())
    table = pa.table({"s": s_arr, "v": v_arr})

    def make_iter_1m(table_, bs):
        for i in range(0, table_.num_rows, bs):
            yield table_.slice(i, bs).to_batches()[0]

    path = external_sort(
        make_iter_1m(table, 50_000), ["s"], ["s", "v"],
        MemoryGuard(max_memory_mb=30), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    # Verify at least first/last sorted
    first_batch = next(iter(read_sorted(path, ["s"])))
    assert first_batch.num_rows > 0
    first_s = first_batch.column("s")[0].as_py()
    assert first_s == "0000000"  # smallest of (i*7919)%1M
'''
    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    print("test_sort.py: +1M float64 test (mirrors bench)")
PY

# ============ Run ============
echo ""
echo "=== pytest tests/test_sort.py -v ==="
set +e
uv run pytest tests/test_sort.py -v
RC1=$?
set -e
if [[ $RC1 -ne 0 ]]; then
  echo "sort: rc=$RC1"
  exit 1
fi

echo ""
echo "=== bench: P5 sort (baseline: +31.9 MB / 30 MB) ==="
set +e
uv run python scripts/bench_p5_sort.py
RCB=$?
set -e

echo ""
echo "=== pytest full fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest full slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.5.2a-fix3: _make_builder — numpy fallback for fixed-width + extended _try (pa/pa.lib/pa.compute); +1M test"

echo ""
echo "==> v0.5.2a-fix3 applied"
echo "    Bench exit code: $RCB (0=OK)"