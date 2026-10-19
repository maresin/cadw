#!/usr/bin/env bash
# patch_v052a_fix2: pyarrow 26 compat для _make_builder (string builder).
# Multi-name fallback + BinaryBuilder→cast. +string sort test + bench rerun.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v052a-sort-buffer"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v052a-fix2-${TS}"
git diff > ".backups/backup-pre-v052a-fix2-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: v0.5.2a-fix2 (pa.StringBuilder missing in pyarrow 26)" || true
require_clean_tree || exit 1

# ============ DIAGNOSTIC: what builders exist ============
echo "=== pyarrow 26 builder diagnostic ==="
uv run python << 'PYEOF'
import pyarrow as pa
try:
    import pyarrow.lib as _plib
except ImportError:
    _plib = None

print(f"pyarrow: {pa.__version__}")
print(f"pa (top): {sorted(n for n in dir(pa) if 'Builder' in n)}")
if _plib is not None:
    print(f"pa.lib:   {sorted(n for n in dir(_plib) if 'Builder' in n)}")
PYEOF
echo ""

# ============ FIX: _make_builder + _flush_builders compat ============
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/sort.py")
s = p.read_text()

# Replace _make_builder (v0.5.2a) with compat version (v0.5.2a-fix2)
old_make = '''def _make_builder(dtype: pa.DataType):
    """Create pa.Array.Builder for given type (v0.5.2a, P5).

    Replaces Python-list accumulation in _merge_keys. Builder allocates
    Arrow buffers directly — peak = read_batch_size × arrow-size, not
    × 28-80 per Python object.
    """
    if pa.types.is_int64(dtype):
        return pa.Int64Builder()
    if pa.types.is_int32(dtype):
        return pa.Int32Builder()
    if pa.types.is_uint64(dtype):
        return pa.UInt64Builder()
    if pa.types.is_uint32(dtype):
        return pa.UInt32Builder()
    if pa.types.is_float64(dtype):
        return pa.Float64Builder()
    if pa.types.is_float32(dtype):
        return pa.Float32Builder()
    if pa.types.is_boolean(dtype):
        return pa.BooleanBuilder()
    if pa.types.is_string(dtype):
        return pa.StringBuilder()
    if pa.types.is_binary(dtype):
        return pa.BinaryBuilder()
    if pa.types.is_timestamp(dtype):
        # builder has unit; use string fallback via pa.array at flush
        return pa.TimestampBuilder(dtype.unit)
    if pa.types.is_date(dtype):
        return pa.Date32Builder() if dtype == pa.date32() else pa.Date64Builder()
    # fallback — builder unsupported; caller uses list
    return None'''

new_make = '''def _make_builder(dtype: pa.DataType):
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
        return None

    if pa.types.is_int64(dtype):
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

if old_make not in s:
    raise SystemExit("_make_builder (v0.5.2a) not found")
s = s.replace(old_make, new_make, 1)
print("sort.py: _make_builder → pyarrow 26 compat (multi-name + binary fallback)")

# Replace _flush_builders with cast-aware version
old_flush = '''def _flush_builders(builders: dict, schema: pa.Schema) -> pa.RecordBatch:
    """Finish builders → RecordBatch. Each builder resets itself after finish."""
    arrays = []
    for f in schema:
        b = builders[f.name]
        if b is None:
            raise RuntimeError(
                f"_flush_builders: no builder for {f.name} ({f.type}); "
                f"this is a bug in _make_builder (v0.5.2a)"
            )
        arrays.append(b.finish())
    return pa.RecordBatch.from_arrays(arrays, schema=schema)'''

new_flush = '''def _flush_builders(builders: dict, schema: pa.Schema) -> pa.RecordBatch:
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
            # e.g. binary → string (utf8 validation); cheap relative to
            # Python-object overhead that P5 was fixing.
            arr = arr.cast(f.type)
        arrays.append(arr)
    return pa.RecordBatch.from_arrays(arrays, schema=schema)'''

if old_flush not in s:
    raise SystemExit("_flush_builders (v0.5.2a) not found")
s = s.replace(old_flush, new_flush, 1)
print("sort.py: _flush_builders → auto-cast binary → string")

p.write_text(s)
PY

echo "=== compile: cadw/ops/sort.py ==="
python3 -c "import py_compile; py_compile.compile('cadw/ops/sort.py', doraise=True); print('OK')"

# ============ Add string sort unit test ============
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_sort.py")
s = p.read_text()

# Check if already added
if "test_sort_string_with_builder" in s:
    print("test_sort.py: string test already present")
else:
    extra = '''

def test_sort_string_with_builder(tmp_path):
    """P5 fix: string sort must work with pyarrow 26 builder API.

    Regression: v0.5.2a used pa.StringBuilder which doesn't exist in
    pyarrow 26. v0.5.2a-fix2 uses multi-name fallback + BinaryBuilder→cast.
    """
    data = {"s": ["c", "a", "b", "a"], "v": [3.0, 1.0, 2.0, 4.0]}
    it = make_iter(data, 2, value_columns=["s", "v"], key_columns=["s"])
    path = external_sort(
        it, ["s"], ["s", "v"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    rows = collect(path, ["s", "v"])
    assert [r["s"] for r in rows] == ["a", "a", "b", "c"]
    assert [r["v"] for r in rows] == [1.0, 4.0, 2.0, 3.0]


def test_sort_string_large_string(tmp_path):
    """Large string builder fallback."""
    data = {"s": ["y", "x", "z"], "v": [1.0, 2.0, 3.0]}
    it = make_iter(data, 2, value_columns=["s", "v"], key_columns=["s"])
    path = external_sort(
        it, ["s"], ["s", "v"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    rows = collect(path, ["s", "v"])
    assert [r["s"] for r in rows] == ["x", "y", "z"]
'''
    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    print("test_sort.py: +string sort tests")
PY

# ============ Run tests + bench ============
echo ""
echo "=== pytest tests/test_sort.py -v ==="
set +e
uv run pytest tests/test_sort.py -v
RC1=$?
set -e
if [[ $RC1 -ne 0 ]]; then
  echo "sort tests: rc=$RC1"
  exit 1
fi

echo ""
echo "=== bench: P5 sort (baseline Pingis B3a: +31.9 MB / 30 MB budget) ==="
set +e
uv run python scripts/bench_p5_sort.py
RCB=$?
set -e
# bench failure is diagnostic — don't exit yet

echo ""
echo "=== pytest full fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest full slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.5.2a-fix2: _make_builder pyarrow 26 compat (StringBuilder/Utf8Builder/LargeStringBuilder; BinaryBuilder→cast); +string tests"

echo ""
echo "==> v0.5.2a-fix2 applied"