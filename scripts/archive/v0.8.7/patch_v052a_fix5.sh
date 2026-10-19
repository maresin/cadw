#!/usr/bin/env bash
# patch_v052a_fix5: добавить _NumpyBackedBuilder (с packbits сразу);
# заменить _make_builder на numpy-fallback версию.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v052a-sort-buffer"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v052a-fix5-${TS}"
git diff > ".backups/backup-pre-v052a-fix5-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: v0.5.2a-fix5 (_NumpyBackedBuilder missing; fix3 lost at write)" || true
require_clean_tree || exit 1

# ============ FIX ============
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/sort.py")
s = p.read_text()

# Sanity
assert "_NumpyBackedBuilder" not in s, "unexpected: _NumpyBackedBuilder already present"
assert "def _make_builder" in s, "_make_builder not found"
assert "def _flush_builders" in s, "_flush_builders not found"

# ---- Step 1: Replace _make_builder head (signature + _try setup) with _NumpyBackedBuilder + new head ----
old_head = '''def _make_builder(dtype: pa.DataType):
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

new_head = '''class _NumpyBackedBuilder:
    """v0.5.2a-fix5: minimal fixed-width builder backed by np.ndarray.

    PyArrow 26 не экспортирует `Float64Builder`/`Int64Builder`/etc.
    (проверено: hasattr(pa, 'Float64Builder') == False). Только
    `StringBuilder` есть в `pa.lib`.

    Peak = capacity × arrow_size (без Python-object overhead: int/float
    в PyArrow = 4-8 B, не 28). Amortized O(n) через geometric grow.

    Null bitmap: bit-packed (np.packbits), LSB-first — совместимо с
    pa.Array.from_buffers.
    """

    __slots__ = ("_dtype", "_arrow_type", "_buf", "_nulls", "_n", "_capacity")

    def __init__(self, arrow_type: pa.DataType, np_dtype):
        import numpy as np
        self._dtype = np_dtype
        self._arrow_type = arrow_type
        self._capacity = 64
        self._buf = np.empty(self._capacity, dtype=np_dtype)
        self._nulls = np.ones(self._capacity, dtype=bool)  # True = null
        self._n = 0

    def _grow(self):
        import numpy as np
        newcap = self._capacity * 2
        new_buf = np.empty(newcap, dtype=self._dtype)
        new_buf[: self._n] = self._buf[: self._n]
        new_nulls = np.ones(newcap, dtype=bool)
        new_nulls[: self._n] = self._nulls[: self._n]
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
        self._buf[self._n] = 0
        self._nulls[self._n] = True
        self._n += 1

    def finish(self) -> pa.Array:
        import numpy as np
        values = self._buf[: self._n].copy()
        mask = self._nulls[: self._n]
        if mask.any():
            packed = np.packbits(mask, bitorder="little")
            arr = pa.Array.from_buffers(
                self._arrow_type, self._n,
                [pa.py_buffer(packed), pa.py_buffer(values)],
            )
        else:
            arr = pa.Array.from_buffers(
                self._arrow_type, self._n,
                [None, pa.py_buffer(values)],
            )
        # Reset
        self._capacity = 64
        self._buf = np.empty(self._capacity, dtype=self._dtype)
        self._nulls = np.ones(self._capacity, dtype=bool)
        self._n = 0
        return arr


def _make_builder(dtype: pa.DataType):
    """Create pa.Array.Builder or numpy fallback (v0.5.2a-fix5).

    PyArrow 26 не экспортирует numeric `*Builder` (hasattr(pa,
    'Float64Builder') == False). Fallback chain:
      1. Native pyarrow *Builder (pa / pa.lib / pa.compute) — для
         string/binary/timestamp/date (если есть).
      2. `_NumpyBackedBuilder` — для fixed-width numeric
         (int/uint/float).
      3. BinaryBuilder для string (cast at flush).

    Returns builder (native or numpy-backed) or None.
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

if old_head not in s:
    raise SystemExit("_make_builder head (fix2) не найден — состояние не совпадает")
s = s.replace(old_head, new_head, 1)
print("sort.py: _NumpyBackedBuilder добавлен; _make_builder head → fix5")

# ---- Step 2: Replace _make_builder body (dispatch by type) ----
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
        return _NumpyBackedBuilder(dtype, np.int64)
    if pa.types.is_date32(dtype):
        return _try("Date32Builder") or _NumpyBackedBuilder(dtype, np.int32)
    if pa.types.is_date64(dtype):
        return _try("Date64Builder") or _NumpyBackedBuilder(dtype, np.int64)
    return None'''

if old_body not in s:
    raise SystemExit("_make_builder body (fix2) не найден")
s = s.replace(old_body, new_body, 1)
print("sort.py: _make_builder body → numpy fallback")

p.write_text(s)
print("sort.py: saved")
PY

# ============ Verify: all markers present ============
echo ""
echo "=== verify sort.py ==="
python3 - <<'PY'
from pathlib import Path
s = Path("cadw/ops/sort.py").read_text()
checks = {
    "_NumpyBackedBuilder class": "class _NumpyBackedBuilder" in s,
    "_make_builder": "def _make_builder" in s,
    "_flush_builders": "def _flush_builders" in s,
    "np.packbits in finish": "np.packbits" in s,
    "_NumpyBackedBuilder float64 branch": "_NumpyBackedBuilder(dtype, np.float64)" in s,
}
for name, ok in checks.items():
    print(f"  {'OK' if ok else 'FAIL'}: {name}")
if not all(checks.values()):
    raise SystemExit("verify failed")
PY

echo ""
echo "=== compile: cadw/ops/sort.py ==="
python3 -c "import py_compile; py_compile.compile('cadw/ops/sort.py', doraise=True); print('OK')"

# ============ Run tests ============
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
uv run python scripts/bench_p5_sort.py 2>&1
RCB=$?
set -e
echo "(bench exit=$RCB)"

echo ""
echo "=== pytest full fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest full slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.5.2a-fix5: _NumpyBackedBuilder + _make_builder numpy fallback (pyarrow 26 no numeric *Builder)"

echo ""
echo "==> v0.5.2a-fix5 applied (bench exit=$RCB)"