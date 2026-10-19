#!/usr/bin/env bash
# patch_v052a_fix4: (a) fix _NumpyBackedBuilder.finish() — np.packbits
# для null bitmap; (b) skip _flush_builders (fix2 версия корректна);
# (c) run tests + bench.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v052a-sort-buffer"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v052a-fix4-${TS}"
git diff > ".backups/backup-pre-v052a-fix4-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: v0.5.2a-fix4 (packbits null bitmap; fix3 partial)" || true
require_clean_tree || exit 1

# ============ DIAGNOSTIC: current state ============
echo "=== current state of sort.py ==="
grep -c "_NumpyBackedBuilder" cadw/ops/sort.py && echo "  → _NumpyBackedBuilder: present"
grep -c "def _make_builder" cadw/ops/sort.py && echo "  → _make_builder: present"
grep -c "def _flush_builders" cadw/ops/sort.py && echo "  → _flush_builders: present"
grep -c "np.packbits" cadw/ops/sort.py || echo "  → np.packbits: MISSING (fix needed)"
grep -n "mask.astype(np.uint8)" cadw/ops/sort.py && echo "  → mask.astype: needs fix (use packbits)"
echo ""

# ============ FIX: _NumpyBackedBuilder.finish() — packbits ============
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/sort.py")
s = p.read_text()

# Ensure _make_builder (fix3) and _NumpyBackedBuilder are present
if "_NumpyBackedBuilder" not in s:
    raise SystemExit(
        "_NumpyBackedBuilder не найден — fix3 не применился к _make_builder. "
        "Проверьте состояние sort.py."
    )
if "def _make_builder" not in s:
    raise SystemExit("_make_builder не найден")

# Replace _NumpyBackedBuilder.finish() — use np.packbits
old_finish = '''    def finish(self) -> pa.Array:
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
        return arr'''

new_finish = '''    def finish(self) -> pa.Array:
        values = self._buf[:self._n].copy()
        mask = self._nulls[:self._n]
        if mask.any():
            # Arrow null bitmap: bit-packed, LSB-first, bit=1 → null.
            # `np.packbits` gives exactly this layout.
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
        # Reset for reuse
        self._capacity = 64
        self._buf = np.empty(self._capacity, dtype=self._dtype)
        self._nulls = np.ones(self._capacity, dtype=bool)
        self._n = 0
        return arr'''

if old_finish not in s:
    # Check if already fixed
    if "np.packbits" in s:
        print("sort.py: _NumpyBackedBuilder.finish already uses np.packbits (no-op)")
    else:
        raise SystemExit(
            "_NumpyBackedBuilder.finish() pattern not found; sort.py в "
            "неожиданном состоянии. Нужна ручная проверка."
        )
else:
    s = s.replace(old_finish, new_finish, 1)
    print("sort.py: _NumpyBackedBuilder.finish — np.packbits (null bitmap bit-packed)")

# _flush_builders: НЕ трогаем (fix2 версия корректна — auto-cast уже есть)
if "_flush_builders" not in s:
    raise SystemExit("_flush_builders отсутствует — fix2 не применился")
print("sort.py: _flush_builders оставлен как есть (fix2 версия корректна)")

p.write_text(s)
PY

echo ""
echo "=== compile: cadw/ops/sort.py ==="
python3 -c "import py_compile; py_compile.compile('cadw/ops/sort.py', doraise=True); print('OK')"

# ============ Add unit test with null numeric (regression for packbits) ============
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_sort.py")
s = p.read_text()

if "test_sort_int_with_nulls_builder_path" in s:
    print("test_sort.py: int+nulls test already present")
else:
    extra = '''

def test_sort_int_with_nulls_builder_path(tmp_path):
    """Numeric with nulls — exercises _NumpyBackedBuilder null bitmap
    (np.packbits). Regression for v0.5.2a-fix4.
    """
    data = {
        "v": pa.array([3, None, 1, None, 2], type=pa.int64()),
        "id": list(range(5)),
    }
    it = make_iter(data, 2, value_columns=["v"], key_columns=["id"])
    path = external_sort(
        it, ["v"], ["v"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    rows = collect(path, ["v"])
    assert [r["v"] for r in rows] == [1, 2, 3, None, None]


def test_sort_float_with_nulls_builder_path(tmp_path):
    """Float64 with nulls — np.packbits null bitmap."""
    data = {
        "v": pa.array([3.5, None, 1.5, None, 2.5], type=pa.float64()),
        "id": list(range(5)),
    }
    it = make_iter(data, 2, value_columns=["v"], key_columns=["id"])
    path = external_sort(
        it, ["v"], ["v"], make_guard(), tmp_path,
        ascending=True, nulls_first=False, log=False,
    )
    rows = collect(path, ["v"])
    assert [r["v"] for r in rows] == [1.5, 2.5, 3.5, None, None]


def test_sort_int_nulls_first_builder_path(tmp_path):
    """Numeric nulls_first — bitmap is at start."""
    data = {
        "v": pa.array([3, None, 1, None, 2], type=pa.int64()),
        "id": list(range(5)),
    }
    it = make_iter(data, 2, value_columns=["v"], key_columns=["id"])
    path = external_sort(
        it, ["v"], ["v"], make_guard(), tmp_path,
        ascending=True, nulls_first=True, log=False,
    )
    rows = collect(path, ["v"])
    assert [r["v"] for r in rows] == [None, None, 1, 2, 3]
'''
    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    print("test_sort.py: +3 nulls builder tests")
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
git commit -m "v0.5.2a-fix4: _NumpyBackedBuilder.finish np.packbits (null bitmap); +3 numeric-nulls tests"

echo ""
echo "==> v0.5.2a-fix4 applied (bench exit=$RCB)"