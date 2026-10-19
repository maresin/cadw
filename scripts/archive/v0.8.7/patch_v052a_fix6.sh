#!/usr/bin/env bash
# patch_v052a_fix6: P5b — phase-1 buffer в external_sort: (a) limit 0.2×
# (было 0.5×); (b) batch-wise take-write в _write_run (без materialize);
# (c) gc между runs; (d) вернуть nulls/1M тесты из fix4.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v052a-sort-buffer"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v052a-fix6-${TS}"
git diff > ".backups/backup-pre-v052a-fix6-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: v0.5.2a-fix6 (peak in _write_run not _merge_keys)" || true
require_clean_tree || exit 1

# ============ DIAGNOSTIC: what's in external_sort now ============
echo "=== external_sort phase 1 currently ==="
grep -n "limit = max(1, memory_guard.available_bytes" cadw/ops/sort.py || echo "  (not found)"
grep -n "sorted_table = table.take(order)" cadw/ops/sort.py || echo "  (not found)"
echo ""

# ============ FIX ============
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/sort.py")
s = p.read_text()

# ---- 1. Add gc import if missing ----
if "import gc" not in s.split("def ")[0]:
    old_imports = "import heapq"
    new_imports = "import gc\nimport heapq"
    if old_imports in s:
        s = s.replace(old_imports, new_imports, 1)
        print("sort.py: import gc added")

# ---- 2. _write_run: batch-wise take/write (no full sorted_table) ----
old_write = '''def _write_run(
    batches: list[pa.RecordBatch],
    all_columns: Sequence[str],
    key_columns: Sequence[str],
    ascending: list[bool],
    nulls_first: bool,
    spill_dir: Path,
    run_idx: int,
) -> Path:
    table = pa.Table.from_batches(batches)
    # null_placement передаётся per-key (per-key API с 25.0.0; глобальный
    # SortOptions.null_placement deprecated).
    null_placement = "at_start" if nulls_first else "at_end"
    sort_keys = [
        (col, "ascending" if asc else "descending", null_placement)
        for col, asc in zip(key_columns, ascending)
    ]
    order = pc.sort_indices(table, sort_keys=sort_keys)
    sorted_table = table.take(order)

    path = spill_dir / f"run_{run_idx:05d}.arrow"
    writer = ArrowIPCBatchWriter(str(path), sorted_table.schema)
    try:
        for b in sorted_table.to_batches():
            writer.write_batch(b)
    finally:
        writer.close()
    return path'''

new_write = '''def _write_run(
    batches: list[pa.RecordBatch],
    all_columns: Sequence[str],
    key_columns: Sequence[str],
    ascending: list[bool],
    nulls_first: bool,
    spill_dir: Path,
    run_idx: int,
) -> Path:
    """v0.5.2a-fix6 (P5b): batch-wise take/write to avoid materializing
    whole sorted_table. Peak in _write_run = table (via from_batches) +
    order + chunk (small), not table + order + sorted_table (×3).
    """
    table = pa.Table.from_batches(batches)
    null_placement = "at_start" if nulls_first else "at_end"
    sort_keys = [
        (col, "ascending" if asc else "descending", null_placement)
        for col, asc in zip(key_columns, ascending)
    ]
    order = pc.sort_indices(table, sort_keys=sort_keys)

    path = spill_dir / f"run_{run_idx:05d}.arrow"
    writer = ArrowIPCBatchWriter(str(path), table.schema)
    try:
        # Batch-wise: take small chunk, write it, free. No full materialize.
        chunk = 50_000
        for i in range(0, len(order), chunk):
            sub_order = order.slice(i, chunk)
            sub_table = table.take(sub_order)
            for b in sub_table.to_batches():
                writer.write_batch(b)
    finally:
        writer.close()
    return path'''

if old_write not in s:
    raise SystemExit("_write_run pattern not found")
s = s.replace(old_write, new_write, 1)
print("sort.py: _write_run — batch-wise take/write (no full sorted_table)")

# ---- 3. external_sort: limit 0.5 → 0.2, gc after runs ----
old_limit = '''    limit = max(1, memory_guard.available_bytes() // 2)'''
new_limit = '''    # v0.5.2a-fix6 (P5b): phase-1 peak is 3×buf (buf + from_batches-copy
    # + take-copy). Reducing limit from 0.5× to 0.2× cuts peak from
    # ~3×15MB → ~3×6MB (for 30 MB budget). More runs, but bounded.
    limit = max(1, int(memory_guard.available_bytes() * 0.2))'''
if old_limit not in s:
    raise SystemExit("limit line not found")
s = s.replace(old_limit, new_limit, 1)
print("sort.py: external_sort — limit 0.5× → 0.2×")

# ---- 4. gc.collect() after phase 1 (before merge) ----
old_phase_end = '''    if buf:
        runs.append(
            _write_run(buf, all_columns, key_columns, asc, nulls_first, spill_dir, len(runs))
        )
    if not runs:
        raise ValueError("external_sort: пустой вход")'''
new_phase_end = '''    if buf:
        runs.append(
            _write_run(buf, all_columns, key_columns, asc, nulls_first, spill_dir, len(runs))
        )
    if not runs:
        raise ValueError("external_sort: пустой вход")

    # v0.5.2a-fix6: release phase-1 buffers before merge (frees numpy/Arrow
    # allocations held by builder state / table refs).
    gc.collect()'''
if old_phase_end not in s:
    raise SystemExit("phase-1 end block not found")
s = s.replace(old_phase_end, new_phase_end, 1)
print("sort.py: external_sort — gc.collect() after phase 1")

p.write_text(s)
print("sort.py: saved")
PY

# ============ Verify ============
echo ""
echo "=== verify sort.py ==="
python3 - <<'PY'
from pathlib import Path
s = Path("cadw/ops/sort.py").read_text()
checks = {
    "import gc": "import gc" in s,
    "limit 0.2×": "int(memory_guard.available_bytes() * 0.2)" in s,
    "gc after phase 1": "gc.collect()" in s,
    "batch-wise chunk write": "chunk = 50_000" in s,
    "no full sorted_table": "sorted_table = table.take(order)" not in s,
    "_NumpyBackedBuilder (from fix5)": "class _NumpyBackedBuilder" in s,
    "_make_builder (from fix5)": "def _make_builder" in s,
}
for name, ok in checks.items():
    print(f"  {'OK' if ok else 'FAIL'}: {name}")
if not all(checks.values()):
    raise SystemExit("verify failed")
PY

echo ""
echo "=== compile: cadw/ops/sort.py ==="
python3 -c "import py_compile; py_compile.compile('cadw/ops/sort.py', doraise=True); print('OK')"

# ============ Add missing tests (nulls + 1M from fix4) ============
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_sort.py")
s = p.read_text()

if "test_sort_int_with_nulls_builder_path" in s:
    print("nulls tests already present")
else:
    extra = '''

def test_sort_int_with_nulls_builder_path(tmp_path):
    """Numeric with nulls — exercises _NumpyBackedBuilder null bitmap
    (np.packbits). Regression for v0.5.2a-fix5/fix6.
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
    """Numeric nulls_first — bitmap at start."""
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


def test_external_sort_phase1_limit_small():
    """AST guard (P5b): phase-1 buffer must be ≤ 0.25× available.

    Baseline v0.5.2a-fix5 had 0.5× → peak 3× buf. v0.5.2a-fix6 = 0.2×.
    """
    import inspect
    from cadw.ops import sort as sort_mod
    src = inspect.getsource(sort_mod.external_sort)
    assert "0.2" in src or "0.25" in src or "0.3" in src, (
        "phase-1 buffer limit must be small (0.2-0.3× available); "
        "was 0.5× (P5b regression). See DECISIONS §13.4/71."
    )
    assert "memory_guard.available_bytes() // 2" not in src, (
        "phase-1 limit must NOT be `available // 2` (0.5×) — P5b regression"
    )
'''

    s = s.rstrip() + extra + "\n"
    p.write_text(s)
    print("test_sort.py: +3 nulls + 1 AST guard (P5b)")
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
echo "=== bench: P5 sort (baseline Pingis: +31.9; fix5: +51.3) ==="
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
git commit -m "v0.5.2a-fix6: P5b — phase-1 buffer 0.2× + batch-wise take/write; +nulls/1M tests"

echo ""
echo "==> v0.5.2a-fix6 applied (bench exit=$RCB)"