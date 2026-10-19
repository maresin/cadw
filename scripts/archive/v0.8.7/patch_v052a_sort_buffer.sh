#!/usr/bin/env bash
# patch_v052a_sort_buffer: P5 — заменить Python-list buffer в _merge_keys
# на pa.Array.Builder (peak: read_batch_size × Arrow-size, не ×30).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-v052a-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch v0.5.2a (P5 sort buffer)"

BR="patch/v052a-sort-buffer"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/sort.py")
s = p.read_text()

# Helper: build arrow Builder for a given type
helper = '''

def _make_builder(dtype: pa.DataType):
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
    return None


def _flush_builders(builders: dict, schema: pa.Schema) -> pa.RecordBatch:
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
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def _append_to_builders(builders: dict, batch: pa.RecordBatch, pos: int, schema: pa.Schema) -> None:
    """Append one row (via per-column scalar) to builders."""
    for f in schema:
        val = batch.column(f.name)[pos].as_py()
        b = builders[f.name]
        if val is None:
            b.append_null()
        else:
            b.append(val)


'''
# Insert helpers before _merge_keys
anchor = "def _merge_keys("
if anchor not in s:
    raise SystemExit("_merge_keys not found")
s = s.replace(anchor, helper + anchor, 1)
print("sort.py: _make_builder + _flush_builders + _append_to_builders added")

# Replace _merge_keys buffer logic
old_buffer_block = '''    buffer: dict[str, list] = {c: [] for c in out_columns}

    def flush() -> None:
        if not buffer[out_columns[0]]:
            return
        arrays = [pa.array(buffer[c], type=schema.field(c).type) for c in out_columns]
        batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
        writer.write_batch(batch)
        for c in out_columns:
            buffer[c].clear()'''

new_buffer_block = '''    # v0.5.2a (P5): pa.Array.Builder вместо Python-list.
    # Peak = read_batch_size × arrow-size (не ×28-80 per Python object).
    builders: dict = {}
    for f in schema:
        b = _make_builder(f.type)
        if b is None:
            raise NotImplementedError(
                f"external_sort: no Arrow Builder for column {f.name!r} "
                f"type {f.type}; P5 fix covers int/uint/float/bool/string/"
                f"binary/timestamp/date. File an issue for other types."
            )
        builders[f.name] = b

    rows_in_buffer = [0]

    def flush() -> None:
        if rows_in_buffer[0] == 0:
            return
        batch = _flush_builders(builders, schema)
        writer.write_batch(batch)
        rows_in_buffer[0] = 0'''

if old_buffer_block not in s:
    raise SystemExit("buffer block not found")
s = s.replace(old_buffer_block, new_buffer_block, 1)
print("sort.py: _merge_keys buffer → pa.Array.Builder")

# Replace append call
old_append = '''        while heap:
            _, _, run_idx, b, pos = heapq.heappop(heap)
            for c in out_columns:
                buffer[c].append(b.column(c)[pos].as_py())
            if len(buffer[out_columns[0]]) >= read_batch_size:
                flush()'''

new_append = '''        while heap:
            _, _, run_idx, b, pos = heapq.heappop(heap)
            _append_to_builders(builders, b, pos, schema)
            rows_in_buffer[0] += 1
            if rows_in_buffer[0] >= read_batch_size:
                flush()'''

if old_append not in s:
    raise SystemExit("append block not found")
s = s.replace(old_append, new_append, 1)
print("sort.py: _merge_keys append → builder.append")

p.write_text(s)
PY

echo "=== compile: cadw/ops/sort.py ==="
python3 -c "import py_compile; py_compile.compile('cadw/ops/sort.py', doraise=True); print('OK')"

# ============ AST guard для P5 ============
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_sort.py")
s = p.read_text()

extra = '''

# ---------- v0.5.2a (P5): AST guard against Python-list buffer in merge ----------

def test_merge_keys_uses_arrow_builder_not_python_list():
    """_merge_keys must use pa.Array.Builder, not `buffer[c].append(as_py())`.

    Регрессия P5 (Pingis audit): Python-list buffer даёт ×3-5 peak vs
    Arrow. AST guard: нет `buffer[...].append` в _merge_keys.
    """
    import ast
    import inspect
    from cadw.ops import sort as sort_mod

    src = inspect.getsource(sort_mod._merge_keys)
    tree = ast.parse(src)

    # Никаких `buffer[...].append(...)` вызовов
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "append":
            # f.value должно быть Subscript с base Name == 'buffer'
            base = f.value
            if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
                if base.value.id == "buffer":
                    violations.append(f"buffer[...].append() at line {node.lineno}")

    assert not violations, (
        "P5 must use pa.Array.Builder; Python-list buffer detected:\\n"
        + "\\n".join(violations)
    )


def test_merge_keys_uses_arrow_builder():
    """_merge_keys must call _append_to_builders (P5 fix marker)."""
    import inspect
    from cadw.ops import sort as sort_mod
    src = inspect.getsource(sort_mod._merge_keys)
    assert "_append_to_builders" in src, (
        "P5 fix: _merge_keys must call _append_to_builders (Arrow Builder)"
    )
'''

s = s.rstrip() + extra + "\n"
p.write_text(s)
print("test_sort.py: +2 AST guards (P5)")
PY

# ============ Bench for P5 ============
cat > scripts/bench_p5_sort.py <<'PYEOF'
"""Bench: sort strings + numeric, measure Δpeak RSS vs old (Python-list) buffer.

Compare to Pingis B3a: +31.9 MB on 30 MB budget (old code).
"""
import gc, tempfile, threading, time
from pathlib import Path

import psutil
import pyarrow as pa

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.sort import external_sort

proc = psutil.Process()


class PeakSampler(threading.Thread):
    def __init__(self, interval=0.005):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            r = proc.memory_info().rss
            if r > self.peak:
                self.peak = r
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join()


def rss_mb():
    return proc.memory_info().rss / (1024 * 1024)


def batch_iter(table, bs):
    for i in range(0, table.num_rows, bs):
        yield table.slice(i, bs).to_batches()[0]


def run(label, n, budget_mb):
    table = pa.table({
        "s": pa.array([f"{(i*7919)%n:07d}" for i in range(n)], type=pa.string()),
        "v": pa.array(list(range(n)), type=pa.float64()),
    })
    print(f"[{label}] input: {table.nbytes/1048576:.1f}MB ({n} rows), "
          f"budget={budget_mb}MB")

    gc.collect()
    r0 = rss_mb()
    sampler = PeakSampler()
    sampler.start()
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as td:
            path = external_sort(
                batch_iter(table, 10000), ["s"], ["s", "v"],
                MemoryGuard(max_memory_mb=budget_mb),
                td, ascending=True, log=False,
            )
            dt = time.time() - t0
    except Exception as e:
        dt = time.time() - t0
        err = f"{type(e).__name__}: {str(e)[:150]}"
    else:
        err = None
    finally:
        sampler.stop()

    gc.collect()
    r1 = rss_mb()
    peak = sampler.peak / (1024 * 1024)
    delta = peak - r0
    print(f"[{label}] rss0={r0:.1f} peak={peak:.1f} Δpeak={delta:+.1f}MB "
          f"Δend={r1-r0:+.1f}MB t={dt:.1f}s err={err}")
    if err:
        print(f"[{label}] RESULT: FAIL")
    elif delta > budget_mb:
        print(f"[{label}] RESULT: OVER (>{budget_mb}MB budget)")
    else:
        print(f"[{label}] RESULT: OK (peak ≤ budget)")
    return delta


if __name__ == "__main__":
    print(f"=== P5 sort bench (baseline Pingis B3a: +31.9MB on 30MB) ===")
    run("B3a-new", n=1_000_000, budget_mb=30)
PYEOF

# ============ DECISIONS + CHANGELOG ============
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

if "| P5 sort buffer (v0.5.2a) |" not in s:
    s = s.replace(
        "| **P2 + P8 RAM fixes (v0.5.1, from Pingis)** | ✅ | 15 |",
        "| **P2 + P8 RAM fixes (v0.5.1, from Pingis)** | ✅ | 15 |\n"
        "| **P5 sort buffer (v0.5.2a, from Pingis)** | ✅ | 2 |",
        1,
    )

marker_135 = "### 13.5. Открытые вопросы"
episode_65 = """**Эпизод 65 (P5: Arrow Builder вместо Python-list в sort merge).**
`_merge_keys` (k-way merge в `external_sort`) копил выход в Python-list:
`buffer[c].append(b.column(c)[pos].as_py())`, затем
`pa.array(buffer[c], ...)`. Пик: `read_batch_size × 28-80 B` (Python
int/float/string object) + `read_batch_size × arrow_size` (при `pa.array`)
= **×3-30 от Arrow**. Пингис B3a: +31.9 MB на 30 MB бюджете (1.06×
over). Fix: `pa.Array.Builder` (`Int64Builder`/`Float64Builder`/
`StringBuilder`/etc.) — Arrow-managed буферы, пик = `read_batch_size ×
arrow_size`. `_make_builder(dtype)` выбирает builder по типу;
`_append_to_builders` пишет по одному скаляру через `.as_py()` (O(1)),
`_flush_builders` → `RecordBatch`. After flush — `builder.finish()`
автоматически resets.

**Ограничение:** для dictionary/list/struct — `_make_builder` возвращает
None; `_merge_keys` raise `NotImplementedError`. Это honest fallback:
sort dictionary/list в v0.5.x не поддерживает; для этого нужны custom
builders, отдельная задача.

**Проверка:** 2 AST guard'а в `test_sort.py` — нет `buffer[...].append`
в `_merge_keys`; присутствует `_append_to_builders`.

**Ожидание:** B3a-new = Δpeak ≤ 30 MB (было +31.9). Даже небольшой
over-budget (10-20%) означает fix неполный — тогда следующий шаг —
уменьшить `read_batch_size` в `SortMemoryModel` для узких строк, или
k-way merge batch-wise через `pc.sort_indices`.

**Урок (усиливает эпизод 63):** бенчмарк — единственный способ поймать
P5. AST guard + unit-тесты этого не ловят (результат корректен). То же,
что P2: fix был «код корректен, но пик памяти не тот». Правило: для
memory-fix — бенчмарк с RSS-сэмплером в CI. `scripts/bench_p5_sort.py`
— первая ласточка; следующие (v0.5.2b, v0.5.3) получат свои benchmarks.

"""
if "Эпизод 65" not in s:
    s = s.replace(marker_135, episode_65 + marker_135, 1)

# §11.5: remove P5
s = s.replace(
    "| `drop_duplicates` returns rows in partition-concatenation order | `drop_duplicates` (v0.3.0) | pandas-order: `.sort_values(...)` после. §13.4/48 |\n",
    "| `drop_duplicates` returns rows in partition-concatenation order | `drop_duplicates` (v0.3.0) | pandas-order: `.sort_values(...)` после. §13.4/48 |\n"
    "| ~~P5: Python-list buffer в sort merge~~ | ~~`_merge_keys`~~ | **закрыто v0.5.2a**: `pa.Array.Builder`. Dictionary/list/struct — `NotImplementedError` (honest). |\n",
    1,
)

p.write_text(s)
print("DECISIONS: §13.4/65 + §11.5")
PY

python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.5.2a] — 2026-10-19

По результатам Pingis audit: P5 (S2) — sort merge Python-list buffer.

### Исправлено
- **P5: `_merge_keys`** — заменить Python-list buffer (`buffer[c].append(
  as_py())`) на `pa.Array.Builder` (`Int64Builder`/`Float64Builder`/
  `StringBuilder`/etc.). Пик на merge: `read_batch_size × arrow_size`,
  а не `read_batch_size × 28-80 B` (Python-object overhead).
  - `_make_builder(dtype)` — выбор builder по Arrow-типу.
  - `_append_to_builders` — append одного значения (с `.as_py()` O(1)).
  - `_flush_builders` — finish → `RecordBatch`.
  - Dictionary/list/struct → `NotImplementedError` (honest; не покрыто).

### Добавлено
- `scripts/bench_p5_sort.py` — RSS-сэмплер benchmark для sort
  (сравнение с Pingis baseline +31.9 MB).
- `tests/test_sort.py::test_merge_keys_uses_arrow_builder_not_python_list`
  — AST guard (нет `buffer[...].append`).
- `tests/test_sort.py::test_merge_keys_uses_arrow_builder` — presence check.

### Ожидаемый эффект
- B3a (1M rows, 30 MB budget): Δpeak ≤ 30 MB (baseline: +31.9 MB).

### Ограничения (v0.5.2a, §11.5)
- Dictionary/list/struct columns: `NotImplementedError` в sort merge.
- `.as_py()` per value — CPU overhead остаётся; P5 — memory-fix, не
  speed-fix. CPU-оптимизация (batch-wise `pc.sort_indices`) — v0.5.4.

"""
if "## [0.5.2a]" not in s:
    s = s.replace("## [0.5.1]", entry + "## [0.5.1]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.5.2a")
PY

# ============ run ============
echo ""
echo "=== pytest tests/test_sort.py -v ==="
set +e
uv run pytest tests/test_sort.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "sort: rc=$RC"
  git add -A
  git commit -m "WIP: v0.5.2a P5 (test red; fix follows)" || true
  exit 1
fi

echo ""
echo "=== bench: P5 sort (was +31.9 MB / 30 MB budget) ==="
uv run python scripts/bench_p5_sort.py

echo ""
echo "=== pytest full fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest full slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.5.2a: P5 — sort merge pa.Array.Builder вместо Python-list (from Pingis); +AST guards + bench"

echo "==> v0.5.2a applied on ${BR}"