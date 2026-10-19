#!/usr/bin/env bash
# patch_v032_rank_bufferable: rank(method='max'|'last'|'average') —
# двухпроходный streaming (pass1: group file; pass2: join). O(1) RAM.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-v032-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch v0.3.2 rank bufferable"

BR="patch/v032-rank-bufferable"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- cadw/ops/rank.py (rewrite) ----------
cat > cadw/ops/rank.py <<'PY'
"""
Rank: external_sort + потоковое вычисление rank.

Алгоритм:
  1. external_sort по `by` (ascending/descending).
  2. Вычисление rank — ОДИН или ДВА прохода (зависит от method):
     - Streamable (`first`/`min`/`dense`): O(1) state, один проход.
     - Bufferable (`max`/`last`/`average`): требуют знать конец
       tie-группы. Двухпроходное решение O(1) RAM:
         Pass 1: scan sorted_by → emit `(first_rank, group_count)`
                 per tie-group на диск. State O(1).
         Pass 2: scan sorted_by + advance по group-файлу (sequential).
                 pos_in_group = global_pos - first_rank + 1.
                 last_rank = first_rank + group_count - 1.
                 max     → last_rank.
                 last    → first_rank + group_count - pos_in_group.
                 average → (first_rank + last_rank) / 2.
  3. external_sort ranks by row_id → original order.
  4. Zip с исходным DataFrame.

Nulls: участвуют как значения (null == null); pandas `skipna` не
поддерживается (v0.3.2).

Ranking formula:
  - `first`:   1-based позиция в sorted order (ties по row_id asc).
  - `min`:     минимум из рангов в tie-группе.
  - `dense`:   1, 2, 3, ... для уникальных значений.
  - `max`:     максимум из рангов в tie-группе.
  - `last`:    обратный min (max = +1, last = -1) по позициям в группе.
  - `average`: (min + max) / 2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.io.batch_writer import ArrowIPCBatchWriter

logger = get_logger("ops.rank")

VALID_STREAMABLE = ("first", "min", "dense")
VALID_BUFFERABLE = ("max", "last", "average")
VALID_METHODS = VALID_STREAMABLE + VALID_BUFFERABLE


def _rank_dtype(method: str, pct: bool) -> pa.DataType:
    """int64 для целочисленных методов без pct; float64 для average или pct."""
    if pct or method == "average":
        return pa.float64()
    return pa.int64()


def _compute_streamable(
    sorted_batches: Iterable[pa.RecordBatch],
    *,
    by: str,
    output_col: str,
    method: str,
    pct: bool,
    total_n: int,
    rank_path: Path,
    memory_guard: MemoryGuard,
) -> Path:
    """One-pass: first/min/dense. O(1) state."""
    dtype = _rank_dtype(method, pct)
    schema = pa.schema([("row_id", pa.uint64()), (output_col, dtype)])
    writer = ArrowIPCBatchWriter(str(rank_path), schema)

    _UNSET = object()
    prev_val = _UNSET
    group_start = 0
    global_pos = 0
    dense_counter = 0

    try:
        for batch in sorted_batches:
            memory_guard.check_batch(batch, name="rank.streamable.input")
            vals = batch.column(by).to_pylist()
            row_ids = batch.column("row_id").to_pylist()

            ranks: list = []
            for v in vals:
                global_pos += 1
                if prev_val is _UNSET:
                    group_start = global_pos
                    dense_counter = 1
                elif v != prev_val:
                    group_start = global_pos
                    dense_counter += 1

                if method == "first":
                    r = global_pos
                elif method == "min":
                    r = group_start
                else:  # dense
                    r = dense_counter

                ranks.append((r / total_n) if pct else r)
                prev_val = v

            rb = pa.RecordBatch.from_arrays(
                [
                    pa.array(row_ids, type=pa.uint64()),
                    pa.array(ranks, type=dtype),
                ],
                schema=schema,
            )
            writer.write_batch(rb)
    finally:
        writer.close()
    return rank_path


def _pass1_groups_to_disk(
    sorted_batches: Iterable[pa.RecordBatch],
    *,
    by: str,
    groups_path: Path,
    memory_guard: MemoryGuard,
) -> Path:
    """Pass 1 of bufferable: scan sorted_by → (first_rank, group_count) file.

    State: prev_val, group_start, global_pos, plus per-batch accumulators
    `starts`/`counts` (bounded by batch size). O(batch_size) RAM.
    """
    schema = pa.schema([
        ("first_rank", pa.int64()),
        ("group_count", pa.int64()),
    ])
    writer = ArrowIPCBatchWriter(str(groups_path), schema)

    _UNSET = object()
    prev_val = _UNSET
    group_start = 0
    global_pos = 0
    starts: list[int] = []
    counts: list[int] = []

    def _flush() -> None:
        nonlocal starts, counts
        if not starts:
            return
        rb = pa.RecordBatch.from_arrays(
            [
                pa.array(starts, type=pa.int64()),
                pa.array(counts, type=pa.int64()),
            ],
            schema=schema,
        )
        writer.write_batch(rb)
        starts = []
        counts = []

    try:
        for batch in sorted_batches:
            memory_guard.check_batch(batch, name="rank.pass1.input")
            vals = batch.column(by).to_pylist()
            for v in vals:
                global_pos += 1
                if prev_val is _UNSET:
                    group_start = global_pos
                elif v != prev_val:
                    starts.append(group_start)
                    counts.append(global_pos - group_start)
                    group_start = global_pos
                prev_val = v
            _flush()

        # Финальная группа (не эмитили её — она «продолжается» до EOF).
        if prev_val is not _UNSET:
            starts.append(group_start)
            counts.append(global_pos - group_start + 1)
            _flush()
    finally:
        writer.close()

    return groups_path


def _iter_groups(path: Path) -> Iterator[tuple[int, int]]:
    """Итерировать (first_rank, group_count) из group-файла."""
    for b in ArrowIPCBatchReader(str(path), ["first_rank", "group_count"]):
        firsts = b.column("first_rank").to_pylist()
        counts = b.column("group_count").to_pylist()
        for f, c in zip(firsts, counts):
            yield (int(f), int(c))


def _pass2_ranks_to_disk(
    sorted_batches: Iterable[pa.RecordBatch],
    *,
    by: str,
    groups_path: Path,
    output_col: str,
    method: str,
    pct: bool,
    total_n: int,
    rank_path: Path,
    memory_guard: MemoryGuard,
) -> Path:
    """Pass 2 of bufferable: scan sorted_by + advance по group file.

    State: prev_val, global_pos, cur_first, cur_count, cur_last. O(1) RAM.
    """
    dtype = _rank_dtype(method, pct)
    schema = pa.schema([("row_id", pa.uint64()), (output_col, dtype)])
    writer = ArrowIPCBatchWriter(str(rank_path), schema)

    groups_iter = _iter_groups(groups_path)
    _UNSET = object()
    prev_val = _UNSET
    global_pos = 0
    cur_first = 0
    cur_count = 0
    cur_last = 0

    def _advance() -> None:
        nonlocal cur_first, cur_count, cur_last
        try:
            f, c = next(groups_iter)
        except StopIteration as exc:
            raise RuntimeError(
                "rank.pass2: group iterator exhausted prematurely — "
                "inconsistent sorted_by / groups file"
            ) from exc
        cur_first = f
        cur_count = c
        cur_last = cur_first + cur_count - 1

    try:
        for batch in sorted_batches:
            memory_guard.check_batch(batch, name="rank.pass2.input")
            vals = batch.column(by).to_pylist()
            row_ids = batch.column("row_id").to_pylist()

            ranks: list = []
            for v, _rid in zip(vals, row_ids):
                global_pos += 1
                if prev_val is _UNSET or v != prev_val:
                    _advance()

                pos_in_group = global_pos - cur_first + 1
                if method == "max":
                    r = cur_last
                elif method == "last":
                    r = cur_first + cur_count - pos_in_group
                elif method == "average":
                    r = (cur_first + cur_last) / 2.0
                else:
                    raise NotImplementedError(f"rank.pass2: method={method!r}")

                ranks.append((r / total_n) if pct else r)
                prev_val = v

            rb = pa.RecordBatch.from_arrays(
                [
                    pa.array(row_ids, type=pa.uint64()),
                    pa.array(ranks, type=dtype),
                ],
                schema=schema,
            )
            writer.write_batch(rb)
    finally:
        writer.close()

    return rank_path


def compute_ranks_to_disk(
    sorted_path: Path,
    *,
    by: str,
    output_col: str,
    method: str,
    pct: bool,
    total_n: int,
    rank_path: Path,
    memory_guard: MemoryGuard,
    log: bool = False,
) -> Path:
    """Stream sorted_by (file) and write (row_id, rank) to rank_path.

    `sorted_path` — Arrow IPC file, columns [by, "row_id"], sorted by `by`.
    Передаём **Path** (не iterable), потому что bufferable-методы требуют
    двух проходов; файл открывается 1–2 раза.

    Streamable (`first`/`min`/`dense`): 1 pass.
    Bufferable (`max`/`last`/`average`): 2 passes (groups → ranks).
    """
    if method not in VALID_METHODS:
        raise NotImplementedError(
            f"rank: method={method!r} not supported. "
            f"Supported: {VALID_METHODS}."
        )
    if pct and total_n <= 0:
        raise ValueError("rank: pct=True requires total_n > 0")

    sorted_path = Path(sorted_path)

    def _batches() -> Iterator[pa.RecordBatch]:
        return ArrowIPCBatchReader(str(sorted_path), [by, "row_id"])

    if method in VALID_STREAMABLE:
        result = _compute_streamable(
            _batches(),
            by=by,
            output_col=output_col,
            method=method,
            pct=pct,
            total_n=total_n,
            rank_path=rank_path,
            memory_guard=memory_guard,
        )
    else:
        # Bufferable: two-pass.
        groups_path = rank_path.parent / (rank_path.stem + "_groups.arrow")
        _pass1_groups_to_disk(
            _batches(),
            by=by,
            groups_path=groups_path,
            memory_guard=memory_guard,
        )
        result = _pass2_ranks_to_disk(
            _batches(),
            by=by,
            groups_path=groups_path,
            output_col=output_col,
            method=method,
            pct=pct,
            total_n=total_n,
            rank_path=rank_path,
            memory_guard=memory_guard,
        )

    if log:
        logger.info(
            "compute_ranks_to_disk: method=%s pct=%s total_n=%d",
            method, pct, total_n,
        )
    return result
PY

# ---------- dataframe.py: rank passes path (not iterable) ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

old = '''        # Step 2: rank per row → (row_id, rank) on disk.
        from cadw.io.batch_reader import ArrowIPCBatchReader as _IPCReader
        rank_pairs_path = compute_ranks_to_disk(
            _IPCReader(str(sorted_by_val), [by, "row_id"]),
            by=by,
            output_col=output_col,
            method=method,
            pct=pct,
            total_n=total_n,
            rank_path=spill / "ranks.arrow",
            memory_guard=self._memory_guard,
            log=False,
        )'''

new = '''        # Step 2: rank per row → (row_id, rank) on disk.
        # compute_ranks_to_disk принимает Path (не iterable) — bufferable
        # методы требуют двух проходов по sorted_by.
        from cadw.io.batch_reader import ArrowIPCBatchReader as _IPCReader
        rank_pairs_path = compute_ranks_to_disk(
            sorted_by_val,
            by=by,
            output_col=output_col,
            method=method,
            pct=pct,
            total_n=total_n,
            rank_path=spill / "ranks.arrow",
            memory_guard=self._memory_guard,
            log=False,
        )'''

if old not in s:
    raise SystemExit("rank step 2 block not found in dataframe.py")
s = s.replace(old, new, 1)

# Update rank docstring: methods list
old_doc = '''        - method ∈ {'first', 'min', 'dense'}:
            - 'first': 1-based позиция в sorted order (ties по row_id asc).
            - 'min':   минимум из рангов в tie-группе.
            - 'dense': 1, 2, 3, ... для уникальных значений.'''
new_doc = '''        - method ∈ {'first', 'min', 'dense', 'max', 'last', 'average'}:
            - 'first':   1-based позиция в sorted order (ties по row_id asc).
            - 'min':     минимум из рангов в tie-группе.
            - 'dense':   1, 2, 3, ... для уникальных значений.
            - 'max':     максимум из рангов в tie-группе.
            - 'last':    обратный min (по позициям в группе).
            - 'average': (min + max) / 2 (float).'''
if old_doc in s:
    s = s.replace(old_doc, new_doc, 1)

# Update "не поддерживаются" note
s = s.replace(
    "        Методы 'max'/'average'/'last' — v0.3.2 (требуют буферизации\n"
    "        tie-группы, O(N) при all-equal).",
    "        Методы 'max'/'last'/'average' реализованы двухпроходным\n"
    "        streaming (O(1) RAM): pass 1 → group file, pass 2 → join.",
    1,
)

# Valid methods check
old_check = '''        if method not in VALID_METHODS:
            raise NotImplementedError(
                f"rank: method={method!r} not supported in v0.3.1. "
                f"Supported: {VALID_METHODS}. "
                f"'max'/'average'/'last' — v0.3.2."
            )'''
new_check = '''        if method not in VALID_METHODS:
            raise NotImplementedError(
                f"rank: method={method!r} not supported. "
                f"Supported: {VALID_METHODS}."
            )'''
if old_check in s:
    s = s.replace(old_check, new_check, 1)

p.write_text(s)
print("dataframe.py: rank — passes path + updated docstring")
PY

# ---------- tests/test_rank.py: add bufferable tests ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_rank.py")
s = p.read_text()

# Обновить импорт VALID_METHODS (нет в тестах, но invalid method тест
# ожидает NotImplementedError для 'max'/'average'/'last' — теперь их нет).
# Заменить блок invalid method test:
old_invalid = '''def test_rank_invalid_method(tmp_path):
    df = make_df(tmp_path, "L", {"k": [1, 2], "v": [1, 2]})
    for bad in ("max", "average", "last", "unknown"):
        with pytest.raises(NotImplementedError, match="method"):
            df.rank(by="k", method=bad)'''
new_invalid = '''def test_rank_invalid_method(tmp_path):
    df = make_df(tmp_path, "L", {"k": [1, 2], "v": [1, 2]})
    for bad in ("unknown", "min_max", "FOO"):
        with pytest.raises(NotImplementedError, match="method"):
            df.rank(by="k", method=bad)'''
if old_invalid in s:
    s = s.replace(old_invalid, new_invalid, 1)

# Дописать bufferable-тесты в конец.
extra = '''

# ====================================================================
# Bufferable: max / last / average (v0.3.2, two-pass streaming)
# ====================================================================

def test_rank_max_with_ties(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["a", "b", "a", "c", "a"], "v": list(range(5))})
    result = df.rank(by="k", method="max")
    got = read_df(result, tmp_path)
    # sorted: a(1), a(2), a(3), b(4), c(5)
    # group a: first=1, last=3; group b: 4; group c: 5
    # original (a,b,a,c,a): 3, 4, 3, 5, 3
    assert got["k_rank"].tolist() == [3, 4, 3, 5, 3]


def test_rank_last_with_ties(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["a", "b", "a", "c", "a"], "v": list(range(5))})
    result = df.rank(by="k", method="last")
    got = read_df(result, tmp_path)
    # sorted: a(pos1,2,3), b(4), c(5)
    # group a: pos_in_group 1→3, 2→2, 3→1 (reverse); b→4; c→5
    # original (a,b,a,c,a): row_id 0(a,pos1)→3, 1(b)→4, 2(a,pos2)→2, 3(c)→5, 4(a,pos3)→1
    assert got["k_rank"].tolist() == [3, 4, 2, 5, 1]


def test_rank_average_with_ties(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["a", "b", "a", "c", "a"], "v": list(range(5))})
    result = df.rank(by="k", method="average")
    got = read_df(result, tmp_path)
    # group a: first=1, last=3 → avg=2.0; b→4.0; c→5.0
    # original (a,b,a,c,a): 2.0, 4.0, 2.0, 5.0, 2.0
    assert got["k_rank"].tolist() == [2.0, 4.0, 2.0, 5.0, 2.0]


def test_rank_max_all_same(tmp_path):
    n = 10
    df = make_df(tmp_path, "L", {"k": ["X"] * n, "v": list(range(n))})
    result = df.rank(by="k", method="max")
    got = read_df(result, tmp_path)
    assert got["k_rank"].tolist() == [n] * n


def test_rank_last_all_same(tmp_path):
    n = 10
    df = make_df(tmp_path, "L", {"k": ["X"] * n, "v": list(range(n))})
    result = df.rank(by="k", method="last")
    got = read_df(result, tmp_path)
    # sorted all same: pos_in_group 1..n, last = n - pos + 1 → n, n-1, ..., 1
    assert got["k_rank"].tolist() == list(range(n, 0, -1))


def test_rank_average_all_same(tmp_path):
    n = 10
    df = make_df(tmp_path, "L", {"k": ["X"] * n, "v": list(range(n))})
    result = df.rank(by="k", method="average")
    got = read_df(result, tmp_path)
    # group first=1, last=n → avg=(1+n)/2
    expected = (1 + n) / 2
    assert got["k_rank"].tolist() == [expected] * n


def test_rank_max_descending(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["b", "a", "c", "a"], "v": list(range(4))})
    result = df.rank(by="k", method="max", ascending=False)
    got = read_df(result, tmp_path)
    # sorted desc: c(1), b(2), a(3), a(4)
    # max: c=1, b=2, a=4 (last in group)
    # original (b,a,c,a): 2, 4, 1, 4
    assert got["k_rank"].tolist() == [2, 4, 1, 4]


def test_rank_max_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(7)
    data = {
        "k": rng.integers(0, 5, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="max")
    got = read_df(result, tmp_path)

    pdf = pd.DataFrame(data)
    expected = pdf["k"].rank(method="max").astype("int64")
    np.testing.assert_array_equal(
        got["k_rank"].to_numpy(), expected.to_numpy()
    )


def test_rank_last_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(7)
    data = {
        "k": rng.integers(0, 5, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="last")
    got = read_df(result, tmp_path)

    pdf = pd.DataFrame(data)
    expected = pdf["k"].rank(method="last").astype("int64")
    np.testing.assert_array_equal(
        got["k_rank"].to_numpy(), expected.to_numpy()
    )


def test_rank_average_pandas_equivalence(tmp_path):
    n = 100
    rng = np.random.default_rng(7)
    data = {
        "k": rng.integers(0, 5, size=n).tolist(),
        "v": list(range(n)),
    }
    df = make_df(tmp_path, "L", data, chunk=10)
    result = df.rank(by="k", method="average")
    got = read_df(result, tmp_path)

    pdf = pd.DataFrame(data)
    expected = pdf["k"].rank(method="average")
    np.testing.assert_allclose(
        got["k_rank"].to_numpy(), expected.to_numpy(), rtol=1e-10
    )


def test_rank_average_pct(tmp_path):
    df = make_df(tmp_path, "L", {"k": ["a", "b", "a", "c"], "v": list(range(4))})
    result = df.rank(by="k", method="average", pct=True)
    got = read_df(result, tmp_path)
    # groups sorted: a(1,2 → avg 1.5), b(3), c(4)
    # original (a,b,a,c): 1.5, 3, 1.5, 4 → /4 = 0.375, 0.75, 0.375, 1.0
    assert got["k_rank"].tolist() == [0.375, 0.75, 0.375, 1.0]


def test_rank_max_dtype_int(tmp_path):
    df = make_df(tmp_path, "L", {"k": [1, 2, 1], "v": [1, 2, 3]})
    result = df.rank(by="k", method="max")
    got = read_df(result, tmp_path)
    assert got["k_rank"].dtype.kind == "i"  # int64


def test_rank_average_dtype_float(tmp_path):
    df = make_df(tmp_path, "L", {"k": [1, 2, 1], "v": [1, 2, 3]})
    result = df.rank(by="k", method="average")
    got = read_df(result, tmp_path)
    assert got["k_rank"].dtype.kind == "f"  # float64


def test_rank_max_after_filter(tmp_path):
    df = make_df(tmp_path, "L", {
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    filtered = df[df["v"] > 2]  # (b,3),(b,4),(c,5)
    result = filtered.rank(by="k", method="max")
    got = read_df(result, tmp_path)
    # sorted: b(1), b(2), c(3) → max: b=2, b=2, c=3
    # output in filtered order: [2, 2, 3]
    assert got["k_rank"].tolist() == [2, 2, 3]
'''

s = s.rstrip() + extra + "\n"
p.write_text(s)
print("test_rank.py: +15 bufferable tests")
PY

# ---------- DECISIONS.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

# §11.1: rank
s = s.replace(
    "| `rank` (v0.3.1: Sort + local stream; first/min/dense) | ✅ | 19 |",
    "| `rank` (v0.3.1/3.2: Sort + stream; first/min/dense/max/last/average) | ✅ | 35 |",
    1,
)

# §11.5: удалить ограничение без max/average/last
s = s.replace(
    "| `rank` без max/average/last | `rank` (v0.3.1) | требуют буферизации tie-группы (O(N) при all-equal); v0.3.2 — двухпроходный streaming или bounded-buffer. §13.4/49 |\n",
    "",
    1,
)

# §13.4/49: обновить статус
s = s.replace(
    "MVP v0.3.1 = streamable trio. v0.3.2: `max`/`average`/`last` через\n"
    "двухпроходный streaming (первый проход: group_end_rank per group → на\n"
    "диск; второй: join с sorted stream) или bounded-buffer с spill при\n"
    "группе > budget.",
    "MVP v0.3.1 = streamable trio. v0.3.2 (закрыто): `max`/`average`/`last`\n"
    "реализованы двухпроходным streaming:\n"
    "  Pass 1: scan sorted_by → `(first_rank, group_count)` per tie-group\n"
    "         на диск. State: prev_val/group_start/global_pos + per-batch\n"
    "         accumulators (O(batch_size)).\n"
    "  Pass 2: scan sorted_by + advance по group file (sequential).\n"
    "         pos_in_group = global_pos - first_rank + 1.\n"
    "         last_rank = first_rank + group_count - 1.\n"
    "         max → last_rank; last → first_rank + count - pos;\n"
    "         average → (first + last) / 2.\n"
    "Итого: O(1) RAM независимо от max tie group size. Плата — два\n"
    "прохода по sorted_by (I/O) + промежуточный group file\n"
    "(`N_groups × 16 B`, обычно << N × row_size). API `compute_ranks_to_disk`\n"
    "принимает Path (не iterable) — файл открывается 1–2 раза.\n"
    "**Урок:** буферизуемое состояние можно эмулировать O(1) RAM\n"
    "**двухпроходным** streaming'ом; цена — 2× I/O и промежуточный файл.\n"
    "Это общий паттерн: `median`/`quantile`/`mode` per-group — та же схема\n"
    "(счётчики → файл → join). Выбор 'однопроходный streamable' vs\n"
    "'двухпроходный bufferable' — по принципу: streamable только если\n"
    "реально O(1)-state; иначе — не притворяться.",
    1,
)

p.write_text(s)
print("DECISIONS: §11.1, §11.5, §13.4/49 updated")
PY

# ---------- CHANGELOG.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.3.2] — 2026-10-19

### Добавлено
- `CadwDataFrame.rank` теперь поддерживает `method='max'|'last'|'average'`
  через **двухпроходный streaming** (O(1) RAM):
  - Pass 1: scan sorted_by → `(first_rank, group_count)` per tie-group
    на диск. State O(1) + per-batch accumulators (O(batch_size)).
  - Pass 2: scan sorted_by + advance по group file (sequential).
    `pos_in_group = global_pos - first_rank + 1`;
    `last_rank = first_rank + group_count - 1`;
    `max → last_rank`; `last → first_rank + count - pos`;
    `average → (first + last) / 2`.
- `VALID_METHODS = ('first', 'min', 'dense', 'max', 'last', 'average')`.
- `_VALID_STREAMABLE` / `_VALID_BUFFERABLE` — разделение методов.
- `_rank_dtype(method, pct)`: int64 для int-методов без pct; float64
  для average или pct.

### Изменено
- `compute_ranks_to_disk(sorted_path: Path, ...)` — принимает **Path**
  вместо `Iterable[RecordBatch]`, потому что bufferable-методы требуют
  двух проходов. Единственный caller — `CadwDataFrame.rank` — обновлён.

### Ограничения (v0.3.2, §11.5)
- **null values**: участвуют как значения (null == null); pandas
  `rank(skipna=True)` не поддерживается.
- **cost**: `max`/`last`/`average` = 2 passes by sorted_by + 1 pass
  by row_id (re-sort). Против 1 pass у streamable.

### Тесты
- 15 новых в `test_rank.py`: max/last/average на asc и desc, pct для
  average, dtype-проверки, after filter, pandas-equivalence для всех
  трёх, all-same для max/last/average.

"""
if "## [0.3.2]" not in s:
    s = s.replace("## [0.3.1]", entry + "## [0.3.1]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.3.2")
PY

# ---------- run ----------
echo "=== pytest tests/test_rank.py -v ==="
set +e
uv run pytest tests/test_rank.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "rank: rc=$RC"
  git add -A
  git commit -m "WIP: v0.3.2 rank bufferable (test red; fix follows)" || true
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.3.2: rank(max/last/average) — двухпроходный streaming (O(1) RAM); 15 tests"

echo "==> v0.3.2 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"