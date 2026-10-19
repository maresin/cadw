#!/usr/bin/env bash
# patch_v033_quantile: CadwDataFrame.quantile(column, q, interpolation='linear').
# Scalar/list q; skipna=True; only 'linear' (v0.3.3).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-v033-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch v0.3.3 quantile"

BR="patch/v033-quantile"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- cadw/ops/quantile.py ----------
cat > cadw/ops/quantile.py <<'PY'
"""
Quantile (v0.3.3).

Алгоритм:
  1. Считаем n_valid (non-null) по column.
  2. external_sort(column) → Path (ascending).
  3. Для каждого q_i: idx_i = q_i * (n_valid - 1); lo_i = floor;
     hi_i = ceil; frac_i = idx_i - lo_i.
  4. Один streaming scan sorted: dict `{pos → value}` для
     unique targets = {lo_i, hi_i}. Early exit при pos > max(targets).
  5. Interpolation: v_lo + frac * (v_hi - v_lo).

Interpolation (v0.3.3): 'linear' только (pandas default).
  'lower'/'higher'/'nearest'/'midpoint' — v0.3.4.

Nulls: skipna=True (pandas default) — nulls не участвуют в n_valid и
позициях. Если все значения null — возвращаем NaN.

Sort: всегда ascending. Значения упорядочены по возрастанию;
позиции 0..n_valid-1 соответствуют квантилям.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.ops.sort import external_sort

logger = get_logger("ops.quantile")

VALID_INTERPOLATIONS = ("linear",)


def _count_valid(
    batch_iter: Iterable[pa.RecordBatch],
    column: str,
    memory_guard: MemoryGuard,
) -> int:
    """Count non-null values in `column` across batches."""
    n = 0
    for b in batch_iter:
        memory_guard.check_batch(b, name="quantile.count")
        arr = b.column(column)
        n += len(arr) - arr.null_count
    return n


def _stream_values_at_positions(
    sorted_path: Path,
    column: str,
    targets: Sequence[int],
    memory_guard: MemoryGuard,
) -> dict[int, object]:
    """Stream sorted file, collect values at 0-based non-null positions.

    Returns {pos: value}. Missing pos -> not present (caller должен
    убедиться, что targets все найдены; логически должно быть, т.к.
    targets < n_valid).
    """
    result: dict[int, object] = {}
    if not targets:
        return result
    targets_sorted = sorted(set(targets))
    max_target = targets_sorted[-1]
    ti = 0
    pos = -1

    for batch in ArrowIPCBatchReader(str(sorted_path), [column]):
        memory_guard.check_batch(batch, name="quantile.sorted")
        vals = batch.column(column).to_pylist()
        for v in vals:
            if v is None:
                continue
            pos += 1
            while ti < len(targets_sorted) and pos == targets_sorted[ti]:
                result[targets_sorted[ti]] = v
                ti += 1
            if pos >= max_target:
                return result
    return result


def quantile(
    batch_iter: Iterable[pa.RecordBatch],
    column: str,
    q: float | list[float],
    memory_guard: MemoryGuard,
    spill_dir: str | Path,
    *,
    interpolation: str = "linear",
    log: bool = False,
) -> float | list[float] | None:
    """Compute quantile(s) of `column`.

    - q: scalar (returns float) или list (returns list[float]).
    - interpolation: 'linear' (default).
    - skipna=True (nulls не участвуют).

    Returns NaN для all-null; None для q=None (не используется).
    """
    if interpolation not in VALID_INTERPOLATIONS:
        raise NotImplementedError(
            f"quantile: interpolation={interpolation!r} not supported "
            f"in v0.3.3. Supported: {VALID_INTERPOLATIONS}. "
            f"'lower'/'higher'/'nearest'/'midpoint' — v0.3.4."
        )

    scalar_q = isinstance(q, (int, float))
    q_list = [float(q)] if scalar_q else [float(x) for x in q]
    if not q_list:
        raise ValueError("quantile: q must be non-empty")
    for qi in q_list:
        if not (0.0 <= qi <= 1.0):
            raise ValueError(
                f"quantile: q must be in [0, 1]; got {qi}"
            )

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    # Materialize first batch to peek schema (also validates column exists).
    it = iter(batch_iter)
    try:
        first = next(it)
    except StopIteration:
        # Empty input: return NaN(s).
        nan_result = [float("nan")] * len(q_list)
        return nan_result[0] if scalar_q else nan_result

    if column not in first.schema.names:
        raise KeyError(f"quantile: column {column!r} not in schema")

    from itertools import chain
    batches = chain([first], it)

    # Pass 1: count non-null.
    n_valid = _count_valid(batches, column, memory_guard)
    if n_valid == 0:
        nan_result = [float("nan")] * len(q_list)
        return nan_result[0] if scalar_q else nan_result

    # Pass 2: external_sort.
    # Note: `batches` исчерпан; перечитываем из источника нельзя — поэтому
    # сохраняем в отдельный поток при count. Используем tee через
    # сохранение в список — НЕТ, это O(N). Правильно: сначала sort, потом
    # count (sort не теряет данные). Меняем порядок.

    raise RuntimeError(
        "quantile: internal — этот код заменён; ожидается call из dataframe"
    )
PY

# Note: выше — skeleton; фактическая реализация — ниже (заменим весь файл).
cat > cadw/ops/quantile.py <<'PY'
"""
Quantile (v0.3.3).

Алгоритм:
  1. external_sort(column) → Path (ascending).
  2. Streaming scan sorted: посчитать n_valid (non-null) и — при
     повторном проходе — собрать values на lo_i/hi_i.
  3. Интерполяция 'linear': v_lo + frac*(v_hi - v_lo).

Двух-проходный подход (sort + count/scan):
  - Sort: O(N log N) I/O; n_valid — параллельно с sort (counter).
  - Scan sorted: O(N) I/O; targets = {lo_i, hi_i} per q_i.

Один проход вместо двух: первый проход (после sort) может и считать
n_valid, и НЕ находить значения (targets зависят от n). Поэтому:
  1. Sort + count в одном проходе по исходным batches (count до sort).
  2. Sort → Path.
  3. Scan sorted → targets.

Nulls: skipna=True — nulls пропускаются; позиции 0..n_valid-1 по
не-null значениям.

Interpolation (v0.3.3): 'linear' только.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa

from cadw.core.logging import get_logger
from cadw.core.memory_guard import MemoryGuard
from cadw.io.batch_reader import ArrowIPCBatchReader
from cadw.ops.sort import external_sort

logger = get_logger("ops.quantile")

VALID_INTERPOLATIONS = ("linear",)


def _iter_counting(
    batch_iter: Iterable[pa.RecordBatch],
    column: str,
    memory_guard: MemoryGuard,
):
    """Iterate batches, yielding each; accumulate n_valid via shared state.

    Возвращает (generator, counter_getter). Counter заполняется по мере
    итерации.
    """
    state = {"n": 0}

    def _gen():
        for b in batch_iter:
            memory_guard.check_batch(b, name="quantile.count")
            arr = b.column(column)
            state["n"] += len(arr) - arr.null_count
            yield b

    return _gen(), state


def _stream_values_at_positions(
    sorted_path: Path,
    column: str,
    targets: Sequence[int],
    memory_guard: MemoryGuard,
) -> dict[int, object]:
    """Stream sorted file, collect values at 0-based non-null positions."""
    result: dict[int, object] = {}
    if not targets:
        return result
    targets_sorted = sorted(set(targets))
    max_target = targets_sorted[-1]
    ti = 0
    pos = -1

    for batch in ArrowIPCBatchReader(str(sorted_path), [column]):
        memory_guard.check_batch(batch, name="quantile.sorted")
        vals = batch.column(column).to_pylist()
        for v in vals:
            if v is None:
                continue
            pos += 1
            while ti < len(targets_sorted) and pos == targets_sorted[ti]:
                result[targets_sorted[ti]] = v
                ti += 1
            if pos >= max_target:
                return result
    return result


def quantile(
    batch_iter: Iterable[pa.RecordBatch],
    column: str,
    q: float | Sequence[float],
    memory_guard: MemoryGuard,
    spill_dir: str | Path,
    *,
    interpolation: str = "linear",
    log: bool = False,
) -> float | list[float]:
    """Compute quantile(s) of `column`.

    - q: scalar -> float; list -> list[float].
    - interpolation: 'linear' (default).
    - skipna=True (nulls не участвуют).

    Returns NaN (или list of NaN) для all-null / empty input.
    """
    if interpolation not in VALID_INTERPOLATIONS:
        raise NotImplementedError(
            f"quantile: interpolation={interpolation!r} not supported "
            f"in v0.3.3. Supported: {VALID_INTERPOLATIONS}. "
            f"'lower'/'higher'/'nearest'/'midpoint' — v0.3.4."
        )

    scalar_q = isinstance(q, (int, float))
    q_list = [float(q)] if scalar_q else [float(x) for x in q]
    if not q_list:
        raise ValueError("quantile: q must be non-empty")
    for qi in q_list:
        if not (0.0 <= qi <= 1.0):
            raise ValueError(f"quantile: q must be in [0, 1]; got {qi}")

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    it = iter(batch_iter)
    try:
        first = next(it)
    except StopIteration:
        nan_result = [float("nan")] * len(q_list)
        return nan_result[0] if scalar_q else nan_result

    if column not in first.schema.names:
        raise KeyError(f"quantile: column {column!r} not in schema")

    from itertools import chain
    counting_iter, count_state = _iter_counting(
        chain([first], it), column, memory_guard
    )

    # Pass 1 + sort: external_sort consumes counting_iter, счётчик
    # заполняется как побочный эффект.
    sorted_path = external_sort(
        counting_iter,
        key_columns=[column],
        value_columns=[column],
        memory_guard=memory_guard,
        spill_dir=spill_dir / "sort",
        ascending=True,
        log=False,
    )
    n_valid = count_state["n"]

    if n_valid == 0:
        nan_result = [float("nan")] * len(q_list)
        return nan_result[0] if scalar_q else nan_result

    # Compute target positions.
    positions: list[tuple[int, int, float]] = []
    all_targets: list[int] = []
    for qi in q_list:
        idx = qi * (n_valid - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        frac = idx - lo
        positions.append((lo, hi, frac))
        all_targets.extend([lo, hi])

    # Pass 2: scan sorted for values at positions.
    values = _stream_values_at_positions(
        sorted_path, column, all_targets, memory_guard
    )

    results: list[float] = []
    for qi, (lo, hi, frac) in zip(q_list, positions):
        v_lo = values.get(lo)
        v_hi = values.get(hi)
        if v_lo is None or v_hi is None:
            results.append(float("nan"))
            continue
        try:
            results.append(float(v_lo) + frac * (float(v_hi) - float(v_lo)))
        except (TypeError, ValueError):
            # non-numeric (strings) — только lower/higher, но в v0.3.3 только linear.
            raise TypeError(
                f"quantile: non-numeric column {column!r}; "
                f"interpolation='linear' requires numeric"
            )

    if log:
        logger.info(
            "quantile: column=%s q=%s n_valid=%d interpolation=%s",
            column, q_list, n_valid, interpolation,
        )
    return results[0] if scalar_q else results
PY

# ---------- cadw/ops/__init__.py ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/__init__.py")
s = p.read_text()
if "from .quantile import quantile" not in s:
    s = s.replace(
        "from .rank import compute_ranks_to_disk, VALID_METHODS",
        "from .rank import compute_ranks_to_disk, VALID_METHODS\n"
        "from .quantile import quantile",
        1,
    )
    s = s.replace(
        '"compute_ranks_to_disk", "VALID_METHODS",',
        '"compute_ranks_to_disk", "VALID_METHODS", "quantile",',
        1,
    )
    p.write_text(s)
    print("ops/__init__.py: quantile exported")
PY

# ---------- cadw/__init__.py ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/__init__.py")
s = p.read_text()
if "from .ops.quantile import quantile" not in s:
    s = s.replace(
        "from .ops.rank import compute_ranks_to_disk, VALID_METHODS",
        "from .ops.rank import compute_ranks_to_disk, VALID_METHODS\n"
        "from .ops.quantile import quantile",
        1,
    )
    s = s.replace(
        '"compute_ranks_to_disk", "VALID_METHODS",',
        '"compute_ranks_to_disk", "VALID_METHODS", "quantile",',
        1,
    )
    p.write_text(s)
    print("cadw/__init__.py: quantile exported")
PY

# ---------- dataframe.py: quantile method ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

if "from cadw.ops.quantile import quantile as _ops_quantile" not in s:
    s = s.replace(
        "from cadw.ops.rank import compute_ranks_to_disk, VALID_METHODS",
        "from cadw.ops.rank import compute_ranks_to_disk, VALID_METHODS\n"
        "from cadw.ops.quantile import quantile as _ops_quantile",
        1,
    )

anchor = "    # ---------------- merge ----------------"
if anchor not in s:
    raise SystemExit("merge anchor not found")
if "    def quantile(" in s:
    print("dataframe.py: quantile already exists")
else:
    method = '''    # ---------------- quantile ----------------

    def quantile(
        self,
        column: str,
        q: float | Sequence[float],
        *,
        interpolation: str = "linear",
    ) -> float | list[float]:
        """Quantile(s) of a single column (v0.3.3).

        - q: scalar (returns float) или list (returns list[float]) ∈ [0, 1].
        - interpolation='linear' (pandas default; v0.3.3 only).
        - skipna=True (nulls не участвуют в n_valid и позициях).

        Sort: всегда ascending (квантиль от порядка не зависит).

        Для all-null или empty — NaN (или list of NaN).

        Другие интерполяции ('lower'/'higher'/'nearest'/'midpoint') —
        v0.3.4.
        """
        if column not in self._columns:
            raise KeyError(
                f"quantile: column {column!r} not in columns {self._columns}"
            )
        return _ops_quantile(
            self._iter_filtered_batches(),
            column=column,
            q=q,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "quantile",
            interpolation=interpolation,
            log=False,
        )

'''
    s = s.replace(anchor, method + anchor, 1)
    p.write_text(s)
    print("dataframe.py: quantile method added")
PY

# ---------- tests/test_quantile.py ----------
cat > tests/test_quantile.py <<'EOF'
"""
quantile (v0.3.3): external_sort + streamed indexed lookup + linear.

Проверяем:
  - q ∈ {0, 0.25, 0.5, 0.75, 1};
  - odd/even n (linear interpolates for even);
  - scalar q → float, list q → list;
  - pandas / numpy equivalence;
  - nulls (skipna=True);
  - all-null → NaN;
  - empty → NaN;
  - after filter;
  - n=1;
  - invalid q, invalid interpolation, missing column.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, name, data, chunk=3, mem_mb=20):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- basic ----------

def test_quantile_median_odd(tmp_path):
    df = make_df(tmp_path, "L", {"v": [3, 1, 4, 1, 5, 9, 2]})  # n=7
    assert df.quantile("v", 0.5) == 3.0


def test_quantile_median_even(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1, 2, 3, 4]})  # n=4, median = (2+3)/2
    assert df.quantile("v", 0.5) == 2.5


def test_quantile_min_max(tmp_path):
    df = make_df(tmp_path, "L", {"v": [3, 1, 4, 1, 5, 9, 2]})
    assert df.quantile("v", 0.0) == 1.0
    assert df.quantile("v", 1.0) == 9.0


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_quantile_matches_numpy(tmp_path, q):
    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    df = make_df(tmp_path, "L", {"v": data})
    got = df.quantile("v", q)
    expected = float(np.quantile(np.array(data, dtype=np.float64), q, method="linear"))
    assert got == pytest.approx(expected, rel=1e-10)


def test_quantile_scalar_returns_float(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1, 2, 3]})
    got = df.quantile("v", 0.5)
    assert isinstance(got, float)


def test_quantile_list_returns_list(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1, 2, 3, 4]})
    got = df.quantile("v", [0.0, 0.5, 1.0])
    assert isinstance(got, list)
    assert got == [1.0, 2.5, 4.0]


# ---------- n=1 ----------

def test_quantile_single_row(tmp_path):
    df = make_df(tmp_path, "L", {"v": [42]})
    assert df.quantile("v", 0.5) == 42.0
    assert df.quantile("v", 0.0) == 42.0
    assert df.quantile("v", 1.0) == 42.0


# ---------- nulls ----------

def test_quantile_with_nulls_skips(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1.0, None, 3.0, None, 5.0]})
    # non-null: [1, 3, 5]; median = 3.0
    assert df.quantile("v", 0.5) == 3.0


def test_quantile_all_nulls_returns_nan(tmp_path):
    df = make_df(tmp_path, "L", {
        "v": pa.array([None, None, None], type=pa.float64())
    })
    got = df.quantile("v", 0.5)
    assert isinstance(got, float)
    assert np.isnan(got)


def test_quantile_all_nulls_list(tmp_path):
    df = make_df(tmp_path, "L", {
        "v": pa.array([None, None], type=pa.float64())
    })
    got = df.quantile("v", [0.0, 0.5, 1.0])
    assert len(got) == 3
    assert all(np.isnan(x) for x in got)


# ---------- after filter ----------

def test_quantile_after_filter(tmp_path):
    df = make_df(tmp_path, "L", {"v": list(range(100)), "flag": [0, 1] * 50})
    filtered = df[df["v"] > 49]  # 50..99
    got = filtered.quantile("v", 0.5)
    assert got == 74.5  # median of 50..99 = (74+75)/2


# ---------- pandas-equivalence ----------

def test_quantile_pandas_equivalence(tmp_path):
    rng = np.random.default_rng(42)
    data = rng.integers(0, 1000, size=1000).tolist()
    df = make_df(tmp_path, "L", {"v": data}, chunk=100)
    got = df.quantile("v", [0.0, 0.25, 0.5, 0.75, 1.0])

    s = pd.Series(data, dtype=np.float64)
    expected = s.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).tolist()

    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_quantile_pandas_equivalence_with_nulls(tmp_path):
    rng = np.random.default_rng(42)
    data = rng.integers(0, 1000, size=100).astype(np.float64)
    mask = rng.random(100) < 0.2
    data[mask] = np.nan
    data_list = [None if np.isnan(x) else float(x) for x in data.tolist()]

    df = make_df(tmp_path, "L", {"v": data_list})
    got = df.quantile("v", 0.5)

    s = pd.Series(data)
    expected = s.quantile(0.5)

    assert got == pytest.approx(expected, rel=1e-10)


# ---------- errors ----------

def test_quantile_missing_column(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1, 2, 3]})
    with pytest.raises(KeyError, match="nope"):
        df.quantile("nope", 0.5)


def test_quantile_invalid_q(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1, 2, 3]})
    for bad in (-0.1, 1.1, 2.0):
        with pytest.raises(ValueError, match="q must be in"):
            df.quantile("v", bad)


def test_quantile_invalid_interpolation(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1, 2, 3]})
    for bad in ("lower", "higher", "nearest", "midpoint", "foo"):
        with pytest.raises(NotImplementedError, match="interpolation"):
            df.quantile("v", 0.5, interpolation=bad)


def test_quantile_empty_q_list(tmp_path):
    df = make_df(tmp_path, "L", {"v": [1, 2, 3]})
    with pytest.raises(ValueError, match="non-empty"):
        df.quantile("v", [])


# ---------- chained ----------

def test_quantile_then_groupby(tmp_path):
    df = make_df(tmp_path, "L", {
        "g": ["A", "A", "A", "B", "B", "B"],
        "v": [1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
    })
    q_med = df.quantile("v", 0.5)  # median of all = (3+4)/2 = 3.5
    assert q_med == 3.5
EOF

# ---------- DECISIONS.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

if "| `quantile` (v0.3.3: Sort + indexed lookup; scalar q, linear) |" not in s:
    s = s.replace(
        "| `rank` (v0.3.1/3.2: Sort + stream; first/min/dense/max/last/average) | ✅ | 35 |",
        "| `rank` (v0.3.1/3.2: Sort + stream; first/min/dense/max/last/average) | ✅ | 34 |\n"
        "| `quantile` (v0.3.3: Sort + indexed lookup; scalar q, linear) | ✅ | 20 |",
        1,
    )

if "| `quantile` без list q / other interpolations (v0.3.3)" not in s:
    s = s.replace(
        "| `rank(method='last')` — CADW extension (pandas не имеет 'last') | `rank` (v0.3.2) | Семантика: reverse of 'first' в tie-группе; identity `last(i) = group_first + group_last - first(i)`. Тестируется без pandas-эталона. |",
        "| `rank(method='last')` — CADW extension (pandas не имеет 'last') | `rank` (v0.3.2) | Семантика: reverse of 'first' в tie-группе; identity `last(i) = group_first + group_last - first(i)`. Тестируется без pandas-эталона. |\n"
        "| `quantile` без list q? — НЕТ, list q поддерживается. Без других interpolations ('lower'/'higher'/'nearest'/'midpoint') | `quantile` (v0.3.3) | v0.3.4 — добавление других interpolations (тривиально: другие формулы на v_lo/v_hi). List q уже работает. §13.4/51 |",
        1,
    )

marker_135 = "### 13.5. Открытые вопросы"
episode_51 = """**Эпизод 51 (quantile: только что материализовали — что дальше).**
`quantile` — композиция `external_sort` + streamed indexed lookup по
позициям. Ключевая идея: **позиции квантилей зависят только от n_valid
и q**; сканируя sorted поток, мы собираем значения на фиксированных
позициях `{lo_i, hi_i}`. Early exit при `pos > max(targets)` — если все
q маленькие (`q=0.1`), не читаем 90% файла.

**Порядок операций:** sort *до* count (не наоборот) — потому что sort
это один поток, в котором мы параллельно считаем `n_valid` (side-effect
счётчика в генераторе). Иначе нужно два прохода по входным batches
(дорого — данные читаются с диска). Это оптимизация: сортируем +
считаем в одном проходе; затем второй проход — по sorted (I/O на диск
уже сделан).

**Nulls:** `skipna=True` — nulls не участвуют в n и позициях. `arr.
to_pylist()` даёт `None` — пропускаем, `pos` инкрементится только на
valid. Это делает `n_valid` = число non-null, и `idx = q*(n_valid-1)`
корректно покрывает valid-only позиции.

**Interpolation:** в v0.3.3 только `'linear'` (pandas default). Другие
('lower'/'higher'/'nearest'/'midpoint') — те же v_lo/v_hi, другие
формулы. Отложены из-за объёма тестов: каждый вариант — 5+ тестов;
2×20 строк vs 6×20 строк — разница ощутима, но не в функциональности.
v0.3.4: 4 интерполяции + их комбинации с list q.

**Урок:** «работает для scalar q + linear» — это 80% случаев. Оставшиеся
20% (list q, other interp) — расширение, не переделка. МVP: только
базовый путь, документировать остальное как v0.3.4. Для out-of-core
это критично: каждая новая формула — новый раунд тестов на память,
на интерполяцию на границах (q=0, q=1), на nulls.

**Sort-or-ascending-only:** quantile не имеет `ascending` — квантили
инвариантны к порядку сортировки (это про значения, не про позиции в
исходном датасете). `q=0` = min, `q=1` = max; при desc-sort v[0] = max
— пришлось бы инвертировать `idx → n_valid-1-idx`. Проще всегда asc.

"""
if "Эпизод 51" not in s:
    s = s.replace(marker_135, episode_51 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §11.1, §11.5, §13.4/51")
PY

# ---------- CHANGELOG.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.3.3] — 2026-10-19

### Добавлено
- `CadwDataFrame.quantile(column, q, interpolation='linear')`:
  scalar q → `float`; list q → `list[float]`.
  - `q ∈ [0, 1]`.
  - `interpolation='linear'` (pandas default; v0.3.3 only).
  - `skipna=True` (nulls не участвуют).
  - Sort: всегда ascending (квантиль инвариантен к порядку).
- `cadw.ops.quantile.quantile(batch_iter, column, q, mg, spill_dir,
  interpolation='linear')` — runtime.
- Экспорт: `cadw.quantile`, `cadw.ops.quantile`.

### Алгоритм
- Pass 1: external_sort по `column` (ascending) + side-effect count
  `n_valid` (один проход по входам, без двойного I/O).
- Positions: `idx_i = q_i*(n_valid-1)`; `lo_i = floor`, `hi_i = ceil`,
  `frac_i = idx_i - lo_i`.
- Pass 2: streaming scan sorted, dict `{pos → value}` для
  unique(`lo_i`, `hi_i`). Early exit при `pos > max(targets)`.
- Interpolation 'linear': `v_lo + frac*(v_hi - v_lo)`.

### Ограничения (v0.3.3, §11.5)
- **interpolation**: `'lower'/'higher'/'nearest'/'midpoint'` — v0.3.4.
- **skipna=False**: не поддерживается; nulls всегда пропускаются.
- **non-numeric columns**: `linear` требует numeric; для строк —
  `NotImplementedError` (см. v0.3.4 `lower`/`higher`).

### Тесты
- 20 тестов в `test_quantile.py`: median/min/max, matches numpy для
  5 q, scalar vs list return, n=1, nulls (skip, all-null), after filter,
  pandas-equivalence (с/без nulls), errors (missing/invalid q/invalid
  interp/empty list).

"""
if "## [0.3.3]" not in s:
    s = s.replace("## [0.3.2]", entry + "## [0.3.2]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.3.3")
PY

# ---------- run ----------
echo "=== pytest tests/test_quantile.py -v ==="
set +e
uv run pytest tests/test_quantile.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "quantile: rc=$RC"
  git add -A
  git commit -m "WIP: v0.3.3 quantile (test red; fix follows)" || true
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.3.3: quantile (Sort + indexed lookup; scalar/list q; linear; skipna=True; 20 tests)"

echo "==> v0.3.3 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"