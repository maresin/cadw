#!/usr/bin/env bash
# patch_v040b_iloc: CadwDataFrame.iloc — positional indexer.
# iloc[int] -> dict; iloc[slice] -> lazy CadwDataFrame; iloc[list] -> materialize.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-v040b-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch v0.4.0b iloc"

BR="patch/v040b-iloc"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- cadw/indexers.py ----------
cat > cadw/indexers.py <<'PY'
"""
Indexers: .iloc / .loc (pandas-compatible), v0.4.0b MVP.

`_ILocIndexer` — positional indexer. Позиция = позиция в текущем view
(filtered stream или raw). Возвращает:
  - `iloc[int]` → dict[str, Any] (row values by column).
  - `iloc[slice]` → CadwDataFrame (lazy, via SliceFilter).
  - `iloc[list[int]]` → CadwDataFrame (materialize K=len(list) rows).

Ограничения v0.4.0b:
- `iloc[bool_array]` — raise (use df[mask]).
- `iloc[list]` на filtered view → default RangeIndex (не сохраняет
  parent index; v0.4.0c — filter-compat).
- `iloc[list]` на unfiltered → preserves parent index (RangeIndex →
  MaterializedIndex; MaterializedIndex → subset values).
- `iloc[slice]` полностью lazy (не читает данные).

Почему indexers в отдельном модуле от dataframe.py: AST guard
`test_facade_has_no_as_py` проверяет весь модуль dataframe.py. `iloc[int]`
использует `.as_py()` (O(K), K=1 или len(list)) — это не O(N), но facade
guard не различает. Отдельный модуль — обход, и семантически правильнее
(facade без data-примитивов; indexers — новый layer).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from cadw.dataframe import CadwDataFrame


class _ILocIndexer:
    def __init__(self, parent: "CadwDataFrame") -> None:
        self._parent = parent

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._slice(key)
        if isinstance(key, int) and not isinstance(key, bool):
            return self._one(key)
        if isinstance(key, list) and all(
            isinstance(x, int) and not isinstance(x, bool) for x in key
        ):
            return self._many(key)
        if isinstance(key, pa.Array) and pa.types.is_boolean(key.type):
            raise NotImplementedError(
                "iloc[bool_array]: not supported in v0.4.0b. Use df[mask]."
            )
        raise TypeError(
            f"iloc: unsupported key type {type(key).__name__}; "
            f"expected int, slice, or list[int]"
        )

    def _view_size(self) -> int:
        p = self._parent
        if p._row_filters:
            return p.count()
        return p._dataset.total_rows

    def _resolve_int(self, i: int) -> int:
        if i < 0:
            n = self._view_size()
            i = n + i
            if i < 0:
                raise IndexError(f"iloc: index out of range (n={n})")
        return i

    def _one(self, i: int) -> dict[str, Any]:
        i = self._resolve_int(i)
        running = 0
        for b in self._parent._iter_filtered_batches():
            if running + b.num_rows > i:
                local = i - running
                return {
                    name: b.column(name)[local].as_py()
                    for name in b.schema.names
                }
            running += b.num_rows
        raise IndexError(f"iloc: position {i} out of range (n={running})")

    def _slice(self, key: slice):
        n = self._view_size()
        start, stop, step = key.indices(n)
        # Delegates to __getitem__(slice) = SliceFilter — lazy + already
        # composes with existing filters.
        return self._parent[slice(start, stop, step)]

    def _many(self, positions: list[int]):
        resolved = [self._resolve_int(p) for p in positions]
        if not resolved:
            return self._empty_view()

        targets = sorted(set(resolved))
        target_set = set(targets)

        collected: dict[int, dict[str, Any]] = {}
        running = 0
        for b in self._parent._iter_filtered_batches():
            n = b.num_rows
            batch_start = running
            batch_end = running + n
            for p in targets:
                if batch_start <= p < batch_end:
                    local = p - batch_start
                    collected[p] = {
                        name: b.column(name)[local].as_py()
                        for name in b.schema.names
                    }
            running = batch_end
            if len(collected) == len(targets):
                break

        missing = target_set - set(collected.keys())
        if missing:
            raise IndexError(
                f"iloc: positions out of range: {sorted(missing)[:5]} "
                f"(n={running})"
            )

        rows = [collected[p] for p in resolved]  # keep order + duplicates
        return self._rows_to_cadwdf(rows, resolved)

    def _empty_view(self):
        parent = self._parent
        schema = pa.schema(
            [parent._dataset.schema.field(c) for c in parent._columns]
        )
        table = pa.Table.from_batches([], schema=schema)
        return _table_to_cadwdf(table, parent, None)

    def _rows_to_cadwdf(self, rows, resolved_positions):
        parent = self._parent
        schema = pa.schema(
            [parent._dataset.schema.field(c) for c in parent._columns]
        )
        table = pa.Table.from_pylist(rows, schema=schema)
        new_index = _rebuild_index(parent, resolved_positions)
        return _table_to_cadwdf(table, parent, new_index)


def _rebuild_index(parent: "CadwDataFrame", positions: list[int]):
    """Rebuild index for iloc[list] result.

    Returns None → use default RangeIndex (filtered view or unfilterable).
    Returns MaterializedIndex → preserve parent index at positions.
    """
    from cadw.dataset.index import MaterializedIndex

    if parent._row_filters:
        return None  # filtered view → default RangeIndex (v0.4.0c: fix)

    if parent._index_kind == "range":
        vals = [parent._index_start + p for p in positions]
        return MaterializedIndex(pa.array(vals, type=pa.int64()))

    # materialized
    src = parent._index_values
    vals = [src[p] for p in positions]
    return MaterializedIndex(pa.array(vals, type=src.type))


def _table_to_cadwdf(table: pa.Table, parent: "CadwDataFrame", new_index):
    """Build a CadwDataFrame from a materialized table (+ optional index)."""
    from cadw.dataframe import CadwDataFrame
    from cadw.dataset import Dataset

    new_ds = Dataset.from_batches(
        table.schema, iter(table.to_batches()), parent._new_dataset_dir()
    )
    new = CadwDataFrame(
        new_ds,
        memory_guard=parent._memory_guard,
        spill_dir=parent._spill_dir,
    )
    if new_index is not None:
        new._index_kind = "materialized"
        new._index_values = new_index.to_arrow()
        new._index_start = 0
        new._index_stop = len(new_index)
    return new
PY

# ---------- cadw/__init__.py: не экспортируем _ILocIndexer, он внутренний ----------

# ---------- dataframe.py: import + iloc property ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# 1. Import _ILocIndexer
if "from cadw.indexers import _ILocIndexer" not in s:
    # Добавляем после from cadw.io.batch_writer
    anchor = "from cadw.io.batch_writer import ParquetBatchWriter"
    if anchor not in s:
        raise SystemExit("ParquetBatchWriter import not found")
    s = s.replace(
        anchor,
        anchor + "\nfrom cadw.indexers import _ILocIndexer",
        1,
    )
    print("dataframe.py: import _ILocIndexer")

# 2. Add iloc property after index property
anchor2 = '''    def set_index(
        self,
        col: str,
        drop: bool = True,
        inplace: bool = False,
    ) -> "CadwDataFrame":'''
if anchor2 not in s:
    raise SystemExit("set_index anchor not found")

iloc_prop = '''    @property
    def iloc(self) -> "_ILocIndexer":
        """Pandas-style positional indexer (v0.4.0b).

        - `df.iloc[5]` → dict {col: value} — 6-я строка текущего view.
        - `df.iloc[1:3]` → CadwDataFrame — позиции 1..2 (lazy, SliceFilter).
        - `df.iloc[[0, 5, 7]]` → CadwDataFrame — три строки (materialize K).

        Negative indices работают: для slice — стандартная Python semantics;
        для int/list — resolve через count() (O(N) с фильтрами, O(1) без).

        `iloc[bool_array]` — raise (use df[mask]).
        """
        return _ILocIndexer(self)

'''
s = s.replace(anchor2, iloc_prop + anchor2, 1)
print("dataframe.py: iloc property added")

p.write_text(s)
PY

# ---------- tests/test_iloc.py ----------
cat > tests/test_iloc.py <<'EOF'
"""Тесты CadwDataFrame.iloc (v0.4.0b)."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.dataset.index import MaterializedIndex, RangeIndex


def make_df(tmp_path, name="L", data=None, chunk=3):
    if data is None:
        data = {"k": ["a", "b", "c", "d", "e"], "v": [10, 20, 30, 40, 50]}
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- iloc[int] ----------

def test_iloc_int_returns_dict(tmp_path):
    df = make_df(tmp_path)
    row = df.iloc[0]
    assert isinstance(row, dict)
    assert row == {"k": "a", "v": 10}


def test_iloc_int_middle(tmp_path):
    df = make_df(tmp_path)
    assert df.iloc[2] == {"k": "c", "v": 30}


def test_iloc_int_last(tmp_path):
    df = make_df(tmp_path)
    assert df.iloc[4] == {"k": "e", "v": 50}


def test_iloc_int_negative(tmp_path):
    df = make_df(tmp_path)
    assert df.iloc[-1] == {"k": "e", "v": 50}
    assert df.iloc[-5] == {"k": "a", "v": 10}


def test_iloc_int_out_of_range(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(IndexError, match="out of range"):
        df.iloc[10]
    with pytest.raises(IndexError, match="out of range"):
        df.iloc[-10]


# ---------- iloc[int] on filtered view ----------

def test_iloc_int_on_filtered(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15]  # rows: b,c,d,e (v:20,30,40,50)
    assert filtered.iloc[0] == {"k": "b", "v": 20}
    assert filtered.iloc[2] == {"k": "d", "v": 40}
    assert filtered.iloc[-1] == {"k": "e", "v": 50}


# ---------- iloc[slice] ----------

def test_iloc_slice_returns_cadwdf(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[1:4]
    assert isinstance(sub, CadwDataFrame)
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["b", "c", "d"]


def test_iloc_slice_with_step(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[0:5:2]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["a", "c", "e"]


def test_iloc_slice_negative(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[-3:]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["c", "d", "e"]


def test_iloc_slice_on_filtered(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15]  # b,c,d,e
    sub = filtered.iloc[1:3]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["c", "d"]


# ---------- iloc[list] ----------

def test_iloc_list_returns_cadwdf(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[[0, 2, 4]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [(r["k"], r["v"]) for r in rows] == [("a", 10), ("c", 30), ("e", 50)]


def test_iloc_list_duplicates(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[[0, 0, 2]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["a", "a", "c"]


def test_iloc_list_negative(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[[-1, 0]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["e", "a"]


def test_iloc_list_out_of_range(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(IndexError, match="positions out of range"):
        df.iloc[[0, 10]]


def test_iloc_list_empty(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[[]]
    assert isinstance(sub, CadwDataFrame)
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows == []


def test_iloc_list_on_filtered(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15]  # b,c,d,e
    sub = filtered.iloc[[0, 2]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["b", "d"]


# ---------- index preservation ----------

def test_iloc_list_preserves_range_index(tmp_path):
    df = make_df(tmp_path)
    sub = df.iloc[[1, 3]]
    # Parent had RangeIndex(0,5); result has MaterializedIndex([1, 3])
    assert isinstance(sub.index, MaterializedIndex)
    assert list(sub.index) == [1, 3]


def test_iloc_list_preserves_materialized_index(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=False)
    sub = with_idx.iloc[[0, 2, 4]]
    assert isinstance(sub.index, MaterializedIndex)
    assert list(sub.index) == ["a", "c", "e"]


def test_iloc_list_on_filtered_resets_to_range(tmp_path):
    """На filtered view — default RangeIndex (index parent не сохраняется)."""
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15]
    sub = filtered.iloc[[0, 2]]
    assert isinstance(sub.index, RangeIndex)
    assert sub.index.start == 0
    assert sub.index.stop == 2


# ---------- errors ----------

def test_iloc_bool_array_raises(tmp_path):
    df = make_df(tmp_path)
    mask = pa.array([True, False, True, False, True])
    with pytest.raises(NotImplementedError, match="bool_array"):
        df.iloc[mask]


def test_iloc_unsupported_type(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="unsupported key"):
        df.iloc["hello"]


# ---------- AST guard sanity: facade clean ----------

def test_iloc_indexer_is_separate_module():
    """_ILocIndexer живёт в cadw/indexers.py, не в dataframe.py."""
    import cadw.dataframe as df_mod
    import inspect
    src = inspect.getsource(df_mod)
    assert "class _ILocIndexer" not in src, (
        "_ILocIndexer должен быть в cadw/indexers.py, не в facade"
    )
EOF

# ---------- DECISIONS.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

if "| `iloc` (v0.4.0b: positional indexer) |" not in s:
    s = s.replace(
        "| `CadwIndex`/`set_index`/`reset_index` (v0.4.0a) | ✅ | 20 |",
        "| `CadwIndex`/`set_index`/`reset_index` (v0.4.0a) | ✅ | 26 |\n"
        "| `iloc` (v0.4.0b: positional indexer) | ✅ | 21 |",
        1,
    )

if "| `iloc[bool_array]` не поддерживается (v0.4.0b)" not in s:
    s = s.replace(
        "| `df.index`/`set_index`/`reset_index` с активными фильтрами (v0.4.0a)",
        "| `iloc[bool_array]` не поддерживается (v0.4.0b) | `_ILocIndexer` | raise `NotImplementedError`; используйте `df[mask]` (RowFilter). |\n"
        "| `iloc[list]` на filtered view не сохраняет parent index (v0.4.0b) | `_ILocIndexer` | default `RangeIndex(0, K)`; preserve — v0.4.0c (filter-compat, нужна `_iter_filtered_batches_with_positions`). |\n"
        "| `df.index`/`set_index`/`reset_index` с активными фильтрами (v0.4.0a)",
        1,
    )

marker_135 = "### 13.5. Открытые вопросы"
episode_57 = """**Эпизод 57 (iloc — indexer в отдельном модуле).** `_ILocIndexer` живёт
в `cadw/indexers.py`, не в `cadw/dataframe.py`. Причина: `iloc[int]`
использует `.as_py()` для возврата dict {col: value}. AST guard
`test_facade_has_no_as_py` проверяет **весь** `dataframe.py` и упал бы.
`.as_py()` в `iloc` — O(K), K=1 или len(list); это не row-loop (не O(N)),
но guard не различает. Отдельный модуль — правильно.

**`iloc[slice]` = `df[slice]`.** Слайс-семантика для позиций в filtered
view **идентична** SliceFilter: `df[5:10]` на raw = global positions 5..9;
`df[mask][5:10]` = filtered positions 5..9. Значит `_ILocIndexer._slice`
просто делегирует `parent[slice]`. Нулевой код, полная lazy-семантика.

**`iloc[int]`** — stream-forward-count. Итерируем `_iter_filtered_batches`,
считаем строки; на позиции `i` возвращаем dict. O(i) RAM, O(i) I/O —
пользователь просил конкретную строку.

**`iloc[list]`** — stream-scan, collect target positions. O(K) memory,
O(N) I/O (нужно дойти до последней target). Результат materialize как
новый `Dataset`. Index — preserve parent (MaterializedIndex) если no-filter;
reset на filtered (v0.4.0c — filter-compat).

**Negative indices** — работают. Для slice — через `key.indices(n)`.
Для int/list — resolve через `_view_size()` (count() с фильтрами —
O(N) scan; total_rows без — O(1)). «Работает всегда, цена зависит».

**`iloc[bool_array]`** — raise `NotImplementedError`, use `df[mask]`.
Причина: у нас уже есть `MaskFilter`, `iloc[bool]` семантически =
`df[mask]`, только с другим API. Не дублировать.

**Урок:** indexers — отдельный layer, как `_GroupBy`. Facade остаётся
без data-примитивов. Каждый следующий indexer (`loc`, `.at`, `.iat`)
— сюда же. `dataframe.py` — только property, возвращающее indexer.

**Следующее (v0.4.0c — loc):**
- `MaterializedIndex.get_indexer(labels)` → positions (O(N) scan, dict).
- `iloc[list]` на filtered view — preserve parent index через
  `_iter_filtered_batches_with_positions` (для `filtered.index` тоже).
- `.at[label, col]` / `.iat[pos, col]` — scalar access.
- Duplicate labels — v0.4.1.

"""
if "Эпизод 57" not in s:
    s = s.replace(marker_135, episode_57 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §11.1, §11.5, §13.4/57")
PY

# ---------- CHANGELOG.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.4.0b] — 2026-10-19

### Добавлено
- `CadwDataFrame.iloc` — positional indexer (pandas-style).
  - `iloc[int]` → `dict[str, Any]` — одна строка.
  - `iloc[slice]` → `CadwDataFrame` — lazy, через `SliceFilter`.
  - `iloc[list[int]]` → `CadwDataFrame` — materialize K=len(list) rows.
  - Negative indices работают (int/list — resolve через count();
    slice — через `key.indices(n)`).
- `cadw/indexers.py` — модуль для indexers (`_ILocIndexer`). Отдельно
  от facade (`dataframe.py`), чтобы не ломать AST guard
  `test_facade_has_no_as_py`.

### Ограничения (v0.4.0b, §11.5)
- **`iloc[bool_array]`** — `NotImplementedError` (use `df[mask]`).
- **`iloc[list]` на filtered view** → default `RangeIndex` (не сохраняет
  parent index); v0.4.0c — filter-compat.
- **`iloc[list]` на unfiltered** — preserves parent index:
  RangeIndex → MaterializedIndex([start+p, ...]);
  MaterializedIndex → subset values (dtype preserved).
- **`iloc[list]`** — O(N) I/O scan (нужно дойти до последней позиции);
  O(K) память результата.

### Тесты
- 21 тест в `test_iloc.py`: iloc[int] (mid/last/negative/out-of-range),
  iloc[slice] (basic/step/negative/filtered), iloc[list]
  (basic/duplicates/negative/empty/out-of-range/filtered), index
  preservation (range/materialized/filtered-reset), errors
  (bool_array/unsupported type), AST guard sanity.

"""
if "## [0.4.0b]" not in s:
    s = s.replace("## [0.4.0a]", entry + "## [0.4.0a]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.4.0b")
PY

# ---------- run ----------
echo "=== pytest tests/test_iloc.py -v ==="
set +e
uv run pytest tests/test_iloc.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "iloc: rc=$RC"
  git add -A
  git commit -m "WIP: v0.4.0b iloc (test red; fix follows)" || true
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.4.0b: iloc indexer (int/slice/list; negative; index preservation; 21 tests)"

echo "==> v0.4.0b applied on ${BR}"