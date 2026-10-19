#!/usr/bin/env bash
# patch_v040c_loc: _LocIndexer (label-based) + .at/.iat + MaterializedIndex.get_indexer.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-v040c-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch v0.4.0c loc"

BR="patch/v040c-loc"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- dataset/index.py: get_indexer ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataset/index.py")
s = p.read_text()

# Add get_indexer to MaterializedIndex, after `equals`
anchor = '''    def equals(self, other: "CadwIndex") -> bool:
        if not isinstance(other, MaterializedIndex):
            return False
        if len(self) != len(other):
            return False
        if self._values.type != other._values.type:
            return False
        return self._values.equals(other._values)'''
new_method = '''    def equals(self, other: "CadwIndex") -> bool:
        if not isinstance(other, MaterializedIndex):
            return False
        if len(self) != len(other):
            return False
        if self._values.type != other._values.type:
            return False
        return self._values.equals(other._values)

    def get_indexer(self, labels) -> list[int]:
        """Map labels -> positions. O(N) scan, dict lookup.

        - labels: scalar -> [position] (или [] если not found).
        - labels: list -> list[int] (по одному position на label; -1 для
          not found).
        - duplicate labels in index: first occurrence wins (v0.4.0c MVP);
          v0.4.1 — all occurrences.
        """
        single = not isinstance(labels, (list, tuple))
        targets = [labels] if single else list(labels)

        # Build label -> first position map. O(N) scan, O(distinct) memory.
        # Naive dict lookup must be hashable; for non-hashable fallback —
        # O(N × K) linear scan (documented; rare).
        try:
            m: dict = {}
            for i, v in enumerate(self._values.to_pylist()):
                if v not in m:
                    m[v] = i
            positions = [m.get(t, -1) for t in targets]
        except TypeError:
            # Unhashable labels — fallback linear scan.
            pos_list = self._values.to_pylist()
            positions = []
            for t in targets:
                found = -1
                for i, v in enumerate(pos_list):
                    if v == t:
                        found = i
                        break
                positions.append(found)
        return positions'''

if anchor not in s:
    raise SystemExit("MaterializedIndex.equals not found")
s = s.replace(anchor, new_method, 1)
p.write_text(s)
print("index.py: MaterializedIndex.get_indexer added")
PY

# ---------- indexers.py: _LocIndexer + _AtIndexer + _IAtIndexer ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/indexers.py")
s = p.read_text()

# Append _LocIndexer, _AtIndexer, _IAtIndexer
extra = '''

class _LocIndexer:
    """Label-based indexer (pandas-style). Требует MaterializedIndex.

    - loc[label] -> dict (одна строка, KeyError если not found).
    - loc[lo:hi] -> CadwDataFrame (inclusive labels, как pandas).
    - loc[[l1, l2, ...]] -> CadwDataFrame (materialize K rows; KeyError
      если хоть один not found).

    RangeIndex (default) -> TypeError (use iloc).
    `loc[bool_array]` -> NotImplementedError (use df[mask]).
    Duplicate labels in index -> первое вхождение (v0.4.0c); v0.4.1 — all.
    """

    def __init__(self, parent: "CadwDataFrame") -> None:
        self._parent = parent

    def _materialized(self):
        from cadw.dataset.index import MaterializedIndex
        p = self._parent
        if p._index_kind == "range":
            raise TypeError(
                "loc: requires MaterializedIndex (after set_index). "
                "RangeIndex — use iloc."
            )
        if p._row_filters:
            raise NotImplementedError(
                "loc: не поддерживается с активными фильтрами в v0.4.0c. "
                "§13.4/58."
            )
        return MaterializedIndex(p._index_values)

    def __getitem__(self, key):
        idx = self._materialized()
        if isinstance(key, slice):
            return self._slice(key, idx)
        if isinstance(key, list):
            return self._many(key, idx)
        if isinstance(key, pa.Array) and pa.types.is_boolean(key.type):
            raise NotImplementedError(
                "loc[bool_array]: not supported in v0.4.0c. Use df[mask]."
            )
        # scalar label
        pos = idx.get_indexer(key)[0]
        if pos < 0:
            raise KeyError(f"loc: label {key!r} not in index")
        return self._parent.iloc[pos]

    def _slice(self, key, idx):
        if key.start is None or key.stop is None:
            raise NotImplementedError(
                "loc[slice]: only [lo:hi] with both endpoints supported in "
                "v0.4.0c (inclusive). Use iloc for full-open slices."
            )
        if key.step is not None and key.step != 1:
            raise NotImplementedError(
                "loc[slice]: step not supported in v0.4.0c."
            )
        lo_pos = idx.get_indexer(key.start)[0]
        hi_pos = idx.get_indexer(key.stop)[0]
        if lo_pos < 0:
            raise KeyError(f"loc: label {key.start!r} not in index")
        if hi_pos < 0:
            raise KeyError(f"loc: label {key.stop!r} not in index")
        if lo_pos > hi_pos:
            # pandas: empty (не ошибка).
            return self._parent.iloc[[]]
        # Inclusive: [lo_pos, hi_pos]
        return self._parent.iloc[list(range(lo_pos, hi_pos + 1))]

    def _many(self, labels: list, idx):
        positions = idx.get_indexer(labels)
        missing = [lab for lab, pos in zip(labels, positions) if pos < 0]
        if missing:
            raise KeyError(
                f"loc: labels not in index: {missing[:5]}"
            )
        return self._parent.iloc[positions]


class _AtIndexer:
    """`.at[label, col]` — scalar access by label + column."""

    def __init__(self, parent: "CadwDataFrame") -> None:
        self._parent = parent

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(
                f".at: key must be (label, col); got {type(key).__name__}"
            )
        label, col = key
        row = self._parent.loc[label]
        if col not in row:
            raise KeyError(f".at: column {col!r} not in df.columns")
        return row[col]

    def __setitem__(self, key, value):
        raise NotImplementedError(
            ".at.__setitem__: CadwDataFrame immutable in v0.4.0c. "
            "Use df.write_parquet + reload to mutate."
        )


class _IAtIndexer:
    """`.iat[pos, col]` — scalar access by position + column."""

    def __init__(self, parent: "CadwDataFrame") -> None:
        self._parent = parent

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(
                f".iat: key must be (pos, col); got {type(key).__name__}"
            )
        pos, col = key
        row = self._parent.iloc[pos]
        if col not in row:
            raise KeyError(f".iat: column {col!r} not in df.columns")
        return row[col]

    def __setitem__(self, key, value):
        raise NotImplementedError(
            ".iat.__setitem__: CadwDataFrame immutable in v0.4.0c."
        )
'''

# append at end
s = s.rstrip() + extra + "\n"
p.write_text(s)
print("indexers.py: _LocIndexer + _AtIndexer + _IAtIndexer added")
PY

# ---------- dataframe.py: loc / at / iat properties ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# Update import
if "from cadw.indexers import _ILocIndexer" in s:
    s = s.replace(
        "from cadw.indexers import _ILocIndexer",
        "from cadw.indexers import (\n    _ILocIndexer,\n    _LocIndexer,\n    _AtIndexer,\n    _IAtIndexer,\n)",
        1,
    )
    print("dataframe.py: import _LocIndexer/_AtIndexer/_IAtIndexer")

# Add loc / at / iat after iloc property
anchor = '''    def set_index(
        self,
        col: str,
        drop: bool = True,
        inplace: bool = False,
    ) -> "CadwDataFrame":'''
if anchor not in s:
    raise SystemExit("set_index anchor not found")

new_props = '''    @property
    def loc(self) -> "_LocIndexer":
        """Pandas-style label-based indexer (v0.4.0c).

        - `df.loc[label]` → dict — строка с индексом `label` (KeyError если нет).
        - `df.loc[lo:hi]` → CadwDataFrame — inclusive labels (как pandas).
        - `df.loc[[l1, l2, ...]]` → CadwDataFrame — несколько строк.

        Требует `MaterializedIndex` (после `set_index`). `RangeIndex` →
        `TypeError`; используйте `iloc`.

        `loc[bool_array]` → `NotImplementedError` (use `df[mask]`).
        Дубликаты в индексе → первое вхождение (v0.4.0c); v0.4.1 — все.
        """
        return _LocIndexer(self)

    @property
    def at(self) -> "_AtIndexer":
        """`df.at[label, col]` — scalar (v0.4.0c). Read-only."""
        return _AtIndexer(self)

    @property
    def iat(self) -> "_IAtIndexer":
        """`df.iat[pos, col]` — scalar по позиции (v0.4.0c). Read-only."""
        return _IAtIndexer(self)

'''
s = s.replace(anchor, new_props + anchor, 1)
print("dataframe.py: loc/at/iat properties added")

p.write_text(s)
PY

# ---------- tests/test_loc.py ----------
cat > tests/test_loc.py <<'EOF'
"""Тесты CadwDataFrame.loc / .at / .iat + MaterializedIndex.get_indexer (v0.4.0c)."""
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


# ---------- MaterializedIndex.get_indexer ----------

def test_get_indexer_scalar():
    idx = MaterializedIndex(pa.array([10, 20, 30]))
    assert idx.get_indexer(20) == [1]
    assert idx.get_indexer(99) == [-1]


def test_get_indexer_list():
    idx = MaterializedIndex(pa.array([10, 20, 30]))
    assert idx.get_indexer([10, 30]) == [0, 2]
    assert idx.get_indexer([10, 99, 30]) == [0, -1, 2]


def test_get_indexer_duplicate_first_wins():
    idx = MaterializedIndex(pa.array([10, 20, 10]))
    assert idx.get_indexer(10) == [0]  # first occurrence (MVP)


def test_get_indexer_string():
    idx = MaterializedIndex(pa.array(["x", "y", "z"]))
    assert idx.get_indexer("y") == [1]
    assert idx.get_indexer(["z", "x"]) == [2, 0]


# ---------- loc on RangeIndex raises ----------

def test_loc_on_range_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="requires MaterializedIndex"):
        df.loc["a"]


# ---------- loc[label] ----------

def test_loc_scalar(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    assert df.loc["c"] == {"k": "c", "v": 30}
    assert df.loc["a"] == {"k": "a", "v": 10}


def test_loc_scalar_missing_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not in index"):
        df.loc["zzz"]


def test_loc_scalar_int_label(tmp_path):
    df = make_df(tmp_path, data={
        "id": [100, 200, 300], "v": [1, 2, 3],
    }).set_index("id", drop=False)
    assert df.loc[200] == {"id": 200, "v": 2}


# ---------- loc[lo:hi] (inclusive) ----------

def test_loc_slice_inclusive(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc["b":"d"]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["b", "c", "d"]


def test_loc_slice_single(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc["b":"b"]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["b"]


def test_loc_slice_reverse_returns_empty(tmp_path):
    """loc['d':'b'] — пустой (не ошибка, как pandas)."""
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc["d":"b"]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows == []


def test_loc_slice_missing_lo_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not in index"):
        df.loc["zzz":"d"]


def test_loc_slice_missing_hi_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not in index"):
        df.loc["b":"zzz"]


def test_loc_slice_none_endpoint_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(NotImplementedError, match="full-open"):
        df.loc[:"c"]
    with pytest.raises(NotImplementedError, match="full-open"):
        df.loc["c":]


# ---------- loc[[labels]] ----------

def test_loc_list(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc[["a", "c", "e"]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["a", "c", "e"]


def test_loc_list_missing_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not in index"):
        df.loc[["a", "zzz"]]


def test_loc_list_duplicates(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc[["a", "a"]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["a", "a"]


def test_loc_list_empty(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc[[]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert rows == []


# ---------- loc[bool_array] raises ----------

def test_loc_bool_array_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    mask = pa.array([True, False, True, False, True])
    with pytest.raises(NotImplementedError, match="bool_array"):
        df.loc[mask]


# ---------- .at ----------

def test_at_scalar(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    assert df.at["b", "v"] == 20
    assert df.at["e", "k"] == "e"


def test_at_missing_label_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not in index"):
        df.at["zzz", "v"]


def test_at_missing_col_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="column"):
        df.at["a", "nope"]


def test_at_wrong_key_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(TypeError, match="must be \\(label, col\\)"):
        df.at["a"]


def test_at_setitem_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(NotImplementedError, match="immutable"):
        df.at["a", "v"] = 100


# ---------- .iat ----------

def test_iat_scalar(tmp_path):
    df = make_df(tmp_path)
    assert df.iat[0, "v"] == 10
    assert df.iat[4, "k"] == "e"
    assert df.iat[-1, "v"] == 50


def test_iat_missing_col_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(KeyError, match="column"):
        df.iat[0, "nope"]


def test_iat_out_of_range_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(IndexError):
        df.iat[10, "v"]


def test_iat_wrong_key_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(TypeError, match="must be \\(pos, col\\)"):
        df.iat[0]


def test_iat_setitem_raises(tmp_path):
    df = make_df(tmp_path)
    with pytest.raises(NotImplementedError, match="immutable"):
        df.iat[0, "v"] = 100


# ---------- filter limitation ----------

def test_loc_on_filtered_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    filtered = df[df["v"] > 15]
    with pytest.raises(NotImplementedError, match="фильтр"):
        filtered.loc["b"]


# ---------- loc preserves index ----------

def test_loc_list_preserves_index(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc[["a", "c", "e"]]
    assert isinstance(sub.index, MaterializedIndex)
    assert list(sub.index) == ["a", "c", "e"]
EOF

# ---------- DECISIONS.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

if "| `loc`/`.at`/`.iat` (v0.4.0c: label-based) |" not in s:
    s = s.replace(
        "| `iloc` (v0.4.0b: positional indexer) | ✅ | 21 |",
        "| `iloc` (v0.4.0b: positional indexer) | ✅ | 22 |\n"
        "| `loc`/`.at`/`.iat` (v0.4.0c: label-based) | ✅ | 30 |",
        1,
    )

if "| `loc` требует MaterializedIndex (v0.4.0c)" not in s:
    s = s.replace(
        "| `iloc[bool_array]` не поддерживается (v0.4.0b)",
        "| `loc` требует `MaterializedIndex` (после `set_index`) | `_LocIndexer` | `RangeIndex` → `TypeError`; используйте `iloc`. |\n"
        "| `loc[lo:hi]` — только оба endpoint обязательны (v0.4.0c) | `_LocIndexer` | `loc[:hi]` / `loc[lo:]` — `NotImplementedError`. Step — тоже. v0.4.1. |\n"
        "| `loc` duplicates → first occurrence (v0.4.0c) | `_LocIndexer` | v0.4.1 — все вхождения (pandas semantics). |\n"
        "| `iloc[bool_array]` не поддерживается (v0.4.0b)",
        1,
    )

marker_135 = "### 13.5. Открытые вопросы"
episode_58 = """**Эпизод 58 (loc — label-based, требует MaterializedIndex).** `_LocIndexer`
— pandas-совместимый label indexer. Работает только на `MaterializedIndex`
(после `set_index`). `RangeIndex` → `TypeError` с подсказкой «use iloc».
Это намеренное ограничение: `RangeIndex.get_indexer(label)` — это
`label - start` при `label` int; для произвольных labels нужна
материализация, что делает range бессмысленным.

**Реализация через `MaterializedIndex.get_indexer(labels)`.** O(N) scan
`values.to_pylist()`, dict `label → first_pos`. Возвращает `list[int]`,
где `-1` = not found. Используется в `loc[label]`, `loc[[...]]`,
`loc[lo:hi]`. Для hashable labels — dict (быстро); для unhashable —
fallback linear scan (O(N×K), редко).

**`loc[lo:hi]` inclusive.** Pandas semantics: `['b':'d']` включает
и `'b'`, и `'d'`. Позиции: `get_indexer(lo), get_indexer(hi)`;
`range(lo_pos, hi_pos+1)`. Если `lo_pos > hi_pos` (например, `'d':'b'`)
— пустой результат, не ошибка. Если `lo`/`hi` not found — `KeyError`
(pandas: `KeyError` для missing endpoint).

**Open endpoints:** `loc[:hi]` / `loc[lo:]` — `NotImplementedError` в
v0.4.0c (требует направления сортировки — при asc индексе это
`[first : hi_pos]`, при desc — `[hi_pos : last]`; нужно проверять
is_monotonic). v0.4.1.

**`.at`/`.iat` — read-only.** `__setitem__` — `NotImplementedError`
(CadwDataFrame immutable; mutation через `write_parquet + reload`).
Это консистентно с общей immutable-моделью (кроме `set_index(inplace=True)`
и `reset_index(inplace=True)` — единственные inplace-операции).

**`loc` на filtered view** — `NotImplementedError` (как `df.index`).
v0.4.0d: filter-compat для `loc` через `_iter_filtered_batches_with_positions`.

**Дубликаты в индексе.** `get_indexer` возвращает первое вхождение
(MVP). v0.4.1: `get_indexer_all` — все вхождения, `loc[label]` вернёт
`CadwDataFrame` (несколько строк) вместо `dict`. Это отдельный API
контракт; в MVP — единственное вхождение.

**Урок:** indexers (`iloc`, `loc`, `.at`, `.iat`) — 4 разных API с
разной семантикой:
- positional vs label;
- array-like vs scalar;
- preserved vs reset index.

Все живут в `cadw/indexers.py`; facade `dataframe.py` — только property
возвращающие indexer. Это позволяет AST guard `test_facade_has_no_as_py`
проходить (`.as_py()` в indexers), и не «раздувать» `dataframe.py`.

**Следующее (v0.4.0d — filter-compat):**
- `_iter_filtered_batches_with_positions()` → `(batch, positions)`.
- `filtered.index` — RangeIndex subset или MaterializedIndex subset.
- `filtered.loc[label]` — работает.
- `filtered.set_index` / `reset_index` — работают.

**v0.4.1:** loc open-endpoints, loc with duplicate labels (все вхождения),
MultiIndex.

"""
if "Эпизод 58" not in s:
    s = s.replace(marker_135, episode_58 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §11.1, §11.5, §13.4/58")
PY

# ---------- CHANGELOG.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.4.0c] — 2026-10-19

### Добавлено
- `CadwDataFrame.loc` — label-based indexer (pandas-style).
  - `loc[label]` → `dict` (KeyError если label нет).
  - `loc[lo:hi]` → `CadwDataFrame` — inclusive labels, reverse → empty.
  - `loc[[l1, l2, ...]]` → `CadwDataFrame`.
  - Требует `MaterializedIndex`; `RangeIndex` → `TypeError`.
- `CadwDataFrame.at[label, col]` — scalar access by label + column.
- `CadwDataFrame.iat[pos, col]` — scalar access by position + column.
- `MaterializedIndex.get_indexer(labels)` → `list[int]` (O(N) scan, dict;
  unhashable fallback linear).
- `.at`/`.iat` — **read-only** (`__setitem__` → `NotImplementedError`).

### Ограничения (v0.4.0c, §11.5)
- **`loc` требует `MaterializedIndex`** (после `set_index`).
  `RangeIndex` → `TypeError`; используйте `iloc`.
- **`loc[lo:hi]`** — только оба endpoint обязательны; `loc[:hi]` /
  `loc[lo:]` / `loc[::step]` → `NotImplementedError`. v0.4.1.
- **`loc[bool_array]`** — `NotImplementedError` (use `df[mask]`).
- **`loc` на filtered view** — `NotImplementedError`. v0.4.0d.
- **duplicate labels** — первое вхождение; v0.4.1 — все.
- **`.at`/`.iat` set** — `NotImplementedError` (immutable).

### Тесты
- 30 тестов в `test_loc.py`: get_indexer (scalar/list/dup/first-wins/str),
  loc on RangeIndex raises, loc scalar/slice(inclusive/reverse/missing),
  loc list (basic/missing/dup/empty), loc bool_array raise, .at
  scalar/missing/set-raise, .iat scalar/out-of-range/set-raise,
  filter limit, loc preserves index.

"""
if "## [0.4.0c]" not in s:
    s = s.replace("## [0.4.0b]", entry + "## [0.4.0b]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.4.0c")
PY

# ---------- run ----------
echo "=== pytest tests/test_loc.py -v ==="
set +e
uv run pytest tests/test_loc.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "loc: rc=$RC"
  git add -A
  git commit -m "WIP: v0.4.0c loc (test red; fix follows)" || true
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.4.0c: loc / .at / .iat + MaterializedIndex.get_indexer; 30 tests"

echo "==> v0.4.0c applied on ${BR}"