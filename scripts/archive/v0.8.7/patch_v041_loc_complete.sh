#!/usr/bin/env bash
# patch_v041_loc_complete: loc open-endpoints + duplicate labels + monotonic API.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-v041-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch v0.4.1 loc-complete"

BR="patch/v041-loc-complete"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- 1. dataset/index.py: monotonic + duplicates + get_positions_all ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataset/index.py")
s = p.read_text()

anchor = '''    def get_indexer(self, labels) -> list[int]:'''
if anchor not in s:
    raise SystemExit("get_indexer not found")

new_methods = '''    def is_monotonic_increasing(self) -> bool:
        """True iff values (skipping nulls) are non-decreasing."""
        v = self._values.to_pylist()
        prev = None
        first = True
        for x in v:
            if x is None:
                continue
            if first:
                prev = x
                first = False
                continue
            try:
                if x < prev:
                    return False
            except TypeError:
                return False
            prev = x
        return True

    def is_monotonic_decreasing(self) -> bool:
        """True iff values (skipping nulls) are non-increasing."""
        v = self._values.to_pylist()
        prev = None
        first = True
        for x in v:
            if x is None:
                continue
            if first:
                prev = x
                first = False
                continue
            try:
                if x > prev:
                    return False
            except TypeError:
                return False
            prev = x
        return True

    def is_monotonic(self) -> bool:
        """True iff increasing OR decreasing (or length <= 1)."""
        if len(self) <= 1:
            return True
        return self.is_monotonic_increasing() or self.is_monotonic_decreasing()

    def has_duplicates(self) -> bool:
        """True iff at least one label appears more than once."""
        v = self._values.to_pylist()
        try:
            return len(set(v)) < len(v)
        except TypeError:
            seen: list = []
            for x in v:
                if x in seen:
                    return True
                seen.append(x)
            return False

    def get_positions_all(self, label) -> list[int]:
        """All 0-based positions matching `label` (v0.4.1).

        O(N) scan, `==` comparison. For non-hashable labels — same cost.
        Возвращает [] если не найдено (проверка на caller).
        """
        v = self._values.to_pylist()
        positions: list[int] = []
        for i, x in enumerate(v):
            try:
                if x == label:
                    positions.append(i)
            except TypeError:
                pass
        return positions

    def get_indexer(self, labels) -> list[int]:'''

s = s.replace(anchor, new_methods, 1)
p.write_text(s)
print("index.py: is_monotonic_*/has_duplicates/get_positions_all added")
PY

# ---------- 2. indexers.py: rewrite _LocIndexer + _AtIndexer ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/indexers.py")
s = p.read_text()

# 2a. Replace entire _LocIndexer class
loc_start = s.find("class _LocIndexer:")
if loc_start < 0:
    raise SystemExit("_LocIndexer class not found")
# find end: next "\n\nclass _AtIndexer" or end of file
loc_end = s.find("\n\nclass _AtIndexer", loc_start)
if loc_end < 0:
    loc_end = len(s)

new_loc = '''class _LocIndexer:
    """Label-based indexer (pandas-style). Требует MaterializedIndex.

    v0.4.1: open endpoints (`loc[:hi]`, `loc[lo:]`, `loc[:]`) на monotonic
    index (asc или desc). Duplicate labels: `loc[label]` → CadwDataFrame
    (несколько строк) вместо dict. `loc[slice]` требует уникального
    индекса (pandas: cannot slice non-unique) и монотонности.
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
            idx = p.index
            if not isinstance(idx, MaterializedIndex):
                raise TypeError(
                    f"loc: filtered index is {type(idx).__name__}, "
                    f"expected MaterializedIndex"
                )
            return idx
        assert p._index_values is not None
        return MaterializedIndex(p._index_values)

    def __getitem__(self, key):
        idx = self._materialized()
        if isinstance(key, slice):
            return self._slice(key, idx)
        if isinstance(key, list):
            return self._many(key, idx)
        if isinstance(key, pa.Array) and pa.types.is_boolean(key.type):
            raise NotImplementedError(
                "loc[bool_array]: not supported. Use df[mask]."
            )
        # scalar label — duplicates → CadwDataFrame; unique → dict.
        positions = idx.get_positions_all(key)
        if not positions:
            raise KeyError(f"loc: label {key!r} not in index")
        if len(positions) == 1:
            return self._parent.iloc[positions[0]]
        return self._parent.iloc[positions]

    def _slice(self, key, idx):
        if key.step is not None and key.step != 1:
            raise NotImplementedError(
                "loc[slice]: step not supported in v0.4.1."
            )
        if idx.has_duplicates():
            raise NotImplementedError(
                "loc[slice]: requires unique index in v0.4.1. "
                "Duplicate labels — use loc[list_of_labels]."
            )

        n = len(idx)
        if key.start is None and key.stop is None:
            # loc[:] — все строки (lazy view).
            return self._parent[slice(0, n)]

        pl = None
        ph = None
        if key.start is not None:
            pl = idx.get_indexer(key.start)[0]
            if pl < 0:
                raise KeyError(f"loc: label {key.start!r} not in index")
        if key.stop is not None:
            ph = idx.get_indexer(key.stop)[0]
            if ph < 0:
                raise KeyError(f"loc: label {key.stop!r} not in index")

        if pl is not None and ph is not None:
            # explicit [lo:hi]
            if idx.is_monotonic_increasing():
                if pl <= ph:
                    return self._parent.iloc[list(range(pl, ph + 1))]
                return self._parent.iloc[[]]
            if idx.is_monotonic_decreasing():
                if ph <= pl:
                    return self._parent.iloc[list(range(ph, pl + 1))]
                return self._parent.iloc[[]]
            raise NotImplementedError(
                "loc[slice]: requires monotonic index (incr or decr)."
            )

        if pl is None:
            # [ : hi] — from start up to hi (inclusive), any monotonic dir.
            if not idx.is_monotonic():
                raise NotImplementedError(
                    "loc[slice]: requires monotonic index (incr or decr)."
                )
            return self._parent.iloc[list(range(0, ph + 1)))

        # [lo : ] — from lo to end.
        if not idx.is_monotonic():
            raise NotImplementedError(
                "loc[slice]: requires monotonic index (incr or decr)."
            )
        return self._parent.iloc[list(range(pl, n)))

    def _many(self, labels: list, idx):
        all_positions: list[int] = []
        missing: list = []
        for lab in labels:
            pos = idx.get_positions_all(lab)
            if not pos:
                missing.append(lab)
            else:
                all_positions.extend(pos)
        if missing:
            raise KeyError(f"loc: labels not in index: {missing[:5]}")
        return self._parent.iloc[all_positions]
'''

s = s[:loc_start] + new_loc + s[loc_end:]
print("indexers.py: _LocIndexer rewritten (open endpoints + duplicates)")

# 2b. Update _AtIndexer — explicit uniqueness check for .at[label, col]
old_at = '''class _AtIndexer:
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
        return row[col]'''

new_at = '''class _AtIndexer:
    """`.at[label, col]` — scalar access by label + column (v0.4.1)."""

    def __init__(self, parent: "CadwDataFrame") -> None:
        self._parent = parent

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(
                f".at: key must be (label, col); got {type(key).__name__}"
            )
        label, col = key
        row = self._parent.loc[label]
        if not isinstance(row, dict):
            raise KeyError(
                f".at: label {label!r} is not unique in index; "
                f"use .loc[{label!r}] for multiple rows"
            )
        if col not in row:
            raise KeyError(f".at: column {col!r} not in df.columns")
        return row[col]'''

if old_at not in s:
    raise SystemExit("_AtIndexer.__getitem__ not found")
s = s.replace(old_at, new_at, 1)
print("indexers.py: _AtIndexer — explicit unique check")

p.write_text(s)
print("indexers.py: готово")
PY

# ---------- 3. tests/test_loc.py: replace old endpoint test + new advanced tests ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_loc.py")
s = p.read_text()

# 3a. Replace test_loc_slice_none_endpoint_raises (now open endpoints work)
old_test = '''def test_loc_slice_none_endpoint_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(NotImplementedError, match="full-open"):
        df.loc[:"c"]
    with pytest.raises(NotImplementedError, match="full-open"):
        df.loc["c":]'''

new_tests = '''def test_loc_slice_open_start(tmp_path):
    """v0.4.1: loc[:'c'] — от начала до c включительно."""
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc[:"c"]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["a", "b", "c"]


def test_loc_slice_open_stop(tmp_path):
    """v0.4.1: loc['c':] — от c до конца."""
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc["c":]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["c", "d", "e"]


def test_loc_slice_open_both(tmp_path):
    """v0.4.1: loc[:] — все строки."""
    df = make_df(tmp_path).set_index("k", drop=False)
    sub = df.loc[:]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["a", "b", "c", "d", "e"]


def test_loc_slice_open_start_missing_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not in index"):
        df.loc[:"zzz"]


def test_loc_slice_open_stop_missing_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not in index"):
        df.loc["zzz":]


def test_loc_slice_descending(tmp_path):
    """v0.4.1: descending monotonic index — loc['b':'d'] returns d,c,b."""
    df = make_df(tmp_path, data={
        "k": ["e", "d", "c", "b", "a"],
        "v": [50, 40, 30, 20, 10],
    }).set_index("k", drop=False)
    # descending index ['e','d','c','b','a']
    sub = df.loc["b":"d"]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["d", "c", "b"]


def test_loc_slice_descending_open(tmp_path):
    """descending + open: loc[:'c'] → ['e','d','c']; loc['c':] → ['c','b','a']."""
    df = make_df(tmp_path, data={
        "k": ["e", "d", "c", "b", "a"],
        "v": [50, 40, 30, 20, 10],
    }).set_index("k", drop=False)
    sub1 = df.loc[:"c"]
    sub2 = df.loc["c":]
    out1 = tmp_path / "o1.parquet"
    out2 = tmp_path / "o2.parquet"
    sub1.write_parquet(out1)
    sub2.write_parquet(out2)
    assert [r["k"] for r in pq.read_table(out1).to_pylist()] == ["e", "d", "c"]
    assert [r["k"] for r in pq.read_table(out2).to_pylist()] == ["c", "b", "a"]


def test_loc_slice_non_monotonic_raises(tmp_path):
    """Не monotonic index → NotImplementedError на slice."""
    df = make_df(tmp_path, data={
        "k": ["c", "a", "b"],
        "v": [30, 10, 20],
    }).set_index("k", drop=False)
    with pytest.raises(NotImplementedError, match="monotonic"):
        df.loc["a":"b"]


# ---------- v0.4.1: duplicate labels ----------

def test_loc_scalar_unique_returns_dict(tmp_path):
    """Уникальный index: loc[label] → dict."""
    df = make_df(tmp_path).set_index("k", drop=False)
    row = df.loc["c"]
    assert isinstance(row, dict)
    assert row == {"k": "c", "v": 30}


def test_loc_scalar_duplicate_returns_cadwdf(tmp_path):
    """Duplicate labels: loc[label] → CadwDataFrame (pandas)."""
    df = make_df(tmp_path, data={
        "k": ["a", "a", "b", "c"],
        "v": [10, 20, 30, 40],
    }).set_index("k", drop=False)
    sub = df.loc["a"]
    assert isinstance(sub, CadwDataFrame)
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [(r["k"], r["v"]) for r in rows] == [("a", 10), ("a", 20)]


def test_loc_scalar_duplicate_three_matches(tmp_path):
    df = make_df(tmp_path, data={
        "k": ["x", "y", "x", "z", "x"],
        "v": [1, 2, 3, 4, 5],
    }).set_index("k", drop=False)
    sub = df.loc["x"]
    assert isinstance(sub, CadwDataFrame)
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["v"] for r in rows] == [1, 3, 5]


def test_loc_list_with_duplicates(tmp_path):
    """loc[['a']] на duplicate index → все вхождения 'a'."""
    df = make_df(tmp_path, data={
        "k": ["a", "a", "b"],
        "v": [1, 2, 3],
    }).set_index("k", drop=False)
    sub = df.loc[["a"]]
    out = tmp_path / "o.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [(r["k"], r["v"]) for r in rows] == [("a", 1), ("a", 2)]


def test_loc_slice_with_duplicates_raises(tmp_path):
    """loc[slice] на duplicate index → NotImplementedError (pandas semantics)."""
    df = make_df(tmp_path, data={
        "k": ["a", "a", "b"],
        "v": [1, 2, 3],
    }).set_index("k", drop=False)
    with pytest.raises(NotImplementedError, match="unique index"):
        df.loc["a":"b"]


def test_at_duplicate_label_raises(tmp_path):
    """`.at[label, col]` на duplicate label → KeyError (honest)."""
    df = make_df(tmp_path, data={
        "k": ["a", "a", "b"],
        "v": [1, 2, 3],
    }).set_index("k", drop=False)
    with pytest.raises(KeyError, match="not unique"):
        df.at["a", "v"]


# ---------- v0.4.1: MaterializedIndex monotonic / has_duplicates / get_positions_all ----------

def test_index_monotonic_increasing():
    idx = MaterializedIndex(pa.array([1, 2, 3, 4]))
    assert idx.is_monotonic_increasing()
    assert not idx.is_monotonic_decreasing()
    assert idx.is_monotonic()


def test_index_monotonic_decreasing():
    idx = MaterializedIndex(pa.array([4, 3, 2, 1]))
    assert not idx.is_monotonic_increasing()
    assert idx.is_monotonic_decreasing()
    assert idx.is_monotonic()


def test_index_not_monotonic():
    idx = MaterializedIndex(pa.array([1, 3, 2, 4]))
    assert not idx.is_monotonic_increasing()
    assert not idx.is_monotonic_decreasing()
    assert not idx.is_monotonic()


def test_index_monotonic_single():
    idx = MaterializedIndex(pa.array([5]))
    assert idx.is_monotonic_increasing()
    assert idx.is_monotonic_decreasing()
    assert idx.is_monotonic()


def test_index_has_duplicates_yes():
    idx = MaterializedIndex(pa.array(["a", "b", "a"]))
    assert idx.has_duplicates()


def test_index_has_duplicates_no():
    idx = MaterializedIndex(pa.array(["a", "b", "c"]))
    assert not idx.has_duplicates()


def test_get_positions_all_single():
    idx = MaterializedIndex(pa.array([10, 20, 30]))
    assert idx.get_positions_all(20) == [1]


def test_get_positions_all_multiple():
    idx = MaterializedIndex(pa.array(["a", "b", "a", "c", "a"]))
    assert idx.get_positions_all("a") == [0, 2, 4]


def test_get_positions_all_missing():
    idx = MaterializedIndex(pa.array([10, 20, 30]))
    assert idx.get_positions_all(99) == []'''

if old_test not in s:
    raise SystemExit("test_loc_slice_none_endpoint_raises not found")
s = s.replace(old_test, new_tests, 1)
p.write_text(s)
print("test_loc.py: open endpoints + duplicates + monotonic tests added")
PY

# ---------- 4. DECISIONS.md: episode 61 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

if "| `loc` open-endpoints + duplicate labels (v0.4.1) |" not in s:
    s = s.replace(
        "| filter-compat index/set/reset (v0.4.0d) | ✅ | 15 |",
        "| filter-compat index/set/reset (v0.4.0d) | ✅ | 16 |\n"
        "| `loc` open-endpoints + duplicate labels (v0.4.1) | ✅ | 22 |",
        1,
    )

# Update §11.5 - remove resolved limits
s = s.replace(
    "| `loc[lo:hi]` — только оба endpoint обязательны (v0.4.0c) | `_LocIndexer` | `loc[:hi]` / `loc[lo:]` — `NotImplementedError`. Step — тоже. v0.4.1. |\n",
    "",
    1,
)
s = s.replace(
    "| `loc` duplicates → first occurrence (v0.4.0c) | `_LocIndexer` | v0.4.1 — все вхождения (pandas semantics). |\n",
    "",
    1,
)

if "| `loc[slice]` на duplicate index (v0.4.1)" not in s:
    s = s.replace(
        "| `loc` требует `MaterializedIndex` (после `set_index`) |",
        "| `loc[slice]` на duplicate index — `NotImplementedError` (v0.4.1) | `_LocIndexer` | pandas: cannot slice non-unique; v0.4.2 — MultiIndex semantics. |\n"
        "| `loc[slice]` step != 1 — `NotImplementedError` (v0.4.1) | `_LocIndexer` | v0.4.2. |\n"
        "| `.at` на duplicate label — `KeyError` (v0.4.1) | `_AtIndexer` | pandas: undefined для duplicates; используйте `loc[[label]]`. |\n"
        "| `loc` требует `MaterializedIndex` (после `set_index`) |",
        1,
    )

marker_135 = "### 13.5. Открытые вопросы"
episode_61 = """**Эпизод 61 (loc complete: open endpoints + duplicates).** v0.4.1 закрывает
последние дыры в `loc`:

1. **Open endpoints.** `loc[:hi]`, `loc[lo:]`, `loc[:]` — работают на
   monotonic index (asc или desc). Pandas использует `slice_locs`, который
   понимает направление; мы делаем проще:
   - `loc[:hi]` → `positions[0 : get_indexer(hi) + 1]` (для любого
     монотонного направления).
   - `loc[lo:]` → `positions[get_indexer(lo) : end]`.
   - `loc[lo:hi]` explicit — для **increasing** `pl <= ph` → `positions[pl:ph+1]`,
     иначе empty; для **decreasing** `ph <= pl` → `positions[ph:pl+1]`,
     иначе empty. Reverse ranges → empty (pandas semantics).
   - Non-monotonic → `NotImplementedError` (pandas 2.x тоже).

2. **Duplicate labels.** `loc[label]`:
   - unique → dict (1 строка);
   - duplicate → `CadwDataFrame` (все вхождения).
   `loc[[labels]]` — все вхождения для каждого label, конкатенированные в
   original order.
   `loc[lo:hi]` на duplicate → `NotImplementedError` (pandas: cannot slice
   non-unique). Step — тоже.
   `.at[label, col]` на duplicate → `KeyError` (pandas: undefined).

**`MaterializedIndex.is_monotonic_*`** — O(N) scan, пропускает nulls.
`has_duplicates()` — set-based, O(N). `get_positions_all(label)` — O(N),
`==` по значениям (не hash).

**Pandas-семантика reverse range:** `df.loc['d':'b']` на ascending index
→ empty (не ошибка). Это уже было в v0.4.0c, теперь тестируется явно.
На descending `df.loc['b':'d']` → `['d','c','b']` (по позиции в index).
Это **общий** принцип: `loc[lo:hi]` на монотонном индексе возвращает
позиции между `pl` и `ph`, упорядоченные по index direction.

**Duplicate label → type change (`dict` → `CadwDataFrame`).** Pandas
делает то же (Series → DataFrame). Это может ломать code, который
делает `df.loc[label]['col']` — для unique это OK, для duplicate
`['col']` вернёт `_ColumnProxy` (не скаляр). Правильно — `.at[label,
col]` для scalar (unique) или `df.loc[label]` для «может быть
несколько». Pandas имеет тот же gotcha (`.loc[a]['b']` deprecated).

**Не в v0.4.1: MultiIndex.** `set_index(['a','b'])`, tuple labels,
`loc[(x, y)]` — отдельная подсистема. v0.4.2. Также: `loc` на
non-monotonic с явными lo/hi без duplicate — pandas 1.x предупреждал,
2.x требует monotonic; мы явно raise.

**Урок:** pandas API эволюционировал от «loc flexible» к «loc only if
monotonic + unique-for-slice». Мы следуем этой строгости — honest
`NotImplementedError` вместо «странных» результатов на non-monotonic.

**Следующее (v0.4.2 — MultiIndex):**
- `MaterializedMultiIndex(values: pa.ChunkedArray × k)` или
  `pa.StructArray`.
- `set_index(['a','b'])` — compose 2 columns в multi-key.
- `loc[(x, y)]` — tuple labels.
- `loc[('a','b'):('c','d')]` — tuple slice, требует лексикографической
  монотонности.
- `reset_index()` — «explode» multi-index в k колонок.

**Следующее (v0.4.3): rolling/resample** — Sort + halo buffer.

"""
if "Эпизод 61" not in s:
    s = s.replace(marker_135, episode_61 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §11.1, §11.5, §13.4/61")
PY

# ---------- 5. CHANGELOG.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.4.1] — 2026-10-19

### Добавлено
- `CadwDataFrame.loc` — open-endpoint slices на monotonic index:
  - `loc[:hi]` → от начала до `hi` (inclusive).
  - `loc[lo:]` → от `lo` до конца.
  - `loc[:]` → все строки (lazy view).
  - descending monotonic тоже работает (`loc['b':'d']` → d,c,b).
- `CadwDataFrame.loc[label]` — duplicate labels → `CadwDataFrame`
  (pandas: Series→DataFrame). Unique → `dict` (без изменений).
- `CadwDataFrame.loc[[labels]]` — все вхождения для каждого label
  (ранее first-wins).
- `MaterializedIndex.is_monotonic_increasing()`,
  `is_monotonic_decreasing()`, `is_monotonic()` — O(N), nulls skipped.
- `MaterializedIndex.has_duplicates()` — O(N), set-based.
- `MaterializedIndex.get_positions_all(label)` — все позиции по `==`.

### Изменено
- `loc[slice]` на **non-monotonic** index → `NotImplementedError`
  (pandas 2.x тоже).
- `loc[slice]` на **duplicate** index → `NotImplementedError`
  (pandas: cannot slice non-unique).
- `loc[slice]` с `step != 1` → `NotImplementedError`.
- `.at[label, col]` на duplicate label → `KeyError` (pandas: undefined;
  используйте `loc[[label]]`).

### Ограничения (v0.4.1, §11.5)
- **MultiIndex** (`set_index(['a','b'])`) — v0.4.2.
- **`loc[slice]` step** — v0.4.2.
- **Non-monotonic + slice** — `NotImplementedError` (pandas-стиль).
- **`.at` на duplicate** — `KeyError`.

### Тесты
- 22 новых в `test_loc.py`: open-endpoints (start/stop/both на
  asc/desc), monotonic API, `get_positions_all`, `has_duplicates`,
  duplicate-label `loc` (dict vs CadwDataFrame), `loc[[duplicates]]`,
  slice на duplicate → raise, `.at` на duplicate → raise.
- Обновлён 1 тест: `test_loc_slice_none_endpoint_raises` → 8 новых.

"""
if "## [0.4.1]" not in s:
    s = s.replace("## [0.4.0d]", entry + "## [0.4.0d]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.4.1")
PY

# ---------- 6. run ----------
echo "=== pytest tests/test_loc.py -v ==="
set +e
uv run pytest tests/test_loc.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "loc: rc=$RC"
  git add -A
  git commit -m "WIP: v0.4.1 loc-complete (test red; fix follows)" || true
  exit 1
fi

echo "=== pytest full fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.4.1: loc open-endpoints (monotonic) + duplicate labels; 22 tests"

echo "==> v0.4.1 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"