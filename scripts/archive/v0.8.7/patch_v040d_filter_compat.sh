#!/usr/bin/env bash
# patch_v040d_filter_compat: filtered.index/set_index/reset_index работают
# через _iter_filtered_batches_with_positions + _index_values (full dataset).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-v040d-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch v0.4.0d filter-compat"

BR="patch/v040d-filter-compat"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- dataframe.py: _iter_filtered_batches_with_positions + filter-compat ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# 1. Import MaskFilter for isinstance checks
if "MaskFilter" not in s.split("from cadw.dataset import")[1][:500]:
    # Update import block
    old = '''from cadw.dataset import (
    Dataset,
    CadwIndex,
    RangeIndex,
    MaterializedIndex,
    RowFilter,
    PredicateFilter,
    SliceFilter,
)'''
    new = '''from cadw.dataset import (
    Dataset,
    CadwIndex,
    RangeIndex,
    MaterializedIndex,
    RowFilter,
    MaskFilter,
    PredicateFilter,
    SliceFilter,
)'''
    if old not in s:
        raise SystemExit("cadw.dataset import block not found")
    s = s.replace(old, new, 1)
    print("dataframe.py: import MaskFilter")

# 2. Add _iter_filtered_batches_with_positions after _iter_filtered_batches
anchor = '''    # ---------------- reading ----------------

    def to_batches('''
if anchor not in s:
    raise SystemExit("reading section anchor not found")

positer = '''    def _iter_filtered_batches_with_positions(self):
        """Yield (filtered_batch, positions_in_raw_dataset).

        positions — numpy int64 array of 0-based позиций в **unfiltered**
        dataset (после последнего set_index/reset_index). Используется для
        восстановления parent index при filter (v0.4.0d).
        """
        if not self._row_filters:
            offset = 0
            for b in self._dataset.iter_batches(columns=self._columns):
                positions = np.arange(offset, offset + b.num_rows, dtype=np.int64)
                yield b, positions
                offset += b.num_rows
            return

        needed = list(self._columns)
        extra: set[str] = set()
        for f in self._row_filters:
            extra |= f.columns_required()
        for c in extra:
            if c not in needed:
                needed.append(c)

        raw_it = self._dataset.iter_batches(columns=needed)

        offset_box = [0]

        def _init():
            for b in raw_it:
                positions = np.arange(
                    offset_box[0],
                    offset_box[0] + b.num_rows,
                    dtype=np.int64,
                )
                yield b, positions
                offset_box[0] += b.num_rows

        it = _init()
        for f in self._row_filters:
            it = _apply_filter_with_positions(f, it)

        for b, positions in it:
            yield b.select(self._columns), positions

'''

s = s.replace(anchor, positer + anchor, 1)
print("dataframe.py: _iter_filtered_batches_with_positions added")

# 3. Update index property — remove raise, use positions
old_index = '''    @property
    def index(self) -> CadwIndex:
        """Индекс строк (v0.4.0a).

        - Без фильтров: RangeIndex(0, N) или MaterializedIndex (после
          set_index).
        - С активными фильтрами: raise NotImplementedError — синхронизация
          RangeIndex с filter требует O(N) scan. v0.4.0b (iloc).
        """
        if self._row_filters:
            raise NotImplementedError(
                "df.index: с активными фильтрами не поддерживается в "
                "v0.4.0a (RangeIndex+filter = O(N) scan). "
                "Сбросьте фильтры через reset_index() перед filter, "
                "или используйте set_index(col) — но и он требует "
                "отсутствия фильтров. См. §13.4/55."
            )
        if self._index_kind == "range":
            return RangeIndex(self._index_start, self._index_stop)
        assert self._index_values is not None
        return MaterializedIndex(self._index_values)'''

new_index = '''    @property
    def index(self) -> CadwIndex:
        """Индекс строк (v0.4.0d — filter-compat).

        - Без фильтров: RangeIndex(0, N) или MaterializedIndex.
        - С фильтрами: MaterializedIndex — subset `_index_values[positions]`
          (O(filtered_n) scan, positions из _iter_filtered_batches_with_positions).

        Pandas-совместимо: df[df.v > 10].index = subset исходного index
        для отфильтрованных строк. §13.4/59.
        """
        if not self._row_filters:
            if self._index_kind == "range":
                return RangeIndex(self._index_start, self._index_stop)
            assert self._index_values is not None
            return MaterializedIndex(self._index_values)

        # Filtered: collect positions and take from _index_values
        pos_chunks: list[np.ndarray] = []
        for _b, pos in self._iter_filtered_batches_with_positions():
            pos_chunks.append(pos)
        if pos_chunks:
            positions = np.concatenate(pos_chunks)
        else:
            positions = np.array([], dtype=np.int64)

        if self._index_kind == "range":
            values = positions + self._index_start
            return MaterializedIndex(pa.array(values, type=pa.int64()))
        assert self._index_values is not None
        values = self._index_values.take(
            pa.array(positions, type=pa.int64())
        )
        return MaterializedIndex(values)'''

if old_index not in s:
    raise SystemExit("index property not found")
s = s.replace(old_index, new_index, 1)
print("dataframe.py: index property — filter-compat")

# 4. Update set_index — remove raise for filters; use full col materialization
old_setindex_raise = '''        if self._row_filters:
            raise NotImplementedError(
                "set_index: не поддерживается с активными фильтрами в "
                "v0.4.0a. Сбросьте фильтры или материализуйте через "
                "df.write_parquet(...) и пересоздайте. §13.4/55."
            )
        if not isinstance(drop, bool):
            raise TypeError("set_index: drop must be bool")

        chunks: list[pa.Array] = []
        for b in self._iter_filtered_batches():
            chunks.append(b.column(col))'''

new_setindex_raise = '''        if not isinstance(drop, bool):
            raise TypeError("set_index: drop must be bool")

        # v0.4.0d: материализуем col для **всего** dataset (не filtered).
        # Это O(N) scan, но `_index_values` всегда full → filtered.index =
        # subset через positions. §13.4/59.
        chunks: list[pa.Array] = []
        for b in self._dataset.iter_batches(columns=[col]):
            chunks.append(b.column(col))'''

if old_setindex_raise not in s:
    raise SystemExit("set_index raise block not found")
s = s.replace(old_setindex_raise, new_setindex_raise, 1)
print("dataframe.py: set_index — filter-compat (full col)")

# 5. Update reset_index — remove raise; use positions for filtered
old_reset_raise = '''        if self._row_filters:
            raise NotImplementedError(
                "reset_index: не поддерживается с активными фильтрами в "
                "v0.4.0a. §13.4/55."
            )
        if not isinstance(drop, bool):
            raise TypeError("reset_index: drop must be bool")

        if self._index_kind == "range":'''

new_reset_raise = '''        if not isinstance(drop, bool):
            raise TypeError("reset_index: drop must be bool")

        if self._index_kind == "range":'''

if old_reset_raise not in s:
    raise SystemExit("reset_index raise block not found")
s = s.replace(old_reset_raise, new_reset_raise, 1)
print("dataframe.py: reset_index — raise removed")

# 6. reset_index range + drop=False при фильтрах — тоже ValueError
#    (index уже RangeIndex — нет смысла выносить).
#    При фильтрах _index_kind всё ещё "range" → уже raise ValueError.
#    OK, никаких изменений.

# 7. reset_index materialized + drop=True при фильтрах:
#    сбрасываем на RangeIndex(0, filtered_n)?
#    Нет — на RangeIndex(0, len(_index_values)) = full dataset n.
#    Потом filter применятся на RangeIndex.
#    Pandas: df.set_index("k")[mask].reset_index(drop=True) → RangeIndex(0, |mask|).
#    Значит _index_stop должен быть filtered_n.
#    Но _index_start/stop применяются ТОЛЬКО к unfiltered view.
#    Для filtered view — reset_index создаёт новый Dataset (filtered rows) + RangeIndex(0, filtered_n).

# Смотрим существующий reset_index materialized drop=True:
old_reset_mat_drop = '''        if drop:
            if inplace:
                self._index_kind = "range"
                self._index_start = 0
                self._index_stop = len(values)
                self._index_values = None
                return self
            new = CadwDataFrame.__new__(CadwDataFrame)
            new._dataset = self._dataset
            new._memory_guard = self._memory_guard
            new._spill_dir = self._spill_dir
            new._columns = list(self._columns)
            new._row_filters = list(self._row_filters)
            new._index_kind = "range"
            new._index_start = 0
            new._index_stop = len(values)
            new._index_values = None
            return new'''

new_reset_mat_drop = '''        if drop:
            if self._row_filters:
                # v0.4.0d: filtered + drop=True → materialize filtered view
                # as new Dataset with RangeIndex(0, filtered_n).
                filtered_n = self.count()
                if inplace:
                    new_ds = Dataset.from_batches(
                        pa.schema(
                            [self._dataset.schema.field(c) for c in self._columns]
                        ),
                        self._iter_filtered_batches(),
                        self._new_dataset_dir(),
                    )
                    self._dataset = new_ds
                    self._row_filters = []
                    self._index_kind = "range"
                    self._index_start = 0
                    self._index_stop = filtered_n
                    self._index_values = None
                    return self
                new_ds = Dataset.from_batches(
                    pa.schema(
                        [self._dataset.schema.field(c) for c in self._columns]
                    ),
                    self._iter_filtered_batches(),
                    self._new_dataset_dir(),
                )
                new = CadwDataFrame.__new__(CadwDataFrame)
                new._dataset = new_ds
                new._memory_guard = self._memory_guard
                new._spill_dir = self._spill_dir
                new._columns = list(self._columns)
                new._row_filters = []
                new._index_kind = "range"
                new._index_start = 0
                new._index_stop = filtered_n
                new._index_values = None
                return new
            if inplace:
                self._index_kind = "range"
                self._index_start = 0
                self._index_stop = len(values)
                self._index_values = None
                return self
            new = CadwDataFrame.__new__(CadwDataFrame)
            new._dataset = self._dataset
            new._memory_guard = self._memory_guard
            new._spill_dir = self._spill_dir
            new._columns = list(self._columns)
            new._row_filters = list(self._row_filters)
            new._index_kind = "range"
            new._index_start = 0
            new._index_stop = len(values)
            new._index_values = None
            return new'''

if old_reset_mat_drop not in s:
    raise SystemExit("reset_index materialized drop=True block not found")
s = s.replace(old_reset_mat_drop, new_reset_mat_drop, 1)
print("dataframe.py: reset_index — filtered drop=True materialize")

# 8. reset_index drop=False при фильтрах — использовать positions
old_reset_drop_false = '''        # drop=False: add "index" or "level_N" column
        name = "index"
        if name in self._columns:
            i = 0
            name = f"level_{i}"
            while name in self._columns:
                i += 1
                name = f"level_{i}"

        new_columns = list(self._columns) + [name]
        new_schema = pa.schema(
            [self._dataset.schema.field(c) for c in self._columns]
            + [pa.field(name, values.type)]
        )

        offset = 0

        def _iter_with_index():
            nonlocal offset
            for b in self._iter_filtered_batches():
                n = b.num_rows
                idx_slice = values.slice(offset, n)
                yield b.append_column(name, idx_slice)
                offset += n'''

new_reset_drop_false = '''        # drop=False: add "index" or "level_N" column
        name = "index"
        if name in self._columns:
            i = 0
            name = f"level_{i}"
            while name in self._columns:
                i += 1
                name = f"level_{i}"

        new_columns = list(self._columns) + [name]
        new_schema = pa.schema(
            [self._dataset.schema.field(c) for c in self._columns]
            + [pa.field(name, values.type)]
        )

        if self._row_filters:
            # v0.4.0d: filtered — index column = _index_values[positions],
            # позиции из _iter_filtered_batches_with_positions.
            def _iter_with_index_filtered():
                for b, positions in self._iter_filtered_batches_with_positions():
                    idx_slice = values.take(
                        pa.array(positions, type=pa.int64())
                    )
                    yield b.append_column(name, idx_slice)
            _iter_with_index_impl = _iter_with_index_filtered
        else:
            offset = 0

            def _iter_with_index_unfiltered():
                nonlocal offset
                for b in self._iter_filtered_batches():
                    n = b.num_rows
                    idx_slice = values.slice(offset, n)
                    yield b.append_column(name, idx_slice)
                    offset += n
            _iter_with_index_impl = _iter_with_index_unfiltered'''

if old_reset_drop_false not in s:
    raise SystemExit("reset_index drop=False block not found")
s = s.replace(old_reset_drop_false, new_reset_drop_false, 1)

# 9. Replace usage of _iter_with_index() with _iter_with_index_impl()
s = s.replace(
    "        new_ds = Dataset.from_batches(\n            new_schema, _iter_with_index(), self._new_dataset_dir()\n        )",
    "        new_ds = Dataset.from_batches(\n            new_schema, _iter_with_index_impl(), self._new_dataset_dir()\n        )",
    1,
)
print("dataframe.py: reset_index — filtered drop=False via positions")

# 10. Add helper _apply_filter_with_positions at module level (bottom)
helper = '''

def _apply_filter_with_positions(f: RowFilter, it):
    """Apply a RowFilter to (batch, positions) stream.

    - MaskFilter: positions[mask] — positions сохраняются для прошедших.
    - SliceFilter: slice по **локальному** counter (filtered-view position),
      но positions сохраняются как в исходном.
    """
    if isinstance(f, MaskFilter):
        for b, positions in it:
            mask = f.mask_for_batch(b)
            mask_np = mask.to_numpy(zero_copy_only=False)
            new_positions = positions[mask_np]
            if len(new_positions) > 0:
                yield b.filter(mask), new_positions
        return
    if isinstance(f, SliceFilter):
        local_pos = 0
        for b, positions in it:
            n = b.num_rows
            batch_local = np.arange(local_pos, local_pos + n, dtype=np.int64)
            mask_np = batch_local >= f.start
            if f.stop is not None:
                mask_np &= batch_local < f.stop
            if f.step > 1:
                mask_np &= ((batch_local - f.start) % f.step == 0)
            new_positions = positions[mask_np]
            if len(new_positions) > 0:
                yield (
                    b.filter(pa.array(mask_np, type=pa.bool_())),
                    new_positions,
                )
            local_pos += n
            if f.stop is not None and local_pos >= f.stop:
                return
        return
    raise TypeError(
        f"_apply_filter_with_positions: unsupported filter "
        f"{type(f).__name__}"
    )

'''
if "_apply_filter_with_positions" not in s:
    s = s.rstrip() + helper + "\n"
    print("dataframe.py: _apply_filter_with_positions helper added")

p.write_text(s)
print("dataframe.py: готово")
PY

# ---------- indexers.py: _many uses with_positions; _LocIndexer works with filtered ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/indexers.py")
s = p.read_text()

# 1. _many: use _iter_filtered_batches_with_positions for filtered, preserve index
old_many = '''    def _many(self, positions: list[int]):
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
        return self._rows_to_cadwdf(rows, resolved)'''

new_many = '''    def _many(self, positions: list[int]):
        resolved = [self._resolve_int(p) for p in positions]
        if not resolved:
            return self._empty_view()

        targets = sorted(set(resolved))
        target_set = set(targets)

        # v0.4.0d: use _iter_filtered_batches_with_positions to also
        # capture raw-dataset positions (for index preservation).
        collected: dict[int, dict[str, Any]] = {}
        raw_pos_collected: dict[int, int] = {}
        running = 0
        for b, raw_positions in self._parent._iter_filtered_batches_with_positions():
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
                    raw_pos_collected[p] = int(raw_positions[local])
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
        raw_pos = [raw_pos_collected[p] for p in resolved]
        return self._rows_to_cadwdf(rows, raw_pos)'''

if old_many not in s:
    raise SystemExit("_many not found")
s = s.replace(old_many, new_many, 1)
print("indexers.py: _many uses with_positions")

# 2. _rebuild_index: now works even with filters (raw_positions from parent._index_values)
old_rebuild = '''def _rebuild_index(parent: "CadwDataFrame", positions: list[int]):
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
    return MaterializedIndex(pa.array(vals, type=src.type))'''

new_rebuild = '''def _rebuild_index(parent: "CadwDataFrame", raw_positions: list[int]):
    """Rebuild index for iloc[list] result (v0.4.0d — filter-compat).

    `raw_positions` — позиции в **unfiltered** dataset (из
    _iter_filtered_batches_with_positions).

    - parent RangeIndex → MaterializedIndex([start + p, ...]).
    - parent MaterializedIndex → subset `_index_values[p]`.
    """
    from cadw.dataset.index import MaterializedIndex

    if parent._index_kind == "range":
        vals = [parent._index_start + p for p in raw_positions]
        return MaterializedIndex(pa.array(vals, type=pa.int64()))

    src = parent._index_values
    vals = [src[p] for p in raw_positions]
    return MaterializedIndex(pa.array(vals, type=src.type))'''

if old_rebuild not in s:
    raise SystemExit("_rebuild_index not found")
s = s.replace(old_rebuild, new_rebuild, 1)
print("indexers.py: _rebuild_index — filter-compat via raw_positions")

# 3. _LocIndexer._materialized: remove filter-raise
old_mat = '''    def _materialized(self):
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
        return MaterializedIndex(p._index_values)'''

new_mat = '''    def _materialized(self):
        from cadw.dataset.index import MaterializedIndex
        p = self._parent
        if p._index_kind == "range":
            raise TypeError(
                "loc: requires MaterializedIndex (after set_index). "
                "RangeIndex — use iloc."
            )
        # v0.4.0d: с фильтрами тоже работаем — filtered.index даёт
        # MaterializedIndex-subset, поверх него get_indexer работает.
        if p._row_filters:
            idx = p.index
            if not isinstance(idx, MaterializedIndex):
                raise TypeError(
                    f"loc: filtered index is {type(idx).__name__}, "
                    f"expected MaterializedIndex"
                )
            return idx
        assert p._index_values is not None
        return MaterializedIndex(p._index_values)'''

if old_mat not in s:
    raise SystemExit("_materialized not found")
s = s.replace(old_mat, new_mat, 1)
print("indexers.py: _LocIndexer._materialized — filter-compat")

p.write_text(s)
print("indexers.py: готово")
PY

# ---------- Update existing tests: filter no longer raises ----------
python3 - <<'PY'
from pathlib import Path

# test_cadw_index.py: rename 3 tests to reflect filter-compat
p = Path("tests/test_cadw_index.py")
s = p.read_text()

# 1. test_index_with_filter_raises → test_index_with_filter_preserves
old1 = '''def test_index_with_filter_raises(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 2]
    with pytest.raises(NotImplementedError, match="фильтр"):
        _ = filtered.index'''
new1 = '''def test_index_with_filter_preserves(tmp_path):
    """v0.4.0d: filtered.index = MaterializedIndex subset."""
    from cadw.dataset.index import MaterializedIndex
    df = make_df(tmp_path)
    filtered = df[df["v"] > 2]  # rows c, d, e → positions 2, 3, 4
    idx = filtered.index
    assert isinstance(idx, MaterializedIndex)
    assert list(idx) == [2, 3, 4]'''
if old1 in s:
    s = s.replace(old1, new1, 1)

# 2. test_set_index_with_filter_raises → test_set_index_with_filter_works
old2 = '''def test_set_index_with_filter_raises(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 2]
    with pytest.raises(NotImplementedError, match="фильтр"):
        filtered.set_index("k")'''
new2 = '''def test_set_index_with_filter_works(tmp_path):
    """v0.4.0d: set_index при фильтрах — материализует full col."""
    from cadw.dataset.index import MaterializedIndex
    df = make_df(tmp_path)
    filtered = df[df["v"] > 2]  # c, d, e
    result = filtered.set_index("k", drop=False)
    # index values = filtered view k values (через positions)
    assert isinstance(result.index, MaterializedIndex)
    assert list(result.index) == ["c", "d", "e"]
    # columns содержат k, v
    assert set(result.columns) == {"k", "v"}'''
if old2 in s:
    s = s.replace(old2, new2, 1)

# 3. test_reset_index_with_filter_raises → test_reset_index_with_filter_works
old3 = '''def test_reset_index_with_filter_raises(tmp_path):
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=True)
    # select without filter works
    sub = with_idx[["v"]]
    filtered = sub[sub["v"] > 2]
    with pytest.raises(NotImplementedError, match="фильтр"):
        filtered.reset_index(drop=True)'''
new3 = '''def test_reset_index_with_filter_works(tmp_path):
    """v0.4.0d: reset_index при фильтрах → RangeIndex(0, filtered_n)."""
    from cadw.dataset.index import RangeIndex
    df = make_df(tmp_path)
    with_idx = df.set_index("k", drop=True)
    sub = with_idx[["v"]]
    filtered = sub[sub["v"] > 2]  # rows v=30, 40, 50 → filtered_n = 3
    restored = filtered.reset_index(drop=True)
    assert isinstance(restored.index, RangeIndex)
    assert restored.index.start == 0
    assert restored.index.stop == 3'''
if old3 in s:
    s = s.replace(old3, new3, 1)

p.write_text(s)
print("test_cadw_index.py: 3 tests updated for filter-compat")

# test_iloc.py: test_iloc_list_on_filtered_resets_to_range → preserves
p2 = Path("tests/test_iloc.py")
s2 = p2.read_text()

old4 = '''def test_iloc_list_on_filtered_resets_to_range(tmp_path):
    """На filtered view — default RangeIndex (index parent не сохраняется)."""
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15]
    sub = filtered.iloc[[0, 2]]
    assert isinstance(sub.index, RangeIndex)
    assert sub.index.start == 0
    assert sub.index.stop == 2'''
new4 = '''def test_iloc_list_on_filtered_preserves_parent(tmp_path):
    """v0.4.0d: filtered view + iloc[list] → preserves parent index via raw positions."""
    from cadw.dataset.index import MaterializedIndex
    df = make_df(tmp_path)  # v: 10,20,30,40,50
    filtered = df[df["v"] > 15]  # rows b(20),c(30),d(40),e(50) → raw positions 1,2,3,4
    sub = filtered.iloc[[0, 2]]  # filtered positions 0, 2 → raw 1, 3
    assert isinstance(sub.index, MaterializedIndex)
    assert list(sub.index) == [1, 3]'''
if old4 in s2:
    s2 = s2.replace(old4, new4, 1)
    p2.write_text(s2)
    print("test_iloc.py: test_iloc_list_on_filtered updated")

# test_loc.py: test_loc_on_filtered_raises → works
p3 = Path("tests/test_loc.py")
s3 = p3.read_text()
old5 = '''def test_loc_on_filtered_raises(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    filtered = df[df["v"] > 15]
    with pytest.raises(NotImplementedError, match="фильтр"):
        filtered.loc["b"]'''
new5 = '''def test_loc_on_filtered_works(tmp_path):
    """v0.4.0d: loc на filtered view работает (через filtered.index)."""
    df = make_df(tmp_path).set_index("k", drop=False)
    filtered = df[df["v"] > 15]  # b, c, d, e
    assert filtered.loc["c"] == {"k": "c", "v": 30}
    assert filtered.loc["e"] == {"k": "e", "v": 50}
    with pytest.raises(KeyError):
        filtered.loc["a"]  # a не в filtered'''
if old5 in s3:
    s3 = s3.replace(old5, new5, 1)
    p3.write_text(s3)
    print("test_loc.py: test_loc_on_filtered updated")

PY

# ---------- tests: new filter-compat tests ----------
cat > tests/test_filter_compat.py <<'EOF'
"""v0.4.0d: filter-compat для index / set_index / reset_index / iloc / loc."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.dataset.index import MaterializedIndex, RangeIndex


def make_df(tmp_path, data=None, chunk=3):
    if data is None:
        data = {"k": ["a", "b", "c", "d", "e"], "v": [10, 20, 30, 40, 50]}
    t = pa.table(data)
    p = Path(tmp_path) / "src.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=tmp_path / "spill",
    )


# ---------- filtered.index (RangeIndex parent) ----------

def test_filtered_index_range(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15]  # b,c,d,e → raw positions 1,2,3,4
    idx = filtered.index
    assert isinstance(idx, MaterializedIndex)
    assert list(idx) == [1, 2, 3, 4]


def test_filtered_index_all_filtered_out(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 1000]
    idx = filtered.index
    assert isinstance(idx, MaterializedIndex)
    assert len(idx) == 0


def test_filtered_index_with_materialized_parent(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    filtered = df[df["v"] > 15]  # b,c,d,e
    idx = filtered.index
    assert isinstance(idx, MaterializedIndex)
    assert list(idx) == ["b", "c", "d", "e"]


def test_filtered_index_slice_filter(tmp_path):
    df = make_df(tmp_path)
    sliced = df[1:4]  # raw positions 1,2,3
    idx = sliced.index
    assert isinstance(idx, MaterializedIndex)
    assert list(idx) == [1, 2, 3]


def test_filtered_index_mask_then_slice(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15][0:2]  # of [1,2,3,4] take [0:2] → raw [1,2]
    idx = filtered.index
    assert isinstance(idx, MaterializedIndex)
    assert list(idx) == [1, 2]


# ---------- filtered.set_index ----------

def test_set_index_on_filtered(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15]  # b,c,d,e
    result = filtered.set_index("k", drop=True)
    assert isinstance(result.index, MaterializedIndex)
    assert list(result.index) == ["b", "c", "d", "e"]
    assert "k" not in result.columns


def test_set_index_then_filter(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    filtered = df[df["v"] > 30]  # d, e
    assert list(filtered.index) == ["d", "e"]


# ---------- filtered.reset_index ----------

def test_reset_index_filtered_drop_true(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=True)
    filtered = df[df["v"] > 15]  # b,c,d,e
    restored = filtered.reset_index(drop=True)
    assert isinstance(restored.index, RangeIndex)
    assert restored.index.start == 0
    assert restored.index.stop == 4  # 4 filtered rows


def test_reset_index_filtered_drop_false(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=True)
    filtered = df[df["v"] > 15]
    restored = filtered.reset_index(drop=False)
    assert isinstance(restored.index, RangeIndex)
    assert restored.index.stop == 4
    assert "index" in restored.columns
    # values
    out = tmp_path / "out.parquet"
    restored.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["index"] for r in rows] == ["b", "c", "d", "e"]


# ---------- filtered.iloc[list] preserves parent ----------

def test_iloc_list_filtered_range_parent(tmp_path):
    df = make_df(tmp_path)
    filtered = df[df["v"] > 15]  # b,c,d,e → raw [1,2,3,4]
    sub = filtered.iloc[[0, 2]]  # filtered 0, 2 → raw 1, 3
    assert isinstance(sub.index, MaterializedIndex)
    assert list(sub.index) == [1, 3]


def test_iloc_list_filtered_materialized_parent(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    filtered = df[df["v"] > 15]  # b,c,d,e
    sub = filtered.iloc[[0, 2]]  # raw 1, 3 → k values b, d
    assert isinstance(sub.index, MaterializedIndex)
    assert list(sub.index) == ["b", "d"]


# ---------- filtered.loc works ----------

def test_loc_filtered(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    filtered = df[df["v"] > 15]  # b, c, d, e
    assert filtered.loc["c"] == {"k": "c", "v": 30}
    with pytest.raises(KeyError):
        filtered.loc["a"]


def test_loc_filtered_slice(tmp_path):
    df = make_df(tmp_path).set_index("k", drop=False)
    filtered = df[df["v"] > 15]  # b, c, d, e
    sub = filtered.loc["c":"d"]
    out = tmp_path / "out.parquet"
    sub.write_parquet(out)
    rows = pq.read_table(out).to_pylist()
    assert [r["k"] for r in rows] == ["c", "d"]


# ---------- no filter — unchanged ----------

def test_no_filter_index_still_range(tmp_path):
    df = make_df(tmp_path)
    assert isinstance(df.index, RangeIndex)


def test_no_filter_set_index_still_works(tmp_path):
    df = make_df(tmp_path)
    result = df.set_index("k", drop=False)
    assert isinstance(result.index, MaterializedIndex)
    assert list(result.index) == ["a", "b", "c", "d", "e"]
EOF

# ---------- DECISIONS.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

if "| `_iter_filtered_batches_with_positions` (v0.4.0d) |" not in s:
    s = s.replace(
        "| `loc`/`.at`/`.iat` (v0.4.0c: label-based) | ✅ | 30 |",
        "| `loc`/`.at`/`.iat` (v0.4.0c: label-based) | ✅ | 31 |\n"
        "| filter-compat index/set/reset (v0.4.0d) | ✅ | 15 |",
        1,
    )

# §11.5: убрать старые ограничения (они закрыты), обновить
s = s.replace(
    "| `df.index`/`set_index`/`reset_index` с активными фильтрами (v0.4.0a) | `CadwDataFrame` | raise NotImplementedError; синхронизация index↔filter — v0.4.0b (iloc, через row_id). §13.4/55. |\n",
    "",
    1,
)
s = s.replace(
    "| `iloc[list]` на filtered view не сохраняет parent index (v0.4.0b) | `_ILocIndexer` | default `RangeIndex(0, K)`; preserve — v0.4.0c (filter-compat, нужна `_iter_filtered_batches_with_positions`). |\n",
    "",
    1,
)
s = s.replace(
    "| `loc` на filtered view — `NotImplementedError`. v0.4.0d.\n",
    "",
    1,
)

marker_135 = "### 13.5. Открытые вопросы"
episode_59 = """**Эпизод 59 (filter-compat — positions-механизм).** v0.4.0a-c
бросали `NotImplementedError` для `filtered.index` / `set_index` /
`reset_index` / `loc` при активных фильтрах. v0.4.0d закрывает это.

**Ключевая идея — `_iter_filtered_batches_with_positions`.** Yield
`(filtered_batch, positions_in_raw_dataset)`, где positions — numpy
int64 массив, 0-based позиции в **unfiltered** dataset (после последнего
set_index/reset_index). Реализация — композиция фильтров:

  Init: (raw_batch, arange(offset, offset+n)).
  Для каждого filter f в chain:
    - MaskFilter: mask = f.mask_for_batch(b); yield b.filter(mask),
      positions[mask];
    - SliceFilter: local_pos counter (filtered view);
      mask = (local_pos..local_pos+n satisfies slice); yield
      b.filter(mask), positions[mask].
  Затем b.select(_columns).

**Position semantics:**
- `MaskFilter` сохраняет raw positions (прошли маску → их raw pos).
- `SliceFilter` берёт позиции по **локальному counter** (позиция в
  filtered view), но raw pos — из накопленного positions.

**`_index_values` — always full dataset (unfiltered).** Filter не трогает
`_index_values`; `_with_filter` копирует его без изменений (уже так).
`filtered.index` = `_index_values[positions]` для MaterializedIndex, или
`positions + _index_start` для RangeIndex.

**Pandas-совместимо:** `df[df.v>10].index` — subset исходного; не
reset_index. `df[df.v>10].set_index("k")` — index = filtered k values
(через `_index_values[positions]` — потому что `_index_values` = полный
k, positions = filtered).

**`set_index(col)` при фильтрах** — материализует col **всего** dataset
(O(N) scan, не только filtered). Плата за корректность: `_index_values`
всегда full → любая комбинация filter/set_index/filter работает через
positions. Для больших N это цена; в v0.4.1 — streaming set_index с
RangeIndex fast path для уникальных col.

**`reset_index` при фильтрах:**
- `drop=True`: materialize filtered view как новый Dataset
  + RangeIndex(0, filtered_n). O(filtered_n) I/O — стандартный
  materialize.
- `drop=False`: добавляет колонку `index` = `_index_values[positions]`
  (для materialized) или `positions + start` (для range).

**`loc` на filtered view** — работает, потому что `_LocIndexer._materialized`
теперь возвращает `filtered.index` (MaterializedIndex). `get_indexer`
поверх filtered index.

**`iloc[list]` на filtered view** — preserves parent index через raw
positions, а не reset. Изменение с v0.4.0b.

**Урок:** позиции — универсальный механизм совместимости filter/index.
Один scan строит `(batch, positions)`; всё остальное — subset через
positions. Pandas делает то же (BlockManager + index → take), но
in-memory; мы делаем streaming с O(1) state.

**Следующее (v0.4.1):**
- loc open-endpoints (`loc[:hi]`, `loc[lo:]`) — требуют is_monotonic.
- Duplicate labels in index — `loc[label]` возвращает все вхождения.
- MultiIndex (`MultiIndex([col1, col2])`).
- Streaming set_index (RangeIndex fast path).

"""
if "Эпизод 59" not in s:
    s = s.replace(marker_135, episode_59 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §11.1, §11.5, §13.4/59")
PY

# ---------- CHANGELOG.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.4.0d] — 2026-10-19

### Добавлено
- `_iter_filtered_batches_with_positions()` — yield `(batch, positions)`,
  positions = numpy int64 в unfiltered dataset. Композиция фильтров:
  MaskFilter → `positions[mask]`; SliceFilter → slice по local counter
  с сохранением raw positions.
- `_apply_filter_with_positions(filter, it)` — helper для filter chain.

### Изменено (filter-compat)
- `df.index` при фильтрах → `MaterializedIndex(_index_values[positions])`
  (pandas-совместимо). Раньше — `NotImplementedError`.
- `set_index(col)` при фильтрах → работает; материализует **полный**
  col (O(N) scan), `_index_values` = full → любой комбинации filter/set
  через positions.
- `reset_index(drop=True)` при фильтрах → materialize filtered view +
  RangeIndex(0, filtered_n).
- `reset_index(drop=False)` при фильтрах → колонка `index` =
  `_index_values[positions]`.
- `iloc[list]` на filtered view — **сохраняет** parent index через
  raw positions (раньше reset). Более pandas-совместимо.
- `loc` на filtered view — работает (через filtered.index).

### Закрыто (из v0.4.0a-c, §11.5)
- ~~`df.index`/`set_index`/`reset_index` с активными фильтрами — raise~~
- ~~`iloc[list]` на filtered view не сохраняет parent index~~
- ~~`loc` на filtered view — NotImplementedError~~

### Тесты
- 15 новых в `test_filter_compat.py`: filtered.index (range/mat/
  slice/mask_then_slice/empty), set_index+filter, filter+set_index,
  reset_index drop=True/False при фильтрах, iloc[list] preserves
  parent (range/materialized), loc на filtered, no-filter unchanged.
- Обновлены 4 теста: `test_index_with_filter_*`, `test_set_index_with_
  filter_*`, `test_reset_index_with_filter_*`, `test_iloc_list_on_
  filtered_*`, `test_loc_on_filtered_*` — теперь проверяют
  сохранение/работу, не raise.

"""
if "## [0.4.0d]" not in s:
    s = s.replace("## [0.4.0c]", entry + "## [0.4.0c]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.4.0d")
PY

# ---------- run ----------
echo "=== pytest tests/test_filter_compat.py -v ==="
set +e
uv run pytest tests/test_filter_compat.py -v
RC1=$?
set -e
if [[ $RC1 -ne 0 ]]; then
  echo "filter_compat: rc=$RC1"
  git add -A
  git commit -m "WIP: v0.4.0d filter-compat (test red; fix follows)" || true
  exit 1
fi

echo "=== pytest tests/test_cadw_index.py tests/test_iloc.py tests/test_loc.py -v ==="
set +e
uv run pytest tests/test_cadw_index.py tests/test_iloc.py tests/test_loc.py -v
RC2=$?
set -e
if [[ $RC2 -ne 0 ]]; then
  echo "regression: rc=$RC2"
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.4.0d: filter-compat (positions); filtered.index/set_index/reset_index/loc; 15 tests + 4 updated"

echo "==> v0.4.0d applied on ${BR}"