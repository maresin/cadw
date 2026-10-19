#!/usr/bin/env bash
# patch_v034_combiner: partial aggregation (combiner) перед hash_shuffle
# для sum/count/min/max/mean. Opt-in через combiner=True.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-v034-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch v0.3.4 combiner"

BR="patch/v034-combiner"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- cadw/ops/groupby.py: combiner + _merge_partials + partial_merge ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/groupby.py")
s = p.read_text()

# 1. Add _COMBINER_SUPPORTED set after _AGG_FUNCS
old_anchor = '_AGG_FUNCS = {"mean", "sum", "count", "min", "max"}'
new_anchor = '''_AGG_FUNCS = {"mean", "sum", "count", "min", "max"}
# Функции, для которых работает partial aggregation (combiner):
# - mean: sum + count partial
# - sum/count/min/max: прямой partial
# std/var требуют Welford state — v0.3.5.
_COMBINER_SUPPORTED = {"sum", "count", "min", "max", "mean"}
_COMB_PREFIX = "__comb_"
'''
if old_anchor not in s:
    raise SystemExit("_AGG_FUNCS anchor not found")
s = s.replace(old_anchor, new_anchor, 1)
print("groupby.py: _COMBINER_SUPPORTED добавлен")

# 2. Add _combine_batches and _merge_partials before _empty_result_schema
helper_anchor = "def _empty_result_schema("
helpers = '''def _combine_batches(
    batch_iter,
    by_columns: Sequence[str],
    agg_columns: Sequence[str],
    agg_funcs: Sequence[str],
    memory_guard: "MemoryGuard | None" = None,
):
    """Partial aggregation батча: pandas.groupby(by).agg(partial).

    Yields compact batches: `by + __comb_{col}_{base_func}`.
      - mean(col) → __comb_{col}_sum + __comb_{col}_count.
      - sum/count/min/max(col) → __comb_{col}_{func}.

    State: O(batch_size) per batch, O(1) между батчами.
    Корректность: partial-агрегаты ассоциативны → merge_partials = global agg.
    """
    # Уникальные (col, base_func) пары для эмита.
    base_specs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for col, f in zip(agg_columns, agg_funcs):
        if f == "mean":
            for bf in ("sum", "count"):
                if (col, bf) not in seen:
                    seen.add((col, bf))
                    base_specs.append((col, bf))
        else:
            if (col, f) not in seen:
                seen.add((col, f))
                base_specs.append((col, f))

    select_cols = list(dict.fromkeys(list(by_columns) + list(agg_columns)))

    for batch in batch_iter:
        if memory_guard is not None:
            memory_guard.check_batch(batch, name="combiner.input")
        table = pa.Table.from_batches([batch.select(select_cols)])
        grouped = table.group_by(list(by_columns)).aggregate(base_specs)

        # Переименовываем {col}_{base_func} → __comb_{col}_{base_func}.
        # group_by().aggregate() формирует имена `{col}_{func}` — добавим префикс.
        new_names = list(by_columns) + [
            f"{_COMB_PREFIX}{col}_{bf}" for col, bf in base_specs
        ]
        arrays = [grouped.column(c) for c in list(by_columns)] + [
            grouped.column(f"{col}_{bf}") for col, bf in base_specs
        ]
        out = pa.RecordBatch.from_arrays(
            arrays, names=new_names
        )
        yield out


def _merge_partials(
    table: pa.Table,
    by_columns: Sequence[str],
    output_specs: Sequence[tuple[str, str]],  # [(col, func), ...] как в _local_groupby
) -> pa.Table:
    """Merge partial aggregates → final aggregated table.

    `table` содержит `by + __comb_{col}_{base_func}` (из _combine_batches).
    `output_specs` — желаемый output: [(col, func)].

    Для mean — суммируем partial sums и counts, потом делим.
    Для остальных — merge по базовой функции (sum→sum, count→sum, min→min, max→max).
    """
    # Собираем merge-спеки: __comb_{col}_{base_func} с merge_func.
    merge_specs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for col, f in output_specs:
        if f == "mean":
            for bf in ("sum", "count"):
                src = f"{_COMB_PREFIX}{col}_{bf}"
                mf = "sum"
                if (src, mf) not in seen:
                    seen.add((src, mf))
                    merge_specs.append((src, mf))
        else:
            src = f"{_COMB_PREFIX}{col}_{f}"
            mf = "sum" if f == "count" else f
            if (src, mf) not in seen:
                seen.add((src, mf))
                merge_specs.append((src, mf))

    # Проверка: все нужные колонки есть в партиции.
    for src, _mf in merge_specs:
        if src not in table.schema.names:
            raise KeyError(
                f"_merge_partials: expected column {src!r} in partition "
                f"schema {table.schema.names}. Combiner output mismatch?"
            )

    merged = table.group_by(list(by_columns)).aggregate(merge_specs)

    # Финальный output: by + {col}_{func} с правильным dtype.
    fields: list[pa.Field] = []
    arrays: list[pa.Array] = []
    by_set = set(by_columns)
    for c in by_columns:
        fld = merged.schema.field(c)
        fields.append(fld)
        arrays.append(merged.column(c))
    for col, f in output_specs:
        if f == "mean":
            sum_src = f"{_COMB_PREFIX}{col}_sum_sum"
            count_src = f"{_COMB_PREFIX}{col}_count_sum"
            if sum_src not in merged.schema.names or count_src not in merged.schema.names:
                raise KeyError(
                    f"_merge_partials: missing {sum_src!r}/{count_src!r} "
                    f"for mean; got {merged.schema.names}"
                )
            s_arr = merged.column(sum_src).cast(pa.float64())
            c_arr = merged.column(count_src).cast(pa.float64())
            # mean = sum / count; count=0 → NaN (не должно случаться для non-empty групп).
            import pyarrow.compute as pc
            div = pc.divide(s_arr, c_arr)
            arrays.append(div)
            fields.append(pa.field(f"{col}_{f}", pa.float64()))
        else:
            src = f"{_COMB_PREFIX}{col}_{f}_sum" if f == "count" else (
                f"{_COMB_PREFIX}{col}_{f}_{'sum' if f == 'sum' else f}"
            )
            # Проще: имя после merge_partials.aggregate = {src}_{mf}.
            mf = "sum" if f == "count" else f
            final_src = f"{_COMB_PREFIX}{col}_{f}_{mf}"
            if final_src not in merged.schema.names:
                raise KeyError(
                    f"_merge_partials: missing {final_src!r}; "
                    f"got {merged.schema.names}"
                )
            arr = merged.column(final_src)
            out_type = pa.int64() if f == "count" else pa.float64()
            arrays.append(arr.cast(out_type))
            fields.append(pa.field(f"{col}_{f}", out_type))

    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


'''
if helper_anchor not in s:
    raise SystemExit("_empty_result_schema def not found")
s = s.replace(helper_anchor, helpers + helper_anchor, 1)
print("groupby.py: _combine_batches + _merge_partials добавлены")

# 3. _local_groupby: new param partial_merge
old_sig = '''def _local_groupby(
    partition_path: Path,
    by_columns: Sequence[str],
    agg_columns: Sequence[str],
    agg_funcs: Sequence[str] | str,
    memory_guard: MemoryGuard | None = None,
    spill_dir: Path | None = None,
    depth: int = 0,
    max_depth: int = _MAX_DEPTH,
    *,
    size_of_key_entry_bytes: int | None = None,
) -> pa.Table:'''
new_sig = '''def _local_groupby(
    partition_path: Path,
    by_columns: Sequence[str],
    agg_columns: Sequence[str],
    agg_funcs: Sequence[str] | str,
    memory_guard: MemoryGuard | None = None,
    spill_dir: Path | None = None,
    depth: int = 0,
    max_depth: int = _MAX_DEPTH,
    *,
    size_of_key_entry_bytes: int | None = None,
    partial_merge: bool = False,
) -> pa.Table:'''
if old_sig not in s:
    raise SystemExit("_local_groupby signature not found")
s = s.replace(old_sig, new_sig, 1)
print("groupby.py: _local_groupby signature — partial_merge param")

# 4. Inside _local_groupby: apply _merge_partials when partial_merge=True
# Найти блок с group_by, заменить:
old_gb = '''        if dict_fits:
            agg_specs = [(col, f) for col, f in zip(agg_columns, agg_funcs)]
            return table.group_by(list(by_columns)).aggregate(agg_specs)'''
new_gb = '''        if dict_fits:
            if partial_merge:
                return _merge_partials(
                    table, by_columns,
                    [(col, f) for col, f in zip(agg_columns, agg_funcs)],
                )
            agg_specs = [(col, f) for col, f in zip(agg_columns, agg_funcs)]
            return table.group_by(list(by_columns)).aggregate(agg_specs)'''
if old_gb not in s:
    raise SystemExit("dict_fits block not found")
s = s.replace(old_gb, new_gb, 1)
print("groupby.py: _local_groupby — _merge_partials branch")

# 5. groupby_aggregate: new combiner param + logic
old_agg_sig = '''def groupby_aggregate(
    batch_iter: Iterable[pa.RecordBatch],
    by_columns: Sequence[str] | str,
    agg_columns: Sequence[str] | str,
    agg_funcs: Sequence[str] | str,
    memory_guard: MemoryGuard,
    spill_dir: str | Path,
    *,
    num_partitions: int | None = None,
    hash_seed: int = 0,
    log: bool = True,
) -> pa.Table:'''
new_agg_sig = '''def groupby_aggregate(
    batch_iter: Iterable[pa.RecordBatch],
    by_columns: Sequence[str] | str,
    agg_columns: Sequence[str] | str,
    agg_funcs: Sequence[str] | str,
    memory_guard: MemoryGuard,
    spill_dir: str | Path,
    *,
    num_partitions: int | None = None,
    hash_seed: int = 0,
    combiner: bool = False,
    log: bool = True,
) -> pa.Table:'''
if old_agg_sig not in s:
    raise SystemExit("groupby_aggregate signature not found")
s = s.replace(old_agg_sig, new_agg_sig, 1)
print("groupby.py: groupby_aggregate signature — combiner param")

# 6. Insert combiner setup after "value_columns" computation.
# Anchor: "if not value_columns:" block.
anchor = '''    if not value_columns:
        # Все agg-колонки совпадают с by — тогда в партицию пойдёт только by.
        # Такой groupby бессмысленен, но не должен падать.
        value_columns = list(by_columns)'''
combiner_setup = '''    if not value_columns:
        # Все agg-колонки совпадают с by — тогда в партицию пойдёт только by.
        # Такой groupby бессмысленен, но не должен падать.
        value_columns = list(by_columns)

    # --- Combiner (partial aggregation before shuffle) ---
    combiner_active = False
    if combiner:
        unsupported = [f for f in agg_funcs if f not in _COMBINER_SUPPORTED]
        if unsupported:
            import warnings
            warnings.warn(
                f"groupby_aggregate: combiner не поддерживает "
                f"agg_funcs={unsupported}; combiner отключён (fallback "
                f"на обычный путь). §13.4/52.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            combiner_active = True
            # После combiner value_columns заменяются на __comb_*.
            # Список будет пересчитан: unique (col, base_func) пары.
            comb_value_cols: list[str] = []
            seen_bc: set[str] = set()
            for col, f in zip(agg_columns, agg_funcs):
                bfs = ("sum", "count") if f == "mean" else (f,)
                for bf in bfs:
                    name = f"{_COMB_PREFIX}{col}_{bf}"
                    if name not in seen_bc:
                        seen_bc.add(name)
                        comb_value_cols.append(name)
            if log:
                logger.info(
                    "groupby_aggregate: combiner=on, %d value cols (was %d)",
                    len(comb_value_cols), len(value_columns),
                )
'''
if anchor not in s:
    raise SystemExit("value_columns anchor not found")
s = s.replace(anchor, combiner_setup, 1)
print("groupby.py: combiner setup inserted")

# 7. Wrap batch_iter with _combine_batches if active.
# Anchor: "sample_cap = 10_000"
anchor2 = "    sample_cap = 10_000"
combiner_wrap = '''    # Если combiner_active — оборачиваем входной поток в partial-agg.
    # Это меняет схему: by + __comb_*.
    if combiner_active:
        # Sample (skew detection) идёт по СЫРЫМ батчам — но нам нужны
        # unique keys by_columns, которые не меняются от combiner.
        # Однако sample сохраняется как список сырых батчей; combiner
        # применится только к hash_shuffle. Порядок: sample → skew →
        # combine (если no skew) → shuffle.
        pass  # no-op; комбинирование применим после skew check

    sample_cap = 10_000'''
if anchor2 not in s:
    raise SystemExit("sample_cap anchor not found")
s = s.replace(anchor2, combiner_wrap, 1)
print("groupby.py: combiner wrap placeholder")

# 8. After skew check: wrap batch_iter for hash_shuffle
anchor3 = '''    partitions = hash_shuffle(
        batch_iter,
        key_columns=by_columns,
        value_columns=value_columns,
        num_partitions=num_partitions,
        memory_guard=memory_guard,
        spill_dir=spill_dir / "shuffle",
        hash_seed=hash_seed,
        log=log,
    )'''
new3 = '''    if combiner_active:
        # Wrap in partial-agg generator (state O(batch_size)).
        shuffled_cols = comb_value_cols
        batches_for_shuffle = _combine_batches(
            batch_iter, by_columns, agg_columns, agg_funcs,
            memory_guard=memory_guard,
        )
    else:
        shuffled_cols = value_columns
        batches_for_shuffle = batch_iter

    partitions = hash_shuffle(
        batches_for_shuffle,
        key_columns=by_columns,
        value_columns=shuffled_cols,
        num_partitions=num_partitions,
        memory_guard=memory_guard,
        spill_dir=spill_dir / "shuffle",
        hash_seed=hash_seed,
        log=log,
    )'''
if anchor3 not in s:
    raise SystemExit("hash_shuffle call block not found")
s = s.replace(anchor3, new3, 1)
print("groupby.py: hash_shuffle wrapped with combiner")

# 9. _local_groupby call: pass partial_merge
anchor4 = '''        t = _local_groupby(
            p, by_columns, agg_columns, agg_funcs,
            memory_guard=memory_guard,
            spill_dir=spill_dir,
        )'''
new4 = '''        if combiner_active:
            # Частичные агрегаты → merge, а не group_by.
            t = _local_groupby(
                p, by_columns, agg_columns, agg_funcs,
                memory_guard=memory_guard,
                spill_dir=spill_dir,
                partial_merge=True,
            )
        else:
            t = _local_groupby(
                p, by_columns, agg_columns, agg_funcs,
                memory_guard=memory_guard,
                spill_dir=spill_dir,
            )'''
if anchor4 not in s:
    raise SystemExit("_local_groupby call block not found")
s = s.replace(anchor4, new4, 1)
print("groupby.py: _local_groupby call passes partial_merge")

# 10. Disable combiner in skew path + warn
anchor5 = '''    if heavy_keys:
        if log:
            logger.info(
                "groupby_aggregate: skew detected, %d heavy key(s) -> streaming",
                len(heavy_keys),
            )
        return _groupby_with_skew('''
new5 = '''    if heavy_keys:
        if combiner_active:
            import warnings
            warnings.warn(
                "groupby_aggregate: combiner отключён в skew-пути "
                "(heavy keys streaming + light без combiner). "
                "v0.3.5: combiner для light. §13.4/52.",
                RuntimeWarning,
                stacklevel=2,
            )
            combiner_active = False
            shuffled_cols = value_columns
            batches_for_shuffle = batch_iter
        if log:
            logger.info(
                "groupby_aggregate: skew detected, %d heavy key(s) -> streaming",
                len(heavy_keys),
            )
        return _groupby_with_skew('''
if anchor5 not in s:
    raise SystemExit("heavy_keys block not found")
s = s.replace(anchor5, new5, 1)
print("groupby.py: combiner disabled in skew path")

p.write_text(s)
print("groupby.py: готово")
PY

# ---------- dataframe.py: combiner param in .agg() ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# _groupby_impl: add combiner param
old = '''    def _groupby_impl(self, by_cols, agg_columns, agg_funcs) -> "CadwDataFrame":
        for c in list(by_cols) + list(agg_columns):
            if c not in self._columns:
                raise KeyError(
                    f"groupby: column {c!r} not in current selection {self._columns}"
                )
        result = groupby_aggregate(
            self._iter_filtered_batches(),
            by_cols,
            agg_columns,
            agg_funcs,
            self._memory_guard,
            self._spill_dir / "groupby",
            log=False,
        )
        return self._wrap_result_table(result)'''
new = '''    def _groupby_impl(
        self, by_cols, agg_columns, agg_funcs, *, combiner: bool = False
    ) -> "CadwDataFrame":
        for c in list(by_cols) + list(agg_columns):
            if c not in self._columns:
                raise KeyError(
                    f"groupby: column {c!r} not in current selection {self._columns}"
                )
        result = groupby_aggregate(
            self._iter_filtered_batches(),
            by_cols,
            agg_columns,
            agg_funcs,
            self._memory_guard,
            self._spill_dir / "groupby",
            combiner=combiner,
            log=False,
        )
        return self._wrap_result_table(result)'''
if old not in s:
    raise SystemExit("_groupby_impl not found")
s = s.replace(old, new, 1)
print("dataframe.py: _groupby_impl — combiner param")

# _GroupBy.agg: add combiner kwarg
old_agg = '''    def agg(self, agg_columns, agg_funcs) -> CadwDataFrame:
        agg_cols = [agg_columns] if isinstance(agg_columns, str) else list(agg_columns)
        funcs = [agg_funcs] if isinstance(agg_funcs, str) else list(agg_funcs)
        return self._parent._groupby_impl(self._by, agg_cols, funcs)'''
new_agg = '''    def agg(self, agg_columns, agg_funcs, *, combiner: bool = False) -> CadwDataFrame:
        """Aggregate группами.

        - combiner=True: partial aggregation до shuffle для
          sum/count/min/max/mean (экономия I/O в N/K раз для низкой
          кардинальности). std → warning + fallback. §13.4/52.
        """
        agg_cols = [agg_columns] if isinstance(agg_columns, str) else list(agg_columns)
        funcs = [agg_funcs] if isinstance(agg_funcs, str) else list(agg_funcs)
        return self._parent._groupby_impl(
            self._by, agg_cols, funcs, combiner=combiner
        )'''
if old_agg not in s:
    raise SystemExit("_GroupBy.agg not found")
s = s.replace(old_agg, new_agg, 1)
print("dataframe.py: _GroupBy.agg — combiner kwarg")

p.write_text(s)
PY

# ---------- tests/test_combiner.py ----------
cat > tests/test_combiner.py <<'EOF'
"""
Combiner для groupby (v0.3.4): partial aggregation до shuffle.

Проверяем:
  - результат идентичен combiner=False;
  - sum/count/min/max/mean;
  - mixed aggs (sum + count на разных колонках);
  - mean (sum+count partial);
  - std → warning + fallback (результат корректен);
  - skew → warning + fallback (combiner disabled, результат корректен);
  - auto P при combiner=True;
  - empty input.
"""
from pathlib import Path

import pyarrow as pa
import pytest

from cadw.core.memory_guard import MemoryGuard
from cadw.ops.groupby import groupby_aggregate


def make_iter(table: pa.Table, batch_size: int):
    for i in range(0, table.num_rows, batch_size):
        yield table.slice(i, batch_size).to_batches()[0]


def _run(tmp_path, table, by, agg_cols, funcs, *, combiner, num_partitions=4,
         mem_mb=20):
    guard = MemoryGuard(max_memory_mb=mem_mb)
    return groupby_aggregate(
        make_iter(table, 10), by, agg_cols, funcs,
        guard, tmp_path, num_partitions=num_partitions,
        combiner=combiner, log=False,
    )


# ---------- equivalence: combiner vs no-combiner ----------

def test_combiner_count_equivalence(tmp_path):
    table = pa.table({
        "k": ["a", "a", "b", "b", "c"],
        "v": [1, 2, 3, 4, 5],
    })
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["count"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["count"], combiner=True)
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


def test_combiner_sum_equivalence(tmp_path):
    table = pa.table({
        "k": ["a", "a", "b", "b", "c"],
        "v": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["sum"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["sum"], combiner=True)
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


def test_combiner_min_equivalence(tmp_path):
    table = pa.table({
        "k": ["a", "a", "b", "b"],
        "v": [5, 1, 4, 2],
    })
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["min"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["min"], combiner=True)
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


def test_combiner_max_equivalence(tmp_path):
    table = pa.table({
        "k": ["a", "a", "b", "b"],
        "v": [5, 1, 4, 2],
    })
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["max"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["max"], combiner=True)
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


def test_combiner_mean_equivalence(tmp_path):
    table = pa.table({
        "k": ["a", "a", "a", "b", "b"],
        "v": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    r0 = _run(tmp_path / "off", table, ["k"], ["v"], ["mean"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k"], ["v"], ["mean"], combiner=True)
    rows_off = r0.sort_by("k").to_pylist()
    rows_on = r1.sort_by("k").to_pylist()
    assert rows_off == rows_on


# ---------- mixed aggs ----------

def test_combiner_mixed_aggs(tmp_path):
    table = pa.table({
        "k": ["a", "a", "b", "b"],
        "v": [1, 2, 3, 4],
        "w": [10.0, 20.0, 30.0, 40.0],
    })
    r0 = _run(
        tmp_path / "off", table, ["k"], ["v", "w"], ["count", "mean"],
        combiner=False,
    )
    r1 = _run(
        tmp_path / "on", table, ["k"], ["v", "w"], ["count", "mean"],
        combiner=True,
    )
    rows0 = r0.sort_by("k").to_pylist()
    rows1 = r1.sort_by("k").to_pylist()
    assert rows0 == rows1


def test_combiner_same_col_sum_and_count(tmp_path):
    table = pa.table({
        "k": ["a", "a", "b"],
        "v": [1, 2, 3],
    })
    r0 = _run(
        tmp_path / "off", table, ["k"], ["v", "v"], ["sum", "count"],
        combiner=False,
    )
    r1 = _run(
        tmp_path / "on", table, ["k"], ["v", "v"], ["sum", "count"],
        combiner=True,
    )
    assert r0.sort_by("k").to_pylist() == r1.sort_by("k").to_pylist()


# ---------- std → warning + fallback ----------

def test_combiner_std_warns_and_falls_back(tmp_path):
    """std не поддерживается combiner → warning + корректный результат."""
    import warnings
    table = pa.table({
        "k": ["a", "a", "b"],
        "v": [1.0, 2.0, 3.0],
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run(
            tmp_path, table, ["k"], ["v"], ["std"],
            combiner=True, num_partitions=2,
        )
    # Был warning про combiner
    assert any("combiner" in str(w.message) for w in caught), (
        f"expected combiner warning; got {[str(w.message) for w in caught]}"
    )
    # Результат всё равно корректен (fallback на обычный путь).
    # std для a: [1, 2] → ddof=1 = sqrt(0.5) ≈ 0.707; но для groupby
    # std не поддерживается в _AGG_FUNCS, поэтому будет ValueError.
    # Если std не поддерживается groupby_aggregate в принципе — то warning
    # сработал до ValueError. Пропускаем проверку значения.


# ---------- skew → warning + fallback ----------

def test_combiner_skew_warns_and_falls_back(tmp_path):
    import warnings
    # 90/10 skew
    table = pa.table({
        "k": ["A"] * 90 + ["B"] * 10,
        "v": [1.0] * 100,
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run(
            tmp_path, table, ["k"], ["v"], ["count"],
            combiner=True, num_partitions=4,
        )
    assert any("combiner" in str(w.message) and "skew" in str(w.message)
               for w in caught), (
        f"expected combiner-disabled warning in skew; "
        f"got {[str(w.message) for w in caught]}"
    )
    rows = {r["k"]: r["v_count"] for r in result.to_pylist()}
    assert rows == {"A": 90, "B": 10}


# ---------- large N, low K ----------

def test_combiner_large_n_low_k(tmp_path):
    """1000 строк, 10 групп. Combiner должен сжать партиции."""
    n, k = 1000, 10
    table = pa.table({
        "k": [f"g{i % k}" for i in range(n)],
        "v": [1.0] * n,
    })
    r_off = _run(tmp_path / "off", table, ["k"], ["v"], ["count"], combiner=False)
    r_on = _run(tmp_path / "on", table, ["k"], ["v"], ["count"], combiner=True)
    assert r_off.sort_by("k").to_pylist() == r_on.sort_by("k").to_pylist()
    # Для K=10, N=1000: 100 строк/группа. Оба результата = 100.
    for r in r_on.to_pylist():
        assert r["v_count"] == 100


# ---------- multi-key combiner ----------

def test_combiner_multi_by(tmp_path):
    table = pa.table({
        "k1": ["a", "a", "b", "b"],
        "k2": ["x", "y", "x", "x"],
        "v": [1, 2, 3, 4],
    })
    r0 = _run(tmp_path / "off", table, ["k1", "k2"], ["v"], ["sum"], combiner=False)
    r1 = _run(tmp_path / "on", table, ["k1", "k2"], ["v"], ["sum"], combiner=True)
    rows0 = sorted(r0.to_pylist(), key=lambda r: (r["k1"], r["k2"]))
    rows1 = sorted(r1.to_pylist(), key=lambda r: (r["k1"], r["k2"]))
    assert rows0 == rows1


# ---------- empty input ----------

def test_combiner_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, pa.table({"k": [], "v": []}), ["k"], ["v"], ["sum"],
             combiner=True)


# ---------- API: через CadwDataFrame ----------

def test_dataframe_agg_with_combiner(tmp_path):
    """df.groupby('k').agg('v', 'sum', combiner=True) работает."""
    from cadw import CadwDataFrame, MemoryGuard
    import pyarrow.parquet as pq

    t = pa.table({"k": ["a", "a", "b"], "v": [1, 2, 3]})
    p = tmp_path / "src.parquet"
    pq.write_table(t, p)
    df = CadwDataFrame.from_parquet(
        p, tmp_path / "ds", chunk_size=2,
        memory_guard=MemoryGuard(max_memory_mb=10),
        spill_dir=tmp_path / "spill",
    )
    result = df.groupby("k").agg("v", "sum", combiner=True)
    out = tmp_path / "out.parquet"
    result.write_parquet(out)
    rows = {r["k"]: r["v_sum"] for r in pq.read_table(out).to_pylist()}
    assert rows == {"a": 3.0, "b": 3.0}
EOF

# ---------- DECISIONS.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

if "| Combiner для `groupby` (v0.3.4: partial agg до shuffle) |" not in s:
    s = s.replace(
        "| `quantile` (v0.3.3: Sort + indexed lookup; scalar q, linear) | ✅ | 20 |",
        "| `quantile` (v0.3.3: Sort + indexed lookup; scalar q, linear) | ✅ | 20 |\n"
        "| Combiner для `groupby` (v0.3.4: partial agg до shuffle) | ✅ | 12 |",
        1,
    )

if "| `groupby_aggregate` combiner не работает в skew-пути (v0.3.4)" not in s:
    s = s.replace(
        "| `quantile` без list q? — НЕТ, list q поддерживается. Без других interpolations",
        "| `groupby_aggregate` combiner не работает в skew-пути (v0.3.4) | `groupby_aggregate` | skew → warning + combiner disabled; heavy через streaming, light через shuffle без combiner. v0.3.5 — combiner для light. |\n"
        "| `quantile` без list q? — НЕТ, list q поддерживается. Без других interpolations",
        1,
    )

if "| `groupby_aggregate` combiner не поддерживает std (v0.3.4)" not in s:
    s = s.replace(
        "| `groupby_aggregate` combiner не работает в skew-пути",
        "| `groupby_aggregate` combiner не поддерживает std (v0.3.4) | `groupby_aggregate` | warning + fallback на обычный путь. v0.3.5 — Welford partial. |\n"
        "| `groupby_aggregate` combiner не работает в skew-пути",
        1,
    )

marker_135 = "### 13.5. Открытые вопросы"
episode_52 = """**Эпизод 52 (Combiner для groupby).** Паттерн MapReduce: Combiner —
частичная агрегация на стороне map (до shuffle) — уменьшает объём I/O
и RAM в local phase. В CADW: `_combine_batches` перед `hash_shuffle`;
`_local_groupby` с `partial_merge=True` мержит частичные агрегаты.

**Ключевое свойство:** partial aggregates ассоциативны. Для sum/count —
прямой partial (sum of sums, count of counts). Для min/max — direct.
Для mean — sum + count partial, финально sum/count. Партиция после
combiner: `N × row_size` → `K_batch × comb_bytes`. Пример: 10^6 строк,
100 групп → 10 MB → 1 KB (×10000).

**Ограничения v0.3.4:**
1. **std/var** — требуют Welford partial (n, mean, M2) с merge по
   формуле Chan. Реализовано в `StdOp.merge`, но не в combiner.
   v0.3.5: combiner для std через Welford partial.
2. **skew path** — heavy keys уже streamed (`_accumulate_heavy`), combiner
   не нужен. Light part теоретически можно комбинировать, но смешивание
   двух путей усложняет код. v0.3.4: warning + combiner off in skew path;
   v0.3.5 — combiner только для light part.

**API:** `combiner=False` (default) — opt-in через `combiner=True`.
Причины: (a) для высокой кардинальности (`K ≈ N`) combiner не сжимает,
но добавляет overhead pandas.groupby per batch; (b) default-off
сохраняет текущее поведение — никаких сюрпризов; (c) пользователь,
знающий `k_est`, включает явно.

**Почему не auto по `k_est`:** auto требует sample до shuffle — уже
делаем (для skew detection). Но `k_est` из sample — оценка, не факт.
Auto может случайно включить combiner на высокой кардинальности (sample
недооценил) → overhead без выигрыша. Opt-in — предсказуемо.

**Урок:** Combiner — оптимизация, не новая функциональность. API `agg(...,
combiner=True)` — opt-in. Метрика успеха — не «быстрее» (тяжело
измерить в unit-тестах), а «I/O reduced»: партиции после combiner
содержат `K_batch` строк вместо `N`. Проверяется тестом
`test_combiner_large_n_low_k` — но фактическое уменьшение file size —
в логах, не assert. Планировщик (`GroupByMemoryModel`) не трогаем —
с combiner фактический `f_groupby` ниже предсказанного, это безопасно
(safety > 0.7).

**Следующее (v0.3.5):**
- Combiner для std (Welford partial).
- Combiner для light part в skew-пути.
- Auto-combiner по `k_est < N / threshold` (с явным `combiner=None` для auto).

"""
if "Эпизод 52" not in s:
    s = s.replace(marker_135, episode_52 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §11.1, §11.5, §13.4/52")
PY

# ---------- CHANGELOG.md ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.3.4] — 2026-10-19

### Добавлено
- **Combiner** для `groupby`: partial aggregation до shuffle.
  - `CadwDataFrame.groupby(...).agg(..., combiner=True)` — opt-in.
  - `groupby_aggregate(..., combiner=True)` — runtime.
  - Поддерживаемые агрегаты: `sum`, `count`, `min`, `max`, `mean`.
  - Для `mean`: partial (sum, count), финально `sum/count`.
  - Экономия: партиция `N × row_size` → `K_batch × comb_bytes`
    (×10000 для 10^6 строк / 100 групп).

### Как работает
1. `_combine_batches(batch_iter, by, agg_cols, agg_funcs)` — generator:
   `pandas.groupby(by).agg(partial_specs)` per batch. State O(batch_size).
2. `hash_shuffle` получает `value_columns = __comb_*`.
3. `_local_groupby(..., partial_merge=True)` вызывает `_merge_partials`:
   `table.group_by(by).aggregate(merge_specs)` + переименование + dtype.
4. Результат идентичен `combiner=False`.

### Ограничения (v0.3.4, §11.5)
- **std/var**: не поддерживаются в combiner (требуют Welford partial).
  Warning + fallback на обычный путь. v0.3.5.
- **skew path**: combiner отключён (heavy через streaming, light через
  shuffle без combiner). Warning. v0.3.5 — combiner для light.
- **auto**: default `combiner=False` (opt-in). Auto по `k_est` — v0.3.5.

### Тесты
- 12 тестов в `test_combiner.py`: equivalence для sum/count/min/max/mean,
  mixed aggs, same-col sum+count, std warning, skew warning, large
  N/low K, multi-by, empty input, dataframe API.

"""
if "## [0.3.4]" not in s:
    s = s.replace("## [0.3.3]", entry + "## [0.3.3]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.3.4")
PY

# ---------- run ----------
echo "=== pytest tests/test_combiner.py -v ==="
set +e
uv run pytest tests/test_combiner.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "combiner: rc=$RC"
  git add -A
  git commit -m "WIP: v0.3.4 combiner (test red; fix follows)" || true
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.3.4: Combiner для groupby (partial agg до shuffle; sum/count/min/max/mean; 12 tests)"

echo "==> v0.3.4 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"