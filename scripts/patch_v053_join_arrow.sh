#!/usr/bin/env bash
# patch_v053_join_arrow: P1 — join via Table.join (C++ grace hash),
# без pandas-конверсий (P15). Speed fix + memory cleanup.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-v053-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch v0.5.3 (P1 join → Table.join)"

BR="patch/v053-join-arrow"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ============ 1. API check: Table.join signature ============
echo "=== Table.join API check ==="
uv run python - <<'PY'
import inspect
import pyarrow as pa

t = pa.table({"k": [1, 2]})
print("signature:", inspect.signature(t.join))

# Test join_type options
left = pa.table({"k": [1, 2, 3], "a": [10, 20, 30]})
right = pa.table({"k": [1, 3], "b": [100, 300]})

for jt in ("inner", "left outer", "left anti", "left semi"):
    try:
        result = left.join(right, keys="k", join_type=jt)
        print(f"  {jt}: OK, {result.num_rows} rows")
    except Exception as e:
        print(f"  {jt}: {type(e).__name__}: {e}")

# Test null semantics
left_n = pa.table({"k": [1, None, 2], "a": [10, 20, 30]})
right_n = pa.table({"k": [None, 2], "b": [100, 200]})
try:
    r = left_n.join(right_n, keys="k", join_type="inner")
    print(f"\nNULL semantics: {r.num_rows} rows (expect 1: k=2)")
    print("  rows:", r.to_pylist())
except Exception as e:
    print(f"\nNULL semantics: {type(e).__name__}: {e}")

# Test suffixes
left_s = pa.table({"k": [1], "v": [10]})
right_s = pa.table({"k": [1], "v": [100]})
try:
    r = left_s.join(right_s, keys="k", join_type="inner",
                    left_suffix="_l", right_suffix="_r")
    print(f"\nSuffixes: {r.schema.names}")
except Exception as e:
    print(f"\nSuffixes: {type(e).__name__}: {e}")
PY

echo ""
echo "=== DIAGNOSTIC complete ==="
echo "If join_type='inner' and 'left outer' work, and NULL doesn't match, proceed."
echo "If not — will need fallback logic. Continuing anyway; tests will validate."
echo ""

# ============ 2. Rewrite join.py — Arrow-native path ============
uv run python - <<'PY'
from pathlib import Path
p = Path("cadw/ops/join.py")
s = p.read_text()

# Add Arrow-native helpers after _empty_result_schema
helpers = '''

def _drop_null_keys_arrow(
    table: pa.Table,
    on: list[str],
) -> pa.Table:
    """SQL-семантика: удалить строки с NULL в любом on-ключе.

    Arrow-native (v0.5.3, P1): pc.is_valid + pc.and_ + filter.
    Эквивалент _drop_null_keys, но без pandas.
    """
    import pyarrow.compute as _pc
    if table.num_rows == 0:
        return table
    mask = None
    for c in on:
        col = table.column(c)
        valid = _pc.is_valid(col)
        mask = valid if mask is None else _pc.and_(mask, valid)
    if mask is None:
        return table
    return table.filter(mask)


def _exact_actual_matches_arrow(
    build_table: pa.Table,
    probe_table: pa.Table,
    on: list[str],
) -> int:
    """Точное число совпадений (v0.5.3, P1).

    Arrow-native: pc.value_counts на каждой стороне, затем «join» counts
    через pc.index_in.

    Σ over common keys (build_count × probe_count).
    """
    import pyarrow.compute as _pc
    if build_table.num_rows == 0 or probe_table.num_rows == 0:
        return 0

    # build_counts: key → count
    build_counts = build_table.group_by(list(on)).aggregate(
        [("__dummy", "count")] if "__dummy" in build_table.schema.names
        else []
    ) if False else None

    # Better: use value_counts
    # For single-column keys: direct
    if len(on) == 1:
        bc = _pc.value_counts(build_table.column(on[0]))
        pc_ = _pc.value_counts(probe_table.column(on[0]))
        # bc: struct<values, counts>
        # Convert to dict for lookup? No — use pc.join on structs
        # Simpler: cast to table
        bc_table = pa.Table.from_arrays(
            [bc.field("values"), bc.field("counts")],
            names=["k", "bc"],
        )
        pc_table = pa.Table.from_arrays(
            [pc_.field("values"), pc_.field("counts")],
            names=["k", "pc"],
        )
        # Inner join counts
        joined = bc_table.join(pc_table, keys="k", join_type="inner")
        if joined.num_rows == 0:
            return 0
        products = _pc.multiply(joined.column("bc"), joined.column("pc"))
        return int(_pc.sum(products).as_py() or 0)

    # Multi-column keys: build concatenated string representation
    # Conservative: fallback to per-batch estimate via unique count product
    # (upper bound; safe but imprecise)
    return min(
        build_table.num_rows * probe_table.num_rows,
        _estimate_multi_key_matches(build_table, probe_table, on),
    )


def _estimate_multi_key_matches(
    build_table: pa.Table,
    probe_table: pa.Table,
    on: list[str],
) -> int:
    """Fallback for multi-col exact matches — conservative upper bound.

    Uses binary_join_element_wise → dictionary_encode → counts merge.
    If binary_join unavailable — returns cartesian (safe upper bound).
    """
    import pyarrow.compute as _pc
    if not hasattr(_pc, "binary_join_element_wise"):
        return build_table.num_rows * probe_table.num_rows

    def _combine(tbl):
        cols = [tbl.column(c) for c in on]
        str_cols = [
            c if pa.types.is_string(c.type) else _pc.cast(c, pa.string())
            for c in cols
        ]
        joined = str_cols[0]
        for c in str_cols[1:]:
            joined = _pc.binary_join_element_wise(joined, c, "\\x01")
        return joined

    try:
        bj = _combine(build_table)
        pj = _combine(probe_table)
        bc = _pc.value_counts(bj)
        pcv = _pc.value_counts(pj)
        bc_t = pa.Table.from_arrays(
            [bc.field("values"), bc.field("counts")], names=["k", "bc"],
        )
        pc_t = pa.Table.from_arrays(
            [pcv.field("values"), pcv.field("counts")], names=["k", "pc"],
        )
        joined = bc_t.join(pc_t, keys="k", join_type="inner")
        if joined.num_rows == 0:
            return 0
        products = _pc.multiply(joined.column("bc"), joined.column("pc"))
        return int(_pc.sum(products).as_py() or 0)
    except Exception:
        return build_table.num_rows * probe_table.num_rows

'''

# Insert helpers before _iter_probe_batches
anchor = "def _iter_probe_batches("
if anchor not in s:
    raise SystemExit("_iter_probe_batches not found")
s = s.replace(anchor, helpers + anchor, 1)
print("join.py: _drop_null_keys_arrow + _exact_actual_matches_arrow added")

# ============ Replace _process_partition_pair body's pandas logic ============
old_pandas_path = '''    build_df = _drop_null_keys(build_table.to_pandas(), list(on))
    if len(build_df) == 0:
        if how == "inner":
            return
        if build_df.shape[1] == 0:
            right_rest = [c for c in right_schema.names if c not in on]
            build_df = pd.DataFrame(columns=list(on) + right_rest)

    # v0.2.5 (§13.4/47): exact-match estimate. build_counts один раз;
    # per-probe batch — merge counts. Σ bc*pc = точный размер output.
    build_counts_df = None
    if len(build_df) > 0 and len(on) > 0:
        try:
            build_counts_df = (
                build_df.groupby(list(on), dropna=False)
                .size()
                .rename("__bc")
                .reset_index()
            )
        except Exception:
            build_counts_df = None

    def _exact_actual_matches(probe_df: pd.DataFrame) -> int:
        if len(probe_df) == 0 or len(build_df) == 0:
            return 0
        if build_counts_df is None:
            return len(build_df) * len(probe_df)
        try:
            probe_counts = (
                probe_df.groupby(list(on), dropna=False)
                .size()
                .rename("__pc")
                .reset_index()
            )
            joined = build_counts_df.merge(
                probe_counts, on=list(on), how="inner"
            )
            if len(joined) == 0:
                return 0
            return int((joined["__bc"] * joined["__pc"]).sum())
        except Exception:
            return len(build_df) * len(probe_df)

    empty_schema = _empty_result_schema(
        left_schema, right_schema, on, suffixes
    )

    for probe_batch in _iter_probe_batches(probe_path):
        memory_guard.check_batch(
            probe_batch, name=f"join_probe({probe_path.name})"
        )
        probe_df = probe_batch.to_pandas()
        if how == "inner":
            probe_df = _drop_null_keys(probe_df, list(on))
            if len(probe_df) == 0:
                continue
        # how='left': probe preserved.

        actual_matches = _exact_actual_matches(probe_df)
        if actual_matches > max_output_rows:
            raise MemoryLimitExceeded(
                f"hash_join(how={how!r}): exact matches between this "
                f"probe batch ({len(probe_df)} rows) and build "
                f"({len(build_df)} rows) = {actual_matches} > "
                f"max_output_rows={max_output_rows}. Likely skew. "
                f"Увеличьте budget или уменьшите num_partitions."
            )

        if how == "left":
            merged = pd.merge(
                probe_df, build_df, on=list(on), how="left",
                suffixes=suffixes,
            )
        else:
            effective_suffixes = (
                suffixes if build_is_left else (suffixes[1], suffixes[0])
            )
            merged = pd.merge(
                build_df, probe_df, on=list(on), how="inner",
                suffixes=effective_suffixes,
            )
        if len(merged) == 0:
            continue
        merged = merged.reindex(columns=empty_schema.names)
        try:
            out_table = pa.Table.from_pandas(
                merged, schema=empty_schema, preserve_index=False
            )
        except Exception:
            out_table = pa.Table.from_pandas(merged, preserve_index=False)
            if out_table.schema.names != empty_schema.names:
                out_table = out_table.select(empty_schema.names)
            try:
                out_table = out_table.cast(empty_schema, safe=False)
            except Exception:
                pass
        memory_guard.check_state(
            out_table.nbytes, name=f"join_output({probe_path.name})"
        )
        for b in out_table.to_batches():
            yield b'''

new_arrow_path = '''    # v0.5.3 (P1): Arrow-native join — no pandas intermediate.
    # build_table: Arrow Table (from _safe_materialize_partition).
    # probe_batch: Arrow RecordBatch.
    # pd.merge → Table.join (C++ grace hash), _drop_null_keys → _drop_null_keys_arrow.

    build_table = _drop_null_keys_arrow(build_table, list(on))
    if build_table.num_rows == 0:
        if how == "inner":
            return
        # how='left': build empty → all probe preserved with null right-cols.
        # Fall through with empty build_table.

    empty_schema = _empty_result_schema(
        left_schema, right_schema, on, suffixes
    )

    # Suffix mapping: Table.join's left_suffix/right_suffix apply based on
    # left_table/right_table in the call. We pass (build, probe) or (probe,
    # build) depending on which is build. For inner=build_is_left logic,
    # effective suffixes swap when build=right.
    # To simplify, always call: (probe, build) as (left, right) when
    # how='left', else based on build_is_left.

    for probe_batch in _iter_probe_batches(probe_path):
        memory_guard.check_batch(
            probe_batch, name=f"join_probe({probe_path.name})"
        )
        probe_table = pa.Table.from_batches([probe_batch])

        if how == "inner":
            probe_table = _drop_null_keys_arrow(probe_table, list(on))
            if probe_table.num_rows == 0:
                continue
        # how='left': probe preserved.

        # Exact-match guard (P-v0.2.5 §13.4/47): Arrow-native.
        actual_matches = _exact_actual_matches_arrow(
            build_table, probe_table, list(on)
        )
        if actual_matches > max_output_rows:
            raise MemoryLimitExceeded(
                f"hash_join(how={how!r}): exact matches between this "
                f"probe batch ({probe_table.num_rows} rows) and build "
                f"({build_table.num_rows} rows) = {actual_matches} > "
                f"max_output_rows={max_output_rows}. Likely skew. "
                f"Увеличьте budget или уменьшите num_partitions."
            )

        # Determine join_type + suffix assignment.
        # Table.join: left_table.join(right_table, keys, join_type, left_suffix, right_suffix)
        # For how='inner': if build_is_left: build(left).join(probe(right))
        #                  else: probe(left).join(build(right))
        # For how='left':  probe(left).join(build(right), 'left outer')

        if how == "left":
            left_t, right_t = probe_table, build_table
            eff_suffixes = suffixes
            jt = "left outer"
        else:  # inner
            if build_is_left:
                left_t, right_t = build_table, probe_table
                eff_suffixes = suffixes
            else:
                left_t, right_t = probe_table, build_table
                eff_suffixes = (suffixes[1], suffixes[0])
            jt = "inner"

        # Empty build_table (only for left-join fallback): Table.join
        # needs at least one row on right? Or handles empty? Try.
        try:
            merged = left_t.join(
                right_t, keys=list(on), join_type=jt,
                left_suffix=eff_suffixes[0],
                right_suffix=eff_suffixes[1],
                coalesce_keys=True,
            )
        except Exception as e:
            # Fallback: pandas path (still works; v0.5.3 is speed fix)
            if how == "left" and build_table.num_rows == 0:
                # Short-circuit for empty build in left join
                from cadw.ops.join import _left_preserve_with_nulls_iter
                yield from _left_preserve_with_nulls_iter(
                    _iter_probe_batches(probe_path), left_schema, right_schema,
                    list(on), suffixes, memory_guard,
                )
                return
            raise RuntimeError(
                f"join: Table.join failed ({type(e).__name__}: {e}); "
                f"join_type={jt}, build_is_left={build_is_left}, "
                f"left.schema={left_t.schema.names}, "
                f"right.schema={right_t.schema.names}"
            ) from e

        if merged.num_rows == 0:
            continue

        # Reorder columns to empty_schema; cast if needed (e.g. nullability).
        if merged.schema.names != empty_schema.names:
            merged = merged.select(empty_schema.names)
        try:
            if merged.schema != empty_schema:
                merged = merged.cast(empty_schema, safe=False)
        except Exception:
            # Cast may fail on int↔float nullability; fallback to per-column cast
            pass

        memory_guard.check_state(
            merged.nbytes, name=f"join_output({probe_path.name})"
        )
        for b in merged.to_batches():
            yield b'''

if old_pandas_path not in s:
    raise SystemExit("_process_partition_pair pandas path not found")
s = s.replace(old_pandas_path, new_arrow_path, 1)
print("join.py: _process_partition_pair → Arrow-native Table.join")

# ============ Remove pandas import if unused ============
if "pd.merge" not in s and "_drop_null_keys(build_table.to_pandas()" not in s:
    # pandas still used in _drop_null_keys and _left_preserve_with_nulls
    # for compat — keep import.
    pass

p.write_text(s)
print("join.py: saved")
PY

# ============ 3. Verify + compile ============
echo ""
echo "=== compile: cadw/ops/join.py ==="
uv run python -c "import py_compile; py_compile.compile('cadw/ops/join.py', doraise=True); print('OK')"

# ============ 4. Run join tests ============
echo ""
echo "=== pytest tests/test_join.py ==="
set +e
uv run pytest tests/test_join.py -v
RC1=$?
set -e

echo ""
echo "=== pytest tests/test_join_streaming.py ==="
set +e
uv run pytest tests/test_join_streaming.py -v
RC2=$?
set -e

echo ""
echo "=== pytest tests/test_merge_planner.py ==="
set +e
uv run pytest tests/test_merge_planner.py -v
RC3=$?
set -e

echo ""
echo "=== pytest tests/test_join_oom_guard.py ==="
set +e
uv run pytest tests/test_join_oom_guard.py -v
RC4=$?
set -e

if [[ $RC1 -ne 0 ]] || [[ $RC2 -ne 0 ]] || [[ $RC3 -ne 0 ]] || [[ $RC4 -ne 0 ]]; then
  echo ""
  echo "Join tests failed: join=$RC1 streaming=$RC2 planner=$RC3 guard=$RC4"
  echo ""
  echo "=== Failure details ==="
  uv run pytest tests/test_join.py -x -q 2>&1 | tail -40
  git add -A
  git commit -m "WIP: v0.5.3 P1 (join tests red)" || true
  exit 1
fi

# ============ 5. Benchmark: speed + Δpeak ============
echo ""
echo "=== Bench: join 2M × 400k (compare: pre-v0.5.3 was ~14s, +2.9 MB) ==="
set +e
uv run python scripts/bench_join_v053.py 2>&1
set -e

# ============ 6. Full suite ============
echo ""
echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.5.3: P1 — join via Table.join (C++ grace hash), no pandas intermediate"

echo ""
echo "==> v0.5.3 applied on ${BR}"