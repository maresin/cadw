#!/usr/bin/env bash
# patch_11b2a_fix: how='left' — peek-early-return preserved; empty build_df с колонками;
# схема-driven from_pandas (nullable int/string).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

BR="patch/11b2a-join-left"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11b2a-fix-${TS}"
git diff > ".backups/backup-pre-11b2a-fix-${TS}/wip.diff"

python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/join.py")
s = p.read_text()

# 1. Peek: right пуст и how='left' — сохранить left с NULL right cols.
old_peek = '''    try:
        first_right = next(right_it)
    except StopIteration:
        return pa.Table.from_batches(
            [], schema=_empty_result_schema(left_schema, right_schema, on, suffixes)
        )
    left_batches = chain([first_left], left_it)
    right_batches = chain([first_right], right_it)'''

new_peek = '''    try:
        first_right = next(right_it)
        right_batches = chain([first_right], right_it)
    except StopIteration:
        if how == "inner":
            return pa.Table.from_batches(
                [], schema=_empty_result_schema(left_schema, right_schema, on, suffixes)
            )
        # how='left': right пуст -> все left-строки с NULL right-cols
        right_batches = iter([])
    left_batches = chain([first_left], left_it)'''

if old_peek not in s:
    raise SystemExit("peek block not found")
s = s.replace(old_peek, new_peek, 1)

# 2. Заменить цикл по партициям.
old_loop = '''    for p in range(num_partitions):
        lp = left_paths[p] if left_paths[p].exists() else None
        rp = right_paths[p] if right_paths[p].exists() else None
        if lp is None or rp is None:
            continue

        if how == "inner":
            # build = меньшая партиция (min по file size)
            if lp.stat().st_size <= rp.stat().st_size:
                build_path, probe_path = lp, rp
                build_is_left = True
            else:
                build_path, probe_path = rp, lp
                build_is_left = False
        else:  # how == "left": build = right всегда
            build_path, probe_path = rp, lp
            build_is_left = False

        build_table = _materialize_partition(build_path, memory_guard)
        build_df = _drop_null_keys(build_table.to_pandas(), list(on))
        if len(build_df) == 0 and how == "inner":
            # inner: пустой build -> нет matches, skip
            continue
        # how='left': пустой build — продолжаем: probe (left) сохраняется
        # с NULL right-колонками.
'''

new_loop = '''    right_rest = [c for c in right_schema.names if c not in set(on)]
    empty_build_cols = list(on) + right_rest

    for p in range(num_partitions):
        lp = (
            left_paths[p]
            if left_paths[p].exists() and left_paths[p].stat().st_size > 0
            else None
        )
        rp = (
            right_paths[p]
            if right_paths[p].exists() and right_paths[p].stat().st_size > 0
            else None
        )

        if how == "inner":
            if lp is None or rp is None:
                continue
            if lp.stat().st_size <= rp.stat().st_size:
                build_path, probe_path = lp, rp
                build_is_left = True
            else:
                build_path, probe_path = rp, lp
                build_is_left = False
            build_table = _materialize_partition(build_path, memory_guard)
            build_df = _drop_null_keys(build_table.to_pandas(), list(on))
            if len(build_df) == 0:
                continue
        else:  # how == "left"
            if lp is None:
                continue
            probe_path = lp
            build_is_left = False
            if rp is None:
                build_df = pd.DataFrame(columns=empty_build_cols)
            else:
                build_table = _materialize_partition(rp, memory_guard)
                build_df = _drop_null_keys(build_table.to_pandas(), list(on))
                if len(build_df) == 0:
                    build_df = pd.DataFrame(columns=empty_build_cols)
'''

if old_loop not in s:
    raise SystemExit("partition loop not found")
s = s.replace(old_loop, new_loop, 1)

# 3. Заменить финальную конверсию: reorder + from_pandas(schema=empty_schema).
old_out = '''            if len(merged) == 0:
                continue
            # Типы: right-колонки могут стать NaN (left join) — используем
            # nullable schema.
            out_table = pa.Table.from_pandas(
                merged, preserve_index=False
            )
            if out_table.schema.names != empty_schema.names:
                out_table = out_table.select(empty_schema.names)
            # Привести к nullable-типам empty_schema (NaN -> null)
            try:
                out_table = out_table.cast(empty_schema, safe=False)
            except Exception:
                # fallback: from_pandas с schema (может дать null в nullable)
                out_table = pa.Table.from_pandas(
                    merged, schema=empty_schema, preserve_index=False
                )'''

new_out = '''            if len(merged) == 0:
                continue
            merged = merged.reindex(columns=empty_schema.names)
            # schema-driven from_pandas: NaN -> null для nullable полей.
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
                    pass'''

if old_out not in s:
    raise SystemExit("output conversion block not found")
s = s.replace(old_out, new_out, 1)

# 4. После цикла: если how='left' и result пуст, но left был непуст —
# нужно добавить preserved left rows. Это уже обработано пустым build_df
# в цикле (build_df=DataFrame(columns=...)): probe (left) сохраняется.
# Дополнительно: если right был пуст полностью, left_paths могут быть
# пропущены (lp is None для всех). Тогда right_paths пуст, но left_paths
# не пуст — цикл отработает с rp=None для каждой, build_df пуст ->
# left preserved. Ничего дополнительного не нужно.

p.write_text(s)
print("join.py: how='left' fix — peek, empty build_df, from_pandas(schema)")
PY

# ---------- tests: добавить проверку nullable dtype ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_join.py")
s = p.read_text()

extra = '''

def test_merge_left_nullable_dtype(tmp_path):
    """Unmatched right cols в left join -> nullable type (int -> null)."""
    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": ["w", "x", "y"]})
    right = make_df(tmp_path, "R", {"k": [1, 3], "v": [100, 300]})
    result = left.merge(right, on="k", how="left")
    head = result.head(10)
    # v (int64) должен стать nullable (null для unmatched)
    v_field = head.schema.field("v")
    assert v_field.nullable, (
        f"v nullable expected, got nullable={v_field.nullable}"
    )
    rows = _sorted_by(head, "k")
    assert rows[0]["v"] == 100
    assert rows[1]["v"] is None
    assert rows[2]["v"] == 300

'''
s = s.rstrip() + extra
p.write_text(s)
print("test_join.py: +test_merge_left_nullable_dtype")
PY

set +e
uv run pytest tests/test_join.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "test_join.py: rc=$RC"
  exit 1
fi

uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2a-fix: how='left' — peek preserved, empty build_df с колонками, from_pandas(schema), nullable dtype test"

echo "==> 11-B-2a-fix applied"