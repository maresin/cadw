#!/usr/bin/env bash
# patch_11b2a_fix2: how='left' + пустой right (нет батчей) —
# short-circuit в _left_preserve_with_nulls, без hash_shuffle.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

BR="patch/11b2a-join-left"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11b2a-fix2-${TS}"
git diff > ".backups/backup-pre-11b2a-fix2-${TS}/wip.diff"

python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/join.py")
s = p.read_text()

# 1. Helper _left_preserve_with_nulls — после _iter_probe_batches
helper_anchor = "def hash_join("
if "_left_preserve_with_nulls" not in s:
    helper = '''def _left_preserve_with_nulls(
    left_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    suffixes: tuple[str, str],
    memory_guard: MemoryGuard,
) -> pa.Table:
    """how='left' при пустом right: left preserved, right-колонки -> null.

    Используется, когда right содержит 0 батчей: hash_shuffle не может
    принять пустой вход (см. §6.7), а left join не требует shuffle, потому
    что совпадений всё равно нет.
    """
    on_set = set(on)
    right_rest = [c for c in right_schema.names if c not in on_set]
    overlap = {c for c in left_schema.names if c not in on_set} & set(right_rest)

    out_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)

    all_batches: list[pa.RecordBatch] = []
    for lb in left_batches:
        memory_guard.check_batch(lb, name="left_preserve.input")
        n = lb.num_rows
        arrays: dict[str, pa.Array] = {}
        for c in on:
            arrays[c] = lb.column(c)
        for f in left_schema:
            if f.name in on_set:
                continue
            name = f.name + suffixes[0] if f.name in overlap else f.name
            arrays[name] = lb.column(f.name)
        for f in right_schema:
            if f.name in on_set:
                continue
            name = f.name + suffixes[1] if f.name in overlap else f.name
            arrays[name] = pa.nulls(n, type=f.type)

        rb = pa.RecordBatch.from_arrays(
            [arrays[c] for c in out_schema.names],
            schema=out_schema,
        )
        if rb.num_rows > 0:
            all_batches.append(rb)

    if not all_batches:
        return pa.Table.from_batches([], schema=out_schema)
    return pa.Table.from_batches(all_batches)


'''
    if helper_anchor not in s:
        raise SystemExit("hash_join def not found")
    s = s.replace(helper_anchor, helper + helper_anchor, 1)
    print("join.py: helper _left_preserve_with_nulls added")

# 2. Peek: right пуст + how='left' -> short-circuit
old_peek = '''    try:
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

new_peek = '''    left_batches = chain([first_left], left_it)
    try:
        first_right = next(right_it)
        right_batches = chain([first_right], right_it)
    except StopIteration:
        if how == "inner":
            return pa.Table.from_batches(
                [], schema=_empty_result_schema(left_schema, right_schema, on, suffixes)
            )
        # how='left': right пуст -> left preserved с null right-cols
        # Без hash_shuffle (пустой вход не принимается, §6.7).
        return _left_preserve_with_nulls(
            left_batches, left_schema, right_schema, list(on), suffixes,
            memory_guard,
        )'''

if old_peek not in s:
    raise SystemExit("peek block not found")
s = s.replace(old_peek, new_peek, 1)
print("join.py: peek — how='left' + пустой right -> short-circuit")

p.write_text(s)
PY

# ---------- run ----------
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
git commit -m "11-B-2a-fix2: how='left' + пустой right -> short-circuit (_left_preserve_with_nulls), без hash_shuffle"

echo "==> 11-B-2a-fix2 applied"