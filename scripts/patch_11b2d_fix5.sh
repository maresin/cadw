#!/usr/bin/env bash
# patch_11b2d_fix5: полностью переписать блок skew check в
# _process_partition_pair — build_counts_df + _exact_actual_matches (12 отступ).
# Сохраняем первым, проверяем после (§13.4/47d).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/11b2d-e2e-join"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11b2d-fix5-${TS}"
git diff > ".backups/backup-pre-11b2d-fix5-${TS}/wip.diff" || true

# ---------- join.py: полностью переписать блок skew check ----------
python3 - <<'PY'
import re
from pathlib import Path

p = Path("cadw/ops/join.py")
s = p.read_text()

# Найти начало блока (Per-partition skew check) и конец (закрывающая ) raise
START_MARKER = "        # Per-partition skew check: unique_build_keys + avg_matches_per_probe."
END_MARKER = "            # how='left': probe preserved.\n"

start_idx = s.find(START_MARKER)
if start_idx < 0:
    # Может быть уже без комментария (если предыдущие версии удалили)
    START_MARKER = "        unique_build_keys = len(build_df)"
    start_idx = s.find(START_MARKER)
    if start_idx < 0:
        raise SystemExit("start marker 'unique_build_keys = len(build_df)' not found")

end_idx = s.find(END_MARKER, start_idx)
if end_idx < 0:
    raise SystemExit("end marker \"# how='left': probe preserved.\" not found")
# Включаем end_marker в заменяемую область: до конца "for probe_batch"
# (плюс начальный отступ с 12 пробелами перед actual_matches в новом коде)
end_idx += len(END_MARKER)

removed = s[start_idx:end_idx]

# Sanity: в заменяемом блоке должны быть и unique_build_keys, и
# _exact_actual_matches (смешанное состояние)
if "unique_build_keys" not in removed:
    raise SystemExit(
        f"expected 'unique_build_keys' in removed block; got:\n{removed[:500]}"
    )
if "_exact_actual_matches" not in removed:
    raise SystemExit(
        f"expected '_exact_actual_matches' in removed block; got:\n{removed[:500]}"
    )
if "for probe_batch in" not in removed:
    raise SystemExit(
        f"expected 'for probe_batch in' in removed block; got:\n{removed[:500]}"
    )

# Полный новый блок. Включает:
#   - build_counts_df (один раз)
#   - _exact_actual_matches как nested function (замыкает build_df, build_counts_df, on)
#   - out_batches / empty_schema
#   - for probe_batch / check_batch / _drop_null_keys / "# how='left': probe preserved."
new_block = '''        # Точная оценка совпадений: build_counts (один раз) × probe_counts
        # (per batch). Это то же, что материализует pd.merge (совпадения, не
        # cartesian). O(build + probe) через groupby на уникальных ключах —
        # дёшево (build ≈ partition size, probe ≈ batch size, тысячи строк).
        #
        # Почему не cartesian: `len(build) × len(probe)` — верхняя граница на
        # совпадения, но для PK-FK (build уникален) даёт 5000 × 2500 = 12.5M
        # при реальном output 2500 — ложные reject'ы. §13.4 эпизод 47.
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

        def _exact_actual_matches(probe_df: "pd.DataFrame") -> int:
            """Точное число совпадений между build_df и этим probe batch.

            Fallback — cartesian, если groupby упал (не должно случаться).
            """
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

        out_batches: list[pa.RecordBatch] = []
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
'''

s = s[:start_idx] + new_block + s[end_idx:]

# После блока "for probe_batch ... # how='left': probe preserved.\n"
# должен идти блок с actual_matches. В новом коде мы закончили на
# "# how='left': probe preserved.\n", поэтому дальше в исходном файле идёт
# старый (неправильный) блок с 16-отступами. Заменим его на корректный 12-отступ.
OLD_CHECK = '''            # Точное число строк, которое pd.merge материализует:
                # sum over common keys(build_count × probe_count).
                actual_matches = _exact_actual_matches(probe_df)
                if actual_matches > max_output_rows:
                    raise MemoryLimitExceeded(
                        f"hash_join(how={how!r}): exact matches between this "
                        f"probe batch ({len(probe_df)} rows) and build "
                        f"({len(build_df)} rows) = {actual_matches} > "
                        f"max_output_rows={max_output_rows}. Likely skew. "
                        f"Увеличьте budget или уменьшите num_partitions."
                    )
'''
NEW_CHECK = '''            actual_matches = _exact_actual_matches(probe_df)
            if actual_matches > max_output_rows:
                raise MemoryLimitExceeded(
                    f"hash_join(how={how!r}): exact matches between this "
                    f"probe batch ({len(probe_df)} rows) and build "
                    f"({len(build_df)} rows) = {actual_matches} > "
                    f"max_output_rows={max_output_rows}. Likely skew. "
                    f"Увеличьте budget или уменьшите num_partitions."
                )
'''

if OLD_CHECK in s:
    s = s.replace(OLD_CHECK, NEW_CHECK, 1)
    print("join.py: старый (16-отступ) actual_matches заменён корректным (12-отступ)")
else:
    # Может быть другой вариант с отступом — попробуем generic
    old_pattern = re.compile(
        r"\n            # Точное число строк[^\n]*\n"
        r"(?:                #[^\n]*\n)*?"
        r"                actual_matches = _exact_actual_matches\(probe_df\)\n"
        r"                if actual_matches > max_output_rows:\n"
        r"                    raise MemoryLimitExceeded\(\n"
        r"(?:                        [^\n]*\n)*?"
        r"                    \)\n"
    )
    if old_pattern.search(s):
        s = old_pattern.sub("\n" + NEW_CHECK, s, count=1)
        print("join.py: actual_matches заменён (regex fallback)")
    else:
        print("WARNING: actual_matches check — не найден в известных формах, "
              "оставляем как есть (возможен дубликат или пропуск)")

# Сохраняем — прогресс не теряется
p.write_text(s)
print("join.py: сохранён")
PY

# ---------- Проверка ПОСЛЕ сохранения ----------
python3 - <<'PY'
import ast
from pathlib import Path

p = Path("cadw/ops/join.py")
s = p.read_text()

# 1. Нет "unique_build_keys" / "avg_matches_per_probe"
assert "unique_build_keys" not in s, "unique_build_keys всё ещё в join.py"
assert "avg_matches_per_probe" not in s, "avg_matches_per_probe всё ещё в join.py"
assert "max_cartesian_rows" not in s, "max_cartesian_rows всё ещё в join.py"
print("join.py: grep checks passed")

# 2. _exact_actual_matches определён внутри _process_partition_pair
tree = ast.parse(s)
found_def = False
found_call_12 = False  # вызов внутри for probe_batch

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_process_partition_pair":
        for inner in ast.walk(node):
            if isinstance(inner, ast.FunctionDef) and inner.name == "_exact_actual_matches":
                found_def = True
assert found_def, "_exact_actual_matches not defined inside _process_partition_pair"
print("join.py: _exact_actual_matches определён внутри _process_partition_pair")

# 3. Компиляция (синтаксис)
try:
    compile(s, "join.py", "exec")
    print("join.py: syntax OK")
except SyntaxError as e:
    raise SystemExit(f"join.py: syntax error — {e}")
PY

# ---------- run ----------
echo "=== pytest tests/test_e2e_join.py -v (slow) ==="
set +e
uv run pytest tests/test_e2e_join.py -v
RC_E2E=$?
set -e
if [[ $RC_E2E -ne 0 ]]; then
  echo "e2e: rc=$RC_E2E"
  exit 1
fi

echo "=== pytest tests/test_join.py tests/test_join_oom_guard.py tests/test_merge_planner.py -v ==="
set +e
uv run pytest tests/test_join.py tests/test_join_oom_guard.py tests/test_merge_planner.py -v
RC_TARGET=$?
set -e
if [[ $RC_TARGET -ne 0 ]]; then
  echo "target: rc=$RC_TARGET"
  exit 1
fi

echo "=== pytest fast ==="
set +e
uv run pytest -q -m "not slow"
RC_FAST=$?
set -e
if [[ $RC_FAST -ne 0 ]]; then
  echo "fast: rc=$RC_FAST"
  exit 1
fi

echo "=== pytest slow (all) ==="
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2d-fix5: _process_partition_pair rewritten (build_counts_df + _exact_actual_matches, 12-отступ); §13.4/47"

echo "==> 11-B-2d-fix5 applied"