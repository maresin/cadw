#!/usr/bin/env bash
# patch_11b2d_fix6: заменить unique_build_keys/avg_matches_per_probe на
# build_counts_df + _exact_actual_matches; исправить отступ actual_matches.
# Save first, verify strict after (§13.4/47d).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/11b2d-e2e-join"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11b2d-fix6-${TS}"
git diff > ".backups/backup-pre-11b2d-fix6-${TS}/wip.diff" || true

# ---------- join.py: STEP 1 + STEP 2 ----------
python3 - <<'PY'
import re
from pathlib import Path

p = Path("cadw/ops/join.py")
s = p.read_text()

# ========== STEP 1: unique_build_keys + avg_matches_per_probe -> build_counts_df + _exact_actual_matches ==========
if "unique_build_keys" not in s and "avg_matches_per_probe" not in s:
    print("[step1] unique_build_keys/avg_matches_per_probe отсутствуют — no-op")
else:
    # Найти начало
    start = -1
    for m in (
        "        # Per-partition skew check: unique_build_keys",
        "        unique_build_keys = len(build_df)",
    ):
        idx = s.find(m)
        if idx >= 0:
            start = idx
            break
    if start < 0:
        raise SystemExit(
            "[step1] start marker not found (unique_build_keys/avg_matches_per_probe "
            "present, but no recognizable block)"
        )

    # Найти конец avg_matches_per_probe = ( ... \n        )
    avg_marker = "        avg_matches_per_probe = ("
    avg_start = s.find(avg_marker, start)
    if avg_start < 0:
        raise SystemExit("[step1] avg_matches_per_probe = ( not found")
    close_m = re.compile(r'\n        \)').search(s, avg_start)
    if not close_m:
        raise SystemExit("[step1] close ) of avg_matches_per_probe not found")
    end1 = close_m.end()
    if s[end1:end1+1] == "\n":
        end1 += 1

    removed = s[start:end1]
    assert "unique_build_keys" in removed
    assert "avg_matches_per_probe" in removed

    new_block1 = '''        # Точная оценка совпадений: build_counts (один раз) × probe_counts
        # (per batch). Это то, что материализует pd.merge (совпадения, не
        # cartesian). §13.4 эпизод 47.
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

        def _exact_actual_matches(probe_df) -> int:
            """Точное число совпадений build_df × probe_df (не cartesian)."""
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

'''
    s = s[:start] + new_block1 + s[end1:]
    print("[step1] unique_build_keys/avg_matches_per_probe заменены build_counts_df + _exact_actual_matches")

# ========== STEP 2: исправить отступ actual_matches (12 или 16) ==========
am_marker = "actual_matches = _exact_actual_matches(probe_df)"
am_idx = s.find(am_marker)
if am_idx < 0:
    raise SystemExit(
        "[step2] actual_matches = _exact_actual_matches(probe_df) not found "
        "(не должно случаться после step1)"
    )

# Начало строки с actual_matches
line_start = s.rfind("\n", 0, am_idx) + 1

# Включить предшествующие комментарии "# Точное число строк" / "# sum over common keys"
prev_line_start = s.rfind("\n", 0, line_start - 1) + 1 if line_start > 0 else 0
prev_line = s[prev_line_start:line_start]
if "# Точное число строк" in prev_line or "# sum over common keys" in prev_line:
    line_start = prev_line_start
    prev_line_start2 = s.rfind("\n", 0, line_start - 1) + 1 if line_start > 0 else 0
    prev_line2 = s[prev_line_start2:line_start]
    if "# Точное число строк" in prev_line2 or "# sum over common keys" in prev_line2:
        line_start = prev_line_start2

# Конец raise MemoryLimitExceeded
raise_idx = s.find("raise MemoryLimitExceeded(", am_idx)
if raise_idx < 0:
    raise SystemExit("[step2] raise MemoryLimitExceeded not found")

# Сбалансировать скобки
depth = 0
close_idx = -1
for i in range(raise_idx + len("raise MemoryLimitExceeded"), len(s)):
    if s[i] == "(":
        depth += 1
    elif s[i] == ")":
        depth -= 1
        if depth == 0:
            close_idx = i
            break
if close_idx < 0:
    raise SystemExit("[step2] close ) of raise not found")

# Включая "\n" после ")"
end2 = close_idx + 1
if end2 < len(s) and s[end2] == "\n":
    end2 += 1

new_block2 = '''            # Точное число строк, которое pd.merge материализует (не cartesian).
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
s = s[:line_start] + new_block2 + s[end2:]
print("[step2] actual_matches блок нормализован к отступу 12")

# СОХРАНЯЕМ (fix §13.4/47d — save before strict check)
p.write_text(s)
print("join.py: сохранён")
PY

# ---------- Verify AFTER save (strict) ----------
echo "=== verify join.py ==="
python3 - <<'PY'
import ast
from pathlib import Path

p = Path("cadw/ops/join.py")
s = p.read_text()

errors = []
if "unique_build_keys" in s:
    errors.append("unique_build_keys")
if "avg_matches_per_probe" in s:
    errors.append("avg_matches_per_probe")
if "max_cartesian_rows" in s:
    errors.append("max_cartesian_rows")

if errors:
    raise SystemExit(f"join.py содержит forbidden names: {errors}")

if "def _exact_actual_matches(" not in s:
    raise SystemExit("join.py: def _exact_actual_matches отсутствует")
if s.count("_exact_actual_matches") < 2:
    raise SystemExit(
        "join.py: _exact_actual_matches встречается < 2 раз "
        "(должно быть определение + вызов)"
    )

# _exact_actual_matches определён ВНУТРИ _process_partition_pair
tree = ast.parse(s)
found = False
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_process_partition_pair":
        for inner in ast.walk(node):
            if isinstance(inner, ast.FunctionDef) and inner.name == "_exact_actual_matches":
                found = True
if not found:
    raise SystemExit(
        "_exact_actual_matches должен быть определён ВНУТРИ _process_partition_pair"
    )

# Синтаксис
try:
    compile(s, "join.py", "exec")
except SyntaxError as e:
    raise SystemExit(f"join.py: SyntaxError — {e}")

print("OK: нет forbidden names; _exact_actual_matches определён и вызывается; syntax OK")
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
git commit -m "11-B-2d-fix6: join.py — build_counts_df + _exact_actual_matches (save-then-strict-check); §13.4/47d"

echo "==> 11-B-2d-fix6 applied"