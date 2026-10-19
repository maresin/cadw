#!/usr/bin/env bash
# patch_11b2d_fix7: точечный фикс по текущему состоянию join.py.
#   - STEP 1: unique_build_keys/avg_matches_per_probe -> build_counts_df + _exact_actual_matches
#   - STEP 2: actual_matches блок -> 8-space indent (для inner и left)
#   - save first, verify after (§13.4/47d)
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/11b2d-e2e-join"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11b2d-fix7-${TS}"
git diff > ".backups/backup-pre-11b2d-fix7-${TS}/wip.diff" || true

# === DIAGNOSTIC ===
echo "=== DIAGNOSTIC cadw/ops/join.py ==="
python3 << 'PY'
from pathlib import Path
lines = Path("cadw/ops/join.py").read_text().split("\n")
for i, line in enumerate(lines):
    if any(m in line for m in (
        "unique_build_keys", "avg_matches_per_probe",
        "actual_matches", "_exact_actual_matches",
        "out_batches =", "Per-partition skew",
    )):
        ind = len(line) - len(line.lstrip())
        print(f"L{i+1:4d} indent={ind:2d}: {line[:100]}")
PY
echo ""

# === FIX ===
python3 << 'PY'
import ast
from pathlib import Path

p = Path("cadw/ops/join.py")
src = p.read_text()
lines = src.split("\n")

# Sanity: not already fixed
if "def _exact_actual_matches" in src and "unique_build_keys" not in src:
    print("Already fixed — no-op")
    raise SystemExit(0)

# ---- STEP 1: replace unique_build_keys block with build_counts_df + nested def ----
uk_idx = next(
    (i for i, l in enumerate(lines) if "unique_build_keys = len(build_df)" in l),
    None,
)
out_idx = next(
    (i for i, l in enumerate(lines) if "out_batches: list[pa.RecordBatch] = []" in l),
    None,
)
if uk_idx is None or out_idx is None or uk_idx >= out_idx:
    raise SystemExit(f"Anchors not found: uk_idx={uk_idx}, out_idx={out_idx}")

# Start = first "# Per-partition skew" comment (or uk_idx if no comment)
block_start = uk_idx
j = uk_idx - 1
while j >= 0 and lines[j].strip().startswith("#"):
    block_start = j
    j -= 1

# Verify the block contains both legacy names
block_text = "\n".join(lines[block_start:out_idx])
assert "unique_build_keys" in block_text
assert "avg_matches_per_probe" in block_text

new_step1 = [
    "    # Точная оценка совпадений: build_counts × probe_counts (не cartesian).",
    "    build_counts_df = None",
    "    if len(build_df) > 0 and len(on) > 0:",
    "        try:",
    "            build_counts_df = (",
    "                build_df.groupby(list(on), dropna=False)",
    "                .size()",
    '                .rename("__bc")',
    "                .reset_index()",
    "            )",
    "        except Exception:",
    "            build_counts_df = None",
    "",
    "    def _exact_actual_matches(probe_df: pd.DataFrame) -> int:",
    '        """Точное число совпадений build × probe. Fallback — cartesian."""',
    "        if len(probe_df) == 0 or len(build_df) == 0:",
    "            return 0",
    "        if build_counts_df is None:",
    "            return len(build_df) * len(probe_df)",
    "        try:",
    "            probe_counts = (",
    "                probe_df.groupby(list(on), dropna=False)",
    "                .size()",
    '                .rename("__pc")',
    "                .reset_index()",
    "            )",
    "            joined = build_counts_df.merge(",
    '                probe_counts, on=list(on), how="inner"',
    "            )",
    "            if len(joined) == 0:",
    "                return 0",
    '            return int((joined["__bc"] * joined["__pc"]).sum())',
    "        except Exception:",
    "            return len(build_df) * len(probe_df)",
    "",
]
lines = lines[:block_start] + new_step1 + lines[out_idx:]
print(f"STEP 1: removed L{block_start+1}-L{out_idx}, inserted build_counts_df + _exact_actual_matches")

# ---- STEP 2: normalize actual_matches block to 8-space indent ----
am_idx = next(
    (i for i, l in enumerate(lines) if "actual_matches = _exact_actual_matches(probe_df)" in l),
    None,
)
if am_idx is None:
    raise SystemExit("actual_matches call not found (unexpected after step1)")

base_indent = len(lines[am_idx]) - len(lines[am_idx].lstrip())
print(f"STEP 2: actual_matches at L{am_idx+1}, base_indent={base_indent}")

# Block start: walk back over contiguous comment lines
step2_start = am_idx
j = am_idx - 1
while j >= 0 and lines[j].strip().startswith("#"):
    step2_start = j
    j -= 1

# Block end: closing paren of raise MemoryLimitExceeded(
raise_idx = next(
    (i for i in range(am_idx, len(lines)) if "raise MemoryLimitExceeded(" in lines[i]),
    None,
)
if raise_idx is None:
    raise SystemExit("raise not found after actual_matches")
depth = 0
close_idx = None
for i in range(raise_idx, len(lines)):
    for ch in lines[i]:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx is not None:
        break
if close_idx is None:
    raise SystemExit("close paren of raise not found")

new_step2 = [
    "        # Точное число строк, которое pd.merge материализует (не cartesian).",
    "        actual_matches = _exact_actual_matches(probe_df)",
    "        if actual_matches > max_output_rows:",
    "            raise MemoryLimitExceeded(",
    '                f"hash_join(how={how!r}): exact matches between this "',
    '                f"probe batch ({len(probe_df)} rows) and build "',
    '                f"({len(build_df)} rows) = {actual_matches} > "',
    '                f"max_output_rows={max_output_rows}. Likely skew. "',
    '                f"Увеличьте budget или уменьшите num_partitions."',
    "            )",
]
lines = lines[:step2_start] + new_step2 + lines[close_idx + 1:]
print(f"STEP 2: normalized L{step2_start+1}-L{close_idx+1} to 8-space")

# ---- SAVE (progress guaranteed) ----
s = "\n".join(lines)
p.write_text(s)
print("Saved")
PY

# === VERIFY AFTER SAVE ===
echo "=== verify join.py ==="
python3 << 'PY'
import ast
from pathlib import Path

p = Path("cadw/ops/join.py")
s = p.read_text()

errors = []
for bad in ("unique_build_keys", "avg_matches_per_probe", "max_cartesian_rows"):
    if bad in s:
        errors.append(f"{bad} still present")
if "def _exact_actual_matches(" not in s:
    errors.append("def _exact_actual_matches missing")
if s.count("_exact_actual_matches") < 2:
    errors.append("_exact_actual_matches occurrences < 2")

if errors:
    print("VERIFY FAILED:")
    for e in errors:
        print(f"  - {e}")
    raise SystemExit(1)

# AST: _exact_actual_matches defined inside _process_partition_pair
tree = ast.parse(s)
found_def = False
found_call_in_for = False
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_process_partition_pair":
        for inner in ast.walk(node):
            if isinstance(inner, ast.FunctionDef) and inner.name == "_exact_actual_matches":
                found_def = True
            if isinstance(inner, ast.Call):
                f = inner.func
                if isinstance(f, ast.Name) and f.id == "_exact_actual_matches":
                    found_call_in_for = True

if not found_def:
    raise SystemExit("_exact_actual_matches not defined inside _process_partition_pair")
if not found_call_in_for:
    raise SystemExit("_exact_actual_matches never called")

try:
    compile(s, "join.py", "exec")
except SyntaxError as e:
    raise SystemExit(f"SyntaxError: {e}")

print("OK: no forbidden names; nested def + call present; syntax OK")
PY

# === TESTS ===
echo ""
echo "=== pytest tests/test_e2e_join.py -v (slow) ==="
set +e
uv run pytest tests/test_e2e_join.py -v
RC_E2E=$?
set -e
if [[ $RC_E2E -ne 0 ]]; then
  echo "e2e: rc=$RC_E2E"
  exit 1
fi

echo "=== pytest tests/test_join_oom_guard.py -v ==="
set +e
uv run pytest tests/test_join_oom_guard.py -v
RC_GUARD=$?
set -e
if [[ $RC_GUARD -ne 0 ]]; then
  echo "guards: rc=$RC_GUARD"
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
git commit -m "11-B-2d-fix7: join.py — build_counts_df + _exact_actual_matches (8-space); §13.4/47"

echo "==> 11-B-2d-fix7 applied"
echo "    Next: squash → tag v0.2.5"