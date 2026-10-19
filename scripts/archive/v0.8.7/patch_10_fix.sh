#!/usr/bin/env bash
# patch_10_fix: NameError total_bytes + AST guard ищет вызов, не комментарий.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

BR="patch/10-materialize-audit"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$BR" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-10-fix-${TS}"
git diff > ".backups/backup-pre-10-fix-${TS}/wip.diff"

# ---- Fix 1: total_bytes -> partition_bytes в error-сообщениях ----
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/groupby.py")
s = p.read_text()

old1 = '''            raise MemoryLimitExceeded(
                f"_local_groupby: partition too large ({total_bytes} B > {safety_limit} B), "
                f"but memory_guard/spill_dir not provided."
            )'''
new1 = '''            raise MemoryLimitExceeded(
                f"_local_groupby: partition too large "
                f"({partition_bytes} B > {safety_limit} B), "
                f"but memory_guard/spill_dir not provided."
            )'''
if old1 in s:
    s = s.replace(old1, new1, 1)
    print("fixed: 'partition too large' message")

old2 = '''            raise MemoryLimitExceeded(
                f"_local_groupby: depth={depth} >= max_depth={max_depth}; "
                f"partition={partition_path.name} size={total_bytes} B > safety={safety_limit} B. "
                f"Вероятен экстремальный skew (см. DECISIONS §11.4 п.3)."
            )'''
new2 = '''            raise MemoryLimitExceeded(
                f"_local_groupby: depth={depth} >= max_depth={max_depth}; "
                f"partition={partition_path.name} size={partition_bytes} B "
                f"> safety={safety_limit} B. "
                f"Вероятен экстремальный skew (см. DECISIONS §11.4 п.3)."
            )'''
if old2 in s:
    s = s.replace(old2, new2, 1)
    print("fixed: 'max_depth' message")

# Проверка, что total_bytes больше нет в _local_groupby
import re
m = re.search(r"def _local_groupby\(.*?(?=\ndef )", s, re.DOTALL)
if m and "total_bytes" in m.group(0):
    raise SystemExit("total_bytes still in _local_groupby")
p.write_text(s)
print("groupby.py: total_bytes не осталось")
PY

# ---- Fix 2: AST guard ищет полные вызовы, не подстроки ----
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_materialize_guard.py")
s = p.read_text()

old = '''    src = inspect.getsource(reduce)
    est_pos = src.find("estimate_state_size")
    init_pos = src.find("op.init()")
    check_pos = src.find("check_state")

    assert est_pos >= 0, "estimate_state_size not called in reduce()"
    assert init_pos >= 0, "op.init() not called in reduce()"
    assert check_pos >= 0, "check_state not called in reduce()"'''

new = '''    src = inspect.getsource(reduce)
    # Ищем ПОЛНЫЕ вызовы, а не подстроки: комментарии в reduce() могут
    # упоминать "op.init()" или "check_state" без вызова — find() на
    # подстроке поймает комментарий и даст ложную позицию.
    est_pos = src.find("memory_guard.estimate_state_size(")
    init_pos = src.find("state = op.init()")
    check_pos = src.find("memory_guard.check_state(")

    assert est_pos >= 0, "memory_guard.estimate_state_size(...) not called in reduce()"
    assert init_pos >= 0, "state = op.init() not called in reduce()"
    assert check_pos >= 0, "memory_guard.check_state(...) not called in reduce()"'''

if old not in s:
    raise SystemExit("test_materialize_guard.py: pattern not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("test_materialize_guard.py: find на полные вызовы")
PY

set +e
uv run pytest -q -m "not slow"
RC_FAST=$?
uv run pytest -q -m slow
RC_SLOW=$?
set -e
if [[ $RC_FAST -ne 0 ]]; then echo "fast: rc=$RC_FAST"; exit 1; fi
if [[ $RC_SLOW -ne 0 ]]; then echo "slow: rc=$RC_SLOW"; exit 1; fi

git add -A
git commit -m "10-fix: partition_bytes в error-сообщениях; AST guard на полные вызовы (не комментарии)"

echo "==> sha256"
sha256sum cadw/ops/groupby.py tests/test_materialize_guard.py
echo "==> 10-fix applied"