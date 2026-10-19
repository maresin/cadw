#!/usr/bin/env bash
# fix_other_leak.sh — убрать .OTHER/ и main.py из индекса после cleanup_v010.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"

echo "=== 1. Убрать .OTHER/ из индекса (файлы на диске останутся) ==="
if git ls-files --error-unmatch .OTHER/ >/dev/null 2>&1; then
  git rm -r --cached .OTHER/
  echo "  OK: .OTHER/ убран из индекса"
else
  echo "  (.OTHER/ не в индексе)"
fi

echo "=== 2. Убрать main.py из индекса + с диска (это print из uv init) ==="
if git ls-files --error-unmatch main.py >/dev/null 2>&1; then
  git rm -f main.py
  echo "  OK: main.py удалён"
fi

echo "=== 3. Обновить .gitignore: .OTHER/ + main.py ==="
python3 - <<'PY'
from pathlib import Path
p = Path(".gitignore")
s = p.read_text()
additions = []
for line in [".OTHER/", "main.py", "*.lance/", "*.parquet"]:
    if line not in s.splitlines():
        additions.append(line)
if additions:
    if not s.endswith("\n"):
        s += "\n"
    s += "\n# Analysis artifacts & experimental data (fix_other_leak)\n"
    for a in additions:
        s += a + "\n"
    p.write_text(s)
    print("  .gitignore: добавлено", additions)
else:
    print("  .gitignore: уже содержит нужные строки")
PY

echo "=== 4. Проверить, что .OTHER/ больше не в индексе ==="
if git ls-files .OTHER/ | head -1; then
  echo "  ERROR: .OTHER/ всё ещё в индексе"
  exit 1
else
  echo "  OK: .OTHER/ не в индексе"
fi

echo "=== 5. Проверить, что .OTHER/ физически на диске (не удалён) ==="
if [[ -d .OTHER ]]; then
  echo "  .OTHER/ на диске: $(du -sh .OTHER/ | cut -f1)"
else
  echo "  WARN: .OTHER/ нет на диске (был удалён?)"
fi

echo "=== 6. pytest (быстрая проверка, что ничего не сломали) ==="
set +e
uv run pytest -q -m "not slow"
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "ERROR: tests failed"; exit 1
fi

echo "=== 7. Commit ==="
git add .gitignore
git commit -m "fix: .OTHER/ и main.py убраны из индекса (были захвачены cleanup_v010 через add -A)"

echo "=== Done ==="
git log --oneline -3
git ls-files | grep -c "^\.OTHER/" || echo "  .OTHER/ в индексе: 0 файлов"