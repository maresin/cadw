#!/usr/bin/env bash
# patch_05-fix: физически удалить _legacy/ (включая __pycache__).
set -euo pipefail

BR="patch/05-remove-legacy"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$BR" ]] || { echo "not on ${BR}"; exit 1; }

# 1. Удалить физически (включая untracked __pycache__)
rm -rf cadw/_legacy tests/_legacy

# 2. Проверить, что директорий нет
[[ ! -d cadw/_legacy ]] || { echo "ERROR: cadw/_legacy still exists"; exit 1; }
[[ ! -d tests/_legacy ]] || { echo "ERROR: tests/_legacy still exists"; exit 1; }
echo "physically removed: cadw/_legacy, tests/_legacy"

# 3. Тесты
set +e
uv run pytest -q -m "not slow"
RC_FAST=$?
uv run pytest -q -m slow
RC_SLOW=$?
set -e
if [[ $RC_FAST -ne 0 ]]; then echo "fast: rc=$RC_FAST"; exit 1; fi
if [[ $RC_SLOW -ne 0 ]]; then echo "slow: rc=$RC_SLOW"; exit 1; fi

# 4. Commit (git rm уже в индексе от patch_05, здесь — только зачистка untracked)
git add -A
git status --short
if git diff --cached --quiet && git diff --quiet; then
  echo "nothing to commit (fix was filesystem-only)"
else
  git commit -m "05-fix: physically remove _legacy/ (untracked __pycache__)" || true
fi

echo "==> 05-fix applied"