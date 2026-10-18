#!/usr/bin/env bash
# patch_05: удалить _legacy/ (заменено примитивами в 02b/03b/03c) + guard.
set -euo pipefail

require_clean_tree() {
  local bt bu
  bt="$(git status --porcelain --untracked-files=no)"
  [[ -z "$bt" ]] || { echo "dirty:"; echo "$bt"; return 1; }
  bu=""
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    case "$f" in
      scripts/*|.backups/*|__pycache__/*|*/__pycache__/*|*.pyc|.pytest_cache/*) : ;;
      *) bu+="  $f"$'\n' ;;
    esac
  done < <(git ls-files --others --exclude-standard)
  [[ -z "$bu" ]] || { echo "untracked:"; echo "$bu"; return 1; }
  return 0
}
require_clean_tree || exit 1

# ---------- 0. Pre-flight: main-код не должен ссылаться на _legacy ----------
echo "==> Pre-flight: grep main code for _legacy references"
if grep -rn "cadw\._legacy\|from \.\._legacy\|_legacy\." \
    --include="*.py" cadw/ 2>/dev/null | grep -v "^cadw/_legacy/"; then
  echo "ERROR: main code references _legacy; fix before removal"
  exit 1
fi
echo "    OK: no references outside cadw/_legacy/"

CUR="$(git rev-parse --abbrev-ref HEAD)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-05-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_05"

BR="patch/05-remove-legacy"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- 1. Remove legacy directories ----------
echo "==> git rm -r cadw/_legacy tests/_legacy"
git rm -r cadw/_legacy 2>/dev/null || true
git rm -r tests/_legacy 2>/dev/null || true
rmdir cadw/_legacy tests/_legacy 2>/dev/null || true

# ---------- 2. Architectural guard ----------
cat > tests/test_architecture.py <<'EOF'
"""Архитектурные guard-тесты (патч 05: legacy удалён)."""
import pathlib

import cadw


def test_no_legacy_package():
    """cadw._legacy удалён; не должно быть попыток его импортировать."""
    root = pathlib.Path(cadw.__file__).parent
    assert not (root / "_legacy").exists(), "cadw/_legacy/ must not exist"


def test_main_code_does_not_reference_legacy():
    """main-код (cadw/*.py) не содержит 'cadw._legacy' или '_legacy.'."""
    root = pathlib.Path(cadw.__file__).parent
    offenders: list[str] = []
    for p in root.rglob("*.py"):
        if "_legacy" in str(p.relative_to(root)):
            continue
        src = p.read_text(encoding="utf-8")
        if "cadw._legacy" in src or "from ._legacy" in src:
            offenders.append(str(p.relative_to(root)))
    assert not offenders, f"main code references _legacy: {offenders}"


def test_legacy_tests_removed():
    """tests/_legacy/ удалён вместе с legacy."""
    tests_root = pathlib.Path(__file__).parent
    assert not (tests_root / "_legacy").exists(), "tests/_legacy/ must not exist"
EOF

# ---------- 3. DECISIONS §11.2 ----------
python3 - <<'PY'
import re
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

old_section = re.compile(
    r"### 11\.2\. Legacy \(заменяется, не удалено\).*?"
    r"(?=### 11\.3\.)",
    re.DOTALL,
)

new_section = """### 11.2. Legacy (удалён, патч 05)

`cadw/_legacy/` и `tests/_legacy/` **удалены** в патче 05. Всё, что там было,
заменено примитивами и фасадом:

| Что было | Замена | Где |
|---|---|---|
| `GroupByAggregator` (построчный цикл) | `groupby_aggregate` | 02a/02b, §6 |
| `UniqueCollector` (терял первый батч) | `df.unique()` = `HashShuffle + count` | 03c |
| `SpillManager` (`merge` перезаписывал) | `hash_shuffle` (партиции не пересекаются → merge не нужен) | 02b, §6 |

Функциональность не потеряна: всё доступно через `CadwDataFrame`.
Guard в `tests/test_architecture.py::test_main_code_does_not_reference_legacy`
ловит регресс, если кто-то снова сошлётся на `_legacy`.

"""

if not old_section.search(s):
    raise SystemExit("§11.2: pattern not found (manual fix required)")

s = old_section.sub(new_section, s)
p.write_text(s)
print("DECISIONS §11.2: legacy -> удалён")
PY

# ---------- 4. README: убрать Legacy-секцию ----------
python3 - <<'PY'
import re
from pathlib import Path
p = Path("README.md")
s = p.read_text()

s = re.sub(
    r"\n## Legacy\n.*?(?=\n## |\Z)",
    "\n",
    s,
    flags=re.DOTALL,
)

# Обновить статус: legacy отсутствует
s = s.replace(
    "Отсутствует: `CadwDataFrame`, `Dataset`, `RowFilter`, `unique`/`join` через\n"
    "новые примитивы (см. `DECISIONS.md` §11.3, §12).",
    "Реализовано: `CadwDataFrame`, `Dataset`, `RowFilter`, ленивые `__getitem__`, "
    "`unique`, `sort_values`, `groupby`, E2E на 200k строк / 20 MB budget. "
    "См. `DECISIONS.md` §11.1.",
    1,
)

p.write_text(s)
print("README: legacy-секция удалена, статус обновлён")
PY

# ---------- 5. Run ----------
echo "==> pytest (fast)"
set +e
uv run pytest -q -m "not slow"
RC_FAST=$?
uv run pytest -q -m slow
RC_SLOW=$?
set -e
if [[ $RC_FAST -ne 0 ]]; then echo "ERROR fast: rc=$RC_FAST"; exit 1; fi
if [[ $RC_SLOW -ne 0 ]]; then echo "ERROR slow: rc=$RC_SLOW"; exit 1; fi

# ---------- 6. Commit ----------
git add -A
git commit -m "05: remove _legacy/ (заменено примитивами 02b/03b/03c) + architectural guard"

echo "==> sha256"
sha256sum tests/test_architecture.py
echo "==> 05 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"
echo "    Hard:   git reset --hard ${BK}"