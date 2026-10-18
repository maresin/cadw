#!/usr/bin/env bash
# patch_03b-fix: AST-based guard (docstring не должен ломать проверку).
set -euo pipefail

BR="patch/03b-dataframe"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$BR" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-03b-fix-${TS}"
git diff > ".backups/backup-pre-03b-fix-${TS}/wip.diff"

python3 - <<'PY'
p = "tests/test_dataframe.py"
s = open(p).read()

old_tail = '''# ---------- facade is honest ----------

def test_facade_has_no_as_py():
    """CadwDataFrame не должен содержать построчных .as_py() на данных."""
    src = inspect.getsource(df_mod)
    assert ".as_py()" not in src, (
        "CadwDataFrame must not call .as_py() on data rows; "
        "delegate to primitives"
    )


def test_facade_has_no_row_loops():
    """Не должно быть for i in range(batch.num_rows)."""
    src = inspect.getsource(df_mod)
    assert "range(batch.num_rows)" not in src
    assert "range(b.num_rows)" not in src
'''

new_tail = '''# ---------- facade is honest (AST-based) ----------
#
# Guard проверяет реальные вызовы в AST, а не текст: строки и docstring,
# упоминающие ".as_py()" как паттерн-запрет, не должны ломать проверку.

import ast


def _call_attr_names(module: ast.Module) -> list[str]:
    """Имена атрибутов, вызываемых как Method(...). Только реальные Call."""
    out: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            out.append(node.func.attr)
    return out


def test_facade_has_no_as_py():
    """CadwDataFrame не должен вызывать .as_py() на данных (AST-level)."""
    src = inspect.getsource(df_mod)
    tree = ast.parse(src)
    bad = [n for n in _call_attr_names(tree) if n == "as_py"]
    assert not bad, (
        f"CadwDataFrame calls .as_py() {len(bad)} time(s); "
        f"delegate to primitives"
    )


def test_facade_has_no_row_loops():
    """Не должно быть for-range с диапазоном batch.num_rows (AST-level)."""
    src = inspect.getsource(df_mod)
    tree = ast.parse(src)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        it = node.iter
        if not (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
        ):
            continue
        for arg in it.args:
            if (
                isinstance(arg, ast.Attribute)
                and arg.attr == "num_rows"
            ):
                offenders.append(ast.unparse(node.target))
    assert not offenders, (
        f"row-level loops over batch.num_rows in facade: {offenders}"
    )
'''

if old_tail not in s:
    raise SystemExit("patch_03b_fix: test tail not found")
s = s.replace(old_tail, new_tail, 1)
open(p, "w").write(s)
print("fixed tests: AST-based facade guard")
PY

set +e
uv run pytest -q
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
    echo "ERROR: tests still failing (rc=${RC})"
    exit 1
fi

git add -A
git commit -m "03b-fix: AST-based facade guard (docstring does not trigger false positive)"

echo "==> sha256"
sha256sum tests/test_dataframe.py
echo "==> 03b-fix applied"