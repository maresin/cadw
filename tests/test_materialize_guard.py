"""
AST guards: «materialize-before-check» — must-fix class of bugs (§11.4).

Правило (§3.10): любая операция, читающая весь источник в память
(`list(reader)`, `Table.from_batches(all_batches)`), обязана
предварительно проверить размер. Исключение — materialize, ограниченный
`batch_size` (один батч, уже проверен `check_batch`).
"""
import ast
import inspect
from pathlib import Path

import pytest

OPS_DIR = Path(__file__).parent.parent / "cadw" / "ops"
FORBIDDEN_READERS = {"ArrowIPCBatchReader", "ParquetBatchReader"}
ALLOWED_FUNCTIONS = {"_read_partition_safe"}


def _iter_py_files(root: Path):
    yield from root.rglob("*.py")


def test_no_list_of_whole_reader_outside_helper():
    """
    `list(<Reader>(...))` разрешён ТОЛЬКО в `_read_partition_safe`.

    Причина: helper делает `partition_path.stat().st_size` ДО `list()`,
    поэтому materialize происходит только если партиция влезает в бюджет.
    В любом другом месте — это materialize-before-check, OOM-риск.
    """
    violations: list[str] = []
    for py in _iter_py_files(OPS_DIR):
        tree = ast.parse(py.read_text())
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            if fn.name in ALLOWED_FUNCTIONS:
                continue
            for node in ast.walk(fn):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "list"
                ):
                    continue
                if not node.args:
                    continue
                inner = node.args[0]
                if not isinstance(inner, ast.Call):
                    continue
                if not (isinstance(inner.func, ast.Name) and inner.func.id in FORBIDDEN_READERS):
                    continue
                violations.append(
                    f"{py.relative_to(OPS_DIR.parent.parent)}:{node.lineno}: "
                    f"list({inner.func.id}(...)) in function {fn.name!r}. "
                    f"Use _read_partition_safe() or partition_path.stat().st_size first."
                )
    assert not violations, (
        "materialize-before-check violations (§11.4 / §3.10):\n" + "\n".join(violations)
    )


def test_reduce_estimate_state_before_init():
    """reduce() должен вызвать estimate_state_size/check_state ДО op.init()."""
    from cadw.ops.reduce import reduce

    src = inspect.getsource(reduce)
    est_pos = src.find("memory_guard.estimate_state_size(")
    init_pos = src.find("state = op.init()")
    check_pos = src.find("memory_guard.check_state(")

    assert est_pos >= 0, "estimate_state_size not called in reduce()"
    assert init_pos >= 0, "op.init() not called in reduce()"
    assert check_pos >= 0, "check_state not called in reduce()"
    assert est_pos < init_pos, (
        f"reduce: estimate_state_size (pos {est_pos}) must come BEFORE "
        f"op.init() (pos {init_pos}); иначе cov(d²)/histogram(bins) OOM до check"
    )
    assert check_pos < init_pos, (
        f"reduce: check_state (pos {check_pos}) must come BEFORE "
        f"op.init() (pos {init_pos})"
    )
