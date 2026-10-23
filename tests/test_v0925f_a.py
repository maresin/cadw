"""v0.9.25f-a: 10 pure helpers extracted to cadw/dataframe_helpers.py.

Phase B diagnostic revealed 48 AST guards block full split (silent
coverage loss). Partial extract: 433 L of pure helpers with no
CadwDataFrame cross-refs. dataframe.py 7172 → ~6740 L. AST guards
intact (single-file facade still scanned).
"""
import ast as _ast
import importlib as _importlib
import inspect as _inspect


HELPER_NAMES = [
    "_ArrowTail", "_RollState", "_ExpandingState", "_EwmState",
    "_group_boundaries_arrow", "_convert_batch", "_concat_batches",
    "_empty_value_counts_table", "_reservoir_positions",
    "_weighted_reservoir_positions",
]

HELPER_CLASSES = {
    "_ArrowTail", "_RollState", "_ExpandingState", "_EwmState",
}


def test_helpers_defined_in_helpers_module():
    """v0.9.25f-a: all 10 helpers defined in cadw/dataframe_helpers.py."""
    from cadw import dataframe_helpers as hm
    src = _inspect.getsource(hm)
    tree = _ast.parse(src)
    defs = {
        node.name for node in tree.body
        if isinstance(node, (_ast.ClassDef, _ast.FunctionDef,
                             _ast.AsyncFunctionDef))
    }
    missing = [n for n in HELPER_NAMES if n not in defs]
    assert not missing, (
        f"helpers missing from dataframe_helpers.py: {missing}"
    )


def test_helpers_re_exported_from_dataframe():
    """Backward compat: `from cadw.dataframe import _RollState` works."""
    import cadw.dataframe as df_mod
    for name in HELPER_NAMES:
        assert hasattr(df_mod, name), (
            f"cadw.dataframe missing re-export of {name!r} — "
            f"backward-compat broken (scripts/tests import these)"
        )


def test_helpers_not_defined_in_facade():
    """v0.9.25f-a belt-and-suspenders: helper definitions must NOT
    reappear in cadw/dataframe.py (would double-define and defeat
    extraction)."""
    import cadw.dataframe as df_mod
    src = _inspect.getsource(df_mod._core)
    tree = _ast.parse(src)
    for name in HELPER_NAMES:
        if name in HELPER_CLASSES:
            hits = [
                n for n in tree.body
                if isinstance(n, _ast.ClassDef) and n.name == name
            ]
        else:
            hits = [
                n for n in tree.body
                if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                and n.name == name
            ]
        assert not hits, (
            f"{name!r} re-defined in cadw/dataframe.py — "
            f"extraction regressed. Move back to "
            f"cadw/dataframe_helpers.py or remove duplicate."
        )


def test_helpers_module_no_cadwdataframe_dep():
    """Pure module invariant: helpers.py imports no CadwDataFrame
    or cadw.dataframe (else circular import)."""
    from cadw import dataframe_helpers as hm
    src = _inspect.getsource(hm)
    tree = _ast.parse(src)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ImportFrom):
            assert node.module != "cadw.dataframe", (
                "helpers.py imports from cadw.dataframe — circular"
            )
            if node.module and "CadwDataFrame" in str(node.module):
                raise AssertionError(
                    f"helpers.py imports {node.module!r} — must be pure"
                )


def test_group_boundaries_arrow_smoke():
    """v0.9.25f-a-fix1b (§13.4/257): runtime smoke test of every extracted
    symbol. Prior dogfood in v0.9.25f-a tested _RollState (numpy-free)
    only — missed that _group_boundaries_arrow needs module-level
    `import numpy as np`. This directly exercises the bug: would
    NameError if numpy not in helpers module scope.
    """
    import pyarrow as pa
    from cadw.dataframe_helpers import _group_boundaries_arrow
    b = pa.RecordBatch.from_arrays(
        [pa.array([1, 1, 2, 2, 2, 3])], names=["g"],
    )
    assert _group_boundaries_arrow(b, ["g"]) == [0, 2, 5, 6]


def test_helpers_module_has_required_imports():
    """v0.9.25f-a-fix2 (§13.4/259): module-level imports cover all
    global Name references in helper defs.

    Uses Python's canonical scope analyzer `symtable`, NOT ast.walk.
    ast.walk + manual local-collection misses:
      - comprehension targets (i, name, b in `[b.col(i) for ...]`)
      - except-as variables (e in `except X as e: ... e.foo()`)
      - AnnAssign targets (heap in `heap: list = []`)
      - lambda parameters
      - walrus (:=) targets
    Each is a real local binding, not a module-level import. Prior
    attempt (v0.9.25f-a-fix1b) false-positived on all 4 kinds;
    symtable handles them all correctly.

    Detection: Name with (a) is_referenced(), (b) is_global() = True
    (looked up in module globals / builtins at runtime), (c) NOT
    bound at module level, (d) NOT a builtin -> missing import.

    Function-local imports (`import numpy as np` inside
    `_reservoir_positions`) don't count as module-level: symtable
    reports them as local within that function only; other functions
    referencing np would still be flagged.

    Recursion into nested scopes catches missing imports at any
    depth — symtable.get_children() gives each nested function /
    class / comprehension its own scope.
    """
    import builtins as _builtins
    import inspect as _inspect
    import symtable as _symtable
    from cadw import dataframe_helpers as hm

    src = _inspect.getsource(hm)
    st = _symtable.symtable(src, "<helpers>", "exec")

    module_names = {sym.get_name() for sym in st.get_symbols()}
    builtin_names = set(dir(_builtins))

    missing: list = []

    def _walk_table(table, path):
        for child in table.get_children():
            child_path = (
                f"{path}.{child.get_name()}" if path else child.get_name()
            )
            for sym in child.get_symbols():
                if not sym.is_referenced():
                    continue
                if not sym.is_global():
                    continue
                name = sym.get_name()
                if name in module_names or name in builtin_names:
                    continue
                missing.append((child_path, name))
            _walk_table(child, child_path)

    _walk_table(st, "")

    assert not missing, (
        f"helpers.py references names not bound at module level "
        f"(would NameError at runtime): {sorted(set(missing))}. "
        f"Add missing imports to cadw/dataframe_helpers.py header. "
        f"§13.4/259: symtable.get_children() + is_global() correctly "
        f"handles comprehensions, except-as, AnnAssign, lambdas, "
        f"walrus — all of which ast.walk misses."
    )


def test_symtable_guard_no_false_positives():
    """v0.9.25f-a-fix3 (§13.4/259/260): symtable guard produces no
    false positives on comprehension / except-as / AnnAssign — the
    3 classes that broke v0.9.25f-a-fix1b's ast.walk guard.

    v0.9.25f-a-fix1b's `test_helpers_module_has_required_imports` used
    ast.walk + manual local-collection → false-positived on:
      - comprehension targets (`[b.col(i) for i in ...]` → i, b)
      - except-as variables (`except X as e: ... e.foo()` → e)
      - AnnAssign targets (`heap: list = []` → heap)
    v0.9.25f-a-fix2 replaced with symtable — this test proves it
    handles all 3 (plus synthetic missing-import detection).

    Previously covered only by fix2's dogfood (subprocess isolation,
    triple-quoted dedent bug aborted it). Now permanent in pytest —
    no subprocess, no dedent issue.
    """
    import builtins as _builtins
    import symtable as _symtable

    builtin_names = set(dir(_builtins))

    def _collect_missing(src: str) -> list:
        st = _symtable.symtable(src, "<test>", "exec")
        module_names = {sym.get_name() for sym in st.get_symbols()}
        missing: list = []

        def _walk(table, path):
            for child in table.get_children():
                child_path = (
                    f"{path}.{child.get_name()}" if path else child.get_name()
                )
                for sym in child.get_symbols():
                    if not sym.is_referenced():
                        continue
                    if not sym.is_global():
                        continue
                    name = sym.get_name()
                    if name in module_names or name in builtin_names:
                        continue
                    missing.append((child_path, name))
                _walk(child, child_path)

        _walk(st, "")
        return missing

    # Case 1: comprehension target
    fake_comp = (
        "import pyarrow as pa\n"
        "def with_comprehension(x, n):\n"
        "    return [x * i for i in range(n)]\n"
    )
    assert _collect_missing(fake_comp) == [], (
        "false positive on comprehension target"
    )

    # Case 2: except-as variable
    fake_except = (
        "def with_except():\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError as e:\n"
        "        return str(e)\n"
    )
    assert _collect_missing(fake_except) == [], (
        "false positive on except-as variable"
    )

    # Case 3: AnnAssign target
    fake_ann = (
        "def with_annassign():\n"
        "    heap: list = []\n"
        "    return len(heap)\n"
    )
    assert _collect_missing(fake_ann) == [], (
        "false positive on AnnAssign target"
    )

    # Case 4: real missing import MUST be detected
    fake_missing = (
        "import pyarrow as pa\n"
        "def uses_numpy(x):\n"
        "    return np.array([1, 2, 3])\n"
    )
    issues = _collect_missing(fake_missing)
    assert any(name == "np" for _, name in issues), (
        f"missing numpy not detected: {issues}"
    )


def test_all_whitelist_paths_valid():
    """v0.9.25f-a-fix4 (§13.4/261): every whitelist key's file path
    exists and defines the named qualname.

    Prevents stale-path class that broke v0.9.25f-a (extract moved
    `_concat_batches` / `_convert_batch` / `_ArrowTail.push` to
    cadw/dataframe_helpers.py; whitelist keys still referenced
    cadw/dataframe.py → orphan in WHITELISTED_FUNCS (dead guard) +
    unguarded classification in diagnostic).

    Two whitelists checked:
      1. tests/test_arrow_call_memory_guard.py::WHITELISTED_FUNCS
      2. scripts/service/diagnose_unguarded_to_pylist.py::WHITELIST

    Both are keyed by `{path}:{qualname}`; qualname may be dotted for
    methods (Class.method). Uses AST qualname collection (same as
    verify_whitelist + _collect_all_qualnames).
    """
    import ast as _ast
    import importlib as _importlib
    import importlib.util as _importlib_util
    from pathlib import Path as _P

    cadw_root = _P("cadw")
    actual: set = set()
    for py in cadw_root.rglob("*.py"):
        try:
            tree = _ast.parse(py.read_text())
        except (SyntaxError, OSError):
            continue
        try:
            rel = py.relative_to(_P(".")).as_posix()
        except ValueError:
            rel = py.as_posix()

        def _walk(node, prefix):
            body = getattr(node, "body", None)
            if body is None:
                return
            for child in body:
                if isinstance(child, _ast.FunctionDef):
                    qn = prefix + child.name
                    actual.add(rel + ":" + qn)
                    _walk(child, qn + ".")
                elif isinstance(child, _ast.ClassDef):
                    cp = prefix + child.name + "."
                    for item in child.body:
                        if isinstance(item, _ast.FunctionDef):
                            qn = cp + item.name
                            actual.add(rel + ":" + qn)
                            _walk(item, qn + ".")
        _walk(tree, "")

    # --- Whitelist 1: WHITELISTED_FUNCS in test_arrow_call_memory_guard
    wl1 = _importlib.import_module(
        "tests.test_arrow_call_memory_guard"
    ).WHITELISTED_FUNCS
    stale1 = [k for k in wl1 if k not in actual]

    # --- Whitelist 2: WHITELIST in diagnostic
    spec = _importlib_util.spec_from_file_location(
        "diag_mod",
        _P("scripts/service/diagnose_unguarded_to_pylist.py"),
    )
    diag = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(diag)
    wl2 = diag.WHITELIST
    stale2 = [k for k in wl2 if k not in actual]

    msg_parts = []
    if stale1:
        msg_parts.append(
            f"WHITELISTED_FUNCS orphan keys: {sorted(stale1)}"
        )
    if stale2:
        msg_parts.append(
            f"diagnostic WHITELIST orphan keys: {sorted(stale2)}"
        )

    assert not msg_parts, (
        "§13.4/261: stale whitelist path(s) after refactor:\n"
        + "\n".join(msg_parts)
        + "\nUpdate to new path in SAME patch that moves the "
        + "symbol."
    )
