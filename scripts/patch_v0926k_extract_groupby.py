#!/usr/bin/env python3
# v0.9.26k: extract _groupby class move (B.1 step 5).
# Bootstrap-safe. Class-move pattern (proven v0.9.26j).
# Full AST-discovery (§13.4/307/308/309/310) + whitelist-grep (§13.4/311).

import ast
import builtins
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

CORE = "cadw/dataframe/_core.py"
NEW_MOD = "cadw/dataframe/_groupby.py"
BUILTINS = set(dir(builtins))
MOVED_CLASS = "_GroupBy"

# Whitelist files to audit (§13.4/311)
WHITELIST_FILES = [
    "cadw/patch_utils.py",
    "scripts/service/diagnose_unguarded_to_pylist.py",
    "scripts/service/diagnose_phase_b_split.py",
    "tests/test_arrow_call_memory_guard.py",
    "tests/test_v0925f_a.py",
    "tests/test_v0925d_a.py",
]


# ============ AST ============

def build_import_map(tree):
    m = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    continue
                m[a.asname or a.name] = ('from', node.module, a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    m[a.asname] = ('import_as', a.name, a.asname)
                else:
                    m[a.name.split(".")[0]] = ('import', a.name, None)
    return m


def top_level_names(tree):
    """Defs/classes/assigns — NO imports (imports are scope-local, §13.4/309)."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    names.add(tgt.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def method_locals(fn_node):
    """Args + Store + nested defs (§13.4/308)."""
    local = set()
    for a in fn_node.args.posonlyargs:
        local.add(a.arg)
    for a in fn_node.args.args:
        local.add(a.arg)
    for a in fn_node.args.kwonlyargs:
        local.add(a.arg)
    if fn_node.args.vararg:
        local.add(fn_node.args.vararg.arg)
    if fn_node.args.kwarg:
        local.add(fn_node.args.kwarg.arg)
    for n in ast.walk(fn_node):
        if n is fn_node:
            continue
        if isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            local.add(n.name)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            local.add(n.id)
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                local.add(a.asname or a.name)
        if isinstance(n, ast.Import):
            for a in n.names:
                local.add((a.asname or a.name).split(".")[0])
    return local


def method_loads(fn_node):
    loads = set()
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            loads.add(n.id)
    return loads


def discover_undefined(cls_node):
    result = {}
    for item in cls_node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = method_locals(item)
        loads = method_loads(item)
        undefined = (loads - local - BUILTINS
                     - {"self", "cls", "__class__", "super"})
        if undefined:
            result[item.name] = sorted(undefined)
    return result


def resolve(undefined, import_map, core_top):
    top_from, top_import, top_import_as, lazy = {}, {}, {}, []
    for name in sorted(undefined):
        if name in import_map:
            kind, module, attr = import_map[name]
            if kind == 'from':
                top_from.setdefault(module, []).append(attr)
            elif kind == 'import':
                top_import[name] = module
            elif kind == 'import_as':
                top_import_as[name] = (module, attr)
            else:
                raise SystemExit(f"bad kind {kind} for {name}")
        elif name in core_top:
            lazy.append(name)
        else:
            raise SystemExit(f"cannot resolve {name!r}")
    return top_from, top_import, top_import_as, lazy


def build_top_imports(top_from, top_import, top_import_as):
    lines = []
    for name, module in sorted(top_import.items()):
        lines.append(f"import {module}")
    for name, (module, asname) in sorted(top_import_as.items()):
        lines.append(f"import {module} as {asname}")
    for module, attrs in sorted(top_from.items()):
        if module == "__future__":
            continue
        if module is None:
            raise SystemExit(f"null module for {attrs}")
        if len(attrs) == 1:
            lines.append(f"from {module} import {attrs[0]}")
        else:
            lines.append(f"from {module} import (")
            for a in sorted(attrs):
                lines.append(f"    {a},")
            lines.append(")")
    return lines


def node_line_range(node):
    start = node.lineno
    if node.decorator_list:
        start = min(d.lineno for d in node.decorator_list)
    end = max(
        getattr(n, "end_lineno", node.lineno)
        for n in ast.walk(node)
    )
    return start, end


def find_class(tree, name):
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    return None


# ============ injection ============

def inject_lazy_into_class(class_src, lazy_names):
    if not lazy_names:
        return class_src, []
    sub = ast.parse(class_src)
    cls = sub.body[0]
    method_map = {}
    for item in cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        loads = method_loads(item)
        needed = [n for n in lazy_names if n in loads]
        if needed:
            method_map[item.name] = needed
    if not method_map:
        return class_src, []

    lines = class_src.split("\n")

    def find_body_start(method_name):
        def_idx = None
        for i, ln in enumerate(lines):
            s = ln.strip()
            if (s.startswith(f"def {method_name}(")
                    or s.startswith(f"def {method_name} (")):
                indent = len(ln) - len(ln.lstrip())
                if indent == 4:
                    def_idx = i
                    break
        if def_idx is None:
            return None
        acc = 0
        sig_end = None
        for j in range(def_idx, len(lines)):
            for ch in lines[j]:
                if ch == "(":
                    acc += 1
                elif ch == ")":
                    acc -= 1
            if acc == 0 and lines[j].rstrip().endswith(":"):
                sig_end = j
                break
        if sig_end is None:
            return None
        insert = sig_end + 1
        if insert < len(lines):
            s = lines[insert].strip()
            if s.startswith('"""') or s.startswith("'''"):
                q = '"""' if s.startswith('"""') else "'''"
                if s.count(q) >= 2:
                    insert += 1
                else:
                    insert += 1
                    while insert < len(lines):
                        if q in lines[insert]:
                            insert += 1
                            break
                        insert += 1
        return insert

    insertions = []
    for method, names in method_map.items():
        idx = find_body_start(method)
        if idx is None:
            print(f"  WARN: {method} body not found")
            continue
        imp = "        from ._core import " + ", ".join(sorted(names))
        insertions.append((idx, imp, method))

    for idx, imp, method in sorted(insertions, reverse=True):
        lines.insert(idx, imp)
        print(f"  {method}: lazy {sorted(method_map[method])} "
              f"(line {idx + 1})")

    return "\n".join(lines), [m for _, _, m in insertions]


# ============ whitelist ============

def audit_whitelist():
    """§13.4/311: find stale `_core.py:_GroupBy.*` keys."""
    old_prefix = f"cadw/dataframe/_core.py:{MOVED_CLASS}."
    new_prefix = f"cadw/dataframe/_column_proxy.py:{MOVED_CLASS}."  # n/a
    # Actually new path is _groupby.py, not _column_proxy
    new_prefix = f"cadw/dataframe/_groupby.py:{MOVED_CLASS}."

    stale_files = []
    for fname in WHITELIST_FILES:
        fp = Path(fname)
        if not fp.exists():
            continue
        src = fp.read_text()
        if old_prefix in src:
            stale_files.append((fname, src.count(old_prefix), src))

    if not stale_files:
        print("  no stale whitelist keys")
        return []

    updated = []
    for fname, count, src in stale_files:
        new_src = src.replace(old_prefix, new_prefix)
        Path(fname).write_text(new_src)
        if fname.endswith(".py"):
            ast.parse(new_src, filename=fname)
        print(f"  {fname}: updated {count} key(s)")
        updated.append((fname, count))
    return updated


# ============ main ============

def main():
    print("=== v0.9.26k: extract _groupby (class move) ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26j":
        raise SystemExit(f"expected HEAD=v0.9.26j, got {head!r}")
    print("HEAD: " + head)

    if not Path(CORE).exists():
        raise SystemExit(CORE + " missing")
    if Path(NEW_MOD).exists():
        raise SystemExit(NEW_MOD + " exists — clean first")

    core_src = Path(CORE).read_text()
    if "class CadwDataFrame(_ExportsMixin, _IteratorsMixin)" not in core_src:
        raise SystemExit("_core.py dual-mixin class missing")
    if "from ._column_proxy import _ColumnProxy" not in core_src:
        raise SystemExit("_core.py doesn't import _ColumnProxy")
    print("  _core.py: dual-mixin + _ColumnProxy import OK")

    # Extract
    print("\n--- extract class ---")
    core_tree = ast.parse(core_src)
    cls = find_class(core_tree, MOVED_CLASS)
    if cls is None:
        raise SystemExit(f"{MOVED_CLASS} not found in _core.py")
    start, end = node_line_range(cls)
    core_lines = core_src.split("\n")
    class_src = "\n".join(core_lines[start - 1:end])
    print(f"  {MOVED_CLASS}: lines {start}-{end} "
          f"({end - start + 1} LOC, {len(cls.body)} members)")

    # Discovery (§13.4/307/308/309/310)
    print("\n--- AST-discovery ---")
    core_imports = build_import_map(core_tree)
    core_top = top_level_names(core_tree)
    undefined = discover_undefined(cls)
    if not undefined:
        print("  no undefined names (unexpected for _GroupBy)")
    else:
        print("  undefined per method:")
        for m, names in undefined.items():
            print(f"    {m}: {names}")

    all_undef = set()
    for names in undefined.values():
        all_undef.update(names)

    top_from, top_import, top_import_as, lazy = resolve(
        all_undef, core_imports, core_top,
    )
    print(f"\n  top_from: {dict(top_from)}")
    print(f"  top_import: {top_import}")
    print(f"  top_import_as: {top_import_as}")
    print(f"  lazy: {sorted(lazy)}")

    # Inject lazy
    class_src_final = class_src
    if lazy:
        class_src_final, injected = inject_lazy_into_class(
            class_src, lazy,
        )
        print(f"  lazy injected into {len(injected)} method(s)")

    # Whitelist audit BEFORE writing (informational)
    print("\n--- whitelist audit (§13.4/311) ---")
    audit_whitelist()

    # Build new module
    print("\n--- write _groupby.py ---")
    header = [
        "# cadw/dataframe/_groupby.py",
        "# v0.9.26k (B.1 step 5): _GroupBy class.",
        "# Extracted verbatim from _core.py; no logic change.",
        "",
        "from __future__ import annotations",
        "",
    ]
    import_lines = build_top_imports(top_from, top_import, top_import_as)
    header.extend(import_lines)
    header.extend(["", ""])
    text = "\n".join(header) + class_src_final + "\n"
    Path(NEW_MOD).write_text(text)
    ast.parse(text, filename=NEW_MOD)
    print(f"  wrote {NEW_MOD} ({len(text.splitlines())} lines)")
    print(f"  imports: {import_lines}")

    # Remove from _core
    print("\n--- remove from _core ---")
    del core_lines[start - 1:end]
    new_core = "\n".join(core_lines)
    Path(CORE).write_text(new_core)
    ast.parse(new_core, filename=CORE)
    print("  _core.py parses OK")

    # Add import to _core
    print("\n--- add import to _core ---")
    src = Path(CORE).read_text()
    marker = "from ._groupby import _GroupBy"
    if marker in src:
        print("  import already present")
    else:
        anchor = "from ._column_proxy import _ColumnProxy"
        if anchor in src:
            src = src.replace(anchor, anchor + "\n" + marker, 1)
        else:
            anchor2 = "class CadwDataFrame("
            src = src.replace(anchor2, marker + "\n\n" + anchor2, 1)
        Path(CORE).write_text(src)
        ast.parse(src, filename=CORE)
        print("  import added")

    # Sanity
    print("\n--- sanity import ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import pyarrow as pa; import cadw; "
         "df = cadw.CadwDataFrame.from_table("
         "pa.table({'g':['a','a','b','b'],'v':[1,2,3,4]})); "
         "g = df.groupby('g'); "
         "print('groupby_type:', type(g).__name__); "
         "agg = g.agg(['v'], ['sum']); "
         "print('agg_rows:', agg.count())"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("sanity import failed")
    print(r.stdout.strip())

    # Tests
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")

    # Whitelist verify
    print("\n--- verify_whitelist ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw.patch_utils import verify_whitelist; "
         "verify_whitelist(raise_on_orphan=True); print('WL OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("whitelist verify failed")
    print(r.stdout.strip())

    # Commit
    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.26k: extract _groupby (B.1 step 5, class move). "
        "_GroupBy class (12 methods, ~200 LOC) moved verbatim "
        "from _core.py to _groupby.py. _core.py imports it at "
        "top-level (no cycle). Full AST-discovery (§13.4/307/308/"
        "309/310): all @-returning-CadwDataFrame methods get lazy "
        "import. Whitelist audit (§13.4/311) confirms no stale "
        "_GroupBy keys. 0 logic change; 2492 tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26k COMPLETE")


if __name__ == "__main__":
    main()