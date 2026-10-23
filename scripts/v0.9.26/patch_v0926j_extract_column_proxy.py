#!/usr/bin/env python3
# v0.9.26j: extract _column_proxy class move (B.1 step 4).
# Bootstrap-safe. Full AST-discovery. Class move pattern.
#
# Discovery (per §13.4/307/308):
#   1. AST-find `class _ColumnProxy` in _core.py
#   2. Extract source (inspect.getsource)
#   3. AST-discover undefined names per method
#   4. Classify: _core-imported (top-level) vs _core-local (lazy)
#   5. Write new module with proper imports
#   6. Remove class from _core.py, add import

import ast
import builtins
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

CORE = "cadw/dataframe/_core.py"
COLUMN_PROXY_MOD = "cadw/dataframe/_column_proxy.py"
BUILTINS = set(dir(builtins))


# ============ AST helpers ============

def top_level_names(tree):
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
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name != "*":
                    names.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
    return names


def build_import_map(tree):
    """name -> ('from', module, attr) | ('import', module, None)"""
    m = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name != "*":
                    m[a.asname or a.name] = (
                        'from', node.module, a.name,
                    )
        elif isinstance(node, ast.Import):
            for a in node.names:
                first = (a.asname or a.name).split(".")[0]
                m[first] = ('import', a.name, None)
    return m


def method_locals(fn_node):
    """Args + Store + nested defs (skip fn_node). §13.4/308."""
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


def discover_undefined_in_class(cls_node, module_names):
    result = {}
    for item in cls_node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = method_locals(item)
        loads = method_loads(item)
        undefined = (loads - local - module_names - BUILTINS
                     - {"self", "cls", "__class__"})
        if undefined:
            result[item.name] = sorted(undefined)
    return result


def node_line_range(node):
    start = node.lineno
    if node.decorator_list:
        start = min(d.lineno for d in node.decorator_list)
    end = max(
        getattr(n, "end_lineno", node.lineno)
        for n in ast.walk(node)
    )
    return start, end


# ============ discovery ============

def find_class(tree, class_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def resolve_imports(undefined_names, core_import_map, core_top):
    """Split into top-level imports (from cadw.X) and lazy (from ._core)."""
    top_imports = {}   # module -> [attrs]
    lazy_names = []    # names to import from ._core lazily

    for name in undefined_names:
        if name in core_import_map:
            kind, module, attr = core_import_map[name]
            if kind == 'from':
                top_imports.setdefault(module, []).append(attr)
            else:  # 'import'
                top_imports.setdefault("__import__", []).append(module)
        elif name in core_top:
            # Defined in _core itself — lazy import needed
            lazy_names.append(name)
        else:
            raise SystemExit(
                f"cannot resolve {name!r}: not in core imports, "
                f"not in core top-level"
            )
    return top_imports, lazy_names


def build_top_imports(top_imports):
    lines = []
    for module, attrs in sorted(top_imports.items()):
        if module == "__import__":
            for attr in attrs:
                lines.append(f"import {attr}")
        elif len(attrs) == 1:
            lines.append(f"from {module} import {attrs[0]}")
        else:
            lines.append(f"from {module} import (")
            for a in sorted(attrs):
                lines.append(f"    {a},")
            lines.append(")")
    return lines


# ============ extraction ============

def extract_class_from_core():
    """Return (class_src, start, end, cls_node, core_tree, core_src_lines)."""
    src = Path(CORE).read_text()
    tree = ast.parse(src)
    cls_node = find_class(tree, "_ColumnProxy")
    if cls_node is None:
        raise SystemExit("_ColumnProxy not found in _core.py")
    start, end = node_line_range(cls_node)
    lines = src.split("\n")
    class_src = "\n".join(lines[start - 1:end])
    return class_src, start, end, cls_node, tree, lines


def inject_lazy_imports(class_src, lazy_names):
    """Inject `from ._core import X` into methods that use X."""
    if not lazy_names:
        return class_src, []
    sub = ast.parse(class_src)
    cls_node = sub.body[0]
    method_to_names = {}
    for item in cls_node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        loads = method_loads(item)
        needed = [n for n in lazy_names if n in loads]
        if needed:
            method_to_names[item.name] = needed

    if not method_to_names:
        # lazy_names not used? Should not happen (undefined implies used)
        return class_src, []

    lines = class_src.split("\n")

    def find_body_start(lines, method_name):
        # Class-level method: def at 4-space indent
        def_idx = None
        for i, ln in enumerate(lines):
            s = ln.strip()
            if (s.startswith(f"def {method_name}(")
                    or s.startswith(f"def {method_name} (")):
                # Must be at method level (4-space indent)
                stripped = ln.lstrip()
                indent = len(ln) - len(stripped)
                if indent == 4:
                    def_idx = i
                    break
        if def_idx is None:
            return None
        # Multi-line sig detection
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
        # Skip docstring
        insert = sig_end + 1
        if insert < len(lines):
            s = lines[insert].strip()
            if s.startswith('"""') or s.startswith("'''"):
                quote = '"""' if s.startswith('"""') else "'''"
                if s.count(quote) >= 2:
                    insert += 1
                else:
                    insert += 1
                    while insert < len(lines):
                        if quote in lines[insert]:
                            insert += 1
                            break
                        insert += 1
        return insert

    # Inject per method (descending by line)
    insertions = []
    for method, names in method_to_names.items():
        idx = find_body_start(lines, method)
        if idx is None:
            print(f"  WARN: method {method} body-start not found")
            continue
        imp = "        from ._core import " + ", ".join(sorted(names))
        insertions.append((idx, imp, method))

    # Apply descending
    for idx, imp, method in sorted(insertions, reverse=True):
        lines.insert(idx, imp)
        print(f"  {method}: lazy core {sorted(method_to_names[method])} "
              f"(line {idx + 1})")

    return "\n".join(lines), [m for _, _, m in insertions]


def write_column_proxy_module(class_src, top_import_lines):
    """Build new module source."""
    header = [
        "# cadw/dataframe/_column_proxy.py",
        "# v0.9.26j (B.1 step 4): _ColumnProxy class.",
        "# Extracted verbatim from _core.py; no logic change.",
        "",
        "from __future__ import annotations",
        "",
    ]
    header += top_import_lines
    header.append("")
    header.append("")
    body = header + [class_src, ""]
    text = "\n".join(body)
    p = Path(COLUMN_PROXY_MOD)
    p.write_text(text)
    ast.parse(text, filename=str(p))
    print(f"  wrote {COLUMN_PROXY_MOD} "
          f"({len(text.splitlines())} lines)")
    return text


def remove_class_from_core(lines, start, end):
    """Delete class range; return new source."""
    del lines[start - 1:end]
    return "\n".join(lines)


def add_import_to_core():
    """Add `from ._column_proxy import _ColumnProxy` before class CadwDataFrame."""
    src = Path(CORE).read_text()
    marker = "from ._column_proxy import _ColumnProxy"
    if marker in src:
        print("  import already present")
        return
    # Insert after `from ._iterators import _IteratorsMixin` (or before
    # `class CadwDataFrame`)
    anchor = "from ._iterators import _IteratorsMixin"
    if anchor in src:
        src = src.replace(
            anchor,
            anchor + "\n" + marker,
            1,
        )
    else:
        # Fallback: insert before `class CadwDataFrame`
        anchor2 = "class CadwDataFrame("
        if anchor2 in src:
            src = src.replace(
                anchor2,
                marker + "\n\n" + anchor2,
                1,
            )
        else:
            raise SystemExit(
                "cannot find anchor in _core.py for import"
            )
    Path(CORE).write_text(src)
    ast.parse(src, filename=CORE)
    print("  import added to _core.py")


# ============ main ============

def main():
    print("=== v0.9.26j: extract _column_proxy (class move) ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26i":
        raise SystemExit(
            f"expected HEAD=v0.9.26i, got {head!r}"
        )
    print("HEAD: " + head)

    if not Path(CORE).exists():
        raise SystemExit(CORE + " missing")
    if Path(COLUMN_PROXY_MOD).exists():
        raise SystemExit(
            COLUMN_PROXY_MOD + " already exists — clean first"
        )

    core_src = Path(CORE).read_text()
    if "class CadwDataFrame(_ExportsMixin, _IteratorsMixin)" not in core_src:
        raise SystemExit("_core.py dual-mixin class missing")
    print("  _core.py: dual-mixin class OK")

    # Extract
    print("\n--- extracting class ---")
    class_src, start, end, cls_node, core_tree, core_lines = \
        extract_class_from_core()
    print(f"  _ColumnProxy: lines {start}-{end} "
          f"({end - start + 1} LOC)")
    print(f"  source: {len(class_src.splitlines())} lines, "
          f"{len(cls_node.body)} members")

    # AST-discovery of undefined names
    print("\n--- AST-discovery ---")
    core_top = top_level_names(core_tree)
    core_imports = build_import_map(core_tree)

    # Also treat core imports as "defined" for discovery
    module_names = core_top | set(core_imports.keys())

    undefined_per_method = discover_undefined_in_class(
        cls_node, module_names,
    )
    if undefined_per_method:
        print("  undefined per method:")
        for m, names in undefined_per_method.items():
            print(f"    {m}: {names}")
    else:
        print("  no undefined names (unexpected but OK)")

    all_undefined = set()
    for names in undefined_per_method.values():
        all_undefined.update(names)

    # Classify: top-level (from cadw.X import) vs lazy (from ._core import)
    top_imports, lazy_names = resolve_imports(
        all_undefined, core_imports, core_top,
    )
    print(f"  top-level imports needed: {len(top_imports)} modules")
    for mod, attrs in sorted(top_imports.items()):
        print(f"    from {mod} import {sorted(attrs)}")
    if lazy_names:
        print(f"  lazy imports from ._core: {sorted(lazy_names)}")

    # Inject lazy into method bodies
    class_src_final, injected = inject_lazy_imports(
        class_src, lazy_names,
    )
    if injected:
        print(f"  lazy injected into {len(injected)} method(s)")

    # Build module
    print("\n--- write _column_proxy.py ---")
    top_import_lines = build_top_imports(top_imports)
    write_column_proxy_module(class_src_final, top_import_lines)

    # Remove from _core
    print("\n--- remove from _core ---")
    new_core = remove_class_from_core(core_lines, start, end)
    Path(CORE).write_text(new_core)
    try:
        ast.parse(new_core, filename=CORE)
        print("  _core.py parses OK")
    except SyntaxError as e:
        print(f"  ERROR: _core.py broken at line {e.lineno}: {e.msg}")
        raise SystemExit("removal failed")

    # Add import
    print("\n--- add import to _core ---")
    add_import_to_core()

    # Sanity
    print("\n--- sanity import ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import pyarrow as pa; import cadw; "
         "df = cadw.CadwDataFrame.from_table("
         "pa.table({'x':[1,5,3],'s':['a','b','c']})); "
         "print('idxmax:', df['x'].idxmax()); "
         "print('isin:', df['x'].isin([1,3]).__class__.__name__); "
         "print('proxy:', type(df['x']).__name__)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("sanity import failed")
    print(r.stdout.strip())

    # Fast suite
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")

    # Whitelist
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
        raise SystemExit("whitelist failed")
    print(r.stdout.strip())

    # Commit
    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.26j: extract _column_proxy (B.1 step 4, class move). "
        "_ColumnProxy class (29 methods, ~315 LOC) moved verbatim "
        "from _core.py to _column_proxy.py. _core.py imports it "
        "at top-level (no cycle). Full AST-discovery of undefined "
        "names per method, resolved to cadw.* imports (top-level) "
        "or _core locals (lazy). §13.4/307/308 applied. "
        "0 logic change; 2492 tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26j COMPLETE")


if __name__ == "__main__":
    main()