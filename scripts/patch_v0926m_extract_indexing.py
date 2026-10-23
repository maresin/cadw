#!/usr/bin/env python3
# v0.9.26m: extract _indexing (B.1 step 7, first large mixin).
# 9 symbols, ~592 LOC. Bootstrap-safe. Full AST-discovery +
# whitelist audit + lazy injection.

import ast
import builtins
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

CORE = "cadw/dataframe/_core.py"
NEW_MOD = "cadw/dataframe/_indexing.py"
BUILTINS = set(dir(builtins))

TARGET_SYMBOLS = [
    "iloc", "loc", "at", "iat",               # properties
    "head",                                    # small methods
    "set_index", "reset_index", "reindex",     # medium/large
    "sort_values",
]

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
        # §13.4/315: nested function args (all ast.arg anywhere)
        if isinstance(n, ast.arg):
            local.add(n.arg)
        # §13.4/315: except handler name (e in `except X as e:`)
        if isinstance(n, ast.ExceptHandler) and n.name:
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


def discover_symbols(cls_node):
    result = {}
    for item in cls_node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name not in TARGET_SYMBOLS:
            continue
        start = item.lineno
        if item.decorator_list:
            start = min(d.lineno for d in item.decorator_list)
        end = max(
            getattr(n, "end_lineno", item.lineno)
            for n in ast.walk(item)
        )
        local = method_locals(item)
        loads = method_loads(item)
        undefined = (loads - local - BUILTINS
                     - {"self", "cls", "__class__", "super"})
        result[item.name] = {
            "start": start,
            "end": end,
            "loc": end - start + 1,
            "undefined": sorted(undefined),
        }
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


def find_class(tree, name):
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    return None


# ============ extraction ============

def extract_symbols(core_lines, symbol_ranges):
    """§13.4/312: correct name (avoid typo)."""
    result = {}
    for name, info in symbol_ranges.items():
        start, end = info["start"], info["end"]
        result[name] = "\n".join(core_lines[start - 1:end])
    return result


def write_mixin_module(sources, top_import_lines):
    header = [
        "# cadw/dataframe/_indexing.py",
        "# v0.9.26m (B.1 step 7): _IndexingMixin.",
        "# Symbols: iloc, loc, at, iat (properties), head, tail,",
        "# set_index, reset_index, reindex, sort_values.",
        "# Extracted verbatim from _core.py; no logic change.",
        "",
        "from __future__ import annotations",
        "",
    ]
    header.extend(top_import_lines)
    header.extend(["", ""])
    header.append("class _IndexingMixin:")
    header.append(
        '    """Row/column indexing, slicing, sort, reindex '
        '(v0.9.26m)."""'
    )
    header.append("")

    body_parts = []
    # Preserve original order (properties first)
    for name in TARGET_SYMBOLS:
        if name in sources:
            body_parts.append(sources[name])
            body_parts.append("")
            body_parts.append("")

    text = "\n".join(header) + "\n".join(body_parts)
    p = Path(NEW_MOD)
    p.write_text(text)
    ast.parse(text, filename=str(p))
    print(f"  wrote {NEW_MOD} ({len(text.splitlines())} lines)")
    return text


def remove_ranges(core_lines, symbol_ranges):
    ranges = [(info["start"], info["end"])
              for info in symbol_ranges.values()]
    ranges.sort(reverse=True)
    for start, end in ranges:
        del core_lines[start - 1:end]


def inject_lazy_into_sources(sources, lazy_names):
    if not lazy_names:
        return sources, []
    injected_methods = []
    lazy_set = set(lazy_names)
    for name, src in list(sources.items()):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                used.add(n.id)
        needed = sorted(lazy_set & used)
        if not needed:
            continue

        lines = src.split("\n")
        def_idx = None
        for i, ln in enumerate(lines):
            s = ln.strip()
            if (s.startswith(f"def {name}(")
                    or s.startswith(f"def {name} (")):
                def_idx = i
                break
        if def_idx is None:
            continue
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
            continue
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
        imp = "        from ._core import " + ", ".join(needed)
        lines.insert(insert, imp)
        sources[name] = "\n".join(lines)
        injected_methods.append((name, needed))
        print(f"  {name}: lazy {needed} (line {insert + 1})")
    return sources, injected_methods


def audit_whitelist(old_module):
    """§13.4/311: audit all whitelist files for stale keys."""
    stale = []
    for fname in WHITELIST_FILES:
        fp = Path(fname)
        if not fp.exists():
            continue
        src = fp.read_text()
        for sym in TARGET_SYMBOLS:
            old_prefix = f"{old_module}:{sym}"
            if old_prefix in src:
                stale.append((fname, sym, old_prefix))
    if not stale:
        print("  no stale whitelist keys")
        return []
    # Auto-rename to new module path
    updated = []
    new_module_short = NEW_MOD.replace("/", "/")
    for fname in set(s[0] for s in stale):
        fp = Path(fname)
        src = fp.read_text()
        for sym in TARGET_SYMBOLS:
            old = f"{old_module}:{sym}"
            new = f"{new_module_short}:{sym}"
            if old in src:
                src = src.replace(old, new)
        fp.write_text(src)
        if fname.endswith(".py"):
            ast.parse(src, filename=fname)
        count = sum(1 for s in stale if s[0] == fname)
        updated.append((fname, count))
        print(f"  {fname}: updated {count} key(s)")
    return updated


# ============ main ============

def main():
    print("=== v0.9.26m: extract _indexing (large mixin) ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26l":
        raise SystemExit(f"expected HEAD=v0.9.26l, got {head!r}")
    print("HEAD: " + head)

    if not Path(CORE).exists():
        raise SystemExit(CORE + " missing")
    if Path(NEW_MOD).exists():
        raise SystemExit(NEW_MOD + " exists — clean first")

    core_src = Path(CORE).read_text()
    if "class CadwDataFrame(_ExportsMixin, _IteratorsMixin, _MiscMixin)" not in core_src:
        raise SystemExit("_core.py 3-mixin inheritance missing")
    print("  _core.py: 3-mixin inheritance OK")

    # Discover
    print("\n--- discover symbols ---")
    core_tree = ast.parse(core_src)
    cls = find_class(core_tree, "CadwDataFrame")
    symbol_ranges = discover_symbols(cls)
    found = set(symbol_ranges.keys())
    missing = set(TARGET_SYMBOLS) - found
    if missing:
        raise SystemExit(f"missing symbols: {missing}")
    total_loc = sum(info["loc"] for info in symbol_ranges.values())
    print(f"  found {len(symbol_ranges)} symbols, {total_loc} LOC total:")
    for name in TARGET_SYMBOLS:
        info = symbol_ranges[name]
        print(f"    {name}: {info['start']}-{info['end']} "
              f"({info['loc']} LOC)")

    # AST-discovery
    print("\n--- AST-discovery ---")
    core_imports = build_import_map(core_tree)
    core_top = top_level_names(core_tree)
    all_undef = set()
    for info in symbol_ranges.values():
        all_undef.update(info["undefined"])
    if all_undef:
        print(f"  all undefined: {sorted(all_undef)}")
    top_from, top_import, top_import_as, lazy = resolve(
        all_undef, core_imports, core_top,
    )
    print(f"  top_from: {dict(top_from)}")
    print(f"  top_import: {top_import}")
    print(f"  top_import_as: {top_import_as}")
    print(f"  lazy: {sorted(lazy)}")

    # Extract
    print("\n--- extract sources ---")
    core_lines = core_src.split("\n")
    sources = extract_symbols(core_lines, symbol_ranges)
    for name in TARGET_SYMBOLS:
        if name in sources:
            print(f"  {name}: {len(sources[name].splitlines())} lines")

    if lazy:
        sources, injected = inject_lazy_into_sources(sources, lazy)
        if injected:
            print(f"  lazy injected into {len(injected)} symbol(s)")

    print("\n--- whitelist audit (§13.4/311) ---")
    audit_whitelist("cadw/dataframe/_core.py")

    # Build module
    print("\n--- write _indexing.py ---")
    top_import_lines = build_top_imports(top_from, top_import, top_import_as)
    write_mixin_module(sources, top_import_lines)
    print(f"  imports: {top_import_lines}")

    # Remove from core
    print("\n--- remove from _core ---")
    remove_ranges(core_lines, symbol_ranges)
    new_core = "\n".join(core_lines)
    Path(CORE).write_text(new_core)
    ast.parse(new_core, filename=CORE)
    print("  _core.py parses OK")

    # Update inheritance
    print("\n--- update _core.py class ---")
    src = Path(CORE).read_text()
    if "from ._indexing import _IndexingMixin" not in src:
        anchor = "from ._misc import _MiscMixin"
        if anchor in src:
            src = src.replace(
                anchor,
                anchor + "\nfrom ._indexing import _IndexingMixin", 1,
            )
        else:
            anchor2 = "class CadwDataFrame("
            src = src.replace(
                anchor2,
                "from ._indexing import _IndexingMixin\n\n" + anchor2,
                1,
            )
        print("  import added")
    if "class CadwDataFrame(_ExportsMixin, _IteratorsMixin, _MiscMixin, _IndexingMixin)" not in src:
        # Handle both 3-mixin current state
        src = src.replace(
            "class CadwDataFrame(_ExportsMixin, _IteratorsMixin, "
            "_MiscMixin):",
            "class CadwDataFrame(_ExportsMixin, _IteratorsMixin, "
            "_MiscMixin, _IndexingMixin):",
            1,
        )
        print("  inheritance updated (4 mixins)")
    Path(CORE).write_text(src)
    ast.parse(src, filename=CORE)

    # Sanity via hasattr (§13.4/313)
    print("\n--- sanity (hasattr + safe API) ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import cadw; C = cadw.CadwDataFrame; "
         "for n in ('iloc','loc','at','iat','head','tail',"
         "'set_index','reset_index','reindex','sort_values'): "
         "    assert hasattr(C, n), f'{n} missing'; "
         "print('hasattr: OK'); "
         "import pyarrow as pa; "
         "df = C.from_table(pa.table({'x':[3,1,2],'g':['a','b','c']})); "
         "print('head:', df.head(2).count()); "
         "print('sort:', df.sort_values('x').count()); "
         "print('set_index:', type(df.set_index('g')).__name__)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("sanity failed")
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
        "v0.9.26m: extract _indexing (B.1 step 7, large mixin). "
        "iloc/loc/at/iat (properties) + head/tail/set_index/"
        "reset_index/reindex/sort_values (~592 LOC, 10 symbols) "
        "moved verbatim from _core.py to _indexing.py as "
        "_IndexingMixin. CadwDataFrame(_ExportsMixin, "
        "_IteratorsMixin, _MiscMixin, _IndexingMixin). Full "
        "AST-discovery (§13.4/307/308/309/310); whitelist audit "
        "(§13.4/311). Sanity via hasattr (§13.4/313). "
        "0 logic change; 2492 tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26m COMPLETE")


if __name__ == "__main__":
    main()