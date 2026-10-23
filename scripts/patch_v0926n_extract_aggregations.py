#!/usr/bin/env python3
# v0.9.26n: extract _aggregations (B.1 step 8, large mixin).
# 11 methods, ~478 LOC. Proven v0.9.26m-fix3 template:
# - Discovery (AST, parent-scope aware)
# - Write module + top-level imports
# - Remove from _core + inheritance
# - POST-WRITE lazy injection (text-based)
# - Whitelist rewrite (class-qualified)
# - Fast suite + whitelist + commit

import ast
import builtins
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

CORE = "cadw/dataframe/_core.py"
NEW_MOD = "cadw/dataframe/_aggregations.py"
BUILTINS = set(dir(builtins))
CLASS_NAME = "_AggregationsMixin"

TARGET_SYMBOLS = [
    "groupby",
    "value_counts", "corr", "nunique", "describe",
    "nlargest", "nsmallest",
    "isna", "isna_sum",
    "quantile",
    "count",
]

WHITELIST_FILES = [
    "cadw/patch_utils.py",
    "scripts/service/diagnose_unguarded_to_pylist.py",
    "scripts/service/diagnose_phase_b_split.py",
    "tests/test_arrow_call_memory_guard.py",
    "tests/test_v0925f_a.py",
    "tests/test_v0925d_a.py",
]

LAZY_IMPORT = "from ._core import CadwDataFrame"


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
    """Args + Store + nested defs + except names + nested args (§13.4/315)."""
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
        if isinstance(n, ast.arg):
            local.add(n.arg)
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
    result = {}
    for name, info in symbol_ranges.items():
        start, end = info["start"], info["end"]
        result[name] = "\n".join(core_lines[start - 1:end])
    return result


def write_mixin_module(sources, top_import_lines):
    header = [
        "# cadw/dataframe/_aggregations.py",
        f"# v0.9.26n (B.1 step 8): {CLASS_NAME}.",
        "# Symbols: groupby, value_counts, corr, nunique, describe,",
        "#   nlargest, nsmallest, isna, isna_sum, quantile, count.",
        "# Extracted verbatim from _core.py; no logic change.",
        "",
        "from __future__ import annotations",
        "",
    ]
    header.extend(top_import_lines)
    header.extend(["", ""])
    header.append(f"class {CLASS_NAME}:")
    header.append(
        '    """Aggregations / statistics mixin (v0.9.26n)."""'
    )
    header.append("")

    body_parts = []
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


# ============ POST-WRITE lazy injection ============

def find_body_start(lines, method_name):
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


def method_uses_cadwdf(method_source_text):
    import textwrap
    try:
        tree = ast.parse(textwrap.dedent(method_source_text))
    except SyntaxError:
        return "CadwDataFrame" in method_source_text
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id == "CadwDataFrame":
            return True
    return False


def post_write_lazy_injection():
    """Text-based lazy injection after module is written."""
    print("\n--- post-write lazy injection ---")
    p = Path(NEW_MOD)
    src = p.read_text()

    tree = ast.parse(src)
    cls_node = next(
        (n for n in tree.body
         if isinstance(n, ast.ClassDef) and n.name == CLASS_NAME),
        None,
    )
    if cls_node is None:
        raise SystemExit(f"{CLASS_NAME} not in {NEW_MOD}")

    lines = src.split("\n")
    to_inject = []

    for item in cls_node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        m_start = item.lineno - 1
        if item.decorator_list:
            m_start = min(d.lineno for d in item.decorator_list) - 1
        m_end = max(
            getattr(n, "end_lineno", item.lineno)
            for n in ast.walk(item)
        )
        method_src = "\n".join(lines[m_start:m_end])

        if not method_uses_cadwdf(method_src):
            continue
        if LAZY_IMPORT in method_src:
            print(f"  {item.name}: already has lazy")
            continue
        insert_idx = find_body_start(lines, item.name)
        if insert_idx is None:
            print(f"  {item.name}: body-start not found")
            continue
        to_inject.append((insert_idx, item.name))

    if not to_inject:
        print("  no injection needed")
        return 0

    to_inject.sort(reverse=True)
    for insert_idx, name in to_inject:
        lines.insert(insert_idx, "        " + LAZY_IMPORT)
        print(f"  {name}: injected at line {insert_idx + 1}")

    new_src = "\n".join(lines)
    ast.parse(new_src, filename=NEW_MOD)
    p.write_text(new_src)
    print(f"  wrote {NEW_MOD} ({len(new_src.splitlines())} lines)")
    return len(to_inject)


# ============ whitelist rewrite (§13.4/318) ============

def rewrite_whitelist():
    print("\n--- whitelist rewrite (§13.4/318) ---")
    old_core = "cadw/dataframe/_core.py"
    new_mod = NEW_MOD
    old_classes = ["CadwDataFrame"]
    new_class = CLASS_NAME

    total = 0
    for fname in WHITELIST_FILES:
        fp = Path(fname)
        if not fp.exists():
            continue
        src = fp.read_text()
        orig = src
        replaced = 0

        for sym in TARGET_SYMBOLS:
            for old_cls in old_classes:
                old_key = f"{old_core}:{old_cls}.{sym}"
                new_key = f"{new_mod}:{new_class}.{sym}"
                count = src.count(old_key)
                if count > 0:
                    src = src.replace(old_key, new_key)
                    replaced += count

        if src != orig:
            fp.write_text(src)
            if fname.endswith(".py"):
                ast.parse(src, filename=fname)
            print(f"  {fname}: {replaced} key(s) rewritten")
            total += replaced
        else:
            print(f"  {fname}: no changes")
    return total


# ============ verify + test + commit ============

def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26m":
        raise SystemExit(f"expected HEAD=v0.9.26m, got {head!r}")
    print("HEAD: " + head)

    if not Path(CORE).exists():
        raise SystemExit(CORE + " missing")
    if Path(NEW_MOD).exists():
        raise SystemExit(NEW_MOD + " exists — clean first")

    core_src = Path(CORE).read_text()
    expected_old = ("class CadwDataFrame(_ExportsMixin, _IteratorsMixin, "
                    "_MiscMixin, _IndexingMixin)")
    if expected_old not in core_src:
        raise SystemExit(f"_core.py doesn't have expected class: "
                         f"{expected_old!r}")
    print("  _core.py: 4-mixin inheritance OK")


def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26n: extract _aggregations (large mixin) ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()

    # Discover
    print("\n--- discover symbols ---")
    core_src = Path(CORE).read_text()
    core_tree = ast.parse(core_src)
    cls = find_class(core_tree, "CadwDataFrame")
    symbol_ranges = discover_symbols(cls)
    found = set(symbol_ranges.keys())
    missing = set(TARGET_SYMBOLS) - found
    if missing:
        raise SystemExit(f"missing symbols: {missing}")
    total_loc = sum(info["loc"] for info in symbol_ranges.values())
    print(f"  found {len(symbol_ranges)} symbols, {total_loc} LOC:")
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
    print(f"  lazy (resolved at POST-WRITE): {sorted(lazy)}")

    # Extract sources
    print("\n--- extract sources ---")
    core_lines = core_src.split("\n")
    sources = extract_symbols(core_lines, symbol_ranges)
    for name in TARGET_SYMBOLS:
        if name in sources:
            print(f"  {name}: {len(sources[name].splitlines())} lines")

    # Write module (WITHOUT lazy injection — post-write later)
    print("\n--- write _aggregations.py ---")
    top_import_lines = build_top_imports(
        top_from, top_import, top_import_as,
    )
    write_mixin_module(sources, top_import_lines)
    print(f"  imports: {top_import_lines}")

    # Remove from core
    print("\n--- remove from _core ---")
    remove_ranges(core_lines, symbol_ranges)
    new_core = "\n".join(core_lines)
    Path(CORE).write_text(new_core)
    ast.parse(new_core, filename=CORE)
    print("  _core.py parses OK")

    # Update class inheritance
    print("\n--- update _core.py class ---")
    src = Path(CORE).read_text()
    if f"from ._aggregations import {CLASS_NAME}" not in src:
        anchor = "from ._indexing import _IndexingMixin"
        if anchor in src:
            src = src.replace(
                anchor,
                anchor + f"\nfrom ._aggregations import {CLASS_NAME}",
                1,
            )
        else:
            anchor2 = "class CadwDataFrame("
            src = src.replace(
                anchor2,
                f"from ._aggregations import {CLASS_NAME}\n\n"
                + anchor2,
                1,
            )
        print("  import added")
    expected_new = ("class CadwDataFrame(_ExportsMixin, "
                    "_IteratorsMixin, _MiscMixin, _IndexingMixin, "
                    f"{CLASS_NAME})")
    expected_old = ("class CadwDataFrame(_ExportsMixin, "
                    "_IteratorsMixin, _MiscMixin, _IndexingMixin)")
    if expected_new not in src:
        src = src.replace(expected_old, expected_new, 1)
        print("  inheritance updated (5 mixins)")
    Path(CORE).write_text(src)
    ast.parse(src, filename=CORE)

    # Post-write lazy injection
    n_lazy = post_write_lazy_injection()

    # Whitelist rewrite
    n_wl = rewrite_whitelist()
    print(f"\n  summary: {n_lazy} lazy injections, "
          f"{n_wl} whitelist key(s) rewritten")

    # Sanity: hasattr only
    print("\n--- sanity: hasattr only (§13.4/313/317) ---")
    sanity_script = (
        "import cadw\n"
        "C = cadw.CadwDataFrame\n"
        "names = ['groupby','value_counts','corr','nunique',\n"
        "         'describe','nlargest','nsmallest','isna',\n"
        "         'isna_sum','quantile','count']\n"
        "missing = [n for n in names if not hasattr(C, n)]\n"
        "assert not missing, f'missing: {missing}'\n"
        "print('hasattr: OK', len(names), 'symbols')\n"
    )
    r = subprocess.run(
        ["uv", "run", "python", "-c", sanity_script],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("hasattr sanity failed")
    print(r.stdout.strip())

    # Tests
    run_tests()

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
        "v0.9.26n: extract _aggregations (B.1 step 8, large "
        "mixin). groupby/value_counts/corr/nunique/describe/"
        "nlargest/nsmallest/isna/isna_sum/quantile/count "
        "(~478 LOC, 11 symbols) moved verbatim from _core.py to "
        f"_aggregations.py as {CLASS_NAME}. CadwDataFrame(5 "
        "mixins). Post-write text-based lazy injection "
        "(§13.4/303/313/317). Whitelist class-qualified rewrite "
        "(§13.4/318). Full AST-discovery (§13.4/307/308/309/310/"
        "315). 0 logic change; tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26n COMPLETE")


if __name__ == "__main__":
    main()