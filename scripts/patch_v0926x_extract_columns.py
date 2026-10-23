#!/usr/bin/env python3
# v0.9.26x: extract _columns (B.1 step 18, missed from v0.9.26j).
# select_dtypes only, 86 LOC. New mixin, proven creation pattern.
# Bootstrap-safe.

import ast
import builtins
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

CORE = "cadw/dataframe/_core.py"
NEW_MOD = "cadw/dataframe/_columns.py"
BUILTINS = set(dir(builtins))
CLASS_NAME = "_ColumnsMixin"

TARGET_SYMBOLS = ["select_dtypes"]

WHITELIST_FILES = [
    "cadw/patch_utils.py",
    "scripts/service/diagnose_unguarded_to_pylist.py",
    "scripts/service/diagnose_phase_b_split.py",
    "tests/test_arrow_call_memory_guard.py",
    "tests/test_v0925f_a.py",
    "tests/test_v0925d_a.py",
]


# ============ AST (proven template) ============

def build_import_map(tree):
    m = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    continue
                nm = a.asname or a.name
                m[nm] = {
                    "kind": "from",
                    "level": node.level,
                    "module": node.module,
                    "name": a.name,
                    "asname": a.asname,
                }
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    m[a.asname] = {
                        "kind": "import_as",
                        "module": a.name,
                        "asname": a.asname,
                    }
                else:
                    m[a.name.split(".")[0]] = {
                        "kind": "import",
                        "module": a.name,
                    }
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
    top_from = []
    top_import = []
    top_import_as = []
    lazy = []
    for name in sorted(undefined):
        if name in import_map:
            info = import_map[name]
            if info["kind"] == "from":
                top_from.append({
                    "level": info["level"],
                    "module": info["module"],
                    "name": info["name"],
                    "asname": info["asname"],
                })
            elif info["kind"] == "import":
                top_import.append(info["module"])
            elif info["kind"] == "import_as":
                top_import_as.append((info["module"], info["asname"]))
        elif name in core_top:
            lazy.append(name)
        else:
            raise SystemExit(f"cannot resolve {name!r}")
    return top_from, top_import, top_import_as, lazy


def build_top_import_lines(top_from, top_import, top_import_as):
    lines = []
    for m in sorted(set(top_import)):
        lines.append(f"import {m}")
    for m, asn in sorted(set(top_import_as)):
        lines.append(f"import {m} as {asn}")
    simple = {}
    aliased = []
    for info in top_from:
        level = info["level"]
        module = info["module"] or ""
        name = info["name"]
        asn = info["asname"]
        if asn:
            aliased.append((level, module, name, asn))
        else:
            simple.setdefault((level, module), []).append(name)
    for (level, module) in sorted(simple.keys()):
        names = sorted(simple[(level, module)])
        dots = "." * level
        target = dots + module
        if len(names) == 1:
            lines.append(f"from {target} import {names[0]}")
        else:
            lines.append(f"from {target} import (")
            for n in names:
                lines.append(f"    {n},")
            lines.append(")")
    for level, module, name, asn in sorted(aliased):
        dots = "." * level
        target = dots + module
        lines.append(f"from {target} import {name} as {asn}")
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
        "# cadw/dataframe/_columns.py",
        f"# v0.9.26x (B.1 step 18): {CLASS_NAME}.",
        "# Symbol: select_dtypes (missed from v0.9.26j C-1 batch).",
        "# Extracted verbatim from _core.py; no logic change.",
        "",
        "from __future__ import annotations",
        "",
    ]
    header.extend(top_import_lines)
    header.extend(["", ""])
    header.append(f"class {CLASS_NAME}:")
    header.append(
        '    """Column selection mixin (v0.9.26x)."""'
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


# ============ EXTENDED post-write lazy injection ============

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


def method_uses_any(method_source_text, names):
    import textwrap
    try:
        tree = ast.parse(textwrap.dedent(method_source_text))
    except SyntaxError:
        return [n for n in names if n in method_source_text]
    used = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            used.add(n.id)
    return [n for n in names if n in used]


def post_write_lazy_injection(lazy_names):
    print("\n--- post-write lazy injection (EXTENDED) ---")
    if not lazy_names:
        print("  no lazy names")
        return 0
    print(f"  lazy pool: {sorted(lazy_names)}")
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
        needed = method_uses_any(method_src, lazy_names)
        if not needed:
            continue
        if f"from ._core import {', '.join(sorted(needed))}" in method_src:
            continue
        insert_idx = find_body_start(lines, item.name)
        if insert_idx is None:
            print(f"  {item.name}: body-start not found")
            continue
        to_inject.append((insert_idx, item.name, needed))
    if not to_inject:
        print("  no injection needed")
        return 0
    to_inject.sort(reverse=True)
    for insert_idx, name, needed in to_inject:
        imp = "        from ._core import " + ", ".join(sorted(needed))
        lines.insert(insert_idx, imp)
        print(f"  {name}: inject {sorted(needed)} at line {insert_idx + 1}")
    new_src = "\n".join(lines)
    ast.parse(new_src, filename=NEW_MOD)
    p.write_text(new_src)
    return len(to_inject)


# ============ whitelist rewrite ============

def rewrite_whitelist():
    print("\n--- whitelist rewrite (§13.4/318/324) ---")
    import re
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
                old_prefix = f"{old_core}:{old_cls}.{sym}"
                new_prefix = f"{new_mod}:{new_class}.{sym}"
                pat = re.compile(
                    re.escape(old_prefix)
                    + r'(?=[.\"\',:\s])'
                )
                src, cnt = pat.subn(new_prefix, src)
                replaced += cnt
        if src != orig:
            fp.write_text(src)
            if fname.endswith(".py"):
                ast.parse(src, filename=fname)
            print(f"  {fname}: {replaced} key(s)")
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
    if head != "v0.9.26w":
        raise SystemExit(f"expected HEAD=v0.9.26w, got {head!r}")
    print("HEAD: " + head)

    if not Path(CORE).exists():
        raise SystemExit(CORE + " missing")
    if Path(NEW_MOD).exists():
        raise SystemExit(NEW_MOD + " exists — clean first")

    core_src = Path(CORE).read_text()
    expected_old = ("class CadwDataFrame(_ExportsMixin, _IteratorsMixin, "
                    "_MiscMixin, _IndexingMixin, _AggregationsMixin, "
                    "_TransformersFillMixin, _TransformersReshapeMixin, "
                    "_RollingMixin, _CumulativeMixin, _PerGroupMixin, "
                    "_ResampleReshapeMixin, _JoinMixin, _MLBindingsMixin)")
    if expected_old not in core_src:
        raise SystemExit(f"expected 13-mixin class missing")
    print("  _core.py: 13-mixin inheritance OK")


def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26x: extract _columns (select_dtypes) ===")
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
    print(f"  top_from ({len(top_from)}):")
    for f in top_from:
        dots = "." * f["level"]
        asn = f" as {f['asname']}" if f["asname"] else ""
        print(f"    from {dots}{f['module'] or ''} "
              f"import {f['name']}{asn}")
    print(f"  top_import: {top_import}")
    print(f"  top_import_as: {top_import_as}")
    print(f"  lazy (post-write): {sorted(lazy)}")

    # Extract
    print("\n--- extract sources ---")
    core_lines = core_src.split("\n")
    sources = extract_symbols(core_lines, symbol_ranges)
    for name in TARGET_SYMBOLS:
        if name in sources:
            print(f"  {name}: {len(sources[name].splitlines())} lines")

    # Write module
    print("\n--- write _columns.py ---")
    top_import_lines = build_top_import_lines(
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
    print(f"  _core.py parses OK ({len(new_core.splitlines())} lines)")

    # Update inheritance
    print("\n--- update _core.py class ---")
    src = Path(CORE).read_text()
    if f"from ._columns import {CLASS_NAME}" not in src:
        anchor = "from ._ml_bindings import _MLBindingsMixin"
        if anchor in src:
            src = src.replace(
                anchor,
                anchor + f"\nfrom ._columns import {CLASS_NAME}", 1,
            )
        else:
            anchor2 = "class CadwDataFrame("
            src = src.replace(
                anchor2,
                f"from ._columns import {CLASS_NAME}\n\n" + anchor2,
                1,
            )
        print("  import added")
    expected_old = ("class CadwDataFrame(_ExportsMixin, "
                    "_IteratorsMixin, _MiscMixin, _IndexingMixin, "
                    "_AggregationsMixin, _TransformersFillMixin, "
                    "_TransformersReshapeMixin, _RollingMixin, "
                    "_CumulativeMixin, _PerGroupMixin, "
                    "_ResampleReshapeMixin, _JoinMixin, "
                    "_MLBindingsMixin)")
    expected_new = ("class CadwDataFrame(_ExportsMixin, "
                    "_IteratorsMixin, _MiscMixin, _IndexingMixin, "
                    "_AggregationsMixin, _TransformersFillMixin, "
                    "_TransformersReshapeMixin, _RollingMixin, "
                    "_CumulativeMixin, _PerGroupMixin, "
                    "_ResampleReshapeMixin, _JoinMixin, "
                    f"_MLBindingsMixin, {CLASS_NAME})")
    if expected_new not in src:
        src = src.replace(expected_old, expected_new, 1)
        print("  inheritance updated (14 mixins)")
    Path(CORE).write_text(src)
    ast.parse(src, filename=CORE)

    # EXTENDED lazy injection
    n_lazy = post_write_lazy_injection(lazy)

    # Whitelist
    n_wl = rewrite_whitelist()
    print(f"\n  summary: {n_lazy} lazy injection(s), "
          f"{n_wl} whitelist key(s)")

    # Sanity
    print("\n--- sanity: hasattr only ---")
    sanity_script = (
        "import cadw\n"
        "C = cadw.CadwDataFrame\n"
        "assert hasattr(C, 'select_dtypes')\n"
        "print('hasattr: OK')\n"
    )
    r = subprocess.run(
        ["uv", "run", "python", "-c", sanity_script],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("sanity failed")
    print(r.stdout.strip())

    run_tests()

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
        "v0.9.26x: extract _columns (B.1 step 18, missed from "
        "v0.9.26j C-1). select_dtypes (~86 LOC) moved verbatim "
        f"from _core.py to _columns.py as {CLASS_NAME}. "
        "CadwDataFrame(14 mixins). Missed in v0.9.26j because "
        "C-1 TARGET_SYMBOLS only had _ColumnProxy methods; "
        "select_dtypes is CadwDataFrame method. 0 logic change; "
        "tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26x COMPLETE")


if __name__ == "__main__":
    main()