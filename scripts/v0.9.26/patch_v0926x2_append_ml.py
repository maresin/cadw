#!/usr/bin/env python3
# v0.9.26x2: append 3 missed fit_* to _ml_bindings.py.
# fit_naive_bayes / fit_lasso / fit_elasticnet, 75 LOC.
# First append pattern (§13.4/329). Bootstrap-safe.

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
ML_MOD = "cadw/dataframe/_ml_bindings.py"
CLASS_NAME = "_MLBindingsMixin"
BUILTINS = set(dir(builtins))

TARGET_SYMBOLS = [
    "fit_naive_bayes",
    "fit_lasso",
    "fit_elasticnet",
]

WHITELIST_FILES = [
    "cadw/patch_utils.py",
    "scripts/service/diagnose_unguarded_to_pylist.py",
    "scripts/service/diagnose_phase_b_split.py",
    "tests/test_arrow_call_memory_guard.py",
    "tests/test_v0925f_a.py",
    "tests/test_v0925d_a.py",
]


# ============ AST helpers ============

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


def collect_module_scope_names(tree):
    """§13.4/329: top-level defs/classes/assigns + import aliases."""
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


def find_class(tree, name):
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    return None


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


def resolve_append(undefined, core_import_map, core_top,
                   ml_scope_names):
    """For each undefined name:
       - if already in _ml_bindings module scope → no import
       - if in core_import_map → top-level import (from cadw.X)
       - if in core_top (module-level in _core) → lazy from ._core
       - else → error"""
    top_from = []
    top_import = []
    top_import_as = []
    lazy = []
    already = []

    for name in sorted(undefined):
        if name in ml_scope_names:
            already.append(name)
            continue
        if name in core_import_map:
            info = core_import_map[name]
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

    return top_from, top_import, top_import_as, lazy, already


def build_import_lines(top_from, top_import, top_import_as):
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


# ============ append to _ml_bindings.py ============

def append_imports(missing_import_lines):
    """Insert new imports after last top-level ImportFrom/Import."""
    if not missing_import_lines:
        print("  no new imports needed")
        return 0
    print(f"  adding {len(missing_import_lines)} import line(s):")
    for ln in missing_import_lines:
        print(f"    {ln}")

    p = Path(ML_MOD)
    src = p.read_text()
    tree = ast.parse(src)
    last_import_end = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            end = getattr(node, "end_lineno", node.lineno)
            last_import_end = max(last_import_end, end)
    if last_import_end == 0:
        raise SystemExit("no imports found in _ml_bindings.py")

    lines = src.split("\n")
    # Insert at index last_import_end (0-based → after line last_import_end)
    insert_at = last_import_end
    for ln in reversed(missing_import_lines):
        lines.insert(insert_at, ln)

    new_src = "\n".join(lines)
    ast.parse(new_src, filename=ML_MOD)
    p.write_text(new_src)
    print(f"  imports inserted at line {insert_at + 1}")
    return len(missing_import_lines)


def append_methods(sources):
    """Insert methods after class end (after cls.end_lineno)."""
    print(f"\n  appending {len(sources)} method(s):")
    for name in TARGET_SYMBOLS:
        if name in sources:
            print(f"    {name}")

    p = Path(ML_MOD)
    src = p.read_text()
    tree = ast.parse(src)
    cls = find_class(tree, CLASS_NAME)
    if cls is None:
        raise SystemExit(f"{CLASS_NAME} not in {ML_MOD}")

    # Idempotency: skip if any target already in class
    existing = {
        item.name for item in cls.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    to_append = [n for n in TARGET_SYMBOLS if n in sources and n not in existing]
    if not to_append:
        print("  all methods already present — idempotent skip")
        return 0

    lines = src.split("\n")
    # cls.end_lineno is 1-based; insert at index = end_lineno
    insert_at = cls.end_lineno
    block = [""]  # blank separator
    for name in to_append:
        block.extend(sources[name].split("\n"))
        block.append("")  # blank between methods
        block.append("")

    lines[insert_at:insert_at] = block
    new_src = "\n".join(lines)
    ast.parse(new_src, filename=ML_MOD)
    p.write_text(new_src)
    print(f"  {len(to_append)} methods appended at line {insert_at + 1}")
    return len(to_append)


# ============ lazy injection ============

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
    if not lazy_names:
        print("  no lazy names needed")
        return 0
    print(f"\n  lazy pool: {sorted(lazy_names)}")
    p = Path(ML_MOD)
    src = p.read_text()
    tree = ast.parse(src)
    cls = find_class(tree, CLASS_NAME)
    if cls is None:
        raise SystemExit(f"{CLASS_NAME} not in {ML_MOD}")
    lines = src.split("\n")
    to_inject = []
    for item in cls.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name not in TARGET_SYMBOLS:
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
    ast.parse(new_src, filename=ML_MOD)
    p.write_text(new_src)
    return len(to_inject)


# ============ remove from _core ============

def remove_from_core(symbol_ranges):
    print("\n--- remove from _core.py ---")
    p = Path(CORE)
    lines = p.read_text().split("\n")
    ranges = [(info["start"], info["end"])
              for info in symbol_ranges.values()]
    ranges.sort(reverse=True)
    for start, end in ranges:
        del lines[start - 1:end]
    new_src = "\n".join(lines)
    Path(CORE).write_text(new_src)
    ast.parse(new_src, filename=CORE)
    print(f"  _core.py parses OK ({len(new_src.splitlines())} lines)")


# ============ whitelist rewrite ============

def rewrite_whitelist():
    print("\n--- whitelist rewrite (§13.4/318/324) ---")
    old_core = "cadw/dataframe/_core.py"
    new_mod = ML_MOD
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
    if head != "v0.9.26x":
        raise SystemExit(f"expected HEAD=v0.9.26x, got {head!r}")
    print("HEAD: " + head)

    for p in (CORE, ML_MOD):
        if not Path(p).exists():
            raise SystemExit(p + " missing")
    print(f"  {CORE} + {ML_MOD}: present")

    ml_src = Path(ML_MOD).read_text()
    if f"class {CLASS_NAME}" not in ml_src:
        raise SystemExit(f"{CLASS_NAME} not in {ML_MOD}")
    print(f"  {CLASS_NAME}: present")


def run_tests():
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26x2: append 3 fit_* to _ml_bindings ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()

    # Discover in _core
    print("\n--- discover in _core ---")
    core_src = Path(CORE).read_text()
    core_tree = ast.parse(core_src)
    core_cls = find_class(core_tree, "CadwDataFrame")
    symbol_ranges = discover_symbols(core_cls)
    missing = set(TARGET_SYMBOLS) - set(symbol_ranges.keys())
    if missing:
        raise SystemExit(f"missing symbols in _core: {missing}")
    total_loc = sum(info["loc"] for info in symbol_ranges.values())
    print(f"  found {len(symbol_ranges)} symbols, {total_loc} LOC:")
    for name in TARGET_SYMBOLS:
        info = symbol_ranges[name]
        print(f"    {name}: {info['start']}-{info['end']} "
              f"({info['loc']} LOC)")

    # Extract sources BEFORE removing from _core
    print("\n--- extract sources ---")
    core_lines = core_src.split("\n")
    sources = {}
    for name, info in symbol_ranges.items():
        sources[name] = "\n".join(
            core_lines[info["start"] - 1:info["end"]]
        )
        print(f"  {name}: {len(sources[name].splitlines())} lines")

    # Discovery: classify undefined names
    print("\n--- classify imports ---")
    core_import_map = build_import_map(core_tree)
    core_top = top_level_names(core_tree)
    ml_tree = ast.parse(Path(ML_MOD).read_text())
    ml_scope = collect_module_scope_names(ml_tree)

    all_undef = set()
    for info in symbol_ranges.values():
        all_undef.update(info["undefined"])
    if all_undef:
        print(f"  all undefined: {sorted(all_undef)}")

    top_from, top_import, top_import_as, lazy, already = resolve_append(
        all_undef, core_import_map, core_top, ml_scope,
    )
    print(f"  already in _ml_bindings scope: {sorted(already)}")
    print(f"  new imports needed: {len(top_from) + len(top_import) + len(top_import_as)}")
    for f in top_from:
        dots = "." * f["level"]
        asn = f" as {f['asname']}" if f["asname"] else ""
        print(f"    from {dots}{f['module'] or ''} "
              f"import {f['name']}{asn}")
    for m in top_import:
        print(f"    import {m}")
    for m, asn in top_import_as:
        print(f"    import {m} as {asn}")
    print(f"  lazy (from ._core): {sorted(lazy)}")

    # 1. Append imports (missing only)
    import_lines = build_import_lines(top_from, top_import, top_import_as)
    append_imports(import_lines)

    # 2. Append methods
    append_methods(sources)

    # 3. Lazy injection
    post_write_lazy_injection(lazy)

    # 4. Remove from _core
    remove_from_core(symbol_ranges)

    # 5. Whitelist rewrite
    n_wl = rewrite_whitelist()
    print(f"\n  summary: {n_wl} whitelist key(s)")

    # Sanity
    print("\n--- sanity: hasattr only ---")
    sanity_script = (
        "import cadw\n"
        "C = cadw.CadwDataFrame\n"
        "names = ['fit_naive_bayes','fit_lasso','fit_elasticnet']\n"
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
        "v0.9.26x2: append fit_naive_bayes/fit_lasso/fit_elasticnet "
        "to _ml_bindings (B.1 step 19). 3 methods (~75 LOC) "
        "missed from v0.9.26v TARGET_SYMBOLS; appended verbatim "
        f"to existing {CLASS_NAME} via §13.4/329 append pattern "
        "(new imports only; idempotent). _core.py 633 → ~558 LOC. "
        "0 logic change; tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26x2 COMPLETE")


if __name__ == "__main__":
    main()