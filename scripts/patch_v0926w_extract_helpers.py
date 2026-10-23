#!/usr/bin/env python3
# v0.9.26w: extract _helpers (B.1 step 17, top-level helpers).
# 5 functions, 142 LOC -> _helpers.py, re-exported from _core.py.
# Update 1 affected test. Bootstrap-safe.

import ast
import builtins
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

CORE = "cadw/dataframe/_core.py"
NEW_MOD = "cadw/dataframe/_helpers.py"
BUILTINS = set(dir(builtins))

TARGET_HELPERS = [
    "_apply_filter_with_positions",
    "_detect_mode",
    "_normalize_scalar",
    "_infer_arrow_type_from_value",
    "_build_out_schema",
]

TEST_PATH = "tests/test_filter_compat.py"


# ============ AST ============

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


def fn_locals(fn_node):
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
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
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


def fn_loads(fn_node):
    loads = set()
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            loads.add(n.id)
    return loads


def discover_helpers(core_tree):
    result = {}
    for item in core_tree.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name not in TARGET_HELPERS:
            continue
        start = item.lineno
        if item.decorator_list:
            start = min(d.lineno for d in item.decorator_list)
        end = max(
            getattr(n, "end_lineno", item.lineno)
            for n in ast.walk(item)
        )
        local = fn_locals(item)
        loads = fn_loads(item)
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
    """Classify: top-level imports vs lazy from ._core."""
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


# ============ extraction ============

def extract_helpers(core_lines, ranges):
    """Extract top-level helper sources verbatim."""
    result = {}
    for name, info in ranges.items():
        start, end = info["start"], info["end"]
        result[name] = "\n".join(core_lines[start - 1:end])
    return result


def write_helpers_module(sources, top_import_lines):
    """Build _helpers.py with 5 functions."""
    header = [
        "# cadw/dataframe/_helpers.py",
        "# v0.9.26w (B.1 step 17): module-level helpers.",
        "# Symbols: _apply_filter_with_positions, _detect_mode,",
        "#   _normalize_scalar, _infer_arrow_type_from_value,",
        "#   _build_out_schema.",
        "# Extracted verbatim from _core.py; re-exported by _core.",
        "",
        "from __future__ import annotations",
        "",
    ]
    header.extend(top_import_lines)
    header.extend(["", ""])
    body_parts = []
    for name in TARGET_HELPERS:
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


def remove_from_core(core_lines, ranges):
    rr = [(info["start"], info["end"]) for info in ranges.values()]
    rr.sort(reverse=True)
    for start, end in rr:
        del core_lines[start - 1:end]


# ============ lazy injection into helpers ============

def find_body_start(lines, fn_name):
    def_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if (s.startswith(f"def {fn_name}(")
                or s.startswith(f"def {fn_name} (")):
            def_idx = i
            break
    if def_idx is None:
        return None
    # Top-level function: sig starts at indent 0
    if len(lines[def_idx]) != len(lines[def_idx].lstrip()):
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


def inject_lazy_in_helpers(lazy_names):
    """Helpers use lazy `from ._core import X` inside function bodies."""
    if not lazy_names:
        return 0
    print(f"\n--- inject lazy in _helpers.py ---")
    print(f"  lazy pool: {sorted(lazy_names)}")
    p = Path(NEW_MOD)
    src = p.read_text()
    tree = ast.parse(src)
    lines = src.split("\n")
    to_inject = []
    lazy_set = set(lazy_names)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        loads = fn_loads(node)
        needed = sorted(loads & lazy_set)
        if not needed:
            continue
        insert_idx = find_body_start(lines, node.name)
        if insert_idx is None:
            print(f"  {node.name}: body-start not found")
            continue
        to_inject.append((insert_idx, node.name, needed))
    if not to_inject:
        print("  no injection needed")
        return 0
    to_inject.sort(reverse=True)
    for insert_idx, name, needed in to_inject:
        imp = "    from ._core import " + ", ".join(needed)
        lines.insert(insert_idx, imp)
        print(f"  {name}: inject {needed} at line {insert_idx + 1}")
    new_src = "\n".join(lines)
    ast.parse(new_src, filename=NEW_MOD)
    p.write_text(new_src)
    return len(to_inject)


# ============ _core.py re-export ============

def re_export_in_core():
    """Add `from ._helpers import (...)` after `from ._ml_bindings ...`."""
    print("\n--- re-export in _core.py ---")
    src = Path(CORE).read_text()
    marker = "from ._helpers import ("
    if marker in src:
        print("  re-export already present")
        return
    imports_block = (
        "from ._helpers import (\n"
        "    _apply_filter_with_positions,\n"
        "    _detect_mode,\n"
        "    _normalize_scalar,\n"
        "    _infer_arrow_type_from_value,\n"
        "    _build_out_schema,\n"
        ")\n"
    )
    anchor = "from ._ml_bindings import _MLBindingsMixin"
    if anchor in src:
        src = src.replace(
            anchor,
            anchor + "\n" + imports_block,
            1,
        )
    else:
        anchor2 = "class CadwDataFrame("
        src = src.replace(anchor2, imports_block + "\n" + anchor2, 1)
    Path(CORE).write_text(src)
    ast.parse(src, filename=CORE)
    print("  re-export added")


# ============ test update (§13.4/325) ============

def fix_test():
    """test_filter_compat: grep _helpers.py instead of _core.py."""
    print("\n--- fix test_filter_compat.py ---")
    p = Path(TEST_PATH)
    src = p.read_text()

    old = "from cadw.dataframe import _core as df_mod"
    new = "from cadw.dataframe import _helpers as df_mod"

    if new in src:
        print("  already fixed")
        return 0
    if old not in src:
        # Maybe already targeting package
        print(f"  WARN: '{old}' not found — inspect manually")
        return 0

    src = src.replace(old, new, 1)
    p.write_text(src)
    ast.parse(src, filename=TEST_PATH)
    print(f"  {old!r} -> {new!r}")
    return 1


# ============ verify + tests + commit ============

def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26v":
        raise SystemExit(f"expected HEAD=v0.9.26v, got {head!r}")
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
    print("=== v0.9.26w: extract _helpers ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()

    # Discover
    print("\n--- discover helpers ---")
    core_src = Path(CORE).read_text()
    core_tree = ast.parse(core_src)
    helper_ranges = discover_helpers(core_tree)
    found = set(helper_ranges.keys())
    missing = set(TARGET_HELPERS) - found
    if missing:
        raise SystemExit(f"missing helpers: {missing}")
    total_loc = sum(info["loc"] for info in helper_ranges.values())
    print(f"  found {len(helper_ranges)} helpers, {total_loc} LOC:")
    for name in TARGET_HELPERS:
        info = helper_ranges[name]
        print(f"    {name}: {info['start']}-{info['end']} "
              f"({info['loc']} LOC)")

    # Discovery undefined
    print("\n--- AST-discovery ---")
    core_imports = build_import_map(core_tree)
    core_top = top_level_names(core_tree)
    all_undef = set()
    for info in helper_ranges.values():
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
    print(f"  lazy (inside helper body): {sorted(lazy)}")

    # Extract
    print("\n--- extract sources ---")
    core_lines = core_src.split("\n")
    sources = extract_helpers(core_lines, helper_ranges)
    for name in TARGET_HELPERS:
        if name in sources:
            print(f"  {name}: {len(sources[name].splitlines())} lines")

    # Write module
    print("\n--- write _helpers.py ---")
    top_import_lines = build_top_import_lines(
        top_from, top_import, top_import_as,
    )
    write_helpers_module(sources, top_import_lines)
    print(f"  imports: {top_import_lines}")

    # Lazy inject
    n_lazy = inject_lazy_in_helpers(lazy)

    # Remove from core
    print("\n--- remove from _core.py ---")
    remove_from_core(core_lines, helper_ranges)
    new_core = "\n".join(core_lines)
    Path(CORE).write_text(new_core)
    ast.parse(new_core, filename=CORE)
    print(f"  _core.py parses OK ({len(new_core.splitlines())} lines)")

    # Re-export in core
    re_export_in_core()

    # Fix test (§13.4/325)
    n_test = fix_test()

    print(f"\n  summary: {n_lazy} lazy injection(s), "
          f"{n_test} test fix")

    # Sanity
    print("\n--- sanity ---")
    sanity_script = (
        "import cadw\n"
        "C = cadw.CadwDataFrame\n"
        "import cadw.dataframe._helpers as h\n"
        "for n in ('_apply_filter_with_positions','_detect_mode',\n"
        "         '_normalize_scalar','_infer_arrow_type_from_value',\n"
        "         '_build_out_schema'):\n"
        "    assert hasattr(h, n), f'_helpers.{n} missing'\n"
        "for n in ('_apply_filter_with_positions','_detect_mode',\n"
        "         '_normalize_scalar','_infer_arrow_type_from_value',\n"
        "         '_build_out_schema'):\n"
        "    assert hasattr(cadw.dataframe._core, n), f'_core.{n} re-export missing'\n"
        "print('helpers + re-export: OK')\n"
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

    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.26w: extract _helpers (B.1 step 17). "
        "_apply_filter_with_positions/_detect_mode/_normalize_scalar/"
        "_infer_arrow_type_from_value/_build_out_schema (~142 LOC, "
        "5 helpers) moved verbatim from _core.py to _helpers.py. "
        "_core.py re-exports (§13.4/325 — lazy imports in "
        "_iterators/_per_group continue via _core). "
        "fix test_filter_compat target: _core → _helpers. "
        "_core.py ~853 → ~626 LOC. 0 logic change; tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26w COMPLETE")


if __name__ == "__main__":
    main()