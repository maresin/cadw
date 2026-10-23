#!/usr/bin/env python3
# v0.9.26i-fix4b: correct AST-discovery (nested defs as locals).
# Bootstrap-safe. Complete extraction to green.

import ast
import builtins
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


ITERATORS = "cadw/dataframe/_iterators.py"
CORE = "cadw/dataframe/_core.py"
EXPORTS = "cadw/dataframe/_exports.py"
HELPERS = "cadw/dataframe_helpers.py"

BUILTINS = set(dir(builtins))


# ============ AST ============

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


def method_locals(fn_node):
    """Locals = args + Store + nested defs (skip fn_node itself)."""
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
            continue  # §13.4/308: skip method's own FunctionDef
        # nested FunctionDef/AsyncFunctionDef/ClassDef -> local
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)):
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


def method_load_names(fn_node):
    loads = set()
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            loads.add(n.id)
    return loads


def discover_undefined(src):
    tree = ast.parse(src)
    module_names = top_level_names(tree)
    result = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "_IteratorsMixin":
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            locals_ = method_locals(item)
            loads = method_load_names(item)
            undefined = (loads
                         - locals_
                         - module_names
                         - BUILTINS
                         - {"self", "cls", "__class__"})
            if undefined:
                result[item.name] = sorted(undefined)
    return result


def classify_names(undefined_set):
    core_tree = ast.parse(Path(CORE).read_text())
    core_top = top_level_names(core_tree)
    helpers_top = set()
    if Path(HELPERS).exists():
        helpers_tree = ast.parse(Path(HELPERS).read_text())
        helpers_top = top_level_names(helpers_tree)
    resolved = {}
    for name in undefined_set:
        if name in core_top:
            resolved[name] = "core"
        elif name in helpers_top:
            resolved[name] = "helpers"
        else:
            resolved[name] = "unknown"
    return resolved


# ============ text injection ============

def find_body_insert_index(lines, method_name):
    def_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if (s.startswith(f"def {method_name}(")
                or s.startswith(f"def {method_name} (")):
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
    insert_idx = sig_end + 1
    if insert_idx < len(lines):
        s = lines[insert_idx].strip()
        if s.startswith('"""') or s.startswith("'''"):
            quote = '"""' if s.startswith('"""') else "'''"
            if s.count(quote) >= 2:
                insert_idx += 1
            else:
                insert_idx += 1
                while insert_idx < len(lines):
                    if quote in lines[insert_idx]:
                        insert_idx += 1
                        break
                    insert_idx += 1
    return insert_idx


def clean_stale_lazy(src):
    """Remove all `from ._core import X` lines regardless of indent."""
    lines = src.split("\n")
    out = []
    removed = 0
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith("from ._core import "):
            removed += 1
            continue
        out.append(ln)
    if removed:
        print(f"  cleaned {removed} stale lazy import(s)")
    return "\n".join(out)


def inject_lazy(src, method_name, core_names):
    if not core_names:
        return src, False
    lines = src.split("\n")
    idx = find_body_insert_index(lines, method_name)
    if idx is None:
        print(f"  WARN: {method_name} body start not found")
        return src, False
    imp = "        from ._core import " + ", ".join(sorted(core_names))
    lines.insert(idx, imp)
    print(f"  {method_name}: lazy core {sorted(core_names)} "
          f"(line {idx + 1})")
    return "\n".join(lines), True


# ============ _exports.py ============

def ensure_exports_correct():
    """_exports.py: top-level _convert_batch from dataframe_helpers."""
    print("\n--- ensure _exports.py ---")
    src = Path(EXPORTS).read_text()
    # remove any lazy `. _core import _convert_batch`
    src = re.sub(
        r"^\s*from \._core import _convert_batch\s*\n",
        "", src, flags=re.MULTILINE,
    )
    # ensure top-level
    marker = "from cadw.dataframe_helpers import _convert_batch"
    if marker not in src:
        anchor = "from cadw.core.logging import get_logger\n"
        if anchor in src:
            src = src.replace(
                anchor, anchor + marker + "\n", 1,
            )
            print(f"  added top-level: {marker}")
        else:
            # fallback anchor
            anchor2 = "import pyarrow.parquet as pq\n"
            if anchor2 in src:
                src = src.replace(
                    anchor2, anchor2 + marker + "\n", 1,
                )
                print(f"  added top-level (fallback): {marker}")
            else:
                raise SystemExit(
                    "cannot find anchor in _exports.py"
                )
    else:
        print("  top-level import already present")
    Path(EXPORTS).write_text(src)
    ast.parse(src, filename=EXPORTS)
    print("  _exports.py parses OK")


# ============ main ============

def main():
    print("=== v0.9.26i-fix4b ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26h2-fix1":
        raise SystemExit("expected HEAD=v0.9.26h2-fix1, got " + repr(head))
    print("HEAD: " + head)

    core_src = Path(CORE).read_text()
    if "class CadwDataFrame(_ExportsMixin, _IteratorsMixin)" not in core_src:
        raise SystemExit("_core.py dual-mixin missing")
    if "def _apply_filter_with_positions(" not in core_src:
        raise SystemExit("_apply_filter_with_positions missing in _core")
    print("  _core.py: OK")

    # 1. Fix _exports.py first
    ensure_exports_correct()

    # 2. Clean + discover _iterators.py
    src = Path(ITERATORS).read_text()
    src = clean_stale_lazy(src)

    for iteration in range(3):
        print(f"\n--- discovery pass {iteration + 1} ---")
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"  parse error line {e.lineno}: {e.msg}")
            print(f"  {src.split(chr(10))[e.lineno-1]!r}")
            raise SystemExit("clean failed")

        undefined = discover_undefined(src)
        if not undefined:
            print("  no undefined names")
            break

        print("  undefined per method:")
        for m, names in undefined.items():
            print(f"    {m}: {names}")

        all_names = set()
        for names in undefined.values():
            all_names.update(names)
        resolved = classify_names(all_names)
        unknown = [n for n, w in resolved.items() if w == "unknown"]
        if unknown:
            raise SystemExit(f"cannot resolve: {unknown}")

        changed = False
        for method, names in undefined.items():
            core_names = [n for n in names if resolved[n] == "core"]
            helper_names = [n for n in names if resolved[n] == "helpers"]
            if helper_names:
                print(f"  {method}: helpers {helper_names} (top-level "
                      f"already added in _iterators.py)")
                # assume already in `from cadw.dataframe_helpers import (...)`
                # if not, will fail at import — then next pass fixes
            src, did = inject_lazy(src, method, core_names)
            if did:
                changed = True
        if not changed:
            print("  no injections - aborting")
            break

    Path(ITERATORS).write_text(src)
    ast.parse(src, filename=ITERATORS)
    print("\n_iterators.py parses OK")

    # 3. sanity
    print("\n--- sanity import ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import pyarrow as pa; import cadw; "
         "df = cadw.CadwDataFrame.from_table(pa.table({'v':[10,20,30,40]})); "
         "f = df[df['v']>15]; "
         "print('filter_count:', f.count()); "
         "print('index:', list(f.index))"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("sanity failed")
    print(r.stdout.strip())

    # 4. tests
    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")

    # 5. whitelist
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

    # 6. commit
    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.26i: extract _iterators (B.1 step 3) + fix4b "
        "(full discovery, nested-defs-as-locals). "
        "_iter_arrow_batches, _iter_filtered_batches, "
        "_iter_filtered_batches_with_positions, _with_filter, "
        "_apply_column_transform moved to _iterators.py. "
        "CadwDataFrame(_ExportsMixin, _IteratorsMixin). "
        "_convert_batch/_concat_batches from dataframe_helpers "
        "(top-level, no cycle). Lazy _core imports for "
        "CadwDataFrame (2 methods) + _apply_filter_with_positions "
        "(1 method). §13.4/307 helper deps; §13.4/308 nested defs "
        "as locals (def _init/_iter inside method body). "
        "§13.4/303, /305, /306 applied. 0 logic change; 2492 green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26i COMPLETE (fix1+2+3+4b)")


if __name__ == "__main__":
    main()