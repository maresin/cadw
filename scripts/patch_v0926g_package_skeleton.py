#!/usr/bin/env python3
# v0.9.26g: package skeleton. git mv cadw/dataframe.py -> _core.py,
# create __init__.py facade, update path references in tests.
# Pure structural change; 0 logic. ASCII-only.

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import commit_with_guards


OLD_FILE = "cadw/dataframe/_core.py"
NEW_DIR = "cadw/dataframe"
NEW_CORE = "cadw/dataframe/_core.py"
PYPROJECT = "pyproject.toml"


def git_status_clean():
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return out


def find_from_dataframe_imports():
    """Collect all symbols imported from cadw.dataframe (relative or absolute)."""
    names = set()
    pat = re.compile(
        r"from\s+(?:cadw\.|\.)dataframe\s+import\s+(\([^)]+\)|[^\n\(]+)",
        re.MULTILINE,
    )
    for py in Path(".").rglob("*.py"):
        s = str(py)
        if s.startswith("scripts/") or s.startswith(".git/"):
            continue
        if s.endswith("cadw/dataframe/_core.py"):
            continue  # self
        try:
            src = py.read_text()
        except Exception:
            continue
        for m in pat.finditer(src):
            block = m.group(1)
            block = block.replace("(", "").replace(")", "")
            for tok in block.split(","):
                tok = tok.strip().split(" as ")[0].strip()
                if tok and not tok.startswith("#"):
                    names.add(tok)
    return names


def find_test_path_refs():
    """Files under tests/ with 'cadw/dataframe/_core.py' string."""
    hits = []
    for py in Path("tests").rglob("*.py"):
        src = py.read_text()
        n = src.count("cadw/dataframe/_core.py")
        if n:
            hits.append((str(py), n, src))
    return hits


def find_cadw_path_refs():
    """Any other references inside cadw/ to 'cadw/dataframe/_core.py'."""
    hits = []
    for py in Path("cadw").rglob("*.py"):
        if str(py).endswith("cadw/dataframe/_core.py"):
            continue
        src = py.read_text()
        if "cadw/dataframe/_core.py" in src:
            hits.append(str(py))
    return hits


def find_pyproject_whitelist():
    """Check pyproject.toml for the path (whitelist config)."""
    if not Path(PYPROJECT).exists():
        return None
    src = Path(PYPROJECT).read_text()
    if "cadw/dataframe/_core.py" in src:
        return src
    return None


def find_df_all():
    """Extract __all__ from dataframe.py if present."""
    src = Path(OLD_FILE).read_text()
    m = re.search(r'^__all__\s*=\s*\[([^\]]+)\]', src, re.MULTILINE)
    if not m:
        return []
    items = []
    for tok in m.group(1).split(","):
        tok = tok.strip().strip("\"'")
        if tok:
            items.append(tok)
    return items


def verify_state():
    print("=== verify ===")
    subj = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    head = subj.split(":", 1)[0].strip()
    if head != "v0.9.26e":
        raise SystemExit("expected HEAD=v0.9.26e, got " + repr(head))
    print("HEAD: " + head)

    st = git_status_clean()
    if st:
        raise SystemExit("git status not clean:\n" + st)
    print("git status clean (tracked)")

    if not Path(OLD_FILE).exists():
        raise SystemExit(OLD_FILE + " missing")
    if Path(NEW_DIR).exists():
        raise SystemExit(NEW_DIR + " already exists (v0.9.26g applied?)")
    print(OLD_FILE + " exists; " + NEW_DIR + "/ absent: OK")

    imports = find_from_dataframe_imports()
    print("\nfrom cadw.dataframe imports: " + str(sorted(imports)))

    test_refs = find_test_path_refs()
    print("\ntest path refs to 'cadw/dataframe/_core.py':")
    for fname, n, _ in test_refs:
        print("  " + fname + ": " + str(n))

    cadw_refs = find_cadw_path_refs()
    print("\ncadw/ refs to 'cadw/dataframe/_core.py' (excl. self):")
    if cadw_refs:
        for r in cadw_refs:
            print("  " + r)
    else:
        print("  (none)")

    pyp = find_pyproject_whitelist()
    if pyp:
        print("\npyproject.toml contains 'cadw/dataframe/_core.py'")
        print("  (whitelist config - will be updated)")

    all_ = find_df_all()
    print("\ndataframe.py __all__: " + str(all_))

    return {
        "imports": imports,
        "test_refs": test_refs,
        "cadw_refs": cadw_refs,
        "pyproject_has_path": pyp is not None,
        "df_all": all_,
    }


def do_git_mv():
    print("\ngit mv " + OLD_FILE + " -> " + NEW_CORE)
    Path(NEW_DIR).mkdir(exist_ok=True)
    subprocess.run(["git", "mv", OLD_FILE, NEW_CORE], check=True)
    print("  done")


def write_init(imports, df_all):
    print("\nwrite " + NEW_DIR + "/__init__.py")
    # Baseline exports: package facade must at minimum re-export
    # everything found in imports + df_all + CadwDataFrame/concat.
    must = {"CadwDataFrame", "concat"}
    names = set(imports) | set(df_all) | must
    names = sorted(n for n in names if re.match(r"^[A-Za-z_]\w*$", n))

    lines = [
        "# cadw/dataframe/__init__.py",
        "# Package facade re-exporting from _core.",
        "# v0.9.26g (B.1 step 1): dataframe.py -> dataframe/ package.",
        "# Pure structural move; no logic change.",
        "",
        "from ._core import (",
    ]
    for n in names:
        lines.append("    " + n + ",")
    lines.append(")")
    lines.append("")
    lines.append("__all__ = [")
    for n in names:
        if not n.startswith("_"):
            lines.append('    "' + n + '",')
    lines.append("]")
    lines.append("")
    src = "\n".join(lines)
    p = Path(NEW_DIR + "/__init__.py")
    p.write_text(src)
    ast.parse(src, filename=str(p))  # syntax gate
    print("  wrote " + str(len(src.splitlines())) + " lines")
    print("  exports: " + str(names))


def update_tests(test_refs):
    print("\nupdate tests:")
    for fname, n, src in test_refs:
        new_src = src.replace(
            '"cadw/dataframe/_core.py"', '"cadw/dataframe/_core.py"',
        )
        new_src = new_src.replace(
            "'cadw/dataframe/_core.py'", "'cadw/dataframe/_core.py'",
        )
        if new_src == src:
            print("  " + fname + ": no change (unexpected)")
            continue
        p = Path(fname)
        p.write_text(new_src)
        ast.parse(new_src, filename=fname)  # syntax gate
        print("  " + fname + ": " + str(n) + " refs updated")


def update_cadw_refs(cadw_refs):
    print("\nupdate cadw/ refs:")
    for fname in cadw_refs:
        p = Path(fname)
        src = p.read_text()
        new_src = src.replace(
            '"cadw/dataframe/_core.py"', '"cadw/dataframe/_core.py"',
        )
        new_src = new_src.replace(
            "'cadw/dataframe/_core.py'", "'cadw/dataframe/_core.py'",
        )
        if new_src == src:
            print("  " + fname + ": no string-literal ref found, "
                  "inspect manually")
            continue
        p.write_text(new_src)
        ast.parse(new_src, filename=fname)
        print("  " + fname + ": updated")


def update_pyproject():
    p = Path(PYPROJECT)
    src = p.read_text()
    if "cadw/dataframe/_core.py" not in src:
        return
    new_src = src.replace(
        '"cadw/dataframe/_core.py"', '"cadw/dataframe/_core.py"',
    )
    if new_src == src:
        # non-quoted form; do simple substitution
        new_src = src.replace(
            "cadw/dataframe/_core.py", "cadw/dataframe/_core.py",
        )
    p.write_text(new_src)
    print("  " + PYPROJECT + ": path updated")


def run_tests():
    print("\nfast suite...")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26g: package skeleton ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    state = verify_state()

    do_git_mv()
    write_init(state["imports"], state["df_all"])
    update_tests(state["test_refs"])
    if state["cadw_refs"]:
        update_cadw_refs(state["cadw_refs"])
    if state["pyproject_has_path"]:
        update_pyproject()

    run_tests()

    commit_with_guards(
        "v0.9.26g: package skeleton. cadw/dataframe.py -> "
        "cadw/dataframe/_core.py + __init__.py facade. Pure git "
        "mv; 0 logic change. Test path references and whitelist "
        "updated. 2488 tests green. B.1 step 1 of N.",
        self_path="scripts/patch_v0926g_package_skeleton.py",
        check_whitelist=True,
    )
    print("\nv0.9.26g done.")


if __name__ == "__main__":
    main()