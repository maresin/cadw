#!/usr/bin/env python3
"""v0.7.8-fix2: import line в cadw/__init__.py (__all__ уже есть, import — нет).

Причина v0.7.8-fix: `has_import = "GBDTMulticlassClassifier" in s` — 
слишком грубая проверка. __all__ уже содержит "GBDTMulticlassClassifier" 
(одна строка replace сработала в v0.7.8), но import-line replace провалился 
(pattern `\\n)` не найден — после `GBDTClassifier,` идёт `GaussianNB,`).

Fix: проверять **точную строку** в import-блоке и в __all__ отдельно.
Используем AST (источник правды для «что импортируется») + line-based 
insert (защита от дублирования).
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
    )
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:25]:
            print(f"    {line}")
    if check and r.returncode != 0:
        print(f"    STDERR: {r.stderr.strip()[-1500:]}")
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         f"import py_compile; py_compile.compile('{path}', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"COMPILE FAIL: {path}")
        print(r.stderr[-800:])
        sys.exit(1)
    print(f"  compile OK: {path}")


# ============ AST diagnostic ============

def diagnose(path: Path):
    """Показать import names (из target-блока) и __all__ names."""
    print(f"\n=== Diagnostic: {path} ===")
    s = path.read_text()
    tree = ast.parse(s)

    # __all__
    all_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant):
                                all_names.add(elt.value)

    print(f"  __all__ count: {len(all_names)}")
    print(f"  GBDTMulticlassClassifier in __all__: "
          f"{'GBDTMulticlassClassifier' in all_names}")
    print(f"  fit_gbdt_multiclass in __all__: "
          f"{'fit_gbdt_multiclass' in all_names}")

    # Imports (from target module)
    mod_name = ".ml" if "cadw/__init__.py" in str(path).replace("\\", "/") else ".gbdt"
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == mod_name.lstrip("."):
            for alias in node.names:
                imported.add(alias.name)
    print(f"  from {mod_name} import: {len(imported)} names")
    print(f"  GBDTMulticlassClassifier imported: "
          f"{'GBDTMulticlassClassifier' in imported}")
    print(f"  fit_gbdt_multiclass imported: "
          f"{'fit_gbdt_multiclass' in imported}")


# ============ Line-based insert ============

def has_exact_line(s: str, target: str) -> bool:
    target_norm = target.strip()
    return any(line.strip() == target_norm for line in s.split("\n"))


def insert_after_exact_line(s: str, anchor: str, new_line: str,
                            *, path: str) -> str:
    """Insert new_line immediately after first line == anchor (stripped).

    Fail if anchor not found (silent no-op protection).
    """
    anchor_norm = anchor.strip()
    lines = s.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == anchor_norm:
            lines.insert(i + 1, new_line)
            return "\n".join(lines)
    raise SystemExit(
        f"[{path}] anchor not found (exact line match): {anchor!r}"
    )


def fix_file(
    path: Path,
    *,
    import_anchor: str,
    import_line: str,
    all_anchor: str,
    all_line: str,
) -> bool:
    """Ensure import_line and all_line are present; add if missing."""
    print(f"\n=== Fix: {path} ===")
    s = path.read_text()
    changed = False

    if not has_exact_line(s, import_line):
        s = insert_after_exact_line(s, import_anchor, import_line,
                                    path=str(path))
        changed = True
        print(f"  OK: inserted import line after {import_anchor.strip()!r}")
    else:
        print(f"  import line already present")

    if not has_exact_line(s, all_line):
        s = insert_after_exact_line(s, all_anchor, all_line,
                                    path=str(path))
        changed = True
        print(f"  OK: inserted __all__ line after {all_anchor.strip()!r}")
    else:
        print(f"  __all__ line already present")

    if changed:
        path.write_text(s)
        compile_check(path)
    else:
        print(f"  (no changes)")

    return changed


def verify_runtime():
    print("\n=== Runtime import verify ===")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw import fit_gbdt_multiclass, GBDTMulticlassClassifier; "
         "from cadw.ml import fit_gbdt_multiclass as f2; "
         "print('OK: cadw.fit_gbdt_multiclass importable')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  FAIL:\n{r.stderr[-1500:]}")
        sys.exit(1)
    print(f"  {r.stdout.strip()}")


def main():
    print("=== v0.7.8-fix2 ===")

    ml_init = Path("cadw/ml/__init__.py")
    cadw_init = Path("cadw/__init__.py")

    diagnose(ml_init)
    diagnose(cadw_init)

    # cadw/ml/__init__.py — likely already OK (from v0.7.8)
    fix_file(
        ml_init,
        import_anchor="    GBDTClassifier, fit_gbdt_classifier,",
        import_line="    GBDTMulticlassClassifier, fit_gbdt_multiclass,",
        all_anchor='    "GBDTClassifier", "fit_gbdt_classifier",',
        all_line='    "GBDTMulticlassClassifier", "fit_gbdt_multiclass",',
    )

    # cadw/__init__.py — import missing (v0.7.8 replaced only __all__)
    fix_file(
        cadw_init,
        import_anchor="    GBDTClassifier, fit_gbdt_classifier,",
        import_line="    GBDTMulticlassClassifier, fit_gbdt_multiclass,",
        all_anchor='    "GBDTClassifier", "fit_gbdt_classifier",',
        all_line='    "GBDTMulticlassClassifier", "fit_gbdt_multiclass",',
    )

    verify_runtime()

    print("\n=== pytest tests/test_gbdt_multiclass.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_gbdt_multiclass.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.8-fix2 (test red)"], check=False)
        sys.exit(1)

    print("\n=== regression: binary GBDT ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_gbdt_regression.py",
         "tests/test_gbdt_classification.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.8-fix2 (regression)"], check=False)
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.8-fix2: import line в cadw/__init__.py "
        "(GBDTMulticlassClassifier, fit_gbdt_multiclass)"])

    print("\n==> v0.7.8-fix2 complete.")


if __name__ == "__main__":
    main()