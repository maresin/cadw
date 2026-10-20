#!/usr/bin/env python3
"""v0.9.1c-fix1: from_json error message clarity (non-existent path).

BUG (1 test): `Pipeline.from_json(tmp_path / "nope")` → error message
"from_json: /tmp/.../nope not found". Test expected "pipeline.json".
Root: ambiguous handling of non-existent path — treated as file, but
user passes directory-like name.

FIX (code-side, not test): if path doesn't exist, assume directory and
show expected `path/pipeline.json` in error. Clearer message.

Behavior matrix:
  - path.is_dir()                        → meta_file = path/pipeline.json
  - path.exists() and path.is_file()    → meta_file = path
  - not path.exists()                   → meta_file = path/pipeline.json
                                          (assume directory, show in msg)

Class (§13.4/150): error message clarity for ambiguous inputs. When a
path can be either file or directory, don't silently pick one — make
the error say what was expected.

GUARDS (before commit): save_load tests + pipeline regression + guards
+ fast + slow.

Run: uv run python scripts/patch_v091c_fix1.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
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


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if not starts:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


def fix_from_json_path_handling():
    print("\n=== cadw/pipeline.py::from_json — non-existent path clarity ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    old = '''        path = Path(path)
        if path.is_dir():
            meta_file = path / "pipeline.json"
        else:
            meta_file = path
        if not meta_file.exists():
            raise FileNotFoundError(
                f"from_json: {meta_file} not found"
            )'''

    new = '''        path = Path(path)
        # v0.9.1c-fix1: unambiguous handling. If path doesn't exist,
        # assume it's a directory (user typically passes "mypipe", not
        # "mypipe/pipeline.json"), and show expected file in error.
        if path.is_dir():
            meta_file = path / "pipeline.json"
        elif path.exists():
            meta_file = path  # file
        else:
            # Non-existent path: assume directory for clearer error
            meta_file = path / "pipeline.json"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"from_json: {meta_file} not found "
                f"(path can be a pipeline directory or the "
                f"pipeline.json file inside it)"
            )'''

    s = replace_line_block(s, old, new, tag="from-json-path")
    p.write_text(s)
    compile_check(p)
    print("  OK")


def append_decisions_150():
    print("\n=== DECISIONS.md: §13.4/150 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 150" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/150 (error message clarity).** Ambiguous path "
        s += "(file vs directory) — make error say what was expected.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 150 (error message clarity for ambiguous inputs).**
`Pipeline.from_json(path)` принимает file или directory. Для несуществующего
path `"nope"` старая логика: `if path.is_dir(): ... else: meta_file =
path` → treated as file → error `"/tmp/.../nope not found"`. Test
ожидал `match="pipeline.json"`. 1 fail.

**Root cause:** ambiguous input handling молча выбирает интерпретацию
(«file») без подсказки, что искали. Для пользователя `from_json("mypipe")`
где `mypipe` — directory, сообщение не помогает — он видит, что
`"mypipe"` не найден, но не понимает, что внутри ищется `pipeline.json`.

**Fix (code-side):** для несуществующего path — **assume directory**,
показать `path/pipeline.json` в error. Явно указать: «path can be a
pipeline directory or the pipeline.json file inside it».

**Урок:** для API с **неоднозначным input** (file-or-directory, scalar-or-
list, str-or-Path) — не «resolve silent», а либо **явный выбор с clear
error**, либо **best-guess + helpful message**. `Path.is_dir()` на
несуществующем path = False → easy to misinterpret; `Path.exists()`
проверять явно.

**Backlog (§11.6):** audit similar ambiguities:
- `CadwDataFrame.from_parquet(path, base_dir)` — path must be file
  (base_dir separate). OK, но неясно что если path directory.
- `Dataset.from_parquet(parquet_path, base_dir)` — same.
- `Pipeline.from_json(path)` — fixed here.
- `write_parquet(path)` — path is file (out), ambiguous if directory
  passed → check в v0.9.2.

**Связанные:** §13.4/146 (first-run bugs — errors not helpful),
§13.4/148 (design intent vs validation — user mental model matters).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/150")


def main():
    print("=== v0.9.1c-fix1 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_from_json_path_handling()
    append_decisions_150()

    print("\n=== pytest: save/load tests (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline_save_load.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** SAVE/LOAD STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: save/load E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline_save_load_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
        sys.exit(1)

    print("\n=== pytest: pipeline regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "tests/test_pipeline_e2e.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PIPELINE REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards via patch_utils ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1200)
    if r.returncode != 0:
        print("\n*** FAST FAILURE ***")
        sys.exit(1)

    print("\n=== pytest slow ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "slow"],
                       timeout=900)
    if r.returncode != 0:
        print("\n*** SLOW FAILURE ***")
        sys.exit(1)

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.9.1c: Pipeline save/load — to_json/from_json (directory "
        "format: pipeline.json + state/<step>.json), per-step "
        "to_dict/from_dict, GBDT full state serialization (JSON, no "
        "pickle), version field + strict_version, LambdaTransform raises "
        "NotImplementedError; §11.5 save/load constraints; §13.4/150 "
        "(error message clarity for ambiguous path)"])

    print("\n==> v0.9.1c-fix1 applied.")
    print("NEXT: v0.9.1d (ColumnTransformer + register_step_type).")


if __name__ == "__main__":
    main()