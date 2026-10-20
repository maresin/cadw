#!/usr/bin/env python3
"""v0.9.1d-fix1: registry namespace + missing ColumnTransformer + test fixes.

BUG 1 (§13.4/151, design): register_step_type("MyCustom", _CustomStep)
registers under key "MyCustom". But to_json writes `type(step).__name__
= "_CustomStep"` — mismatch. from_json looks up "_CustomStep" → not
found. Two independent namespaces (registry key vs class name).

FIX: register_step_type sets `cls.__serialization_name__ = name`.
to_json prefers `__serialization_name__` over `type(step).__name__`.
Built-ins pre-registered with their canonical name. Backward compat:
if `__serialization_name__` absent, fall back to class name.

BUG 2 (test vs code): test_to_dict_lambda_nested_raises expected
"does not implement to_dict". But LambdaTransform HAS to_dict (which
raises NotImplementedError with honest message). Code is correct,
test regex wrong.

FIX: test regex → "LambdaTransform cannot" (matches actual message).

BUG 3 (missing registration): ColumnTransformer defined AFTER
_STEP_REGISTRY pre-registration block → never registered. 

FIX: `_STEP_REGISTRY["ColumnTransformer"] = ColumnTransformer` after
class definition.

BONUS: SyntaxWarning "invalid escape sequence '\\+'" in
test_column_transformer.py — non-raw string. Fix with r-prefix.

GUARDS: new tests + pipeline regression + guards + fast + slow.

Run: uv run python scripts/patch_v091d_fix1.py
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


# ============ Fix 1: register_step_type sets __serialization_name__ ============

def fix_register_sets_serialization_name():
    print("\n=== cadw/pipeline.py::register_step_type — __serialization_name__ ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    old = '''    if name in _STEP_REGISTRY:
        warnings.warn(
            f"register_step_type: {name!r} already registered as "
            f"{_STEP_REGISTRY[name].__name__}; overwriting with "
            f"{cls.__name__}",
            RuntimeWarning, stacklevel=2,
        )
    _STEP_REGISTRY[name] = cls'''

    new = '''    if name in _STEP_REGISTRY:
        warnings.warn(
            f"register_step_type: {name!r} already registered as "
            f"{_STEP_REGISTRY[name].__name__}; overwriting with "
            f"{cls.__name__}",
            RuntimeWarning, stacklevel=2,
        )
    _STEP_REGISTRY[name] = cls
    # v0.9.1d-fix1 (§13.4/151): set `__serialization_name__` so `to_json`
    # writes THIS name (registry key), not `type(step).__name__` (class
    # name). Resolves namespace mismatch: registration key = serialization
    # key. Registering "MyCustom" for class `_CustomStep` → to_json writes
    # "MyCustom"; from_json finds it in registry.
    setattr(cls, "__serialization_name__", name)'''

    s = replace_line_block(s, old, new, tag="register-serialization-name")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 2: pre-registration sets __serialization_name__ ============

def fix_preregistration():
    print("\n=== cadw/pipeline.py: pre-registration + ColumnTransformer ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    old = '''# Pre-register built-in wrappers
_STEP_REGISTRY["CastStep"] = CastStep
_STEP_REGISTRY["DropnaStep"] = DropnaStep
_STEP_REGISTRY["GBDTStep"] = GBDTStep'''

    new = '''# Pre-register built-in wrappers. `__serialization_name__` set to the
# canonical name so `to_json` writes it (and `from_json` finds it).
# `ColumnTransformer` is registered at the bottom of this module,
# AFTER its class definition (see below).
_STEP_REGISTRY["CastStep"] = CastStep
_STEP_REGISTRY["DropnaStep"] = DropnaStep
_STEP_REGISTRY["GBDTStep"] = GBDTStep
for _pre_name, _pre_cls in list(_STEP_REGISTRY.items()):
    setattr(_pre_cls, "__serialization_name__", _pre_name)'''

    s = replace_line_block(s, old, new, tag="preregistration")
    p.write_text(s)
    compile_check(p)
    print("  OK: pre-registration loop")


# ============ Fix 3: register ColumnTransformer after class ============

def register_column_transformer():
    print("\n=== cadw/pipeline.py: register ColumnTransformer ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    if '_STEP_REGISTRY["ColumnTransformer"] = ColumnTransformer' in s:
        print("  already present — skip")
        return

    # Append after ColumnTransformer's __repr__ (last method of class)
    anchor = '''    def __repr__(self) -> str:
        parts = ", ".join(
            f"{name}={list(cols)}" for name, cols, _ in self._transformers
        )
        return (
            f"ColumnTransformer([{parts}], remainder={self._remainder!r})"
        )'''

    new = anchor + '''


# v0.9.1d-fix1 (§13.4/151): register ColumnTransformer AFTER its class
# definition (pre-registration block runs before class exists).
_STEP_REGISTRY["ColumnTransformer"] = ColumnTransformer
ColumnTransformer.__serialization_name__ = "ColumnTransformer"'''

    s = replace_line_block(s, anchor, new, tag="register-ct-after-class")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 4: to_json uses __serialization_name__ ============

def fix_to_json_type_name():
    print("\n=== cadw/pipeline.py::to_json — __serialization_name__ ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    old = '''            entry = {
                "name": name,
                "type": type(step).__name__,
                "inputs": self._step_inputs[
                    self._step_names.index(name)
                ],
                "params": params,
                "has_state": state is not None,
            }'''

    new = '''            # v0.9.1d-fix1: use __serialization_name__ (set by
            # register_step_type / pre-registration) if present; fall
            # back to class name for backward compat.
            type_name = getattr(
                type(step), "__serialization_name__", type(step).__name__
            )
            entry = {
                "name": name,
                "type": type_name,
                "inputs": self._step_inputs[
                    self._step_names.index(name)
                ],
                "params": params,
                "has_state": state is not None,
            }'''

    s = replace_line_block(s, old, new, tag="to-json-type-name")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 5: test regex for LambdaTransform ============

def fix_lambda_test_regex():
    print("\n=== tests/test_column_transformer.py: LambdaTransform regex ===")
    p = Path("tests/test_column_transformer.py")
    s = p.read_text()

    old = '''def test_to_dict_lambda_nested_raises():
    ct = ColumnTransformer([
        ("lam", ["a"], LambdaTransform(
            "a", "sq", lambda arr: pc.multiply(arr, arr), pa.int64(),
        )),
    ])
    with pytest.raises(NotImplementedError, match="does not implement to_dict"):
        ct.to_dict()'''

    new = '''def test_to_dict_lambda_nested_raises():
    """LambdaTransform.to_dict raises NotImplementedError (§13.4/151).

    v0.9.1d-fix1: LambdaTransform DOES implement to_dict (which raises
    NotImplementedError). Message: "LambdaTransform cannot be serialized".
    Prior test expected "does not implement to_dict" (wrong — that's for
    custom steps without to_dict method).
    """
    ct = ColumnTransformer([
        ("lam", ["a"], LambdaTransform(
            "a", "sq", lambda arr: pc.multiply(arr, arr), pa.int64(),
        )),
    ])
    with pytest.raises(NotImplementedError, match="LambdaTransform cannot"):
        ct.to_dict()'''

    s = replace_line_block(s, old, new, tag="lambda-test-regex")
    p.write_text(s)
    print("  OK")


# ============ Fix 6: SyntaxWarning '\+' ============

def fix_syntax_warning():
    print("\n=== tests/test_column_transformer.py: raw-string regex ===")
    p = Path("tests/test_column_transformer.py")
    s = p.read_text()

    old = 'with pytest.raises(TypeError, match="must have fit \\+ transform"):'
    new = 'with pytest.raises(TypeError, match=r"must have fit \\+ transform"):'

    if old not in s:
        print("  pattern not found — skip")
        return
    s = s.replace(old, new, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 7: §13.4/151 ============

def append_decisions_151():
    print("\n=== DECISIONS.md: §13.4/151 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 151" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/151 (registry key vs class name).** "
        s += "register_step_type(name, cls) must set cls.__serialization_name__ "
        s += "= name; to_json uses __serialization_name__ else class name.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 151 (registry key vs class name — namespace mismatch).**
v0.9.1d: `register_step_type("MyCustom", _CustomStep)` регистрирует под
ключом `"MyCustom"`. Но `to_json` пишет `type = type(step).__name__ =
"_CustomStep"` — **class name**, не registry key. `from_json` ищет
`"_CustomStep"` в registry → не находит. 3 tests fail.

**Тот же design flaw, что был в v0.9.1c (`_STEP_REGISTRY` hardcoded
dict) — но другая грань.** Там я устранил hardcoded registry, но не
устранил **namespace mismatch** между тем, что регистрируется, и тем,
что пишется в JSON.

**Fix (v0.9.1d-fix1):**
- `register_step_type` sets `cls.__serialization_name__ = name`
  (attribute-based namespace binding).
- `to_json` prefers `getattr(type(step), "__serialization_name__",
  type(step).__name__)` — backward compat для старых pipeline.json.
- Built-in types pre-registered с `__serialization_name__` = canonical
  name (CastStep, DropnaStep, GBDTStep, ColumnTransformer).
- `ColumnTransformer` регистрируется **после** class definition
  (pre-registration block runs before class exists — NameError).

**Сопутствующие fixes (в том же прогоне):**
- test regex `"does not implement to_dict"` → `"LambdaTransform cannot"`
  (LambdaTransform **имеет** to_dict — он raises с honest message;
  test был wrong, code correct).
- `SyntaxWarning: invalid escape sequence '\\+'` — non-raw regex string,
  `match="must have fit \\+ transform"` → raw `match=r"..."`.

**Урок (расширение §13.4/144, /149):** если API использует **lookup key**
(registry, dict, enum) — **key at write и key at read должны быть одной
природы**. В v0.9.1d: write by class name (implicit), read by registry
key (explicit). Fix: явная `__serialization_name__` как единая точка.

**Метаправило:** для **lookup-based API** (registry, plugin system,
event bus) — обязательное `assert write_key == read_key` в тесте на
round-trip. Здесь это round-trip `to_json → from_json` — but only for
built-ins (registry key = class name). Custom step would catch it. Что
не хватило: **тест с custom registered name ≠ class name**.

**Backlog (§11.6):** audit similar lookups:
- `_STEP_REGISTRY` (fixed here).
- `_MODELS` в `memory_guard.py` — `register_model(model.op_name)`:
  registry key = `model.op_name`, no round-trip. Меньше риск, но
  `get_model(name)` uses registry directly — OK.
- `_VARIANTS` в `naive_bayes.py` — enum-like, no write side.

**Связанные:** §13.4/144 (write/read key mismatch, round 2 → 144's rule
needed), §13.4/150 (error message clarity).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/151")


# ============ main ============

def main():
    print("=== v0.9.1d-fix1 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_register_sets_serialization_name()
    fix_preregistration()
    register_column_transformer()
    fix_to_json_type_name()
    fix_lambda_test_regex()
    fix_syntax_warning()
    append_decisions_151()

    print("\n=== pytest: new registry + CT tests (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_register_step_type.py",
         "tests/test_column_transformer.py",
         "tests/test_column_transformer_pipeline.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: pipeline regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py",
         "tests/test_pipeline_dag.py",
         "tests/test_pipeline_e2e.py",
         "tests/test_pipeline_save_load.py",
         "tests/test_pipeline_save_load_e2e.py",
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
        "v0.9.1d: ColumnTransformer (sklearn-style column fanout, "
        "row-preserving, remainder drop/passthrough) + register_step_type "
        "(module-level _STEP_REGISTRY, custom steps for from_json); "
        "cadw/__init__.py exports Pipeline + wrappers; §13.4/151 "
        "(registry key vs class name — __serialization_name__)"])

    print("\n==> v0.9.1d-fix1 applied.")
    print("NEXT: v0.9.1e (apply(fn) — prototype + implementation).")


if __name__ == "__main__":
    main()