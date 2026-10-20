#!/usr/bin/env python3
"""v0.9.1b-fix1: 3 in-place fixes (design + validation-order + cleanup).

BUG 1 (design, §13.4/148): add_step validation forbade `inputs=[]` after
first step. Confused "explicit source" (redundant for first) with
"multiple sources" (essential for DAG fanout). My DAG tests use
`a inputs=[]`, `b inputs=[]` — both source. Validation killed the
feature. Fix: remove the check. `inputs=[]` on any step = source.

BUG 2 (validation-order): add_step("post", inputs=["model"]) where model
is Predictor. My order:
  1. Predictor-after check fires ("cannot add step after Predictor")
  2. inputs-contains-Predictor check never reached
Test expected "cannot depend on Predictor" (more specific error).
Fix: reorder — inputs validation BEFORE Predictor-after check.

BUG 3 (latent code smell): `_merge_fanout._iter` had garbage expression
`if c in seen and c not in set(b.schema.names[:0])` — second part always
True (empty set). Worked by accident. Simplify to `if c in base_cols_set:
continue`.

GUARDS (before commit): test_pipeline_dag + backward compat + guards +
fast + slow.

Run: uv run python scripts/patch_v091b_fix1.py
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


# ============ Fix 1: _merge_fanout._iter cleanup ============

def fix_merge_fanout_iter():
    print("\n=== cadw/pipeline.py: _merge_fanout._iter cleanup ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    old = '''    # Build output schema: base (dfs[0]) + new columns from others
    base_cols = list(dfs[0].columns)
    seen = set(base_cols)
    out_fields = [dfs[0]._dataset.schema.field(c) for c in base_cols]
    for d in dfs[1:]:
        for c in d.columns:
            if c in seen:
                continue
            out_fields.append(d._dataset.schema.field(c))
            seen.add(c)
    out_schema = pa.schema(out_fields)

    BATCH = 10_000

    def _iter():
        iters = [d._iter_arrow_batches(batch_size=BATCH) for d in dfs]
        for batches in zip(*iters):
            n = batches[0].num_rows
            for b in batches[1:]:
                if b.num_rows != n:
                    raise RuntimeError(
                        f"_merge_fanout: batch size mismatch at streaming: "
                        f"{n} vs {b.num_rows}"
                    )
            first_b = batches[0]
            arrays = [first_b.column(c) for c in base_cols]
            for b in batches[1:]:
                for c in b.schema.names:
                    if c in seen and c not in set(
                        b.schema.names[:0]
                    ):  # c in base_cols → already taken
                        continue
                    # Append new column (not in base_cols)
                    if c in base_cols:
                        continue
                    arrays.append(b.column(c))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)'''

    new = '''    # Build output schema: base (dfs[0]) + new columns from others
    base_cols = list(dfs[0].columns)
    base_cols_set = set(base_cols)
    seen = set(base_cols)
    out_fields = [dfs[0]._dataset.schema.field(c) for c in base_cols]
    for d in dfs[1:]:
        for c in d.columns:
            if c in seen:
                continue
            out_fields.append(d._dataset.schema.field(c))
            seen.add(c)
    out_schema = pa.schema(out_fields)

    BATCH = 10_000

    def _iter():
        iters = [d._iter_arrow_batches(batch_size=BATCH) for d in dfs]
        for batches in zip(*iters):
            n = batches[0].num_rows
            for b in batches[1:]:
                if b.num_rows != n:
                    raise RuntimeError(
                        f"_merge_fanout: batch size mismatch at streaming: "
                        f"{n} vs {b.num_rows}"
                    )
            first_b = batches[0]
            arrays = [first_b.column(c) for c in base_cols]
            # v0.9.1b-fix1: simple check — append only columns NOT in
            # base_cols. (Prior garbage: `c in seen and c not in set(
            # b.schema.names[:0])` — always True on second part.)
            for b in batches[1:]:
                for c in b.schema.names:
                    if c in base_cols_set:
                        continue
                    arrays.append(b.column(c))
            yield pa.RecordBatch.from_arrays(arrays, schema=out_schema)'''

    s = replace_line_block(s, old, new, tag="merge-fanout-iter")
    p.write_text(s)
    compile_check(p)
    print("  OK: simplified append condition (base_cols_set)")


# ============ Fix 2: reorder validation + remove inputs=[] restriction ============

def fix_add_step_validation():
    print("\n=== cadw/pipeline.py: add_step — remove inputs=[] restriction, reorder ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    old = '''        new_is_pred = _is_predictor(step)
        if new_is_pred and any(_is_predictor(s) for s in self._steps):
            raise RuntimeError(
                "add_step: pipeline already has a Predictor; "
                "cannot add another (model must be last)"
            )
        if self._steps and _is_predictor(self._steps[-1]) and not new_is_pred:
            raise RuntimeError(
                "add_step: cannot add step after a Predictor "
                "(model must be last)"
            )
        # v0.9.1b: validate inputs
        if inputs is not None:
            if not isinstance(inputs, (list, tuple)):
                raise TypeError(
                    f"add_step: inputs must be list or None; "
                    f"got {type(inputs).__name__}"
                )
            for inp in inputs:
                if not isinstance(inp, str):
                    raise TypeError(
                        f"add_step: inputs must contain str; "
                        f"got {type(inp).__name__}"
                    )
                if inp not in self._step_names:
                    raise ValueError(
                        f"add_step: inputs refers to unknown step {inp!r}; "
                        f"existing={self._step_names}"
                    )
                inp_step = self._steps[self._step_names.index(inp)]
                if _is_predictor(inp_step):
                    raise RuntimeError(
                        f"add_step: cannot depend on Predictor {inp!r} "
                        f"(model must be sink)"
                    )
            if len(inputs) == 0 and self._steps:
                raise ValueError(
                    "add_step: inputs=[] (explicit source) is only allowed "
                    "for the first step; leave inputs=None for chain"
                )
            inputs_stored = list(inputs)
        else:
            inputs_stored = None  # chain (derived at execution)
        self._step_names.append(name)
        self._steps.append(step)
        self._step_inputs.append(inputs_stored)
        return self'''

    new = '''        new_is_pred = _is_predictor(step)
        if new_is_pred and any(_is_predictor(s) for s in self._steps):
            raise RuntimeError(
                "add_step: pipeline already has a Predictor; "
                "cannot add another (model must be last)"
            )

        # v0.9.1b-fix1 (§13.4/148): validate inputs BEFORE Predictor-after
        # check. Specific error ("cannot depend on Predictor") should fire
        # before generic ("cannot add step after Predictor") when user
        # passes inputs=["model"].
        # Also: inputs=[] is VALID for any step (multiple sources — main
        # DAG feature). v0.9.1b originally forbade inputs=[] after first
        # step — that killed fanout (`a inputs=[]`, `b inputs=[]`).
        if inputs is not None:
            if not isinstance(inputs, (list, tuple)):
                raise TypeError(
                    f"add_step: inputs must be list or None; "
                    f"got {type(inputs).__name__}"
                )
            for inp in inputs:
                if not isinstance(inp, str):
                    raise TypeError(
                        f"add_step: inputs must contain str; "
                        f"got {type(inp).__name__}"
                    )
                if inp not in self._step_names:
                    raise ValueError(
                        f"add_step: inputs refers to unknown step {inp!r}; "
                        f"existing={self._step_names}"
                    )
                inp_step = self._steps[self._step_names.index(inp)]
                if _is_predictor(inp_step):
                    raise RuntimeError(
                        f"add_step: cannot depend on Predictor {inp!r} "
                        f"(model must be sink)"
                    )
            inputs_stored = list(inputs)
        else:
            inputs_stored = None  # chain (derived at execution)

        # Predictor-after check (moved AFTER inputs validation)
        if self._steps and _is_predictor(self._steps[-1]) and not new_is_pred:
            raise RuntimeError(
                "add_step: cannot add step after a Predictor "
                "(model must be last)"
            )

        self._step_names.append(name)
        self._steps.append(step)
        self._step_inputs.append(inputs_stored)
        return self'''

    s = replace_line_block(s, old, new, tag="add-step-reorder")
    p.write_text(s)
    compile_check(p)
    print("  OK: inputs=[] allowed anywhere; reordered validation")


# ============ Fix 3: §13.4/148 ============

def append_decisions_148():
    print("\n=== DECISIONS.md: §13.4/148 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 148" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append via warning")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/148 (design intent vs validation).** v0.9.1b add_step "
        s += "forbade inputs=[] after first step. Confused 'explicit source' "
        s += "with 'multiple sources' (DAG fanout). Fix in v0.9.1b-fix1.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 148 (design intent vs validation — DAG multiple sources).**
v0.9.1b: `add_step` validation запретил `inputs=[]` после первого шага:
```python
if len(inputs) == 0 and self._steps:
    raise ValueError("explicit source only valid for first step")
```
Я перепутал **«explicit source»** (избыточно для первого шага — там
`inputs=None` уже даёт source) с **«multiple sources»** (главная цель
DAG — fanout от raw). Мои же тесты используют `a inputs=[]`,
`b inputs=[]` — оба source. Validation убил фичу в самом патче,
который эту фичу добавляет. 6 tests fail.

**Различие семантик:**
- Sequence (v0.9.0): «step N+1 идёт после step N» — один linear source.
- DAG (v0.9.1b): «любой step может читать raw независимо» — **multiple
  sources by design**. `inputs=[]` на любом шаге = source, игнорируя prev.

**Fix (v0.9.1b-fix1):** убрать проверку. `inputs=[]` валиден на любом
шаге. `inputs=None` (default) = chain to prev (backward compat).

**Сопутствующий bug 2 (validation order):**
`add_step("post", inputs=["model"])` где `model` = Predictor. Мой
порядок: Predictor-after check → RuntimeError «cannot add step after
Predictor»; inputs validation не достигается. Но `inputs=["model"]` —
это семантически **более специфичная** ошибка («зависимость от
Predictor»). Fix: reorder — inputs validation до Predictor-after.

**Сопутствующий bug 3 (latent code smell):**
`_merge_fanout._iter` имел `if c in seen and c not in set(b.schema.names[:0])`.
`b.schema.names[:0]` = пустой list → `set()` → условие сводится к `c in
seen`. Работает случайно. Заменено на `if c in base_cols_set: continue`.

**Урок (расширение 146-147):** validation в новом API должен пройти
через **все репрезентативные use cases** *до коммита*. В v0.9.1b я
написал тесты для: sequence, fan-in, fanout, diamond, mismatch,
unknown input, Predictor-as-input, explicit source validation, topo
order (chain, fanout), transform-stops-at-model, predict, repr DAG.
Первый прогон — 6 fails. Правило: **«написать 1 тест на каждую
семантическую ветку + прогон до коммита»** — но этого не хватило,
потому что **тесты и validation были от одной модели** (я писал и
то, и другое). Нужна **независимая проверка** — либо prototype (A/B
против pandas groupby — не применимо к DAG), либо **явное документирование
design intent** до написания validation.

**Backlog (§11.6, v0.9.1c+):** перед `add_step`-style API — сначала
docstring «## API contract» с явными примерами, **затем** validation.
Не наоборот. Для DAG это спасло бы: «inputs=[] → source, может быть
несколько» — одной строки в docstring хватило бы.

**Связанные:** §13.4/146 (3 first-run bugs), §13.4/147 (partial
contract).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/148")


# ============ main ============

def main():
    print("=== v0.9.1b-fix1 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_merge_fanout_iter()
    fix_add_step_validation()
    append_decisions_148()

    print("\n=== pytest: DAG tests (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_pipeline_dag.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** DAG TESTS STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: backward compat + E2E ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline.py", "tests/test_pipeline_e2e.py",
         "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** BACKWARD COMPAT FAILED ***")
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
        "v0.9.1b: DAG Pipeline — add_step(inputs=[...]), fan-in via "
        "_merge_fanout (column union, same-row precondition), "
        "topological execution (fit/transform/predict), LambdaTransform "
        "test helper, DAG-aware _repr_html_; §11.5 fan-in MVP note; "
        "§13.4/148 (multiple sources in DAG, not just first step)"])

    print("\n==> v0.9.1b applied.")
    print("NEXT: v0.9.1c (save/load to_json/from_json).")


if __name__ == "__main__":
    main()
