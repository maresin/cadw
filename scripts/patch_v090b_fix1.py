#!/usr/bin/env python3
"""v0.9.0b-fix1: Predictor contract + DECISIONS anchor.

BUG 1 (§13.4/147 — partial contract):
  `add_step` требует `fit + transform`. `GBDTStep` — Predictor
  (fit + predict), без transform. 6 tests fail при
  `add_step("model", GBDTStep(...))`.

  Контракт Pipeline: Predictor = Transformer с добавленным `predict`.
  Но `transform` не обязателен для Predictor — его никогда не вызывают
  (`Pipeline.transform` останавливается перед Predictor,
  `Pipeline.predict` вызывает `step.predict` на финальном шаге).

  Fix: `add_step` validation — `fit + (transform OR predict)`.

  Плюс proactive fix: `add_step` также должен ловить «добавили
  non-Predictor после Predictor» — иначе `fit` (RuntimeError) станет
  первым местом обнаружения.

BUG 2 (§13.4/144-подобный — anchor not found):
  DECISIONS §11.6 anchor not found. Заголовок мог измениться между
  версиями (### 11.6. vs ## 11.6.). Fix: fallback anchors.

GUARDS (before commit):
  test_pipeline + test_pipeline_e2e (slow) + changelog + arrow +
  no_pandas + fast + slow.

Run: uv run python scripts/patch_v090b_fix1.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


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


# ============ Fix 1: add_step contract + Predictor-after check ============

def fix_add_step_contract():
    print("\n=== cadw/pipeline.py::add_step — fit + (transform OR predict) ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    old = '''        if not (hasattr(step, "fit") and hasattr(step, "transform")):
            raise TypeError(
                f"add_step: step must have fit() and transform(); "
                f"got {type(step).__name__}"
            )
        if _is_predictor(step) and self._steps:
            # Predictor must be last; if it's not the first, error would
            # fire later in fit/transform, but check upfront for clear msg.
            if any(_is_predictor(s) for s in self._steps):
                raise RuntimeError(
                    "add_step: pipeline already has a Predictor at end; "
                    "cannot add another"
                )'''

    new = '''        # v0.9.0b-fix1: Transformer contract = fit + (transform OR predict).
        # A Predictor is a step with fit + predict; its transform is never
        # called (Pipeline.transform stops before Predictor; Pipeline.predict
        # uses step.predict on the last step). Requiring transform on
        # Predictors broke GBDTStep (has fit+predict, no transform).
        has_fit = hasattr(step, "fit")
        has_transform = hasattr(step, "transform")
        has_predict = hasattr(step, "predict")
        if not (has_fit and (has_transform or has_predict)):
            raise TypeError(
                f"add_step: step must have fit() and (transform() or "
                f"predict()); got {type(step).__name__}"
            )
        # v0.9.0b-fix1: Predictor must be LAST. Two checks:
        #   (a) new step is Predictor, existing steps contain Predictor → error
        #   (b) existing last is Predictor, new step is non-Predictor → error
        new_is_pred = _is_predictor(step)
        if new_is_pred and any(_is_predictor(s) for s in self._steps):
            raise RuntimeError(
                "add_step: pipeline already has a Predictor; "
                "cannot add another (model must be last)"
            )
        if self._steps and _is_predictor(self._steps[-1]) and not new_is_pred:
            raise RuntimeError(
                "add_step: cannot add step after a Predictor "
                "(model must be last)"
            )'''

    s = replace_line_block(s, old, new, tag="add-step-contract")
    p.write_text(s)
    compile_check(p)
    print("  OK: fit + (transform OR predict); Predictor must be last")


# ============ Fix 2: DECISIONS §11.6 anchor fallback ============

DECISIONS_ADDENDUM = '''**v0.9.0b (Pipeline v1):**
- `Pipeline.plan(df)` — **fit feasibility only** (`sum(step.state_bytes)
  ≤ state_limit`). `spill_boundaries` always empty (deferred v0.9.1).
  Per-batch transform guard — **per-step** (each step's `MemoryGuard`
  handles its own batch).
- `Pipeline.transform(df)` **stops before Predictor** (model is last).
  `Pipeline.predict(df)` requires last step to be Predictor.
- `Pipeline.fit(df, target=None)` — target required if pipeline contains
  Predictor. `plan` called first → `PipelineMemoryError` if not feasible
  (honest fail-fast, §9.3).
- **Predictor contract:** `fit + predict` (transform optional; Predictor
  never goes through `.transform()`).
- **State in-memory** (bounded by K × #features per step). `to_json` /
  `from_json` — v0.9.1.
- **`ColumnTransformer`** — v0.9.1 (fanout by columns).
- **DAG** (via `inputs=[...]`) — v0.9.1.
- **`TimeSeriesFeatures`** composite — v0.9.2.

'''


def fix_decisions_anchor():
    print("\n=== DECISIONS.md: §11.6 with anchor fallback ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "v0.9.0b (Pipeline v1)" in s:
        print("  already present — skip")
        return

    # Try multiple anchors
    candidates = [
        "### 11.6. Технический долг",
        "## 11.6. Технический долг",
        "11.6. Технический долг",
    ]
    anchor = None
    for c in candidates:
        if c in s:
            anchor = c
            break

    if anchor is None:
        print("  WARNING: no §11.6 anchor found (###/##/plain); "
              "appending at end of file")
        if not s.endswith("\n"):
            s += "\n"
        s += "\n" + DECISIONS_ADDENDUM
    else:
        s = s.replace(anchor, DECISIONS_ADDENDUM + anchor, 1)
        print(f"  OK: anchored at {anchor!r}")
    p.write_text(s)


# ============ §13.4/147 ============

def append_decisions_147():
    print("\n=== DECISIONS.md: §13.4/147 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 147" in s:
        print("  already present — skip")
        return

    marker = "### 13.5. Открытые вопросы"
    if marker not in s:
        print("  WARNING: §13.5 marker not found — skip")
        return

    episode = '''**Эпизод 147 (partial contract — Predictor ≠ Transformer).**
v0.9.0b: `add_step` валидировал `fit + transform`. `GBDTStep` —
Predictor (fit + predict, без transform). 6 tests fail.

**Класс — partial contract** (как §13.4/146 bug 1 «partial update within
one class», но здесь — partial contract между двумя классами):
- `Transformer` Protocol декларирует `fit + transform`.
- `Predictor` Protocol — `fit + transform + predict` (в v0.9.0b коде).
- Но `GBDTStep` реализует `fit + predict` (без transform).
- `Pipeline.transform` **останавливается перед** Predictor → Predictor
  никогда не вызывает `.transform()`.
- `add_step` проверяет `transform` → Predictor'ы (GBDTStep) не проходят
  validation.

**Fix (v0.9.0b-fix1):** `add_step` validation —
`has_fit and (has_transform or has_predict)`. Протокол `Transformer`
остаётся `fit + transform` (для pure-transform steps); `Predictor` —
`fit + predict` (для model steps); `Pipeline` принимает оба, различает
по `_is_predictor(step) = hasattr(step, "predict")`.

**Proactive fix:** `add_step` также ловит «Predictor not last» в обоих
направлениях:
- new step is Predictor + существующие contain Predictor → RuntimeError.
- existing last is Predictor + new step non-Predictor → RuntimeError.

Раньше это ловилось только на `fit` (`if is_pred and i != len(self._steps) - 1`),
что поздно. Теперь — fail-fast при `add_step`.

**Урок:** Protocol `Transformer` в v0.9.0b коде декларирует `fit +
transform + predict` для `Predictor` (строгое подмножество полного
набора). Реально `predict`-only steps **не нуждаются** в `transform`.
Contract должен быть **minimal**: `Predictor = fit + predict`, а
`Pipeline.transform` проверяет `_is_predictor` и пропускает последний
шаг. Декларируй больше, чем нужно → validation ломается на классах,
которые не реализуют ненужное.

**Правило:** для каждого Protocol/ABC — спросить «какой минимум
методов реально вызывается?». Всё, что «для симметрии» — либо убрать,
либо сделать optional. `Transformer` (pure transform) vs `Predictor`
(model) — не подмножество, а два разных минимальных контракта.

**Backlog (v0.9.1):** отделить `Transformer` и `Predictor` Protocols
строго (не `Predictor ⊃ Transformer`). `Predictor` Protocol: `fit,
predict, estimate_state_bytes, state_bytes` — без `transform`.
`Pipeline` типизирует `steps: list[Transformer | Predictor]`.

**Связанные:** §13.4/128, /129, /139-146.

'''
    s = s.replace(marker, episode + marker, 1)
    p.write_text(s)
    print("  OK: §13.4/147")


# ============ main ============

def main():
    print("=== v0.9.0b-fix1 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_add_step_contract()
    fix_decisions_anchor()
    append_decisions_147()

    print("\n=== pytest: pipeline tests (was failing) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_pipeline.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PIPELINE TESTS STILL RED ***")
        sys.exit(1)

    print("\n=== pytest: pipeline E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pipeline_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_no_pandas_in_ops.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** GUARD FAILURE ***")
        sys.exit(1)

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

    # Amend WIP (b5df8eb... no wait — v0.9.0b was never committed!)
    # Check git status:
    r_status = subprocess.run(["git", "status", "--porcelain"],
                              capture_output=True, text=True)
    if r_status.stdout.strip():
        sh(["git", "add", "-A"])
        sh(["git", "commit", "-m",
            "v0.9.0b: Pipeline skeleton — Transformer/Predictor protocols, "
            "Pipeline (add_step/fit/transform/predict/fit_transform/plan), "
            "CastStep/DropnaStep/GBDTStep wrappers, plan-first feasibility, "
            "E2E time-series GBDT; §13.4/147 (Predictor contract minimal)"])

    print("\n==> v0.9.0b-fix1 applied.")
    print("NEXT: v0.9.0c (Jupyter _repr_html_, df.plot accessor).")


if __name__ == "__main__":
    main()