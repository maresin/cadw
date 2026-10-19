#!/usr/bin/env bash
# patch_11a3_fix: spill_boundaries = reserved for v0.2 (always empty в v0.1.x);
# allow_spill = no-op; нарушение invariant → RuntimeError (bug, не spill).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

BR="patch/11a3-planner"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11a3-fix-${TS}"
git diff > ".backups/backup-pre-11a3-fix-${TS}/wip.diff"

# ---- Fix 1: planner.py — spill reserved, invariant violation = bug ----
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/core/planner.py")
s = p.read_text()

# 1a. Docstring plan_pipeline
old_doc = '''    """Планирует цепочку.

    Для каждой op:
      1. model = get_model(op.name) — если op.name есть в реестре.
         Если нет ("filter", "with_column") — только propagate.
      2. params = model.solve(...); если None — попытка spill.
      3. estimate = model.estimate(params); если > budget — попытка spill.
      4. output_chars = propagate_chars(op, input, params).
    """'''
new_doc = '''    """Планирует цепочку (predict-only, §9.3).

    Для каждой op:
      1. model = get_model(op.name) — если op.name есть в реестре.
         Если нет ("filter", "with_column") — только propagate.
      2. params = model.solve(...). Если None — PipelineMemoryError.
      3. estimate = model.estimate(params). Инвариант solve гарантирует
         estimate <= budget; нарушение — RuntimeError (баг модели).
      4. output_chars = propagate_chars(op, input, params).

    `spill_boundaries` **reserved for v0.2** (chunked execution):
    в v0.1.x всегда пуст, `predicted_outcome` ∈ {"fits", "does_not_fit"}.
    `allow_spill` — no-op в v0.1.x, параметр сохранён для API v0.2.
    """'''
if old_doc not in s:
    raise SystemExit("plan_pipeline docstring not found")
s = s.replace(old_doc, new_doc, 1)

# 1b. Заменить блок spill-логики
old_spill = '''            if estimate > budget:
                if not allow_spill:
                    raise PipelineMemoryError(
                        step_idx=idx,
                        step_name=op.name,
                        estimate_bytes=estimate,
                        available_bytes=available_bytes,
                        describe=model.describe(),
                    )
                spill_boundaries.append(idx)'''
new_spill = '''            if estimate > budget:
                # Инвариант solve нарушен: params пришли из solve,
                # значит estimate должен быть <= budget. Это баг модели,
                # а не повод для spill (spill не уменьшает estimate для
                # той же op на тех же данных).
                raise RuntimeError(
                    f"planner invariant violated at step {idx} "
                    f"({op.name!r}): solve() returned params with "
                    f"estimate={estimate} > budget={budget}. "
                    f"Model: {model.describe()}"
                )'''
if old_spill not in s:
    raise SystemExit("spill logic not found")
s = s.replace(old_spill, new_spill, 1)

# 1c. Outcome: только fits / does_not_fit в v0.1.x
old_outcome = '''    peak = max((s.estimate_bytes for s in steps), default=0)
    if spill_boundaries:
        outcome = "fits_with_spill"
    else:
        outcome = "fits"

    return PipelinePlan(
        steps=steps,
        peak_bytes=int(peak),
        spill_boundaries=list(spill_boundaries),
        estimated_io=int(estimated_io_total),
        predicted_outcome=outcome,'''
new_outcome = '''    peak = max((s.estimate_bytes for s in steps), default=0)
    # v0.1.x: spill_boundaries всегда пуст (см. docstring);
    # outcome ∈ {"fits", "does_not_fit"}. "fits_with_spill" — v0.2.
    outcome = "fits"

    return PipelinePlan(
        steps=steps,
        peak_bytes=int(peak),
        spill_boundaries=list(spill_boundaries),
        estimated_io=int(estimated_io_total),
        predicted_outcome=outcome,'''
if old_outcome not in s:
    raise SystemExit("outcome block not found")
s = s.replace(old_outcome, new_outcome, 1)

p.write_text(s)
print("planner.py: spill reserved (v0.2); invariant violation = RuntimeError")
PY

# ---- Fix 2: HashShuffle.solve — финальная проверка estimate ≤ budget ----
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/core/models_shuffle.py")
s = p.read_text()

old = '''        batch_size = remaining // (data_chars.row_size_bytes + c_hash)
        if batch_size < 1:
            return None
        return HashShuffleParams(
            batch_size=int(batch_size),
            num_partitions=int(p),
            writer_buffer_bytes=int(writer_buffer),
        )'''
new = '''        batch_size = remaining // (data_chars.row_size_bytes + c_hash)
        if batch_size < 1:
            return None
        params = HashShuffleParams(
            batch_size=int(batch_size),
            num_partitions=int(p),
            writer_buffer_bytes=int(writer_buffer),
        )
        # Инвариант: estimate(params) <= budget. Округление вниз может
        # дать estimate на несколько байт больше; при строгом превышении
        # уменьшаем batch_size на 1, если всё ещё больше — None.
        if self.estimate(params, data_chars) > budget:
            params.batch_size -= 1
            if params.batch_size < 1:
                return None
            if self.estimate(params, data_chars) > budget:
                return None
        return params'''
if old not in s:
    raise SystemExit("HashShuffle.solve tail not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("models_shuffle.py: финальная проверка estimate <= budget")
PY

# ---- Fix 3: тесты — заменить 2 неверных ----
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_planner.py")
s = p.read_text()

old_tests = '''def test_pipeline_spill_boundary_when_estimate_exceeds_budget():
    """
    filter пропускает (estimate=0), groupby возвращает estimate > budget.
    Ожидается: fits_with_spill, spill_boundaries=[1] (перед groupby).
    """
    dc = _dc(n_rows=1_000_000, k_est=1_000_000, size_of_key_entry=80)
    ops = [OpSpec(name="filter"), OpSpec(name="groupby")]
    # available = 10 MB; groupby с K=1M и size_of_key=80 -> estimate ~25 MB
    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7, allow_spill=True)
    assert plan.predicted_outcome == "fits_with_spill"
    assert plan.spill_boundaries == [1]
    assert plan.steps[1].spill_before is True


def test_pipeline_no_spill_when_not_allowed():
    dc = _dc(n_rows=1_000_000, k_est=1_000_000, size_of_key_entry=80)
    ops = [OpSpec(name="filter"), OpSpec(name="groupby")]
    with pytest.raises(PipelineMemoryError):
        plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7, allow_spill=False)'''

new_tests = '''def test_pipeline_spill_boundaries_empty_in_v0_1():
    """spill_boundaries reserved for v0.2: в v0.1.x всегда пуст."""
    dc = _dc(n_rows=1_000_000, k_est=1_000_000, size_of_key_entry=80)
    ops = [OpSpec(name="filter"), OpSpec(name="groupby")]
    plan = plan_pipeline(ops, dc, 10 * 1024 * 1024, safety=0.7)
    assert plan.spill_boundaries == []
    assert plan.predicted_outcome == "fits"
    assert all(not s.spill_before for s in plan.steps)


def test_pipeline_solve_none_raises_regardless_of_allow_spill():
    """solve()=None -> PipelineMemoryError, allow_spill не помогает (v0.1.x)."""
    dc = _dc()
    ops = [OpSpec(name="reduce", op_context={"state_bytes": 200 * 1024 * 1024})]
    for allow in (True, False):
        with pytest.raises(PipelineMemoryError):
            plan_pipeline(
                ops, dc, 10 * 1024 * 1024, safety=0.7, allow_spill=allow
            )


def test_pipeline_invariant_violation_raises_runtime_error(monkeypatch):
    """Если solve вернул params с estimate > budget -> RuntimeError (баг модели)."""
    from cadw.core import planner as planner_mod
    from cadw.core.models_reduce import ReduceMemoryModel, ReduceParams

    class FakeModel(ReduceMemoryModel):
        op_name = "fake_reduce"

        def solve(self, available_bytes, data_chars, *, safety=0.7, op_context=None):
            # Возвращаем params, у которых estimate заведомо > budget
            return ReduceParams(batch_size=10**9, state_bytes=0)

        def estimate(self, params, data_chars):
            return 10**12  # 1 TB

        def describe(self):
            return "FakeModel (always huge)"

    monkeypatch.setattr(
        planner_mod, "get_model", lambda name: FakeModel()
    )
    dc = _dc()
    with pytest.raises(RuntimeError, match="invariant violated"):
        planner_mod.plan_pipeline(
            [OpSpec(name="fake_reduce")], dc, 10 * 1024 * 1024, safety=0.7
        )'''

if old_tests not in s:
    raise SystemExit("old tests block not found")
s = s.replace(old_tests, new_tests, 1)
p.write_text(s)
print("test_planner.py: 2 неверных теста заменены 3 корректными")
PY

# ---- Fix 4: DECISIONS §9.3 — уточнить spill ----
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

old = '''3. **Spill-boundary** = `write_parquet(temp) + read_from_temp`. Композиция,
   не новая операция. `peak` = `max(write, read, step_i)`.'''
new = '''3. **Spill-boundary** = `write_parquet(temp) + read_from_temp`. Композиция,
   не новая операция. `peak` = `max(write, read, step_i)`.
   **Статус v0.1.x: reserved for v0.2.** `solve` по инварианту возвращает
   params с `estimate ≤ available × safety`, поэтому `spill_boundaries`
   всегда пуст; `predicted_outcome ∈ {"fits", "does_not_fit"}`. Если после
   `solve` `estimate > budget` — это нарушение инварианта модели
   (`RuntimeError`), а не повод для spill. Реальный spill (chunked
   execution) — v0.2.'''

if old not in s:
    raise SystemExit("§9.3 spill text not found")
s = s.replace(old, new, 1)

# Обновить §9.8 статус
s = s.replace(
    "| §9.3 | `PipelinePlan`, `plan_pipeline` | ⏳ 11-A-3 |",
    "| §9.3 | `PipelinePlan`, `plan_pipeline` | ✅ 11-A-3 (spill — v0.2) |",
    1,
)
p.write_text(s)
print("DECISIONS §9.3 / §9.8: spill reserved for v0.2")
PY

# ---- Fix 5: убрать allow_spill из MemoryGuard.plan_pipeline? Оставляем как no-op ----
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/core/memory_guard.py")
s = p.read_text()
old = '''    def plan_pipeline(self, ops, data_chars: DataChars, *, allow_spill: bool = True):
        """Запланировать цепочку (§9.3). Raises PipelineMemoryError."""'''
new = '''    def plan_pipeline(self, ops, data_chars: DataChars, *, allow_spill: bool = True):
        """Запланировать цепочку (§9.3). Raises PipelineMemoryError.

        `allow_spill` — no-op в v0.1.x (spill_boundaries reserved for v0.2).
        """'''
if old not in s:
    raise SystemExit("MemoryGuard.plan_pipeline docstring not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("memory_guard.py: allow_spill — no-op (v0.1.x)")
PY

# ---- Run ----
set +e
uv run pytest tests/test_planner.py -v
RC_MODELS=$?
set -e
if [[ $RC_MODELS -ne 0 ]]; then
  echo "test_planner.py: rc=$RC_MODELS"
  exit 1
fi

uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-A-3-fix: spill_boundaries reserved for v0.2 (always empty); invariant violation = RuntimeError; HashShuffle финальная проверка"

echo "==> sha256"
sha256sum cadw/core/planner.py cadw/core/models_shuffle.py tests/test_planner.py
echo "==> 11-A-3-fix applied"