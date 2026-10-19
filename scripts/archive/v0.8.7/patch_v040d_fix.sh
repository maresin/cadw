#!/usr/bin/env bash
# patch_v040d_fix: добавить _apply_filter_with_positions helper (module-level).
# Проверка была по подстроке → call site маскировал отсутствие def.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/v040d-filter-compat"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-v040d-fix-${TS}"
git diff > ".backups/backup-pre-v040d-fix-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: v0.4.0d-fix (helper undefined; substring check false-negative)" || true
require_clean_tree || exit 1

python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# Strict check: define vs call. Call site inside class method exists; def at
# module level does NOT.
if "def _apply_filter_with_positions(" in s:
    print("dataframe.py: helper already defined — no-op")
else:
    if "_apply_filter_with_positions(" not in s:
        raise SystemExit(
            "expected call site '_apply_filter_with_positions(' to exist "
            "(from v0.4.0d patch); got none"
        )

    helper = '''

def _apply_filter_with_positions(f: RowFilter, it):
    """Apply a RowFilter to (batch, positions) stream (v0.4.0d).

    - MaskFilter: positions[mask] — raw positions of rows that passed.
    - SliceFilter: slice по **локальному** counter (filtered-view position);
      raw positions — из накопленного массива.
    """
    if isinstance(f, MaskFilter):
        for b, positions in it:
            mask = f.mask_for_batch(b)
            mask_np = mask.to_numpy(zero_copy_only=False)
            new_positions = positions[mask_np]
            if len(new_positions) > 0:
                yield b.filter(mask), new_positions
        return
    if isinstance(f, SliceFilter):
        local_pos = 0
        for b, positions in it:
            n = b.num_rows
            batch_local = np.arange(local_pos, local_pos + n, dtype=np.int64)
            mask_np = batch_local >= f.start
            if f.stop is not None:
                mask_np &= batch_local < f.stop
            if f.step > 1:
                mask_np &= ((batch_local - f.start) % f.step == 0)
            new_positions = positions[mask_np]
            if len(new_positions) > 0:
                yield (
                    b.filter(pa.array(mask_np, type=pa.bool_())),
                    new_positions,
                )
            local_pos += n
            if f.stop is not None and local_pos >= f.stop:
                return
        return
    raise TypeError(
        f"_apply_filter_with_positions: unsupported filter "
        f"{type(f).__name__}"
    )
'''

    s = s.rstrip() + helper + "\n"
    p.write_text(s)
    print("dataframe.py: _apply_filter_with_positions helper added (def at module level)")

# Verify def now present
s2 = p.read_text()
assert "def _apply_filter_with_positions(" in s2, "helper still not defined"
print("verify: def _apply_filter_with_positions present")
PY

# ---------- AST guard against this class of bug ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_filter_compat.py")
s = p.read_text()

extra = '''

# ---------- AST guard: no undefined module-level helper calls ----------

def test_helper_is_defined_not_just_called():
    """Регрессия v0.4.0d: `if "name" not in s` видел call site и не
    добавлял def. Guard: `def _apply_filter_with_positions(` присутствует.
    """
    import inspect
    from cadw import dataframe as df_mod
    src = inspect.getsource(df_mod)
    assert "def _apply_filter_with_positions(" in src, (
        "helper _apply_filter_with_positions not defined (only called) — "
        "NameError at runtime. §13.4/60."
    )
    assert "_apply_filter_with_positions(" in src
'''

s = s.rstrip() + extra + "\n"
p.write_text(s)
print("test_filter_compat.py: +AST guard def-not-just-call")
PY

# ---------- DECISIONS: episode 60 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

marker_135 = "### 13.5. Открытые вопросы"
episode_60 = """**Эпизод 60 (substring check маскирует отсутствие def).** Патч v0.4.0d
добавлял helper `_apply_filter_with_positions` условно:
`if "_apply_filter_with_positions" not in s: s = s + helper`. К моменту
проверки **call site** helper'а уже был в `s` (из шага 2 — метода
`_iter_filtered_batches_with_positions`). Проверка по подстроке увидела
call, решила «уже есть» — def не добавлен. 12 из 15 тестов упали на
`NameError: name '_apply_filter_with_positions' is not defined`.

**Fix:** проверка по `"def _apply_filter_with_positions("` (не просто имя).
Плюс AST guard `test_helper_is_defined_not_just_called`.

**Урок (усиливает эпизод 44/47b):** `if name not in src` — плохая
проверка существования функции. Различайте **call site** и **definition**:
- `f"def {name}(" in src` — определение;
- `f"{name}(" in src` — может быть call.

Тот же класс, что `str.replace` silent no-op (эпизод 44), но через
**false negative** — не «не заметили ошибку замены», а «не заметили
отсутствия».

**Правило:** для условной вставки **функции** — проверять по `def `.
Для условной вставки **константы** — по `NAME = `. Для условной вставки
**метода класса** — AST walk по `FunctionDef` внутри `ClassDef`. Grep
по имени — только для «вызов присутствует / отсутствует», не для
«определение присутствует».

**Более широкое правило:** любой `if X not in src` в патч-скрипте
нужно сопроводить комментарием, **что именно** `X` означает —
call, definition, import. Если X встречается в нескольких формах —
проверка должна быть по форме, не по имени.

"""
if "Эпизод 60" not in s:
    s = s.replace(marker_135, episode_60 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §13.4/60")
PY

# ---------- run ----------
echo "=== pytest tests/test_filter_compat.py -v ==="
set +e
uv run pytest tests/test_filter_compat.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "filter_compat fix: rc=$RC"
  exit 1
fi

echo "=== pytest tests/test_cadw_index.py tests/test_iloc.py tests/test_loc.py -v ==="
set +e
uv run pytest tests/test_cadw_index.py tests/test_iloc.py tests/test_loc.py -v
RC2=$?
set -e
if [[ $RC2 -ne 0 ]]; then
  echo "regression: rc=$RC2"
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.4.0d-fix: _apply_filter_with_positions defined (substring check false-negative); +AST guard; §13.4/60"

echo "==> v0.4.0d-fix applied"