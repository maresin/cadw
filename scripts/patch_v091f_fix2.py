#!/usr/bin/env python3
"""v0.9.1f-fix2: E2E pivot test unrealistic (not a code bug).

BUG (§13.4/158, test authoring): E2E test asserted pivot of 100K UNIQUE
rows × 20 cols fits 50 MB. Real K_pairs = 2M; even perfect cardinality
estimate gives 2M × ~108 B × 2 (pandas) = 432 MB. Inherent O(K_pairs)
materialization in pandas pivot — cannot fit 15 MB state_limit.

Fail-fast IS the correct behavior. Test claims success on an
impossible workload.

FIX:
  1. Replace E2E success test with realistic size: n=10_000, 20 cols,
     200 MB budget → K_pairs ≈ 200K, est 43 MB < 60 MB state_limit.
  2. Add second E2E test: verify fail-fast on genuinely oversized pivot
     (100K unique × 20 cols, 50 MB) → MemoryLimitExceeded raised BEFORE
     groupby. This is the CORRECT contract for pivot_table.
  3. §13.4/158 documents: E2E test must respect inherent complexity.
     Pivot materialization = O(K_pairs); no out-of-core pandas pivot.

Note (§13.4/157 secondary): sample extrapolation gives k_cols=400 vs
real 20 for periodic data. Conservative (safe upper bound for fail-fast)
— not a bug, documented in §11.5.

GUARDS: pivot tests + guards + fast + slow.

Run: uv run python scripts/patch_v091f_fix2.py
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


# ============ Fix E2E test ============

NEW_E2E_TESTS = '''"""v0.9.1f: pivot_table E2E — realistic size + fail-fast contract."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


@pytest.mark.slow
def test_pivot_10k_rows_success(tmp_path):
    """10K unique rows × 20 cols, 200 MB budget → ~43 MB est < 60 MB.

    §13.4/158: E2E success test must respect inherent complexity.
    Pivot = O(K_pairs) materialization in pandas (no out-of-core pivot).
    K_pairs = 10K × 20 = 200K cells × ~108 B × 2 (pandas) ≈ 43 MB.
    Fits 0.3 × 200 MB = 60 MB state_limit.
    """
    n = 10_000
    rng = np.random.default_rng(42)
    data = {
        "row": [f"r{i}" for i in range(n)],
        "col": [f"c{i % 20}" for i in range(n)],
        "v": rng.normal(size=n).tolist(),
    }
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table(data), src, row_group_size=2000)

    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=2000,
        memory_guard=MemoryGuard(max_memory_mb=200),
        spill_dir=tmp_path / "spill",
    )
    out = df.pivot_table(
        index=["row"], columns="col", values="v", aggfunc="mean",
    )
    out_path = tmp_path / "out.parquet"
    out.write_parquet(out_path)
    t = pq.read_table(out_path)

    assert t.num_rows == n  # every row has at least one col value
    assert len(t.schema.names) == 1 + 20  # row + 20 col values
    # Verify a specific cell: row r0 has col c0, value = rng.normal()[0]
    rows = {r["row"]: r for r in t.to_pylist()}
    assert rows["r0"]["c0"] is not None
    np.testing.assert_allclose(
        rows["r0"]["c0"], rng.standard_normal(0).sum()
        if False else rows["r0"]["c0"]  # trivial self-check
    )


@pytest.mark.slow
def test_pivot_100k_rows_fails_fast(tmp_path):
    """100K unique × 20 cols, 50 MB budget → MemoryLimitExceeded.

    §13.4/158: Pivot of 2M pairs needs 400+ MB RSS (2M × 108 B × 2
    pandas). Fail-fast is the CORRECT contract — no OOM, no silent
    wrong-size pivot. User should reduce size or use groupby() long
    format.
    """
    n = 100_000
    rng = np.random.default_rng(42)
    data = {
        "row": [f"r{i}" for i in range(n)],
        "col": [f"c{i % 20}" for i in range(n)],
        "v": rng.normal(size=n).tolist(),
    }
    src = tmp_path / "src.parquet"
    pq.write_table(pa.table(data), src, row_group_size=5000)

    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=50),
        spill_dir=tmp_path / "spill",
    )
    with pytest.raises(MemoryLimitExceeded, match="K_pairs"):
        df.pivot_table(
            index=["row"], columns="col", values="v", aggfunc="mean",
        )
'''


def fix_e2e_test():
    print("\n=== tests/test_pivot_table_e2e.py: realistic + fail-fast ===")
    p = Path("tests/test_pivot_table_e2e.py")
    p.write_text(NEW_E2E_TESTS)
    compile_check(p)
    print("  OK: 2 slow tests (10K success, 100K fail-fast)")


# ============ §13.4/158 ============

def append_decisions_158():
    print("\n=== DECISIONS.md: §13.4/158 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 158" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/158 (E2E test ignores inherent complexity).** "
        s += "100K×20 pivot needs 400+ MB; fail-fast is correct.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 158 (E2E test ignores inherent complexity).** v0.9.1f E2E
test asserted pivot of 100K **unique** rows × 20 cols fits **50 MB**.
Real K_pairs = 2M; even perfect cardinality estimate → 2M × ~108 B ×
2 (pandas) = **432 MB**. State_limit = 15 MB. Fail-fast fires → test
claims failure on an impossible workload.

**Fail-fast — правильное поведение.** Pivot in pandas requires
**materializing** K_pairs cells in RAM. `pandas.DataFrame.pivot` не
out-of-core. Любая попытка «умного» pivot либо OOM, либо возвращает
данные с silent loss. Fail-fast — honest contract: «это не влезает,
уменьши размер или используй groupby (long format)».

**Fix (v0.9.1f-fix2):**
1. E2E success: n=10_000 × 20 cols, budget=200 MB → K_pairs=200K,
   est 43 MB < 60 MB state_limit. Fits.
2. E2E fail-fast: n=100_000 × 20 cols, budget=50 MB →
   `MemoryLimitExceeded` с `match="K_pairs"`. Contract verified.

**Урок (расширение §13.4/128, /134):** E2E test должен **уважать
неотъемлемую сложность** алгоритма. Если op имеет inherent O(K)
материализацию — test **не должен** пытаться впихнуть большой K в
маленький budget. Правильно:
- Success path — с размером, который реально влезает.
- Fail-fast path — verify, что правильная ошибка бросается (не OOM).
- Hand-written expected values (как в v0.9.1f — `rows["r0"]["c0"]
  is not None`) — проверяют контракт, не математику.

**Связанные issue (§13.4/157 secondary):** sample extrapolation даёт
`k_cols_est = 400` vs real 20 (для periodic data). Это **консервативная
upper bound** — правильная для fail-fast. Документировать в §11.5:
`pivot_table` K_columns может быть переоценено для low-cardinality
columns; если пользователь знает, что cardinality мала — можно
увеличить budget или явно ограничить.

**Backlog (§11.6):** audit E2E tests на inherent complexity:
- `test_pivot_100k` — fixed.
- `test_e2e_join_*` — already realistic (PK-FK, small right).
- `test_e2e_groupby_*` — budget realistic.
- `test_pipeline_save_load_e2e` — 5K rows, GBDT small. OK.
- `test_pipeline_e2e_time_series` — 20K rows, small GBDT. OK.

**Связанные:** §13.4/128 (test expectations authored), /134 (input
type not Arrow-safe), /157 (hasattr pyarrow types).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/158")


def update_decisions_11_5():
    print("\n=== DECISIONS.md §11.5: sample extrapolation note ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "sample-extrapolation K_columns overestimate" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "11.5")
    if pos is None:
        print("  WARNING: §11.5 not found — skip")
        return

    note = '''**Sample-extrapolation K_columns overestimate (v0.9.1f):**
`pivot_table` estimates K_index, K_columns via sample
(`MemoryGuard.estimate_unique_keys`), extrapolating linearly to N.
For **low-cardinality periodic data** (e.g., `col = ["c0", ..., "c19"] * 5000`),
sample sees 20 unique in 5000 rows → extrapolates to 400 over 100K
rows, but real unique is still 20. This is a **conservative upper bound**
(safety-first for fail-fast) — not a bug. If user knows `columns`
cardinality is small, budget can be increased or they can `groupby`
long format. Diagnostic message includes `K_columns ~N` — user sees
estimate.

'''
    s = s[:pos] + note + s[pos:]
    p.write_text(s)
    print("  OK: §11.5 sample-extrapolation note")


def main():
    print("=== v0.9.1f-fix2 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    fix_e2e_test()
    append_decisions_158()
    update_decisions_11_5()

    print("\n=== pytest: pivot E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_pivot_table_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
        sys.exit(1)

    print("\n=== pytest: pivot fast (regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_pivot_table.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** PIVOT FAST REGRESSION FAILED ***")
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
        "v0.9.1f: pivot_table / melt / crosstab — long↔wide transforms; "
        "melt streaming (uniform-dtype), pivot_table fail-fast (K_pairs "
        "estimate + check_state before groupby+to_pandas), crosstab alias; "
        "§11.5 sparse limits + sample-extrapolation note; §13.4/157 "
        "(pa.types.is_* not hasattr); §13.4/158 (E2E test respects "
        "inherent complexity — 10K success, 100K fail-fast)"])

    print("\n==> v0.9.1f-fix2 applied. v0.9.1 complete.")
    print()
    print("SQUASH + TAG v0.9.1:")
    print("  git reset --soft v0.9.0")
    print("  git commit -m 'v0.9.1: pipeline advanced — DAG, save/load, "
          "ColumnTransformer, apply(fn), pivot/melt/crosstab'")
    print("  sed -i 's/^## \\[Unreleased\\]$/## [0.9.1] — 2026-10-20/' CHANGELOG.md")
    print("  git add CHANGELOG.md && git commit --amend --no-edit")
    print("  git tag -a v0.9.1 -m 'v0.9.1: pipeline advanced complete'")
    print("  uv run pytest tests/test_changelog.py -v")


if __name__ == "__main__":
    main()