#!/usr/bin/env python3
"""v0.9.2a-fix2: log_loss validation + E2E test semantics split.

BUG (§13.4/163, validation gap + test-authoring):
  E2E test uses continuous y (rng.normal) for log_loss. Formula
  `-(y·log(p) + (1-y)·log(1-p))` assumes y ∈ {0, 1}. With y=2.0:
  `-(2·log(p) + (1-2)·log(1-p))` = negative — mathematically invalid.

  TWO issues:
  1. Production: MetricsOp doesn't validate y ∈ {0, 1} for log_loss.
     Silent garbage output.
  2. Test-authoring: E2E test mixes regression semantics (continuous y)
     and classification semantics (binary y) in one call.

FIX:
  1. MetricsOp.update (log_loss branch): validate y ∈ {0, 1}; raise
     ValueError on first invalid batch with clear message. Cost
     O(batch) check, honest per-batch validation.
  2. E2E test: split into regression (continuous y) + log_loss (binary
     y_bin from `(y > 0)`). Two metrics() calls. Both verified.
  3. New fast test: test_metrics_log_loss_rejects_continuous_y.
  4. §13.4/163 — aggregate formulas need domain validation at Op boundary.
  5. Dogfood: no stale assertions after validation added (existing fast
     log_loss tests use binary y already — pass).

GUARDS: metrics tests + E2E + full suite + guards.

Run: uv run python scripts/patch_v092a_fix2.py
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


# ============ Fix 1: log_loss y ∈ {0, 1} validation ============

def add_log_loss_validation():
    print("\n=== cadw/ops/metrics.py: log_loss y∈{0,1} validation ===")
    p = Path("cadw/ops/metrics.py")
    s = p.read_text()

    old = '''        # Log loss (binary; y must be {0,1}, p in [0,1])
        if self.y_proba_col is not None:
            y = _to_f64(batch.column(self.y_true_col))
            p = _to_f64(batch.column(self.y_proba_col))
            mask = ~np.isnan(y) & ~np.isnan(p)
            if mask.any():
                y_m = y[mask]
                p_m = p[mask]
                eps = 1e-15
                p_c = np.clip(p_m, eps, 1.0 - eps)
                ll = -(y_m * np.log(p_c)
                       + (1.0 - y_m) * np.log(1.0 - p_c))
                state["sum_log_loss"] += float(ll.sum())
                state["n_proba"] += int(mask.sum())

        return state'''

    new = '''        # Log loss (binary; y must be {0,1}, p in [0,1])
        if self.y_proba_col is not None:
            y = _to_f64(batch.column(self.y_true_col))
            p = _to_f64(batch.column(self.y_proba_col))
            mask = ~np.isnan(y) & ~np.isnan(p)
            if mask.any():
                y_m = y[mask]
                p_m = p[mask]
                # §13.4/163: validate y ∈ {0, 1} — log_loss formula
                # assumes binary; continuous y gives mathematically
                # invalid (often negative) output silently.
                bad_mask = (y_m != 0.0) & (y_m != 1.0)
                if bad_mask.any():
                    bad_vals = y_m[bad_mask][:5].tolist()
                    raise ValueError(
                        f"MetricsOp: log_loss requires binary y_true "
                        f"∈ {{0, 1}}; got values {bad_vals} "
                        f"(first 5 of {int(bad_mask.sum())} invalid). "
                        f"Use `encode_categorical` for multiclass or "
                        f"threshold predictions. §13.4/163."
                    )
                eps = 1e-15
                p_c = np.clip(p_m, eps, 1.0 - eps)
                ll = -(y_m * np.log(p_c)
                       + (1.0 - y_m) * np.log(1.0 - p_c))
                state["sum_log_loss"] += float(ll.sum())
                state["n_proba"] += int(mask.sum())

        return state'''

    s = replace_line_block(s, old, new, tag="log-loss-validate")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 2: E2E test semantic split ============

def fix_e2e_test():
    print("\n=== tests/test_metrics_e2e.py: split regression + log_loss ===")
    p = Path("tests/test_metrics_e2e.py")
    s = p.read_text()

    old = '''@pytest.mark.slow
def test_metrics_100k_rows(tmp_path):
    """100K rows — metrics reduce independently of N."""
    n = 100_000
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    yp = y + rng.normal(scale=0.3, size=n)
    p = 1.0 / (1.0 + np.exp(-y))  # arbitrary proba in (0,1)

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({"y": y.tolist(), "yp": yp.tolist(), "p": p.tolist()}),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=30),
        spill_dir=tmp_path / "spill",
    )
    res = df.metrics(
        y_true="y", y_pred="yp", y_proba="p",
        metrics=["r2", "rmse", "mae", "log_loss"],
    )

    # Reference (numpy)
    ss_res = float(((y - yp) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ref_r2 = 1.0 - ss_res / ss_tot
    ref_rmse = float(np.sqrt(ss_res / n))
    ref_mae = float(np.abs(y - yp).mean())

    assert res["r2"] == pytest.approx(ref_r2, rel=1e-9)
    assert res["rmse"] == pytest.approx(ref_rmse, rel=1e-9)
    assert res["mae"] == pytest.approx(ref_mae, rel=1e-9)
    assert res["log_loss"] > 0  # proba doesn't match y
'''

    new = '''@pytest.mark.slow
def test_metrics_100k_regression(tmp_path):
    """100K rows — regression metrics (r2/rmse/mae), continuous y.

    §13.4/163: separate regression semantics (continuous y) from
    log_loss semantics (binary y). Prior test mixed both in one call —
    log_loss formula produced invalid negative output for continuous y.
    """
    n = 100_000
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    yp = y + rng.normal(scale=0.3, size=n)

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({"y": y.tolist(), "yp": yp.tolist()}),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=30),
        spill_dir=tmp_path / "spill",
    )
    res = df.metrics(
        y_true="y", y_pred="yp", metrics=["r2", "rmse", "mae"],
    )

    ss_res = float(((y - yp) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ref_r2 = 1.0 - ss_res / ss_tot
    ref_rmse = float(np.sqrt(ss_res / n))
    ref_mae = float(np.abs(y - yp).mean())

    assert res["r2"] == pytest.approx(ref_r2, rel=1e-9)
    assert res["rmse"] == pytest.approx(ref_rmse, rel=1e-9)
    assert res["mae"] == pytest.approx(ref_mae, rel=1e-9)


@pytest.mark.slow
def test_metrics_100k_log_loss_binary(tmp_path):
    """100K rows — log_loss with binary y.

    Binary y_bin = (y > 0); proba p = sigmoid(y) — calibrated-ish.
    """
    n = 100_000
    rng = np.random.default_rng(42)
    y = rng.normal(size=n)
    y_bin = (y > 0).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-y))  # sigmoid(y) — well-calibrated

    src = tmp_path / "src.parquet"
    pq.write_table(
        pa.table({"yb": y_bin.tolist(), "p": p.tolist()}),
        src, row_group_size=5000,
    )
    df = CadwDataFrame.from_parquet(
        src, tmp_path / "ds", chunk_size=5000,
        memory_guard=MemoryGuard(max_memory_mb=30),
        spill_dir=tmp_path / "spill",
    )
    res = df.metrics(y_true="yb", y_proba="p", metrics=["log_loss"])

    # Reference (numpy, matches sklearn)
    eps = 1e-15
    p_c = np.clip(p, eps, 1.0 - eps)
    ref_ll = float(-(y_bin * np.log(p_c)
                     + (1 - y_bin) * np.log(1 - p_c)).mean())

    assert res["log_loss"] == pytest.approx(ref_ll, rel=1e-9)
    assert 0 < res["log_loss"] < 1.0  # binary CE ∈ (0, ∞); sigmoid-calibrated ~0.35
'''

    s = replace_line_block(s, old, new, tag="e2e-split")
    p.write_text(s)
    compile_check(p)
    print("  OK: regression + log_loss separate tests")


# ============ Fix 3: new fast test — reject continuous y ============

def add_reject_test():
    print("\n=== tests/test_metrics.py: log_loss rejects continuous y ===")
    p = Path("tests/test_metrics.py")
    s = p.read_text()

    if "test_metrics_log_loss_rejects_continuous_y" in s:
        print("  already present — skip")
        return

    new_test = '''

def test_metrics_log_loss_rejects_continuous_y(tmp_path):
    """§13.4/163: log_loss requires binary y; continuous y → ValueError.

    Formula `-(y·log(p) + (1-y)·log(1-p))` assumes y ∈ {0, 1}.
    With y=2.0, output is negative — mathematically invalid. Must
    reject with clear message.
    """
    df = make_df(tmp_path, {
        "y": [0.0, 1.0, 2.0],  # contains invalid 2.0
        "p": [0.1, 0.9, 0.5],
    })
    with pytest.raises(ValueError, match="requires binary y_true"):
        df.metrics(y_true="y", y_proba="p", metrics=["log_loss"])
'''

    if not s.endswith("\n"):
        s += "\n"
    s += new_test
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Fix 4: §13.4/163 ============

def append_decisions_163():
    print("\n=== DECISIONS.md: §13.4/163 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    if "Эпизод 163" in s:
        print("  already present — skip")
        return

    from cadw.patch_utils import find_section_anchor
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        print("  WARNING: §13.5 not found — append")
        s = s.rstrip() + "\n\n"
        s += "**§13.4/163 (aggregate formulas need domain validation).** "
        s += "log_loss assumes y∈{0,1}; continuous y → negative garbage. "
        s += "Validate at Op boundary.\n"
        p.write_text(s)
        return

    episode = '''**Эпизод 163 (aggregate formulas need domain validation).**
v0.9.2a: E2E test uses continuous `y` (rng.normal) для `log_loss`.
Формула `-(y·log(p) + (1-y)·log(1-p))` предполагает `y ∈ {0, 1}`.
С `y = 2.0`: `-(2·log(p) + (1-2)·log(1-p))` → **negative** output
(assert `log_loss > 0` fails: got -0.203).

**Две независимые проблемы:**

1. **Production validation gap:** `MetricsOp` не валидирует `y ∈
   {0, 1}` для log_loss. Формула молча даёт мусор. Пользователь
   получает число, не подозревая, что оно неправильное.

2. **Test-authoring semantic mixing:** E2E test использовал **одну
   `y` (continuous)** для regression (r2/rmse/mae, где continuous OK)
   и для log_loss (где binary required). Один столбец — два разных
   семантических контракта. Test «passed» на regression и дал garbage
   на log_loss.

**Fix (v0.9.2a-fix2):**
- `MetricsOp.update` log_loss branch: `bad_mask = (y_m != 0) & (y_m
  != 1)` → raise `ValueError` с первыми 5 invalid values + счётчик +
  рекомендация (`encode_categorical` для multiclass).
- E2E: split на `test_metrics_100k_regression` (continuous y,
  r2/rmse/mae) + `test_metrics_100k_log_loss_binary` (binary y_bin =
  y > 0, proba = sigmoid(y)). Две независимые семантики.
- New fast test: `test_metrics_log_loss_rejects_continuous_y` —
  `y=[0, 1, 2]` → ValueError.

**Правило (§13.4/163, расширение 162):** aggregate formula имеет
**domain contract** (`log_loss`: y ∈ {0,1}, p ∈ (0,1); `r2`: y, y_pred
— continuous; `accuracy`: labels). Contract должен валидироваться
**at Op boundary** — где формула применяется, не у caller. Иначе
silent garbage.

**Метаправило (§13.4/163, расширение 128, /134, /158):** test,
смешивающий семантики (continuous + binary в одном столбце),
не может поймать validation bugs — потому что test сам не осознаёт
границу. Правильно: **один test = одна семантика**. Если нужно
проверить два режима — два test'а.

**Backlog (§11.6):** audit **всех** aggregate-формул на domain
validation:
- `log_loss` — fixed (y ∈ {0,1}).
- `r2/mse/rmse/mae` — no domain (любые finite float).
- `accuracy/f1/...` (v0.9.2b) — labels; validation в K×K confusion
  matrix (class set must match).
- `roc_auc` (v0.9.2c) — binary y; same log_loss validation.
- `groupby().agg` `mean/sum/...` — numeric required (existing
  validation).
- `std/var` — numeric, n >= 1 (existing).
- `PCA` CovarianceOp — numeric, d >= 1 (existing).

**Связанные:** §13.4/128 (test expectations authored), /134 (input
type not Arrow-safe), /158 (E2E respects complexity), /162 (redundant
check masks specific).

'''
    s = s[:pos] + episode + s[pos:]
    p.write_text(s)
    print("  OK: §13.4/163")


def main():
    print("=== v0.9.2a-fix2 ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    add_log_loss_validation()
    fix_e2e_test()
    add_reject_test()
    append_decisions_163()

    print("\n=== pytest: metrics tests (regression) ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_metrics.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** METRICS FAST FAILED ***")
        sys.exit(1)

    print("\n=== pytest: metrics E2E (slow) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_metrics_e2e.py", "-v", "-m", "slow"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** E2E FAILED ***")
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
        "v0.9.2a: metrics — regression + binary log_loss; §13.4/162-163 "
        "(redundant pre-check removed; log_loss validates y ∈ {0,1}; "
        "E2E splits regression vs log_loss semantics)"])

    print("\n==> v0.9.2a-fix2 applied.")
    print("NEXT: v0.9.2b (classification metrics — confusion matrix).")


if __name__ == "__main__":
    main()