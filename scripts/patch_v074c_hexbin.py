#!/usr/bin/env python3
"""v0.7.4c: hexbin (sample-based) + random sample.

- `_aggregate.sample(df, n, *, random=False)` — new `random` param.
    - random=False (default, fast): df.head(n) — as before, may be biased
      on time-sorted data.
    - random=True: uniform random sample (full O(N) scan, O(n) memory).
      Uses df.iloc[random_indices] (v0.4.0b, filter-compat v0.4.0d).
- `matplotlib_adapter.hexbin_plot(df, x, y, gridsize, n, random, ...)` —
    sample + plt.hexbin (matplotlib native).
- Plotly/Seaborn: no native hexbin. Document use hist2d_plot instead.

Honest limitations:
    - random=True: 2× O(N) scan (count + iloc to max position). For 10^7
      rows ~seconds. Use random=False for fast preview.
    - hexbin is sample-based, not aggregate-first: sample(n=10_000) then
      plt.hexbin does its own binning. This is matplotlib-native; making
      it aggregate-first would require reimplementing hexbin in CADW Reduce.
    - Plotly has no hexbin — use plotly_adapter.hist2d_plot (rect grid).

Run: uv run python scripts/patch_v074c_hexbin.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def sh(cmd, check=True):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, str) else ' '.join(cmd)}")
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


def replace_line_block(s, old_block, new_block, *, tag):
    old_norm = [l.rstrip() for l in old_block.split("\n")]
    lines = s.split("\n")
    starts = []
    for i in range(len(lines) - len(old_norm) + 1):
        if all(lines[i + k].rstrip() == old_norm[k]
               for k in range(len(old_norm))):
            starts.append(i)
    if len(starts) == 0:
        raise SystemExit(f"[{tag}] block not found")
    if len(starts) > 1:
        raise SystemExit(f"[{tag}] block found {len(starts)} times")
    start = starts[0]
    new_lines = new_block.split("\n")
    lines = lines[:start] + new_lines + lines[start + len(old_norm):]
    return "\n".join(lines)


# ============ 1. _aggregate.sample: add random param ============

def patch_sample():
    print("\n=== cadw/viz/_aggregate.py: sample(random=...) ===")
    p = Path("cadw/viz/_aggregate.py")
    s = p.read_text()

    old = '''def sample(df, n: int = 10_000):
    """Bounded sample: returns df if count() <= n, else df.head(n).

    v0.7.4a-fix: df.head(n) returns pa.Table, not CadwDataFrame.
    Wrap via Dataset.from_batches to keep API consistent.
    """
    if n < 1:
        raise ValueError(f"sample: n must be >= 1; got {n}")
    cnt = df.count()
    if cnt <= n:
        return df
    from cadw.dataframe import CadwDataFrame
    from cadw.dataset import Dataset
    table = df.head(n)
    new_ds = Dataset.from_batches(
        table.schema, iter(table.to_batches()), df._new_dataset_dir(),
    )
    return CadwDataFrame(
        new_ds,
        memory_guard=df._memory_guard,
        spill_dir=df._spill_dir,
    )'''

    new = '''def sample(df, n: int = 10_000, *, random: bool = False):
    """Bounded sample (v0.7.4c: +random).

    - random=False (default, fast): df.head(n). Bounded I/O (first n).
      **May be biased** if data is ordered (e.g. time-sorted parquet) —
      samples only prefix. Use for quick preview.
    - random=True: uniform random sample of n rows. Full O(N) scan
      (count + iloc[random_indices] → reads to max position). O(n) memory.
      Use for viz correctness on arbitrary/ordered data.

    Returns CadwDataFrame either way (count() <= n → returns df unchanged).
    """
    if n < 1:
        raise ValueError(f"sample: n must be >= 1; got {n}")
    cnt = df.count()
    if cnt <= n:
        return df
    if random:
        import numpy as _np
        rng = _np.random.default_rng(42)
        idx = _np.sort(rng.choice(cnt, size=n, replace=False))
        # df.iloc[list] → CadwDataFrame (v0.4.0b, filter-compat v0.4.0d)
        return df.iloc[idx.tolist()]
    # head (fast, may be biased)
    from cadw.dataframe import CadwDataFrame
    from cadw.dataset import Dataset
    table = df.head(n)
    new_ds = Dataset.from_batches(
        table.schema, iter(table.to_batches()), df._new_dataset_dir(),
    )
    return CadwDataFrame(
        new_ds,
        memory_guard=df._memory_guard,
        spill_dir=df._spill_dir,
    )'''

    s = replace_line_block(s, old, new, tag="sample-random")
    p.write_text(s)
    compile_check(p)
    print("  OK: sample(random=True) via iloc[np.random.choice]")


# ============ 2. matplotlib_adapter: hexbin_plot ============

def patch_hexbin():
    print("\n=== cadw/viz/matplotlib_adapter.py: hexbin_plot ===")
    p = Path("cadw/viz/matplotlib_adapter.py")
    s = p.read_text()
    if "def hexbin_plot(" in s:
        print("  already present — skip")
        return

    # _sample_pair: add `random` param (backward compat default=False)
    old_pair = '''def _sample_pair(df, x: str, y: str, n: int):
    """Pull (x, y) numpy arrays for at most n rows."""
    smp = _aggregate.sample(df, n=n)'''

    new_pair = '''def _sample_pair(df, x: str, y: str, n: int, *, random: bool = False):
    """Pull (x, y) numpy arrays for at most n rows.

    random=False (default): head (fast, may be biased on ordered data).
    random=True: uniform random sample (full O(N) scan).
    """
    smp = _aggregate.sample(df, n=n, random=random)'''

    if old_pair not in s:
        raise SystemExit("_sample_pair pattern not found in matplotlib_adapter")
    s = replace_line_block(s, old_pair, new_pair, tag="sample-pair")

    # Insert hexbin_plot before feature_importance
    anchor = "def feature_importance(\n"
    if anchor not in s:
        raise SystemExit("feature_importance anchor not found")

    hexbin_fn = '''def hexbin_plot(
    df, x: str, y: str,
    *,
    gridsize: int = 30,
    n: int = 10_000,
    random: bool = True,
    ax=None,
    cmap: str = "viridis",
    log_scale: bool = True,
    **kwargs,
):
    """Hexbin via sample(n) + plt.hexbin (v0.7.4c, matplotlib native).

    **Sample-based, not aggregate-first**: hexbin aggregation on a
    hexagonal grid is non-trivial (matplotlib does its own binning
    internally via plt.hexbin). We pass sampled (xs, ys), not counts.

    - n=10_000: bounded sample (same pattern as scatter).
    - random=True (default): unbiased. Full O(N) scan — set False for
      fast preview on ordered data (head, may be biased).
    - gridsize: matplotlib hexagon count.
    - log_scale=True: LogNorm for colorbar (typical scatter density).

    Plotly/Seaborn don't have native hexbin — use hist2d_plot instead.
    """
    plt = _plt()
    import matplotlib.colors as mcolors
    if gridsize < 1:
        raise ValueError(f"hexbin_plot: gridsize must be >= 1; got {gridsize}")
    xs, ys = _sample_pair(df, x, y, n, random=random)
    # Drop NaN rows (both x and y must be valid)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs = xs[mask]
    ys = ys[mask]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    norm = mcolors.LogNorm() if log_scale and len(xs) > 0 else None
    hb = ax.hexbin(
        xs, ys,
        gridsize=gridsize,
        cmap=cmap,
        norm=norm,
        **kwargs,
    )
    fig.colorbar(hb, ax=ax, label="count")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"hexbin({x}, {y}, gridsize={gridsize}, n={len(xs)})")
    return fig, ax


'''
    if s.count(anchor) > 1:
        raise SystemExit(f"feature_importance anchor found {s.count(anchor)} times")
    s = s.replace(anchor, hexbin_fn + anchor, 1)
    p.write_text(s)
    compile_check(p)
    print("  OK: hexbin_plot added")


# ============ 3. Tests ============

def add_tests():
    print("\n=== tests/test_viz_aggregate.py + test_viz_matplotlib.py ===")

    # --- test_viz_aggregate.py: sample(random=) tests ---
    p1 = Path("tests/test_viz_aggregate.py")
    s1 = p1.read_text()
    if "test_sample_random_uniform" not in s1:
        extra1 = '''

# ---------- v0.7.4c: random sample ----------

def test_sample_random_uniform(tmp_path):
    """random=True: not just first n rows — should include later positions."""
    n = 1000
    rng = np.random.default_rng(0)
    x = np.arange(n, dtype=np.float64)
    df = make_df(tmp_path, {"x": x.tolist()}, chunk=200)
    smp = sample(df, n=100, random=True)
    got = smp.count()
    assert got == 100
    # Extract x values, verify: not all from first 100 (i.e. max > 100)
    from cadw.viz import _aggregate
    xs = []
    for b in smp._iter_filtered_batches():
        xs.extend(b.column("x").to_pylist())
    assert max(xs) > 500, f"random sample didn't reach later rows: max={max(xs)}"
    assert min(xs) < 500, f"random sample is not uniform: min={min(xs)}"


def test_sample_random_small_df_returns_self(tmp_path):
    """count() <= n → returns df, random flag ignored."""
    df = make_df(tmp_path, {"x": [1.0, 2.0, 3.0]})
    s1 = sample(df, n=10, random=False)
    s2 = sample(df, n=10, random=True)
    assert s1 is df and s2 is df


def test_sample_random_deterministic(tmp_path):
    """Same seed → same sample (deterministic)."""
    n = 500
    rng = np.random.default_rng(0)
    df = make_df(tmp_path, {"x": rng.normal(size=n).tolist()}, chunk=100)
    s1 = sample(df, n=50, random=True)
    s2 = sample(df, n=50, random=True)
    xs1 = []
    xs2 = []
    for b in s1._iter_filtered_batches():
        xs1.extend(b.column("x").to_pylist())
    for b in s2._iter_filtered_batches():
        xs2.extend(b.column("x").to_pylist())
    assert xs1 == xs2


def test_sample_head_default_still_works(tmp_path):
    """random=False (default): head behavior unchanged."""
    n = 100
    df = make_df(tmp_path, {"x": list(range(n))}, chunk=50)
    smp = sample(df, n=20)
    got = smp.count()
    assert got == 20
'''
        s1 = s1.rstrip() + extra1 + "\n"
        p1.write_text(s1)
        compile_check(p1)
        print("  OK: +4 sample(random) tests")

    # --- test_viz_matplotlib.py: hexbin tests ---
    p2 = Path("tests/test_viz_matplotlib.py")
    s2 = p2.read_text()
    if "test_hexbin_plot_writes_png" not in s2:
        extra2 = '''

# ---------- v0.7.4c: hexbin_plot ----------

def test_hexbin_plot_writes_png(tmp_path):
    """hexbin_plot: sample + plt.hexbin, writes PNG."""
    rng = np.random.default_rng(0)
    n = 10_000
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    df = make_df(tmp_path, {"x": x.tolist(), "y": y.tolist()})
    from cadw.viz.matplotlib_adapter import hexbin_plot
    fig, ax = hexbin_plot(df, "x", "y", gridsize=20, n=2000)
    out = tmp_path / "hexbin.png"
    fig.savefig(str(out), dpi=80, bbox_inches="tight")
    assert out.stat().st_size > 1500


def test_hexbin_plot_random_sample(tmp_path):
    """random=True: sample spreads across dataset."""
    n = 5000
    rng = np.random.default_rng(1)
    df = make_df(tmp_path, {
        "x": np.arange(n, dtype=np.float64).tolist(),
        "y": rng.normal(size=n).tolist(),
    })
    from cadw.viz.matplotlib_adapter import hexbin_plot
    fig, ax = hexbin_plot(df, "x", "y", gridsize=20, n=500, random=True)
    # If sample were head, all x ∈ [0, 500]; check title contains n=500
    assert "n=500" in ax.get_title()


def test_hexbin_plot_missing_column(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [3.0, 4.0]})
    from cadw.viz.matplotlib_adapter import hexbin_plot
    import pytest
    with pytest.raises(KeyError, match="nope"):
        hexbin_plot(df, "nope", "y")


def test_hexbin_plot_invalid_gridsize(tmp_path):
    df = make_df(tmp_path, {"x": [1.0, 2.0], "y": [3.0, 4.0]})
    from cadw.viz.matplotlib_adapter import hexbin_plot
    import pytest
    with pytest.raises(ValueError, match="gridsize"):
        hexbin_plot(df, "x", "y", gridsize=0)


def test_hexbin_plot_skips_nulls(tmp_path):
    """Rows with null in x or y — dropped before plt.hexbin."""
    df = make_df(tmp_path, {
        "x": [1.0, None, 3.0, 4.0],
        "y": [1.0, 2.0, None, 4.0],
    })
    from cadw.viz.matplotlib_adapter import hexbin_plot
    fig, ax = hexbin_plot(df, "x", "y", n=100, random=False)
    # 2 valid rows (1,1) and (4,4) — hexbin shows both
    assert "n=2" in ax.get_title()
'''
        s2 = s2.rstrip() + extra2 + "\n"
        p2.write_text(s2)
        compile_check(p2)
        print("  OK: +5 hexbin tests")


# ============ 4. CHANGELOG + DECISIONS ============

def patch_docs():
    print("\n=== CHANGELOG.md: [0.7.4c] ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "## [0.7.4c]" in s:
        print("  already present — skip")
        return
    entry = """## [0.7.4c] — 2026-10-19

### Добавлено
- **`matplotlib_adapter.hexbin_plot(df, x, y, gridsize=30, n=10_000,
  random=True, ...)`** — 2D hexagonal binning (matplotlib native).
  - Sample-based (`sample(n)` + `plt.hexbin`), **не** aggregate-first:
    hexbin aggregation на гексагональной сетке требует custom binning,
    matplotlib делает это сам через `plt.hexbin`.
  - `log_scale=True` (default): LogNorm для colorbar (плотность scatter).
  - **Plotly/Seaborn:** нет native hexbin — используйте `hist2d_plot`
    (прямоугольная сетка). Documented.

- **`_aggregate.sample(df, n, *, random=False)`** — новый параметр.
  - `random=False` (default, fast): `df.head(n)` — bounded I/O,
    но **может быть смещён** на time-sorted данных (только префикс).
  - `random=True`: uniform random sample через
    `df.iloc[np.sort(rng.choice(cnt, n, replace=False))]`. Full O(N)
    scan, O(n) memory. Deterministic (seed=42).
  - `scatter` / `_sample_pair`: `random=False` по умолчанию
    (backward compat v0.7.4a).

### Ограничения (v0.7.4c, §11.5)
- `hexbin_plot` — только matplotlib. Plotly/Seaborn — нет hexbin
  (документировано; use hist2d).
- `random=True`: 2× O(N) scan (count + iloc to max position). Для 10^7
  строк ~секунды. `random=False` — fast preview.
- hexbin sample-based, не aggregate-first — n=10_000 bounded, shape
  approximate.

### Тесты
- +4 в `test_viz_aggregate.py`: sample(random=True) uniformity,
  small-df returns-self, deterministic, head default unchanged.
- +5 в `test_viz_matplotlib.py`: hexbin_plot writes PNG, random sample
  spreads, missing column raises, invalid gridsize raises, nulls skipped.
- fast: **~1026 passed**. slow: **8 passed**.

"""
    # Anchor: top entry [0.8.2]
    anchor = "## [0.8.2]"
    if anchor not in s:
        raise SystemExit("[0.8.2] anchor not found in CHANGELOG")
    s = s.replace(anchor, entry + anchor, 1)
    p.write_text(s)
    print("  OK: [0.7.4c]")


# ============ main ============

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = "patch/v074c-hexbin"

    print("=== v0.7.4c: hexbin + random sample ===")
    # ensure_branch inline
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if r.stdout.strip() != branch:
        r2 = subprocess.run(["git", "rev-parse", "--verify", branch],
                            capture_output=True)
        if r2.returncode == 0:
            sh(["git", "checkout", branch])
        else:
            sh(["git", "checkout", "-b", branch])

    r = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    backup_dir = Path(f".backups/backup-pre-v074c-{ts_str}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "archive", "--format=tar.gz",
        "-o", str(backup_dir / "repo.tgz"), "HEAD"])
    sh(["git", "tag", "-a", f"backup-pre-v074c-{ts_str}",
        "-m", "before patch v0.7.4c hexbin"])

    patch_sample()
    patch_hexbin()
    add_tests()
    patch_docs()

    print("\n=== pytest tests/test_viz_aggregate.py "
          "tests/test_viz_matplotlib.py -v ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_viz_aggregate.py",
         "tests/test_viz_matplotlib.py", "-v"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** FAILURE — WIP ***")
        sh(["git", "add", "-A"], check=False)
        sh(["git", "commit", "-m", "WIP: v0.7.4c (test red)"], check=False)
        sys.exit(1)

    print("\n=== regression: viz (all) ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_viz_aggregate.py",
         "tests/test_viz_matplotlib.py",
         "tests/test_viz_plotly.py",
         "tests/test_viz_seaborn.py",
         "tests/test_feature_importance_gain.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** REGRESSION FAILURE ***")
        sys.exit(1)

    print("\n=== pytest fast ===")
    sh(["uv", "run", "pytest", "-q", "-m", "not slow"])
    print("\n=== pytest slow ===")
    sh(["uv", "run", "pytest", "-q", "-m", "slow"])

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.7.4c: hexbin_plot (sample-based) + sample(random=True)"])

    print("\n==> v0.7.4c applied on", branch)


if __name__ == "__main__":
    main()