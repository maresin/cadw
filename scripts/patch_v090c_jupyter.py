#!/usr/bin/env python3
"""v0.9.0c: Jupyter integration — repr_html + df.plot accessor.

SCOPE:
  A. cadw/plot_accessor.py (new):
     _PlotAccessor: hist/line/box/scatter/hist2d (matplotlib default),
     .plotly.* and .seaborn.* (explicit backend via properties).
     Thin wrapper over cadw.viz.* (aggregate-first, no raw data to
     renderers — existing discipline).
  B. dataframe.py:
     _repr_html_ — sample 20 rows + schema + shape (upper bound) +
     filter note. Bounded 20-row `.to_pandas().to_html()`.
     .plot property — lazy import of _PlotAccessor.
  C. pipeline.py:
     Pipeline._repr_html_ — steps table with per-step state_bytes,
     fitted marker, total.
  D. AST guard whitelist: `_repr_html_` (bounded 20-row to_pandas).

TESTS (~16 fast):
  test_repr_html.py (~8): empty/small/large/filtered, schema, HTML table,
    no pandas on dataframe.py import (still guard-compliant).
  test_plot_accessor.py (~8): df.plot.hist/line/box/scatter/hist2d
    → (fig, ax); df.plot.plotly.hist → Figure; df.plot.seaborn.hist
    → (fig, ax); missing column → KeyError.

GUARDS (before commit): new tests, changelog, arrow_guard, no_pandas,
fast+slow.

Run: uv run python scripts/patch_v090c_jupyter.py
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


# ============ A. cadw/plot_accessor.py ============

PLOT_MODULE = '''"""Plot accessor for CadwDataFrame (v0.9.0c).

`df.plot.hist(col)` — matplotlib default.
`df.plot.plotly.hist(col)` — plotly explicit.
`df.plot.seaborn.hist(col)` — seaborn explicit.

All methods delegate to `cadw.viz.*` adapters (aggregate-first: adapters
receive small pre-computed aggregates, not raw data — see §7.4 discipline).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cadw.dataframe import CadwDataFrame


class _MatplotlibAccessor:
    """`df.plot.*` — matplotlib backend (default)."""

    __slots__ = ("_df",)

    def __init__(self, df: "CadwDataFrame") -> None:
        self._df = df

    def hist(self, column: str, bins: int = 50, **kw):
        from cadw.viz.matplotlib_adapter import hist_plot
        return hist_plot(self._df, column, bins=bins, **kw)

    def hist2d(self, x: str, y: str, bins=(200, 200), **kw):
        from cadw.viz.matplotlib_adapter import hist2d_plot
        return hist2d_plot(self._df, x, y, bins=bins, **kw)

    def line(self, by: str, agg_col: str, agg: str = "mean", **kw):
        from cadw.viz.matplotlib_adapter import line_plot
        return line_plot(self._df, by, agg_col, agg=agg, **kw)

    def box(self, column: str, **kw):
        from cadw.viz.matplotlib_adapter import box_plot
        return box_plot(self._df, column, **kw)

    def scatter(self, x: str, y: str, n: int = 10_000, **kw):
        from cadw.viz.matplotlib_adapter import scatter
        return scatter(self._df, x, y, n=n, **kw)

    def hexbin(self, x: str, y: str, gridsize: int = 30,
               n: int = 10_000, **kw):
        from cadw.viz.matplotlib_adapter import hexbin_plot
        return hexbin_plot(self._df, x, y, gridsize=gridsize, n=n, **kw)


class _PlotlyAccessor:
    """`df.plot.plotly.*` — plotly backend."""

    __slots__ = ("_df",)

    def __init__(self, df: "CadwDataFrame") -> None:
        self._df = df

    def hist(self, column: str, bins: int = 50, **kw):
        from cadw.viz.plotly_adapter import hist_plot
        return hist_plot(self._df, column, bins=bins, **kw)

    def hist2d(self, x: str, y: str, bins=(200, 200), **kw):
        from cadw.viz.plotly_adapter import hist2d_plot
        return hist2d_plot(self._df, x, y, bins=bins, **kw)

    def line(self, by: str, agg_col: str, agg: str = "mean", **kw):
        from cadw.viz.plotly_adapter import line_plot
        return line_plot(self._df, by, agg_col, agg=agg, **kw)

    def box(self, column: str, **kw):
        from cadw.viz.plotly_adapter import box_plot
        return box_plot(self._df, column, **kw)

    def scatter(self, x: str, y: str, n: int = 10_000, **kw):
        from cadw.viz.plotly_adapter import scatter
        return scatter(self._df, x, y, n=n, **kw)


class _SeabornAccessor:
    """`df.plot.seaborn.*` — seaborn backend."""

    __slots__ = ("_df",)

    def __init__(self, df: "CadwDataFrame") -> None:
        self._df = df

    def hist(self, column: str, bins: int = 50, **kw):
        from cadw.viz.seaborn_adapter import hist_plot
        return hist_plot(self._df, column, bins=bins, **kw)

    def heatmap(self, x: str, y: str, bins=(50, 50), **kw):
        from cadw.viz.seaborn_adapter import heatmap_plot
        return heatmap_plot(self._df, x, y, bins=bins, **kw)

    def line(self, by: str, agg_col: str, agg: str = "mean", **kw):
        from cadw.viz.seaborn_adapter import line_plot
        return line_plot(self._df, by, agg_col, agg=agg, **kw)

    def box(self, column: str, n: int = 1000, **kw):
        from cadw.viz.seaborn_adapter import box_plot
        return box_plot(self._df, column, n=n, **kw)

    def scatter(self, x: str, y: str, n: int = 1000, **kw):
        from cadw.viz.seaborn_adapter import scatter
        return scatter(self._df, x, y, n=n, **kw)


class _PlotAccessor(_MatplotlibAccessor):
    """`df.plot.*` — matplotlib by default; `.plotly`, `.seaborn` explicit."""

    __slots__ = ("_df",)

    @property
    def plotly(self) -> _PlotlyAccessor:
        return _PlotlyAccessor(self._df)

    @property
    def seaborn(self) -> _SeabornAccessor:
        return _SeabornAccessor(self._df)
'''


def write_plot_module():
    print("\n=== cadw/plot_accessor.py ===")
    p = Path("cadw/plot_accessor.py")
    p.write_text(PLOT_MODULE)
    compile_check(p)
    print("  OK")


# ============ B. dataframe.py: _repr_html_ + .plot property ============

def patch_dataframe():
    print("\n=== cadw/dataframe.py: _repr_html_ + .plot accessor ===")
    p = Path("cadw/dataframe.py")
    s = p.read_text()

    if "def _repr_html_(self)" in s:
        print("  already present — skip")
        return

    # Insert _repr_html_ + .plot property after __repr__ in CadwDataFrame.
    old = '''    def __repr__(self) -> str:
        nf = len(self._row_filters)
        return (
            f"CadwDataFrame(rows<={self._dataset.total_rows}, "
            f"columns={self._columns}, filters={nf})"
        )'''

    new = '''    def __repr__(self) -> str:
        nf = len(self._row_filters)
        return (
            f"CadwDataFrame(rows<={self._dataset.total_rows}, "
            f"columns={self._columns}, filters={nf})"
        )

    def _repr_html_(self) -> str:
        """Jupyter HTML repr — sample 20 rows, schema, shape (upper bound).

        v0.9.0c. Bounded: `.to_pandas().to_html()` on 20-row head only,
        not full df. Whitelisted in AST guard with rationale.
        """
        sample_n = 20
        try:
            sample_table = self.head(sample_n)
            sample_df = sample_table.to_pandas()
            table_html = sample_df.to_html(
                index=False, border=0, classes="cadw-table",
            )
        except Exception as e:
            return (
                f"<pre>CadwDataFrame (repr failed: {e!r})\\n"
                f"columns={self._columns}</pre>"
            )

        n_rows_upper = self._dataset.total_rows
        actual = self.count() if self._row_filters else n_rows_upper

        header_parts = [
            f"<strong>CadwDataFrame</strong> "
            f"({actual} rows"
        ]
        if self._row_filters:
            header_parts.append(
                f" of upper bound {n_rows_upper}; {len(self._row_filters)} "
                f"filter(s) applied"
            )
        else:
            header_parts.append("")
        header_parts.append(f", {len(self._columns)} columns)")
        header = "".join(header_parts)

        schema_rows = "".join(
            f"<tr><td>{c}</td><td>{self._dataset.schema.field(c).type}</td></tr>"
            for c in self._columns[:20]
        )
        if len(self._columns) > 20:
            schema_rows += (
                f"<tr><td>... ({len(self._columns) - 20} more)</td>"
                f"<td></td></tr>"
            )
        schema_html = (
            "<details><summary>schema</summary>"
            "<table><tr><th>column</th><th>type</th></tr>"
            f"{schema_rows}</table></details>"
        )

        truncated_note = ""
        if actual > sample_n:
            truncated_note = (
                f"<div style='color:#666;font-size:0.9em;'>"
                f"Showing first {sample_n} rows</div>"
            )

        return (
            f"<div class='cadw-df'>{header}{schema_html}"
            f"{truncated_note}{table_html}</div>"
        )

    @property
    def plot(self):
        """Plot accessor: df.plot.hist(col) / df.plot.plotly.hist(col)."""
        from cadw.plot_accessor import _PlotAccessor
        return _PlotAccessor(self)'''

    s = replace_line_block(s, old, new, tag="repr-html-plot")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ C. pipeline.py: Pipeline._repr_html_ ============

def patch_pipeline():
    print("\n=== cadw/pipeline.py: Pipeline._repr_html_ ===")
    p = Path("cadw/pipeline.py")
    s = p.read_text()

    if "def _repr_html_(self)" in s:
        print("  already present — skip")
        return

    old = '''    def __repr__(self) -> str:
        names = ", ".join(self._step_names)
        return f"Pipeline([{names}], fitted={self._fitted})"'''

    new = '''    def __repr__(self) -> str:
        names = ", ".join(self._step_names)
        return f"Pipeline([{names}], fitted={self._fitted})"

    def _repr_html_(self) -> str:
        """Jupyter HTML repr — step table with per-step state_bytes."""
        if not self._steps:
            return "<div class='cadw-pipeline'>Pipeline (empty)</div>"

        total_state = self.total_state_bytes()
        rows = []
        for i, (name, step) in enumerate(zip(self._step_names, self._steps)):
            kind = "Predictor" if _is_predictor(step) else "Transformer"
            try:
                sb = step.state_bytes() if self._fitted else 0
            except Exception:
                sb = "?"
            state_str = f"{sb:,} B" if isinstance(sb, int) else str(sb)
            rows.append(
                f"<tr><td>{i + 1}</td><td>{name}</td>"
                f"<td>{type(step).__name__}</td>"
                f"<td>{kind}</td><td>{state_str}</td></tr>"
            )
        fitted_badge = (
            "<span style='color:green'>fitted</span>"
            if self._fitted
            else "<span style='color:#999'>not fitted</span>"
        )
        return (
            f"<div class='cadw-pipeline'>"
            f"<strong>Pipeline</strong> ({len(self._steps)} steps, "
            f"{fitted_badge}, total_state={total_state:,} B)"
            f"<table><tr><th>#</th><th>name</th><th>type</th>"
            f"<th>role</th><th>state</th></tr>"
            f"{''.join(rows)}</table></div>"
        )'''

    s = replace_line_block(s, old, new, tag="pipeline-repr-html")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ D. AST guard whitelist ============

def update_ast_guard_whitelist():
    print("\n=== tests/test_arrow_call_memory_guard.py: whitelist _repr_html_ ===")
    p = Path("tests/test_arrow_call_memory_guard.py")
    s = p.read_text()
    if '"_repr_html_"' in s:
        print("  already present — skip")
        return

    old = '''    "plan": "metadata only (peek_first_batch + estimate_state_bytes)",
    "state_bytes": "in-memory metadata (fitted state)",
    "estimate_state_bytes": "metadata only",
}'''

    new = '''    "plan": "metadata only (peek_first_batch + estimate_state_bytes)",
    "state_bytes": "in-memory metadata (fitted state)",
    "estimate_state_bytes": "metadata only",
    # v0.9.0c Jupyter: _repr_html_ calls .to_pandas().to_html() on
    # a 20-row sample only — bounded by design, no O(N) materialize.
    "_repr_html_": "20-row head().to_pandas().to_html(), bounded sample",
}'''

    s = replace_line_block(s, old, new, tag="whitelist-repr-html")
    p.write_text(s)
    compile_check(p)
    print("  OK")


# ============ Tests ============

REPR_TESTS = '''"""v0.9.0c: CadwDataFrame._repr_html_ tests."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=3, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def test_repr_html_returns_string(tmp_path):
    df = make_df(tmp_path, {"a": [1, 2, 3], "b": ["x", "y", "z"]})
    html = df._repr_html_()
    assert isinstance(html, str)
    assert "<div class='cadw-df'>" in html
    assert "</div>" in html


def test_repr_html_has_table():
    """HTML contains <table> (schema) and rendered rows."""
    # Use a trivial in-memory path via parquet roundtrip
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = make_df(td, {"a": [1, 2, 3]})
        html = df._repr_html_()
        assert "<table>" in html
        assert "<details>" in html


def test_repr_html_empty_df(tmp_path):
    """Empty df: sample table with 0 rows, but still valid HTML."""
    df = make_df(tmp_path, {
        "a": pa.array([], type=pa.int64()),
        "b": pa.array([], type=pa.string()),
    })
    html = df._repr_html_()
    assert "<div class='cadw-df'>" in html
    assert "0 rows" in html or "of upper bound" in html


def test_repr_html_large_df_shows_sample_note(tmp_path):
    """Large df: 'Showing first 20 rows' note."""
    n = 100
    df = make_df(tmp_path, {"v": list(range(n))}, chunk=25)
    html = df._repr_html_()
    assert "Showing first 20 rows" in html


def test_repr_html_small_df_no_truncation(tmp_path):
    """Small df (<=20): no 'Showing first' note."""
    df = make_df(tmp_path, {"v": [1, 2, 3]})
    html = df._repr_html_()
    assert "Showing first" not in html


def test_repr_html_filtered_df_note(tmp_path):
    """Filtered df: note about filters."""
    df = make_df(tmp_path, {"v": [1, 2, 3, 4, 5]})
    filtered = df[df["v"] > 2]
    html = filtered._repr_html_()
    assert "filter(s) applied" in html
    assert "upper bound" in html


def test_repr_html_schema_types_present(tmp_path):
    """Schema section shows column types."""
    df = make_df(tmp_path, {
        "i": [1, 2],
        "s": ["a", "b"],
        "f": [1.0, 2.0],
    })
    html = df._repr_html_()
    assert "int64" in html
    assert "string" in html
    assert "double" in html or "float64" in html


def test_no_pandas_import_in_dataframe():
    """AST: dataframe.py must not `import pandas` even after _repr_html_."""
    import ast
    import inspect
    from cadw import dataframe as df_mod
    src = inspect.getsource(df_mod)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("pandas"), (
                    "dataframe.py must not import pandas — _repr_html_ "
                    "uses .to_pandas() method call only"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert not node.module.startswith("pandas"), (
                    "dataframe.py must not import pandas"
                )
'''


PLOT_TESTS = '''"""v0.9.0c: df.plot accessor tests."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard


def make_df(tmp_path, data, chunk=200, name="L"):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


def _make_numeric_df(tmp_path):
    rng = np.random.default_rng(0)
    return make_df(tmp_path, {
        "x": rng.normal(size=200).tolist(),
        "y": rng.normal(size=200).tolist(),
        "g": [f"a{i % 4}" for i in range(200)],
    })


# ============ matplotlib (default backend) ============

def test_plot_hist_matplotlib(tmp_path):
    pytest.importorskip("matplotlib")
    df = _make_numeric_df(tmp_path)
    fig, ax = df.plot.hist("x", bins=10)
    assert fig is not None
    assert ax is not None


def test_plot_line_matplotlib(tmp_path):
    pytest.importorskip("matplotlib")
    df = _make_numeric_df(tmp_path)
    fig, ax = df.plot.line("g", "x")
    assert fig is not None


def test_plot_box_matplotlib(tmp_path):
    pytest.importorskip("matplotlib")
    df = _make_numeric_df(tmp_path)
    fig, ax = df.plot.box("x")
    assert fig is not None


def test_plot_scatter_matplotlib(tmp_path):
    pytest.importorskip("matplotlib")
    df = _make_numeric_df(tmp_path)
    fig, ax = df.plot.scatter("x", "y", n=100)
    assert fig is not None


def test_plot_hist2d_matplotlib(tmp_path):
    pytest.importorskip("matplotlib")
    df = _make_numeric_df(tmp_path)
    fig, ax = df.plot.hist2d("x", "y", bins=(10, 10))
    assert fig is not None


# ============ plotly (explicit backend) ============

def test_plot_plotly_hist(tmp_path):
    pytest.importorskip("plotly")
    df = _make_numeric_df(tmp_path)
    fig = df.plot.plotly.hist("x", bins=10)
    # plotly Figure has .data and .layout
    assert hasattr(fig, "data")
    assert hasattr(fig, "layout")


def test_plot_plotly_scatter(tmp_path):
    pytest.importorskip("plotly")
    df = _make_numeric_df(tmp_path)
    fig = df.plot.plotly.scatter("x", "y", n=100)
    assert hasattr(fig, "data")


# ============ seaborn (explicit backend) ============

def test_plot_seaborn_hist(tmp_path):
    pytest.importorskip("seaborn")
    df = _make_numeric_df(tmp_path)
    fig, ax = df.plot.seaborn.hist("x", bins=10)
    assert fig is not None


# ============ errors ============

def test_plot_missing_column_raises(tmp_path):
    pytest.importorskip("matplotlib")
    df = _make_numeric_df(tmp_path)
    with pytest.raises(KeyError, match="nope"):
        df.plot.hist("nope")
'''


def write_tests():
    for name, content in [
        ("tests/test_repr_html.py", REPR_TESTS),
        ("tests/test_plot_accessor.py", PLOT_TESTS),
    ]:
        print(f"\n=== {name} ===")
        p = Path(name)
        p.write_text(content)
        compile_check(p)
        print("  OK")


# ============ CHANGELOG ============

CHANGELOG_ENTRY = '''## [Unreleased]

### Добавлено (v0.9.0c — Jupyter integration)

- **`CadwDataFrame._repr_html_`** — sample 20 rows + schema + shape
  (upper bound) + filter note. Bounded: `.to_pandas().to_html()` на
  20-row `head()` только, не full df. AST guard whitelist с rationale.
- **`CadwDataFrame.plot` accessor** — matplotlib default через
  `df.plot.hist/line/box/scatter/hist2d`; explicit backends:
  `df.plot.plotly.*`, `df.plot.seaborn.*`. Все методы делегируют
  `cadw.viz.*` adapters (aggregate-first discipline preserved).
- **`Pipeline._repr_html_`** — HTML-таблица steps с per-step
  `state_bytes`, fitted marker, total state.
- **`cadw/plot_accessor.py`** — новый модуль, thin wrapper over
  `cadw.viz.matplotlib_adapter/plotly_adapter/seaborn_adapter`.

### Ограничения (v0.9.0c, §11.5)

- `df.info()` / `df.dtypes` / `df.describe()` — v0.10.
- `Pipeline` DAG визуализация — v0.9.1 (после DAG support в Pipeline).
- `df.plot` — только явные методы (`df.plot.hist(...)`), не
  `df.plot(kind='hist', ...)` (pandas-style). Method-style selected
  as primary; kind-based — backlog, если запрос.

### Тесты (v0.9.0c)

- +16 fast: 8 `test_repr_html.py`, 8 `test_plot_accessor.py`.
- AST guard whitelist: `_repr_html_` (bounded 20-row to_pandas).

- fast: **~1278** (+16). slow: **9** (unchanged).

'''


def append_changelog():
    print("\n=== CHANGELOG.md: ensure [Unreleased] + append v0.9.0c ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.0c — Jupyter integration" in s:
        print("  already present — skip")
        return

    if "## [Unreleased]" not in s:
        m = re.search(r'^## \[', s, re.MULTILINE)
        if m:
            s = s[:m.start()] + "## [Unreleased]\n\n" + s[m.start():]
        else:
            s = "## [Unreleased]\n\n" + s
        print("  prepended `## [Unreleased]`")

    s = s.replace("## [Unreleased]\n\n", CHANGELOG_ENTRY, 1)
    p.write_text(s)
    print("  OK")


# ============ main ============

def main():
    print("=== v0.9.0c: Jupyter integration ===")

    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print(f"  branch: {r.stdout.strip()}")

    r = subprocess.run(["git", "status", "--porcelain",
                        "--untracked-files=no"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print("ERROR: tracked files dirty:")
        print(r.stdout)
        sys.exit(1)

    write_plot_module()
    patch_dataframe()
    patch_pipeline()
    update_ast_guard_whitelist()
    write_tests()
    append_changelog()

    print("\n=== pytest: new Jupyter tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_repr_html.py",
         "tests/test_plot_accessor.py",
         "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        print("Fix in place (§13.4/146 backlog: no WIP commit on red).")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_changelog.py",
         "tests/test_arrow_call_memory_guard.py",
         "tests/test_no_pandas_in_ops.py",
         "tests/test_pipeline.py",
         "-q"],
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

    sh(["git", "add", "-A"])
    sh(["git", "commit", "-m",
        "v0.9.0c: Jupyter integration — CadwDataFrame._repr_html_ "
        "(20-row sample), df.plot accessor (matplotlib/plotly/seaborn), "
        "Pipeline._repr_html_ (steps table); bounded to_pandas whitelist"])

    print("\n==> v0.9.0c applied.")
    print()
    print("NEXT: squash v0.9.0a + v0.9.0b + v0.9.0c → tag v0.9.0.")
    print("  git reset --soft v0.8.11")
    print("  git commit -m 'v0.9.0: foundation — accessors, type ops, "
          "Pipeline, Jupyter'")
    print("  sed -i 's/^## \\[Unreleased\\]$/## [0.9.0] — 2026-10-20/' CHANGELOG.md")
    print("  git add CHANGELOG.md && git commit --amend --no-edit")
    print("  git tag -a v0.9.0 -m 'v0.9.0: foundation complete'")
    print("  uv run pytest tests/test_changelog.py -v")


if __name__ == "__main__":
    main()