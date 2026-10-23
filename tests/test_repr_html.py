"""v0.9.0c: CadwDataFrame._repr_html_ tests."""
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



def _iter_dataframe_sources():
    """v0.9.26h2 (§13.4/292): iterate all dataframe package sources.
    Replaces single-file scans; new extracted modules auto-covered.
    """
    import pathlib as _p
    pkg = _p.Path("cadw/dataframe")
    if pkg.is_dir():
        for f in sorted(pkg.rglob("*.py")):
            yield f, f.read_text()
    else:
        # legacy single-file fallback (should not trigger post-v0.9.26g)
        legacy = _p.Path("cadw/dataframe.py")
        if legacy.exists():
            yield legacy, legacy.read_text()

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
    """AST: dataframe.py must not have MODULE-LEVEL `import pandas`.

    v0.9.1e-fix3 (§13.4/156): updated to module-level only.

    Lazy `import pandas as pd` inside a function (e.g., `_emit_group` in
    `_per_group_apply` — needed to construct pd.Series/pd.DataFrame from
    user fn output) is legitimate: it keeps `import cadw` light, pandas
    only loads when `_per_group_apply` is actually invoked. Only
    module-level import (eager pandas load on `import cadw`) is forbidden.

    Prior version walked the entire AST (`ast.walk`) — brittle for
    legitimate lazy imports. See §13.4/155, /156.
    """
    import ast
    import inspect
    from cadw import dataframe as df_mod
    src = inspect.getsource(df_mod._core)
    tree = ast.parse(src)
    # Only check module-level nodes (top-level body), not nested.
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("pandas"), (
                    "dataframe.py must not MODULE-LEVEL import pandas "
                    "(lazy imports inside functions OK — see §13.4/156)"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert not node.module.startswith("pandas"), (
                    "dataframe.py must not MODULE-LEVEL import pandas"
                )
