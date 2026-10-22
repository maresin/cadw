"""Stage 8: run the README 'Time series' example block verbatim."""
import re
import numpy as np
import pyarrow as pa
import pytest

from cadw import CadwDataFrame


def _readme_example():
    s = open("README.md", encoding="utf-8").read()
    start = s.find("## Time series (v0.9.24)")
    assert start >= 0, "Time series section missing"
    # First python code fence after the section header.
    m = re.search(r"```python\n(.*?)```", s[start:], re.S)
    assert m, "README has no python example in the Time series section"
    return m.group(1)


def test_readme_example_block_executes():
    code = _readme_example()
    # Strip the leading `import cadw` and `from cadw.ts import (...)`
    # since we import explicitly; run the rest against a synthetic df.
    # Build a synthetic df named exactly as the README uses ('df').
    n = 60
    ts = [1700000000000000000 + i * 3600 * 10**9 for i in range(n)]
    rng = np.random.default_rng(0)
    df = CadwDataFrame.from_table(pa.table({
        "y": pa.array(rng.normal(0, 1, n).cumsum() + 100.0,
                      type=pa.float64()),
        "ts": pa.array(ts, type=pa.timestamp("ns")),
        "g": pa.array(["a"] * n, type=pa.string()),
    }))
    ns = {"df": df, "np": np}
    # Execute the code block. print() calls are harmless.
    exec(compile(code, "<README>Time-series", "exec"), ns)
    # Post-conditions: forecasters and THF actually produced results.
    assert "th" in ns
    assert "g" in ns and hasattr(ns["g"], "predict")
    assert ns["m"].predict(7) is not None
    assert len(ns["th"].predict(30)) == 30
