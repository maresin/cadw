"""v0.9.34 B1: refill linear (fixed 6fa5e49)."""
import ast, inspect, textwrap, time
import pyarrow as pa
from cadw.ops.sort import _RunStream


def test_uses_parts_list():
    src = textwrap.dedent(inspect.getsource(_RunStream.refill))
    fn = ast.parse(src).body[0]
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    code = chr(10).join(ast.unparse(n) for n in body)
    assert "parts.append" in code
    assert "concat_tables(parts)" in code
    assert "concat_tables([self.buffer" not in code


def test_bench_linear_vs_quadratic():
    N = 5000
    ts = [pa.table({"x": [i]}) for i in range(N)]
    t0 = time.perf_counter()
    acc = ts[0]
    for t in ts[1:]:
        acc = pa.concat_tables([acc, t])
    old = time.perf_counter() - t0
    t0 = time.perf_counter()
    pa.concat_tables(ts)
    new = time.perf_counter() - t0
    assert new * 3 < old, (old, new)
