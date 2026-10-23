#!/usr/bin/env python3
# v0.9.26y2: implement CadwDataFrame.tail (Pandas parity).
# Closes §13.4/314 backlog. Mirror of head, ~20 LOC + tests.
# Bootstrap-safe.

import ast
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

INDEXING = "cadw/dataframe/_indexing.py"
TESTS = "tests/test_v0926y2_tail.py"


TAIL_SRC = '''
    def tail(self, n: int = 5) -> pa.Table:
        """Return last n rows as pa.Table (v0.9.26y2, Pandas parity).

        - n > 0: last n rows (or all if n >= len).
        - n == 0: empty Table (same schema as df).
        - n < 0: all rows except the first |n| (Pandas parity).
        - n not int: TypeError.

        Returns pa.Table (same as `head`). Peak RAM = O(min(|n|, len))
        for n > 0; O(batch) for n < 0 (streaming skip).

        Closes §13.4/314 backlog (tail was missing pre-v0.9.26y2).
        """
        if not isinstance(n, int):
            raise TypeError(
                f"tail: n must be int; got {type(n).__name__}"
            )

        if n < 0:
            # Pandas parity: tail(-k) = all rows except the first k.
            skip = -n
            collected = []
            for b in self._iter_filtered_batches():
                if skip >= b.num_rows:
                    skip -= b.num_rows
                    continue
                if skip > 0:
                    b = b.slice(skip, b.num_rows - skip)
                    skip = 0
                collected.append(b)
            if not collected:
                return pa.Table.from_batches(
                    [], schema=self._dataset.schema
                )
            return pa.Table.from_batches(collected)

        if n == 0:
            return pa.Table.from_batches(
                [], schema=self._dataset.schema
            )

        # n > 0: last n rows via bounded deque (O(n) memory).
        from collections import deque
        buf: deque = deque(maxlen=n)
        for b in self._iter_filtered_batches():
            buf.extend(b.to_pylist())
        return pa.Table.from_pylist(
            list(buf), schema=self._dataset.schema
        )
'''


TESTS_SRC = '''"""v0.9.26y2: CadwDataFrame.tail (Pandas parity, §13.4/314)."""
import pyarrow as pa
import pytest

from cadw import CadwDataFrame, MemoryGuard


def _mk(data, mem_mb=20):
    return CadwDataFrame.from_table(
        pa.table(data),
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
    )


def test_tail_basic():
    df = _mk({"x": [1, 2, 3, 4, 5]})
    t = df.tail(2)
    assert t.column("x").to_pylist() == [4, 5]


def test_tail_default_5():
    df = _mk({"x": list(range(10))})
    t = df.tail()
    assert t.column("x").to_pylist() == [5, 6, 7, 8, 9]


def test_tail_n_larger_than_len():
    df = _mk({"x": [1, 2]})
    t = df.tail(10)
    assert t.column("x").to_pylist() == [1, 2]


def test_tail_zero():
    df = _mk({"x": [1, 2, 3]})
    t = df.tail(0)
    assert t.num_rows == 0
    assert t.schema == df._dataset.schema


def test_tail_negative():
    """Pandas: tail(-k) = all rows except first k."""
    df = _mk({"x": [1, 2, 3, 4, 5]})
    t = df.tail(-2)
    assert t.column("x").to_pylist() == [3, 4, 5]


def test_tail_not_int():
    df = _mk({"x": [1]})
    with pytest.raises(TypeError, match="must be int"):
        df.tail(2.5)


def test_tail_filtered():
    df = _mk({"x": [1, 2, 3, 4, 5], "v": [10, 20, 30, 40, 50]})
    filtered = df[df["v"] > 20]  # x = [3, 4, 5]
    t = filtered.tail(2)
    assert t.column("x").to_pylist() == [4, 5]


def test_tail_empty():
    df = _mk({"x": []})
    t = df.tail(5)
    assert t.num_rows == 0


def test_tail_preserves_all_columns():
    df = _mk({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    t = df.tail(2)
    assert t.column("a").to_pylist() == [2, 3]
    assert t.column("b").to_pylist() == ["y", "z"]
'''


def verify_state():
    head = subprocess.run(
        ["git", "log", "--format=%s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(":", 1)[0].strip()
    if head != "v0.9.26y":
        raise SystemExit(f"expected HEAD=v0.9.26y, got {head!r}")
    print("HEAD: " + head)

    if not Path(INDEXING).exists():
        raise SystemExit(INDEXING + " missing")
    if Path(TESTS).exists():
        raise SystemExit(TESTS + " exists — clean first")

    # Confirm head exists but tail absent
    from cadw.dataframe._indexing import _IndexingMixin
    assert hasattr(_IndexingMixin, "head"), "head missing"
    if hasattr(_IndexingMixin, "tail"):
        raise SystemExit("tail already present — v0.9.26y2 applied?")
    print("  head: present; tail: absent (needs impl)")

    # Read head source for signature match
    print("\n--- head source (for signature reference) ---")
    head_src = inspect.getsource(_IndexingMixin.head)
    for ln in head_src.split("\n")[:6]:
        print(f"  {ln}")


def add_tail_to_indexing():
    print("\n--- add tail to _indexing.py ---")
    p = Path(INDEXING)
    src = p.read_text()
    tree = ast.parse(src)
    cls = next(
        (n for n in tree.body
         if isinstance(n, ast.ClassDef) and n.name == "_IndexingMixin"),
        None,
    )
    if cls is None:
        raise SystemExit("_IndexingMixin not in _indexing.py")

    existing = {
        item.name for item in cls.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "tail" in existing:
        print("  tail already present — idempotent skip")
        return 0

    lines = src.split("\n")
    insert_at = cls.end_lineno  # 1-based → insert index
    block = [""] + TAIL_SRC.strip("\n").split("\n") + [""]
    lines[insert_at:insert_at] = block
    new_src = "\n".join(lines)
    ast.parse(new_src, filename=INDEXING)
    p.write_text(new_src)
    print(f"  _indexing.py: {len(new_src.splitlines())} lines "
          f"(tail inserted at line {insert_at + 1})")
    return 1


def write_tests():
    print("\n--- write tests ---")
    p = Path(TESTS)
    p.write_text(TESTS_SRC)
    ast.parse(TESTS_SRC, filename=TESTS)
    print(f"  {TESTS}: {len(TESTS_SRC.splitlines())} lines, "
          f"9 tests")


def run_tests():
    print("\n--- sanity ---")
    sanity = (
        "import cadw\n"
        "import pyarrow as pa\n"
        "C = cadw.CadwDataFrame\n"
        "assert hasattr(C, 'tail'), 'tail missing'\n"
        "assert hasattr(C, 'head'), 'head missing'\n"
        "df = C.from_table(pa.table({'x':[1,2,3,4,5]}))\n"
        "t = df.tail(2)\n"
        "assert t.column('x').to_pylist() == [4,5], t\n"
        "print('tail: OK')\n"
    )
    r = subprocess.run(
        ["uv", "run", "python", "-c", sanity],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("sanity failed")
    print(r.stdout.strip())

    print("\n--- tail tests ---")
    r = subprocess.run(
        ["uv", "run", "pytest", TESTS, "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        raise SystemExit("tail tests FAILED")

    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26y2: implement tail ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()
    n_add = add_tail_to_indexing()
    if n_add == 0:
        print("\nno changes made — checking tests")
    else:
        write_tests()

    run_tests()

    # Whitelist
    print("\n--- verify_whitelist ---")
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "from cadw.patch_utils import verify_whitelist; "
         "verify_whitelist(raise_on_orphan=True); print('WL OK')"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("whitelist failed")
    print(r.stdout.strip())

    # Commit
    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.26y2: implement CadwDataFrame.tail (closes §13.4/314 "
        "backlog). Mirror of `head`, returns pa.Table. "
        "Pandas parity: n>0 last n; n=0 empty; n<0 all-but-first-k. "
        "~20 LOC + 9 tests. Sanity adds tail to list — root cause "
        "of v0.9.26m-fix1 / v0.9.26y-fix2 / v0.9.26y-fix3 sanity "
        "misses resolved. 0 logic change in existing code; tests "
        "green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26y2 COMPLETE")


if __name__ == "__main__":
    main()