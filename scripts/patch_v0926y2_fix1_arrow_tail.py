#!/usr/bin/env python3
# v0.9.26y2-fix1: rewrite tail Arrow-native (no to_pylist).
# §13.4/336: guard-compliant by construction. Mirror head.

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

INDEXING = "cadw/dataframe/_indexing.py"


TAIL_NEW = '''    def tail(self, n: int = 5) -> pa.Table:
        """Return last n rows as pa.Table (v0.9.26y2, Pandas parity).

        - n > 0: last n rows (or all if n >= len).
        - n == 0: empty Table (same schema as df).
        - n < 0: all rows except the first |n| (Pandas parity).
        - n not int: TypeError.

        Returns pa.Table (same as `head`). Arrow-native — no
        to_pylist (guard-compliant, §13.4/336).

        n > 0: two-pass (count then skip first total-n).
        n < 0: one-pass (skip first |n|).
        """
        if not isinstance(n, int):
            raise TypeError(
                f"tail: n must be int; got {type(n).__name__}"
            )

        schema = pa.schema(
            [self._dataset.schema.field(c) for c in self._columns]
        )

        if n == 0:
            return pa.Table.from_batches([], schema=schema)

        if n < 0:
            skip = -n
        else:
            # n > 0: two-pass — count total rows, skip first total-n
            total = self.count()
            if n >= total:
                skip = 0
            else:
                skip = total - n

        batches: list[pa.RecordBatch] = []
        rows_skipped = 0
        for b in self._iter_filtered_batches():
            if rows_skipped + b.num_rows <= skip:
                rows_skipped += b.num_rows
                continue
            if rows_skipped < skip:
                offset = skip - rows_skipped
                b = b.slice(offset, b.num_rows - offset)
                rows_skipped = skip
            batches.append(b)

        if not batches:
            return pa.Table.from_batches([], schema=schema)
        return pa.Table.from_batches(batches, schema=schema)
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

    src = Path(INDEXING).read_text()
    if "def tail(" not in src:
        raise SystemExit("tail not in _indexing.py — v0.9.26y2 didn't apply")
    if "to_pylist" not in src:
        print("  WARN: no to_pylist found? inspect manually")
    print("  tail present (guard violation — will rewrite)")


def rewrite_tail():
    """Replace existing `def tail(...)` body with Arrow-native."""
    print("\n--- rewrite tail Arrow-native ---")
    p = Path(INDEXING)
    src = p.read_text()
    tree = ast.parse(src)
    cls = next(
        (n for n in tree.body
         if isinstance(n, ast.ClassDef) and n.name == "_IndexingMixin"),
        None,
    )
    if cls is None:
        raise SystemExit("_IndexingMixin not found")

    tail_node = None
    for item in cls.body:
        if (isinstance(item, ast.FunctionDef)
                and item.name == "tail"):
            tail_node = item
            break
    if tail_node is None:
        raise SystemExit("tail method not found in class body")

    start = tail_node.lineno
    if tail_node.decorator_list:
        start = min(d.lineno for d in tail_node.decorator_list)
    end = max(
        getattr(n, "end_lineno", tail_node.lineno)
        for n in ast.walk(tail_node)
    )
    print(f"  current tail: lines {start}-{end} "
          f"({end - start + 1} LOC)")

    lines = src.split("\n")
    new_block = TAIL_NEW.rstrip("\n").split("\n")
    lines[start - 1:end] = new_block

    new_src = "\n".join(lines)
    ast.parse(new_src, filename=INDEXING)
    p.write_text(new_src)
    print(f"  _indexing.py: {len(new_src.splitlines())} lines "
          f"({len(new_block)} LOC tail)")


def run_tests():
    print("\n--- sanity ---")
    sanity = (
        "import cadw\n"
        "import pyarrow as pa\n"
        "C = cadw.CadwDataFrame\n"
        "df = C.from_table(pa.table({'x':[1,2,3,4,5]}))\n"
        "assert df.tail(2).column('x').to_pylist() == [4,5]\n"
        "assert df.tail(0).num_rows == 0\n"
        "assert df.tail(-2).column('x').to_pylist() == [3,4,5]\n"
        "assert df.tail(10).column('x').to_pylist() == [1,2,3,4,5]\n"
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
        ["uv", "run", "pytest", "tests/test_v0926y2_tail.py", "-q"],
        timeout=300,
    )
    if r.returncode != 0:
        raise SystemExit("tail tests FAILED")

    print("\n--- guard test ---")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_v0925d_a.py::test_no_unguarded_or_low_factor_to_pylist",
         "-q"],
        timeout=120,
    )
    if r.returncode != 0:
        raise SystemExit("guard test still fails")

    print("\n--- fast suite ---")
    r = subprocess.run(
        ["uv", "run", "pytest", "-q", "-m", "not slow"],
        timeout=1800,
    )
    if r.returncode != 0:
        raise SystemExit("fast suite FAILED")


def main():
    print("=== v0.9.26y2-fix1: Arrow-native tail ===")
    br = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("branch: " + br)

    verify_state()
    rewrite_tail()
    run_tests()

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

    print("\n--- commit ---")
    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        "v0.9.26y2: implement CadwDataFrame.tail (closes §13.4/314 "
        "backlog) + fix1 (Arrow-native, guard-compliant). "
        "Pandas parity: n>0 last n (two-pass); n=0 empty; n<0 "
        "all-but-first-k (one-pass). Arrow-native via "
        "pa.RecordBatch.slice, no to_pylist (§13.4/336 — "
        "_indexing.py scanned by v0.9.25d-b2 guard). Mirror of "
        "`head` pattern. ~40 LOC + 9 tests. Sanity adds tail to "
        "list — root cause of v0.9.26m-fix1 / v0.9.26y-fix2 / "
        "v0.9.26y-fix3 sanity misses resolved. 0 logic change in "
        "existing code; tests green."
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)

    log = subprocess.run(
        ["git", "log", "--format=%h %s", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("\n" + log)
    print("\nv0.9.26y2 COMPLETE (fix1)")


if __name__ == "__main__":
    main()