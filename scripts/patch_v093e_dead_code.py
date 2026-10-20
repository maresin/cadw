#!/usr/bin/env python3
# v0.9.3e: C1 (S2) dead shadowed code in _local_groupby.
#
# Bug: `_local_groupby` (cadw/ops/groupby.py) contains TWO
# `def _read_for_resplit()` defs at module-nested scope. First is
# shadowed by second. First was intended optimization (reuse in-memory
# `table` when partition file fits but dict overflows, avoiding disk
# re-read). Never activated -> never tested -> dead code.
#
# Current behavior: always re-read from disk (correct, just
# suboptimal). Preserve it. Delete first (dead) def.
# Optimization as backlog 11.6.
#
# Fix via AST (not text-match — Russian em-dash and other unicode in
# comments made text-match fragile in prior v0.9.3a rounds; see
# 13.4/174).
#
# Run: uv run python scripts/patch_v093e_dead_code.py

from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def sh(cmd, check=True):
    print("  $ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().splitlines()[:20]:
            print("    " + line)
    if check and r.returncode != 0:
        print("    STDERR: " + r.stderr.strip()[-1500:])
        sys.exit(r.returncode)
    return r.returncode


def compile_check(path):
    r = subprocess.run(
        ["uv", "run", "python", "-c",
         "import py_compile; py_compile.compile('" + str(path) +
         "', doraise=True)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("COMPILE FAIL: " + str(path))
        print(r.stderr[-800:])
        sys.exit(1)
    print("  compile OK: " + str(path))


# ============ Change 1: delete shadowed def via AST ============

def delete_shadowed_def():
    print("\n=== groupby.py: delete shadowed _read_for_resplit ===")
    p = Path("cadw/ops/groupby.py")
    text = p.read_text()
    lines = text.split("\n")

    # Parse whole file
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        raise SystemExit("groupby.py parse failed: " + str(e))

    # Find _local_groupby top-level FunctionDef
    local_gb_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_local_groupby":
            local_gb_node = node
            break
    if local_gb_node is None:
        raise SystemExit("_local_groupby FunctionDef not found")

    # Find all _read_for_resplit FunctionDef nodes inside
    rfrs = []
    for node in ast.walk(local_gb_node):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "_read_for_resplit"):
            rfrs.append(node)
    rfrs.sort(key=lambda n: n.lineno)

    if len(rfrs) == 0:
        print("  no _read_for_resplit found - unexpected, skip")
        return
    if len(rfrs) == 1:
        print("  only 1 _read_for_resplit - already fixed, skip")
        return
    if len(rfrs) > 2:
        print("  WARNING: " + str(len(rfrs)) + " defs found (>2), "
              "deleting first, keeping rest")

    first = rfrs[0]
    # AST lineno/end_lineno are 1-indexed; end_lineno is inclusive.
    # Convert to 0-indexed slice [start : end_excl]:
    #   start = first.lineno - 1
    #   end_excl = first.end_lineno
    start_idx = first.lineno - 1
    end_excl = first.end_lineno

    print("  first def: lines " + str(first.lineno) + "-" +
          str(first.end_lineno))
    print("  second def: lines " + str(rfrs[1].lineno) + "-" +
          str(rfrs[1].end_lineno))
    print("  deleting lines " + str(start_idx + 1) + "-" + str(end_excl) +
          " (0-indexed " + str(start_idx) + ":" + str(end_excl) + ")")
    for i in range(start_idx, end_excl):
        preview = lines[i][:80]
        print("    " + repr(preview))

    # Replace with comment header (same indent level as def)
    comment = [
        "    # v0.9.3e (C1, S2): prior (shadowed) `_read_for_resplit`",
        "    # def removed. It was intended optimization: reuse in-memory",
        "    # `table` when partition file fits but dict overflows,",
        "    # avoiding disk re-read. Deferred to backlog (11.6) -- never",
        "    # activated (second identical-name def shadowed it) and thus",
        "    # never tested. See 13.4/177.",
    ]
    lines[start_idx:end_excl] = comment

    p.write_text("\n".join(lines))
    compile_check(p)
    print("  OK (first def deleted, comment inserted)")


# ============ Change 2: tests ============

TESTS_LINES = [
    '# v0.9.3e: C1 dead code cleanup tests.',
    '# Prior _local_groupby had two identical-name _read_for_resplit',
    '# defs (first shadowed by second). Dead code removed via AST;',
    '# current behavior (re-read from disk) preserved.',
    '',
    'from pathlib import Path',
    '',
    'import pyarrow as pa',
    'import pytest',
    '',
    'from cadw.core.memory_guard import MemoryGuard',
    'from cadw.io.batch_writer import ArrowIPCBatchWriter',
    'from cadw.ops.groupby import _local_groupby',
    '',
    '',
    'def write_partition(path, table, chunk=100):',
    '    w = ArrowIPCBatchWriter(str(path), table.schema)',
    '    try:',
    '        for b in table.to_batches(max_chunksize=chunk):',
    '            w.write_batch(b)',
    '    finally:',
    '        w.close()',
    '    return path',
    '',
    '',
    'def test_exactly_one_read_for_resplit_def():',
    '    # AST: _local_groupby has EXACTLY ONE _read_for_resplit def.',
    '    import ast',
    '    import textwrap',
    '    import importlib',
    '    import inspect',
    '    gb_mod = importlib.import_module("cadw.ops.groupby")',
    '    src = inspect.getsource(gb_mod._local_groupby)',
    '    tree = ast.parse(textwrap.dedent(src))',
    '    count = 0',
    '    for node in ast.walk(tree):',
    '        if (isinstance(node, ast.FunctionDef)',
    '                and node.name == "_read_for_resplit"):',
    '            count += 1',
    '    assert count == 1, (',
    '        "expected exactly 1 _read_for_resplit def in _local_groupby, "',
    '        "got " + str(count)',
    '    )',
    '',
    '',
    'def test_no_shadowed_def_ast():',
    '    # AST: no two FunctionDef nodes with same name in same scope.',
    '    import ast',
    '    import textwrap',
    '    import importlib',
    '    import inspect',
    '    gb_mod = importlib.import_module("cadw.ops.groupby")',
    '    src = inspect.getsource(gb_mod._local_groupby)',
    '    tree = ast.parse(textwrap.dedent(src))',
    '    # Collect names of immediate body FunctionDefs (top-level of method',
    '    # body). Nested defs inside other defs are OK (function-scope).',
    '    method_node = tree.body[0]',
    '    seen = {}',
    '    for node in method_node.body:',
    '        if isinstance(node, ast.FunctionDef):',
    '            if node.name in seen:',
    '                raise AssertionError(',
    '                    "duplicate same-scope def: " + node.name +',
    '                    " (lines " + str(seen[node.name]) + " and " +',
    '                    str(node.lineno) + ")"',
    '                )',
    '            seen[node.name] = node.lineno',
    '',
    '',
    'def test_comment_present_ast():',
    '    # AST: comment marker for v0.9.3e present near old def location.',
    '    import importlib',
    '    import inspect',
    '    gb_mod = importlib.import_module("cadw.ops.groupby")',
    '    src = inspect.getsource(gb_mod._local_groupby)',
    '    assert "v0.9.3e" in src, (',
    '        "expected v0.9.3e comment marker in _local_groupby"',
    '    )',
    '',
    '',
    'def test_dict_overflow_recursion_still_works(tmp_path):',
    '    # Behavioral: dict-overflow recursion path still functional.',
    '    # N=1000, K=1000 unique int32 keys. row_size=8. File ~10 KB.',
    '    # Budget 100 KB -> safety 50 KB. File fits, dict overflow (80 KB',
    '    # > 50 KB) -> hash_shuffle split.',
    '    n = 1000',
    '    table = pa.table({',
    '        "cat": pa.array(list(range(n)), type=pa.int32()),',
    '        "val": pa.array(list(range(n)), type=pa.int32()),',
    '    })',
    '    p = write_partition(tmp_path / "p.arrow", table)',
    '    guard = MemoryGuard(max_memory_bytes=100_000)',
    '    result = _local_groupby(',
    '        p, ["cat"], ["val"], ["count"],',
    '        memory_guard=guard,',
    '        spill_dir=tmp_path,',
    '        size_of_key_entry_bytes=80,',
    '    )',
    '    assert result.num_rows == n',
    '',
]


def write_tests():
    print("\n=== tests/test_groupby_dead_code.py ===")
    p = Path("tests/test_groupby_dead_code.py")
    p.write_text("\n".join(TESTS_LINES) + "\n")
    compile_check(p)
    print("  OK")


# ============ Change 3: CHANGELOG ============

CHANGELOG_LINES = [
    "### Исправлено (v0.9.3e - shadowed _read_for_resplit, S2)",
    "",
    "**C1 (S2, из v0.9.2 audit):** `_local_groupby` содержал",
    "**два** `def _read_for_resplit()` подряд. Первый (intended",
    "оптимизация: reuse in-memory `table` когда partition file fits",
    "но dict overflow — избегать disk re-read) был shadowed вторым",
    "и никогда не выполнялся.",
    "",
    "**Почему это важно:**",
    "- Dead code — будущий читатель думает что optimization работает.",
    "- Optimization никогда не активировалась (второй def перекрыл) →",
    "  никогда не тестировалась → потенциально buggy. Правильно не",
    "  активировать без отдельного prototype.",
    "",
    "**Fix (AST-based, не text-match):**",
    "",
    "- Delete first (dead) `_read_for_resplit`.",
    "- Current behavior (re-read from disk) сохранён — correct, если",
    "  suboptimal.",
    "- Comment marker на месте удалённого def.",
    "",
    "**Почему AST, не text-match:** unicode em-dash в Russian comments",
    "и escape sequences ломали text-match в v0.9.3a (6 round fix).",
    "AST `FunctionDef` nodes — прецизионные `lineno`/`end_lineno`,",
    "устойчивы к content.",
    "",
    "**Backlog (11.6):**",
    "",
    "- Restore optimization: `_read_for_resplit` reuse `table` если",
    "  available (file fits but dict overflow). Prototype needed:",
    "  verify correctness (in-memory split vs disk re-read — должны",
    "  дать same partition content). v0.9.4.",
    "",
    "**Тесты (v0.9.3e):** +4 fast (`test_groupby_dead_code.py`):",
    "exactly 1 `_read_for_resplit` def, no duplicate same-scope defs,",
    "comment marker present, dict-overflow recursion still works.",
    "",
    "- fast: **~1587** (+4). slow: **20** (unchanged).",
    "",
]


def append_changelog():
    print("\n=== CHANGELOG v0.9.3e ===")
    p = Path("CHANGELOG.md")
    s = p.read_text()
    if "v0.9.3e - shadowed" in s:
        print("  already present - skip")
        return
    from cadw.patch_utils import append_unreleased_entry
    entry = "\n".join(CHANGELOG_LINES)
    new_s = append_unreleased_entry(s, entry)
    p.write_text(new_s)
    print("  OK")


# ============ Change 4: DECISIONS 13.4/177 ============

EPISODE_177_LINES = [
    "**Эпизод 177 (shadowed def - dead code + dead optimization).**",
    "v0.9.3e: C1 из v0.9.2 audit.",
    "",
    "**Bug.** `_local_groupby` (cadw/ops/groupby.py):",
    "```python",
    "def _read_for_resplit(): ...    # optimization: reuse `table`",
    "def _read_for_resplit(): ...    # shadows first (identical name)",
    "```",
    "Второй def перекрыл первый. Первый (optimization) никогда не",
    "выполнялся. Optimization не активна (suboptimal disk re-read) и",
    "не тестировалась.",
    "",
    "**Правило (13.4/177, расширение 152, 160, 167):**",
    "",
    "- **Duplicate same-name def в same scope = всегда shadowing**,",
    "  не fallback. Python не «попробует сначала один, потом другой»;",
    "  второй безусловно переопределяет первый.",
    "- **Dead code + dead optimization.** Если optimization shadowed —",
    "  она не работает. Future reader, глядя на первый def, думает",
    "  «optimization есть». Должна быть либо активна (delete second),",
    "  либо удалена (delete first + backlog). Оставлять = misleading.",
    "- **AST guard для same-scope duplicates:** `test_no_shadowed_def_",
    "  ast` — для каждой FunctionDef в теле другой функции: name",
    "  уникален. Extended: ClassDef methods, module-level defs — same",
    "  check.",
    "",
    "**Урок (meta).** Text-match как способ fix'а упал в v0.9.3a",
    "6 раз (unicode em-dash `—` vs ASCII `->`, escape `\\n`, `'''`",
    "vs `\"\"\"` in templates). **AST-based fix** (`ast.walk` +",
    "`node.lineno`/`end_lineno`) — прецизионный и content-agnostic.",
    "Для любого deletion/insertion по структуре кода использовать",
    "AST, не regex/string.",
    "",
    "**Backlog (11.6):**",
    "- Restore optimization: `_read_for_resplit` reuse `table` если",
    "  available. Prototype needed (correctness: in-memory split == disk",
    "  re-read). v0.9.4.",
    "- Extend `test_no_shadowed_def_ast` на весь `cadw/` — grep AST",
    "  на duplicate top-level defs, duplicate method names per class.",
    "  Одна проверка всех — cheap, high-value.",
    "",
    "**Связанные:** 13.4/152 (stale), 160 (stale regex round 5),",
    "167 (partial migration).",
    "",
]


def append_episode_13_4_177():
    print("\n=== DECISIONS 13.4/177 ===")
    p = Path("DECISIONS.md")
    s = p.read_text()
    from cadw.patch_utils import find_section_anchor

    if "Эпизод 177" in s:
        print("  13.4/177 already present - skip")
        return
    pos = find_section_anchor(s, "13.5")
    if pos is None:
        raise SystemExit("DECISIONS: 13.5 not found")
    block = "\n".join(EPISODE_177_LINES)
    s = s[:pos] + block + "\n" + s[pos:]
    p.write_text(s)
    print("  added 13.4/177")


# ============ main ============

def main():
    print("=== v0.9.3e: C1 dead code ===")
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                       capture_output=True, text=True, check=True)
    print("  branch: " + r.stdout.strip())

    delete_shadowed_def()
    write_tests()
    append_changelog()
    append_episode_13_4_177()

    print("\n=== pytest: dead code tests ===")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/test_groupby_dead_code.py", "-v"],
        timeout=300,
    )
    if r.returncode != 0:
        print("\n*** NEW TESTS FAILED ***")
        sys.exit(1)

    print("\n=== pytest: groupby regression ===")
    r = subprocess.run(
        ["uv", "run", "pytest",
         "tests/test_groupby_v2.py",
         "tests/test_groupby_per_group.py",
         "tests/test_mustfix_dict_check.py",
         "tests/test_mustfix_local_groupby.py",
         "tests/test_mustfix_skew.py",
         "-q"],
        timeout=600,
    )
    if r.returncode != 0:
        print("\n*** GROUPBY REGRESSION FAILED ***")
        sys.exit(1)

    print("\n=== pytest: guards ===")
    from cadw.patch_utils import assert_guards_green
    assert_guards_green([
        "tests/test_changelog.py",
        "tests/test_arrow_call_memory_guard.py",
        "tests/test_no_pandas_in_ops.py",
        "tests/test_patch_utils.py",
    ])

    print("\n=== pytest fast ===")
    r = subprocess.run(["uv", "run", "pytest", "-q", "-m", "not slow"],
                       timeout=1800)
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
        "v0.9.3e: C1 dead shadowed _read_for_resplit removed "
        "(AST-based, not text-match) — current disk-reread behavior "
        "preserved; intended in-memory optimization deferred to 11.6 "
        "backlog; 13.4/177 (shadowed def = dead code + dead optimization; "
        "AST for structural edits)"])

    print("\n==> v0.9.3e applied.")
    print("==> v0.9.3 COMPLETE (5 patches: a,b,c,d,e).")
    print("NEXT: v0.9.3 release (CHANGELOG bump + tag)")


if __name__ == "__main__":
    main()
