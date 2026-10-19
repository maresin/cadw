#!/usr/bin/env bash
# patch_11b2a_join_left: hash_join(how='inner'|'left') + merge(how='left').
# Не трогает память/planner — это 11-B-2b. E2E — 11-B-2c.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-11b2a-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_11b2a"

BR="patch/11b2a-join-left"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- cadw/ops/join.py: hash_join(how) ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/join.py")
s = p.read_text()

# 1. Docstring модуля
s = s.replace(
    '"""\nInner join (v0.2.1).',
    '"""\nHash join: inner (v0.2.1), left (v0.2.2).',
    1,
)

# 2. _empty_result_schema: nullable=True (важно для how='left' с NaN)
s = s.replace(
    '''    fields: list[pa.Field] = []
    for c in on:
        fields.append(left_schema.field(c))
    for f in left_rest:
        name = f.name + suffixes[0] if f.name in overlap else f.name
        fields.append(pa.field(name, f.type))
    for f in right_rest:
        name = f.name + suffixes[1] if f.name in overlap else f.name
        fields.append(pa.field(name, f.type))
    return pa.schema(fields)''',
    '''    fields: list[pa.Field] = []
    for c in on:
        f = left_schema.field(c)
        fields.append(pa.field(c, f.type, nullable=True))
    for f in left_rest:
        name = f.name + suffixes[0] if f.name in overlap else f.name
        fields.append(pa.field(name, f.type, nullable=True))
    for f in right_rest:
        name = f.name + suffixes[1] if f.name in overlap else f.name
        fields.append(pa.field(name, f.type, nullable=True))
    return pa.schema(fields)''',
    1,
)

# 3. Переименовать inner_join -> hash_join + добавить how; inner_join = обёртка
old_func_start = '''def inner_join(
    left_batches: Iterable[pa.RecordBatch],
    right_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    suffixes: tuple[str, str] = ("_x", "_y"),
    num_partitions: int = 8,
    hash_seed: int = 0,
    max_output_rows: Optional[int] = None,
) -> pa.Table:
    """Grace hash inner join. Возвращает pa.Table."""
    if not on:
        raise ValueError("inner_join: on must be non-empty")
    for c in on:
        if c not in left_schema.names:
            raise KeyError(f"inner_join: on column {c!r} not in left schema")
        if c not in right_schema.names:
            raise KeyError(f"inner_join: on column {c!r} not in right schema")'''

new_func_start = '''def hash_join(
    left_batches: Iterable[pa.RecordBatch],
    right_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    how: str = "inner",
    suffixes: tuple[str, str] = ("_x", "_y"),
    num_partitions: int = 8,
    hash_seed: int = 0,
    max_output_rows: Optional[int] = None,
) -> pa.Table:
    """Grace hash join. how ∈ {'inner', 'left'} (v0.2.2).

    - how='inner': build = меньшая партиция (по file size); NULL-ключи
      отбрасываются с обеих сторон (SQL).
    - how='left':  build = right всегда (left — preserved); NULL-ключи
      отбрасываются только из build (right), probe (left) сохраняется
      с NULL в right-колонках.
    """
    if how not in ("inner", "left"):
        raise NotImplementedError(
            f"hash_join: how={how!r}; only 'inner' and 'left' supported "
            f"in v0.2.x. 'right'/'outer' — v0.3."
        )
    if not on:
        raise ValueError("hash_join: on must be non-empty")
    for c in on:
        if c not in left_schema.names:
            raise KeyError(f"hash_join: on column {c!r} not in left schema")
        if c not in right_schema.names:
            raise KeyError(f"hash_join: on column {c!r} not in right schema")'''

if old_func_start not in s:
    raise SystemExit("inner_join def not found")
s = s.replace(old_func_start, new_func_start, 1)

# 4. Заменить цикл по партициям: логика build/probe + pd.merge по how
old_loop = '''    for p in range(num_partitions):
        lp = left_paths[p] if left_paths[p].exists() else None
        rp = right_paths[p] if right_paths[p].exists() else None
        if lp is None or rp is None:
            continue

        # build = меньшая партиция по file size (proxy для памяти)
        if lp.stat().st_size <= rp.stat().st_size:
            build_path, probe_path = lp, rp
        else:
            build_path, probe_path = rp, lp

        build_table = _materialize_partition(build_path, memory_guard)
        build_df = _drop_null_keys(build_table.to_pandas(), list(on))
        if len(build_df) == 0:
            continue

        for probe_batch in _iter_probe_batches(probe_path):
            memory_guard.check_batch(
                probe_batch, name=f"join_probe({probe_path.name})"
            )
            probe_df = _drop_null_keys(probe_batch.to_pandas(), list(on))
            if len(probe_df) == 0:
                continue
            # Верхняя оценка output до materialize
            if len(build_df) * len(probe_df) > max_output_rows:
                raise MemoryLimitExceeded(
                    f"inner_join: estimated output rows "
                    f"{len(build_df)} * {len(probe_df)} = "
                    f"{len(build_df) * len(probe_df)} > max_output_rows="
                    f"{max_output_rows}. Likely skew (DECISIONS §11.4)."
                )
            merged = pd.merge(
                build_df, probe_df, on=list(on), how="inner", suffixes=suffixes
            )
            if len(merged) == 0:
                continue
            out_table = pa.Table.from_pandas(merged, preserve_index=False)
            # Привести порядок колонок к empty_schema
            if out_table.schema.names != empty_schema.names:
                out_table = out_table.select(empty_schema.names)
            memory_guard.check_state(
                out_table.nbytes, name=f"join_output(p={p})"
            )
            for b in out_table.to_batches():
                all_batches.append(b)'''

new_loop = '''    for p in range(num_partitions):
        lp = left_paths[p] if left_paths[p].exists() else None
        rp = right_paths[p] if right_paths[p].exists() else None
        if lp is None or rp is None:
            continue

        if how == "inner":
            # build = меньшая партиция (min по file size)
            if lp.stat().st_size <= rp.stat().st_size:
                build_path, probe_path = lp, rp
                build_is_left = True
            else:
                build_path, probe_path = rp, lp
                build_is_left = False
        else:  # how == "left": build = right всегда
            build_path, probe_path = rp, lp
            build_is_left = False

        build_table = _materialize_partition(build_path, memory_guard)
        build_df = _drop_null_keys(build_table.to_pandas(), list(on))
        if len(build_df) == 0 and how == "inner":
            # inner: пустой build -> нет matches, skip
            continue
        # how='left': пустой build — продолжаем: probe (left) сохраняется
        # с NULL right-колонками.

        for probe_batch in _iter_probe_batches(probe_path):
            memory_guard.check_batch(
                probe_batch, name=f"join_probe({probe_path.name})"
            )
            probe_df = probe_batch.to_pandas()
            if how == "inner":
                # inner: NULL-ключи в probe отбрасываются (SQL)
                probe_df = _drop_null_keys(probe_df, list(on))
                if len(probe_df) == 0:
                    continue
            # how='left': probe (left) сохраняется целиком; NULL-ключи
            # дадут NULL в right-колонках (SQL: NULL не совпадает, но
            # left preserved).

            # Верхняя оценка output до materialize (cartesian guard)
            if len(build_df) * len(probe_df) > max_output_rows:
                raise MemoryLimitExceeded(
                    f"hash_join(how={how!r}): estimated output rows "
                    f"{len(build_df)} * {len(probe_df)} = "
                    f"{len(build_df) * len(probe_df)} > max_output_rows="
                    f"{max_output_rows}. Likely skew (DECISIONS §11.4)."
                )

            if how == "left":
                # left = probe, right = build; left preserved
                merged = pd.merge(
                    probe_df, build_df, on=list(on), how="left",
                    suffixes=suffixes,
                )
            else:
                # inner: build + probe; при build=right — поменять суффиксы
                effective_suffixes = (
                    suffixes if build_is_left else (suffixes[1], suffixes[0])
                )
                merged = pd.merge(
                    build_df, probe_df, on=list(on), how="inner",
                    suffixes=effective_suffixes,
                )
            if len(merged) == 0:
                continue
            # Типы: right-колонки могут стать NaN (left join) — используем
            # nullable schema.
            out_table = pa.Table.from_pandas(
                merged, preserve_index=False
            )
            if out_table.schema.names != empty_schema.names:
                out_table = out_table.select(empty_schema.names)
            # Привести к nullable-типам empty_schema (NaN -> null)
            try:
                out_table = out_table.cast(empty_schema, safe=False)
            except Exception:
                # fallback: from_pandas с schema (может дать null в nullable)
                out_table = pa.Table.from_pandas(
                    merged, schema=empty_schema, preserve_index=False
                )
            memory_guard.check_state(
                out_table.nbytes, name=f"join_output(p={p})"
            )
            for b in out_table.to_batches():
                all_batches.append(b)'''

if old_loop not in s:
    raise SystemExit("join loop not found")
s = s.replace(old_loop, new_loop, 1)

# 5. inner_join — обёртка
s = s.rstrip() + '''


def inner_join(
    left_batches: Iterable[pa.RecordBatch],
    right_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    suffixes: tuple[str, str] = ("_x", "_y"),
    num_partitions: int = 8,
    hash_seed: int = 0,
    max_output_rows: Optional[int] = None,
) -> pa.Table:
    """Inner join. Обёртка над hash_join(how='inner')."""
    return hash_join(
        left_batches, right_batches, left_schema, right_schema, on,
        memory_guard, spill_dir,
        how="inner", suffixes=suffixes, num_partitions=num_partitions,
        hash_seed=hash_seed, max_output_rows=max_output_rows,
    )


def left_join(
    left_batches: Iterable[pa.RecordBatch],
    right_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    suffixes: tuple[str, str] = ("_x", "_y"),
    num_partitions: int = 8,
    hash_seed: int = 0,
    max_output_rows: Optional[int] = None,
) -> pa.Table:
    """Left join. Обёртка над hash_join(how='left')."""
    return hash_join(
        left_batches, right_batches, left_schema, right_schema, on,
        memory_guard, spill_dir,
        how="left", suffixes=suffixes, num_partitions=num_partitions,
        hash_seed=hash_seed, max_output_rows=max_output_rows,
    )
'''

p.write_text(s)
print("join.py: hash_join(how='inner'|'left'), left_join, inner_join wrapper")
PY

# ---------- cadw/ops/__init__.py ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/__init__.py")
s = p.read_text()
if "from .join import" in s and "left_join" not in s:
    s = s.replace(
        "from .join import inner_join",
        "from .join import hash_join, inner_join, left_join",
        1,
    )
    s = s.replace(
        '"inner_join",',
        '"hash_join", "inner_join", "left_join",',
        1,
    )
    p.write_text(s)
    print("ops/__init__.py: hash_join, left_join exported")
else:
    print("ops/__init__.py: already exported or not found")
PY

# ---------- dataframe.py: merge(how='inner'|'left') ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

if "from cadw.ops.join import inner_join, left_join" not in s:
    s = s.replace(
        "from cadw.ops.join import inner_join",
        "from cadw.ops.join import hash_join",
        1,
    )
    print("dataframe.py: import hash_join")

old_merge = '''    def merge(
        self,
        right: "CadwDataFrame",
        on: str | Sequence[str],
        how: str = "inner",
        *,
        suffixes: tuple[str, str] = ("_x", "_y"),
        num_partitions: int = 8,
    ) -> "CadwDataFrame":
        """Inner join (v0.2.1). how ∈ {'inner'} только.

        - on: имена общих колонок в обеих сторонах.
        - suffixes: для пересекающихся не-on колонок.
        - num_partitions: число партиций HashShuffle (обе стороны).
        """
        if how != "inner":
            raise NotImplementedError(
                f"merge: how={how!r}; only 'inner' supported in v0.2.1. "
                f"'left'/'right'/'outer' — v0.2.2."
            )
        if not isinstance(right, CadwDataFrame):
            raise TypeError("merge: right must be CadwDataFrame")
        on_list = [on] if isinstance(on, str) else list(on)
        for c in on_list:
            if c not in self.columns:
                raise KeyError(f"merge: on column {c!r} not in self")
            if c not in right.columns:
                raise KeyError(f"merge: on column {c!r} not in right")

        left_schema = pa.schema(
            [self._dataset.schema.field(c) for c in self.columns]
        )
        right_schema = pa.schema(
            [right._dataset.schema.field(c) for c in right.columns]
        )

        result_table = inner_join(
            self._iter_filtered_batches(),
            right._iter_filtered_batches(),
            left_schema,
            right_schema,
            on=on_list,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "join",
            suffixes=suffixes,
            num_partitions=num_partitions,
        )
        return self._wrap_result_table(result_table)'''

new_merge = '''    def merge(
        self,
        right: "CadwDataFrame",
        on: str | Sequence[str],
        how: str = "inner",
        *,
        suffixes: tuple[str, str] = ("_x", "_y"),
        num_partitions: int = 8,
    ) -> "CadwDataFrame":
        """Hash join (v0.2.2). how ∈ {'inner', 'left'}.

        - how='inner': только совпадающие ключи.
        - how='left':  все строки self сохраняются; right-колонки -> NULL
          при отсутствии совпадения.
        - on: имена общих колонок в обеих сторонах.
        - suffixes: для пересекающихся не-on колонок.
        - num_partitions: число партиций HashShuffle (обе стороны).
        """
        if how not in ("inner", "left"):
            raise NotImplementedError(
                f"merge: how={how!r}; only 'inner', 'left' supported in "
                f"v0.2.x. 'right'/'outer' — v0.3."
            )
        if not isinstance(right, CadwDataFrame):
            raise TypeError("merge: right must be CadwDataFrame")
        on_list = [on] if isinstance(on, str) else list(on)
        for c in on_list:
            if c not in self.columns:
                raise KeyError(f"merge: on column {c!r} not in self")
            if c not in right.columns:
                raise KeyError(f"merge: on column {c!r} not in right")

        left_schema = pa.schema(
            [self._dataset.schema.field(c) for c in self.columns]
        )
        right_schema = pa.schema(
            [right._dataset.schema.field(c) for c in right.columns]
        )

        result_table = hash_join(
            self._iter_filtered_batches(),
            right._iter_filtered_batches(),
            left_schema,
            right_schema,
            on=on_list,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir / "join",
            how=how,
            suffixes=suffixes,
            num_partitions=num_partitions,
        )
        return self._wrap_result_table(result_table)'''

if old_merge not in s:
    raise SystemExit("merge method not found")
s = s.replace(old_merge, new_merge, 1)
p.write_text(s)
print("dataframe.py: merge(how='inner'|'left') via hash_join")
PY

# ---------- tests/test_join.py: 5 left-тестов + обновить test_merge_unsupported_how ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_join.py")
s = p.read_text()

old_unsupported = '''def test_merge_unsupported_how(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1], "a": [10]})
    right = make_df(tmp_path, "R", {"k": [1], "b": [100]})
    with pytest.raises(NotImplementedError, match="only 'inner'"):
        left.merge(right, on="k", how="left")'''

new_unsupported = '''def test_merge_unsupported_how(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1], "a": [10]})
    right = make_df(tmp_path, "R", {"k": [1], "b": [100]})
    for bad_how in ("right", "outer", "cross"):
        with pytest.raises(NotImplementedError, match="only 'inner', 'left'"):
            left.merge(right, on="k", how=bad_how)'''

if old_unsupported not in s:
    raise SystemExit("test_merge_unsupported_how not found")
s = s.replace(old_unsupported, new_unsupported, 1)

# Добавить 5 left-тестов в конец
left_tests = '''

# ---------- how='left' ----------

def test_merge_left_basic(tmp_path):
    left = make_df(tmp_path, "L", {
        "k": [1, 2, 3, 4], "a": ["w", "x", "y", "z"],
    })
    right = make_df(tmp_path, "R", {"k": [1, 3, 5], "b": ["p", "q", "r"]})
    result = left.merge(right, on="k", how="left")
    head = result.head(10)
    assert head.num_rows == 4
    rows = _sorted_by(head, "k")
    assert rows[0] == {"k": 1, "a": "w", "b": "p"}
    assert rows[1] == {"k": 2, "a": "x", "b": None}
    assert rows[2] == {"k": 3, "a": "y", "b": "q"}
    assert rows[3] == {"k": 4, "a": "z", "b": None}


def test_merge_left_null_in_left_key_preserved(tmp_path):
    left = make_df(tmp_path, "L", {
        "k": [1, None, 2], "a": ["w", "x", "y"],
    })
    right = make_df(tmp_path, "R", {"k": [1, 2], "b": ["p", "q"]})
    result = left.merge(right, on="k", how="left")
    head = result.head(10)
    assert head.num_rows == 3
    rows = sorted(head.to_pylist(), key=lambda r: (r["k"] is None, r["k"] or 0))
    # Строки: k=None (a=x, b=None), k=1 (a=w, b=p), k=2 (a=y, b=q)
    by_a = {r["a"]: r for r in rows}
    assert by_a["w"]["b"] == "p"
    assert by_a["x"]["b"] is None
    assert by_a["y"]["b"] == "q"


def test_merge_left_null_in_right_key_not_matched(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": ["w", "x", "y"]})
    right = make_df(tmp_path, "R", {"k": [None, 2], "b": ["p", "q"]})
    result = left.merge(right, on="k", how="left")
    head = result.head(10)
    assert head.num_rows == 3
    rows = _sorted_by(head, "k")
    assert rows[0]["b"] is None
    assert rows[1]["b"] == "q"
    assert rows[2]["b"] is None


def test_merge_left_empty_right(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2], "a": [10, 20]})
    right_data = {
        "k": pa.array([], type=pa.int64()),
        "b": pa.array([], type=pa.string()),
    }
    right = make_df(tmp_path, "R", right_data)
    result = left.merge(right, on="k", how="left")
    head = result.head(10)
    assert head.num_rows == 2
    rows = _sorted_by(head, "k")
    assert rows[0]["b"] is None
    assert rows[1]["b"] is None


def test_merge_left_after_filter(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [1, 2, 3], "b": [100, 200, 300]})
    filtered = left[left["a"] > 15]  # k=2,3
    result = filtered.merge(right, on="k", how="left")
    head = result.head(10)
    assert head.num_rows == 2
    rows = _sorted_by(head, "k")
    assert rows[0] == {"k": 2, "a": 20, "b": 200}
    assert rows[1] == {"k": 3, "a": 30, "b": 300}

'''
s = s.rstrip() + left_tests
p.write_text(s)
print("test_join.py: +5 left tests, test_merge_unsupported_how updated")
PY

# ---------- DECISIONS §11.1 / §12.6 / §13.4/37 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

# §11.1: обновить строку inner_join -> hash_join
s = s.replace(
    "| `inner_join` (v0.2.1: HashShuffle обеих сторон + pd.merge) | ✅ | 9 |",
    "| `hash_join` inner/left (v0.2.2: HashShuffle обеих сторон + pd.merge) | ✅ | 15 |",
    1,
)

# §12.6: v0.2 progress
s = s.replace(
    "§12.6. v0.2 (в работе: inner_join ✅)",
    "§12.6. v0.2 (в работе: inner+left ✅; JoinMemoryModel, E2E — 11-B-2b/2c)",
    1,
)

# §13.4 эпизод 37
marker_135 = "### 13.5. Открытые вопросы"
episode_37 = """**Эпизод 37 (left join и nullable-типы).** Для `how='left'` build = right
всегда (left — preserved), probe (left) не проходит `_drop_null_keys`:
NULL-ключ в left сохраняется с NULL в right-колонках (SQL: NULL не
совпадает, но left preserved). При этом `pd.merge(how='left')` даёт NaN
в right-колонках для unmatched, и `pa.Table.from_pandas` без schema
конвертирует int64 NaN -> float64 — несовместимо с `empty_schema`.
**Fix:** `_empty_result_schema` — `nullable=True` для всех полей;
`out_table.cast(empty_schema, safe=False)` (NaN в nullable-int
конвертируется в null). Fallback — `pa.Table.from_pandas(merged,
schema=empty_schema)`. **Урок:** для left/outer join **все** output
колонки должны быть nullable, иначе materialize unmatched падает на
типах. Right/outer (v0.3) — та же проблема, тот же фикс.

"""
if "Эпизод 37" not in s:
    s = s.replace(marker_135, episode_37 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §11.1, §12.6, §13.4/37")
PY

# ---------- CHANGELOG ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.2.1] — 2026-10-19

### Добавлено
- `CadwDataFrame.merge(..., how='left')`: left join, все строки self
  сохраняются; right-колонки -> NULL при отсутствии совпадения.
- `cadw.ops.join.hash_join(how='inner'|'left')` — общий runtime;
  `inner_join`, `left_join` — обёртки.
- SQL-семантика NULL:
  - inner: `_drop_null_keys` на обе стороны;
  - left: `_drop_null_keys` только на build (right); probe (left)
    сохраняется (NULL в left key -> NULL right cols).
- `_empty_result_schema` — nullable=True для всех полей (left join с
  unmatched: NaN в int/string -> null, cast safe=False).

### Ограничения v0.2.2
- `how='right'`, `how='outer'`, `left_on`/`right_on`, `on=index` — v0.3.
- `JoinMemoryModel` + `propagate_chars("join")` + E2E — 11-B-2b/2c.

"""
if "## [0.2.1]" not in s:
    if "## [0.2.0]" in s:
        s = s.replace("## [0.2.0]", entry + "## [0.2.0]", 1)
    else:
        s = s.replace("## [0.1.0]", entry + "## [0.1.0]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.2.1")
PY

# ---------- run ----------
echo "=== pytest tests/test_join.py ==="
set +e
uv run pytest tests/test_join.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "test_join.py: rc=$RC"
  git add -A
  git commit -m "WIP: 11-B-2a (test red; fix follows)" || true
  exit 1
fi

uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2a: hash_join(how='inner'|'left'), merge(how='left'), nullable schema, 15 tests"

echo "==> 11-B-2a applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"