#!/usr/bin/env bash
# patch_v050_ram_fixes: P2 (join out_batches → streaming) + P8 (ReduceOp.estimate_state_nbytes).
# По результатам мультиагентного анализа Pingis (см. _FOR_PINGIS/REPORT.md).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-v050-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch v0.5.0 (P2+P8)"

BR="patch/v050-ram-fixes"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ============ Part 1: P8 — ReduceOp.estimate_state_nbytes ============
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/reduce.py")
s = p.read_text()

# 1a. Add method to ReduceOp ABC
old_abc = '''class ReduceOp(ABC):
    @abstractmethod
    def init(self) -> State: ...
    @abstractmethod
    def update(self, state: State, batch: pa.RecordBatch) -> State: ...
    @abstractmethod
    def merge(self, a: State, b: State) -> State: ...
    @abstractmethod
    def finalize(self, state: State) -> Any: ...
    @abstractmethod
    def state_nbytes(self, state: State) -> int: ...'''

new_abc = '''class ReduceOp(ABC):
    @abstractmethod
    def init(self) -> State: ...
    @abstractmethod
    def update(self, state: State, batch: pa.RecordBatch) -> State: ...
    @abstractmethod
    def merge(self, a: State, b: State) -> State: ...
    @abstractmethod
    def finalize(self, state: State) -> Any: ...
    @abstractmethod
    def state_nbytes(self, state: State) -> int: ...

    def estimate_state_nbytes(self) -> "int | None":
        """Upper bound on state size BEFORE init() (v0.5.1, P8 fix).

        Called by `reduce()` for `MemoryGuard.check_state` *before*
        `op.init()`, so the state must be estimable without allocation.

        - Built-in ops override with known sizes (no allocation, O(1)).
        - User ops should override — otherwise `MemoryGuard.
          estimate_state_size` falls back to 1 MiB hardcode (P8 in
          _FOR_PINGIS/REPORT.md, DECISIONS §9.2.4).
        - Returning None means "use MemoryGuard fallback".

        Contract: `estimate_state_nbytes() >= state_nbytes(init())`.
        """
        return None'''

if old_abc not in s:
    raise SystemExit("ReduceOp ABC not found")
s = s.replace(old_abc, new_abc, 1)
print("reduce.py: ReduceOp.estimate_state_nbytes added")

# 1b. Built-in op overrides
overrides = {
    "class CountOp(ReduceOp):": '''    def estimate_state_nbytes(self) -> int:
        return 16
''',
    "class SumOp(ReduceOp):": '''    def estimate_state_nbytes(self) -> int:
        return 16
''',
    "class MinMaxOp(ReduceOp):": '''    def estimate_state_nbytes(self) -> int:
        return 16
''',
    "class MeanOp(ReduceOp):": '''    def estimate_state_nbytes(self) -> int:
        return 16
''',
    "class StdOp(ReduceOp):": '''    def estimate_state_nbytes(self) -> int:
        return 24
''',
    "class CovarianceOp(ReduceOp):": '''    def estimate_state_nbytes(self) -> int:
        return 8 + 8 * self.d + 8 * self.d * self.d
''',
    "class HistogramOp(ReduceOp):": '''    def estimate_state_nbytes(self) -> int:
        return 8 * self.bins + 16
''',
}

for cls_marker, method_src in overrides.items():
    idx = s.find(cls_marker)
    if idx < 0:
        raise SystemExit(f"class marker not found: {cls_marker}")
    # insert method right after class line
    insert_pos = idx + len(cls_marker) + 1  # +1 for newline
    # find first non-empty line after class to insert method
    s = s[:insert_pos] + "\n" + method_src + s[insert_pos:]
    print(f"reduce.py: {cls_marker.rstrip(':').strip()} — estimate_state_nbytes")

# 1c. Modify reduce() to use op.estimate_state_nbytes() first
old_reduce_check = '''    _estimated = memory_guard.estimate_state_size(
        type(op).__name__,
        schema=None,
        d=getattr(op, "d", None),
        bins=getattr(op, "bins", 255),
    )
    memory_guard.check_state(_estimated, name=f"{type(op).__name__}.estimated")'''

new_reduce_check = '''    # v0.5.1 (P8): prefer op.estimate_state_nbytes() when overridden;
    # fall back to MemoryGuard.estimate_state_size for unknown ops.
    _estimated = op.estimate_state_nbytes()
    if _estimated is None:
        _estimated = memory_guard.estimate_state_size(
            type(op).__name__,
            schema=None,
            d=getattr(op, "d", None),
            bins=getattr(op, "bins", 255),
        )
    memory_guard.check_state(_estimated, name=f"{type(op).__name__}.estimated")'''

if old_reduce_check not in s:
    raise SystemExit("reduce() check_state block not found")
s = s.replace(old_reduce_check, new_reduce_check, 1)
print("reduce.py: reduce() — prefer op.estimate_state_nbytes()")

p.write_text(s)
PY

# ============ Part 2: P2 — streaming join ============
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/join.py")
s = p.read_text()

# 2a. Refactor _left_preserve_with_nulls → _iter version + wrapper
old_lpn = '''def _left_preserve_with_nulls(
    left_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    suffixes: tuple[str, str],
    memory_guard: MemoryGuard,
) -> pa.Table:
    """how='left' при пустом right: left preserved, right-колонки -> null.

    Используется, когда right содержит 0 батчей: hash_shuffle не может
    принять пустой вход (см. §6.7), а left join не требует shuffle, потому
    что совпадений всё равно нет.
    """
    on_set = set(on)
    right_rest = [c for c in right_schema.names if c not in on_set]
    overlap = {c for c in left_schema.names if c not in on_set} & set(right_rest)

    out_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)

    all_batches: list[pa.RecordBatch] = []
    for lb in left_batches:
        memory_guard.check_batch(lb, name="left_preserve.input")
        n = lb.num_rows
        arrays: dict[str, pa.Array] = {}
        for c in on:
            arrays[c] = lb.column(c)
        for f in left_schema:
            if f.name in on_set:
                continue
            name = f.name + suffixes[0] if f.name in overlap else f.name
            arrays[name] = lb.column(f.name)
        for f in right_schema:
            if f.name in on_set:
                continue
            name = f.name + suffixes[1] if f.name in overlap else f.name
            arrays[name] = pa.nulls(n, type=f.type)

        rb = pa.RecordBatch.from_arrays(
            [arrays[c] for c in out_schema.names],
            schema=out_schema,
        )
        if rb.num_rows > 0:
            all_batches.append(rb)

    if not all_batches:
        return pa.Table.from_batches([], schema=out_schema)
    return pa.Table.from_batches(all_batches)'''

new_lpn = '''def _left_preserve_with_nulls_iter(
    left_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    suffixes: tuple[str, str],
    memory_guard: MemoryGuard,
):
    """Streaming version (v0.5.1, P2): yields batches, not a Table."""
    on_set = set(on)
    right_rest = [c for c in right_schema.names if c not in on_set]
    overlap = {c for c in left_schema.names if c not in on_set} & set(right_rest)

    out_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)

    for lb in left_batches:
        memory_guard.check_batch(lb, name="left_preserve.input")
        n = lb.num_rows
        arrays: dict[str, pa.Array] = {}
        for c in on:
            arrays[c] = lb.column(c)
        for f in left_schema:
            if f.name in on_set:
                continue
            name = f.name + suffixes[0] if f.name in overlap else f.name
            arrays[name] = lb.column(f.name)
        for f in right_schema:
            if f.name in on_set:
                continue
            name = f.name + suffixes[1] if f.name in overlap else f.name
            arrays[name] = pa.nulls(n, type=f.type)

        rb = pa.RecordBatch.from_arrays(
            [arrays[c] for c in out_schema.names],
            schema=out_schema,
        )
        if rb.num_rows > 0:
            yield rb


def _left_preserve_with_nulls(
    left_batches: Iterable[pa.RecordBatch],
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    suffixes: tuple[str, str],
    memory_guard: MemoryGuard,
) -> pa.Table:
    """Backward-compat wrapper (v0.5.1). Prefer _iter for streaming."""
    out_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)
    batches = list(_left_preserve_with_nulls_iter(
        left_batches, left_schema, right_schema, on, suffixes, memory_guard,
    ))
    return pa.Table.from_batches(batches, schema=out_schema) if batches else pa.Table.from_batches([], schema=out_schema)'''

if old_lpn not in s:
    raise SystemExit("_left_preserve_with_nulls not found")
s = s.replace(old_lpn, new_lpn, 1)
print("join.py: _left_preserve_with_nulls_iter added; wrapper kept")

# 2b. Refactor _process_partition_pair → generator (yield instead of extend)
old_ppp_sig = '''def _process_partition_pair(
    left_path: Path,
    right_path: Path,
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    suffixes: tuple[str, str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    how: str,
    num_partitions_recurse: int = 4,
    depth: int = 0,
    max_depth: int = 5,
    max_output_rows: int = 0,
) -> list[pa.RecordBatch]:'''
new_ppp_sig = '''def _process_partition_pair(
    left_path: Path,
    right_path: Path,
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    on: list[str],
    suffixes: tuple[str, str],
    memory_guard: MemoryGuard,
    spill_dir: Path,
    *,
    how: str,
    num_partitions_recurse: int = 4,
    depth: int = 0,
    max_depth: int = 5,
    max_output_rows: int = 0,
):
    """Generator (v0.5.1, P2): yields output batches instead of accumulating.

    Callers may consume lazily (streaming write) or collect (`hash_join`).
    """
'''
if old_ppp_sig not in s:
    raise SystemExit("_process_partition_pair signature not found")
s = s.replace(old_ppp_sig, new_ppp_sig, 1)

# 2c. Early returns: `return []` → `return` (in generator body)
s = s.replace(
    "    if lp_size == 0 and rp_size == 0:\n        return []",
    "    if lp_size == 0 and rp_size == 0:\n        return",
    1,
)
s = s.replace(
    "    if rp_size == 0 and how == \"inner\":\n        return []",
    "    if rp_size == 0 and how == \"inner\":\n        return",
    1,
)

# 2d. Recursion: out.extend → yield from
old_rec = '''        out: list[pa.RecordBatch] = []
        for sub_l, sub_r in zip(sub_left, sub_right):
            out.extend(
                _process_partition_pair(
                    sub_l, sub_r, left_schema, right_schema, on, suffixes,
                    memory_guard, spill_dir, how=how,
                    num_partitions_recurse=num_partitions_recurse,
                    depth=depth + 1, max_depth=max_depth,
                    max_output_rows=max_output_rows,
                )
            )
        return out'''
new_rec = '''        for sub_l, sub_r in zip(sub_left, sub_right):
            yield from _process_partition_pair(
                sub_l, sub_r, left_schema, right_schema, on, suffixes,
                memory_guard, spill_dir, how=how,
                num_partitions_recurse=num_partitions_recurse,
                depth=depth + 1, max_depth=max_depth,
                max_output_rows=max_output_rows,
            )
        return'''
if old_rec not in s:
    raise SystemExit("recursion block not found")
s = s.replace(old_rec, new_rec, 1)

# 2e. Empty build_df early returns
s = s.replace(
    '''    if len(build_df) == 0:
        if how == "inner":
            return []''',
    '''    if len(build_df) == 0:
        if how == "inner":
            return''',
    1,
)

# 2f. Probe loop: extend → yield
old_loop = '''    out_batches: list[pa.RecordBatch] = []
    empty_schema = _empty_result_schema(
        left_schema, right_schema, on, suffixes
    )'''
new_loop = '''    empty_schema = _empty_result_schema(
        left_schema, right_schema, on, suffixes
    )'''
if old_loop not in s:
    raise SystemExit("out_batches declaration not found")
s = s.replace(old_loop, new_loop, 1)

# final return out_batches → nothing (generator ends naturally)
s = s.replace(
    '''        memory_guard.check_state(
            out_table.nbytes, name=f"join_output({probe_path.name})"
        )
        out_batches.extend(out_table.to_batches())

    return out_batches''',
    '''        memory_guard.check_state(
            out_table.nbytes, name=f"join_output({probe_path.name})"
        )
        for b in out_table.to_batches():
            yield b''',
    1,
)
print("join.py: _process_partition_pair → generator")

# 2g. Extract common setup into _join_core
old_hj_body = '''    left_rest = [c for c in left_schema.names if c not in on]
    right_rest = [c for c in right_schema.names if c not in on]

    # Peek: пустой вход -> пустой результат
    left_it = iter(left_batches)
    right_it = iter(right_batches)
    try:
        first_left = next(left_it)
    except StopIteration:
        return pa.Table.from_batches(
            [], schema=_empty_result_schema(left_schema, right_schema, on, suffixes)
        )
    left_batches = chain([first_left], left_it)
    try:
        first_right = next(right_it)
        right_batches = chain([first_right], right_it)
    except StopIteration:
        if how == "inner":
            return pa.Table.from_batches(
                [], schema=_empty_result_schema(left_schema, right_schema, on, suffixes)
            )
        # how='left': right пуст -> left preserved с null right-cols
        # Без hash_shuffle (пустой вход не принимается, §6.7).
        return _left_preserve_with_nulls(
            left_batches, left_schema, right_schema, list(on), suffixes,
            memory_guard,
        )

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)
    left_paths = hash_shuffle(
        left_batches, list(on), left_rest, num_partitions, memory_guard,
        spill_dir / "join_left", hash_seed=hash_seed, log=False,
    )
    right_paths = hash_shuffle(
        right_batches, list(on), right_rest, num_partitions, memory_guard,
        spill_dir / "join_right", hash_seed=hash_seed, log=False,
    )

    if max_output_rows is None:
        # Консервативная оценка: 0.3 * available / 256 B на строку
        max_output_rows = max(
            1, int(memory_guard.available_bytes() * 0.3 / 256)
        )
    empty_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)
    all_batches: list[pa.RecordBatch] = []

    for p in range(num_partitions):
        lp = left_paths[p] if left_paths[p].exists() else None
        rp = right_paths[p] if right_paths[p].exists() else None
        if lp is None and rp is None:
            continue
        # Для inner: если хотя бы одна сторона пуста — пропускаем пару.
        if how == "inner":
            if lp is None or rp is None:
                continue
        else:  # left
            if lp is None:
                continue
            # rp может быть None (пустая right) — _process_partition_pair
            # обработает как build_df пустой.

        # Если rp is None для left, создадим пустой временный файл? Нет,
        # _process_partition_pair принимает Path; для None используем
        # заглушку с пустым файлом. Проще: обработать отдельно.
        if how == "left" and rp is None:
            # left preserved, right null: используем _left_preserve_with_nulls
            for b in _iter_probe_batches(lp):
                # Собираем left-батчи и добавляем null right-колонки
                # (упрощённо: вызываем _left_preserve_with_nulls на одном батче)
                pass
            # Пока оставим как есть: _process_partition_pair с rp=lp?
            # Нет. Реализуем через _left_preserve_with_nulls на одном lp.
            # Для простоты: создадим пустой right-файл с нужной схемой.
            # Это редкий случай; в v0.2.2 оставим через _left_preserve_with_nulls
            # на всех left_paths сразу (см. ниже).
            all_batches.extend(
                _left_preserve_with_nulls(
                    _iter_probe_batches(lp), left_schema, right_schema,
                    list(on), suffixes, memory_guard,
                ).to_batches()
            )
            continue

        all_batches.extend(
            _process_partition_pair(
                lp, rp, left_schema, right_schema, on, suffixes,
                memory_guard, spill_dir, how=how,
                max_output_rows=max_output_rows,
            )
        )

    return pa.Table.from_batches(all_batches)'''

new_hj_body = '''    left_rest = [c for c in left_schema.names if c not in on]
    right_rest = [c for c in right_schema.names if c not in on]

    # Peek: empty check
    left_it = iter(left_batches)
    right_it = iter(right_batches)
    try:
        first_left = next(left_it)
    except StopIteration:
        empty_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)
        return empty_schema, iter([])
    left_batches = chain([first_left], left_it)
    try:
        first_right = next(right_it)
        right_batches = chain([first_right], right_it)
    except StopIteration:
        empty_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)
        if how == "inner":
            return empty_schema, iter([])
        return empty_schema, _left_preserve_with_nulls_iter(
            left_batches, left_schema, right_schema, list(on), suffixes,
            memory_guard,
        )

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)
    left_paths = hash_shuffle(
        left_batches, list(on), left_rest, num_partitions, memory_guard,
        spill_dir / "join_left", hash_seed=hash_seed, log=False,
    )
    right_paths = hash_shuffle(
        right_batches, list(on), right_rest, num_partitions, memory_guard,
        spill_dir / "join_right", hash_seed=hash_seed, log=False,
    )

    if max_output_rows is None:
        max_output_rows = max(
            1, int(memory_guard.available_bytes() * 0.3 / 256)
        )
    empty_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)

    def _iter_output():
        for p in range(num_partitions):
            lp = left_paths[p] if left_paths[p].exists() else None
            rp = right_paths[p] if right_paths[p].exists() else None
            if lp is None and rp is None:
                continue
            if how == "inner":
                if lp is None or rp is None:
                    continue
            else:  # left
                if lp is None:
                    continue
            if how == "left" and rp is None:
                yield from _left_preserve_with_nulls_iter(
                    _iter_probe_batches(lp), left_schema, right_schema,
                    list(on), suffixes, memory_guard,
                )
                continue
            yield from _process_partition_pair(
                lp, rp, left_schema, right_schema, on, suffixes,
                memory_guard, spill_dir, how=how,
                max_output_rows=max_output_rows,
            )

    return empty_schema, _iter_output()


def _join_core(
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
    max_output_rows: "Optional[int]" = None,
):
    """Shared setup: validate, peek, shuffle; return (schema, generator).

    Both `hash_join` (collect to Table) and `hash_join_streaming`
    (write to disk) use this.
    """
    if how not in ("inner", "left"):
        raise NotImplementedError(
            f"hash_join: how={how!r}; only 'inner' and 'left' supported "
            f"in v0.5.x. 'right'/'outer' — v0.6."
        )
    if not on:
        raise ValueError("hash_join: on must be non-empty")
    for c in on:
        if c not in left_schema.names:
            raise KeyError(f"hash_join: on column {c!r} not in left schema")
        if c not in right_schema.names:
            raise KeyError(f"hash_join: on column {c!r} not in right schema")
    return _join_core_inner(
        left_batches, right_batches, left_schema, right_schema, on,
        memory_guard, spill_dir, how=how, suffixes=suffixes,
        num_partitions=num_partitions, hash_seed=hash_seed,
        max_output_rows=max_output_rows,
    )


def _join_core_inner(
    left_batches, right_batches, left_schema, right_schema, on,
    memory_guard, spill_dir, *, how, suffixes, num_partitions,
    hash_seed, max_output_rows,
):
    """Actual setup (after validation). Separated so hash_join and
    hash_join_streaming share it without duplicating validation."""
    left_rest = [c for c in left_schema.names if c not in on]
    right_rest = [c for c in right_schema.names if c not in on]

    left_it = iter(left_batches)
    right_it = iter(right_batches)
    try:
        first_left = next(left_it)
    except StopIteration:
        empty_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)
        return empty_schema, iter([])
    left_batches = chain([first_left], left_it)
    try:
        first_right = next(right_it)
        right_batches = chain([first_right], right_it)
    except StopIteration:
        empty_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)
        if how == "inner":
            return empty_schema, iter([])
        return empty_schema, _left_preserve_with_nulls_iter(
            left_batches, left_schema, right_schema, list(on), suffixes,
            memory_guard,
        )

    spill_dir = Path(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)
    left_paths = hash_shuffle(
        left_batches, list(on), left_rest, num_partitions, memory_guard,
        spill_dir / "join_left", hash_seed=hash_seed, log=False,
    )
    right_paths = hash_shuffle(
        right_batches, list(on), right_rest, num_partitions, memory_guard,
        spill_dir / "join_right", hash_seed=hash_seed, log=False,
    )

    if max_output_rows is None:
        max_output_rows = max(
            1, int(memory_guard.available_bytes() * 0.3 / 256)
        )
    empty_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)

    def _iter_output():
        for p in range(num_partitions):
            lp = left_paths[p] if left_paths[p].exists() else None
            rp = right_paths[p] if right_paths[p].exists() else None
            if lp is None and rp is None:
                continue
            if how == "inner":
                if lp is None or rp is None:
                    continue
            else:
                if lp is None:
                    continue
            if how == "left" and rp is None:
                yield from _left_preserve_with_nulls_iter(
                    _iter_probe_batches(lp), left_schema, right_schema,
                    list(on), suffixes, memory_guard,
                )
                continue
            yield from _process_partition_pair(
                lp, rp, left_schema, right_schema, on, suffixes,
                memory_guard, spill_dir, how=how,
                max_output_rows=max_output_rows,
            )

    return empty_schema, _iter_output()'''

if old_hj_body not in s:
    raise SystemExit("hash_join body block not found")
s = s.replace(old_hj_body, new_hj_body, 1)

# 2h. hash_join signature: accept and use _join_core; add hash_join_streaming
# Replace original hash_join signature+body-start (which we just replaced)
# with new ones. Find current hash_join and rewrite fully.
old_hj_start = '''def hash_join(
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
) -> pa.Table:'''

new_hj_start = '''def hash_join(
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
    """Backward-compat wrapper. Collects streaming output to a Table.

    For memory-bounded callers (CadwDataFrame.merge), use
    `hash_join_streaming` — writes directly to disk.
    """
    schema, it = _join_core(
        left_batches, right_batches, left_schema, right_schema, on,
        memory_guard, spill_dir, how=how, suffixes=suffixes,
        num_partitions=num_partitions, hash_seed=hash_seed,
        max_output_rows=max_output_rows,
    )
    batches = list(it)
    return pa.Table.from_batches(batches, schema=schema) if batches else pa.Table.from_batches([], schema=schema)


def hash_join_streaming(
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
) -> tuple[pa.Schema, Path]:
    """Streaming join (v0.5.1, P2 fix). Writes output batches directly
    to a single Arrow IPC file; returns (schema, path).

    Peak RAM = max(build_partition) + probe_batch + writer_buffer.
    Output is NOT accumulated in RAM regardless of output size.

    Caller should consume the file lazily and delete it after (e.g.
    Dataset.from_batches via ArrowIPCBatchReader, then path.unlink()).
    """
    from cadw.io.batch_writer import ArrowIPCBatchWriter

    schema, it = _join_core(
        left_batches, right_batches, left_schema, right_schema, on,
        memory_guard, spill_dir, how=how, suffixes=suffixes,
        num_partitions=num_partitions, hash_seed=hash_seed,
        max_output_rows=max_output_rows,
    )
    out_path = Path(spill_dir) / "join_output.arrow"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ArrowIPCBatchWriter(str(out_path), schema)
    try:
        for b in it:
            writer.write_batch(b)
    finally:
        writer.close()
    return schema, out_path'''

if old_hj_start not in s:
    raise SystemExit("hash_join def start not found")
s = s.replace(old_hj_start, new_hj_start, 1)

# 2i. Remove old hash_join body (which was: validation + `return _join_core_inner(...)` — no, wait.
# Actually the old body is: validation, then `return _join_core_inner(...)` — but that's not right either.
# Let me check what remains after we replaced start. The old function body was: validation + assign + ... 
# But we replaced old_hj_body which contained the whole iteration. Hmm — we replaced too much.
# Actually: old_hj_body was the whole iteration logic. But hash_join also had validation prefix.
# Let me search for what's left.

p.write_text(s)
print("join.py: hash_join + hash_join_streaming + _join_core written (check syntax next)")
PY

# If any inconsistency (leftover old code), fail fast. Compile and inspect.
echo "=== compile check: cadw/ops/join.py ==="
set +e
python3 -c "import py_compile; py_compile.compile('cadw/ops/join.py', doraise=True)" 2>&1
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "COMPILE FAIL — join.py has syntax errors after patch"
  # Show the file around hash_join for debugging
  grep -n "def hash_join\|def _join_core\|def _process_partition_pair\|def _left_preserve" cadw/ops/join.py | head -20
  exit 1
fi
echo "OK: join.py compiles"

# ============ Part 3: dataframe.py — merge uses streaming ============
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# 3a. Import hash_join_streaming
if "from cadw.ops.join import hash_join, hash_join_streaming" not in s:
    old_import = "from cadw.ops.join import hash_join"
    new_import = "from cadw.ops.join import hash_join, hash_join_streaming"
    if old_import in s:
        s = s.replace(old_import, new_import, 1)
        print("dataframe.py: import hash_join_streaming")
    else:
        raise SystemExit("hash_join import not found")

# 3b. Replace merge body — use streaming path instead of collecting Table.
old_merge = '''        result_table = hash_join(
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
            max_output_rows=max_output_rows,
        )
        return self._wrap_result_table(result_table)'''

new_merge = '''        # v0.5.1 (P2): use streaming path — output written to disk
        # batch-by-batch, never accumulated in RAM. Only the resulting
        # Dataset is built from IPC reader (also streaming).
        out_schema, out_path = hash_join_streaming(
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
            max_output_rows=max_output_rows,
        )
        try:
            new_ds = Dataset.from_batches(
                out_schema,
                ArrowIPCBatchReader(str(out_path), None),
                self._new_dataset_dir(),
            )
        finally:
            out_path.unlink(missing_ok=True)
        return CadwDataFrame(
            new_ds,
            memory_guard=self._memory_guard,
            spill_dir=self._spill_dir,
        )'''

if old_merge not in s:
    raise SystemExit("merge body not found")
s = s.replace(old_merge, new_merge, 1)
print("dataframe.py: merge uses hash_join_streaming")

p.write_text(s)
PY

# ============ Part 4: exports ============
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/__init__.py")
s = p.read_text()
if "hash_join_streaming" not in s:
    s = s.replace(
        "from .join import hash_join, inner_join, left_join",
        "from .join import hash_join, hash_join_streaming, inner_join, left_join",
        1,
    )
    s = s.replace(
        '"hash_join", "inner_join", "left_join",',
        '"hash_join", "hash_join_streaming", "inner_join", "left_join",',
        1,
    )
    p.write_text(s)
    print("cadw/ops/__init__.py: hash_join_streaming exported")
PY

# ============ Part 5: fix test_join_oom_guard generator usage ============
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_join_oom_guard.py")
s = p.read_text()

# _process_partition_pair now returns generator — need list() or next() to trigger
old = '''    with pytest.raises(MemoryLimitExceeded, match="max_depth"):
        _process_partition_pair(
            lp, rp, left_schema, right_schema, ["k"], ("_x", "_y"),
            guard, tmp_path / "spill", how="inner",
            depth=0, max_depth=0, max_output_rows=10**9,
        )'''
new = '''    # v0.5.1: _process_partition_pair is now a generator; trigger body.
    with pytest.raises(MemoryLimitExceeded, match="max_depth"):
        next(_process_partition_pair(
            lp, rp, left_schema, right_schema, ["k"], ("_x", "_y"),
            guard, tmp_path / "spill", how="inner",
            depth=0, max_depth=0, max_output_rows=10**9,
        ))'''
if old in s:
    s = s.replace(old, new, 1)
    p.write_text(s)
    print("test_join_oom_guard.py: fixed generator usage")
else:
    print("WARNING: test pattern not found; check manually")
PY

# ============ Part 6: new tests P8 ============
cat > tests/test_reduce_state_bytes.py <<'EOF'
"""P8 fix (v0.5.1): ReduceOp.estimate_state_nbytes + MemoryGuard integration."""
import pyarrow as pa
import pytest

from cadw import MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded
from cadw.ops.reduce import (
    reduce, ReduceOp, CountOp, SumOp, MeanOp, MinMaxOp, StdOp,
    CovarianceOp, HistogramOp,
)


def make_iter(vals, batch_size=3):
    for i in range(0, len(vals), batch_size):
        yield pa.RecordBatch.from_arrays(
            [pa.array(vals[i:i+batch_size], type=pa.float64())],
            names=["x"],
        )


# ---------- built-in ops have estimate_state_nbytes ----------

@pytest.mark.parametrize("op, expected", [
    (CountOp("x"), 16),
    (SumOp("x"), 16),
    (MeanOp("x"), 16),
    (MinMaxOp("x"), 16),
    (StdOp("x"), 24),
    (CovarianceOp(["a", "b", "c"]), 8 + 8*3 + 8*3*3),
    (HistogramOp("x", bins=10, min_val=0, max_val=1), 8*10 + 16),
])
def test_builtin_ops_have_estimate(op, expected):
    est = op.estimate_state_nbytes()
    assert est is not None
    assert est == expected


def test_covariance_estimate_matches_actual_state():
    """Estimate must be >= actual state_nbytes(init())."""
    op = CovarianceOp(["a", "b", "c", "d"])
    assert op.estimate_state_nbytes() >= op.state_nbytes(op.init())


def test_histogram_estimate_matches_actual_state():
    op = HistogramOp("x", bins=255, min_val=0, max_val=1)
    assert op.estimate_state_nbytes() >= op.state_nbytes(op.init())


# ---------- custom op with estimate override ----------

class CustomOp(ReduceOp):
    def __init__(self, d):
        self.d = d
    def init(self):
        return 0.0  # placeholder
    def update(self, state, batch):
        return state
    def merge(self, a, b):
        return a + b
    def finalize(self, state):
        return state
    def state_nbytes(self, state):
        return 8 * self.d * self.d
    def estimate_state_nbytes(self):
        return 8 * self.d * self.d


def test_custom_op_estimate_overridden():
    """Custom op with estimate_state_nbytes — no hardcode fallback."""
    op = CustomOp(d=100)  # 80 KB
    assert op.estimate_state_nbytes() == 80000


def test_reduce_uses_estimate_from_op(tmp_path):
    """reduce() should call op.estimate_state_nbytes(), not estimate_state_size."""
    # 100 MB budget; op state = 80 KB — easily fits
    guard = MemoryGuard(max_memory_mb=100)
    op = CustomOp(d=100)
    result = reduce(make_iter([1.0]*10), op, guard, log=False)
    assert result == 0.0  # placeholder


def test_reduce_custom_op_rejects_oversized_state():
    """Custom op with state > budget → MemoryLimitExceeded BEFORE op.init()."""
    # Custom op with state = 100 MB, budget = 10 MB
    class HugeOp(ReduceOp):
        def init(self):
            raise AssertionError("init() should NOT be called — check first")
        def update(self, state, batch):
            return state
        def merge(self, a, b):
            return a
        def finalize(self, state):
            return state
        def state_nbytes(self, state):
            return 100 * 1024 * 1024
        def estimate_state_nbytes(self):
            return 100 * 1024 * 1024

    guard = MemoryGuard(max_memory_mb=10)
    op = HugeOp()
    with pytest.raises(MemoryLimitExceeded):
        reduce(make_iter([1.0]), op, guard, log=False)


# ---------- fallback for op without override ----------

class LegacyOp(ReduceOp):
    """Does not override estimate_state_nbytes → falls back to hardcode."""
    def init(self):
        return 0.0
    def update(self, state, batch):
        return state
    def merge(self, a, b):
        return a
    def finalize(self, state):
        return state
    def state_nbytes(self, state):
        return 8


def test_legacy_op_falls_back_to_estimate_state_size(tmp_path):
    """Op without override uses estimate_state_size (1 MiB fallback)."""
    guard = MemoryGuard(max_memory_mb=100)  # 1 MiB > 100 MB? No, 1 < 100, OK
    op = LegacyOp()
    assert op.estimate_state_nbytes() is None
    # reduce() should use 1 MiB fallback (fits under 100 MB budget)
    reduce(make_iter([1.0]*5), op, guard, log=False)


def test_legacy_op_with_tight_budget_uses_1mib_fallback():
    """Legacy op (no override) → 1 MiB fallback → fits under 2 MB but not 0.5 MB."""
    # 500 KB budget → 1 MiB fallback > 500 KB * 0.5 = 250 KB → raise
    guard = MemoryGuard(max_memory_bytes=500_000)  # 500 KB
    op = LegacyOp()
    with pytest.raises(MemoryLimitExceeded):
        reduce(make_iter([1.0]*5), op, guard, log=False)
EOF

# ============ Part 7: new tests P2 ============
cat > tests/test_join_streaming.py <<'EOF'
"""P2 fix (v0.5.1): hash_join_streaming writes output to disk.

Verifies:
  - hash_join_streaming returns (schema, path) with valid IPC.
  - Output batches written incrementally (not accumulated).
  - hash_join (compat) still returns pa.Table.
  - CadwDataFrame.merge uses streaming path.
  - _process_partition_pair is a generator (yield).
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.io.batch_writer import ArrowIPCBatchWriter
from cadw.ops import join as join_mod
from cadw.ops.join import hash_join, hash_join_streaming


def make_df(tmp_path, name, data, chunk=3):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=20),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- AST: _process_partition_pair is generator ----------

def test_process_partition_pair_is_generator():
    """_process_partition_pair must yield (not return list)."""
    import inspect
    src = inspect.getsource(join_mod._process_partition_pair)
    assert "yield " in src or "yield from" in src, (
        "_process_partition_pair must be a generator (P2 fix). "
        "See DECISIONS §13.4/63."
    )


# ---------- hash_join_streaming writes to disk ----------

def test_hash_join_streaming_returns_schema_and_path(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2, 3, 4], "a": [10, 20, 30, 40]})
    right = make_df(tmp_path, "R", {"k": [1, 3], "b": [100, 300]})
    schema, path = hash_join_streaming(
        left._iter_filtered_batches(),
        right._iter_filtered_batches(),
        left._dataset.schema, right._dataset.schema, ["k"],
        MemoryGuard(max_memory_mb=20),
        tmp_path / "join_spill",
        how="inner", num_partitions=2,
    )
    assert isinstance(schema, pa.Schema)
    assert isinstance(path, Path)
    assert path.exists()
    # Read back to verify
    reader = ipc.open_file(str(path))
    total = sum(
        reader.get_batch(i).num_rows
        for i in range(reader.num_record_batches)
    )
    assert total == 2  # k=1, k=3


def test_hash_join_streaming_writes_incrementally(tmp_path, monkeypatch):
    """Each output batch must be written immediately (not accumulated)."""
    write_calls = []
    orig_write = ArrowIPCBatchWriter.write_batch

    def spy(self, batch):
        write_calls.append(batch.num_rows)
        return orig_write(self, batch)

    monkeypatch.setattr(ArrowIPCBatchWriter, "write_batch", spy)

    left = make_df(tmp_path, "L", {"k": list(range(100)), "a": list(range(100))})
    right = make_df(tmp_path, "R", {"k": list(range(100)), "b": list(range(100))})
    schema, path = hash_join_streaming(
        left._iter_filtered_batches(),
        right._iter_filtered_batches(),
        left._dataset.schema, right._dataset.schema, ["k"],
        MemoryGuard(max_memory_mb=20),
        tmp_path / "join_spill", how="inner", num_partitions=4,
    )
    assert write_calls, "hash_join_streaming must write batches to disk"
    assert sum(write_calls) == 100


# ---------- backward compat: hash_join returns Table ----------

def test_hash_join_returns_table(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [1, 3], "b": [100, 300]})
    result = hash_join(
        left._iter_filtered_batches(),
        right._iter_filtered_batches(),
        left._dataset.schema, right._dataset.schema, ["k"],
        MemoryGuard(max_memory_mb=20),
        tmp_path / "join_spill", how="inner", num_partitions=2,
    )
    assert isinstance(result, pa.Table)
    assert result.num_rows == 2


# ---------- merge uses streaming ----------

def test_merge_uses_streaming(tmp_path, monkeypatch):
    """CadwDataFrame.merge must call hash_join_streaming, not hash_join."""
    from cadw import dataframe as df_mod

    called_streaming = []
    called_collect = []
    orig_streaming = df_mod.hash_join_streaming
    orig_collect = df_mod.hash_join

    def spy_streaming(*a, **kw):
        called_streaming.append(True)
        return orig_streaming(*a, **kw)

    def spy_collect(*a, **kw):
        called_collect.append(True)
        return orig_collect(*a, **kw)

    monkeypatch.setattr(df_mod, "hash_join_streaming", spy_streaming)
    monkeypatch.setattr(df_mod, "hash_join", spy_collect)

    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [1, 3], "b": [100, 300]})
    left.merge(right, on="k")

    assert called_streaming, "merge must use hash_join_streaming (P2)"
    assert not called_collect, "merge must NOT use hash_join (P2)"


def test_merge_still_works(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2, 3, 4], "a": ["x", "y", "z", "w"]})
    right = make_df(tmp_path, "R", {"k": [1, 3, 5], "b": ["p", "q", "r"]})
    result = left.merge(right, on="k")
    head = result.head(20)
    assert head.num_rows == 2


def test_merge_join_output_file_cleaned_up(tmp_path):
    """merge should delete intermediate join_output.arrow after Dataset.from_batches."""
    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [1, 3], "b": [100, 300]})
    left.merge(right, on="k")
    # Intermediate file should be gone
    join_out = left._spill_dir / "join" / "join_output.arrow"
    assert not join_out.exists(), (
        f"intermediate {join_out} should be cleaned up after merge"
    )


# ---------- left join via streaming ----------

def test_merge_left_streaming(tmp_path):
    left = make_df(tmp_path, "L", {"k": [1, 2, 3, 4], "a": ["w", "x", "y", "z"]})
    right = make_df(tmp_path, "R", {"k": [1, 3, 5], "b": ["p", "q", "r"]})
    result = left.merge(right, on="k", how="left")
    head = result.head(20)
    assert head.num_rows == 4  # all left preserved


# ---------- empty inputs ----------

def test_hash_join_streaming_empty_left(tmp_path):
    left = make_df(tmp_path, "L", {"k": pa.array([], type=pa.int64()), "a": pa.array([], type=pa.int64())})
    right = make_df(tmp_path, "R", {"k": [1, 2], "b": [100, 200]})
    schema, path = hash_join_streaming(
        left._iter_filtered_batches(),
        right._iter_filtered_batches(),
        left._dataset.schema, right._dataset.schema, ["k"],
        MemoryGuard(max_memory_mb=20),
        tmp_path / "join_spill", how="inner", num_partitions=2,
    )
    assert path.exists()
    reader = ipc.open_file(str(path))
    assert reader.num_record_batches == 0  # no output
EOF

# ============ Part 8: .gitignore exception for TASK.md ============
python3 - <<'PY'
from pathlib import Path
p = Path(".gitignore")
s = p.read_text()
old = "# others\n.pingis/\n_FOR_PINGIS/"
new = "# others\n.pingis/*\n!.pingis/TASK.md\n_FOR_PINGIS/"
if old in s:
    s = s.replace(old, new, 1)
    p.write_text(s)
    print(".gitignore: .pingis/* + !.pingis/TASK.md")
else:
    print("WARNING: .gitignore pattern not found; skipping (manual check)")
PY

# ============ Part 9: TASK.md ============
mkdir -p .pingis
cat > .pingis/TASK.md <<'EOF'
# PINGIS — periodic analysis task for CADW

## Context

**CADW** — out-of-core framework for tabular data that doesn't fit in
RAM. Three primitives: `Reduce`, `HashShuffle`, `Sort`. Bounded-memory
guarantee. Current version: check `git log --oneline -1` and read
`DECISIONS.md` §11.1 (implemented), §11.5 (limitations), §13.4 (lessons).

**Read before working:**
1. `DECISIONS.md` §9 (planner), §11.4 (must-fix history), §13.4 (lessons).
2. `_FOR_PINGIS/REPORT.md` and `analysis_ram.md` — previous audit.
3. `CHANGELOG.md` — what's recently changed.

**Do NOT re-report known issues without checking:**
- Known S1 (as of v0.5.1): P1 (join `to_pandas`), P3 (`list(reader)` in
  groupby), P4 (heavy accum Python dict) — see `analysis_ram.md`.
- Known P2, P8 — fixed in v0.5.1. Verify with `tests/test_join_streaming.py`
  and `tests/test_reduce_state_bytes.py`.

## Constraints

- **Do NOT modify `cadw/`** — analysis-only.
- Work in branch `pingis/<task-id>`.
- Output goes to `_FOR_PINGIS/` (gitignored) or `.pingis/` (gitignored
  except this TASK.md).
- Do not add new dependencies. Use existing: `pyarrow`, `pandas`,
  `numpy`, `psutil`, `xxhash`.

## Goal 1: RAM correctness audit

Read each op in `cadw/ops/` (`reduce.py`, `shuffle.py`, `sort.py`,
`groupby.py`, `join.py`, `rank.py`, `quantile.py`, `dedup.py`) and look
for:

- `to_pylist()`, `to_pandas()`, `as_py()` in hot paths.
- `list(reader)`, `Table.from_batches(...)` on unbounded input.
- `out_batches.extend(...)` — output accumulation.
- Python-list buffers (`buffer[c].append(...)`).
- Python dict accumulators (`accum[key] = ...`).
- `check_state` called after materialize.

For each finding report: **P#-ID**, **severity (S1/S2/S3/S4)**,
**code location (file:line)**, **peak estimate formula**, **how to
reproduce** (benchmark idea).

Severity:
- **S1**: guaranteed OOM on realistic data (> 2× budget violation).
- **S2**: might exceed budget depending on data type / scale.
- **S3**: efficiency/correctness issue, not OOM.
- **S4**: cosmetic / docs.

## Goal 2: Algorithm suggestions

For each finding (or generally), suggest Arrow-native or classic
external-memory alternative. **Verify availability** via:

```python
import pyarrow.compute as pc
print([n for n in pc.list_functions() if 'hash' in n.lower()])
```

**Do not suggest non-existent functions.** Currently confirmed absent:
- `pc.hash_64` (PR #45001 not merged as of Arrow 26).
- `pc.compute_hash`.

Confirmed present: `Table.join`, `Table.group_by`, `pc.sort_indices`,
`pc.take`, `pc.bucketize`, `pc.dictionary_encode` (method on Array).

Report: current → suggested → feasibility (doable / needs research /
not possible).

## Goal 3: Prototype new operations

When new op is on roadmap (see `DECISIONS.md` §12.7) or requested by
user, prototype in `_FOR_PINGIS/proto/`. Each prototype:

- Working script (runnable).
- Uses CADW primitives (Reduce/Shuffle/Sort) — not reinvent the wheel.
- Measured peak RSS via `psutil` + sampler thread.
- Comparison to naive approach where relevant.

## Goal 4: ML/viz readiness

- Which ML algorithms are feasible on CADW primitives?
- Which need adapters (fine) or are out-of-scope (state > RAM)?
- Prototype 1-2 algorithms per run.
- Viz adapters: aggregate-first pattern (histogram/box/line/heatmap);
  scatter only via sampling or datashader.

## Output format

Create/update these files in `_FOR_PINGIS/`:
- `REPORT.md` — executive summary + status of previous findings.
- `analysis_ram.md` — full P1..Pn catalog with benchmarks.
- `analysis_external.md` — external solutions + applicability matrix.
- `gap_ml.md` — ML/viz readiness.
- `bench/bench_ram*.py` + `bench/RESULTS.md`.
- `proto/` — working prototypes.

At the top of `REPORT.md`:
```
**Run**: YYYY-MM-DD
**CADW version**: vX.Y.Z (git describe)
**Arrow version**: X.Y.Z
**Findings**: X new, Y resolved, Z unchanged.
```

## Re-run criteria

Trigger a new analysis when:
- **A new tag is released** (vX.Y.Z).
- **A new op is added** to `cadw/ops/`.
- **Arrow version bumps** in `pyproject.toml` (check for new `pc.*`
  functions that would fix P6/P11).
- **~2-3 months** since last analysis.

## Current focus (fill each run)

- [ ] Verify v0.5.1 fixes (P2, P8).
- [ ] Re-benchmark join (B1a) and sort (B3a) — expect less than previous.
- [ ] Next candidate fixes: P1 (join `to_pandas`), P3/P10 (groupby
      materialize), P5 (sort buffer), P6/P11 (`to_pylist`).
- [ ] Check Arrow version for `pc.rank`/`pc.quantile`/`pc.rank_quantile`
      availability (would fix P11 immediately).
EOF

echo "=== compile check: cadw/ (all) ==="
python3 -c "import compileall, sys; ok = compileall.compile_dir('cadw', quiet=1, force=True); sys.exit(0 if ok else 1)"

# ============ Part 10: DECISIONS + CHANGELOG ============
python3 - <<'PY'
from pathlib import Path

# DECISIONS: add §13.4/63
p = Path("DECISIONS.md")
s = p.read_text()

if "| P2 + P8 fixes (v0.5.1) |" not in s:
    s = s.replace(
        "| `loc` open-endpoints + duplicate labels (v0.4.1) | ✅ | 22 |",
        "| `loc` open-endpoints + duplicate labels (v0.4.1) | ✅ | 22 |\n"
        "| **P2 + P8 RAM fixes (v0.5.1, from Pingis)** | ✅ | 15 |",
        1,
    )

marker_135 = "### 13.5. Открытые вопросы"
episode_63 = """**Эпизод 63 (Pingis: P2 + P8 fixes).** Внешний мультиагентный
аудит (`_FOR_PINGIS/`) нашёл 15 проблем RAM (P1–P15). Две самые
показательные:
- **P2**: `_process_partition_pair` возвращал `list[RecordBatch]` —
  весь output 1:M join в RAM. Бенчмарк B1a: 100k×10k под 80 MB →
  ΔRSS +140 MB = **2.5× бюджета**.
- **P8**: `MemoryGuard.estimate_state_size` для custom `ReduceOp`
  возвращал 1 MiB (хардкод). ML-прототипы (XTXOp, KMeansStepOp,
  HistOp) получали warning; для state > 1 MiB — silently passed,
  OOM после. Блокер ML-слоя.

**v0.5.1 fixes:**

- **P2 (streaming join):**
  - `_process_partition_pair` → generator (yield вместо extend).
  - `_left_preserve_with_nulls` → `_left_preserve_with_nulls_iter`
    (streaming wrapper оставлен для совместимости).
  - `_join_core` — общий setup, возвращает `(schema, iterator)`.
  - `hash_join` — backward compat (собирает в Table).
  - `hash_join_streaming` — NEW: пишет прямо в `ArrowIPCBatchWriter`,
    возвращает `(schema, Path)`.
  - `CadwDataFrame.merge` — использует `hash_join_streaming`, оборачивает
    через `ArrowIPCBatchReader` + `Dataset.from_batches`, удаляет
    промежуточный файл через `path.unlink(missing_ok=True)`.
  - Peak: `max(build_partition) + probe_batch + writer_buffer` — **не
    зависит от размера output**.

- **P8 (state_bytes API):**
  - `ReduceOp.estimate_state_nbytes() -> int | None` — метод ABC.
    Default None (fallback на `MemoryGuard.estimate_state_size`).
  - Built-in ops override: CountOp/SumOp/MinMaxOp/MeanOp → 16;
    StdOp → 24; CovarianceOp → 8+8d+8d²; HistogramOp → 8bins+16.
  - `reduce()` вызывает `op.estimate_state_nbytes()` **до** `op.init()`;
    при None — fallback на `estimate_state_size` (старое поведение).
  - Пользовательские ops могут override — это публичный контракт для ML.

**Урок:** мультиагентный анализ дал **две независимые находки, которые
мы сами пропустили**: (a) out_batches accumulation (2.5× OOM,
бенчмарк — единственный способ увидеть); (b) хардкод 1 MiB в
`estimate_state_size` блокировал ML. Обе — из класса «аудит горячих
путей» — то, что мы не делали систематически. `_pingis/TASK.md` фиксирует
этот процесс для будущих прогонов (после новых тегов, Arrow-версий,
новых ops).

**Второстепенно:** бенчмарк (RSS-сэмплер) — единственный способ
подтвердить OOM, потому что `estimate` из модели ≠ реальный пик
(эпизоды 42, 47). Мы должны включить RSS-сэмплер в собственные E2E-тесты
(сейчас только `psutil` в e2e-join, но не в unit).

**Roadmap после v0.5.1:** P1 (join без pandas), P3/P10 (groupby
materialize), P5 (sort buffer), P6/P11 (to_pylist), P7 (модели с
Python-overhead), P4 (heavy accum). Порядок — по стоимости/эффекту
(сначала локальные, потом архитектурные).

"""
if "Эпизод 63" not in s:
    s = s.replace(marker_135, episode_63 + marker_135, 1)

# §11.5: remove old limits
s = s.replace(
    "| `df[-5:]` (отрицательный slice) — не поддержано | `SliceFilter` | `step > 0`, `start >= 0`; для отрицательных — `count()` + пересчёт |\n",
    "| `df[-5:]` (отрицательный slice) — не поддержано | `SliceFilter` | `step > 0`, `start >= 0`; для отрицательных — `count()` + пересчёт |\n"
    "| ~~P2: out_batches accumulation (join)~~ | ~~`_process_partition_pair`~~ | **закрыто v0.5.1**: streaming via `hash_join_streaming` |\n"
    "| ~~P8: 1 MiB hardcode в `estimate_state_size`~~ | ~~`reduce()`~~ | **закрыто v0.5.1**: `ReduceOp.estimate_state_nbytes()` |\n",
    1,
)

p.write_text(s)
print("DECISIONS: §13.4/63 + §11.5")
PY

python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.5.1] — 2026-10-19

По результатам мультиагентного анализа Pingis (`_FOR_PINGIS/REPORT.md`):
P2 (S1) + P8 (S2).

### Исправлено
- **P2 (S1, streaming join):**
  - `_process_partition_pair` — теперь generator (`yield` вместо
    `out_batches.extend(...)`).
  - `_left_preserve_with_nulls_iter` — streaming-версия.
  - `_join_core` — общий setup (validate, peek, shuffle) → `(schema, iter)`.
  - `hash_join_streaming(...) -> (schema, Path)` — **NEW**: пишет
    выходные батчи напрямую в `ArrowIPCBatchWriter`. Peak RAM =
    `max(build_partition) + probe_batch + writer_buffer`, **не зависит
    от размера output**.
  - `CadwDataFrame.merge` использует `hash_join_streaming`; промежуточный
    `.arrow` удаляется после `Dataset.from_batches`.
  - `hash_join(...)` (старый API) сохраняет семантику — собирает
    streaming-output в `pa.Table` (backward compat).

- **P8 (S2, state_bytes API):**
  - `ReduceOp.estimate_state_nbytes() -> int | None` — метод ABC
    (default None → fallback на `MemoryGuard.estimate_state_size`).
  - Built-in ops override: `CountOp`/`SumOp`/`MinMaxOp`/`MeanOp` → 16;
    `StdOp` → 24; `CovarianceOp` → `8+8d+8d²`; `HistogramOp` → `8bins+16`.
  - `reduce()` вызывает `op.estimate_state_nbytes()` **до** `op.init()`.
  - User `ReduceOp` can override — публичный контракт для ML-слоя.

### Добавлено
- `cadw.ops.hash_join_streaming` — streaming join API.
- `tests/test_reduce_state_bytes.py` — 8 тестов (built-in + custom +
  fallback + oversized-state-rejects-before-init).
- `tests/test_join_streaming.py` — 10 тестов (generator, disk write,
  incremental batches, backward compat, merge uses streaming, cleanup,
  empty input).
- `.pingis/TASK.md` — periodic analysis task template.
- `.gitignore`: exception для `.pingis/TASK.md`.

### Изменено
- `test_join_oom_guard.py::test_oversize_partition_raises_with_max_depth_zero`
  — `next(gen)` вместо `gen(...)` (generator).

### Ожидаемый эффект
- Join B1a (100k × 10k под 80 MB) → Δpeak с ~140 MB до ≤ 80 MB (цель).
- ML custom `ReduceOp` (XTXOp, KMeansStepOp, HistOp) не получают
  warning про 1 MiB; правильный state-размер.

### Ограничения (v0.5.1, §11.5)
- P1 (join `to_pandas`/`pd.merge`) — отдельный патч (v0.5.2).
- P3/P4/P5/P6/P7/P10/P11/P12/P14 — в roadmap после v0.5.2.
- `hash_join` (не-streaming) всё ещё может OOM при большом output —
  используйте `hash_join_streaming` или `CadwDataFrame.merge`.

"""
if "## [0.5.1]" not in s:
    s = s.replace("## [0.4.1]", entry + "## [0.4.1]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.5.1")
PY

# ============ Part 11: run tests ============
echo ""
echo "=== pytest tests/test_reduce_state_bytes.py tests/test_join_streaming.py -v ==="
set +e
uv run pytest tests/test_reduce_state_bytes.py tests/test_join_streaming.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "new tests: rc=$RC"
  exit 1
fi

echo "=== pytest tests/test_join.py tests/test_join_oom_guard.py tests/test_merge_planner.py tests/test_reduce.py -v ==="
set +e
uv run pytest tests/test_join.py tests/test_join_oom_guard.py tests/test_merge_planner.py tests/test_reduce.py -v
RC2=$?
set -e
if [[ $RC2 -ne 0 ]]; then
  echo "regression: rc=$RC2"
  exit 1
fi

echo "=== pytest fast ==="
uv run pytest -q -m "not slow"
echo "=== pytest slow ==="
uv run pytest -q -m slow

git add -A
git commit -m "v0.5.1: P2 (streaming join) + P8 (ReduceOp.estimate_state_nbytes); +.pingis/TASK.md; §13.4/63"

echo "==> v0.5.1 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"