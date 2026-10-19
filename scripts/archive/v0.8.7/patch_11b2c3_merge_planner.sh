#!/usr/bin/env bash
# patch_11b2c3_merge_planner: интеграция JoinMemoryModel в merge;
# разделение max_output_rows (expected) и max_cartesian_rows (worst case).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-11b2c3-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_11b2c3"

BR="patch/11b2c3-merge-planner"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- 1. models_join.py: c_cartesian + max_cartesian_rows ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/core/models_join.py")
s = p.read_text()

# 1a. JoinParams: новое поле
s = s.replace(
    '''@dataclass
class JoinParams(Params):
    build_side: str         # "left" | "right"
    how: str                # "inner" | "left"
    num_partitions: int
    probe_batch_size: int
    build_partition_bytes: int
    max_output_rows: int
    right_chars: DataChars
    probe_row_size: int
    output_row_size: int
    avg_matches: float''',
    '''@dataclass
class JoinParams(Params):
    build_side: str         # "left" | "right"
    how: str                # "inner" | "left"
    num_partitions: int
    probe_batch_size: int
    build_partition_bytes: int
    max_output_rows: int
    max_cartesian_rows: int   # worst case (для runtime защиты от skew)
    right_chars: DataChars
    probe_row_size: int
    output_row_size: int
    avg_matches: float''',
    1,
)

# 1b. __init__: c_cartesian
s = s.replace(
    '''        build_ratio: float = 0.4,       # CALIBRATION DEFAULT (§9.5)
        probe_ratio: float = 0.3,       # CALIBRATION DEFAULT
        output_ratio: float = 0.3,      # CALIBRATION DEFAULT
        c_merge_ratio: float = 2.0,     # pandas temp ≈ 2 × output
    ) -> None:''',
    '''        build_ratio: float = 0.4,       # CALIBRATION DEFAULT (§9.5)
        probe_ratio: float = 0.3,       # CALIBRATION DEFAULT
        output_ratio: float = 0.3,      # CALIBRATION DEFAULT
        c_merge_ratio: float = 2.0,     # pandas temp ≈ 2 × output
        c_cartesian: float = 10.0,      # max_cartesian = c_cartesian × max_output
    ) -> None:''',
    1,
)

s = s.replace(
    '''        if c_merge_ratio < 0:
            raise ValueError("c_merge_ratio must be >= 0")
        self.build_ratio = float(build_ratio)
        self.probe_ratio = float(probe_ratio)
        self.output_ratio = float(output_ratio)
        self.c_merge_ratio = float(c_merge_ratio)''',
    '''        if c_merge_ratio < 0:
            raise ValueError("c_merge_ratio must be >= 0")
        if c_cartesian < 1.0:
            raise ValueError("c_cartesian must be >= 1.0")
        self.build_ratio = float(build_ratio)
        self.probe_ratio = float(probe_ratio)
        self.output_ratio = float(output_ratio)
        self.c_merge_ratio = float(c_merge_ratio)
        self.c_cartesian = float(c_cartesian)''',
    1,
)

# 1c. solve: вычислить max_cartesian_rows
s = s.replace(
    '''        max_output_rows = max(
            1, int(output_budget / max(1.0, row_size_out * (1.0 + self.c_merge_ratio)))
        )

        params = JoinParams(
            build_side=build_side,
            how=how,
            num_partitions=int(P),
            probe_batch_size=int(probe_batch_size),
            build_partition_bytes=int(build_partition_bytes),
            max_output_rows=int(max_output_rows),
            right_chars=right_chars,''',
    '''        max_output_rows = max(
            1, int(output_budget / max(1.0, row_size_out * (1.0 + self.c_merge_ratio)))
        )
        # worst-case: cartesian product. Не связан с output_budget (merge
        # материализует только совпадения, не cartesian); c_cartesian —
        # множитель от expected для detection катастрофического skew.
        max_cartesian_rows = max(
            max_output_rows + 1,
            int(self.c_cartesian * max_output_rows),
        )

        params = JoinParams(
            build_side=build_side,
            how=how,
            num_partitions=int(P),
            probe_batch_size=int(probe_batch_size),
            build_partition_bytes=int(build_partition_bytes),
            max_output_rows=int(max_output_rows),
            max_cartesian_rows=int(max_cartesian_rows),
            right_chars=right_chars,''',
    1,
)

# 1d. describe: упомянуть c_cartesian
s = s.replace(
    '''            f"output_ratio={self.output_ratio:.2f}, "
            f"c_merge={self.c_merge_ratio:.2f}"
        )''',
    '''            f"output_ratio={self.output_ratio:.2f}, "
            f"c_merge={self.c_merge_ratio:.2f}, "
            f"c_cartesian={self.c_cartesian:.2f}"
        )''',
    1,
)

p.write_text(s)
print("models_join.py: c_cartesian + max_cartesian_rows")
PY

# ---------- 2. join.py: max_cartesian_rows + per-partition unique check ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/join.py")
s = p.read_text()

# 2a. hash_join signature: max_cartesian_rows
s = s.replace(
    '''    hash_seed: int = 0,
    max_output_rows: Optional[int] = None,
) -> pa.Table:
    """Grace hash join. how ∈ {'inner', 'left'} (v0.2.2).''',
    '''    hash_seed: int = 0,
    max_output_rows: Optional[int] = None,
    max_cartesian_rows: Optional[int] = None,
) -> pa.Table:
    """Grace hash join. how ∈ {'inner', 'left'} (v0.2.2).''',
    1,
)

# 2b. hash_join: default max_cartesian_rows из max_output_rows
s = s.replace(
    '''    if max_output_rows is None:
        # Консервативная оценка: 0.3 * available / 256 B на строку
        max_output_rows = max(
            1, int(memory_guard.available_bytes() * 0.3 / 256)
        )

    empty_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)''',
    '''    if max_output_rows is None:
        # Консервативная оценка: 0.3 * available / 256 B на строку
        max_output_rows = max(
            1, int(memory_guard.available_bytes() * 0.3 / 256)
        )
    if max_cartesian_rows is None:
        # Default: 10× expected (c_cartesian из JoinMemoryModel)
        max_cartesian_rows = max_output_rows * 10

    empty_schema = _empty_result_schema(left_schema, right_schema, on, suffixes)''',
    1,
)

# 2c. _process_partition_pair: новый параметр + per-partition check
s = s.replace(
    '''    depth: int = 0,
    max_depth: int = 5,
    max_output_rows: int = 0,
) -> list[pa.RecordBatch]:''',
    '''    depth: int = 0,
    max_depth: int = 5,
    max_output_rows: int = 0,
    max_cartesian_rows: int = 0,
) -> list[pa.RecordBatch]:''',
    1,
)

s = s.replace(
    '''                    num_partitions_recurse=num_partitions_recurse,
                    depth=depth + 1, max_depth=max_depth,
                    max_output_rows=max_output_rows,
                )
            )
        return out''',
    '''                    num_partitions_recurse=num_partitions_recurse,
                    depth=depth + 1, max_depth=max_depth,
                    max_output_rows=max_output_rows,
                    max_cartesian_rows=max_cartesian_rows,
                )
            )
        return out''',
    1,
)

# 2d. _process_partition_pair: перед probe loop — вычислить unique_build_keys
s = s.replace(
    '''    build_df = _drop_null_keys(build_table.to_pandas(), list(on))
    if len(build_df) == 0:
        if how == "inner":
            return []
        if build_df.shape[1] == 0:
            right_rest = [c for c in right_schema.names if c not in on]
            build_df = pd.DataFrame(columns=list(on) + right_rest)
    # how='left': пустой build — продолжаем (left preserved).

    out_batches: list[pa.RecordBatch] = []''',
    '''    build_df = _drop_null_keys(build_table.to_pandas(), list(on))
    if len(build_df) == 0:
        if how == "inner":
            return []
        if build_df.shape[1] == 0:
            right_rest = [c for c in right_schema.names if c not in on]
            build_df = pd.DataFrame(columns=list(on) + right_rest)
    # how='left': пустой build — продолжаем (left preserved).

    # Per-partition skew check: unique build keys + avg matches per probe.
    # Дешёвая операция на уже материализованном build_df (pandas .nunique
    # или drop_duplicates + shape). Верхняя оценка через это позволяет
    # ловить partition-level skew, который модель не видела (она
    # глобальная).
    unique_build_keys = len(build_df)
    if len(build_df) > 0 and len(on) > 0:
        try:
            if len(on) == 1:
                unique_build_keys = int(build_df[on[0]].nunique(dropna=True))
            else:
                unique_build_keys = int(
                    build_df.drop_duplicates(subset=list(on)).shape[0]
                )
            unique_build_keys = max(1, unique_build_keys)
        except Exception:
            unique_build_keys = len(build_df)
    avg_matches_per_probe = (
        len(build_df) / max(1, unique_build_keys)
        if unique_build_keys > 0 else 1.0
    )

    out_batches: list[pa.RecordBatch] = []''',
    1,
)

# 2e. Заменить старую проверку (cartesian > max_output_rows) на две
s = s.replace(
    '''        if len(build_df) * len(probe_df) > max_output_rows:
            raise MemoryLimitExceeded(
                f"hash_join(how={how!r}): estimated output rows "
                f"{len(build_df)} * {len(probe_df)} = "
                f"{len(build_df) * len(probe_df)} > max_output_rows="
                f"{max_output_rows}. Likely skew."
            )''',
    '''        # 1. Expected output (avg_matches × probe) против max_output_rows.
        #    Ловит per-partition skew: если в этой партиции build дублируется
        #    сильнее, чем в среднем по датасету.
        expected_probe_output = int(
            len(probe_df) * max(1.0, avg_matches_per_probe)
        )
        if expected_probe_output > max_output_rows:
            raise MemoryLimitExceeded(
                f"hash_join(how={how!r}): expected output from this "
                f"partition ≈ {expected_probe_output} rows "
                f"({len(probe_df)} probe × {avg_matches_per_probe:.2f} "
                f"avg_matches) > max_output_rows={max_output_rows}. "
                f"Likely partition-level skew."
            )
        # 2. Cartesian worst case (build × probe) против max_cartesian_rows.
        #    Ловит экстремальный skew, где даже probe-строка с уникальным
        #    ключом матчит много build-строк.
        cartesian = len(build_df) * len(probe_df)
        if cartesian > max_cartesian_rows:
            raise MemoryLimitExceeded(
                f"hash_join(how={how!r}): cartesian worst case "
                f"{len(build_df)} × {len(probe_df)} = {cartesian} > "
                f"max_cartesian_rows={max_cartesian_rows}. "
                f"Likely extreme skew."
            )''',
    1,
)

# 2f. Обновить вызовы _process_partition_pair в hash_join (передать max_cartesian)
s = s.replace(
    '''        all_batches.extend(
            _process_partition_pair(
                lp, rp, left_schema, right_schema, on, suffixes,
                memory_guard, spill_dir, how=how,
                max_output_rows=max_output_rows,
            )
        )''',
    '''        all_batches.extend(
            _process_partition_pair(
                lp, rp, left_schema, right_schema, on, suffixes,
                memory_guard, spill_dir, how=how,
                max_output_rows=max_output_rows,
                max_cartesian_rows=max_cartesian_rows,
            )
        )''',
    1,
)

# 2g. inner_join / left_join обёртки: пробросить max_cartesian_rows
for wrapper in ("inner_join", "left_join"):
    old_sig = f'''def {wrapper}(
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
) -> pa.Table:'''
    new_sig = f'''def {wrapper}(
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
    max_cartesian_rows: Optional[int] = None,
) -> pa.Table:'''
    s = s.replace(old_sig, new_sig, 1)

s = s.replace(
    '''        how="inner", suffixes=suffixes, num_partitions=num_partitions,
        hash_seed=hash_seed, max_output_rows=max_output_rows,
    )''',
    '''        how="inner", suffixes=suffixes, num_partitions=num_partitions,
        hash_seed=hash_seed, max_output_rows=max_output_rows,
        max_cartesian_rows=max_cartesian_rows,
    )''',
    1,
)
s = s.replace(
    '''        how="left", suffixes=suffixes, num_partitions=num_partitions,
        hash_seed=hash_seed, max_output_rows=max_output_rows,
    )''',
    '''        how="left", suffixes=suffixes, num_partitions=num_partitions,
        hash_seed=hash_seed, max_output_rows=max_output_rows,
        max_cartesian_rows=max_cartesian_rows,
    )''',
    1,
)

p.write_text(s)
print("join.py: max_cartesian_rows + per-partition unique_build check")
PY

# ---------- 3. dataframe.py: use_planner + sampling + mg.plan ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# 3a. import: data_chars_from_sample, estimate_unique_keys helper
if "from cadw.core.models import data_chars_from_sample" not in s:
    s = s.replace(
        "from cadw.core.memory_guard import MemoryGuard",
        "from cadw.core.memory_guard import MemoryGuard\n"
        "from cadw.core.models import DataChars, data_chars_from_sample",
        1,
    )

# 3b. Заменить merge()
old_merge = '''    def merge(
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

new_merge = '''    def merge(
        self,
        right: "CadwDataFrame",
        on: str | Sequence[str],
        how: str = "inner",
        *,
        suffixes: tuple[str, str] = ("_x", "_y"),
        num_partitions: int | None = None,
        use_planner: bool = True,
    ) -> "CadwDataFrame":
        """Hash join (v0.2.4). how ∈ {'inner', 'left'}.

        - how='inner': только совпадающие ключи.
        - how='left':  все строки self сохраняются; right-колонки -> NULL
          при отсутствии совпадения.
        - on: имена общих колонок в обеих сторонах.
        - suffixes: для пересекающихся не-on колонок.
        - num_partitions: число партиций HashShuffle; None — из
          JoinMemoryModel (если use_planner), иначе дефолт 8.
        - use_planner: использовать JoinMemoryModel.solve() для выбора
          num_partitions / max_output_rows / max_cartesian_rows. Если
          False — runtime использует старые дефолты (0.3 * available / 256).
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

        # --- Planner: sample first batch, plan, override defaults. ---
        max_output_rows: int | None = None
        max_cartesian_rows: int | None = None

        if use_planner:
            left_sample = _peek_first(self._iter_filtered_batches())
            right_sample = _peek_first(right._iter_filtered_batches())

            # Пустая сторона: короткое замыкание (runtime обработает).
            if left_sample is not None and right_sample is not None:
                left_chars = _data_chars_for_join(
                    left_sample, self._dataset.total_rows,
                    on_list, self._memory_guard,
                )
                right_chars = _data_chars_for_join(
                    right_sample, right._dataset.total_rows,
                    on_list, self._memory_guard,
                )
                params = self._memory_guard.plan(
                    "join", left_chars,
                    op_context={"right_chars": right_chars, "how": how},
                )
                if num_partitions is None:
                    num_partitions = params.num_partitions
                max_output_rows = params.max_output_rows
                max_cartesian_rows = params.max_cartesian_rows
            else:
                # Одна из сторон пуста — planner не нужен; hash_join
                # короткозамкнёт. num_partitions не важен.
                if num_partitions is None:
                    num_partitions = 8
        else:
            if num_partitions is None:
                num_partitions = 8

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
            max_output_rows=max_output_rows,
            max_cartesian_rows=max_cartesian_rows,
        )
        return self._wrap_result_table(result_table)'''

if old_merge not in s:
    raise SystemExit("old merge not found")
s = s.replace(old_merge, new_merge, 1)

# 3c. Helper-функции в конце модуля, перед _GroupBy
helpers = '''

def _peek_first(gen) -> "pa.RecordBatch | None":
    """Первый батч генератора или None (для sampling в merge)."""
    for b in gen:
        return b
    return None


def _data_chars_for_join(
    sample: pa.RecordBatch,
    n_total: int,
    on_cols: list[str],
    memory_guard: MemoryGuard,
) -> DataChars:
    """DataChars для join: k_est — из unique на on-колонках sample, с
    экстраполяцией на n_total.

    size_of_key_entry — консервативно row_size + 80 (agg state).
    avg_key_size — средний размер ключа (repr().encode()).
    """
    if sample.num_rows == 0:
        raise ValueError("_data_chars_for_join: empty sample")
    # k_est: по on-колонкам sample, экстраполяция на total
    on_sample = sample.select([c for c in on_cols if c in sample.schema.names])
    k_est = memory_guard.estimate_unique_keys(on_sample, n_total)

    # avg_key_size из first row repr().encode() — консервативно
    if sample.num_rows > 0:
        first = {c: sample.column(c)[0].as_py() for c in on_cols
                 if c in sample.schema.names}
        avg_key_size = len(repr(tuple(first.values())).encode())
    else:
        avg_key_size = 16

    return data_chars_from_sample(
        sample, n_total,
        k_est=k_est,
        avg_key_size_bytes=avg_key_size,
        size_of_key_entry_bytes=avg_key_size + 80,
    )

'''
anchor = "class _GroupBy:"
if anchor not in s:
    raise SystemExit("_GroupBy not found")
s = s.replace(anchor, helpers + anchor, 1)

p.write_text(s)
print("dataframe.py: merge with use_planner + sampling + mg.plan(join)")
PY

# ---------- 4. tests/test_merge_planner.py ----------
cat > tests/test_merge_planner.py <<'EOF'
"""Тесты интеграции JoinMemoryModel в merge (11-B-2c-3)."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cadw import CadwDataFrame, MemoryGuard
from cadw.core.memory_guard import MemoryLimitExceeded


def make_df(tmp_path, name, data, chunk=3, mem_mb=20):
    t = pa.table(data)
    p = Path(tmp_path) / f"{name}.parquet"
    pq.write_table(t, p)
    return CadwDataFrame.from_parquet(
        p, Path(tmp_path) / f"ds_{name}", chunk_size=chunk,
        memory_guard=MemoryGuard(max_memory_mb=mem_mb),
        spill_dir=Path(tmp_path) / f"spill_{name}",
    )


# ---------- planner используется по умолчанию ----------

def test_merge_uses_planner_num_partitions(tmp_path):
    """num_partitions=None + use_planner=True → модель выбирает P."""
    left = make_df(tmp_path, "L", {"k": [1, 2, 3, 4], "a": ["x", "y", "z", "w"]})
    right = make_df(tmp_path, "R", {"k": [1, 3, 5], "b": ["p", "q", "r"]})
    # Не падаем, merge возвращает результат (plan выбрал P)
    result = left.merge(right, on="k")  # use_planner=True default
    assert result.shape[0] == 2


def test_merge_use_planner_false(tmp_path):
    """use_planner=False → дефолт num_partitions=8, работает."""
    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [1, 3], "b": [100, 300]})
    result = left.merge(right, on="k", use_planner=False)
    assert result.shape[0] == 2


def test_merge_explicit_num_partitions_overrides_planner(tmp_path):
    """num_partitions явно → planner всё равно считает limits."""
    left = make_df(tmp_path, "L", {"k": [1, 2, 3], "a": [10, 20, 30]})
    right = make_df(tmp_path, "R", {"k": [1, 3], "b": [100, 300]})
    result = left.merge(right, on="k", num_partitions=2)
    assert result.shape[0] == 2


# ---------- early failure когда план не сходится ----------

def test_merge_raises_when_plan_fails(tmp_path):
    """Огромные стороны под 1 MB → MemoryLimitExceeded от mg.plan."""
    left = make_df(
        tmp_path, "L",
        {"k": list(range(10_000)), "a": list(range(10_000))},
        chunk=1000, mem_mb=20,
    )
    right = make_df(
        tmp_path, "R",
        {"k": list(range(10_000)), "b": list(range(10_000))},
        chunk=1000, mem_mb=20,
    )
    # merge использует mg из left (20 MB); для join двух 10k-row сторон
    # с row_size ~16 B это должно влезть... Уменьшим бюджет до 100 KB.
    left._memory_guard = MemoryGuard(max_memory_bytes=100_000)
    with pytest.raises(MemoryLimitExceeded):
        left.merge(right, on="k")


# ---------- planner + skew: runtime защищает ----------

def test_merge_cartesian_skew_raises(tmp_path):
    """Один тяжёлый ключ: 1000×1000 cartesian → MemoryLimitExceeded."""
    # Left: 1000 строк с одним ключом; right: 1000 строк с тем же ключом
    left = make_df(
        tmp_path, "L",
        {"k": ["HOT"] * 1000, "a": list(range(1000))},
        chunk=200, mem_mb=20,
    )
    right = make_df(
        tmp_path, "R",
        {"k": ["HOT"] * 1000, "b": list(range(1000))},
        chunk=200, mem_mb=20,
    )
    with pytest.raises(MemoryLimitExceeded, match="cartesian|expected"):
        left.merge(right, on="k")


# ---------- planner + left join + PK-FK ----------

def test_merge_left_planner(tmp_path):
    """Left join через planner: PK-FK, ожидаемый output = len(left)."""
    left = make_df(
        tmp_path, "L",
        {"k": [1, 2, 3, 4], "a": ["w", "x", "y", "z"]},
    )
    right = make_df(tmp_path, "R", {"k": [1, 3, 5], "b": ["p", "q", "r"]})
    result = left.merge(right, on="k", how="left")
    assert result.shape[0] == 4


# ---------- empty side ----------

def test_merge_empty_right_with_planner(tmp_path):
    """Пустой right + planner → inner даёт пусто."""
    left = make_df(tmp_path, "L", {"k": [1, 2], "a": [10, 20]})
    right_data = {
        "k": pa.array([], type=pa.int64()),
        "b": pa.array([], type=pa.string()),
    }
    right = make_df(tmp_path, "R", right_data)
    result = left.merge(right, on="k")
    assert result.shape[0] == 0


def test_merge_empty_right_left_with_planner(tmp_path):
    """Пустой right + planner + how='left' → left preserved."""
    left = make_df(tmp_path, "L", {"k": [1, 2], "a": [10, 20]})
    right_data = {
        "k": pa.array([], type=pa.int64()),
        "b": pa.array([], type=pa.string()),
    }
    right = make_df(tmp_path, "R", right_data)
    result = left.merge(right, on="k", how="left")
    assert result.shape[0] == 2
EOF

# ---------- 5. DECISIONS: §11.1 + §13.4/42 closed + эпизод 43 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

# §11.1: строка
if "| `CadwDataFrame.merge` + `JoinMemoryModel` (v0.2.4) |" not in s:
    s = s.replace(
        '| `propagate_chars("join")` (v0.2.3: planner chain с join) | ✅ | 9 |',
        '| `propagate_chars("join")` (v0.2.3: planner chain с join) | ✅ | 9 |\n'
        '| `CadwDataFrame.merge` + `JoinMemoryModel` (v0.2.4) | ✅ | 8 |',
        1,
    )

# §12.6: progress
s = s.replace(
    "§12.6. v0.2 (в работе: inner+left ✅, JoinMemoryModel ✅, propagate ✅; merge integration, E2E — 11-B-2c-3/2d)",
    "§12.6. v0.2 (в работе: inner+left ✅, JoinMemoryModel ✅, propagate ✅, merge+planner ✅; E2E — 11-B-2d)",
    1,
)

# Эпизод 42 → closed
s = s.replace(
    "**Статус:** 11-B-2c-3 — согласовать: либо runtime использует\n"
    "`max_output_rows` как expected (а cartesian вынести в отдельный лимит\n"
    "`max_cartesian_rows`), либо модель использует cartesian как upper bound\n"
    "(over-conservative, отвергает валидные join'ы). Мой выбор — первое:\n"
    "runtime проверяет `len(build) × len(probe) > max_cartesian_rows` (с\n"
    "большим запасом, `c_cartesian ≈ 10×` от expected) и `expected_output >\n"
    "max_output_rows`. Это разделяет «защиту от OOM» (expected) и «защиту от\n"
    "skew» (cartesian). **Урок:** модель предсказывает **ожидаемый** пик,\n"
    "runtime защищает от **worst case**. Это два разных числа, их нельзя\n"
    "смешивать в одном параметре.",
    "**Статус (закрыто 11-B-2c-3):** разделены `max_output_rows` (expected,\n"
    "из `output_budget / row_size_out`) и `max_cartesian_rows` (`c_cartesian\n"
    "× max_output_rows`, default 10×). Runtime проверяет:\n"
    "- `expected_probe_output = len(probe) × avg_matches_per_probe >\n"
    "  max_output_rows` → raise (per-partition skew);\n"
    "- `len(build) × len(probe) > max_cartesian_rows` → raise (extreme skew);\n"
    "где `avg_matches_per_probe = len(build) / unique_build_keys` — оценка\n"
    "по **фактической** build-партиции, а не глобальная. Это позволяет\n"
    "ловить partition-level skew, который глобальная модель не видит.\n"
    "`CadwDataFrame.merge(use_planner=True)` пробрасывает `num_partitions`,\n"
    "`max_output_rows`, `max_cartesian_rows` из `mg.plan(\"join\", ...)`.\n"
    "**Урок:** модель предсказывает **ожидаемый** пик, runtime защищает\n"
    "от **worst case**. Это два разных числа, их нельзя смешивать в одном\n"
    "параметре. Плюс — runtime может compute more precise estimate per\n"
    "partition (unique_build_keys), чем глобальная модель.",
    1,
)

# Эпизод 43
marker_135 = "### 13.5. Открытые вопросы"
episode_43 = """**Эпизод 43 (sample-based plan в merge).** `CadwDataFrame.merge` теперь
сэмплирует первый батч с обеих сторон (`_peek_first`), вычисляет `DataChars`
через `_data_chars_for_join` (`k_est` из `MemoryGuard.estimate_unique_keys`
на on-колонках sample, `avg_key_size` из repr().encode() первой строки),
и вызывает `mg.plan("join", left_chars, op_context={"right_chars": ...})`.
Плюсы: пользователь получает раннюю ошибку `MemoryLimitExceeded`, если
план не сходится, а не OOM в середине merge. Минусы: (a) sample — это
первый батч, может быть нерепрезентативным; (b) `k_est` экстраполируется
наивно (`unique_in_sample × n_total / sample_rows`), при сильно скошенном
sample — over-/under-estimate; (c) `_iter_filtered_batches` теперь
вызывается дважды (sample + real), но sample = первый батч, цена
пренебрежима. **Урок:** planner в операции — баланс между точностью
(нужен sample) и стоимостью (лишний проход). Первый батч — разумный
компромисс: не читает всё, даёт порядок величины. Если пользователь
знает cardinality лучше, `use_planner=False` + явный `num_partitions`.

"""
if "Эпизод 43" not in s:
    s = s.replace(marker_135, episode_43 + marker_135, 1)

p.write_text(s)
print("DECISIONS: §11.1, §12.6, §13.4/42 closed + /43")
PY

# ---------- 6. CHANGELOG: v0.2.4 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
entry = """## [0.2.4] — 2026-10-19

### Исправлено (§13.4 эпизод 42)
- Модель `JoinMemoryModel` возвращает два лимита: `max_output_rows`
  (expected, из `output_budget / row_size_out`) и
  `max_cartesian_rows` (`c_cartesian × max_output_rows`, default 10×).
  Runtime `hash_join` разделяет проверки:
  - `expected_probe_output = len(probe) × avg_matches_per_probe >
    max_output_rows` → raise (per-partition skew);
  - `len(build) × len(probe) > max_cartesian_rows` → raise (extreme cartesian).
  Ранее `len(build) × len(probe) > max_output_rows` ошибочно сравнивал
  cartesian с expected → `plan_pipeline` мог сказать `"fits"`, а `merge` —
  упасть.

### Добавлено
- `JoinMemoryModel.c_cartesian` (default 10.0) — множитель для worst-case.
- `JoinParams.max_cartesian_rows`.
- `hash_join(..., max_cartesian_rows=None)`; default `10 × max_output_rows`.
- `CadwDataFrame.merge(..., num_partitions=None, use_planner=True)`:
  - `use_planner=True` (default): sample first batch каждой стороны,
    `mg.plan("join", ...)`, параметры пробрасываются в `hash_join`.
  - `use_planner=False`: старые дефолты (`num_partitions=8`,
    `max_output_rows = 0.3 × available / 256`).
  - `num_partitions` явно → override модели, но `max_output_rows` /
    `max_cartesian_rows` всё равно из модели.
- Per-partition `unique_build_keys` (pandas `.nunique`/`drop_duplicates`
  на build) — уточнённый `avg_matches_per_probe`, недоступный глобальной
  модели.

### Тесты
- 8 новых в `test_merge_planner.py`: planner default, `use_planner=False`,
  explicit `num_partitions`, early-fail план, cartesian-skew reject,
  left-planner, empty-side inner/left.

"""
if "## [0.2.4]" not in s:
    if "## [0.2.3]" in s:
        s = s.replace("## [0.2.3]", entry + "## [0.2.3]", 1)
    else:
        s = s.replace("## [0.2.2]", entry + "## [0.2.2]", 1)
    p.write_text(s)
    print("CHANGELOG: v0.2.4")
PY

# ---------- run ----------
echo "=== pytest tests/test_merge_planner.py -v ==="
set +e
uv run pytest tests/test_merge_planner.py -v
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "test_merge_planner.py: rc=$RC"
  git add -A
  git commit -m "WIP: 11-B-2c-3 (test red; fix follows)" || true
  exit 1
fi

echo "=== full fast + slow ==="
uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2c-3: merge + planner integration; max_output_rows vs max_cartesian_rows split; 8 tests"

echo "==> 11-B-2c-3 applied on ${BR}"
echo "    Next: squash → tag v0.2.4"