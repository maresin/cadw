#!/usr/bin/env bash
# patch_11b2d_fix: заменить cartesian-check на точную оценку совпадений
# (build_counts × probe_counts merge). Удалить max_cartesian_rows.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

BR="patch/11b2d-e2e-join"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11b2d-fix-${TS}"
git diff > ".backups/backup-pre-11b2d-fix-${TS}/wip.diff" || true
git add -A
git commit -m "WIP: 11-B-2d (cartesian false-positive; exact match estimate follows)" || true
require_clean_tree || exit 1

# ---------- 1. join.py: заменить avg/cartesian на exact-match ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/join.py")
s = p.read_text()

# 1a. Сигнатура _process_partition_pair: убрать max_cartesian_rows
old_sig = '''    depth: int = 0,
    max_depth: int = 5,
    max_output_rows: int = 0,
    max_cartesian_rows: int = 0,
) -> list[pa.RecordBatch]:'''
new_sig = '''    depth: int = 0,
    max_depth: int = 5,
    max_output_rows: int = 0,
) -> list[pa.RecordBatch]:'''
if old_sig in s:
    s = s.replace(old_sig, new_sig, 1)
    print("join.py: _process_partition_pair signature updated")

# 1b. Рекурсивный вызов: убрать max_cartesian_rows
s = s.replace(
    '''                        depth=depth + 1, max_depth=max_depth,
                        max_output_rows=max_output_rows,
                        max_cartesian_rows=max_cartesian_rows,
                    )
                )
            return out''',
    '''                        depth=depth + 1, max_depth=max_depth,
                        max_output_rows=max_output_rows,
                    )
                )
            return out''',
    1,
)

# 1c. Удалить блок unique_build_keys/avg_matches_per_probe, добавить
#     точное вычисление build_counts (до цикла).
old_block = '''    # Per-partition skew check: unique_build_keys + avg_matches_per_probe.
    # Ловит partition-level skew, которого глобальная модель не видит:
    # после hash_shuffle все строки с одним ключом идут в одну партицию,
    # и там build может дублироваться сильнее, чем в среднем.
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

    out_batches: list[pa.RecordBatch] = []'''

new_block = '''    # Точная оценка совпадений: build_counts (один раз) × probe_counts
    # (per batch). Это то же, что материализует pd.merge (совпадения, не
    # cartesian). O(build + probe) через groupby на уникальных ключах —
    # дёшево (build ≈ partition size, probe ≈ batch size, тысячи строк).
    #
    # Почему не cartesian: `len(build) × len(probe)` — верхняя граница на
    # совпадения, но для PK-FK (build уникален) даёт 5000 × 2500 = 12.5M
    # при реальном output 2500 — ложные reject'ы. §13.4 эпизод 47.
    build_counts_df = None
    if len(build_df) > 0 and len(on) > 0:
        try:
            build_counts_df = (
                build_df.groupby(list(on), dropna=False)
                .size()
                .rename("__bc")
                .reset_index()
            )
        except Exception:
            build_counts_df = None

    def _exact_actual_matches(probe: "pd.DataFrame") -> int:
        """Точное число совпадений между build_df и этим probe batch.

        Если build_counts_df недоступен (группировка упала) — консервативная
        верхняя граница cartesian.
        """
        if len(probe) == 0 or len(build_df) == 0:
            return 0
        if build_counts_df is None:
            return len(build_df) * len(probe)
        try:
            probe_counts = (
                probe.groupby(list(on), dropna=False)
                .size()
                .rename("__pc")
                .reset_index()
            )
            joined_counts = build_counts_df.merge(
                probe_counts, on=list(on), how="inner"
            )
            if len(joined_counts) == 0:
                return 0
            return int((joined_counts["__bc"] * joined_counts["__pc"]).sum())
        except Exception:
            return len(build_df) * len(probe)

    out_batches: list[pa.RecordBatch] = []'''

if old_block not in s:
    raise SystemExit("unique_build_keys block not found")
s = s.replace(old_block, new_block, 1)
print("join.py: unique_build_keys -> build_counts_df + _exact_actual_matches")

# 1d. Заменить два check'а (expected + cartesian) на один exact-match.
old_checks = '''            # 1. Expected output (avg_matches × probe) против max_output_rows.
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
                )'''

new_checks = '''            # Точное число строк, которое pd.merge материализует:
            # sum over common keys(build_count × probe_count).
            actual_matches = _exact_actual_matches(probe_df)
            if actual_matches > max_output_rows:
                raise MemoryLimitExceeded(
                    f"hash_join(how={how!r}): exact matches between this "
                    f"probe batch ({len(probe_df)} rows) and build "
                    f"({len(build_df)} rows) = {actual_matches} > "
                    f"max_output_rows={max_output_rows}. Likely skew. "
                    f"Увеличьте budget или уменьшите num_partitions."
                )'''

if old_checks not in s:
    raise SystemExit("cartesian/expected checks block not found")
s = s.replace(old_checks, new_checks, 1)
print("join.py: cartesian+avg checks -> exact match check")

# 1e. hash_join signature: убрать max_cartesian_rows
old_hs = '''    hash_seed: int = 0,
    max_output_rows: Optional[int] = None,
    max_cartesian_rows: Optional[int] = None,
) -> pa.Table:
    """Grace hash join. how ∈ {'inner', 'left'} (v0.2.2).'''
new_hs = '''    hash_seed: int = 0,
    max_output_rows: Optional[int] = None,
) -> pa.Table:
    """Grace hash join. how ∈ {'inner', 'left'} (v0.2.2).'''
if old_hs in s:
    s = s.replace(old_hs, new_hs, 1)

# 1f. hash_join: убрать default max_cartesian_rows и передачу
s = s.replace(
    '''    if max_cartesian_rows is None:
        # Default: 10× expected (c_cartesian из JoinMemoryModel)
        max_cartesian_rows = max_output_rows * 10

    empty_schema''',
    '''    empty_schema''',
    1,
)

s = s.replace(
    '''                max_output_rows=max_output_rows,
                max_cartesian_rows=max_cartesian_rows,
            )
        )''',
    '''                max_output_rows=max_output_rows,
            )
        )''',
    1,
)

# 1g. inner_join/left_join: убрать max_cartesian_rows
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
    max_cartesian_rows: Optional[int] = None,
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
) -> pa.Table:'''
    s = s.replace(old_sig, new_sig, 1)

s = s.replace(
    '''        hash_seed=hash_seed, max_output_rows=max_output_rows,
        max_cartesian_rows=max_cartesian_rows,
    )''',
    '''        hash_seed=hash_seed, max_output_rows=max_output_rows,
    )''',
)

p.write_text(s)
print("join.py: hash_join / inner_join / left_join — max_cartesian_rows removed")
PY

# ---------- 2. models_join.py: убрать max_cartesian_rows, c_cartesian ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/core/models_join.py")
s = p.read_text()

# 2a. JoinParams: убрать поле
s = s.replace(
    '''    max_output_rows: int
    max_cartesian_rows: int   # worst case (для runtime защиты от skew)
    right_chars: DataChars''',
    '''    max_output_rows: int
    right_chars: DataChars''',
    1,
)

# 2b. __init__: убрать c_cartesian
s = s.replace(
    '''        c_merge_ratio: float = 2.0,     # pandas temp ≈ 2 × output
        c_cartesian: float = 10.0,      # max_cartesian = c_cartesian × max_output
    ) -> None:''',
    '''        c_merge_ratio: float = 2.0,     # pandas temp ≈ 2 × output
    ) -> None:''',
    1,
)
s = s.replace(
    '''        if c_cartesian < 1.0:
            raise ValueError("c_cartesian must be >= 1.0")
        self.build_ratio = float(build_ratio)''',
    '''        self.build_ratio = float(build_ratio)''',
    1,
)
s = s.replace(
    '''        self.c_merge_ratio = float(c_merge_ratio)
        self.c_cartesian = float(c_cartesian)''',
    '''        self.c_merge_ratio = float(c_merge_ratio)''',
    1,
)

# 2c. solve: убрать max_cartesian_rows
s = s.replace(
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
    1,
)

# 2d. describe: убрать c_cartesian
s = s.replace(
    '''            f"c_merge={self.c_merge_ratio:.2f}, "
            f"c_cartesian={self.c_cartesian:.2f}"
        )''',
    '''            f"c_merge={self.c_merge_ratio:.2f}"
        )''',
    1,
)

p.write_text(s)
print("models_join.py: max_cartesian_rows + c_cartesian removed")
PY

# ---------- 3. dataframe.py: не передавать max_cartesian_rows ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/dataframe.py")
s = p.read_text()

# 3a. Убрать переменную и её установку
s = s.replace(
    '''        max_output_rows: int | None = None
        max_cartesian_rows: int | None = None

        if use_planner:''',
    '''        max_output_rows: int | None = None

        if use_planner:''',
    1,
)
s = s.replace(
    '''                if num_partitions is None:
                    num_partitions = params.num_partitions
                max_output_rows = params.max_output_rows
                max_cartesian_rows = params.max_cartesian_rows''',
    '''                if num_partitions is None:
                    num_partitions = params.num_partitions
                max_output_rows = params.max_output_rows''',
    1,
)

# 3b. Убрать из вызова hash_join
s = s.replace(
    '''            num_partitions=num_partitions,
            max_output_rows=max_output_rows,
            max_cartesian_rows=max_cartesian_rows,
        )
        return self._wrap_result_table(result_table)''',
    '''            num_partitions=num_partitions,
            max_output_rows=max_output_rows,
        )
        return self._wrap_result_table(result_table)''',
    1,
)

p.write_text(s)
print("dataframe.py: max_cartesian_rows removed from merge")
PY

# ---------- 4. tests: обновить guard на exact-match ----------
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_join_oom_guard.py")
s = p.read_text()

# Заменить test_process_partition_pair_defines_avg_matches_before_use
old_test = '''def test_process_partition_pair_defines_avg_matches_before_use():
    """avg_matches_per_probe определён ДО использования (AST, lineno).

    Регрессия 11-B-2c-3: шаг 2d патча молча не сработал (str.replace с
    несуществующим шаблоном — no-op), и avg_matches_per_probe использовался
    без определения. NameError в runtime. Guard фиксирует порядок.
    """
    from cadw.ops.join import _process_partition_pair
    import ast

    src = inspect.getsource(_process_partition_pair)
    tree = ast.parse(inspect.cleandoc(src))

    # Найти все Store и Load для имени "avg_matches_per_probe"
    stores: list[int] = []
    loads: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "avg_matches_per_probe":
            if isinstance(node.ctx, ast.Store):
                stores.append(node.lineno)
            elif isinstance(node.ctx, ast.Load):
                loads.append(node.lineno)

    assert stores, (
        "avg_matches_per_probe never assigned in _process_partition_pair "
        "(NameError at runtime)"
    )
    assert loads, (
        "avg_matches_per_probe never read in _process_partition_pair"
    )
    first_store = min(stores)
    first_load = min(loads)
    assert first_store < first_load, (
        f"avg_matches_per_probe: first Store at line {first_store}, "
        f"first Load at line {first_load} — use before assignment"
    )'''

new_test = '''def test_actual_matches_checked_before_pd_merge():
    """Exact-match guard вызывается ДО pd.merge в _process_partition_pair.

    Регрессия 11-B-2d: cartesian check (`len(build) × len(probe) >
    max_cartesian_rows`) давал ложные reject'ы для PK-FK (build уникален).
    Fix: точная оценка `actual_matches = _exact_actual_matches(probe)`,
    сравнение с `max_output_rows` — то же, что `pd.merge` материализует.
    AST guard фиксирует порядок: check ДО merge.
    """
    from cadw.ops.join import _process_partition_pair
    import ast

    src = inspect.getsource(_process_partition_pair)
    tree = ast.parse(inspect.cleandoc(src))

    # `_exact_actual_matches` — вызов функции; `actual_matches` — её результат
    exact_call_lines: list[int] = []
    actual_load_lines: list[int] = []
    pd_merge_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "_exact_actual_matches":
                exact_call_lines.append(node.lineno)
            if isinstance(f, ast.Attribute) and f.attr == "merge":
                # pd.merge или build_counts_df.merge — оба после check
                if isinstance(f.value, ast.Name) and f.value.id == "pd":
                    pd_merge_lines.append(node.lineno)
        if (
            isinstance(node, ast.Name)
            and node.id == "actual_matches"
            and isinstance(node.ctx, ast.Load)
        ):
            actual_load_lines.append(node.lineno)

    assert exact_call_lines, (
        "_exact_actual_matches not called in _process_partition_pair"
    )
    assert pd_merge_lines, "pd.merge not called in _process_partition_pair"
    first_check = min(exact_call_lines)
    first_merge = min(pd_merge_lines)
    assert first_check < first_merge, (
        f"_exact_actual_matches at line {first_check} must precede "
        f"pd.merge at line {first_merge}"
    )


def test_no_cartesian_check_in_join():
    """cartesian check (`len(build) × len(probe) > max_cartesian_rows`) удалён.

    Регрессия 11-B-2d: cartesian check ложно режет PK-FK. Правильный
    guard — exact matches. AST guard фиксирует отсутствие старой
    переменной/параметра.
    """
    from cadw.ops import join as join_mod
    src = inspect.getsource(join_mod)
    assert "max_cartesian_rows" not in src, (
        "max_cartesian_rows must be removed (cartesian is not an OOM bound); "
        "use exact-match estimate"
    )'''

if old_test not in s:
    raise SystemExit("old avg_matches AST test not found")
s = s.replace(old_test, new_test, 1)
p.write_text(s)
print("test_join_oom_guard.py: 2 новых AST guard'а")
PY

# ---------- 5. DECISIONS: эпизод 47 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

marker_135 = "### 13.5. Открытые вопросы"
episode_47 = """**Эпизод 47 (cartesian — не OOM-граница).** После 11-B-2c-3 runtime
проверял `len(build) × len(probe) > max_cartesian_rows` с
`max_cartesian_rows = 10 × max_output_rows`. Для PK-FK join 100k × 10k
(8 партиций): build ~5000 уникальных, probe ~2500 → cartesian = 12.6M
при **реальном output 2500 строк** (каждый probe-ключ матчит ровно один
build-ключ). `max_cartesian_rows = 407k` → ложный `MemoryLimitExceeded`.
**Причина:** `pd.merge` материализует **только совпадения**, не
cartesian. `len(build) × len(probe)` — верхняя граница совпадений, но
для PK-FK (build уникален) она в тысячи раз больше реального output.
**Fix:** точная оценка `actual_matches = sum over common keys(build_count
× probe_count)` через `build_df.groupby(on).size()` (один раз до цикла)
× `probe_df.groupby(on).size()` (per batch) × `merge`. O(build + probe) на
уникальных ключах (не на всех строках) — дёшево. Это **точный** размер
`pd.merge` output, а не эвристика. `max_cartesian_rows` удалён из
`JoinParams`, `JoinMemoryModel`, `hash_join`, `merge`. **Урок:** для
защиты от OOM нужно оценивать **что материализуется** (совпадения), а
не **что могло бы материализоваться в худшем случае** (cartesian).
Cartesian — легитимная верхняя граница в общем случае (no index), но
для hash join на известных ключах точная оценка дешевле и корректнее.
Верхняя граница должна быть **звуковой** (никогда не ниже реального
output), а cartesian over-reject'ит; exact match никогда не over-reject'ит
и никогда не under-reject'ит (это и есть то, что `pd.merge` сделает).
Фильтр по ключам — это **единственная** точка, где мы знаем структуру
до `pd.merge`, используем её.

"""
if "Эпизод 47" not in s:
    s = s.replace(marker_135, episode_47 + marker_135, 1)

# Обновить §11.1 (количество тестов E2E — 3 slow; join OOM — 7+2 guard)
p.write_text(s)
print("DECISIONS: §13.4/47")
PY

# ---------- 6. CHANGELOG: fix в v0.2.5 ----------
python3 - <<'PY'
from pathlib import Path
p = Path("CHANGELOG.md")
s = p.read_text()
old_marker = "## [0.2.5] — 2026-10-19"
if old_marker in s and "### Исправлено (§13.4 эпизод 47)" not in s:
    entry_fix = """### Исправлено (§13.4 эпизод 47)
- `hash_join`: заменён cartesian check (`len(build) × len(probe) >
  max_cartesian_rows`) на **точную** оценку совпадений:
  `actual_matches = sum over common keys(build_count × probe_count)`
  через `groupby.size()` на уникальных ключах. O(build + probe),
  дёшево. Устраняет ложные reject'ы для PK-FK (build уникален, cartesian
  = 12.6M при реальном output 2500).
- Удалены: `max_cartesian_rows` из `JoinParams`, `JoinMemoryModel`,
  `hash_join`, `merge`; `c_cartesian` из `JoinMemoryModel`.
- 2 AST guard'а: `test_actual_matches_checked_before_pd_merge`,
  `test_no_cartesian_check_in_join`.

"""
    s = s.replace(old_marker, old_marker + "\n\n" + entry_fix.rstrip() + "\n", 1)
    p.write_text(s)
    print("CHANGELOG: v0.2.5 fix note")
PY

# ---------- run ----------
echo "=== pytest tests/test_e2e_join.py -v ==="
set +e
uv run pytest tests/test_e2e_join.py -v
RC_E2E=$?
set -e
if [[ $RC_E2E -ne 0 ]]; then
  echo "test_e2e_join.py: rc=$RC_E2E"
  exit 1
fi

echo "=== full fast + slow ==="
uv run pytest -q -m "not slow"
uv run pytest -q -m slow

git add -A
git commit -m "11-B-2d-fix: exact-match estimate вместо cartesian; удалён max_cartesian_rows; §13.4/47"

echo "==> 11-B-2d-fix applied"