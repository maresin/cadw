#!/usr/bin/env bash
# patch_11a2_fix: GroupBy.estimate учитывает recurse_depth (leaf = K/P/4^d);
# test_sort_none_when_too_tight — меньший budget.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

BR="patch/11a2-models-extended"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-11a2-fix-${TS}"
git diff > ".backups/backup-pre-11a2-fix-${TS}/wip.diff"

# ---- Fix 1: GroupBy.estimate — leaf by p_recurse^depth ----
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/core/models_groupby.py")
s = p.read_text()

old = '''        f_shuffle = self.shuffle_model.estimate(params.shuffle, data_chars)

        k_per = data_chars.k_est / params.shuffle.num_partitions
        partition_bytes = k_per * data_chars.row_size_bytes
        dict_bytes = k_per * data_chars.size_of_key_entry_bytes
        f_local = partition_bytes + dict_bytes + self.c_temp_ratio * dict_bytes

        return int(max(f_shuffle, f_local))'''

new = '''        f_shuffle = self.shuffle_model.estimate(params.shuffle, data_chars)

        # Estimate — для листа дерева рекурсии. После `recurse_depth` split'ов
        # каждая подпартиция содержит K/P/p_recurse^d ключей. Родительская
        # партиция при split не материализуется целиком: hash_shuffle
        # обрабатывает батчами, RAM ≤ f_shuffle (на подпартиции ≤ f_shuffle
        # для полного датасета, т.к. подпартиция меньше). Поэтому лист
        # доминирует в пике локальной агрегации.
        p_recurse = params.p_recurse if params.recurse_depth > 0 else 1
        divisor = p_recurse ** params.recurse_depth
        k_per_leaf = data_chars.k_est / params.shuffle.num_partitions / divisor
        partition_leaf = k_per_leaf * data_chars.row_size_bytes
        dict_leaf = k_per_leaf * data_chars.size_of_key_entry_bytes
        f_local = partition_leaf + dict_leaf + self.c_temp_ratio * dict_leaf

        return int(max(f_shuffle, f_local))'''

if old not in s:
    raise SystemExit("GroupBy.estimate pattern not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("models_groupby.py: estimate учитывает recurse_depth (leaf = K/P/4^d)")
PY

# ---- Fix 2: test_sort_none_when_too_tight — меньший budget ----
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_models_extended.py")
s = p.read_text()

old = '''def test_sort_none_when_too_tight():
    m = SortMemoryModel()
    dc = _dc(row_size=1000)
    # 100 KB budget, 1 KB rows: read_avail = 21 KB; read_batch_target=1000 -> нужно 1 MB
    params = m.solve(100 * 1024, dc, safety=0.7)
    # read_avail // (1000 * 1000) = 0 -> k = max(2, 0) = 2; read_buffer = 10 KB; read_batch = 10
    # f_phase2 = 2*10KB + 2*32 + 10*1000 = 20KB + 64 + 10KB ≈ 30 KB < 70 KB -> passes
    # Для точного None нужен ещё меньше budget:
    params2 = m.solve(10 * 1024, dc, safety=0.7)
    # 10 KB * 0.7 = 7 KB; read_avail = 2.1 KB; read_avail < 2 * 1000 -> None
    assert params2 is None'''

new = '''def test_sort_none_when_too_tight():
    m = SortMemoryModel()
    dc = _dc(row_size=1000)
    # available=5 KB -> budget=3.5 KB; read_avail=1.05 KB < 2 * 1 KB -> None
    params = m.solve(5 * 1024, dc, safety=0.7)
    assert params is None'''

if old not in s:
    raise SystemExit("test_sort_none_when_too_tight pattern not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("test_models_extended.py: test_sort_none_when_too_tight — 5 KB budget")
PY

# ---- Run ----
set +e
uv run pytest tests/test_models_extended.py -v
RC_MODELS=$?
uv run pytest -q -m "not slow"
RC_FAST=$?
set -e
if [[ $RC_MODELS -ne 0 ]]; then echo "test_models_extended: rc=$RC_MODELS"; exit 1; fi
if [[ $RC_FAST -ne 0 ]]; then echo "fast: rc=$RC_FAST"; exit 1; fi

git add -A
git commit -m "11-A-2-fix: GroupBy.estimate — leaf (K/P/4^d); test_sort_none — 5 KB budget"

echo "==> sha256"
sha256sum cadw/core/models_groupby.py tests/test_models_extended.py
echo "==> 11-A-2-fix applied"