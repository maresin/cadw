#!/usr/bin/env bash
# patch_07: API availability guard + null_placement per-key (FutureWarning).
set -euo pipefail
source "$(dirname "$0")/_lib.sh"
require_clean_tree || exit 1

CUR="$(git branch --show-current)"
TS="$(date +%Y%m%d_%H%M%S)"
BK="backup-pre-07-${TS}"
mkdir -p ".backups/${BK}"
git archive --format=tar.gz -o ".backups/${BK}/repo.tgz" HEAD
git tag -a "${BK}" -m "before patch_07"

BR="patch/07-hygiene"
if git rev-parse --verify "${BR}" >/dev/null 2>&1; then
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "${BR}" ]] || exit 1
else
  git checkout -b "${BR}"
fi

# ---------- A. API availability guard ----------
cat > tests/test_api_availability.py <<'EOF'
"""
Pre-flight guard: API PyArrow / sklearn / LightGBM / xxhash, которые CADW
использует. Если внешняя библиотека изменит сигнатуру — этот файл упадёт
первым с понятным сообщением, а не 30 тестов с AttributeError.

Основание: DECISIONS.md §13.4 — 24 эпизода отладки, ~40% из них —
API-контракты. Классы: параметры (`type_` → `type`), методы
(`arr.length` → `len(arr)`, `num_data` → `num_data()`), отсутствующие
функции (`pc.hash`, `serialize_schema`), deprecated kwargs
(`IncrementalPCA(random_state=)`), required kwargs (`classes=`, `n_features=`).
"""
import inspect
import pytest


# ---------- PyArrow ----------

def test_pyarrow_version_and_imports():
    import pyarrow as pa
    import pyarrow.compute as pc
    assert pa.__version__ >= "26.0.0", f"pyarrow>=26 required, got {pa.__version__}"
    for name in ("add", "subtract", "multiply", "divide",
                 "shift_left", "shift_right", "bit_wise_and", "bit_wise_xor",
                 "and_", "or_", "invert",
                 "greater", "greater_equal", "less", "less_equal", "equal", "not_equal",
                 "sum", "mean", "min_max",
                 "sort_indices"):
        assert hasattr(pc, name), f"pc.{name} missing in pyarrow {pa.__version__}"


def test_pa_array_type_kwarg():
    """pa.array(values, type=...) — параметр `type`, не `type_`."""
    import pyarrow as pa
    sig = inspect.signature(pa.array)
    assert "type" in sig.parameters, "pa.array must accept type= kwarg"


def test_pa_array_has_len_not_length():
    """len(arr), не arr.length."""
    import pyarrow as pa
    arr = pa.array([1, 2, 3])
    assert len(arr) == 3
    assert not hasattr(arr, "length"), "pa.Array should not have .length; use len()"


def test_schema_serialize():
    """pa.Schema.serialize() — метод, не pa.ipc.serialize_schema."""
    import pyarrow as pa
    schema = pa.schema([("x", pa.int64())])
    assert hasattr(schema, "serialize")
    assert callable(schema.serialize)


def test_ipc_read_schema():
    import pyarrow.ipc as ipc
    assert hasattr(ipc, "read_schema")


def test_table_group_by():
    """pa.Table.group_by — есть; RecordBatch.group_by — нет."""
    import pyarrow as pa
    t = pa.table({"x": [1, 2], "y": [3.0, 4.0]})
    assert hasattr(t, "group_by")
    b = t.to_batches()[0]
    assert not hasattr(b, "group_by"), "RecordBatch.group_by should not exist; use Table"


def test_sort_indices_null_placement_per_key():
    """null_placement должен приниматься в sort_keys (per-key), а не в SortOptions."""
    import pyarrow as pa
    import pyarrow.compute as pc
    t = pa.table({"x": [3, None, 1]})
    # per-key null_placement: (name, order, null_placement)
    order = pc.sort_indices(t, sort_keys=[("x", "ascending", "at_end")])
    assert order.to_pylist() == [2, 0, 1]


# ---------- xxhash ----------

def test_xxhash_accepts_bytes():
    """xxh64 принимает bytes, не str."""
    import xxhash
    h = xxhash.xxh64(b"abc", seed=0).intdigest()
    assert isinstance(h, int)
    with pytest.raises(TypeError):
        xxhash.xxh64("abc", seed=0)


# ---------- sklearn ----------

def test_sklearn_version():
    import sklearn
    assert sklearn.__version__ >= "1.4", f"sklearn>=1.4 required, got {sklearn.__version__}"


def test_sgd_partial_fit_classes_kwarg():
    """SGDClassifier.partial_fit требует classes= на первом вызове."""
    from sklearn.linear_model import SGDClassifier
    import numpy as np
    model = SGDClassifier(loss="log_loss", random_state=0)
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    # classes — обязательный kwarg в API; проверим, что передача работает
    model.partial_fit(X, y, classes=[0, 1])
    assert hasattr(model, "classes_")


def test_standard_scaler_partial_fit():
    from sklearn.preprocessing import StandardScaler
    assert hasattr(StandardScaler(), "partial_fit")


def test_minibatch_kmeans_partial_fit():
    from sklearn.cluster import MiniBatchKMeans
    assert hasattr(MiniBatchKMeans(), "partial_fit")


def test_incremental_pca_partial_fit():
    from sklearn.decomposition import IncrementalPCA
    model = IncrementalPCA(n_components=1)
    assert hasattr(model, "partial_fit")


def test_incremental_pca_signature_no_random_state():
    """IncrementalPCA.__init__ не принимает random_state; partial_fit — без n_features."""
    from sklearn.decomposition import IncrementalPCA
    init_sig = inspect.signature(IncrementalPCA.__init__)
    assert "random_state" not in init_sig.parameters, (
        "IncrementalPCA now accepts random_state? Update test if API changed"
    )
    fit_sig = inspect.signature(IncrementalPCA.partial_fit)
    assert "n_features" not in fit_sig.parameters, (
        "IncrementalPCA.partial_fit now requires n_features? Update DataFrame.partial_fit"
    )


def test_pca_has_no_partial_fit():
    """PCA (batch-only) не должен иметь partial_fit; если появился — можно упростить."""
    from sklearn.decomposition import PCA
    assert not hasattr(PCA, "partial_fit"), (
        "PCA now has partial_fit? Consider simplifying CadwDataFrame.partial_fit"
    )


# ---------- LightGBM ----------

def test_lightgbm_version_and_dataset():
    import lightgbm as lgb
    assert lgb.__version__ >= "4.0", f"lightgbm>=4.0 required, got {lgb.__version__}"
    assert hasattr(lgb, "Dataset")
    assert hasattr(lgb, "train")


def test_lgb_dataset_num_data_is_method():
    """ds.num_data() — метод, не атрибут."""
    import lightgbm as lgb
    assert hasattr(lgb.Dataset, "num_data")
    assert callable(lgb.Dataset.num_data), "num_data must be callable, not attribute"
    assert hasattr(lgb.Dataset, "num_feature")
    assert callable(lgb.Dataset.num_feature)
EOF

# ---------- B. Fix null_placement per-key in sort.py ----------
python3 - <<'PY'
from pathlib import Path
p = Path("cadw/ops/sort.py")
s = p.read_text()

old = '''    sort_keys = [
        (col, "ascending" if asc else "descending")
        for col, asc in zip(key_columns, ascending)
    ]
    null_placement = "at_start" if nulls_first else "at_end"
    order = pc.sort_indices(table, sort_keys=sort_keys, null_placement=null_placement)'''

new = '''    # null_placement передаётся per-key (per-key API с 25.0.0; глобальный
    # SortOptions.null_placement deprecated).
    null_placement = "at_start" if nulls_first else "at_end"
    sort_keys = [
        (col, "ascending" if asc else "descending", null_placement)
        for col, asc in zip(key_columns, ascending)
    ]
    order = pc.sort_indices(table, sort_keys=sort_keys)'''

if old not in s:
    raise SystemExit("sort.py: pattern for _write_run not found")
s = s.replace(old, new, 1)
p.write_text(s)
print("sort.py: null_placement per-key (FutureWarning removed)")
PY

# ---------- C. Cosmetic: .gitignore section header ----------
python3 - <<'PY'
from pathlib import Path
p = Path(".gitignore")
s = p.read_text()
s = s.replace(
    "# Analysis artifacts & experimental data (fix_other_leak)\nmain.py",
    "# uv init artifact (should not be tracked)\nmain.py",
    1,
)
p.write_text(s)
print(".gitignore: cosmetic header for main.py")
PY

# ---------- D. DECISIONS.md update ----------
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

# 11.6: null_placement убран
s = s.replace(
    "| `FutureWarning: null_placement deprecated` в `Sort` | шум в тестах; заменить на per-sort-key |\n",
    "| ~~`FutureWarning: null_placement deprecated`~~ | ~~шум в тестах~~ → **закрыто** (патч 07: per-key `sort_keys`) |\n",
    1,
)

# 11.6: check_api убран из техдолга
s = s.replace(
    "| `scripts/check_api.py` отсутствует | 24 эпизода API-контрактов в §13.4 |\n",
    "| ~~`scripts/check_api.py` отсутствует~~ | ~~24 эпизода API-контрактов~~ → **закрыто** (патч 07: `tests/test_api_availability.py`, 18 assertions) |\n",
    1,
)

# 11.1: добавить строку
s = s.replace(
    "| Architecture guards (legacy, AST) | ✅ | 3 |",
    "| Architecture guards (legacy, AST) | ✅ | 3 |\n"
    "| API availability guard (07, 18 assertions) | ✅ | 18 |",
    1,
)

# 11.6: убрать "24 эпизода" из итога, если есть
s = s.replace(
    "24 эпизода отладки, 4 класса",
    "25 эпизодов отладки, 4 класса",
    1,
)

p.write_text(s)
print("DECISIONS.md updated: §11.1 + §11.6")
PY

# ---------- Run ----------
echo "=== pytest (fast) ==="
set +e
uv run pytest -q -m "not slow"
RC_FAST=$?
uv run pytest -q -m slow
RC_SLOW=$?
set -e
if [[ $RC_FAST -ne 0 ]]; then echo "fast: rc=$RC_FAST"; exit 1; fi
if [[ $RC_SLOW -ne 0 ]]; then echo "slow: rc=$RC_SLOW"; exit 1; fi

# Проверить, что warnings ушли
echo "=== check warnings (должно быть 0 FutureWarning) ==="
uv run pytest -q -m "not slow" 2>&1 | grep -c "FutureWarning" || echo "  0 FutureWarning"

git add -A
git commit -m "07: API availability guard (18 assertions) + null_placement per-key (FutureWarning removed)"

echo "==> sha256"
sha256sum tests/test_api_availability.py cadw/ops/sort.py
echo "==> 07 applied on ${BR}"
echo "    Revert: git checkout ${CUR} && git branch -D ${BR}"