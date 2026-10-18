#!/usr/bin/env bash
# patch_06c-fix: PCA -> IncrementalPCA (API sklearn; PCA partial_fit не имеет).
set -euo pipefail

BR="patch/06c-sklearn-extras"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$BR" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-06c-fix-${TS}"
git diff > ".backups/backup-pre-06c-fix-${TS}/wip.diff"

# ---- A. tests: PCA -> IncrementalPCA ----
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_sklearn_extras.py")
s = p.read_text()

# 1. import
if "IncrementalPCA" not in s:
    s = s.replace(
        "from sklearn.decomposition import PCA",
        "from sklearn.decomposition import IncrementalPCA",
        1,
    )

# 2. Все использования PCA(...) -> IncrementalPCA(...)
s = s.replace("pca = PCA(", "pca = IncrementalPCA(")

# 3. Комментарии про PCA.partial_fit -> IncrementalPCA
s = s.replace(
    '"""PCA требует n_features в первом вызове (API sklearn, не CADW)."""',
    '"""IncrementalPCA требует n_features в первом вызове (API sklearn, не CADW)."""',
)
s = s.replace(
    "# n_features=4 — количество X-колонок, требуемое sklearn на первом вызове",
    "# n_features=4 — количество X-колонок, требуемое IncrementalPCA на первом вызове",
)

p.write_text(s)
print("patched tests: PCA -> IncrementalPCA")
PY

# ---- B. DECISIONS: PCA -> IncrementalPCA в §11.5 ----
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()
if "`PCA.partial_fit` требует `n_features`" in s:
    s = s.replace(
        "| `PCA.partial_fit` требует `n_features` в первом вызове | "
        "`CadwDataFrame.partial_fit(pca, ..., n_features=d)` | "
        "передайте `n_features=len(X_cols)` в kwargs (API sklearn) |",
        "| `IncrementalPCA.partial_fit` требует `n_features` в первом вызове "
        "(а `PCA` не имеет `partial_fit` вообще) | "
        "`CadwDataFrame.partial_fit(ipca, ..., n_features=d)` | "
        "передайте `n_features=len(X_cols)`; для batch-only `PCA` — "
        "`to_batches(format='numpy')` → `np.vstack` → `PCA.fit` (не out-of-core) |",
        1,
    )
    p.write_text(s)
    print("DECISIONS §11.5: PCA -> IncrementalPCA")
PY

# ---- C. README: ML-секция, PCA -> IncrementalPCA ----
python3 - <<'PY'
from pathlib import Path
p = Path("README.md")
s = p.read_text()

ml_old = """### sklearn: partial_fit (потоковое обучение)

```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

guard = MemoryGuard(max_memory_mb=100)
df = CadwDataFrame.from_parquet("data.parquet", "cache/", memory_guard=guard)

scaler = StandardScaler()
df.partial_fit(scaler, X_cols=["x1", "x2"], y_col=None, batch_size=5000)

model = SGDClassifier()
for _ in range(3):
    df.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=5000,
                   classes=[0, 1])
```

Батчи конвертируются в numpy на лету, память ограничена `MemoryGuard`.
"""

ml_new = """### sklearn: partial_fit (потоковое обучение)

```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA

guard = MemoryGuard(max_memory_mb=100)
df = CadwDataFrame.from_parquet("data.parquet", "cache/", memory_guard=guard)

# Модели без y
scaler = StandardScaler()
df.partial_fit(scaler, X_cols=["x1", "x2"], y_col=None, batch_size=5000)

kmeans = MiniBatchKMeans(n_clusters=5)
df.partial_fit(kmeans, X_cols=["x1", "x2"], y_col=None, batch_size=5000)

ipca = IncrementalPCA(n_components=2)
df.partial_fit(ipca, X_cols=["x1", "x2"], y_col=None, batch_size=5000,
               n_features=2)

# Модель с y (классификация)
model = SGDClassifier()
for _ in range(3):
    df.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=5000,
                   classes=[0, 1])
```

Батчи конвертируются в numpy на лету, память ограничена `MemoryGuard`.

**Ограничения API sklearn** (не CADW):
- `SGDClassifier.partial_fit` требует `classes=[0, 1]` на первом вызове.
- `IncrementalPCA.partial_fit` требует `n_features=d` на первом вызове.
- `PCA` (не `Incremental`) не имеет `partial_fit`: для batch-only
  используйте `to_batches(format='numpy')` → `np.vstack` → `PCA.fit`
  (это материализует весь датасет, не out-of-core).
"""

if ml_old in s:
    s = s.replace(ml_old, ml_new, 1)
    p.write_text(s)
    print("README: ML section updated (IncrementalPCA + API limits)")
else:
    print("WARN: README ML section pattern not found; manual fix required")
PY

# ---- Run ----
set +e
uv run pytest -q -m "not slow"
RC_FAST=$?
uv run pytest -q -m slow
RC_SLOW=$?
set -e
if [[ $RC_FAST -ne 0 ]]; then echo "fast: rc=$RC_FAST"; exit 1; fi
if [[ $RC_SLOW -ne 0 ]]; then echo "slow: rc=$RC_SLOW"; exit 1; fi

git add -A
git commit -m "06c-fix: PCA -> IncrementalPCA (PCA не имеет partial_fit; API sklearn)"

echo "==> sha256"
sha256sum tests/test_sklearn_extras.py README.md DECISIONS.md
echo "==> 06c-fix applied"