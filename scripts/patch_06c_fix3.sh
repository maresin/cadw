#!/usr/bin/env bash
# patch_06c-fix3: убрать n_features из IncrementalPCA.partial_fit (его нет в API).
set -euo pipefail

BR="patch/06c-sklearn-extras"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$BR" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-06c-fix3-${TS}"
git diff > ".backups/backup-pre-06c-fix3-${TS}/wip.diff"

# ---- A. tests: убрать n_features из partial_fit для IncrementalPCA ----
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_sklearn_extras.py")
s = p.read_text()

# 1. Убрать параметр n_features=... из вызовов df.partial_fit для pca
s = s.replace(
    "        batch_size=100, n_features=4,\n",
    "        batch_size=100,\n",
)
s = s.replace(
    "        batch_size=200, n_features=2,\n",
    "        batch_size=200,\n",
)
s = s.replace(
    "        batch_size=5_000, n_features=2,\n",
    "        batch_size=5_000,\n",
)

# 2. Обновить docstring-комментарии
s = s.replace(
    '"""IncrementalPCA требует n_features в первом вызове (API sklearn, не CADW)."""',
    '"""IncrementalPCA.partial_fit: n_features не нужен (выводится из X.shape[1])."""',
)
s = s.replace(
    "    # n_features=4 — количество X-колонок, требуемое IncrementalPCA на первом вызове\n",
    "    # IncrementalPCA выводит n_features из X.shape[1] первого батча\n",
)

p.write_text(s)
print("patched tests: убран n_features из IncrementalPCA.partial_fit")
PY

# ---- B. DECISIONS: убрать неверное утверждение про n_features у IncrementalPCA ----
python3 - <<'PY'
from pathlib import Path
p = Path("DECISIONS.md")
s = p.read_text()

old_row = (
    "| `IncrementalPCA.partial_fit` требует `n_features` в первом вызове "
    "(а `PCA` не имеет `partial_fit` вообще) | "
    "`CadwDataFrame.partial_fit(ipca, ..., n_features=d)` | "
    "передайте `n_features=len(X_cols)`; для batch-only `PCA` — "
    "`to_batches(format='numpy')` → `np.vstack` → `PCA.fit` (не out-of-core) |"
)

new_row = (
    "| `PCA` (batch-only) не имеет `partial_fit`; "
    "`IncrementalPCA.partial_fit(X)` — без `n_features` "
    "(выводится из `X.shape[1]`) | "
    "`CadwDataFrame.partial_fit(ipca, X_cols=[...])` | "
    "для batch-only `PCA` — `to_batches(format='numpy')` → `np.vstack` → "
    "`PCA.fit` (не out-of-core). "
    "**Важно:** `n_features` — API `LatentDirichletAllocation`/`SGDOneClassSVM`, "
    "не PCA-семейства |"
)

if old_row in s:
    s = s.replace(old_row, new_row, 1)
    p.write_text(s)
    print("DECISIONS §11.5: исправлена строка про IncrementalPCA")
else:
    print("WARN: DECISIONS row not found; manual check")
PY

# ---- C. README: убрать n_features из примера IncrementalPCA ----
python3 - <<'PY'
from pathlib import Path
p = Path("README.md")
s = p.read_text()

old = (
    "ipca = IncrementalPCA(n_components=2)\n"
    "df.partial_fit(ipca, X_cols=[\"x1\", \"x2\"], y_col=None, batch_size=5000,\n"
    "               n_features=2)\n"
)
new = (
    "ipca = IncrementalPCA(n_components=2)\n"
    "df.partial_fit(ipca, X_cols=[\"x1\", \"x2\"], y_col=None, batch_size=5000)\n"
)

if old in s:
    s = s.replace(old, new, 1)
    p.write_text(s)
    print("README: убран n_features из IncrementalPCA-примера")
else:
    print("WARN: README example not found; manual check")
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
git commit -m "06c-fix3: убран n_features из IncrementalPCA.partial_fit (его нет в API; выводится из X.shape[1])"

echo "==> sha256"
sha256sum tests/test_sklearn_extras.py DECISIONS.md README.md
echo "==> 06c-fix3 applied"