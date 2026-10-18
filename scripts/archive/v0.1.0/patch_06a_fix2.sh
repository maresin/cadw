#!/usr/bin/env bash
# patch_06a-fix2: sklearn API — classes= на первом partial_fit (test fix only).
set -euo pipefail

BR="patch/06a-ml-numpy"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$BR" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-06a-fix2-${TS}"
git diff > ".backups/backup-pre-06a-fix2-${TS}/wip.diff"

python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_ml_integration.py")
s = p.read_text()

# 1. test_partial_fit_sgd: classes=[0, 1]
old = '''    model = SGDClassifier(loss="log_loss", random_state=0)
    df.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=50)'''
new = '''    model = SGDClassifier(loss="log_loss", random_state=0)
    df.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=50,
                   classes=[0, 1])'''
if old not in s:
    raise SystemExit("test_partial_fit_sgd: pattern not found")
s = s.replace(old, new, 1)

# 2. test_partial_fit_after_filter: classes=[0, 1]
old = '''    model = SGDClassifier(loss="log_loss", random_state=0)
    filtered.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=50)'''
new = '''    model = SGDClassifier(loss="log_loss", random_state=0)
    filtered.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=50,
                         classes=[0, 1])'''
if old not in s:
    raise SystemExit("test_partial_fit_after_filter: pattern not found")
s = s.replace(old, new, 1)

# 3. E2E: classes=[0, 1] на первом вызове (sklearn принимает и на последующих)
old = '''    model = SGDClassifier(loss="log_loss", random_state=0)
    for _ in range(3):
        df.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=5_000)'''
new = '''    model = SGDClassifier(loss="log_loss", random_state=0)
    for epoch in range(3):
        df.partial_fit(model, X_cols=["x1", "x2"], y_col="y", batch_size=5_000,
                       classes=[0, 1])'''
if old not in s:
    raise SystemExit("E2E: pattern not found")
s = s.replace(old, new, 1)

p.write_text(s)
print("patched 3 tests: classes=[0, 1] for SGDClassifier.partial_fit")
PY

set +e
uv run pytest -q -m "not slow"
RC_FAST=$?
uv run pytest -q -m slow
RC_SLOW=$?
set -e
if [[ $RC_FAST -ne 0 ]]; then echo "fast: rc=$RC_FAST"; exit 1; fi
if [[ $RC_SLOW -ne 0 ]]; then echo "slow: rc=$RC_SLOW"; exit 1; fi

git add -A
git commit -m "06a-fix2: sklearn classes= API contract in tests (no code change)"

echo "==> sha256"
sha256sum tests/test_ml_integration.py
echo "==> 06a-fix2 applied"