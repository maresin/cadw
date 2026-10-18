#!/usr/bin/env bash
# patch_06c-fix2: IncrementalPCA не принимает random_state (детерминирован).
set -euo pipefail

BR="patch/06c-sklearn-extras"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$BR" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-06c-fix2-${TS}"
git diff > ".backups/backup-pre-06c-fix2-${TS}/wip.diff"

python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_sklearn_extras.py")
s = p.read_text()

before = s.count("IncrementalPCA(n_components=")
s = s.replace(
    "IncrementalPCA(n_components=2, random_state=0)",
    "IncrementalPCA(n_components=2)",
)
after = s.count("IncrementalPCA(n_components=2, random_state=0)")

p.write_text(s)
print(f"replaced IncrementalPCA(... random_state=0): {before - after} occurrence(s); "
      f"remaining with random_state: {after}")
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
git commit -m "06c-fix2: IncrementalPCA без random_state (детерминирован, API sklearn)"

echo "==> sha256"
sha256sum tests/test_sklearn_extras.py
echo "==> 06c-fix2 applied"