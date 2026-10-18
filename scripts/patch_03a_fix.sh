#!/usr/bin/env bash
# patch_03a-fix: Schema.serialize() вместо pa.ipc.serialize_schema.
set -euo pipefail

BR="patch/03a-dataset"
[[ "$(git rev-parse --abbrev-ref HEAD)" == "$BR" ]] || { echo "not on ${BR}"; exit 1; }

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p ".backups/backup-pre-03a-fix-${TS}"
git diff > ".backups/backup-pre-03a-fix-${TS}/wip.diff"

python3 - <<'PY'
p = "cadw/dataset/dataset.py"
s = open(p).read()

old = '''    def _save_meta(self) -> None:
        assert self._schema is not None
        schema_bytes = (
            pa.ipc.serialize_schema(self._schema).serialize().to_pybytes()
        )'''
new = '''    def _save_meta(self) -> None:
        assert self._schema is not None
        # PyArrow 26: сериализация схемы — Schema.serialize() (не
        # pa.ipc.serialize_schema, которого в этом API нет).
        schema_bytes = self._schema.serialize().to_pybytes()'''

if old not in s:
    raise SystemExit("_save_meta: pattern not found")
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("fixed _save_meta: Schema.serialize()")
PY

set +e
uv run pytest -q
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
    echo "ERROR: tests still failing (rc=${RC})"
    exit 1
fi

git add -A
git commit -m "03a-fix: Schema.serialize() (pyarrow 26 API)"

echo "==> sha256"
sha256sum cadw/dataset/dataset.py
echo "==> 03a-fix applied"