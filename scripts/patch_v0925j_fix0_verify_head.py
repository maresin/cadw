#!/usr/bin/env python3
# v0.9.25j-fix0 — verify_state version check is substring-fragile.
#
# DIAGNOSIS: v0.9.25j aborted at verify_state with
#   "v0.9.25j already committed — patch misplaced"
# But HEAD is genuinely v0.9.25i (no v0.9.25j commit exists).
#
# ROOT: `git log --oneline -1` shows the whole one-line commit
# message. v0.9.25i's body contains
#   "DEFERRED: ffill/bfill limit= → v0.9.25j; reindex..."
# so `"v0.9.25j" in log` → True, though HEAD is v0.9.25i.
# Substring presence ≠ subject identity.
#
# §13.4/270 (NEW): version-string presence check is unreliable. Prior
# commit bodies legitimately reference "→ vX.Y.Z" in DEFERRED and
# Next sections. Must match SUBJECT-PREFIX pattern:
#   `re.match(r'^[a-f0-9]+ (v\d+\.\d+\.\d+\w*):', log)`
# Exact-match, not substring.
#
# Parallel traps: §13.4/265 ("wait, X" comment vs stale assert),
# §13.4/269 (to_pylist in guarded scope → null_count). Same meta-
# lesson: "presence ≠ identity". For "is X committed?" use subject
# (authoritative); for "is X used?" use AST (semantic); never
# substring.
#
# FIX: replace 5-line version check in v0.9.25j's verify_state with
# 8-line regex subject extractor. Feature-absence check (already
# present, later in verify_state) remains authoritative.
#
# STATE: HEAD=v0.9.25i, v0.9.25j aborted (0 mutations).
#
# Usage:
#   uv run python scripts/patch_v0925j_fix0_verify_head.py
#   uv run python scripts/patch_v0925j_cut_qcut_duplicated.py
#
# TODO (v0.9.26): consolidate into
# patch_utils.verify_head_version(expected_version) helper — reused by
# all verify_state()s; extract subject-prefix via regex, raise on
# mismatch.

from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cadw.patch_utils import (
    apply_or_fail, assert_compiles, check_no_invalid_escape,
)


TARGET = "scripts/patch_v0925j_cut_qcut_duplicated.py"

# Old block: 5 lines of version-substring checks. Present in the
# aborted v0.9.25j verify_state.
OLD = '''    if "v0.9.25i" not in log:
        raise SystemExit(
            f"expected HEAD=v0.9.25i; got {log[:200]!r}"
        )
    if "v0.9.25j" in log:
        raise SystemExit("v0.9.25j already committed — patch misplaced")'''

# New block: extract subject-prefix version via regex, exact-match.
# Pattern `^[a-f0-9]+ (v\d+\.\d+\.\d+\w*):` — hash + space + version
# token + literal colon. Captures version in group(1). Only matches
# when version appears as the subject prefix (not as deferred
# reference in body).
NEW = '''    # v0.9.25j-fix0 (§13.4/270): version-string presence check false-
    # positives when prior commit body references "→ vX.Y.Z" for
    # deferred/next work (v0.9.25i's body: "DEFERRED: ... → v0.9.25j").
    # Extract SUBJECT-PREFIX version via regex; exact-match.
    import re as _re
    _m = _re.match(r'^[a-f0-9]+ (v\\d+\\.\\d+\\.\\d+\\w*):', log)
    _head_version = _m.group(1) if _m else None
    if _head_version != "v0.9.25i":
        raise SystemExit(
            f"expected HEAD version v0.9.25i; got {_head_version!r} "
            f"(log prefix: {log[:120]!r})"
        )'''


def verify_state():
    src = Path(TARGET).read_text()
    if NEW in src:
        raise SystemExit(
            "fix0 already applied — nothing to do (re-run "
            "patch_v0925j_cut_qcut_duplicated.py directly)"
        )
    if OLD not in src:
        raise SystemExit(
            f"OLD anchor not found in {TARGET}. Script may have "
            f"been manually edited or reverted — inspect."
        )
    print("  broken version check present in v0.9.25j script; "
          "ready to patch")

    # Sanity: regex behaves as expected
    import re as _re
    fake_log = (
        "f41dd2e v0.9.25i: Phase C batch C-3a — ffill + bfill + "
        "reindex ... DEFERRED: ffill/bfill limit= → v0.9.25j; ..."
    )
    m = _re.match(r'^[a-f0-9]+ (v\d+\.\d+\.\d+\w*):', fake_log)
    if m is None or m.group(1) != "v0.9.25i":
        raise SystemExit(
            f"regex sanity failed: matched {m.group(1) if m else None!r}, "
            f"expected 'v0.9.25i'. Fix pattern before applying."
        )
    print("  regex sanity: fake_log → group(1)='v0.9.25i' "
          "(deferred 'v0.9.25j' ignored) ✓")


def main():
    print("=== v0.9.25j-fix0: verify_state substring → subject-prefix ===")
    verify_state()

    print("\n=== apply fix0 to scripts/patch_v0925j_cut_qcut_duplicated.py ===")
    applied = apply_or_fail(TARGET, OLD, NEW, label="verify_head")
    if applied:
        assert_compiles(TARGET)
        check_no_invalid_escape(TARGET)
    print("  applied" if applied else "  already applied — skip")

    print("\n=== next step ===")
    print("  uv run python scripts/patch_v0925j_cut_qcut_duplicated.py")


if __name__ == "__main__":
    main()