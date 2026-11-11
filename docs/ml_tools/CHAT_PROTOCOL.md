# Chat Protocol — Developer <-> AI Session

**Version:** 1.0 (2026-10-31, v0.9.102)
**Purpose:** session rhythm. Reduces context waste, prevents terminal
crashes, keeps evidence clean.
**Companions:** `AI_CONSTITUTION.md` (Tier 1), `STABILITY_PROTOCOL.md`
(Tier 2). Does not override any rule.

**Hierarchy:** Tier 2 (methodological). Complements
`AI_CONSTITUTION.md` (Tier 1). Does NOT override §1 principles,
§3 patterns, §5 anti-patterns. On conflict — `AI_CONSTITUTION.md`
wins. If you believe a rule here contradicts the constitution,
stop and ask: which should change?

---

## §1. Session structure — 4 phases, never mixed

```
[1. Probe]    read-only commands -> output pasted back by human
[2. Patch]    write script to /tmp, run it; verify in SAME block
[3. Review]   human reads summary + diff; approves or corrects
[4. Commit]   git add + commit + tag (only after explicit "go")
```

Do not mix probe + patch in one block.

---

## §2. Patch scripts — always file in /tmp, never inline

BAD: inline heredoc (`python - <<EOF`) for 200+ line patch — cannot
re-run on error, cannot inspect after failure.

GOOD: write to file first, then run.

```bash
cat > /tmp/patch_102a_stage1.py <<'INNER_DELIM'
import sys
from pathlib import Path
# 200 lines
INNER_DELIM

uv run python /tmp/patch_102a_stage1.py
```

File remains on disk. If it fails — inspect, patch, re-run.

**Delimiter discipline:** the OUTER delimiter of `cat > ... <<'OUTER'`
MUST be unique (e.g. `CADW_STAGE_N_UNIQUE_EOF`). If the content being
written contains its own heredoc examples (`<<'PYEOF'`, `<<'MSG'`),
any `PYEOF` at start-of-line will prematurely close an OUTER heredoc
named `PYEOF`. This is exactly how v0.9.102 stage-a failed first try.

---

## §3. Verification — separate block, tail only

Patch block runs alone. Verify block runs separately:

```bash
uv run python -c "import cadw; print(cadw.__version__)"
uv run pytest -q tests/test_estimate_memory.py > /tmp/v.log 2>&1
echo "exit: $?"
tail -6 /tmp/v.log
```

Never mix `cat > patch.py; python patch.py; pytest` in one block.

---

## §4. Heavy tests — redirect, do not pipe (AP13)

BAD: `pytest ... | tail` — crashes terminal on fail.
GOOD: `pytest ... > /tmp/log 2>&1; echo $?; tail`.

**Rule:** full fast (~10 min) or full (~30 min) — always redirect.
Same for `measure_peak` matrix runs.

**When to run full fast:** only before `git commit`. Otherwise — focused
tests.

**When to run slow:** only if user asks.

---

## §5. Report format — what to paste back

Include: `tail -N /tmp/log`, `grep -c '^FAILED'`, `git diff --stat`,
`git status --short`, error + last traceback frame, exit code.

Exclude: full log, every passing test, full diff (unless reviewing),
full `git log`, full traceback (unless deep).

**Format:**

```
=== <phase> output ===
exit: <code>
<tail -N output>
```

One block per phase. Clean, small.

**Do NOT paste back:**
- The patch script itself (already visible in AI's previous message)
- Full YAML diff if `--stat` shows correct counts

---

## §6. Common patterns

### §6.1. Probe phase — read-only

`grep -n 'def fit_X' -A 40`, `yaml.safe_load`, `pytest -q > /tmp/log`.

Never edit files. Only read.

### §6.2. Patch phase — write + verify

Write to `/tmp/patch_NNN.py`, run, verify in SAME block. If patch fails —
stop, do not chain.

### §6.3. Review phase — diff only, no code

`git diff --stat`, `git diff <file> | head -40`. AI sees the rest from
its own patch script.

### §6.4. Commit phase — after explicit "go"

`git add`, `git diff --cached --stat`, `git commit -F - <<'COMMITMSG'`,
`git tag`. Only after human says "go" / "принимаю" / "commit".
**Never** auto-commit.

---



**TAG IN SAME BLOCK AS COMMIT** (v0.9.213 addendum):

```bash
git commit -m "vX.Y.Z: release" && git tag vX.Y.Z
```

Do NOT separate `git commit` and `git tag` across chat messages.
v0.9.212 tag was lost because commit was in one message, tag expected
in next — but user moved on without running tag. Recovery required
retroactive `git tag vX.Y.Z <hash>` (works, but flags session non-atomic).

Verification before moving on:
- `git describe --tags` = `vX.Y.Z` (exact, not `vX.Y.(Z-1)-N-g...`).

### §6.5. Pre-tag verification (AP31)

Before `git tag`, ALL must pass:

1. `git status --short` → **empty** (nothing uncommitted).
2. `git diff <prev-tag>..HEAD --name-only` → contains **all expected files** for the release.
3. For each feature in CHANGELOG: `git show HEAD:<file> | grep <marker>` (e.g. new class/method/function name) → must be present.
4. `git describe --tags` → shows the new tag.

**Why:** tests run against **working tree**, not against tag. A tag can be honest (tests pass) but **dishonest** (tag missing files). v0.9.194 shipped this way — Part B (Theta+AutoTheta merge) was uncommitted at tag time; fixed in v0.9.194-d.

**Anti-pattern:** `git tag` after `git commit` without `git status --short`. Never trust "tests are green" as proof of tag correctness.

### §6.5.1. AP31 probe pitfalls (v0.9.194-196)

AP31's first iterations produced 3 false positives of its own. Before trusting an AP31 probe, verify it against these 5 classes:

**P1. Verify with `git show HEAD:file`, not `cat file`.** `cat` reads working tree; uncommitted changes look tagged. If `git status --short` is non-empty, investigate first — do not proceed to `git tag`.

**P2. Substring vs exact match.** Renamed attributes have substring collisions: `_y_centered_recent_` contains `_y_centered_`. A grep/`in` check fails incorrectly. Use word-boundary regex:
- Python: `re.search(r'(?<!\w)_y_centered_(?!\w)', text)` — before non-word AND after non-word.
- Or split on whitespace + `.` + `,`: `set(text.replace('.',' ').replace(',',' ').split())`.

**P3. Booleans must be monotone.** `all_ok = all_ok and not stale` must be **outside** if/else branches. Putting it inside `else:` means `stale=True` never triggers overall FAIL. Test the probe with a deliberately-bad input to confirm it fails.

**P4. awk range with shared start/end pattern.** `/name: X/,/^  - name:/` — if the start line also matches end pattern (e.g. `name: X` at indent matching), range is empty. Use Python YAML parse instead: `yaml.safe_load(git show ...)` and walk `.get("subtypes")[i]["models"]`.

**P5. Patch scripts embedding YAML in f-strings must escape braces.** f-string `f"{axis: n}"` → NameError (axis undefined). Either `f"...{{axis: n}}..."` (escape) or `.format(i=indent)` with `{{` `}}`. Prefer `.replace()` or `.format()` over raw f-string for YAML content. v0.9.196 P54 crashed on this; fix was `.format(i=indent)` with `{{...}}`.

**When the probe fails on `>>> ALL: FAIL`:** do not tag. Fix the probe first (if false positive) or the artifact (if real). Log the false positive class here.

**Long-term:** `scripts/verify_tag.py` (backlog v0.9.197+) automates §6.5 with Python YAML parse — no awk, no f-string.

## §7. When AI is uncertain

Format:

```
QUESTION: <one sentence>
OPTIONS:
  A) <option> — <rationale>
  B) <option> — <rationale>
RECOMMENDATION: <A|B> because <reason>
```

Do not proceed until human chooses. Exception: trivially reversible
choices (log format, file naming) — decide + mention.

**When brief contradicts AI_CONSTITUTION:**

- Constitution wins.
- State explicitly: "Brief says X, constitution says Y. I follow Y
  because §<rule>."
- If human disagrees — ask them to update constitution or issue override.

---

## §8. Anti-patterns (each cost us a session)

| Anti-pattern | Correct |
|---|---|
| Inline heredoc for 200+ line patch | `cat > /tmp/patch.py` then run |
| Outer heredoc delimiter repeats content delimiter | Unique outer (e.g. `CADW_STAGE_N_UNIQUE_EOF`) |
| `pytest ... | tail` with pipefail (AP13) | `> /tmp/log 2>&1; echo $?; tail` |
| Paste full patch script in "output" | Paste only tail of log |
| Patch + verify in one block | Two blocks: apply, then verify |
| Full fast after every minor change | Focused during dev; full before commit |
| Auto-commit after green tests | Wait for explicit "go" |
| Silent decision on ambiguous scope | QUESTION/OPTIONS/RECOMMENDATION |
| Brief overrides constitution | Constitution wins; state conflict |

---

## §9. Session-start checks (auto-generated in snapshot)

At top of every snapshot, after reading order:

```
# SESSION START CHECKLIST:
# [ ] Read AI_CONSTITUTION §1 order
# [ ] Read STABILITY_PROTOCOL.md (formula method)
# [ ] Read CHAT_PROTOCOL.md (this file)
# [ ] Check DECISION_LOG.md — has current topic come up before?
# [ ] If session brief contradicts AI_CONSTITUTION -> ask, don't proceed
# [ ] First action: probe (read-only), not patch
```
