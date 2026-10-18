# scripts/_lib.sh — общие проверки для патч-скриптов.
require_clean_tree() {
  local bad_tracked bad_untracked
  bad_tracked="$(git status --porcelain --untracked-files=no)"
  if [[ -n "$bad_tracked" ]]; then
    echo "ERROR: tracked files have uncommitted changes:" >&2
    echo "$bad_tracked" >&2
    return 1
  fi
  bad_untracked=""
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    case "$f" in
      scripts/*|.backups/*|__pycache__/*|*/__pycache__/*|*.pyc|.pytest_cache/*)
        : ;;
      *)
        bad_untracked+="  $f"$'\n' ;;
    esac
  done < <(git ls-files --others --exclude-standard)
  if [[ -n "$bad_untracked" ]]; then
    echo "ERROR: untracked files outside scripts/, .backups/, caches:" >&2
    echo "$bad_untracked" >&2
    return 1
  fi
  return 0
}
