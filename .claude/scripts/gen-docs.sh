#!/usr/bin/env bash
# PostToolUse hook: regenerate docs/notes/module_diagram.md after any src/*.py edit.
# Reads the tool-call JSON on stdin; only fires for edits to files under src/.
set -euo pipefail

payload="$(cat)"
file_path="$(printf '%s' "$payload" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("file_path",""))' 2>/dev/null || true)"

case "$file_path" in
  */src/*.py|src/*.py)
    cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0
    # Regenerate the ASCII module diagram + inventory (deterministic; never blocks).
    python -m src.gen_module_docs >/dev/null 2>&1 || true
    ;;
esac
exit 0
