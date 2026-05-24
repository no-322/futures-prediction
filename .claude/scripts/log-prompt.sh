#!/bin/bash
# Append every user prompt to prompts/log.md.
# Reads JSON from stdin with a .prompt field.

set -e
mkdir -p "$CLAUDE_PROJECT_DIR/prompts"

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')

if [ -n "$PROMPT" ]; then
  {
    echo ""
    echo "---"
    echo "## $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "$PROMPT"
  } >> "$CLAUDE_PROJECT_DIR/prompts/log.md"
fi

exit 0