#!/bin/bash
# Append every user prompt to prompts/log.md.
# Reads JSON from stdin with a .prompt field.

set -e
mkdir -p "$CLAUDE_PROJECT_DIR/prompts"

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')

# Strip system-injected XML blocks before logging (multiline, so use perl slurp mode)
CLEANED=$(printf '%s' "$PROMPT" | perl -0777 -pe 's/<task-notification>.*?<\/task-notification>//gs')
CLEANED=$(printf '%s' "$CLEANED" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

if [ -n "$CLEANED" ]; then
  {
    echo ""
    echo "---"
    echo "## $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "$CLEANED"
  } >> "$CLAUDE_PROJECT_DIR/prompts/log.md"
fi

exit 0