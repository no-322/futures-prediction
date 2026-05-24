#!/bin/bash
# After Claude edits a file in src/, run the matching test file.
# Exits 2 on test failure so Claude sees the error and fixes it.

set -e
INPUT=$(cat)
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty')

if [[ "$FILE" == *"/src/"*".py" ]]; then
  BASENAME=$(basename "$FILE" .py)
  TEST_FILE="$CLAUDE_PROJECT_DIR/tests/test_${BASENAME}.py"

  if [ -f "$TEST_FILE" ]; then
    cd "$CLAUDE_PROJECT_DIR"
    if ! pytest "$TEST_FILE" -q; then
      echo "Tests failed in $TEST_FILE — fix before continuing." >&2
      exit 2
    fi
  fi
fi

exit 0