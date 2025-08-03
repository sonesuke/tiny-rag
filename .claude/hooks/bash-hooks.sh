#!/bin/bash

# Read JSON input from stdin
INPUT=$(cat)

# Extract relevant fields from the JSON input
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only process if this is a Bash tool call
if [[ "$TOOL_NAME" == "Bash" ]]; then
    # Check if the command starts with 'python' (but not 'uv run python' or 'uv python')
    if [[ "$COMMAND" =~ ^python[[:space:]] ]] || [[ "$COMMAND" == "python" ]]; then
        if ! [[ "$COMMAND" =~ ^uv[[:space:]]+(run[[:space:]]+)?python ]]; then
            # Return JSON response to block the tool use
            cat <<EOF
{
  "decision": "block",
  "message": "Error: Direct python execution is not allowed in this project.\nPlease use 'uv run python' instead of 'python'.\nExample: uv run python script.py"
}
EOF
            exit 0
        fi
    fi
    
    # Check if the command is 'git commit'
    if [[ "$COMMAND" =~ ^git[[:space:]]+commit ]]; then
        echo "Running tests and linting before commit..." >&2
        echo "This may take a moment..." >&2
        
        # Find the project root by looking for pyproject.toml
        PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
        if [[ -z "$PROJECT_ROOT" ]]; then
            cat <<EOF
{
  "decision": "block",
  "message": "Error: Not in a git repository"
}
EOF
            exit 0
        fi
        
        # Change to project directory
        cd "$PROJECT_ROOT"
        
        # Run make test
        echo "Running make test..." >&2
        if ! make test >&2; then
            cat <<EOF
{
  "decision": "block",
  "message": "Error: Tests failed. Please fix failing tests before committing."
}
EOF
            exit 0
        fi
        
        # Run make lint
        echo "Running make lint..." >&2
        if ! make lint >&2; then
            cat <<EOF
{
  "decision": "block",
  "message": "Error: Linting failed. Please fix linting issues before committing."
}
EOF
            exit 0
        fi
        
        echo "All checks passed! Proceeding with commit..." >&2
    fi
fi

# Allow the tool to proceed
cat <<EOF
{
  "decision": "approve"
}
EOF