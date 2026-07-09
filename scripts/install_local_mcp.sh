#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -e .
mkdir -p workspaces
BIN="$ROOT/.venv/bin/vidmcp"
echo "Installed: $BIN"
echo "Add to Claude Desktop config:"
cat <<JSON
{
  "mcpServers": {
    "vidmcp": {
      "command": "$BIN",
      "env": {
        "VIDMCP_SAM_BACKEND": "mock",
        "VIDMCP_WORKSPACE_ROOT": "$ROOT/workspaces"
      }
    }
  }
}
JSON
