#!/bin/bash
# 从 OpenClaw 拉取所有 session JSONL 到 data/raw/
set -euo pipefail

REMOTE="openclaw"
REMOTE_DIR="~/.openclaw/agents/main/sessions/"
LOCAL_DIR="$(dirname "$0")/../data/raw/"

mkdir -p "$LOCAL_DIR"

echo "Pulling active sessions..."
scp "${REMOTE}:${REMOTE_DIR}*.jsonl" "$LOCAL_DIR" 2>/dev/null || true

echo "Pulling deleted sessions..."
scp "${REMOTE}:${REMOTE_DIR}*.deleted.*" "$LOCAL_DIR" 2>/dev/null || true

count=$(ls -1 "$LOCAL_DIR" | wc -l | tr -d ' ')
echo "Done. $count files pulled to $LOCAL_DIR"
