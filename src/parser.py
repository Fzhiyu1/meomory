"""Parse OpenClaw session JSONL files into structured format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _extract_text(content: Any) -> str:
    """Extract text from message content (str, list, or other)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "toolCall":
                tool_name = item.get("tool") or item.get("name") or "unknown_tool"
                parts.append(f"[tool:{tool_name}]")
        return " ".join(parts)
    return str(content) if content else ""


def parse_session(path: str | Path) -> dict[str, Any]:
    """Parse a single session JSONL file into structured data.

    Returns a dict with keys: id, source_file, timestamp, model, turns,
    total_tokens, turn_count.
    """
    path = Path(path)

    session_id: str = ""
    session_timestamp: str = ""
    model: str = ""
    turns: list[dict[str, Any]] = []
    total_tokens: int = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rtype = record.get("type")

            if rtype == "session":
                session_id = record.get("sessionId", "") or record.get("id", "")
                session_timestamp = record.get("timestamp", "")

            elif rtype == "model_change":
                # Take the last model_change as the primary model
                model = record.get("modelId", "")

            elif rtype == "message":
                msg = record.get("message", {})
                role = msg.get("role", "")
                content = msg.get("content")
                text = _extract_text(content)

                # Skip empty text turns
                if not text.strip():
                    continue

                # Extract tokens from assistant usage
                tokens: int | None = None
                usage = msg.get("usage")
                if usage and isinstance(usage, dict):
                    tok = usage.get("totalTokens")
                    if tok is not None:
                        tokens = int(tok)
                        total_tokens += tokens

                timestamp = record.get("timestamp", "")

                turns.append(
                    {
                        "role": role,
                        "text": text,
                        "timestamp": timestamp,
                        "tokens": tokens,
                    }
                )

    return {
        "id": session_id,
        "source_file": path.name,
        "timestamp": session_timestamp,
        "model": model,
        "turns": turns,
        "total_tokens": total_tokens,
        "turn_count": len(turns),
    }


def parse_all_sessions(raw_dir: str | Path) -> list[dict[str, Any]]:
    """Parse all .jsonl* files in a directory.

    Picks up both active (.jsonl) and deleted (.jsonl.deleted.*) files.
    """
    raw_dir = Path(raw_dir)
    results: list[dict[str, Any]] = []

    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue
        # Match *.jsonl and *.jsonl.deleted.* etc.
        if ".jsonl" not in path.name:
            continue
        try:
            result = parse_session(path)
            results.append(result)
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"Warning: skipping {path.name}: {exc}")

    return results
