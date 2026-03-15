"""Tests for session JSONL parser."""

import json
import tempfile
from pathlib import Path

from src.parser import parse_session, parse_all_sessions


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_session_lines() -> list[dict]:
    """Build a minimal but realistic session JSONL sequence."""
    return [
        {
            "type": "session",
            "sessionId": "sess-abc-123",
            "timestamp": "2026-03-10T08:00:00Z",
            "cwd": "/home/user/project",
        },
        {
            "type": "model_change",
            "provider": "anthropic",
            "modelId": "claude-sonnet-4-20250514",
        },
        {
            "type": "message",
            "message": {
                "role": "user",
                "content": "Hello, can you help me?",
            },
            "timestamp": "2026-03-10T08:00:01Z",
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Sure, I can help!"},
                    {"type": "toolCall", "tool": "read_file", "args": {}},
                ],
                "usage": {"totalTokens": 150},
            },
            "timestamp": "2026-03-10T08:00:02Z",
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "content": "file contents here",
            },
            "timestamp": "2026-03-10T08:00:03Z",
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": "Here is the result.",
                "usage": {"totalTokens": 80},
            },
            "timestamp": "2026-03-10T08:00:04Z",
        },
    ]


class TestParseSession:
    def test_returns_correct_id_and_model(self, tmp_path: Path):
        path = tmp_path / "sess-abc-123.jsonl"
        _write_jsonl(path, _make_session_lines())

        result = parse_session(path)

        assert result["id"] == "sess-abc-123"
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["timestamp"] == "2026-03-10T08:00:00Z"

    def test_turns_extracted_correctly(self, tmp_path: Path):
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, _make_session_lines())

        result = parse_session(path)
        roles = [t["role"] for t in result["turns"]]

        assert "user" in roles
        assert "assistant" in roles
        assert "toolResult" in roles

        # User message text
        user_turns = [t for t in result["turns"] if t["role"] == "user"]
        assert user_turns[0]["text"] == "Hello, can you help me?"

    def test_assistant_content_list_joined(self, tmp_path: Path):
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, _make_session_lines())

        result = parse_session(path)
        assistant_turns = [t for t in result["turns"] if t["role"] == "assistant"]

        # First assistant turn has list content: text + toolCall
        first = assistant_turns[0]
        assert "Sure, I can help!" in first["text"]
        assert "read_file" in first["text"]

    def test_token_count_correct(self, tmp_path: Path):
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, _make_session_lines())

        result = parse_session(path)

        assert result["total_tokens"] == 230  # 150 + 80
        assert result["turn_count"] == 4  # user + assistant + toolResult + assistant

    def test_empty_content_turns_skipped(self, tmp_path: Path):
        lines = [
            {
                "type": "session",
                "sessionId": "sess-empty",
                "timestamp": "2026-03-10T09:00:00Z",
            },
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": "",
                },
                "timestamp": "2026-03-10T09:00:01Z",
            },
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": "Real message",
                },
                "timestamp": "2026-03-10T09:00:02Z",
            },
        ]
        path = tmp_path / "empty.jsonl"
        _write_jsonl(path, lines)

        result = parse_session(path)
        assert result["turn_count"] == 1
        assert result["turns"][0]["text"] == "Real message"


class TestParseAllSessions:
    def test_parses_multiple_files(self, tmp_path: Path):
        for name in ["a.jsonl", "b.jsonl"]:
            _write_jsonl(
                tmp_path / name,
                [
                    {
                        "type": "session",
                        "sessionId": name.replace(".jsonl", ""),
                        "timestamp": "2026-03-10T10:00:00Z",
                    },
                    {
                        "type": "message",
                        "message": {"role": "user", "content": "hi"},
                        "timestamp": "2026-03-10T10:00:01Z",
                    },
                ],
            )

        results = parse_all_sessions(tmp_path)
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {"a", "b"}

    def test_parses_deleted_sessions(self, tmp_path: Path):
        _write_jsonl(
            tmp_path / "old.jsonl.deleted.1234",
            [
                {
                    "type": "session",
                    "sessionId": "old",
                    "timestamp": "2026-03-10T10:00:00Z",
                },
                {
                    "type": "message",
                    "message": {"role": "user", "content": "hi"},
                    "timestamp": "2026-03-10T10:00:01Z",
                },
            ],
        )

        results = parse_all_sessions(tmp_path)
        assert len(results) == 1
        assert results[0]["id"] == "old"
