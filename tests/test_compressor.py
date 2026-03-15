"""Tests for L2→L1→L0 compression pipeline."""

from src.compressor import compress_session_to_l1, compress_l1_to_l0


SAMPLE_PARSED = {
    "id": "test-001",
    "source_file": "test-001.jsonl",
    "timestamp": "2026-03-14T10:00:00Z",
    "model": "gpt-5.4",
    "turns": [
        {
            "role": "user",
            "text": "飞书机器人不回复了怎么办",
            "timestamp": "2026-03-14T10:00:01Z",
            "tokens": None,
        },
        {
            "role": "assistant",
            "text": "让我检查一下 gateway 状态",
            "timestamp": "2026-03-14T10:00:05Z",
            "tokens": 50,
        },
        {
            "role": "toolResult",
            "text": "active (running)",
            "timestamp": "2026-03-14T10:00:08Z",
            "tokens": None,
        },
        {
            "role": "assistant",
            "text": "gateway 正常运行，问题可能在 nginx 反代或 mihomo 代理",
            "timestamp": "2026-03-14T10:00:12Z",
            "tokens": 80,
        },
    ],
    "total_tokens": 130,
    "turn_count": 4,
}


class TestCompressSessionToL1:
    def test_returns_list_with_at_least_one_fragment(self):
        result = compress_session_to_l1(SAMPLE_PARSED)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_l1_fragment_has_required_fields(self):
        result = compress_session_to_l1(SAMPLE_PARSED)
        for frag in result:
            assert "id" in frag
            assert "cue" in frag
            assert "ts" in frag
            assert "source" in frag
            assert "body" in frag
            assert frag["id"].startswith("l1-")

    def test_l1_body_within_500_chars(self):
        result = compress_session_to_l1(SAMPLE_PARSED)
        for frag in result:
            assert len(frag["body"]) <= 500

    def test_l1_cue_max_15(self):
        result = compress_session_to_l1(SAMPLE_PARSED)
        for frag in result:
            assert len(frag["cue"]) <= 15

    def test_l1_id_format(self):
        result = compress_session_to_l1(SAMPLE_PARSED)
        # id should be l1-{first 8 chars of session id}-{seq:03d}
        first = result[0]
        assert first["id"] == "l1-test-001-000"

    def test_l1_source_references_session(self):
        result = compress_session_to_l1(SAMPLE_PARSED)
        for frag in result:
            assert frag["source"] == "session/test-001"

    def test_l1_ts_is_date_only(self):
        result = compress_session_to_l1(SAMPLE_PARSED)
        for frag in result:
            # Should be YYYY-MM-DD format, no time
            assert len(frag["ts"]) == 10
            assert frag["ts"].count("-") == 2


class TestCompressL1ToL0:
    def test_returns_correct_format(self):
        l1_fragments = compress_session_to_l1(SAMPLE_PARSED)
        l0 = compress_l1_to_l0(l1_fragments, SAMPLE_PARSED)

        assert l0["id"].startswith("l0-")
        assert "cue" in l0
        assert "ts" in l0
        assert "source" in l0
        assert "turns" in l0
        assert "body" in l0

    def test_l0_body_within_200_chars(self):
        l1_fragments = compress_session_to_l1(SAMPLE_PARSED)
        l0 = compress_l1_to_l0(l1_fragments, SAMPLE_PARSED)
        assert len(l0["body"]) <= 200

    def test_l0_cue_max_10(self):
        l1_fragments = compress_session_to_l1(SAMPLE_PARSED)
        l0 = compress_l1_to_l0(l1_fragments, SAMPLE_PARSED)
        assert len(l0["cue"]) <= 10

    def test_l0_turns_matches_l1_count(self):
        l1_fragments = compress_session_to_l1(SAMPLE_PARSED)
        l0 = compress_l1_to_l0(l1_fragments, SAMPLE_PARSED)
        assert l0["turns"] == len(l1_fragments)

    def test_l0_empty_session(self):
        empty_parsed = {
            "id": "empty-001",
            "source_file": "empty.jsonl",
            "timestamp": "2026-03-14T10:00:00Z",
            "model": "gpt-5.4",
            "turns": [],
            "total_tokens": 0,
            "turn_count": 0,
        }
        l1 = compress_session_to_l1(empty_parsed)
        l0 = compress_l1_to_l0(l1, empty_parsed)
        assert l0["body"] == "(空 session)"
