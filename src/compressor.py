"""L2→L1→L0 rule-based compression pipeline.

L2 = parsed session (from parser.py)
L1 = per-segment fragments (one per user message)
L0 = session-level summary
"""

from __future__ import annotations

import re

STOP_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "are",
        "was",
        "not",
        "but",
        "you",
        "your",
        "can",
        "will",
        "let",
        "has",
        "have",
        "been",
        "just",
        "more",
        "some",
        "than",
        "its",
        "also",
        "into",
    }
)

_CUE_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b")


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters, appending '...' if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _extract_cues(text: str, max_count: int = 15) -> list[str]:
    """Extract English technical terms as cue keywords.

    Returns up to max_count unique lowercase terms, preserving first-seen order,
    with stop words filtered out.
    """
    tokens = _CUE_RE.findall(text)
    seen: set[str] = set()
    cues: list[str] = []
    for tok in tokens:
        lower = tok.lower()
        if lower in STOP_WORDS or lower in seen:
            continue
        seen.add(lower)
        cues.append(lower)
        if len(cues) >= max_count:
            break
    return cues


def compress_session_to_l1(parsed: dict) -> list[dict]:
    """Compress a parsed session (L2) into a list of L1 fragments.

    Segments the session by user messages: each user message starts a new
    segment that includes all subsequent non-user turns until the next user
    message.
    """
    session_id = parsed["id"]
    id_prefix = session_id[:8]
    model = parsed.get("model", "")
    timestamp = parsed.get("timestamp", "")
    ts_date = timestamp[:10] if timestamp else ""

    # Split turns into segments by user messages
    segments: list[list[dict]] = []
    for turn in parsed["turns"]:
        if turn["role"] == "user":
            segments.append([turn])
        else:
            if segments:
                segments[-1].append(turn)
            # else: non-user turn before any user message, discard

    fragments: list[dict] = []
    for idx, segment in enumerate(segments):
        user_text = ""
        assistant_texts: list[str] = []
        tool_texts: list[str] = []

        for turn in segment:
            if turn["role"] == "user":
                user_text = _truncate(turn["text"], 200)
            elif turn["role"] == "assistant":
                assistant_texts.append(_truncate(turn["text"], 200))
            elif turn["role"] == "toolResult":
                tool_texts.append(_truncate(turn["text"], 100))

        # Build body
        parts: list[str] = []
        if user_text:
            parts.append(f"用户: {user_text}")
        if assistant_texts:
            parts.append(f"动作: {' → '.join(assistant_texts)}")
        if tool_texts:
            parts.append(f"工具返回: {'; '.join(tool_texts)}")

        body = _truncate("\n".join(parts), 500)

        # Extract cues from all text in this segment
        all_text = " ".join(t["text"] for t in segment)
        cues = _extract_cues(all_text, max_count=15)

        fragments.append(
            {
                "id": f"l1-{id_prefix}-{idx:03d}",
                "cue": cues,
                "ts": ts_date,
                "source": f"session/{session_id}",
                "model": model,
                "body": body,
            }
        )

    return fragments


def compress_l1_to_l0(l1_fragments: list[dict], session_meta: dict) -> dict:
    """Compress L1 fragments into a single L0 session summary."""
    session_id = session_meta["id"]
    id_prefix = session_id[:8]
    timestamp = session_meta.get("timestamp", "")
    ts_date = timestamp[:10] if timestamp else ""

    # Merge cues: preserve order, deduplicate, max 10
    seen: set[str] = set()
    merged_cues: list[str] = []
    for frag in l1_fragments:
        for cue in frag.get("cue", []):
            if cue not in seen:
                seen.add(cue)
                merged_cues.append(cue)
                if len(merged_cues) >= 10:
                    break
        if len(merged_cues) >= 10:
            break

    # Body: truncated first L1 body, or empty session marker
    if l1_fragments:
        body = _truncate(l1_fragments[0]["body"], 200)
    else:
        body = "(空 session)"

    return {
        "id": f"l0-{id_prefix}",
        "cue": merged_cues,
        "ts": ts_date,
        "source": f"session/{session_id}",
        "turns": len(l1_fragments),
        "body": body,
    }
