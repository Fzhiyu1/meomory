"""L2→L1→L0 compression pipeline.

L2 = parsed session (from parser.py)
L1 = per-segment fragments (one per user message)
L0 = session-level summary

支持两种模式：
- 规则压缩（默认，无外部依赖）
- LLM 精压（调用 GPT-5.4，产出可行动状态）
"""

from __future__ import annotations

import json
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


# ─── LLM 精压 ────────────────────────────────────────────────────

L1_SYSTEM_PROMPT = """\
你是一个记忆压缩器。你的任务是把一段 AI 对话压缩成一条可行动的记忆片段。

输出要求（严格遵守）：
1. 用 JSON 格式输出，包含两个字段：
   - "cue": 一个关键词数组（3-8个），用于未来检索匹配。关键词必须是具体的技术术语或实体名称（如 "cliproxyapi"、"mihomo"、"飞书机器人"），不要用 UUID、路径、通用词。
   - "body": 一段 200 字以内的压缩摘要。
2. body 的内容必须是**知识**，不是对话记录：
   - ✅ "飞书不回复时排障路径：gateway → nginx → cliproxyapi → mihomo → 上游"
   - ❌ "用户问了飞书不回复怎么办，AI说让我检查一下"
3. body 要**可行动**——读到这条，就知道下次遇到同样问题该怎么做。
4. 如果对话内容是纯寒暄、心跳检查或没有实质内容，body 写 "SKIP"。
5. 术语保持原样，不要翻译或换说法。
6. 只输出 JSON，不要其他内容。"""

L0_SYSTEM_PROMPT = """\
你是一个记忆压缩器。你的任务是把多条 L1 记忆片段压缩成一条极致摘要（L0）。

输出要求（严格遵守）：
1. 用 JSON 格式输出，包含两个字段：
   - "cue": 一个关键词数组（3-8个），覆盖所有 L1 片段的核心主题。
   - "body": 一句话（最多 100 字），概括这个 session 在做什么、结论是什么。
2. 如果所有 L1 都是 SKIP，body 写 "SKIP"。
3. 只输出 JSON，不要其他内容。"""


def compress_session_to_l1_llm(parsed: dict) -> list[dict]:
    """用 LLM 将 parsed session 压缩为高质量 L1 片段。"""
    from src.llm_client import chat

    session_id = parsed["id"]
    id_prefix = session_id[:8]
    model = parsed.get("model", "")
    timestamp = parsed.get("timestamp", "")
    ts_date = timestamp[:10] if timestamp else ""

    # 按 user 消息分段
    segments: list[list[dict]] = []
    for turn in parsed["turns"]:
        if turn["role"] == "user":
            segments.append([turn])
        elif segments:
            segments[-1].append(turn)

    fragments: list[dict] = []
    for idx, segment in enumerate(segments):
        # 组装对话文本给 LLM
        conv_lines = []
        for turn in segment:
            role_label = {"user": "用户", "assistant": "AI", "toolResult": "工具返回"}.get(
                turn["role"], turn["role"]
            )
            text = turn["text"][:500]
            conv_lines.append(f"[{role_label}] {text}")

        conv_text = "\n".join(conv_lines)
        if not conv_text.strip():
            continue

        prompt = f"请压缩以下对话片段：\n\n{conv_text}"

        try:
            raw = chat(prompt, system=L1_SYSTEM_PROMPT, max_tokens=400)
            # 解析 JSON（兼容 markdown code block）
            result = _parse_llm_json(raw)
            body = result.get("body", "").strip()
            cues = result.get("cue", [])

            if body == "SKIP" or not body:
                continue

            fragments.append({
                "id": f"l1-{id_prefix}-{idx:03d}",
                "cue": cues[:8],
                "ts": ts_date,
                "source": f"session/{session_id}",
                "model": model,
                "body": _truncate(body, 300),
            })
        except Exception as e:
            # LLM 失败时降级为规则压缩
            print(f"  LLM compress failed for {id_prefix}-{idx:03d}: {e}, falling back to rule-based")
            rule_frags = compress_session_to_l1(parsed)
            if idx < len(rule_frags):
                fragments.append(rule_frags[idx])

    return fragments


def compress_l1_to_l0_llm(l1_fragments: list[dict], session_meta: dict) -> dict:
    """用 LLM 将 L1 片段压缩为 L0 极致摘要。"""
    from src.llm_client import chat

    session_id = session_meta["id"]
    id_prefix = session_id[:8]
    timestamp = session_meta.get("timestamp", "")
    ts_date = timestamp[:10] if timestamp else ""

    if not l1_fragments:
        return {
            "id": f"l0-{id_prefix}",
            "ts": ts_date,
            "source": f"session/{session_id}",
            "turns": 0,
            "body": "(空 session)",
        }

    # 非 SKIP 的 L1 片段
    valid = [f for f in l1_fragments if f.get("body") != "SKIP"]
    if not valid:
        return {
            "id": f"l0-{id_prefix}",
            "ts": ts_date,
            "source": f"session/{session_id}",
            "turns": len(l1_fragments),
            "body": "SKIP",
        }

    l1_text = "\n\n".join(
        f"[{f['id']}] cue={f['cue']}\n{f['body']}" for f in valid
    )
    prompt = f"请将以下 L1 记忆片段压缩为一条 L0 摘要：\n\n{l1_text}"

    try:
        raw = chat(prompt, system=L0_SYSTEM_PROMPT, max_tokens=200)
        result = _parse_llm_json(raw)
        body = result.get("body", "").strip()
        cues = result.get("cue", [])

        if body == "SKIP" or not body:
            body = "SKIP"

        return {
            "id": f"l0-{id_prefix}",
            "cue": cues[:8],
            "ts": ts_date,
            "source": f"session/{session_id}",
            "turns": len(l1_fragments),
            "body": _truncate(body, 150),
        }
    except Exception as e:
        print(f"  LLM L0 compress failed for {id_prefix}: {e}, falling back")
        return compress_l1_to_l0(l1_fragments, session_meta)


def _parse_llm_json(raw: str) -> dict:
    """解析 LLM 输出的 JSON，兼容 markdown code block。"""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)
