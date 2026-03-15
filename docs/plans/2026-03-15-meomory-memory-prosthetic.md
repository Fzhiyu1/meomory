# meomory 记忆假肢原型 — 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 从 OpenClaw session 数据中压缩出分层记忆片段，构建一个基于向量匹配的"图书馆版"记忆激活管线，为后续升级到关联记忆（DGD/CMS）打好基础。

**Architecture:** 本地 Python 项目，分三个独立模块：(1) session 数据拉取与解析器 (2) L2→L1→L0 压缩管线 (3) 向量匹配激活层。阶段一先在本地跑通整个管线，产出可检验的记忆片段和匹配效果。不涉及 OpenClaw gateway 集成（那是阶段一的最后一步，等管线验证后再做）。

**Tech Stack:** Python 3.14, SSH/SCP 拉取数据, LLM API 做压缩（通过 OpenClaw 的 relay 或直接调 Claude）, qwen3-embedding（OpenClaw 上的 Ollama 容器 :11435）做向量编码

---

## 整体架构四阶段概览

```
阶段一：跑通管线（本计划详细展开）
  T1: 项目初始化
  T2: Session 数据拉取与解析
  T3: L2→L1 压缩（原始session → 交互摘要片段）
  T4: L1→L0 压缩（摘要片段 → 极致压缩的状态行）
  T5: 向量编码与存储
  T6: 匹配查询接口
  T7: 端到端验证 + recall_memory 工具原型
  → 产出：可用的"图书馆版"记忆系统

阶段二：从图书馆升级成记忆
  - 上下文门控（相关度加权注入）
  - Zipfian 频率统计（高频提升到 L0）
  - thinking cue 集成
  → 产出：有门控、有分层、有扩散激活

阶段三：加在线学习
  - DGD 更新规则替换 cosine 匹配
  - 收集训练对数据
  → 产出：匹配器从经验中学习

阶段四：加多时间尺度
  - CMS 多频率层
  → 产出：完整的可在线学习记忆模块
```

---

## 阶段一详细计划

### 目录结构（最终态）

```
meomory/
├── docs/plans/                    # 计划文档
├── data/
│   ├── raw/                       # L3: 从 OpenClaw 拉下来的原始 session JSONL
│   └── parsed/                    # L2: 解析后的结构化 session JSON
├── memory/
│   ├── l1/                        # L1 片段（带 cue 的交互摘要）
│   ├── l0/                        # L0 片段（极致压缩的状态行）
│   └── vectors/                   # 向量编码存储
├── src/
│   ├── __init__.py
│   ├── puller.py                  # T2: 从 OpenClaw 拉取 session 数据
│   ├── parser.py                  # T2: 解析 JSONL 为结构化格式
│   ├── compressor.py              # T3+T4: L2→L1→L0 压缩管线
│   ├── embedder.py                # T5: 调用 qwen3-embedding 做向量编码
│   ├── store.py                   # T5: 向量存储与检索
│   └── matcher.py                 # T6: 匹配查询接口
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_compressor.py
│   ├── test_embedder.py
│   ├── test_store.py
│   └── test_matcher.py
├── scripts/
│   ├── pull_sessions.sh           # 一键拉取脚本
│   └── run_pipeline.py            # 端到端管线脚本
├── pyproject.toml
└── .gitignore
```

---

### Task 1: 项目初始化

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

**Step 1: 初始化 git 仓库**

```bash
cd /Users/fangzhiyu/run/meomory
git init
```

**Step 2: 创建 pyproject.toml**

```toml
[project]
name = "meomory"
version = "0.1.0"
description = "Memory prosthetic for LLM state restoration"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]
```

依赖说明：
- `httpx`：调用 qwen3-embedding 的 Ollama HTTP API 和 LLM API 做压缩
- 不用 `openai` SDK，直接 HTTP 调，减少依赖

**Step 3: 创建 .gitignore**

```
__pycache__/
*.pyc
.venv/
data/raw/
data/parsed/
memory/vectors/
.env
```

注意：`data/raw/` 和 `data/parsed/` 不进 git（包含用户隐私数据），但 `memory/l1/` 和 `memory/l0/` 进 git（压缩后的片段是项目产出）。

**Step 4: 创建目录结构和空 __init__.py**

```bash
mkdir -p src tests data/raw data/parsed memory/l1 memory/l0 memory/vectors scripts
touch src/__init__.py tests/__init__.py
```

**Step 5: 安装依赖**

```bash
cd /Users/fangzhiyu/run/meomory
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Step 6: 验证 pytest 能跑**

Run: `source .venv/bin/activate && pytest tests/ -v`
Expected: `no tests ran` (0 collected, no error)

**Step 7: Commit**

```bash
git add pyproject.toml .gitignore src/ tests/ docs/
git commit -m "init: meomory project skeleton"
```

---

### Task 2: Session 数据拉取与解析

**Files:**
- Create: `src/puller.py`
- Create: `src/parser.py`
- Create: `tests/test_parser.py`
- Create: `scripts/pull_sessions.sh`

**Step 1: 创建拉取脚本 `scripts/pull_sessions.sh`**

```bash
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
```

**Step 2: 运行拉取脚本验证**

```bash
chmod +x scripts/pull_sessions.sh
bash scripts/pull_sessions.sh
ls data/raw/ | head -5
```

Expected: 看到 JSONL 文件被拉到 `data/raw/`

**Step 3: 写 parser 的 failing test**

```python
# tests/test_parser.py
import json
import tempfile
from pathlib import Path
from src.parser import parse_session


def make_session_file(events: list[dict]) -> Path:
    """创建临时 JSONL 文件用于测试"""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for e in events:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
    f.close()
    return Path(f.name)


SAMPLE_EVENTS = [
    {"type": "session", "version": 3, "id": "test-001", "timestamp": "2026-03-14T10:00:00Z", "cwd": "/tmp"},
    {"type": "model_change", "id": "m1", "parentId": None, "timestamp": "2026-03-14T10:00:00Z", "provider": "relay", "modelId": "gpt-5.4"},
    {"type": "message", "id": "msg1", "parentId": "m1", "timestamp": "2026-03-14T10:00:01Z",
     "message": {"role": "user", "content": [{"type": "text", "text": "你好"}], "timestamp": 1000}},
    {"type": "message", "id": "msg2", "parentId": "msg1", "timestamp": "2026-03-14T10:00:05Z",
     "message": {"role": "assistant", "content": [{"type": "text", "text": "你好！有什么可以帮你的？"}],
                 "model": "gpt-5.4", "provider": "relay", "usage": {"input": 100, "output": 20, "totalTokens": 120}}},
    {"type": "message", "id": "msg3", "parentId": "msg2", "timestamp": "2026-03-14T10:00:10Z",
     "message": {"role": "toolResult", "content": "文件内容...", "timestamp": 1000}},
]


def test_parse_session_basic():
    path = make_session_file(SAMPLE_EVENTS)
    result = parse_session(path)

    assert result["id"] == "test-001"
    assert result["model"] == "gpt-5.4"
    assert len(result["turns"]) > 0


def test_parse_session_extracts_turns():
    path = make_session_file(SAMPLE_EVENTS)
    result = parse_session(path)

    # 应该把连续的 user → assistant → toolResult 组织成 turn
    user_msgs = [t for t in result["turns"] if t["role"] == "user"]
    assert len(user_msgs) == 1
    assert "你好" in user_msgs[0]["text"]


def test_parse_session_counts_tokens():
    path = make_session_file(SAMPLE_EVENTS)
    result = parse_session(path)

    assert result["total_tokens"] > 0
```

**Step 4: 运行 test 验证 fail**

Run: `source .venv/bin/activate && pytest tests/test_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.parser'`

**Step 5: 实现 parser.py**

```python
# src/parser.py
"""解析 OpenClaw session JSONL 为结构化格式"""
import json
from pathlib import Path


def parse_session(path: Path) -> dict:
    """解析单个 session JSONL 文件，返回结构化数据。

    返回格式:
    {
        "id": str,
        "source_file": str,
        "timestamp": str,           # session 开始时间
        "model": str,               # 主模型
        "turns": [                   # 按时间序排列的消息
            {"role": "user"|"assistant"|"toolResult",
             "text": str,            # 提取的文本内容
             "timestamp": str,
             "tokens": int|None}
        ],
        "total_tokens": int,
        "turn_count": int,
    }
    """
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    # 提取 session 元信息
    session_event = next((e for e in events if e.get("type") == "session"), {})
    model_event = next((e for e in events if e.get("type") == "model_change"), {})

    turns = []
    total_tokens = 0

    for event in events:
        if event.get("type") != "message":
            continue

        msg = event.get("message", {})
        role = msg.get("role", "")
        text = _extract_text(msg.get("content", ""))
        tokens = msg.get("usage", {}).get("totalTokens", 0) if isinstance(msg.get("usage"), dict) else 0
        total_tokens += tokens

        if text.strip():
            turns.append({
                "role": role,
                "text": text,
                "timestamp": event.get("timestamp", ""),
                "tokens": tokens or None,
            })

    return {
        "id": session_event.get("id", path.stem),
        "source_file": path.name,
        "timestamp": session_event.get("timestamp", ""),
        "model": model_event.get("modelId", "unknown"),
        "turns": turns,
        "total_tokens": total_tokens,
        "turn_count": len(turns),
    }


def _extract_text(content) -> str:
    """从 message content 中提取纯文本"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "toolCall":
                    name = item.get("name", item.get("function", {}).get("name", "?"))
                    parts.append(f"[tool:{name}]")
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content)


def parse_all_sessions(raw_dir: Path) -> list[dict]:
    """解析目录下所有 session 文件"""
    sessions = []
    for path in sorted(raw_dir.glob("*.jsonl*")):
        try:
            sessions.append(parse_session(path))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: skipping {path.name}: {e}")
    return sessions
```

**Step 6: 运行 test 验证 pass**

Run: `source .venv/bin/activate && pytest tests/test_parser.py -v`
Expected: 3 passed

**Step 7: Commit**

```bash
git add src/parser.py tests/test_parser.py scripts/pull_sessions.sh
git commit -m "feat: session puller and JSONL parser"
```

---

### Task 3: L2→L1 压缩（session → 交互摘要片段）

**Files:**
- Create: `src/compressor.py`
- Create: `tests/test_compressor.py`

L1 片段格式（每个片段一个 markdown 文件）：

```markdown
---
id: l1-{session_id}-{index}
cue: [关键词1, 关键词2, ...]
ts: 2026-03-14
source: session/{session_id}
model: gpt-5.4
---
{200字以内的交互摘要：这轮对话在做什么、关键决策、结果}
```

**Step 1: 写 compressor 的 failing test**

```python
# tests/test_compressor.py
from src.compressor import compress_session_to_l1, compress_l1_to_l0


SAMPLE_PARSED = {
    "id": "test-001",
    "source_file": "test-001.jsonl",
    "timestamp": "2026-03-14T10:00:00Z",
    "model": "gpt-5.4",
    "turns": [
        {"role": "user", "text": "飞书机器人不回复了怎么办", "timestamp": "2026-03-14T10:00:01Z", "tokens": None},
        {"role": "assistant", "text": "让我检查一下 gateway 状态", "timestamp": "2026-03-14T10:00:05Z", "tokens": 50},
        {"role": "toolResult", "text": "active (running)", "timestamp": "2026-03-14T10:00:08Z", "tokens": None},
        {"role": "assistant", "text": "gateway 正常运行，问题可能在 nginx 反代或 mihomo 代理", "timestamp": "2026-03-14T10:00:12Z", "tokens": 80},
    ],
    "total_tokens": 130,
    "turn_count": 4,
}


def test_compress_session_to_l1_returns_fragments():
    fragments = compress_session_to_l1(SAMPLE_PARSED)
    assert isinstance(fragments, list)
    assert len(fragments) >= 1


def test_l1_fragment_has_required_fields():
    fragments = compress_session_to_l1(SAMPLE_PARSED)
    frag = fragments[0]
    assert "id" in frag
    assert "cue" in frag
    assert "ts" in frag
    assert "source" in frag
    assert "body" in frag
    assert frag["id"].startswith("l1-")


def test_l1_fragment_body_is_concise():
    fragments = compress_session_to_l1(SAMPLE_PARSED)
    for frag in fragments:
        # L1 片段应该在 500 字以内
        assert len(frag["body"]) <= 500, f"Fragment too long: {len(frag['body'])} chars"
```

**Step 2: 运行 test 验证 fail**

Run: `source .venv/bin/activate && pytest tests/test_compressor.py::test_compress_session_to_l1_returns_fragments -v`
Expected: FAIL

**Step 3: 实现 compressor.py（本地规则压缩，不依赖 LLM API）**

阶段一的压缩先用**规则方式**，不调 LLM API。这样管线不依赖外部服务就能跑通，验证数据结构和流程。后续可以在此基础上加 LLM 精压。

```python
# src/compressor.py
"""L2→L1→L0 压缩管线

阶段一：规则压缩（不依赖 LLM API）
后续阶段：可替换为 LLM 驱动的压缩
"""
import re
from datetime import datetime


def compress_session_to_l1(parsed: dict) -> list[dict]:
    """将一个 parsed session 压缩为 L1 片段列表。

    策略：按 user 消息分段，每段生成一个 L1 片段。
    每个片段包含：用户意图、assistant 动作、结果。
    """
    turns = parsed.get("turns", [])
    if not turns:
        return []

    # 按 user 消息分段
    segments = []
    current = []
    for turn in turns:
        if turn["role"] == "user" and current:
            segments.append(current)
            current = []
        current.append(turn)
    if current:
        segments.append(current)

    fragments = []
    for i, segment in enumerate(segments):
        user_text = ""
        assistant_texts = []
        tool_texts = []

        for turn in segment:
            if turn["role"] == "user":
                user_text = _truncate(turn["text"], 200)
            elif turn["role"] == "assistant":
                assistant_texts.append(_truncate(turn["text"], 200))
            elif turn["role"] == "toolResult":
                tool_texts.append(_truncate(turn["text"], 100))

        # 提取 cue（关键词）
        all_text = user_text + " " + " ".join(assistant_texts)
        cues = _extract_cues(all_text)

        # 组装 body
        body_parts = []
        if user_text:
            body_parts.append(f"用户: {user_text}")
        if assistant_texts:
            body_parts.append(f"动作: {' → '.join(assistant_texts)}")
        if tool_texts:
            body_parts.append(f"工具返回: {'; '.join(tool_texts)}")

        body = "\n".join(body_parts)

        ts = segment[0].get("timestamp", parsed.get("timestamp", ""))[:10]

        fragments.append({
            "id": f"l1-{parsed['id'][:8]}-{i:03d}",
            "cue": cues,
            "ts": ts,
            "source": f"session/{parsed['id']}",
            "model": parsed.get("model", "unknown"),
            "body": _truncate(body, 500),
        })

    return fragments


def compress_l1_to_l0(l1_fragments: list[dict], session_meta: dict) -> dict:
    """将一组 L1 片段压缩为一个 L0 状态行。

    L0 是整个 session 的极致摘要，一两句话。
    """
    if not l1_fragments:
        return {
            "id": f"l0-{session_meta.get('id', 'unknown')[:8]}",
            "ts": session_meta.get("timestamp", "")[:10],
            "source": f"session/{session_meta.get('id', '')}",
            "body": "(空 session)",
        }

    # 合并所有 L1 的 cue
    all_cues = []
    for frag in l1_fragments:
        all_cues.extend(frag.get("cue", []))
    # 去重保序
    seen = set()
    unique_cues = []
    for c in all_cues:
        if c not in seen:
            seen.add(c)
            unique_cues.append(c)

    # 取第一个和最后一个 L1 片段的 body 做摘要
    first_body = l1_fragments[0].get("body", "")
    summary = _truncate(first_body, 200)

    return {
        "id": f"l0-{session_meta.get('id', 'unknown')[:8]}",
        "cue": unique_cues[:10],
        "ts": session_meta.get("timestamp", "")[:10],
        "source": f"session/{session_meta.get('id', '')}",
        "turns": len(l1_fragments),
        "body": summary,
    }


def _truncate(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _extract_cues(text: str) -> list[str]:
    """从文本中提取关键词作为 cue。

    阶段一：简单的规则提取（英文技术术语 + 中文关键词）
    后续：可替换为 LLM 提取或 TF-IDF
    """
    # 提取英文技术术语（2字符以上的连续英文+数字+连字符）
    en_terms = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b', text)
    # 转小写去重
    en_terms = list(dict.fromkeys(t.lower() for t in en_terms))

    # 过滤常见停用词
    stopwords = {"the", "and", "for", "that", "this", "with", "from", "are", "was",
                 "not", "but", "you", "your", "can", "will", "let", "has", "have",
                 "been", "just", "more", "some", "than", "its", "also", "into"}
    en_terms = [t for t in en_terms if t not in stopwords]

    return en_terms[:15]
```

**Step 4: 运行 test 验证 pass**

Run: `source .venv/bin/activate && pytest tests/test_compressor.py -v`
Expected: 3 passed

**Step 5: 补充 L0 压缩 test**

在 `tests/test_compressor.py` 末尾追加：

```python
def test_compress_l1_to_l0():
    fragments = compress_session_to_l1(SAMPLE_PARSED)
    l0 = compress_l1_to_l0(fragments, SAMPLE_PARSED)
    assert l0["id"].startswith("l0-")
    assert "body" in l0
    assert len(l0["body"]) <= 200
```

**Step 6: 运行 all tests**

Run: `source .venv/bin/activate && pytest tests/ -v`
Expected: All passed

**Step 7: Commit**

```bash
git add src/compressor.py tests/test_compressor.py
git commit -m "feat: L2→L1→L0 rule-based compression pipeline"
```

---

### Task 4: 向量编码与存储

**Files:**
- Create: `src/embedder.py`
- Create: `src/store.py`
- Create: `tests/test_embedder.py`
- Create: `tests/test_store.py`

向量编码调用 OpenClaw 上的 qwen3-embedding（Ollama 容器 `100.117.243.72:11435`）。

**Step 1: 写 embedder failing test**

```python
# tests/test_embedder.py
import pytest
from src.embedder import get_embedding, EMBED_DIM


def test_get_embedding_returns_vector():
    """需要 OpenClaw 上的 Ollama 可达"""
    vec = get_embedding("这是一段测试文本")
    assert isinstance(vec, list)
    assert len(vec) == EMBED_DIM
    assert all(isinstance(v, float) for v in vec)


def test_get_embedding_different_texts():
    """不同文本应该返回不同向量"""
    v1 = get_embedding("飞书机器人排障")
    v2 = get_embedding("量子物理实验")
    # cosine similarity 应该较低
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(a * a for a in v2) ** 0.5
    cosine = dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0
    assert cosine < 0.95, f"Vectors too similar: {cosine}"
```

**Step 2: 实现 embedder.py**

```python
# src/embedder.py
"""调用 qwen3-embedding 做向量编码

通过 SSH 隧道访问 OpenClaw 上的 Ollama 容器 (100.117.243.72:11435)。
本地需要能 ssh openclaw。
"""
import httpx

OLLAMA_URL = "http://100.117.243.72:11435"
EMBED_MODEL = "qwen3-embedding"
EMBED_DIM = 4096
TIMEOUT = 30.0


def get_embedding(text: str) -> list[float]:
    """获取单条文本的 embedding 向量"""
    resp = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings", [])
    if not embeddings:
        raise ValueError(f"No embeddings returned: {data}")
    return embeddings[0]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """批量获取 embedding"""
    resp = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=TIMEOUT * 2,
    )
    resp.raise_for_status()
    return resp.json().get("embeddings", [])
```

**Step 3: 验证 Ollama 可达后运行 test**

先确认网络通：
```bash
curl -s http://100.117.243.72:11435/api/tags | python3 -m json.tool | head -5
```

如果不通（Tailscale 跨网），走 SSH 隧道：
```bash
ssh -fN -L 11435:127.0.0.1:11435 openclaw
```
然后把 `OLLAMA_URL` 改为 `http://127.0.0.1:11435`。

Run: `source .venv/bin/activate && pytest tests/test_embedder.py -v`
Expected: 2 passed

**Step 4: 写 store failing test**

```python
# tests/test_store.py
import tempfile
from pathlib import Path
from src.store import VectorStore


def test_store_add_and_query():
    store = VectorStore()

    store.add("frag-001", [1.0, 0.0, 0.0], {"body": "飞书排障"})
    store.add("frag-002", [0.0, 1.0, 0.0], {"body": "nginx 配置"})
    store.add("frag-003", [0.9, 0.1, 0.0], {"body": "gateway 重启"})

    results = store.query([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "frag-001"
    assert results[1]["id"] == "frag-003"


def test_store_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"

        store = VectorStore()
        store.add("frag-001", [1.0, 0.0], {"body": "test"})
        store.save(path)

        store2 = VectorStore.load(path)
        results = store2.query([1.0, 0.0], top_k=1)
        assert results[0]["id"] == "frag-001"
```

**Step 5: 实现 store.py**

```python
# src/store.py
"""向量存储与检索

阶段一：纯 Python 实现（内存中 brute-force cosine）。
数据量小（<1000 片段），不需要 FAISS/ANN。
"""
import json
import math
from pathlib import Path


class VectorStore:
    def __init__(self):
        self._entries: list[dict] = []  # [{id, vector, meta}]

    def add(self, id: str, vector: list[float], meta: dict):
        self._entries.append({"id": id, "vector": vector, "meta": meta})

    def query(self, vector: list[float], top_k: int = 5) -> list[dict]:
        scored = []
        for entry in self._entries:
            sim = _cosine_similarity(vector, entry["vector"])
            scored.append({"id": entry["id"], "score": sim, **entry["meta"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self._entries, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        store = cls()
        with open(path) as f:
            store._entries = json.load(f)
        return store

    def __len__(self):
        return len(self._entries)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
```

**Step 6: 运行 all tests**

Run: `source .venv/bin/activate && pytest tests/ -v`
Expected: All passed

**Step 7: Commit**

```bash
git add src/embedder.py src/store.py tests/test_embedder.py tests/test_store.py
git commit -m "feat: embedding client and vector store"
```

---

### Task 5: 匹配查询接口

**Files:**
- Create: `src/matcher.py`
- Create: `tests/test_matcher.py`

**Step 1: 写 matcher failing test**

```python
# tests/test_matcher.py
from src.matcher import MemoryMatcher


def test_matcher_match_returns_fragments():
    matcher = MemoryMatcher()
    # 手动添加一些 L1 片段（绕过 embedding，用固定向量）
    matcher.store.add("l1-001", [1.0, 0.0, 0.0], {
        "body": "飞书机器人不回复排障：gateway → nginx → mihomo",
        "cue": ["飞书", "gateway", "排障"],
    })
    matcher.store.add("l1-002", [0.0, 1.0, 0.0], {
        "body": "知识库记录：记忆假肢概念",
        "cue": ["知识库", "记忆"],
    })

    # 模拟 query（直接传向量）
    results = matcher.query_by_vector([0.9, 0.1, 0.0], top_k=1)
    assert len(results) == 1
    assert "飞书" in results[0]["body"]


def test_matcher_format_for_injection():
    matcher = MemoryMatcher()
    matcher.store.add("l1-001", [1.0, 0.0], {
        "body": "排障路径：gateway → nginx → mihomo",
        "cue": ["飞书", "排障"],
    })

    results = matcher.query_by_vector([1.0, 0.0], top_k=1)
    injected = matcher.format_for_injection(results)

    assert isinstance(injected, str)
    assert "排障路径" in injected
    assert len(injected) < 2000  # 注入文本不能太长
```

**Step 2: 实现 matcher.py**

```python
# src/matcher.py
"""记忆匹配查询接口

给定一段文本（用户消息或 thinking 片段），
返回最相关的记忆片段，并格式化为可注入上下文的文本。
"""
from src.store import VectorStore
from src.embedder import get_embedding


class MemoryMatcher:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()

    def match(self, text: str, top_k: int = 5) -> list[dict]:
        """给定文本，返回最匹配的记忆片段"""
        vector = get_embedding(text)
        return self.query_by_vector(vector, top_k)

    def query_by_vector(self, vector: list[float], top_k: int = 5) -> list[dict]:
        """给定向量，返回最匹配的记忆片段"""
        return self.store.query(vector, top_k)

    def format_for_injection(self, results: list[dict], max_chars: int = 1500) -> str:
        """将匹配结果格式化为可注入上下文的文本"""
        if not results:
            return ""

        lines = ["[相关记忆]"]
        total = 0
        for r in results:
            body = r.get("body", "")
            cue = ", ".join(r.get("cue", []))
            entry = f"- [{cue}] {body}"
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)

        return "\n".join(lines)
```

**Step 3: 运行 all tests**

Run: `source .venv/bin/activate && pytest tests/ -v`
Expected: All passed

**Step 4: Commit**

```bash
git add src/matcher.py tests/test_matcher.py
git commit -m "feat: memory matcher with injection formatting"
```

---

### Task 6: 端到端管线脚本

**Files:**
- Create: `scripts/run_pipeline.py`

这个脚本串联所有模块：拉数据 → 解析 → 压缩 → 编码 → 存储 → 验证匹配。

**Step 1: 实现端到端管线**

```python
#!/usr/bin/env python3
"""meomory 端到端管线：从 raw session 到可查询的记忆库"""
import json
import sys
from pathlib import Path

# 确保 src 在 path 上
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import parse_all_sessions
from src.compressor import compress_session_to_l1, compress_l1_to_l0
from src.embedder import get_embedding
from src.store import VectorStore
from src.matcher import MemoryMatcher

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
L1_DIR = ROOT / "memory" / "l1"
L0_DIR = ROOT / "memory" / "l0"
VECTOR_PATH = ROOT / "memory" / "vectors" / "l1.json"


def main():
    # Step 1: 解析
    print("=== Step 1: Parsing sessions ===")
    sessions = parse_all_sessions(RAW_DIR)
    print(f"  Parsed {len(sessions)} sessions, {sum(s['turn_count'] for s in sessions)} turns total")

    # 过滤空 session
    sessions = [s for s in sessions if s["turn_count"] > 0]
    print(f"  Non-empty: {len(sessions)}")

    # Step 2: L2→L1 压缩
    print("\n=== Step 2: Compressing L2→L1 ===")
    all_l1 = []
    for session in sessions:
        fragments = compress_session_to_l1(session)
        all_l1.extend(fragments)
    print(f"  Generated {len(all_l1)} L1 fragments")

    # 保存 L1 片段
    L1_DIR.mkdir(parents=True, exist_ok=True)
    for frag in all_l1:
        path = L1_DIR / f"{frag['id']}.json"
        with open(path, "w") as f:
            json.dump(frag, f, ensure_ascii=False, indent=2)

    # Step 3: L1→L0 压缩
    print("\n=== Step 3: Compressing L1→L0 ===")
    l0_entries = []
    for session in sessions:
        session_l1 = [f for f in all_l1 if f["source"] == f"session/{session['id']}"]
        l0 = compress_l1_to_l0(session_l1, session)
        l0_entries.append(l0)

    L0_DIR.mkdir(parents=True, exist_ok=True)
    for entry in l0_entries:
        path = L0_DIR / f"{entry['id']}.json"
        with open(path, "w") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
    print(f"  Generated {len(l0_entries)} L0 entries")

    # Step 4: 向量编码
    print(f"\n=== Step 4: Embedding {len(all_l1)} L1 fragments ===")
    store = VectorStore()
    for i, frag in enumerate(all_l1):
        text = frag["body"]
        try:
            vec = get_embedding(text)
            store.add(frag["id"], vec, {
                "body": frag["body"],
                "cue": frag["cue"],
                "ts": frag["ts"],
                "source": frag["source"],
            })
            if (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(all_l1)}")
        except Exception as e:
            print(f"  Warning: failed to embed {frag['id']}: {e}")

    VECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.save(VECTOR_PATH)
    print(f"  Saved {len(store)} vectors to {VECTOR_PATH}")

    # Step 5: 验证匹配
    print("\n=== Step 5: Validation ===")
    matcher = MemoryMatcher(store)

    test_queries = [
        "飞书机器人不回复了",
        "nginx 反代配置",
        "记忆系统设计",
    ]
    for q in test_queries:
        results = matcher.match(q, top_k=3)
        print(f"\n  Query: '{q}'")
        for r in results:
            print(f"    [{r['score']:.3f}] {r['body'][:60]}...")

    print("\n=== Pipeline complete ===")
    print(f"  L1 fragments: {len(all_l1)} → {L1_DIR}")
    print(f"  L0 entries: {len(l0_entries)} → {L0_DIR}")
    print(f"  Vectors: {len(store)} → {VECTOR_PATH}")


if __name__ == "__main__":
    main()
```

**Step 2: 确保 data/raw/ 有数据后运行**

```bash
source .venv/bin/activate
python3 scripts/run_pipeline.py
```

Expected:
- 看到各步骤输出
- `memory/l1/` 下有 JSON 片段文件
- `memory/l0/` 下有 JSON 摘要文件
- `memory/vectors/l1.json` 存在
- 验证查询返回有意义的结果

**Step 3: Commit**

```bash
git add scripts/run_pipeline.py memory/l1/ memory/l0/
git commit -m "feat: end-to-end pipeline script with validation"
```

---

### Task 7: recall_memory 工具原型

**Files:**
- Create: `src/tool.py`

这是给 OpenClaw 或任何 LLM 使用的 `recall_memory` 工具的独立原型。阶段一先做一个命令行版本验证效果，gateway 集成放到管线跑通之后。

**Step 1: 实现 tool.py**

```python
# src/tool.py
"""recall_memory 工具原型

用法（命令行测试）:
    python -m src.tool "飞书机器人不回复了"

用法（作为模块导入）:
    from src.tool import recall_memory
    result = recall_memory("飞书机器人不回复了")
"""
import sys
import json
from pathlib import Path
from src.store import VectorStore
from src.matcher import MemoryMatcher
from src.embedder import get_embedding

VECTOR_PATH = Path(__file__).parent.parent / "memory" / "vectors" / "l1.json"


def recall_memory(query: str, top_k: int = 5) -> str:
    """给定查询文本，返回格式化的相关记忆"""
    if not VECTOR_PATH.exists():
        return "[记忆库未初始化，请先运行 scripts/run_pipeline.py]"

    store = VectorStore.load(VECTOR_PATH)
    matcher = MemoryMatcher(store)
    results = matcher.match(query, top_k=top_k)
    return matcher.format_for_injection(results)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "测试查询"
    print(recall_memory(query))
```

**Step 2: 端到端验证**

```bash
source .venv/bin/activate
python3 -m src.tool "飞书机器人不回复怎么排查"
```

Expected: 输出 `[相关记忆]` 格式的匹配结果

**Step 3: Commit**

```bash
git add src/tool.py
git commit -m "feat: recall_memory CLI tool prototype"
```

---

## 阶段一完成标准

- [ ] `scripts/pull_sessions.sh` 能从 OpenClaw 拉取所有 session
- [ ] `scripts/run_pipeline.py` 能一键跑通 解析→压缩→编码→存储
- [ ] `memory/l1/` 下有结构化的 L1 片段文件
- [ ] `memory/l0/` 下有极致压缩的 L0 摘要文件
- [ ] `python3 -m src.tool "查询文本"` 能返回有意义的匹配结果
- [ ] 所有 test 通过：`pytest tests/ -v`

## 阶段一之后的衔接

阶段一跑通后，进入阶段二前有两个可选的快速改进：

1. **LLM 精压**：在 `compressor.py` 中加一个 `compress_with_llm()` 函数，调 LLM API 做更高质量的 L1/L0 压缩（规则压缩是 baseline，LLM 压缩是升级）
2. **Gateway 集成**：在 OpenClaw gateway 的消息处理管线中插入 `recall_memory` 调用，实现真正的被动注入

---

## 阶段二详细计划：从图书馆升级成记忆

**前置条件**：阶段一完成，pipeline 能跑通，有可用的 L1 向量库。

### Task 8: LLM 驱动的高质量压缩

**目标**：用 LLM 替代规则压缩，产出更高质量的 L1/L0 片段。

**Files:**
- Modify: `src/compressor.py` — 新增 `compress_session_to_l1_llm()` 和 `compress_l1_to_l0_llm()`
- Create: `src/llm_client.py` — 封装 LLM API 调用（通过 OpenClaw 的 relay 或直接调 Claude）
- Modify: `scripts/run_pipeline.py` — 加 `--llm` flag 切换压缩模式

**设计要点**：
- LLM 压缩的 prompt 核心指令：提取意图、动作、结果、关键术语，保持术语一致性
- L1 的 cue 字段改由 LLM 提取（比正则更准确）
- L0 由 LLM 从所有 L1 中做跨片段压缩（不只是取第一条）
- 保留规则压缩作为 fallback（LLM 不可用时降级）

### Task 9: 上下文门控

**目标**：匹配结果不再等权注入，用当前上下文对每条结果做相关度加权。

**Files:**
- Modify: `src/matcher.py` — 新增 `gated_match()` 方法
- Create: `tests/test_gating.py`

**设计要点**：
- 门控公式参考 Engram：`α = σ(RMSNorm(h)ᵀ RMSNorm(k) / √d)`
- 简化版本：用 query 向量和每条结果向量的 cosine 做 sigmoid 门控
- 低于阈值（如 0.3）的结果直接丢弃，不注入
- `format_for_injection()` 按门控值排序，高相关度的排前面

### Task 10: 频率统计与 Zipfian 分层

**目标**：追踪每条记忆片段被命中的频率，高频片段自动提升到 L0。

**Files:**
- Create: `src/frequency.py` — 频率追踪器
- Modify: `src/store.py` — 每次 query 后更新命中计数
- Create: `scripts/promote_hot.py` — 把高频 L1 提升到 L0 的脚本

**设计要点**：
- 每条片段记录：命中次数、最近命中时间、命中上下文摘要
- 定期（或每 N 次查询后）扫描：命中次数 top-K 的 L1 片段提升到 L0
- L0 有容量上限（如 20 条），超出时按命中频率淘汰最低的
- 这就是 Zipfian 分布的自然结果：少数高频片段驻留 L0，大量长尾留在 L1

### Task 11: Thinking cue 集成

**目标**：用上一轮模型 thinking 内容作为当前轮的深层记忆激活线索。

**Files:**
- Create: `src/thinking_cue.py` — 从 thinking 内容提取 cue
- Modify: `src/matcher.py` — 支持多来源 cue 合并匹配

**设计要点**：
- 从 OpenClaw session 的 assistant message 中提取 thinking 内容（如果模型支持 thinking 且有输出）
- 将 thinking 中的关键术语作为额外 cue，和用户消息的 cue 合并
- 两组 cue 的匹配结果做加权合并：用户消息权重 0.6，thinking 权重 0.4
- 如果上一轮没有 thinking（如 thinkingLevel=off），退化为纯消息匹配

### Task 12: Gateway 中间件集成

**目标**：在 OpenClaw gateway 的消息处理管线中插入前意识激活层，实现真正的被动注入。

**Files:**
- 需要修改 OpenClaw 源码（`/home/fzhiyu/projects/openclaw-dev`）
- 或通过 OpenClaw 的插件系统注入

**设计要点**：
- 在用户消息到达 LLM 之前：提取 cue → 匹配记忆 → 注入到 system prompt
- 注入格式参考阶段一的 `format_for_injection()`
- 注入内容占上下文预算不超过 20-25%（参考 Engram 的分配法则）
- 需要确认 OpenClaw 的插件/中间件机制支持这种注入

### 阶段二完成标准

- [ ] LLM 精压产出的 L1/L0 片段质量明显优于规则压缩
- [ ] 门控过滤掉低相关度结果，注入质量提升
- [ ] 频率统计运行，高频片段自动驻留 L0
- [ ] thinking cue 能产生比纯消息匹配更精准的结果
- [ ] Gateway 中间件能在真实对话中被动注入记忆

---

## 阶段三详细计划：加在线学习

**前置条件**：阶段二完成，gateway 集成运行，有实际使用数据积累。

### Task 13: DGD 更新规则实现

**目标**：实现 Delta Gradient Descent 作为关联记忆模块的在线更新机制。

**Files:**
- Create: `src/dgd.py` — DGD 更新规则
- Create: `tests/test_dgd.py`

**设计要点**：
- 核心公式：`M_new = M_old * (α*I - η * k @ k^T) - η * error @ k^T`
- M 是一个 `[embed_dim, embed_dim]` 的权重矩阵（qwen3-embedding dim=4096，可能需要降维到 256-512 先做原型）
- key = 当前上下文的 embedding（qwen3-embedding 编码）
- value = 被激活记忆片段的 embedding
- 遗忘项 `(α*I - η * k @ k^T)` 自动衰减旧映射
- 先在合成数据上验证 DGD 的记忆/遗忘行为正确

### Task 14: DGD 替换 cosine 匹配

**目标**：用 DGD 关联记忆做 L1 层的激活，替换 brute-force cosine。

**Files:**
- Create: `src/associative_memory.py` — 封装 DGD 关联记忆模块
- Modify: `src/matcher.py` — 新增 `AssociativeMemoryMatcher`
- Create: `tests/test_associative_memory.py`

**设计要点**：
- 初始化：用所有 L1 片段的 (cue_embedding, fragment_embedding) 对训练初始 M
- 在线更新：每次查询后，根据哪些记忆被实际使用（用户反馈或后续对话确认），更新 M
- 查询：`M @ query_embedding → 激活向量`，然后在 L1 片段中找最近邻
- 保留 cosine 作为 L2 层的 fallback

### Task 15: 训练对数据收集

**目标**：从实际运行中自动收集 (context → activated_memory → was_useful) 三元组。

**Files:**
- Create: `src/collector.py` — 数据收集器
- Create: `data/training/` — 训练数据目录

**设计要点**：
- 每次记忆注入后，记录：查询上下文、注入了哪些片段、注入后的对话走向
- "有用"的信号：注入后对话继续深入同一话题（正反馈）
- "无用"的信号：模型忽略注入内容、或用户换了话题（弱负反馈）
- 数据格式：`{context_embedding, memory_id, score, was_useful, timestamp}`
- 这些数据为阶段四的 CMS 训练和 DGD 调参提供基础

### 阶段三完成标准

- [ ] DGD 在合成数据上验证通过（能记忆、能遗忘、能泛化）
- [ ] DGD 替换 cosine 后，匹配准确率不降（至少持平）
- [ ] 经过一段使用后，DGD 的匹配质量比初始状态提升（越用越准）
- [ ] 训练对数据持续积累

---

## 阶段四详细计划：加多时间尺度

**前置条件**：阶段三完成，DGD 在线学习运行稳定，有足够的训练对数据。

### Task 16: CMS 多频率层实现

**目标**：实现连续记忆系统（CMS），让不同频率的 DGD 层捕捉不同时间尺度的记忆模式。

**Files:**
- Create: `src/cms.py` — CMS 多频率层
- Create: `tests/test_cms.py`

**设计要点**：
- 三层 CMS 架构：
  - 高频层（每轮对话更新）：捕捉"这次对话在聊什么"
  - 中频层（每天/每 N 次对话更新）：捕捉"最近的工作重心"
  - 低频层（每周/每 M 次更新）：捕捉"这个用户的长期模式"
- 每层是一个独立的 DGD 关联记忆模块
- 查询时：三层结果按频率加权合并
- 更新调度器：根据步数决定哪一层在本轮更新
- 灾难性遗忘通过分层自然解决：高频层快速适应也快速遗忘，低频层保持长期稳定

### Task 17: CMS 与现有管线集成

**目标**：把 CMS 替换为 matcher 的核心匹配引擎。

**Files:**
- Modify: `src/matcher.py` — 新增 `CMSMatcher`
- Modify: `src/tool.py` — 切换到 CMS matcher
- Create: `scripts/train_cms.py` — 用收集的训练对数据初始化 CMS

**设计要点**：
- 查询流程：`query → CMS 三层并行查询 → 加权合并 → 门控过滤 → 格式化注入`
- CMS 状态持久化：每层的权重矩阵 M 保存为 numpy/pickle 文件
- 热启动：用阶段三收集的训练对数据初始化 CMS 各层
- 冷启动降级：如果 CMS 未初始化，退化为 cosine 匹配

### Task 18: 完整系统验证

**目标**：端到端验证 CMS 记忆系统在真实场景中的效果。

**设计要点**：
- 对比实验：cosine baseline vs DGD 单层 vs CMS 三层
- 指标：匹配准确率、注入后对话质量（人工评估）、latency
- 长时间运行测试：跑一周，观察 CMS 各层的行为是否符合预期
  - 高频层：每天的热点话题变化
  - 中频层：一周内的工作重心变化
  - 低频层：稳定的长期模式（如用户的运维习惯）

### 阶段四完成标准

- [ ] CMS 三层在合成数据上验证通过
- [ ] CMS 集成到完整管线，端到端能跑
- [ ] 对比实验显示 CMS > DGD 单层 > cosine baseline
- [ ] 长时间运行表现稳定，各层行为符合预期

---

## 整体架构演进总结

```
阶段一（图书馆）         阶段二（智能图书馆）       阶段三（会学习）         阶段四（多时间尺度）

session → 规则压缩      session → LLM 精压        同左                  同左
  → L1/L0 片段           → 高质量 L1/L0

cosine 匹配             cosine + 门控过滤         DGD 关联记忆          CMS 三层
  → 等权注入              → 加权注入                → 在线学习            → 短/中/长期
                         + thinking cue             + 训练对收集          + 分频更新
                         + 频率统计
                         + gateway 注入

检索: O(n)              检索: O(n) + 过滤         检索: O(1) 矩阵乘     检索: O(1) × 3 层
学习: 不学              学习: 不学                学习: 在线 DGD         学习: 分频 DGD
遗忘: 不忘              遗忘: 频率淘汰            遗忘: DGD 衰减项       遗忘: 分层衰减
```
