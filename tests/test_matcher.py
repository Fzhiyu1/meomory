"""MemoryMatcher 单元测试（纯本地，不调 embedding API，用固定向量）"""

from src.store import VectorStore
from src.matcher import MemoryMatcher


def test_matcher_match_returns_fragments():
    """添加 2 个片段到 store，用 query_by_vector 查询，验证返回最相关的。"""
    store = VectorStore()

    # 片段 1：与查询向量高度相似
    store.add("m1", [1.0, 0.0, 0.0], {"body": "今天和小明吃了火锅", "cue": ["火锅", "小明"]})
    # 片段 2：与查询向量正交
    store.add("m2", [0.0, 0.0, 1.0], {"body": "读完了三体第三部", "cue": ["三体"]})

    matcher = MemoryMatcher(store=store)

    # 查询向量接近片段 1
    results = matcher.query_by_vector([0.9, 0.1, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0]["id"] == "m1"
    assert results[0]["meta"]["body"] == "今天和小明吃了火锅"
    assert results[0]["score"] > results[1]["score"]


def test_matcher_format_for_injection():
    """验证格式化输出包含正确内容，长度 < 2000。"""
    store = VectorStore()
    store.add("m1", [1.0, 0.0, 0.0], {"body": "今天和小明吃了火锅", "cue": ["火锅", "小明"]})
    store.add("m2", [0.0, 1.0, 0.0], {"body": "读完了三体第三部", "cue": ["三体"]})

    matcher = MemoryMatcher(store=store)
    results = matcher.query_by_vector([0.8, 0.6, 0.0], top_k=2)

    formatted = matcher.format_for_injection(results)

    assert formatted.startswith("[相关记忆]")
    assert "火锅" in formatted
    assert "小明" in formatted
    assert "三体" in formatted
    assert "今天和小明吃了火锅" in formatted
    assert "读完了三体第三部" in formatted
    assert len(formatted) < 2000


def test_matcher_format_empty():
    """空结果应返回空字符串。"""
    matcher = MemoryMatcher()
    assert matcher.format_for_injection([]) == ""


def test_matcher_format_respects_max_chars():
    """超长内容应被 max_chars 截断。"""
    # 构造一个结果列表，每条内容很长
    results = [
        {"id": f"m{i}", "score": 0.9 - i * 0.1, "meta": {"body": "A" * 200, "cue": [f"cue{i}"]}}
        for i in range(20)
    ]
    matcher = MemoryMatcher()
    formatted = matcher.format_for_injection(results, max_chars=500)

    # 不算首行 "[相关记忆]"，正文总长度应不超过 500
    lines = formatted.split("\n")
    assert lines[0] == "[相关记忆]"
    body_text = "\n".join(lines[1:])
    assert len(body_text) <= 500 + len(lines) - 1  # 容许换行符
    # 不应包含全部 20 条
    assert lines.count("- ") < 20  # 不到 20 行条目
