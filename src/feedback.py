"""自动反馈层：观察 agent 回复，推断哪些注入的记忆被"用上了"，更新 DGD。

agent 完全无感。它不知道有记忆被注入，也不知道自己的回复在被观察。
"""
import json
import math
from pathlib import Path

from src.embedder import get_embedding
from src.projection import project
from src.dgd import AssociativeMemory

# 回复和记忆的 cosine 高于此值 → 判定为"被用上了"
USE_THRESHOLD = 0.50


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na * nb > 0 else 0


def auto_feedback(
    query_text: str,
    response_text: str,
    injected_memories: list[dict],
    mem: AssociativeMemory,
    proj_matrix: list[list[float]],
) -> list[dict]:
    """一次完整的自动反馈循环。

    Args:
        query_text: 用户原始查询
        response_text: agent 的回复
        injected_memories: 注入的记忆列表，每条需包含 "proj_vec" 字段
        mem: DGD 关联记忆实例
        proj_matrix: 投影矩阵

    Returns:
        反馈结果列表 [{"title", "sim", "used", "updated"}]
    """
    # embed + project query 和 response（截断回复到 500 字符避免超时）
    q_vec = get_embedding(query_text)
    q_proj = _normalize(project(q_vec, proj_matrix))

    r_vec = get_embedding(response_text[:500])
    r_proj = _normalize(project(r_vec, proj_matrix))

    results = []
    for memory in injected_memories:
        mem_proj = memory["proj_vec"]
        sim = _cosine(r_proj, mem_proj)
        used = sim >= USE_THRESHOLD

        if used:
            mem.update(q_proj, mem_proj)

        results.append({
            "title": memory.get("title", "?"),
            "sim": round(sim, 3),
            "used": used,
            "updated": used,
        })

    return results
