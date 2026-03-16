"""巩固模块：离线压缩合并，输入层 → 概念层。

遗忘在这里产生——不是靠 TTL 删除，是靠压缩中细节的丢失。
"""
import json
import time
from pathlib import Path

from src.store import VectorStore
from src.embedder import get_embedding


async def consolidate(store: VectorStore, llm_chat, min_group_size: int = 2, similarity_threshold: float = 0.6):
    """扫描输入层片段，找相关的合并，生成概念层片段。

    Args:
        store: 向量库
        llm_chat: async LLM 调用函数
        min_group_size: 至少多少条相关片段才触发合并
        similarity_threshold: cosine 相似度阈值，高于此值认为"相关"

    Returns:
        consolidation_report: {merged: int, new_concepts: int, details: [...]}
    """
    import math

    def cosine(a, b):
        d = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return d / (na * nb) if na * nb > 0 else 0

    # Step 1: 找出所有输入层片段
    input_entries = []
    for i, entry in enumerate(store._entries):
        meta = entry.get("meta", {})
        if meta.get("layer") == "input" and not meta.get("consolidated"):
            input_entries.append((i, entry))

    if len(input_entries) < min_group_size:
        return {"merged": 0, "new_concepts": 0, "details": [], "msg": "not enough input entries"}

    # Step 2: 找相关组（简单贪心聚类）
    used = set()
    groups = []

    for idx_a, (pos_a, entry_a) in enumerate(input_entries):
        if idx_a in used:
            continue
        group = [(pos_a, entry_a)]
        used.add(idx_a)

        for idx_b, (pos_b, entry_b) in enumerate(input_entries):
            if idx_b in used:
                continue
            sim = cosine(entry_a["vector"], entry_b["vector"])
            if sim >= similarity_threshold:
                group.append((pos_b, entry_b))
                used.add(idx_b)

        if len(group) >= min_group_size:
            groups.append(group)

    if not groups:
        return {"merged": 0, "new_concepts": 0, "details": [], "msg": "no similar groups found"}

    # Step 3: 用 LLM 压缩每组
    report = {"merged": 0, "new_concepts": 0, "details": []}

    for group in groups:
        bodies = [e["meta"].get("body", "") for _, e in group]
        combined = "\n---\n".join(bodies)

        prompt = f"""以下是多条相关的记忆碎片，请去重合并。

要求：
1. 去掉重复的信息，保留所有不重复的具体细节（人名、数字、事件、专有名词）
2. 不要概括或抽象——"Howard 用 common field cricket 打赌"比"Howard 参与了一次打赌"好
3. 如果有矛盾，保留转变过程和原因
4. 合并后不超过 500 字
5. 只输出合并后的内容，不要其他

记忆碎片：
{combined}"""

        try:
            compressed = await llm_chat(prompt, max_tokens=300)
        except Exception as e:
            report["details"].append({"error": str(e), "group_size": len(group)})
            continue

        # 创建概念层片段
        concept_id = f"concept-{int(time.time())}-{report['new_concepts']:04d}"
        vec = get_embedding(compressed[:500])

        source_ids = [e["id"] for _, e in group]
        store.add(concept_id, vec, {
            "body": compressed[:500],
            "source": "consolidation",
            "layer": "concept",
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "compressed_from": source_ids,
        })

        # 标记原始片段为"已压缩"
        for pos, entry in group:
            entry["meta"]["consolidated"] = True
            entry["meta"]["consolidated_into"] = concept_id

        report["merged"] += len(group)
        report["new_concepts"] += 1
        report["details"].append({
            "concept_id": concept_id,
            "sources": source_ids,
            "compressed": compressed[:200],
        })

    return report
