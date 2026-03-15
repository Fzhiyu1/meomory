"""向量存储与检索 — 纯 Python brute-force cosine 实现。

数据量 <1000，无需 FAISS；用 JSON 序列化做持久化。
"""

import json
import math
from pathlib import Path


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """内存向量存储，支持 cosine 相似度检索和 JSON 持久化。"""

    def __init__(self) -> None:
        self._entries: list[dict] = []  # [{id, vector, meta}]

    def add(self, id: str, vector: list[float], meta: dict) -> None:
        """添加一条向量记录。"""
        self._entries.append({"id": id, "vector": vector, "meta": meta})

    def query(self, vector: list[float], top_k: int = 5) -> list[dict]:
        """按 cosine similarity 降序返回 [{id, score, meta}]。"""
        scored = []
        for entry in self._entries:
            score = _cosine_similarity(vector, entry["vector"])
            scored.append({"id": entry["id"], "score": score, "meta": entry["meta"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def save(self, path: Path) -> None:
        """将存储内容序列化为 JSON 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """从 JSON 文件反序列化恢复存储。"""
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            store._entries = json.load(f)
        return store

    def __len__(self) -> int:
        return len(self._entries)
