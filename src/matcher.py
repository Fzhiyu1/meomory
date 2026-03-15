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
        """给定文本，返回最匹配的记忆片段（调用 embedding API）"""
        vector = get_embedding(text)
        return self.query_by_vector(vector, top_k)

    def query_by_vector(self, vector: list[float], top_k: int = 5) -> list[dict]:
        """给定向量，返回最匹配的记忆片段（不调 API，用于测试）"""
        return self.store.query(vector, top_k)

    def format_for_injection(self, results: list[dict], max_chars: int = 1500) -> str:
        """将匹配结果格式化为可注入上下文的文本。

        格式：
        [相关记忆]
        - [cue1, cue2] body内容
        - [cue3] body内容

        总字符数不超过 max_chars。
        """
        if not results:
            return ""
        lines = ["[相关记忆]"]
        total = 0
        for r in results:
            meta = r.get("meta", r)
            body = meta.get("body", "")
            cue = ", ".join(meta.get("cue", []))
            entry = f"- [{cue}] {body}"
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)
        return "\n".join(lines)
