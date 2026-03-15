"""记忆匹配查询接口

给定一段文本（用户消息或 thinking 片段），
返回最相关的记忆片段，并格式化为可注入上下文的文本。
"""

from src.store import VectorStore
from src.embedder import get_embedding

# 低于此阈值的结果不返回
DEFAULT_THRESHOLD = 0.40


class MemoryMatcher:
    def __init__(self, store: VectorStore | None = None, threshold: float = DEFAULT_THRESHOLD):
        self.store = store or VectorStore()
        self.threshold = threshold

    def match(self, text: str, top_k: int = 5) -> list[dict]:
        """给定文本，返回最匹配的记忆片段（调用 embedding API）"""
        vector = get_embedding(text)
        return self.query_by_vector(vector, top_k)

    def query_by_vector(self, vector: list[float], top_k: int = 5) -> list[dict]:
        """给定向量，返回最匹配的记忆片段（带门控过滤）"""
        raw = self.store.query(vector, top_k)
        # 门控：低于阈值的不返回
        return [r for r in raw if r["score"] >= self.threshold]

    def format_for_injection(self, results: list[dict], max_chars: int = 1500) -> str:
        """将匹配结果格式化为可注入上下文的文本。"""
        if not results:
            return "[无相关记忆]"

        lines = ["[相关记忆]"]
        total = 0
        for r in results:
            meta = r.get("meta", {})
            title = meta.get("title", r.get("id", ""))
            body = meta.get("body", "")
            cues = meta.get("cue", [])
            score = r.get("score", 0)
            category = meta.get("category", "")

            # 格式：标题 + 分数 + 正文摘要
            header = f"- **{title}** ({score:.2f})"
            if category:
                header += f" [{category}]"
            # body 截断到 200 字符
            body_short = body[:200] + "..." if len(body) > 200 else body
            entry = f"{header}\n  {body_short}"

            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)

        return "\n".join(lines)
