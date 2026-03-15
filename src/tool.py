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
