"""DGD 交互式记忆工具

用法：
  python3 -m src.tool_dgd                # 进入交互模式
  python3 -m src.tool_dgd "查询内容"      # 单次查询（无反馈）

交互模式下：
  输入查询 → 看结果 → 输入有用的编号（如 1 或 1,3）→ DGD 学习 → 下一轮
  输入 q 退出并保存
"""
import json
import math
import sys
from pathlib import Path

from src.store import VectorStore
from src.embedder import get_embedding
from src.projection import project
from src.dgd import AssociativeMemory

ROOT = Path(__file__).parent.parent
VECTOR_PATH = ROOT / "memory" / "vectors" / "l1.json"
DGD_PATH = ROOT / "memory" / "vectors" / "dgd.json"
PROJ_PATH = ROOT / "memory" / "vectors" / "proj.json"

THRESHOLD = 0.30  # DGD 空间的阈值（投影后分数偏低）


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na * nb > 0 else 0


class DGDMatcher:
    def __init__(self):
        self.store = VectorStore.load(VECTOR_PATH)
        self.mem = AssociativeMemory.load(DGD_PATH)
        with open(PROJ_PATH) as f:
            self.proj = json.load(f)

        # 预投影所有文档向量
        self.entries = self.store._entries
        self.proj_vectors = [
            _normalize(project(e["vector"], self.proj))
            for e in self.entries
        ]

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        """查询并返回 DGD 匹配结果。"""
        q_vec = get_embedding(text)
        q_proj = _normalize(project(q_vec, self.proj))

        # DGD: M @ query → activation
        activation = self.mem.query(q_proj)

        # 在文档向量中找最接近 activation 的
        scored = []
        for i, pv in enumerate(self.proj_vectors):
            sim = _cosine(activation, pv)
            if sim >= THRESHOLD:
                meta = self.entries[i]["meta"]
                scored.append({
                    "rank": 0,
                    "score": sim,
                    "index": i,
                    "title": meta.get("title", self.entries[i]["id"]),
                    "category": meta.get("category", ""),
                    "body": meta.get("body", "")[:150],
                    "proj_vec": pv,
                })
        scored.sort(key=lambda x: x["score"], reverse=True)

        for rank, item in enumerate(scored[:top_k]):
            item["rank"] = rank + 1

        return scored[:top_k]

    def feedback(self, q_proj: list[float], useful_indices: list[int]):
        """用户反馈：指定哪些结果有用，DGD 在线更新。"""
        for idx in useful_indices:
            target = self.proj_vectors[idx]
            self.mem.update(q_proj, target)

    def save(self):
        self.mem.save(DGD_PATH)


def single_query(text: str):
    """单次查询模式。"""
    matcher = DGDMatcher()
    results = matcher.query(text, top_k=5)

    if not results:
        print("[无相关记忆]")
        return

    print("[DGD 记忆匹配]")
    for r in results:
        print(f"  {r['rank']}. [{r['score']:.3f}] [{r['category']}] {r['title']}")
        print(f"     {r['body'][:100]}")
    print()


def interactive():
    """交互模式：查询 → 反馈 → 学习。"""
    print("=== DGD 交互式记忆工具 ===")
    print("输入查询内容，看到结果后输入有用的编号（如 1 或 1,3）")
    print("输入 q 退出并保存，输入 n 跳过反馈\n")

    matcher = DGDMatcher()
    query_count = 0
    update_count = 0

    while True:
        try:
            text = input("查询> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text or text.lower() == "q":
            break

        # 查询
        q_vec = get_embedding(text)
        q_proj = _normalize(project(q_vec, matcher.proj))
        results = matcher.query(text, top_k=5)
        query_count += 1

        if not results:
            print("  [无相关记忆]\n")
            continue

        print()
        for r in results:
            print(f"  {r['rank']}. [{r['score']:.3f}] {r['title']}")
            print(f"     {r['body'][:80]}")
        print()

        # 反馈
        try:
            fb = input("有用的编号(如 1,3) / n跳过 / q退出> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if fb.lower() == "q":
            break
        if fb.lower() == "n" or not fb:
            print()
            continue

        # 解析编号
        try:
            ranks = [int(x.strip()) for x in fb.split(",")]
            useful_indices = []
            for rank in ranks:
                for r in results:
                    if r["rank"] == rank:
                        useful_indices.append(r["index"])
            if useful_indices:
                matcher.feedback(q_proj, useful_indices)
                update_count += len(useful_indices)
                print(f"  ✓ DGD 已更新（强化了 {len(useful_indices)} 条关联）\n")
            else:
                print("  未找到对应编号\n")
        except ValueError:
            print("  无法解析编号，跳过\n")

    # 保存
    if update_count > 0:
        matcher.save()
        print(f"\n已保存。本次 {query_count} 次查询，{update_count} 次 DGD 更新。")
    else:
        print(f"\n本次 {query_count} 次查询，无更新。")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        single_query(" ".join(sys.argv[1:]))
    else:
        interactive()
