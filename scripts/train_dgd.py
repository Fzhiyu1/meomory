#!/usr/bin/env python3
"""在知识库数据上训练 DGD 关联记忆，并与 cosine baseline 对比。"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.store import VectorStore
from src.projection import create_projection_matrix, project
from src.dgd import AssociativeMemory
from src.embedder import get_embedding

ROOT = Path(__file__).parent.parent
VECTOR_PATH = ROOT / "memory" / "vectors" / "l1.json"
DGD_PATH = ROOT / "memory" / "vectors" / "dgd.json"
PROJ_PATH = ROOT / "memory" / "vectors" / "proj.json"

DIM = 256


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na * nb > 0 else 0


def main():
    # 加载向量库
    print("=== Loading vector store ===")
    store = VectorStore.load(VECTOR_PATH)
    entries = store._entries
    print(f"  {len(entries)} documents")

    # 提取原始向量和 meta
    raw_vectors = [e["vector"] for e in entries]
    metas = [e["meta"] for e in entries]
    ids = [e["id"] for e in entries]

    # 创建投影矩阵
    print(f"\n=== Creating projection matrix (4096 → {DIM}) ===")
    proj = create_projection_matrix(input_dim=4096, output_dim=DIM, seed=42)

    # 投影所有向量
    print("  Projecting vectors...")
    proj_vectors = [project(v, proj) for v in raw_vectors]

    # 归一化
    def normalize(v):
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n > 0 else v

    proj_vectors = [normalize(v) for v in proj_vectors]

    # 保存投影矩阵
    with open(PROJ_PATH, "w") as f:
        json.dump(proj, f)
    print(f"  Saved projection matrix to {PROJ_PATH}")

    # 训练 DGD
    print(f"\n=== Training DGD (dim={DIM}, {len(proj_vectors)} pairs) ===")
    mem = AssociativeMemory(dim=DIM, alpha=1.0, eta=0.01)

    # 用文档向量同时作为 key 和 value（自关联记忆）
    # 这样 M @ doc_vec ≈ doc_vec，DGD 学的是"哪些方向重要"
    mem.train(proj_vectors, proj_vectors, epochs=20)
    mem.save(DGD_PATH)
    print(f"  Saved DGD model to {DGD_PATH}")

    # 对比测试
    print("\n=== Comparison: cosine vs DGD ===")
    test_queries = [
        "记忆系统怎么设计",
        "遗忘是怎么回事",
        "Hope 模型怎么用",
        "自由能原理",
        "DeepSeek Engram",
        "什么是可行动状态",
        "知识库的作用是什么",
        "搅屎棍",
    ]

    for q in test_queries:
        q_vec = get_embedding(q)
        q_proj = normalize(project(q_vec, proj))

        # Cosine baseline（在投影空间）
        cosine_scores = []
        for i, pv in enumerate(proj_vectors):
            sim = cosine(q_proj, pv)
            cosine_scores.append((sim, i))
        cosine_scores.sort(reverse=True)

        # DGD
        activation = mem.query(q_proj)
        dgd_scores = []
        for i, pv in enumerate(proj_vectors):
            sim = cosine(activation, pv)
            dgd_scores.append((sim, i))
        dgd_scores.sort(reverse=True)

        print(f"\n  Query: '{q}'")
        print(f"  {'Cosine Top-3':<50} {'DGD Top-3'}")
        print(f"  {'-'*50} {'-'*50}")
        for rank in range(3):
            c_score, c_idx = cosine_scores[rank]
            d_score, d_idx = dgd_scores[rank]
            c_title = metas[c_idx].get("title", ids[c_idx])[:25]
            d_title = metas[d_idx].get("title", ids[d_idx])[:25]
            print(f"  [{c_score:.3f}] {c_title:<43} [{d_score:.3f}] {d_title}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
