#!/usr/bin/env python3
"""测试 Mem0 + DGD Reranker 集成。

在 LoCoMo 数据上：
1. 将对话片段存入 Mem0
2. 用 Mem0 检索 + 原生 rerank
3. 用 Mem0 检索 + DGD rerank（无反馈）
4. 用 Mem0 检索 + DGD rerank（有反馈，模拟在线学习）
对比三者的 P@1
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mem0 import Memory
from src.mem0_integration.dgd_reranker import DGDReranker
from src.bench.datasets import Dataset


def setup_mem0():
    """配置 Mem0 使用 DeepSeek LLM + Ollama embedding。"""
    config = {
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": "deepseek-chat",
                "api_key": "sk-4682199cd4bb4d69826a838cd318578c",
                "temperature": 0.0,
            }
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "qwen3-embedding",
                "ollama_base_url": "http://100.94.126.19:11434",
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "meomory_test",
                "host": "localhost",
                "port": 6333,
            }
        },
    }

    # 如果 qdrant 不可用，fallback 到内存模式
    try:
        m = Memory.from_config(config)
        return m
    except Exception:
        # 用默认的内存向量库
        config_simple = {
            "llm": {
                "provider": "deepseek",
                "config": {
                    "model": "deepseek-chat",
                    "api_key": "sk-4682199cd4bb4d69826a838cd318578c",
                    "temperature": 0.0,
                }
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "qwen3-embedding",
                    "ollama_base_url": "http://100.94.126.19:11434",
                }
            },
        }
        try:
            m = Memory.from_config(config_simple)
            return m
        except Exception as e:
            print(f"Mem0 init failed: {e}")
            return None


def test_without_mem0():
    """不用 Mem0，直接测试 DGD reranker 在 LoCoMo 数据上的效果。

    模拟场景：
    - embedding 检索返回 top-10 候选
    - DGD reranker 对 top-10 重排序
    - 评测 P@1
    """
    import math
    import numpy as np

    print("=== 测试 DGD Reranker（不依赖 Mem0 服务）===\n")

    ds = Dataset.load_locomo_full(0)  # 用第一段对话
    print(f"Dataset: {ds.name}, {len(ds.fragments)} fragments, {len(ds.questions)} questions\n")

    # 加载 embedding
    cache_file = Path("experiments/cache/LoCoMo-full-all_embed.json")
    with open(cache_file) as f:
        cached = json.load(f)

    # 只取第一段对话的 embedding
    n_frags = len(ds.fragments)
    n_qs = len(ds.questions)
    frag_vecs = cached["frag_vecs"][:n_frags]
    q_vecs = cached["q_vecs"][:n_qs]

    frag_np = np.array(frag_vecs, dtype=np.float32)
    frag_norms = np.linalg.norm(frag_np, axis=1, keepdims=True)
    frag_norms[frag_norms == 0] = 1
    frag_np = frag_np / frag_norms

    # 初始化 DGD reranker（使用真实 embedding，dim=4096）
    reranker = DGDReranker({"dim": 256})

    # 投影矩阵（4096 → 256）
    from src.projection import create_projection_matrix, project

    def _norm(v):
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n > 0 else v

    proj = create_projection_matrix(4096, 256, seed=42)
    proj_frags = [_norm(project(v, proj)) for v in frag_vecs]
    proj_qs = [_norm(project(v, proj)) for v in q_vecs]

    # === Test 1: 纯 cosine 检索 ===
    hits_cosine = 0
    for qi, q in enumerate(ds.questions):
        q_np = np.array(q_vecs[qi], dtype=np.float32)
        q_np = q_np / (np.linalg.norm(q_np) + 1e-8)
        scores = frag_np @ q_np
        top1 = int(np.argmax(scores))
        if top1 in q["answer_indices"]:
            hits_cosine += 1
    p1_cosine = hits_cosine / n_qs
    print(f"1. 纯 Cosine 检索:          P@1 = {p1_cosine:.1%}")

    # === Test 2: Cosine top-10 → DGD rerank（无反馈）===
    hits_dgd_cold = 0
    reranker_cold = DGDReranker({"dim": 256})
    for qi, q in enumerate(ds.questions):
        q_np = np.array(q_vecs[qi], dtype=np.float32)
        q_np = q_np / (np.linalg.norm(q_np) + 1e-8)
        scores = frag_np @ q_np
        top10_idx = np.argsort(scores)[-10:][::-1].tolist()

        # 构建 Mem0 风格的文档列表
        docs = [{"memory": ds.fragments[idx]["body"], "idx": idx} for idx in top10_idx]

        # 用投影后的向量做 DGD rerank
        # 手动设置 embedding 为投影向量（跳过 hash embedding）
        q_proj = proj_qs[qi]
        activation = reranker_cold.memory.query(q_proj)
        for doc in docs:
            doc_proj = proj_frags[doc["idx"]]
            score = sum(a * b for a, b in zip(activation, doc_proj))
            na = math.sqrt(sum(a * a for a in activation))
            nb = math.sqrt(sum(b * b for b in doc_proj))
            doc["rerank_score"] = score / (na * nb) if na * nb > 0 else 0
        docs.sort(key=lambda d: d["rerank_score"], reverse=True)

        if docs[0]["idx"] in q["answer_indices"]:
            hits_dgd_cold += 1
    p1_dgd_cold = hits_dgd_cold / n_qs
    print(f"2. Cosine→DGD rerank(冷启动): P@1 = {p1_dgd_cold:.1%}")

    # === Test 3: Cosine top-10 → DGD rerank（有反馈，多轮学习）===
    from src.funsearch.sandbox import compile_class
    from src.funsearch.specification import INITIAL_IMPLEMENTATION

    # 用进化出的算法
    evolved_code = '''
class AssociativeMemory:
    def __init__(self, dim, alpha=1.0, eta=0.01, momentum=0.9):
        self.dim = dim
        self.alpha = alpha
        self.eta = eta
        self.momentum = momentum
        self.M = [[0.0 for _ in range(dim)] for _ in range(dim)]
        self.vel_M = [[0.0 for _ in range(dim)] for _ in range(dim)]

    def query(self, key):
        dim = self.dim
        return [sum(self.M[i][j] * key[j] for j in range(dim)) for i in range(dim)]

    def update(self, key, target):
        dim = self.dim
        eta = self.eta
        mom = self.momentum
        activation = self.query(key)
        error = [activation[i] - target[i] for i in range(dim)]
        for i in range(dim):
            for j in range(dim):
                error_term = -eta * error[i] * key[j]
                hebb_term = eta * target[i] * key[j]
                force = error_term + hebb_term
                self.vel_M[i][j] = mom * self.vel_M[i][j] + force
                self.M[i][j] += self.vel_M[i][j]
'''
    cls = compile_class(evolved_code.strip())
    mem = cls(dim=256, alpha=1.0, eta=0.01)

    for n_rounds in [1, 3, 5, 10]:
        # 重新初始化
        mem = cls(dim=256, alpha=1.0, eta=0.01)

        # 训练 N 轮
        for r in range(n_rounds):
            for qi, q in enumerate(ds.questions):
                target_idx = q["answer_indices"][0]
                mem.update(proj_qs[qi], proj_frags[target_idx])

        # 评测
        hits = 0
        for qi, q in enumerate(ds.questions):
            q_proj = proj_qs[qi]
            activation = mem.query(q_proj)

            q_np2 = np.array(q_vecs[qi], dtype=np.float32)
            q_np2 = q_np2 / (np.linalg.norm(q_np2) + 1e-8)
            cosine_scores = frag_np @ q_np2
            top10_idx = np.argsort(cosine_scores)[-10:][::-1].tolist()

            best_score = -999
            best_idx = top10_idx[0]
            for idx in top10_idx:
                doc_proj = proj_frags[idx]
                score = sum(a * b for a, b in zip(activation, doc_proj))
                na = math.sqrt(sum(a * a for a in activation))
                nb = math.sqrt(sum(b * b for b in doc_proj))
                s = score / (na * nb) if na * nb > 0 else 0
                if s > best_score:
                    best_score = s
                    best_idx = idx

            if best_idx in q["answer_indices"]:
                hits += 1

        p1 = hits / n_qs
        print(f"3. Cosine→进化DGD({n_rounds:2d}轮反馈):  P@1 = {p1:.1%}")

    print()


if __name__ == "__main__":
    test_without_mem0()
