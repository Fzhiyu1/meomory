"""Mem0 DGD Reranker: 将进化出的关联记忆作为 Mem0 的 reranker 插件。

用法:
    from mem0 import Memory
    config = {
        "reranker": {
            "provider": "dgd",
            "config": {
                "dim": 256,
                "alpha": 1.0,
                "eta": 0.01,
                "state_path": "dgd_state.json",  # 持久化学习状态
            }
        }
    }
    m = Memory.from_config(config)
"""
import json
import math
from pathlib import Path
from typing import List, Dict, Any


class EvolvedAssociativeMemory:
    """进化出的关联记忆（gen005-island3-001, P@1=72.4%/500q, 42%/1976q）。

    核心创新：零初始化 + Hebbian 强化 + 动量。
    由 9B 模型通过 FunSearch 群岛进化自动发现。
    """

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

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"M": self.M, "vel_M": self.vel_M, "dim": self.dim,
                       "alpha": self.alpha, "eta": self.eta, "momentum": self.momentum}, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.M = data["M"]
        self.vel_M = data["vel_M"]


def _get_embedding(text, embedder=None):
    """获取文本 embedding。如果有 embedder 用它，否则用简单 hash。"""
    if embedder:
        return embedder.embed(text)
    # Fallback: 简单字符级 hash embedding（仅用于测试）
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    dim = 256
    vec = [(b / 128.0) - 1.0 for b in h * (dim // len(h) + 1)][:dim]
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n > 0 else vec


class DGDReranker:
    """Mem0 compatible DGD reranker。

    实现 BaseReranker 接口：rerank(query, documents, top_k) -> documents。
    内部使用进化出的 AssociativeMemory 进行在线学习 reranking。
    """

    def __init__(self, config=None):
        config = config or {}
        self.dim = config.get("dim", 256)
        self.alpha = config.get("alpha", 1.0)
        self.eta = config.get("eta", 0.01)
        self.state_path = config.get("state_path", None)

        self.memory = EvolvedAssociativeMemory(
            dim=self.dim, alpha=self.alpha, eta=self.eta,
        )

        if self.state_path and Path(self.state_path).exists():
            self.memory.load(self.state_path)

        self._embedder = None

    def set_embedder(self, embedder):
        """设置 embedding 函数（复用 Mem0 的 embedder）。"""
        self._embedder = embedder

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """Rerank documents using evolved associative memory.

        Args:
            query: 搜索查询文本
            documents: Mem0 返回的文档列表，每个含 'memory' 字段
            top_k: 返回前 K 个

        Returns:
            重排序的文档列表，添加 'rerank_score' 字段
        """
        if not documents:
            return documents

        # 获取 query embedding
        q_vec = _get_embedding(query, self._embedder)

        # 投影到 DGD 维度（如果 embedding dim != self.dim）
        if len(q_vec) != self.dim:
            q_vec = self._project(q_vec)

        # DGD query: 通过关联记忆变换
        activation = self.memory.query(q_vec)

        # 对每个文档计算 rerank score
        for doc in documents:
            text = doc.get("memory", str(doc))
            doc_vec = _get_embedding(text, self._embedder)
            if len(doc_vec) != self.dim:
                doc_vec = self._project(doc_vec)

            # cosine(activation, doc_vec) 作为 rerank score
            score = self._cosine(activation, doc_vec)
            doc["rerank_score"] = score

        # 按 rerank_score 降序排列
        documents.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)

        if top_k:
            documents = documents[:top_k]

        return documents

    def feedback(self, query: str, relevant_doc: str):
        """用户反馈：告诉 DGD 哪个文档是正确的，在线更新。

        这是 DGD 的核心价值——每次反馈让 reranking 更准。
        """
        q_vec = _get_embedding(query, self._embedder)
        target_vec = _get_embedding(relevant_doc, self._embedder)

        if len(q_vec) != self.dim:
            q_vec = self._project(q_vec)
        if len(target_vec) != self.dim:
            target_vec = self._project(target_vec)

        self.memory.update(q_vec, target_vec)

        # 持久化
        if self.state_path:
            self.memory.save(self.state_path)

    def _project(self, vec):
        """简单投影：截断或 padding 到 self.dim。"""
        if len(vec) >= self.dim:
            v = vec[:self.dim]
        else:
            v = vec + [0.0] * (self.dim - len(vec))
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n > 0 else v

    @staticmethod
    def _cosine(a, b):
        d = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return d / (na * nb) if na * nb > 0 else 0
