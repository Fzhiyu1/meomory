"""Embedder 集成测试（需要网络访问 Ollama 服务）"""
import pytest

from src.embedder import get_embedding, EMBED_DIM
from src.store import _cosine_similarity


@pytest.mark.network
def test_get_embedding_returns_vector():
    """检查返回向量长度 == EMBED_DIM，元素都是 float。"""
    vec = get_embedding("今天天气真好")
    assert len(vec) == EMBED_DIM
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.network
def test_get_embedding_different_texts():
    """两个语义不同的文本的 cosine similarity 应 < 0.95。"""
    vec_a = get_embedding("量子力学的基本原理")
    vec_b = get_embedding("今天中午吃了红烧肉")
    sim = _cosine_similarity(vec_a, vec_b)
    assert sim < 0.95, f"语义不同的文本 cosine 相似度过高: {sim}"
