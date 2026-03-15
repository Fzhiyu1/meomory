"""VectorStore 单元测试（纯本地，无网络依赖）"""
import math
from pathlib import Path

from src.store import VectorStore, _cosine_similarity


def test_store_add_and_query():
    """添加 3 个向量，查询返回按 cosine similarity 降序排列。"""
    store = VectorStore()

    # 构造 3 个简单向量
    v_query = [1.0, 0.0, 0.0]
    v_close = [0.9, 0.1, 0.0]  # 与 query 最接近
    v_mid = [0.5, 0.5, 0.0]
    v_far = [0.0, 0.0, 1.0]  # 与 query 正交

    store.add("close", v_close, {"label": "close"})
    store.add("mid", v_mid, {"label": "mid"})
    store.add("far", v_far, {"label": "far"})

    assert len(store) == 3

    results = store.query(v_query, top_k=3)
    assert len(results) == 3
    assert results[0]["id"] == "close"
    assert results[1]["id"] == "mid"
    assert results[2]["id"] == "far"

    # score 应该递减
    assert results[0]["score"] > results[1]["score"] > results[2]["score"]
    # far 向量与 query 正交，cosine 应该接近 0
    assert abs(results[2]["score"]) < 1e-9

    # top_k 限制
    results_top1 = store.query(v_query, top_k=1)
    assert len(results_top1) == 1
    assert results_top1[0]["id"] == "close"


def test_store_save_and_load(tmp_path: Path):
    """保存到临时文件再加载，查询结果应一致。"""
    store = VectorStore()
    store.add("a", [1.0, 0.0], {"tag": "alpha"})
    store.add("b", [0.0, 1.0], {"tag": "beta"})
    store.add("c", [0.7, 0.7], {"tag": "gamma"})

    save_path = tmp_path / "store.json"
    store.save(save_path)
    assert save_path.exists()

    loaded = VectorStore.load(save_path)
    assert len(loaded) == 3

    q = [1.0, 0.0]
    orig_results = store.query(q, top_k=3)
    load_results = loaded.query(q, top_k=3)

    assert len(orig_results) == len(load_results)
    for o, l in zip(orig_results, load_results):
        assert o["id"] == l["id"]
        assert math.isclose(o["score"], l["score"], rel_tol=1e-9)
        assert o["meta"] == l["meta"]
