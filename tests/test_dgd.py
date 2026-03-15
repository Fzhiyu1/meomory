"""测试 DGD 关联记忆"""
from src.dgd import AssociativeMemory
from src.projection import create_projection_matrix, project
import math


def _normalize(v):
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v] if n > 0 else v


def _cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0


def test_identity_initialization():
    """初始 M 是单位矩阵，query 应该返回输入本身。"""
    mem = AssociativeMemory(dim=4)
    result = mem.query([1.0, 0.0, 0.0, 0.0])
    assert result == [1.0, 0.0, 0.0, 0.0]


def test_train_memorizes():
    """训练后应该能回忆 key-value 对。"""
    mem = AssociativeMemory(dim=4, alpha=1.0, eta=0.05)
    keys = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    values = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
    mem.train(keys, values, epochs=50)

    # 查询 key[0] 应该接近 value[0]
    result = mem.query([1, 0, 0, 0])
    sim = _cosine(result, [0, 0, 0, 1])
    assert sim > 0.8, f"Expected high similarity, got {sim}"


def test_forgetting():
    """新的关联应该逐渐覆盖旧的（遗忘）。"""
    mem = AssociativeMemory(dim=4, alpha=0.95, eta=0.1)
    key = [1, 0, 0, 0]
    old_value = [0, 1, 0, 0]
    new_value = [0, 0, 1, 0]

    # 先学旧关联
    for _ in range(30):
        mem.update(key, old_value)
    old_result = mem.query(key)
    old_sim = _cosine(old_result, old_value)

    # 再学新关联
    for _ in range(30):
        mem.update(key, new_value)
    new_result = mem.query(key)
    new_sim = _cosine(new_result, new_value)

    # 新关联应该比旧关联更强
    assert new_sim > _cosine(new_result, old_value), "New association should be stronger"


def test_generalization():
    """类似的 key 应该产出类似的 activation。"""
    mem = AssociativeMemory(dim=4, alpha=1.0, eta=0.05)
    mem.train([[1, 0, 0, 0]], [[0, 0, 0, 1]], epochs=30)

    # 相似的 key
    result = mem.query([0.9, 0.1, 0, 0])
    sim = _cosine(result, [0, 0, 0, 1])
    assert sim > 0.5, f"Expected generalization, got {sim}"


def test_projection_preserves_order():
    """投影后 cosine 排序应该基本保持。"""
    proj = create_projection_matrix(input_dim=16, output_dim=8, seed=42)

    a = [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]
    b = [0.9,0.1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]  # similar to a
    c = [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0]  # different from a

    pa, pb, pc = project(a, proj), project(b, proj), project(c, proj)

    sim_ab = _cosine(pa, pb)
    sim_ac = _cosine(pa, pc)
    assert sim_ab > sim_ac, f"Projection should preserve relative similarity: ab={sim_ab} ac={sim_ac}"
