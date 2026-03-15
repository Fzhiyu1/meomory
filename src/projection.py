"""随机投影降维：4096 → target_dim"""
import math
import random


def create_projection_matrix(input_dim: int = 4096, output_dim: int = 256, seed: int = 42) -> list[list[float]]:
    """创建随机投影矩阵 [output_dim × input_dim]。
    使用高斯随机投影，每个元素 ~ N(0, 1/output_dim)。
    """
    rng = random.Random(seed)
    scale = 1.0 / math.sqrt(output_dim)
    matrix = []
    for _ in range(output_dim):
        row = [rng.gauss(0, scale) for _ in range(input_dim)]
        matrix.append(row)
    return matrix


def project(vector: list[float], proj_matrix: list[list[float]]) -> list[float]:
    """将 input_dim 维向量投影到 output_dim 维。"""
    result = []
    for row in proj_matrix:
        val = sum(a * b for a, b in zip(row, vector))
        result.append(val)
    return result


def project_batch(vectors: list[list[float]], proj_matrix: list[list[float]]) -> list[list[float]]:
    """批量投影。"""
    return [project(v, proj_matrix) for v in vectors]
